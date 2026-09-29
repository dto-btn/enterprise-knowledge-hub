"""
Laws-Lois knowledge service implementation.

Ingests Acts and Regulations from the Department of Justice's Laws-Lois website:
    List of Consolidated Acts (English):
        https://laws-lois.justice.gc.ca/eng/acts/{CAPITAL_LETTER OR "Num"}.html
    Individual Act:
        https://laws-lois.justice.gc.ca/eng/acts/{page}/index.html
    List of Consolidated Regulations:
        https://laws-lois.justice.gc.ca/eng/regulations/{CAPITAL_LETTER OR "Num"}.html
    Individual Regulation:
        https://laws-lois.justice.gc.ca/eng/regulations/{page}/index.html
    XML (both Acts and Regulations):
        https://laws-lois.justice.gc.ca/eng/XML/{page}.xml

Discovers all Act/Regulation pages from the alphabetical index pages, fetches
the consolidated XML for each, and yields it as a KnowledgeItem for downstream
processing (embedding) and storage.
"""
import hashlib
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from repository.knowledge_laws_model import KnowledgeBaseLaws
from services.database.law_item_service import LawItemService
from services.knowledge.base import KnowledgeService
from services.knowledge.models import KnowledgeItem
from services.knowledge.laws.models import LawItemRaw, LawItemProcessed

# Index pages listing all Acts/Regulations, one per starting letter (plus "Num" for
# instruments whose title starts with a number).
LAWS_ACTS_INDEX_URL = "https://laws-lois.justice.gc.ca/eng/acts/{letter}.html"
LAWS_REGULATIONS_INDEX_URL = "https://laws-lois.justice.gc.ca/eng/regulations/{letter}.html"
# Consolidated XML for a single Act or Regulation, keyed by its page (e.g. "A-1", "SOR-86-1026")
LAWS_XML_URL = "https://laws-lois.justice.gc.ca/eng/XML/{page}.xml"

_INDEX_LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["Num"]

# Polite delay between fetches (seconds)
_FETCH_DELAY = float(os.getenv("LAWS_FETCH_DELAY", "1.0"))
_REQUEST_TIMEOUT = int(os.getenv("LAWS_REQUEST_TIMEOUT", "30"))


@dataclass
class LawEntry:
    """A single Act/Regulation discovered from an index page."""
    page: str
    title: str
    kind: str  # "acts" or "regulations"


@dataclass
class LawsKnowledgeService(KnowledgeService):
    """Knowledge service for Laws-Lois Acts and Regulations."""

    def __init__(self, queue_service, logger, run_history_service):
        super().__init__(queue_service=queue_service, logger=logger,
                         run_history_service=run_history_service, service_name="laws")
        self._session: requests.Session | None = None
        self._law_item_service = LawItemService(logger)

    def get_query_instruction(self) -> str:
        return (
            "Instruct: Given a query, retrieve relevant Government of Canada acts and "
            "regulations that answer the query\n"
            "Query: "
        )

    @property
    def session(self) -> requests.Session:
        """Lazy-initialized requests session with common headers."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": "EnterpriseKnowledgeHub/1.0 (GC Internal)",
                "Accept-Language": "en-CA,en;q=0.9",
            })
        return self._session

    def get_batch_size(self) -> int:
        return int(os.getenv("LAWS_PROCESS_BATCH_SIZE", "32"))

    def _get_run_id(self) -> int:
        """Generate a run ID based on the current date (one run per day is expected)."""
        today = datetime.now().strftime("%Y-%m-%d")
        digest = hashlib.sha256(f"laws-{today}".encode()).digest()
        return int.from_bytes(digest[:4], "big", signed=False) & 0x7FFFFFFF

    @staticmethod
    def _get_page_id(page: str) -> int:
        """Generate a numeric page_id from a law's page (e.g. "A-1", "SOR-86-1026")."""
        digest = hashlib.sha256(page.encode()).digest()
        return int.from_bytes(digest[:4], "big", signed=False) & 0x7FFFFFFF

    # ─── INGEST STAGE ────────────────────────────────────────────────────────────

    def fetch_from_source(self) -> Iterator[LawItemRaw]:
        """Fetch the Acts and Regulations indices, then yield each individual law."""
        self.logger.info("Fetching Laws-Lois Acts and Regulations indices.")

        entries = self._fetch_law_entries("acts") + self._fetch_law_entries("regulations")
        self.logger.info("Discovered %d laws from Laws-Lois indices.", len(entries))

        for entry in entries:
            try:
                item = self._fetch_law(entry)
                if item is None:
                    continue

                # Skip laws already stored with the same or newer last_modified_date
                if self._law_item_service.record_is_up_to_date(
                    item.page_id, item.source, item.last_modified_date
                ):
                    self.logger.debug("Law %s (%s) is up to date, skipping.", entry.page, item.name)
                    continue

                # Delete stale chunks before re-ingesting updated law
                self._law_item_service.delete_by_page_id_source(item.page_id, item.source)
                yield item
            except Exception:
                self.logger.exception("Failed to fetch law page=%s, skipping.", entry.page)
                continue

            # Be polite to the server
            time.sleep(_FETCH_DELAY)

    def _fetch_law_entries(self, kind: str) -> list[LawEntry]:
        """Fetch every alphabetical index page for a given kind and collect its law entries."""
        base_url = LAWS_ACTS_INDEX_URL if kind == "acts" else LAWS_REGULATIONS_INDEX_URL

        entries: list[LawEntry] = []
        seen_pages: set[str] = set()
        for letter in _INDEX_LETTERS:
            url = base_url.format(letter=letter)
            response = self.session.get(url, timeout=_REQUEST_TIMEOUT)
            if response.status_code == 404:
                continue
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")
            for anchor in soup.find_all("a", class_="TocTitle", href=True):
                # href looks like "A-1/index.html" — the page is the first path segment.
                page = anchor["href"].split("/")[0]
                title = anchor.get_text(strip=True)
                if not page or not title or page in seen_pages:
                    continue
                seen_pages.add(page)
                entries.append(LawEntry(page=page, title=title, kind=kind))

        return entries

    def _fetch_law(self, entry: LawEntry) -> LawItemRaw | None:
        """Fetch a single law's consolidated XML and parse its title, content, and modified date."""
        url = LAWS_XML_URL.format(page=entry.page)
        self.logger.debug("Fetching law XML: %s", url)

        response = self.session.get(url, timeout=_REQUEST_TIMEOUT)
        if response.status_code == 404:
            self.logger.debug("No XML found for page=%s, skipping.", entry.page)
            return None
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "xml")

        # Acts are rooted at <Statute>, Regulations at <Regulation>.
        root = soup.find(["Statute", "Regulation"])
        if root is None:
            self.logger.warning("Unrecognized XML root for page=%s, skipping.", entry.page)
            return None

        title_tag = root.find("ShortTitle") or root.find("LongTitle")
        title = title_tag.get_text(strip=True) if title_tag else entry.title

        # Identification carries bibliographic metadata (titles, bill history, chapter
        # numbers) and RecentAmendments is a changelog — neither is useful chunk content.
        for tag in root.find_all(["Identification", "RecentAmendments"]):
            tag.decompose()

        content = root.get_text(separator="\n", strip=True)
        if not content or len(content.strip()) < 50:
            self.logger.debug("Skipping law %s (%s) — content too short.", entry.page, title)
            return None

        last_modified = self._extract_last_modified(root)

        return LawItemRaw(
            name=title,
            content=content,
            page_id=self._get_page_id(entry.page),
            source="laws",
            last_modified_date=last_modified,
        )

    def _extract_last_modified(self, root) -> datetime | None:
        """Extract the last-amended date from the root element's lims:* attributes."""
        for attr in ("lims:current-date", "lims:lastAmendedDate", "lims:pit-date"):
            value = root.get(attr)
            if not value:
                continue
            try:
                return datetime.strptime(value, "%Y-%m-%d")
            except ValueError:
                continue
        return None

    # ─── EMIT / PROCESS / STORE (queue plumbing) ─────────────────────────────────

    def emit_fetched_item(self, item: LawItemRaw) -> None:
        """Chunk the raw item and write chunks to the ingest queue."""
        # Lazy import: avoids GPU initialisation during ingest-only runs
        from provider.embedding.qwen3.embedder_factory import get_embedder  # pylint: disable=import-outside-toplevel
        embedder = get_embedder()

        max_tokens = getattr(embedder, "max_seq_length", None)
        chunks = embedder.chunk_text_by_tokens(item.content, max_tokens=max_tokens)
        num_chunks = len(chunks)

        for idx, chunk_text in enumerate(chunks, start=1):
            chunk_item = LawItemRaw(
                name=item.name,
                content=chunk_text,
                page_id=item.page_id,
                source=item.source,
                last_modified_date=item.last_modified_date,
                chunk_index=idx,
                chunk_count=num_chunks,
            )
            self.queue_service.write(self._ingest_queue_name(), chunk_item)

    def process_item(self, knowledge_item: KnowledgeItem) -> None:
        """Process a single item — compute embeddings and emit to processed queue."""
        from provider.embedding.qwen3.embedder_factory import get_embedder  # pylint: disable=import-outside-toplevel
        import numpy as np  # pylint: disable=import-outside-toplevel

        embedder = get_embedder()
        content = knowledge_item['content'] if isinstance(knowledge_item, dict) else knowledge_item.content

        embeddings = embedder.embed([content])
        vec = np.asarray(embeddings)[0]

        item_data = knowledge_item if isinstance(knowledge_item, dict) else knowledge_item.model_dump()
        processed = LawItemProcessed(
            name=item_data['name'],
            content=item_data['content'],
            page_id=item_data['page_id'],
            source=item_data['source'],
            last_modified_date=item_data.get('last_modified_date'),
            chunk_index=item_data.get('chunk_index', 1),
            chunk_count=item_data.get('chunk_count', 1),
            embeddings=vec,
        )
        self.emit_processed_item(processed)

    def emit_processed_item(self, item: LawItemProcessed) -> None:
        """Write processed item to the processed queue."""
        self.queue_service.write(self._processed_queue_name(), item)

    def store_item(self, item: KnowledgeItem) -> None:
        """Store the processed item into the database."""
        validated = LawItemProcessed.model_validate(item)
        record_to_insert = KnowledgeBaseLaws.from_item(validated)
        self._law_item_service.insert(record_to_insert.as_mapping())