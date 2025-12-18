import bz2
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import threading
import time

from dotenv import load_dotenv
from provider.embedding.base import EmbeddingBackendProvider
from provider.embedding.embeddingUtil import EmbeddingUtil
from provider.embedding.torch import TorchEmbeddingBackend
from services.knowledge.base import KnowledgeService
from services.knowledge.models import WikipediaItem

load_dotenv()

PROGRESS_SUFFIX: str = ".progress"
INDEX_FILENAME = re.compile(r"(?P<prefix>.+)-index(?P<chunk>\d*)\.txt\.bz2")

@dataclass
class WikipediaKnowedgeService(KnowledgeService):
    """Knowledge service for Wikipedia ingestion."""

    _ignored_title_prefixes: tuple[str, ...] = (
            "Draft:",
            "Category:",
            "File:",
            "Wikipedia:",
            "Ébauche:",
            "Catégorie:",
            "Fichier:",
            "Wikipédia:",
            "Portal:"
        )

    _content_folder_path: Path = Path(os.getenv("WIKIPEDIA_CONTENT_FOLDER",
                                    "./content/wikipedia")).expanduser().resolve()

    _progress_flush_interval: int = 1000 # for the .progress file we track line number we stpped.

    def __init__(self, queue_service, logger):
        super().__init__(queue_service=queue_service, logger=logger, service_name="wikipedia")

    def process(
        self, 
        max_batch_size: int,
        model: str,
        isGGUF: bool,
        device: str,
        max_seq_length: int,
        max_batch_cap: int,
        overlap_tokens: int,
        limit: int | None = None,
        process_done_event: threading.Event | None = None,
        idle_sleep: float | None = None
    ) -> None:
        
        processed = 0
        effective_limit = self._effective_limit(limit, self.process_limit)
        input_queue = self.process_queue_name
        start_time = time.perf_counter()
        
        #init backend.  in constructor?  check with sequence of events on where this needs to be init
        backend = TorchEmbeddingBackend(model, device, max_seq_length)
        
        # get max batch size.  #random for now
        # max_batch_size = 512 
        max_batch_size = EmbeddingUtil.detect_max_batch_size(max_seq_length, device, max_batch_cap=max_batch_cap)
        self.logger.info("Processing ingested data. (%s)", self.service_name)
        # Placeholder for processing logic
        try:
            i = 0
            print('try')
            for batch in EmbeddingUtil.batched(self.process_queue(backend,
                                                                  process_done_event,
                                                                  idle_sleep,
                                                                  overlap_tokens), max_batch_size):
                print(i)
                i = i + 1
                if (i > 0):
                    print('break')
                    break
            # elapsed = max(time.perf_counter() - start_time, 1e-6)
            # rate = processed / elapsed
            # self.logger.info(
            #     "Vectorized %s items from %s (limit=%s, %.1f msg/s)",
            #     processed,
            #     input_queue,
            #     effective_limit or "unbounded",
            #     rate,
            # )
            # return processed
            
            
            # item = self.queue_service.read(self.service_name + ".ingest")
            # print('raw item')
            # print(item)
            # self.process_queue(item)
        except:
            print() #to fix
            
    def process_queue(
        self, 
        backend: EmbeddingBackendProvider,         
        process_done_event: threading.Event | None = None,
        idle_sleep: float | None = None,
        overlap_tokens: int = 64
    ):
        """Process ingested WikipediaItem from the queue."""
        
        sleep_interval = self._normalize_idle_sleep(idle_sleep)
        tokenizer = backend.tokenizer
        max_tokens = backend.max_seq_len
        
        for item in self.queue_service.read(self.service_name + ".ingest"): #should change this read to a read without ackloeldge
            try:
                payload: WikipediaItem = WikipediaItem(**item)
            except:
                continue #to be added
            try:
                for chunk in EmbeddingUtil.article_to_chunks(payload, tokenizer, max_tokens, overlap_tokens):
                    yield chunk
            except:
                continue #to be added
            #acknoiledge
            #increment stats
                
        

        # payload = json.loads(item)
        # print("item================")
        # print(item)
        # itemdict = item.to_dict()
        # print('to_dict')
        # print(itemdict)
        # backend = TorchEmbeddingBackend(model_name="test", device="cpu")
        # chunk = EmbeddingUtil.article_to_chunks(itemdict, backend.tokenizer)
        # print("chunk")
        # print(chunk)
        
            
        #self.logger.debug("Processing Wikipedia item: %s", item.title)
        # add vector logic here.


    def fetch_from_source(self) -> Iterator[WikipediaItem]:
        """Read data from Wikipedia index.txt.bz2 source.queue_service

            The content will be first entrypoint in the main .bz2 multistream file.
                Ex: 345,6789,Fruits (Read more on the doc from the README.md from content/ folder)
        """
        for index_path in self._discover_index_files():
            self.logger.info("Reading data from Wikipedia source: %s", index_path)

            # Load last processed line number
            start_line = self._load_progress(index_path)
            if start_line > 0:
                self.logger.info("Resuming from line %d for %s", start_line, index_path.name)

            temp_last_byteoffset: int | None = None
            current_line = 0

            with bz2.open(index_path, mode='rt') as index_file:
                for line in index_file:
                    current_line += 1

                    # Parse the offset first
                    try:
                        offset_str, _, _ = line.strip().split(":", 2)
                        offset = int(offset_str)
                    except ValueError:
                        self.logger.warning("Skipping malformed line %d in %s", current_line, index_path.name)
                        continue

                    # Skip already processed lines
                    if current_line <= start_line:
                        temp_last_byteoffset = offset
                        continue

                    # Process the line
                    try:
                        last_byteoffset = temp_last_byteoffset
                        temp_last_byteoffset = offset

                        if last_byteoffset is None:
                            last_byteoffset = offset
                            length = offset
                        else:
                            length = offset - last_byteoffset

                        if last_byteoffset != offset:
                            match = INDEX_FILENAME.match(index_path.name)
                            prefix = match.group("prefix")
                            chunk = match.group("chunk")
                            dump_name = f"{prefix}{chunk if chunk else ''}.xml.bz2"
                            dump_path = index_path.with_name(dump_name)

                            with open(dump_path, 'rb') as dump_file:
                                dump_file.seek(last_byteoffset)
                                data = dump_file.read(length)
                                try:
                                    decompressed = bz2.decompress(data)
                                    xml_content = decompressed.decode("utf-8", errors="ignore")
                                    for page_match in re.finditer(r"<page>(.*?)</page>", xml_content, re.DOTALL):
                                        page_xml = page_match.group(0)
                                        if not self._should_ignore_page(page_xml):
                                            item = self._parse_page_xml(page_xml)
                                            if item:
                                                yield item
                                except Exception as exc:
                                    self.logger.error(
                                        "Failed to decompress chunk from %s between offsets %s and %s: %s",
                                        dump_name, last_byteoffset, offset, exc
                                    )
                    except Exception as exc:
                        self.logger.error("Error processing line %d in %s: %s", current_line, index_path.name, exc)
                    finally:
                        # Save progress periodically
                        if current_line % self._progress_flush_interval == 0:
                            self._save_progress(index_path, current_line)

                # Final progress save at end of file
                self._save_progress(index_path, current_line)
                self.logger.info("Completed %s at line %d", index_path.name, current_line)

    def _save_progress(self, index_path: Path, line_number: int) -> None:
        """Save current progress (line number) to a small file."""
        progress_path = index_path.with_suffix(index_path.suffix + PROGRESS_SUFFIX)
        progress_path.write_text(str(line_number))

    def _load_progress(self, index_path: Path) -> int:
        """Load the last processed line number. Returns 0 if no progress file exists."""
        progress_path = index_path.with_suffix(index_path.suffix + PROGRESS_SUFFIX)
        if not progress_path.exists():
            return 0
        try:
            return int(progress_path.read_text().strip())
        except (ValueError, OSError):
            return 0

    def _discover_index_files(self) -> Iterator[Path]:
        """Discover index files in the content folder."""
        self.logger.debug("Searching for index files in ---> %s", self._content_folder_path)
        for node in sorted(self._content_folder_path.rglob("*.txt*")):
            if not node.is_file():
                continue
            if node.suffix == PROGRESS_SUFFIX:
                continue  # Skip progress files
            match = INDEX_FILENAME.match(node.name)
            if not match:
                self.logger.debug("Skipping index file with unknown pattern: %s", node.name)
                continue
            yield node

    def _should_ignore_page(self, xml_page: str) -> bool:
        """Check if a page should be ignored based on title or type."""
        if re.search(r"<redirect\s", xml_page):
            return True
        title_match = re.search(r"<title>([^<]+)</title>", xml_page)
        if title_match:
            title = title_match.group(1)
            for prefix in self._ignored_title_prefixes:
                if title.startswith(prefix):
                    return True
        return False

    def _parse_page_xml(self, xml_page: str) -> WikipediaItem | None:
        """Parse a Wikipedia page XML and extract relevant fields."""
        # Extract title
        title_match = re.search(r"<title>([^<]+)</title>", xml_page)
        title = title_match.group(1) if title_match else ""

        # Extract page ID
        pid_match = re.search(r"<id>(\d+)</id>", xml_page)
        pid = int(pid_match.group(1)) if pid_match else 0

        # Extract content (wiki markup text)
        text_match = re.search(r"<text[^>]*>([^<]*(?:<(?!/text>)[^<]*)*)</text>", xml_page, re.DOTALL)
        content = text_match.group(1) if text_match else ""

        # Extract last modified date (timestamp)
        timestamp_match = re.search(r"<timestamp>([^<]+)</timestamp>", xml_page)
        last_modified_date = None
        if timestamp_match:
            try:
                last_modified_date = datetime.fromisoformat(timestamp_match.group(1).replace("Z", "+00:00"))
            except ValueError:
                pass

        if not title or not content:
            return None

        return WikipediaItem(
            name=title,
            title=title,
            content=content,
            last_modified_date=last_modified_date,
            pid=pid,
        )