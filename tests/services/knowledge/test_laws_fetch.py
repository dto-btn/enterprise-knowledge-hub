"""Unit tests for Laws-Lois page-fetching logic (isolated from DB, queue, embeddings).

Run live tests (real HTTP calls to Laws-Lois) with:
    uv run -m pytest tests/services/knowledge/test_laws_fetch.py -v -s -k live
"""
# pylint: disable=protected-access

import enum
import sys
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

# Python 3.10 compat: backfill StrEnum if missing
if not hasattr(enum, "StrEnum"):
    class _StrEnum(str, enum.Enum):
        """Minimal StrEnum backfill for Python < 3.11."""
    enum.StrEnum = _StrEnum

# Stub out heavy/unavailable modules before importing the service
_stub = MagicMock()
for _mod in (
    "playhouse", "playhouse.postgres_ext",
    "peewee",
    "pgvector", "pgvector.peewee",
    "repository.database", "repository.base_model",
    "repository.knowledge_laws_model", "repository.knowledge_laws",
    "services.database.law_item_service",
    "services.database.run_history_service",
    "services.queue.queue_worker", "services.queue.queue_service",
    "provider.embedding.qwen3.embedder_factory",
    "torch",
):
    sys.modules.setdefault(_mod, MagicMock())

from services.knowledge.laws.laws import LawEntry, LawsKnowledgeService  # pylint: disable=wrong-import-position


# ── Sample HTML/XML fixtures ──────────────────────────────────────────────────

ACTS_INDEX_HTML = """
<html><body>
<div class="contentBlock"><ul class="wb-filter wet-boew-zebra titlesList">
<li><span class="objTitle"><a class="TocTitle" href="A-1/index.html">
Access to Information Act
</a></span></li>
<li><span class="objTitle"><a class="TocTitle" href="A-0.6/index.html">
Accessible Canada Act
</a></span></li>
<li><span class="objTitle"><a class="TocTitle" href="A-1/index.html">
Access to Information Act (duplicate)
</a></span></li>
</ul></div>
</body></html>
"""

SINGLE_ACT_INDEX_HTML = """
<html><body>
<div class="contentBlock"><ul class="wb-filter wet-boew-zebra titlesList">
<li><span class="objTitle"><a class="TocTitle" href="A-1/index.html">
Access to Information Act
</a></span></li>
</ul></div>
</body></html>
"""

REGULATIONS_INDEX_HTML = """
<html><body>
<div class="contentBlock"><ul class="wb-filter wet-boew-zebra titlesList">
<li><span class="objTitle"><a class="TocTitle" href="SOR-86-1026/index.html">
Ferry Cable Regulations
</a></span></li>
</ul></div>
</body></html>
"""

EMPTY_INDEX_HTML = """
<html><body><div class="contentBlock"><ul class="titlesList"></ul></div></body></html>
"""

STATUTE_XML = """<?xml version="1.0"?><Statute lims:current-date="2026-06-14"
lims:lastAmendedDate="2026-06-14" xmlns:lims="http://justice.gc.ca/lims">
<Identification><ShortTitle>Access to Information Act</ShortTitle></Identification>
<Body><Section><Text>The purpose of this Act is to enhance the accountability and
transparency of federal institutions in order to promote an open and democratic
society and to enable public debate.</Text></Section></Body></Statute>"""

REGULATION_XML = """<?xml version="1.0"?><Regulation lims:current-date="2019-08-29"
xmlns:lims="http://justice.gc.ca/lims">
<Identification><ShortTitle>Ferry Cable Regulations</ShortTitle></Identification>
<Order><Provision><Text>These Regulations respecting ferry cables in navigable waters
establish requirements for the marking and maintenance of ferry cables to ensure
the safety of navigation.</Text></Provision></Order></Regulation>"""

# No lims:current-date/lastAmendedDate — exercises the lims:pit-date fallback.
STATUTE_XML_PIT_DATE_ONLY = """<?xml version="1.0"?><Statute lims:pit-date="2024-03-01"
xmlns:lims="http://justice.gc.ca/lims">
<Identification><ShortTitle>Directive Act</ShortTitle></Identification>
<Body><Section><Text>This Act establishes requirements for the release of information
and data under the Government of Canada's commitment to transparent operations.</Text>
</Section></Body></Statute>"""

# Content too short to pass the 50-char minimum.
STATUTE_XML_TOO_SHORT = """<?xml version="1.0"?><Statute lims:current-date="2025-01-01"
xmlns:lims="http://justice.gc.ca/lims">
<Identification><ShortTitle>Short Act</ShortTitle></Identification>
<Body><Section><Text>Too short.</Text></Section></Body></Statute>"""

UNRECOGNIZED_XML_ROOT = """<?xml version="1.0"?><SomethingElse/>"""


class TestLawsFetch(unittest.TestCase):
    """Tests for _fetch_law_entries and _fetch_law in isolation."""

    def _build_service(self) -> LawsKnowledgeService:
        """Create a LawsKnowledgeService with all dependencies mocked."""
        queue_service = MagicMock()
        logger = MagicMock()
        run_history_service = MagicMock()
        run_history_service.select_first_instance_of_run_id.return_value = None

        svc = LawsKnowledgeService(
            queue_service=queue_service,
            logger=logger,
            run_history_service=run_history_service,
        )
        # Mock the DB service so no real DB calls are made
        svc._law_item_service = MagicMock()
        return svc

    # ── _fetch_law_entries ────────────────────────────────────────────────────

    def test_fetch_law_entries_extracts_unique_pages(self):
        """Should extract unique (page, title) pairs from <a class='TocTitle'> elements."""
        svc = self._build_service()

        acts_resp = MagicMock()
        acts_resp.status_code = 200
        acts_resp.text = ACTS_INDEX_HTML
        acts_resp.raise_for_status = MagicMock()

        empty_resp = MagicMock()
        empty_resp.status_code = 200
        empty_resp.text = EMPTY_INDEX_HTML
        empty_resp.raise_for_status = MagicMock()

        # First call returns acts, all subsequent index calls (B..Z, Num) are empty.
        svc.session.get = MagicMock(side_effect=[acts_resp] + [empty_resp] * 26)

        entries = svc._fetch_law_entries("acts")

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0], LawEntry(page="A-1", title="Access to Information Act", kind="acts"))
        self.assertEqual(entries[1].page, "A-0.6")

    def test_fetch_law_entries_skips_404_index_pages(self):
        """Should tolerate 404s from index pages without raising (e.g. no acts starting with X)."""
        svc = self._build_service()

        not_found_resp = MagicMock()
        not_found_resp.status_code = 404

        svc.session.get = MagicMock(return_value=not_found_resp)

        entries = svc._fetch_law_entries("regulations")

        self.assertEqual(entries, [])

    def test_fetch_law_entries_regulations(self):
        """Should parse regulation entries the same way as acts."""
        svc = self._build_service()

        regs_resp = MagicMock()
        regs_resp.status_code = 200
        regs_resp.text = REGULATIONS_INDEX_HTML
        regs_resp.raise_for_status = MagicMock()

        empty_resp = MagicMock()
        empty_resp.status_code = 200
        empty_resp.text = EMPTY_INDEX_HTML
        empty_resp.raise_for_status = MagicMock()

        svc.session.get = MagicMock(side_effect=[regs_resp] + [empty_resp] * 26)

        entries = svc._fetch_law_entries("regulations")

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].page, "SOR-86-1026")
        self.assertEqual(entries[0].kind, "regulations")

    # ── _fetch_law ────────────────────────────────────────────────────────────

    def test_fetch_law_parses_statute_xml(self):
        """Should extract title, content, page_id and last_modified from a Statute XML doc."""
        svc = self._build_service()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = STATUTE_XML.encode("utf-8")
        mock_response.raise_for_status = MagicMock()
        svc.session.get = MagicMock(return_value=mock_response)

        entry = LawEntry(page="A-1", title="Access to Information Act", kind="acts")
        item = svc._fetch_law(entry)

        self.assertIsNotNone(item)
        self.assertEqual(item.name, "Access to Information Act")
        self.assertEqual(item.source, "laws")
        self.assertIn("accountability", item.content)
        self.assertEqual(item.last_modified_date, datetime(2026, 6, 14))
        # page_id is a deterministic hash of the page.
        self.assertEqual(item.page_id, LawsKnowledgeService._get_page_id("A-1"))

    def test_fetch_law_parses_regulation_xml(self):
        """Should extract title/content from a Regulation XML doc (different root tag)."""
        svc = self._build_service()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = REGULATION_XML.encode("utf-8")
        mock_response.raise_for_status = MagicMock()
        svc.session.get = MagicMock(return_value=mock_response)

        entry = LawEntry(page="SOR-86-1026", title="Ferry Cable Regulations", kind="regulations")
        item = svc._fetch_law(entry)

        self.assertIsNotNone(item)
        self.assertEqual(item.name, "Ferry Cable Regulations")
        self.assertIn("navigable waters", item.content)
        self.assertEqual(item.last_modified_date, datetime(2019, 8, 29))

    def test_fetch_law_strips_identification_metadata(self):
        """Should exclude bibliographic metadata (Identification/RecentAmendments) from content."""
        svc = self._build_service()

        xml_with_metadata = """<?xml version="1.0"?><Statute lims:current-date="2026-06-14"
xmlns:lims="http://justice.gc.ca/lims">
<Identification><ShortTitle>Access to Information Act</ShortTitle>
<RunningHead>Access to Information</RunningHead></Identification>
<Body><Section><Text>The purpose of this Act is to enhance the accountability and
transparency of federal institutions in order to promote an open and democratic
society.</Text></Section></Body>
<RecentAmendments><Amendment>2024, c. 1</Amendment></RecentAmendments></Statute>"""

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = xml_with_metadata.encode("utf-8")
        mock_response.raise_for_status = MagicMock()
        svc.session.get = MagicMock(return_value=mock_response)

        entry = LawEntry(page="A-1", title="Access to Information Act", kind="acts")
        item = svc._fetch_law(entry)

        self.assertIsNotNone(item)
        # Title is still extracted from Identification before it's stripped
        self.assertEqual(item.name, "Access to Information Act")
        self.assertIn("accountability", item.content)
        self.assertNotIn("Running", item.content)
        self.assertNotIn("2024, c. 1", item.content)

    def test_fetch_law_falls_back_to_pit_date(self):
        """Should fall back to lims:pit-date when current-date/lastAmendedDate are absent."""
        svc = self._build_service()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = STATUTE_XML_PIT_DATE_ONLY.encode("utf-8")
        mock_response.raise_for_status = MagicMock()
        svc.session.get = MagicMock(return_value=mock_response)

        entry = LawEntry(page="D-1", title="Directive Act", kind="acts")
        item = svc._fetch_law(entry)

        self.assertIsNotNone(item)
        self.assertEqual(item.last_modified_date, datetime(2024, 3, 1))

    def test_fetch_law_returns_none_on_404(self):
        """Should return None when the XML endpoint 404s for a page."""
        svc = self._build_service()

        mock_response = MagicMock()
        mock_response.status_code = 404
        svc.session.get = MagicMock(return_value=mock_response)

        entry = LawEntry(page="Z-99", title="Nonexistent Act", kind="acts")
        item = svc._fetch_law(entry)

        self.assertIsNone(item)

    def test_fetch_law_skips_short_content(self):
        """Should return None when the extracted content is too short (< 50 chars)."""
        svc = self._build_service()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = STATUTE_XML_TOO_SHORT.encode("utf-8")
        mock_response.raise_for_status = MagicMock()
        svc.session.get = MagicMock(return_value=mock_response)

        entry = LawEntry(page="S-1", title="Short Act", kind="acts")
        item = svc._fetch_law(entry)

        self.assertIsNone(item)

    def test_fetch_law_unrecognized_root(self):
        """Should return None when the XML root is neither <Statute> nor <Regulation>."""
        svc = self._build_service()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = UNRECOGNIZED_XML_ROOT.encode("utf-8")
        mock_response.raise_for_status = MagicMock()
        svc.session.get = MagicMock(return_value=mock_response)

        entry = LawEntry(page="?-1", title="Weird", kind="acts")
        item = svc._fetch_law(entry)

        self.assertIsNone(item)

    def test_get_page_id_is_deterministic(self):
        """Same page must always hash to the same page_id."""
        first = LawsKnowledgeService._get_page_id("A-1")
        second = LawsKnowledgeService._get_page_id("A-1")
        different = LawsKnowledgeService._get_page_id("C-46")

        self.assertEqual(first, second)
        self.assertNotEqual(first, different)

    # ── fetch_from_source (integration of fetch + skip logic) ─────────────────

    @patch("time.sleep", return_value=None)
    def test_fetch_from_source_skips_up_to_date(self, _mock_sleep):
        """Should skip laws the DB says are up-to-date and yield the rest."""
        svc = self._build_service()

        def fake_get(url, timeout=None):  # pylint: disable=unused-argument
            response = MagicMock()
            response.status_code = 200
            response.raise_for_status = MagicMock()
            if url == "https://laws-lois.justice.gc.ca/eng/acts/A.html":
                response.text = SINGLE_ACT_INDEX_HTML
            elif url == "https://laws-lois.justice.gc.ca/eng/regulations/A.html":
                response.text = REGULATIONS_INDEX_HTML
            elif url.startswith("https://laws-lois.justice.gc.ca/eng/acts/"):
                response.text = EMPTY_INDEX_HTML
            elif url.startswith("https://laws-lois.justice.gc.ca/eng/regulations/"):
                response.text = EMPTY_INDEX_HTML
            elif url == "https://laws-lois.justice.gc.ca/eng/XML/A-1.xml":
                response.content = STATUTE_XML.encode("utf-8")
            elif url == "https://laws-lois.justice.gc.ca/eng/XML/SOR-86-1026.xml":
                response.content = REGULATION_XML.encode("utf-8")
            return response

        svc.session.get = MagicMock(side_effect=fake_get)

        # Mark the act (A-1) as up-to-date, the regulation as stale
        page_id_a1 = LawsKnowledgeService._get_page_id("A-1")

        def is_up_to_date(page_id, _source, _last_mod):
            return page_id == page_id_a1
        svc._law_item_service.record_is_up_to_date.side_effect = is_up_to_date

        items = list(svc.fetch_from_source())

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].name, "Ferry Cable Regulations")
        svc._law_item_service.delete_by_page_id_source.assert_called_once_with(
            items[0].page_id, "laws"
        )


if __name__ == "__main__":
    unittest.main()


class TestLawsLiveFetch(unittest.TestCase):
    """Live tests that hit real Laws-Lois URLs — run with `pytest -s -k live` to observe output."""

    def _build_service(self) -> LawsKnowledgeService:
        queue_service = MagicMock()
        logger = MagicMock()
        run_history_service = MagicMock()
        svc = LawsKnowledgeService(
            queue_service=queue_service,
            logger=logger,
            run_history_service=run_history_service,
        )
        svc._law_item_service = MagicMock()
        return svc

    def test_live_fetch_acts_index_and_first_laws(self):
        """Fetch the real 'A' acts index, print discovered entries, then fetch first 3 laws."""
        svc = self._build_service()

        response = svc.session.get(
            "https://laws-lois.justice.gc.ca/eng/acts/A.html", timeout=30
        )
        response.raise_for_status()
        from bs4 import BeautifulSoup  # pylint: disable=import-outside-toplevel
        soup = BeautifulSoup(response.text, "lxml")
        entries = [
            LawEntry(page=a["href"].split("/")[0], title=a.get_text(strip=True), kind="acts")
            for a in soup.find_all("a", class_="TocTitle", href=True)
        ]

        print(f"\n{'=' * 70}")
        print(f"ACTS INDEX 'A': discovered {len(entries)} entries")
        print(f"First 5: {[(e.page, e.title) for e in entries[:5]]}")
        print(f"{'=' * 70}")

        self.assertGreater(len(entries), 0, "Expected at least one act from the 'A' index")

        fetched = 0
        for entry in entries[:3]:
            print(f"\n{'─' * 70}")
            print(f"Fetching page={entry.page} ...")
            item = svc._fetch_law(entry)

            if item is None:
                print("  → returned None (skipped or empty)")
                continue

            fetched += 1
            print(f"  title          : {item.name}")
            print(f"  page_id        : {item.page_id}")
            print(f"  source         : {item.source}")
            print(f"  last_modified  : {item.last_modified_date}")
            print(f"  content length : {len(item.content)} chars")
            print(f"  content preview: {item.content[:300]}...")
            print(f"{'─' * 70}")

        self.assertGreater(fetched, 0, "Expected at least one law to parse successfully")
