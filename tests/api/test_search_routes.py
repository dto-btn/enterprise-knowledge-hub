"""Tests for the knowledge search route and Wikipedia result enrichment."""
import logging
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

import router.root.search_retrieve_endpoints as search_module
from services.database.wiki_item_service import WikipediaArticleService


class TestSearchRoute(unittest.TestCase):
    """Contract tests for GET /{slug}/search."""

    def setUp(self) -> None:
        app = FastAPI()
        app.include_router(search_module.router, prefix="/database")
        self.client = TestClient(app)

        self.service = MagicMock()
        registry = MagicMock()
        registry.get.return_value = SimpleNamespace(model_name="test-model", query_instruction="q")
        embedder = MagicMock()
        embedder.embed.return_value = [0.1, 0.2]

        patches = [
            patch.dict(search_module._SEARCH_REGISTRY, {"wikipedia": self.service}),  # pylint: disable=protected-access
            patch.object(search_module, "_source_registry_service", registry),
            patch.object(search_module, "get_embedder", return_value=embedder),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_returns_citation_metadata(self) -> None:
        """Results should carry id, source, language and url."""
        self.service.search_by_embedding.return_value = [{
            "id": 42, "source": "frwiki", "name": "Ottawa", "content": "c",
            "chunk_index": 0, "similarity": 0.9,
            "language": "fr", "url": "https://fr.wikipedia.org/?curid=42",
        }]

        response = self.client.get("/database/wikipedia/search", params={"query": "Ottawa"})

        self.assertEqual(response.status_code, 200)
        result = response.json()["results"][0]
        self.assertEqual(result["id"], 42)
        self.assertEqual(result["source"], "frwiki")
        self.assertEqual(result["language"], "fr")
        self.assertEqual(result["url"], "https://fr.wikipedia.org/?curid=42")

    def test_source_filter_is_passed_through(self) -> None:
        """The optional source param should reach the service."""
        self.service.search_by_embedding.return_value = []

        response = self.client.get("/database/wikipedia/search",
                                   params={"query": "x", "limit": 5, "source": "enwiki"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total"], 0)
        self.service.search_by_embedding.assert_called_once_with(
            [0.1, 0.2], limit=5, source="enwiki")

    def test_source_filter_defaults_to_none(self) -> None:
        """Without a source param, both languages are searched."""
        self.service.search_by_embedding.return_value = []

        self.client.get("/database/wikipedia/search", params={"query": "x"})

        self.assertIsNone(self.service.search_by_embedding.call_args.kwargs["source"])

    def test_metadata_fields_are_optional(self) -> None:
        """Rows without the new fields (e.g. tbs-policies) still validate."""
        self.service.search_by_embedding.return_value = [{
            "name": "Policy", "content": "c", "chunk_index": 1, "similarity": 0.5,
        }]

        response = self.client.get("/database/wikipedia/search", params={"query": "x"})

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["results"][0]["url"])

    def test_query_text_is_not_logged(self) -> None:
        """User query text must not appear in logs."""
        self.service.search_by_embedding.return_value = []

        with self.assertLogs(search_module.logger, level=logging.INFO) as logs:
            self.client.get("/database/wikipedia/search", params={"query": "secret prompt"})

        self.assertNotIn("secret prompt", "\n".join(logs.output))


class TestWikipediaSearchEnrichment(unittest.TestCase):
    """Language and URL derivation in WikipediaArticleService."""

    def setUp(self) -> None:
        self.service = WikipediaArticleService(logging.getLogger(__name__))
        self.service._repository = MagicMock()  # pylint: disable=protected-access

    def _search(self, rows: list[dict]) -> list[dict]:
        self.service._repository.search_by_embedding.return_value = rows  # pylint: disable=protected-access
        return self.service.search_by_embedding([0.1], limit=3, source="enwiki")

    def test_english_and_french_rows(self) -> None:
        """enwiki/frwiki map to en/fr with curid URLs."""
        rows = self._search([{"id": 1, "source": "enwiki"}, {"id": 2, "source": "frwiki"}])

        self.assertEqual(rows[0]["language"], "en")
        self.assertEqual(rows[0]["url"], "https://en.wikipedia.org/?curid=1")
        self.assertEqual(rows[1]["language"], "fr")
        self.assertEqual(rows[1]["url"], "https://fr.wikipedia.org/?curid=2")
        self.service._repository.search_by_embedding.assert_called_once_with(  # pylint: disable=protected-access
            [0.1], limit=3, source="enwiki")

    def test_unknown_or_missing_source(self) -> None:
        """Rows with no recognised source get no language or URL."""
        rows = self._search([{"id": 3, "source": None}, {"id": 4, "source": "other"}])

        for row in rows:
            self.assertIsNone(row["language"])
            self.assertIsNone(row["url"])


if __name__ == "__main__":
    unittest.main()
