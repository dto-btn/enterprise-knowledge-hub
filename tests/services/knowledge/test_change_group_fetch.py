"""Unit tests for the local SharePoint PDF ingestion service."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock

from services.knowledge.change_group.change_group import ChangeGroupKnowledgeService


class TestChangeGroupFetch(unittest.TestCase):
    """Smoke tests for Change Group PDF discovery and extraction."""

    def test_fetch_from_source_reads_pdf_files_in_folder(self):
        """The service should discover .pdf files from the configured directory."""
        shared_dir = Path("/tmp/change-group-test")
        shared_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = shared_dir / "sample-document.pdf"
        pdf_path.write_bytes(b"dummy-pdf")

        service = ChangeGroupKnowledgeService(
            queue_service=MagicMock(),
            logger=MagicMock(),
            run_history_service=MagicMock(),
            root_folder=str(shared_dir),
            service_name="change-group",
        )
        service.extract_text_from_pdf = MagicMock(return_value="This policy covers accessibility and governance.")

        items = list(service.fetch_from_source())

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].name, "sample-document.pdf")
        self.assertIn("accessibility", items[0].content)
        assert service.extract_text_from_pdf.called

        pdf_path.unlink(missing_ok=True)
        shared_dir.rmdir()
