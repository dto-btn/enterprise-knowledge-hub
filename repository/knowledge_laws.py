"""Postgres/pgvector repository for Laws-Lois (Acts and Regulations) knowledge base."""
from __future__ import annotations

from datetime import datetime

from repository.base import EmbeddingRepository
from repository.knowledge_laws_model import KnowledgeBaseLaws


class KnowledgeLawsRepository(EmbeddingRepository):
    """Repository to read/write laws records."""

    id_field_name = "page_id"

    def __init__(self):
        super().__init__(KnowledgeBaseLaws)

    def get_by_page_id_source(self, page_id: int, source: str) -> list[KnowledgeBaseLaws]:
        """Get all chunks for a given page_id and source."""
        return self.get_chunks_by_id_source(page_id, source)

    def get_by_page_id_source_modified_date(self, page_id: int, source: str,
                                            last_date_modified: datetime) -> KnowledgeBaseLaws | None:
        """Get a record by page_id and source if it was modified after last_date_modified."""
        return self.get_by_id_source_modified_date(page_id, source, last_date_modified)

    def delete_by_page_id_source(self, page_id: int, source: str) -> None:
        """Delete all chunks for a given page_id and source."""
        self.delete_by_id_source(page_id, source)
