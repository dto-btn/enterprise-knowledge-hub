"""Service layer for Laws-Lois (Acts and Regulations) knowledge items."""
from datetime import datetime
import logging
from dataclasses import dataclass

from repository.knowledge_laws_model import KnowledgeBaseLaws
from repository.knowledge_laws import KnowledgeLawsRepository


@dataclass
class LawItemService:
    """Service to manage Laws-Lois embeddings."""

    logger: logging.Logger
    _repository: KnowledgeLawsRepository

    def __init__(self, logger):
        self._logger = logger
        self._repository = KnowledgeLawsRepository()

    def insert(self, row: dict) -> KnowledgeBaseLaws:
        """Insert a record."""
        return self._repository.create(
            page_id=row['page_id'],
            chunk_index=row['chunk_index'],
            name=row['name'],
            content=row['content'],
            last_modified_date=row['last_modified_date'],
            embedding=row['embedding'],
            source=row['source'],
        )

    def search_by_embedding(self, embedding: list[float], limit: int = 100) -> list[dict]:
        """Semantic search over laws by embedding similarity."""
        return self._repository.search_by_embedding(embedding, limit=limit)

    def delete_by_page_id_source(self, page_id: int, source: str) -> None:
        """Delete all chunks for a page_id and source."""
        self._repository.delete_by_page_id_source(page_id, source)

    def record_is_up_to_date(self, page_id: int, source: str, last_date_modified: datetime) -> bool:
        """Return True if the record exists and is at least as recent as last_date_modified."""
        if last_date_modified is None:
            return False
        result = self._repository.get_by_page_id_source_modified_date(page_id, source, last_date_modified)
        return result is not None
