"""Service layer for TBS policy knowledge items."""
from datetime import datetime
import logging
from dataclasses import dataclass

from provider.embedding.qwen3.embedder_factory import get_embedder
from repository.knowledge_tbs_policies_model import KnowledgeBaseTBSPolicies
from repository.knowledge_tbs_policies import KnowledgeTBSPoliciesRepository


@dataclass
class TBSPolicyItemService:
    """Service to manage TBS policy embeddings."""

    logger: logging.Logger
    _repository: KnowledgeTBSPoliciesRepository

    def __init__(self, logger):
        self._logger = logger
        self._repository = KnowledgeTBSPoliciesRepository()

    @property
    def embedder(self):
        """Get embedder (lazy GPU init on first call)."""
        return get_embedder()

    def search(self, query: str, limit: int = 10) -> list:
        """Search TBS policies by query embedding (asymmetric retrieval)."""
        query_embedding = self.embedder.embed(query, is_query=True)
        return self._repository.search_by_embedding(query_embedding, limit)

    def insert(self, row: dict) -> KnowledgeBaseTBSPolicies:
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

    def delete_by_page_id_source(self, page_id: int, source: str) -> None:
        """Delete all chunks for a page_id and source."""
        self._repository.delete_by_page_id_source(page_id, source)

    def record_is_up_to_date(self, page_id: int, source: str, last_date_modified: datetime) -> bool:
        """Return True if the record exists and is at least as recent as last_date_modified."""
        if last_date_modified is None:
            return False
        result = self._repository.get_by_page_id_source_modified_date(page_id, source, last_date_modified)
        return result is not None
