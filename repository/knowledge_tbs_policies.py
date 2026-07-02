"""Postgres/pgvector repository for TBS policies knowledge base."""
from __future__ import annotations

from datetime import datetime
import numpy as np
from peewee import Expression, SQL

from repository.base import BaseRepository
from repository.knowledge_tbs_policies_model import KnowledgeBaseTBSPolicies


class KnowledgeTBSPoliciesRepository(BaseRepository):
    """Repository to read/write TBS policy records."""

    def __init__(self):
        super().__init__(KnowledgeBaseTBSPolicies)

    def search_by_embedding(
        self,
        embedding: list[float],
        limit: int = 100,
        probes: int = 100,
    ) -> list[dict]:
        """Search for similar embeddings using pgvector cosine distance."""
        embedding_vector = embedding[0] if isinstance(embedding[0], (list, tuple, np.ndarray)) else embedding
        db = self.model._meta.database  # pylint: disable=protected-access

        def cosine_distance(column, emb):
            if hasattr(emb, 'tolist'):
                emb = emb.tolist()
            return Expression(column, '<=>', SQL("%s::vector", [emb]))

        with db.atomic():
            db.execute_sql(f"SET LOCAL ivfflat.probes = {int(probes)}")
            query = (self.model.select(
                        self.model.name,
                        self.model.content,
                        self.model.chunk_index,
                        (1 - cosine_distance(self.model.embedding, embedding_vector)).alias('similarity')
                    )
                    .order_by(cosine_distance(self.model.embedding, embedding_vector))
                    .limit(limit))
            results = list(query.dicts())

        return results

    def get_by_page_id_source(self, page_id: int, source: str) -> list[KnowledgeBaseTBSPolicies]:
        """Get all chunks for a given page_id and source."""
        query = (self.model.select().where(
                     (self.model.page_id == page_id) &
                     (self.model.source == source)
                 ).order_by(self.model.chunk_index))
        return list(query)

    def get_by_page_id_source_modified_date(self, page_id: int, source: str,
                                            last_date_modified: datetime) -> KnowledgeBaseTBSPolicies | None:
        """Get a record by page_id and source if it was modified after last_date_modified."""
        query = (self.model.select().where(
                     (self.model.page_id == page_id) &
                     (self.model.source == source) &
                     (self.model.last_modified_date >= last_date_modified)
                 )
                 .get_or_none())
        return query

    def delete_by_page_id_source(self, page_id: int, source: str) -> None:
        """Delete all chunks for a given page_id and source."""
        query = (self.model.delete().where(
                     (self.model.page_id == page_id) &
                     (self.model.source == source)
                 ))
        query.execute()
