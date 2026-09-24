"""Postgres/pgvector repository for local SharePoint PDF knowledge base."""
from __future__ import annotations

from datetime import datetime

from repository.base import EmbeddingRepository
from repository.knowledge_change_group_model import KnowledgeBaseChangeGroup

KB_TABLE_NAME = "kb_change_group"


class ChangeGroupRepository(EmbeddingRepository):
    """Repository for Change Group PDF records."""

    id_field_name = "file_path"

    def __init__(self):
        super().__init__(KnowledgeBaseChangeGroup)

    def get_first_by_file_source(self, file_path: str, source: str) -> KnowledgeBaseChangeGroup | None:
        query = (
            self.model.select()
            .where((self.model.file_path == file_path) & (self.model.source == source))
            .get_or_none()
        )
        return query

    def get_by_file_source(self, file_path: str, source: str) -> list[KnowledgeBaseChangeGroup]:
        return self.get_chunks_by_id_source(file_path, source)

    def get_by_file_source_modified_date(self, file_path: str, source: str, last_date_modified: datetime) -> KnowledgeBaseChangeGroup | None:
        return self.get_by_id_source_modified_date(file_path, source, last_date_modified)

    def delete_by_file_source(self, file_path: str, source: str) -> None:
        self.delete_by_id_source(file_path, source)
