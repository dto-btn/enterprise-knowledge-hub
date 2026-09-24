"""Service layer for local SharePoint PDF knowledge items."""
from datetime import datetime
import logging
from dataclasses import dataclass

from repository.knowledge_change_group_model import KnowledgeBaseChangeGroup
from repository.knowledge_change_group import ChangeGroupRepository


@dataclass
class ChangeGroupService:
    """Service to manage Change Group PDF embeddings."""

    logger: logging.Logger
    _repository: ChangeGroupRepository

    def __init__(self, logger):
        self._logger = logger
        self._repository = ChangeGroupRepository()

    def insert(self, row: dict) -> KnowledgeBaseChangeGroup:
        return self._repository.create(
            file_path=row['file_path'],
            chunk_index=row['chunk_index'],
            name=row['name'],
            content=row['content'],
            last_modified_date=row['last_modified_date'],
            embedding=row['embedding'],
            source=row['source'],
        )

    def search_by_embedding(self, embedding: list[float], limit: int = 100) -> list[dict]:
        return self._repository.search_by_embedding(embedding, limit=limit)

    def delete_by_file_source(self, file_path: str, source: str) -> None:
        self._repository.delete_by_file_source(file_path, source)

    def record_is_up_to_date(self, file_path: str, source: str, last_date_modified: datetime) -> bool:
        if last_date_modified is None:
            return False
        result = self._repository.get_by_file_source_modified_date(file_path, source, last_date_modified)
        return result is not None
