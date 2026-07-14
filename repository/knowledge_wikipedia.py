"""Postgres/pgvector repository for Wikipedia knowledge base."""
from __future__ import annotations

from datetime import datetime

from repository.base import EmbeddingRepository
from repository.knowledge_wikipedia_model import KnowledgeBaseWikipedia

KB_TABLE_NAME = "kb_wikipedia"

class KnowledgeWikipediaRepository(EmbeddingRepository):
    """Repository to write Wikipedia records"""

    id_field_name = "pid"

    def __init__(self):
        super().__init__(KnowledgeBaseWikipedia)

    def get_first_by_title_source(self, title: str, source: str) -> KnowledgeBaseWikipedia | None:
        """Query for based on title"""

        query = (self.model.select().where(
                     (self.model.name == title) &
                     (self.model.source == source)
                 )
                 .get_or_none())

        return query

    def get_by_pid_source(self, pid: int, source: str) -> list[KnowledgeBaseWikipedia]:
        """Query for entire record chunks based on article title/name"""
        return self.get_chunks_by_id_source(pid, source)

    def get_by_pid_source_modified_date(self, pid: int, source: str,
                                        last_date_modified: datetime) -> KnowledgeBaseWikipedia | None:
        """
        Queries the database for the documents with pid and checks if the date is currently
        more recent than the one in the database

        --- Query needs to return a record (it needs to exists) AND make sure current db date is
            greater or equal to current passed date

        Returns True if the record EXISTS AND is UP TO DATE, False otherwise
        """
        return self.get_by_id_source_modified_date(pid, source, last_date_modified)

    def delete_by_pid_source(self, pid: int, source: str) -> None:
        """Delete chunks for a given pid and source"""
        self.delete_by_id_source(pid, source)
