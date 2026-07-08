"""
Wikipedia article service — retrieval and persistence operations for kb_wikipedia.
"""
from datetime import datetime
import logging
from dataclasses import dataclass

from repository.knowledge_wikipedia_model import KnowledgeBaseWikipedia
from repository.knowledge_wikipedia import KnowledgeWikipediaRepository


@dataclass
class WikipediaArticleService():
    """Service to retrieve and manage Wikipedia knowledge items."""

    logger: logging.Logger
    _repository: KnowledgeWikipediaRepository

    def __init__(self, logger):
        self._logger = logger
        self._repository = KnowledgeWikipediaRepository()

    def get_article_content_by_title(self, title: str, source: str) -> str:
        """Get article content based on title and source"""
        self._logger.info("Retrieving article chunks for title=%r source=%r", title, source)

        # get pid of article then get all by pid?  feels like this can be combined.  revisit after refactoring
        article = self._repository.get_first_by_title_source(title, source)

        if article is None:
            return None

        chunks = self._repository.get_by_pid_source(article.pid, source)
        combined_content = "\n\n".join(chunk.content for chunk in chunks)
        return (article.name, combined_content)

    def delete_by_pid_source(self, pid: int, source: str) -> None:
        """Delete all records by PID and source"""
        self._repository.delete_by_pid_source(pid, source)

    def insert(self, row: KnowledgeBaseWikipedia) -> KnowledgeBaseWikipedia:
        """Insert a record"""
        return self._repository.create(
                            pid=row['pid'],
                            chunk_index=row['chunk_index'],
                            name=row['name'],
                            content=row['content'],
                            last_modified_date=row['last_modified_date'],
                            embedding=row['embedding'],
                            source=row['source']
                        )

    def record_is_up_to_date(self, pid: int, source: str, last_date_modified: datetime) -> bool:
        """
        Queries the database for the documents with pid and checks if the date is currently
        more recent than the one in the database

        --- Query needs to return a record (it needs to exists) AND make sure current db date is
            greater or equal to current passed date

        Returns True if the record EXISTS AND is UP TO DATE, False otherwise

        """
        result = self._repository.get_by_pid_source_modified_date(pid, source, last_date_modified)

        if result:
            return True
        return False
