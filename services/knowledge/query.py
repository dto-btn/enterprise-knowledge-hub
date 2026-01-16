"""Service layer to query embedding in persistance layer"""
from dataclasses import dataclass
import logging

from services.db.model import DocumentRecord
from services.db.postgrespg import WikipediaPgRepository

@dataclass
class QueryService():
    """Service to query wiki embeddings"""

    logger: logging.Logger

    def __init__(self, repository: WikipediaPgRepository | None = None):
        self._repository = repository or WikipediaPgRepository.from_env()

    def get_article_content_by_title(self, title: str) -> list[DocumentRecord]:
        """Get article contentn based on title"""

        result = self._repository.get_record_content(title)
        return result
