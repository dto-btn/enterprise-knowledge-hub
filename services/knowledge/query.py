"""Service layer to query embedding in persistance layer"""
from dataclasses import dataclass
import logging
from services.db.postgrespg import WikipediaPgRepository
from provider.embedding.qwen3.embedder_factory import get_embedder

@dataclass
class QueryService():
    """Service to query wiki embeddings"""

    logger: logging.Logger

    def __init__(self, repository: WikipediaPgRepository | None = None):
        self._repository = repository or WikipediaPgRepository.from_env()

    @property
    def embedder(self):
        """Get embedder"""
        return get_embedder()

    def test(self):
        """Test method to verify service layer functionality."""
        print("servicelayer ok")
        result = self._repository.get_record()
        print("result")
        print(result)

    def search(self, query: str, limit: int =10):
        """Search Wikipedia articles by query embedding."""
        query_embedding = self.embedder.embed(query)
        results = self._repository.search_by_embedding(query_embedding, limit)
        return results
