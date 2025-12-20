"""Embedding service to handle embedding operations."""
from typing import List
from dataclasses import dataclass
import logging
import numpy as np
from provider.embedding.base import EmbeddingBackendProvider

@dataclass
class EmbeddingService:
    """Service to manage embedding operations."""
    embedding_provider: EmbeddingBackendProvider
    logger: logging.Logger

    def embed(self, text: List[str]) -> np.ndarray:
        """Read messages from the specified queue."""
        return self.embedding_provider.embed(text)
