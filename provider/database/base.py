"""
Database Provider base
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence, Iterable
import numpy as np

@dataclass(frozen=True)
class VectorRecord:
    """One row to store in the vector index."""
    id: str
    embedding: Sequence[float]
    text: str
    metadata: dict
    
@dataclass
class VectorDatabaseProvider(ABC):
    """
    Database Provider base
    """
    def upsert(self, records: Iterable[VectorRecord], *, batch_size: int = 256) -> int:
        """Insert for database"""
        raise NotImplementedError