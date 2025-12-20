"""database service to handle database operations."""
from typing import List, Sequence, Dict, Any, Tuple
from dataclasses import dataclass
import logging
import numpy as np
from provider.database.base import VectorDatabaseProvider

@dataclass
class DatabaseService:
    """Service to manage database operations."""
    database_provider: VectorDatabaseProvider
    logger: logging.Logger

    def bulk_upsert(self, records: Sequence[Tuple[str, str, List[float], Dict[str, Any]]]) -> int:
        """Read messages from the specified queue."""
        return self.database_provider.upsert(records)

    def upsert_batch(self, embeddings: np.ndarray, batches: List[Dict[str, Any]]) -> None:
        """package to upsert"""
        records = []
        for vec, ch in zip(embeddings, batches):
            records.append((
                ch["id"],
                ch["content"],
                vec.tolist(),
                ch.get("metadata", {}),
            ))

        self.bulk_upsert(records)
