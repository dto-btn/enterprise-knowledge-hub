"""database service to handle database operations."""
from typing import List, Iterable, Sequence, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass
import logging
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
        records = []
        for vec, ch in zip(embeddings, batches):
            records.append((
                ch["id"],
                ch["content"],
                vec.tolist(),
                ch.get("metadata", {}),
            ))
            
        self.bulk_upsert(records)
        