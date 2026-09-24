"""Persistence model for the Change Group knowledge base."""
from __future__ import annotations

import os

from dotenv import load_dotenv
from peewee import SQL, IntegerField, TextField
import numpy as np
from torch import Tensor

from repository.base_model import BaseEmbeddingModel, VectorField
from services.knowledge.change_group.models import ChangeGroupItemProcessed

load_dotenv()

KB_TABLE_NAME = "kb_change_group"


class KnowledgeBaseChangeGroup(BaseEmbeddingModel):
    """kb_change_group model"""

    file_path: str = TextField()
    chunk_index: int = IntegerField()
    name: str = TextField()
    content: str = TextField()
    embedding: list[float] = VectorField(dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", str(512))))
    source: str | None = TextField(null=True)

    class Meta:  # pylint: disable=too-few-public-methods
        db_table = KB_TABLE_NAME
        constraints = [
            SQL(
                'CONSTRAINT change_group_file_path_source_chunk_index_key '
                'UNIQUE (file_path, source, chunk_index)'
            )
        ]
        indexes = [
            SQL('CREATE INDEX IF NOT EXISTS change_group_embedding_index '
                'ON kb_change_group USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);'),
            SQL('CREATE INDEX IF NOT EXISTS change_group_file_path_idx ON kb_change_group (file_path);'),
            SQL('CREATE INDEX IF NOT EXISTS change_group_source_idx ON kb_change_group (source);'),
        ]

    @classmethod
    def from_item(cls, item: ChangeGroupItemProcessed) -> KnowledgeBaseChangeGroup:
        embedding = cls._to_floats(item.embeddings)
        return cls(
            file_path=item.file_path,
            chunk_index=item.chunk_index,
            name=item.name,
            content=item.content,
            last_modified_date=item.last_modified_date,
            embedding=embedding,
            source=item.source,
        )

    def as_mapping(self) -> dict[str, object]:
        return {
            "file_path": self.file_path,
            "chunk_index": self.chunk_index,
            "name": self.name,
            "content": self.content,
            "last_modified_date": self.last_modified_date,
            "embedding": self.embedding,
            "source": self.source,
        }

    @staticmethod
    def _to_floats(raw_embedding: object) -> list[float]:
        if raw_embedding is None:
            raise ValueError("Embeddings are required for storage.")
        if isinstance(raw_embedding, Tensor):
            return raw_embedding.detach().cpu().flatten().tolist()
        if isinstance(raw_embedding, np.ndarray):
            return raw_embedding.flatten().tolist()
        if isinstance(raw_embedding, (list, tuple)):
            return [float(x) for x in raw_embedding]
        raise TypeError(f"Unsupported embedding type: {type(raw_embedding)!r}")
