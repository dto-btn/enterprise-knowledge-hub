"""Data models for Change Group PDF items."""
from __future__ import annotations

from datetime import datetime

from pydantic import ConfigDict, field_serializer, field_validator
import numpy as np
import torch

from services.knowledge.models import KnowledgeItem, encode_embeddings, decode_embeddings

Tensor = torch.Tensor


class ChangeGroupItemRaw(KnowledgeItem):
    """Knowledge item representing one PDF file or extracted document chunk."""

    content: str = ""
    source: str = "change-group"
    file_path: str = ""
    last_modified_date: datetime | None = None
    chunk_index: int = 1
    chunk_count: int = 1


class ChangeGroupItemProcessed(ChangeGroupItemRaw):
    """PDF document chunk with computed embeddings."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    embeddings: np.ndarray | Tensor | None = None

    @field_serializer("embeddings")
    def serialize_embeddings(self, value):
        return encode_embeddings(value)

    @field_validator("embeddings", mode="before")
    @classmethod
    def _val_embedding(cls, value):
        if value is None or isinstance(value, (np.ndarray, Tensor)):
            return value
        if isinstance(value, dict):
            return decode_embeddings(value)
        raise TypeError(f"Invalid embedding value type: {type(value)!r}")
