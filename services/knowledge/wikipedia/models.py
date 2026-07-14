"""Data models for Wikipedia items."""
from datetime import datetime
from enum import StrEnum
from pydantic import ConfigDict, field_serializer, field_validator

import torch
import numpy as np

from services.knowledge.models import KnowledgeItem, encode_embeddings, decode_embeddings

Tensor = torch.Tensor

class Source(StrEnum):
    """Enumeration of knowledge item sources."""
    WIKIPEDIA_EN = "enwiki"
    WIKIPEDIA_FR = "frwiki"
    #MYSSCPLUS = "mysscplus

class WikipediaItemRaw(KnowledgeItem):
    """Knowledge item representing a Wikipedia page."""
    content: str = ""  # Wiki markup content
    last_modified_date: datetime | None = None
    pid: int = 0
    source: Source | None = Source.WIKIPEDIA_EN
    chunk_index: int = 1
    chunk_count: int = 1
    # fields for processing article initially (seeing if we ignore it or not, etc)
    is_namespace_0: bool = True # we only process articles in ns 0
    is_redirect: bool = False
    has_wikilinks: bool = True  # Wikipedia only counts ns-0 pages with >=1 internal wikilink as "articles"

class WikipediaItemProcessed(WikipediaItemRaw):
    """Knowledge item representing a Wikipedia page stored in a database."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    embeddings: np.ndarray | Tensor | None = None

    @field_serializer("embeddings")
    def serialize_embeddings(self, value):
        """custom serializer for embeddings prop"""
        return encode_embeddings(value)

    @field_validator("embeddings", mode="before")
    @classmethod
    def _val_embedding(cls, value):
        if value is None or isinstance(value, (np.ndarray, Tensor)):
            return value
        if isinstance(value, dict):
            return decode_embeddings(value)
        raise TypeError(f"Invalid embedding value type: {type(value)!r}")
