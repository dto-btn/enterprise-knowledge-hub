"""Pydantic response models for the knowledge query API."""
from pydantic import BaseModel


class SearchResult(BaseModel):
    """A single chunk returned by a semantic search."""
    name: str
    content: str
    chunk_index: int
    similarity: float
    id: int | None = None
    source: str | None = None
    language: str | None = None
    url: str | None = None


class SearchResponse(BaseModel):
    """Full response envelope for a semantic search request."""
    query: str
    source: str
    model: str
    total: int
    results: list[SearchResult]
