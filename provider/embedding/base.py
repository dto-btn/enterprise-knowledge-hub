"""Base interface for embedding backends."""
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from provider.embedding.tokenizer import ThreadTokenizer

# Default Qwen3 query instruction — used as a fallback when no source-specific
# instruction is provided.  The query endpoint uses the instruction stored in
# kb_source_registry so this constant is only a safety net.
QWEN3_DEFAULT_QUERY_INSTRUCTION = (
    "Instruct: Given a query, retrieve relevant passages that answer the query\nQuery: "
)

# Keep the old name as an alias so existing callers don't break.
QWEN3_QUERY_INSTRUCTION = QWEN3_DEFAULT_QUERY_INSTRUCTION


class EmbeddingBackendProvider(ABC):
    """Contract for embedding providers to implement."""
    model: Any
    model_name: str
    device: str
    max_seq_length: int
    dimensions: int
    tokenizer: ThreadTokenizer | None = None
    batch_size: int = 32

    @abstractmethod
    def embed(
        self,
        text: Any,
        is_query: bool = False,
        instruction: str | None = None,
    ) -> np.ndarray:
        """Generate embeddings for text.

        Args:
            text: The text to embed.
            is_query: If True and no instruction is given, prepend the default
                      query instruction (backward-compat fallback).
            instruction: Task-specific instruction prefix.  When provided it
                         takes priority over is_query and is prepended verbatim.
                         Retrieve from kb_source_registry at query time so the
                         instruction always matches what was used at ingest.
        """
        raise NotImplementedError

    @abstractmethod
    def chunk_text_by_tokens(self, text: str, max_tokens: int = None, overlap_tokens: int = 10) -> list[str]:
        """Split text into chunks based on token count with overlap."""
        raise NotImplementedError

    def is_model_loaded(self) -> bool:
        """Returns true is model is loaded"""
        return getattr(self, "model", None) is not None

    def get_tokenizer(self) -> Any:
        """
        Get tokenizer for current thread
        """
        return self.tokenizer.get()

    def get_batch_size(self) -> int:
        """
        Get batch size for embedding generation, which may be dynamic based on model/device
        """
        return self.batch_size
