import numpy as np
from abc import ABC, abstractmethod
from typing import List
from typing import Any

class EmbeddingBackendProvider(ABC):
    model: Any
    model_name: str
    device: str
    max_seq_length: int

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """used to create embeddings for a text input"""
        raise NotImplementedError

    def chunk_text_by_tokens(self, text: str, max_tokens: int = None, overlap_tokens: int = 200) -> list[str]:
        """Split text into chunks based on token count with overlap."""
        if max_tokens is None:
            max_tokens = self.model.max_seq_length

        # Tokenize the entire text
        tokens = self.model.tokenizer.encode(text, add_special_tokens=False)

        # If text fits in one chunk, return as-is
        if len(tokens) <= max_tokens:
            return [text]

        chunks = []
        start_idx = 0

        while start_idx < len(tokens):
            # Get chunk of tokens
            end_idx = min(start_idx + max_tokens, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode back to text
            chunk_text = self.model.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)

            # Move forward with overlap
            start_idx += max_tokens - overlap_tokens

        return chunks
