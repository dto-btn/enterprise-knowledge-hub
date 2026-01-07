import logging
import os
from typing import Union

import numpy as np
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from provider.embedding.base import EmbeddingBackendProvider

load_dotenv()

class Qwen3SentenceTransformer(EmbeddingBackendProvider):
    """
    Qwen3 Sentence Transformer embedding provider.
    """
    def __init__(self):
        self.model = SentenceTransformer(
            "Qwen/Qwen3-Embedding-0.6B",
            model_kwargs={"device_map": torch.device("mps") if torch.backends.mps.is_available() else "auto",
                        "dtype": torch.float32 if torch.backends.mps.is_available() else torch.float16},
            tokenizer_kwargs={"padding_side": "left"},
        )
        self.model.max_seq_length = int(os.getenv("WIKIPEDIA_EMBEDDING_MODEL_MAX_LENGTH", "4096"))
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Model loaded on device: %s", self.model.device)
        self.logger.debug("Model max sequence length: %d", self.model.max_seq_length)

    def embed(self, sentences: str, instruction: Union[str, None] = None, dim: int = int(os.getenv("WIKIPEDIA_EMBEDDING_MODEL_MAX_DIM", "1024"))) -> np.ndarray:
        chunks = super().chunk_text_by_tokens(sentences, max_tokens=self.model.max_seq_length)
        self.logger.debug("Split into %d chunks", len(chunks))

        # Encode the string chunks
        embeddings = self.model.encode(
            chunks,
            convert_to_tensor=False,
            show_progress_bar=bool(os.getenv("MODEL_SHOW_PROGRESS", "True").lower() == "true"),
            batch_size=int(os.getenv("WIKIPEDIA_EMBEDDING_MODEL_BATCH_SIZE", "1")),  # Lower batch size for potentially large chunks
            truncate_dim=dim
        )

        # Aggressive cleanup for MPS
        if os.getenv("WIKIPEDIA_EMBEDDING_MODEL_CLEANUP", "False").lower() == "true":
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

        return embeddings