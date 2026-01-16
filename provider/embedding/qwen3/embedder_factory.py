"""Factory to choose embedder"""
import os
from functools import lru_cache

@lru_cache(maxsize=1)
def get_embedder():
    """get embedder type based on env os"""
    if os.getenv("WIKIPEDIA_EMBEDDING_MODEL_BACKEND", "LLAMA").upper() == "SENTENCE_TRANSFORMER":
        from provider.embedding.qwen3.sentence_transformer import Qwen3SentenceTransformer
        return Qwen3SentenceTransformer()

    from provider.embedding.qwen3.llama_embed import Qwen3LlamaCpp
    return Qwen3LlamaCpp()
