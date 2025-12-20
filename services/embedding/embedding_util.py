"""
Utility class
"""
import os
from typing import Dict, Iterable, Any
from transformers import PreTrainedTokenizerBase, AutoTokenizer, AutoModel
import torch

class EmbeddingUtil:
    """
    Provides utility functions for embedding
    """
    @staticmethod
    def chunk_text_by_tokens(text: str,
                             tokenizer: PreTrainedTokenizerBase, max_tokens: int = 512, overlap_tokens: int = 64):
        """
        This module provides functions for calculating basic arithmetic operations.
        """
        tokens = tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        start = 0

        while start < len(tokens):
            end = start + max_tokens
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            if end >= len(tokens):
                break

            # move forward but keep an overlap
            start = end - overlap_tokens

        return chunks

    #make this abstract?  the yield is wiki specific
    @staticmethod
    def article_to_chunks(article: Dict, tokenizer, max_tokens=512, overlap_tokens=64):
        """
        Chunks articles
        """
        chunks = EmbeddingUtil.chunk_text_by_tokens(article.content, tokenizer, max_tokens, overlap_tokens)
        for i, chunk_text in enumerate(chunks):
            yield {
                "id": f'{article.pid}-{i}',
                "content": chunk_text,
                "metadata": {
                    "article_id": article.pid,
                    "title": article.title,
                    "chunk_index": i,
                },
            }

    @staticmethod
    def batched (iterable: Iterable[Dict], batch_size: int):
        """
        Write meaningful stuff
        """
        batch = []
        for item in iterable:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
