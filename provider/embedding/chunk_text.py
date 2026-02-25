"""Tiktoken tokenizer to chunk text by tokens"""
import tiktoken

def chunk_text_by_tokens(
    text: str,
    max_tokens: int,
    overlap_tokens: int = 10
) -> list[str]:
    """Split text into chunks based on token count with overlap using tiktoken."""

    tt = tiktoken.get_encoding("cl100k_base")
    # Encode entire text
    tokens = tt.encode(text)

    # If text fits in one chunk
    if len(tokens) <= max_tokens:
        return [text]

    chunks = []
    start_idx = 0

    while start_idx < len(tokens):
        end_idx = min(start_idx + max_tokens, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]

        # Decode back to text
        chunk_text = tt.decode(chunk_tokens)
        chunks.append(chunk_text)

        # Move forward with overlap
        start_idx += max_tokens - overlap_tokens

    return chunks
