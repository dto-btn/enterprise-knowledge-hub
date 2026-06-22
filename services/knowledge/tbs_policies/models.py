"""Data models for TBS Policy items."""
from datetime import datetime
from typing import Any, Literal
import base64

from pydantic import ConfigDict, field_serializer, field_validator
import numpy as np
import torch

from services.knowledge.models import KnowledgeItem

Tensor = torch.Tensor


def _encode_embeddings(embedding: np.ndarray | Tensor | None) -> dict[str, Any] | None:
    """Convert embeddings into a JSON-serializable dict."""
    if embedding is None:
        return None

    if isinstance(embedding, Tensor):
        embedding_np = embedding.detach().to("cpu").numpy()
        kind: Literal["torch"] = "torch"
    elif isinstance(embedding, np.ndarray):
        embedding_np = embedding
        kind = "numpy"
    else:
        raise TypeError(f"Unsupported embeddings type: {type(embedding)!r}")

    embedding_np = np.ascontiguousarray(embedding_np)
    raw = embedding_np.tobytes(order="C")
    data_b64 = base64.b64encode(raw).decode("ascii")

    return {
        "kind": kind,
        "dtype": str(embedding_np.dtype),
        "shape": list(embedding_np.shape),
        "data_b64": data_b64,
    }


def _decode_embeddings(payload: dict[str, Any] | None) -> np.ndarray | Tensor | None:
    """Reverse of _encode_embeddings."""
    if payload is None:
        return None

    kind = payload["kind"]
    dtype = np.dtype(payload["dtype"])
    shape = tuple(payload["shape"])
    raw = base64.b64decode(payload["data_b64"].encode("ascii"))

    arr = np.frombuffer(raw, dtype=dtype).reshape(shape)

    if kind == "numpy":
        return arr
    if kind == "torch":
        return torch.from_numpy(arr)

    raise ValueError(f"Unknown embeddings kind: {kind!r}")


class TBSPolicyItemRaw(KnowledgeItem):
    """Knowledge item representing a single TBS policy page."""
    content: str = ""
    page_id: int = 0
    source: str = "tbs-policies"
    last_modified_date: datetime | None = None
    chunk_index: int = 1
    chunk_count: int = 1


class TBSPolicyItemProcessed(TBSPolicyItemRaw):
    """TBS policy item with computed embeddings."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    embeddings: np.ndarray | Tensor | None = None

    @field_serializer("embeddings")
    def serialize_embeddings(self, value):
        """Custom serializer for embeddings prop."""
        return _encode_embeddings(value)

    @field_validator("embeddings", mode="before")
    @classmethod
    def _val_embedding(cls, value):
        if value is None or isinstance(value, (np.ndarray, Tensor)):
            return value
        if isinstance(value, dict):
            return _decode_embeddings(value)
        raise TypeError(f"Invalid embedding value type: {type(value)!r}")
