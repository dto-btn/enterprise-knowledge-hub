"""Base data model for knowledge items."""
from enum import StrEnum
from typing import Any, Literal
import base64

from pydantic import BaseModel
import numpy as np
import torch

Tensor = torch.Tensor


def encode_embeddings(embedding: np.ndarray | Tensor | None) -> dict[str, Any] | None:
    """Convert embeddings into a JSON-serializable dict."""
    if embedding is None:
        return None

    if isinstance(embedding, Tensor):
        # Move to CPU + detach, then convert to numpy for consistent encoding
        embedding_np = embedding.detach().to("cpu").numpy()
        kind: Literal["torch"] = "torch"
    elif isinstance(embedding, np.ndarray):
        embedding_np = embedding
        kind = "numpy"
    else:
        raise TypeError(f"Unsupported embeddings type: {type(embedding)!r}")

    # Ensure contiguous so .tobytes() matches shape/dtype correctly
    embedding_np = np.ascontiguousarray(embedding_np)
    raw = embedding_np.tobytes(order="C")
    data_b64 = base64.b64encode(raw).decode("ascii")

    return {
        "kind": kind,  # tells you what to rebuild as (numpy vs torch)
        "dtype": str(embedding_np.dtype),  # e.g. "float32"
        "shape": list(embedding_np.shape),
        "data_b64": data_b64,
    }


def decode_embeddings(payload: dict[str, Any] | None) -> np.ndarray | Tensor | None:
    """Reverse of encode_embeddings."""
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


class KnowledgeItem(BaseModel):
    """Base class for knowledge items that will be pushed to the queue."""

    name: str = ""

class RunStatus(StrEnum):
    """Enumeration of run status (and cronjob status) for the run_history table."""
    RUN_STARTED = "Run Started"
    INGESTION_STARTED = "Ingestion Started"
    INGESTION_COMPLETED = "Ingestion Completed"
    PROCESSING_STARTED = "Processing Started"
    PROCESSING_COMPLETED = "Processing Completed"
    STORING_STARTED = "Storing Started"
    STORING_COMPLETED = "Storing Completed"
    RUN_ENDED = "Run Completed"
    DUMP_LINK_UPDATED = "New Dump Link Detected and Downloaded"
    BATCH_AVERAGE_TIME = "Processing Batch Average Time"
    RUN_STOPPED = "Run Manually Stopped"
    RUN_CONTINUED = "Run Continued"
