"""Response models for the public API."""
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class RunHistoryResponse(BaseModel):
    """A single row from run_history."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: int | None
    service_name: str
    status: str
    metadata: Any | None
    timestamp: datetime


class RunMetricResponse(BaseModel):
    """A single row from run_metrics."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    run_history_id: int
    metadata: Any | None
    timestamp: datetime