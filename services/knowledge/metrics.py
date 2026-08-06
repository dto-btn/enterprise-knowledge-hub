"""Progress metric helpers shared by knowledge services."""

import os
import time
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class StageProgressState:
    """Per-stage state used for progress updates"""

    last_count: float
    last_update: float


@dataclass
class ProgressMetricsTracker:
    """Build and throttle progress metadata updates for run history logs"""

    enabled: bool = False
    update_every_n_items: int = 500
    update_every_seconds: float = 2.0
    _state: dict[str, StageProgressState] = field(default_factory=dict)

    def __init__(self) -> None:
        """Initialize tracker, defaulting to environment-derived config."""
        self.from_env()
        self._state = {}

    def from_env(self) -> None:
        """Read tracker config from environment variables."""
        self.enabled = os.getenv("SVC_KB_PROGRESS_METRICS_ENABLED", "false").lower() in ("1", "true", "yes")
        self.update_every_n_items = int(os.getenv("SVC_KB_PROGRESS_UPDATE_EVERY_N_ITEMS", "500"))
        self.update_every_seconds = float(os.getenv("SVC_KB_PROGRESS_UPDATE_EVERY_SECONDS", "2.0"))

    def start_stage(self, stage: str, stage_start: float, total: int | None = None) -> dict[str, object] | None:
        """Initialize stage tracking and optionally return initial metadata"""
        self._state[stage] = StageProgressState(last_count=0.0, last_update=stage_start)
        if not self.enabled:
            return None

        return self.build_progress_metadata(
            stage=stage,
            status="running",
            completed=0,
            total=total,
            stage_start=stage_start,
            now_perf=stage_start,
        )

    def build_progress_metadata(self, stage: str, status: str, completed: int, total: int | None, stage_start: float,
                                now_perf: float | None = None) -> dict[str, object]:
        """Build progress metadata payload for a running stage"""
        if now_perf is None:
            now_perf = time.perf_counter()

        elapsed_seconds = max(0.0, now_perf - stage_start)
        throughput = completed / elapsed_seconds if elapsed_seconds > 0 else 0.0

        return {
            "stage": stage,
            "status": status,
            "completed": completed,
            "total": total,
            "throughput": throughput,
            "elapsed_seconds": elapsed_seconds,
            "updated_at": datetime.now().isoformat(),
        }

    def maybe_progress_metadata(self, stage: str, completed: int, stage_start: float, total: int | None = None, 
                                stage_status: str = "running", force: bool = False) -> dict[str, object] | None:
        """Return progress metadata for a stage, if an update is ready"""
        if not self.enabled:
            return None

        now_perf = time.perf_counter()
        state = self._state.setdefault(
            stage,
            StageProgressState(last_count=0.0, last_update=stage_start),
        )
        count_delta = completed - int(state.last_count)
        time_delta = now_perf - state.last_update

        should_update = force or (
            count_delta >= self.update_every_n_items
            or time_delta >= self.update_every_seconds
        )
        if not should_update:
            return None

        metadata = self.build_progress_metadata(
            stage=stage,
            status=stage_status,
            completed=completed,
            total=total,
            stage_start=stage_start,
            now_perf=now_perf,
        )
        state.last_count = float(completed)
        state.last_update = now_perf
        return metadata
