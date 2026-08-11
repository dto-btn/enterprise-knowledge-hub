"""Run metrics table repository"""
from datetime import datetime

from repository.base import BaseRepository
from repository.run_metrics_model import RunMetric


class RunMetricsRepository(BaseRepository):
    """Repository for run_metrics table."""

    def __init__(self):
        super().__init__(RunMetric)

    def update_for_history(self, run_history_id: int, metadata: dict | None, timestamp: datetime) -> bool:
        """Update the existing metric row for a given run_history_id."""
        rows_updated = (
            RunMetric.update(metadata=metadata, timestamp=timestamp)
            .where(RunMetric.run_history == run_history_id)
            .execute()
        )
        return rows_updated > 0
