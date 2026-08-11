"""Run metrics service class for run_metrics table"""
from datetime import datetime

from repository.run_metrics import RunMetricsRepository
from repository.run_metrics_model import RunMetric


class RunMetricsService:
    """Run metrics service class for run_metrics table"""

    def __init__(self):
        self._repository = RunMetricsRepository()

    def insert_metric(self, run_history_id: int, metadata: dict | None, timestamp: datetime) -> RunMetric:
        """Insert a metric row linked to a run_history row."""
        return self._repository.create(
            run_history=run_history_id,
            metadata=metadata,
            timestamp=timestamp,
        )

    def update_metric(self, run_history_id: int, metadata: dict | None, timestamp: datetime) -> bool:
        """Update the existing metric row for a run_history row in-place."""
        return self._repository.update_for_history(run_history_id, metadata=metadata, timestamp=timestamp)

    def run_metrics_table_rows(self) -> list[RunMetric]:
        """Get all metrics rows."""
        return self._repository.list_all()
