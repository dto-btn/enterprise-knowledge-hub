"""Run metrics API endpoints."""
from fastapi import APIRouter

from router.api.models import RunMetricResponse
from services.database.run_metrics_service import RunMetricsService

router = APIRouter()
run_metrics_service = RunMetricsService()


@router.get("/run-metrics", response_model=list[RunMetricResponse])
def list_run_metrics():
    """Return all run metrics rows"""
    return run_metrics_service.run_metrics_table_rows()
