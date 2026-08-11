"""Run history API endpoints."""
import logging

from fastapi import APIRouter

from router.api.models import RunHistoryResponse
from services.database.run_history_service import RunHistoryService

router = APIRouter()
logger = logging.getLogger(__name__)
_run_history_service = RunHistoryService(logger)


@router.get("/run-history", response_model=list[RunHistoryResponse])
def list_run_history():
    """Return all run history rows."""
    return _run_history_service.run_history_table_rows()