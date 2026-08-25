"""Public JSON API router aggregation."""
from fastapi import APIRouter

from router.api.run_history import router as run_history_router
from router.api.run_metrics import router as run_metrics_router

router = APIRouter()
router.include_router(run_history_router)
router.include_router(run_metrics_router)
