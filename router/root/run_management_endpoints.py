"""
Endpoints for knowledge run management (start / stop).

To add a new knowledge source, instantiate it and insert one entry into _REGISTRY.
No other changes are required.
"""
import logging
import os
from collections.abc import Callable

from dotenv import load_dotenv
from fastapi import APIRouter, BackgroundTasks, HTTPException

from provider.queue.rabbitmq import RabbitMQProvider
from router.root.run_state import RunState
from services.database.run_history_service import RunHistoryService
from services.knowledge.base import KnowledgeService
from services.knowledge.tbs_policies.tbs_policies import TBSPoliciesKnowledgeService
from services.knowledge.wikipedia.wikipedia import WikipediaKnowledgeService
from services.queue.queue_service import QueueService

load_dotenv()

logger = logging.getLogger(__name__)
router = APIRouter()

KNOWLEDGE_BASE = "/knowledge"

# ── Shared infrastructure ─────────────────────────────────────────────────────

_queue_service = QueueService(
    queue_provider=RabbitMQProvider(url=os.getenv("RABBITMQ_URL"), logger=logger),
    logger=logger,
)
_run_history_service = RunHistoryService(logger)

# ── Service registry ──────────────────────────────────────────────────────────
# Add a new knowledge source here — one entry, nothing else to change.

_REGISTRY: dict[str, tuple[KnowledgeService, RunState]] = {
    "wikipedia": (
        WikipediaKnowledgeService(
            queue_service=_queue_service,
            logger=logger,
            run_history_service=_run_history_service,
        ),
        RunState(),
    ),
    "tbs-policies": (
        TBSPoliciesKnowledgeService(
            queue_service=_queue_service,
            logger=logger,
            run_history_service=_run_history_service,
        ),
        RunState(),
    ),
}


def _get_or_404(slug: str) -> tuple[KnowledgeService, RunState]:
    entry = _REGISTRY.get(slug)
    if entry is None:
        known = ", ".join(sorted(_REGISTRY))
        raise HTTPException(status_code=404, detail=f"Unknown knowledge source '{slug}'. Known: {known}")
    return entry


def _make_run_task(service: KnowledgeService, state: RunState) -> Callable:
    """Return a background-task callable that resets state when the run finishes."""
    def _task(run_id: int | None = None) -> None:
        try:
            service.run(run_id)
        finally:
            state.stop()
    return _task


# ── Generic run / stop endpoints ──────────────────────────────────────────────

@router.get("/{slug}/run")
def knowledge_run(slug: str, background_tasks: BackgroundTasks, run_id: int | None = None):
    """
    Start a knowledge ingestion run for the given source (e.g. wikipedia, tbs-policies).
    An optional run_id overrides the service's computed run ID.
    """
    service, state = _get_or_404(slug)
    if not state.try_start():
        return {
            "message": f"'{slug}' run already in progress.",
            "details": "Follow progress at frontend/status",
        }
    background_tasks.add_task(_make_run_task(service, state), run_id)
    return {
        "message": f"'{slug}' run started.",
        "details": "Follow progress at frontend/status",
    }


@router.get("/{slug}/stop")
async def knowledge_stop(slug: str):
    """Stop the currently running ingestion for the given source."""
    service, state = _get_or_404(slug)
    if not state.is_running():
        return {"message": f"No '{slug}' run is currently in progress."}
    run_id = service.request_stop()
    state.stop()
    return {"message": f"Stop requested for '{slug}'.", "run_id": run_id}
