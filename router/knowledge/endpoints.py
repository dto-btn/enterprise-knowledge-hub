"""
Main endpoints for knowledge management (creation/delete/update).
"""
import logging
import os
from typing import Literal
from dotenv import load_dotenv
from fastapi import APIRouter, Query
from fastapi import BackgroundTasks

from provider.queue.rabbitmq import RabbitMQProvider
from provider.database.pgvector_db import PgVectorProvider
from provider.embedding.torch_backend import TorchEmbeddingBackendProvider
from router.knowledge.run_state import RunState
from services.knowledge.wikipedia import WikipediaKnowedgeService
from services.queue.queue_service import QueueService
from services.embedding.embedding_service import EmbeddingService
from services.database.database_service import DatabaseService

load_dotenv()

logger = logging.getLogger(__name__)

router = APIRouter()

KNOWLEDGE_BASE = "/knowledge"
DSN = "postgres://admin:admin@localhost:5432/rag"

# initialize the queue service here
_queue_service = QueueService(queue_provider=RabbitMQProvider(url=os.getenv("RABBITMQ_URL"),
                                                              logger=logger), logger=logger)
_embedding_service = EmbeddingService(embedding_provider=TorchEmbeddingBackendProvider(model_name="Qwen3-Embedding-0.6B-Q8_0",
                                                                               logger=logger, device="cpu"),
                                      logger=logger)
_database_service = DatabaseService(database_provider=PgVectorProvider(dsn=DSN, logger=logger), logger=logger)
_wikipedia_service = WikipediaKnowedgeService(queue_service=_queue_service, embedding_service=_embedding_service,
                                              database_service=_database_service, logger=logger)
_wikipedia_state = RunState()


def _run_wikipedia_task():
    """Wrapper that manages the running state flag."""
    try:
        _wikipedia_service.run()
    finally:
        _wikipedia_state.stop()

def _run_wikipedia_vector():
    print('running wiki function')
    _wikipedia_service.process()

@router.get("/wikipedia/run")
def wikipedia_run(background_tasks: BackgroundTasks):
    """Endpoint to trigger Wikipedia full run"""
    if not _wikipedia_state.try_start():
        return {
            "message": "Wikipedia run already in progress.",
            "details": f"Follow progress here {KNOWLEDGE_BASE}/wikipedia/status"
        }

    background_tasks.add_task(_run_wikipedia_task)
    return {
        "message": "Wikipedia run started.",
        "details": f"Follow progress here {KNOWLEDGE_BASE}/wikipedia/status"
    }

@router.get("/wikipedia/vectorize")
def wikipedia_vectorize():
    """Endpoint to trigger Wikipedia vectorize test"""
    print('running vecotr===========')
    _run_wikipedia_vector()



@router.get("/wikipedia/status")
def wikipedia_stats(
    rate_window: Literal[5, 10] = Query(
        default=5,
        description="Time window in seconds for rate calculations (5 or 10)"
    )
):
    """Return in-memory ingestion stats plus live queue depths."""
    # Update the rate window before getting stats
    _wikipedia_service.stats.set_rate_window(rate_window)

    return {
        "running": _wikipedia_state.is_running(),
        "stats": _wikipedia_service.stats.get_stats()
    }
