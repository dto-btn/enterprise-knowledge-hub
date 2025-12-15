import logging
import os
from dotenv import load_dotenv
from fastapi import APIRouter
from fastapi import BackgroundTasks

from provider.queue.rabbitmq import RabbitMQProvider
from services.queue.queue_service import QueueService

load_dotenv()

logger = logging.getLogger(__name__)

router = APIRouter()

KNOWLEDGE_BASE = "/knowledge"

# initialize the queue service here
queue_service = QueueService(RabbitMQProvider(url=os.getenv("RABBITMQ_URL")))
wikipedia_service = KnowledgeService(WikipediaKnowedgeService(queue_service=queue_service, logger=logger))

@router.get("/wikipedia/run")
def wikipedia_run(background_tasks: BackgroundTasks):
    """Endpoint to trigger Wikipedia full run"""
    logger.info("Starting Wikipedia ingestion run")
    #background_tasks.add_task(_wikipedia_service.run)
    return {
        "message": "Wikipedia run started.",
        "details": f"Follow progress here {KNOWLEDGE_BASE}/wikipedia/status"
        }

@router.get("/wikipedia/status")
def wikipedia_stats():
    """Return in-memory ingestion stats plus live queue depths."""
    return {"stats": "empty #todo"}
