from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from services.queue.queue_service import QueueService


@dataclass
class KnowledgeService(ABC):
    """Abstract base class for knowledge services."""
    queue_service: QueueService
    logger: logging.Logger
    service_name: str

    @abstractmethod
    def run(self) -> None:
        """Run the knowledge ingestion process."""
        raise NotImplementedError

    def ingest(self) -> None:
        """Ingest data into the knowledge base."""
        self.logger.info("Ingesting data into the knowledge base.")
        ingest_queue = self.queue_service.get(self.service_name + "ingest_queue")