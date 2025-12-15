from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from services.queue.queue_service import QueueService


@dataclass
class KnowledgeService(ABC):
    """Abstract base class for knowledge services."""
    queue_service: QueueService
    logger: logging.Logger

    @abstractmethod
    def run(self) -> None:
        """Run the knowledge ingestion process."""
        raise NotImplementedError