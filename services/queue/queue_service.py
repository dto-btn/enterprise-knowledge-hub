from dataclasses import dataclass
import logging
from provider.queue.base import QueueProvider

@dataclass
class QueueService:
    """Service to manage queue operations."""
    queue_provider: QueueProvider
    logger: logging.Logger

    def get(self, queue_name: str) -> dict[str, object]:
        """Read the status of the specified queue."""
        self.queue_provider.get(queue_name)

    def put(self, queue_name: str, message: dict[str, object]) -> None:
        """Write a message to the specified queue."""
        self.queue_provider.put(queue_name, message)