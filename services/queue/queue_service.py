from dataclasses import dataclass
import logging
from provider.queue.base import QueueProvider

@dataclass
class QueueService:
    """Service to manage queue operations."""
    queue_provider: QueueProvider
    logger: logging.Logger

    def read(self, queue_name: str) -> dict[str, object]:
        """Read the status of the specified queue."""
        return self.queue_provider.read(queue_name)

    def write(self, queue_name: str, message: dict[str, object]) -> None:
        """Write a message to the specified queue."""
        return self.queue_provider.write(queue_name, message)