import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QueueProvider(ABC):
    """Abstract base class for queue configurations"""
    url: str

    def __post_init__(self) -> None:
        """Called after dataclass initialization to run startup."""
        self.startup()

    @abstractmethod
    def startup(self) -> None:
        """Startup actions for the queue provider."""
        raise NotImplementedError

    @abstractmethod
    def get(self, queue_name: str) -> dict[str, object]:
        """Read from the specified queue."""
        raise NotImplementedError

    @abstractmethod
    def put(self, queue_name: str, message: dict[str, object]) -> None:
        """Write to the specified queue."""
        raise NotImplementedError
