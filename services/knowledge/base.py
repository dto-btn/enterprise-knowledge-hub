from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import logging
from services.queue.queue_service import QueueService


@dataclass
class KnowledgeService(ABC):
    """Abstract base class for knowledge services."""
    queue_service: QueueService
    logger: logging.Logger
    service_name: str

    def run(self) -> None:
        """Run the knowledge ingestion process."""
        self.logger.info("Running knowledge ingestion for %s", self.service_name)
        self.ingest()

    @abstractmethod
    def read(self) -> Iterator[dict[str, object]]:
        """Read data from a source that can be anything and will pass the message to the ingest queue."""
        raise NotImplementedError("Subclasses must implement the read method.")

    def ingest(self) -> None:
        """Ingest data into the knowledge base."""
        self.logger.info("Ingesting data into the knowledge base. (%s)", self.service_name)
        try:
            for item in self.read():
                self.queue_service.write(self.service_name + ".ingest", item)
        except Exception as e:
            self.logger.exception("Error during ingestion for %s: %s", self.service_name, e)
        finally:
            self.logger.info("Done processing with ingestion for %s", self.service_name)
