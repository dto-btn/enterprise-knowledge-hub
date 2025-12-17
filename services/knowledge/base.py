from abc import ABC
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

    def ingest(self) -> None:
        """Ingest data into the knowledge base."""
        self.logger.info("Ingesting data into the knowledge base. (%s)", self.service_name)
        items = [{"data": "sample data 1"}, {"data": "sample data 2"}]
        try:
            for item in items:
                self.queue_service.write(self.service_name + ".ingest", item)
                self.logger.debug("Ingested message: %s", item)
        except Exception as e:
            self.logger.exception("Error during ingestion for %s: %s", self.service_name, e)
        finally:
            self.logger.info("Done processing with ingestion for %s", self.service_name)
