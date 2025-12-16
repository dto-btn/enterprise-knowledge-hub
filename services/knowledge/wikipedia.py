from dataclasses import dataclass
from services.knowledge.base import KnowledgeService


@dataclass
class WikipediaKnowedgeService(KnowledgeService):
    """Knowledge service for Wikipedia ingestion."""

    def __init__(self, queue_service, logger):
        super().__init__(queue_service=queue_service, logger=logger, service_name="wikipedia")

    def run(self) -> None:
        """Run the Wikipedia ingestion process."""
        self.logger.info("Starting Wikipedia ingestion process.")
        # Implementation of the Wikipedia ingestion logic
        self.logger.info("Wikipedia ingestion process completed.")
