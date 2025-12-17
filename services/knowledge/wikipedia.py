from dataclasses import dataclass
from services.knowledge.base import KnowledgeService


@dataclass
class WikipediaKnowedgeService(KnowledgeService):
    """Knowledge service for Wikipedia ingestion."""

    def __init__(self, queue_service, logger):
        super().__init__(queue_service=queue_service, logger=logger, service_name="wikipedia")
