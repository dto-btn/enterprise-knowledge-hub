from dataclasses import dataclass

from services.knowledge.base import KnowledgeService


@dataclass
class TBSPoliciesKnowledgeService(KnowledgeService):
    """Knowledge service for TBS policies."""
    def __init__(self, queue_service, logger, run_history_service):
        super().__init__(queue_service=queue_service, logger=logger,
                         run_history_service=run_history_service, service_name="tbs-policies")