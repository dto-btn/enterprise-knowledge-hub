"""Service layer for the knowledge-source embedding registry."""
from __future__ import annotations

from datetime import datetime, timezone

from repository.kb_source_registry import KbSourceRegistryRepository
from repository.kb_source_registry_model import KbSourceRegistry


class KbSourceRegistryService:
    """Register and retrieve embedding metadata per knowledge source."""

    def __init__(self) -> None:
        self._repo = KbSourceRegistryRepository()

    def register(
        self,
        source: str,
        model_name: str,
        dimensions: int,
        query_instruction: str,
    ) -> None:
        """Create or update the registry entry for a source."""
        self._repo.upsert(
            source=source,
            model_name=model_name,
            dimensions=dimensions,
            query_instruction=query_instruction,
            last_ingested_at=datetime.now(timezone.utc),
        )

    def get(self, source: str) -> KbSourceRegistry | None:
        """Return the registry entry for a source, or None if not yet ingested."""
        return self._repo.get_by_source(source)
