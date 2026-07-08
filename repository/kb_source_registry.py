"""Repository for the knowledge-source embedding registry."""
from __future__ import annotations

from datetime import datetime

from repository.kb_source_registry_model import KbSourceRegistry


class KbSourceRegistryRepository:
    """Read/write source embedding metadata."""

    def upsert(
        self,
        source: str,
        model_name: str,
        dimensions: int,
        query_instruction: str,
        last_ingested_at: datetime,
    ) -> None:
        """Insert or update the registry entry for a source."""
        KbSourceRegistry.insert(
            source=source,
            model_name=model_name,
            dimensions=dimensions,
            query_instruction=query_instruction,
            last_ingested_at=last_ingested_at,
        ).on_conflict(
            conflict_target=[KbSourceRegistry.source],
            update={
                KbSourceRegistry.model_name: model_name,
                KbSourceRegistry.dimensions: dimensions,
                KbSourceRegistry.query_instruction: query_instruction,
                KbSourceRegistry.last_ingested_at: last_ingested_at,
            },
        ).execute()

    def get_by_source(self, source: str) -> KbSourceRegistry | None:
        """Return the registry entry for a source, or None if not found."""
        return KbSourceRegistry.get_or_none(KbSourceRegistry.source == source)
