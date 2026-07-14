"""
Endpoints for querying the knowledge database.

Search uses a slug-keyed service registry (same pattern as run management).
The embedding model and query instruction are read from kb_source_registry so
the query vector is always produced by the same model that ingested the data.

To add a new searchable source, insert one entry into _SEARCH_REGISTRY.
"""
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from provider.embedding.qwen3.embedder_factory import get_embedder
from router.root.models import SearchResponse, SearchResult
from services.database.kb_source_registry_service import KbSourceRegistryService
from services.database.tbs_policy_item_service import TBSPolicyItemService
from services.database.wiki_item_service import WikipediaArticleService

router = APIRouter()
logger = logging.getLogger(__name__)

_source_registry_service = KbSourceRegistryService()
_wikipedia_service = WikipediaArticleService(logger)
_tbs_policy_service = TBSPolicyItemService(logger)

# ── Search service registry ───────────────────────────────────────────────────
# Add a new searchable source here — one entry, nothing else to change.

_SEARCH_REGISTRY: dict[str, Any] = {
    "wikipedia":    _wikipedia_service,
    "tbs-policies": _tbs_policy_service,
}


# ── Generic search ────────────────────────────────────────────────────────────

@router.get("/{slug}/search", response_model=SearchResponse)
def knowledge_search(
    slug: str,
    query: str = Query(..., description="Natural language search query"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results"),
):
    """
    Semantic search over any ingested knowledge source.

    The query is embedded using the same model and task instruction that was
    active when the source was last ingested (stored in kb_source_registry).
    Known sources: wikipedia, tbs-policies.
    """
    service = _SEARCH_REGISTRY.get(slug)
    if service is None:
        known = ", ".join(sorted(_SEARCH_REGISTRY))
        raise HTTPException(status_code=404,
                            detail=f"Unknown source '{slug}'. Known sources: {known}")

    source_meta = _source_registry_service.get(slug)
    if source_meta is None:
        raise HTTPException(
            status_code=503,
            detail=f"Source '{slug}' has no registry entry — run ingestion first.",
        )

    logger.info("Search: source=%r query=%r limit=%d model=%s",
                slug, query, limit, source_meta.model_name)

    embedder = get_embedder()
    embedding = embedder.embed(query, instruction=source_meta.query_instruction)

    raw = service.search_by_embedding(embedding, limit=limit)
    results = [SearchResult(**row) for row in raw]

    return SearchResponse(
        query=query,
        source=slug,
        model=source_meta.model_name,
        total=len(results),
        results=results,
    )


# ── Wikipedia article retrieval (Wikipedia-specific, no generic equivalent) ──

@router.get("/wikipedia/retrieve/{title}")
def retrieve_wikipedia_article(title: str, source: str = Query("enwiki")):
    """Retrieve the full concatenated content of a Wikipedia article by title."""
    logger.info("Wikipedia retrieve: title=%r source=%r", title, source)
    result = _wikipedia_service.get_article_content_by_title(title, source)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Article '{title}' not found")
    return result
