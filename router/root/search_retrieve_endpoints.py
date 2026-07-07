"""
Endpoints for querying the knowledge database.
"""
import logging

from fastapi import APIRouter, HTTPException, Query
from services.database.knowledge_item_service import KnowledgeItemService
from services.database.tbs_policy_item_service import TBSPolicyItemService

router = APIRouter()
logger = logging.getLogger(__name__)

_wikipedia_service = KnowledgeItemService(logger)
_tbs_policies_service = TBSPolicyItemService(logger)


@router.get("/wikipedia/search")
def search_wikipedia(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Number of results to return"),
):
    """Search Wikipedia articles by semantic similarity."""
    logger.info("Wikipedia search: query=%r limit=%d", query, limit)
    results = _wikipedia_service.search(query, limit)
    return {"query": query, "results": results}


@router.get("/wikipedia/retrieve/{title}")
def retrieve_wikipedia_article(title: str, source: str = Query("enwiki")):
    """Retrieve the full content of a Wikipedia article by title."""
    logger.info("Wikipedia retrieve: title=%r source=%r", title, source)
    result = _wikipedia_service.get_article_content_by_title(title, source)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Article '{title}' not found")
    return result


@router.get("/tbs-policies/search")
def search_tbs_policies(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Number of results to return"),
):
    """Search TBS policy documents by semantic similarity."""
    logger.info("TBS policies search: query=%r limit=%d", query, limit)
    results = _tbs_policies_service.search(query, limit)
    return {"query": query, "results": results}
