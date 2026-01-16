"""
Docstring for router.database_interaction.endpoints
"""
from fastapi import APIRouter

from services.db.model import DocumentRecord
from services.knowledge.query import QueryService


router = APIRouter()

_query_service = QueryService()

@router.get("/retrieve/{title}")
def retrieve_wiki_articles(
    title:str
) -> list[DocumentRecord]:
    """Get wiki article content"""
    print("hit test endpoint success")
    return _query_service.get_article_content_by_title(title)
