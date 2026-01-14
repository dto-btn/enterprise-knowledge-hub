from fastapi import APIRouter

from services.db.model import DocumentRecord
from services.knowledge.query import QueryService


router = APIRouter()

# @router.get("/search")
#     def search_database(
#         query: str = Query(..., description="Search query")
#         # , limit: int = Query(10, description="Number of results to return")
#     )

_query_service = QueryService()

@router.get("/test/{title}")
def retrieve_wiki_articles(
    title:str
) -> list[DocumentRecord]:
    """Get wiki article content"""
    print("hit test endpoint success")
    return _query_service.get_article_content_by_title(title)
