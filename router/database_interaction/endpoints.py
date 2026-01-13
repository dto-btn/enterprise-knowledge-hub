from fastapi import APIRouter, Query

router = APIRouter()

@router.get("/search")
    def search_database(
        query: str = Query(..., description="Search query")
        # , limit: int = Query(10, description="Number of results to return")
    ):
        return query
        pass