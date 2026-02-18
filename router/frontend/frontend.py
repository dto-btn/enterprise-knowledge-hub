"""
Frontend router for serving static files.
"""
from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()

@router.get("/")
def dev_frontend():
    """Serve the index.html file."""
    return FileResponse("router/frontend/index.html")

# Here we can add frontend endpoints related to status of db (size, number of articles, etc.)
