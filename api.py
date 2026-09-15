from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from search_engine import search


app = FastAPI(
    title="Semantic Search Engine",
    description="AI-powered semantic search using embeddings and FAISS",
    version="1.0.0"
)


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.get("/search")
def search_documents(
    q: str = Query(
        ...,
        min_length=1,
        description="Search query"
    ),
    top_k: int = Query(
        3,
        ge=1,
        le=10,
        description="Number of results to return"
    )
):

    # Remove unnecessary whitespace
    query = q.strip()

    # Reject empty queries
    if not query:
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty."
        )

    try:

        results = search(
            query,
            top_k=top_k
        )

        return {
            "query": query,
            "results": results
        }

    except Exception as error:

        print("Search error:", error)

        raise HTTPException(
            status_code=500,
            detail="An error occurred while performing the search."
        )