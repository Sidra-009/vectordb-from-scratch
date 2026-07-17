"""
Module 2: REST API Layer for SidraDB
This module exposes the BruteForceDB engine as a web service using FastAPI.
Provides endpoints for adding vectors, searching, and checking stats.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple
import uvicorn

# Import the brute-force engine from src folder
from src.brute_engine import BruteForceDB

# -------------------- Initialize App & Database --------------------

app = FastAPI(
    title="SidraDB API",
    description="A high-performance vector database API built from scratch. Supports brute-force cosine similarity search.",
    version="0.1.0"
)

# Global instance of our vector database (stored in memory)
db = BruteForceDB()

# -------------------- Request/Response Models (Pydantic Schemas) --------------------

class VectorAddRequest(BaseModel):
    """
    Schema for adding a new vector to the database.
    """
    id: str                         # Unique identifier for the vector
    vector: List[float]             # The actual embedding (list of numbers)
    metadata: Optional[str] = ""    # Optional text like original sentence

class VectorSearchRequest(BaseModel):
    """
    Schema for searching similar vectors.
    """
    vector: List[float]             # Query vector to search against
    top_k: Optional[int] = 5        # Number of nearest neighbors to return

class SearchResultItem(BaseModel):
    """
    Schema for a single search result item.
    """
    id: str
    similarity: float
    metadata: Optional[str] = None

class SearchResponse(BaseModel):
    """
    Schema for the entire search response.
    """
    results: List[SearchResultItem]
    total_vectors: int

class StatsResponse(BaseModel):
    """
    Schema for database statistics.
    """
    total_vectors: int

# -------------------- API Endpoints --------------------

@app.get("/health", tags=["System"])
async def health_check():
    """
    Simple health check endpoint to verify the API is running.
    """
    return {"status": "healthy", "message": "SidraDB API is up and running!"}

@app.post("/add", tags=["Vector Operations"])
async def add_vector(request: VectorAddRequest):
    """
    Add a new vector to the database.
    
    - **id**: Unique identifier (string). If ID already exists, it will overwrite the old vector.
    - **vector**: List of floats (e.g., [0.1, 0.5, 0.9, ...]).
    - **metadata**: Optional text description of the vector.
    """
    try:
        db.add(request.id, request.vector, request.metadata)
        return {
            "status": "success",
            "message": f"Vector with id '{request.id}' added successfully.",
            "total_vectors": db.size()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add vector: {str(e)}")

@app.post("/search", tags=["Vector Operations"], response_model=SearchResponse)
async def search_vector(request: VectorSearchRequest):
    """
    Search for the top_k most similar vectors to the provided query vector.
    
    - **vector**: Query embedding to find neighbors for.
    - **top_k**: Number of results to return (default 5).
    """
    if db.size() == 0:
        return SearchResponse(results=[], total_vectors=0)
    
    try:
        # Perform brute-force search
        raw_results: List[Tuple[str, float]] = db.search(request.vector, request.top_k)
        
        # Format results with metadata
        formatted_results = []
        for idx, score in raw_results:
            meta = db.get_metadata(idx)
            formatted_results.append(
                SearchResultItem(
                    id=idx,
                    similarity=round(score, 4),  # Round to 4 decimal places for readability
                    metadata=meta
                )
            )
        
        return SearchResponse(
            results=formatted_results,
            total_vectors=db.size()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/stats", tags=["System"], response_model=StatsResponse)
async def get_stats():
    """
    Get current statistics about the database (total vectors stored).
    """
    return StatsResponse(total_vectors=db.size())

# -------------------- Run the Server (Optional) --------------------

if __name__ == "__main__":
    """
    Run the server directly using: python api/main.py
    """
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)