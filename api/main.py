"""
Module 2 & 3: REST API Layer for VectorDB with Persistence
This module exposes the BruteForceDB engine as a web service using FastAPI.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple
import uvicorn
import os

from src.brute_engine import BruteForceDB

# -------------------- Initialize App & Database --------------------

app = FastAPI(
    title="VectorDB API",
    description="A high-performance vector database API built from scratch. Supports brute-force cosine similarity search and disk persistence.",
    version="0.2.0"
)

# Global instance of our vector database
db = BruteForceDB()
# File to persist data (Generic name)
DB_FILE = "vector_db_data.pkl"

# -------------------- Startup & Shutdown Events (Persistence) --------------------

@app.on_event("startup")
async def startup_event():
    """Load the database from disk when the server starts."""
    db.load(DB_FILE)

@app.on_event("shutdown")
async def shutdown_event():
    """Save the database to disk when the server shuts down."""
    db.save(DB_FILE)
    print("[Persistence] Database saved on shutdown.")

# -------------------- Request/Response Models --------------------

class VectorAddRequest(BaseModel):
    id: str
    vector: List[float]
    metadata: Optional[str] = ""

class VectorSearchRequest(BaseModel):
    vector: List[float]
    top_k: Optional[int] = 5

class SearchResultItem(BaseModel):
    id: str
    similarity: float
    metadata: Optional[str] = None

class SearchResponse(BaseModel):
    results: List[SearchResultItem]
    total_vectors: int

class StatsResponse(BaseModel):
    total_vectors: int

# -------------------- API Endpoints --------------------

@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "message": "VectorDB API is up and running!"}

@app.post("/add", tags=["Vector Operations"])
async def add_vector(request: VectorAddRequest):
    try:
        db.add(request.id, request.vector, request.metadata)
        # Save to disk after every addition
        db.save(DB_FILE)
        return {
            "status": "success",
            "message": f"Vector with id '{request.id}' added successfully.",
            "total_vectors": db.size()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add vector: {str(e)}")

@app.post("/search", tags=["Vector Operations"], response_model=SearchResponse)
async def search_vector(request: VectorSearchRequest):
    if db.size() == 0:
        return SearchResponse(results=[], total_vectors=0)
    
    try:
        raw_results = db.search(request.vector, request.top_k)
        formatted_results = []
        for idx, score in raw_results:
            meta = db.get_metadata(idx)
            formatted_results.append(
                SearchResultItem(
                    id=idx,
                    similarity=round(score, 4),
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
    return StatsResponse(total_vectors=db.size())

# -------------------- Run the Server --------------------

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)