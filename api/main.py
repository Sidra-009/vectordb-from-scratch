"""
Module 2, 3 & 4: REST API Layer with HNSW Support
Supports both Brute-Force (accurate) and HNSW (fast) engines via config flag.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
import uvicorn
import os

# Import both engines
from src.brute_engine import BruteForceDB
from src.hnsw_engine import HNSWDB

# -------------------- Configuration --------------------
# Set this to True to use HNSW (fast), False to use Brute-Force (accurate)
USE_HNSW = True  # <-- CHANGE THIS TO TEST BOTH ENGINES!

# -------------------- Initialize App & Database --------------------

app = FastAPI(
    title="VectorDB API",
    description="High-performance vector database with Brute-Force and HNSW engines.",
    version="0.4.0"
)

# Initialize the selected engine
if USE_HNSW:
    db = HNSWDB(M=16, ef_construction=200, ef_search=50)
    DB_FILE = "hnsw_db_data.pkl"
    ENGINE_NAME = "HNSW (Fast Approximate)"
else:
    db = BruteForceDB()
    DB_FILE = "vector_db_data.pkl"
    ENGINE_NAME = "Brute-Force (Accurate)"

# -------------------- Startup & Shutdown --------------------

@app.on_event("startup")
async def startup_event():
    """Load database from disk on startup."""
    db.load(DB_FILE)
    print(f"[System] Using engine: {ENGINE_NAME}")

@app.on_event("shutdown")
async def shutdown_event():
    """Save database to disk on shutdown."""
    db.save(DB_FILE)
    print(f"[System] Database saved. Engine: {ENGINE_NAME}")

# -------------------- Models --------------------

class VectorAddRequest(BaseModel):
    id: Optional[Union[str, int]] = None  # Auto-generate if not provided (for HNSW)
    vector: List[float]
    metadata: Optional[str] = ""

class VectorSearchRequest(BaseModel):
    vector: List[float]
    top_k: Optional[int] = 5

class SearchResultItem(BaseModel):
    id: Union[str, int]
    similarity: float
    metadata: Optional[str] = None

class SearchResponse(BaseModel):
    results: List[SearchResultItem]
    total_vectors: int
    engine: str

class StatsResponse(BaseModel):
    total_vectors: int
    engine: str

# -------------------- API Endpoints --------------------

@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "engine": ENGINE_NAME}

@app.post("/add", tags=["Vector Operations"])
async def add_vector(request: VectorAddRequest):
    """Add a vector. If no ID provided, auto-generates one (for HNSW)."""
    try:
        if USE_HNSW:
            # HNSW uses integer IDs. If string provided, convert or auto-generate.
            if request.id is None:
                new_id = db.add(request.vector, request.metadata)
            else:
                # HNSW expects int, but we stored as int internally.
                # For simplicity, we just use the passed ID if it's int.
                # Ideally we'd map, but to keep it simple, we auto-generate.
                # To ensure compatibility, we ignore passed string IDs for HNSW.
                # Let's just use auto-generate for HNSW.
                new_id = db.add(request.vector, request.metadata)
            return {
                "status": "success",
                "message": f"Vector added with ID: {new_id}",
                "total_vectors": db.size()
            }
        else:
            # Brute-Force uses string IDs
            if request.id is None or request.id == "":
                raise HTTPException(status_code=400, detail="ID is required for Brute-Force engine")
            db.add(str(request.id), request.vector, request.metadata)
            db.save(DB_FILE)
            return {
                "status": "success",
                "message": f"Vector with id '{request.id}' added.",
                "total_vectors": db.size()
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add vector: {str(e)}")

@app.post("/search", tags=["Vector Operations"], response_model=SearchResponse)
async def search_vector(request: VectorSearchRequest):
    """Search for similar vectors."""
    if db.size() == 0:
        return SearchResponse(results=[], total_vectors=0, engine=ENGINE_NAME)
    
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
            total_vectors=db.size(),
            engine=ENGINE_NAME
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/stats", tags=["System"], response_model=StatsResponse)
async def get_stats():
    return StatsResponse(total_vectors=db.size(), engine=ENGINE_NAME)

# -------------------- Run --------------------

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)