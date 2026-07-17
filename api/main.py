"""
Module 2, 3, 4 & 5: REST API Layer with Text Embedding (AI) Support
Now supports adding and searching via natural language text!
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
import uvicorn
import os

# Import engines and embedder
from src.brute_engine import BruteForceDB
from src.hnsw_engine import HNSWDB
from src.embedder import Embedder

# -------------------- Configuration --------------------
# Set to True for HNSW (fast), False for Brute-Force (accurate)
USE_HNSW = True
# AI Model name (small and fast)
MODEL_NAME = "all-MiniLM-L6-v2"

# -------------------- Initialize App & Database --------------------

app = FastAPI(
    title="VectorDB API",
    description="High-performance vector database with AI text embeddings. Search by meaning, not just keywords!",
    version="1.0.0"  # Full release!
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

# AI Embedder (will be loaded on startup)
embedder = None

# -------------------- Startup & Shutdown --------------------

@app.on_event("startup")
async def startup_event():
    """Load database from disk and initialize AI model on startup."""
    global embedder
    db.load(DB_FILE)
    # Load the AI model
    embedder = Embedder(MODEL_NAME)
    print(f"[System] Using engine: {ENGINE_NAME}")
    print(f"[System] AI Model ready: {MODEL_NAME} (Dimension: {embedder.dimension})")

@app.on_event("shutdown")
async def shutdown_event():
    """Save database to disk on shutdown."""
    db.save(DB_FILE)
    print(f"[System] Database saved. Engine: {ENGINE_NAME}")

# -------------------- Request/Response Models --------------------

class VectorAddRequest(BaseModel):
    id: Optional[str] = None
    vector: List[float]
    metadata: Optional[str] = ""

class VectorSearchRequest(BaseModel):
    vector: List[float]
    top_k: Optional[int] = 5

class TextAddRequest(BaseModel):
    """Request model for adding text directly."""
    text: str
    id: Optional[str] = None
    metadata: Optional[str] = ""

class TextSearchRequest(BaseModel):
    """Request model for searching via text."""
    text: str
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
    return {
        "status": "healthy",
        "engine": ENGINE_NAME,
        "ai_model": MODEL_NAME,
        "total_vectors": db.size()
    }

# -------------------- RAW VECTOR ENDPOINTS (Same as before) --------------------

@app.post("/add", tags=["Vector Operations (Raw)"])
async def add_vector(request: VectorAddRequest):
    """Add a raw vector (list of numbers)."""
    try:
        if USE_HNSW:
            new_id = db.add(request.vector, request.metadata)
            db.save(DB_FILE)
            return {
                "status": "success",
                "message": f"Vector added with ID: {new_id}",
                "total_vectors": db.size()
            }
        else:
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

@app.post("/search", tags=["Vector Operations (Raw)"], response_model=SearchResponse)
async def search_vector(request: VectorSearchRequest):
    """Search using a raw vector (list of numbers)."""
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

# -------------------- ✨ NEW: AI TEXT EMBEDDING ENDPOINTS (Module 5) ✨ --------------------

@app.post("/add_text", tags=["AI Text Operations"])
async def add_text(request: TextAddRequest):
    """
    Add a text document to the database.
    The AI model will automatically convert it to a vector.
    
    Example: {"text": "Biryani is very spicy and tasty", "metadata": "food"}
    """
    global embedder
    if embedder is None:
        raise HTTPException(status_code=503, detail="AI Model is still loading. Please wait.")
    
    try:
        # Convert text to vector
        vector = embedder.encode(request.text)
        
        # Add to database
        if USE_HNSW:
            new_id = db.add(vector, request.metadata or request.text)
        else:
            # For brute force, use provided ID or generate one
            id_str = request.id or f"text_{db.size() + 1}"
            db.add(id_str, vector, request.metadata or request.text)
            new_id = id_str
        
        db.save(DB_FILE)
        return {
            "status": "success",
            "message": f"Text added with ID: {new_id}",
            "text": request.text,
            "vector_dimension": len(vector),
            "total_vectors": db.size()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add text: {str(e)}")

@app.post("/search_text", tags=["AI Text Operations"], response_model=SearchResponse)
async def search_text(request: TextSearchRequest):
    """
    Search for similar documents using a text query.
    The AI model will convert your query to a vector and find the nearest neighbors.
    
    Example: {"text": "I want something sweet", "top_k": 3}
    """
    global embedder
    if embedder is None:
        raise HTTPException(status_code=503, detail="AI Model is still loading. Please wait.")
    
    if db.size() == 0:
        return SearchResponse(results=[], total_vectors=0, engine=ENGINE_NAME)
    
    try:
        # Convert query text to vector
        query_vector = embedder.encode(request.text)
        
        # Search the database
        raw_results = db.search(query_vector, request.top_k)
        
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
        raise HTTPException(status_code=500, detail=f"Text search failed: {str(e)}")

# -------------------- Stats Endpoint --------------------

@app.get("/stats", tags=["System"], response_model=StatsResponse)
async def get_stats():
    return StatsResponse(total_vectors=db.size(), engine=ENGINE_NAME)

# -------------------- Run --------------------

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)