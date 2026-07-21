"""
VectorDB API - FastAPI backend with HNSW indexing and AI embeddings.
"""

import os
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "cache")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.getcwd(), "cache")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union
import uvicorn

from src.brute_engine import BruteForceDB
from src.hnsw_engine import HNSWDB
from src.embedder import Embedder

# config
USE_HNSW = True
MODEL_NAME = "all-MiniLM-L6-v2"

# app
app = FastAPI(
    title="VectorDB API",
    description="High-performance vector database with AI text embeddings. Search by meaning, not just keywords!",
    version="1.0.0"
)

# cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# engine
if USE_HNSW:
    db = HNSWDB(M=16, ef_construction=200, ef_search=50)
    DB_FILE = "hnsw_db_data"
    ENGINE_NAME = "HNSW (Fast Approximate)"
else:
    db = BruteForceDB()
    DB_FILE = "vector_db_data"
    ENGINE_NAME = "Brute-Force (Accurate)"

embedder = None

# startup
@app.on_event("startup")
async def startup_event():
    global embedder
    db.load(DB_FILE)
    embedder = Embedder(MODEL_NAME)
    print(f"[System] Using engine: {ENGINE_NAME}")
    print(f"[System] AI Model ready: {MODEL_NAME} (Dimension: {embedder.dimension})")

# shutdown
@app.on_event("shutdown")
async def shutdown_event():
    db.save(DB_FILE)
    print(f"[System] Database saved. Engine: {ENGINE_NAME}")

# models
class VectorAddRequest(BaseModel):
    id: Optional[str] = None
    vector: List[float]
    metadata: Optional[str] = ""

class VectorSearchRequest(BaseModel):
    vector: List[float]
    top_k: Optional[int] = 5
    filter_metadata: Optional[str] = None

class TextAddRequest(BaseModel):
    text: str
    id: Optional[str] = None
    metadata: Optional[str] = ""

class TextSearchRequest(BaseModel):
    text: str
    top_k: Optional[int] = 5
    filter_metadata: Optional[str] = None

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

# health
@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "engine": ENGINE_NAME,
        "ai_model": MODEL_NAME,
        "total_vectors": db.size()
    }

# raw add
@app.post("/add", tags=["Vector Operations (Raw)"])
async def add_vector(request: VectorAddRequest):
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

# raw search
@app.post("/search", tags=["Vector Operations (Raw)"], response_model=SearchResponse)
async def search_vector(request: VectorSearchRequest):
    if db.size() == 0:
        return SearchResponse(results=[], total_vectors=0, engine=ENGINE_NAME)
    try:
        raw_results = db.search(request.vector, request.top_k)

        # post-filtering
        if request.filter_metadata:
            filtered = []
            for idx, score in raw_results:
                meta = db.get_metadata(idx)
                if meta and request.filter_metadata.lower() in meta.lower():
                    filtered.append((idx, score))
            raw_results = filtered

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

# ai add
@app.post("/add_text", tags=["AI Text Operations"])
async def add_text(request: TextAddRequest):
    global embedder
    if embedder is None:
        raise HTTPException(status_code=503, detail="AI Model is still loading. Please wait.")
    try:
        vector = embedder.encode(request.text)
        if USE_HNSW:
            new_id = db.add(vector, request.metadata or request.text)
        else:
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

# ai search
@app.post("/search_text", tags=["AI Text Operations"], response_model=SearchResponse)
async def search_text(request: TextSearchRequest):
    global embedder
    if embedder is None:
        raise HTTPException(status_code=503, detail="AI Model is still loading. Please wait.")
    if db.size() == 0:
        return SearchResponse(results=[], total_vectors=0, engine=ENGINE_NAME)
    try:
        query_vector = embedder.encode(request.text)
        raw_results = db.search(query_vector, request.top_k)

        # post-filtering
        if request.filter_metadata:
            filtered = []
            for idx, score in raw_results:
                meta = db.get_metadata(idx)
                if meta and request.filter_metadata.lower() in meta.lower():
                    filtered.append((idx, score))
            raw_results = filtered

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

# stats
@app.get("/stats", tags=["System"], response_model=StatsResponse)
async def get_stats():
    return StatsResponse(total_vectors=db.size(), engine=ENGINE_NAME)

# run
if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)