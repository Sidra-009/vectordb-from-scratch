# Build Your Own Vector Database

A step-by-step, educational implementation of a Vector Database from scratch. 
This project demonstrates how modern AI search engines (like Pinecone, Qdrant) work underneath the hood.

## 🚀 Roadmap (Modules)
- [ ] **Module 0**: Project Setup (Structure, Dependencies)
- [ ] **Module 1**: Brute-Force Engine (Cosine Similarity, Naive Search)
- [ ] **Module 2**: REST API Layer (FastAPI integration)
- [ ] **Module 3**: Persistence Layer (Save/Load to disk with Pickle)
- [ ] **Module 4**: HNSW Index (Hierarchical Navigable Small World for fast search)
- [ ] **Module 5**: AI Integration (Semantic search using Sentence-Transformers)

## 🛠️ Tech Stack
- **Language**: Python 3.9+
- **Libraries**: NumPy, FastAPI, Uvicorn, Sentence-Transformers
- **Testing**: Pytest

## 🏃 How to Run (After Module 2)
1. Install dependencies: `pip install -r requirements.txt`
2. Run the server: `uvicorn api.main:app --reload`
3. Open browser: `http://localhost:8000/docs` for Swagger UI


## 📖 Documentation

- **[HNSW Internals Explained](docs/HNSW_EXPLAINED.md)** — Deep dive into our HNSW implementation, parameters (`M`, `ef_construction`, `ef_search`), distance metric, architecture diagram, and known limitations vs. production systems.