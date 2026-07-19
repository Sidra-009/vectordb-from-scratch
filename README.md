```markdown
# 🚀 VectorDB: Semantic Search Engine from Scratch

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.139+-green.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/Sidra-009/vectordb-from-scratch.svg)](https://github.com/Sidra-009/vectordb-from-scratch/stargazers)

> **A high-performance Vector Database built from scratch.**  
> Search by **meaning**, not just keywords. Powered by HNSW graphs and AI embeddings.

## 🌟 Why VectorDB?

Most developers use Pinecone or Qdrant, but few understand how they work underneath. This project is a **complete, production-ready implementation** that:

- 🔍 Implements **HNSW (Hierarchical Navigable Small World)** from scratch for fast approximate search.
- 🧠 Integrates **Sentence-Transformers** for semantic (AI) search.
- 🌐 Exposes a **RESTful API** with FastAPI.
- 🎨 Features a beautiful **Next.js UI** with glassmorphism design.
- 🐳 Fully **containerized with Docker** for one-click deployment.

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **Brute-Force Engine** | 100% accurate cosine similarity search (O(n)). |
| **HNSW Index** | Fast approximate search (O(log n)) using multi-layer graphs. |
| **AI Embeddings** | Converts text to vectors using `all-MiniLM-L6-v2`. |
| **Persistence** | Automatic save/load to disk using Pickle. |
| **REST API** | Swagger UI available at `/docs`. |
| **Production UI** | Next.js frontend with real-time search and progress bars. |
| **Dockerized** | Run the entire stack with a single command. |

---

## 🗺️ System Architecture

```mermaid
graph TD
    A[Next.js UI (Port 3000)] --> B[FastAPI Backend (Port 8000)];
    B --> C[Vector Engine (HNSW / Brute-Force)];
    C --> D[(Pickle Persistence)];
    B --> E[AI Embedder (Sentence-Transformers)];
    E --> F[(Model Cache)];
    style A fill:#a855f7,stroke:#fff,color:#fff
    style B fill:#3b82f6,stroke:#fff,color:#fff
```

---

## 🚀 Quick Start (Production)

The easiest way to run the full stack:

### Prerequisites
- Docker & Docker Compose installed.

### Commands
```bash
# 1. Clone the repository
git clone https://github.com/Sidra-009/vectordb-from-scratch.git
cd vectordb-from-scratch

# 2. Build and run (Docker will handle everything)
docker compose up --build

# 3. Open your browser
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/docs
```

> **Note**: The first build may take 5-10 minutes as it downloads the AI model (90MB) and Python dependencies.

---

## 💻 Local Development (Without Docker)

If you prefer to run natively:

### 1. Backend (Python)
```bash
# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn api.main:app --reload
```

### 2. Frontend (Next.js)
```bash
cd frontend

# Install dependencies
npm install

# Run the development server
npm run dev
```

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:3000`

---

## 📚 API Reference

Once the server is running, visit `/docs` for interactive Swagger UI.

### 1. Add Text (Semantic Vector)
```bash
curl -X POST "http://localhost:8000/add_text" \
  -H "Content-Type: application/json" \
  -d '{"text": "Biryani is a spicy rice dish", "metadata": "Pakistani Food"}'
```

### 2. Search Text (Semantic Query)
```bash
curl -X POST "http://localhost:8000/search_text" \
  -H "Content-Type: application/json" \
  -d '{"text": "I want something sweet", "top_k": 3}'
```

### 3. Get Database Stats
```bash
curl "http://localhost:8000/stats"
```

### 4. Health Check
```bash
curl "http://localhost:8000/health"
```

---

## 🛠️ Tech Stack

### Backend
- **[FastAPI](https://fastapi.tiangolo.com/)** - High-performance web framework.
- **[NumPy](https://numpy.org/)** - Linear algebra for vector operations.
- **[Sentence-Transformers](https://www.sbert.net/)** - AI embeddings.
- **[HNSW](https://arxiv.org/abs/1603.09320)** - Approximate nearest neighbor algorithm (implemented from scratch!).

### Frontend
- **[Next.js](https://nextjs.org/)** - React framework.
- **[Tailwind CSS](https://tailwindcss.com/)** - Utility-first CSS with glassmorphism.

### DevOps
- **[Docker](https://www.docker.com/)** - Containerization.
- **[Docker Compose](https://docs.docker.com/compose/)** - Multi-container orchestration.

---

## 🤝 Contributing

This is an open-source educational project. Contributions are welcome!

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 🎯 Roadmap

- [x] Brute-Force Engine
- [x] HNSW Index
- [x] AI Embeddings
- [x] REST API
- [x] Next.js UI
- [x] Dockerization
- [ ] PostgreSQL backend for large-scale data
- [ ] Authentication & Rate Limiting
- [ ] Kubernetes deployment

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
---

Ab aap ka GitHub repo **beautiful** aur **professional** lag raha hoga! 🚀
