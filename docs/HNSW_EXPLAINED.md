# HNSW Internals Explained

This document explains how our custom **HNSW (Hierarchical Navigable Small World)** implementation works under the hood. 

> **Why this matters:** Most developers use libraries like FAISS or services like Pinecone. Very few have implemented the core algorithm themselves. This document proves we understand the math and data structures behind production-grade vector search.

---

## 1. What is HNSW?

HNSW is a graph-based algorithm for **Approximate Nearest Neighbor (ANN)** search. It builds a multi-layer graph where:

- **Layer 0 (Bottom):** Contains **all** vectors. Search here is 100% accurate but slow (O(n)).
- **Layer 1 (Middle):** Contains a **subset** of vectors (skip list style). Search here is faster but slightly less accurate.
- **Layer 2 (Top):** Contains only a **few** vectors (the "landmarks"). Search here is extremely fast but can miss distant neighbors.

When searching, we start at the **top layer**, greedily find the nearest node, drop down to the next layer, refine the search, and continue until we reach Layer 0. This gives us **O(log n)** search complexity.

---

## 2. Our Implementation Parameters

In `api/main.py` and `src/hnsw_engine.py`, we initialized our HNSW index with:

```python
HNSWDB(M=16, ef_construction=200, ef_search=50)
What do these parameters mean?
Parameter	Value	Explanation
M	16	Maximum number of bi-directional connections (neighbors) each node can have per layer. Higher M = better accuracy, but more memory and slower insertion. We chose 16 as a balanced default (FAISS also uses 16 for its HNSW default).
ef_construction	200	Dynamic list size used while building the index. Higher values build a better-quality graph (higher recall), but increase build time. 200 is a sweet spot for up to 100k vectors.
ef_search	50	Dynamic list size used during a search query. Higher values increase accuracy (recall) but increase search latency. The exact recall depends on the dataset and graph quality, so the benchmark script reports the measured Recall@10 directly.
Trade-off Summary:
Speed vs Accuracy: Lower ef_search -> faster but less accurate. Higher ef_search -> slower but more accurate.

Memory vs Build Time: Lower M -> less memory, faster build, but lower recall. Higher M -> more memory, slower build, higher recall.

3. Distance Metric: Cosine Similarity
We use Cosine Similarity as our distance metric.

python
def _cosine_similarity(self, vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2)
Why Cosine over Euclidean?

Our embeddings come from all-MiniLM-L6-v2 (Sentence-Transformers). This model is trained to output normalized embeddings (vectors lie on a unit sphere). For normalized vectors, cosine similarity is equivalent to Euclidean distance (rank-wise), but cosine gives an intuitive 0..1 similarity score (1 = identical, 0 = orthogonal).

In the graph, we convert similarity to distance using distance = 1 - similarity to maintain a proper metric space.

4. Architecture (Mermaid Diagram)
Below is a simplified visualization of our 3-layer HNSW graph with M=2 (for illustration). In reality, M=16 and the number of layers depends on the dataset size.


















Search Flow (Example query "Q"):

Start at Layer 2: Compare Q with A and B. A is closer. Move to A.

Drop to Layer 1: From A, explore its neighbors (C). Compare Q with C and B. C is closer. Move to C.

Drop to Layer 0: From C, explore its neighbors (E, F). Compare Q with all neighbors. Return the top-K closest nodes.

5. Known Limitations (vs. Production HNSW)
Our implementation is built for learning and demonstration. It is not intended to replace FAISS or Pinecone in production. Here are the current limitations:

Limitation	Explanation	Production Alternative
Delete Support is Basic	The implementation now supports deleting vectors by ID and cleaning up neighbor references. It is still a learning-oriented graph implementation, so production-grade tombstone or rewire strategies would be the next step.
Single-threaded Build	Index insertion is purely sequential (no parallelization).	FAISS uses OpenMP for multi-threaded builds.
No Dynamic ef Tuning	ef_search is fixed at 50. In production, you often tune this per-query based on latency SLAs.	Pinecone/Qdrant allow per-query ef override.
Memory-bound	All vectors and neighbor lists are stored in Python dictionaries (RAM only).	Production systems use memory-mapped files (mmap) or custom C++ allocators.
No Replication/Sharding	Single-node only. No distributed search.	Elasticsearch, Qdrant, and Pinecone support distributed deployments.
Basic Persistence	While we hardened persistence (JSON + .npy + checksums), it doesn't handle concurrent writers.	RocksDB/LMDB for transactional persistence.
6. Future Improvements
If we wanted to take this to production, we would focus on:

C++ Rewrite: Core HNSW logic in C++ with Python bindings (pybind11) for 10x speed boost.

Disk-based Indexing: Use memory-mapped files to handle billions of vectors.

Multi-threading: Parallelize insertion and search.

Binary Quantization (PQ): Reduce memory footprint by compressing vectors.

This implementation was built as a learning exercise to open the "black box" of vector search algorithms.