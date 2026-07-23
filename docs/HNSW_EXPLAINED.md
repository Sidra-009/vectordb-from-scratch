# HNSW Internals Explained

This document explains how our custom **HNSW (Hierarchical Navigable Small World)** implementation works under the hood.

> **Why this matters:** Most developers use libraries like FAISS or services like Pinecone. Very few have implemented the core algorithm themselves. This document proves we understand the math and data structures behind production-grade vector search.

---

## 1. What is HNSW?

HNSW is a graph-based algorithm for **Approximate Nearest Neighbor (ANN)** search. It builds a multi-layer graph where:

- **Layer 0 (Bottom):** Contains **all** vectors. Search here is 100% accurate but slow (O(n)).
- **Layer 1 (Middle):** Contains a **subset** of vectors (skip list style). Search here is faster but slightly less accurate.
- **Layer 2 (Top):** Contains only a **few** vectors (the "landmarks"). Search here is extremely fast but can miss distant neighbors.

When searching, we start at the **top layer**, greedily find the nearest node, drop down to the next layer, refine the search, and continue until we reach Layer 0. This gives us approximately **O(log n)** search complexity.

---

## 2. Our Implementation Parameters

In `api/main.py` and `src/hnsw_engine.py`, we initialize our HNSW index with:

```python
HNSWDB(M=16, ef_construction=200, ef_search=50)
```

What do these parameters mean?

| Parameter | Value | Explanation |
|---|---|---|
| `M` | 16 | Maximum number of bi-directional connections (neighbors) each node can have per layer. Higher `M` = better accuracy, but more memory and slower insertion. We chose 16 as a balanced default (FAISS also uses 16 as its HNSW default). |
| `ef_construction` | 200 | Dynamic candidate list size used while building the index. Higher values build a better-quality graph (higher recall), but increase build time. |
| `ef_search` | 50 | Dynamic candidate list size used during a search query. Higher values increase accuracy (recall) but increase search latency. The exact recall depends on the dataset and graph quality, so the benchmark script reports the measured Recall@10 directly rather than a fixed assumed number. |
| Max level | capped at a fixed ceiling | The random level generator that decides how "tall" a node is in the graph is capped, so an unlucky random draw can't create a pathologically tall, disconnected graph on a small dataset. |

**Trade-off Summary:**
- **Speed vs Accuracy:** Lower `ef_search` → faster but less accurate. Higher `ef_search` → slower but more accurate.
- **Memory vs Build Time:** Lower `M` → less memory, faster build, but lower recall. Higher `M` → more memory, slower build, higher recall.

---

## 3. Distance Metric: Cosine Similarity

We use Cosine Similarity as our distance metric.

```python
def _cosine_similarity(self, vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2)
```

**Why Cosine over Euclidean?**

Our embeddings come from `all-MiniLM-L6-v2` (Sentence-Transformers). This model is trained to output normalized embeddings (vectors lie on a unit sphere). For normalized vectors, cosine similarity is rank-equivalent to Euclidean distance, but cosine gives an intuitive 0–1 similarity score (1 = identical, 0 = orthogonal).

In the graph, we convert similarity to distance using `distance = 1 - similarity` to maintain a proper metric space.

---

## 4. How Search Actually Traverses the Graph

Instead of a diagram, here's the search flow described step by step for an example query `Q`:

1. **Start at the top layer** at a fixed entry point node.
2. **Greedily walk** to whichever neighbor is closest to `Q`, repeating until no neighbor improves the distance — this is a proper best-first search, not a single-hop check.
3. **Drop down one layer** using the closest node found so far as the new entry point.
4. **Repeat** the greedy walk at each layer, refining the candidate set, until reaching Layer 0.
5. **At Layer 0**, expand across the full local neighborhood using the `ef_search` candidate list and return the top-K closest nodes found.

---

## 5. Known Limitations (vs. Production HNSW)

Our implementation is built for learning and demonstration. It is not intended to replace FAISS or Pinecone in production. Current limitations:

| Limitation | Explanation | Production Alternative |
|---|---|---|
| Delete support is basic | Supports deleting vectors by ID and cleaning up neighbor references, but it's a learning-oriented implementation — production-grade tombstoning or rewiring isn't implemented. | FAISS supports deletion via ID mapping. Pinecone supports delete by ID. |
| Single-threaded build | Index insertion is purely sequential (no parallelization). | FAISS uses OpenMP for multi-threaded builds. |
| No dynamic `ef` tuning | `ef_search` is fixed at construction time. | Pinecone/Qdrant allow per-query `ef` override. |
| Memory-bound | All vectors and neighbor lists are stored in Python dictionaries (RAM only). | Production systems use memory-mapped files (mmap) or custom C++ allocators. |
| No replication/sharding | Single-node only. No distributed search. | Elasticsearch, Qdrant, and Pinecone support distributed deployments. |
| Insert performance | Distance calculations are pure Python/NumPy per-pair, not batch-vectorized, so insert time scales poorly on larger datasets. | Production systems vectorize distance computation or use SIMD/C++ kernels. |
| Basic persistence | Persistence (JSON + `.npy` + checksums) is hardened against corruption but doesn't handle concurrent writers. | RocksDB/LMDB for transactional persistence. |

---

## 6. Future Improvements

If we wanted to take this to production, we would focus on:

- **C++ rewrite:** Core HNSW logic in C++ with Python bindings (pybind11) for a significant speed boost.
- **Vectorized distance computation:** Batch distance calculations with NumPy/BLAS instead of per-pair Python loops.
- **Disk-based indexing:** Use memory-mapped files to handle billions of vectors.
- **Multi-threading:** Parallelize insertion and search.
- **Binary quantization (PQ):** Reduce memory footprint by compressing vectors.

This implementation was built as a learning exercise to open the "black box" of vector search algorithms.