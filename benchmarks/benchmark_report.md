# VectorDB Benchmark Results

## Search Latency Comparison (p50, p95, p99)

| Vectors | Method | p50 (ms) | p95 (ms) | p99 (ms) | Insert Time (s) | Recall@10 (%) |
|---------|--------|----------|----------|----------|-----------------|----------------|
| 1,000 | Brute-Force | 8.16 | 13.28 | 14.55 | 0.04 |  |
| 1,000 | HNSW (ours) | 10.50 | 15.61 | 18.87 | 41.82 | 76.60% |
| 5,000 | Brute-Force | 58.11 | 84.03 | 98.86 | 0.17 |  |
| 5,000 | HNSW (ours) | 10.68 | 20.65 | 23.55 | 466.58 | 37.00% |

## Recall@10

- **1,000 vectors**: average Recall@10 = **76.60%**
- **5,000 vectors**: average Recall@10 = **37.00%**

## Speedup: HNSW vs Brute-Force

- **1,000 vectors**: HNSW is **0.8x faster** than brute-force.
- **5,000 vectors**: HNSW is **5.4x faster** than brute-force.