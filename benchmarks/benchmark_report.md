# VectorDB Benchmark Results

## Search Latency Comparison (p50, p95, p99)

| Vectors | Method | p50 (ms) | p95 (ms) | p99 (ms) | Insert Time (s) |
|---------|--------|----------|----------|----------|-----------------|
| 10,000 | Brute-Force | 134.17 | 296.71 | 323.19 | 0.26 |
| 10,000 | HNSW (ours) | 0.56 | 1.13 | 1.30 | 44.34 |
| 100,000 | Brute-Force | 1345.46 | 2533.86 | 2873.07 | 2.61 |
| 100,000 | HNSW (ours) | 0.70 | 0.74 | 0.78 | 418.20 |

## Speedup: HNSW vs Brute-Force

- **10,000 vectors**: HNSW is **240.0x faster** than brute-force.
- **100,000 vectors**: HNSW is **1917.8x faster** than brute-force.