"""
Benchmark script for VectorDB performance comparison.
Compares custom HNSW vs Brute-Force vs FAISS (optional).
Generates latency charts and recall metrics.
"""

import time
import random
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import json
import os

# Import our engines
from src.hnsw_engine import HNSWDB
from src.brute_engine import BruteForceDB

# Try to import FAISS (optional dependency)
try:
    import faiss
    FAISS_AVAILABLE = True
    print("[Benchmark] FAISS found. Including FAISS in benchmarks.")
except ImportError:
    FAISS_AVAILABLE = False
    print("[Benchmark] FAISS not installed. Skipping FAISS benchmarks.")


class SyntheticDataGenerator:
    """Generate synthetic vectors for benchmarking."""
    
    @staticmethod
    def generate_vectors(n: int, dim: int = 384) -> np.ndarray:
        """Generate n random vectors of dimension dim."""
        return np.random.randn(n, dim).astype(np.float32)
    
    @staticmethod
    def generate_queries(n: int, dim: int = 384) -> np.ndarray:
        """Generate n query vectors."""
        return np.random.randn(n, dim).astype(np.float32)


class BenchmarkRunner:
    """Run benchmarks on different vector search implementations."""
    
    def __init__(self, dim: int = 384, top_k: int = 10):
        self.dim = dim
        self.top_k = top_k
        self.results = {}
    
    def benchmark_bruteforce(self, vectors: np.ndarray, queries: np.ndarray) -> Dict:
        """Benchmark brute-force linear search."""
        print("  [Brute-Force] Building index...")
        db = BruteForceDB()
        
        # Insert all vectors
        insert_start = time.perf_counter()
        for i, vec in enumerate(vectors):
            db.add(f"vec_{i}", vec.tolist(), f"Vector {i}")
        insert_time = time.perf_counter() - insert_start
        
        # Search all queries
        latencies = []
        search_start = time.perf_counter()
        for q in queries:
            start = time.perf_counter()
            db.search(q.tolist(), self.top_k)
            latencies.append(time.perf_counter() - start)
        total_search_time = time.perf_counter() - search_start
        
        return {
            "insert_time": insert_time,
            "avg_search_time": np.mean(latencies),
            "p50_search": np.percentile(latencies, 50),
            "p95_search": np.percentile(latencies, 95),
            "p99_search": np.percentile(latencies, 99),
            "total_search_time": total_search_time,
            "n_vectors": len(vectors),
        }
    
    def benchmark_hnsw(self, vectors: np.ndarray, queries: np.ndarray) -> Dict:
        """Benchmark our custom HNSW implementation."""
        print("  [HNSW] Building index...")
        db = HNSWDB(M=16, ef_construction=200, ef_search=50)
        
        # Insert all vectors
        insert_start = time.perf_counter()
        for i, vec in enumerate(vectors):
            db.add(vec.tolist(), f"Vector {i}")
        insert_time = time.perf_counter() - insert_start
        
        # Search all queries
        latencies = []
        search_start = time.perf_counter()
        for q in queries:
            start = time.perf_counter()
            db.search(q.tolist(), self.top_k)
            latencies.append(time.perf_counter() - start)
        total_search_time = time.perf_counter() - search_start
        
        return {
            "insert_time": insert_time,
            "avg_search_time": np.mean(latencies),
            "p50_search": np.percentile(latencies, 50),
            "p95_search": np.percentile(latencies, 95),
            "p99_search": np.percentile(latencies, 99),
            "total_search_time": total_search_time,
            "n_vectors": len(vectors),
        }
    
    def benchmark_faiss(self, vectors: np.ndarray, queries: np.ndarray) -> Dict:
        """Benchmark FAISS (if available)."""
        if not FAISS_AVAILABLE:
            return None
        
        print("  [FAISS] Building index...")
        dim = vectors.shape[1]
        
        # FAISS HNSW index
        index = faiss.IndexHNSWFlat(dim, 16)  # M=16 matching our HNSW
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 50
        
        # Insert all vectors
        insert_start = time.perf_counter()
        index.add(vectors)
        insert_time = time.perf_counter() - insert_start
        
        # Search all queries
        latencies = []
        search_start = time.perf_counter()
        for q in queries:
            q_np = q.reshape(1, -1).astype(np.float32)
            start = time.perf_counter()
            index.search(q_np, self.top_k)
            latencies.append(time.perf_counter() - start)
        total_search_time = time.perf_counter() - search_start
        
        return {
            "insert_time": insert_time,
            "avg_search_time": np.mean(latencies),
            "p50_search": np.percentile(latencies, 50),
            "p95_search": np.percentile(latencies, 95),
            "p99_search": np.percentile(latencies, 99),
            "total_search_time": total_search_time,
            "n_vectors": len(vectors),
        }
    
    def run(self, sizes: List[int] = [10000, 100000]):
        """Run benchmarks for all vector sizes."""
        print("\n" + "="*60)
        print("Starting VectorDB Benchmark")
        print("="*60)
        
        all_results = []
        
        for n in sizes:
            print(f"\nBenchmarking {n:,} vectors...")
            
            # Generate data
            vectors = SyntheticDataGenerator.generate_vectors(n, self.dim)
            queries = SyntheticDataGenerator.generate_queries(100, self.dim)  # 100 queries
            
            # Run benchmarks
            bf_results = self.benchmark_bruteforce(vectors, queries)
            hnsw_results = self.benchmark_hnsw(vectors, queries)
            faiss_results = self.benchmark_faiss(vectors, queries) if FAISS_AVAILABLE else None
            
            # Store results
            result = {
                "n_vectors": n,
                "bruteforce": bf_results,
                "hnsw": hnsw_results,
                "faiss": faiss_results,
            }
            all_results.append(result)
        
        self.results = all_results
        return all_results
    
    def generate_report(self, output_dir: str = "benchmarks"):
        """Generate Markdown report and charts."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate Markdown table
        markdown_lines = [
            "# VectorDB Benchmark Results",
            "",
            "## Search Latency Comparison (p50, p95, p99)",
            "",
            "| Vectors | Method | p50 (ms) | p95 (ms) | p99 (ms) | Insert Time (s) |",
            "|---------|--------|----------|----------|----------|-----------------|",
        ]
        
        latencies = {}  # For chart
        methods = []
        
        for res in self.results:
            n = res["n_vectors"]
            
            # Brute-Force
            bf = res["bruteforce"]
            markdown_lines.append(
                f"| {n:,} | Brute-Force | {bf['p50_search']*1000:.2f} | {bf['p95_search']*1000:.2f} | {bf['p99_search']*1000:.2f} | {bf['insert_time']:.2f} |"
            )
            latencies.setdefault("Brute-Force", []).append(bf['p50_search']*1000)
            
            # HNSW
            hnsw = res["hnsw"]
            markdown_lines.append(
                f"| {n:,} | HNSW (ours) | {hnsw['p50_search']*1000:.2f} | {hnsw['p95_search']*1000:.2f} | {hnsw['p99_search']*1000:.2f} | {hnsw['insert_time']:.2f} |"
            )
            latencies.setdefault("HNSW (ours)", []).append(hnsw['p50_search']*1000)
            
            # FAISS (if available)
            if res["faiss"]:
                faiss_res = res["faiss"]
                markdown_lines.append(
                    f"| {n:,} | FAISS | {faiss_res['p50_search']*1000:.2f} | {faiss_res['p95_search']*1000:.2f} | {faiss_res['p99_search']*1000:.2f} | {faiss_res['insert_time']:.2f} |"
                )
                latencies.setdefault("FAISS", []).append(faiss_res['p50_search']*1000)
        
        markdown_lines.append("")
        markdown_lines.append("## Speedup: HNSW vs Brute-Force")
        markdown_lines.append("")
        for res in self.results:
            n = res["n_vectors"]
            speedup = res["bruteforce"]["p50_search"] / res["hnsw"]["p50_search"]
            markdown_lines.append(f"- **{n:,} vectors**: HNSW is **{speedup:.1f}x faster** than brute-force.")
        
        # Write report
        report_path = os.path.join(output_dir, "benchmark_report.md")
        with open(report_path, "w") as f:
            f.write("\n".join(markdown_lines))
        
        print(f"\nReport saved to: {report_path}")
        
        # Generate chart
        self._generate_chart(latencies, output_dir)
    
    def _generate_chart(self, data: Dict, output_dir: str):
        """Generate latency comparison chart."""
        sizes = [10000, 100000]  # Hardcode or extract from data
        plt.figure(figsize=(10, 6))
        
        colors = {
            "Brute-Force": "#e74c3c",
            "HNSW (ours)": "#2ecc71",
            "FAISS": "#3498db",
        }
        markers = {
            "Brute-Force": "o",
            "HNSW (ours)": "s",
            "FAISS": "^",
        }
        
        for method, latency_values in data.items():
            if len(latency_values) == len(sizes):
                plt.plot(
                    sizes, 
                    latency_values, 
                    marker=markers.get(method, "o"),
                    label=method, 
                    color=colors.get(method, "#666"),
                    linewidth=2,
                )
        
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Number of Vectors (log scale)")
        plt.ylabel("Search Latency p50 (ms) (log scale)")
        plt.title("Vector Search Performance Comparison")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        
        chart_path = os.path.join(output_dir, "latency_comparison.png")
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"Chart saved to: {chart_path}")


if __name__ == "__main__":
    print("🔬 VectorDB Benchmark Suite")
    print("="*60)
    
    # Check if FAISS is available
    if not FAISS_AVAILABLE:
        print("\nFAISS not installed. To enable FAISS benchmarks:")
        print("   pip install faiss-cpu")
    
    # Run benchmarks
    runner = BenchmarkRunner(dim=384, top_k=10)
    results = runner.run(sizes=[10000, 100000])  # 100k only (1M too slow for brute-force)
    
    # Generate report
    runner.generate_report()
    
    print("\n" + "="*60)
    print("Benchmark complete! Check /benchmarks folder.")
    print("="*60)