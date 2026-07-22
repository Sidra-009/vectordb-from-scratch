"""
Concurrency stress test for HNSWDB.
Run with: python tests/test_concurrency.py
"""

import sys
import os
import threading
import time
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.hnsw_engine import HNSWDB

def worker_add(db, count, start_id):
    """Thread worker to add vectors."""
    for i in range(count):
        vec = [random.random() for _ in range(384)]
        db.add(vec, f"test_{start_id + i}")
    print(f"[Worker-{start_id}] Added {count} vectors.")

def worker_search(db, count):
    """Thread worker to search vectors."""
    for _ in range(count):
        vec = [random.random() for _ in range(384)]
        db.search(vec, top_k=5)
    print(f"[Worker-Search] Performed {count} searches.")

def test_concurrency():
    """Run concurrent add and search operations."""
    print("🚀 Starting Concurrency Test...")
    db = HNSWDB()

    threads = []

    # 2 threads adding 100 vectors each (total 200)
    t1 = threading.Thread(target=worker_add, args=(db, 100, 0))
    t2 = threading.Thread(target=worker_add, args=(db, 100, 100))
    threads.append(t1)
    threads.append(t2)

    # 2 threads searching 50 times each
    t3 = threading.Thread(target=worker_search, args=(db, 50))
    t4 = threading.Thread(target=worker_search, args=(db, 50))
    threads.append(t3)
    threads.append(t4)

    # Start all threads
    for t in threads:
        t.start()

    # Wait for all threads to finish
    for t in threads:
        t.join()

    # Verify data integrity
    final_count = db.size()
    print(f"Final vector count: {final_count}")

    if final_count == 200:
        print("Concurrency Test PASSED! No crashes, data consistent.")
    else:
        print(f"Concurrency Test FAILED! Expected 200, got {final_count}.")

if __name__ == "__main__":
    test_concurrency()