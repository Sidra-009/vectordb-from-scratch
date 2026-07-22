import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.hnsw_engine import HNSWDB


def test_delete_removes_vector_and_reduces_size():
    db = HNSWDB()
    vectors = [
        [1.0, 0.0, 0.0] + [0.0] * 381,
        [0.0, 1.0, 0.0] + [0.0] * 381,
        [0.0, 0.0, 1.0] + [0.0] * 381,
        [0.5, 0.5, 0.0] + [0.0] * 381,
        [0.0, 0.5, 0.5] + [0.0] * 381,
    ]

    for i, vec in enumerate(vectors):
        db.add(vec, f"vec_{i}")

    before_size = db.size()
    deleted = db.delete(1)
    after_size = db.size()

    results = db.search(vectors[1], top_k=5)
    returned_ids = [idx for idx, _ in results]

    assert deleted is True
    assert 1 not in returned_ids
    assert after_size == before_size - 1
