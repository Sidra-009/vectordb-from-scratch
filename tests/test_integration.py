"""
Integration tests for VectorDB API.
Run with: pytest tests/test_integration.py -v
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.hnsw_engine import HNSWDB
from src.embedder import Embedder


class TestHNSWDB:
    """Test suite for HNSWDB engine."""

    def test_empty_search(self):
        """Search on empty database should return empty list."""
        db = HNSWDB()
        results = db.search([0.0] * 384, top_k=5)
        assert results == []

    def test_add_and_search(self):
        """Basic add and search functionality."""
        db = HNSWDB()
        vec1 = [1.0, 0.0, 0.0] + [0.0] * 381
        vec2 = [0.0, 1.0, 0.0] + [0.0] * 381
        vec3 = [0.5, 0.5, 0.0] + [0.0] * 381

        db.add(vec1, "test1")
        db.add(vec2, "test2")
        db.add(vec3, "test3")

        results = db.search([1.0, 0.0, 0.0] + [0.0] * 381, top_k=2)
        assert len(results) == 2
        assert results[0][0] == 0  # ID 0 should be closest

    def test_duplicate_insertion(self):
        """HNSW generates new IDs for each insertion."""
        db = HNSWDB()
        vec = [0.5] * 384
        id1 = db.add(vec, "first")
        id2 = db.add(vec, "second")
        assert id1 != id2
        assert db.size() == 2

    def test_very_short_text_embedding(self):
        """Test embedding for 1-word text."""
        embedder = Embedder()
        emb = embedder.encode("Hello")
        assert len(emb) == 384
        assert isinstance(emb, list)

    def test_very_long_text_embedding(self):
        """Test embedding for long paragraph."""
        embedder = Embedder()
        long_text = "This is a very long sentence. " * 50
        emb = embedder.encode(long_text)
        assert len(emb) == 384
        assert isinstance(emb, list)

    def test_unicode_text_embedding(self):
        """Test Unicode/Urdu/Arabic text handling."""
        embedder = Embedder()
        urdu_text = "بریانی ایک مزیدار ڈش ہے"
        emb = embedder.encode(urdu_text)
        assert len(emb) == 384
        assert isinstance(emb, list)

    def test_metadata_filter_no_matches(self):
        """Search with filter that doesn't match anything."""
        db = HNSWDB()
        db.add([0.5] * 384, "Pakistani Food")
        db.add([0.6] * 384, "Pakistani Dessert")

        # Search with filter that doesn't match (post-filtering)
        results = db.search([0.5] * 384, top_k=5)
        # Filter manually (simulating API post-filtering)
        filtered = []
        for idx, score in results:
            meta = db.get_metadata(idx)
            if meta and "Italian" in meta:
                filtered.append((idx, score))
        assert len(filtered) == 0

    def test_metadata_filter_with_matches(self):
        """Search with filter that matches some results."""
        db = HNSWDB()
        db.add([0.5] * 384, "Pakistani Food")
        db.add([0.6] * 384, "Pakistani Dessert")
        db.add([0.7] * 384, "Italian Food")

        results = db.search([0.5] * 384, top_k=5)
        filtered = []
        for idx, score in results:
            meta = db.get_metadata(idx)
            if meta and "Pakistani" in meta:
                filtered.append((idx, score))

        assert len(filtered) >= 1
        assert "Pakistani" in db.get_metadata(filtered[0][0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
