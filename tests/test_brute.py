"""
Unit tests for the BruteForceDB engine.
Run with: pytest tests/test_brute.py -v
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pytest
from src.brute_engine import BruteForceDB

def test_add_and_search():
    """Test basic insertion and search functionality."""
    db = BruteForceDB()
    
    db.add("1", [1.0, 0.0], "Vector A")
    db.add("2", [0.0, 1.0], "Vector B")
    db.add("3", [0.5, 0.5], "Vector C")
    
    results = db.search([1.0, 0.0], top_k=2)
    
    assert len(results) == 2
    assert results[0][0] == "1"
    assert results[0][1] == 1.0
    assert results[1][0] == "3"
    assert db.size() == 3

def test_empty_search():
    """Test search behavior when database is empty."""
    db = BruteForceDB()
    results = db.search([1.0, 1.0])
    assert results == []

def test_cosine_similarity_edge_cases():
    """Test edge cases like zero vectors."""
    db = BruteForceDB()
    db.add("zero", [0.0, 0.0], "Zero Vector")
    
    results = db.search([1.0, 1.0], top_k=1)
    assert results[0][1] == 0.0