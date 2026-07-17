"""
Unit tests for the BruteForceDB engine.
Run with: pytest tests/test_brute.py -v
"""

import sys
import os
# Add the src directory to path to import the module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pytest
from src.brute_engine import BruteForceDB

def test_add_and_search():
    """Test basic insertion and search functionality."""
    db = BruteForceDB()
    
    # Add some sample vectors (Dimension 2 for simplicity)
    db.add("1", [1.0, 0.0], "Vector A - Horizontal")
    db.add("2", [0.0, 1.0], "Vector B - Vertical")
    db.add("3", [0.5, 0.5], "Vector C - Diagonal 45 deg")
    
    # Search for a vector exactly like "Vector A"
    results = db.search([1.0, 0.0], top_k=2)
    
    # Assertions
    assert len(results) == 2, "Should return top 2 results"
    assert results[0][0] == "1", "Most similar should be itself (ID 1)"
    assert results[0][1] == 1.0, "Cosine similarity with itself should be 1.0"
    assert results[1][0] == "3", "Second most similar should be [0.5, 0.5]"
    
    # Check size
    assert db.size() == 3, "Database size should be 3"

def test_empty_search():
    """Test search behavior when database is empty."""
    db = BruteForceDB()
    results = db.search([1.0, 1.0])
    assert results == [], "Search on empty DB should return empty list"

def test_cosine_similarity_edge_cases():
    """Test edge cases like zero vectors."""
    db = BruteForceDB()
    db.add("zero", [0.0, 0.0], "Zero Vector")
    
    results = db.search([1.0, 1.0], top_k=1)
    # Similarity should be 0.0 to avoid division by zero
    assert results[0][1] == 0.0, "Similarity with zero vector should be 0.0"