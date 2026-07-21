"""
Module 1 & 3: Brute-Force Vector Search Engine with Persistence
This module implements a naive vector database with disk persistence using JSON + NPY.
"""

import numpy as np
from typing import List, Tuple, Optional
import os
from src.persistence import VectorDBPersistence  # <-- NEW IMPORT

class BruteForceDB:
    """
    A naive vector database implementation using brute-force search.
    Stores vectors in memory and computes cosine similarity against all vectors
    for every search query. O(n) complexity.
    Supports saving and loading to/from disk.
    """

    def __init__(self):
        # Dictionary to store vector ID -> numpy array
        self.vectors: dict[str, np.ndarray] = {}
        # Dictionary to store vector ID -> original text/metadata
        self.metadata: dict[str, str] = {}

    # ... (add, _cosine_similarity, search, size, get_metadata methods same)

    
    # Persistence Methods (NEW: JSON + NPY)
    
    def save(self, filepath: str = "vector_db_data") -> None:
        """
        Save the current database state to disk using JSON + NPY format.
        """
        persister = VectorDBPersistence(filepath)
        success = persister.save(self.vectors, self.metadata)

        if success:
            print(f"[Persistence] Database saved to {filepath}.meta.json and .vectors.npy")
        else:
            print(f"[Persistence] ERROR: Failed to save database!")

    def load(self, filepath: str = "vector_db_data") -> bool:
        """
        Load the database state from disk using JSON + NPY format.
        
        Returns:
            True if file loaded successfully, False if file not found.
        """
        persister = VectorDBPersistence(filepath)
        vectors_loaded, metadata_loaded = persister.load()

        if not vectors_loaded:
            print(f"[Persistence] No existing database found. Starting fresh.")
            return False

        self.vectors = vectors_loaded
        self.metadata = metadata_loaded

        print(f"[Persistence] Loaded from {filepath}.meta.json and .vectors.npy ({len(self.vectors)} vectors)")
        return True
