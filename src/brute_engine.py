"""
Module 1 & 3: Brute-Force Vector Search Engine with Persistence
This module implements a naive vector database with disk persistence using pickle.
"""

import numpy as np
from typing import List, Tuple, Optional
import pickle
import os

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

    def add(self, id: str, vector: List[float], metadata: str = "") -> None:
        """
        Add a vector to the database.
        
        Args:
            id: Unique identifier for the vector.
            vector: List of floats representing the vector/embedding.
            metadata: Optional string (e.g., original text) associated with the vector.
        """
        self.vectors[id] = np.array(vector)
        self.metadata[id] = metadata

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate the cosine similarity between two vectors.
        Formula: dot(A, B) / (||A|| * ||B||)
        """
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        
        return dot_product / (norm_vec1 * norm_vec2)

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find the top_k most similar vectors to the query vector.
        """
        query_np = np.array(query_vector)
        results = []
        
        for idx, vec in self.vectors.items():
            sim_score = self._cosine_similarity(query_np, vec)
            results.append((idx, sim_score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def size(self) -> int:
        """Return the number of vectors stored in the database."""
        return len(self.vectors)

    def get_metadata(self, id: str) -> Optional[str]:
        """Retrieve metadata for a specific vector ID."""
        return self.metadata.get(id, None)

    # ===================== Persistence Methods (Module 3) =====================

    def save(self, filepath: str = "vector_db_data.pkl") -> None:
        """
        Save the current database state to a pickle file on disk.
        """
        data = {
            "vectors": self.vectors,
            "metadata": self.metadata
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"[Persistence] Database saved to {filepath}")

    def load(self, filepath: str = "vector_db_data.pkl") -> bool:
        """
        Load the database state from a pickle file on disk.
        """
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                self.vectors = data["vectors"]
                self.metadata = data["metadata"]
            print(f"[Persistence] Database loaded from {filepath} ({self.size()} vectors)")
            return True
        else:
            print(f"[Persistence] No existing database found at {filepath}. Starting fresh.")
            return False