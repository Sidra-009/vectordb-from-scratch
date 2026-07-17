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
        # Convert list to numpy array for faster mathematical operations
        self.vectors[id] = np.array(vector)
        self.metadata[id] = metadata

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate the cosine similarity between two vectors.
        Formula: dot(A, B) / (||A|| * ||B||)
        
        Cosine similarity ranges from -1 (opposite) to 1 (identical direction).
        For text embeddings (which are usually positive), it's often between 0 and 1.
        """
        # Calculate dot product
        dot_product = np.dot(vec1, vec2)
        # Calculate magnitudes (norms)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        # Avoid division by zero (if vector is all zeros)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        
        return dot_product / (norm_vec1 * norm_vec2)

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find the top_k most similar vectors to the query vector.
        
        Args:
            query_vector: List of floats to search against.
            top_k: Number of results to return.
        
        Returns:
            List of tuples (id, similarity_score) sorted by similarity descending.
        """
        # Convert query to numpy array
        query_np = np.array(query_vector)
        results = []
        
        # Iterate over every vector in the database (This is the Brute-Force part)
        for idx, vec in self.vectors.items():
            sim_score = self._cosine_similarity(query_np, vec)
            results.append((idx, sim_score))
        
        # Sort by similarity score (highest first) and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def size(self) -> int:
        """Return the number of vectors stored in the database."""
        return len(self.vectors)

    def get_metadata(self, id: str) -> Optional[str]:
        """Retrieve metadata for a specific vector ID."""
        return self.metadata.get(id, None)

    # ===================== NEW: Persistence Methods (Module 3) =====================

    def save(self, filepath: str = "sidradb_data.pkl") -> None:
        """
        Save the current database state to a pickle file on disk.
        
        Args:
            filepath: Path where the .pkl file will be saved (default: sidradb_data.pkl)
        """
        # Prepare data to be serialized
        data = {
            "vectors": self.vectors,   # dict of id -> numpy array
            "metadata": self.metadata  # dict of id -> string
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"[Persistence] Database saved to {filepath}")

    def load(self, filepath: str = "sidradb_data.pkl") -> bool:
        """
        Load the database state from a pickle file on disk.
        
        Args:
            filepath: Path to the .pkl file to load.
            
        Returns:
            True if file loaded successfully, False if file not found.
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