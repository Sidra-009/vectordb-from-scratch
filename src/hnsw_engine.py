"""
Module 4: HNSW (Hierarchical Navigable Small World) Index
This module implements an approximate nearest neighbor search using a multi-layer graph.
"""

import threading
import numpy as np
import math
import random
import os
from typing import List, Tuple, Optional, Dict, Set
from src.persistence import VectorDBPersistence


class HNSWDB:
    """
    A vector database using Hierarchical Navigable Small World graphs.
    Supports fast approximate nearest neighbor search.
    """

    def __init__(self, M: int = 16, ef_construction: int = 200, ef_search: int = 50):
        """
        Initialize the HNSW index.

        Args:
            M: Maximum number of neighbors per node per layer.
            ef_construction: Dynamic list size during index building.
            ef_search: Dynamic list size during search.
        """
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        # Core data structures
        self.vectors: Dict[int, np.ndarray] = {}
        self.metadata: Dict[int, str] = {}
        self.neighbors: Dict[int, Dict[int, Set[int]]] = {}
        self.levels: Dict[int, int] = {}
        self.entry_point: Optional[int] = None
        self.max_level = 0
        self.next_id = 0

        # Thread-safety lock
        self._lock = threading.RLock()

        # HNSW constants
        self.LEVEL_MULTIPLIER = 1 / math.log(M)

    def _random_level(self) -> int:
        """
        Generate a random level for a new node using exponential distribution.
        Higher levels have exponentially fewer nodes.
        """
        return int(-math.log(random.random()) * self.LEVEL_MULTIPLIER)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        Returns a value between -1 and 1.
        """
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def _search_layer(self, query: np.ndarray, entry_id: int, level: int, ef: int) -> List[int]:
        """
        Perform a greedy search on a single layer to find the ef nearest neighbors.
        This is the core traversal algorithm of HNSW.
        """
        visited = set()
        candidates = []

        # Ensure entry_id has neighbors entry
        if entry_id not in self.neighbors:
            self.neighbors[entry_id] = {}

        entry_dist = 1 - self._cosine_similarity(query, self.vectors[entry_id])
        candidates.append((entry_dist, entry_id))
        visited.add(entry_id)

        while True:
            candidates.sort(key=lambda x: x[0])
            closest = candidates[0]

            current_neighbors = self.neighbors[closest[1]].get(level, set())
            new_candidates = []

            for neighbor_id in current_neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    dist = 1 - self._cosine_similarity(query, self.vectors[neighbor_id])
                    candidates.append((dist, neighbor_id))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[:ef]

            if not new_candidates:
                break
            if len(candidates) < 2:
                break

            break

        return [c[1] for c in candidates]

    def add(self, vector: List[float], metadata: str = "") -> int:
        """
        Add a new vector to the HNSW index.
        Returns the auto-generated ID.
        Thread-safe with RLock.
        """
        with self._lock:
            idx = self.next_id
            self.next_id += 1

            vec_np = np.array(vector)
            self.vectors[idx] = vec_np
            self.metadata[idx] = metadata

            level = self._random_level()
            self.levels[idx] = level
            self.neighbors[idx] = {}

            for l in range(level + 1):
                self.neighbors[idx][l] = set()

            if self.entry_point is None:
                self.entry_point = idx
                self.max_level = level
                return idx

            curr_id = self.entry_point
            curr_level = self.max_level

            for l in range(curr_level, level, -1):
                candidates = self._search_layer(vec_np, curr_id, l, 1)
                if candidates:
                    curr_id = candidates[0]

            for l in range(min(level, self.max_level), -1, -1):
                neighbors_l = self._search_layer(vec_np, curr_id, l, self.ef_construction)

                for neighbor_id in neighbors_l:
                    self.neighbors[idx][l].add(neighbor_id)

                    # Ensure neighbor_id exists in self.neighbors
                    if neighbor_id not in self.neighbors:
                        self.neighbors[neighbor_id] = {}
                    if l not in self.neighbors[neighbor_id]:
                        self.neighbors[neighbor_id][l] = set()
                    self.neighbors[neighbor_id][l].add(idx)

                    if len(self.neighbors[neighbor_id][l]) > self.M:
                        neighbor_vec = self.vectors[neighbor_id]
                        neighbor_connections = list(self.neighbors[neighbor_id][l])
                        neighbor_connections.sort(
                            key=lambda x: 1 - self._cosine_similarity(neighbor_vec, self.vectors[x])
                        )
                        self.neighbors[neighbor_id][l] = set(neighbor_connections[:self.M])

                if neighbors_l:
                    curr_id = neighbors_l[0]

            if level > self.max_level:
                self.max_level = level
                self.entry_point = idx

            return idx

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for the top_k nearest neighbors using the HNSW graph.
        Thread-safe with RLock.
        """
        with self._lock:
            if self.entry_point is None or len(self.vectors) == 0:
                return []

            query_np = np.array(query_vector)

            curr_id = self.entry_point
            curr_level = self.max_level

            for l in range(curr_level, 0, -1):
                candidates = self._search_layer(query_np, curr_id, l, 1)
                if candidates:
                    curr_id = candidates[0]

            candidates = self._search_layer(query_np, curr_id, 0, self.ef_search)

            results = []
            for idx in candidates[:top_k]:
                sim = self._cosine_similarity(query_np, self.vectors[idx])
                results.append((idx, sim))

            results.sort(key=lambda x: x[1], reverse=True)
            return results

    def size(self) -> int:
        """Return the number of vectors in the database."""
        return len(self.vectors)

    def get_metadata(self, idx: int) -> Optional[str]:
        """Retrieve metadata for a specific vector ID."""
        return self.metadata.get(idx, None)

    # ===================== Persistence Methods =====================

    def save(self, filepath: str = "hnsw_db_data") -> None:
        """
        Save the entire HNSW index to disk using JSON + NPY format.
        Thread-safe with RLock.

        Args:
            filepath: Base path (without extension).
                     Creates {filepath}.meta.json and {filepath}.vectors.npy
        """
        with self._lock:
            vectors_to_save = {}
            for idx, vec in self.vectors.items():
                vectors_to_save[idx] = vec

            metadata_to_save = self.metadata

            graph_data = {
                "neighbors": self.neighbors,
                "levels": self.levels,
                "entry_point": self.entry_point,
                "max_level": self.max_level,
                "next_id": self.next_id,
            }

            full_metadata = {
                "data": metadata_to_save,
                "graph": graph_data,
            }

            persister = VectorDBPersistence(filepath)
            success = persister.save(vectors_to_save, full_metadata)

            if success:
                print(f"[HNSW Persistence] Database saved to {filepath}.meta.json and .vectors.npy")
            else:
                print(f"[HNSW Persistence] ERROR: Failed to save database!")

    def load(self, filepath: str = "hnsw_db_data") -> bool:
        """
        Load the HNSW index from disk using JSON + NPY format.
        Thread-safe with RLock.

        Args:
            filepath: Base path (without extension).

        Returns:
            True if loaded successfully, False if no existing data.
        """
        with self._lock:
            persister = VectorDBPersistence(filepath)
            vectors_loaded, metadata_loaded = persister.load()

            if not vectors_loaded:
                print(f"[HNSW Persistence] No existing database found. Starting fresh.")
                return False

            self.vectors = vectors_loaded

            if "data" in metadata_loaded:
                self.metadata = metadata_loaded["data"]
            else:
                self.metadata = metadata_loaded

            if "graph" in metadata_loaded:
                graph_data = metadata_loaded["graph"]
                self.neighbors = graph_data.get("neighbors", {})
                self.levels = graph_data.get("levels", {})
                self.entry_point = graph_data.get("entry_point", None)
                self.max_level = graph_data.get("max_level", 0)
                self.next_id = graph_data.get("next_id", max(self.vectors.keys()) + 1 if self.vectors else 0)

            # Ensure every vector has an entry in neighbors
            for vid in self.vectors.keys():
                if vid not in self.neighbors:
                    self.neighbors[vid] = {}

            print(f"[HNSW Persistence] Loaded from {filepath}.meta.json and .vectors.npy ({len(self.vectors)} vectors)")
            return True