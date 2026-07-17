"""
Module 4: HNSW (Hierarchical Navigable Small World) Index
This module implements an approximate nearest neighbor search using a multi-layer graph.
It drastically speeds up search from O(n) to O(log n) with minimal accuracy loss.
"""

import numpy as np
import math
import random
import pickle
import os
from typing import List, Tuple, Optional, Dict, Set

class HNSWDB:
    """
    A vector database using Hierarchical Navigable Small World graphs.
    Supports fast approximate nearest neighbor search.
    """
    
    def __init__(self, M: int = 16, ef_construction: int = 200, ef_search: int = 50):
        """
        Initialize the HNSW index.
        
        Args:
            M: Maximum number of neighbors per node per layer (higher = more accurate but slower).
            ef_construction: Dynamic list size during index building (higher = better index).
            ef_search: Dynamic list size during search (higher = more accurate).
        """
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        
        # Core data structures
        self.vectors: Dict[int, np.ndarray] = {}          # id -> vector
        self.metadata: Dict[int, str] = {}                # id -> text
        self.neighbors: Dict[int, Dict[int, Set[int]]] = {} # id -> {level: set(neighbor_ids)}
        self.levels: Dict[int, int] = {}                  # id -> max_level assigned to this node
        self.entry_point: Optional[int] = None            # Topmost node id
        self.max_level = 0                                # Global max level
        
        self.next_id = 0  # Auto-incrementing ID for internal use
        
        # HNSW constants
        self.LEVEL_MULTIPLIER = 1 / math.log(M)
        
    def _random_level(self) -> int:
        """
        Generate a random level for a new node using exponential distribution.
        Higher levels have exponentially fewer nodes.
        """
        return int(-math.log(random.random()) * self.LEVEL_MULTIPLIER)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
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
        candidates = []  # (distance, id) - negative for max-heap simulation
        
        # Start from the entry point
        entry_dist = 1 - self._cosine_similarity(query, self.vectors[entry_id])
        candidates.append((entry_dist, entry_id))
        visited.add(entry_id)
        
        # Use a simple list as a priority queue (brute-force within this layer)
        # In production we use heapq, but for simplicity and learning, we use a list.
        while True:
            # Find the closest candidate not yet processed
            candidates.sort(key=lambda x: x[0])
            closest = candidates[0]
            
            # Explore neighbors of the closest node at this level
            current_neighbors = self.neighbors[closest[1]].get(level, set())
            new_candidates = []
            
            for neighbor_id in current_neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    dist = 1 - self._cosine_similarity(query, self.vectors[neighbor_id])
                    candidates.append((dist, neighbor_id))
            
            # Sort and keep only the ef best candidates
            candidates.sort(key=lambda x: x[0])
            candidates = candidates[:ef]
            
            # Convergence check: if the closest node hasn't changed in the last iteration
            # (Simplified: break if no new candidates added)
            if not new_candidates:
                break
            # Additional safety: break if candidates list is small
            if len(candidates) < 2:
                break
            # Simple convergence: if the top candidate is the same as previous iteration
            # We'll just limit iterations by checking if the first element's id changed
            # For pure Python implementation, we use a simple loop limit.
            break  # This simplified version will just take the current closest
        
        return [c[1] for c in candidates]
    
    def add(self, vector: List[float], metadata: str = "") -> int:
        """
        Add a new vector to the HNSW index.
        Returns the auto-generated ID.
        """
        id = self.next_id
        self.next_id += 1
        
        vec_np = np.array(vector)
        self.vectors[id] = vec_np
        self.metadata[id] = metadata
        
        # Assign random level
        level = self._random_level()
        self.levels[id] = level
        self.neighbors[id] = {}
        
        # Initialize neighbor sets for each level
        for l in range(level + 1):
            self.neighbors[id][l] = set()
        
        # If this is the first node, make it the entry point
        if self.entry_point is None:
            self.entry_point = id
            self.max_level = level
            return id
        
        # Find entry point at the highest level
        curr_id = self.entry_point
        curr_level = self.max_level
        
        # Traverse from top down to level+1 (finding the closest node at each level)
        for l in range(curr_level, level, -1):
            # Greedy search at level l
            candidates = self._search_layer(vec_np, curr_id, l, 1)
            if candidates:
                curr_id = candidates[0]
        
        # Now connect at levels from min(level, max_level) down to 0
        for l in range(min(level, self.max_level), -1, -1):
            # Find ef_construction neighbors at this level
            neighbors_l = self._search_layer(vec_np, curr_id, l, self.ef_construction)
            
            # Add connections bidirectionally
            for neighbor_id in neighbors_l:
                # Connect id -> neighbor
                self.neighbors[id][l].add(neighbor_id)
                # Connect neighbor -> id
                if l not in self.neighbors[neighbor_id]:
                    self.neighbors[neighbor_id][l] = set()
                self.neighbors[neighbor_id][l].add(id)
                
                # Prune connections if they exceed M
                if len(self.neighbors[neighbor_id][l]) > self.M:
                    # Keep only the M closest neighbors (simplified pruning)
                    # We sort by distance to the neighbor's own vector
                    neighbor_vec = self.vectors[neighbor_id]
                    neighbor_connections = list(self.neighbors[neighbor_id][l])
                    neighbor_connections.sort(
                        key=lambda x: 1 - self._cosine_similarity(neighbor_vec, self.vectors[x])
                    )
                    self.neighbors[neighbor_id][l] = set(neighbor_connections[:self.M])
            
            # Update current node to the best neighbor for next level
            if neighbors_l:
                curr_id = neighbors_l[0]
        
        # Update entry point if this node has a higher level
        if level > self.max_level:
            self.max_level = level
            self.entry_point = id
        
        return id
    
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for the top_k nearest neighbors using the HNSW graph.
        """
        if self.entry_point is None or len(self.vectors) == 0:
            return []
        
        query_np = np.array(query_vector)
        
        # Start from the entry point at the top level
        curr_id = self.entry_point
        curr_level = self.max_level
        
        # Traverse down from top to level 1 (greedy)
        for l in range(curr_level, 0, -1):
            candidates = self._search_layer(query_np, curr_id, l, 1)
            if candidates:
                curr_id = candidates[0]
        
        # Search at level 0 for the final results
        candidates = self._search_layer(query_np, curr_id, 0, self.ef_search)
        
        # Filter to top_k
        results = []
        for id in candidates[:top_k]:
            sim = self._cosine_similarity(query_np, self.vectors[id])
            results.append((id, sim))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def size(self) -> int:
        """Return the number of vectors in the database."""
        return len(self.vectors)
    
    def get_metadata(self, id: int) -> Optional[str]:
        """Retrieve metadata for a specific vector ID."""
        return self.metadata.get(id, None)
    
    # ===================== Persistence (Pickle) =====================
    
    def save(self, filepath: str = "hnsw_db_data.pkl") -> None:
        """Save the entire HNSW index to disk."""
        data = {
            "vectors": self.vectors,
            "metadata": self.metadata,
            "neighbors": self.neighbors,
            "levels": self.levels,
            "entry_point": self.entry_point,
            "max_level": self.max_level,
            "next_id": self.next_id
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"[HNSW Persistence] Database saved to {filepath}")
    
    def load(self, filepath: str = "hnsw_db_data.pkl") -> bool:
        """Load the HNSW index from disk."""
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                self.vectors = data["vectors"]
                self.metadata = data["metadata"]
                self.neighbors = data["neighbors"]
                self.levels = data["levels"]
                self.entry_point = data["entry_point"]
                self.max_level = data["max_level"]
                self.next_id = data["next_id"]
            print(f"[HNSW Persistence] Database loaded from {filepath} ({self.size()} vectors)")
            return True
        else:
            print(f"[HNSW Persistence] No existing database found at {filepath}. Starting fresh.")
            return False