"""
Migration script: Convert pickle-based database to new JSON + NPY format.
Run once to migrate existing data.
"""

import pickle
import os
from src.persistence import VectorDBPersistence
from src.hnsw_engine import HNSWDB
from src.brute_engine import BruteForceDB


def migrate_hnsw_pickle():
    """Migrate HNSW database from pickle to new format."""
    pickle_file = "hnsw_db_data.pkl"
    if not os.path.exists(pickle_file):
        print(f"[Migration] No pickle file found: {pickle_file}")
        return

    print(f"[Migration] Loading from {pickle_file}...")
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    db = HNSWDB()
    db.vectors = data.get("vectors", {})
    db.metadata = data.get("metadata", {})
    db.neighbors = data.get("neighbors", {})
    db.levels = data.get("levels", {})
    db.entry_point = data.get("entry_point", None)
    db.max_level = data.get("max_level", 0)
    db.next_id = data.get("next_id", 0)

    db.save("hnsw_db_data")
    print("[Migration] Migration complete! New files: hnsw_db_data.meta.json and hnsw_db_data.vectors.npy")
    print("[Migration] You can now delete the old .pkl file if everything works.")


def migrate_bruteforce_pickle():
    """Migrate Brute-Force database from pickle to new format."""
    pickle_file = "vector_db_data.pkl"
    if not os.path.exists(pickle_file):
        print(f"[Migration] No pickle file found: {pickle_file}")
        return

    print(f"[Migration] Loading from {pickle_file}...")
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    db = BruteForceDB()
    db.vectors = data.get("vectors", {})
    db.metadata = data.get("metadata", {})

    db.save("vector_db_data")
    print("[Migration] Migration complete! New files: vector_db_data.meta.json and vector_db_data.vectors.npy")
    print("[Migration] You can now delete the old .pkl file if everything works.")


if __name__ == "__main__":
    print("=" * 50)
    print("   VectorDB Migration Tool")
    print("   Pickle -> JSON + NPY")
    print("=" * 50)

    if os.path.exists("hnsw_db_data.pkl"):
        migrate_hnsw_pickle()
    elif os.path.exists("vector_db_data.pkl"):
        migrate_bruteforce_pickle()
    else:
        print("[Migration] No pickle files found. Nothing to migrate.")