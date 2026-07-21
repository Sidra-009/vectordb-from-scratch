"""
Persistence layer for VectorDB.
Replaces raw Pickle with JSON + NumPy .npy format.
Features: atomic writes, checksum validation, crash recovery.
"""

import json
import os
import shutil
import tempfile
import hashlib
import numpy as np
from typing import Dict, Any, Tuple, Optional


def _convert_sets_to_lists(obj):
    """Recursively convert all sets to lists."""
    if isinstance(obj, dict):
        return {k: _convert_sets_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [_convert_sets_to_lists(item) for item in obj]
    elif isinstance(obj, set):
        return list(obj)
    else:
        return obj


class VectorDBPersistence:
    """
    Handles saving and loading of vector database to disk.
    Uses JSON for metadata and .npy for vector arrays.
    Supports atomic writes and integrity checks.
    """

    def __init__(self, filepath: str):
        self.base_path = filepath
        self.meta_path = f"{filepath}.meta.json"
        self.vectors_path = f"{filepath}.vectors.npy"
        self.checksum_path = f"{filepath}.checksum"

    def save(self, vectors: Dict, metadata: Dict) -> bool:
        """Save database with atomic writes."""
        tmp_vectors_path = None
        tmp_meta_path = None

        try:
            metadata_serializable = _convert_sets_to_lists(metadata)

            ids = list(vectors.keys())
            vector_list = [vectors[id] for id in ids]
            vector_array = np.array(vector_list, dtype=np.float32)

            meta_data = {
                "ids": ids,
                "metadata": metadata_serializable,
                "dimension": vector_array.shape[1] if len(ids) > 0 else 0,
                "num_vectors": len(ids),
                "version": "2.0",
            }

            # Write vectors
            with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".npy") as tmp_v:
                np.save(tmp_v, vector_array)
                tmp_vectors_path = tmp_v.name

            # Write metadata
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp_m:
                json.dump(meta_data, tmp_m, indent=2)
                tmp_meta_path = tmp_m.name

            # Checksum
            checksum = self._compute_checksum(tmp_vectors_path, tmp_meta_path)

            # Atomic rename
            os.replace(tmp_vectors_path, self.vectors_path)
            os.replace(tmp_meta_path, self.meta_path)

            with open(self.checksum_path, "w") as f:
                json.dump({"checksum": checksum, "version": "2.0"}, f)

            return True

        except Exception as e:
            print(f"[Persistence] ERROR: {e}")
            # Clean up temp files
            for path in [tmp_vectors_path, tmp_meta_path]:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except:
                        pass
            return False

    def load(self) -> Tuple[Dict, Dict]:
        """Load database with integrity check."""
        if not os.path.exists(self.meta_path) or not os.path.exists(self.vectors_path):
            print("[Persistence] No existing database found. Starting fresh.")
            return {}, {}

        try:
            if not self._verify_checksum():
                print("[Persistence] Checksum failed. Attempting recovery...")
                if self._try_recovery():
                    print("[Persistence] Recovery successful.")
                else:
                    print("[Persistence] Recovery failed. Starting empty.")
                    return {}, {}

            with open(self.meta_path, "r") as f:
                meta_data = json.load(f)

            ids = meta_data["ids"]
            metadata = meta_data["metadata"]
            vector_array = np.load(self.vectors_path)

            vectors = {}
            for i, idx in enumerate(ids):
                vectors[idx] = vector_array[i]

            print(f"[Persistence] Loaded {len(vectors)} vectors")
            return vectors, metadata

        except Exception as e:
            print(f"[Persistence] Load error: {e}")
            return {}, {}

    def _compute_checksum(self, v_path: str, m_path: str) -> str:
        sha256 = hashlib.sha256()
        with open(v_path, "rb") as f:
            sha256.update(f.read())
        with open(m_path, "rb") as f:
            sha256.update(f.read())
        return sha256.hexdigest()

    def _verify_checksum(self) -> bool:
        if not os.path.exists(self.checksum_path):
            return True
        try:
            with open(self.checksum_path, "r") as f:
                stored = json.load(f)
            expected = stored.get("checksum", "")
            actual = self._compute_checksum(self.vectors_path, self.meta_path)
            return actual == expected
        except:
            return False

    def _try_recovery(self) -> bool:
        backup_files = [f for f in os.listdir(".") if f.startswith(self.base_path) and f.endswith(".bak")]
        if not backup_files:
            return False
        for backup in backup_files:
            if ".meta" in backup:
                shutil.copy(backup, self.meta_path)
            elif ".vectors" in backup:
                shutil.copy(backup, self.vectors_path)
        return self._verify_checksum()