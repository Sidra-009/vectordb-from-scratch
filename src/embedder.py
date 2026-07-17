"""
Module 5: Text Embedding Service
This module loads a Sentence-Transformer model to convert text into vectors.
"""

from sentence_transformers import SentenceTransformer
from typing import List

class Embedder:
    """
    A wrapper class for Sentence-Transformer models.
    Converts raw text into numerical vectors (embeddings).
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder with a specific model.
        
        Args:
            model_name: Name of the Sentence-Transformer model.
                       'all-MiniLM-L6-v2' is small (~80MB) and fast, good for development.
        """
        print(f"[AI Model] Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"[AI Model] Loaded successfully. Embedding dimension: {self.dimension}")

    def encode(self, text: str) -> List[float]:
        """
        Convert a single text string into a vector (list of floats).
        
        Args:
            text: The input text (can be Roman Urdu, English, Urdu).
            
        Returns:
            List of floats representing the embedding.
        """
        if not text or text.strip() == "":
            print("[AI Model] Warning: Empty text received, returning zero vector.")
            return [0.0] * self.dimension
        
        # Convert to embedding and convert numpy array to list
        embedding = self.model.encode(text, normalize_embeddings=True)  # Normalized for cosine similarity
        return embedding.tolist()

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Convert multiple texts to vectors (for bulk processing).
        """
        if not texts:
            return []
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]