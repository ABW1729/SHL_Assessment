"""
Global model loading - load once, reuse everywhere
"""
from sentence_transformers import CrossEncoder, SentenceTransformer
import numpy as np

# Load models ONCE globally
_sentence_transformer = None
_cross_encoder = None

def get_sentence_transformer():
    """Get or initialize SentenceTransformer model globally"""
    global _sentence_transformer
    if _sentence_transformer is None:
        _sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
    return _sentence_transformer

def get_cross_encoder():
    """Get or initialize CrossEncoder model globally"""
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder

# Cache for pre-computed embeddings
_embedding_cache = {}

def compute_embeddings(texts, use_cache=True):
    """
    Compute embeddings for texts, using cache if available.
    
    Args:
        texts: List of strings to embed
        use_cache: Whether to use/store in cache
        
    Returns:
        embeddings: Normalized embeddings array
    """
    embedder = get_sentence_transformer()
    
    if not use_cache:
        return embedder.encode(texts, normalize_embeddings=True)
    
    # Create cache key from texts
    cache_key = tuple(texts)
    
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]
    
    # Compute and cache
    embeddings = embedder.encode(list(texts), normalize_embeddings=True)
    _embedding_cache[cache_key] = embeddings
    
    return embeddings

def clear_cache():
    """Clear embedding cache"""
    global _embedding_cache
    _embedding_cache = {}
