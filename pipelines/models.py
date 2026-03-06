
import numpy as np
# Cache for pre-computed embeddings
_embedding_cache = {}

def compute_embeddings(texts, use_cache=True):
    """
    Compute embeddings for texts using Hugging Face Inference API, using cache if available.
    Args:
        texts: List of strings to embed
        use_cache: Whether to use/store in cache
    Returns:
        embeddings: Normalized embeddings array
    """
    import requests
    import os
    HF_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
    HF_TOKEN = os.getenv("HF_TOKEN")  # Set your token in environment
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    cache_key = tuple(texts)
    if use_cache and cache_key in _embedding_cache:
        return _embedding_cache[cache_key]
    payload = {"inputs": list(texts)}
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    embeddings = np.array(result, dtype=np.float32)
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norm, 1e-8, None)
    if use_cache:
        _embedding_cache[cache_key] = embeddings
    return embeddings

def clear_cache():
    """Clear embedding cache"""
    global _embedding_cache
    _embedding_cache = {}
