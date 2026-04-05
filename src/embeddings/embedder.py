from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np


@dataclass(slots=True)
class EmbeddingModel:
    name: str
    dimension: int
    pooling: Literal["mean", "cls"] = "mean"


# Standard embedding models
MINILM_L6_V2 = EmbeddingModel(name="sentence-transformers/all-MiniLM-L6-v2", dimension=384)
BGE_BASE_EN = EmbeddingModel(name="BAAI/bge-base-en-v1.5", dimension=768)


def generate_embeddings(
    texts: list[str],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    normalize: bool = True,
) -> list[list[float]]:
    """Generate embeddings for a list of texts using a HuggingFace sentence-transformer model.
    
    Args:
        texts: List of text strings to embed.
        model_name: HuggingFace model identifier.
        normalize: Whether to L2-normalize embeddings.
    
    Returns:
        List of embedding vectors (one per input text).
    """
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Embeddings require sentence-transformers to be installed: pip install sentence-transformers"
        ) from exc
    
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, normalize_embeddings=normalize, convert_to_numpy=True)
    
    if isinstance(embeddings, np.ndarray):
        return embeddings.tolist()
    return embeddings


def get_embedding_dimension(model_name: str) -> int:
    """Get the output dimension of an embedding model without loading it fully."""
    
    if "minilm" in model_name.lower() or "all-minilm" in model_name.lower():
        return 384
    if "bge-base" in model_name.lower():
        return 768
    if "bge-large" in model_name.lower():
        return 1024
    if "openai" in model_name.lower() or "text-embedding" in model_name.lower():
        return 1536
    raise ValueError(f"Unknown embedding model: {model_name}")
