from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.chunking.chunker import Chunk
from src.embeddings.embedder import generate_embeddings
from src.utils.config import RESULTS_DIR, ensure_project_directories


@dataclass(slots=True)
class EmbeddedChunk:
    source_name: str
    source_path: str
    extension: str
    chunk_id: int
    chunk_type: str
    text: str
    start_index: int
    end_index: int
    embedding: list[float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class EmbeddingExperimentResult:
    model_name: str
    chunk_count: int
    embedding_dimension: int
    average_embedding_L2_norm: float
    min_embedding_L2_norm: float
    max_embedding_L2_norm: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _embedding_norms(embeddings: list[list[float]]) -> tuple[float, float, float]:
    """Compute L2 norms for embeddings."""
    
    import numpy as np

    norms = [np.linalg.norm(e) for e in embeddings]
    if not norms:
        return 0.0, 0.0, 0.0
    return float(np.mean(norms)), float(min(norms)), float(max(norms))


def embed_chunks(
    chunks: list[Chunk],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> list[EmbeddedChunk]:
    """Embed a list of chunks using the specified model."""

    texts = [chunk.text for chunk in chunks]
    embeddings = generate_embeddings(texts, model_name=model_name)

    embedded: list[EmbeddedChunk] = []
    for chunk, embedding in zip(chunks, embeddings):
        embedded.append(
            EmbeddedChunk(
                source_name=chunk.source_name,
                source_path=chunk.source_path,
                extension=chunk.extension,
                chunk_id=chunk.chunk_id,
                chunk_type=chunk.chunk_type,
                text=chunk.text,
                start_index=chunk.start_index,
                end_index=chunk.end_index,
                embedding=embedding,
            )
        )
    return embedded


def run_embedding_experiment(
    chunks: list[Chunk],
    *,
    model_name: str,
    output_dir: Path | None = None,
) -> EmbeddingExperimentResult:
    """Run an embedding experiment and optionally persist the result."""

    ensure_project_directories()
    output_dir = output_dir or (RESULTS_DIR / "comparisons")
    output_dir.mkdir(parents=True, exist_ok=True)

    embedded_chunks = embed_chunks(chunks, model_name=model_name)
    embeddings = [c.embedding for c in embedded_chunks]
    average_norm, min_norm, max_norm = _embedding_norms(embeddings)

    result = EmbeddingExperimentResult(
        model_name=model_name,
        chunk_count=len(embedded_chunks),
        embedding_dimension=len(embeddings[0]) if embeddings else 0,
        average_embedding_L2_norm=average_norm,
        min_embedding_L2_norm=min_norm,
        max_embedding_L2_norm=max_norm,
    )

    safe_model_name = model_name.replace("/", "_").replace("-", "_")
    output_path = output_dir / f"embedding_{safe_model_name}.json"
    output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result
