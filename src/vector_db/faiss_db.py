from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.embeddings.compare import EmbeddedChunk
from src.utils.config import RESULTS_DIR, ensure_project_directories


class FAISSVectorDB:
    """FAISS-based vector database for fast similarity search."""

    def __init__(self, dimension: int, index_path: Path | None = None):
        """Initialize FAISS index.
        
        Args:
            dimension: Embedding dimension.
            index_path: Optional path to load/save index.
        """
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "FAISS requires faiss-cpu or faiss-gpu to be installed: pip install faiss-cpu"
            ) from exc

        self.faiss = faiss
        self.dimension = dimension
        self.index_path = index_path
        self.index = self.faiss.IndexFlatL2(dimension)
        self.metadata: list[dict[str, Any]] = []

    def add(self, chunks: list[EmbeddedChunk]) -> None:
        """Add embedded chunks to the index."""
        if not chunks:
            return

        import numpy as np

        embeddings = np.array([chunk.embedding for chunk in chunks], dtype=np.float32)
        self.index.add(embeddings)
        self.metadata.extend([asdict(chunk) for chunk in chunks])

    def search(self, query_embedding: list[float], k: int = 5) -> list[EmbeddedChunk]:
        """Search for top-k similar chunks."""
        import numpy as np

        query = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query, k)

        results: list[EmbeddedChunk] = []
        for idx in indices[0]:
            if 0 <= idx < len(self.metadata):
                meta = self.metadata[int(idx)]
                results.append(
                    EmbeddedChunk(
                        source_name=meta["source_name"],
                        source_path=meta["source_path"],
                        extension=meta["extension"],
                        chunk_id=meta["chunk_id"],
                        chunk_type=meta["chunk_type"],
                        text=meta["text"],
                        start_index=meta["start_index"],
                        end_index=meta["end_index"],
                        embedding=meta["embedding"],
                    )
                )
        return results

    def save(self, output_dir: Path | None = None) -> Path:
        """Save index and metadata to disk."""
        ensure_project_directories()
        output_dir = output_dir or (RESULTS_DIR / "vector_db" / "faiss")
        output_dir.mkdir(parents=True, exist_ok=True)

        index_path = output_dir / "index.faiss"
        metadata_path = output_dir / "metadata.json"

        self.faiss.write_index(self.index, str(index_path))
        metadata_path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")
        return output_dir

    @staticmethod
    def load(index_dir: Path) -> FAISSVectorDB:
        """Load index and metadata from disk."""
        import faiss

        index_path = index_dir / "index.faiss"
        metadata_path = index_dir / "metadata.json"

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Index or metadata not found in {index_dir}")

        index = faiss.read_index(str(index_path))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        db = FAISSVectorDB(dimension=index.d)
        db.index = index
        db.metadata = metadata
        return db
