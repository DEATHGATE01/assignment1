from __future__ import annotations

from pathlib import Path
from typing import Any

from src.embeddings.compare import EmbeddedChunk
from src.utils.config import RESULTS_DIR, ensure_project_directories


class ChromaVectorDB:
    """ChromaDB-based vector database with persistent storage and metadata filtering."""

    def __init__(self, collection_name: str = "default", persist_dir: Path | None = None):
        """Initialize ChromaDB client and collection.
        
        Args:
            collection_name: Name of the collection.
            persist_dir: Directory for persistent storage.
        """
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "ChromaDB requires chromadb to be installed: pip install chromadb"
            ) from exc

        ensure_project_directories()
        persist_dir = persist_dir or (RESULTS_DIR / "vector_db" / "chroma")
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.chroma = chromadb
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.collection_name = collection_name
        self.persist_dir = persist_dir

    def add(self, chunks: list[EmbeddedChunk]) -> None:
        """Add embedded chunks to the collection."""
        if not chunks:
            return

        ids = [f"{chunk.source_name}_{chunk.chunk_id}" for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks]
        metadatas = [
            {
                "source_name": chunk.source_name,
                "source_path": chunk.source_path,
                "extension": chunk.extension,
                "chunk_type": chunk.chunk_type,
                "chunk_id": str(chunk.chunk_id),
                "start_index": str(chunk.start_index),
                "end_index": str(chunk.end_index),
            }
            for chunk in chunks
        ]
        documents = [chunk.text for chunk in chunks]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

    def search(self, query_embedding: list[float], k: int = 5) -> list[EmbeddedChunk]:
        """Search for top-k similar chunks."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )

        chunks: list[EmbeddedChunk] = []
        if results["ids"] and len(results["ids"]) > 0:
            for idx, doc_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][idx]
                document = results["documents"][0][idx]
                embedding = results["embeddings"][0][idx] if results["embeddings"] else [0.0]

                chunks.append(
                    EmbeddedChunk(
                        source_name=meta["source_name"],
                        source_path=meta["source_path"],
                        extension=meta["extension"],
                        chunk_id=int(meta["chunk_id"]),
                        chunk_type=meta["chunk_type"],
                        text=document,
                        start_index=int(meta["start_index"]),
                        end_index=int(meta["end_index"]),
                        embedding=embedding,
                    )
                )
        return chunks

    def search_by_text(self, query_text: str, k: int = 5) -> list[EmbeddedChunk]:
        """Search using raw text (ChromaDB will embed it internally if models are loaded)."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=k,
        )

        chunks: list[EmbeddedChunk] = []
        if results["ids"] and len(results["ids"]) > 0:
            for idx, doc_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][idx]
                document = results["documents"][0][idx]
                embedding = results["embeddings"][0][idx] if results["embeddings"] else [0.0]

                chunks.append(
                    EmbeddedChunk(
                        source_name=meta["source_name"],
                        source_path=meta["source_path"],
                        extension=meta["extension"],
                        chunk_id=int(meta["chunk_id"]),
                        chunk_type=meta["chunk_type"],
                        text=document,
                        start_index=int(meta["start_index"]),
                        end_index=int(meta["end_index"]),
                        embedding=embedding,
                    )
                )
        return chunks

    def get_collection_info(self) -> dict[str, Any]:
        """Get information about the collection."""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "chunk_count": count,
            "persist_dir": str(self.persist_dir),
        }
