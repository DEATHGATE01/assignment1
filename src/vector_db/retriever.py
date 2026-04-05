from __future__ import annotations

from typing import Literal, Union

from src.embeddings.compare import EmbeddedChunk
from src.embeddings.embedder import generate_embeddings
from src.vector_db.chroma_db import ChromaVectorDB
from src.vector_db.faiss_db import FAISSVectorDB


VectorDBType = Union[FAISSVectorDB, ChromaVectorDB]


class VectorRetriever:
    """Unified retrieval interface for FAISS and ChromaDB."""

    def __init__(
        self,
        db: VectorDBType,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """Initialize retriever with a vector database.
        
        Args:
            db: FAISS or ChromaDB instance.
            embedding_model: Model to use for query embedding.
        """
        self.db = db
        self.embedding_model = embedding_model

    def retrieve(self, query: str, k: int = 5) -> list[EmbeddedChunk]:
        """Retrieve top-k chunks for a query using embeddings.
        
        Args:
            query: Query text.
            k: Number of results to return.
        
        Returns:
            List of top-k relevant chunks.
        """
        query_embedding = generate_embeddings(
            [query],
            model_name=self.embedding_model,
            normalize=True,
        )

        if isinstance(self.db, ChromaVectorDB):
            return self.db.search(query_embedding[0], k=k)
        else:
            return self.db.search(query_embedding[0], k=k)

    def retrieve_by_embedding(self, embedding: list[float], k: int = 5) -> list[EmbeddedChunk]:
        """Retrieve top-k chunks using a pre-computed embedding.
        
        Args:
            embedding: Query embedding vector.
            k: Number of results to return.
        
        Returns:
            List of top-k relevant chunks.
        """
        return self.db.search(embedding, k=k)
