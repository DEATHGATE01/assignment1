from __future__ import annotations

from src.embeddings.compare import EmbeddedChunk
from src.vector_db.retriever import VectorRetriever


class SemanticSearch:
    """Semantic search using vector embeddings and similarity metrics."""

    def __init__(self, retriever: VectorRetriever):
        """Initialize semantic search with a retriever.
        
        Args:
            retriever: VectorRetriever instance for similarity search.
        """
        self.retriever = retriever

    def search(
        self,
        query: str,
        k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[EmbeddedChunk]:
        """Perform semantic search.
        
        Args:
            query: Query text.
            k: Number of results to return.
            min_similarity: Minimum similarity threshold (0.0 to 1.0).
        
        Returns:
            Ranked list of relevant chunks.
        """
        results = self.retriever.retrieve(query, k=k * 2)

        # Filter by similarity threshold if applicable
        if min_similarity > 0.0:
            import numpy as np

            from src.embeddings.embedder import generate_embeddings

            query_emb = generate_embeddings(
                [query],
                model_name=self.retriever.embedding_model,
                normalize=True,
            )[0]

            filtered: list[EmbeddedChunk] = []
            for chunk in results:
                similarity = np.dot(query_emb, chunk.embedding)
                if similarity >= min_similarity:
                    filtered.append(chunk)
            results = filtered[:k]

        return results[:k]

    def batch_search(
        self,
        queries: list[str],
        k: int = 5,
    ) -> dict[str, list[EmbeddedChunk]]:
        """Perform batch semantic search.
        
        Args:
            queries: List of query texts.
            k: Number of results per query.
        
        Returns:
            Dictionary mapping queries to result chunks.
        """
        return {query: self.search(query, k=k) for query in queries}
