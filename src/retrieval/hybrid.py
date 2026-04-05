from __future__ import annotations

from typing import Optional

from src.embeddings.compare import EmbeddedChunk
from src.retrieval.reranker import CrossEncoderReranker, RerankedChunk
from src.retrieval.semantic_search import SemanticSearch
from src.vector_db.retriever import VectorRetriever


class HybridRetrieval:
    """Hybrid retrieval combining semantic search and optional cross-encoder reranking."""

    def __init__(
        self,
        retriever: VectorRetriever,
        use_reranker: bool = False,
        reranker_model: str = "BAAI/bge-reranker-base",
    ):
        """Initialize hybrid retrieval.
        
        Args:
            retriever: VectorRetriever instance.
            use_reranker: Whether to apply cross-encoder reranking.
            reranker_model: Cross-encoder model to use if reranking.
        """
        self.semantic_search = SemanticSearch(retriever)
        self.use_reranker = use_reranker
        self.reranker = (
            CrossEncoderReranker(reranker_model) if use_reranker else None
        )

    def retrieve(
        self,
        query: str,
        k: int = 5,
    ) -> list[EmbeddedChunk] | list[RerankedChunk]:
        """Retrieve relevant chunks using semantic search and optional reranking.
        
        Args:
            query: Query text.
            k: Number of results to return.
        
        Returns:
            List of chunks (EmbeddedChunk or RerankedChunk if reranked).
        """
        # Retrieve with semantic search
        chunks = self.semantic_search.search(query, k=k * 2)

        if not self.use_reranker or not self.reranker:
            return chunks[:k]

        # Apply reranking
        reranked = self.reranker.rerank(query, chunks, top_k=k)
        return reranked

    def batch_retrieve(
        self,
        queries: list[str],
        k: int = 5,
    ) -> dict[str, list[EmbeddedChunk] | list[RerankedChunk]]:
        """Batch retrieval for multiple queries.
        
        Args:
            queries: List of queries.
            k: Number of results per query.
        
        Returns:
            Dictionary mapping queries to retrieved chunks.
        """
        return {query: self.retrieve(query, k=k) for query in queries}
