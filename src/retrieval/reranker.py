from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.embeddings.compare import EmbeddedChunk


@dataclass(slots=True)
class RerankedChunk:
    chunk: EmbeddedChunk
    original_rank: int
    rerank_score: float
    reranked: bool


class CrossEncoderReranker:
    """Rerank chunks using a cross-encoder model."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """Initialize cross-encoder reranker.
        
        Args:
            model_name: HuggingFace model identifier for cross-encoder.
        """
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "Cross-encoder reranking requires sentence-transformers: pip install sentence-transformers"
            ) from exc

        self._model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        chunks: list[EmbeddedChunk],
        top_k: Optional[int] = None,
    ) -> list[RerankedChunk]:
        """Rerank chunks using cross-encoder similarity.
        
        Args:
            query: Query text.
            chunks: List of retrieved chunks to rerank.
            top_k: Return only top-k after reranking (default: return all).
        
        Returns:
            Reranked chunks with scores.
        """
        if not chunks:
            return []

        self._load_model()

        # Prepare pairs: (query, chunk_text)
        pairs = [(query, chunk.text) for chunk in chunks]

        # Get scores
        scores = self._model.predict(pairs)

        # Build results
        reranked: list[RerankedChunk] = []
        for idx, (chunk, score) in enumerate(zip(chunks, scores)):
            reranked.append(
                RerankedChunk(
                    chunk=chunk,
                    original_rank=idx,
                    rerank_score=float(score),
                    reranked=True,
                )
            )

        # Sort by score descending
        reranked.sort(key=lambda x: x.rerank_score, reverse=True)

        if top_k:
            reranked = reranked[:top_k]

        return reranked

    def batch_rerank(
        self,
        queries: list[str],
        all_chunks: list[list[EmbeddedChunk]],
        top_k: Optional[int] = None,
    ) -> dict[str, list[RerankedChunk]]:
        """Batch rerank multiple queries.
        
        Args:
            queries: List of queries.
            all_chunks: List of chunk lists (one per query).
            top_k: Return top-k per query.
        
        Returns:
            Dictionary mapping queries to reranked chunks.
        """
        return {
            query: self.rerank(query, chunks, top_k=top_k)
            for query, chunks in zip(queries, all_chunks)
        }
