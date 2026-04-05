from __future__ import annotations

from typing import Optional

from src.embeddings.compare import EmbeddedChunk
from src.retrieval.reranker import RerankedChunk


class PromptBuilder:
    """Builder for RAG prompts with retrieved context injection."""

    @staticmethod
    def build_rag_prompt(
        query: str,
        retrieved_chunks: list[EmbeddedChunk] | list[RerankedChunk],
        system_role: str = "domain expert assistant",
        include_sources: bool = True,
    ) -> str:
        """Build a RAG prompt with context injection.
        
        Args:
            query: User query.
            retrieved_chunks: List of retrieved chunks to inject as context.
            system_role: Role description for the assistant.
            include_sources: Whether to include source citations in context.
        
        Returns:
            Formatted prompt string.
        """
        context_parts: list[str] = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            text = chunk.chunk.text if isinstance(chunk, RerankedChunk) else chunk.text
            source = chunk.chunk.source_name if isinstance(chunk, RerankedChunk) else chunk.source_name

            if include_sources:
                context_parts.append(f"[Document {idx}: {source}]\n{text}")
            else:
                context_parts.append(text)

        context = "\n\n".join(context_parts)

        prompt = f"""You are a {system_role}.

Answer the question using ONLY the provided context below.
If the answer is not found in the context, explicitly state "The answer is not available in the provided context."
Do not make assumptions or provide information from your training data.

Context:
{context}

Question:
{query}

Answer:"""
        return prompt

    @staticmethod
    def build_qa_prompt(
        query: str,
        context: str,
    ) -> str:
        """Build a simple question-answering prompt.
        
        Args:
            query: User query.
            context: Context text.
        
        Returns:
            Formatted prompt string.
        """
        return f"""Context:
{context}

Question: {query}

Answer:"""

    @staticmethod
    def build_multi_turn_prompt(
        query: str,
        retrieved_chunks: list[EmbeddedChunk] | list[RerankedChunk],
        history: Optional[list[tuple[str, str]]] = None,
    ) -> str:
        """Build a multi-turn RAG prompt with conversation history.
        
        Args:
            query: Latest query.
            retrieved_chunks: Retrieved context chunks.
            history: Conversation history as list of (user_msg, assistant_msg) tuples.
        
        Returns:
            Formatted prompt string.
        """
        context_parts: list[str] = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            text = chunk.chunk.text if isinstance(chunk, RerankedChunk) else chunk.text
            source = chunk.chunk.source_name if isinstance(chunk, RerankedChunk) else chunk.source_name
            context_parts.append(f"[{source}]\n{text}")

        context = "\n\n".join(context_parts)

        prompt = "You are a helpful assistant."

        if history:
            prompt += "\n\nConversation History:"
            for user_msg, assistant_msg in history:
                prompt += f"\nUser: {user_msg}\nAssistant: {assistant_msg}"

        prompt += f"""

Context Information:
{context}

User: {query}
Assistant:"""
        return prompt

    @staticmethod
    def build_summary_prompt(
        text: str,
        max_length: int = 200,
    ) -> str:
        """Build a prompt for summarizing retrieved content.
        
        Args:
            text: Text to summarize.
            max_length: Maximum length of summary in words.
        
        Returns:
            Formatted prompt string.
        """
        return f"""Write a concise summary of the following text in at most {max_length} words.

Text:
{text}

Summary:"""

    @staticmethod
    def build_relevance_check_prompt(
        query: str,
        text: str,
    ) -> str:
        """Build a prompt for checking if text is relevant to a query.
        
        Args:
            query: Query to check relevance against.
            text: Text to evaluate.
        
        Returns:
            Formatted prompt string.
        """
        return f"""Given the query below, determine if the provided text is relevant to answering it.
Respond with only "Yes" or "No".

Query: {query}

Text:
{text}

Relevant:"""
