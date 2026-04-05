from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.embeddings.compare import EmbeddedChunk
from src.ingestion.cleaner import clean_text
from src.ingestion.loader import load_documents
from src.ingestion.parser import parse_document
from src.llm.gemma import GemmaModel
from src.llm.llama import LlamaModel
from src.llm.prompt_builder import PromptBuilder
from src.retrieval.hybrid import HybridRetrieval
from src.retrieval.reranker import RerankedChunk
from src.utils.config import IngestionConfig, PROCESSED_DATA_DIR, ensure_project_directories
from src.vector_db.retriever import VectorRetriever


@dataclass(slots=True)
class ProcessedDocument:
    source_name: str
    source_path: str
    extension: str
    text: str


@dataclass(slots=True)
class RAGResult:
    query: str
    answer: str
    prompt: str
    retrieved_chunks: list[dict[str, Any]]
    reranked: bool
    llm_model: str
    retrieval_k: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RAGPipelineConfig:
    query_top_k: int = 5
    use_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"
    llm_provider: str = "llama"
    llm_model_name: str = "llama2"
    llm_api_url: str = "http://localhost:11434/api/generate"
    include_sources: bool = True
    system_role: str = "domain expert assistant"


class RAGPipeline:
    """End-to-end RAG pipeline for retrieval, prompt construction, and generation."""

    def __init__(
        self,
        retriever: VectorRetriever,
        llm: Any,
        prompt_builder: type[PromptBuilder] = PromptBuilder,
        config: RAGPipelineConfig | None = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.config = config or RAGPipelineConfig()
        self.hybrid_retrieval = HybridRetrieval(
            retriever,
            use_reranker=self.config.use_reranker,
            reranker_model=self.config.reranker_model,
        )

    def retrieve(self, query: str, top_k: int | None = None) -> list[EmbeddedChunk] | list[RerankedChunk]:
        """Retrieve relevant chunks for a query."""

        return self.hybrid_retrieval.retrieve(query, k=top_k or self.config.query_top_k)

    def build_prompt(
        self,
        query: str,
        retrieved_chunks: list[EmbeddedChunk] | list[RerankedChunk],
    ) -> str:
        """Build the generation prompt using retrieved context."""

        return self.prompt_builder.build_rag_prompt(
            query=query,
            retrieved_chunks=retrieved_chunks,
            system_role=self.config.system_role,
            include_sources=self.config.include_sources,
        )

    def generate(
        self,
        query: str,
        top_k: int | None = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> RAGResult:
        """Run the full RAG pipeline for a single query."""

        retrieved_chunks = self.retrieve(query, top_k=top_k)
        prompt = self.build_prompt(query, retrieved_chunks)
        answer = self.llm.generate(prompt, max_tokens=max_tokens, temperature=temperature)

        return RAGResult(
            query=query,
            answer=answer,
            prompt=prompt,
            retrieved_chunks=[self._serialize_chunk(chunk) for chunk in retrieved_chunks],
            reranked=bool(self.config.use_reranker),
            llm_model=getattr(self.llm, "model_name", self.llm.__class__.__name__),
            retrieval_k=top_k or self.config.query_top_k,
        )

    @staticmethod
    def _serialize_chunk(chunk: EmbeddedChunk | RerankedChunk) -> dict[str, Any]:
        """Serialize a retrieved chunk for reporting and debugging."""

        if isinstance(chunk, RerankedChunk):
            return {
                "source_name": chunk.chunk.source_name,
                "source_path": chunk.chunk.source_path,
                "extension": chunk.chunk.extension,
                "chunk_id": chunk.chunk.chunk_id,
                "chunk_type": chunk.chunk.chunk_type,
                "text": chunk.chunk.text,
                "start_index": chunk.chunk.start_index,
                "end_index": chunk.chunk.end_index,
                "score": chunk.rerank_score,
                "original_rank": chunk.original_rank,
            }

        return {
            "source_name": chunk.source_name,
            "source_path": chunk.source_path,
            "extension": chunk.extension,
            "chunk_id": chunk.chunk_id,
            "chunk_type": chunk.chunk_type,
            "text": chunk.text,
            "start_index": chunk.start_index,
            "end_index": chunk.end_index,
            "score": None,
            "original_rank": None,
        }


def build_llm(provider: str = "llama", model_name: str | None = None, api_url: str | None = None) -> Any:
    """Factory for LLaMA and Gemma wrappers."""

    provider = provider.lower().strip()
    if provider == "llama":
        return LlamaModel(model_name=model_name or "llama2", api_url=api_url or "http://localhost:11434/api/generate")
    if provider == "gemma":
        return GemmaModel(model_name=model_name or "gemma:7b", api_url=api_url or "http://localhost:11434/api/generate")
    raise ValueError(f"Unsupported LLM provider: {provider}")


def ingest_documents(config: IngestionConfig | None = None) -> list[ProcessedDocument]:
    """Load, parse, and clean documents from the raw data directory."""

    config = config or IngestionConfig()
    ensure_project_directories()

    raw_documents = load_documents(config.raw_dir, recursive=config.recursive)
    processed_documents: list[ProcessedDocument] = []

    for document in raw_documents:
        parsed_text = parse_document(document.path)
        cleaned_text = clean_text(
            parsed_text,
            remove_reference_sections=config.remove_reference_sections,
            normalize_whitespace=config.normalize_whitespace,
        )
        processed_documents.append(
            ProcessedDocument(
                source_name=document.source_name,
                source_path=str(document.path),
                extension=document.extension,
                text=cleaned_text,
            )
        )

    return processed_documents


def save_processed_documents(documents: list[ProcessedDocument], output_dir: Path | None = None) -> list[Path]:
    """Persist cleaned documents as JSON files in the processed data directory."""

    ensure_project_directories()
    output_dir = output_dir or PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for document in documents:
        output_path = output_dir / f"{document.source_name}.json"
        output_path.write_text(json.dumps(asdict(document), indent=2, ensure_ascii=False), encoding="utf-8")
        saved_paths.append(output_path)
    return saved_paths


def run_ingestion_pipeline(config: IngestionConfig | None = None) -> list[Path]:
    """Convenience wrapper that runs phase 1 ingestion end-to-end."""

    processed_documents = ingest_documents(config=config)
    return save_processed_documents(processed_documents, output_dir=(config.processed_dir if config else None))


def run_rag_pipeline(
    query: str,
    retriever: VectorRetriever,
    *,
    llm: Any | None = None,
    pipeline_config: RAGPipelineConfig | None = None,
    top_k: int | None = None,
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> RAGResult:
    """Convenience wrapper that executes the full RAG pipeline."""

    config = pipeline_config or RAGPipelineConfig()
    model = llm or build_llm(
        provider=config.llm_provider,
        model_name=config.llm_model_name,
        api_url=config.llm_api_url,
    )
    pipeline = RAGPipeline(retriever=retriever, llm=model, config=config)
    return pipeline.generate(query, top_k=top_k, max_tokens=max_tokens, temperature=temperature)
