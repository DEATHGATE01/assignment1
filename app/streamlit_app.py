from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except Exception:  # pragma: no cover
    get_script_run_ctx = lambda: None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__" and get_script_run_ctx() is None:
    print("This is a Streamlit app. Run it with: streamlit run app/streamlit_app.py")
    raise SystemExit(0)

from src.chunking.experiments import chunk_documents
from src.embeddings.compare import embed_chunks
from src.llm.prompt_builder import PromptBuilder
from src.pipeline.rag_pipeline import RAGPipelineConfig, build_llm, ingest_documents, run_rag_pipeline
from src.retrieval.hybrid import HybridRetrieval
from src.retrieval.reranker import RerankedChunk
from src.utils.config import IngestionConfig
from src.vector_db.faiss_db import FAISSVectorDB
from src.vector_db.retriever import VectorRetriever


@dataclass(slots=True)
class DemoSettings:
    chunk_strategy: str
    chunk_size: int
    overlap: int
    embedding_model: str
    llm_provider: str
    llm_model_name: str
    llm_api_url: str
    use_reranker: bool
    reranker_model: str
    top_k: int
    retrieve_only: bool


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #102a43;
            --muted: #486581;
            --sand: #f8f5ef;
            --linen: #fffdfa;
            --accent: #d97706;
            --accent-soft: #ffedd5;
            --ok: #0f766e;
            --border: #d9e2ec;
        }
        .stApp {
            background: radial-gradient(1200px 500px at 80% -20%, #ffe7cc 0%, transparent 60%),
                        linear-gradient(180deg, var(--linen) 0%, var(--sand) 100%);
            color: var(--ink);
            font-family: "Source Sans Pro", "Segoe UI", sans-serif;
        }
        .hero {
            border: 1px solid var(--border);
            background: rgba(255, 255, 255, 0.7);
            border-radius: 16px;
            padding: 1.2rem 1.1rem;
            box-shadow: 0 6px 18px rgba(16, 42, 67, 0.07);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            font-family: "Georgia", "Times New Roman", serif;
            font-size: 1.8rem;
            color: var(--ink);
            letter-spacing: 0.2px;
        }
        .hero p {
            margin: 0.5rem 0 0 0;
            color: var(--muted);
            line-height: 1.45;
        }
        .metric-strip {
            border-left: 4px solid var(--accent);
            background: var(--accent-soft);
            border-radius: 10px;
            padding: 0.55rem 0.8rem;
            color: #7c2d12;
            font-size: 0.92rem;
            margin-top: 0.4rem;
        }
        .section-title {
            font-family: "Georgia", "Times New Roman", serif;
            font-size: 1.18rem;
            color: var(--ink);
            margin-top: 0.3rem;
            margin-bottom: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Research Notes Q&A Studio</h1>
            <p>
                Build a local knowledge index from your documents, run retrieval-only checks,
                and generate grounded answers with source snippets you can inspect.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar() -> DemoSettings:
    with st.sidebar:
        st.markdown("### Configuration")

        chunk_strategy = st.selectbox("Chunking strategy", ["fixed", "sentence", "semantic"], index=0)
        chunk_size = st.slider("Chunk size", min_value=100, max_value=600, value=300, step=50)
        overlap = st.slider("Chunk overlap", min_value=0, max_value=200, value=50, step=10)

        embedding_model = st.selectbox(
            "Embedding model",
            [
                "sentence-transformers/all-MiniLM-L6-v2",
                "BAAI/bge-base-en-v1.5",
            ],
            index=0,
        )

        st.markdown("### Generation")
        llm_provider = st.selectbox("LLM provider", ["llama", "gemma"], index=0)
        default_model_name = "llama2" if llm_provider == "llama" else "gemma:7b"
        llm_model_name = st.text_input("LLM model name", value=default_model_name)
        llm_api_url = st.text_input("LLM API URL", value="http://localhost:11434/api/generate")

        st.markdown("### Retrieval")
        use_reranker = st.checkbox("Use reranker", value=False)
        reranker_model = st.text_input("Reranker model", value="BAAI/bge-reranker-base")
        top_k = st.slider("Top K", min_value=1, max_value=10, value=5, step=1)

        retrieve_only = st.checkbox("Retrieve-only mode", value=False)

    return DemoSettings(
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
        overlap=overlap,
        embedding_model=embedding_model,
        llm_provider=llm_provider,
        llm_model_name=llm_model_name,
        llm_api_url=llm_api_url,
        use_reranker=use_reranker,
        reranker_model=reranker_model,
        top_k=top_k,
        retrieve_only=retrieve_only,
    )


def _serialize_chunk(chunk: Any) -> dict[str, Any]:
    if isinstance(chunk, RerankedChunk):
        return {
            "source_name": chunk.chunk.source_name,
            "source_path": chunk.chunk.source_path,
            "chunk_id": chunk.chunk.chunk_id,
            "chunk_type": chunk.chunk.chunk_type,
            "text": chunk.chunk.text,
            "score": chunk.rerank_score,
        }

    return {
        "source_name": chunk.source_name,
        "source_path": chunk.source_path,
        "chunk_id": chunk.chunk_id,
        "chunk_type": chunk.chunk_type,
        "text": chunk.text,
        "score": None,
    }


def _build_retriever(settings: DemoSettings) -> VectorRetriever:
    documents = ingest_documents(IngestionConfig())
    if not documents:
        raise RuntimeError("No supported documents found in data/raw")

    chunks = chunk_documents(
        documents,
        strategy=settings.chunk_strategy,
        chunk_size=settings.chunk_size,
        overlap=settings.overlap,
    )
    if not chunks:
        raise RuntimeError("No chunks produced from ingested documents")

    embedded_chunks = embed_chunks(chunks, model_name=settings.embedding_model)
    dimension = len(embedded_chunks[0].embedding)

    db = FAISSVectorDB(dimension=dimension)
    db.add(embedded_chunks)

    st.session_state.index_stats = {
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "embedding_dimension": dimension,
        "embedding_model": settings.embedding_model,
        "chunk_strategy": settings.chunk_strategy,
    }

    return VectorRetriever(db=db, embedding_model=settings.embedding_model)


def _run_query(query: str, settings: DemoSettings) -> tuple[str, str, list[dict[str, Any]], str]:
    retriever = st.session_state.retriever
    if retriever is None:
        raise RuntimeError("Build the index first")

    if settings.retrieve_only:
        hybrid = HybridRetrieval(
            retriever,
            use_reranker=settings.use_reranker,
            reranker_model=settings.reranker_model,
        )
        retrieved = hybrid.retrieve(query, k=settings.top_k)
        prompt = PromptBuilder.build_rag_prompt(
            query=query,
            retrieved_chunks=retrieved,
            system_role="domain expert assistant",
            include_sources=True,
        )
        return (
            "Retrieve-only mode is enabled, so no LLM call was made.",
            prompt,
            [_serialize_chunk(chunk) for chunk in retrieved],
            "none",
        )

    config = RAGPipelineConfig(
        query_top_k=settings.top_k,
        use_reranker=settings.use_reranker,
        reranker_model=settings.reranker_model,
        llm_provider=settings.llm_provider,
        llm_model_name=settings.llm_model_name,
        llm_api_url=settings.llm_api_url,
        include_sources=True,
    )
    llm = build_llm(
        provider=settings.llm_provider,
        model_name=settings.llm_model_name,
        api_url=settings.llm_api_url,
    )
    rag_result = run_rag_pipeline(
        query,
        retriever,
        llm=llm,
        pipeline_config=config,
        top_k=settings.top_k,
    )
    return rag_result.answer, rag_result.prompt, rag_result.retrieved_chunks, rag_result.llm_model


def _render_retrieved_chunks(retrieved_chunks: list[dict[str, Any]]) -> None:
    st.markdown('<div class="section-title">Retrieved Context</div>', unsafe_allow_html=True)
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        title = f"{idx}. {chunk.get('source_name', 'unknown')} (chunk {chunk.get('chunk_id', '?')})"
        with st.expander(title):
            if chunk.get("score") is not None:
                st.caption(f"Rerank score: {chunk.get('score'):.4f}")
            st.caption(chunk.get("source_path", ""))
            st.write(chunk.get("text", ""))


def _render_evaluation_panel() -> None:
    st.divider()
    st.markdown('<div class="section-title">Evaluation Artifacts</div>', unsafe_allow_html=True)

    comparisons_dir = Path("results/comparisons")
    if not comparisons_dir.exists():
        st.caption("results/comparisons directory does not exist yet.")
        return

    report_files = sorted(comparisons_dir.glob("evaluation_*.json"))
    if not report_files:
        st.caption("No evaluation report files found yet in results/comparisons.")
        return

    selected = st.selectbox("Select evaluation report", report_files, format_func=lambda p: p.name)
    if not selected:
        return

    try:
        content = json.loads(selected.read_text(encoding="utf-8"))
        st.json(content)
    except Exception as exc:
        st.error(f"Could not read report: {exc}")


def main() -> None:
    st.set_page_config(page_title="RAG Demo", layout="wide")
    _inject_styles()
    _render_header()

    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "index_stats" not in st.session_state:
        st.session_state.index_stats = {}

    settings = _render_sidebar()

    left_col, right_col = st.columns([1, 1])
    with left_col:
        if st.button("Build / Rebuild Index", type="primary"):
            with st.spinner("Building vector index from data/raw..."):
                try:
                    st.session_state.retriever = _build_retriever(settings)
                    st.success("Index built successfully.")
                except Exception as exc:
                    st.session_state.retriever = None
                    st.error(f"Index build failed: {exc}")

    with right_col:
        stats = st.session_state.index_stats
        if stats:
            st.markdown(
                (
                    '<div class="metric-strip">'
                    f"Docs: {stats['document_count']} | "
                    f"Chunks: {stats['chunk_count']} | "
                    f"Dim: {stats['embedding_dimension']} | "
                    f"Chunking: {stats['chunk_strategy']}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        else:
            st.info("Build an index first to run queries.")

    query = st.text_input(
        "Question",
        placeholder="Example: What is retrieval-augmented generation?",
    )

    if st.button("Run Query"):
        if not query.strip():
            st.warning("Enter a query first.")
        elif st.session_state.retriever is None:
            st.warning("Build the index first.")
        else:
            try:
                answer, prompt, retrieved_chunks, model_name = _run_query(query, settings)

                st.markdown('<div class="section-title">Answer</div>', unsafe_allow_html=True)
                st.write(answer)
                st.caption(f"Model: {model_name} | Top K: {settings.top_k}")

                _render_retrieved_chunks(retrieved_chunks)

                with st.expander("Prompt Sent To LLM"):
                    st.code(prompt)
            except Exception as exc:
                st.error(f"Query failed: {exc}")

    _render_evaluation_panel()


if __name__ == "__main__":
    main()
