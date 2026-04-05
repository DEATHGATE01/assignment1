# RAG Project

This repository contains a modular Retrieval-Augmented Generation project built phase by phase.

## Current status

Phase 1 is implemented:

- raw document discovery
- text extraction for TXT, MD, PDF, and DOCX files
- cleaning and normalization
- persistence of cleaned documents to JSON

Phase 2 is also implemented:

- fixed-size token chunking
- sentence-aware chunking
- semantic chunking approximation with overlap
- chunking experiment summaries

Phase 3 is also implemented:

- embeddings using HuggingFace sentence-transformers
- support for MiniLM L6 v2 (384-dim) and BGE Base (768-dim) models
- embedding normalization and statistics
- embedding experiment summaries

Phase 4 is also implemented:

- FAISS vector database with fast similarity search
- ChromaDB with persistent storage and metadata filtering
- unified VectorRetriever interface for both databases
- query embedding and retrieval

Phase 5 is also implemented:

- semantic search using vector similarity
- cross-encoder reranking (optional, using BAAI/bge-reranker-base)
- hybrid retrieval combining semantic search with optional reranking
- batch retrieval support

Phase 6 is also implemented:

- RAG prompt building with context injection
- multi-turn conversation support
- LLaMA-3.1 model wrapper via Ollama
- Gemma model wrapper via Ollama
- prompt templates for QA, summarization, and relevance checking

Phase 7 is also implemented:

- end-to-end RAG pipeline orchestration
- retriever -> prompt builder -> LLM flow
- configurable query top-k and reranking support
- wrapper for single-query RAG execution

Phase 8 is also implemented:

- retrieval metrics: precision@k, hit rate@k, MRR
- generation metrics: relevance, grounding, hallucination estimate
- end-to-end aggregate scoring
- experiment report generation and config comparison output

Phase 9 is also implemented:

- Streamlit demo to build/rebuild index from raw documents
- retrieval-only mode and full LLM mode
- answer view with retrieved sources and optional rerank scores
- prompt inspection panel for transparency
- evaluation report viewer for generated JSON artifacts

## Folder layout

- data/raw/ - source documents
- data/processed/ - cleaned output
- src/ingestion/ - loader, parser, cleaner
- src/pipeline/ - ingestion and RAG pipeline
- src/evaluation/ - metrics and experiment reports
- app/ - Streamlit demo

## Usage

1. Put source files in data/raw/
2. Run the app: streamlit run app/streamlit_app.py
3. Build index from the sidebar settings
4. Query in retrieve-only mode or full LLM mode
5. Review evaluation reports in results/comparisons

## Next phases

Final step: report packaging and submission-ready comparison writeup.
