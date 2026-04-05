Assignment-1: Retrieval-Augmented Generation (RAG) for Domain-Speciβic
Question Answering

Objective: The objective of this assignment is to design and implement a Retrieval-
Augmented Generation (RAG) system capable of answering user queries using a domain-
speciβic knowledge base.

Problem Statement: Large Language Models sometimes generate incorrect responses
due to lack of domain knowledge. Retrieval-Augmented Generation (RAG) mitigates this
limitation by retrieving relevant documents from an external knowledge source and
injecting them into the prompt context before generating a response.
Students must design and implement a RAG-based application for a domain of their choice.
• Healthcare knowledge assistant
• Legal document assistant
• University course assistant
• Research paper assistant
• Financial advisory assistant
• Technical documentation assistant
• Customer support knowledge base
Required System Components
• Document ingestion and preprocessing
• Text chunking strategy (use different chunk sizes)
• Embedding generation (use different embedding models – Huggingface and others, try
different embedding dimensions, etc.)
• Vector database storage and retrieval (use at least two vector database)
• Prompt engineering with retrieved context (semantic search and ranking)
• LLM-based answer generation (use different LLMs – e.g. LLaMA-3.1, Gemma-3, GPT style
open models, etc.)
Deliverables
• Complete source code with modular design
• Experimental analysis report (8–10 pages)
• Comparative evaluation of at least two system conβigurations
• Example output demonstrating the system (screenshots, etc.)

-------------------------------------------------------------------------------------------------------------------------------

Here’s a **clean, implementation-first plan** for your RAG assignment that will actually stand out during evaluation (not just “working,” but engineered + defensible in viva).

---

# 🔷 1. Project Direction (Pick This — Strong + Scalable)

### **Domain: Research Paper Assistant (Recommended)**

Why:

* Easy dataset availability (arXiv, Semantic Scholar)
* Naturally fits RAG (long documents, semantic retrieval)
* Strong scope for experimentation (chunking + embeddings matter here)

Alternative (if you want simpler): **University Course Assistant**

---

# 🔷 2. System Architecture (High-Level)

```
User Query
   ↓
Query Embedding
   ↓
Vector DB Retrieval (Top-K)
   ↓
Re-ranking (optional but high scoring)
   ↓
Prompt Construction (context injection)
   ↓
LLM Generation
   ↓
Final Answer + Sources
```

---

# 🔷 3. Tech Stack (Deliberate Choices for Marks)

### Embeddings (compare 2)

* `sentence-transformers/all-MiniLM-L6-v2` (fast baseline)
* `BAAI/bge-base-en-v1.5` (strong semantic performance)

### Vector DB (must use 2)

* **FAISS** (local, fast baseline)
* **ChromaDB** (persistent + metadata filtering)

### LLMs (compare 2)

* `LLaMA 3.1 (8B Instruct)` (via Ollama or API)
* `Gemma 2/3` OR `Mistral 7B`

### Optional Boost (for extra marks)

* Cross-encoder reranker: `bge-reranker-base`
* LangChain OR LlamaIndex (but keep modular)

---

# 🔷 4. Directory Structure (Production-Level)

```
rag-project/
│
├── data/
│   ├── raw/                   # PDFs, scraped text
│   ├── processed/             # cleaned + chunked
│
├── src/
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── parser.py
│   │   └── cleaner.py
│   │
│   ├── chunking/
│   │   ├── chunker.py         # multiple strategies
│   │   └── experiments.py
│   │
│   ├── embeddings/
│   │   ├── embedder.py        # HF models
│   │   └── compare.py
│   │
│   ├── vector_db/
│   │   ├── faiss_db.py
│   │   ├── chroma_db.py
│   │   └── retriever.py
│   │
│   ├── retrieval/
│   │   ├── semantic_search.py
│   │   ├── reranker.py
│   │   └── hybrid.py
│   │
│   ├── llm/
│   │   ├── llama.py
│   │   ├── gemma.py
│   │   └── prompt_builder.py
│   │
│   ├── pipeline/
│   │   └── rag_pipeline.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── experiments.py
│   │
│   └── utils/
│       └── config.py
│
├── notebooks/
│   └── experiments.ipynb
│
├── app/
│   ├── streamlit_app.py       # demo UI
│
├── results/
│   ├── logs/
│   ├── comparisons/
│
├── report/
│   └── report.pdf
│
├── requirements.txt
└── README.md
```

---

# 🔷 5. Phase-Wise Execution Plan (Strict Timeline)

## ✅ Phase 1 — Data + Ingestion (Day 1)

* Collect 20–50 research papers (PDFs)
* Extract text using:

  * `PyMuPDF` or `pdfplumber`
* Clean:

  * Remove references, citations noise
  * Normalize whitespace

**Output:** clean text corpus

---

## ✅ Phase 2 — Chunking Strategy (Day 1–2)

Implement **3 chunking methods**:

1. Fixed size (e.g., 300 tokens)
2. Overlapping chunks (300 + 50 overlap)
3. Semantic chunking (sentence boundaries)

**Why important:** You’ll compare this in report.

---

## ✅ Phase 3 — Embeddings (Day 2)

* Generate embeddings using:

  * MiniLM (384 dim)
  * BGE (768 dim)

Store:

```
{
  text,
  embedding,
  metadata (source, page)
}
```

---

## ✅ Phase 4 — Vector DB (Day 3)

Implement both:

### FAISS

* Fast similarity search
* No metadata filtering

### ChromaDB

* Persistent storage
* Metadata-based retrieval

**Compare:**

* Retrieval speed
* Accuracy (qualitative)

---

## ✅ Phase 5 — Retrieval + Ranking (Day 3–4)

Pipeline:

1. Top-K retrieval (k=5 or 10)
2. Optional reranker (boost marks)

---

## ✅ Phase 6 — Prompt Engineering (Day 4)

Template:

```
You are a domain expert assistant.

Answer the question using ONLY the provided context.
If the answer is not in the context, say "Not found".

Context:
{retrieved_chunks}

Question:
{query}

Answer:
```

Test:

* hallucination control
* context grounding

---

## ✅ Phase 7 — LLM Integration (Day 4–5)

Use:

* LLaMA 3.1
* Gemma/Mistral

Compare:

* response quality
* hallucination
* latency

---

## ✅ Phase 8 — Pipeline Integration (Day 5)

Create single function:

```python
def rag_pipeline(query):
    chunks = retrieve(query)
    ranked = rerank(chunks)
    prompt = build_prompt(ranked)
    answer = generate(prompt)
    return answer
```

---

## ✅ Phase 9 — Evaluation (IMPORTANT for marks)

Metrics:

* Retrieval Accuracy (manual)
* Response Relevance
* Hallucination Rate

Create comparison table:

| Config | Chunking | Embedding | DB | LLM | Result |
| ------ | -------- | --------- | -- | --- | ------ |

---

## ✅ Phase 10 — Demo App (Day 5)

Use **Streamlit**:

* Input query
* Show:

  * Answer
  * Retrieved chunks (very important)

---

# 🔷 6. Two Required System Configurations (MANDATORY)

### Config A (Baseline)

* Fixed chunking
* MiniLM
* FAISS
* LLaMA

### Config B (Improved)

* Overlapping chunking
* BGE embeddings
* ChromaDB
* Reranker + Gemma

---

# 🔷 7. What Will Impress Evaluator (Critical)

* Showing retrieved context alongside answers
* Clear comparison (not just code)
* Explaining:

  * why chunk size matters
  * why embedding dimension affects retrieval
* Demonstrating hallucination control

---

# 🔷 8. Report Structure (8–10 Pages)

1. Introduction
2. Problem Statement
3. Architecture
4. Methodology
5. Experiments
6. Results Comparison
7. Observations
8. Conclusion
