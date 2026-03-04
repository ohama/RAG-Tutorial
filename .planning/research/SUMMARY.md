# Project Research Summary

**Project:** Local RAG Educational Tutorial (Jupyter Notebook, Apple Silicon MLX)
**Domain:** Local Retrieval-Augmented Generation (RAG) system — educational, bilingual Korean/English
**Researched:** 2026-03-04
**Confidence:** HIGH

## Executive Summary

This project is a fully local, Korean-language RAG tutorial built on Apple Silicon (M4 Max, 128GB), using MLX for LLM inference and sentence-transformers for embedding. Experts build this type of project by explicitly separating the two RAG pipelines — offline ingestion (document loading, chunking, embedding, vector storage) and runtime query (embedding, retrieval, prompt construction, generation) — and by keeping all components modular so each can be understood and tested in isolation. The primary educational deliverable is a side-by-side RAG on/off comparison notebook, not just a working system. The critical design constraint is "no framework abstractions": avoid LangChain and LlamaIndex because they hide the exact mechanics the tutorial is meant to teach.

The recommended stack is well-defined and all current as of March 2026: mlx-lm 0.30.7 with `mlx-community/Qwen2.5-7B-Instruct-4bit` for generation, `BAAI/bge-m3` via sentence-transformers for multilingual embedding (mandatory for Korean content), ChromaDB 1.5.2 for local persistent vector storage, pymupdf4llm 0.3.4 for PDF parsing, and langchain-text-splitters 1.1.1 (standalone) for chunking. All components run fully offline with no cloud API dependencies. The stack is beginner-friendly in terms of API surface, and all libraries have active maintenance and strong documentation.

The most dangerous risks are MLX-specific and silent: the Qwen3.5 KV cache hybrid-architecture crash, Metal GPU memory bloat accumulating across Jupyter cells, and using an English-only embedding model for a Korean corpus (all retrievals appear to work but return wrong results). The Korean language constraint adds a non-obvious requirement to the embedding layer that must be addressed before indexing — using `BAAI/bge-m3` instead of the typical `all-MiniLM-L6-v2` default. All five critical pitfalls have known, low-cost fixes that must be baked in from day one rather than retrofitted.

---

## Key Findings

### Recommended Stack

The stack is designed around maximum educational transparency and full offline operation on Apple Silicon. MLX (Apple's own framework) and mlx-lm provide direct Metal GPU access for LLM inference with no daemon or server layer. Sentence-transformers handles embedding via the Apple MPS backend automatically. ChromaDB requires no server process, persists to disk as SQLite, and provides metadata filtering that pure FAISS cannot. All libraries are installed via pip into a Python 3.11 virtual environment.

Full details: [.planning/research/STACK.md](.planning/research/STACK.md)

**Core technologies:**
- `mlx` 0.31.0 + `mlx-lm` 0.30.7: Apple Silicon LLM inference — the only first-party tool for running Qwen MLX models; releases every 2-3 weeks; model-agnostic API
- `sentence-transformers` 5.2.3 + `BAAI/bge-m3`: Multilingual embedding — mandatory for Korean content; 100+ language training, 8192-token context window; runs on Apple MPS automatically
- `chromadb` 1.5.2: Local vector database — embedded (no server), persists to disk, metadata filtering, best-documented local vector DB
- `pymupdf4llm` 0.3.4: PDF parsing to Markdown — preserves headers/tables, 0.12s/page, designed for RAG workflows
- `langchain-text-splitters` 1.1.1: Document chunking — standalone (no full LangChain needed), `RecursiveCharacterTextSplitter` is the proven RAG standard
- `mlx-community/Qwen2.5-7B-Instruct-4bit`: Generation model — ~4.5GB RAM at 4-bit, well-tested mlx-community conversion, strong quality with headroom on 128GB for embeddings + vector DB

**Model note:** The project specifies "Qwen3.5 MLX" — as of March 2026, `Qwen2.5-7B-Instruct-4bit` is the stable, production-ready choice. `Qwen/Qwen3-8B-MLX-4bit` is available as an upgrade path with no code changes required (mlx-lm API is model-agnostic).

**What to avoid:** Full LangChain, full LlamaIndex, Ollama, OpenAI API for embeddings, all-MiniLM-L6-v2 (English-only embedding), FAISS standalone, pypdf2.

### Expected Features

Full details: [.planning/research/FEATURES.md](.planning/research/FEATURES.md)

**Must have (table stakes — P1):**
- Document loading for Markdown, PDF, TXT — entry point for any RAG demo
- Fixed-size chunking with configurable size and overlap — baseline every RAG tutorial covers
- Local embedding generation via sentence-transformers — must be local and visible
- ChromaDB vector store — simple enough to understand in one notebook cell
- Top-k similarity retrieval with source metadata — retrieval with citations
- Prompt construction showing retrieved context explicitly — learner must see the full prompt
- Qwen3.5 MLX generation step — the local LLM output
- RAG on vs. RAG off comparison notebook — the core educational deliverable
- Source citations displayed with each answer — grounds output in retrieved documents
- 3-5 curated sample documents — learners can run everything immediately
- Korean language explanations throughout notebook Markdown cells

**Should have (differentiators — P2, add after core works):**
- Chunking strategy comparison notebook (fixed vs. recursive vs. semantic)
- Embedding cosine similarity score visualization (bar chart per chunk)
- Retrieval failure examples (intentionally bad queries that demonstrate when RAG breaks)
- User document drop-in directory (`docs/` with auto-discovery)

**Defer to v2+:**
- RAGAS-style qualitative evaluation (requires pre-made ground-truth Q&A pairs)
- Interactive chunk parameter sensitivity demo (ipywidgets sliders)
- Hybrid search (BM25 + dense vector) — teach dense-only first
- Korean-specific embedding model comparison

**Anti-features (do not build):** Web UI (Gradio/Streamlit), full LangChain/LlamaIndex, cloud embedding APIs, multi-turn conversational memory, Docker/containerization, multimodal support.

### Architecture Approach

The system uses two completely separate pipelines — an offline ingestion pipeline (load → chunk → embed → persist to ChromaDB) and a runtime query pipeline (embed query → retrieve from ChromaDB → build prompt → generate with mlx-lm) — that share only one dependency: the embedding model. The embedding model must be identical in both pipelines or retrieval degrades silently. All components are implemented as standalone Python modules in a `src/` directory, with numbered Jupyter notebooks in `tutorial/` that demonstrate each component individually before combining them in an end-to-end notebook.

Full details: [.planning/research/ARCHITECTURE.md](.planning/research/ARCHITECTURE.md)

**Major components:**
1. **Loader** (`src/ingestion/loader.py`) — parse MD/PDF/TXT into `Document` objects with source metadata; metadata must be preserved through the entire pipeline for source citation
2. **Chunker** (`src/ingestion/chunker.py`) — split documents into overlapping `Chunk` objects; 512 tokens, 128-token overlap (25%), Korean-aware sentence splitting via KSS
3. **Embedder** (`src/embedding/embedder.py`) — shared singleton; both ingestion and query import from here; changing this requires full re-indexing
4. **Vector Store** (`src/store/vector_store.py`) — ChromaDB wrapper with `PersistentClient`; ingestion writes, query reads (never both in same call path)
5. **Retriever** (`src/retrieval/retriever.py`) — query embedding + ChromaDB `.query()` returning top-k chunks with similarity scores and metadata
6. **Prompt Builder** (`src/retrieval/prompt_builder.py`) — pure string assembly (no I/O, no DB); prompt template as a config value, not hardcoded
7. **Generator** (`src/generation/generator.py`) — thin mlx-lm wrapper; keeping this separate makes LLM backend swappable
8. **RAG Facade** (`src/rag.py`) — high-level `rag.ingest(path)` + `rag.query("question")` for notebook cells that don't need to show internals

**Key patterns:**
- Shared Embedder Singleton: only one class, imported by both pipelines
- Document/Chunk objects carry metadata through the whole pipeline
- Two-stage pipeline separation: ingestion writes once, query reads only
- Prompt template as a config value, not inline string

### Critical Pitfalls

Full details: [.planning/research/PITFALLS.md](.planning/research/PITFALLS.md)

1. **Qwen3.5 KV cache initialization crash on turn 2** — Always use `mlx_lm.utils.make_prompt_cache(model)` instead of manual `[KVCache() for _ in range(n_layers)]`. Qwen3.5 is a hybrid attention model; manual init does not account for mixed layer types. First turn works; second turn crashes with `TypeError`. Fix is one line.

2. **MLX Metal cache memory bloat killing Jupyter kernels** — Call `mx.metal.clear_cache()` after every RAG inference call. Without it, memory grows linearly across queries in a long-running Jupyter session (first query: 3s, tenth: 45s, kernel eventually dies). Bake this into the inference wrapper from day one.

3. **English-only embedding model silently failing on Korean content** — Use `BAAI/bge-m3` (100+ language training) instead of `all-MiniLM-L6-v2` (English-only). Korean queries will retrieve English chunks and never surface Korean documents. No error is raised — retrieval appears to work but returns semantically random results for Korean text.

4. **Chunking destroying Korean sentence context** — Use `chunk_size=512` tokens with `chunk_overlap=128` (25%). Use KSS (Korean Sentence Splitter) not NLTK punkt (English-only). Validate chunk output by printing 20 random chunks before embedding.

5. **Metal GPU memory cap at 75% causing silent performance collapse** — Load embedding model, embed entire corpus, persist to ChromaDB, then `del embedding_model; mx.metal.clear_cache()` before loading the LLM. Never run both models simultaneously. Monitor with `mx.metal.get_peak_memory()`.

---

## Implications for Roadmap

Based on research, the project naturally decomposes into 4 phases following the feature dependency chain and pitfall-to-phase mapping from PITFALLS.md.

### Phase 1: Environment and Model Validation
**Rationale:** Three of the five critical pitfalls must be eliminated before any application code is written: the Qwen3.5 KV cache crash, the Metal memory architecture constraint, and embedding model selection. These cannot be retrofitted — they affect the foundation. Stack installation and model validation are also gating dependencies for everything else.
**Delivers:** Working Python environment, validated model loading (Qwen2.5-7B-Instruct-4bit via mlx-lm), verified KV cache pattern, confirmed sequential model loading strategy, verified `BAAI/bge-m3` Korean retrieval, baseline performance measurements.
**Addresses:** FEATURES.md — "Local embedding with sentence-transformers model," "Qwen3.5 MLX generation step"
**Avoids:** Pitfall 1 (KV cache crash), Pitfall 3 (English-only embedding), Pitfall 5 (Metal memory cap)
**Research flag:** Standard patterns — mlx-lm model loading is well-documented; no additional research-phase needed.

### Phase 2: Document Ingestion Pipeline
**Rationale:** The ingestion pipeline (load → chunk → embed → store) is the write side of the system and must be complete before any retrieval or comparison demo is possible. Chunking quality directly determines embedding quality, which determines retrieval quality. Metadata must be designed into this phase — retrofitting source citation later requires a full re-index.
**Delivers:** Working ingestion for Markdown, PDF (via pymupdf4llm), and TXT; validated chunking with 512/128 overlap and Korean-aware splitting; ChromaDB persistent collection with source metadata; sample documents indexed and ready.
**Addresses:** FEATURES.md — document loading, fixed-size chunking, local embedding generation, vector store, source citations (metadata foundation), sample documents
**Avoids:** Pitfall 4 (chunking destroying Korean context), Anti-pattern 5 (re-embedding on every query), Technical debt: skip embedding persistence
**Research flag:** Standard patterns — ingestion pipeline is well-documented. One area to validate: pymupdf4llm table handling behavior on any PDF sample documents chosen.

### Phase 3: Core RAG Pipeline and Comparison Notebook
**Rationale:** This is the core educational deliverable. With ingestion complete, the query pipeline (retrieve → build prompt → generate) can be built and the RAG on/off comparison notebook assembled. The memory bloat pitfall must be handled here (bake `clear_cache()` into the inference wrapper from the first cell).
**Delivers:** Working retrieval with similarity scores and source citations, prompt builder with visible context injection, RAG on/off comparison notebook (the primary artifact), Korean explanations throughout all Markdown cells.
**Addresses:** FEATURES.md — all P1 features: similarity retrieval with metadata, prompt construction (visible), Qwen3.5 MLX generation, RAG on/off comparison, source citation display, Korean explanations
**Avoids:** Pitfall 2 (MLX cache memory bloat), Anti-pattern 1 (different embedding models at ingest vs query), Anti-pattern 3 (top-k too large)
**Research flag:** Standard patterns — RAG query pipeline is well-documented. The Qwen3.5 thinking-mode stripping behavior should be validated during implementation (strip `<think>` blocks before displaying and before adding to chat history).

### Phase 4: Differentiation Notebooks (P2 Features)
**Rationale:** Once the core pipeline is working and demonstrably educational, the P2 differentiators can be added as standalone notebooks. These build on the working ingestion and query infrastructure without modifying it. Each notebook is independently runnable and adds a specific learning dimension.
**Delivers:** Chunking strategy comparison notebook, embedding score visualization, retrieval failure examples, user document drop-in directory with auto-discovery loading.
**Addresses:** FEATURES.md — all P2 features: chunking strategy comparison, embedding score visualization, retrieval failure examples, user document drop-in support
**Avoids:** Scope creep into P3 features (RAGAS evaluation, interactive widgets, hybrid search)
**Research flag:** Chunking strategy comparison (fixed vs. recursive vs. semantic) may benefit from a focused research pass on semantic chunking implementations — this is the most implementation-ambiguous P2 feature.

### Phase Ordering Rationale

- **Environment before ingestion:** Pitfalls 1, 3, and 5 are all environment-level issues. If the embedding model is wrong, all indexed data must be deleted and re-embedded. If the KV cache pattern is wrong, it is invisible until multi-turn conversation — which happens in Phase 3. Validating these in Phase 1 costs an hour; fixing them after Phase 3 costs days.
- **Ingestion before query:** The feature dependency chain (FEATURES.md) is deterministic — you cannot retrieve what has not been indexed. The query pipeline has no code to write until the vector store has data.
- **Core comparison notebook before differentiators:** The RAG on/off comparison is the stated "primary educational deliverable." P2 features add value only if P1 is solid. Building P2 on a shaky P1 creates compounding debt.
- **Metadata from day one:** Source citation is a P1 feature, but it depends on metadata being stored in ChromaDB at ingestion time. If metadata is omitted from Phase 2, Phase 3 requires a full re-index. This cross-phase dependency is the most common missed dependency in RAG tutorials.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 4 (chunking strategy comparison):** Semantic chunking implementations vary significantly; the best Python library choice for sentence-boundary semantic chunking on Korean/English mixed text is not fully resolved. Recommend a targeted research pass before implementing the comparison notebook.

Phases with standard patterns (skip research-phase):
- **Phase 1:** MLX model loading, KV cache, Metal memory — all documented with official sources and known fixes.
- **Phase 2:** ChromaDB ingestion, sentence-transformers, pymupdf4llm — all well-documented with clear APIs.
- **Phase 3:** RAG query pipeline, prompt construction, mlx-lm generation — standard patterns with no unresolved questions.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All package versions verified against PyPI as of Feb-Mar 2026; official mlx-lm and mlx-community HuggingFace repos confirmed; alternatives explicitly evaluated and rejected |
| Features | HIGH | Feature set derived from analysis of comparable RAG tutorial repositories and RAG production failure literature; feature dependencies explicitly mapped; MVP definition is unambiguous |
| Architecture | HIGH | Two-pipeline separation is the authoritative pattern documented by LlamaIndex, DEV Community deep-dives, and production RAG guides; component boundaries are clean and testable |
| Pitfalls | HIGH | All five critical pitfalls sourced from specific GitHub issues, arXiv papers, and official MLX documentation; prevention strategies are concrete and code-level |

**Overall confidence:** HIGH

### Gaps to Address

- **Qwen3 vs Qwen2.5 naming ambiguity:** The project spec says "Qwen3.5 MLX" but as of March 2026, `Qwen/Qwen3-8B-MLX-4bit` is available alongside `mlx-community/Qwen2.5-7B-Instruct-4bit`. The recommended default is Qwen2.5-7B for stability, but the choice should be confirmed with the project owner before Phase 1. No code changes needed to switch — mlx-lm API is model-agnostic.
- **Korean document corpus selection:** Sample documents must be chosen before Phase 2 begins. They need to cover a topic the model lacks parametric knowledge of, and must include questions with known correct answers if Phase 4 evaluations are planned. This is a content decision, not a technical one, but it gates Phase 2.
- **KSS (Korean Sentence Splitter) compatibility:** KSS is the recommended Korean-aware sentence splitter, but its compatibility with Python 3.11 and current pip should be verified in Phase 1 environment setup. If KSS is unavailable, a fallback strategy (e.g., splitting on Korean sentence-ending punctuation `。`, `？`, `！`) must be prepared.
- **Semantic chunking for Phase 4:** The best approach for Korean/English mixed-language semantic chunking is not definitively resolved. A short research pass during Phase 4 planning is recommended before committing to an implementation.

---

## Sources

### Primary (HIGH confidence)
- [mlx-lm PyPI](https://pypi.org/project/mlx-lm/) — version 0.30.7 confirmed (Feb 12, 2026)
- [mlx PyPI](https://pypi.org/project/mlx/) — version 0.31.0 confirmed
- [chromadb PyPI](https://pypi.org/project/chromadb/) — version 1.5.2 confirmed (Feb 27, 2026)
- [pymupdf4llm PyPI](https://pypi.org/project/pymupdf4llm/) — version 0.3.4 confirmed (Feb 14, 2026)
- [sentence-transformers PyPI](https://pypi.org/project/sentence-transformers/) — version 5.2.3 confirmed (Feb 17, 2026)
- [langchain-text-splitters PyPI](https://pypi.org/project/langchain-text-splitters/) — version 1.1.1 confirmed (Feb 18, 2026)
- [nomic-ai/nomic-embed-text-v1.5 HuggingFace](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) — model card confirms Apple MPS support
- [BAAI/bge-m3 HuggingFace](https://huggingface.co/BAAI/bge-m3) — multilingual embedding, Korean support confirmed
- [mlx-community Qwen2.5 HuggingFace](https://huggingface.co/collections/mlx-community/qwen25) — Qwen2.5-7B-Instruct-4bit availability confirmed
- [MLX LM Memory Usage Issue #1025](https://github.com/ml-explore/mlx-examples/issues/1025) — KV cache memory bloat root cause and fix
- [Qwen3.5 MLX KV Cache Bug Issue #37](https://github.com/QwenLM/Qwen3.5/issues/37) — hybrid attention cache crash
- [Seven Failure Points When Engineering a RAG System (arXiv 2401.05856)](https://arxiv.org/html/2401.05856v1) — RAG failure mode taxonomy
- [LlamaIndex: Building RAG from Scratch](https://developers.llamaindex.ai/python/examples/low_level/oss_ingestion_retrieval/) — ingestion vs retrieval pipeline breakdown
- [ChromaDB Persistent Client Docs](https://docs.trychroma.com/docs/run-chroma/persistent-client) — persistence and schema migration caveats
- [MLX Unified Memory Documentation](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html) — Metal memory cap behavior

### Secondary (MEDIUM confidence)
- [DEV Community: RAG Pipeline Deep Dive](https://dev.to/derrickryangiggs/rag-pipeline-deep-dive-ingestion-chunking-embedding-and-vector-search-2877) — component data flow with Python examples
- [Best Chunking Strategies for RAG 2025 (Firecrawl)](https://www.firecrawl.dev/blog/best-chunking-strategies-rag) — chunk size and overlap recommendations
- [Optimizing RAG with Hybrid Search and Reranking (VectorHub)](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking) — recall vs. precision failure patterns
- [I Tested 7 Python PDF Extractors (2025 Edition)](https://onlyoneaman.medium.com/i-tested-7-python-pdf-extractors-so-you-dont-have-to-2025-edition-c88013922257) — PDF parser performance benchmarks
- [NirDiamant/RAG_Techniques (GitHub)](https://github.com/NirDiamant/RAG_Techniques) — advanced RAG patterns reference
- [Apple Silicon Limitations with Local LLM](https://stencel.io/posts/apple-silicon-limitations-with-usage-on-local-llm%20.html) — Metal 75% memory cap behavior
- [Qwen3 Best Settings (jan.ai)](https://www.jan.ai/post/qwen3-settings) — thinking mode temperature warnings

### Tertiary (LOW confidence — needs validation during implementation)
- KSS (Korean Sentence Splitter) Python 3.11 compatibility — assumed from package documentation; verify during Phase 1 setup
- Semantic chunking for Korean/English mixed text — no authoritative source found; requires validation during Phase 4 planning

---
*Research completed: 2026-03-04*
*Ready for roadmap: yes*
