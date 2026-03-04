# Architecture Research

**Domain:** Local RAG (Retrieval-Augmented Generation) System
**Researched:** 2026-03-04
**Confidence:** HIGH

## Standard Architecture

A local RAG system has two distinct pipelines that share the embedding model but run at different times:

- **Ingestion pipeline** — runs offline (on demand or in batch). Transforms documents into vector embeddings stored in a vector database.
- **Query pipeline** — runs at runtime. Embeds the user query, searches the vector store, assembles context, and calls the local LLM.

These two pipelines must use the **same embedding model** or retrieval quality degrades silently.

### System Overview

```
INGESTION PIPELINE (offline / on-demand)
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  [Raw Docs]                                                         │
│  .md / .pdf / .txt                                                  │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐   │
│  │   Loader    │────▶│   Chunker   │────▶│      Embedder       │   │
│  │             │     │             │     │  (sentence-         │   │
│  │ parse raw   │     │ split text  │     │   transformers)     │   │
│  │ files into  │     │ into chunks │     │  text → vector      │   │
│  │ plain text  │     │ with overlap│     │  [0.12, -0.34, ...] │   │
│  └─────────────┘     └─────────────┘     └──────────┬──────────┘   │
│                                                      │             │
│                                                      ▼             │
│                                          ┌─────────────────────┐   │
│                                          │    Vector Store     │   │
│                                          │    (ChromaDB)       │   │
│                                          │  chunk + vector +   │   │
│                                          │  metadata persisted │   │
│                                          └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘

QUERY PIPELINE (runtime / interactive)
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  [User Query]                                                       │
│  "What is RAG?"                                                     │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐   │
│  │   Embedder  │────▶│  Retriever  │────▶│  Prompt Builder     │   │
│  │  (same      │     │             │     │                     │   │
│  │   model as  │     │ similarity  │     │ system prompt +     │   │
│  │  ingestion) │     │ search top-k│     │ context chunks +    │   │
│  │  query →    │     │ + metadata  │     │ user question       │   │
│  │  vector     │     │   filter    │     │ → full prompt str   │   │
│  └─────────────┘     └──────┬──────┘     └──────────┬──────────┘   │
│                             │                       │             │
│                    ┌────────┘                       ▼             │
│                    │                    ┌─────────────────────┐   │
│                    │   (reads from)     │   LLM Generator     │   │
│                    ▼                    │   (mlx-lm /         │   │
│          ┌─────────────────────┐        │    Qwen3.5 MLX)     │   │
│          │    Vector Store     │        │  prompt → response  │   │
│          │    (ChromaDB)       │        └──────────┬──────────┘   │
│          └─────────────────────┘                   │             │
│                                                    ▼             │
│                                          [Answer to User]        │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| Loader | Parse raw file formats (MD, PDF, TXT) into clean text strings with source metadata | `pypdf`, `pathlib.read_text()`, `python-docx` |
| Chunker | Split text into overlapping chunks; preserve sentence boundaries where possible | `langchain.text_splitter`, or custom recursive splitter |
| Embedder | Convert text strings to dense float vectors; used by both pipelines | `sentence-transformers` (`all-MiniLM-L6-v2` or `bge-small-en-v1.5`) |
| Vector Store | Persist and query chunk vectors with metadata; support similarity search | ChromaDB (persistent, local, no server required) |
| Retriever | Take query vector, return top-k most semantically similar chunks | ChromaDB `.query()` with optional metadata filter |
| Prompt Builder | Assemble system prompt + retrieved context + user question into a single string | Pure Python string formatting |
| LLM Generator | Accept a prompt string, return a completion string; runs fully locally | `mlx-lm` with Qwen3.5 model on Apple Silicon |

## Recommended Project Structure

```
RAG/
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py          # parse MD, PDF, TXT → Document objects
│   │   ├── chunker.py         # split Document text → list of chunks
│   │   └── pipeline.py        # orchestrate: load → chunk → embed → store
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retriever.py       # embed query, search vector store, return chunks
│   │   └── prompt_builder.py  # assemble final prompt string from chunks + query
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   └── generator.py       # call mlx-lm, return response string
│   │
│   ├── store/
│   │   ├── __init__.py
│   │   └── vector_store.py    # ChromaDB wrapper: init, add, query, delete
│   │
│   ├── embedding/
│   │   ├── __init__.py
│   │   └── embedder.py        # sentence-transformers wrapper; shared by both pipelines
│   │
│   └── rag.py                 # high-level RAG class: ties ingestion + query together
│
├── tutorial/
│   ├── 01_document_loading.ipynb
│   ├── 02_chunking_strategies.ipynb
│   ├── 03_embeddings_and_vector_store.ipynb
│   ├── 04_retrieval.ipynb
│   ├── 05_prompt_engineering.ipynb
│   ├── 06_generation_with_mlx.ipynb
│   └── 07_end_to_end_rag.ipynb
│
├── docs/                      # sample documents for demos
│   ├── sample_markdown.md
│   ├── sample_paper.pdf
│   └── sample_notes.txt
│
├── data/
│   └── chroma_db/             # ChromaDB persistence directory (gitignored)
│
├── tests/
│   ├── test_loader.py
│   ├── test_chunker.py
│   ├── test_embedder.py
│   └── test_retriever.py
│
├── pyproject.toml
└── README.md
```

### Structure Rationale

- **`src/ingestion/`** — Separates offline pipeline into its own module. Can be run as a script independently: `python -m src.ingestion.pipeline --docs ./docs`. The `pipeline.py` orchestrator is the only file that knows about all ingestion steps.
- **`src/retrieval/`** — Separates runtime pipeline. The `retriever.py` knows about the vector store; `prompt_builder.py` knows nothing about storage, only about text assembly. This split makes prompt templates easy to iterate independently.
- **`src/generation/`** — Thin wrapper around `mlx-lm`. Keeps LLM backend swappable (if someone wants to try Ollama instead, only this file changes).
- **`src/store/`** — ChromaDB operations isolated here. If the vector DB changes (e.g., swap to FAISS), only this module changes.
- **`src/embedding/`** — Single place for the embedding model. Both ingestion and query pipelines import from here, guaranteeing the same model is used. This is the most critical shared dependency.
- **`src/rag.py`** — Public facade for simple use cases. Lets notebooks call `rag.ingest(path)` and `rag.query("question")` without knowing the internal structure.
- **`tutorial/`** — Numbered notebooks that walk through each component individually, then combine them. Educational clarity is first-class, not an afterthought.

## Architectural Patterns

### Pattern 1: Shared Embedder Singleton

**What:** Both the ingestion pipeline and query pipeline import the same `Embedder` class from `src/embedding/embedder.py`. The model is loaded once and reused.

**When to use:** Always. Using different model instances or different model names between ingestion and query time is the most silent failure mode in RAG.

**Trade-offs:** Model loads into memory once (~90MB for MiniLM). On 128GB unified memory this is negligible. Bigger models (e.g., bge-m3 at ~500MB) are also fine.

**Example:**
```python
# src/embedding/embedder.py
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name  # stored for metadata/auditing

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, convert_to_list=True)

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
```

### Pattern 2: Document Object with Metadata

**What:** Every chunk carries its source metadata (file path, page number, chunk index) through the entire pipeline, so the retriever can return source attribution alongside the text.

**When to use:** Always. Without metadata, you cannot cite sources, filter by file, or debug which documents are being retrieved.

**Trade-offs:** Slightly more data per chunk in ChromaDB. Negligible at any scale you'd run locally.

**Example:**
```python
# src/ingestion/loader.py
from dataclasses import dataclass

@dataclass
class Document:
    text: str
    source: str       # file path
    page: int | None  # for PDFs
    doc_type: str     # "markdown" | "pdf" | "txt"

# Chunks produced by chunker inherit this metadata
@dataclass
class Chunk:
    text: str
    source: str
    page: int | None
    chunk_index: int
    doc_type: str
```

### Pattern 3: Two-Stage Pipeline Separation (Ingest vs Query)

**What:** Keep ingestion and query as completely separate code paths with no shared state except the vector store. Ingestion can run as a CLI script or scheduled job. The query pipeline reads from the store but never writes.

**When to use:** Always. Mixing ingest and query logic leads to accidental re-embedding on every query, which is slow and wastes compute.

**Trade-offs:** Requires a persistent vector store (ChromaDB handles this). Running ChromaDB in-memory would break the separation.

**Example:**
```python
# Ingest (run once, or when docs change)
from src.ingestion.pipeline import ingest_documents
ingest_documents(docs_path="./docs", collection_name="my_docs")

# Query (runs on every user question)
from src.rag import RAG
rag = RAG(collection_name="my_docs")
answer = rag.query("What is RAG?")
```

### Pattern 4: Prompt Template as a Config Value

**What:** The system prompt and context injection template live in a dedicated `prompt_builder.py`, not inline in the generator or retriever. Prompt text is a parameter, not hardcoded.

**When to use:** From the start. Prompt iteration is the highest-leverage optimization in RAG. If prompts are buried inside other logic, iteration is painful.

**Example:**
```python
# src/retrieval/prompt_builder.py
SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question
using only the provided context. If the answer is not in the context, say so.
Always cite the source document name."""

def build_prompt(query: str, chunks: list[Chunk]) -> str:
    context = "\n\n".join(
        f"[Source: {c.source}]\n{c.text}" for c in chunks
    )
    return f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}"
```

## Data Flow

### Ingestion Data Flow

```
Raw file on disk
    │
    ▼  [Loader]
Document(text=..., source="./docs/paper.pdf", page=3, doc_type="pdf")
    │
    ▼  [Chunker]
[
  Chunk(text="first 512 chars...", source="./docs/paper.pdf", chunk_index=0),
  Chunk(text="chars 462-974...", source="./docs/paper.pdf", chunk_index=1),
  ...
]
    │
    ▼  [Embedder]
[
  (Chunk, vector=[0.12, -0.34, ...]),  # 384 dims for MiniLM
  (Chunk, vector=[0.05,  0.11, ...]),
  ...
]
    │
    ▼  [Vector Store (ChromaDB)]
Persisted to disk at ./data/chroma_db/
  - id: "paper_pdf_chunk_0"
  - embedding: [0.12, -0.34, ...]
  - document: "first 512 chars..."
  - metadata: {source: "...", page: 3, chunk_index: 0}
```

### Query Data Flow

```
User question: "What is retrieval-augmented generation?"
    │
    ▼  [Embedder] (same model as ingestion)
query_vector = [0.08, -0.29, ...]  # 384 dims
    │
    ▼  [Retriever] (ChromaDB .query())
top_k=5 chunks with similarity scores:
  - Chunk("RAG is a technique...", score=0.91, source="paper.pdf")
  - Chunk("Retrieval systems...", score=0.87, source="notes.md")
  ...
    │
    ▼  [Prompt Builder]
"""You are a helpful assistant. Answer using only the provided context.

Context:
[Source: paper.pdf]
RAG is a technique that combines...

[Source: notes.md]
Retrieval systems work by...

Question: What is retrieval-augmented generation?"""
    │
    ▼  [LLM Generator] (mlx-lm, Qwen3.5)
"Retrieval-Augmented Generation (RAG) is a technique that combines..."
    │
    ▼
User sees the answer
```

### Key Data Flows Summary

1. **Ingestion:** File path → Document → Chunks → Vectors → ChromaDB (write once)
2. **Query:** User string → Query vector → Similar chunks → Prompt string → LLM response (read-only on ChromaDB)
3. **Shared dependency:** Embedder is the only component used by both pipelines — changes to the embedding model require re-ingestion

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 0-10k chunks | Default: ChromaDB local persistence, MiniLM embedder, single process |
| 10k-200k chunks | Same architecture, but run ingestion as a background job; add metadata pre-filtering to reduce retrieval set before vector search |
| 200k+ chunks | Swap ChromaDB for FAISS with HNSW index, or consider pgvector; consider a larger embedding model (bge-m3) if retrieval quality degrades |

### Scaling Priorities

1. **First bottleneck (retrieval quality):** Bad chunks, not bad search. If answers are poor, improve chunking strategy first — add overlap, try semantic chunking — before changing the embedding model or vector DB.
2. **Second bottleneck (speed):** Embedding at query time is fast (~14ms for MiniLM). The LLM generation dominates latency. On M4 Max, Qwen3.5 at 4-bit is fast enough for interactive use. If latency matters, reduce max_new_tokens or use a smaller model.

## Anti-Patterns

### Anti-Pattern 1: Different Embedding Models at Ingest vs Query Time

**What people do:** Use `all-MiniLM-L6-v2` during ingestion, then switch to `bge-small-en-v1.5` for querying (or forget which model was used).

**Why it's wrong:** Vector spaces are model-specific. Comparing a query vector from model A against chunk vectors from model B produces semantically meaningless similarity scores. Retrieval appears to "work" (no errors) but returns wrong chunks. This is the hardest bug to detect.

**Do this instead:** Store the embedding model name as metadata in ChromaDB's collection settings on creation. Assert at query time that the collection model matches the configured embedder.

### Anti-Pattern 2: Chunk Size Too Large, No Overlap

**What people do:** Set chunk size to 2000 characters with no overlap to "keep context together." Or chunk at paragraph boundaries without regard for length.

**Why it's wrong:** Large chunks dilute the embedding signal — the vector represents too many ideas at once, weakening similarity search. No overlap means a sentence split across chunk boundaries loses coherent meaning for retrieval.

**Do this instead:** Target 512–800 characters per chunk with 10–15% overlap (50–100 chars). For Markdown, split at heading boundaries first, then by character limit. For PDF, split by page-then-character.

### Anti-Pattern 3: Retrieving Too Many Chunks (Large k)

**What people do:** Set `top_k=20` to "make sure the answer is in there."

**Why it's wrong:** LLMs have a "lost in the middle" problem — they reliably use information at the beginning and end of context, but ignore information buried in the middle. 20 chunks of ~600 chars each is ~12k characters; most is ignored. Also inflates the prompt and LLM inference time.

**Do this instead:** Start with `top_k=4` or `top_k=5`. If retrieval quality is poor, fix chunking and embedding first, not k. Only increase k with a re-ranker that selects the best 3-5 from a larger candidate pool.

### Anti-Pattern 4: Monolithic RAG Class (All Logic in One File)

**What people do:** Build one `rag.py` that loads files, splits text, embeds, stores, retrieves, builds prompts, and calls the LLM.

**Why it's wrong:** Every component becomes untestable in isolation. Swapping the LLM or vector store requires touching one giant file. Notebooks can't demonstrate individual concepts.

**Do this instead:** Follow the project structure above. `src/rag.py` is just a thin facade that imports from the individual modules. Each module has a single responsibility and can be tested or demonstrated independently.

### Anti-Pattern 5: Re-embedding on Every Query

**What people do:** Load documents, embed them, and store them in-memory (e.g., a plain list), without persisting to disk. Every time the app starts, it re-ingests all documents.

**Why it's wrong:** Wasted compute on every startup. Slow for any non-trivial document set. Encourages mixing ingestion and query logic.

**Do this instead:** Use ChromaDB with a persistent directory (`client = chromadb.PersistentClient(path="./data/chroma_db")`). Ingestion runs once; querying reads from disk.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| mlx-lm (Qwen3.5) | Python API: `mlx_lm.load()` + `mlx_lm.generate()` | Runs fully locally; no network call; uses Apple Neural Engine via unified memory |
| sentence-transformers | Python API: `SentenceTransformer(model_name).encode()` | Downloads model from HuggingFace on first run; cached locally at `~/.cache/huggingface/` |
| ChromaDB | Python API: `chromadb.PersistentClient(path=...)` | Embedded DB, no server process; persistence directory committed to `.gitignore` |
| pypdf | Python API: `PdfReader(path)` | Local PDF parsing; no external calls |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| Loader → Chunker | `list[Document]` passed directly | Pure data handoff, no shared state |
| Chunker → Embedder | `list[Chunk]` with text extracted | Embedder only needs `chunk.text`; other fields preserved via Python objects |
| Embedder → Vector Store | `list[Chunk]`, `list[list[float]]` paired | Vector store stores both; embedder doesn't know about storage |
| Vector Store → Retriever | ChromaDB client instance (dependency injected) | Retriever calls `.query()`; never calls `.add()` |
| Retriever → Prompt Builder | `list[Chunk]` (retrieved) + query string | Prompt builder is pure: no I/O, no DB calls |
| Prompt Builder → Generator | Prompt string | Single string in, single string out; cleanest boundary in the system |

## Sources

- [LlamaIndex: Building RAG from Scratch (Open-source)](https://developers.llamaindex.ai/python/examples/low_level/oss_ingestion_retrieval/) — Authoritative breakdown of ingestion vs retrieval pipeline steps with code
- [DEV Community: RAG Pipeline Deep Dive — Ingestion, Chunking, Embedding, Vector Search](https://dev.to/derrickryangiggs/rag-pipeline-deep-dive-ingestion-chunking-embedding-and-vector-search-2877) — Component-by-component data flow with Python examples
- [Morphik: Guide to Open-Source RAG Frameworks 2025](https://www.morphik.ai/blog/guide-to-oss-rag-frameworks-for-developers) — Six-component production RAG anatomy
- [Capella Solutions: FAISS vs Chroma](https://www.capellasolutions.com/blog/faiss-vs-chroma-lets-settle-the-vector-database-debate) — ChromaDB recommended for local/educational RAG under 200k vectors
- [ML-Explore: mlx-lm GitHub](https://github.com/ml-explore/mlx-lm) — Official MLX LM Python API for local LLM inference on Apple Silicon
- [Data Science Collective: RAG Architectures Complete Guide 2025](https://medium.com/data-science-collective/rag-architectures-a-complete-guide-for-2025-daf98a2ede8c) — Anti-patterns including lost-in-middle, chunk size, and metadata filtering
- [Weaviate: Verba Open Source Modular RAG](https://weaviate.io/blog/verba-open-source-rag-app) — Modular component architecture reference
- [Hacker News: Don't use all-MiniLM-L6-v2 for new embeddings](https://news.ycombinator.com/item?id=46081800) — Community discussion on embedding model selection tradeoffs

---
*Architecture research for: Local RAG System (Mac Studio M4 Max, MLX, Python)*
*Researched: 2026-03-04*
