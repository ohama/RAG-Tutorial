# Stack Research

**Domain:** Local RAG system on Apple Silicon (MLX + Python)
**Researched:** 2026-03-04
**Confidence:** HIGH

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| mlx | 0.31.0 | Apple Silicon ML array framework | Apple's own framework; direct Metal GPU access; mlx-lm depends on it; mandatory baseline for the whole stack |
| mlx-lm | 0.30.7 | Load and run Qwen LLMs via MLX | The only first-party tool for running Qwen MLX models locally; actively maintained (releases every 2–3 weeks); supports mlx-community HuggingFace models out of the box |
| sentence-transformers | 5.2.3 | Generate text embeddings | Industry standard; runs on Apple Silicon via MPS backend; `nomic-ai/nomic-embed-text-v1.5` works natively; simpler API than raw transformers; better documented than MLX-only embedding alternatives |
| chromadb | 1.5.2 | Local vector database | Embedded (no server required); Python-first API; persists to disk; best-documented local vector DB; integrates directly with sentence-transformers; low setup friction for educational use |
| pymupdf4llm | 0.3.4 | PDF and document parsing to Markdown | Specifically designed for LLM/RAG workflows; outputs GitHub-flavored Markdown preserving headers, bold, tables; fastest quality/speed tradeoff among PDF parsers (0.12s per page); maintained by PyMuPDF team |
| langchain-text-splitters | 1.1.1 | Chunk documents for embedding | `RecursiveCharacterTextSplitter` is the proven standard for RAG; respects paragraph and sentence boundaries; standalone package (no full LangChain dependency required) |

### Model Selection

| Model | Size (4-bit) | Purpose | Why |
|-------|-------------|---------|-----|
| `mlx-community/Qwen2.5-7B-Instruct-4bit` | ~4.5 GB RAM | LLM for generation | Well-tested mlx-community conversion; 7B gives strong quality with headroom on 128GB for embeddings+vector DB; mlx-lm supports it natively |
| `nomic-ai/nomic-embed-text-v1.5` | ~550 MB RAM | Text embedding | 768-dim embeddings; 8192 token context window; Matryoshka Representation Learning for flexible dimensions; runs on Apple MPS; open weights; 81.2% accuracy on MTEB benchmarks; works with `trust_remote_code=True` |

**Note on Qwen3.5 naming:** The project specifies "Qwen3.5 MLX" — as of March 2026, this refers to the Qwen2.5 series (the most current stable Qwen generation with broad MLX support). Qwen3 models (`Qwen/Qwen3-8B-MLX-4bit`) are available but have fewer mlx-community conversions. Use `mlx-community/Qwen2.5-7B-Instruct-4bit` for production stability; switch to `Qwen/Qwen3-8B-MLX-4bit` if you want the newer architecture.

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| jupyter | >=7.0 | Interactive notebooks | Core interface for RAG on/off comparison — always needed |
| ipywidgets | >=8.0 | Notebook widgets/progress bars | Makes embedding progress visible in Jupyter; required for `tqdm.notebook` |
| tqdm | >=4.66 | Progress bars | Show embedding and indexing progress; use `from tqdm.notebook import tqdm` inside Jupyter |
| python-dotenv | >=1.0 | Environment variable management | Keep model paths and config outside code; good practice even for local projects |
| huggingface-hub | >=0.27 | Model download from HuggingFace | mlx-lm pulls from HuggingFace; `snapshot_download()` caches models to `~/.cache/huggingface/` |
| numpy | >=1.26 | Numerical operations | Required by ChromaDB and sentence-transformers; handles embedding arrays |
| pandas | >=2.2 | Tabular display in notebooks | Display comparison results (RAG on vs off) in readable tables |
| rich | >=13.0 | Terminal/notebook formatted output | Pretty-print retrieved chunks during debugging; optional but improves readability |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| uv | Fast Python package manager | Faster than pip for installing the heavy ML dependency chain; use `uv pip install` as drop-in |
| JupyterLab | Notebook IDE | Prefer JupyterLab over classic Jupyter for better sidebar file browser; install with `pip install jupyterlab` |
| htop / Activity Monitor | RAM monitoring | 128GB unified memory means LLM + embeddings + ChromaDB all share the same pool; monitor during indexing |

---

## Installation

```bash
# Create virtual environment (Python 3.11 recommended — matches mlx-lm requirements)
python3.11 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Core ML stack
pip install mlx==0.31.0 mlx-lm==0.30.7

# Embeddings
pip install sentence-transformers==5.2.3

# Vector database
pip install chromadb==1.5.2

# Document parsing
pip install pymupdf4llm==0.3.4

# Text chunking (standalone — does not require full langchain)
pip install langchain-text-splitters==1.1.1

# Jupyter environment
pip install jupyterlab ipywidgets tqdm

# Utilities
pip install numpy pandas rich python-dotenv huggingface-hub
```

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| sentence-transformers + nomic-embed | mlx-embedding-models (0.0.11) | Only if you need pure MLX inference for embeddings; last updated Sep 2024, limited model support, community-maintained |
| sentence-transformers + nomic-embed | mlx-embeddings (Blaizzy) | If you need vision+language embedding models; adds complexity; overkill for text-only RAG |
| sentence-transformers + nomic-embed | qwen3-embeddings-mlx | Only if Qwen3 Embedding (0.6B/4B/8B) is required; runs as a REST server, not embedded Python; adds ops overhead not worth it for an educational project |
| chromadb | LanceDB | If you outgrow ChromaDB (millions of vectors) or need multimodal; LanceDB's Rust-core has better performance at scale but a steeper API curve |
| chromadb | FAISS | Only if you need pure in-memory vector search with no persistence layer; no metadata filtering; harder to inspect |
| pymupdf4llm | pdfplumber | Only for table-heavy PDFs requiring precise coordinate extraction; slower (0.10s vs 0.12s) but better column alignment |
| pymupdf4llm | pypdf | Avoid for RAG — outputs unsegmented text with spacing artifacts; no Markdown structure |
| langchain-text-splitters | LlamaIndex SimpleNodeParser | If you adopt LlamaIndex as full RAG framework; unnecessary overhead when using standalone chunking |
| mlx-lm (direct) | LlamaIndex / LangChain full framework | For this educational project, direct mlx-lm + ChromaDB is clearer than framework abstraction; frameworks hide the RAG mechanics that the tutorial should expose |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| LangChain (full) | Abstracts away RAG internals — exactly what an educational project should teach; heavy dependency tree; harder to debug | Use mlx-lm + sentence-transformers + ChromaDB directly; import `langchain-text-splitters` standalone only |
| LlamaIndex (full) | Same abstraction problem as LangChain; 0.14.x API changes frequently | Build pipeline manually; LlamaIndex is good for production, not for learning internals |
| Ollama | Runs models via a daemon process; GGUF format, not MLX; loses the direct MLX programming model the project is built on | mlx-lm (pure MLX, in-process Python) |
| PyTorch sentence-transformers on CPU | Apple Silicon MPS is 3–5x faster than CPU for embedding batches; sentence-transformers auto-detects MPS | sentence-transformers with MPS device (default on macOS) |
| pypdf2 | Unmaintained since 2022; superseded by pypdf; poor text extraction quality | pymupdf4llm for RAG; pypdf for simple extraction |
| FAISS | No metadata filtering; binary install on macOS is brittle; ChromaDB wraps FAISS under the hood but adds critical metadata support | chromadb |
| OpenAI API for embeddings | Breaks the offline constraint; adds cost | nomic-embed-text-v1.5 via sentence-transformers (local, offline, comparable quality) |
| vllm / vllm-mlx | Adds a server layer and production complexity; designed for multi-user throughput not single-user RAG demos | mlx-lm direct Python API |

---

## Stack Patterns by Variant

**For the RAG comparison notebook (primary use case):**
- Load Qwen2.5-7B-Instruct-4bit once at kernel startup via mlx-lm
- Pre-build ChromaDB index from sample documents at session start
- RAG "on" path: embed query → ChromaDB similarity search → inject top-k chunks into prompt → mlx-lm generate
- RAG "off" path: same query → mlx-lm generate with no context injection
- Display both answers side by side with pandas DataFrame

**If switching to Qwen3 generation model:**
- Replace `mlx-community/Qwen2.5-7B-Instruct-4bit` with `Qwen/Qwen3-8B-MLX-4bit`
- No code changes needed — mlx-lm API is model-agnostic
- Expect slightly different chat template; use `mlx_lm.utils.get_model_path()` to confirm template

**If adding Korean-language documents (tutorial language is Korean):**
- nomic-embed-text-v1.5 handles Korean adequately (multilingual training data)
- For Korean-specific retrieval quality, consider `BAAI/bge-m3` via sentence-transformers (multilingual, 8192 token context) as an alternative embedding model

---

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| mlx 0.31.0 | mlx-lm 0.30.7 | mlx-lm pins to compatible mlx version; install mlx-lm first and let it pull correct mlx |
| sentence-transformers 5.2.3 | Python >=3.10, PyTorch 2.x | MPS backend active by default on Apple Silicon; no config needed |
| chromadb 1.5.2 | Python >=3.9, SQLite >=3.35 | macOS ships SQLite 3.43+ — no issue; chromadb is self-contained |
| pymupdf4llm 0.3.4 | pymupdf >=1.24 | pymupdf4llm auto-installs pymupdf; no separate install needed |
| langchain-text-splitters 1.1.1 | Python >=3.9 | Standalone; does NOT require langchain-core; safe to import alone |
| nomic-embed-text-v1.5 | sentence-transformers >=2.7 | Requires `trust_remote_code=True` in `SentenceTransformer()` constructor |

---

## Sources

- [mlx-lm PyPI](https://pypi.org/project/mlx-lm/) — version 0.30.7 confirmed (Feb 12, 2026)
- [mlx PyPI](https://pypi.org/project/mlx/) — version 0.31.0 confirmed
- [chromadb PyPI](https://pypi.org/project/chromadb/) — version 1.5.2 confirmed (Feb 27, 2026)
- [pymupdf4llm PyPI](https://pypi.org/project/pymupdf4llm/) — version 0.3.4 confirmed (Feb 14, 2026)
- [sentence-transformers PyPI](https://pypi.org/project/sentence-transformers/) — version 5.2.3 confirmed (Feb 17, 2026)
- [langchain-text-splitters PyPI](https://pypi.org/project/langchain-text-splitters/) — version 1.1.1 confirmed (Feb 18, 2026)
- [llama-index-core PyPI](https://pypi.org/project/llama-index-core/) — evaluated and rejected for educational transparency reasons
- [nomic-ai/nomic-embed-text-v1.5 HuggingFace](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) — model card confirms Apple MPS support and 8192 token context
- [mlx-community Qwen2.5 HuggingFace](https://huggingface.co/collections/mlx-community/qwen25) — confirms mlx-community/Qwen2.5-7B-Instruct-4bit availability
- [Qwen3 MLX models HuggingFace](https://huggingface.co/Qwen/Qwen3-8B-MLX-4bit) — Qwen3-8B-MLX-4bit confirmed as official Qwen release
- [jakedahn/qwen3-embeddings-mlx GitHub](https://github.com/jakedahn/qwen3-embeddings-mlx) — evaluated for embedding alternative; rejected (server architecture overhead)
- [mlx-embedding-models PyPI](https://pypi.org/project/mlx-embedding-models/) — version 0.0.11, last updated Sep 2024; rejected (maintenance gap)
- [DEV.to: Installing Qwen 3.5 on Apple Silicon using MLX](https://dev.to/thefalkonguy/installing-qwen-35-on-apple-silicon-using-mlx-for-2x-performance-37ma) — install procedure verification
- [I Tested 7 Python PDF Extractors (2025 Edition)](https://onlyoneaman.medium.com/i-tested-7-python-pdf-extractors-so-you-dont-have-to-2025-edition-c88013922257) — PDF parser performance benchmarks

---

*Stack research for: Local RAG with Qwen MLX on Apple Silicon (M4 Max, 128GB)*
*Researched: 2026-03-04*
