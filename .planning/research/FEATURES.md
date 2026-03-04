# Feature Research

**Domain:** Local RAG Educational Tutorial (Jupyter Notebook, Apple Silicon MLX)
**Researched:** 2026-03-04
**Confidence:** HIGH

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features a learner expects in any RAG tutorial. Missing these = tutorial feels incomplete or untrustworthy.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Document loading (Markdown, PDF, TXT) | RAG is pointless without documents to retrieve from — this is the entry point | LOW | Use `pypdf`, `pathlib`; keep loaders simple and readable |
| Fixed-size chunking with overlap | Every RAG tutorial demonstrates this; it is the baseline everyone learns first | LOW | Chunk size 400–512 tokens, 10–20% overlap; expose parameters as variables |
| Embedding generation (local model) | Core concept of RAG — must be visible and understandable | MEDIUM | Use sentence-transformers (e.g., `all-MiniLM-L6-v2`) via ONNX or CPU; avoid cloud APIs |
| Vector storage and similarity search | Retrieval cannot happen without an index to search | MEDIUM | ChromaDB or FAISS — both run fully local with no server needed |
| Context injection into prompt | The actual "augmented" part of RAG — learner must see exactly what is injected | LOW | Show the full constructed prompt in a notebook cell output |
| LLM generation with Qwen3.5 MLX | The generation step using the local model | MEDIUM | Load via `mlx-lm`; expose generation parameters (temperature, max_tokens) |
| RAG on vs. RAG off comparison | The primary educational deliverable; stated as core project value | MEDIUM | Side-by-side notebook cell with identical query, one with context, one without |
| Sample documents included | Learners need something to run immediately — "bring your own docs" is a barrier to entry | LOW | 3–5 well-chosen Markdown/TXT files on a clear topic the model lacks parametric knowledge of |
| Source citations in answers | Shows learner that retrieved chunks drove the answer — grounding is the point | LOW | Display chunk text + source filename + chunk index after every RAG answer |
| Korean-language explanations in notebooks | Stated as a project requirement; the audience reads Korean | LOW | All Markdown cells in Korean; all code + comments in English |

### Differentiators (Competitive Advantage)

Features that make this tutorial stand out versus generic RAG tutorials.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Chunking strategy comparison notebook | Most tutorials use one strategy; showing fixed vs. recursive vs. semantic side-by-side teaches *why* it matters | MEDIUM | Single notebook cell per strategy, same query, display retrieved chunks + score |
| Chunk parameter sensitivity demo | Interactive (ipywidgets slider or parameter cell) showing how chunk_size and overlap change retrieved results | MEDIUM | Even a static multi-parameter comparison table is valuable; full widgets are a bonus |
| Embedding score visualization | Display cosine similarity scores for every retrieved chunk, not just top answer | LOW | `matplotlib` bar chart of scores per chunk; makes "similarity search" tangible |
| Retrieval failure examples | Show queries that fail with small chunks, succeed with larger ones, or vice versa | LOW | 2–3 intentionally bad queries; demonstrates when RAG breaks, not just when it works |
| MLX-specific performance notes | Apple Silicon / unified memory advantages explained; shows why local is viable | LOW | One markdown cell per notebook explaining what MLX does and why it is fast here |
| Progressive complexity across tutorial files | Tutorial organized from simplest RAG (00_basics) to advanced (03_evaluation) | MEDIUM | Mirrors how a learner's mental model builds; each file standalone-runnable |
| RAGAS-style qualitative evaluation cell | Show faithfulness and relevance assessment without requiring a full RAGAS install | MEDIUM | Manual or LLM-as-judge evaluation in a notebook cell; teaches *how* to think about evaluation |
| User document drop-in support | Clear `docs/` directory with a README explaining how to add personal files | LOW | Structured directory + loading code that auto-discovers files; no code changes needed |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem useful but undermine educational clarity for this project.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Web UI (Gradio, Streamlit) | Looks polished and shareable | Hides the RAG pipeline behind abstraction; learner sees buttons, not prompts and retrieval | Use Jupyter with clear cell structure and outputs; the notebook IS the interface |
| LangChain or LlamaIndex framework | "Everyone uses it," reduces boilerplate | Hides every step that is the point of the tutorial; `chain.invoke()` teaches nothing about RAG | Implement document loading, chunking, embedding, retrieval, and generation explicitly with plain Python |
| Cloud embedding APIs (OpenAI, Cohere) | Higher quality embeddings, easy to call | Requires API keys, internet, and cost — breaks "fully local" constraint; also hides what embedding is | Use a local sentence-transformers model; explain that cloud models work the same way |
| Production error handling | Makes code robust | Verbose try/except blocks and retry logic obscure the happy path a learner needs to see | One sentence per risk area in a markdown cell; minimal inline error handling |
| Async / streaming generation | Feels more responsive | Adds concurrency complexity that is irrelevant to learning RAG concepts | Synchronous generation is clearer; note that production systems add streaming |
| Multi-turn conversational memory | "Chat with your docs" is compelling | Requires state management that distracts from the retrieval concept being taught | Single-turn Q&A is sufficient to demonstrate RAG; note conversational RAG as a future extension |
| Graph RAG or knowledge graph retrieval | Advanced and exciting | Requires a separate database (Neo4j), complex schema; appropriate for an advanced follow-on | Dense vector retrieval first; mention Graph RAG as a differentiator in a "what next" section |
| Automatic hyperparameter tuning | Seems efficient | Removes the learner from the decision-making process; the *point* is to see how parameters affect results | Explicit parameter cells the learner changes manually |
| Docker / containerization | Makes setup reproducible | Adds operational complexity for a local, educational project; Mac Studio setup is the environment | A `requirements.txt` and clear setup instructions in the tutorial are sufficient |
| Multimodal (image, audio) document support | Expands scope | Completely different embedding pipeline; dilutes the text RAG concept | Out of scope; mention in a "future directions" cell |

---

## Feature Dependencies

```
[Document Loading]
    └──requires──> [Document Parsing + Cleaning]
                       └──requires──> [Chunking]
                                          └──requires──> [Embedding Generation]
                                                             └──requires──> [Vector Store Index]
                                                                                └──requires──> [Similarity Search / Retrieval]
                                                                                                   └──requires──> [Prompt Construction]
                                                                                                                      └──requires──> [LLM Generation (Qwen3.5 MLX)]

[RAG on/off Comparison]
    └──requires──> [Full pipeline above] + [Baseline LLM call without retrieval]

[Source Citation Display]
    └──requires──> [Similarity Search] (chunk source metadata must be preserved)

[Chunking Strategy Comparison]
    └──requires──> [Embedding Generation] (need same embedding to compare chunk strategies)

[Embedding Score Visualization]
    └──requires──> [Similarity Search] (scores come from retrieval step)

[RAGAS-style Evaluation]
    └──requires──> [Full RAG pipeline] + [Ground-truth Q&A pairs in sample docs]

[User Document Drop-in]
    └──requires──> [Document Loading] (must support arbitrary files, not hardcoded paths)
```

### Dependency Notes

- **Chunking requires Document Loading:** Chunk strategy has no meaning until text is loaded and parsed; loading must handle Markdown, PDF, TXT before chunking is demonstrated.
- **Embedding requires Chunking:** The embedding model runs on chunks, not raw documents; chunk quality directly determines embedding quality — this sequencing is the core educational insight.
- **Similarity Search requires Vector Store Index:** The index must be built before any retrieval demo; building the index is Phase 1, retrieval is Phase 2.
- **RAG on/off comparison requires the full pipeline:** This is the capstone demo; it cannot be built before all components exist. Plan it as the final notebook or section.
- **Source citation requires metadata preservation:** If chunk source info is not stored in the vector DB at index time, it cannot be retrieved later. This must be designed into the ingestion step from the start — it is a hidden dependency that causes rewrites if missed.
- **Evaluation requires pre-made ground truth:** RAGAS-style evaluation needs questions with known correct answers. Sample documents must be chosen with this in mind from the start.

---

## MVP Definition

### Launch With (v1)

Minimum viable tutorial — what validates that the project works and teaches the concept.

- [ ] Document loading for Markdown, PDF, TXT — needed for any demo
- [ ] Fixed-size chunking with configurable size and overlap — baseline every RAG tutorial covers
- [ ] Local embedding with a sentence-transformers model — must be local; embedding is a core concept
- [ ] ChromaDB or FAISS vector store (local, no server) — simple enough to understand in one cell
- [ ] Top-k similarity retrieval with source metadata returned — retrieval with citations
- [ ] Prompt construction showing retrieved context explicitly — the learner must see the full prompt
- [ ] Qwen3.5 MLX generation step — the local LLM output
- [ ] RAG on vs. RAG off comparison notebook — the core educational deliverable
- [ ] Source citations displayed with each answer — grounds the output in retrieved documents
- [ ] 3–5 curated sample documents — so learners can run everything immediately
- [ ] Korean explanations throughout — required by project context

### Add After Validation (v1.x)

Features to add once core pipeline is working and demonstrably educational.

- [ ] Chunking strategy comparison notebook — add when baseline is solid; teaches strategy tradeoffs
- [ ] Embedding score visualization — add when retrieval is working; makes similarity tangible
- [ ] Retrieval failure examples — add after the success path is clear; teaches robustness
- [ ] User document drop-in directory — add once the loading code is proven with sample docs

### Future Consideration (v2+)

Features to defer until the core tutorial is validated.

- [ ] RAGAS-style evaluation notebook — requires ground-truth Q&A pairs; time-consuming to design well; valuable but not MVP
- [ ] Chunk parameter sensitivity demo (interactive) — ipywidgets add complexity; a static parameter table is sufficient for v1
- [ ] Hybrid search (BM25 + dense) — conceptually important but requires an additional library (`rank-bm25`); teach dense-only first
- [ ] Korean language embedding model comparison — valuable for Korean-document use cases but adds scope; defer

---

## Feature Prioritization Matrix

| Feature | Learner Value | Implementation Cost | Priority |
|---------|---------------|---------------------|----------|
| Document loading (Markdown, PDF, TXT) | HIGH | LOW | P1 |
| Fixed-size chunking | HIGH | LOW | P1 |
| Local embedding generation | HIGH | MEDIUM | P1 |
| Vector store (ChromaDB or FAISS) | HIGH | MEDIUM | P1 |
| Similarity retrieval with metadata | HIGH | MEDIUM | P1 |
| Prompt construction (visible) | HIGH | LOW | P1 |
| Qwen3.5 MLX generation | HIGH | MEDIUM | P1 |
| RAG on/off comparison notebook | HIGH | MEDIUM | P1 |
| Source citation display | HIGH | LOW | P1 |
| Sample documents | HIGH | LOW | P1 |
| Korean explanations | HIGH | LOW | P1 |
| Chunking strategy comparison | HIGH | MEDIUM | P2 |
| Embedding score visualization | MEDIUM | LOW | P2 |
| Retrieval failure examples | MEDIUM | LOW | P2 |
| User document drop-in support | MEDIUM | LOW | P2 |
| RAGAS-style evaluation | MEDIUM | MEDIUM | P3 |
| Chunk parameter sensitivity (interactive) | MEDIUM | MEDIUM | P3 |
| Hybrid search (BM25 + dense) | MEDIUM | MEDIUM | P3 |

**Priority key:**
- P1: Must have for tutorial launch
- P2: Should have, add when core pipeline works
- P3: Nice to have, future consideration

---

## Competitor Feature Analysis

RAG educational projects surveyed: LLM Zoomcamp RAG prototype, NirDiamant/RAG_Techniques, guyernest/advanced-rag, typical LangChain/LlamaIndex quickstart tutorials.

| Feature | Typical Tutorial | Advanced RAG Repo | Our Approach |
|---------|------------------|-------------------|--------------|
| Framework usage | LangChain or LlamaIndex | Mixed (LangChain + custom) | Explicit plain Python — no framework abstraction |
| Embedding | OpenAI API or HuggingFace | Varies | Local sentence-transformers, MLX-compatible |
| LLM | OpenAI GPT / Anthropic | Various cloud APIs | Fully local Qwen3.5 MLX |
| RAG comparison demo | Rarely included | Sometimes | Core deliverable — on/off comparison notebook |
| Chunking strategies | One strategy (fixed) | Multiple advanced | Fixed baseline + comparison across strategies |
| Source citations | Sometimes | Usually | Always shown; mandatory for educational grounding |
| Evaluation metrics | Rarely in beginner tutorials | RAGAS in advanced | Simple qualitative in v1; RAGAS-style in v2+ |
| Language | English | English | Korean explanations, English code |
| Hardware | Cloud / any | Any | Apple Silicon (MLX) specific |

**Our differentiation:** Fully local execution, Korean-language explanations, explicit no-framework pipeline, and a RAG on/off comparison as the primary artifact — not just a working system, but a visible learning experience.

---

## Sources

- [RAG Pipeline Deep Dive: Ingestion, Chunking, Embedding, and Vector Search (DEV Community)](https://dev.to/derrickryangiggs/rag-pipeline-deep-dive-ingestion-chunking-embedding-and-vector-search-2877)
- [Chunking Strategies for RAG (DataCamp)](https://www.datacamp.com/blog/chunking-strategies)
- [Best Chunking Strategies for RAG in 2025 (Firecrawl)](https://www.firecrawl.dev/blog/best-chunking-strategies-rag)
- [Document Chunking for RAG: 9 Strategies Tested (LLM Practical Experience Hub)](https://langcopilot.com/posts/2025-10-11-document-chunking-for-rag-practical-guide)
- [RAGAS Evaluation Metrics Documentation](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)
- [RAG Evaluation Metrics Guide 2025 (FutureAGI)](https://futureagi.com/blogs/rag-evaluation-metrics-2025)
- [Mastering RAG Evaluation 2025 (Tredence)](https://www.tredence.com/blog/understanding-rag-systems-the-future-of-ai-interactions)
- [Six Lessons Learned Building RAG Systems in Production (Towards Data Science)](https://towardsdatascience.com/six-lessons-learned-building-rag-systems-in-production/)
- [Designing RAG Systems: Patterns, Tradeoffs, and Code Examples (Medium, Jan 2026)](https://medium.com/@kyle.zarif/designing-rag-systems-patterns-tradeoffs-and-code-examples-95c33a8b2df7)
- [Citation-Aware RAG: Fine-Grained Citations in Retrieval (Tensorlake)](https://www.tensorlake.ai/blog/rag-citations)
- [Optimizing RAG with Hybrid Search and Reranking (VectorHub)](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking)
- [Prototype RAG in Jupyter Notebook (Hung's Notebook)](https://hung.bearblog.dev/llm-zoomcamp-1-rag/)
- [MLX Documentation (Apple)](https://ml-explore.github.io/mlx/)
- [NirDiamant/RAG_Techniques (GitHub)](https://github.com/NirDiamant/RAG_Techniques)

---
*Feature research for: Local RAG Educational Tutorial (Mac Studio M4 Max, Qwen3.5 MLX)*
*Researched: 2026-03-04*
