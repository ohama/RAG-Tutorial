# Pitfalls Research

**Domain:** Local RAG system — MLX + Apple Silicon, Korean/English bilingual, Jupyter notebook, educational
**Researched:** 2026-03-04
**Confidence:** HIGH

---

## Critical Pitfalls

### Pitfall 1: Qwen3.5 KV Cache Initialization Bug with MLX

**What goes wrong:**
When initializing KV cache manually with `[KVCache() for _ in range(num_layers)]` for Qwen3.5, multi-turn conversations crash during the prefill stage on the second turn. The crash is a `TypeError` because `KVCache.make_mask()` is called with wrong arguments — Qwen3.5 is a hybrid model using both linear attention (ArrayCache) and full attention (KVCache), and manual initialization does not account for this mixed architecture.

**Why it happens:**
Tutorial code copied from simpler Qwen2/Qwen3 examples uses a uniform cache initialization pattern. Qwen3.5 introduced hybrid attention layers but the breaking change was subtle. The signature mismatch `cache.make_mask(h.shape[1])` vs. the required `cache.make_mask(h.shape[1], window_size, return_array)` crashes silently on the second conversation turn, not the first, making it look like a conversation state bug.

**How to avoid:**
Always use `mlx_lm.utils.make_prompt_cache(model)` instead of manually constructing cache arrays. This function introspects the model architecture and creates the correct cache type per layer automatically. Never copy cache initialization code from examples targeting different model families.

```python
# WRONG — crashes on second turn for Qwen3.5
cache = [KVCache() for _ in range(model.n_layers)]

# CORRECT — works for hybrid attention models
from mlx_lm.utils import make_prompt_cache
cache = make_prompt_cache(model)
```

**Warning signs:**
- First conversation turn works; second turn crashes with `TypeError`
- Error mentions `make_mask` or wrong number of arguments
- Only happens with Qwen3.5 (not Qwen3 or earlier Qwen2)

**Phase to address:**
Model loading and inference setup phase — validate multi-turn cache before building RAG pipeline on top.

---

### Pitfall 2: MLX KV Cache Memory Bloat Killing Jupyter Kernels

**What goes wrong:**
MLX's internal cache grows linearly with token count during prompt processing. An 8B model with a 10k token prompt can consume 26GB of app memory instead of the expected 4–5GB. In a Jupyter notebook where the kernel stays alive across cells, repeated RAG queries accumulate cache without release, eventually killing the kernel or swapping to disk, making inference 10–100x slower.

**Why it happens:**
MLX aggressively caches intermediate computations during prompt processing for potential reuse. With RAG queries, each query has a different context (different retrieved chunks), so there is almost zero cache reuse, but the cache still grows. Jupyter notebooks compound this because the Python process lives for the entire session — no automatic cleanup occurs between cells.

**How to avoid:**
Call `mx.metal.clear_cache()` after each RAG inference call. Structure the notebook so model loading happens once in an early cell, and each query cell explicitly clears the cache after getting the response. Monitor memory with `mx.metal.get_active_memory()`.

```python
import mlx.core as mx

def rag_query(model, tokenizer, question, context_chunks):
    prompt = build_prompt(question, context_chunks)
    response = generate(model, tokenizer, prompt)
    mx.metal.clear_cache()  # CRITICAL: prevent accumulation
    return response
```

**Warning signs:**
- Activity Monitor shows MLX memory growing across queries
- Generation speed degrades significantly after 5–10 queries
- Jupyter kernel crashes mid-session with no error message
- First query: 3s. Tenth query: 45s.

**Phase to address:**
Core RAG pipeline implementation — bake `clear_cache()` into the inference wrapper from day one.

---

### Pitfall 3: Mismatched Embedding and Generation Model Families Causing Retrieval Failure

**What goes wrong:**
Using an English-dominant embedding model (e.g., `all-MiniLM-L6-v2`) for a Korean+English corpus produces semantically meaningless vectors for Korean text. Korean queries retrieve English chunks, and Korean documents are never surfaced, even when directly relevant. The system "works" — it returns results — but retrieval quality for Korean content is near random.

**Why it happens:**
Most embedding model tutorials target English. Developers see good results in English testing and assume multilingual capability. `all-MiniLM-L6-v2` has extremely limited Korean training data. The failure is silent: no error, just wrong retrievals that look plausible.

**How to avoid:**
Use `BAAI/bge-m3` as the embedding model. It was trained on 100+ languages including Korean and supports up to 8192 token inputs. Verify multilingual retrieval quality explicitly with Korean-only test queries during setup. Run a 5-query sanity check: Korean query → must retrieve Korean document.

Note: BGE-M3 uses the XLM-RoBERTa tokenizer, which handles Korean subword tokenization correctly. Do not substitute any English-only tokenizer.

**Warning signs:**
- Korean queries return only English documents
- Cosine similarity scores for Korean↔Korean pairs are lower than Korean↔English pairs
- Retrieval results look correct in English tests but feel wrong in bilingual use

**Phase to address:**
Embedding model selection phase — test Korean retrieval before indexing the full corpus.

---

### Pitfall 4: Fixed Chunk Size Destroying Cross-Sentence and Korean Sentence Context

**What goes wrong:**
Default chunking with `chunk_size=500` and no overlap splits documents at character/token boundaries that break Korean sentences mid-morpheme or split a concept across two chunks with no overlap. A question about a concept that spans two paragraphs retrieves chunk A or chunk B, never both, so the LLM gets incomplete context and either hallucinates or says "I don't know."

**Why it happens:**
Most chunking tutorials use English examples where word boundaries are clear. Korean uses agglutinative morphology — splitting at token 500 may cut a grammatical particle off its stem, creating malformed text. Without overlap, boundary information is permanently lost. Developers use default settings from LangChain/LlamaIndex without understanding the language-specific implications.

**How to avoid:**
- Use `chunk_size=512` tokens with `chunk_overlap=128` tokens (25% overlap minimum).
- For Korean text, use a Korean-aware sentence splitter (KSS — Korean Sentence Splitter) rather than NLTK's punkt tokenizer, which is English-only.
- Test chunking output by printing 5 random chunks and verifying they are complete thoughts.
- Consider semantic chunking for documents with clear section structure.

```python
# English-only sentence splitting — WRONG for Korean
from nltk.tokenize import sent_tokenize  # punkt: English only

# Korean-aware splitting
import kss
sentences = kss.split_sentences(korean_text)
```

**Warning signs:**
- Chunks end mid-sentence (visible in debug output)
- Questions about concepts that span multiple paragraphs return wrong answers
- Korean chunks contain morpheme fragments at boundaries

**Phase to address:**
Document ingestion and chunking phase — validate chunk quality before embedding.

---

### Pitfall 5: MLX Metal GPU Memory Cap at 75% Causing Silent Performance Collapse

**What goes wrong:**
Apple's Metal API caps GPU memory allocation at approximately 75% of unified RAM by default. On a 128GB M4 Max system, this means ~96GB maximum for GPU operations. When model + KV cache + embeddings approach this limit, inference does not fail with an OOM error — it silently falls back to CPU or swaps, causing 10–50x performance degradation with no clear error message.

**Why it happens:**
Developers assume 128GB means 128GB available for the model. The 75% cap is a system-level Metal safeguard. Additionally, running the embedding model and LLM simultaneously (e.g., in a single Jupyter session) can push total GPU memory past the cap.

**How to avoid:**
- Load embedding model first, embed the entire corpus, persist to ChromaDB, then unload the embedding model before loading the LLM.
- Monitor GPU memory: `mx.metal.get_peak_memory()` and `mx.metal.get_active_memory()`.
- For very large models, apply `sysctl iogpu.wired_limit_mb=<value>` to raise the cap (requires admin rights).
- Keep LLM + KV cache under 80GB total to maintain headroom.

**Warning signs:**
- Generation speed drops from 30 tok/s to 2 tok/s mid-session
- Activity Monitor shows GPU usage dropping to near zero during inference
- `mx.metal.get_active_memory()` approaching 96GB on a 128GB system

**Phase to address:**
Model loading architecture phase — design sequential (not simultaneous) model loading from the start.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Skip embedding persistence, recompute every session | Simpler notebook code | 2–10 min startup penalty per session, inconsistent results if model updates | Never — always persist to ChromaDB with `persist_directory` |
| Use `client = chromadb.Client()` (in-memory) instead of `PersistentClient` | No directory setup | All embeddings lost on kernel restart | Only for throwaway experiments, never in tutorial code |
| Use `enable_thinking=True` with greedy decoding (`temperature=0`) | Deterministic outputs | Infinite repetition loops, model gets stuck | Never with thinking mode — use temp=0.6 minimum |
| Load embedding model and LLM simultaneously in same notebook session | Fewer cells, simpler UX | Both compete for GPU memory, may push past 75% Metal cap | Never on 128GB — sequence them |
| Default `top_k=4` retrieval without testing recall | Less latency | Misses relevant chunks that ranked 5th–10th | Only after measuring recall@10 and confirming 4 is sufficient |
| Skip chunk overlap | Simpler chunking code | Boundary information lost, cross-chunk concepts fail | Never for production; acceptable only for initial prototyping |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| ChromaDB versioning | Accessing data created with Chroma 0.4.x using 0.5.x auto-migrates the SQLite schema — irreversible | Pin ChromaDB version in `requirements.txt`; test version upgrades on a copy of the data |
| ChromaDB collection reuse | Using `get_or_create_collection` then adding documents again re-adds duplicate embeddings silently | Check `collection.count()` before adding; only add if count == 0, or use upsert with document IDs |
| MLX model loading | Calling `mlx_lm.load()` twice in the same kernel without unloading doubles memory usage | Keep model reference as module-level variable; check if already loaded before calling load() |
| BGE-M3 on MLX | BGE-M3 runs via `sentence-transformers`, not via `mlx_lm` — cannot run both on GPU simultaneously without care | Load BGE-M3 for embedding phase, persist results, then use `del embedding_model; mx.metal.clear_cache()` before loading LLM |
| Qwen3.5 thinking mode chat history | Including `<think>...</think>` blocks in subsequent conversation turns causes context bloat | Strip think blocks from chat history; store only the final answer turn |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Naive top-k only retrieval without hybrid search | Korean proper nouns, document codes, and technical terms never retrieved correctly | Add BM25 sparse retrieval alongside vector search (hybrid) | Whenever exact-match terms matter — names, IDs, codes |
| top_k too large (>20 results to LLM) | "Lost in the middle" — relevant answer buried at position 10 is ignored by LLM | Use top_k=5–8 for context, top_k=20 for retrieval then rerank to top 5 | Any corpus with >500 documents |
| top_k too small (<3 results) | Multi-part questions missing context from second required document | Retrieve top_k=10–20, rerank to top 5–8 | Multi-hop questions, documents with distributed information |
| Recomputing embeddings every notebook run | Session startup takes 2–10 minutes | Always use `PersistentClient` and check `collection.count()` before embedding | Any corpus >100 documents |
| Long Qwen3.5 thinking traces in context | KV cache grows with each turn, context fills up | Strip think blocks; use rotating context window | Multi-turn conversations >5 turns |
| No reranking (pure vector search) | Low precision — semantically similar but irrelevant chunks retrieved | Add CrossEncoder reranker after retrieval | Any real corpus; becomes noticeable at >200 documents |

---

## "Looks Done But Isn't" Checklist

- [ ] **Embedding persistence:** Embeddings are computed but not saved — verify `persist_directory` is set AND `collection.count()` survives kernel restart
- [ ] **Korean retrieval:** System tested with English queries only — run 5 Korean-only queries and verify relevant Korean documents surface
- [ ] **KV cache cleanup:** Each RAG query cell works, but no `mx.metal.clear_cache()` call — verify memory does not grow across 20 sequential queries
- [ ] **Chunk boundaries:** Chunks look reasonable at first glance — print 10 random chunks and verify none end mid-sentence (especially Korean)
- [ ] **Multi-turn cache:** Single-turn queries work — run a 3-turn conversation to verify Qwen3.5 does not crash on turn 2
- [ ] **Thinking mode stripping:** Responses contain `<think>` blocks in tutorial output — verify these are stripped before displaying to user and before adding to chat history
- [ ] **Hallucination against retrieved context:** LLM produces answers — verify a sample of answers are actually grounded in retrieved chunks, not parametric memory
- [ ] **PDF table handling:** PDF ingestion "works" — check that tables are not chunked as garbled whitespace-separated text

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Wrong embedding model (English-only for Korean) | HIGH | Re-embed entire corpus with correct multilingual model; ChromaDB collection must be deleted and rebuilt |
| KV cache memory bloat | LOW | Add `mx.metal.clear_cache()` to query function; restart kernel to clear accumulated state |
| ChromaDB schema migration from version upgrade | MEDIUM | Backup data directory before upgrade; if migrated, cannot downgrade — must re-index from source documents |
| Duplicate embeddings from double-add | MEDIUM | Delete collection, re-create, re-embed; add ID-based upsert guard |
| Qwen3.5 KV cache crash on turn 2 | LOW | Replace manual cache init with `make_prompt_cache(model)` |
| Chunk overlap omitted, poor retrieval | HIGH | Re-chunk all documents with overlap; re-embed; rebuild index |
| Greedy decoding with thinking mode | LOW | Add `temperature=0.6` to generation call; no data loss |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Qwen3.5 KV cache crash | Model loading setup | Run 3-turn conversation test before building RAG |
| MLX memory bloat in Jupyter | Core RAG inference wrapper | Query 20 times; verify `get_active_memory()` stays stable |
| Wrong embedding model for Korean | Embedding model selection | Run Korean-only retrieval test before full indexing |
| Chunking destroying Korean context | Document ingestion | Print 20 random chunks; verify no mid-sentence breaks |
| Metal GPU memory cap | Architecture/model loading design | Monitor `get_peak_memory()` during simultaneous model use |
| No embedding persistence | Early notebook structure | Kill and restart kernel; verify results load in <5s |
| Lost in the middle | Retrieval tuning | Test with answer deliberately placed at position 5–8 in context |
| Hallucination despite RAG | Answer generation | Manual spot-check: 10 answers vs. source chunks |
| PDF table destruction | Document ingestion | Compare original PDF table with what was chunked |
| Thinking mode in chat history | Qwen3.5 configuration | Inspect conversation history object for `<think>` tags |

---

## Sources

- [Seven Failure Points When Engineering a Retrieval Augmented Generation System (arXiv 2401.05856)](https://arxiv.org/html/2401.05856v1) — authoritative taxonomy of RAG failure modes
- [MLX LM Memory Usage Issue #1025](https://github.com/ml-explore/mlx-examples/issues/1025) — cache bloat root cause and fix
- [Qwen3.5 MLX KV Cache Bug Issue #37](https://github.com/QwenLM/Qwen3.5/issues/37) — hybrid attention cache initialization crash
- [MLX Unified Memory Documentation](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html) — official memory architecture
- [Benchmarking On-Device Machine Learning on Apple Silicon with MLX (arXiv 2510.18921)](https://arxiv.org/abs/2510.18921) — sustained inference and thermal behavior
- [Ten Failure Modes of RAG Nobody Talks About (DEV Community)](https://dev.to/kuldeep_paul/ten-failure-modes-of-rag-nobody-talks-about-and-how-to-detect-them-systematically-7i4) — embedding drift, citation hallucination, context position bias
- [BAAI/bge-m3 on Hugging Face](https://huggingface.co/BAAI/bge-m3) — multilingual embedding model capabilities and Korean support
- [Optimizing RAG with Hybrid Search & Reranking (VectorHub)](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking) — recall vs. precision failure patterns
- [Best Chunking Strategies for RAG in 2025 (Firecrawl)](https://www.firecrawl.dev/blog/best-chunking-strategies-rag) — chunk size and overlap recommendations
- [Qwen3 Best Settings (jan.ai)](https://www.jan.ai/post/qwen3-settings) — thinking mode temperature and greedy decoding warnings
- [ChromaDB Persistent Client Docs](https://docs.trychroma.com/docs/run-chroma/persistent-client) — persistence and schema migration caveats
- [PDF Hell and Practical RAG Applications (Unstract)](https://unstract.com/blog/pdf-hell-and-practical-rag-applications/) — PDF parsing failure modes
- [Apple Silicon Limitations with Local LLM](https://stencel.io/posts/apple-silicon-limitations-with-usage-on-local-llm%20.html) — Metal memory cap behavior

---
*Pitfalls research for: Local RAG system — MLX + Apple Silicon, bilingual Korean/English, Jupyter notebook*
*Researched: 2026-03-04*
