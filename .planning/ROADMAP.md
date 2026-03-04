# Roadmap: Local RAG with Qwen3.5 MLX

## Overview

Four phases build a fully local Korean-language RAG tutorial on Apple Silicon. The project starts by validating the environment and models (the three hardest-to-retrofit decisions), proceeds through document ingestion and the core RAG pipeline, and culminates in the primary educational deliverable: a side-by-side RAG on/off comparison notebook with step-by-step Korean tutorials.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Environment & Model Validation** - Working MLX environment with validated Qwen + embedding models
- [ ] **Phase 2: Document Ingestion Pipeline** - Load, chunk, embed, and persist local documents to ChromaDB
- [ ] **Phase 3: Core RAG Pipeline & Comparison** - Retrieval, generation, and the RAG on/off comparison notebook
- [ ] **Phase 4: Tutorial Documentation** - Step-by-step Korean tutorial series in tutorial/

## Phase Details

### Phase 1: Environment & Model Validation
**Goal**: Learner can load both models, run inference, and confirm Korean retrieval works before writing any application code
**Depends on**: Nothing (first phase)
**Requirements**: ENV-01, ENV-02, ENV-03
**Success Criteria** (what must be TRUE):
  1. Qwen2.5-7B-Instruct-4bit loads and generates text in a Jupyter cell with no crash on repeated calls
  2. BAAI/bge-m3 generates embeddings for Korean text and returns cosine similarity scores
  3. Sequential model loading (embed → clear cache → generate) works without hitting Metal memory cap
  4. A Markdown notebook cell explains MLX unified memory and Apple Silicon GPU acceleration in Korean
**Plans**: TBD

Plans:
- [ ] 01-01: Environment setup and stack installation (Python venv, all packages, model downloads)
- [ ] 01-02: Model validation notebook (Qwen KV cache, bge-m3 Korean, Metal memory lifecycle)

### Phase 2: Document Ingestion Pipeline
**Goal**: Learner can ingest Markdown, PDF, and TXT files into a ChromaDB collection with source metadata preserved for citation
**Depends on**: Phase 1
**Requirements**: DOC-01, DOC-02, DOC-03, DOC-04
**Success Criteria** (what must be TRUE):
  1. Running the ingestion script indexes all three document formats (Markdown, PDF, TXT) without error
  2. Each chunk in ChromaDB carries the original filename and position metadata
  3. Chunk boundaries respect Korean sentence structure (no mid-sentence splits visible on inspection)
  4. The 3-5 sample documents are included and index successfully out of the box
  5. chunk_size and chunk_overlap parameters are exposed and documented in the notebook cell
**Plans**: TBD

Plans:
- [ ] 02-01: Document loader and chunker modules (src/ingestion/, KSS Korean splitting)
- [ ] 02-02: Embedder and ChromaDB store modules (src/embedding/, src/store/, persistent collection)
- [ ] 02-03: Sample documents and ingestion notebook

### Phase 3: Core RAG Pipeline & Comparison
**Goal**: Learner can run the same question with RAG off and RAG on side-by-side and observe the quality difference, with source citations visible
**Depends on**: Phase 2
**Requirements**: RET-01, RET-02, RET-03, RET-04, TUT-01, TUT-02
**Success Criteria** (what must be TRUE):
  1. The comparison notebook runs end-to-end with no errors (RAG off answer and RAG on answer produced from one cell)
  2. The injected context block is printed in the notebook so the learner sees exactly what goes into the prompt
  3. Each RAG answer shows the source filename and chunk number it drew from
  4. All Markdown explanation cells are in Korean; all code and inline comments are in English
  5. mx.metal.clear_cache() is called after every inference call and memory stays stable across 10+ queries
**Plans**: TBD

Plans:
- [ ] 03-01: Retriever and prompt builder modules (src/retrieval/, similarity search, prompt template)
- [ ] 03-02: Generator module and RAG facade (src/generation/, src/rag.py, Metal cache management)
- [ ] 03-03: RAG on/off comparison notebook (the primary educational deliverable)

### Phase 4: Tutorial Documentation
**Goal**: Learner can follow a numbered Korean tutorial series that walks through every component of the pipeline from first principles
**Depends on**: Phase 3
**Requirements**: TUT-03
**Success Criteria** (what must be TRUE):
  1. The tutorial/ directory contains numbered documents covering each pipeline stage in order
  2. A learner with no prior RAG experience can follow the tutorial from start to finish using only the tutorial files
  3. Each tutorial document links to the corresponding source module or notebook cell it explains
**Plans**: TBD

Plans:
- [ ] 04-01: Korean tutorial series (tutorial/ directory, numbered step-by-step guides)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Environment & Model Validation | 0/2 | Not started | - |
| 2. Document Ingestion Pipeline | 0/3 | Not started | - |
| 3. Core RAG Pipeline & Comparison | 0/3 | Not started | - |
| 4. Tutorial Documentation | 0/1 | Not started | - |

---
*Roadmap created: 2026-03-04*
*Last updated: 2026-03-04 after initial creation*
