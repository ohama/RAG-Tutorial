# Local RAG with Qwen3.5 MLX

## Documentation

[RAG Tutorial](https://ohama.github.io/RAG-Tutorial/)

Mac Studio M4 Max (128GB)에서 Qwen3.5 MLX 모델을 사용한 로컬 RAG(Retrieval-Augmented Generation) 학습 프로젝트.

RAG가 LLM 답변 품질에 어떤 차이를 만드는지 직접 체험할 수 있습니다.

## What You'll Learn

- RAG 파이프라인의 각 단계: 문서 로딩 → 청킹 → 임베딩 → 벡터 저장 → 검색 → 생성
- RAG 유무에 따른 답변 품질 차이 (Jupyter 노트북에서 나란히 비교)
- Apple Silicon에서 MLX를 활용한 완전 로컬 LLM 실행

## Stack

| Component | Choice | Why |
|-----------|--------|-----|
| LLM | Qwen2.5-7B-Instruct-4bit (MLX) | 128GB에서 로컬 실행, Apple Silicon 최적화 |
| Embedding | BAAI/bge-m3 | 한국어 + 영어 지원 다국어 모델 |
| Vector DB | ChromaDB | 서버 불필요, 로컬 영구 저장, Python 네이티브 |
| Documents | Markdown, PDF, TXT | 로컬 파일 파싱 |
| Framework | None (plain Python) | 교육 목적 — 모든 단계가 보여야 함 |

## Project Structure

```
src/                  # RAG implementation
  ingestion/          # Document loading, chunking
  embedding/          # Vector embedding
  store/              # ChromaDB vector store
  retrieval/          # Similarity search
  generation/         # LLM generation with MLX
  rag.py              # RAG pipeline facade
tutorial/             # Step-by-step Korean tutorial
notebooks/            # Jupyter notebooks (RAG on/off comparison)
docs/                 # Sample documents for RAG
```

## Requirements

- Mac with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- ~10GB disk space (models + vector DB)

## Quick Start

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download models (first run only)
python -m src.setup

# Run the RAG comparison notebook
jupyter notebook notebooks/rag_comparison.ipynb
```

## Key Design Decisions

- **No LangChain / LlamaIndex** — every pipeline step is explicit, not hidden behind abstractions
- **No cloud APIs** — everything runs locally, no API keys needed
- **Korean explanations, English code** — 설명은 한국어, 코드/주석은 영어
- **Clarity over performance** — educational code that's easy to read and understand
