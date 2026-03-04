# 02. 환경 설정

## 필요한 것들

이 프로젝트를 실행하려면 다음이 필요합니다:

| 필요한 것 | 역할 | 비유 |
|----------|------|------|
| Apple Silicon Mac | 하드웨어 | 도서관 건물 |
| Python 3.11+ | 프로그래밍 언어 | 사서의 언어 |
| MLX + mlx-lm | LLM 실행 엔진 | 학생의 뇌 |
| sentence-transformers | 임베딩 모델 | 색인 카드 작성기 |
| ChromaDB | 벡터 데이터베이스 | 서가 |
| pymupdf4llm | PDF 변환 | 책 스캐너 |
| Jupyter | 노트북 환경 | 실험실 |

## Step 1: Python 환경 만들기

### 가상환경이란?

Python 프로젝트마다 필요한 패키지가 다릅니다. 가상환경은 프로젝트별로 **독립적인 Python 환경**을 만들어줍니다.

비유하면, 실험실마다 다른 도구 세트를 갖추는 것과 같습니다. 한 실험실의 도구를 바꿔도 다른 실험실에는 영향이 없습니다.

```bash
# 터미널에서 프로젝트 폴더로 이동
cd ~/vibe-coding/RAG

# Python 버전 확인 (3.11 이상이어야 합니다)
python3 --version
# Python 3.11.x 또는 3.12.x

# 가상환경 생성 (.venv라는 폴더가 만들어집니다)
python3 -m venv .venv

# 가상환경 활성화
source .venv/bin/activate

# 활성화되면 프롬프트 앞에 (.venv)가 표시됩니다:
# (.venv) ~/vibe-coding/RAG $
```

> **중요**: 터미널을 새로 열 때마다 `source .venv/bin/activate`를 실행해야 합니다. 그래야 이 프로젝트의 Python 환경을 사용합니다.

### pip 업그레이드

```bash
pip install --upgrade pip
```

## Step 2: 패키지 설치

### 하나씩 설치하며 이해하기

각 패키지가 무엇을 하는지 알고 설치합시다:

```bash
# 1. MLX — Apple Silicon을 위한 ML 프레임워크
#    Apple이 만들었습니다. PyTorch/TensorFlow 대신 사용합니다.
pip install mlx

# 2. mlx-lm — MLX 위에서 LLM을 쉽게 쓸 수 있게 해주는 도구
#    모델 다운로드, 로딩, 텍스트 생성을 간단하게 해줍니다.
pip install mlx-lm

# 3. sentence-transformers — 텍스트를 벡터(임베딩)로 변환
#    우리는 BAAI/bge-m3 모델을 여기서 실행합니다.
pip install sentence-transformers

# 4. ChromaDB — 벡터를 저장하고 검색하는 데이터베이스
#    설치만 하면 바로 쓸 수 있습니다. 별도 서버가 필요 없습니다.
pip install chromadb

# 5. pymupdf4llm — PDF 파일을 텍스트로 변환
#    PDF의 복잡한 레이아웃을 깔끔하게 텍스트로 바꿔줍니다.
pip install pymupdf4llm

# 6. langchain-text-splitters — 텍스트를 조각(chunk)으로 나누는 도구
#    주의: LangChain 전체가 아니라, 텍스트 분할 기능만 사용합니다.
pip install langchain-text-splitters

# 7. Jupyter — 코드를 한 줄씩 실행하며 결과를 볼 수 있는 노트북 환경
pip install jupyter ipykernel
```

### 한 줄로 한꺼번에 설치

위 패키지를 한 번에 설치할 수도 있습니다:

```bash
pip install mlx mlx-lm sentence-transformers chromadb pymupdf4llm langchain-text-splitters jupyter ipykernel
```

### requirements.txt 만들기

나중에 환경을 다시 만들 때 편하도록 패키지 목록을 파일로 저장합니다:

```
# requirements.txt
mlx>=0.21.0
mlx-lm>=0.20.0
sentence-transformers>=3.0.0
chromadb>=1.0.0
pymupdf4llm>=0.3.0
langchain-text-splitters>=0.3.0
jupyter
ipykernel
```

```bash
# 나중에 이 파일로 한 번에 설치할 수 있습니다:
pip install -r requirements.txt
```

## Step 3: 모델 다운로드

우리 프로젝트에서 사용하는 AI 모델은 두 개입니다:

### LLM: Qwen2.5-7B-Instruct (MLX 4-bit)

질문에 답변하는 모델입니다. 약 **4.5GB**입니다.

```python
# Python에서 실행하면 자동으로 다운로드됩니다
python3 -c "
from mlx_lm import load
print('Downloading Qwen model...')
model, tokenizer = load('mlx-community/Qwen2.5-7B-Instruct-4bit')
print('Done! Model loaded successfully.')
"
```

> **"4-bit"란?** 모델의 숫자 정밀도를 줄여서 크기를 작게 만든 것입니다. 원래 모델은 ~14GB이지만, 4-bit로 압축하면 ~4.5GB가 됩니다. 품질은 약간 떨어지지만 학습용으로 충분합니다.

> **"Instruct"란?** 사용자의 지시를 따르도록 추가 학습된 모델입니다. 일반 모델은 텍스트를 이어서 쓰기만 하지만, Instruct 모델은 질문에 답변하는 형태로 동작합니다.

### 임베딩 모델: BAAI/bge-m3

텍스트를 벡터로 변환하는 모델입니다. 약 **2.3GB**입니다.

```python
python3 -c "
from sentence_transformers import SentenceTransformer
print('Downloading embedding model...')
model = SentenceTransformer('BAAI/bge-m3')
print('Done! Embedding model loaded successfully.')
"
```

> **왜 bge-m3인가?** 이 모델은 100개 이상의 언어를 지원합니다. 한국어 문서와 영어 문서를 모두 다룰 수 있습니다. 영어만 지원하는 모델(예: `all-MiniLM-L6-v2`)을 쓰면 **한국어 검색이 조용히 실패**합니다 — 에러 없이 엉뚱한 결과가 나옵니다.

## Step 4: 설치 확인

모든 것이 제대로 설치되었는지 확인하는 스크립트입니다:

```python
# check_install.py — 이 코드를 실행해보세요

print("=" * 50)
print("RAG Tutorial — Installation Check")
print("=" * 50)

# 1. MLX
import mlx.core as mx
print(f"\n✓ MLX version: {mx.__version__}")
print(f"  Metal (GPU) available: {mx.metal.is_available()}")
print(f"  Memory limit: {mx.metal.memory_limit() / 1e9:.1f} GB")

# 2. ChromaDB
import chromadb
print(f"\n✓ ChromaDB version: {chromadb.__version__}")

# 3. sentence-transformers
import sentence_transformers
print(f"\n✓ sentence-transformers version: {sentence_transformers.__version__}")

# 4. mlx-lm
import mlx_lm
print(f"\n✓ mlx-lm OK")

# 5. pymupdf4llm
import pymupdf4llm
print(f"\n✓ pymupdf4llm OK")

# 6. langchain-text-splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter
print(f"\n✓ langchain-text-splitters OK")

print("\n" + "=" * 50)
print("All packages installed successfully!")
print("=" * 50)
```

### 기대하는 출력

```
==================================================
RAG Tutorial — Installation Check
==================================================

✓ MLX version: 0.31.0
  Metal (GPU) available: True
  Memory limit: 96.0 GB

✓ ChromaDB version: 1.5.2

✓ sentence-transformers version: 5.2.3

✓ mlx-lm OK

✓ pymupdf4llm OK

✓ langchain-text-splitters OK

==================================================
All packages installed successfully!
==================================================
```

> **Metal memory limit가 96GB?** 128GB Mac인데 96GB만 표시되는 것은 정상입니다. macOS가 GPU(Metal)에 전체 메모리의 약 75%만 할당합니다. 나머지 25%는 OS와 다른 프로세스가 사용합니다.

## Step 5: Jupyter 커널 등록

Jupyter 노트북에서 우리 가상환경을 사용할 수 있도록 등록합니다:

```bash
python -m ipykernel install --user --name=rag-tutorial --display-name="RAG Tutorial"
```

이제 Jupyter에서 노트북을 만들 때 "RAG Tutorial" 커널을 선택할 수 있습니다.

### Jupyter 실행 테스트

```bash
jupyter notebook
```

브라우저가 열리면 새 노트북을 만들고, 커널을 "RAG Tutorial"로 선택한 뒤, 다음 코드를 실행해보세요:

```python
import mlx.core as mx
print(f"Hello from MLX! GPU available: {mx.metal.is_available()}")
```

## 문제 해결

### "pip: command not found"

가상환경이 활성화되지 않았습니다:
```bash
source .venv/bin/activate
```

### "Metal is not available"

Apple Silicon이 아닌 Intel Mac에서는 MLX의 GPU 가속을 사용할 수 없습니다. `M1` 이상의 칩이 필요합니다.

### 모델 다운로드가 느림

모델은 Hugging Face에서 다운로드됩니다. 첫 번째 다운로드 시 인터넷 연결이 필요합니다. 이후에는 로컬 캐시에서 로딩되므로 인터넷이 필요 없습니다.

캐시 위치: `~/.cache/huggingface/hub/`

---

이전: [01. RAG의 전체 그림](01-rag-big-picture.md) | 다음: [03. LLM 만나기](03-meet-your-llm.md)
