# 06. 의미를 숫자로 (Embedding)

## 이 단계의 위치

```
                               ★ 여기!
 ① 로딩        ② 청킹          ③ 임베딩         ④ 저장
┌──────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 문서  │ →  │ 텍스트    │ →  │ 벡터     │ →  │ 벡터 DB  │
│ 파일  │    │ 조각들    │    │ 변환     │    │ 저장     │
└──────┘    └──────────┘    └──────────┘    └──────────┘
```

임베딩은 RAG에서 **가장 핵심적인 개념**입니다. 이 단계를 이해하면 RAG의 "검색이 어떻게 작동하는지"가 완전히 이해됩니다.

## 컴퓨터는 텍스트를 모른다

인간은 텍스트를 읽으면 의미를 바로 이해합니다:
- "고양이가 소파에 앉아 있다" — 아, 고양이가 소파 위에 있구나
- "강아지가 침대에 누워 있다" — 비슷한 상황이네

하지만 컴퓨터에게 텍스트는 그냥 **글자의 나열**입니다. "고양이"와 "강아지"가 비슷한 개념이라는 것을 모릅니다.

그래서 텍스트의 **의미**를 컴퓨터가 이해할 수 있는 **숫자**로 변환해야 합니다.

## 벡터란? (지도 좌표 비유)

벡터(vector)를 가장 쉽게 이해하는 방법은 **지도의 좌표**를 떠올리는 것입니다.

### 2차원 좌표

서울의 식당들을 좌표로 표시한다고 생각해보세요:

```
맛 (y축)
  │
  │        ★ 고급 한식당
  │
  │   ★ 동네 한식당
  │
  │                    ★ 고급 이탈리안
  │
  │              ★ 동네 피자집
  │
  └──────────────────────── 가격 (x축)
     저렴                    비쌈
```

이 좌표에서:
- **가까운 점** = 비슷한 식당 (동네 한식당 ↔ 동네 피자집)
- **먼 점** = 다른 식당 (동네 피자집 ↔ 고급 한식당)

### 텍스트도 마찬가지

텍스트의 "의미"를 좌표(벡터)로 표현합니다. 단, 2차원이 아니라 **1024차원**입니다.

```
2차원 좌표: (3.5, 7.2)                  ← 2개의 숫자
텍스트 벡터: (0.12, -0.34, 0.56, ..., 0.78)  ← 1024개의 숫자
```

1024개의 숫자가 텍스트의 다양한 의미적 측면을 표현합니다. 사람이 각 차원이 무엇을 뜻하는지 해석하기는 어렵지만, 핵심 원리는 같습니다:

**의미가 비슷한 텍스트 → 비슷한 벡터 → 벡터 공간에서 가까운 위치**

```
"고양이가 소파에 앉아 있다"  → [0.82, -0.15, 0.43, ..., 0.67]
"강아지가 침대에 누워 있다"  → [0.79, -0.18, 0.41, ..., 0.64]  ← 가까움!
"주식 시장이 폭락했다"      → [-0.23, 0.91, -0.56, ..., 0.12] ← 멀다!
```

## 임베딩 모델이란?

텍스트를 벡터로 변환해주는 AI 모델입니다. 수억 개의 텍스트를 학습하여 "비슷한 의미의 텍스트는 비슷한 벡터를 만들도록" 훈련되었습니다.

```
              임베딩 모델
"MLX는 빠르다" ──────→ [0.12, -0.34, 0.56, ..., 0.78]
```

### 우리가 사용하는 모델: BAAI/bge-m3

| 특성 | 값 |
|------|------|
| 이름 | BAAI/bge-m3 |
| 벡터 차원 | 1024 |
| 지원 언어 | 100개 이상 (한국어 + 영어 포함) |
| 모델 크기 | ~2.3GB |
| 용도 | 텍스트 유사도 검색 |

### 왜 이 모델인가?

**가장 중요한 이유: 한국어를 지원합니다.**

많이 알려진 `all-MiniLM-L6-v2`는 영어에 최적화되어 있습니다. 이 모델로 한국어를 임베딩하면:

```
영어 전용 모델 (all-MiniLM-L6-v2):
  "고양이가 앉아 있다"  → [0.12, 0.56, ...]
  "강아지가 누워 있다"  → [0.78, -0.34, ...]  ← 전혀 다른 벡터!
  → 한국어 의미를 이해하지 못함
  → 검색 결과가 엉뚱함
  → 에러가 나지 않기 때문에 원인을 찾기 어려움!

다국어 모델 (bge-m3):
  "고양이가 앉아 있다"  → [0.82, -0.15, ...]
  "강아지가 누워 있다"  → [0.79, -0.18, ...]  ← 비슷한 벡터!
  → 한국어 의미를 정확히 이해
  → 검색 결과가 정확
```

> **경고**: 영어 전용 임베딩 모델을 한국어에 쓰면, **에러 없이 조용히 실패**합니다. 벡터는 생성되지만 의미가 없는 벡터입니다. 검색 결과가 엉뚱하게 나와도 왜 그런지 알기 어렵습니다. 한국어 문서를 다룬다면 **반드시** 다국어 모델을 사용하세요.

## 코드로 임베딩하기

### 기본 사용법

```python
from sentence_transformers import SentenceTransformer

# 모델 로딩 (첫 실행 시 ~2.3GB 다운로드)
embed_model = SentenceTransformer("BAAI/bge-m3")

# 텍스트 하나를 벡터로 변환
text = "MLX는 Apple Silicon에 최적화된 프레임워크입니다"
vector = embed_model.encode(text)

print(f"Input text: {text}")
print(f"Vector shape: {vector.shape}")      # (1024,) → 1024개의 숫자
print(f"First 5 values: {vector[:5]}")       # [0.023, -0.118, ...]
print(f"Vector type: {type(vector)}")        # numpy.ndarray
```

### 여러 텍스트를 한 번에

```python
texts = [
    "Apple Silicon의 통합 메모리",
    "M4 칩의 성능",
    "오늘 점심 메뉴",
]

# 배치로 한 번에 임베딩 (하나씩보다 훨씬 빠름)
vectors = embed_model.encode(texts)

print(f"Input: {len(texts)} texts")
print(f"Output: {vectors.shape}")  # (3, 1024) → 3개 텍스트 × 1024차원
```

## 코사인 유사도: 벡터끼리 비교하기

두 벡터가 얼마나 비슷한지 측정하는 방법이 **코사인 유사도**입니다.

### 직관적 이해

두 화살표의 **방향**이 얼마나 같은지를 봅니다.

```
같은 방향 (유사도 ≈ 1.0):      반대 방향 (유사도 ≈ -1.0):
    ↗ ↗                             ↗ ↙

수직 (유사도 ≈ 0.0):
    ↗
    ↑
```

| 코사인 유사도 | 의미 |
|-------------|------|
| 0.9 ~ 1.0 | 거의 같은 의미 |
| 0.7 ~ 0.9 | 관련 있는 내용 |
| 0.3 ~ 0.7 | 약간의 관련성 |
| 0.0 ~ 0.3 | 거의 관련 없음 |

### 코드로 확인

```python
import numpy as np

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors.

    Returns a value between -1 (opposite) and 1 (identical).
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

```python
# 세 문장의 유사도 비교
texts = [
    "MLX는 Apple Silicon에 최적화된 프레임워크입니다",     # [0] 기준
    "Apple의 ML 프레임워크는 M4 칩에서 빠르게 동작합니다",  # [1] 비슷한 의미
    "오늘 서울의 날씨는 맑고 기온은 15도입니다",            # [2] 관련 없음
]

vectors = embed_model.encode(texts)

# [0] vs [1]: 비슷한 내용
sim_01 = cosine_similarity(vectors[0], vectors[1])
print(f"MLX 설명 vs Apple ML: {sim_01:.4f}")    # ~0.85 (높음!)

# [0] vs [2]: 관련 없는 내용
sim_02 = cosine_similarity(vectors[0], vectors[2])
print(f"MLX 설명 vs 날씨:     {sim_02:.4f}")    # ~0.15 (낮음!)
```

**이것이 RAG 검색의 원리입니다:**
1. 질문을 벡터로 변환합니다
2. 저장된 모든 청크 벡터와 코사인 유사도를 계산합니다
3. 유사도가 가장 높은 청크를 반환합니다

## 다국어 임베딩의 신기한 점

bge-m3는 **다른 언어의 같은 의미**도 비슷한 벡터로 변환합니다:

```python
texts = [
    "MLX는 Apple Silicon에 최적화되어 있습니다",                # 한국어
    "MLX is optimized for Apple Silicon",                     # 영어 (같은 의미)
    "오늘 비가 올 예정입니다",                                  # 한국어 (다른 의미)
]

vectors = embed_model.encode(texts)

sim_same_meaning = cosine_similarity(vectors[0], vectors[1])  # 한↔영 같은 의미
sim_diff_meaning = cosine_similarity(vectors[0], vectors[2])  # 한↔한 다른 의미

print(f"한국어 vs 영어 (같은 의미): {sim_same_meaning:.4f}")  # ~0.90 (매우 높음!)
print(f"한국어 vs 한국어 (다른 의미): {sim_diff_meaning:.4f}")  # ~0.15 (낮음)
```

한국어로 질문해도 영어 문서에서 관련 내용을 찾을 수 있고, 그 반대도 가능합니다!

## 청크 임베딩하기

이전 단계에서 만든 청크들을 벡터로 변환합니다:

```python
def embed_chunks(
    chunks: list[dict],
    model: SentenceTransformer,
) -> list[dict]:
    """Add embedding vectors to chunks.

    Takes chunks from chunk_document() and adds an 'embedding' field
    to each chunk dictionary.
    """
    # Extract texts for batch encoding
    texts = [chunk["text"] for chunk in chunks]

    # Batch encoding is much faster than one-by-one
    vectors = model.encode(texts, show_progress_bar=True)

    # Attach vectors to chunks
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector

    print(f"  Embedded {len(chunks)} chunks (dim={vectors.shape[1]})")
    return chunks
```

```python
# 사용 예
embed_model = SentenceTransformer("BAAI/bge-m3")
chunks = load_and_chunk_directory("docs/")
chunks = embed_chunks(chunks, embed_model)

# 확인
print(f"Chunk 0 text: {chunks[0]['text'][:80]}...")
print(f"Chunk 0 vector: {chunks[0]['embedding'][:5]}...")  # 첫 5개 숫자
print(f"Chunk 0 vector shape: {chunks[0]['embedding'].shape}")  # (1024,)
```

## 메모리 관리

임베딩 모델은 약 2.3GB의 메모리를 사용합니다. Qwen 모델(~4.5GB)과 동시에 로딩하면 ~7GB가 됩니다.

128GB Mac에서는 문제없지만, 메모리가 적은 시스템에서는 순차적으로 사용해야 합니다:

```python
import gc
import mlx.core as mx

# 1. 임베딩 수행
embed_model = SentenceTransformer("BAAI/bge-m3")
chunks = embed_chunks(chunks, embed_model)

# 2. 임베딩 모델 해제 (메모리 절약)
del embed_model
gc.collect()
mx.metal.clear_cache()

# 3. 이제 LLM 로딩 가능
from mlx_lm import load
model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
```

## 실습: 임베딩의 힘 느껴보기

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("BAAI/bge-m3")

# 질문과 여러 청크의 유사도를 비교합니다
query = "Apple Silicon의 통합 메모리란?"

chunk_texts = [
    "Apple Silicon은 CPU와 GPU가 같은 메모리를 공유합니다. 이를 통합 메모리라고 합니다.",
    "MLX는 lazy evaluation을 사용하여 계산을 미루고 메모리를 절약합니다.",
    "Python 3.11은 CPython의 속도를 10-60% 향상시켰습니다.",
    "ChromaDB는 로컬에서 실행되는 벡터 데이터베이스입니다.",
]

# 질문과 청크를 모두 임베딩
all_texts = [query] + chunk_texts
vectors = model.encode(all_texts)
query_vec = vectors[0]

# 각 청크와 질문의 유사도 계산
print(f"Query: {query}\n")
for i, text in enumerate(chunk_texts):
    sim = np.dot(query_vec, vectors[i+1]) / (
        np.linalg.norm(query_vec) * np.linalg.norm(vectors[i+1])
    )
    bar = "█" * int(sim * 30)
    print(f"  [{sim:.3f}] {bar}")
    print(f"  {text[:60]}...")
    print()
```

첫 번째 청크(통합 메모리 설명)의 유사도가 가장 높게 나올 것입니다. 이것이 RAG 검색이 작동하는 원리입니다!

## 핵심 정리

1. **임베딩** = 텍스트의 의미를 숫자 벡터로 변환하는 것
2. 의미가 비슷한 텍스트 → 비슷한 벡터 → 벡터 공간에서 **가까운 위치**
3. **코사인 유사도**로 두 벡터의 비슷함을 측정 (1에 가까울수록 비슷)
4. 한국어 문서에는 반드시 **다국어 모델** (bge-m3) 사용
5. 영어 전용 모델을 한국어에 쓰면 **에러 없이 조용히 실패** — 가장 위험한 함정
6. 배치(batch)로 임베딩하면 하나씩보다 훨씬 빠르다

---

이전: [05. 텍스트 자르기](05-chunking.md) | 다음: [07. 벡터 저장소](07-vector-store.md)
