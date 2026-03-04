# 05. 텍스트 자르기 (Chunking)

## 이 단계의 위치

```
                ★ 여기!
 ① 로딩        ② 청킹          ③ 임베딩         ④ 저장
┌──────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 문서  │ →  │ 텍스트    │ →  │ 벡터     │ →  │ 벡터 DB  │
│ 파일  │    │ 조각들    │    │ 변환     │    │ 저장     │
└──────┘    └──────────┘    └──────────┘    └──────────┘
```

## 왜 자르는가?

도서관 비유로 돌아갑시다.

학생이 "통합 메모리란?" 이라고 물었을 때, 사서가 어떻게 해야 할까요?

**방법 A**: 책 한 권을 통째로 건네준다
- 학생: "이 300페이지 중에서 어디를 봐야 하나요...?"
- 시간 낭비, 관련 없는 내용이 대부분

**방법 B**: 관련된 3페이지만 펼쳐준다
- 학생: "아, 여기에 딱 나와있네요!"
- 정확하고 빠름

RAG도 마찬가지입니다. 문서 전체를 LLM에 넣는 것은 비효율적입니다:

1. **LLM 한계** — LLM은 한 번에 처리할 수 있는 텍스트 양이 제한됩니다 (컨텍스트 윈도우)
2. **검색 정밀도** — 작은 조각이 큰 문서보다 더 정확하게 "관련 있음/없음"을 판별할 수 있습니다
3. **비용** — 보내는 텍스트가 많을수록 처리 시간과 메모리가 더 듭니다

그래서 문서를 **적당한 크기의 조각(chunk)**으로 나눕니다.

## 종이를 자르는 것처럼

10장짜리 문서를 500자씩 자른다고 생각해보세요.

```
전체 문서 (5000자):

"RAG는 검색 증강 생성입니다. ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
LLM이 답변을 생성할 때,                                   │
외부 문서에서 관련 정보를                                  │ 조각 1
먼저 검색한 뒤 그 정보를                                  │ (500자)
바탕으로 답변합니다..." ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘

"임베딩은 텍스트를 벡터로 ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
변환하는 과정입니다.                                      │
의미가 비슷한 텍스트는                                    │ 조각 2
비슷한 벡터를 가집니다.                                   │ (500자)
이 성질 덕분에..." ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘

...
```

## 두 가지 핵심 파라미터

### chunk_size: 조각의 크기

한 조각의 최대 글자 수입니다.

```
chunk_size = 100:  짧은 조각 → 많은 조각 개수
chunk_size = 500:  적당한 조각 ★ 권장
chunk_size = 2000: 긴 조각 → 적은 조각 개수
```

### chunk_overlap: 겹침 영역

인접한 두 조각이 일부 내용을 공유하도록 합니다.

**왜 겹쳐야 하는가?** 문장이 두 조각의 경계에 걸쳐서 잘리면, 의미가 손실됩니다.

```
겹침 없이 자르면 (overlap = 0):

조각 1: "...Apple Silicon의 통합 메모"
조각 2: "리는 CPU와 GPU가 공유합니다..."

→ "통합 메모리"가 두 조각으로 나뉘었습니다!
  조각 1에서 검색하면 불완전한 정보
  조각 2에서 검색하면 맥락이 없음
```

```
겹침 있게 자르면 (overlap = 50):

조각 1: "...Apple Silicon의 통합 메모리는 CPU와 GPU가"
                                    ↕ 겹침 영역 (50자)
조각 2: "통합 메모리는 CPU와 GPU가 공유합니다. 이는..."

→ "통합 메모리"가 양쪽 조각에 모두 포함됩니다!
  어느 조각을 검색해도 완전한 정보를 얻을 수 있습니다
```

## 코드로 구현하기

`langchain-text-splitters` 패키지를 사용합니다. LangChain 프레임워크 전체가 아니라, 텍스트 분할 기능만 독립적으로 사용합니다.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """Split text into overlapping chunks.

    Uses RecursiveCharacterTextSplitter which tries to split
    at natural boundaries (paragraphs > lines > sentences > words).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_text(text)
    return chunks
```

### RecursiveCharacterTextSplitter가 하는 일

이 splitter는 텍스트를 **가능한 한 자연스러운 경계**에서 자릅니다.

`separators` 리스트의 순서대로 시도합니다:

```
1차 시도: "\n\n" (빈 줄) → 문단 단위로 나누기
   조각이 chunk_size 이하면 OK!
   아직 크면 ↓

2차 시도: "\n" (줄바꿈) → 줄 단위로 나누기
   조각이 chunk_size 이하면 OK!
   아직 크면 ↓

3차 시도: ". " (마침표+공백) → 문장 단위로 나누기
   조각이 chunk_size 이하면 OK!
   아직 크면 ↓

4차 시도: " " (공백) → 단어 단위로 나누기
   최후의 수단
```

예시:

```python
text = """첫 번째 문단입니다. 이 문단은 짧습니다.

두 번째 문단입니다. 이 문단은 조금 더 길어서 여러 문장을 포함합니다.
중요한 내용이 여기에 있습니다. 그리고 더 많은 설명이 이어집니다.

세 번째 문단입니다."""

chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
for i, chunk in enumerate(chunks):
    print(f"[Chunk {i}] ({len(chunk)} chars): {chunk}")
```

## 메타데이터 보존

조각을 만들 때 **이 조각이 어디서 왔는지** 정보를 함께 저장해야 합니다.

```python
def chunk_document(
    doc: dict,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict]:
    """Chunk a document and preserve source metadata.

    Each chunk carries:
    - The original document's metadata (source filename, format)
    - Its own position info (chunk_index, chunk_total)
    """
    text_chunks = chunk_text(doc["text"], chunk_size, chunk_overlap)

    chunks = []
    for i, text in enumerate(text_chunks):
        chunks.append({
            "text": text,
            "metadata": {
                **doc["metadata"],            # source, format, title...
                "chunk_index": i,             # 이 조각의 순서
                "chunk_total": len(text_chunks),  # 전체 조각 수
            }
        })

    return chunks
```

나중에 RAG 답변에서 이렇게 사용됩니다:

```
답변: "Apple Silicon의 통합 메모리는 CPU와 GPU가 같은 메모리를 공유합니다."
출처: mlx-guide.md, chunk 3/8
```

## chunk_size 선택 가이드

어떤 크기가 좋을까요? 상황마다 다르지만, 직관적으로 이해할 수 있습니다:

### 너무 작으면 (chunk_size = 100)

```
조각: "통합 메모리는 CPU와 GPU가 같은 메모리를 공유합니다."

→ 이것만 보면: 무엇의 통합 메모리? 어떤 맥락?
→ 검색은 잘 되지만, LLM에게 주기엔 맥락이 부족
```

### 너무 크면 (chunk_size = 2000)

```
조각: "MLX 소개... [500자] ...통합 메모리 설명... [500자]
       ...설치 방법... [500자] ...코드 예제... [500자]"

→ 모든 내용이 다 들어있지만, "통합 메모리"를 물어봤는데
   설치 방법, 코드 예제까지 딸려옴
→ 검색 정밀도가 떨어짐
```

### 적당하면 (chunk_size = 500)

```
조각: "MLX의 핵심 특징은 통합 메모리입니다.
       Apple Silicon은 CPU와 GPU가 같은 메모리를 공유합니다.
       일반 시스템에서는 데이터를 복사해야 하지만,
       Apple Silicon에서는 이 복사가 필요 없어 대규모 모델
       실행에 유리합니다."

→ 충분한 맥락 + 집중된 내용
→ 검색도 잘 되고, LLM이 이해하기도 좋음
```

| chunk_size | 장점 | 단점 | 적합한 경우 |
|-----------|------|------|-----------|
| 100-200 | 정밀한 검색 | 맥락 부족 | 짧은 FAQ |
| 400-600 | 균형 | - | **대부분의 경우 (권장)** |
| 1000-2000 | 풍부한 맥락 | 검색 정밀도 낮음 | 긴 논문 |

> **시작은 500으로.** 결과를 보면서 조정하세요.

## 한국어 청킹 주의사항

한국어는 영어와 특성이 다릅니다:

### 1. 토큰 수의 차이

같은 의미를 표현해도 한국어가 더 많은 토큰을 소비합니다:

```
영어: "The unified memory architecture"     →  약 5 tokens
한국어: "통합 메모리 아키텍처"                  →  약 10 tokens
```

이 때문에 한국어 문서는 같은 chunk_size에서 실제로 담기는 **의미의 양**이 적습니다.

> **팁**: 한국어 문서가 많다면 `chunk_size`를 600-800으로 약간 늘려보세요.

### 2. 문장 경계

영어는 마침표(`.`)로 문장이 끝나지만, 한국어는 `다.`, `요.`, `죠.`, `까?` 등 다양하게 끝납니다. `RecursiveCharacterTextSplitter`의 `. ` 기준만으로는 한국어 문장 경계를 완벽히 잡지 못할 수 있습니다.

## 전체 파이프라인: 로딩 → 청킹

지금까지 배운 것을 연결합니다:

```python
def load_and_chunk_directory(
    dir_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict]:
    """Load all documents and split into chunks.

    This is the complete ingestion pipeline up to chunking:
    Files → Text → Chunks (with metadata)
    """
    # Step 1: Load all documents
    docs = load_directory(dir_path)

    # Step 2: Chunk each document
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
        print(f"  {doc['metadata']['source']}: {len(chunks)} chunks")

    print(f"\n  Total: {len(all_chunks)} chunks from {len(docs)} documents")
    return all_chunks
```

```python
# 실행 예
chunks = load_and_chunk_directory("docs/", chunk_size=500, chunk_overlap=50)

#   Loaded: mlx-guide.md (2,340 chars)
#   Loaded: notes.txt (890 chars)
#
#   Total: 2 documents loaded
#   mlx-guide.md: 6 chunks
#   notes.txt: 2 chunks
#
#   Total: 8 chunks from 2 documents
```

## 실습: 청킹 결과 관찰하기

```python
chunks = load_and_chunk_directory("docs/", chunk_size=500, chunk_overlap=50)

# 각 조각의 내용과 메타데이터 확인
for i, chunk in enumerate(chunks[:5]):  # 처음 5개만
    print(f"\n{'=' * 50}")
    print(f"Chunk {i}")
    print(f"  Source: {chunk['metadata']['source']}")
    print(f"  Index:  {chunk['metadata']['chunk_index']} / {chunk['metadata']['chunk_total']}")
    print(f"  Length: {len(chunk['text'])} chars")
    print(f"  Text:   {chunk['text'][:150]}...")
```

다양한 chunk_size로 바꿔보며 결과가 어떻게 달라지는지 관찰해보세요:

```python
# chunk_size를 바꿔보세요
for size in [200, 500, 1000]:
    chunks = load_and_chunk_directory("docs/", chunk_size=size)
    print(f"  chunk_size={size}: {len(chunks)} chunks\n")
```

## 핵심 정리

1. 문서를 **작은 조각(chunk)**으로 나눠야 검색 정밀도가 높아진다
2. **chunk_size**는 조각의 크기, **chunk_overlap**은 겹침 영역이다
3. 겹침이 없으면 **문장이 잘려서** 의미가 손실될 수 있다
4. `RecursiveCharacterTextSplitter`는 문단 → 줄 → 문장 → 단어 순으로 자연스러운 경계에서 자른다
5. 각 조각에 **메타데이터**(출처, 순서)를 반드시 보존해야 한다
6. 시작은 **chunk_size=500, overlap=50**으로

---

이전: [04. 문서 준비](04-document-loading.md) | 다음: [06. 의미를 숫자로](06-embedding.md)
