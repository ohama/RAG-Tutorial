# 08. 검색하기 (Retrieval)

## 이 단계의 위치

이제 질문 단계(Query Pipeline)로 넘어갑니다!

```
═══════════════ 질문 단계 ═══════════════

 ★ 여기!
 ⑤ 질문임베딩    ⑥ 검색          ⑦ 프롬프트      ⑧ 생성
┌──────────┐  ┌──────────┐    ┌──────────┐   ┌──────────┐
│ 질문을    │→ │ 비슷한   │ →  │ 질문 +   │ → │ LLM이    │ → 답변
│ 벡터로   │  │ 벡터 찾기 │    │ 관련 문서 │   │ 답변 생성 │
└──────────┘  └──────────┘    └──────────┘   └──────────┘
```

준비 단계에서 문서를 벡터로 변환하여 ChromaDB에 저장했습니다.
이제 질문이 들어오면 **관련 있는 조각을 찾아오는** 과정을 배웁니다.

## 검색의 원리 (복습)

[06. 임베딩](06-embedding.md)에서 배운 것을 떠올려보세요:

1. 비슷한 의미의 텍스트는 **비슷한 벡터**를 가진다
2. 비슷한 벡터는 벡터 공간에서 **가까이** 있다
3. 코사인 유사도로 **가까운 정도**를 측정한다

검색은 이 원리를 그대로 활용합니다:

```
질문: "Apple Silicon의 통합 메모리란?"
    ↓ 임베딩
질문 벡터: [0.45, -0.12, 0.78, ...]
    ↓ ChromaDB에서 가장 가까운 벡터 찾기

결과:
  1위: "Apple Silicon은 CPU와 GPU가 같은 메모리를..."  (distance: 0.08)
  2위: "통합 메모리를 활용하여 데이터 복사 없이..."     (distance: 0.15)
  3위: "MLX는 Apple Silicon에 최적화된..."             (distance: 0.22)
```

## 검색 함수 만들기

```python
from sentence_transformers import SentenceTransformer
import chromadb

class Retriever:
    """Find relevant document chunks for a query.

    This is the 'librarian' of our RAG system.
    Given a question, it finds the most relevant document chunks.
    """

    def __init__(
        self,
        embed_model_name: str = "BAAI/bge-m3",
        db_path: str = "./chroma_db",
        collection_name: str = "my_documents",
    ):
        # Load the same embedding model used during indexing
        self.embed_model = SentenceTransformer(embed_model_name)

        # Connect to the existing ChromaDB
        client = chromadb.PersistentClient(path=db_path)
        self.collection = client.get_collection(collection_name)

        print(f"Retriever ready: {self.collection.count()} chunks indexed")

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Find the top_k most relevant chunks for a query.

        Steps:
        1. Convert query to vector (same model as indexing!)
        2. Search ChromaDB for nearest vectors
        3. Return chunks with text, metadata, and similarity score
        """
        # Step 1: Embed the query
        query_vec = self.embed_model.encode(query).tolist()

        # Step 2: Search
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Step 3: Format results
        matches = []
        for i in range(len(results["documents"][0])):
            matches.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "rank": i + 1,
            })

        return matches
```

```python
# 사용 예
retriever = Retriever()
results = retriever.retrieve("통합 메모리란?", top_k=3)

for r in results:
    print(f"[Rank {r['rank']}] distance={r['distance']:.4f}")
    print(f"  Source: {r['metadata']['source']}")
    print(f"  Text: {r['text'][:100]}...")
    print()
```

## top_k: 몇 개를 가져올까?

`top_k`는 검색할 청크의 수입니다. 이 값에 따라 답변의 품질이 달라집니다.

### 비유: 사서에게 "관련 자료 몇 개 가져다 주세요"

```
top_k = 1: "가장 관련 있는 페이지 1개만요"
  → 핵심만 있지만, 중요한 정보를 놓칠 수 있음

top_k = 3: "관련 자료 3개 정도요" ★ 보통 이것이 적당
  → 충분한 맥락, 관련 없는 것이 섞일 확률 낮음

top_k = 10: "관련될 만한 자료 10개요"
  → 정보는 많지만, 관련 없는 자료가 섞이기 시작
  → LLM이 중간의 정보를 무시하는 "Lost in the Middle" 문제 발생
```

### "Lost in the Middle" 문제

LLM은 프롬프트의 **처음과 끝**에 주목하고, **중간을 무시**하는 경향이 있습니다.

```
컨텍스트에 10개 청크를 넣으면:

[chunk 1] ← LLM이 잘 읽음
[chunk 2]
[chunk 3]
[chunk 4]
[chunk 5] ← 여기가 가장 중요한 정보인데... LLM이 놓침!
[chunk 6]
[chunk 7]
[chunk 8]
[chunk 9]
[chunk 10] ← LLM이 잘 읽음
```

이것은 LLM의 알려진 한계입니다. 해결 방법:
- **top_k를 3-5로 유지** — 가장 간단하고 효과적
- 나중에 고급 기법: 재순위(reranking), 역순 정렬 등

> **권장**: `top_k=3`으로 시작하세요.

## 프롬프트 구성: RAG의 핵심

검색된 청크를 질문과 합쳐서 LLM에게 보내는 프롬프트를 만듭니다.

이것이 RAG의 "Augmented" 부분입니다 — 검색 결과로 LLM의 능력을 **증강**합니다.

```python
def build_rag_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """Build a prompt that includes retrieved context.

    This is the heart of RAG: combining the question with
    relevant document chunks so the LLM can answer based on facts.
    """
    # Format each chunk with source information
    context_parts = []
    for chunk in retrieved_chunks:
        source = chunk["metadata"].get("source", "unknown")
        idx = chunk["metadata"].get("chunk_index", "?")
        context_parts.append(
            f"[Source: {source}, Chunk {idx}]\n{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""다음 참고 문서를 바탕으로 질문에 답변하세요.

규칙:
- 참고 문서에 있는 정보만 사용하세요
- 문서에 없는 내용은 "문서에서 해당 정보를 찾을 수 없습니다"라고 답하세요
- 답변 끝에 참고한 문서의 출처를 표시하세요

## 참고 문서

{context}

## 질문

{query}

## 답변"""

    return prompt
```

### 프롬프트의 각 부분이 하는 일

```
┌─────────────────────────────────────────────────┐
│ "참고 문서를 바탕으로 답변하세요"                    │ ← ① 역할 지정
│                                                 │
│ "문서에 없는 내용은 없다고 답하세요"                 │ ← ② 환각 방지
│                                                 │
│ "출처를 표시하세요"                                │ ← ③ 출처 추적
│                                                 │
│ [검색된 청크 1]                                   │
│ [검색된 청크 2]                                   │ ← ④ 컨텍스트
│ [검색된 청크 3]                                   │
│                                                 │
│ "질문: MLX의 장점은?"                              │ ← ⑤ 질문
└─────────────────────────────────────────────────┘
```

① **역할 지정**: "문서를 참고하라" → 자기 기억 대신 주어진 문서를 사용하도록
② **환각 방지**: "없으면 없다고 해라" → 지어내기 방지
③ **출처 추적**: 어떤 청크에서 왔는지 표시
④ **컨텍스트**: 실제 검색된 문서 조각들
⑤ **질문**: 사용자의 원래 질문

## 프롬프트 시각화

RAG의 가장 큰 장점 중 하나는 **LLM에 무엇이 전달되는지 확인**할 수 있다는 것입니다:

```python
def retrieve_and_show(retriever, query, top_k=3):
    """Retrieve chunks and display everything the LLM will see.

    This function makes the RAG process transparent:
    you can see exactly what context the LLM receives.
    """
    # Retrieve
    chunks = retriever.retrieve(query, top_k=top_k)

    # Show retrieved chunks
    print("=" * 60)
    print(f"QUERY: {query}")
    print("=" * 60)

    print("\n📚 RETRIEVED CHUNKS:\n")
    for chunk in chunks:
        similarity = 1 - chunk["distance"]  # distance → similarity
        bar = "█" * int(similarity * 20)
        print(f"  [{chunk['rank']}] similarity={similarity:.3f} {bar}")
        print(f"      Source: {chunk['metadata'].get('source', '?')}")
        print(f"      Text: {chunk['text'][:120]}...")
        print()

    # Build prompt
    prompt = build_rag_prompt(query, chunks)

    print("=" * 60)
    print("📝 FULL PROMPT TO LLM:")
    print("=" * 60)
    print(prompt)
    print("=" * 60)

    return prompt, chunks
```

이 함수를 Jupyter 노트북에서 실행하면, **LLM이 보는 모든 것**을 눈으로 확인할 수 있습니다. 이것이 프레임워크를 사용하지 않는 이유입니다 — 블랙박스가 없습니다.

## 검색 품질 진단

### 결과가 좋을 때

```
Query: "통합 메모리란?"
[1] similarity=0.92  ████████████████████
    "Apple Silicon은 CPU와 GPU가 같은 메모리를 공유합니다..."

→ 질문과 직접적으로 관련된 내용이 높은 유사도로 반환됨 ✓
```

### 결과가 나쁠 때

```
Query: "통합 메모리란?"
[1] similarity=0.35  ███████
    "Python 3.11은 CPython의 속도를 향상시켰습니다..."

→ 관련 없는 내용이 반환됨 ✗
```

**원인과 해결**:

| 증상 | 가능한 원인 | 해결 |
|------|-----------|------|
| 관련 없는 결과 | 임베딩 모델이 한국어 미지원 | bge-m3 확인 |
| 유사도가 전체적으로 낮음 | chunk_size가 너무 작아 맥락 부족 | chunk_size 늘리기 |
| 같은 내용이 여러 번 | overlap이 너무 큼 | overlap 줄이기 |
| 필요한 정보가 안 나옴 | 문서에 해당 내용이 없음 | 문서 확인 |

## 실습

```python
# 1. Retriever 생성
retriever = Retriever()

# 2. 다양한 질문으로 검색 테스트
queries = [
    "MLX의 통합 메모리란 무엇인가?",
    "ChromaDB는 왜 사용하나?",
    "Python 가상환경은 어떻게 만드나?",
    "내일 날씨는?",  # 문서에 없는 질문
]

for query in queries:
    prompt, chunks = retrieve_and_show(retriever, query, top_k=3)
    print("\n" + "─" * 60 + "\n")
```

마지막 질문("내일 날씨는?")의 결과를 주목하세요. 유사도가 매우 낮을 것입니다. 이 경우 RAG 시스템은 "문서에서 해당 정보를 찾을 수 없습니다"라고 답해야 합니다.

## 핵심 정리

1. 검색은 **질문 벡터와 가장 가까운 청크 벡터를 찾는** 것이다
2. `top_k=3`이 대부분의 경우 적당하다
3. 너무 많은 청크를 가져오면 **Lost in the Middle** 문제가 생긴다
4. 프롬프트에 **역할 지정, 환각 방지, 출처 추적** 규칙을 넣는다
5. `retrieve_and_show()`로 LLM에 전달되는 **모든 것을 확인**할 수 있다
6. 검색 결과가 나쁘면 임베딩 모델, chunk_size, 문서 자체를 점검한다

---

이전: [07. 벡터 저장소](07-vector-store.md) | 다음: [09. RAG 파이프라인 조립](09-rag-pipeline.md)
