# 10. RAG의 차이 — 최종 비교

## 이 프로젝트의 목표

이 튜토리얼 시리즈의 최종 목표입니다:

**같은 질문을 RAG 없이 / RAG와 함께 물어보고, 답변 품질이 어떻게 달라지는지 직접 비교합니다.**

## 비교 방법

하나의 질문에 대해 두 가지 답변을 나란히 생성합니다:

```
                     같은 질문
                    ↙         ↘
            RAG OFF              RAG ON
         (LLM 기억만)          (문서 검색 후)
              ↓                     ↓
           답변 A                답변 B
              ↓                     ↓
         ┌──────────┐        ┌──────────┐
         │ 일반적인  │        │ 정확하고  │
         │ 답변     │        │ 근거 있는 │
         │ (환각?)  │        │ 답변     │
         │          │        │ (출처O)  │
         └──────────┘        └──────────┘
```

## 비교 함수

```python
def compare_rag(rag_pipeline, query: str, top_k: int = 3):
    """Compare answers with and without RAG side by side.

    This is the core educational artifact of this project.
    Shows exactly what difference RAG makes.
    """
    print("=" * 70)
    print(f"  QUESTION: {query}")
    print("=" * 70)

    # ── RAG OFF ──
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  RAG OFF — LLM의 학습 데이터만으로 답변                       │")
    print("└─────────────────────────────────────────────────────────────┘\n")

    answer_without = rag_pipeline.ask_without_rag(query)
    print(answer_without)

    # ── RAG ON ──
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  RAG ON — 문서 검색 후 답변                                  │")
    print("└─────────────────────────────────────────────────────────────┘\n")

    # Show retrieved chunks
    chunks = rag_pipeline.retrieve(query, top_k=top_k)
    print("📚 검색된 문서 조각:")
    for i, chunk in enumerate(chunks):
        similarity = 1 - chunk["distance"]
        source = chunk["metadata"].get("source", "?")
        idx = chunk["metadata"].get("chunk_index", "?")
        print(f"  [{i+1}] {source} chunk {idx} (유사도: {similarity:.3f})")
        print(f"      \"{chunk['text'][:100]}...\"\n")

    answer_with = rag_pipeline.generate_answer(query, chunks)
    print("💬 답변:")
    print(answer_with)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    print(f"  RAG OFF: {len(answer_without)} chars")
    print(f"  RAG ON:  {len(answer_with)} chars")
    print(f"  Sources: {', '.join(c['metadata'].get('source','?') for c in chunks)}")
    print("=" * 70)

    return {
        "query": query,
        "answer_without_rag": answer_without,
        "answer_with_rag": answer_with,
        "chunks": chunks,
    }
```

## 비교 실험

### 실험 1: 문서에 있는 내용

문서에 명확히 적혀 있는 내용을 물어봅니다.

```python
rag = RAGPipeline()

compare_rag(rag, "Apple Silicon의 통합 메모리란 무엇이고 왜 중요한가요?")
```

**기대 결과:**

| | RAG OFF | RAG ON |
|---|---|---|
| 정확성 | 일반적인 설명 (대체로 맞지만 구체성 부족) | 문서의 구체적 내용 인용 |
| 출처 | 없음 | "mlx-guide.md에 따르면..." |
| 환각 위험 | 있음 (세부 사항을 지어낼 수 있음) | 낮음 (문서 내용 기반) |

### 실험 2: LLM이 모르는 내용

LLM의 학습 데이터에 없을 가능성이 높은 내용을 물어봅니다.

```python
compare_rag(rag, "우리 팀의 코드 리뷰 규칙은 어떻게 되나요?")
```

이 실험을 위해서는 팀 규칙 문서를 미리 인덱싱해 두어야 합니다.

**기대 결과:**

| | RAG OFF | RAG ON |
|---|---|---|
| 답변 | 일반적인 코드 리뷰 모범 사례 | 실제 팀 규칙 |
| 정확성 | 우리 팀 규칙과 다름 | 문서와 일치 |

### 실험 3: 문서에 없는 내용

인덱싱된 문서에 전혀 없는 내용을 물어봅니다.

```python
compare_rag(rag, "내일 서울의 날씨는 어떤가요?")
```

**기대 결과:**

| | RAG OFF | RAG ON |
|---|---|---|
| 답변 | "일반적으로 서울의 날씨는..." (지어냄) | "문서에서 해당 정보를 찾을 수 없습니다" |

RAG ON이 "모른다"고 정직하게 답하는 것이 오히려 좋은 결과입니다!

### 실험 4: 여러 질문 연속 비교

```python
queries = [
    "MLX에서 lazy evaluation이란?",
    "벡터 데이터베이스의 역할은?",
    "128GB Mac에서 어떤 크기의 모델을 실행할 수 있나요?",
]

results = []
for query in queries:
    result = compare_rag(rag, query)
    results.append(result)
    print("\n" + "━" * 70 + "\n")
```

## 비교 결과 분석 가이드

각 비교 결과를 보면서 다음을 확인하세요:

### 1. 정확성 (Accuracy)

```
RAG OFF: "MLX는 PyTorch와 유사한 API를 제공합니다" ← 일반론 (맞을 수도 틀릴 수도)
RAG ON:  "MLX는 NumPy와 유사한 API를 제공합니다"  ← 문서에 있는 정확한 내용
```

### 2. 구체성 (Specificity)

```
RAG OFF: "대규모 모델을 실행할 수 있습니다"        ← 막연함
RAG ON:  "128GB 메모리로 70B 모델을 실행할 수 있습니다" ← 구체적 수치
```

### 3. 출처 (Source Citation)

```
RAG OFF: (출처 없음)                              ← 어디서 온 정보인지 모름
RAG ON:  (출처: mlx-guide.md, chunk 3)           ← 확인 가능
```

### 4. 환각 (Hallucination)

```
RAG OFF: "MLX는 2023년 12월에 처음 공개되었습니다" ← 진짜인지 확인 불가
RAG ON:  문서에 출시일 정보가 없으면 언급하지 않음   ← 안전
```

## 메모리 안정성 확인

비교 실험을 연속으로 실행한 후 메모리가 안정적인지 확인합니다:

```python
import mlx.core as mx

print(f"After all comparisons:")
print(f"  Active memory: {mx.metal.get_active_memory() / 1e9:.1f} GB")
print(f"  Peak memory:   {mx.metal.get_peak_memory() / 1e9:.1f} GB")
```

메모리가 계속 증가하지 않고 안정적으로 유지되어야 합니다.

## 정리: RAG가 만드는 차이

| 측면 | RAG 없이 | RAG와 함께 |
|------|---------|-----------|
| 일반 지식 | ✓ 잘 답변 | ✓ 잘 답변 (+ 문서 근거) |
| 내 문서 내용 | ✗ 모름 / 지어냄 | ✓ 정확히 답변 |
| 최신 정보 | ✗ 학습 시점까지만 | ✓ 최신 문서 있으면 가능 |
| 환각 | ✗ 그럴듯하게 지어냄 | ✓ 문서 기반, 없으면 "모름" |
| 출처 | ✗ 확인 불가 | ✓ 어떤 문서의 어디에서 왔는지 |
| 신뢰도 | 낮음 | 높음 |

## 더 나아가기

이 튜토리얼에서 배운 기본 RAG를 개선할 수 있는 방향들:

### 청킹 전략 비교
- 고정 크기 vs 재귀 분할 vs 의미 기반 청킹
- 같은 질문에 어떤 청킹이 더 좋은 결과를 내는지 비교

### 하이브리드 검색
- 벡터 검색 + 키워드 검색(BM25) 결합
- 고유명사, 코드명 등 키워드가 중요한 경우에 유리

### 검색 실패 분석
- 어떤 질문에서 검색이 실패하는지 패턴 찾기
- chunk_size, top_k 파라미터 최적화

### 평가 지표
- 답변의 정확성을 수치화 (RAGAS 프레임워크 등)
- "잘 만든 RAG"와 "못 만든 RAG"의 차이를 정량적으로 측정

## 축하합니다!

RAG의 모든 단계를 직접 구현하고, RAG가 만드는 차이를 눈으로 확인했습니다.

```
00. RAG란? ✓             → 개념 이해
01. 전체 그림 ✓          → 파이프라인 구조
02. 환경 설정 ✓          → 도구 준비
03. LLM 만나기 ✓         → LLM의 능력과 한계
04. 문서 로딩 ✓          → 파일 → 텍스트
05. 청킹 ✓              → 텍스트 → 조각
06. 임베딩 ✓             → 조각 → 벡터
07. 벡터 저장소 ✓        → 벡터 → ChromaDB
08. 검색 ✓              → 질문 → 관련 조각 찾기
09. 파이프라인 ✓         → 전체 연결
10. RAG 비교 ✓           → RAG on/off 차이 확인 ← 지금 여기!
```

핵심을 한 문장으로:

**RAG는 LLM에게 "참고 자료를 건네주는 것"이고, 이것만으로 답변의 정확성, 구체성, 신뢰도가 극적으로 향상됩니다.**

---

이전: [09. RAG 파이프라인 조립](09-rag-pipeline.md)

---

## 전체 튜토리얼 목록

| # | 제목 | 배우는 것 |
|---|------|----------|
| [00](00-what-is-rag.md) | RAG란 무엇인가? | RAG 개념, 왜 필요한가 |
| [01](01-rag-big-picture.md) | RAG의 전체 그림 | 두 파이프라인, 각 단계 |
| [02](02-environment-setup.md) | 환경 설정 | Python, MLX, 모델 설치 |
| [03](03-meet-your-llm.md) | LLM 만나기 | Qwen 모델, LLM 한계 체험 |
| [04](04-document-loading.md) | 문서 준비 | MD, PDF, TXT 파싱 |
| [05](05-chunking.md) | 텍스트 자르기 | 청킹, 오버랩, 파라미터 |
| [06](06-embedding.md) | 의미를 숫자로 | 벡터, 코사인 유사도 |
| [07](07-vector-store.md) | 벡터 저장소 | ChromaDB 사용법 |
| [08](08-retrieval.md) | 검색하기 | 유사도 검색, 프롬프트 구성 |
| [09](09-rag-pipeline.md) | 파이프라인 조립 | 전체 RAG 연결 |
| [10](10-rag-comparison.md) | RAG의 차이 | RAG on/off 비교 |
