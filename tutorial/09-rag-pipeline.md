# 09. RAG 파이프라인 조립

## 드디어 연결할 시간

지금까지 배운 모든 조각을 하나로 합칩니다:

```
                        전체 RAG 파이프라인
═══════════════════════════════════════════════════════

 [문서 로딩] → [청킹] → [임베딩] → [ChromaDB 저장]     ← 준비 (한 번)

 [질문] → [질문 임베딩] → [검색] → [프롬프트 구성] → [LLM 생성] → [답변]
                                                      ← 질문 (매번)
```

도서관 비유의 완성:
1. 사서가 책을 정리해두었습니다 (준비 단계 완료)
2. 학생이 질문합니다
3. 사서가 관련 페이지를 찾아줍니다
4. 학생이 그 페이지를 읽고 답변합니다

## RAG 파이프라인 클래스

```python
from mlx_lm import load, generate
from sentence_transformers import SentenceTransformer
import chromadb
import mlx.core as mx

class RAGPipeline:
    """Complete RAG pipeline: retrieve relevant chunks and generate answers.

    Connects all the pieces we've built:
    - Retriever: finds relevant document chunks
    - Generator: produces answers using Qwen + retrieved context
    """

    def __init__(
        self,
        llm_model: str = "mlx-community/Qwen2.5-7B-Instruct-4bit",
        embed_model: str = "BAAI/bge-m3",
        db_path: str = "./chroma_db",
        collection_name: str = "my_documents",
    ):
        print("Loading models...")

        # Load embedding model (for query embedding)
        print("  Loading embedding model...")
        self.embed_model = SentenceTransformer(embed_model)

        # Load LLM (for answer generation)
        print("  Loading LLM...")
        self.llm, self.tokenizer = load(llm_model)

        # Connect to vector store
        client = chromadb.PersistentClient(path=db_path)
        self.collection = client.get_collection(collection_name)

        print(f"  Ready! {self.collection.count()} chunks indexed")
        print(f"  GPU memory: {mx.metal.get_active_memory() / 1e9:.1f} GB")

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Find relevant chunks for a query."""
        query_vec = self.embed_model.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for i in range(len(results["documents"][0])):
            chunks.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return chunks

    def generate_answer(
        self,
        query: str,
        context_chunks: list[dict],
        max_tokens: int = 500,
        temp: float = 0.3,
    ) -> str:
        """Generate an answer using the LLM with retrieved context."""
        # Build context string
        context_parts = []
        for chunk in context_chunks:
            source = chunk["metadata"].get("source", "unknown")
            idx = chunk["metadata"].get("chunk_index", "?")
            context_parts.append(
                f"[Source: {source}, Chunk {idx}]\n{chunk['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # Build chat messages
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Answer based ONLY on the provided documents. "
                    "If the answer is not in the documents, say so. "
                    "Cite sources. Answer in Korean."
                ),
            },
            {
                "role": "user",
                "content": f"## 참고 문서\n\n{context}\n\n## 질문\n\n{query}",
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        response = generate(
            self.llm, self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temp,
        )

        # Clean up GPU cache (essential for stable memory)
        mx.metal.clear_cache()

        return response

    def ask(self, query: str, top_k: int = 3, show_context: bool = True) -> str:
        """Complete RAG pipeline: retrieve + generate.

        This is the main function users call.
        """
        # Step 1: Retrieve relevant chunks
        chunks = self.retrieve(query, top_k=top_k)

        # Step 2: Show what was retrieved (transparency!)
        if show_context:
            print(f"📚 Retrieved {len(chunks)} chunks:\n")
            for i, chunk in enumerate(chunks):
                similarity = 1 - chunk["distance"]
                source = chunk["metadata"].get("source", "?")
                print(f"  [{i+1}] {source} (similarity: {similarity:.3f})")
                print(f"      {chunk['text'][:80]}...")
                print()

        # Step 3: Generate answer
        answer = self.generate_answer(query, chunks)

        return answer

    def ask_without_rag(self, query: str, max_tokens: int = 500) -> str:
        """Ask the LLM directly without RAG (for comparison).

        Same question, but the LLM uses only its training data.
        """
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer in Korean.",
            },
            {"role": "user", "content": query},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        response = generate(
            self.llm, self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=0.3,
        )

        mx.metal.clear_cache()
        return response
```

## 파이프라인 사용하기

```python
# RAG 파이프라인 생성
rag = RAGPipeline()

# RAG로 질문
answer = rag.ask("Apple Silicon의 통합 메모리란 무엇인가요?")
print("\n💬 Answer:")
print(answer)
```

출력 예:

```
📚 Retrieved 3 chunks:

  [1] mlx-guide.md (similarity: 0.921)
      Apple Silicon은 CPU와 GPU가 같은 메모리를 공유합니다. 이를 통합 메모리...

  [2] mlx-guide.md (similarity: 0.856)
      128GB 통합 메모리를 가진 Mac Studio에서는 70B 파라미터 모델도 실행할...

  [3] mlx-guide.md (similarity: 0.734)
      MLX는 Apple이 Apple Silicon을 위해 만든 머신러닝 프레임워크입니다...

💬 Answer:
Apple Silicon의 통합 메모리(Unified Memory)란, CPU와 GPU가 같은 물리적
메모리를 공유하는 아키텍처입니다. 일반적인 시스템에서는 CPU 메모리(RAM)와
GPU 메모리(VRAM)가 별도로 존재하여 데이터를 복사해야 하지만, Apple Silicon
에서는 이 복사가 필요 없습니다. 이는 특히 대규모 모델을 실행할 때 큰 이점을
제공하며, 128GB 통합 메모리를 가진 Mac Studio에서는 70B 파라미터 모델도
실행할 수 있습니다.

(출처: mlx-guide.md)
```

## 내부 동작 살펴보기

파이프라인의 각 단계에서 무슨 일이 일어나는지 시각적으로 보여줍니다:

```python
def ask_verbose(rag, query):
    """Show every step of the RAG pipeline in detail."""

    print("=" * 60)
    print(f"QUESTION: {query}")
    print("=" * 60)

    # Step 1: Retrieve
    print("\n── Step 1: RETRIEVE ──")
    chunks = rag.retrieve(query, top_k=3)
    for i, c in enumerate(chunks):
        sim = 1 - c["distance"]
        print(f"  [{i+1}] sim={sim:.3f} | {c['metadata'].get('source','?')}")
        print(f"      \"{c['text'][:80]}...\"")

    # Step 2: Build prompt
    print("\n── Step 2: BUILD PROMPT ──")
    context_parts = []
    for c in chunks:
        source = c["metadata"].get("source", "?")
        idx = c["metadata"].get("chunk_index", "?")
        context_parts.append(f"[Source: {source}, Chunk {idx}]\n{c['text']}")
    context = "\n---\n".join(context_parts)
    print(f"  Context length: {len(context)} chars")
    print(f"  Chunks included: {len(chunks)}")

    # Step 3: Generate
    print("\n── Step 3: GENERATE ──")
    answer = rag.generate_answer(query, chunks)
    print(f"  Answer length: {len(answer)} chars")
    print(f"  GPU memory: {mx.metal.get_active_memory() / 1e9:.1f} GB")

    # Result
    print("\n── ANSWER ──")
    print(answer)
    print("=" * 60)

    return answer
```

```python
ask_verbose(rag, "MLX에서 메모리 관리는 어떻게 하나요?")
```

이 함수를 통해 RAG의 **모든 내부 동작**을 볼 수 있습니다. 블랙박스가 없습니다.

## 문서에 없는 질문 처리

RAG는 문서에 없는 내용에 대해 "모릅니다"라고 답해야 합니다:

```python
answer = rag.ask("내일 서울 날씨는 어떤가요?")
```

```
📚 Retrieved 3 chunks:

  [1] mlx-guide.md (similarity: 0.123)  ← 유사도가 매우 낮음
      ...

💬 Answer:
문서에서 날씨에 관한 정보를 찾을 수 없습니다.
인덱싱된 문서는 MLX 프레임워크에 관한 기술 문서입니다.
```

이것이 RAG의 장점입니다. LLM이 자기 기억으로 지어내는 대신, "문서에 없다"고 정직하게 답합니다.

## 에러 없이 동작하는지 확인

```python
# 여러 질문을 연속으로 테스트
test_queries = [
    "MLX란 무엇인가요?",
    "통합 메모리의 장점은?",
    "Lazy evaluation이란?",
    "오늘 점심 메뉴는?",  # 문서에 없는 질문
]

for query in test_queries:
    print(f"\n{'─' * 40}")
    answer = rag.ask(query, show_context=False)
    print(f"Q: {query}")
    print(f"A: {answer[:200]}...")

    # 메모리 확인
    mem = mx.metal.get_active_memory() / 1e9
    print(f"   [Memory: {mem:.1f} GB]")
```

메모리가 질문마다 안정적으로 유지되는지 확인하세요. `clear_cache()`가 제대로 동작하고 있다면, 메모리가 계속 증가하지 않을 것입니다.

## 핵심 정리

1. **RAGPipeline**은 Retriever(검색) + Generator(생성)를 하나로 합친 것이다
2. `ask()` 하나로 전체 파이프라인이 동작한다: 검색 → 프롬프트 구성 → 생성
3. `show_context=True`로 검색 결과를 투명하게 확인할 수 있다
4. `ask_without_rag()`로 RAG 없는 답변을 별도로 얻을 수 있다 (비교용)
5. 문서에 없는 질문에 "없다"고 답하는 것이 RAG의 장점이다
6. 매 생성 후 `mx.metal.clear_cache()`로 메모리를 안정적으로 유지한다

---

이전: [08. 검색하기](08-retrieval.md) | 다음: [10. RAG의 차이](10-rag-comparison.md)
