# 07. 벡터 저장소 (Vector Store)

## 이 단계의 위치

```
                                              ★ 여기!
 ① 로딩        ② 청킹          ③ 임베딩         ④ 저장
┌──────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 문서  │ →  │ 텍스트    │ →  │ 벡터     │ →  │ 벡터 DB  │
│ 파일  │    │ 조각들    │    │ 변환     │    │ 저장     │
└──────┘    └──────────┘    └──────────┘    └──────────┘
```

임베딩으로 만든 벡터를 **저장**하고, 나중에 비슷한 벡터를 **빠르게 찾을** 수 있게 하는 단계입니다.

## 왜 별도의 데이터베이스가 필요한가?

### 일반 데이터베이스 vs 벡터 데이터베이스

일반 데이터베이스(MySQL, PostgreSQL)는 **정확한 일치**로 검색합니다:

```sql
-- "정확히 이 제목인 문서를 찾아줘"
SELECT * FROM docs WHERE title = 'MLX Guide'
```

하지만 RAG에서는 **의미적 유사성**으로 검색해야 합니다:

```
-- "MLX의 장점과 비슷한 의미를 가진 문서 조각을 찾아줘"
-- 이건 SQL로는 불가능합니다!
```

벡터 데이터베이스는 "이 벡터와 가장 가까운 벡터를 찾아줘"라는 질문에 답할 수 있습니다.

### 비유: 도서관의 서가 정리

사서가 색인 카드를 정리할 때:

- **일반 서가**: 가나다 순, 날짜 순 — 정확한 제목을 알아야 찾을 수 있음
- **벡터 서가**: 비슷한 내용끼리 가까운 서랍에 — "이것과 비슷한 내용"으로 찾을 수 있음

## ChromaDB: 우리의 벡터 저장소

이 프로젝트에서는 **ChromaDB**를 사용합니다.

### 왜 ChromaDB인가?

| 특성 | ChromaDB | 다른 옵션 |
|------|----------|----------|
| 서버 필요? | X — Python 안에서 바로 실행 | Pinecone, Weaviate: 서버 필요 |
| 설치 | `pip install chromadb` 끝 | 일부는 Docker 필요 |
| 데이터 저장 | 디스크에 자동 저장 | FAISS: 수동 저장/로드 |
| 메타데이터 검색 | 지원 | FAISS: 미지원 |
| 학습 비용 | 매우 낮음 | 다양 |

> ChromaDB는 "파일 하나 열듯이" 쓸 수 있는 벡터 DB입니다. 서버를 띄울 필요 없고, 프로그램을 종료해도 데이터가 디스크에 남습니다.

### ChromaDB의 핵심 개념

```
ChromaDB Client
  └── Collection (컬렉션) = 테이블
        └── 문서 하나하나 = 행
              ├── id:        고유 식별자
              ├── embedding: 벡터
              ├── document:  원본 텍스트
              └── metadata:  메타데이터 (출처, 형식 등)
```

- **Client**: ChromaDB에 연결하는 객체
- **Collection**: 벡터들의 모음 (일반 DB의 "테이블"에 해당)
- **Document**: 하나의 청크 (벡터 + 텍스트 + 메타데이터)

## 코드로 사용하기

### 1. 클라이언트 생성

```python
import chromadb

# PersistentClient = 데이터를 디스크에 저장 (프로그램 종료 후에도 유지)
client = chromadb.PersistentClient(path="./chroma_db")

# 참고: 테스트용으로 메모리에만 저장하려면:
# client = chromadb.Client()  # 프로그램 종료 시 데이터 사라짐
```

### 2. 컬렉션 생성

```python
# 컬렉션 만들기 (이미 있으면 가져오기)
collection = client.get_or_create_collection(
    name="my_documents",
    metadata={"hnsw:space": "cosine"},  # 코사인 유사도 사용
)
```

> **`hnsw:space`란?** 벡터 간 거리를 계산하는 방식입니다. `cosine`은 코사인 유사도, `l2`는 유클리드 거리입니다. 대부분의 임베딩 모델은 코사인 유사도에 맞춰 학습되었으므로 `cosine`을 사용합니다.

### 3. 벡터 추가

```python
collection.add(
    ids=["chunk_0", "chunk_1", "chunk_2"],
    embeddings=[
        [0.1, 0.2, 0.3, ...],   # chunk_0의 벡터
        [0.4, 0.5, 0.6, ...],   # chunk_1의 벡터
        [0.7, 0.8, 0.9, ...],   # chunk_2의 벡터
    ],
    documents=[
        "MLX는 Apple Silicon에 최적화되어 있습니다.",
        "통합 메모리를 통해 CPU와 GPU가 같은 메모리를 공유합니다.",
        "ChromaDB는 로컬 벡터 데이터베이스입니다.",
    ],
    metadatas=[
        {"source": "mlx-guide.md", "chunk_index": 0},
        {"source": "mlx-guide.md", "chunk_index": 1},
        {"source": "chromadb-intro.md", "chunk_index": 0},
    ],
)
```

| 매개변수 | 역할 | 비유 |
|---------|------|------|
| `ids` | 각 항목의 고유 ID | 색인 카드 번호 |
| `embeddings` | 벡터 (숫자 배열) | 카드의 좌표 |
| `documents` | 원본 텍스트 | 카드에 적힌 내용 |
| `metadatas` | 부가 정보 | 카드 뒷면 메모 (출처 등) |

### 4. 유사도 검색

```python
results = collection.query(
    query_embeddings=[query_vector],   # 질문의 벡터
    n_results=3,                       # 상위 3개 반환
    include=["documents", "metadatas", "distances"],
)
```

결과 구조:

```python
# results는 딕셔너리입니다
results["documents"][0]   # ["텍스트1", "텍스트2", "텍스트3"]
results["metadatas"][0]   # [{"source": "guide.md", ...}, ...]
results["distances"][0]   # [0.12, 0.34, 0.67]  ← 거리 (낮을수록 비슷)
```

> **주의: distance vs similarity**
> - 코사인 **유사도**(similarity): 높을수록 비슷 (1.0 = 동일)
> - ChromaDB의 **거리**(distance): **낮을수록** 비슷 (0.0 = 동일)
> - 관계: `distance = 1 - similarity`

### distance 값 해석

| distance | 의미 | 설명 |
|----------|------|------|
| 0.0 ~ 0.2 | 매우 유사 | 거의 같은 내용 |
| 0.2 ~ 0.5 | 관련 있음 | 같은 주제의 다른 측면 |
| 0.5 ~ 0.8 | 약한 관련 | 간접적으로 관련 |
| 0.8 ~ 1.0 | 관련 없음 | 다른 주제 |

## 전체 저장 함수

이전 단계의 임베딩된 청크를 ChromaDB에 저장합니다:

```python
import chromadb

def store_chunks(
    chunks: list[dict],
    db_path: str = "./chroma_db",
    collection_name: str = "my_documents",
) -> chromadb.Collection:
    """Store embedded chunks in ChromaDB.

    Each chunk must have 'text', 'metadata', and 'embedding' fields.
    Returns the ChromaDB collection for later querying.
    """
    client = chromadb.PersistentClient(path=db_path)

    # Delete existing collection if re-indexing
    try:
        client.delete_collection(collection_name)
    except ValueError:
        pass  # Collection doesn't exist yet

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Prepare batch data
    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        # Create unique ID: "filename_chunkindex"
        source = chunk["metadata"].get("source", "unknown")
        idx = chunk["metadata"].get("chunk_index", i)
        chunk_id = f"{source}_{idx}"

        ids.append(chunk_id)
        embeddings.append(chunk["embedding"].tolist())  # numpy → list
        documents.append(chunk["text"])

        # ChromaDB metadata must be str, int, float, or bool
        clean_meta = {
            k: v for k, v in chunk["metadata"].items()
            if isinstance(v, (str, int, float, bool))
        }
        metadatas.append(clean_meta)

    # Add all at once
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"  Stored {len(chunks)} chunks in '{collection_name}'")
    print(f"  Database path: {db_path}")
    return collection
```

## 컬렉션 관리

```python
# 저장된 청크 수 확인
print(f"Total chunks: {collection.count()}")

# 컬렉션 삭제 (재인덱싱할 때)
client.delete_collection("my_documents")

# 특정 문서의 청크만 삭제
collection.delete(
    where={"source": "old_document.md"}
)

# 특정 문서에서만 검색
results = collection.query(
    query_embeddings=[query_vec],
    n_results=3,
    where={"source": "mlx-guide.md"},  # mlx-guide.md 내에서만 검색
)
```

## 전체 인덱싱 파이프라인

지금까지 배운 모든 단계를 연결합니다:

```python
from sentence_transformers import SentenceTransformer

# === 전체 인덱싱 파이프라인 ===

# 1. 문서 로딩 + 청킹
print("Step 1: Loading and chunking documents...")
chunks = load_and_chunk_directory("docs/", chunk_size=500, chunk_overlap=50)

# 2. 임베딩
print("\nStep 2: Embedding chunks...")
embed_model = SentenceTransformer("BAAI/bge-m3")
chunks = embed_chunks(chunks, embed_model)

# 3. ChromaDB에 저장
print("\nStep 3: Storing in ChromaDB...")
collection = store_chunks(chunks)

print(f"\nDone! {collection.count()} chunks indexed and ready for search.")
```

```
Step 1: Loading and chunking documents...
  Loaded: mlx-guide.md (2,340 chars)
  Loaded: notes.txt (890 chars)
  Total: 2 documents loaded
  mlx-guide.md: 6 chunks
  notes.txt: 2 chunks
  Total: 8 chunks from 2 documents

Step 2: Embedding chunks...
  Embedded 8 chunks (dim=1024)

Step 3: Storing in ChromaDB...
  Stored 8 chunks in 'my_documents'
  Database path: ./chroma_db

Done! 8 chunks indexed and ready for search.
```

이 과정은 **한 번만** 하면 됩니다. `./chroma_db/` 폴더에 데이터가 저장되므로, 프로그램을 종료해도 데이터가 유지됩니다.

## 실습

```python
# 1. 인덱싱 실행 (위 코드)

# 2. 테스트 검색
query = "통합 메모리란?"
query_vec = embed_model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_vec],
    n_results=3,
    include=["documents", "metadatas", "distances"],
)

print(f"Query: {query}\n")
for i in range(len(results["documents"][0])):
    doc = results["documents"][0][i]
    meta = results["metadatas"][0][i]
    dist = results["distances"][0][i]
    print(f"--- Result {i+1} (distance: {dist:.4f}) ---")
    print(f"Source: {meta['source']}, chunk {meta.get('chunk_index', '?')}")
    print(f"Text: {doc[:150]}...")
    print()
```

검색 결과를 보면서 확인하세요:
- 질문과 관련된 청크가 상위에 나오는가?
- distance 값이 낮은(비슷한) 것이 먼저 나오는가?
- 출처 정보가 정확한가?

## 핵심 정리

1. **벡터 저장소**는 벡터를 저장하고 비슷한 벡터를 빠르게 찾아주는 특수 DB이다
2. **ChromaDB**는 설치만 하면 바로 쓸 수 있는 로컬 벡터 DB이다
3. 각 항목은 `id` + `embedding` + `document` + `metadata`로 구성된다
4. ChromaDB의 distance는 **낮을수록 비슷**하다 (similarity와 반대)
5. 인덱싱은 **한 번만** 하면 되고, 데이터는 디스크에 영구 저장된다

---

이전: [06. 의미를 숫자로](06-embedding.md) | 다음: [08. 검색하기](08-retrieval.md)
