# 04. 문서 준비와 로딩

## 이 단계의 위치

```
 ★ 여기!
 ① 로딩        ② 청킹          ③ 임베딩         ④ 저장
┌──────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 문서  │ →  │ 텍스트    │ →  │ 벡터     │ →  │ 벡터 DB  │
│ 파일  │    │ 조각들    │    │ 변환     │    │ 저장     │
└──────┘    └──────────┘    └──────────┘    └──────────┘
```

RAG의 모든 것은 **문서**에서 시작합니다. 문서가 없으면 검색할 것도 없습니다.

이 단계에서는 다양한 형식의 파일을 열어서 **순수한 텍스트**를 꺼내는 방법을 배웁니다.

## 문서 로딩이란?

비유하면, 도서관에 새 책이 도착한 것과 같습니다.

- 어떤 책은 **일반 소설** (텍스트 파일) — 그냥 읽으면 됩니다
- 어떤 책은 **사진이 있는 잡지** (마크다운) — 본문 텍스트만 뽑아야 합니다
- 어떤 책은 **복잡한 보고서** (PDF) — 표, 그림, 각주를 걸러내고 텍스트만 추출해야 합니다

결국 목표는 하나입니다: **깨끗한 텍스트**를 얻는 것.

```
guide.md  ──→ "MLX는 Apple이 만든..."     (텍스트)
report.pdf ──→ "2024년 분기 실적은..."     (텍스트)
notes.txt  ──→ "오늘 회의에서 결정된..."    (텍스트)
```

## 텍스트 파일 (.txt) 로딩

가장 간단합니다. 파일을 열어서 읽으면 끝입니다.

```python
from pathlib import Path

def load_txt(file_path: str) -> dict:
    """Load a plain text file.

    Returns a dictionary with the text content and metadata.
    Metadata is used later for source citation.
    """
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")

    return {
        "text": text,
        "metadata": {
            "source": path.name,     # "notes.txt"
            "format": "txt",
        }
    }
```

**왜 딕셔너리(dict)로 반환하는가?**

텍스트만 반환하면 이 텍스트가 **어디서 왔는지** 알 수 없습니다. 나중에 RAG가 "이 정보는 notes.txt에서 가져왔습니다"라고 출처를 표시하려면, 처음부터 메타데이터를 함께 가지고 다녀야 합니다.

```python
# 사용 예
doc = load_txt("docs/notes.txt")
print(doc["text"][:100])      # 텍스트 내용
print(doc["metadata"])         # {'source': 'notes.txt', 'format': 'txt'}
```

## 마크다운 파일 (.md) 로딩

마크다운은 `#`, `**`, `-` 같은 문법 기호가 포함되어 있습니다. 하지만 이 기호들도 내용의 일부이므로 그대로 둡니다. 대신 제목(heading)을 메타데이터로 추출합니다.

```python
def load_markdown(file_path: str) -> dict:
    """Load a markdown file.

    Keeps markdown formatting as-is (headings, bold, lists).
    Extracts the first heading as the document title.
    """
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")

    # Extract title from first heading (# Title)
    title = None
    for line in text.split("\n"):
        if line.startswith("# "):
            title = line[2:].strip()
            break

    return {
        "text": text,
        "metadata": {
            "source": path.name,
            "format": "markdown",
            "title": title,           # "MLX Guide" 등
        }
    }
```

**왜 마크다운 문법을 제거하지 않는가?**

`#` (제목), `**` (강조) 같은 기호는 LLM이 문서의 구조를 이해하는 데 도움이 됩니다. 제목이 있으면 "이 섹션의 주제가 뭔지" 알 수 있기 때문입니다.

## PDF 파일 (.pdf) 로딩

PDF는 가장 복잡합니다. PDF는 원래 **인쇄**를 위한 형식이지 텍스트 추출을 위한 형식이 아닙니다.

### PDF의 문제점

```
PDF 파일 내부:
┌──────────────────────┐
│ 제목 (큰 글씨)        │
│ 본문 텍스트           │  ← 이것만 필요
│ [이미지]              │  ← 불필요
│ 각주 (작은 글씨)      │  ← 섞이면 혼란
│ 페이지 번호           │  ← 불필요
└──────────────────────┘

일반 PDF 파서로 읽으면:
"제목본문 텍스트이미지 설명각주페이지 번호"
→ 줄바꿈, 공백이 엉망이 됩니다
```

### pymupdf4llm으로 깔끔하게 변환

`pymupdf4llm`은 PDF의 레이아웃을 분석해서 **마크다운 형태**로 깔끔하게 변환합니다.

```python
import pymupdf4llm

def load_pdf(file_path: str) -> dict:
    """Load a PDF file using pymupdf4llm.

    Converts PDF layout to clean markdown text.
    Handles tables, headings, and multi-column layouts.
    """
    path = Path(file_path)
    md_text = pymupdf4llm.to_markdown(str(path))

    return {
        "text": md_text,
        "metadata": {
            "source": path.name,
            "format": "pdf",
        }
    }
```

```python
# 사용 예
doc = load_pdf("docs/report.pdf")
print(doc["text"][:300])
# 깔끔한 마크다운 형태로 출력됩니다:
# # Quarterly Report
#
# ## Revenue
#
# | Quarter | Amount |
# |---------|--------|
# | Q1      | $1.2M  |
# | Q2      | $1.5M  |
```

## 통합 로더: 파일 확장자로 자동 선택

파일 확장자를 보고 적절한 로더를 자동으로 선택하는 함수입니다:

```python
def load_document(file_path: str) -> dict:
    """Load a document based on its file extension.

    Supported formats: .txt, .md, .pdf
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    loaders = {
        ".txt": load_txt,
        ".md": load_markdown,
        ".pdf": load_pdf,
    }

    loader = loaders.get(ext)
    if loader is None:
        raise ValueError(
            f"Unsupported format: '{ext}'\n"
            f"Supported formats: {list(loaders.keys())}"
        )

    return loader(file_path)
```

이 함수 하나로 어떤 형식이든 같은 방식으로 로딩할 수 있습니다:

```python
doc1 = load_document("docs/guide.md")    # 마크다운 로더 자동 선택
doc2 = load_document("docs/report.pdf")  # PDF 로더 자동 선택
doc3 = load_document("docs/notes.txt")   # 텍스트 로더 자동 선택
```

## 디렉토리 전체 로딩

`docs/` 폴더에 있는 모든 문서를 한 번에 로딩합니다:

```python
def load_directory(dir_path: str) -> list[dict]:
    """Load all supported documents from a directory.

    Scans the directory for .txt, .md, .pdf files and loads them all.
    Returns a list of document dictionaries.
    """
    path = Path(dir_path)
    supported = {".txt", ".md", ".pdf"}

    documents = []
    for file_path in sorted(path.iterdir()):
        if file_path.suffix.lower() in supported and file_path.is_file():
            doc = load_document(str(file_path))
            char_count = len(doc["text"])
            documents.append(doc)
            print(f"  Loaded: {file_path.name} ({char_count:,} chars)")

    print(f"\n  Total: {len(documents)} documents loaded")
    return documents
```

```python
# 사용 예
docs = load_directory("docs/")
#   Loaded: guide.md (3,420 chars)
#   Loaded: paper.pdf (12,500 chars)
#   Loaded: notes.txt (890 chars)
#
#   Total: 3 documents loaded
```

## 메타데이터가 중요한 이유

메타데이터 없이 텍스트만 저장하면 어떻게 될까요?

```
RAG 답변: "코드 리뷰는 최소 2명이 승인해야 합니다."

메타데이터 없음: 이 정보가 어디서 왔는지 알 수 없음
메타데이터 있음: (출처: team-rules.md, chunk 3) ← 신뢰할 수 있음!
```

메타데이터는 **처음부터** 설계해야 합니다. 나중에 "출처를 추가하고 싶다"고 해도, 이미 저장된 벡터에는 메타데이터가 없기 때문에 **전체를 다시 인덱싱**해야 합니다.

> **교훈**: 인덱싱은 다시 하면 되지만, 설계는 처음에 제대로 해야 합니다.

## 샘플 문서 만들기

튜토리얼을 따라하려면 검색할 문서가 필요합니다. `docs/` 폴더에 다음과 같은 샘플 문서를 만들어 둡니다:

```bash
mkdir -p docs
```

실제 프로젝트에서는 샘플 문서가 포함될 예정이지만, 지금은 간단한 텍스트 파일을 만들어 테스트해도 됩니다:

```python
# 샘플 문서 생성 예시
sample = """# Apple MLX Framework

## 개요

MLX는 Apple이 Apple Silicon을 위해 만든 머신러닝 프레임워크입니다.
NumPy와 유사한 API를 제공하면서, Apple Silicon의 통합 메모리를 활용합니다.

## 핵심 특징

### 통합 메모리 (Unified Memory)

Apple Silicon은 CPU와 GPU가 같은 메모리를 공유합니다.
일반적인 시스템에서는 CPU 메모리와 GPU 메모리 사이에 데이터를 복사해야 하지만,
Apple Silicon에서는 이 복사가 필요 없습니다.

이는 특히 대규모 모델을 실행할 때 큰 이점을 제공합니다.
128GB 통합 메모리를 가진 Mac Studio에서는 70B 파라미터 모델도 실행할 수 있습니다.

### Lazy Evaluation

MLX는 계산을 즉시 수행하지 않고, 결과가 필요할 때까지 미룹니다.
이를 통해 불필요한 계산을 피하고, 메모리를 효율적으로 사용합니다.
"""

Path("docs/mlx-guide.md").write_text(sample, encoding="utf-8")
```

## 실습

```python
# 1. 샘플 문서 생성 (위의 코드 실행)

# 2. 문서 로딩
docs = load_directory("docs/")

# 3. 각 문서 내용 확인
for doc in docs:
    print(f"\n{'=' * 50}")
    print(f"Source: {doc['metadata']['source']}")
    print(f"Format: {doc['metadata']['format']}")
    print(f"Length: {len(doc['text']):,} characters")
    print(f"\nFirst 200 chars:")
    print(doc['text'][:200])
```

## 핵심 정리

1. 문서 로딩은 파일을 열어서 **깨끗한 텍스트**를 꺼내는 과정이다
2. `.txt`는 그대로, `.md`는 구조 유지, `.pdf`는 레이아웃 변환이 필요하다
3. **메타데이터**(파일명, 형식)를 반드시 함께 저장해야 나중에 출처를 표시할 수 있다
4. 파일 확장자로 로더를 자동 선택하면 어떤 형식이든 같은 방식으로 처리할 수 있다

---

이전: [03. LLM 만나기](03-meet-your-llm.md) | 다음: [05. 텍스트 자르기](05-chunking.md)
