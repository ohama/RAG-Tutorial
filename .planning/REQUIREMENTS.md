# Requirements: Local RAG with Qwen3.5 MLX

**Defined:** 2026-03-04
**Core Value:** RAG가 LLM 답변 품질에 어떤 차이를 만드는지 직접 체험하고 이해할 수 있는 실행 가능한 학습 예제

## v1 Requirements

### Environment & Model

- [ ] **ENV-01**: Qwen3.5 MLX 모델을 로컬에서 로드하고 텍스트 생성
- [ ] **ENV-02**: MLX 메모리 관리 — mx.metal.clear_cache() 등 Apple Silicon 메모리 최적화 적용
- [ ] **ENV-03**: MLX/Apple Silicon 성능 설명 마크다운 셀 포함 (통합 메모리, GPU 가속 등)

### Document Pipeline

- [ ] **DOC-01**: 로컬 문서 로딩 — Markdown, PDF, TXT 3가지 형식 지원
- [ ] **DOC-02**: 고정 크기 청킹 + 오버랩 — chunk_size, overlap 파라미터 노출
- [ ] **DOC-03**: 청크에 원본 파일명/위치 메타데이터 보존 (인용용)
- [ ] **DOC-04**: 바로 실행 가능한 3-5개 샘플 문서 포함

### Retrieval & Generation

- [ ] **RET-01**: BAAI/bge-m3 로컬 임베딩 모델로 벡터 생성
- [ ] **RET-02**: ChromaDB에 임베딩 저장 및 유사도 검색 (로컬 영구 저장)
- [ ] **RET-03**: 검색된 컨텍스트가 프롬프트에 주입되는 과정 시각화
- [ ] **RET-04**: 답변에 출처 파일명/청크 정보 인용 표시

### Comparison & Tutorial

- [ ] **TUT-01**: RAG on/off 동일 질문 비교 Jupyter 노트북
- [ ] **TUT-02**: 노트북 설명은 한국어, 코드/주석은 영어
- [ ] **TUT-03**: 단계별 튜토리얼 문서 (tutorial/)

## v2 Requirements

### Advanced Retrieval

- **ADV-01**: 청킹 전략 비교 노트북 (fixed vs recursive vs semantic)
- **ADV-02**: 임베딩 유사도 점수 시각화 (matplotlib 차트)
- **ADV-03**: 검색 실패 사례 예시 (RAG가 언제 깨지는지)
- **ADV-04**: 사용자 문서 추가 지원 (docs/ 디렉토리에 파일 추가 → 자동 인식)

### Evaluation

- **EVAL-01**: RAGAS 스타일 정성 평가 노트북
- **EVAL-02**: 하이브리드 검색 (BM25 + dense vector) 데모

## Out of Scope

| Feature | Reason |
|---------|--------|
| Web UI (Gradio, Streamlit) | RAG 파이프라인을 추상화로 숨김 — 학습 목적에 부적합 |
| LangChain / LlamaIndex | 모든 단계를 숨겨서 RAG를 가르치는 목적에 반함 |
| Cloud API 임베딩 (OpenAI, Cohere) | "완전 로컬" 제약 위반 |
| 멀티모달 (이미지, 오디오) | 텍스트 RAG에 집중 |
| 멀티턴 대화 메모리 | 상태 관리가 RAG 학습에서 주의를 분산 |
| Docker / 컨테이너화 | 로컬 교육용 프로젝트에 불필요한 복잡성 |
| 파인튜닝 | 기본 모델 + RAG로 충분한 효과 시연 |
| 프로덕션 에러 핸들링 | 학습 예제의 가독성 저해 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ENV-01 | - | Pending |
| ENV-02 | - | Pending |
| ENV-03 | - | Pending |
| DOC-01 | - | Pending |
| DOC-02 | - | Pending |
| DOC-03 | - | Pending |
| DOC-04 | - | Pending |
| RET-01 | - | Pending |
| RET-02 | - | Pending |
| RET-03 | - | Pending |
| RET-04 | - | Pending |
| TUT-01 | - | Pending |
| TUT-02 | - | Pending |
| TUT-03 | - | Pending |

**Coverage:**
- v1 requirements: 14 total
- Mapped to phases: 0
- Unmapped: 14 ⚠️

---
*Requirements defined: 2026-03-04*
*Last updated: 2026-03-04 after initial definition*
