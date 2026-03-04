# Local RAG with Qwen3.5 MLX

## What This Is

Mac Studio M4 Max (128GB)에서 Qwen3.5 MLX 모델을 사용해 로컬 문서에 대한 RAG(Retrieval-Augmented Generation) 시스템을 구축하는 학습용 프로젝트. 튜토리얼과 실제 동작하는 구현체를 모두 포함하며, RAG 유무에 따른 답변 품질 차이를 Jupyter 노트북으로 시각적으로 비교한다.

## Core Value

RAG가 LLM 답변 품질에 어떤 차이를 만드는지 직접 체험하고 이해할 수 있는 실행 가능한 학습 예제.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Qwen3.5 MLX 모델을 로컬에서 로드하고 텍스트 생성
- [ ] 로컬 문서(Markdown, PDF, TXT) 파싱 및 청킹
- [ ] 문서 임베딩 생성 및 벡터 DB 저장
- [ ] 유사도 검색으로 관련 문서 청크 검색
- [ ] 검색된 컨텍스트를 프롬프트에 주입하여 답변 생성 (RAG pipeline)
- [ ] RAG on/off 비교 Jupyter 노트북
- [ ] 데모용 샘플 문서 세트 포함
- [ ] 사용자 문서 추가 가능한 구조
- [ ] 단계별 한국어 튜토리얼 (tutorial/)

### Out of Scope

- 웹 UI — 학습 목적이므로 Jupyter 노트북으로 충분
- 멀티모달 (이미지, 오디오) — 텍스트 RAG에 집중
- 클라우드 배포 — 완전 로컬 환경
- 파인튜닝 — 기본 모델 + RAG로 충분한 효과 시연
- 프로덕션 수준 에러 핸들링 — 학습 예제에 불필요

## Context

- **하드웨어**: Mac Studio M4 Max, 16-core CPU, 40-core GPU, 128GB unified memory, 2TB SSD
- **LLM**: Qwen3.5 MLX 버전 (Apple Silicon 최적화)
- **프레임워크**: MLX (Apple의 ML 프레임워크)
- **언어**: Python
- **문서 타입**: Markdown, PDF, TXT (로컬 파일)
- **출력 형태**: Jupyter 노트북에서 RAG on/off 비교
- **튜토리얼 언어**: 설명은 한국어, 코드/주석은 영어

## Constraints

- **Hardware**: Mac Studio M4 Max 128GB — 모델 크기와 벡터 DB가 메모리에 적합해야 함
- **Runtime**: MLX 프레임워크 — Apple Silicon 전용, PyTorch/TensorFlow 아님
- **Offline**: 완전 로컬 실행 — 외부 API 호출 없음
- **Educational**: 코드가 읽기 쉽고 이해하기 좋아야 함 — 성능보다 명확성 우선

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Qwen3.5 MLX 사용 | 128GB에서 로컬 실행 가능한 고성능 모델, Apple Silicon 최적화 | — Pending |
| Jupyter 노트북으로 비교 | 단계별 실행과 시각적 비교에 최적 | — Pending |
| 임베딩 모델 미정 | 리서치 후 결정 (로컬 실행, MLX 호환성 고려) | — Pending |
| 벡터 DB 미정 | 리서치 후 결정 (가볍고 로컬 친화적인 것) | — Pending |

---
*Last updated: 2026-03-04 after initialization*
