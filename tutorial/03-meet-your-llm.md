# 03. LLM 만나기

## 왜 LLM부터 시작하나?

RAG를 만들기 전에, 먼저 LLM이 **무엇을 잘하고 무엇을 못하는지** 직접 체험해봐야 합니다.

LLM의 한계를 직접 느껴봐야, 나머지 RAG 단계들이 **왜 필요한지** 자연스럽게 이해됩니다.

## LLM이란?

LLM(Large Language Model)은 **다음 단어를 예측하는 모델**입니다.

```
입력: "오늘 날씨가"
모델: "좋습니다" (확률 35%)
      "맑습니다" (확률 28%)
      "흐립니다" (확률 15%)
      ...
```

수조 개의 텍스트를 읽고 "이 단어 다음에 어떤 단어가 올 확률이 높은가"를 학습한 것입니다. 놀라운 것은, 이 단순한 원리로 질문에 답하고, 글을 쓰고, 코드를 작성할 수 있다는 점입니다.

> **중요한 오해**: LLM은 "이해"하거나 "알고" 있는 것이 아닙니다. 학습 데이터의 패턴을 바탕으로 그럴듯한 다음 단어를 생성하는 것입니다. 이 차이가 환각(hallucination)의 원인입니다.

## 첫 번째 대화

Jupyter 노트북을 열고, 커널을 "RAG Tutorial"로 선택한 뒤 실행합니다.

### 모델 로딩

```python
from mlx_lm import load, generate
import mlx.core as mx

# 모델 로딩 (첫 실행 시 약 1-2분 소요)
model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")

print(f"Model loaded!")
print(f"GPU memory used: {mx.metal.get_active_memory() / 1e9:.1f} GB")
```

### 간단한 질문

```python
# LLM에게 질문하는 함수
def ask(question, system_prompt="You are a helpful assistant. Answer in Korean."):
    """Ask a question to the LLM."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=500,
        temp=0.7,
    )

    # Clean up GPU memory
    mx.metal.clear_cache()

    return response
```

```python
# 첫 번째 질문
answer = ask("RAG란 무엇인가요? 간단히 설명해주세요.")
print(answer)
```

LLM이 RAG에 대해 잘 설명할 것입니다. 이건 학습 데이터에 포함된 일반 지식이기 때문입니다.

## LLM의 한계 체험하기

이제 LLM이 **못하는 것**을 직접 확인합시다.

### 실험 1: 내 문서에 대한 질문

아래와 같은 내용의 문서가 있다고 상상해보세요:

```
# 우리 팀 개발 규칙
- 코드 리뷰는 최소 2명이 승인해야 머지 가능
- 브랜치 이름은 feature/JIRA-번호-설명 형식
- 배포는 매주 화요일 오후 3시
- 핫픽스는 팀장 승인 후 즉시 배포 가능
```

이 문서를 LLM에게 보여준 적이 없습니다. 질문해봅시다:

```python
answer = ask("우리 팀의 코드 리뷰 규칙은 어떻게 되나요?")
print(answer)
```

**예상 결과**: LLM은 **일반적인** 코드 리뷰 모범 사례를 답변합니다. 우리 팀의 규칙("최소 2명 승인")은 모릅니다.

### 실험 2: 최신 정보

```python
answer = ask("2026년 2월에 출시된 MLX의 새로운 기능은 무엇인가요?")
print(answer)
```

**예상 결과**: LLM은 학습 데이터에 없는 정보이므로, 오래된 정보를 답하거나 "잘 모르겠습니다"라고 할 것입니다.

### 실험 3: 환각 (Hallucination)

```python
answer = ask("김철수 교수의 2024년 양자 컴퓨팅 논문의 핵심 결론은?")
print(answer)
```

**예상 결과**: LLM은 존재하지 않는 논문에 대해 **그럴듯한 답변을 지어냅니다**. 자신감 있게 틀린 정보를 말합니다. 이것이 환각(hallucination)입니다.

## RAG가 이 문제를 어떻게 해결하는가

위 실험의 결과를 정리하면:

| 질문 유형 | LLM만 | LLM + RAG |
|----------|-------|-----------|
| 일반 지식 | ✓ 잘 답변 | ✓ 잘 답변 |
| 내 문서 내용 | ✗ 모름, 일반론 | ✓ 문서 검색 후 정확히 답변 |
| 최신 정보 | ✗ 오래된 정보 | ✓ 최신 문서가 있으면 답변 |
| 환각 위험 | ✗ 높음 | ✓ 낮음 (문서에 없으면 "없다"고 답) |

RAG는 질문 전에 관련 문서를 찾아서 LLM에게 함께 보내줍니다. LLM은 자기 기억 대신 **주어진 문서를 근거로** 답변합니다.

## MLX 메모리 관리 — 중요!

Apple Silicon에서 LLM을 돌릴 때 반드시 알아야 할 것이 있습니다.

### 캐시를 정리하지 않으면

```python
# 이 코드를 Jupyter에서 실행하며 메모리를 관찰하세요

for i in range(5):
    answer = ask(f"숫자 {i}에 대해 재미있는 사실을 알려주세요.")
    memory_gb = mx.metal.get_active_memory() / 1e9
    print(f"질문 {i}: memory = {memory_gb:.2f} GB")
```

메모리가 계속 증가하는 것을 볼 수 있습니다! 이것은 MLX가 KV cache를 저장하기 때문입니다.

### 해결: 매번 캐시 정리

위의 `ask()` 함수에서 이미 `mx.metal.clear_cache()`를 호출하고 있습니다. 이것이 **필수**입니다.

```python
# ask() 함수 안에서:
response = generate(model, tokenizer, prompt=prompt, max_tokens=500, temp=0.7)
mx.metal.clear_cache()  # ← 이 줄이 없으면 메모리가 계속 증가합니다!
```

> **규칙**: `generate()` 호출 후에는 반드시 `mx.metal.clear_cache()`를 호출하세요. 특히 RAG에서는 매번 다른 컨텍스트를 사용하므로, 이전 캐시가 재사용되지 않고 메모리만 낭비합니다.

### Metal 메모리 한도

128GB Mac에서 GPU가 사용할 수 있는 메모리는 약 **96GB** (75%)입니다.

```python
print(f"Memory limit: {mx.metal.memory_limit() / 1e9:.1f} GB")
# 출력: Memory limit: 96.0 GB
```

이 한도를 넘으면 에러가 나는 게 아니라, **조용히 CPU로 전환**됩니다. 속도가 10~50배 느려지지만 경고가 없습니다. 메모리를 주시하는 습관을 들이세요.

## 생성 파라미터 이해하기

`generate()` 함수의 주요 파라미터입니다:

### temperature (temp)

모델의 **창의성**을 조절합니다.

```python
# temp=0.0 — 항상 같은 답변 (결정적)
answer = ask("1+1은?", temp=0.0)

# temp=0.7 — 적당히 다양한 답변 (기본값)
answer = ask("시 한 편 써줘", temp=0.7)

# temp=1.0 — 매우 다양한 답변 (창의적, 때로 엉뚱)
answer = ask("시 한 편 써줘", temp=1.0)
```

| temp | 성격 | 적합한 용도 |
|------|------|-----------|
| 0.0 | 결정적, 일관됨 | 사실 기반 Q&A, 코드 |
| 0.3-0.5 | 약간의 변화 | RAG 답변 (사실 + 자연스러움) |
| 0.7 | 균형 | 일반 대화 |
| 1.0+ | 창의적, 예측 불가 | 스토리텔링, 브레인스토밍 |

> **RAG에서는 temp=0.3 정도를 추천**합니다. 문서의 정보를 충실히 반영하되, 자연스러운 문장을 만들어야 하기 때문입니다.

### max_tokens

생성할 최대 토큰(단어 조각) 수입니다.

```python
# 짧은 답변
answer = ask("MLX란?", max_tokens=50)   # ~2-3문장

# 긴 답변
answer = ask("MLX란?", max_tokens=500)  # ~1-2 문단
```

> **토큰이란?** 단어를 모델이 처리하는 단위로 쪼갠 것입니다. 영어에서는 대략 1단어 = 1토큰이지만, 한국어에서는 1글자가 1-3토큰이 될 수 있습니다. 그래서 한국어는 같은 내용이라도 더 많은 토큰이 필요합니다.

## 실습: LLM과 대화하기

아래 질문들을 하나씩 LLM에게 물어보세요. 어떤 질문에 잘 답하고, 어떤 질문에 못 답하는지 직접 느껴보세요.

```python
# 잘 답하는 질문들 (일반 지식)
ask("Python에서 리스트와 튜플의 차이는?")
ask("뉴턴의 운동 법칙을 설명해줘")
ask("김치찌개 레시피를 알려줘")

# 못 답하는 질문들 (개인/내부/최신 정보)
ask("내 프로젝트의 데이터베이스 스키마는?")
ask("우리 회사 연봉 테이블은?")
ask("어제 내가 쓴 메모 내용은?")
```

이 격차가 RAG의 존재 이유입니다. 다음 튜토리얼부터 이 격차를 메우는 과정을 시작합니다.

## 핵심 정리

1. LLM은 **다음 단어를 예측**하는 모델이다
2. 학습 데이터에 있는 일반 지식은 잘 답하지만, **내 문서/최신 정보는 모른다**
3. 모르는 것도 **그럴듯하게 지어내는** 것이 위험 (환각)
4. `mx.metal.clear_cache()`를 매번 호출해야 메모리가 안정적이다
5. RAG는 이 한계를 극복하기 위해 **관련 문서를 찾아서 LLM에게 보여주는** 방법이다

---

이전: [02. 환경 설정](02-environment-setup.md) | 다음: [04. 문서 준비](04-document-loading.md)
