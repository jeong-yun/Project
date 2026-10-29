# 26-Agent-Office 구현 계획

> 직급별(사원 → 대리 → 차장 → 부장 → 팀장) AI 에이전트가 답변을 단계적으로 검토하고,
> 사람이 최종안을 채택하면 그 결과가 각 에이전트에게 피드백되는 "AI 사무실" 프레임워크.
> 현재 단계는 **도메인에 무관한 기본 바탕(general 팀)** 을 만들고, 이후 팀(주식 검토, 보안 데이터 생성, 여행 계획서 등)을 플러그인처럼 추가한다.

작성일: 2026-10-19

---

## 1. 목표 요약

| # | 요구사항 | 설계 방향 |
|---|---|---|
| 1 | 사원/대리/차장/부장/팀장 에이전트 | 직급을 코드가 아닌 **설정(YAML)** 으로 정의 → 직급 추가/삭제 가능 |
| 2 | 클릭한 에이전트부터 상위 직급이 순차 검토 → 사람 채택 → 피드백 | **LangGraph** 워크플로우 + Human-in-the-loop(interrupt) + 피드백 메모리 |
| 3 | 다른 PC에서도 실행 가능한 README | 요구 사양, 설치, .env 설정, 실행, 트러블슈팅 |
| 4 | today_worked.md | 날짜별 진행 로그 + 누적 키워드·성과 요약 |
| 5 | 직급이 낮을수록 낮은 모델 | `config/models.yaml` 로 직급↔모델 매핑, API 키는 `.env` |
| 6 | 분야별 전문가 팀은 추후 | `teams/<team_id>/team.yaml` 로 팀 추가 (코드 수정 없이) |
| 7 | 에이전트 이미지 교체 가능 | `assets/avatars/` 파일 교체 또는 UI 업로드, 팀별 오버라이드 |
| 8 | 확장성 | 설정 기반, Provider 추상화, Tool 레지스트리, 실행 이력 DB |

---

## 2. 기술 스택 (기존 레포 관례에 맞춤)

- **Python 3.11+** (현재 PC: 3.11.9)
- **UI**: Streamlit (19-Streamlit, 23-My_Project와 동일한 방식)
- **오케스트레이션**: LangGraph (검토 체인 + 사람 승인 단계 interrupt + 체크포인트)
- **LLM 연동**: LangChain 통합 패키지
  - `langchain-openai` (ChatGPT), `langchain-anthropic` (Claude), `langchain-google-genai` (Gemini), Upstage Solar 는 OpenAI 호환 API 로 `langchain-openai` 사용
- **저장소**: SQLite (실행 이력·피드백), 파일 시스템 (이미지·설정)
- **추적**: LangSmith (`langchain_teddynote.logging`, 선택)
- 폴더 전용 `requirements.txt` 를 별도로 둔다 (루트 requirements는 패키지가 많고 버전이 오래됨).

---

## 3. 폴더 구조 (안)

```
26-Agent-Office/
├── app.py                      # Streamlit 메인 (사무실 화면: 에이전트 카드)
├── pages/
│   ├── 1_history.py            # 실행 이력 / 채택 결과 조회
│   ├── 2_feedback.py           # 에이전트별 누적 피드백(교훈) 조회·편집
│   └── 3_settings.py           # 모델 매핑, 아바타 이미지 교체, 팀 선택
├── core/
│   ├── config.py               # YAML 로더 + 검증 (pydantic)
│   ├── llm_factory.py          # provider/model → ChatModel 생성, 키 없을 때 fallback
│   ├── agent.py                # RankAgent: 답변(answer) / 검토(review) 메서드
│   ├── schemas.py              # ReviewResult 등 구조화 출력 스키마
│   ├── workflow.py             # LangGraph: ask → review* → human_decision → feedback
│   ├── feedback.py             # 채택 결과 → 에이전트별 교훈 생성·저장·프롬프트 주입
│   ├── storage.py              # SQLite (runs, steps, feedback)
│   └── tools/                  # 팀별 툴 레지스트리 (추후: 주가 조회, 웹검색 등)
│       └── __init__.py
├── config/
│   ├── ranks.yaml              # 직급 정의 (순서, 이름, 역할 설명, 기본 아바타)
│   ├── models.yaml             # 직급 → provider/model/temperature + fallback
│   └── settings.yaml           # 검토 정책(전체 검토/승인 시 조기 종료), 피드백 개수 등
├── teams/
│   ├── general/                # 기본 바탕 팀 (현재 단계)
│   │   ├── team.yaml           # 팀 설명, 직급별 페르소나, 검토 기준, 출력 형식
│   │   └── avatars/            # (선택) 팀 전용 아바타 → 없으면 공용 사용
│   └── _template/              # 새 팀 만들 때 복사하는 템플릿
│       └── team.yaml
├── assets/
│   └── avatars/                # 공용 아바타: staff.png, assistant_manager.png ...
├── data/                       # (gitignore) runs.db, 업로드 이미지 등
├── tests/                      # 설정 검증, 워크플로우(모의 LLM) 테스트
├── .env.example                # 필요한 API 키 목록 (값 없음)
├── requirements.txt
├── README.md                   # 다른 PC 설치·실행 가이드
├── today_worked.md             # 날짜별 진행 로그
└── PLAN.md                     # 이 문서
```

---

## 4. 핵심 설계

### 4.1 직급 정의 — `config/ranks.yaml`

```yaml
ranks:            # 위에서 아래로 = 낮은 직급 → 높은 직급 (검토 순서)
  - id: staff
    name: 사원
    role: 초안 작성. 빠르고 폭넓게 아이디어를 낸다.
    avatar: assets/avatars/staff.png
  - id: assistant_manager
    name: 대리
    role: 사실관계·누락 검토 및 보완
  - id: deputy_gm
    name: 차장
    role: 논리 구조·리스크 검토
  - id: gm
    name: 부장
    role: 실행 가능성·우선순위 검토
  - id: team_lead
    name: 팀장
    role: 최종 품질 판단, 결론 정리
```
- 직급 추가(예: 과장)는 YAML에 한 줄 추가 + models.yaml 매핑만 하면 됨.

### 4.2 모델 매핑 — `config/models.yaml` (직급이 낮을수록 저가 모델)

| 직급 | 1순위 (예시) | fallback (예시) |
|---|---|---|
| 사원 | Upstage `solar-mini` | Gemini Flash-Lite |
| 대리 | Gemini Flash | GPT mini 계열 |
| 차장 | GPT 중급 (예: gpt-4.1) | Upstage `solar-pro` |
| 부장 | Claude Sonnet | Gemini Pro |
| 팀장 | Claude Opus | GPT 최상위 |

- 모델명은 각 사가 자주 바꾸므로 **코드에 하드코딩하지 않고 YAML에만** 둔다. 구현 시점에 각 사 최신 모델명 확인.
- `llm_factory` 는 해당 provider의 API 키가 `.env` 에 없으면 fallback 으로 자동 전환하고 UI에 표시.
- 현재 `.env` 에는 OpenAI / Google / Anthropic 키만 있음 → **Upstage 키 추가 필요** (없으면 fallback 동작).
- 직급별 토큰/비용을 기록해 "직급별 모델 차등"의 효과를 수치로 보여줄 수 있게 한다.

### 4.3 검토 워크플로우 — LangGraph

```
[사용자가 '대리' 카드 클릭 + 질문 입력]
        │
        ▼
  answer(대리) ──► review(차장) ──► review(부장) ──► review(팀장)
                                                       │
                                                       ▼
                                          human_decision (interrupt)
                                          - 단계별 답변 비교 화면
                                          - 하나 채택 / 직접 수정 / 전체 반려
                                          - 점수(1~5) + 코멘트
                                                       │
                                                       ▼
                                   feedback → 참여한 모든 에이전트에 교훈 저장
```

- **시작 직급 = 클릭한 에이전트**, 검토자 = 그보다 상위 직급 전부 (팀장 클릭 시 검토 없이 바로 사람 판단).
- 각 검토자는 구조화 출력(Pydantic)을 반환:
  ```python
  class ReviewResult(BaseModel):
      verdict: Literal["approve", "revise"]
      score: int            # 1~10
      issues: list[str]     # 지적 사항
      revised_answer: str   # 보완된 답변 (approve면 이전 답변 유지)
      comment: str          # 하위 직급에게 남기는 한 줄 코멘트
  ```
- 검토자에게는 원 질문 + 직전 답변 + 이전 검토 이력 요약을 전달.
- 정책 옵션(`settings.yaml`): `full_chain`(항상 끝까지) / `early_stop`(상위자가 approve 연속 시 종료).
- UI에는 단계가 끝날 때마다 실시간 표시(스트리밍), 각 단계의 모델명·소요시간·토큰 표시.
- LangGraph 체크포인터(SQLite)로 사람 승인 대기 중 브라우저를 닫아도 이어서 진행 가능.

### 4.4 사람 채택 → 피드백 루프

1. 사람이 최종안 채택 (어느 직급 답변인지 + 수정본 + 점수 + 코멘트).
2. `feedback.py` 가 에이전트별 교훈을 생성 (저가 모델 사용):
   - 예: 사원 → "주식 추천 시 근거 데이터 출처와 리스크를 함께 제시할 것 (차장 검토에서 보완됨, 사람이 차장안 채택)"
   - 채택된 직급은 긍정 피드백, 그 이후 검토자가 과하게 바꿨다면 그것도 피드백.
3. SQLite `feedback` 테이블에 `team_id, rank_id, lesson, score, created_at` 저장.
4. 다음 실행 시 해당 팀·직급의 **최근/중요 교훈 N개를 시스템 프롬프트에 주입** (간단한 장기 기억).
5. `2_feedback.py` 페이지에서 교훈 열람·수정·삭제 가능 (잘못된 학습 방지).
6. (확장) 교훈이 쌓이면 벡터DB 검색으로 질문과 관련된 교훈만 주입.

### 4.5 팀(전문 분야) 확장 — `teams/<team_id>/team.yaml`

```yaml
id: general
name: 기본 팀
description: 범용 질의응답 기본 팀
personas:                     # 직급별 전문가 페르소나 (없으면 ranks.yaml 기본값)
  staff: "당신은 성실한 신입 사원입니다..."
  team_lead: "당신은 20년 경력의 팀장입니다..."
review_criteria:              # 검토자가 공통으로 확인할 체크리스트
  - 사실 정확성
  - 누락된 관점
  - 실행 가능성
output_format: markdown
tools: []                     # 추후: [stock_price, web_search] 등
model_overrides: {}           # 팀별로 직급 모델 변경 가능
disclaimer: ""                # 예: 주식팀 → "투자 권유가 아님"
```
- 추후 팀 예시: `stock_review`(주가/재무 조회 툴 + 투자 면책 문구), `security_data`(보안 로그/데이터 생성), `travel_plan`(일정표 출력 형식).
- 새 팀 추가 = `_template` 복사 → yaml 수정 → (선택) 툴 등록. 앱 재시작 시 자동 인식.

### 4.6 에이전트 이미지 교체

- 우선순위: `teams/<team>/avatars/<rank>.png` → `assets/avatars/<rank>.png` → 기본 이모지(👤).
- `3_settings.py` 에서 이미지 업로드 시 해당 경로에 저장 (png/jpg/webp, 크기 자동 리사이즈).
- 사용자가 직접 파일만 교체해도 반영 (README에 파일명 규칙 명시).
- 초기에는 저작권 문제 없는 단색/이니셜 placeholder 이미지 생성 스크립트 제공.

### 4.7 기타 확장성 고려

- **Provider 추상화**: 새 LLM(예: Ollama 로컬, Azure, 25-FreeLLMAPI 무료 모델)도 factory에 한 곳만 추가.
- **이력 저장**: 모든 실행(질문, 단계별 답변·검토, 채택 결과, 토큰/비용)을 SQLite에 저장 → 이력 페이지, CSV/Markdown 내보내기.
- **비용 가드**: 실행 전 예상 단계 수 표시, 일일 토큰 상한 설정(옵션).
- **보안**: `.env` 는 git 제외, `.env.example` 만 커밋. 업로드 이미지 확장자/크기 검증.
- **테스트**: 모의 LLM(FakeListChatModel)으로 API 키 없이 워크플로우 테스트.
- **향후**: 팀 간 협업(여러 팀 병렬 실행), FastAPI 백엔드 분리, Docker 배포.

---

## 5. README.md 포함 내용 (다른 PC용)

- 프로젝트 소개 + 동작 흐름 그림
- **요구 사양**
  - OS: Windows 10/11, macOS, Linux
  - Python 3.11 이상
  - RAM 8GB 이상 권장 (로컬 모델 미사용, 모든 추론은 API 호출 → GPU 불필요)
  - 디스크 1GB 내외 (가상환경 포함)
  - 인터넷 연결 필수 (각 LLM API 호출)
- **필요 API 키**: OPENAI / ANTHROPIC / GOOGLE / UPSTAGE (+ 선택 LANGSMITH) 와 발급처 링크, 일부만 있어도 fallback 동작 설명
- 설치: `python -m venv .venv` → 활성화(Windows/macOS 각각) → `pip install -r requirements.txt`
- `.env.example` → `.env` 복사 후 키 입력
- 실행: `streamlit run app.py`
- 아바타 이미지 교체 방법, 새 팀 추가 방법, 모델 변경 방법
- 트러블슈팅 (키 누락, 포트 충돌, 한글 경로 이슈 등)

## 6. today_worked.md 형식

```markdown
## 2026-10-19
- 작업: 26-Agent-Office 기획 및 아키텍처 설계 (PLAN.md)
- 기술: LangGraph, Multi-LLM(OpenAI/Claude/Gemini/Upstage), Streamlit
- 성과/의미: 직급형 다단계 검토 + Human-in-the-loop 피드백 구조 설계
- 다음 할 일: 프로젝트 골격 생성, LLM factory 구현
```
- 하단에 **"누적 요약"** 섹션 유지 (핵심 역량 키워드, 정량 성과: 예) 직급별 모델 차등으로 비용 X% 절감, 검토 단계별 품질 점수 향상 등).

---

## 7. 단계별 진행 (마일스톤)

| 단계 | 내용 | 완료 기준 |
|---|---|---|
| **P0. 골격** | 폴더 구조, requirements, .env.example, README 초안, today_worked.md, .gitignore | `streamlit run app.py` 빈 화면 실행 |
| **P1. 코어** | config 로더, llm_factory(+fallback), RankAgent 단일 답변 | 직급별 모델로 단일 질문 응답 성공 |
| **P2. 검토 체인** | LangGraph 워크플로우, ReviewResult 구조화 출력, 단계 스트리밍 | 사원 클릭 → 팀장까지 검토 결과 출력 |
| **P3. 사람 채택·피드백** | interrupt 기반 채택 화면, 교훈 생성/저장/프롬프트 주입 | 채택 후 다음 실행에 교훈 반영 확인 |
| **P4. UI 완성** | 에이전트 카드(이미지·클릭), 단계 타임라인, 비교 뷰, 이력/피드백/설정 페이지, 이미지 업로드 | 이미지 교체 즉시 반영 |
| **P5. 팀 확장 검증** | `_template` 및 샘플 팀 1개(예: stock_review, 툴 1개 포함) | yaml 추가만으로 새 팀 동작 |
| **P6. 마무리** | 테스트(모의 LLM), README 최종화, 다른 PC(또는 새 venv)에서 설치 검증 | 클린 환경 설치·실행 성공 |

각 단계 종료 시 today_worked.md 에 기록.

---

## 8. 결정이 필요한 사항 (권장안)

1. **UI**: Streamlit (권장, 기존 레포와 동일·빠름) vs React+FastAPI (화려하지만 개발량 큼)
2. **검토 방식**: 상위자가 *보완된 답변을 다시 작성* (권장) vs 코멘트만 남김
3. **검토 범위**: 항상 팀장까지 (권장, 기본) vs 상위자 승인 시 조기 종료 (옵션으로 제공)
4. **폴더명**: `26-Agent-Office` (변경 가능)
5. **Upstage API 키** 발급 여부 (없으면 사원/차장 fallback 모델 사용)

---

## 9. 변경 이력 (계획 대비)

- **2026-10-19 — 구조 변경: 일렬 직급 체인 → 조직도(Supervisor → Specialist 트리)**
  - 이유: 사용자가 supervisor(팀장)와 대화하고, supervisor 가 필요한 전문가에게 업무를 나누는 구조가 실제 팀 운영·확장(팀별 전문가 추가)에 더 맞음
  - 4.1 직급 정의는 그대로 '모델·기본 이미지 기준'으로 쓰고, 실제 에이전트는 team.yaml 의 `org` 트리로 정의
  - 4.3 검토 워크플로우: `brief → research → work(할 일 스택) → human_decision → feedback` 으로 대체 (조기 종료 정책 제거)
  - 추가: Desired-State 브리프, 팀 스킬, 적대적 리뷰어·비평가·자가 교정, 결정론적 하네스, 단계별 평가 하네스(evals/), 미결정 실행 자동 삭제
  - 모델: Claude 크레딧 부족으로 `disabled_providers: [anthropic]`, GPT-5.4/5.5 · Gemini 3.5/3.8 Flash 로 구성
- 최신 구조·사용법은 README.md 가 기준이다.
