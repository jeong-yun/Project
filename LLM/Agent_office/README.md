# 🏢 Agent Office

사용자는 **supervisor(팀장)** 에게 일을 맡기고, supervisor 가 질문을 보고 필요한 **전문가 에이전트**들에게 업무를 나눠 주는
조직도(트리) 구조의 멀티 에이전트 프레임워크입니다. 결과는 아래에서 위로 검토·통합되고, 사람이 최종안을 채택하면
참여한 모든 에이전트가 교훈을 받아 다음 업무에 반영합니다.

```
                👤 사용자  ⟷  [팀장] supervisor
                              /            \
                    [부장·리서치]        [부장·실행]
                         |                    |
                    [차장·분석]          [대리·실행안]
                     /       \                |
           [대리·팩트체크] [사원·초안]     [사원·사례]

  ① 요청 정리   질문 → '최종 상태 정의(목표·완료 기준·제약·형식)' 브리프
  ② 자료 조사   웹 검색 → 모든 에이전트가 근거 자료로 사용 (출처 [번호] 표시)
  ③ 위 → 아래  관리자가 필요한 하위 전문가를 골라 업무 분배
  ④ 맨 아래    실무자가 초안 작성 (+ 하네스: 코드·쿼리 결정론적 검증, 실패 시 재작성)
  ⑤ 아래 → 위  관리자가 적대적 리뷰어로 검토(1명)·통합(여러 명) (+ 비평가 채점, 불합격 시 자가 교정)
  ⑥ 👤 결정    사람이 채택(수정 가능)/반려 → 에이전트별 교훈 저장 → 다음 실행 프롬프트에 주입
```

### 팀

| 팀 | 업무 (화면의 📋 업무 선택) | 근거 자료 |
|---|---|---|
| 기본 팀 | 자유 질문 (예시 4종) | 웹 검색 |
| 📈 주식 팀 | ① 분야별 종목 추천 ② 주가 전망 ③ 보유 종목 전략(수량·매입가) ④ 포트폴리오 구성 | **시세 데이터**(Yahoo Finance: 현재가·수익률·이동평균·변동성·평가손익을 코드로 계산) + 최근 1개월 뉴스·리포트 |
| ✈️ 여행 팀 | 여행 계획서 (국내/해외, 출발·도착일, 인원, 예산, 스타일) | **여행 달력**(N박 M일·날짜별 요일을 코드로 계산) + 목적지 날씨·행사·교통 검색 |

> 진행 상황은 [today_worked.md](today_worked.md), 초기 계획은 [PLAN.md](PLAN.md) 참고.

---

## 1. 요구 사양

| 항목 | 요구 사항 |
|---|---|
| OS | Windows 10/11, macOS, Linux |
| Python | **3.11 이상** (개발 환경: 3.11.9) |
| CPU / RAM | 일반 PC 수준, RAM 8GB 이상 권장 |
| GPU | 불필요 (모든 추론은 API 호출) |
| 디스크 | 약 1GB (가상환경 포함) |
| 네트워크 | 인터넷 연결 필수 (LLM API 호출, 웹 검색) |
| 브라우저 | Chrome / Edge / Safari 최신 버전 |

## 2. 필요 API 키

| Provider | 환경 변수 | 발급처 | 현재 사용처 |
|---|---|---|---|
| Google (Gemini) | `GOOGLE_API_KEY` | https://aistudio.google.com/apikey | 사원·대리, 보조 작업(브리프·검색어·비평·교훈) |
| OpenAI (ChatGPT) | `OPENAI_API_KEY` | https://platform.openai.com/api-keys | 차장·부장·팀장 |
| Anthropic (Claude) | `ANTHROPIC_API_KEY` | https://console.anthropic.com/ | **현재 비활성** (크레딧 충전 후 사용) |
| Upstage (Solar) | `UPSTAGE_API_KEY` | https://console.upstage.ai/ | 대안 모델 (선택) |
| LangSmith (선택) | `LANGSMITH_API_KEY` | https://smith.langchain.com/ | 실행 추적 |

- **일부 키만 있어도 동작합니다.** 사용할 수 없는 provider 는 `config/models.yaml` 의 대안(fallback) 모델로 자동 대체됩니다.
- 키가 있어도 호출이 실패하면(크레딧 부족, 할당량 초과, 모델 단종 등) 다음 후보로 넘어가고, 타임라인의 "검증 기록"에 사유가 남습니다.
- **Claude 를 다시 쓰려면**: `config/models.yaml` 의 `disabled_providers: [anthropic]` 에서 anthropic 을 지우고, 부장·팀장 primary 를 Claude 모델로 바꾸면 됩니다.
- Gemini **Pro** 계열은 무료 등급 할당량이 0 입니다 (Flash 계열은 무료 사용 가능).
- 웹 검색은 DuckDuckGo(`ddgs` 패키지), 주가 데이터는 Yahoo Finance(`yfinance` 패키지)를 쓰므로 별도 키가 필요 없습니다.
- 비용: 기본 팀 전체(팀장부터)에 질문하면 LLM 호출 약 12~15회 (브리프 1 + 검색어 1 + 분배 2 + 초안 3 + 검토·통합 5 + 비평 1 + 교훈 1).

## 3. 설치

```bash
# 1) 이 폴더로 이동 (GitHub jeong-yun/Project 저장소 기준)
cd Project/LLM/Agent_office

# 2) 가상환경 생성
python -m venv .venv

# 3) 가상환경 활성화
#   Windows (PowerShell)
.venv\Scripts\Activate.ps1
#   macOS / Linux
source .venv/bin/activate

# 4) 패키지 설치
pip install -r requirements.txt

# 5) API 키 설정: .env.example 을 .env 로 복사 후 값 입력
#   Windows
copy .env.example .env
#   macOS / Linux
cp .env.example .env
```

> `.env` 는 이 폴더 안의 것을 우선 읽고, 없는 값은 바로 위 폴더의 `.env` 에서 보충합니다.
> **`.env` 는 절대 git 에 올리지 않습니다** — `.gitignore` 에 들어 있고, 저장소에는 값이 빈 `.env.example` 만 올립니다.

## 4. 실행

```bash
streamlit run app.py
```

브라우저에서 http://localhost:8501 이 열립니다. 포트를 바꾸려면 `streamlit run app.py --server.port 8600`.

## 5. 사용법 — 좋은 질문은 '과정'이 아니라 '최종 상태'를 정의한다

| | 예시 |
|---|---|
| ❌ 과정 지시 | "1) HBF 검색하고 2) 정리한 다음 3) 표로 만들어줘" |
| ✅ 최종 상태 정의 | **목표** 팀 회의에서 5분 안에 공유할 HBF 설명 자료 / **완료 기준** 정식 명칭·정의, HBM 과 비교 표, 사실마다 출처 / **제약** 비전공자용, A4 1장 / **형식** 요약 → 비교 표 → 핵심 3개 → 출처 |

- 화면의 **💡 질문 예시**를 고르면 질문과 최종 상태가 채워집니다 (개념 조사·의사결정·계획·코드).
- **🎯 원하는 최종 상태**를 비워 두면 보조 모델이 질문에서 목표·완료 기준을 추론해 채웁니다 (`🤖 빈 항목은 AI 가 정리`).
- **완료 기준**은 검토자·비평가의 채점 기준이 됩니다 → "정식 명칭과 출처를 제시한다"처럼 검증 가능한 문장으로 쓰세요.
- 과정(누가 무엇을 조사할지)은 supervisor 가 스스로 나눕니다.
- 결과를 그대로 받아들이지 말고, 타임라인의 **검토·비평 기록**(취약점·논리 비약)을 확인한 뒤 채택하세요.
- **문의 대상**에서 다른 칸을 고르면 그 에이전트가 supervisor 가 되어 그 아래 팀만 일합니다 (비용·시간 절약).
- 주식·여행 팀은 **📋 업무 선택**에서 업무를 고르고 입력칸만 채우면 됩니다. 목표·완료 기준은 업무 양식에 미리 정의돼 있어
  자동으로 '최종 상태' 브리프가 됩니다 (예: 주가 전망 → "상승·기본·하락 시나리오와 확률, 기준일 인용").
- 모든 에이전트는 **오늘 날짜**를 기준으로 '현재·최근'을 판단하고, 주식 팀 뉴스 검색은 최근 1개월로 제한됩니다.

## 6. 검증 장치

| 장치 | 내용 | 설정 |
|---|---|---|
| CoT 강제 | 검토·비평 출력에서 분석(analysis)을 점수보다 먼저 쓰게 강제 | `core/schemas.py` (필드 순서) |
| 퓨샷 루브릭 | 점수 기준 + 실제 고득점/저득점 예시(HBF≠HBM)로 채점 영점 조정 | `config/verification.yaml` `review_rubric` (팀별: team.yaml `rubric`) |
| 구조화 단계 분할 | 분배(Delegation) → 검토(ReviewResult) → 비평(CriticResult) 을 JSON 키로 분리 | `core/schemas.py` |
| 적대적 리뷰어 | 검토자가 결과를 믿지 않고 취약점·논리 오류·완료 기준 미충족부터 찾음 | `core/agent.py`, 스킬 `adversarial_review.md` |
| 비평가 (LLM-as-a-Judge) | 검토자의 '지적 → 수정' 중간 과정의 논리 비약만 따로 채점 | `config/verification.yaml` `critic` |
| 자가 교정 | 비평 불합격 시 "A 부분이 B 규칙에 어긋난다. 수정하라" 를 체인 중간에 넣어 재작성 | `critic.max_self_corrections` |
| 하네스 (RLVR 응용) | 코드·SQL·JSON 을 실제 인터프리터/샌드박스로 검증 → 실패 시 0점 + 에러 로그를 돌려줘 재작성 | team.yaml `harness` |
| 단계별 평가 하네스 | 검색·분배·초안·검토·최종 단계마다 유닛 테스트 → 어느 노드에서 실패했는지 추적 (PRM vs ORM) | `evals/` |

### 하네스 검증기 (team.yaml `harness.validators`)

| 이름 | 하는 일 |
|---|---|
| `python_syntax` | 답변 속 ```python 블록을 `compile()` 로 문법 검사 (실행하지 않음) |
| `sql_sandbox` | ```sql 블록을 메모리 SQLite 에 스키마(`options.sql_schema`)를 만들고 EXPLAIN 컴파일 + 선택적 시간 제한 실행 (`sql_execute`, `sql_timeout_sec`) |
| `json_keys` | ```json 블록 형식 + 단계 분할 필수 키(`options.required_keys`) 검사 |
| `required_terms` | 업무 필수 항목(예: 시나리오·손절·목표가·예산)이 답변에 있는지 검사 (`"A\|B"` 는 둘 중 하나) |
| `itinerary_days` | 여행 일정이 1일차~마지막 날까지 모든 날을 다루는지 검사 (일수는 여행 달력 툴이 계산해 전달) |

업무 양식(team.yaml `tasks`)마다 `validators`·`harness_options` 를 따로 지정할 수 있습니다 (예: 주가 전망은 `required_terms: [기준일, 시나리오, 상승, 하락, 확률]`).

새 검증기(예: ClickHouse 테스트 DB 실행)는 `core/harness/validators.py` 에 `Validator` 를 상속한 클래스를 만들고 `@register` 를 붙이면 됩니다.

### 단계별 평가 실행

```bash
python -m evals.run_eval evals/cases/general.yaml   # 기본 팀 (HBF 등)
python -m evals.run_eval evals/cases/stock.yaml     # 주식 팀 (삼성전자 전망·보유 전략)
python -m evals.run_eval evals/cases/travel.yaml    # 여행 팀 (오사카 3박 4일)
```

케이스는 `question`(자유 질문) 또는 `task` + `inputs`(업무 양식) 로 쓰고, 단계는 `tools · research · plan · draft · review · integrate · final` 중에서 검사합니다.

케이스 작성법은 `evals/cases/general.yaml`, 보안 로그 분석(페이로드 복원 → 시그니처 매칭 → 위험도) 예시는 `evals/cases/_security_template.yaml` 참고.
결과는 화면 표와 `data/evals/<시각>.json` 으로 저장됩니다.

## 7. 커스터마이징

### 조직도 바꾸기 (전문가 추가)
`teams/<팀>/team.yaml` 의 `org` 에서 `children` 에 칸을 추가합니다. `rank`(직급 → 모델·기본 이미지), `specialty`(전문 분야) 를 지정합니다.
`org` 를 지우면 `config/ranks.yaml` 순서대로 팀장 → 부장 → 차장 → 대리 → 사원 한 줄 조직도가 자동으로 만들어집니다.

### 팀 스킬 (반복 업무 지침)
`teams/<팀>/skills/*.md` 에 지침을 쓰면 그 팀의 모든 에이전트 시스템 프롬프트에 들어갑니다 (예: `answer_style.md`, `adversarial_review.md`).
스킬은 사람이 쓰는 고정 지침, 교훈은 사람의 결정에서 자동으로 쌓이는 학습입니다.

### 에이전트 이미지 바꾸기 (먼저 찾은 것 사용)
1. `teams/<팀>/avatars/<에이전트id>.png` — 에이전트 한 명만 (예: `lead.png`)
2. `teams/<팀>/avatars/<직급id>.png` — 그 팀의 해당 직급 전체
3. `assets/avatars/<직급id>.png` — 공용 기본 이미지
- 같은 이름에 `_working` 을 붙인 파일(예: `lead_working.png`)은 **답변·검토 중**일 때 표시됩니다.
- png/jpg/jpeg/webp, 정사각형 권장. 기본 이미지 다시 만들기: `python scripts/make_placeholder_avatars.py --force`
- 직급 id: `staff`(사원), `assistant_manager`(대리), `deputy_gm`(차장), `gm`(부장), `team_lead`(팀장)

### 모델 바꾸기
`config/models.yaml` 에서 직급별 `primary` / `fallback` 을 수정합니다. 특정 팀만 바꾸려면 team.yaml 의 `model_overrides`.

### 새 팀 추가
`teams/_template` 폴더를 `teams/<새_팀_id>` 로 복사하고 `team.yaml`(조직도·검토 기준·하네스·업무 양식) 과 `skills/` 를 작성합니다.
`teams/stock`, `teams/travel` 이 완성된 예시입니다.

### 업무 양식 추가 (team.yaml `tasks`)
업무 하나 = 입력칸(`fields`) + 질문 템플릿 + 목표·완료 기준 + 하네스. 앱을 새로고침하면 **📋 업무 선택**에 나타납니다.

```yaml
tasks:
  - id: forecast
    title: 2) 주가 전망
    fields:                                  # type: text | textarea | number | date | select
      - {key: stock, label: 종목 이름 또는 코드, required: true}
      - {key: horizon, label: 전망 기간, type: select, options: [1주, 1개월, 3개월]}
    question: "{stock} 종목의 향후 {horizon} 주가를 전망해줘."   # {key} = 입력값, 빈 값이 든 줄은 자동 삭제
    goal: "{stock} 종목의 {horizon} 시나리오별 전망 보고서"
    success_criteria: ["상승·기본·하락 시나리오와 확률을 제시한다"]
    validators: [required_terms]             # 이 업무에서 추가로 돌릴 하네스
    harness_options: {required_terms: [시나리오, 확률]}
```
`start_date`·`end_date` 입력칸이 있으면 `{nights}`·`{days}` 가 자동으로 계산되어 템플릿에서 쓸 수 있습니다.

### 데이터 툴 추가 (team.yaml `tools`)
`core/tools/` 에 `run(question, inputs, models) -> ToolResult | None` 함수를 만들고 `core/tools/__init__.py` 의 `TOOLS` 에 등록합니다.
`inputs` 는 업무 양식의 입력값입니다. 결과는 모든 에이전트의 근거 자료와 화면 타임라인에 들어갑니다.
현재 툴: `stock_data`(주가·평가손익), `travel_calendar`(여행 날짜·요일). 팀별 검색 요령·기간은 team.yaml `search`(`hint`, `news_timelimit` 등).

## 8. 저장 정책

| 결과 | 처리 |
|---|---|
| 사람이 **채택/반려**한 결과 | `data/runs.db` 에 보관 + 에이전트별 교훈 저장 |
| 결정 없이 **결과 버리기** | 즉시 삭제 (체크포인트 포함) |
| 결정 없이 같은 대상에게 **새 질문** | 직전 결과 삭제 |
| 오류로 중단된 결과 | 저장하지 않음 |
| 앱 시작 시 남아 있는 미결정 결과 | `settings.yaml` `storage.purge_undecided_after_hours`(기본 1시간)보다 오래된 것 삭제 |

## 9. 폴더 구조와 코드 읽는 순서

```
Agent_office/
├── app.py                  ⑥ 화면 (조직도·업무 양식·질문·타임라인·결정) — Streamlit
├── core/
│   ├── schemas.py          ① 모든 데이터 형식 (설정·LLM 출력·실행 결과) — 여기부터 읽기
│   ├── config.py           ② 설정 읽기·검증, 조직도·아바타·스킬
│   ├── llm_factory.py      ③ provider/model → ChatModel, 대안 모델 전환
│   ├── agent.py            ④ 에이전트 한 명: 분배(plan)·초안(draft)·검토(review) 프롬프트
│   ├── workflow.py         ⑤ LangGraph: brief → research → work⟲ → human_decision → feedback
│   ├── brief.py            요청 정리(최종 상태 정의) + 질문 예시
│   ├── tasks.py            팀 업무 양식 → 질문·브리프·하네스 옵션
│   ├── tools/              데이터 툴: stock_data(주가), travel_calendar(여행 달력)
│   ├── search.py           웹 검색(검색어 생성 → DuckDuckGo 웹+뉴스)
│   ├── verification.py     루브릭·비평가·자가 교정
│   ├── feedback.py         사람 결정 → 에이전트별 교훈
│   ├── storage.py          SQLite (실행 이력·교훈, 미결정 삭제)
│   └── harness/            결정론적 검증기(validators) + 단계별 평가(evaluation)
├── config/                 ranks / models / settings / verification (.yaml)
├── teams/                  팀별 team.yaml (조직도·업무 양식) + skills/ + avatars/
│                           (general = 기본, stock = 주식, travel = 여행, _template = 템플릿)
├── evals/                  단계별 평가 실행기 + 케이스
├── assets/avatars/         공용 에이전트 이미지 (+ _working 이미지)
├── scripts/                기본 아바타 생성
├── GIT_GUIDE.md            Git/GitHub 에 올리기 단계별 가이드
├── tests/                  모의 LLM 테스트 (API 키·비용 없이 실행)
└── data/                   runs.db · checkpoints.db · evals/ (git 제외, 자동 생성)
```

## 10. 테스트

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

테스트는 가짜 모델·가짜 검색을 쓰므로 API 키·비용·인터넷 없이 실행됩니다.

## 11. Git 에 올리기 (새 저장소로 분리하는 경우)

> 처음 올리는 경우 **[GIT_GUIDE.md](GIT_GUIDE.md)** 의 단계별 가이드(GitHub 저장소 만들기 → 폴더 위치 → init → 메일 설정 →
> .env 제외 확인 → 커밋 → push → 다른 PC 에서 받기 → 문제 해결)를 따라 하세요. 아래는 요약입니다.

```bash
# 이 폴더를 독립 저장소로 (langchain-kr 안에 둔 채로도 가능)
cd 26-Agent-Office
git init -b main
git config user.name "이름"          # 이 저장소에만 적용 (--global 아님)
git config user.email "메일@example.com"
git add .
git status                           # .env, data/ 가 목록에 없는지 반드시 확인
git commit -m "Agent Office 초기 버전"
git remote add origin https://github.com/<계정>/<저장소>.git
git push -u origin main
```

- `.env`(API 키)와 `data/`(실행 기록·교훈)는 `.gitignore` 로 제외됩니다. 다른 PC 에서는 `.env.example` 을 복사해 키를 넣습니다.
- langchain-kr 안에 둔 채 독립 저장소로 쓰면, 바깥 저장소가 이 폴더를 추적하지 않도록 `../.git/info/exclude` 에 `26-Agent-Office/` 를 한 줄 추가하세요 (커밋되지 않는 로컬 설정).

## 12. 트러블슈팅

| 증상 | 해결 |
|---|---|
| 조직도 칸에 `모델 없음` | `.env` 에 해당 직급의 primary 또는 fallback provider 키를 입력 후 앱 재시작 |
| `ModuleNotFoundError` | 가상환경이 활성화됐는지 확인 후 `pip install -r requirements.txt` |
| `streamlit` 명령을 찾을 수 없음 | `python -m streamlit run app.py` 로 실행 |
| 포트 8501 사용 중 | `--server.port` 옵션으로 다른 포트 지정 |
| PowerShell 에서 Activate.ps1 실행 차단 | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` 후 재시도 |
| 검증 기록에 `credit balance is too low` | Anthropic 크레딧 충전 (그 전까지는 대안 모델이 답변) |
| 검증 기록에 `404 ... no longer available` | 모델 단종 — 에러 메시지가 안내하는 새 모델명으로 `config/models.yaml` 수정 |
| 검증 기록에 `429 ... quota` | 무료 등급 할당량 초과 — 결제 등록 또는 다른 모델로 변경 |
| 웹 검색 자료 0건 | 네트워크 확인. 검색 실패해도 답변은 계속 진행됩니다 (자료 없이) |
| 업데이트 후 화면에 AttributeError | 코드가 바뀌는 중에 실행한 경우 — 앱을 재시작하세요 |
