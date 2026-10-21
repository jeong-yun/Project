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
| 📈 주식 팀 | ① 분야별 종목 추천(1~5점 매력도) ② 종목 판단(7개 항목 → 매수·보유·매도) ③ 주가 전망 ④ 매매 전략(분할 매수·목표가·손절가·손익비) ⑤ 보유 종목 추가 매매(물타기·불타기·익절·손절) ⑥ 포트폴리오 구성 | **시세 데이터**(Yahoo Finance: 현재가·이동평균·재무 지표·컨센서스·평가손익·평단 시뮬레이션을 코드로 계산) + 최근 1개월 공시·리포트·뉴스 (블로그·커뮤니티 제외) + **코드 검산**(가중 점수·평균가·손익비) |
| ✈️ 여행 팀 | 여행 계획서 (국내/해외, 출발·도착일, 인원, 예산, 스타일) + **🗺 움직이는 경로 지도** | **여행 달력**(N박 M일·날짜별 요일을 코드로 계산) + 목적지 날씨·행사·교통 검색 |

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
| 네트워크 | 인터넷 연결 필수 (LLM API 호출, 웹 검색, 주가·지도 데이터) |
| 브라우저 | Chrome / Edge / Safari 최신 버전 |
| Git | 코드 내려받기·올리기용 (Windows: https://git-scm.com/download/win) |

## 2. 필요 API 키

| Provider | 환경 변수 | 발급처 | 현재 사용처 |
|---|---|---|---|
| Google (Gemini) | `GOOGLE_API_KEY` | https://aistudio.google.com/apikey | 사원·대리, 보조 작업(브리프·검색어·비평·교훈) |
| OpenAI (ChatGPT) | `OPENAI_API_KEY` | https://platform.openai.com/api-keys | 차장·부장·팀장 |
| Anthropic (Claude) | `ANTHROPIC_API_KEY` | https://console.anthropic.com/ | **현재 비활성** (크레딧 충전 후 사용) |
| Upstage (Solar) | `UPSTAGE_API_KEY` | https://console.upstage.ai/ | 대안 모델 (선택) |
| FreeLLMAPI (무료 모델) | `FREELLMAPI_API_KEY`, `FREELLMAPI_BASE_URL` | 내 PC 의 FreeLLMAPI 대시보드 (아래 2-1) | 유료 키가 모두 안 될 때 마지막 대안 / '🆓 무료 모델만 사용' |
| LangSmith (선택) | `LANGSMITH_API_KEY` | https://smith.langchain.com/ | 실행 추적 |

- **일부 키만 있어도 동작합니다.** 사용할 수 없는 provider 는 `config/models.yaml` 의 대안(fallback) 모델로 자동 대체됩니다.
- 키가 있어도 호출이 실패하면(크레딧 부족, 할당량 초과, 모델 단종 등) 다음 후보로 넘어가고, 타임라인의 "검증 기록"에 사유가 남습니다.
- **Claude 를 다시 쓰려면**: `config/models.yaml` 의 `disabled_providers: [anthropic]` 에서 anthropic 을 지우고, 부장·팀장 primary 를 Claude 모델로 바꾸면 됩니다.
- Gemini **Pro** 계열은 무료 등급 할당량이 0 입니다 (Flash 계열은 무료 사용 가능).
- 웹 검색은 DuckDuckGo(`ddgs` 패키지), 주가 데이터는 Yahoo Finance(`yfinance` 패키지)를 쓰므로 별도 키가 필요 없습니다.
- 비용: 기본 팀 전체(팀장부터)에 질문하면 LLM 호출 약 12~15회 (브리프 1 + 검색어 1 + 분배 2 + 초안 3 + 검토·통합 5 + 비평 1 + 교훈 1).

## 2-1. 무료 모델 (FreeLLMAPI) — 유료 키를 쓸 수 없을 때

`langchain-kr/25-FreeLLMAPI-FreeModels` 에서 설치한 **FreeLLMAPI**(Google·Groq 등 무료 티어를 OpenAI 호환 주소 하나로 묶는 내 PC 의 프록시)를
`freellm` provider 로 연결했습니다. 직급마다 유료 후보가 모두 실패(키 없음·크레딧 부족·장애)하면 무료 모델로 넘어가고,
사이드바의 **🆓 무료 모델만 사용** 을 켜면 처음부터 무료 모델만 씁니다.

1. FreeLLMAPI 서버 켜기: Docker Desktop 실행 후 `docker compose -f C:/Users/<사용자>/freellmapi/docker-compose.yml up -d`
2. `.env` 에 `FREELLMAPI_BASE_URL=http://127.0.0.1:3001/v1`, `FREELLMAPI_API_KEY=freellmapi-…`(대시보드 Keys 페이지의 통합 키)
3. 사이드바 provider 상태에 `✅ freellm` 이 보이면 준비 완료 (`🔌 서버 꺼짐`이면 1번 확인). 평가도 `python -m evals.run_eval … --free`

**직급별 무료 모델 배정** (2026-10-20 실측: FreeLLMAPI 카탈로그의 지능 순위·무료 한도 + 직접 호출한 속도·구조화 출력 성공 여부)

| 직급 | 1순위 → 대안 | 이유 |
|---|---|---|
| 사원 | gpt-oss-20b → gemini-3.1-flash-lite → gemini-3.5-flash-lite | 가장 많이 호출되는 직급 → 하루 1,000회·0.5초로 가장 빠른 Groq gpt-oss-20b. 긴 프롬프트(분당 8천 토큰 초과)는 문맥이 큰 Flash-Lite 로 |
| 대리 | gemini-3.5-flash-lite → gpt-oss-120b → gemini-3.1-flash-lite | 검토(구조화 출력)가 빠르고 안정적(0.8초, 두 방식 모두 성공). 지능 순위 63 |
| 차장 | gpt-oss-120b → gemini-3.5-flash → gemini-3.1-flash-lite | 추론형 120B, 하루 1,000회라 여러 번 쓰는 실무 분석에 적합. 긴 입력은 Gemini 로 |
| 부장 | gemini-3.6-flash → gemini-3.5-flash → gpt-oss-120b | 하위 결과를 모아 통합 → 긴 입력(분당 25만 토큰)·지능 순위 39·5~7초 |
| 팀장 | gemini-3.7-flash → gemini-3.8-flash → gemini-3.6-flash → gpt-oss-120b | 최종 품질 담당·호출 수가 가장 적음 → 동작이 확인된 최상위 모델(순위 29, 37~52초로 느림). 3.8 은 순위 25 이지만 실측 시 시간 초과 |
| 보조 작업 | gpt-oss-120b → gemini-3.1-flash-lite | 브리프·검색어·비평·교훈 — 호출이 많아 하루 1,000회 한도가 필요 |

- **제외한 무료 모델**: qwen3.8-27b(분당 1천 토큰 — 긴 프롬프트 불가), gemma-4(3만 토큰 문맥·느림·구조화 출력 실패), gemini-2.5 계열(404), auto(자동 선택 — 응답 형식 오류).
- **한도 주의**: Gemini 무료 모델은 **모델마다 하루 20회·분당 10회**, Groq gpt-oss 는 **분당 8천 토큰**이라 하루 실행 횟수가 적습니다.
  곧 풀리는 한도(150초 이내, 예: 분당 토큰)는 그만큼 기다렸다 같은 모델로 다시 하고, 하루 한도가 찬 모델은 10분 동안 건너뜁니다.
  직급마다 Google(Gemini)과 Groq(gpt-oss)를 섞어 둔 것은 실측 중 Gemini 가 503 '요청 폭주'로 동시에 막힌 적이 있기 때문입니다.
- 무료 티어 입력은 제공자 학습에 쓰일 수 있습니다 — 회사·개인 민감 데이터에는 유료 키를 쓰세요.

## 3. 설치 — 새 PC 에서 처음 시작하기

코드는 GitHub `jeong-yun/Project` 저장소의 `LLM/Agent_office` 폴더에 있습니다. 아래는 Windows PowerShell 기준입니다.

```powershell
# 1) 저장소 내려받기 (원하는 폴더에서)
git clone https://github.com/jeong-yun/Project.git
cd Project\LLM\Agent_office

# 2) 가상환경 만들기 → 켜기 (Python 3.11 이상)
python -m venv .venv
.venv\Scripts\Activate.ps1
#   macOS / Linux:  python3 -m venv .venv  →  source .venv/bin/activate

# 3) 패키지 설치
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4) API 키 설정: .env.example 을 .env 로 복사한 뒤 메모장에서 키 입력
Copy-Item .env.example .env
notepad .env
#   macOS / Linux:  cp .env.example .env

# 5) 실행 → 브라우저에서 http://localhost:8501
streamlit run app.py
```

> `.env` 는 이 폴더 안의 것을 우선 읽고, 없는 값은 바로 위 폴더의 `.env` 에서 보충합니다.
> **`.env` 는 절대 git 에 올리지 않습니다** — `.gitignore` 에 들어 있고, 저장소에는 값이 빈 `.env.example` 만 올립니다.
> 잘 설치됐는지 확인하려면 `pip install -r requirements-dev.txt` 후 `python -m pytest -q` (API 키 없이 실행, 모두 passed 면 정상).

### git 으로 옮겨지지 않는 것 (직접 챙기기)

| 항목 | 내용 | 다른 PC 로 옮기는 방법 |
|---|---|---|
| `.env` | API 키 | 새로 만들어 키 입력 (메신저·메일로 키 파일을 보내지 않는 것을 권장) |
| `data/runs.db` | 에이전트별 **교훈**·채택/반려 기록 | 이 파일을 USB·클라우드 드라이브로 같은 위치에 복사 → 교훈이 그대로 이어짐 |
| `data/geocode_cache.json` | 여행 지도 좌표 캐시 | 선택 (없으면 처음 한 번 다시 검색) |
| `results/` | 채택해서 저장한 결과 파일(.md, 지도 .html) | 폴더째 복사 |
| `.venv/` | 가상환경 | 옮기지 말고 PC 마다 새로 만들기 (2~3단계) |

### 이미 설치한 PC 에서 최신 코드로 업데이트

```powershell
cd Project\LLM\Agent_office
git pull
.venv\Scripts\Activate.ps1
pip install -r requirements.txt   # 새 패키지가 추가됐을 수 있으므로 매번 실행
streamlit run app.py
```

> 두 PC 에서 번갈아 작업할 때는 **작업 시작 전 `git pull`, 끝나면 `git add` → `git commit` → `git push`** 순서를 지키면 충돌이 생기지 않습니다.
> 자세한 Git 사용법은 [GIT_GUIDE.md](GIT_GUIDE.md) 참고.

## 4. 실행

```bash
streamlit run app.py
```

브라우저에서 http://localhost:8501 이 열립니다. 포트를 바꾸려면 `streamlit run app.py --server.port 8600`.

## 5. 화면 보는 법

```
① 문의 대상      팀장(기본) / 다른 칸을 고르면 그 아래 팀만 일함
② 업무 맡기기     팀 업무 양식 또는 자유 질문 — 일이 없을 때는 맨 위, 업무가 있으면 결과 아래
③ 조직도         👤 사용자 ⟷ 팀장 → 전문가들 — 칸마다 지금 상태(✍️ 작성 중 / 📝 초안 완료 / 🧩 통합 8점 …)
   └ 질문 카드: 질문 + 진행 단계 줄 (🎯 요청 정리 › 🌐 자료 수집 › 📋 업무 분배 › ✍️ 작성 › 🔍 검토·통합 › ⚖️ 검증 › 👤 결정)
④ 에이전트 패널   조직도 칸을 클릭했을 때만 — 맡은 업무 · 지금 쓰는 초안 · 결과·검토 내용(스크롤 상자) · 🖼 이미지 바꾸기
⑤ 전체 진행      '📜 전체 진행·근거 자료 보기'를 켰을 때만 — 전체 단계 / 브리프·근거 자료 / 🗺 여행 지도(여행 팀)
⑥ 최종 결과·결정  채택(수정 가능) / 반려 / 버리기 — ⬇ 결과 받기, 채택하면 results/<팀>/ 에 파일로 저장
                 결정 뒤 '✖ 이 결과 닫고 새 업무 시작'을 누르면 처음 화면으로

사이드바: 팀 선택 · 🔔 진행 중인 업무(모든 팀, 누르면 이동) · 🆓 무료 모델 · 🕘 지난 업무(누르면 내용·결정·교훈 창)
          · 📂 저장된 결과 파일 · ⚙️ 모델·검증 상태(접힘) · 📁 개념 카탈로그(개념 편집 팀, GitHub 링크·올리지 않은 커밋)
```

- 업무는 **백그라운드에서 실행**되고 화면은 1초마다 자동으로 갱신됩니다. 진행 중에 다른 에이전트를 눌러 봐도 작업은 끊기지 않습니다.
- 긴 내용은 각 상자 안에서 스크롤되므로 페이지 전체를 내려가지 않아도 됩니다.
- 진행 중인 업무는 서버에 보관되므로 **새로 고침·다른 팀 이동·브라우저를 다시 열어도** 그대로 보입니다 (주소의 `?team=` 으로 보던 팀도 유지).
  서버(Streamlit)를 껐다 켜면 진행 중이던 작업은 멈추지만, **결정 대기** 상태였던 업무는 되살아나 이어서 채택·반려할 수 있습니다.
- 업무가 끝나면 화면 오른쪽 아래 알림과 **브라우저 탭 제목**(⏳ 진행 중 / 🔔 결정 대기)으로 알려 줍니다.

## 5-1. 사용법 — 좋은 질문은 '과정'이 아니라 '최종 상태'를 정의한다

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
| 비평가 (LLM-as-a-Judge) | 검토자의 '지적 → 수정' 중간 과정의 논리 비약만 따로 채점. 데이터 툴·웹 검색 근거 자료도 함께 받아 숫자·출처·기준일을 대조 | `config/verification.yaml` `critic` (`evidence_chars`: 근거 자료 글자 예산) |
| 비평 합격 기준 | 점수 ≥ `min_score` 이면 합격. 지적마다 심각도(severity: none/minor/severe)를 매겨 **severe**(사실 오류·근거와 다른 숫자·올바른 사실 누락)일 때만 불합격 | `critic.fail_on` (`severe` 기본 / `any` = 지적이 하나라도 있으면 불합격, 이전 동작) |
| 자가 교정 | 비평 불합격 시 "A 부분이 B 규칙에 어긋난다. 수정하라" 를 체인 중간에 넣어 재작성. 교정본이 더 나빠지면 **비평 점수가 가장 높은 답변을 유지**하고, 끝까지 불합격이면 검증 기록에 "최종 비평가 여전히 불합격"과 시도별 점수를 남김 | `critic.max_self_corrections` |
| 빈 초안 감지 | 모델이 오류 없이 빈 응답(조각 0개·공백뿐)을 주면 실패로 보고 같은 업무를 다음 후보 모델로 다시 씀 | `core/agent.py` |
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
| `stock_plan` | 주식 답변 끝 ```json 요약의 가중 점수·결정 밴드·분할 매수 평균가·손익비·손절폭·물타기 평단·추가 매수 한도를 코드로 다시 계산 → 틀리면 정답 값과 함께 재작성 (`core/stock_calc.py`) |

`harness.final_only` 에 적은 검증기는 최종 답변(문의 대상 에이전트)에만 돌립니다 — 하위 실무자는 업무 일부만 쓰므로 전체 형식을 요구하지 않습니다.
team.yaml `verification.critic` 으로 팀별 비평 설정을 덮어쓸 수 있습니다 (주식 팀: 모든 검토 단계 비평, 자가 교정 최대 2회, 점수-분석 불일치·근거 없는 수치 규칙).

업무 양식(team.yaml `tasks`)마다 `validators`·`harness_options` 를 따로 지정할 수 있습니다 (예: 주가 전망은 `required_terms: [기준일, 시나리오, 상승, 하락, 확률]`).

새 검증기(예: ClickHouse 테스트 DB 실행)는 `core/harness/validators.py` 에 `Validator` 를 상속한 클래스를 만들고 `@register` 를 붙이면 됩니다.

### 단계별 평가 실행

```bash
python -m evals.run_eval evals/cases/general.yaml   # 기본 팀 (HBF 등)
python -m evals.run_eval evals/cases/stock.yaml     # 주식 팀 (삼성전자 전망·보유 전략·판단·매매 전략)
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

### 에이전트 이미지 바꾸기
**화면에서 바꾸기 (권장)**: 조직도에서 에이전트를 클릭 → 패널의 **🖼 이 에이전트 이미지 바꾸기** → 그림 올리기 → 💾 저장.
- 가운데를 정사각형으로 잘라 256×256 으로 저장하고, '작업 중' 그림을 따로 올리지 않으면 말풍선(…)을 붙여 자동으로 만듭니다.
- **↩ 기본 이미지로** 를 누르면 원래(직급 공용) 이미지로 돌아갑니다. png/jpg/webp, 5MB 이하.

**파일로 바꾸기** (먼저 찾은 것 사용):
1. `teams/<팀>/avatars/<에이전트id>.png` — 에이전트 한 명만 (예: `lead.png`) ← 화면에서 저장하면 이 위치
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
    instructions: |                          # (선택) 사고 절차·루브릭·퓨샷·JSON 형식 → 브리프 [배경]으로 모든 에이전트에게
      근거를 먼저 쓰고 점수는 마지막에 쓴다 …
    validators: [required_terms]             # 이 업무에서 추가로 돌릴 하네스
    harness_options: {required_terms: [시나리오, 확률]}   # 문자열 값의 {key} 도 입력값으로 바뀜
```
`start_date`·`end_date` 입력칸이 있으면 `{nights}`·`{days}` 가 자동으로 계산되어 템플릿에서 쓸 수 있습니다.

### 데이터 툴 추가 (team.yaml `tools`)
`core/tools/` 에 `run(question, inputs, models) -> ToolResult | None` 함수를 만들고 `core/tools/__init__.py` 의 `TOOLS` 에 등록합니다.
`inputs` 는 업무 양식의 입력값입니다. 결과는 모든 에이전트의 근거 자료와 화면 타임라인에 들어갑니다.
현재 툴: `stock_data`(주가·재무 지표·컨센서스·평가손익·추가 매수 한도), `travel_calendar`(여행 날짜·요일). 팀별 검색 요령·기간은 team.yaml `search`(`hint`, `news_timelimit` 등).

## 7-1. 주식 팀 — 판단을 돕는 장치

| 장치 | 내용 |
|---|---|
| 종목 식별 | 시장(KOSPI/KOSDAQ/NASDAQ…)·업종을 시세 툴이 먼저 확인 |
| 재무 지표 | EPS·ROE·영업이익률·성장률·부채비율·유동비율·배당성향·베타 (Yahoo 집계값 — DART 교차 확인 안내, 없으면 `확인 불가`) |
| 컨센서스 극단값 주의 | 평균 목표가가 현재가와 60% 이상 차이 나거나 최고/최저가 2배 이상이면 ⚠️ 표시 → 최신 리포트 확인 전에는 근거로 쓰지 않음 |
| 손절 참고가 | 단기 3~5% · 중기 8~10% · 장기 15% 이상 가격을 코드로 계산 |
| 보유 포지션 | 평가손익·본전 상승률, 현재/최대 비중으로 **추가 매수 가능 금액**, 25·50·100% 추가 시 **새 평균 단가** 시뮬레이션 |
| 근거 먼저 → 점수 | 추천·판단 업무는 리스크 → 기회 → 점수 순서 강제 + 1점/5점(추천), 4.35 매수/1.55 매도(판단) 기준점 예시 |
| 출처 규칙 | 모든 수치 `[출처, YYYY-MM-DD]`, 우선순위 DART → KRX → 네이버 금융·FnGuide → 증권사 리포트 → 뉴스, 블로그·카페·종목토론방은 검색 결과에서 제외 (team.yaml `search.exclude_domains`) |
| 🧮 코드 검산 | 최종 결과 아래에 LLM 이 아닌 코드가 다시 계산한 점수표·평균 매수가·손익비·평단을 따로 표시 (채택 시 저장 파일에도 포함) |

판단 업무 결정 밴드: 종합 = Σ(점수×가중치)÷100 → **3.75 이상 매수 / 2.50~3.74 보유 / 2.50 미만 매도**, `데이터 부족` 3개 이상이면 **판단 보류**.

## 7-2. 📚 개념 편집 팀 (IT 개념 카탈로그)

사이드바 **팀 선택 → 개념 편집 팀** — 다른 팀과 똑같이 조직도·에이전트 패널(칸을 누르면 그 에이전트의 진행 상황)·이미지 편집·타임라인을 쓰면서, GitHub [jeong-yun/CS](https://github.com/jeong-yun/CS/tree/main/Term/Every_day) 의 IT 개념 카탈로그에 항목을 작성(추가·수정)·삭제합니다. 사이드바 **📁 개념 카탈로그** 에서 폴더 위치와 git 상태를 확인합니다.

1. CS 저장소를 clone: `git clone https://github.com/jeong-yun/CS.git` (기본 위치: 프로젝트 폴더 옆, `config/settings.yaml` 의 `concept.catalog_dir`)
2. **✍️ 작성**에 개념명·초안 입력 → 추가/수정은 **자동 판정**: 같은 개념이 독립 항목으로 있거나 다른 항목의 `7. 변형·확장 상세` 소절에 들어 있으면 **수정**(그 항목/부모 항목), 없으면 **추가** (판정은 화면에서 바꿀 수 있음). **관련 기존 항목**도 코드가 찾아 함께 보냄
3. **개념 편집 팀**(편집장 → 수석에디터 → 팩트체크·계보분석)이 'IT 개념 프롬프트'(`teams/concept/skills/it_concept_prompt.md`) 규칙으로
   1단계 비교(이중 해리) → 2단계 검증 로그 → 카탈로그 형식 최종본 → GitHub 요약 → 적용 계획(JSON) 을 작성
   - 형식은 하네스 `concept_entry` 가 코드로 검사: `<thought>`가 먼저인지, 1~6절·등장 배경·관계의 유사/차이 짝, 개조식, 절대 표현, 참고문헌 형식, 적용 계획
4. **사람 승인** (최종본·적용 계획 수정 가능) → 파일 반영
   - 추가: 해당 분류 파일 맨 위 / 수정: 제자리 교체 + 수정일 추가(작성일은 기존 그대로, 분류가 바뀌면 이동) / 삭제·통합: 원문을 `old_term.txt` 로
   - `0. Summary.md` 로드맵·한 줄 요약 표 갱신, `basic_Term.txt` 에 기초 용어 추가, 다른 항목의 관계 문장 교체(정확히 일치할 때만)
5. 채택 후 결과 영역의 **📚 카탈로그 반영** → **💾 커밋** → **🚀 GitHub 에 푸시** (버튼을 눌러야만 올라감)

## 7-3. 여행 경로 지도 (여행 팀)

여행 팀의 최종 계획서가 나오면 백그라운드에서 지도를 만듭니다 (약 30초~1분, 그동안 결과를 보고 결정해도 됩니다).

| 요소 | 설명 |
|---|---|
| 기호 마커 | 🏛 문화·역사 · 🚶 걷기·산책 · 🍜 음식 · ☕ 카페 · 🛍 쇼핑 · 🌳 자연 · 🎢 테마파크·체험 · 🌃 야경·전망 · 🏨 숙소 · 🚆 교통 — 마커의 숫자는 `일차-순서` |
| 움직이는 경로선 | 일차별 색, 점선이 이동 방향으로 흐름 |
| ▶ 재생 | 시간 순서(여행 날짜 + 오전·오후·저녁)대로 경로가 그려지며 이동을 보여 줌 |
| 일차 선택 | 전체 일정 / n일차 만 보기 + 방문지 표(장소·시간·메모·위치 확인 여부) |

- 방문지는 보조 모델이 계획서에서 뽑고, 좌표는 OpenStreetMap(Nominatim, 키 불필요)으로 확인합니다. 모델이 추정한 위치와 30km 이상 다르면
  동명이인 장소로 보고 추정 좌표를 쓰며 표에 '추정'으로 표시합니다. 검색 결과는 `data/geocode_cache.json` 에 저장해 다시 묻지 않습니다.
- 채택할 때 계획을 직접 고쳤거나 다른 에이전트의 안을 채택하면, 채택한 내용으로 지도를 다시 만들어 저장합니다.
- 다른 팀에서도 쓰려면 team.yaml 에 `features: [route_map]` 을 넣습니다.

## 8. 저장 정책

| 결과 | 처리 |
|---|---|
| 사람이 **채택**한 결과 | `data/runs.db` 기록 + 교훈 저장 + **`results/<팀>/<날짜-시각>_<질문>.md` 파일** (여행 팀은 `_지도.html` 도) — 화면의 ⬇ 받기, 사이드바 📂 저장된 결과에서 내려받기 |
| 사람이 **반려**한 결과 | `data/runs.db` 기록 + 교훈 저장 (파일은 만들지 않음) |
| 결정 없이 **결과 버리기** | 즉시 삭제 (체크포인트 포함) |
| 결정 없이 같은 대상에게 **새 질문** | 직전 결과 삭제 |
| 오류로 중단된 결과 | 저장하지 않음 |
| 앱 시작 시 남아 있는 미결정 결과 | `settings.yaml` `storage.purge_undecided_after_hours`(기본 1시간)보다 오래된 것 삭제 |

## 9. 폴더 구조와 코드 읽는 순서

```
Agent_office/
├── app.py                  ⑥ 화면 (클릭형 조직도·에이전트 패널·탭·결정·업무 양식) — Streamlit
├── concept_ui.py           개념 편집 팀 화면 조각 (카탈로그 사이드바·작성/삭제 입력·결과 탭·채택 후 반영 → 커밋 → 푸시)
├── core/
│   ├── schemas.py          ① 모든 데이터 형식 (설정·LLM 출력·실행 결과) — 여기부터 읽기
│   ├── config.py           ② 설정 읽기·검증, 조직도·아바타·스킬
│   ├── llm_factory.py      ③ provider/model → ChatModel, 대안 모델 전환
│   ├── agent.py            ④ 에이전트 한 명: 분배(plan)·초안(draft)·검토(review) 프롬프트
│   ├── workflow.py         ⑤ LangGraph: brief → research → work⟲ → human_decision → feedback
│   ├── brief.py            요청 정리(최종 상태 정의) + 질문 예시
│   ├── live.py             백그라운드 실행 + 실시간 진행 상태 (화면이 1초마다 읽어 그림)
│   ├── avatars.py          에이전트 이미지 저장·자동 '작업 중' 이미지·되돌리기
│   ├── exporter.py         채택 결과 → results/<팀>/ 파일 (.md, 지도 .html)
│   ├── route.py            여행 계획서 → 방문지 → OSM 좌표 → 움직이는 지도(folium)
│   ├── tasks.py            팀 업무 양식 → 질문·브리프·하네스 옵션
│   ├── stock_calc.py       주식 숫자 검산 (가중 점수·결정 밴드·평균가·손익비·평단)
│   ├── catalog.py          IT 개념 카탈로그(CS 저장소) 읽기·관련 항목 찾기·반영·git
│   ├── concept.py          개념 작성 요청·결과 나누기·형식 검사
│   ├── tools/              데이터 툴: stock_data(주가), travel_calendar(여행 달력)
│   ├── search.py           웹 검색(검색어 생성 → DuckDuckGo 웹+뉴스)
│   ├── verification.py     루브릭·비평가·자가 교정
│   ├── feedback.py         사람 결정 → 에이전트별 교훈
│   ├── storage.py          SQLite (실행 이력·교훈, 미결정 삭제)
│   └── harness/            결정론적 검증기(validators) + 단계별 평가(evaluation)
├── config/                 ranks / models / settings / verification (.yaml)
├── teams/                  팀별 team.yaml (조직도·업무 양식) + skills/ + avatars/
│                           (general = 기본, stock = 주식, travel = 여행, concept = 개념 편집, _template = 템플릿)
├── evals/                  단계별 평가 실행기 + 케이스
├── results/                채택한 결과 파일 (git 제외 — 개인 정보 보호. 올리려면 .gitignore 에서 results/ 삭제)
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

## 11. Git — 수정한 내용 올리기

코드는 GitHub `jeong-yun/Project` 저장소(`main` 브랜치)의 `LLM/Agent_office` 폴더에서 관리합니다.

```powershell
cd Project            # 저장소 최상위 폴더
git pull              # 다른 PC 에서 올린 내용 먼저 받기
git status            # 바뀐 파일 확인 — .env, data/, results/ 가 없어야 정상 (.gitignore 로 제외)
git add LLM/Agent_office
git commit -m "무엇을 왜 바꿨는지 한 줄"
git push
```

- 처음 쓰는 PC 에서는 커밋 작성자를 한 번 설정합니다 (이 저장소에만 적용): `git config user.name "이름"`, `git config user.email "메일"`
- 처음 push 할 때 브라우저 GitHub 로그인 창이 뜹니다 (Git Credential Manager).
- 저장소 만들기·폴더 이름 바꾸기·실수로 `.env` 를 올렸을 때 대처 등 자세한 내용은 [GIT_GUIDE.md](GIT_GUIDE.md).

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
| `503 ... high demand` (요청 폭주) | 해당 모델 서버가 일시적으로 붐빔 — 앱이 자동으로 2초·5초 뒤 같은 모델로 다시 시도하고, 그래도 안 되면 대안 모델로 처음부터 다시 씁니다. 실무자 한 명이 끝내 실패해도 "⚠️ 작성 실패" 로 표시하고 팀은 계속 진행합니다 |
| 웹 검색 자료 0건 | 네트워크 확인. 검색 실패해도 답변은 계속 진행됩니다 (자료 없이) |
| 업데이트 후 화면에 AttributeError | 코드가 바뀌는 중에 실행한 경우 — 앱을 재시작하세요 |
