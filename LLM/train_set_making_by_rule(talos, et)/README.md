# rule_data — Snort 룰 수집·파싱 도구

> **공개본 안내**: 이 폴더는 코드만 담았다. 데이터(01~10 폴더), 사용자 룰, 기존 학습 로그, 보고 문서, 담당자 검토용 도구(10_review), 사용자 룰 수집기(1_1a)는 넣지 않았다. `.env.example` 을 `.env` 로 복사해 키를 넣고 쓴다.

Cisco Talos와 Proofpoint ET Open의 Snort 룰을 매일 자동으로 받고, 선별한 HTTP 룰을 정규화·통합한 뒤 **IPS/WAF 학습데이터 생성용 LLM 입력**(`05_llm_input/`)으로 만든다.

- 파이썬 3.8 이상, 기본 라이브러리만 사용 (`pip install` 불필요)
- 매일 자동 수집: 화면의 **수집 에이전트**(또는 Windows 작업 스케줄러)
- **화면(UI)**: `python 0_0_run_ui.py` → 수집 에이전트·학습데이터 생성 실행, 진행률(현재 룰 중 몇 %), 룰별 상태 플래그

## 문서 (`docs/`)

| 폴더 | 용도 | 파일 |
| --- | --- | --- |
| `docs/코드참조/` | **코드가 읽거나, 코드를 만들 때 참조하는 문서** | [선별조건.md](docs/코드참조/선별조건.md) — 룰 선별·중복 제거 조건 (O/X를 고치면 다음 실행 때 자동 반영)<br>[LLM_프롬프트.md](docs/코드참조/LLM_프롬프트.md) — LLM 지시문 (`05_llm_input`에 prompt.md로 복사됨)<br>[LLM입력_형식.md](docs/코드참조/LLM입력_형식.md) — LLM 입력 레코드 필드 명세<br>[method_매핑.md](docs/코드참조/method_매핑.md) — 공격 유형 → IPS/WAF method 이름 (1.4가 읽음)<br>[LLM_설정.md](docs/코드참조/LLM_설정.md) — 생성·검토 모델 선택 (3.0이 읽음)<br>[LLM_검토_프롬프트.md](docs/코드참조/LLM_검토_프롬프트.md) — 검토 모델 지시문 |
| `docs/보고용/` | **사람이 읽는 보고·근거 문서** | [룰_선별기준.md](docs/보고용/룰_선별기준.md) — 선별·중복 제거 판단 근거와 수치<br>[진행_설계_요약.md](docs/보고용/진행_설계_요약.md) — 진행 내용과 설계(라벨·생성·품질·속도·ML 피처) 요약 |

## 빠른 시작

1. 설정 파일을 만들고 OINKCODE를 입력한다. (없으면 ET만 받는다)
   ```powershell
   Copy-Item .env.example .env
   ```
   OINKCODE 발급: snort.org 회원가입 → 로그인 → 계정 메뉴 → **Oinkcode** 탭

2. 한 번 실행해 본다. 첫 실행은 Talos(약 149MB) 때문에 2~3분 걸린다.
   ```powershell
   python 1_0_run_daily.py
   ```

3. 화면을 연다. 브라우저에서 수집 에이전트를 켜고(매일 16시), 학습데이터 생성을 시작·중지하고, 진행률을 본다.
   ```powershell
   python 0_0_run_ui.py
   ```
   화면 없이 쓰려면: 수집 에이전트 `python 1_0_run_daily.py --agent --at 16:00`, 또는 작업 스케줄러 등록
   `powershell -ExecutionPolicy Bypass -File .\2_0_register_task.ps1 -At 16:00` (둘 다 켜도 수집은 한 번에 하나만 돈다).

## 구성

**코드 (파일 이름의 번호 = 실행 순서)**

| 순서 | 파일 | 역할 | 결과 폴더 |
| --- | --- | --- | --- |
| 0.0 | `0_0_run_ui.py` | **화면.** 수집 에이전트·학습데이터 생성 실행·중지, 진행률, 룰별 상태 (127.0.0.1 에서만) | — |
| 0.0a | `0_0a_runtime.py` | (보조) 실행 잠금·중지 요청·상태 파일. 직접 실행하지 않음 | `.run/` |
| 1.0 | `1_0_run_daily.py` | **진입점.** 1.1 → 1.2 → 1.3 → 1.4를 차례로 실행. `--agent` 면 매일 정해진 시각에 반복(수집 에이전트) | — |
| 1.1a | (공개본에 없음) | 사용자 룰 수집기 (사내 시스템 전용). `.rules` 파일을 `00_custom_rules/` 에 직접 넣어도 된다 | `00_custom_rules/*.rules` |
| 1.1 | `1_1_fetch_rules.py` | 수집: 룰 압축 파일 다운로드 (바뀐 것만) | `01_raw/` |
| 1.2 | `1_2_parse_rules.py` | 파싱: 압축 해제, 선별, HTTP 필터링, 변경분, 통합·중복 제거 | `02_extracted/` ~ `04_integrated/` |
| 1.2a | `1_2a_rule_selection.py` | (1.2 보조) `docs/코드참조/선별조건.md`를 읽는 로더. 직접 실행하지 않음 | — |
| 1.3 | `1_3_build_llm_input.py` | LLM 입력: 학습데이터 생성에 필요한 필드만 추출, 25개씩 배치 분할 | `05_llm_input/` |
| 1.4 | `1_4_build_dataset.py` | 데이터셋: LLM 응답 → IPS/WAF 학습 로그 + 줄별 label | `08_dataset/` |
| 2.0 | `2_0_register_task.ps1` | (선택) 1.0을 매일 자동 실행하도록 작업 스케줄러에 등록 (수집 에이전트 대신) | — |
| 3.0 | `3_0_run_generation.py` | **(따로 실행, API 비용)** LLM 데이터 생성 진입점. 3.1 → 3.4 → 1.4 → 3.5 를 차례로 실행. `--status` 는 진행 현황만 | — |
| 3.0a | `3_0a_common.py` | (3.x 보조) 설정·LLM 호출·코드 검사 공통 기능. 직접 실행하지 않음 | — |
| 3.1a | `3_1a_existing_coverage.py` | 기존 학습 로그(06)에서 룰별 기존 샘플 수 계산 (비용 0, 입력이 그대로면 건너뜀) | `07_llm_output/<날짜>/_0_existing/` |
| 3.1 | `3_1_prepare_requests.py` | LLM 입력 준비: 생성할 룰만 골라 '호출 1회 = 요청 파일 1개'로 나눔 (비용 0) | `07_llm_output/<날짜>/_1_requests/` |
| 3.2 | `3_2_generate.py` | 데이터 생성: UpStage Solar → 룰당 씨앗 샘플(anomalies 10, normal 5) + 코드 검사 | `_2_generated/` |
| 3.3 | `3_3_review.py` | 데이터 검토: ChatGPT gpt-5 → pass / fix / drop, 공격 유형 확정, 코드 재검사 | `_3_reviewed/` |
| 3.4 | `3_4_augment.py` | 데이터 증강: 조건과 무관한 부분만 바꿔 룰당 anomalies 50(기존 샘플 수만큼 차감) / normal 10 (비용 0). normal 은 다른 공격 룰에도 안 걸리는 것만 | `07_llm_output/<날짜>/*.jsonl` |
| 3.5 | `3_5_progress.py` | 진행 현황: 현재 룰 중 학습데이터를 만든 %, 룰별 상태 플래그, 어디까지 했는지, 남은 시간·토큰 (비용 0) | `07_llm_output/_progress/` |
| 4.0 | `4_0_run_features.py` | **(따로 실행, 비용 0)** ML 피처 진입점. 4.1 → 4.2 를 차례로 실행. 1.0·3.0 과 별개 | — |
| 4.1 | `4_1_build_features.py` | ML 피처: 학습 로그의 **payload 만** → 공격·정상을 확실히 나타내는 피처 표 (공격 흔적·회피 신호, 신뢰할 수 있는 공격 룰 탐지 수, 개별 룰·정규식) | `09_features/<이름>/` |
| 4.2 | `4_2_evaluate_features.py` | 피처 점검: 피처 묶음별 간단 모델(LogReg·RF) 성능, 중요 피처, 룰 자체 성능 비교 | `09_features/<이름>/eval_*.md` |

각 단계는 따로 실행해도 되고(`python 1_2_parse_rules.py`), 바뀐 것이 없으면 스스로 건너뛴다.

**폴더와 파일**

```
rule_data/
 ├─ 1_0_run_daily.py ~ 2_0_register_task.ps1   ← 코드 (위 표)
 ├─ README.md            ← 이 문서
 ├─ docs/코드참조/       ← 코드가 읽는 md, 형식 명세
 ├─ docs/보고용/         ← 사람용 보고 문서
 ├─ 00_custom_rules/     ← 사용자 룰(선택): 직접 만든 Snort 룰 .rules 파일 (README 참고)
 ├─ .env.example         ← 설정 양식 (.env 로 복사해서 사용)
 ├─ .env                 ← OINKCODE 등 비밀 설정 — 공유 금지
 │
 │  (아래는 실행하면 자동 생성, 번호 = 처리 순서)
 ├─ 01_raw/<et|talos>/<날짜>/        ← ① 원본 압축 파일                       (1.1)
 ├─ 02_extracted/<et|talos>/*.rules  ← ② 압축 해제한 룰 (최신본만)            (1.2)
 ├─ 03_processed/<et|talos>/<날짜>/  ← ③ 소스별 선별 HTTP 룰                  (1.2)
 ├─ 04_integrated/<날짜>/            ← ④ ET + Talos 통합, 중복 제거           (1.2)
 ├─ 05_llm_input/<날짜>/             ← ⑤ LLM 입력                             (1.3)
 ├─ 06_기존 데이터/                  ← ⑥ 기존 IPS/WAF 학습 로그 (형식·method 기준, 직접 넣은 파일)
 ├─ 07_llm_output/<날짜>/            ← ⑦ LLM 데이터: _1_requests → _2_generated → _3_reviewed → _4_augmented, 최종 *.jsonl (3.x)
 ├─ 08_dataset/<날짜>/               ← ⑧ IPS/WAF 수집 형식 로그 + *_label.jsonl                        (1.4)
 ├─ 09_features/<이름>/              ← ⑨ ML 피처 표 features.csv + 피처 설명·점검 결과                (4.x)
 ├─ 10_review/                       ← ⑩ 담당자 검토용 (따로 실행): review_sample.py + <이름>/검토용_샘플.md 등. 본 파이프라인과 별개
 ├─ .run/                            ← 실행 중 표시(잠금)·중지 요청·에이전트 상태·화면에서 시작한 작업 출력 (0_0a_runtime.py)
 ├─ state.json           ← 마지막으로 받은 버전 (1.1)
 ├─ parse_state.json     ← 마지막으로 파싱한 버전, 선별 조건 지문 (1.2)
 └─ logs/                ← run_daily.log, fetch.log, parse.log, build_llm_input.log, build_dataset.log
```

## 실행 옵션

| 명령 | 설명 |
| --- | --- |
| `python 1_0_run_daily.py` | 수집 → 파싱·통합 → LLM 입력 (각 단계는 바뀐 것이 없으면 건너뜀) |
| `python 1_0_run_daily.py --skip-fetch` | 수집 빼고 실행 |
| `python 1_0_run_daily.py --force` | 바뀐 것이 없어도 모든 단계 다시 실행 |
| `python 1_3_build_llm_input.py --force --batch-size 50` | LLM 입력만 다시 생성 (배치 크기 변경) |
| `python 1_1_fetch_rules.py --source et` | ET만 수집 (`talos`, `all` 가능) |
| `python 1_2_parse_rules.py --source talos --force` | Talos만 다시 파싱 |

자동 실행 해제:
```powershell
Unregister-ScheduledTask -TaskName SnortRuleFetch -Confirm:$false
```

## 설정 (.env)

| 키 | 기본값 | 설명 |
| --- | --- | --- |
| `SNORT_OINKCODE` | (없음) | Talos 다운로드 키. 비우면 Talos 건너뜀 |
| `TALOS_SNORT_VERSION` | `29200` | Talos 스냅샷 버전 (Snort 2.9.20) |
| `ET_BASE_URL` | `https://rules.emergingthreats.net/open/snort-2.9.0` | ET Open 경로 |

## 동작 방식

**수집 (`1_1_fetch_rules.py`)**
- ET는 `version.txt`, Talos는 `.md5`를 먼저 받아 `state.json`의 값과 비교한다. 같으면 건너뛰고, 다르면 받는다.
- 다운로드에 성공했을 때만 `state.json`을 갱신한다. 실패하면 다음 실행 때 다시 시도된다.
- 로그에 찍히는 OINKCODE는 `****`로 가린다.

**파싱 (`1_2_parse_rules.py`)**
- ET는 `emerging-all.rules`, Talos는 `rules/*.rules`만 대상으로 한다. `so_rules`, `preproc_rules`는 제외한다.
- 주석 처리된 비활성 룰도 `enabled: false`로 포함한다.
- HTTP 룰로 판단하는 기준: `http_*` 수식어, pcre HTTP 플래그(U/I/P/H/D/M/C/K/S/Y), `metadata: service http`, `$HTTP_PORTS`, `urilen`
- method 추출 순서: `http_method` content → pcre `/M` → 원시 content(`"POST "`). 근거가 없으면 `methods: []`이고 `method_hint`에 추정값을 넣는다.
- 변경 판단: 키 `소스:gid:sid` 기준으로 rev 또는 룰 본문 해시(`sig`)가 바뀌면 "변경"이다. 활성/비활성 전환만 된 경우는 제외한다.

## LLM 입력 (`05_llm_input/<날짜>/`)

IPS/WAF 학습데이터를 만들 때 LLM에 넣는 파일이다. 통합 결과에서 필요한 필드만 남겨 레코드가 약 40% 작다 (평균 2.2KB → 1.3KB).

| 파일 | 내용 |
| --- | --- |
| `rules_all.jsonl` | 통합 룰 전체. 처음 한 번 학습데이터를 만들 때 사용 |
| `rules_delta.jsonl` | 그날 신규·변경·선별 추가 룰. 매일 추가분 |
| `batches/all/`, `batches/delta/` | 위 두 파일을 100개씩 나눈 것. **LLM 1회 호출 = prompt.md + 배치 파일 1개** |
| `prompt.md` | LLM 지시문 (룰마다 공격 3개 + 정상 2개 HTTP 메시지 생성, JSON Lines 출력) |
| `summary.json` | 개수, 공격 유형·요청/응답·소스별 분포 |

레코드 필드는 [LLM입력_형식.md](docs/코드참조/LLM입력_형식.md) 참고. 주요 필드: `id`, `msg`, `attack_type`(추정 공격 유형), `side`, `methods`, `conditions`(탐지 조건), `rule`(정답지).

## LLM 데이터 생성 (`3_0_run_generation.py`)

API 비용이 들어서 `1_0_run_daily.py`에 넣지 않고 따로 실행한다. `3_0`을 실행하면 아래 단계가 차례로 실행된다.
모델·샘플 수는 [LLM_설정.md](docs/코드참조/LLM_설정.md), 키는 `.env` (`UPSTAGE_API_KEY`, `OPENAI_API_KEY`, 나중에 `ANTHROPIC_API_KEY`).

```
3.1 준비(룰 3개씩 요청 파일) → 3.2 생성(Solar: 룰당 씨앗 anomalies 10 / normal 5) → 코드 검사
  → 3.3 검토(gpt-5: pass/fix/drop, 공격 유형 확정) → 코드 재검사
  → 3.4 증강(헤더·부가 파라미터 변형, 룰당 anomalies 50 / normal 10) → 1.4 IPS·WAF 로그
```

| 명령 | 설명 |
| --- | --- |
| `python 3_0_run_generation.py --dry-run` | 호출 없이 요청 준비와 건수만 확인 (비용 0) |
| `python 3_0_run_generation.py --generator mock --reviewer mock` | 키 없이 전체 흐름 시험 |
| `python 3_0_run_generation.py` | 변경분(delta) 요청 **1개**만 생성·검토·증강 (기본) |
| `python 3_0_run_generation.py --set all --max-requests 30` | 전체 중 아직 안 한 요청 30개 |
| `python 3_0_run_generation.py --max-requests 0` | 남은 요청 전부 (비용 주의) |
| `python 3_2_generate.py --max-requests 5` | 단계별로 따로 실행할 수도 있다 (3_1 ~ 3_4 각각) |

**속도 (2026-10-20)**: LLM 호출이 끝나기를 기다리는 시간이 대부분이라, 요청을 **동시에** 보내고 **생성된 요청부터 바로 검토**한다 (`LLM_설정.md` 의 `## 실행`).

| 설정 | 기본 | 내용 |
| --- | --- | --- |
| `생성 동시 요청 수` | 10 | Solar 에 동시에 보내는 요청 수의 **최대값** (1 = 예전처럼 하나씩). 4개부터 시작해 성공할 때마다 늘리고, 속도 제한(429)·과부하(503)면 반으로 줄인다 (자동 조절) |
| `검토 동시 요청 수` | 12 | gpt-5 에 동시에 보내는 요청 수의 최대값 (자동 조절은 같음) |
| `생성·검토 동시 진행` | 예 | 3.2 와 3.3 을 함께 돌림 (`3_3_review.py --follow <생성 PID>`). `아니오` = 생성을 다 한 뒤 검토 |
| `검토 추론 강도` | (비움) | gpt-5 의 `reasoning_effort`. `low` 면 검토가 더 빠르지만 정확도가 낮아질 수 있어 소량으로 비교 후 결정 |

- 실제: 요청 10개 약 7.5분, 100개 약 61분 (동시 3·4 일 때, 예전 방식이면 100개 약 7시간). 최대 10·12 로 올린 뒤에는 API 한도가 허락하면 남은 약 5,800개를 약 15시간에 처리하는 속도 (요청 1개 평균 생성 92초·검토 97초 기준).
- API 속도 제한(429)에 걸리면 기다렸다 다시 보내고(`재시도 횟수`와 따로 최대 6번) 동시 요청 수를 자동으로 줄인다. 실행을 시작하면 로그(`3_2_generate.log`, `3_3_review.log`)에 계정의 **API 한도**(분당 요청·토큰)가 남으니, 여유가 크면 최대값을 더 올린다.
- **요청 줄 자동 수정**: 생성 모델이 URI 에 공백을 그대로 쓰는 등 요청 줄 형식이 깨진 샘플(생성 샘플의 약 6%)은 공백을 `%20` 으로 고쳐 다시 검사한다 (실제 클라이언트가 보내는 형태). note 에 `[요청 줄 자동 수정]` 표시.
- **정규식 예시** (`정규식 예시` = 예): 생성 요청의 룰마다 정규식 조건에 실제로 맞는 짧은 예시를 코드로 만들어 `regex_examples` 로 붙인다. 생성 모델이 정규식을 잘못 읽어(예: XSS `<.+\/script` 에 보통의 `</script>`, RFI `=\s*(https?|ftps?|php)\:\/` 에 `data://`) 코드 검사에 실패하고 gpt-5 가 고치던 일을 줄인다. 효과는 화면의 '생성 품질'(코드 검사 통과율·검토가 고친 샘플 수)로 이전 실행과 비교한다.
- **요청 수 = 이번에 생성·검토할 요청 수**: 이전 실행에서 생성만 되고 검토 안 된 요청이 있으면 그것부터 검토하고 그만큼 덜 생성한다. 검토도 요청 수까지만 한다 (예전에는 대기분을 모두 검토해 15개를 요청해도 22개를 검토).
- **반복 루프 대책**: Solar 가 `%00%00…`, `..\..\…` 를 출력 한도까지 반복해 응답을 버리는 일이 잦아서(파일럿에서 요청의 약 30%), 긴 반복은 `{{REPEAT:문자열:횟수}}` 로 쓰게 하고 코드가 펼친다 (`LLM_프롬프트.md` 생성 규칙 11). 그래도 잘리면 **잘린 룰만** 반복 주의를 붙여 다시 요청한다 (예전에는 요청 전체를 다시).
- 각 단계는 이미 끝난 요청을 건너뛴다. 중간에 멈춰도 다시 실행하면 이어서 진행한다.
- **입력 날짜가 바뀌어도 이어서 진행한다.** 1_0(매일 수집)이 새 `05_llm_input/<날짜>` 를 만들면 3_0 은 새 `07_llm_output/<날짜>` 에서 일하는데,
  - 다른 날짜 폴더에서 이미 생성한 룰은 **룰 내용(원본 룰 텍스트)이 같으면 다시 만들지 않는다.** 내용이 바뀐 룰(rev 변경 등)만 다시 생성한다.
  - 지난 날짜 폴더에 남은 검토(3.3)·증강(3.4) 대기분도 함께 처리한다 (`--date` 를 주면 그 날짜만). 학습 로그는 날짜 폴더마다 `08_dataset/<날짜>/` 에 만든다.
  - 룰이 바뀌어 다시 생성하면 옛 버전 샘플은 그 날짜의 데이터셋에 그대로 남는다.
- 생성하지 않는 룰: `informational`(정보성), HTTP 응답 룰, **기존 학습 로그(06)에 이미 1000건 이상 있는 룰** (`LLM_설정.md`)
- 기존 샘플이 **1~50건인 룰**은 기존 payload 예시를 LLM 에 주고 비슷하지만 똑같지 않은 샘플을 만들게 하며, 3.4 에서 기존 데이터와 payload 가 똑같은 샘플은 뺀다.
- **부족한 만큼만 생성**: anomalies 목표(50)에서 기존 샘플 수를 뺀다 (예: 기존 10건 → 40개). `LLM_설정.md`의 `기존 데이터 차감`.
- **라벨은 룰 기준**: anomalies = 룰 조건을 모두 만족(탐지), normal = 조건을 하나 이상 어김(미탐지)인 샘플만 남긴다. 룰로 확인할 수 없는 샘플도 뺀다 (`판정 불가 처리`). method(공격 유형)는 로그 형식 때문에 기록만 하고 판단에는 쓰지 않는다.
- **증강(3.4)**: LLM 씨앗의 공격 부분은 그대로 두고, 룰 조건과 무관한 Host·User-Agent·부가 헤더·끝 파라미터만 바꾼다. 변형마다 코드 검사를 다시 통과한 것만 남기고 중복은 뺀다. normal 이 모자라면 공격 씨앗의 User-Agent 를 일반 브라우저 UA 로 바꿔 룰이 탐지하지 않는 것을 normal 로 쓴다 (스캐너·악성 UA 룰 등). 공격 기법의 다양성은 씨앗 수가 정하고, 증강은 맥락의 다양성을 더한다.
- 결과 확인: `07_llm_output/<날짜>/_2_generated|_3_reviewed|_4_augmented/<요청>.meta.json` (토큰 사용량, 샘플 수, 수정·제외 수, 목표 미달 룰), `_3_reviewed/<요청>.rejected.jsonl` (빠진 샘플과 이유)

## 담당자 검토용 샘플 (`10_review/`)

보안 담당자가 LLM 생성 품질을 확인하도록, **학습 데이터에 없는 룰**로 소량 샘플을 만든다. 코드와 결과를 이 폴더 하나에 둔다.
**본 파이프라인(1_0 매일 수집, 3_0 데이터 생성, 4_0 피처)은 이 코드를 실행하지 않고, 결과도 본 데이터셋(07 맨 위·08)에 들어가지 않는다.**

```powershell
python 10_review/review_sample.py --dry-run                                   # 고를 룰만 확인 (비용 0)
python 10_review/review_sample.py --rules talos:1:66993,et:1:2072460,custom:1:1007117 --anomalies 1 --normal 1
python 10_review/review_sample.py --name <폴더> --report-only                  # 저장된 결과로 문서만 다시 (비용 0)
```

- 결과: `10_review/<이름>/검토용_샘플.md` (담당자에게 전달), `samples.jsonl`, `ips_review.log`(8필드 예시), `_2_generated/`, `_3_reviewed/`
- 로그: `10_review/logs/review_sample.log` (검토용 코드·결과·로그를 모두 이 폴더에 둔다)
- 생성·검토는 3.2·3.3 과 같은 코드(헤더 현실화·교차 검사 포함)를 쓰고, 증강(3.4)은 하지 않는다.

## 화면 (`0_0_run_ui.py`)

`python 0_0_run_ui.py` 를 실행하면 브라우저에 http://127.0.0.1:8765 가 열린다 (이 PC 에서만 접속 가능).
8765 를 다른 프로그램(예: 다른 Streamlit 앱)이 쓰고 있으면 다음 빈 포트(8766 …)로 열고, 실행 창에 주소를 보여 준다.

| 페이지 | 하는 일 |
| --- | --- |
| 1. 진행 현황 (첫 화면) | **이번 실행 진행률**(아래) + 생성 대상 룰 중 **완료 %**, 현재 룰 중 **학습데이터 보유 %**(완료 + 기존 충분), 상태 플래그별 수, **어디까지**(배치 N까지 빠짐없이 생성), 출처·공격 유형별 %, 남은 요청·예상 시간·토큰 |
| 2. 생성·수집 | **학습데이터 생성**: 이번 실행 진행률, 건수 확인(호출 없음), 파일럿 5개, 생성 시작(세트·요청 수·입력 날짜·모델), **중지**(지금 요청까지 마치고), 강제 중지, 실행 로그<br>요청 수를 넣으면 **예상 시간·예상 토큰**(생성·검토 모델별 입력/출력)을 바로 보여 준다 (생성·검토는 지금까지 생성한 요청의 실제 평균(재시도 포함)과 동시 요청 수, 준비·마무리는 지난 실행 기록. 아직 생성을 돌리지 않은 입력 날짜면 기존 데이터 대조·요청 파일 만들기 시간을 더함. 0 = 남은 전부)<br>**수집**: 수집 에이전트 시작·중지(매일 정해진 시각, 놓친 수집은 바로 1회, 실패 시 1시간 뒤 재시도), 지금 1회 수집, 다음·마지막 수집 시각 |
| 3. 룰별 상태 | 상태 플래그·출처·검색으로 찾기, CSV 내려받기 |
| 4. 학습데이터 | 지금까지 만든 학습데이터(08_dataset)를 날짜(또는 전체)·장비(IPS/WAF)·label·룰 id·payload 검색·**label 정합성**으로 보기, 로그 내려받기<br>**log**: 수집 형식(`\|\|\|` 8필드) 로그 줄 그대로<br>**데이터 프레임**: `index`, `payload`, `method`, `label`, `label 사유`, `검증 결과`, `label 정합성`, `탐지 피처`<br>**전체 검증**: 파일의 모든 줄의 label 정합성을 계산해 요약(정합·정합(룰 기준)·확인 필요·불일치 수와 %)을 보여 주고, 정합성으로 걸러 볼 수 있게 한다 (결과는 `08_dataset/<날짜>/<장비>_verify.json`, 줄별 확인 결과는 `<장비>_frame.jsonl`. 로그나 룰 입력 날짜가 바뀌면 다시)<br>**내려받기**: log 보기 = **로그 내려받기**(그 날짜의 로그 파일 그대로). 데이터 프레임 보기 = **데이터 프레임 내려받기 (CSV)**: 지금 조건(날짜·장비·label·룰·검색·정합성)에 맞는 **모든 줄**을 표의 열 그대로 + `label 정합성 사유`·`rule_id`·`date` (UTF-8 BOM 이라 엑셀에서 바로 열림, payload 는 로그와 같은 한 줄). 전체 검증을 해 두면 저장된 줄별 결과로 바로 만들고(약 5천 줄 1초), 아니면 줄마다 다시 확인한다(1천 줄에 약 15초 - 1천 줄이 넘으면 먼저 물어봄) |
| 5. 피처 | ML 피처(4.1) 전체 37개 + 개별 룰 피처의 이름·뜻·예·값·**패턴**(4.1 코드에서 그대로 읽음), 묶음·검색으로 찾기<br>**켜진 줄 수**: 학습데이터에서 피처가 켜진 anomalies·normal 줄 수와 "켜졌을 때 anomalies 비율" (전체 검증과 같은 작업으로 계산) |

**이번 실행 진행률** (1·2 페이지, 메뉴의 `● 24%`, 브라우저 탭 제목 `(24%)`):

- 실행 중인 생성(3_0)을 건드리지 않고 **로그만 읽어** 계산한다 (`logs/3_0_run_generation.log`, `3_1a_existing_coverage.log`, `3_2_generate.log`, `3_3_review.log`, 새로 쓰인 부분만 읽음). 명령줄에서 시작한 실행도 보인다.
- **%** = 걸리는 시간 기준. 준비(3.1a·3.1) + 생성·검토 + 마무리(3.4·1.4·3.5).
  - 생성·검토: 남은 요청 수 × 이번 실행에서 **요청이 끝나는 실제 간격**(최근 20개, 동시 처리 반영. 끝난 것이 5개 미만이면 요청 1개 평균 시간 ÷ 지금 동시 수). 생성·검토를 함께 돌리면 검토는 '생성 끝 → 검토 끝' 시간 ÷ 동시 수로 잰다 (검토가 생성을 기다린 시간이 섞이지 않게).
  - 준비·마무리: **지난 실행 3번의 단계별 실제 시간**(중앙값, 증강은 요청 수에 비례). 새 입력 날짜의 첫 실행은 3.1a 가 기존 로그를 새로 대조하므로 그 진행 줄(`… N / M payload 대조`)로 잰다.
  - 처리 중인 요청·단계는 지난 시간만큼 채운다. 응답 해석 실패·재시도가 많으면 남은 시간이 늘어난다.
- 단계별 상태(✔ 끝남 · ● 진행 중 · ○ 대기 · – 건너뜀 · ✘ 실패·중단)와 생성 `n/N`·검토 `n/N`, **지금 무엇을 하는지**(예: `생성 12/25번째 요청 처리 중 · … 응답을 해석하지 못해 다시 요청 중`), 성공·실패 수, 요청당 시간, 사용 토큰(성공한 요청 기준), 남은 시간·끝날 시각.
- 끝난 뒤에는 마지막 실행의 결과(끝남 / 중단됨, 걸린 시간)를 보여 준다.

**피처 설명**: 4. 학습데이터의 `탐지 피처` 이름에 **마우스를 올리면** 그 피처의 뜻·예·이 줄에서 켜진 이유(룰 id 등)가 말풍선으로 나오고, **누르면** 5. 피처 페이지의 그 줄로 간다.

데이터 프레임의 열:

- **label 사유**: 어떤 룰의 탐지 조건을 모두 만족했는지(anomalies), 어떤 조건이 불일치했는지(normal) + 생성 메모
- **검증 결과**: 생성 때의 검증(코드 검사 결과, 검토 모델 판정 pass/fix, 증강 샘플 여부)과 **지금 현재 룰로 다시 확인한 결과**(anomalies: 조건 n/n 일치, normal: 이 룰에 탐지 안 됨 + 다른 공격 룰·공격 흔적 없음). 문제가 있으면 ✘ 로 표시하고 줄을 붉게 칠한다.
- **label 정합성** (라벨이 증거와 맞는지 + 그 사유):

  | 판정 | anomalies | normal |
  | --- | --- | --- |
  | 정합 | 룰 조건 모두 일치 + 범용 공격 흔적이나 다른 공격 룰도 공격으로 봄 | 룰 조건 불일치 + 공격 흔적·공격 룰 없음 |
  | 정합 (룰 기준) | 룰 조건은 모두 일치하지만 범용 공격 흔적·다른 공격 룰은 없음 → 이 룰의 고유 패턴(경로·파라미터·UA)으로만 공격. 정책성 룰(예: `ET POLICY …`)이 많으면 보안 관점에서 다시 볼 대상 | - |
  | 확인 필요 | 룰이 현재 목록에 없거나 판단 불가 조건(정규식 해석 불가) | 같음 |
  | 불일치 | 룰 조건 불일치 → 룰 기준으로 공격이 아님 | 룰에 탐지되거나 공격 흔적·공격 룰이 있음 → 보안 관점에서 정상이 아님 |
- **탐지 피처**: ML 피처(4.1) 중 이 payload 에서 켜지는 것과 값 — `k_` 공격 흔적·회피 신호, `ra_<유형>` 신뢰할 수 있는 공격 룰(모호한 룰 제외) 탐지 수와 룰 id, `r_<룰>_all`·`r_<룰>_re<n>` 이 룰·정규식 조건. normal 은 켜진 피처가 없어야 정상으로 볼 근거가 된다.
- 1.4 가 `*_label.jsonl` 에 줄마다 생성 단계 기록(origin, review, local_check, note)을 남기고, 화면은 보는 페이지의 줄만 읽어 다시 확인한다 (수십만 줄이어도 빠름).

- 작업은 별도 프로세스로 돈다. **화면을 닫아도 수집 에이전트·생성은 계속**되고, 다시 열면 이어서 보인다. 명령줄에서 시작한 작업도 보이고 중지할 수 있다.
  - 화면은 실행 창에서 **Ctrl+C** 로 끄고 다시 켜면 된다 (작업은 따로 돌아 영향 없음).
  - 화면에서 시작한 작업은 화면을 띄운 터미널(VS Code 등)에서 떼어 내 시작한다 (2026-10-19 부터). 그 전에 시작한 작업은 **화면을 띄운 터미널 창이나 VS Code 를 닫으면 같이 끝날 수 있으니** 작업이 끝날 때까지 닫지 않는다. 며칠 걸리는 생성은 일반 PowerShell 창에서 화면을 띄우는 것이 안전하다.
- **수집 에이전트는 켜 두는 것**이다: 매일 정해진 시각에 수집하고 그 사이에는 '대기' 한다 (화면: 파란 테두리 점 + '지금은 다음 수집까지 대기 중'). 수집 중일 때만 초록 점·메뉴의 실행 표시가 켜진다. 한 번만 수집하려면 '지금 1회 수집'.
- **수집 에이전트가 도는 중에도 학습데이터 생성을 함께 돌릴 수 있다.**
  - 1.3 은 입력을 임시 폴더에 다 쓴 뒤 한 번에 바꾼다 → 생성 쪽은 반쯤 쓴 입력을 읽지 않는다.
  - 3_0 은 시작할 때 고른 입력 날짜를 끝까지 쓰고, 이미 만든 룰은 날짜가 바뀌어도 다시 만들지 않는다.
  - 생성은 한 번에 하나만(`.run/generation.lock`), 수집도 한 번에 하나만(`.run/collect.lock`) 돈다. 두 번 시작하면 나중 것은 바로 끝난다.
  - 1.4(학습 로그 변환)는 두 쪽에서 동시에 돌아도 임시 파일이 겹치지 않는다. 결과 파일을 바꿀 때 다른 프로세스(다른 1.4·화면)가 잠깐 열고 있으면 기다렸다 다시 바꾼다 (Windows).

### 룰별 상태 플래그 (`07_llm_output/_progress/rule_status.csv`, `3_0_run_generation.py --status`)

| 플래그 | 상태 | 뜻 |
| --- | --- | --- |
| `DONE` | 완료 | 생성 → 검토 → 증강까지 끝나 학습 로그(08)에 들어감 |
| `REVIEWED` | 증강 대기 | 검토까지 끝남 (다음 3_0 실행 때 증강·로그 변환) |
| `GENERATED` | 검토 대기 | 생성까지 끝남 (다음 3_0 실행 때 검토) |
| `QUEUED` | 요청 대기 | 요청 파일에 있고 아직 생성 전 |
| `TODO` | 미생성 | 생성 대상인데 아직 요청도 없음 |
| `CHANGED` | 룰 변경 | 예전 내용으로 만들었는데 룰이 바뀜 → 다시 생성 대상 |
| `MISSING` | 응답 빠짐 | 생성 응답에 없었음 → 다음 실행 때 다시 요청 |
| `EMPTY` | 샘플 없음 | 검토·코드 검사에서 샘플이 모두 빠짐 → 확인 필요 |
| `FAILED` | 생성 실패 | 응답 해석 실패가 2번 반복돼 자동으로 건너뛰는 중 (`3_2_generate.py --force` 로 다시) |
| `EXISTING` | 기존 충분 | 기존 학습 로그에 충분(1000건 이상) → 생성하지 않음 |
| `EXCLUDED` | 제외 | 정보성·HTTP 응답 방향 룰 → 생성 대상 아님 |

- 완료율 = `DONE` / 생성 대상(전체 − `EXCLUDED` − `EXISTING`), 학습데이터 보유율 = (`DONE` + `EXISTING`) / 전체 룰
- '어디까지': 최신 입력의 `all` 요청을 배치 번호 순으로 보고, 앞에서부터 빠짐없이 생성이 끝난 배치 번호와 가장 뒤의 생성 배치를 보여 준다.

## 학습 로그 (`08_dataset/<날짜>/`)

**IPS·WAF 파서(`|||` 구분자, 8필드)가 읽는 형식**이다. 같은 샘플을 두 장비 로그로 함께 만든다.
기존 학습 로그(`06_기존 데이터/ips_*.log`, `waf_*.log`)와 같은 형식이다.

```
2026-10-15 00:00:00|||105.204.55.81|||62071|||192.168.20.20|||80|||Command Injection|||GET /gena.cgi?service=Layer3Forwarding%3Bid HTTP/1.1
Host: 192.168.20.20
...|||anomalies
```
- event_time, s_ip, s_port, d_ip, d_port, method, payload, reason(= label)
- IPS(`ips_<yymmdd>_snort.log`)와 WAF(`waf_<yymmdd>_snort.log`)의 차이는 d_port 기본값(IPS 80, WAF 8080)과 method 이름(장비별 탐지 이름)뿐이다.
- 두 장비 모두 마지막 필드에 label(anomalies/normal)을 쓴다. 줄별 원본 룰 id는 `*_label.jsonl`(line, id, label, attack_type, method)에 같은 줄 순서로 둔다.
- method는 매핑 md의 `IPS method 방식` / `WAF method 방식`으로 고른다: `mapping`(장비 탐지 이름) / `rule_name`(Snort 룰 이름) / 고정 태그.
- method·포트 설정은 [method_매핑.md](docs/코드참조/method_매핑.md). method 는 학습 정답이 아니며(정답은 label) 로그 형식상 채운다.
- 검사에서 빠진 샘플(라벨 오류, `|||` 포함, 매핑 제외 등)은 `rejected.jsonl`
- **문서용 주소 바꾸기** (2026-10-20, 매핑 md 의 `문서용 주소 바꾸기` = 예): 생성 지시문의 안전 규칙 때문에 LLM 출력에 들어간 문서용 IP(`192.0.2.x`·`198.51.100.x`·`203.0.113.x`)·`example.com`·`evil.php` 같은 자리표시 값을, 학습 로그를 만들 때 샘플마다 다른 공인 IP·도메인·파일 이름으로 바꾼다 (같은 샘플 안의 같은 값은 같은 값으로, 본문이 바뀌면 Content-Length 도). 실제 트래픽에는 없는 값이라 그대로 두면 모델이 '이 주소 = 공격' 으로 배울 수 있어서다. LLM 출력(07)은 그대로이고, 바꾼 뒤 룰 판정이 달라지면 원래대로 둔다. `summary.json` 의 `addr_fixed`, 줄별 `*_label.jsonl` 의 `addr_fixed` 에 바꾼 수.

## 소스별 파싱 결과 (`03_processed/<소스>/<날짜>/`)

| 파일 | 내용 |
| --- | --- |
| `llm_queue.jsonl` | 신규 + 변경 HTTP 룰. LLM 페이로드 생성의 입력 (한 줄 = 룰 1개) |
| `delta.json` | 신규 / 변경 / 삭제 룰 목록 (key, rev, msg) |
| `http_rules.jsonl` | HTTP 룰 전체 (ET 약 69MB, Talos 약 64MB). 다음 실행 때 비교 기준 |
| `summary.json` | 룰 수, 요청/응답 분포, method 분포, 파싱 오류 샘플 |

`llm_queue.jsonl`의 주요 필드:

| 필드 | 설명 |
| --- | --- |
| `key`, `sid`, `rev`, `msg` | 룰 식별 정보 |
| `side` | `request` / `response` / `unknown` |
| `methods`, `methods_excluded`, `method_hint` | 룰이 요구하는 method, 제외 method, 추정 method |
| `contents` | `value`(디코딩된 값), `buffer`(`http_uri` 등), `nocase`, `depth`, `offset`, `distance`, `within`, `negated` |
| `pcres` | `pattern`, `flags`, `buffer`, `negated` |
| `flow`, `classtype`, `cve` | 부가 정보 |
| `change` | `new` / `updated` |
| `rule` | 원본 룰 텍스트 |

`http_rules.jsonl` 읽기 (파이썬):
```python
import json
with open("03_processed/et/2026-10-16/http_rules.jsonl", encoding="utf-8") as f:
    rules = [json.loads(line) for line in f]
```

용량 참고: `http_rules.jsonl`은 룰이 바뀐 날마다 새 날짜 폴더에 쌓인다 (ET는 거의 매일, 하루 약 70MB). 오래된 날짜 폴더는 직접 지워도 되지만, `parse_state.json`이 가리키는 폴더는 **남겨 둔다**: `dir`(최신 결과, 다음 비교 기준)와 `baseline_dir`(같은 버전을 다시 파싱할 때의 비교 기준). 다른 단계(04·05·07)도 최신 날짜 폴더만 읽는다 (2026-10-19 정리 기록: 진행보고.md).

## 문제 해결

| 증상 | 확인할 것 |
| --- | --- |
| `[talos] SNORT_OINKCODE 미설정` | `.env` 파일 이름과 `SNORT_OINKCODE=` 줄 확인 |
| `HTTP 403` / `HTTP 422` (Talos) | OINKCODE 오류이거나 다운로드 횟수 제한. 잠시 후 재시도 |
| 다운로드 실패 반복 | 사내 프록시나 방화벽이 snort.org, emergingthreats.net을 막는지 확인 |
| 자동 실행이 안 됨 | 작업 스케줄러 → `SnortRuleFetch` → 마지막 실행 결과 확인 (0x0 정상, 0x1 실패). Windows에 로그인한 상태에서만 실행됨 |
| 파싱 코드를 고친 뒤 다시 돌리고 싶음 | `python 1_0_run_daily.py --skip-fetch --force` |
| 선별 조건을 바꾸고 싶음 | `docs/코드참조/선별조건.md`의 O/X 수정 후 `python 1_0_run_daily.py` (자동 재파싱) |
| `선별 조건 읽기 실패` 로그 | `docs/코드참조/선별조건.md`의 해당 줄 '선택' 열이 O 또는 X인지 확인 |
| LLM 지시문을 바꾸고 싶음 | `docs/코드참조/LLM_프롬프트.md` 수정 후 `python 1_3_build_llm_input.py --force` |
| 처음부터 다시 받고 싶음 | `state.json`, `parse_state.json`, `03_processed/` 삭제 후 실행 (`parse_state.json`만 지우면 `03_processed/`의 summary로 자동 복구됨) |
| `parse_state.json`이 비었거나 깨짐 | 자동 처리됨: `.broken`으로 옮기고 `03_processed/`의 summary로 복구한 뒤 계속 진행 |

## 주의 사항

- `.env`에는 개인 OINKCODE가 들어 있으므로 공유하거나 커밋하지 않는다. (`.gitignore`에 포함됨)
- Talos 룰은 재배포 제한이 있다. 내부 용도 외 공개 시 라이선스를 확인한다.
- Talos 무료(Registered) 룰은 유료 구독자보다 30일 늦게 공개된다.
- 처음 한 번은 `05_llm_input/<날짜>/rules_all.jsonl`(또는 `batches/all/`)로 학습데이터를 만들고, 이후에는 `rules_delta.jsonl`(매일 추가분)만 쓰면 된다.
