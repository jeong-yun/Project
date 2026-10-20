# LLM 설정 (생성 모델 / 검토 모델 / 샘플 수)

**이 파일은 3_x 단계(`3_0_run_generation.py` 와 `3_1`~`3_4`)가 읽는다.** 명령줄 옵션(`--generator`, `--reviewer`)을 주면 그 값이 우선한다.

- API 키는 이 파일이 아니라 **`.env`**에 넣는다 (`UPSTAGE_API_KEY=...`, `OPENAI_API_KEY=...`). 이 파일은 공유해도 된다.
- 모델 이름은 계정에서 쓸 수 있는 이름으로 바꾼다 (**확인 필요**).
- 표 형식(`|` 구분)과 제목(`## 모델`, `## provider`, `## 실행`)은 지우지 않는다.

## 모델

| 역할 | provider | model | temperature | max_tokens | 설명 |
| --- | --- | --- | --- | --- | --- |
| 생성 | upstage | solar-pro2 | 0.8 | 12000 | 룰 → 씨앗 샘플 생성 (`LLM_프롬프트.md`). 다양성을 위해 temperature 를 조금 높게 |
| 검토 | openai | gpt-5 | | 32000 | 생성 결과 검토·수정 (`LLM_검토_프롬프트.md`). gpt-5 는 추론 토큰도 max_tokens 에 포함되므로 넉넉하게. provider 를 `none` 으로 하면 검토 생략 |

- `temperature`를 비워 두면 보내지 않는다 (gpt-5 같은 추론 모델은 temperature 를 지원하지 않음).
- 검토 비용을 줄이고 싶으면 `gpt-5-mini`로 바꾼다 (같은 계열의 작은 모델, 더 싸고 빠르지만 판단 정확도가 낮을 수 있음).
- 나중에 Claude 를 쓰려면 provider 를 `anthropic`, model 을 예: `claude-sonnet-5-5`로 바꾸고 `.env`에 `ANTHROPIC_API_KEY`를 넣는다.

## provider

| provider | API 주소 | 키 이름 (.env) | 형식 |
| --- | --- | --- | --- |
| upstage | https://api.upstage.ai/v1/chat/completions | UPSTAGE_API_KEY | openai |
| openai | https://api.openai.com/v1/chat/completions | OPENAI_API_KEY | openai |
| anthropic | https://api.anthropic.com/v1/messages | ANTHROPIC_API_KEY | anthropic |
| mock | - | - | mock |

- `형식`: `openai` = OpenAI 호환 Chat Completions (UpStage Solar도 이 형식), `anthropic` = Claude Messages API, `mock` = 키 없이 흐름만 시험하는 가짜 응답

## 실행

| 항목 | 값 | 설명 |
| --- | --- | --- |
| 생성 anomalies 수 | 10 | LLM 이 룰마다 만드는 공격 **씨앗** 샘플 수. 공격 기법·인코딩의 다양성은 이 수가 결정 |
| 생성 normal 수 | 5 | LLM 이 룰마다 만드는 정상(오탐 학습용) 씨앗 샘플 수 |
| 목표 anomalies 수 | 50 | 3.4 증강 후 룰마다 목표 공격 샘플 수 (헤더·부가 파라미터 변형으로 채움, LLM 비용 없음) |
| 목표 normal 수 | 10 | 3.4 증강 후 룰마다 목표 정상 샘플 수 |
| 기존 데이터 충분 기준 | 1000 | 기존 학습 로그(06)에서 이 룰에 탐지되는 샘플이 이 수 이상이면 **생성하지 않는다** (3.1a 가 계산) |
| 기존 데이터 참고 기준 | 50 | 기존 샘플이 1 ~ 이 수인 룰은 기존 payload 예시를 LLM 에 주고 **비슷하지만 똑같지 않은** 샘플을 만들게 한다 |
| 참고 예시 수 | 3 | 위 경우 LLM 에 줄 기존 payload 예시 수 |
| 호출당 룰 수 | 3 | LLM 한 번에 넣을 룰 수. 룰당 출력 ≈ 씨앗 15개 × 약 200토큰 = 3천 토큰 → 3개면 약 9천 토큰 |
| 생성 제외 attack_type | informational | 이 공격 유형의 룰은 LLM 을 부르지 않는다 (쉼표로 여러 개). `method_매핑.md`에서 `제외`인 유형과 맞춘다. 비우려면 `-` |
| 생성 대상 side | request | 이 방향의 룰만 생성 (`request`, `response`, `unknown`). 기존 학습 로그(06)는 IPS·WAF 모두 요청만 있음 |
| 기존 데이터 차감 | 예 | `예` = anomalies 목표에서 기존 학습 로그(06)의 이 룰 샘플 수를 뺀다 (부족한 만큼만 생성). 예: 기존 10건 → 40개. 기존이 목표 이상이면 증강 없이 LLM 씨앗만 |
| 로컬 검사 실패 처리 | drop | 라벨은 **룰 기준**: anomalies = 룰 조건을 모두 만족(탐지), normal = 조건을 하나 이상 어김(미탐지). `drop` = 어긋난 샘플은 뺌, `keep` = 남기고 표시만 |
| normal 교차 검사 | 예 | `예` = normal 샘플을 선별된 **전체** HTTP 룰과 대조해 다른 공격 룰이 탐지하거나, 공격 흔적(`;uname`, `UNION SELECT`, `<script`, `../../` 등 4.1 의 `k_` 피처)이 있으면 뺀다 (이 룰만 피한 '룰 우회 공격'을 정상으로 학습하지 않게) |
| 교차 검사 제외 attack_type | informational | 교차 검사에서 무시할 룰 유형 (예: Host 가 IP 인 요청을 기록하는 정보성 룰). 쉼표로 여러 개 |
| 교차 검사 제외 룰 이름 | \saccess$ | 교차 검사에서 무시할 룰 이름 (정규식). 기본: `... access` 로 끝나는 접근 기록 룰 (예: `SERVER-WEBAPP modules.php access`) |
| 헤더 현실화 | 예 | `예` = 생성·검토 결과의 잘린 User-Agent(`Mozilla/5.0` 만)·없는 버전(`curl/8.0`)을 실제 문자열로 바꾸고, 클라이언트 기본 헤더(브라우저: Accept·Accept-Encoding·Accept-Language·Connection)와 정확한 Content-Length 를 넣는다. 바꾼 뒤 룰 판정이 달라지면 원래대로 둔다 (2026-10-16 검토 의견) |
| 정규식 예시 | 예 | `예` = 생성 요청의 룰마다 정규식 조건에 **실제로 맞는 짧은 예시**(코드가 만들고 확인)를 `regex_examples` 로 붙인다. 생성 모델이 정규식을 잘못 읽어 코드 검사에 실패하는 일을 줄인다 (예: XSS 룰 52개의 `<.+\/script` 는 보통의 `</script>` 와 맞지 않음, RFI 룰 58개의 `=\s*(https?\|ftps?\|php)\:\/` 에 `data://` 를 씀). `아니오` = 붙이지 않음 (비교용) |
| 판정 불가 처리 | drop | 룰 조건을 코드로 확인할 수 없는 샘플(정규식 해석 불가). `drop` = 뺌 (룰이 탐지하는지 확인된 것만 남김), `keep` = 남김 |
| 재시도 횟수 | 3 | API 오류(5xx·네트워크)·JSON 형식 오류 시 다시 시도. 생성 응답이 출력 한도에서 잘리면(반복 루프) 잘린 룰만 다시 요청. 속도 제한(429)은 이 횟수와 따로 최대 6번 기다렸다 다시 보냄 |
| 응답 대기 시간 | 600 | 초 (gpt-5 는 추론 시간이 길 수 있음) |
| 생성 동시 요청 수 | 10 | 생성 모델에 **동시에** 보내는 요청 수의 **최대값** (1 = 예전처럼 하나씩). 4개부터 시작해 성공할 때마다 1개씩 늘리고, 속도 제한(429)·과부하(503)가 오면 반으로 줄인 뒤 2분 동안 늘리지 않는다 (자동 조절, 로그의 '동시 요청 a → b'). 시작할 때 로그의 'API 한도'(계정의 분당 요청·토큰 한도)를 보고 정한다 |
| 검토 동시 요청 수 | 12 | 검토 모델에 동시에 보내는 요청 수의 최대값 (자동 조절은 위와 같음). gpt-5 는 요청마다 '입력 + max_tokens' 를 분당 토큰 한도에서 미리 잡으므로, 429 가 자주 나면 이 값이나 검토 max_tokens 를 줄인다 |
| 생성·검토 동시 진행 | 예 | `예` = 3.2(생성)와 3.3(검토)을 함께 돌려 **생성된 요청부터 바로 검토**. `아니오` = 생성을 모두 마친 뒤 검토 (예전 방식) |
| 검토 추론 강도 |  | 추론 모델(gpt-5 등)의 `reasoning_effort`: `minimal`·`low`·`medium`·`high`. 비우면 보내지 않음 (모델 기본 = `medium`). `low` 로 하면 검토가 더 빨라지지만 판단 정확도가 낮아질 수 있어, 소량으로 비교해 보고 정한다. OpenAI 검토 모델에만 적용 |
