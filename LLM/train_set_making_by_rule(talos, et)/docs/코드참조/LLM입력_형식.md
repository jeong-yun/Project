# LLM 입력 레코드 형식 (05_llm_input)

`1_3_build_llm_input.py`가 만드는 `rules_all.jsonl`, `rules_delta.jsonl`, `batches/*/batch_NNNN.jsonl`의 한 줄(룰 1개) 형식이다.
코드를 고치거나 다음 단계(LLM 호출, 결과 검증) 코드를 만들 때 참조한다.

## 필드

| 필드 | 항상 있음 | 예 | 설명 |
| --- | :---: | --- | --- |
| `id` | O | `et:1:2072375` | `소스:gid:sid`. 결과를 원본 룰과 연결하는 키 |
| `source` | O | `et` / `talos` | 룰 출처 |
| `sid`, `rev` | O | `2072375`, `1` | Snort 룰 번호와 개정 번호 |
| `rule_name` | O | `ET WEB_SPECIFIC_APPS D-Link ...` | 룰 이름 (Snort msg) |
| `category` | O | `WEB_SPECIFIC_APPS`, `server-webapp` | 선별 카테고리 (ET: 룰 이름 두 번째 단어, Talos: 파일 이름) |
| `attack_type` | O | `command_injection` | 룰 이름 기반 **추정** 공격 유형 (아래 목록). 학습 라벨 참고용 |
| `classtype` | O | `attempted-admin` | Snort classtype |
| `cve` | | `["CVE-2024-23625"]` | 관련 CVE |
| `mitre` | | `{"tactic": "TA0001", "technique": "T1190", "technique_name": "..."}` | MITRE ATT&CK (ET만 제공) |
| `severity` | | `Major` | 심각도 (ET만 제공) |
| `enabled` | O | `true` | 기본 정책에서 켜진 룰인지 (Talos는 대부분 false) |
| `side` | O | `request` / `response` / `unknown` | HTTP 요청 룰인지 응답 룰인지 |
| `methods` | O | `["GET"]` / `["ANY"]` | 룰이 지정한 HTTP method. `["ANY"]` = 룰이 method를 검사하지 않음 (어떤 method든 가능) |
| `method_hint` | | `["POST"]` | `methods`가 `["ANY"]`일 때 가장 자연스러운 method (body 검사 있으면 POST, 없으면 GET) |
| `methods_excluded` | | `["POST"]` | 이 method면 탐지 안 됨 |
| `conditions` | O | 아래 참고 | 탐지 조건 (content + 정규식) |
| `urilen` | | `>200` | URI 길이 조건 |
| `requires_state` | | `["isset,ET.wikiloader"]` | 앞선 요청이 있어야 탐지됨 (flowbits) |
| `change` | | `new` / `updated` / `selection_added` | `rules_delta`에만 있음. 신규·변경·선별 조건 변경으로 추가 |
| `rule` | O | `alert tcp ...` | 원본 Snort 룰 (정답지) |

## conditions 항목

```json
{"type": "content", "in": "uri", "value": "/gena.cgi?", "position": {"depth": 10}}
{"type": "content", "in": "body", "value": "cmd=", "nocase": true}
{"type": "content", "in": "header", "value": "curl", "negated": true}
{"type": "regex", "in": "uri", "pattern": "service\\x3d[^\\x26]*?...", "flags": "U"}
```

| 키 | 설명 |
| --- | --- |
| `type` | `content`(문자열 포함) / `regex`(정규식, Snort pcre) |
| `in` | 검사 위치: `uri`, `raw_uri`, `header`, `raw_header`, `cookie`, `raw_cookie`, `body`, `method`, `status_code`, `status_msg`, `response_body`, `base64_data`, `payload` |
| `value` | 디코딩된 값. 출력할 수 없는 바이트는 `\xNN` |
| `nocase` | 대소문자 무시 (true일 때만 있음) |
| `negated` | 이 값이 없어야 함 (true일 때만 있음) |
| `position` | `depth`, `offset`, `distance`, `within` 중 있는 것만 |
| `pattern`, `flags` | 정규식과 플래그 (U=uri, H=header, P=body, i=대소문자 무시 등). `type: "regex"`에만 있음 |

## attack_type 값

`sql_injection`, `xss`, `xxe`, `ssrf`, `deserialization`, `code_injection`, `command_injection`, `file_inclusion`, `path_traversal`, `file_upload`, `webshell`, `auth_bypass`, `rce`, `buffer_overflow`, `info_disclosure`, `scanner`, `dos`, `csrf`, `open_redirect`, `informational`(공격 키워드 없는 ET INFO·POLICY 룰), `other`

룰 이름의 키워드를 위 순서대로 비교해 처음 맞는 값을 쓴다 (`1_3_build_llm_input.py`의 `ATTACK_TYPES`). 추정값이므로 최종 라벨이 필요하면 검증 단계에서 확인한다.

## 빠지는 룰

- `conditions`가 비어 있는 룰(content·정규식 없음): 페이로드를 만들 수 없어서 뺀다. 목록은 `summary.json`의 `skipped_no_condition`
- 중복 제거로 통합 결과에서 빠진 룰: `04_integrated/<날짜>/duplicates.json`

## 생성 결과의 label

LLM 출력은 룰 하나당 JSON 한 줄이다. 지시문은 `LLM_프롬프트.md`.

| 필드 | 설명 |
| --- | --- |
| `id` | 입력 룰의 `id` |
| `attack_type` | LLM이 **확정**한 공격 유형(IPS/WAF 공격 유형). 입력 추정값이 맞으면 그대로, `other`이거나 틀리면 목록에서 다시 고름 |
| `attack_type_note` | 입력 추정값과 다르게 정했을 때만 그 이유 |
| `samples[].label` | `anomalies`(그 룰에 탐지되는 공격) / `normal`(탐지되지 않는 정상) |
| `samples[].attack_type` | `anomalies`는 확정한 공격 유형, `normal`은 `none` |
| `samples[].http` | HTTP 메시지 원문 |
| `samples[].note` | 만족한 조건 / 일부러 어긴 조건 |

## 변경 이력

- 2026-10-15: `msg` → `rule_name`, 조건 `type: "pcre"` → `"regex"`, `method_inferred` + 추정값 → `methods: ["ANY"]` + `method_hint`, 생성 label `attack/benign` → `anomalies/normal`
- 2026-10-15: 생성 결과에 공격 유형(`attack_type`)을 룰 단위·샘플 단위로 추가. 입력의 추정값(`other` 포함)을 LLM이 확정
