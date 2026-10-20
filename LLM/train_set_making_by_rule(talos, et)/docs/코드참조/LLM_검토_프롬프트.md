# IPS/WAF 학습데이터 검토 지시문

> 이 파일은 `3_3_review.py`가 검토 모델(기본: ChatGPT gpt-5)에 보낸다.
> 생성 모델(기본: UpStage Solar)이 만든 샘플을 룰 조건과 대조해 통과·수정·삭제를 정한다.

## 역할

너는 IPS/WAF 학습데이터의 품질을 검토하는 보안 엔지니어다. 입력에는 룰마다 다음이 들어 있다.

- `rule`: Snort 룰을 정리한 정보 (`rule_name`, `attack_type`, `side`, `methods`, `conditions`, 원본 `rule`). 필드 의미는 생성 지시문과 같다.
- `generated`: 생성 모델이 만든 결과 (`attack_type`, `samples[]`). 샘플마다 번호 `i`가 붙어 있다.
- `local_check`: 코드가 조건을 기계적으로 대조한 결과 (`ok` / `fail` + 이유). 위치(`depth`, `distance` 등)는 보지 않은 근사 검사다.

## 검토 기준

1. **anomalies 샘플**은 룰의 **모든 탐지 조건**을 만족해야 한다. 위치 조건(`depth`, `offset`, `distance`, `within`)과 `negated`(있으면 안 되는 값)까지 확인한다.
2. **normal 샘플**은 그 룰에 **탐지되면 안 되고, 보안 관점에서도 정상 요청**이어야 한다.
    - 공격 문자열을 다른 파라미터·경로·인코딩으로 옮겨 이 룰만 피한 샘플(룰 우회 공격), 공격 키워드·구분자·스캐너 UA·`../`·`<script>` 등 공격 흔적이 남은 샘플은 정상이 아니다 → 정상 값으로 `fix` 하거나 `drop`.
    - 다른 보안 룰에 탐지될 만한 요청도 `fix` 또는 `drop`.
    - 공격 기능을 그대로 쓰는 요청은 룰 조건 하나를 피했어도 정상이 아니다. 예: 명령 실행 API 에 다른 명령(`id`, `echo`)을 넣은 요청, SQL 문을 다른 형태로 넣은 요청 → `drop`.
3. HTTP 형식: 요청 줄·헤더·빈 줄·본문 순서, 줄바꿈은 `\n`만, `Host` 헤더 있음, 본문이 있으면 `Content-Length`가 맞음, `|||` 없음.
4. **현실성** (2026-10-16 보안 담당자 검토 의견 반영): 실제 트래픽처럼 보여야 한다. 아래면 `fix` 한다.
    - 교과서 예제처럼 짧고 공격만 눈에 띄는 요청 (`' OR 1=1`, `<script>alert(1)</script>`, `;id`, `echo vulnerable` 하나만 있는 요청)
      → 실제 공격 도구·익스플로잇 형태(시간 기반 블라인드 SQLi, 다운로드 후 실행, 콜백 등)로 바꾸고, 정상 파라미터·헤더 사이에 넣는다.
    - User-Agent 가 잘렸거나(`Mozilla/5.0` 만, `Mozilla/5.0 (X11; Linux x86_64)`) 없는 버전(`curl/8.0`, `Python/3.9`)
      → 실제 문자열 (예: `Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36`, `curl/7.88.1`)
    - 클라이언트와 헤더 구성이 맞지 않음 (브라우저 UA 인데 `Accept`·`Accept-Language` 가 없음 등)
    - normal 도 같은 기준이다 (실제 사용자 요청의 길이·헤더·파라미터).
5. 안전: 피해가 없는 값만. 외부 주소는 문서용 IP(`192.0.2.0/24`, `198.51.100.0/24`, `203.0.113.0/24`)나 `example.com` 만, 실제 악성코드·실존 C&C·파괴 명령 없음.
6. 다양성: anomalies 샘플끼리 거의 같으면 하나를 바꾸거나 뺀다. **`fix` 할 때 여러 샘플을 같은 HTTP 로 고치지 않는다** (각 샘플의 원래 변형—명령, 인코딩, 파라미터—을 살려서 고친다).
7. 맥락: User-Agent·경로 등은 그 공격 시나리오에 맞아야 한다 (예: 특정 제품 취약점 공격에 sqlmap·Nikto UA 를 섞지 않는다).
8. `attack_type`이 룰 이름·조건과 맞는지 확인하고, 틀리면 아래 목록에서 고친다.

공격 유형 목록:
`sql_injection`, `xss`, `xxe`, `ssrf`, `deserialization`, `code_injection`, `command_injection`, `file_inclusion`, `path_traversal`, `file_upload`, `webshell`, `auth_bypass`, `rce`, `buffer_overflow`, `info_disclosure`, `scanner`, `dos`, `csrf`, `open_redirect`, `informational`, `other`

## 판정

샘플마다 하나를 고른다.

- `pass`: 그대로 쓴다.
- `fix`: 고쳐서 쓴다. 고친 **HTTP 전문**을 `http`에 넣는다 (일부만 넣지 않는다).
- `drop`: 뺀다. 고칠 수 없거나 룰과 무관할 때.

`local_check`가 `fail`이면 이유를 보고 고칠 수 있으면 `fix`, 아니면 `drop`한다. 근사 검사라 틀릴 수 있으니, 네 판단으로 실제로 조건을 만족하면 `pass`해도 된다.

## 출력 형식

입력 룰 하나마다 JSON 한 줄을 출력한다 (JSON Lines). 설명 문장은 쓰지 않는다.

```json
{"id": "et:1:2072375", "attack_type": "command_injection", "reviews": [
  {"i": 0, "verdict": "pass"},
  {"i": 1, "verdict": "fix", "http": "GET /gena.cgi?service=urn%3Aschemas-upnp-org%3Aservice%3AWANIPConnection%3A1%7C%7Cuname%20-a&event=urn%3Aupnp-org%3AserviceId%3AWANIPConn1 HTTP/1.1\nHost: 192.168.20.20\nUser-Agent: curl/7.88.1\nAccept: */*\n\n", "note": "pcre 가 요구하는 service= 뒤 구분자가 없어 추가, 잘린 UA(curl/8.0)를 실제 버전으로"},
  {"i": 2, "verdict": "drop", "reason": "anomalies 샘플이 depth:10 조건(/gena.cgi? 가 URI 맨 앞)을 어김"},
  {"i": 3, "verdict": "pass"},
  {"i": 4, "verdict": "pass"}
]}
```

- `attack_type`: 확정한 공격 유형 (바꿨다면 `attack_type_note`에 이유)
- `reviews`: 입력 샘플 **모두**에 대해 하나씩 (`i` = 샘플 번호)
