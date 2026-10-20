# method 매핑과 IPS/WAF 로그 형식 설정

**이 파일은 `1_4_build_dataset.py`가 읽는다.** LLM이 만든 샘플을 IPS/WAF 학습 로그로 바꿀 때 쓰는 값이다.
표를 고치고 `python 1_4_build_dataset.py --force`를 실행하면 반영된다.

| 장비 | 파서 | 형식 |
| --- | --- | --- |
| IPS | 구분자 `\|\|\|`, 8필드 | event_time, s_ip, s_port, d_ip, d_port, method, payload, reason(=label) — 기존 학습 로그(`06_기존 데이터/ips_*.log`)와 같음 |
| WAF | 구분자 `\|\|\|`, 8필드 | IPS 와 같음 (d_port 기본 8080, method 는 WAF 이름) — 기존 학습 로그(`06_기존 데이터/waf_*.log`)과 같음 |

- 매핑 표의 method 이름은 예시다. 쓰는 장비의 탐지 이름으로 바꾼다.
- `제외`로 적으면 그 공격 유형의 샘플은 로그에 넣지 않는다.
- 표 형식(`|` 구분)과 제목(`## 설정`, `## 매핑`)은 지우지 않는다.

## 설정

| 항목 | 값 | 설명 |
| --- | --- | --- |
| IPS method 방식 | mapping | `mapping` = 아래 표의 IPS method (기존 학습 로그의 탐지 이름) / `rule_name` = Snort 룰 이름 그대로 (예: `ET WEB_SPECIFIC_APPS D-Link gena.cgi ... Command Injection Attempt`) / 그 외 값 = 모든 anomalies를 그 값으로 통일 (예: `26_snort_rule`) |
| WAF method 방식 | mapping | 위와 같음 (WAF method) |
| normal method | normal | normal 샘플의 method (기존 학습 로그의 정상 = `normal`) |
| 기본 목적지 IP | 192.168.10.31 | Host 헤더가 IP가 아닐 때 쓸 목적지 IP |
| IPS 포트 | 80 | IPS 로그의 기본 목적지 포트 (기존 IPS 로그의 99%) |
| WAF 포트 | 8080 | WAF 로그의 기본 목적지 포트 (기존 WAF 로그의 99%) |
| payload 줄바꿈 | \n | payload 안의 줄바꿈 표기. `\n` = 두 글자 `\n`(기존 학습 로그와 같음), `space` = 공백 하나 |
| 문서용 주소 바꾸기 | 예 | `예` = 학습 로그를 만들 때 생성 지시문의 안전용 자리표시 값(문서용 IP `192.0.2.x`·`198.51.100.x`·`203.0.113.x`, `example.com` 등, `evil.php`·`malicious.com` 같은 이름)을 샘플마다 다른 실제 같은 공인 IP·도메인·파일 이름으로 바꾼다. 실제 트래픽에는 없는 값이라 그대로 두면 모델이 '이 주소 = 공격' 으로 배울 수 있다. LLM 출력(07)은 그대로, 바꾼 뒤 룰 판정이 달라지면 원래대로. `아니오` = 바꾸지 않음 |

## 매핑

`WAF rule_name`·`위험도` 열은 예전 WAF 23필드 형식용으로, 8필드로 바뀐 뒤에는 쓰지 않는다 (참고용으로 남겨 둠).

| attack_type | IPS method | WAF method | WAF rule_name | 위험도 |
| --- | --- | --- | --- | --- |
| sql_injection | SQL Injection | SQL Injection | SQL 인젝션 | 상 |
| xss | Cross Site Scripting | Cross Site Scripting | 크로스 사이트 스크립트 | 중 |
| command_injection | Command Injection | Command Injection | 명령 주입 | 상 |
| code_injection | Code Injection | Code Injection | 코드 주입 | 상 |
| deserialization | Deserialization | Deserialization | 역직렬화 | 상 |
| rce | Remote Code Execution | Remote Code Execution | 원격 코드 실행 | 상 |
| path_traversal | Path Traversal | Path Traversal | 경로 이동 | 상 |
| file_inclusion | File Inclusion | File Inclusion | 파일 포함 | 상 |
| file_upload | File Upload | File Upload | 파일 업로드 | 상 |
| webshell | Web Shell | Web Shell | 웹셸 | 상 |
| auth_bypass | Authentication Bypass | Authentication Bypass | 인증 우회 | 중 |
| info_disclosure | Information Disclosure | Information Disclosure | 정보 노출 | 중 |
| scanner | Scanner | Scanner | 스캐너 | 하 |
| buffer_overflow | Buffer Overflow | Buffer Overflow | 버퍼 오버플로 | 상 |
| dos | Denial of Service | Denial of Service | 서비스 거부 | 중 |
| csrf | CSRF | CSRF | CSRF | 중 |
| ssrf | SSRF | SSRF | SSRF | 상 |
| xxe | XXE | XXE | XXE | 상 |
| open_redirect | Open Redirect | Open Redirect | 오픈 리다이렉트 | 하 |
| other | Other Attack | Other Attack | 기타 공격 | 중 |
| informational | 제외 | 제외 | | |
