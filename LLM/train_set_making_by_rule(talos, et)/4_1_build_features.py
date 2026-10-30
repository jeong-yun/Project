"""[4.1 피처 생성] IPS/WAF 학습 로그(8필드)의 payload → 표준 ML 탐지 모델용 피처 표(CSV).

[원칙]  (2026-10-16 결정)
  - payload 값만 쓴다. 시각·IP·포트·method 필드는 쓰지 않는다.
  - 표준 모델(여러 환경 공용)이므로 환경에 따라 달라지는 값은 쓰지 않는다 (Host IP, 헤더 수·길이, Cookie·Referer 유무 등).
  - **공격 또는 정상을 확실하게 나타내는 피처만** 쓴다. 모호한 것은 뺀다:
      × 길이·헤더 수·확장자·Content-Type·method (정상에도 흔함, 데이터 출처 편향의 원인이었음)
      × 특수문자 수·엔트로피·대소문자 비율 (& = % ' 등은 정상 요청에도 많음)
      × 정보성 룰 탐지, 확인 못 한 부분이 있는 룰 탐지(possible), 룰 조건 일부만 일치(경로만 일치 등)
      × 기존 데이터(06)의 normal 에도 걸리는 모호한 룰 (정밀도 90% 미만, --min-precision)
      × 판단 불가(-1) 값이 나오는 정규식

[피처 묶음]  (자세한 설계·시험 결과: docs/보고용/피처_설계.md)
  1. k_  공격 흔적 : 공격 유형별 키워드 수 (SQL, XSS, 명령, 경로 이동, 파일 포함, 템플릿, XXE, 역직렬화, Shellshock, 민감 파일)
                     + 회피·이상 신호 (이중 URL 인코딩, %u 인코딩, 제어문자·NULL 바이트, 스캐너 UA)
                     표준모델 피처(표준 34)와 같은 내용은 빼고 센다 (feature_values, FEATURE_PARTS - 2026-10-29).
                     라벨 검증(화면 4)·증강 normal 교차 검사(3.4)는 모든 갈래를 쓴다 (trace_features, KEYWORDS).
  2. ra_ 공격 룰   : 선별된 전체 HTTP 룰 중 '공격 룰'(정보성 제외)과 대조 → 공격 유형별 '모든 조건이 일치한 룰 수'
  3. r_  개별 룰   : --rules 로 지정한 룰의 '전체 일치'(룰이 탐지) + 정규식(pcre) 조건별 일치

[주의: 라벨 누수]  LLM 생성 데이터(08)의 라벨은 '그 룰이 탐지하는가' 로 정했다.
  생성 룰 자신의 r_·ra_ 피처는 라벨과 거의 같다. 룰 단위로 나눠 평가하거나(4_2 --group rule_id) 다른 출처 데이터로 평가한다.

[출력]  09_features/<이름>/   (이름 기본: 입력 날짜 폴더 이름, --name 으로 지정)
     ├─ features.csv       ← 한 줄 = 로그 한 줄. 앞 열: file, line, rule_id, payload_hash, label, y(anomalies=1) / 뒤: 피처
     ├─ feature_list.json  ← 피처 이름 → 묶음·설명
     └─ summary.json       ← 줄 수, 라벨 분포, 피처 수, 제외한 룰 수, 걸린 시간

[사용법]
    python 4_1_build_features.py                                   # 08_dataset 최근 날짜
    python 4_1_build_features.py --rules et:1:2053468               # 개별 룰 피처 추가
    python 4_1_build_features.py --input "06_기존 데이터/ips_260202_raw.log" --sample 30000 --name existing_ips --rules et:1:2053468
    python 4_1_build_features.py --no-rule-agg                      # 공격 룰(ra_) 피처 빼기 (빠름)
"""

import argparse
import collections
import csv
import datetime as dt
import importlib.util
import json
import random
import re
import sys
import threading
import time
import warnings
from pathlib import Path
from urllib.parse import unquote_plus

_spec = importlib.util.spec_from_file_location("llm_common", Path(__file__).resolve().parent / "3_0a_common.py")
C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)
warnings.filterwarnings("ignore", category=FutureWarning)

DATASET_DIR = C.BASE_DIR / "08_dataset"
FEATURE_DIR = C.BASE_DIR / "09_features"
SEP = "|||"

# 공격 흔적 키워드: 정상 요청에는 거의 나오지 않는 형태만 (디코딩한 payload 를 소문자로 보고 센다).
# 라벨 검증(화면 4)·증강 normal 교차 검사(3.4)가 쓰는 '흔적' 이다 (trace_features). ML 피처 k_ 는 아래 FEATURE_PARTS.
KEYWORDS = {
    "sql": r"union[\s/*+]+(?:all[\s/*+]+)?select\b|\bselect\b.{1,100}\bfrom\b.{1,100}\bwhere\b|"
           r"['\"]\s*(?:or|and)\s+['\"]?\w+['\"]?\s*(?:=|like)\s*['\"]?\w|\b(?:or|and)\s+\d+\s*=\s*\d+|"
           r"sleep\s*\(\s*\d|benchmark\s*\(|waitfor\s+delay|information_schema|xp_cmdshell|sp_password|sp_executesql|"
           r"\bchar\s*\(\s*\d+\s*\)|group_concat\s*\(|extractvalue\s*\(|updatexml\s*\(|load_file\s*\(|into\s+(?:out|dump)file|"
           r"['\"]\s*(?:--|#|/\*)|;\s*(?:drop|insert|update|delete|exec)\b",
    "xss": r"<script\b|</script>|javascript\s*:|vbscript\s*:|\bon(?:error|load|mouseover|focus|click|toggle|animationstart)\s*=|"
           r"<svg[^>]*\bon|<img[^>]+\bon\w+\s*=|<iframe\b|document\.(?:cookie|location|domain)|"
           r"\b(?:alert|prompt|confirm)\s*\(|string\.fromcharcode|<body[^>]+\bon",
    "cmd": r"(?:[;|`]|&&|\|\|)\s*(?:id|whoami|uname|cat|ls|wget|curl|nc|ncat|bash|sh|ping|nslookup|ifconfig|net\s+user)"
           r"(?![\w=])|\$\(\s*(?:id|whoami|uname|cat|ls|wget|curl|echo|sleep)\b|`\s*(?:id|whoami|uname|cat|ls|wget|curl)\b|"
           r"/bin/(?:ba)?sh\b|cmd(?:\.exe)?\s*/c\b|"
           r"powershell(?:\.exe)?\s+-|busybox\s|chmod\s+\+?[0-7x]+\s",
    "traversal": r"(?:\.\./){2,}|(?:\.\.\\){2,}|(?:%2e%2e|\.%2e|%2e\.)(?:%2f|/|%5c|\\)|%c0%ae|%252e%252e|"
                 r"/etc/(?:passwd|shadow)\b|[\\/](?:win|boot|system)\.ini\b|/proc/self/",
    "inclusion": r"(?:php|expect|zip|phar|glob)://|file:///|php://(?:input|filter)|data:(?:text|application)/[^,]*;base64,|"
                 r"[?&]\w+=(?:https?|ftp)://[^&\s]+\.(?:txt|php|sh|pl)\?",
    "template": r"\$\{jndi:|\$\{(?:env|sys|java|lower|upper|::-)|\{\{[^}]*(?:\.__|\(|\[|\*|config|self)[^}]*\}\}|"
                r"#\{[^}]*\(|%\{[^}]*\(|<%=\s*\w",
    "xxe": r"<!entity\b|<!doctype[^>]+\[\s*<!entity|\bsystem\s+['\"](?:file|http|php|expect)",
    "deserial": r"\bro0ab|aced0005|\bo:\d+:\"\w+\":\d+:\{|__proto__|constructor\[prototype\]|java\.lang\.runtime|"
                r"ysoserial|\"\$type\"\s*:|!!python/object",
    "shellshock": r"\(\)\s*\{\s*:\s*;\s*\}\s*;",
    "sensitive": r"/\.git/(?:config|head)|/\.env\b|wp-config\.php|web\.config\b|/\.aws/credentials|\bid_rsa\b|"
                 r"/\.htpasswd|/\.ssh/",
}
KEYWORD_RE = {k: re.compile(v, re.I | re.S) for k, v in KEYWORDS.items()}
UA_SCANNER = re.compile(r"sqlmap|nikto|nmap scripting|masscan|zgrab|nuclei|acunetix|nessus|openvas|wpscan|dirbuster|"
                        r"gobuster|w3af|arachni|netsparker|wfuzz|ffuf|jaeles|havij|commix|fimap", re.I)

# ML 피처 k_ (4.1 피처 표·화면 5·5.x 피처 실험): 표준모델 피처(표준 34)와 같은 내용은 넣지 않는다 (2026-10-29).
# 위 흔적 정규식을 갈래로 나눠 표준 피처와 대조했다 (11_supervised/experiments/03_피처겹침: 갈래 예시 문자열에서 표준 피처가
# 켜지는지 + 기존 40만·신규 61만 행에서 갈래가 맞은 샘플 중 같은 계열 표준 피처도 켜진 비율).
#   FEATURE_DROPPED  표준 피처가 그대로 봄 (예시에서 켜지고 데이터 99% 이상 함께 켜짐) → ML 피처에서 뺌. 라벨 검증에는 그대로 씀
#   "enc"            표준 피처에 같은 내용이 있지만, URL 인코딩되면 표준 피처(원문을 봄)가 못 봄
#                    (표준이 놓친 샘플이 1% 이상이고 그 대부분이 디코딩해야 보이는 것) → 디코딩해야만 보일 때만 센다
#   "new"            표준 피처에 없는 내용 → 모두 센다
# (패턴, 처리). php:// 처럼 한 갈래 안에 표준에 있는 것·없는 것이 섞이면 나눴다.
# 어느 표준 피처가 같은 내용을 보는지는 표준 피처 이름을 코드에 적지 않으려고 로컬 파일(OVERLAP_FILE, 분석이 만듦)에 둔다.
FEATURE_PARTS = {
    "sql": [
        (r"['\"]\s*(?:or|and)\s+['\"]?\w+['\"]?\s*(?:=|like)\s*['\"]?\w", "enc"),
        (r"\b(?:or|and)\s+\d+\s*=\s*\d+", "enc"),
        (r"benchmark\s*\(", "enc"),
        (r"\bchar\s*\(\s*\d+\s*\)", "enc"),
        (r";\s*(?:drop|insert|update|delete|exec)\b", "enc"),
        (r"xp_cmdshell|sp_password|sp_executesql", "new"),         # xp_cmdshell 은 표준이 일부 모양만 봄 (원문에서도 23% 놓침)
        (r"group_concat\s*\(|updatexml\s*\(|load_file\s*\(", "new"),     # load_file( 은 표준이 원문에서도 3.5% 놓침
        (r"into\s+(?:out|dump)file", "new"),
        (r"['\"]\s*(?:#|/\*)", "new"),
    ],
    "xss": [
        (r"javascript\s*:", "enc"),
        (r"\bon(?:error|load|mouseover|focus|toggle|animationstart)\s*=", "enc"),
        (r"<img[^>]+\bon\w+\s*=|<body[^>]+\bon", "enc"),
        (r"<iframe\b", "enc"),
        (r"\b(?:alert|prompt|confirm)\s*\(", "enc"),
        (r"<script\b|</script>|vbscript\s*:|\bonclick\s*=", "new"),
        (r"document\.(?:cookie|location|domain)|string\.fromcharcode", "new"),
    ],
    "cmd": [   # 명령 연결: 표준이 원문에서 보는 명령(cat·wget·whoami…)은 인코딩만, 표준에 없는 명령(id·ls·bash…)은 새 내용
        (r"/bin/(?:ba)?sh\b", "enc"),
        (r"chmod\s+\+?[0-7x]+\s", "enc"),
        (r"(?:[;|`]|&&|\|\|)\s*(?:whoami|uname|cat|wget|curl|nc|ping|nslookup|net\s+user)(?![\w=])", "enc"),
        (r"\$\(\s*(?:whoami|uname|cat|wget|curl|echo)\b|`\s*(?:whoami|uname|cat|wget|curl)\b", "enc"),
        (r"(?:[;|`]|&&|\|\|)\s*(?:id|ls|ncat|bash|sh|ifconfig)(?![\w=])", "new"),
        (r"\$\(\s*(?:id|ls|sleep)\b|`\s*(?:id|ls)\b", "new"),
        (r"cmd(?:\.exe)?\s*/c\b|busybox\s", "new"),
    ],
    "traversal": [
        (r"(?:%2e%2e|\.%2e|%2e\.)(?:%2f|/|%5c|\\)", "enc"),
        (r"[\\/](?:win|boot|system)\.ini\b", "enc"),
        (r"/proc/self/", "enc"),
    ],
    "inclusion": [
        (r"file:///|php://(?:input|filter)", "enc"),
        (r"(?:zip|glob)://|php://(?!input|filter)", "new"),
        (r"data:(?:text|application)/[^,]*;base64,|[?&]\w+=(?:https?|ftp)://[^&\s]+\.(?:txt|php|sh|pl)\?", "new"),
    ],
    "template": [
        (KEYWORDS["template"], "new"),                                  # 표준은 일부 모양만 봄 (데이터 덮임 66~80%)
    ],
    "xxe": [
        (KEYWORDS["xxe"], "new"),
    ],
    "deserial": [
        (r"\bo:\d+:\"\w+\":\d+:\{", "enc"),
        (r"\bro0ab|aced0005|__proto__|constructor\[prototype\]|ysoserial|\"\$type\"\s*:|!!python/object", "new"),
    ],
    "shellshock": [
        (KEYWORDS["shellshock"], "new"),
    ],
    "sensitive": [
        (r"/\.ssh/", "enc"),
        (r"web\.config\b", "new"),                                      # 표준 34 에는 없음 (신규 후보 9 의 취약 경로 사전에는 있음)
    ],
}
FEATURE_DROPPED = {     # 흔적 갈래 중 표준 피처가 그대로 보는 것
    "sql": [r"union[\s/*+]+(?:all[\s/*+]+)?select\b", r"\bselect\b.{1,100}\bfrom\b.{1,100}\bwhere\b", r"sleep\s*\(\s*\d",
            r"waitfor\s+delay", r"information_schema", r"extractvalue\s*\(", r"['\"]\s*--"],
    "xss": [r"<svg[^>]*\bon"],
    "cmd": [r"powershell(?:\.exe)?\s+-"],
    "inclusion": [r"(?:expect|phar)://"],
    "traversal": [r"(?:\.\./){2,}", r"(?:\.\.\\){2,}", r"%c0%ae", r"%252e%252e", r"/etc/(?:passwd|shadow)\b"],
    "deserial": [r"java\.lang\.runtime"],
    "sensitive": [r"/\.git/(?:config|head)", r"/\.env\b", r"wp-config\.php", r"/\.aws/credentials", r"\bid_rsa\b", r"/\.htpasswd"],
}
UA_DROPPED = ["sqlmap", "nikto", "nmap scripting", "masscan", "zgrab", "nuclei", "wpscan", "dirbuster", "gobuster", "arachni", "ffuf"]
UA_SCANNER_FEATURE = re.compile("|".join(t for t in UA_SCANNER.pattern.split("|") if t not in UA_DROPPED), re.I)
OVERLAP_FILE = FEATURE_DIR / "_feature_overlap.json"     # 갈래·도구 → 같은 내용을 보는 표준 피처 (03_피처겹침/notes.py 가 만듦)


def overlap_notes():
    """화면 4·5 표시용 로컬 대응표 (표준 피처는 짧은 이름 = 장비_payload_ 를 뺀 것). 파일이 없으면 빈 사전.
    parts: 갈래 패턴 → 표준 피처, ua: 도구 → 표준 피처, families: add_* → 같은 계열 표준 피처,
    generic: 장비 → 범용 표준 피처 (기준 정상의 10% 이상에서 켜짐 → 줄별 표시에서 뺌),
    base_rates: 장비 → 표준 피처 → [기준 공격 %, 기준 정상 %], titles: 표준 피처 → 한 줄 설명."""
    try:
        d = json.loads(OVERLAP_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        d = {}
    return {k: d.get(k) or {} for k in ("parts", "ua", "families", "generic", "base_rates", "titles")}


def std_short(name, dev):
    """표준 피처 열 이름 → 짧은 이름 (로컬 대응표의 키)."""
    p = f"{dev}_payload_"
    return name[len(p):] if name.startswith(p) else name


def _join_re(parts, how):
    rx = "|".join(p for p, h in parts if h == how)
    return re.compile(rx, re.I | re.S) if rx else None


FEATURE_RE = {k: (_join_re(v, "new"), _join_re(v, "enc")) for k, v in FEATURE_PARTS.items()}
EVASION = {
    "double_enc": "기호의 이중 URL 인코딩 (%252e %252f %253c 등)", "unicode_enc": "%u 유니코드 인코딩",
    "ctrl_bytes": "제어문자·NULL 바이트 (\\x00 등, 줄바꿈·탭·파일 업로드 본문 제외)", "scanner_ua": "스캐너·공격 도구 User-Agent",
}
ATTACK_TYPES = ["sql_injection", "xss", "xxe", "ssrf", "deserialization", "code_injection", "command_injection",
                "file_inclusion", "path_traversal", "file_upload", "webshell", "auth_bypass", "rce", "buffer_overflow",
                "info_disclosure", "scanner", "dos", "csrf", "open_redirect", "other"]
NON_ATTACK_NAME = re.compile(r"\saccess$", re.I)       # '... access' 접근 기록 룰


# ── 1. 공격 흔적 ────────────────────────────────────────────────────────

def decode(s):
    """URL 디코딩 두 번 (이중 인코딩까지). 실패하면 원문."""
    try:
        return unquote_plus(unquote_plus(s))
    except Exception:
        return s


def trace_features(http):
    """라벨 검증·증강 normal 교차 검사용 공격 흔적 (모든 갈래). ML 피처는 feature_values."""
    text = decode(http).lower()
    f = {f"k_{k}": len(rx.findall(text)) for k, rx in KEYWORD_RE.items()}
    f.update(_evasion(http, UA_SCANNER))
    return f


def feature_values(http):
    """ML 신규 피처 add_* (FEATURE_NAMES): 표준모델 피처가 못 잡는 것만 센다 (2026-10-29: k_ → add_ 로 이름을 바꿈).
    "new" 갈래는 디코딩한 payload 에서 맞은 수, "enc" 갈래는 디코딩해야만 보이는 수(디코딩 뒤 − 원문), 스캐너 UA 는 표준에 없는 도구만."""
    text, low = decode(http).lower(), http.lower()
    f = {}
    for k, (new, enc) in FEATURE_RE.items():
        n = len(new.findall(text)) if new else 0
        if enc:
            n += max(0, len(enc.findall(text)) - len(enc.findall(low)))
        f[f"add_{k}"] = n
    f.update({"add_" + k[2:]: v for k, v in _evasion(http, UA_SCANNER_FEATURE).items()})
    return f


FEATURE_NAMES = [f"add_{k}" for k in FEATURE_PARTS] + ["add_double_enc", "add_unicode_enc", "add_ctrl_bytes", "add_scanner_ua"]
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")     # 글자 코드 < 32, 줄바꿈·탭(\n \r \t) 제외


def _evasion(http, ua_re):
    """회피·이상 신호 4개 (원문 payload 에서). 스캐너 UA 는 ua_re (흔적 = 모든 도구, ML 피처 = 표준에 없는 도구)."""
    f = {}
    low = http.lower()
    f["k_double_enc"] = len(re.findall(r"%25(?:2e|2f|5c|3c|3e|27|22|00|3b|7c)", low))   # 기호를 이중 인코딩
    f["k_unicode_enc"] = len(re.findall(r"%u[0-9a-f]{4}", low))
    # 제어문자·NULL 바이트: 파일 업로드(multipart) 본문의 바이너리는 정상 파일 내용이므로 세지 않는다
    # (요청 줄·헤더·일반 본문에 들어간 NULL 바이트 등만 확실한 이상 신호)
    head, _, body = http.partition("\n\n")
    scope = head if "multipart/form-data" in head.lower() else http
    f["k_ctrl_bytes"] = len(_CTRL_RE.findall(scope)) + len(re.findall(r"\\x(?:0[0-8bcef]|1[0-9a-f])|%00", scope.lower()))
    ua = next((l.split(":", 1)[1] for l in http.split("\n")[1:] if l.lower().startswith("user-agent:")), "")
    f["k_scanner_ua"] = int(bool(ua_re.search(ua)))
    return f


# ── 2~3. 룰 피처 ────────────────────────────────────────────────────────

def is_attack_rule(r):
    """공격 룰인지: 정보성 유형·ET 심각도 Informational·접근 기록 룰은 공격 판정이 아니다."""
    return (r.get("attack_type") != "informational" and r.get("severity") != "Informational"
            and not NON_ATTACK_NAME.search(r.get("rule_name", "")))


def rule_agg_features(matcher, types, http):
    """공격 룰과 대조 → 공격 유형별 '모든 조건이 일치한 룰 수'. 확인 못 한 부분이 있는 탐지(possible)는 세지 않는다."""
    f = {f"ra_{t}": 0 for t in ATTACK_TYPES}
    for rid, kind in matcher.match(http):
        if kind == "match":
            f[f"ra_{types.get(rid, 'other')}"] += 1
    f["ra_total"] = sum(f.values())
    return f


def rule_slug(rid):
    return re.sub(r"[^0-9A-Za-z]+", "_", rid)


def single_rule_plan(rule):
    """개별 룰 피처 계획: 전체 일치 + 정규식 조건(해석 가능한 것만). → [(피처 이름, 조건 번호 또는 None)]"""
    slug = rule_slug(rule["id"])
    plan = [(f"r_{slug}_all", None)]
    for i, c in enumerate(rule["conditions"]):
        if c["type"] == "regex" and not c.get("negated"):
            flags = sum(fl for ch, fl in (("i", re.I), ("s", re.S), ("m", re.M), ("x", re.X)) if ch in c.get("flags", ""))
            if C._compile(c["pattern"], flags) is not None:
                plan.append((f"r_{slug}_re{i}", i))
    return plan


def single_rule_features(rule, plan, http):
    parts = C.split_http(http)
    res = [C.check_condition(c, parts) for c in rule["conditions"]]
    out = {}
    for name, i in plan:
        out[name] = int(all(r is True for r in res)) if i is None else int(res[i] is True)
    return out


def describe(name, rules_by_id):
    """feature_list.json 용 설명."""
    if name.startswith("add_"):
        k = name[4:]
        if k in EVASION:
            return "신규: 회피·이상 신호", EVASION[k] + (" (표준 피처에 없는 도구만)" if k == "scanner_ua" else "")
        return "신규: 표준이 못 잡는 공격 흔적", "|".join(p for p, _ in FEATURE_PARTS[k])[:120]
    if name.startswith("k_"):
        k = name[2:]
        return ("라벨 검증 흔적", EVASION[k]) if k in EVASION else ("라벨 검증 흔적", KEYWORDS[k][:120])
    if name.startswith("ra_"):
        return "공격 룰", ("모든 공격 유형 합계" if name == "ra_total" else f"{name[3:]} 공격 룰 중 모든 조건이 일치한 룰 수")
    for rid, r in rules_by_id.items():
        slug = rule_slug(rid)
        if name.startswith(f"r_{slug}_"):
            tail = name[len(slug) + 3:]
            if tail == "all":
                return "개별 룰", f"{rid} 탐지 (모든 조건 일치) - {r['rule_name']}"
            c = r["conditions"][int(tail[2:])]
            return "개별 룰", f"{rid} 정규식 조건{tail[2:]} 일치: {c['pattern'][:90]}"
    return "기타", name


def rule_quality(attack, log, min_precision=0.9):
    """공격 룰마다 기존 학습 로그(06)의 normal 에 몇 번 걸리는지 세고, '확실한 공격 룰'만 고른다.

    정밀도 = 기존 anomalies 탐지 수 / (anomalies 탐지 수 + normal 탐지 수)  (anomalies 수는 3.1a 의 coverage.json)
    normal 에 한 번이라도 걸리고 정밀도가 min_precision 미만이면 '모호한 룰' 로 빼낸다.
    기존 데이터에 한 번도 안 나온 룰은 판단할 근거가 없으므로 그대로 둔다 (공격 룰로 분류된 룰).
    결과는 09_features/_rule_quality.json 에 저장하고, 입력(룰·06 로그)이 같으면 다시 계산하지 않는다.
    """
    logs = sorted(C.BASE_DIR.joinpath("06_기존 데이터").glob("*.log"))
    in_dir = C.latest_input_dir(None)
    cov_file = C.output_dir(in_dir.name) / "_0_existing" / "coverage.json"
    if not logs or not cov_file.exists():
        log.warning("[4.1] 기존 데이터(06) 또는 3.1a 결과가 없어 룰 신뢰도 확인을 건너뜀")
        return {r["id"] for r in attack}, {}
    key = C.payload_hash("|".join(sorted(r["id"] + r.get("rule", "") for r in attack))
                         + "|".join(f"{p.name}:{p.stat().st_size}" for p in logs))
    cache = FEATURE_DIR / "_rule_quality.json"
    if cache.exists():
        q = json.loads(cache.read_text(encoding="utf-8"))
        if q.get("key") == key:
            normal_hits = q["normal_hits"]
        else:
            q = None
    else:
        q = None
    if q is None:
        normals = set()
        for p in logs:
            with open(p, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    f = line.rstrip("\n").split(SEP)
                    if len(f) == 8 and f[7] == "normal":
                        normals.add(f[6])
        log.info("[4.1] 룰 신뢰도: 기존 normal 고유 payload %d개와 공격 룰 대조 (처음 한 번, 몇 분 걸림)", len(normals))
        matcher, normal_hits, t0 = C.RuleMatcher(attack), collections.Counter(), time.time()
        for i, pl in enumerate(normals, 1):
            for rid, kind in matcher.match(C.norm_payload(pl)):
                if kind == "match":
                    normal_hits[rid] += 1
            if i % 50000 == 0:
                log.info("  ... %d / %d (%.0f초)", i, len(normals), time.time() - t0)
        FEATURE_DIR.mkdir(exist_ok=True)
        cache.write_text(json.dumps({"key": key, "normal_payloads": len(normals), "normal_hits": dict(normal_hits)},
                                    ensure_ascii=False), encoding="utf-8")
    cov = json.loads(cov_file.read_text(encoding="utf-8"))
    keep, dropped = set(), {}
    for r in attack:
        n, a = normal_hits.get(r["id"], 0), cov.get(r["id"], {}).get("count", 0)
        if n and a / (a + n) < min_precision:
            dropped[r["id"]] = {"anomalies": a, "normal": n, "rule_name": r["rule_name"]}
        else:
            keep.add(r["id"])
    return keep, dropped


# ── 입력 ───────────────────────────────────────────────────────────────

def read_logs(paths, sample, seed):
    """8필드 로그 → [(file, line, payload, label, rule_id)]. payload(7번째 필드)와 label 만 쓴다.
    *_label.jsonl 이 옆에 있으면 rule_id 를 붙인다 (평가 때 룰 단위로 나누는 데만 사용)."""
    rows = []
    for p in paths:
        rid_by_line = {}
        lab = p.with_name(p.stem + "_label.jsonl")
        if lab.exists():
            for l in open(lab, encoding="utf-8"):
                j = json.loads(l)
                rid_by_line[j["line"]] = j["id"]
        with open(p, encoding="utf-8", errors="replace") as fh:
            for n, line in enumerate(fh, 1):
                f = line.rstrip("\n").split(SEP)
                if len(f) != 8 or f[7] not in ("anomalies", "normal"):
                    continue
                rows.append((p.name, n, f[6], f[7], rid_by_line.get(n, "")))
    if sample and len(rows) > sample:
        rows = random.Random(seed).sample(rows, sample)
    return rows


# ── 화면 표시용 피처 (화면 4 줄별 '탐지 피처'·화면 5 켜진 줄 수) ─────────────────
# 표준모델 피처(표준 34: 이름·정규식은 실행할 때 표준모델 피처 파일에서 읽는다 - 공개 코드에 적지 않음) + 신규 피처 add_*.
# 표준 피처 값은 화면 6 비교(5.5)의 피처 캐시(payload 지문 → 값)를 다시 쓰고, 캐시에 없는 payload 만 계산한다.
#   python 4_1_build_features.py --view ips   → 09_features/_view/ips.parquet (지문 → 표준·add_ 값), ips_counts.json (켜진 줄 수)
VIEW_DIR = FEATURE_DIR / "_view"
VIEW_LOCK = "features_view"
_mods = {}
_mods_lock = threading.Lock()


def _sup_mods():
    """5.0a(버전·신규 학습 데이터 읽기)·5.0b(표준모델 피처 파일 읽기·계산). pandas 를 쓰므로 필요할 때만 읽는다.
    잠금 안에서 둘 다 읽은 뒤 한꺼번에 넣는다: 화면을 켠 직후 요청 여러 개가 함께 와도 한 번만 읽고, 반쯤 찬 표를 보지 않는다
    (2026-10-30: 5.0b 를 읽는 몇 초 사이에 온 두 번째 요청이 KeyError 'F5' → 화면 5 첫 로드 'Failed to fetch')."""
    with _mods_lock:
        if not _mods:
            got = {}
            for key, f in (("C5", "5_0a_common.py"), ("F5", "5_0b_features.py")):
                spec = importlib.util.spec_from_file_location(f"view_{key}", Path(__file__).resolve().parent / f)
                m = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)
                got[key] = m
            _mods.update(got)
        return _mods["C5"], _mods["F5"]


def _legacy(dev):
    C5, F5 = _sup_mods()
    code = C5.feature_code(C5.load_versions(), dev)
    return code, F5.load_legacy(code)


def std_features(dev):
    """표준모델 피처 [(열 이름, 종류 comb·count·ratio, 정규식 목록)] (표준모델 피처 파일에서, 5.0b 와 같은 이름순)."""
    _, F5 = _sup_mods()
    _, ns = _legacy(dev)
    out = []
    for kind in ("comb", "count", "ratio"):
        out += [(n, kind, list(rx)) for n, rx in ns[f"regex_dict_{kind}"].items()]
    order = {n: i for i, n in enumerate(F5.legacy_names(ns))}
    return sorted(out, key=lambda x: order[x[0]])


def view_compute(raws, dev):
    """학습 로그 payload 필드 목록 → 표준 34 + add_* 값 (DataFrame, 행 순서 그대로). 5.5 와 같은 payload 변환."""
    import pandas as pd
    _, F5 = _sup_mods()
    _, ns = _legacy(dev)
    texts = [F5.raw_to_payload(r) for r in raws]
    std = F5.generate_fast(pd.Series(texts), ns).reset_index(drop=True)
    add = pd.DataFrame([feature_values(C.norm_payload(t)) for t in texts], columns=FEATURE_NAMES)
    return pd.concat([std, add], axis=1)


def view_cache(dev):
    """표시 캐시 (payload 지문 pk → 표준 34·add_* 값). 없으면 None."""
    import pandas as pd
    p = VIEW_DIR / f"{dev}.parquet"
    return pd.read_parquet(p).set_index("pk") if p.exists() else None


def view_features(raws, dev, cache=None, hide=()):
    """줄별 표시 피처 [[(열 이름, 값)]]: 켜진 표준 피처(hide = 범용 피처 제외) + 켜진 add_*.
    cache(view_cache) 에 있는 payload 는 그 값을 쓰고, 없는 것만 계산한다 (화면 한 페이지 100줄 약 0.3초)."""
    C5, _ = _sup_mods()
    keys = [C5.payload_key(r) for r in raws]
    vals = [None] * len(raws)
    if cache is not None:
        hit = [i for i, k in enumerate(keys) if k in cache.index]
        for i, r in zip(hit, cache.loc[[keys[i] for i in hit]].itertuples(index=False)):
            vals[i] = r
        cols = list(cache.columns)
    miss = [i for i, v in enumerate(vals) if v is None]
    if miss:
        X = view_compute([raws[i] for i in miss], dev)
        if cache is not None:
            X = X[cols]
        cols = list(X.columns)
        for i, r in zip(miss, X.itertuples(index=False)):
            vals[i] = r
    return [[(c, float(x)) for c, x in zip(cols, r) if x and c not in hide] for r in vals]


def build_view(dev, log):
    """신규 학습 데이터(08) 모든 줄의 표시 캐시와 켜진 줄 수 (화면 5 '켜진 줄 수 다시 계산'). 이미 계산한 payload 는 건너뜀."""
    import pandas as pd
    spec = importlib.util.spec_from_file_location("runtime_v", Path(__file__).resolve().parent / "0_0a_runtime.py")
    RT = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(RT)
    holder = RT.acquire_lock(VIEW_LOCK, by="4_1", dev=dev, step="시작")
    if holder:
        log.error("[4.1] 표시용 피처 계산이 이미 돌고 있습니다: %s", holder)
        return 1
    try:
        C5, F5 = _sup_mods()
        code, ns = _legacy(dev)
        std_cols = F5.legacy_names(ns)
        cols = std_cols + FEATURE_NAMES
        parts, dates = [], []
        for d in C5.dataset_dates():
            RT.update_lock(VIEW_LOCK, step=f"읽기 {d}")
            got, st = C5.read_dataset(d, dev)
            if st is None:
                continue
            dates.append(d)
            parts.append(pd.DataFrame({"label": got["label"], "raw": got["payload_raw"]}))
        R = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["label", "raw"])
        R["pk"] = [C5.payload_key(p) for p in R["raw"]]
        uniq = R.drop_duplicates("pk")
        old = view_cache(dev)
        if old is not None and list(old.columns) != cols:          # 표준모델 피처 파일이나 add_* 가 바뀜 → 처음부터
            old = None
        todo = uniq[~uniq["pk"].isin(old.index)] if old is not None else uniq
        fdir = C5.CACHE / "features" / dev / F5.code_hash(code)
        fparts = sorted(fdir.glob("part-*.parquet"))
        std_cache = (pd.concat([pd.read_parquet(p, columns=["pk"] + std_cols) for p in fparts], ignore_index=True)
                     .drop_duplicates("pk", keep="last").set_index("pk") if fparts else None)
        log.info("[4.1] 표시용 피처 %s: %d줄, payload %d개 중 새로 계산 %d개 (5.5 피처 캐시 %d개)", dev, len(R), len(uniq), len(todo),
                 0 if std_cache is None else len(std_cache))
        new, t0 = [], time.time()
        for s in range(0, len(todo), 5000):
            ch = todo.iloc[s:s + 5000]
            raws = ch["raw"].tolist()
            texts = [F5.raw_to_payload(r) for r in raws]
            inc = ch["pk"].isin(std_cache.index).to_numpy() if std_cache is not None else [False] * len(ch)
            std = pd.DataFrame(0.0, index=range(len(ch)), columns=std_cols)
            if any(inc):
                std.loc[[i for i, b in enumerate(inc) if b], std_cols] = std_cache.loc[ch["pk"][inc]].to_numpy()
            if not all(inc):
                rest = [i for i, b in enumerate(inc) if not b]
                std.loc[rest, std_cols] = F5.generate_fast(pd.Series([texts[i] for i in rest]), ns)[std_cols].to_numpy()
            add = pd.DataFrame([feature_values(C.norm_payload(t)) for t in texts], columns=FEATURE_NAMES)
            new.append(pd.concat([pd.DataFrame({"pk": ch["pk"].to_numpy()}), std.astype("float32"), add.astype("int16")], axis=1))
            RT.update_lock(VIEW_LOCK, step="계산", done=s + len(ch), total=len(todo))
            log.info("  ... %d / %d (%.0f초)", s + len(ch), len(todo), time.time() - t0)
        V = pd.concat(([old.reset_index()] if old is not None else []) + new, ignore_index=True)
        V = V[V["pk"].isin(set(uniq["pk"]))].drop_duplicates("pk", keep="last")
        VIEW_DIR.mkdir(parents=True, exist_ok=True)
        tmp = VIEW_DIR / f"{dev}.parquet.tmp"
        V.to_parquet(tmp, index=False)
        tmp.replace(VIEW_DIR / f"{dev}.parquet")
        M = R[["pk", "label"]].merge(V, on="pk", how="left")
        y = M["label"].eq("anomalies").to_numpy()
        on = M[cols].fillna(0).gt(0).to_numpy()
        out = {"dev": dev, "dates": dates, "lines": len(R), "labels": {"anomalies": int(y.sum()), "normal": int((~y).sum())},
               "counts": {c: [int((on[:, j] & y).sum()), int((on[:, j] & ~y).sum())] for j, c in enumerate(cols)},
               "computed": len(todo), "computed_at": dt.datetime.now().isoformat(timespec="seconds")}
        (VIEW_DIR / f"{dev}_counts.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        log.info("[4.1] 표시용 피처 %s 끝 → %s (%.0f초)", dev, VIEW_DIR, time.time() - t0)
        return 0
    finally:
        RT.release_lock(VIEW_LOCK)


def main(argv=None):
    ap = argparse.ArgumentParser(description="학습 로그 payload → 표준 ML 피처 표")
    ap.add_argument("--input", nargs="*", help="로그 파일들 (기본: 08_dataset 최근 날짜의 *_snort.log)")
    ap.add_argument("--date", help="08_dataset 날짜 폴더 (기본: 가장 최근)")
    ap.add_argument("--name", help="출력 폴더 이름 (기본: 날짜 또는 첫 입력 파일 이름)")
    ap.add_argument("--rules", help="개별 룰 피처를 만들 룰 id (쉼표 구분, 예: et:1:2053468)")
    ap.add_argument("--no-rule-agg", action="store_true", help="공격 룰(ra_) 피처를 만들지 않는다")
    ap.add_argument("--min-precision", type=float, default=0.9,
                    help="ra_ 에 쓸 룰의 최소 정밀도 (기존 데이터 기준, 기본 0.9). 0 이면 신뢰도 확인 안 함")
    ap.add_argument("--sample", type=int, default=0, help="무작위로 이 줄 수만 (0 = 전부)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--view", choices=("ips", "waf"),
                    help="화면 표시용 피처만 (표준모델 피처 + add_*, 신규 학습 데이터 모든 줄) → 09_features/_view/")
    args = ap.parse_args(argv)
    log = C.setup_logging("4_1_build_features")
    if args.view:
        return build_view(args.view, log)
    started = time.time()

    if args.input:
        paths = [Path(p) if Path(p).is_absolute() else C.BASE_DIR / p for p in args.input]
        name = args.name or paths[0].stem
    else:
        days = sorted(p for p in DATASET_DIR.iterdir()
                      if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name)) if DATASET_DIR.is_dir() else []
        day = DATASET_DIR / args.date if args.date else (days[-1] if days else None)
        if day is None or not day.is_dir():
            log.error("08_dataset 에 날짜 폴더가 없습니다 (먼저 3_0_run_generation.py)")
            return 1
        paths = sorted(day.glob("*_snort.log"))
        name = args.name or day.name
    missing = [p for p in paths if not p.exists()]
    if missing or not paths:
        log.error("입력 로그가 없습니다: %s", missing or "(없음)")
        return 1
    rows = read_logs(paths, args.sample, args.seed)
    log.info("[4.1] 입력 %s → %d줄", ", ".join(p.name for p in paths), len(rows))

    in_dir = C.latest_input_dir(None)
    all_rules = C.read_jsonl(in_dir / "rules_all.jsonl") if in_dir else []
    by_id = {r["id"]: r for r in all_rules}
    picked, plans = {}, {}
    for rid in (args.rules.split(",") if args.rules else []):
        rid = rid.strip()
        if rid not in by_id:
            log.error("룰 %s 가 %s 에 없습니다", rid, in_dir)
            return 1
        if not is_attack_rule(by_id[rid]):
            log.error("룰 %s 는 정보성 룰이라 공격 피처로 쓰지 않습니다 (%s)", rid, by_id[rid]["rule_name"])
            return 1
        picked[rid], plans[rid] = by_id[rid], single_rule_plan(by_id[rid])
        log.info("[4.1] 개별 룰 %s: 피처 %s", rid, [n for n, _ in plans[rid]])
    matcher, attack, dropped = None, [r for r in all_rules if is_attack_rule(r)], {}
    if not args.no_rule_agg and attack:
        if args.min_precision > 0:
            keep, dropped = rule_quality(attack, log, args.min_precision)
            log.info("[4.1] 룰 신뢰도: 기존 normal 에도 걸리는 모호한 룰 %d개 제외 (정밀도 %.0f%% 미만)",
                     len(dropped), args.min_precision * 100)
            attack = [r for r in attack if r["id"] in keep]
        matcher = C.RuleMatcher(attack)
        types = {r["id"]: (r.get("attack_type") if r.get("attack_type") in ATTACK_TYPES else "other") for r in attack}
        log.info("[4.1] 공격 룰 피처: 전체 %d개 중 공격 룰 %d개 (정보성 %d개 제외) → 대조 가능 %d개",
                 len(all_rules), len(attack), len(all_rules) - len(attack), len(matcher.rules))

    out_dir = FEATURE_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = None
    with open(out_dir / "features.csv.tmp", "w", encoding="utf-8", newline="") as fh:
        w = None
        for i, (fname, line, payload, label, rid) in enumerate(rows, 1):
            http = C.norm_payload(payload)
            f = feature_values(http)
            if matcher:
                f.update(rule_agg_features(matcher, types, http))
            for r in picked.values():
                f.update(single_rule_features(r, plans[r["id"]], http))
            head = {"file": fname, "line": line, "rule_id": rid, "payload_hash": C.payload_hash(payload),
                    "label": label, "y": int(label == "anomalies")}
            if w is None:
                columns = list(f)
                w = csv.writer(fh)
                w.writerow(list(head) + columns)
            w.writerow(list(head.values()) + [f[c] for c in columns])
            if i % 20000 == 0:
                log.info("  ... %d / %d줄 (%.0f초)", i, len(rows), time.time() - started)
    (out_dir / "features.csv.tmp").replace(out_dir / "features.csv")

    flist = {c: dict(zip(("group", "desc"), describe(c, picked))) for c in (columns or [])}
    (out_dir / "feature_list.json").write_text(json.dumps(flist, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {"generated_at": dt.datetime.now().isoformat(timespec="seconds"),
               "inputs": [str(p.relative_to(C.BASE_DIR)) if p.is_relative_to(C.BASE_DIR) else str(p) for p in paths],
               "rows": len(rows), "label": dict(collections.Counter(r[3] for r in rows)),
               "features": len(columns or []),
               "groups": dict(collections.Counter(v["group"] for v in flist.values())),
               "rules": list(picked), "attack_rules": len(attack) if matcher else 0,
               "excluded_rules": (len(all_rules) - len(attack)) if matcher else 0,
               "excluded_ambiguous_rules": dropped,
               "seconds": round(time.time() - started, 1)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("[4.1] %s: %d줄 × 피처 %d개 %s (%.0f초)", out_dir, len(rows), summary["features"], summary["groups"],
             summary["seconds"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
