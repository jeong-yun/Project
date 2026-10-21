"""[1.4 데이터셋] LLM 출력 → IPS/WAF 학습 로그으로 변환.  (1_0_run_daily.py 가 실행)

[IPS 로그 형식] 구분자 '|||' 파서, 필드 8개 (0~7)  ← 기존 학습 로그(06_기존 데이터/ips_*.log)와 같은 형식
    [0]event_time(yyyy-MM-dd HH:mm:ss) [1]s_ip [2]s_port [3]d_ip [4]d_port [5]method [6]payload [7]reason(=label)
    2026-01-01 00:26:20|||203.0.113.50|||32656|||192.168.10.30|||80|||Cross Site Scripting|||GET /board/view.php?... HTTP/1.1\\nHost: ...|||anomalies

[WAF 로그 형식] IPS 와 같은 8필드 (기존 학습 로그 waf_*.log 와 같음)
    [0]event_time [1]s_ip [2]s_port [3]d_ip [4]d_port(기본 8080) [5]method(WAF 이름) [6]payload [7]reason(=label)
    2026-01-01 00:13:40|||198.51.100.60|||60742|||192.168.10.30|||8080|||normal|||POST /login.php HTTP/1.1\nHost: ...|||normal

[payload] HTTP 메시지를 한 줄로. 줄바꿈은 두 글자 '\\n' (기존 학습 로그와 같음. 매핑 md 에서 공백으로 바꿀 수 있음)
  요청 줄(첫 줄)의 글자 역슬래시 뒤에 n 이나 제어문자 표기(x00~x1f, x7f)가 오면 역슬래시를 '%5C' 로 적는다
  (그대로 두면 로그를 읽을 때 줄바꿈·제어문자로 읽혀 요청 줄이 잘림. 실제 로그에서도 URL 의 역슬래시는 흔히 %5C).

[label]
  - IPS·WAF 모두 마지막 필드(reason)에 anomalies / normal 을 직접 쓴다.
  - 두 장비 모두 *_label.jsonl 에 줄별로 원본 룰 id·attack_type·method 와 생성 단계 검증 기록(origin: llm/augmented,
    review: 검토 판정, local_check: 코드 검사, note: 생성 메모)을 남긴다 (추적·화면의 학습데이터 페이지용).
  - sk: 샘플 지문 (룰 id·샘플 번호·LLM 출력 http). 다시 변환해도 같은 샘플이면 같다 → 화면의 수정·삭제가 이것으로 샘플을 찾음.

[수정·삭제] 화면(4. 학습데이터)에서 저장한 것을 08_dataset/<날짜>/edits.json 에서 읽어 변환할 때마다 적용한다.
  {sk: {"action": "delete"}} = 학습 로그에서 뺌 (rejected.jsonl 에 '사용자 삭제'),
  {sk: {"action": "edit", "label": ..., "http": ...}} = label·payload 를 바꿔서 씀 (label 파일의 edited 에 표시).
  LLM 출력(07)은 그대로이므로 edits.json 에서 지우면 원래대로 돌아온다.

[method]  docs/코드참조/method_매핑.md 의 'IPS method 방식' / 'WAF method 방식'
  - mapping   : 공격 유형 → 장비 탐지 이름 (매핑 표)
  - rule_name : Snort 룰 이름 그대로 (예: ET WEB_SPECIFIC_APPS D-Link gena.cgi ... Command Injection Attempt)
  - 그 외 값  : 모든 anomalies 의 method 를 그 값으로 통일 (예: 26_snort_rule)

[전체 흐름]
  1. 07_llm_output/<날짜>/*.jsonl (LLM 응답) 을 읽는다.
  2. 화면에서 저장한 수정·삭제(edits.json)를 적용하고, 샘플을 검사한다: label, 빈 http, '|||' 포함, attack_type 목록, 매핑 제외
  3. method_매핑.md 로 IPS·WAF method 를 정한다.
  4. 시각(하루에 고르게), 출발IP(공인 대역, 샘플마다 고정), 목적IP/포트(Host 헤더 또는 기본값)를 채운다.
  5. 08_dataset/<날짜>/ 에 IPS·WAF 로그와 label 파일을 쓴다.

[출력]
  08_dataset/<날짜>/
     ├─ ips_<yymmdd>_snort.log         ← IPS 파서 형식 (8필드)
     ├─ ips_<yymmdd>_snort_label.jsonl ← 줄별 정답·원본 룰 id
     ├─ waf_<yymmdd>_snort.log         ← WAF 파서 형식 (8필드)
     ├─ waf_<yymmdd>_snort_label.jsonl ← 줄별 정답·원본 룰 id
     ├─ rejected.jsonl                 ← 검사에서 빠진 샘플과 이유 (화면에서 삭제한 샘플 포함)
     ├─ edits.json                     ← 화면에서 저장한 수정·삭제 (1.4 는 읽기만)
     └─ summary.json

[사용법]
    python 1_4_build_dataset.py                       # LLM 출력이 바뀐 날짜만 변환
    python 1_4_build_dataset.py --force               # 다시 변환
    python 1_4_build_dataset.py --date 2026-10-20 --force   # 그 날짜만 (화면의 '학습 로그에 반영')
    python 1_4_build_dataset.py --llm-output 폴더 --out 폴더   # 다른 위치 (테스트용)
"""

import argparse
import datetime as dt
import hashlib
import importlib.util
import ipaddress
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LLM_INPUT_DIR = BASE_DIR / "05_llm_input"
LLM_OUTPUT_DIR = BASE_DIR / "07_llm_output"
DATASET_DIR = BASE_DIR / "08_dataset"
MAPPING_FILE = BASE_DIR / "docs" / "코드참조" / "method_매핑.md"
LOG_DIR = BASE_DIR / "logs"

SEP = "|||"
LABELS = {"anomalies", "normal"}
FORMAT = 2              # 출력 형식 판 (바뀌면 다음 실행에서 모든 날짜를 다시 변환). 2 = 요청 줄 역슬래시 %5C, sk, 수정·삭제
EDITS_NAME = "edits.json"
# 요청 줄에서 로그 표기(줄바꿈 '\n', 제어문자 '\x00')로 읽힐 글자 역슬래시
AMBIG_BS = re.compile(r"\\(?=n|x(?:0[0-9b-f]|1[0-9a-f]|7f))")
ATTACK_TYPES = {
    "sql_injection", "xss", "xxe", "ssrf", "deserialization", "code_injection", "command_injection",
    "file_inclusion", "path_traversal", "file_upload", "webshell", "auth_bypass", "rce", "buffer_overflow",
    "info_disclosure", "scanner", "dos", "csrf", "open_redirect", "informational", "other",
}

log = logging.getLogger("build_dataset")


def setup_logging():
    LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(LOG_DIR / "build_dataset.log", encoding="utf-8")
    fh.setFormatter(fmt)
    log.setLevel(logging.INFO)
    log.addHandler(fh)
    if sys.stdout is not None:  # pythonw(작업 스케줄러) 실행 시에는 콘솔이 없음
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        log.addHandler(sh)


# ── method_매핑.md 읽기 ────────────────────────────────────────────────

DEFAULT_SETTINGS = {
    "IPS method 방식": "mapping", "WAF method 방식": "mapping", "normal method": "normal",
    "기본 목적지 IP": "192.168.10.31", "IPS 포트": "80", "WAF 포트": "8080", "payload 줄바꿈": "\\n",
    "문서용 주소 바꾸기": "예",
}


def load_mapping(path=MAPPING_FILE):
    """'## 설정', '## 매핑' 표를 읽는다.

    반환: (settings, {attack_type: {"ips": , "waf": }})  (매핑 표의 나머지 열은 읽지 않음)
    """
    settings, mapping, section = dict(DEFAULT_SETTINGS), {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("## "):
            section = {"설정": "settings", "매핑": "mapping"}.get(s[3:].strip())
            continue
        if section is None or not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        cells = [c[1:-1] if len(c) >= 2 and c[0] == c[-1] == "`" else c for c in cells]
        if not cells or set(cells[0]) <= set("-: ") or cells[0] in ("항목", "attack_type"):
            continue
        if section == "settings" and len(cells) >= 2:
            settings[cells[0]] = cells[1]
        elif section == "mapping" and len(cells) >= 3:
            mapping[cells[0]] = {"ips": cells[1], "waf": cells[2]}
    if not mapping:
        raise ValueError(f"{path.name} 에서 '## 매핑' 표를 찾지 못했습니다")
    return settings, mapping


# ── 값 만들기 ───────────────────────────────────────────────────────────

def _replace(tmp, dst, tries=10):
    """임시 파일 → 결과 파일. Windows 에서는 다른 프로세스(동시에 도는 다른 1.4, 화면)가 결과 파일을 잠깐 열고 있으면
    바꾸기가 실패하므로 조금 기다렸다 다시 (수집과 학습데이터 생성이 같은 날짜의 로그를 거의 동시에 다시 만들 때)."""
    for k in range(tries):
        try:
            tmp.replace(dst)
            return
        except PermissionError:
            if k == tries - 1:
                raise
            time.sleep(0.5 + 0.5 * k)


def _h(*parts):
    return int(hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest(), 16)


def src_ip(seed):
    """공인 IPv4 하나 (사설·예약 대역 제외). 같은 seed → 같은 IP."""
    n = seed
    while True:
        ip = ipaddress.IPv4Address(n % (2 ** 32))
        if ip.is_global and 1 <= int(str(ip).split(".")[0]) <= 223:
            return str(ip)
        n = _h(n)


HOST_RE = re.compile(r"(?:^|\n)Host:\s*([^\s:]+)(?::(\d+))?", re.I)


def host_of(http):
    m = HOST_RE.search(http)
    return (m.group(1), m.group(2)) if m else ("", None)


def dst_of(http, default_ip):
    """Host 헤더가 IPv4 면 (IP, 포트 또는 None), 아니면 (기본 IP, None)."""
    host, port = host_of(http)
    try:
        ipaddress.IPv4Address(host)
        return host, port
    except ValueError:
        return default_ip, None


def normalize_http(http):
    """줄바꿈을 LF 로 통일하고 끝의 빈 줄을 정리한다. (헤더만 있는 요청은 빈 줄 하나로 끝나게)"""
    s = http.replace("\r\n", "\n").replace("\r", "\n")
    head, sep, body = s.partition("\n\n")
    head = head.rstrip("\n")
    return head + "\n\n" + body if sep else head + "\n\n"


# ── 문서용 주소 → 실제처럼 보이는 임의 주소 ────────────────────────────
# 생성 지시문은 안전을 위해 외부 주소(다운로드·콜백·유출)를 문서용 IP(RFC 5737)·example 도메인으로만 쓰게 한다.
# 실제 트래픽에는 없는 값인데 생성 anomalies 의 32.5% 에 들어 있고(normal 은 2%, 기존 학습 로그 0%), 몇 개의 IP 가
# 수천 번 반복돼(198.51.100.23 이 1,998번) 모델이 '이 주소 = 공격' 으로 배울 수 있다. 그래서 학습 로그를 만들 때
# 샘플마다 다른 공인 IP·도메인으로 바꾼다. 'evil.php'·'malicious.com' 같은 자리표시 이름도 흔한 실제 이름으로.
# LLM 출력(07)은 그대로 두고 학습 로그(08)만 바꾸며, 바꾼 뒤 룰 판정(코드 검사)이 달라지면 원래대로 둔다.

_START = r"(?:\b|(?<=%[0-9A-Fa-f]{2}))"                  # 단어 경계 또는 %XX 바로 뒤 (bob%40example.com, %2F%2Fevil.com)
# 앞이 숫자·'숫자.' 이면 더 긴 숫자의 일부라 제외 (1.192.0.2.10). %XX 뒤(%20192.0.2.1)·글자. 뒤(smtp.192.0.2.1)는 포함
DOC_IP_RE = re.compile(r"(?:(?<=%[0-9A-Fa-f]{2})|(?<=[A-Za-z_\-]\.)|(?<![\d.]))(?:192\.0\.2|198\.51\.100|203\.0\.113)\.\d{1,3}(?!\d)")
# 생성 모델의 반복 루프로 문서용 IP 가 구분자 없이 이어진 것 (198.51.100.23198.51.100.23…) → IP 하나로
_DOC = r"(?:192\.0\.2|198\.51\.100|203\.0\.113)\."
DOC_IP_RUN_RE = re.compile(rf"(?:{_DOC}\d{{1,3}}?(?={_DOC}))+{_DOC}\d+")
DOC_DOMAIN_RE = re.compile(_START + r"((?:[A-Za-z0-9-]+\.)*?)(example\.(?:com|net|org)|[A-Za-z0-9-]+\.example)\b(?!\.[A-Za-z])",
                           re.I)
FAKE_DOMAIN_RE = re.compile(_START + r"((?:[A-Za-z0-9-]+\.)*?)(?:evil|malicious|attacker|hacker)"
                            r"(?:-?(?:site|server|domain|host))?\.(?:com|net|org)\b", re.I)
FAKE_FILE_RE = re.compile(_START + r"(?:evil|malicious|attacker|hacker)(?=\.(?:php\d?|phtml|phar|txt|js|sh|jsp|aspx?|exe|html?|"
                          r"xml|dtd|svg|py|pl|cgi|war|jar|zip|gif|jpe?g|png)\b)", re.I)
_DOMAIN_WORDS = ["cdn", "update", "static", "img", "api", "files", "dl", "sync", "mirror", "web", "app", "cloud", "data",
                 "srv", "host", "node", "edge", "media", "assets", "get", "box", "hub", "panel", "mail", "load", "soft"]
_TLDS = ["com", "net", "org", "info", "xyz", "top", "ru", "cn", "io", "biz", "online", "site", "club", "pw", "cc"]
_FILE_NAMES = ["shell", "cmd", "x", "up", "1", "a", "img", "cache", "mini", "conf", "s", "info", "bot", "sh", "init",
               "ajax", "class", "lib", "wp-cache", "upload", "tmp", "inc", "db", "test1", "log"]


def fake_domain(seed):
    r = random.Random(seed)
    name = r.choice(_DOMAIN_WORDS) + r.choice(["", "-", ""]) + r.choice(_DOMAIN_WORDS + [str(r.randint(1, 99))])
    if r.random() < 0.4:
        name += "".join(r.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(r.randint(2, 4)))
    return f"{name}.{r.choice(_TLDS)}"


def realistic_addresses(http, seed):
    """문서용 IP·example 도메인·evil 류 이름 → 샘플마다 다른 실제 같은 값. 같은 샘플 안에서 같은 값은 같은 값으로.
    → (바꾼 http, 바꾼 곳 수). 본문이 바뀌면 Content-Length 를 다시 맞춘다."""
    count = 0

    def ip(m):
        nonlocal count
        count += 1
        return src_ip(_h(seed, "ip", m.group(0)))

    def ip_run(m):
        nonlocal count
        count += 1
        return src_ip(_h(seed, "ip", m.group(0)[:15]))

    def domain(m):
        nonlocal count
        count += 1
        sub, dom = m.group(1), m.group(2)
        if dom.lower().endswith(".example"):                # smtp.example → smtp.<도메인>
            sub, dom = sub + dom[:-len("example")], "example"
        return sub + fake_domain(_h(seed, "dom", dom.lower()))

    def fake_dom(m):
        nonlocal count
        count += 1
        return m.group(1) + fake_domain(_h(seed, "fake", m.group(0).lower()))

    def fake_file(m):
        nonlocal count
        count += 1
        return random.Random(_h(seed, "file", m.group(0).lower())).choice(_FILE_NAMES)

    head, sep, body = http.partition("\n\n")
    out = []
    for part in (head, body):
        part = DOC_IP_RUN_RE.sub(ip_run, part)
        part = DOC_IP_RE.sub(ip, part)
        part = DOC_DOMAIN_RE.sub(domain, part)
        part = FAKE_DOMAIN_RE.sub(fake_dom, part)
        out.append(FAKE_FILE_RE.sub(fake_file, part))
    new_head, new_body = out
    if sep and new_body != body:                           # 본문 길이가 바뀌었으면 Content-Length 를 다시
        new_head = re.sub(r"(?im)^(Content-Length:[ \t]*)\d+", lambda m: f"{m.group(1)}{len(new_body.encode('utf-8'))}",
                          new_head, count=1)
    return new_head + sep + new_body, count


_check = {}


def same_verdict(old, new, label, conditions):
    """바꾸기 전후의 코드 검사(룰 조건 대조) 결과가 같은지 (3_0a_common.local_check)."""
    if "C" not in _check:
        spec = importlib.util.spec_from_file_location("llm_common_14", BASE_DIR / "3_0a_common.py")
        _check["C"] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_check["C"])
    lc, rule = _check["C"].local_check, {"conditions": conditions}
    return lc({"label": label, "http": old}, rule)["result"] == lc({"label": label, "http": new}, rule)["result"]


# ── LLM 출력 읽기 ───────────────────────────────────────────────────────

def load_edits(out_dir):
    """화면에서 저장한 수정·삭제 {sk: {...}} (없으면 빈 dict)."""
    try:
        return json.loads((out_dir / EDITS_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def edits_mtime(out_dir):
    try:
        return (out_dir / EDITS_NAME).stat().st_mtime
    except OSError:
        return 0


def read_llm_outputs(folder):
    """*.jsonl 을 읽어 룰 결과 목록을 돌려준다. ``` 줄, 빈 줄은 무시."""
    results, bad = [], []
    for f in sorted(folder.glob("*.jsonl")):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            s = line.strip()
            if not s or s.startswith("```"):
                continue
            try:
                obj = json.loads(s)
                if not isinstance(obj, dict) or "samples" not in obj:
                    raise ValueError("samples 없음")
                results.append((f.name, i, obj))
            except ValueError as e:
                bad.append({"file": f.name, "line": i, "reason": f"JSON 오류: {e}", "text": s[:300]})
    return results, bad


def input_rules():
    """05_llm_input 최신 rules_all.jsonl 의 {id: {"attack_type", "conditions", "rule_name"}}."""
    # 완성된 날짜 폴더만 (1.3 이 쓰는 중인 임시 폴더 .<날짜>.tmp… 와 summary.json 이 없는 폴더는 뺀다)
    dirs = sorted(d for d in LLM_INPUT_DIR.glob("*") if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d.name)
                  and (d / "rules_all.jsonl").exists() and (d / "summary.json").exists())
    out = {}
    if dirs:
        with open(dirs[-1] / "rules_all.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                out[r["id"]] = {"attack_type": r["attack_type"], "conditions": r["conditions"],
                                "rule_name": r.get("rule_name") or r.get("msg", "")}
    return out


# ── 변환 ────────────────────────────────────────────────────────────────

def pick_method(mode, mapped, rule_name):
    """method 방식에 따라 method 이름을 고른다. mapping=매핑 표, rule_name=Snort 룰 이름, 그 외=고정값."""
    if mode == "mapping":
        return mapped
    if mode == "rule_name":
        return rule_name or mapped
    return mode


def collect_events(results, rejected, st, mapping, rules, edits=None):
    """LLM 결과 → 이벤트 목록. 잘못된 샘플은 rejected 에 이유와 함께 넣는다. edits = 화면에서 저장한 수정·삭제."""
    events, edits = [], edits or {}
    for fname, lineno, obj in results:
        rid = obj.get("id", "")
        rule = rules.get(rid, {})
        rule_type = obj.get("attack_type") or rule.get("attack_type", "other")
        for k, smp in enumerate(obj.get("samples", [])):
            label = (smp.get("label") or "").strip()
            http = smp.get("http") or ""
            atype = (smp.get("attack_type") or rule_type).strip()
            seed = _h(rid, k, http)                     # 출발 IP·포트·주소 바꾸기의 씨앗 (LLM 출력 그대로 기준)
            sk = f"{seed:040x}"[:16]                    # 샘플 지문: 화면의 수정·삭제가 이것으로 샘플을 찾는다
            ref = {"file": fname, "line": lineno, "id": rid, "sample": k, "sk": sk}
            edit, edited = edits.get(sk) or {}, []
            if edit.get("action") == "delete":
                rejected.append(dict(ref, reason="사용자 삭제" + (f": {edit['why']}" if edit.get("why") else ""))); continue
            if edit.get("action") == "edit":
                if edit.get("label") in LABELS and edit["label"] != label:
                    label = edit["label"]
                    edited.append("label")
                    if label == "anomalies":                # normal 샘플의 attack_type 은 none → 룰의 공격 유형
                        atype = (rule_type or "other").strip()
                if edit.get("http") and edit["http"] != http:
                    http = edit["http"]
                    edited.append("payload")
            if label not in LABELS:
                rejected.append(dict(ref, reason=f"label 오류: {label!r}")); continue
            if not http.strip():
                rejected.append(dict(ref, reason="http 비어 있음")); continue
            if SEP in http:
                rejected.append(dict(ref, reason="payload 에 구분자 '|||' 포함")); continue
            if label == "anomalies":
                if atype not in ATTACK_TYPES:
                    rejected.append(dict(ref, reason=f"attack_type 목록 밖: {atype!r}")); continue
                m = mapping.get(atype)
                if not m or not m["ips"] or not m["waf"]:
                    rejected.append(dict(ref, reason=f"매핑 없음: {atype}")); continue
                if "제외" in (m["ips"], m["waf"]):
                    rejected.append(dict(ref, reason=f"매핑에서 제외: {atype}")); continue
                name = rule.get("rule_name", "").replace(SEP, " ")
                ev = {"ips_method": pick_method(st["IPS method 방식"], m["ips"], name),
                      "waf_method": pick_method(st["WAF method 방식"], m["waf"], name)}
            else:
                atype = "none"
                ev = {"ips_method": st["normal method"], "waf_method": st["normal method"]}
            http = normalize_http(http)
            n_addr = 0
            if "payload" not in edited and st["문서용 주소 바꾸기"].strip() in ("예", "yes", "y", "true", "1"):
                new, n_addr = realistic_addresses(http, seed)
                if n_addr and not same_verdict(http, new, label, rule.get("conditions", [])):
                    new, n_addr = http, -1                  # 바꾸면 룰 판정이 달라짐 → 원래대로 (표시만)
                http = new
            ev.update(id=rid, label=label, attack_type=atype, http=http, addr_fixed=n_addr,
                      conditions=rule.get("conditions", []), seed=seed, sk=sk, edited="·".join(edited),
                      # 생성 단계의 검증 기록 (화면의 '학습데이터' 페이지가 label 사유·검증 결과로 보여 줌)
                      origin=smp.get("origin", ""), review=smp.get("review", ""),
                      local_check=smp.get("local_check", ""), note=str(smp.get("note", ""))[:300])
            events.append(ev)
    return events


def one_line(http, st):
    """HTTP 메시지를 한 줄 payload 로. 줄바꿈 표기는 매핑 md 의 'payload 줄바꿈' (기본 두 글자 '\\n')."""
    nl = " " if st["payload 줄바꿈"].lower() == "space" else "\\n"
    # 요청 줄의 글자 역슬래시 + n·제어문자 표기 → %5C (그대로 두면 로그를 읽을 때 줄바꿈으로 읽혀 요청 줄이 잘림)
    head, sep, rest = http.rstrip("\n").partition("\n")
    s = AMBIG_BS.sub("%5C", head) + sep + rest
    # 줄바꿈 외 제어문자(바이너리 바이트)는 기존 학습 로그처럼 '\x00' 글자 표기로 (로그 한 줄이 깨지지 않게)
    s = re.sub(r"[\x00-\x09\x0b-\x1f\x7f]", lambda m: f"\\x{ord(m.group()):02x}", s)
    return s.replace("\n", nl)


def check_fields(fields, dev):
    for f in fields:                                        # 필드 안에 구분자·개행이 있으면 형식이 깨짐
        if SEP in f or "\n" in f:
            raise ValueError(f"{dev} 필드에 구분자 또는 개행 포함")
    return SEP.join(fields)


def ips_line(ev, ts, st):
    """IPS: event_time|||s_ip|||s_port|||d_ip|||d_port|||method|||payload|||label (8필드)"""
    fields = [ts.strftime("%Y-%m-%d %H:%M:%S"), ev["s_ip"], str(ev["s_port"]), ev["d_ip"], str(ev["ips_port"]),
              ev["ips_method"], one_line(ev["http"], st), ev["label"]]
    return check_fields(fields, "IPS")


def waf_line(ev, ts, st):
    """WAF: IPS 와 같은 8필드. 포트·method 만 WAF 값"""
    fields = [ts.strftime("%Y-%m-%d %H:%M:%S"), ev["s_ip"], str(ev["s_port"]), ev["d_ip"], str(ev["waf_port"]),
              ev["waf_method"], one_line(ev["http"], st), ev["label"]]
    return check_fields(fields, "WAF")


def build(day_dir, out_dir, st, mapping, rules):
    results, rejected = read_llm_outputs(day_dir)
    edits, e_mtime = load_edits(out_dir), edits_mtime(out_dir)
    try:
        day0 = dt.datetime.strptime(day_dir.name, "%Y-%m-%d")
    except ValueError:
        day0 = dt.datetime.combine(dt.date.today(), dt.time())
    yymmdd = day0.strftime("%y%m%d")
    events = collect_events(results, rejected, st, mapping, rules, edits)

    n = max(len(events), 1)
    for i, ev in enumerate(events):
        ev["ts"] = day0 + dt.timedelta(seconds=int(i * 86400 / n))
        ev["s_ip"], ev["s_port"] = src_ip(ev["seed"]), 1024 + ev["seed"] % 64511
        ev["d_ip"], hport = dst_of(ev["http"], st["기본 목적지 IP"])
        ev["ips_port"] = int(hport or st["IPS 포트"])
        ev["waf_port"] = int(hport or st["WAF 포트"])

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"ips": (ips_line, "ips_method"), "waf": (waf_line, "waf_method")}
    names = {}
    for dev, (fmt, mkey) in outputs.items():
        logf = out_dir / f"{dev}_{yymmdd}_snort.log"
        labf = out_dir / f"{dev}_{yymmdd}_snort_label.jsonl"
        names[dev] = [logf.name, labf.name]
        # 임시 파일 이름에 PID 를 넣는다 (수집 에이전트의 1.4 와 학습데이터 생성의 1.4 가 동시에 돌아도 부딪히지 않게)
        tmp_log = logf.with_name(f"{logf.name}.{os.getpid()}.tmp")
        tmp_lab = labf.with_name(f"{labf.name}.{os.getpid()}.tmp")
        with open(tmp_log, "w", encoding="utf-8", newline="\n") as lf, \
                open(tmp_lab, "w", encoding="utf-8", newline="\n") as tf:
            line_no = 0
            for ev in events:
                try:
                    text = fmt(ev, ev["ts"], st)
                except ValueError as e:
                    rejected.append({"id": ev["id"], "device": dev, "reason": str(e)})
                    continue
                line_no += 1
                lf.write(text + "\n")
                rec = {"line": line_no, "id": ev["id"], "label": ev["label"], "attack_type": ev["attack_type"],
                       "method": ev[mkey], "origin": ev["origin"], "review": ev["review"],
                       "local_check": ev["local_check"], "note": ev["note"], "sk": ev["sk"]}
                if ev["edited"]:                            # 화면에서 고친 것 (label·payload)
                    rec["edited"] = ev["edited"]
                if ev["addr_fixed"]:                        # 문서용 주소를 바꾼 곳 수 (-1 = 룰 판정이 달라져 그대로 둠)
                    rec["addr_fixed"] = ev["addr_fixed"]
                tf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        _replace(tmp_log, logf)
        _replace(tmp_lab, labf)

    with open(out_dir / "rejected.jsonl", "w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "llm_output_dir": str(day_dir),
        "llm_output_files": sorted(p.name for p in day_dir.glob("*.jsonl")),
        "llm_output_mtime": max((p.stat().st_mtime for p in day_dir.glob("*.jsonl")), default=0),
        "mapping_mtime": MAPPING_FILE.stat().st_mtime, "format": FORMAT, "edits_mtime": e_mtime,
        "edits": {"saved": len(edits), "deleted": sum(1 for r in rejected if r.get("reason", "").startswith("사용자 삭제")),
                  "edited": sum(1 for e in events if e["edited"])},
        "rules": len(results), "events": len(events), "rejected": len(rejected),
        "label": dict(Counter(e["label"] for e in events)),
        "attack_type": dict(Counter(e["attack_type"] for e in events).most_common()),
        "ips_method": dict(Counter(e["ips_method"] for e in events).most_common()),
        "waf_method": dict(Counter(e["waf_method"] for e in events).most_common()),
        "reject_reasons": dict(Counter(r["reason"].split(":")[0] for r in rejected)),
        "addr_fixed": {"samples": sum(1 for e in events if e["addr_fixed"] > 0), "places": sum(max(0, e["addr_fixed"]) for e in events),
                       "kept_original": sum(1 for e in events if e["addr_fixed"] < 0)},
        "files": names,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("[데이터셋] %s: 룰 %d개 → 이벤트 %d건 (anomalies %d, normal %d), 제외 %d건",
             out_dir, len(results), len(events), summary["label"].get("anomalies", 0),
             summary["label"].get("normal", 0), len(rejected))


def main(argv=None):
    parser = argparse.ArgumentParser(description="LLM 출력 → IPS/WAF 수집 로그 변환")
    parser.add_argument("--force", action="store_true", help="이미 변환한 날짜도 다시 변환")
    parser.add_argument("--llm-output", type=Path, default=LLM_OUTPUT_DIR, help="LLM 출력 폴더 (날짜 폴더들의 상위)")
    parser.add_argument("--out", type=Path, default=DATASET_DIR, help="데이터셋 출력 폴더")
    parser.add_argument("--date", help="이 날짜 폴더만 (예: 2026-10-20)")
    args = parser.parse_args(argv)
    setup_logging()

    try:
        st, mapping = load_mapping()
    except (ValueError, FileNotFoundError) as e:
        log.error("method 매핑 읽기 실패 - 중단: %s", e)
        return 1

    # 날짜 폴더만 (07_llm_output/_progress 같은 진행 현황 폴더는 빼기)
    day_dirs = sorted(d for d in args.llm_output.glob("*") if d.is_dir()
                      and re.fullmatch(r"\d{4}-\d{2}-\d{2}", d.name) and any(d.glob("*.jsonl"))
                      and (not args.date or d.name == args.date))
    if not day_dirs:
        log.info("LLM 출력 없음 (%s/<날짜>/*.jsonl) - 건너뜀", args.llm_output)
        return 0
    rules = input_rules()
    for day in day_dirs:
        out = args.out / day.name
        prev = {}
        if (out / "summary.json").exists():
            prev = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        latest = max(p.stat().st_mtime for p in day.glob("*.jsonl"))
        if (not args.force and prev.get("llm_output_mtime") == latest
                and prev.get("mapping_mtime") == MAPPING_FILE.stat().st_mtime
                and prev.get("format", 1) == FORMAT and prev.get("edits_mtime", 0) == edits_mtime(out)):
            log.info("[데이터셋] %s 변경 없음 - 건너뜀", day.name)
            continue
        build(day, out, st, mapping, rules)
    return 0


if __name__ == "__main__":
    sys.exit(main())
