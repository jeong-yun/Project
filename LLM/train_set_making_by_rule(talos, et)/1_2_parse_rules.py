"""[1.2 파싱·통합] Snort 룰 파싱/정규화 + 선별·HTTP 필터링 + 변경분 추출 + 통합·중복 제거.  (1_0_run_daily.py 가 실행)

[전체 흐름]
  1. state.json(1_1_fetch_rules.py 가 기록)에서 소스별 최신 다운로드 파일을 찾는다.
  2. parse_state.json 과 비교해 '새 버전'이거나 '선별 조건(선별조건.md)이 바뀐' 소스만 처리한다.
  3. 압축 파일에서 .rules 만 꺼내 02_extracted/ 에 저장한다. (최신본만 유지)
  4. 룰 한 줄 한 줄을 JSON 구조로 변환한다. (주석 처리된 비활성 룰도 enabled=false 로 포함)
  5. 선별조건.md 에서 선택(O)한 카테고리 + HTTP 룰만 골라내고, HTTP method 를 추출한다.
  6. 직전 파싱 결과와 비교해 신규/변경/삭제 룰을 뽑는다.
     (선별 조건 변경으로 새로 들어온 카테고리의 룰은 '선별 추가'로 따로 표시)
  7. 신규 + 변경 + 선별 추가 룰을 LLM 입력용(llm_queue.jsonl)으로 저장한다.
  8. ET + Talos 결과를 04_integrated/<오늘>/ 에 JSON 하나로 합치고, 선별 보고서(md)를 만든다.

[폴더 - 번호 = 처리 순서]
  01_raw/<source>/<날짜>/        ← (1_1_fetch_rules.py) 원본 압축 파일
  02_extracted/<source>/*.rules  ← 압축 해제한 원본 룰 텍스트 (최신 버전으로 덮어씀)
  03_processed/<source>/<날짜>/
     ├─ http_rules.jsonl     ← 선별 HTTP 룰 전체 (정규화 JSON, 한 줄 = 룰 1개)
     ├─ delta.json           ← 직전 대비 신규/변경/삭제/선별 추가 룰 목록 (key, rev, msg)
     ├─ llm_queue.jsonl      ← 신규 + 변경 + 선별 추가 룰 내용 (LLM 페이로드 생성 입력용)
     └─ summary.json         ← 개수 통계, 카테고리별 수, method 분포, 파싱 오류 샘플
  04_integrated/<오늘>/          ← 이번 실행에서 파싱한 소스가 있을 때만 생성
     ├─ http_rules.json      ← ET + Talos 선별 HTTP 룰 전체 (각 소스의 최신 결과를 합침, 중복 제거)
     ├─ llm_queue.json       ← 오늘 파싱된 소스의 llm_queue (중복 제거)
     ├─ duplicates.json      ← Talos·ET 양쪽에 있어서 뺀 룰 + 같다고 판단한 짝 + 기준
     └─ selection_report.md  ← 카테고리별 룰 수, 선택/제외 여부와 이유, 중복 제거 결과
     (중복 제거 설정은 선별조건.md 의 '## 중복 제거' 표. 바뀌면 통합 결과를 다시 만든다)

  ※ 룰이 바뀐 날마다 새 날짜 폴더에 쌓인다.
     (예전 버전에서 만든 http_rules.jsonl.gz 도 비교 기준으로 그대로 읽을 수 있다)

[사용법]
    python 1_2_parse_rules.py                 # 새로 받은 파일만 파싱
    python 1_2_parse_rules.py --source et     # ET 만
    python 1_2_parse_rules.py --force         # 이미 파싱한 파일도 다시 파싱

- 첫 실행은 비교 대상이 없으므로 모든 HTTP 룰이 delta_new 로 잡힌다.
- 파이썬 기본 라이브러리만 사용한다.
"""

import argparse
import collections
import datetime as dt
import gzip
import hashlib
import json
import logging
import re
import sys
import tarfile
import zipfile
from pathlib import Path

import importlib.util

# 선별조건.md 를 읽는 로더 (같은 폴더의 1_2a_rule_selection.py)
# 파일 이름이 숫자로 시작해서 일반 import 를 쓸 수 없으므로, 파일 경로로 직접 불러온다.
_sel_path = Path(__file__).resolve().parent / "1_2a_rule_selection.py"
_spec = importlib.util.spec_from_file_location("rule_selection", _sel_path)
sel = importlib.util.module_from_spec(_spec)
sys.modules["rule_selection"] = sel
_spec.loader.exec_module(sel)

# ── 경로 설정 (1_1_fetch_rules.py 와 같은 기준 폴더, 번호 = 처리 순서) ─────────
BASE_DIR = Path(__file__).resolve().parent
EXTRACT_DIR = BASE_DIR / "02_extracted"
PROCESSED_DIR = BASE_DIR / "03_processed"
INTEGRATED_DIR = BASE_DIR / "04_integrated"
CUSTOM_DIR = BASE_DIR / "00_custom_rules"           # 사용자가 직접 넣는 룰 (예: 보안 공유 정책의 Snort 룰)

# 파싱 규칙 버전. HTTP 판별·method 추출 같은 파싱 로직을 고치면 이 값을 바꾼다.
# → 다음 실행 때 같은 룰 버전이라도 자동으로 다시 파싱된다.
#   2026-10-16: 원시 TCP 로 HTTP 요청을 검사하는 룰(raw_http_request)도 HTTP 룰로 판별
PARSER_VERSION = "2026-10-16c"
#   2026-10-16b: 사용자 룰 중복 제거용으로 Talos·ET 의 '모든' HTTP 룰 비교 색인(dedup_index.jsonl) 저장
LOG_DIR = BASE_DIR / "logs"
FETCH_STATE_FILE = BASE_DIR / "state.json"        # 1_1_fetch_rules.py 가 쓰는 파일 (읽기만 함)
PARSE_STATE_FILE = BASE_DIR / "parse_state.json"  # 이 파일이 쓰는 상태 파일

log = logging.getLogger("parse_rules")

# ── 룰 문법 관련 상수 ───────────────────────────────────────────────────

# 룰 헤더: action proto src_addr src_port 방향 dst_addr dst_port (옵션들)
# 예) alert tcp any any -> $HOME_NET $HTTP_PORTS (msg:"..."; ...)
# 줄 맨 앞의 "# " 은 비활성(주석 처리된) 룰을 뜻한다.
RULE_RE = re.compile(
    r"^(?P<comment>#\s*)?"
    r"(?P<action>alert|log|pass|drop|reject|sdrop)\s+"
    r"(?P<proto>\S+)\s+(?P<src>\S+)\s+(?P<sport>\S+)\s+"
    r"(?P<dir>->|<>)\s+"
    r"(?P<dst>\S+)\s+(?P<dport>\S+)\s*"
    r"\((?P<opts>.*)\)\s*$"
)

# content 바로 뒤에 붙어서 "어느 HTTP 영역을 검사할지" 정하는 수식어
HTTP_BUFFERS = {
    "http_uri", "http_raw_uri", "http_header", "http_raw_header",
    "http_cookie", "http_raw_cookie", "http_client_body",
    "http_method", "http_stat_code", "http_stat_msg",
}
# 요청(request) 쪽 버퍼 / 응답(response) 쪽 버퍼
REQUEST_BUFFERS = {"http_uri", "http_raw_uri", "http_method", "http_client_body",
                   "http_cookie", "http_raw_cookie"}
RESPONSE_BUFFERS = {"http_stat_code", "http_stat_msg"}

# content 위치/옵션 수식어 (직전 content 에 적용됨)
CONTENT_MODIFIERS = {"nocase", "rawbytes", "depth", "offset", "distance", "within", "fast_pattern"}

# 이 키워드 이후의 content 는 해당 버퍼를 검사함 (sticky buffer)
STICKY_BUFFERS = {"file_data": "file_data", "base64_data": "base64_data", "pkt_data": "raw"}

# pcre 의 HTTP 플래그 → 검사 버퍼  (예: pcre:"/.../U" → http_uri)
PCRE_HTTP_FLAGS = {
    "U": "http_uri", "I": "http_raw_uri", "P": "http_client_body",
    "H": "http_header", "D": "http_raw_header", "M": "http_method",
    "C": "http_cookie", "K": "http_raw_cookie", "S": "http_stat_code",
    "Y": "http_stat_msg",
}

# 헤더 포트에 이 변수가 있으면 HTTP 룰로 본다.
HTTP_PORT_VARS = ("$HTTP_PORTS", "$FILE_DATA_PORTS")

# method 추출 시 비교할 알려진 HTTP method 목록 (WebDAV, UPnP 포함)
KNOWN_METHODS = [
    "GET", "POST", "PUT", "HEAD", "DELETE", "OPTIONS", "PATCH", "TRACE", "CONNECT",
    "PROPFIND", "PROPPATCH", "MKCOL", "COPY", "MOVE", "LOCK", "UNLOCK", "SEARCH",
    "REPORT", "MERGE", "SUBSCRIBE", "UNSUBSCRIBE", "NOTIFY", "M-SEARCH", "M-POST",
    "BITS_POST", "CCM_POST",
]


# ════════════════════════════════════════════════════════════════════════
#  공통 유틸
# ════════════════════════════════════════════════════════════════════════

def setup_logging():
    """로그를 logs/parse.log 와 화면에 출력한다. (1_1_fetch_rules.py 와 같은 형식)"""
    LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(LOG_DIR / "parse.log", encoding="utf-8")
    fh.setFormatter(fmt)
    log.setLevel(logging.INFO)
    log.addHandler(fh)
    if sys.stdout is not None:  # pythonw(작업 스케줄러) 실행 시에는 콘솔이 없음
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        log.addHandler(sh)


def read_json(path, default):
    """JSON 파일을 읽는다. 없거나, 비었거나, 깨져 있으면 default 를 돌려준다.

    깨진 파일은 '<이름>.broken' 으로 옮겨 두고 경고를 남긴다. (원인 확인용)
    """
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        path.replace(path.with_name(path.name + ".broken"))
        log.warning("%s 이(가) 비었거나 깨짐 (%s) - %s.broken 으로 옮김", path.name, e, path.name)
        return default


def write_json(path, data):
    """JSON 저장. 임시 파일에 다 쓴 뒤 이름을 바꿔, 도중에 끊겨도 빈 파일이 남지 않게 한다."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def rebuild_parse_state():
    """parse_state.json 이 없거나 깨졌을 때, 03_processed/ 의 summary.json 들로 상태를 복구한다.

    소스마다 가장 최근에 파싱한 summary(parsed_at 기준)를 골라
    marker / dir / baseline_dir 를 되살린다.
    → 이미 파싱한 버전을 다시 '전부 신규'로 덮어쓰는 일을 막는다.
    """
    state = {}
    for summary_file in PROCESSED_DIR.glob("*/*/summary.json"):
        try:
            s = json.loads(summary_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        src = s.get("source")
        if src and s.get("parsed_at", "") > state.get(src, {}).get("parsed_at", ""):
            state[src] = {
                "marker": s["marker"],
                "dir": str(summary_file.parent.relative_to(BASE_DIR)),
                "baseline_dir": s.get("baseline"),
                "selection": s.get("selection"),
                "parsed_at": s["parsed_at"],
            }
    return state


def _open(path, mode):
    """확장자가 .gz 이면 gzip 으로, 아니면 일반 텍스트 파일로 연다."""
    if str(path).endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def write_jsonl(path, records):
    """JSON Lines 형식으로 저장 (한 줄 = 레코드 1개). 큰 파일도 한 줄씩 읽을 수 있다."""
    with _open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with _open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


# ════════════════════════════════════════════════════════════════════════
#  1. 압축 해제
# ════════════════════════════════════════════════════════════════════════

def extract_rules(source, archive, out_dir):
    """압축 파일에서 .rules 파일만 꺼내 out_dir 에 저장하고 {파일명: 텍스트} 를 돌려준다.

    - ET    : emerging-all.rules.zip 안의 emerging-all.rules
    - Talos : tar.gz 안의 rules/*.rules  (so_rules = 바이너리 룰, preproc_rules = 디코더 룰은 제외)

    보안: extractall() 을 쓰지 않고 파일명(basename)만 사용해 직접 저장한다.
          → 압축 안에 '../' 같은 경로가 있어도 다른 폴더에 파일이 써지지 않는다.
    용량: 최신본만 유지하므로, 저장 전에 out_dir 의 기존 .rules 파일을 지운다.
          (과거 버전은 raw/ 의 압축 파일로 언제든 다시 풀 수 있음)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.rules"):
        old.unlink()
    files = {}
    if source == "et":
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if name.endswith(".rules"):
                    files[Path(name).name] = z.read(name).decode("utf-8", "replace")
    else:
        with tarfile.open(archive, "r:gz") as t:
            for m in t.getmembers():
                if m.isfile() and m.name.startswith("rules/") and m.name.endswith(".rules"):
                    files[Path(m.name).name] = t.extractfile(m).read().decode("utf-8", "replace")
    for name, text in files.items():
        (out_dir / name).write_text(text, encoding="utf-8")
    return files


# ════════════════════════════════════════════════════════════════════════
#  2. 룰 파싱 / 정규화
# ════════════════════════════════════════════════════════════════════════

def split_options(opts):
    """옵션 문자열을 ';' 기준으로 나눈다.

    단, 따옴표 안의 ';' 와 역슬래시로 이스케이프된 '\\;' 는 구분자가 아니다.
    예) msg:"a;b"; content:"x\\;y"; sid:1;  →  ['msg:"a;b"', 'content:"x\\;y"', 'sid:1']

    일부 룰은 pcre 안에 이스케이프 안 된 '"' 가 있다. (예: pcre:"/name="abc/m";)
    그래서 따옴표는
      - 여는 따옴표: 바로 앞(공백 제외)이 ':' 또는 '!' 일 때만
      - 닫는 따옴표: 바로 뒤(공백 제외)가 ';' / ',' / 문자열 끝 일 때만
    으로 인정한다.
    """
    parts, buf, in_quote, escaped = [], [], False, False
    for i, ch in enumerate(opts):
        if escaped:
            buf.append(ch)
            escaped = False
            continue
        if ch == "\\":
            buf.append(ch)
            escaped = True
            continue
        if ch == '"':
            if not in_quote:
                prev = "".join(buf).rstrip()[-1:]
                in_quote = prev in (":", "!")
            else:
                nxt = opts[i + 1:].lstrip()[:1]
                in_quote = nxt not in (";", ",", "")
        if ch == ";" and not in_quote:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def unquote(value):
    """'!"abc"' → (True, 'abc')   /  '"abc"' → (False, 'abc')   ('!' = 부정 조건)"""
    value = value.strip()
    negated = value.startswith("!")
    if negated:
        value = value[1:].strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        value = value[1:-1]
    return negated, value


def decode_content(s):
    """Snort content 문자열을 실제 bytes 로 변환한다.

    - |3f 3d| 처럼 파이프 사이는 16진수 → b'?='
    - \\" \\\\ \\; \\: 는 이스케이프 문자 → 해당 문자 그대로
    예) '/gena.cgi|3f|' → b'/gena.cgi?'
    """
    out = bytearray()
    i, hex_mode = 0, False
    while i < len(s):
        ch = s[i]
        if ch == "|":
            hex_mode = not hex_mode
            i += 1
            continue
        if hex_mode:
            pair = s[i:i + 2]
            if re.fullmatch(r"[0-9A-Fa-f]{2}", pair):
                out.append(int(pair, 16))
                i += 2
            else:
                i += 1  # 공백 등은 무시
            continue
        if ch == "\\" and i + 1 < len(s):
            i += 1
            ch = s[i]
        out.extend(ch.encode("utf-8"))
        i += 1
    return bytes(out)


def bytes_to_text(b):
    """bytes 를 사람이/LLM 이 읽기 쉬운 문자열로. 출력 불가 문자는 \\xNN 으로 표기.
    역슬래시도 \\x5c 로 쓴다 (그대로 두면 |5C|x0d 같은 '글자 \\x0d' 가 CR 바이트 표기와 구분되지 않음)."""
    return "".join(chr(c) if 0x20 <= c <= 0x7E and c != 0x5C else "\\x%02x" % c for c in b)


def to_int(v):
    """'10' → 10, 변수명(byte_extract 결과 등)은 문자열 그대로."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


def parse_pcre(value):
    """pcre:"/패턴/플래그" → {pattern, flags, negated, buffer}"""
    negated, body = unquote(value)
    m = re.match(r"^/(.*)/([A-Za-z]*)$", body, re.S)
    if not m:
        return {"pattern": body, "flags": "", "negated": negated, "buffer": None}
    pattern, flags = m.group(1), m.group(2)
    buffer = next((PCRE_HTTP_FLAGS[f] for f in flags if f in PCRE_HTTP_FLAGS), None)
    return {"pattern": pattern, "flags": flags, "negated": negated, "buffer": buffer}


def parse_rule(line, source, filename):
    """룰 한 줄을 정규화된 딕셔너리로 변환한다. 룰이 아니면 None.

    content 처리 방식 (Snort 2 문법):
        content:"GET"; http_method; content:"/gena.cgi|3f|"; http_uri; depth:10;
        → 수식어(http_method, depth 등)는 '바로 앞 content' 에 적용된다.
    """
    m = RULE_RE.match(line.strip())
    if not m:
        return None

    rule = {
        "key": None, "source": source, "gid": 1, "sid": None, "rev": None,
        "enabled": m.group("comment") is None,     # '#' 으로 시작하면 비활성 룰
        "file": filename,
        "action": m.group("action"), "proto": m.group("proto"),
        "src_addr": m.group("src"), "src_port": m.group("sport"),
        "direction": m.group("dir"),
        "dst_addr": m.group("dst"), "dst_port": m.group("dport"),
        "msg": None, "classtype": None, "flow": [],
        "references": [], "cve": [], "metadata": {},
        "contents": [], "pcres": [], "other_options": {},
    }
    sticky = "raw"          # 현재 sticky buffer (file_data 이후면 'file_data')
    last_content = None     # 수식어를 붙일 대상 content

    for part in split_options(m.group("opts")):
        name, _, value = part.partition(":")
        name, value = name.strip(), value.strip()

        if name in ("content", "uricontent"):
            negated, raw = unquote(value)
            last_content = {
                "value": bytes_to_text(decode_content(raw)),  # 디코딩된 실제 값
                "raw": raw,                                   # 룰에 적힌 원문
                "negated": negated,
                "buffer": "http_uri" if name == "uricontent" else sticky,
                "nocase": False,
            }
            rule["contents"].append(last_content)
        elif name in HTTP_BUFFERS and last_content is not None:
            last_content["buffer"] = name
        elif name in CONTENT_MODIFIERS and last_content is not None:
            if name in ("nocase", "rawbytes"):
                last_content[name] = True
            elif name == "fast_pattern":
                last_content["fast_pattern"] = value or True
            else:
                last_content[name] = to_int(value)
        elif name in STICKY_BUFFERS:
            sticky = STICKY_BUFFERS[name]
            last_content = None
        elif name == "pcre":
            p = parse_pcre(value)
            if p["buffer"] is None:
                p["buffer"] = sticky
            # R(상대) = 바로 앞 content 가 끝난 자리부터 검사 (^ 도 그 자리). 룰 원문 순서의 그 content 를 남긴다
            # (contents·pcres 를 따로 모으면 순서를 잃음, 2026-10-30). sticky buffer 뒤 content 가 없으면 버퍼 처음 기준
            if "R" in p["flags"] and last_content is not None and not last_content["negated"]:
                p["after"] = {"value": last_content["value"], "buffer": last_content["buffer"],
                              "nocase": bool(last_content.get("nocase"))}
            rule["pcres"].append(p)
        elif name == "msg":
            rule["msg"] = unquote(value)[1]
        elif name in ("sid", "rev", "gid"):
            rule[name] = to_int(value)
        elif name == "classtype":
            rule["classtype"] = value
        elif name == "flow":
            rule["flow"] = [v.strip() for v in value.split(",")]
        elif name == "reference":
            rtype, _, rval = value.partition(",")
            rule["references"].append({"type": rtype.strip(), "value": rval.strip()})
            if rtype.strip().lower() == "cve":
                cve = rval.strip()
                rule["cve"].append(cve if cve.upper().startswith("CVE-") else "CVE-" + cve)
        elif name == "metadata":
            # "service http, policy balanced-ips drop" → {"service": ["http"], "policy": [...]}
            for item in value.split(","):
                k, _, v = item.strip().partition(" ")
                if k:
                    rule["metadata"].setdefault(k, []).append(v.strip())
        else:
            # flowbits, urilen, byte_test 등 나머지 옵션은 원문 그대로 보관
            rule["other_options"].setdefault(name, []).append(value)

    if rule["sid"] is None:
        return None

    # 고유 키: 소스:gid:sid  (ET/Talos 는 SID 대역이 달라 겹치지 않음)
    rule["key"] = f"{source}:{rule['gid']}:{rule['sid']}"
    # 내용 지문: 활성/비활성 여부('#')를 제외한 룰 본문의 해시 → 변경 감지용
    body = line.strip().lstrip("#").strip()
    rule["sig"] = hashlib.sha1(body.encode("utf-8")).hexdigest()
    rule["raw"] = body
    return rule


# ════════════════════════════════════════════════════════════════════════
#  3. HTTP 판별 + method 추출
# ════════════════════════════════════════════════════════════════════════

def methods_matching_content(value, nocase):
    """http_method content 하나가 허용하는 method 집합.

    - 값이 알려진 method 와 정확히 같으면 그 method 하나 (예: "GET" → {GET})
    - 일부만 있으면 그 문자열을 포함하는 method 들 (예: "P" → {POST, PUT, PATCH, ...})
    - 알려진 목록에 없으면 값 그대로 (특이 method 공격 등)
    """
    value = value.strip()   # "propfind " 처럼 끝 공백이 붙은 경우 정리
    cmp_v = value.upper() if nocase else value
    if cmp_v.upper() in KNOWN_METHODS:
        return {cmp_v.upper() if nocase else value}
    found = {mt for mt in KNOWN_METHODS if cmp_v in (mt if not nocase else mt.upper())}
    return found or {value}


def methods_matching_pcre(p):
    """pcre /M 패턴에 실제로 매칭되는 method 집합. (예: /^(?:GET|POST)$/M → {GET, POST})"""
    flags = re.I if "i" in p["flags"] else 0
    try:
        rx = re.compile(p["pattern"], flags)
    except re.error:
        return set()
    return {mt for mt in KNOWN_METHODS if rx.search(mt)}


_RAW_METHOD_RE = re.compile(r"(?:GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS) /", re.I)
_RAW_PCRE_RE = re.compile(r"\^\(?(?:\?:)?(?:GET|POST|PUT)")


def raw_http_request(rule):
    """HTTP 키워드 없이 원시 TCP 로 HTTP 요청을 검사하는 룰이면 True.

    - content 가 'POST /...' 처럼 요청 줄로 시작하거나 ' HTTP/1.' 를 포함
    - content 가 '/' 로 시작하고 depth 64 이하 (패킷 맨 앞의 요청 경로)
    - pcre 가 ^(GET|POST|PUT ...) 로 시작
    """
    for c in rule["contents"]:
        if c["buffer"] != "raw" or c["negated"]:
            continue
        v = c["value"]
        if _RAW_METHOD_RE.match(v) or " HTTP/1." in v or v.startswith("HTTP/1."):
            return True
        d = c.get("depth")
        if v.startswith(("/", " /")) and isinstance(d, int) and d <= 64:
            return True
    return any(p["buffer"] == "raw" and not p["negated"] and _RAW_PCRE_RE.match(p["pattern"])
               for p in rule["pcres"])


def analyze_http(rule):
    """룰이 HTTP 룰인지 판별하고, 요청/응답 구분과 method 를 추출한다.

    HTTP 판별 근거 (하나라도 해당되면 HTTP 룰):
      - content 에 http_* 수식어 사용 (http_uri, http_header ...)
      - pcre 에 HTTP 플래그 사용 (U, H, P, M ...)
      - metadata: service http
      - 헤더 포트에 $HTTP_PORTS / $FILE_DATA_PORTS
      - urilen 옵션 사용
      - HTTP 키워드 없이 원시 TCP 로 HTTP 요청을 검사 (raw_http_request)
          예) content:"POST /api/v2/files/ HTTP|2f|"   pcre:"/^(GET|POST)\\x20\\x2fapi.../"
          → 비표준 포트(3000, 7860, 8265 등)를 쓰는 AI 서비스(Langflow, Flowise 등) 룰이 이 형태가 많다

    method 추출 우선순위:
      1) http_method content      → method_source = "http_method"
      2) pcre /M                  → method_source = "pcre_M"
      3) 원시 content 가 "POST " 처럼 method 로 시작 → method_source = "raw_content"
      4) 근거 없음                → method_source = "none", methods = []
         (참고용 method_hint: 요청 body 검사가 있으면 POST, 아니면 GET)
    """
    reasons, buffers = set(), set()

    for c in rule["contents"]:
        if c["buffer"] in HTTP_BUFFERS:
            reasons.add("content_buffer")
            buffers.add(c["buffer"])
        elif c["buffer"] == "file_data":
            buffers.add("file_data")
    for p in rule["pcres"]:
        if p["buffer"] in HTTP_BUFFERS:
            reasons.add("pcre_flag")
            buffers.add(p["buffer"])
    if "http" in [v.lower() for v in rule["metadata"].get("service", [])]:
        reasons.add("metadata_service")
    if any(v in rule["src_port"] or v in rule["dst_port"] for v in HTTP_PORT_VARS):
        reasons.add("http_ports")
    if "urilen" in rule["other_options"]:
        reasons.add("urilen")
        buffers.add("http_uri")
    raw_req = raw_http_request(rule)
    if raw_req:
        reasons.add("raw_http_request")

    # 요청/응답 구분: flow 방향이 먼저, 없으면 사용 버퍼로 판단
    # (2026-10-30: to_client 응답 룰이 http_cookie(Set-Cookie)·원시 HTTP content 때문에 요청으로 분류되던 것 22룰)
    flow = set(rule["flow"])
    if flow & {"to_client", "from_server"}:
        side = "response"
    elif flow & {"to_server", "from_client"} or buffers & REQUEST_BUFFERS or raw_req:
        side = "request"
    elif buffers & RESPONSE_BUFFERS or "file_data" in buffers:
        side = "response"
    else:
        side = "unknown"

    # method 추출
    allowed, excluded, source = None, set(), "none"
    for c in rule["contents"]:
        if c["buffer"] != "http_method":
            continue
        s = methods_matching_content(c["value"], c["nocase"])
        if c["negated"]:
            excluded |= s
        else:
            allowed = s if allowed is None else allowed & s   # content 여러 개 = AND 조건
            source = "http_method"
    for p in rule["pcres"]:
        if p["buffer"] != "http_method":
            continue
        s = methods_matching_pcre(p)
        if p["negated"]:
            excluded |= s
        elif s:
            allowed = s if allowed is None else allowed & s
            if source == "none":
                source = "pcre_M"
    if allowed is None:
        for c in rule["contents"]:
            if c["buffer"] == "raw" and not c["negated"]:
                head = c["value"].split(" ", 1)[0].upper()
                if head in KNOWN_METHODS and c["value"].upper().startswith(head + " ") \
                        and (c.get("offset", 0) == 0 or "depth" in c):
                    allowed, source = {head}, "raw_content"
                    break

    methods = sorted((allowed or set()) - excluded)
    hint = methods or (["POST"] if "http_client_body" in buffers else ["GET"])
    return {
        "is_http": bool(reasons),
        "reasons": sorted(reasons),
        "side": side,
        "buffers": sorted(buffers),
        "methods": methods,                      # 룰이 명시한 method (없으면 빈 목록 = 제한 없음)
        "methods_excluded": sorted(excluded),    # content:!"POST"; http_method 같은 제외 조건
        "method_source": source,
        "method_hint": hint if source == "none" else methods,
    }


# ════════════════════════════════════════════════════════════════════════
#  4. 변경분 추출
# ════════════════════════════════════════════════════════════════════════

def compute_delta(current, baseline):
    """현재 HTTP 룰과 직전 HTTP 룰을 key 기준으로 비교한다.

    - new     : 직전에 없던 key
    - updated : key 는 같지만 rev 또는 내용 지문(sig)이 바뀜
                (활성/비활성 전환만 된 경우는 내용이 같으므로 제외)
    - removed : 직전에 있었는데 지금 없는 key
    """
    new, updated = [], []
    for key, r in current.items():
        old = baseline.get(key)
        if old is None:
            new.append(r)
        elif old["rev"] != r["rev"] or old["sig"] != r["sig"]:
            updated.append(dict(r, prev_rev=old["rev"]))
    removed = [baseline[k] for k in baseline if k not in current]
    return new, updated, removed


def to_llm_record(r, change):
    """LLM 입력용으로 필요한 필드만 남긴 간단한 레코드.

    change: "new"(신규) / "updated"(rev·내용 변경) / "selection_added"(선별 조건 변경으로 새로 포함)
    """
    h = r["http"]
    return {
        "key": r["key"], "sid": r["sid"], "rev": r["rev"], "msg": r["msg"],
        "category": r["category"],
        "side": h["side"], "methods": h["methods"], "method_hint": h["method_hint"],
        "methods_excluded": h["methods_excluded"],
        "flow": r["flow"], "classtype": r["classtype"], "cve": r["cve"],
        "contents": [{k: v for k, v in c.items() if k != "raw"} for c in r["contents"]],
        "pcres": r["pcres"],
        "change": change,
        "rule": r["raw"],
    }


# ════════════════════════════════════════════════════════════════════════
#  5. 소스 하나 처리
# ════════════════════════════════════════════════════════════════════════

def custom_fetch_info():
    """00_custom_rules/*.rules 의 '버전' = 파일 이름·내용 해시. 파일이 없으면 None."""
    files = sorted(CUSTOM_DIR.glob("*.rules")) if CUSTOM_DIR.is_dir() else []
    if not files:
        return None
    h = hashlib.sha1()
    for f in files:
        h.update(f.name.encode("utf-8") + b"\0" + f.read_bytes() + b"\0")
    return {"marker": h.hexdigest()[:12], "file": CUSTOM_DIR.name, "custom": True}


def process_source(source, fetch_info, parse_state, force):
    """소스 하나(et / talos / custom)를 압축 해제 → 파싱 → 필터링 → 변경분 추출까지 처리한다.

    custom = 00_custom_rules/*.rules (사용자가 직접 넣은 룰). 압축 해제 없이 파일을 그대로 읽는다.
    """
    marker = fetch_info["marker"]
    archive = BASE_DIR / fetch_info["file"]
    prev = parse_state.get(source, {})
    selection = sel.fingerprint() + "/" + PARSER_VERSION   # 선별조건.md 의 O/X 지문 + 파싱 규칙 버전

    # 같은 버전 + 같은 선별 조건 + 같은 파싱 규칙이면 건너뜀. 하나라도 바뀌면 같은 룰 버전이라도 다시 파싱
    if prev.get("marker") == marker and not force:
        if prev.get("selection") == selection:
            log.info("[%s] 이미 파싱한 버전 (%s) - 건너뜀", source, marker)
            return False
        log.info("[%s] 선별 조건 또는 파싱 규칙 변경 감지 (%s -> %s) - 다시 파싱",
                 source, prev.get("selection"), selection)
    if not archive.exists():
        raise RuntimeError(f"압축 파일 없음: {fetch_info['file']}")

    # 같은 버전을 --force 로 다시 파싱하면, 비교 대상은 '그 이전' 결과를 그대로 사용
    baseline_dir = prev.get("baseline_dir") if prev.get("marker") == marker else prev.get("dir")

    if fetch_info.get("custom"):
        date = dt.date.today().isoformat()          # 사용자 룰은 처리한 날짜
    else:
        date = archive.parent.name                  # 01_raw/<source>/<날짜>/ 의 날짜
    out_dir = PROCESSED_DIR / source / date
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 압축 해제 (사용자 룰은 파일을 그대로 읽음)
    if fetch_info.get("custom"):
        files = {f.name: f.read_text(encoding="utf-8", errors="replace") for f in sorted(CUSTOM_DIR.glob("*.rules"))}
        log.info("[%s] 사용자 룰 파일: %d개 (%s)", source, len(files), ", ".join(files))
    else:
        files = extract_rules(source, archive, EXTRACT_DIR / source)
        log.info("[%s] 압축 해제: .rules %d개", source, len(files))

    # 2) 파싱 + 3) 선별 + HTTP 판별
    current, errors, index = {}, [], []
    total = enabled = 0
    # 카테고리별 통계 (보고서용): {카테고리: {"rules": 전체, "http": HTTP 룰, "selected": 선택 여부}}
    cat_stats = {}
    if source == "talos":   # 룰이 0개인 빈 파일도 보고서에 보이도록 미리 등록
        for fname in files:
            cat = fname.rsplit(".rules", 1)[0]
            cat_stats[cat] = {"rules": 0, "http": 0,
                              "selected": sel.is_selected({"source": source, "file": fname})}
    for fname, text in sorted(files.items()):
        for lineno, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if not s:
                continue
            if not RULE_RE.match(s):
                # 주석이 아닌데 룰 형식이 아니면 파싱 오류로 기록 (설명 주석은 무시)
                if not s.startswith("#"):
                    errors.append(f"{fname}:{lineno}: {s[:200]}")
                continue
            rule = parse_rule(s, source, fname)
            if rule is None:
                errors.append(f"{fname}:{lineno}: {s[:200]}")
                continue
            total += 1
            enabled += rule["enabled"]
            rule["http"] = analyze_http(rule)              # (사용자 룰 카테고리는 요청/응답 방향도 보므로 먼저)
            rule["category"] = sel.rule_category(rule)
            selected = sel.is_selected(rule)

            st = cat_stats.setdefault(rule["category"], {"rules": 0, "http": 0,
                                                         "selected": sel.category_selected(source, rule["category"])})
            st["rules"] += 1
            st["http"] += rule["http"]["is_http"]

            # 선별 카테고리 + HTTP 룰만 남긴다
            if selected and rule["http"]["is_http"]:
                current[rule["key"]] = rule
            # 사용자 룰 중복 제거용 색인: 선별과 상관없이 모든 HTTP 룰 (Talos·ET)
            if source != "custom" and rule["http"]["is_http"]:
                index.append(dedup_index_entry(rule, selected))

    http_rules = sorted(current.values(), key=lambda r: r["sid"])
    if source != "custom":
        write_jsonl(out_dir / "dedup_index.jsonl", index)

    # 직전 결과에 어떤 카테고리가 들어 있었는지 (덮어쓰기 전에 읽어 둠)
    # → 선별조건.md 에서 새로 O 로 바꾼 카테고리의 룰을 '선별 추가'로 LLM 입력에 넣기 위함
    prev_cats = None
    if prev.get("dir"):
        for name in ("http_rules.jsonl", "http_rules.jsonl.gz"):
            path = BASE_DIR / prev["dir"] / name
            if path.exists():
                prev_cats = {sel.rule_category(r) for r in read_jsonl(path)}
                break

    write_jsonl(out_dir / "http_rules.jsonl", http_rules)

    # 4) 변경분 추출 - 직전 결과(baseline)의 http_rules 와 비교
    #    (예전 버전이 만든 .gz 도 읽을 수 있도록 두 이름을 차례로 찾는다)
    #    직전 결과도 '지금의 선별 기준'으로 거른 뒤 비교한다.
    #    → 선별 기준을 바꿔도, 빠진 카테고리가 '삭제'로 대량 잡히지 않는다.
    baseline = {}
    if baseline_dir:
        for name in ("http_rules.jsonl", "http_rules.jsonl.gz"):
            path = BASE_DIR / baseline_dir / name
            if path.exists():
                baseline = {r["key"]: r for r in read_jsonl(path) if sel.is_selected(r)}
                break
    new, updated, removed = compute_delta(current, baseline)

    # 선별 추가: 신규/변경은 아니지만, 직전 결과에 없던 카테고리라서 이번에 처음 들어온 룰
    done = {r["key"] for r in new + updated}
    added = []
    if prev_cats is not None:
        added = [r for r in http_rules if r["key"] not in done and r["category"] not in prev_cats]

    # delta.json 은 목록만 (내용 전체는 llm_queue.jsonl / http_rules 에 있음)
    write_json(out_dir / "delta.json", {
        "baseline": baseline_dir,
        "selection": selection,
        "new": [{"key": r["key"], "rev": r["rev"], "msg": r["msg"]} for r in new],
        "updated": [{"key": r["key"], "prev_rev": r["prev_rev"], "rev": r["rev"], "msg": r["msg"]}
                    for r in updated],
        "removed": [{"key": r["key"], "rev": r["rev"], "msg": r["msg"]} for r in removed],
        "selection_added": [{"key": r["key"], "category": r["category"], "msg": r["msg"]} for r in added],
    })
    write_jsonl(out_dir / "llm_queue.jsonl",
                [to_llm_record(r, "new") for r in new]
                + [to_llm_record(r, "updated") for r in updated]
                + [to_llm_record(r, "selection_added") for r in added])

    # 요약 통계
    method_counter = collections.Counter(m for r in http_rules for m in r["http"]["methods"])
    summary = {
        "source": source, "marker": marker, "archive": fetch_info["file"],
        "parsed_at": dt.datetime.now().isoformat(timespec="seconds"),
        "baseline": baseline_dir,
        "selection": selection,
        "rules_total": total, "rules_enabled": enabled,
        "http_rules": len(http_rules),          # 선별 카테고리 + HTTP 룰 수
        "by_category": {c: cat_stats[c] for c in sorted(cat_stats)},
        "http_rules_enabled": sum(r["enabled"] for r in http_rules),
        "http_side": dict(collections.Counter(r["http"]["side"] for r in http_rules)),
        "method_source": dict(collections.Counter(r["http"]["method_source"] for r in http_rules)),
        "methods": dict(method_counter.most_common()),
        "delta": {"new": len(new), "updated": len(updated), "removed": len(removed),
                  "selection_added": len(added)},
        "parse_errors": len(errors), "parse_error_samples": errors[:20],
    }
    write_json(out_dir / "summary.json", summary)

    parse_state[source] = {
        "marker": marker,
        "dir": str(out_dir.relative_to(BASE_DIR)),
        "baseline_dir": baseline_dir,
        "selection": selection,
        "parsed_at": summary["parsed_at"],
    }
    log.info("[%s] 파싱 완료: 전체 %d (활성 %d) / 선별 HTTP %d / 신규 %d, 변경 %d, 삭제 %d, 선별 추가 %d / 오류 %d",
             source, total, enabled, len(http_rules), len(new), len(updated), len(removed), len(added),
             len(errors))
    return True


# ════════════════════════════════════════════════════════════════════════
#  6. 통합 폴더 (ET + Talos 합치기) + 선별 보고서
# ════════════════════════════════════════════════════════════════════════

def write_json_array(path, header, key, records):
    """큰 JSON 을 '레코드 한 줄에 하나' 형태로 저장한다. (에디터에서 보기 쉽고, 유효한 JSON)

    결과 모양:
        {"generated_at": ..., ...header...,
         "<key>": [
        {...룰 1...},
        {...룰 2...}
        ]}
    """
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        head = json.dumps(header, ensure_ascii=False)[:-1]      # 마지막 '}' 제거
        f.write(head + (", " if header else "") + json.dumps(key) + ": [\n")
        f.write(",\n".join(json.dumps(r, ensure_ascii=False) for r in records))
        f.write("\n]}\n")
    tmp.replace(path)


# ── 중복 제거 (Talos ↔ ET) ─────────────────────────────────────────────

_NORM_BUF = {"http_raw_uri": "http_uri", "http_raw_header": "http_header", "http_raw_cookie": "http_cookie"}


def detect_sig(r):
    """기준1 '탐지 내용 동일'용 지문: content(검사 위치·값·부정, 대소문자 무시) + pcre(패턴·부정).
    content 수식어(depth, nocase 등)는 무시한다. content 와 pcre 가 모두 없으면 None."""
    cs = tuple(sorted((_NORM_BUF.get(c["buffer"], c["buffer"]), c["value"].lower(), c["negated"])
                      for c in r["contents"]))
    ps = tuple(sorted((p["pattern"], p["negated"]) for p in r["pcres"]))
    return (cs, ps) if (cs or ps) else None


def uri_tokens(r):
    """기준2 '같은 경로'용: URI 검사 값 중 경로·파라미터 형태인 것.

    - '/', '.', '=' 중 하나를 포함하고 영숫자가 3자 이상 (예: /ecp/, login.cgi, mesajno=)
    - 'select', 'union' 같은 SQL 키워드나 '/../' 같은 기호뿐인 값은 제외 → 범용 룰과 잘못 묶이는 것 방지
    """
    out = set()
    for c in r["contents"]:
        v = c["value"].lower()
        if (_NORM_BUF.get(c["buffer"], c["buffer"]) == "http_uri" and not c["negated"]
                and any(ch in v for ch in "/.=") and sum(ch.isalnum() for ch in v) >= 3):
            out.add(v)
    return out


def dedup_index_entry(rule, selected):
    """사용자 룰 중복 제거용 색인 한 줄 (탐지 내용 지문 해시, CVE, URI 경로 값, 선별 여부)."""
    sig = detect_sig(rule)
    return {"key": rule["key"], "msg": rule["msg"], "selected": selected,
            "sig": hashlib.sha1(json.dumps(sig, ensure_ascii=False).encode()).hexdigest()[:16] if sig else None,
            "cve": [c.upper() for c in rule["cve"]], "uri": sorted(uri_tokens(rule))}


def find_custom_duplicates(rules, parse_state, opts):
    """사용자 룰(custom) 중 Talos·ET 의 룰과 같은 것 → {key: 정보}. Talos·ET 쪽이 선별에서 빠진 룰이어도 같으면 뺀다.
    (사내 탐지정책의 일부 채널은 Snort.org 유료 룰 = Talos 를 옮긴 것이 대부분)"""
    if not opts["enabled"]:
        return {}
    by_sig, by_cve = {}, {}
    for src in ("talos", "et"):
        st = parse_state.get(src)
        path = BASE_DIR / st["dir"] / "dedup_index.jsonl" if st else None
        if not path or not path.exists():
            continue
        for e in read_jsonl(path):
            if e["sig"] and opts["exact"]:
                by_sig.setdefault(e["sig"], e)
            if opts["cve_uri"]:
                for c in e["cve"]:
                    by_cve.setdefault(c, []).append(e)
    dups = {}
    for r in rules:
        if r["source"] != "custom":
            continue
        e = dedup_index_entry(r, True)
        m = by_sig.get(e["sig"]) if e["sig"] else None
        crit = "탐지 내용 동일"
        if m is None and opts["cve_uri"]:
            toks = set(e["uri"])
            for c in e["cve"]:
                for cand in by_cve.get(c, []):
                    shared = toks & set(cand["uri"])
                    if shared:
                        m, crit = cand, f"같은 CVE({c}) + 같은 경로({sorted(shared)[0]})"
                        break
                if m:
                    break
        if m:
            dups[r["key"]] = {"rule": r, "matches": [{"key": m["key"], "msg": m["msg"], "criterion": crit,
                                                      "selected": m["selected"]}]}
    return dups


def find_duplicates(rules, opts):
    """남길 소스(keep)의 룰과 중복되는, 다른 소스의 룰을 찾는다.

    반환: {제거할 key: {"rule": 룰, "matches": [{"key", "msg", "criterion"}, ...]}}
    """
    if not opts["enabled"]:
        return {}
    keep = opts["keep"]
    kept = [r for r in rules if r["source"] == keep]
    # 중복 제거는 Talos ↔ ET 사이에서만 (사용자 룰 custom 은 그대로 둔다)
    others = [r for r in rules if r["source"] in ("et", "talos") and r["source"] != keep]

    by_sig, by_cve = collections.defaultdict(list), collections.defaultdict(list)
    for r in kept:
        sig = detect_sig(r)
        if sig:
            by_sig[sig].append(r)
        for cve in r["cve"]:
            by_cve[cve.upper()].append(r)

    dups = {}
    for r in others:
        matches = []
        if opts["exact"]:
            sig = detect_sig(r)
            for k in by_sig.get(sig, []) if sig else []:
                matches.append({"key": k["key"], "msg": k["msg"], "criterion": "탐지 내용 동일"})
        if opts["cve_uri"] and not matches:
            toks = uri_tokens(r)
            seen = set()
            for cve in r["cve"]:
                for k in by_cve.get(cve.upper(), []):
                    shared = toks & uri_tokens(k)
                    if shared and k["key"] not in seen:
                        seen.add(k["key"])
                        matches.append({"key": k["key"], "msg": k["msg"],
                                        "criterion": f"같은 CVE({cve.upper()}) + 같은 경로({sorted(shared)[0]})"})
        if matches:
            dups[r["key"]] = {"rule": r, "matches": matches}
    return dups


def build_integrated(parse_state):
    """ET + Talos 의 최신 결과를 04_integrated/<오늘>/ 에 합치고, 양쪽에 같은 룰이 있으면 한쪽만 남긴다.

    - http_rules.json : 각 소스 최신 03_processed 의 http_rules 를 합친 것 (중복 제거 후)
    - llm_queue.json  : 오늘 파싱된 소스의 llm_queue 를 합친 것 (중복 제거 후)
    - duplicates.json : 중복으로 뺀 룰과, 그 룰과 같다고 판단한 남긴 쪽 룰 + 판단 기준
    - selection_report.md : 카테고리별 선택/제외 현황, 중복 제거 결과
    """
    today = dt.date.today().isoformat()
    out_dir = INTEGRATED_DIR / today
    out_dir.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now().isoformat(timespec="seconds")
    opts = sel.dedup_options()

    rules, queue, sources, summaries = [], [], {}, {}
    for source in ("et", "talos", "custom"):
        st = parse_state.get(source)
        if not st:
            continue
        pdir = BASE_DIR / st["dir"]
        if not (pdir / "http_rules.jsonl").exists():
            log.warning("[통합] %s 결과 없음 (%s) - 통합에서 제외", source, st["dir"])
            continue
        src_rules = read_jsonl(pdir / "http_rules.jsonl")
        rules += src_rules
        summaries[source] = read_json(pdir / "summary.json", {})
        sources[source] = {"marker": st["marker"], "processed_dir": st["dir"], "http_rules": len(src_rules)}
        if st.get("parsed_at", "")[:10] == today:      # 오늘 파싱된 소스의 변경분만 LLM 입력으로
            items = read_jsonl(pdir / "llm_queue.jsonl")
            queue += [dict(r, source=source) for r in items]
            sources[source]["llm_queue"] = len(items)

    # 중복 제거: Talos ↔ ET, 그리고 사용자 룰 ↔ Talos·ET(선별과 상관없이 전체)
    dups = find_duplicates(rules, opts)
    dups.update(find_custom_duplicates(rules, parse_state, opts))
    rules = [r for r in rules if r["key"] not in dups]
    queue_before = len(queue)
    queue = [r for r in queue if r["key"] not in dups]
    for src in sources:
        sources[src]["after_dedup"] = sum(1 for r in rules if r["source"] == src)

    dedup_info = {"options": opts, "removed": len(dups),
                  "removed_from_llm_queue": queue_before - len(queue),
                  "by_criterion": dict(collections.Counter(
                      "탐지 내용 동일" if d["matches"][0]["criterion"] == "탐지 내용 동일" else "같은 CVE + 같은 경로"
                      for d in dups.values())),
                  "by_source": dict(collections.Counter(d["rule"]["source"] for d in dups.values()))}

    write_json_array(out_dir / "http_rules.json",
                     {"generated_at": now, "sources": sources, "dedup": dedup_info, "count": len(rules)},
                     "rules", rules)
    write_json_array(out_dir / "llm_queue.json",
                     {"generated_at": now, "sources": {s: v for s, v in sources.items() if "llm_queue" in v},
                      "dedup_removed": dedup_info["removed_from_llm_queue"], "count": len(queue)},
                     "items", queue)
    write_json_array(out_dir / "duplicates.json",
                     {"generated_at": now, "options": opts, "count": len(dups)}, "removed",
                     [{"key": d["rule"]["key"], "category": d["rule"]["category"], "msg": d["rule"]["msg"],
                       "cve": d["rule"]["cve"], "same_as": d["matches"]}
                      for d in sorted(dups.values(), key=lambda d: d["rule"]["sid"])])
    (out_dir / "selection_report.md").write_text(selection_report(summaries, now, dedup_info, dups,
                                                                  {r["key"].lower() for r in rules}),
                                                 encoding="utf-8")
    log.info("[통합] %s: http_rules %d개 (중복 제거 %d), llm_queue %d개 (중복 제거 %d)",
             out_dir.relative_to(BASE_DIR), len(rules), len(dups), len(queue), dedup_info["removed_from_llm_queue"])


def selection_report(summaries, now, dedup_info=None, dups=None, present=None):
    """summary.json 의 카테고리 통계 + 중복 제거 결과로 선별 보고서(markdown)를 만든다."""
    lines = [f"# 룰 선별 결과 ({now[:10]})", "",
             f"선별 조건은 `docs/코드참조/선별조건.md` (지문 {sel.fingerprint()}), 상세 근거는 `docs/보고용/룰_선별기준.md` 참고. "
             "'추출' = 선택 카테고리의 HTTP 룰 = 03_processed/ 와 04_integrated/ 에 들어간 룰 수.", ""]
    for source, title in (("talos", "Cisco Talos"), ("et", "Proofpoint ET Open"),
                          ("custom", "사용자 추가 룰 (00_custom_rules)")):
        s = summaries.get(source)
        if not s or "by_category" not in s:
            continue
        cats = s["by_category"]
        chosen = {c: v for c, v in cats.items() if v["selected"]}
        lines += [f"## {title} (버전 {s['marker'][:12]})", "",
                  f"전체 {s['rules_total']:,}개 중 선택 카테고리 {sum(v['rules'] for v in chosen.values()):,}개, "
                  f"그중 HTTP 룰 **{s['http_rules']:,}개**를 추출했다.", "",
                  "| 카테고리 | 선택 | 전체 룰 | HTTP 룰 | 추출 | 이유 |",
                  "| --- | :---: | ---: | ---: | ---: | --- |"]
        # 선택된 것 먼저(추출 수 많은 순), 그 다음 제외된 것(전체 룰 많은 순)
        order = sorted(cats.items(), key=lambda kv: (not kv[1]["selected"],
                                                     -(kv[1]["http"] if kv[1]["selected"] else kv[1]["rules"]),
                                                     kv[0]))
        for c, v in order:
            mark = "O" if v["selected"] else "-"
            got = f"{v['http']:,}" if v["selected"] else "0"
            lines.append(f"| {c} | {mark} | {v['rules']:,} | {v['http']:,} | {got} | {sel.reason(source, c)} |")
        lines.append("")

    # 개별 룰 (카테고리와 상관없이 넣거나 뺀 룰)
    ind = sel.individual_rules()
    if ind:
        lines += ["## 개별 룰", "", "| 룰 키 | 선택 | 통합 결과에 있음 | 이유 |", "| --- | :---: | :---: | --- |"]
        for k, (chosen, why) in sorted(ind.items()):
            has = "-" if present is None else ("O" if k in present else "X")
            lines.append(f"| {k} | {'O' if chosen else 'X'} | {has} | {why} |")
        lines.append("")

    # 중복 제거 결과
    if dedup_info is not None:
        o = dedup_info["options"]
        lines += ["## Talos ↔ ET 중복 제거", ""]
        if not o["enabled"]:
            lines += ["중복 제거 사용 안 함 (`선별조건.md`의 '사용' = X).", ""]
        else:
            other = "et" if o["keep"] == "talos" else "talos"
            crit = [n for n, on in (("탐지 내용 동일", o["exact"]), ("같은 CVE + 같은 경로", o["cve_uri"])) if on]
            lines += [f"{o['keep']} 룰을 남기고, 그와 같은 {other} 룰 **{dedup_info['removed']:,}개**를 통합 결과에서 뺐다 "
                      f"(llm_queue 에서 {dedup_info['removed_from_llm_queue']:,}개). 기준: {', '.join(crit) or '없음'}. "
                      "전체 목록은 `duplicates.json`.", "",
                      "| 기준 | 제거 수 |", "| --- | ---: |"]
            for k, v in sorted(dedup_info["by_criterion"].items(), key=lambda kv: -kv[1]):
                lines.append(f"| {k} | {v:,} |")
            cat = collections.Counter(d["rule"]["category"] for d in dups.values())
            if cat:
                lines += ["", "| 제거된 룰의 카테고리 | 제거 수 |", "| --- | ---: |"]
                lines += [f"| {c} | {n:,} |" for c, n in cat.most_common()]
            lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Snort 룰 파싱/HTTP 필터링/변경분 추출")
    parser.add_argument("--source", choices=["et", "talos", "custom", "all"], default="all")
    parser.add_argument("--force", action="store_true", help="이미 파싱한 버전도 다시 파싱")
    args = parser.parse_args(argv)

    setup_logging()
    # 선별조건.md 읽기 - 형식이 잘못됐으면 아무것도 바꾸지 않고 종료 (기존 결과 유지)
    try:
        sel.reload()
    except (ValueError, FileNotFoundError) as e:
        log.error("선별 조건 읽기 실패 - 파싱 중단 (기존 결과는 그대로): %s", e)
        return 1
    log.info("선별 조건: 선별조건.md (지문 %s)", sel.fingerprint())
    fetch_state = read_json(FETCH_STATE_FILE, {})
    parse_state = read_json(PARSE_STATE_FILE, None)
    if parse_state is None:
        # 첫 실행이거나 파일이 깨진 경우: 기존 파싱 결과가 있으면 그것으로 복구
        parse_state = rebuild_parse_state()
        if parse_state:
            write_json(PARSE_STATE_FILE, parse_state)
            log.info("parse_state.json 복구: %s",
                     ", ".join(f"{k}={v['dir']}" for k, v in parse_state.items()))

    sources = ["et", "talos", "custom"] if args.source == "all" else [args.source]
    failed = False
    parsed_now = []
    for source in sources:
        if source == "custom":
            info = custom_fetch_info()
            if info is None:
                if "custom" in parse_state:          # 사용자 룰을 모두 지웠으면 통합 결과에서도 뺀다
                    del parse_state["custom"]
                    write_json(PARSE_STATE_FILE, parse_state)
                    parsed_now.append("custom")
                    log.info("[custom] 사용자 룰 파일이 없어짐 - 통합 결과에서 제외")
                continue
        elif source not in fetch_state:
            log.info("[%s] 다운로드 기록 없음 - 건너뜀 (먼저 1_1_fetch_rules.py 실행)", source)
            continue
        else:
            info = fetch_state[source]
        try:
            if process_source(source, info, parse_state, args.force):
                write_json(PARSE_STATE_FILE, parse_state)
                parsed_now.append(source)
        except Exception as e:
            failed = True
            log.exception("[%s] 파싱 실패: %s", source, e)

    # 통합 폴더는 '새로 파싱한 소스가 있을 때' 또는 '중복 제거 설정이 바뀌었을 때' 다시 만든다
    dedup_fp = sel.dedup_fingerprint()
    integ = parse_state.get("integrated", {})
    if parsed_now or integ.get("dedup") != dedup_fp:
        if not parsed_now:
            log.info("[통합] 중복 제거 설정 변경 감지 (%s -> %s) - 통합 결과 다시 생성", integ.get("dedup"), dedup_fp)
        try:
            build_integrated(parse_state)
            parse_state["integrated"] = {"dedup": dedup_fp,
                                         "built_at": dt.datetime.now().isoformat(timespec="seconds")}
            write_json(PARSE_STATE_FILE, parse_state)
        except Exception as e:
            failed = True
            log.exception("[통합] 실패: %s", e)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
