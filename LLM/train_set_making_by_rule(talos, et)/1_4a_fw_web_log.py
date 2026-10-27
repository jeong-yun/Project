"""[1.4a FW·웹 로그 형식] IPS 신규 학습 데이터 줄 → 방화벽(FW)·웹(nginx) 로그 형식.  (0_0_run_ui.py 화면 4 가 씀)

화면 4(신규 학습 데이터)에서 장비를 '방화벽(FW)'·'웹 로그'로 고르면 IPS 학습 로그(08_dataset/<날짜>/ips_*_snort.log)와
label 파일의 줄을 이 형식으로 바꿔 보여 주고 CSV 로 내려받게 한다. '파일로 저장'(명령줄 --save)은 날짜 전체를 아래 [저장 파일]로
쓴다 (1.4 학습 로그·6. 확정 파일은 그대로).
IPS·WAF 학습 로그는 payload 가 같고 목적 포트·method 만 달라 IPS 를 원본으로 쓴다.
형식·열 순서·환경 값(장비 이름·IP·도메인 등)은 docs/코드참조/FW_웹로그_형식.md 에 둔다 (없으면 아래 DEFAULTS 의 일반 예시 값).

[FW 로그]  syslog(RFC 5424) 머리 + CSV 34열 (md '## FW 열': 예시 로그 2건의 값과 SIEM 파싱 결과를 맞춰 정함)
    <14>1 2026-10-27T00:05:30.458276Z [fw4_allow] [<장비 IP>] <시작>,<끝>,<세션 시간>,<장비>,<룰 id>,<룰 이름>,<NAT 룰 id>,<NAT 룰 이름>,<s_ip>,-,<s_port>,…
  - 5-tuple 은 원본 그대로, protocol·앱은 목적 포트별 (md '## FW 포트')
  - 바이트·패킷: 보낸 쪽은 요청 길이로, 받은 쪽은 만든 응답 크기로 TCP 근사 (SYN·ACK·FIN 포함)
  - payload 가 없다 → 공격 근거가 로그에 없다. label 은 원본 샘플의 것을 보여 줄 뿐 FW 줄로는 판단할 수 없다

[웹 로그]  syslog(RFC 3164) 머리 + nginx combined + "$http_x_forwarded_for""[$request_body]"
    <190>Oct 26 18:00:52 <호스트> nginx: <s_ip> - - [26/Oct/2026:18:00:52 +0900] "GET /… HTTP/1.1" 200 773 "-" "<UA>" "-""[-]"
  - 남는 것: 요청 줄·User-Agent·Referer·X-Forwarded-For·본문, Basic 인증의 사용자 이름($remote_user)
  - 사라지는 것: 그 밖의 헤더 (Host·Cookie·Content-Type·사용자 정의 헤더 등)
    → anomalies 인데 룰 조건이 사라지는 헤더에서만 맞으면 '공격 근거 사라짐' (남는 부분만으로 룰 조건을 다시 대조)
  - 값 표기는 nginx 기본(escape=default): " \\ 와 글자 코드 32 미만·126 초과 바이트(UTF-8) → \\xHH (대문자). 본문 줄바꿈 → \\x0A

[label 과 무관하게]  원본에 없는 값(세션 시간, 응답 코드·크기, 응답 헤더 크기, 시각의 소수점)은 샘플 지문(sk)으로만 정한다.
  label·공격 유형에 따라 달라지면 그 값이 정답을 알려 주는 정보(누수)가 된다.

[저장 파일]  08_dataset/<날짜>/ - 화면의 조건과 상관없이 그 날짜 전체, IPS 학습 로그와 같은 줄 순서
    <fw|web>_<yymmdd>.log          장비가 보내는 원문 로그 한 줄씩 (화면의 'FW 로그'·'웹 로그' 열)
    <fw|web>_<yymmdd>_label.jsonl  줄별 line, id(원본 룰), label, attack_type, sk, evidence (+ 웹: evidence_why, lost_headers)
        evidence: none = FW 로그에 근거 없음, kept·lost = 웹 로그에 근거 남음·사라짐, check = 확인 못 함, na = normal
  - 이름이 *_snort.log 가 아니라 화면의 IPS·WAF 보기·4.1·5.x 는 읽지 않는다. 1.4 로 학습 로그를 다시 만들면 다시 저장한다

[명령줄]
    python 1_4a_fw_web_log.py 2026-10-26 web 3          # 그 날짜 IPS 로그 앞 3줄을 웹 로그로
    python 1_4a_fw_web_log.py 2026-10-26 web --count    # 모든 줄의 공격 근거 상태 수
    python 1_4a_fw_web_log.py 2026-10-26 fw --save      # 파일로 저장 (날짜에 all 이면 모든 날짜)
"""

import argparse
import base64
import datetime as dt
import functools
import hashlib
import importlib.util
import ipaddress
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SETTINGS_FILE = BASE_DIR / "docs" / "코드참조" / "FW_웹로그_형식.md"
DATASET_DIR = BASE_DIR / "08_dataset"
DEVICES = {"fw": "FW", "web": "웹"}

# 설정 md 가 없을 때 쓰는 일반 예시 값 (회사 환경 값은 md 에만 둔다)
DEFAULTS = {
    "공통": {"시간대": "+09:00", "사설 IP 방향": "IN", "공인 IP 방향": "OUT"},
    "FW 설정": {"syslog 머리": "<14>1", "로그 종류": "fw4_allow", "sublog": "allow", "action": "accept",
               "장비 IP": "192.0.2.1", "장비 이름": "FW-01", "device_name": "FW", "룰 id": "1",
               "룰 이름": "WEB_SERVICE_IN", "NAT 룰 id": "0", "NAT 룰 이름": "", "zone": "External", "interface": "eth1",
               "세션 시간(초)": "0-5", "로그 전송 지연(초)": "0-10", "MSS": "1460", "사설 IP 국가 코드": "B2",
               "공인 IP 국가": "-", "TCP 플래그": "-"},
    "웹 설정": {"syslog 머리": "<190>", "호스트 이름": "localhost.localdomain", "태그": "nginx",
               "웹 서버 IP": "192.0.2.10", "웹 서버 이름": "WEB-01", "도메인": "www.example.com", "포트": "80",
               "형식": '{remote_addr} - {remote_user} [{time_local}] "{request}" {status} {body_bytes_sent} '
                      '"{http_referer}" "{http_user_agent}" "{http_x_forwarded_for}""[{request_body}]"'},
}
DEFAULT_PORTS = [["기본", "6", "http", "-", "Network-Protocol"], ["443", "6", "https", "-", "Network-Protocol"]]
DEFAULT_COLUMNS = [   # (값, 파싱 이름) - md '## FW 열' 과 같음
    ("{start}", "event_time"), ("{end}", "?"), ("{duration}", "duration"), ("{device_name}", "device_name"),
    ("{rule_id}", "rule_id"), ("{rule_name}", "rule_name"), ("{nat_rule_id}", "nat_rule_id"),
    ("{nat_rule_name}", "nat_rule_name"),
    ("{s_ip}", "s_ip"), ("-", "?"), ("{s_port}", "s_port"), ("{d_ip}", "d_ip"), ("{d_port}", "d_port"),
    ("{protocol}", "protocol"), ("{app_name}", "app_name"), ("{app_protocol}", "app_protocol"),
    ("{app_category}", "app_category"), ("no", "?"), ("{zone_name}", "zone_name"), ("{interface}", "interface"),
    ("{sent_packets}", "sent_packets"), ("{recv_packets}", "recv_packets"), ("{sent_bytes}", "sent_bytes"),
    ("{recv_bytes}", "recv_bytes"), ("{total_packets}", "total_packets"), ("{total_bytes}", "total_bytes"),
    (" ", "?"), ("{tcp_flag}", "tcp_flag"), ("-", "?"), ("no", "is_ssl"), ("no", "is_sslvpn"), ("-", "?"),
    ("{s_country}", "s_country"), ("{d_country}", "d_country"),
]
DEFAULT_RESPONSES = [["200", "85", "300-30000"], ["404", "7", "153"], ["302", "4", "145"], ["304", "3", "0"],
                     ["500", "1", "177"]]
LITERAL = {"(빈 값)": "", "(공백)": " "}
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
WEB_KEPT = ("user-agent", "referer", "x-forwarded-for")     # 웹 로그에 남는 헤더 (값만, 정해진 자리에)
# 룰 검사 위치 중 웹 로그에도 그대로 있는 곳 (요청 줄·본문). 그 밖(header·cookie·payload 등)은 남는 부분만으로 다시 대조
SAME_IN_WEB = {"method", "uri", "raw_uri", "body", "status_code", "status_msg", "response_body"}
EV_TEXT = {"none": "없음", "kept": "남음", "lost": "사라짐", "check": "확인 못 함", "na": ""}
CTRL_RE = re.compile(r"\\x(0[0-9b-f]|1[0-9a-f]|7f)")       # 1.4 가 글자로 적은 제어문자 (줄바꿈은 두 글자 \n)
NGX_RE = re.compile(r'[\x00-\x1f"\\\x7f-\xff]')
PLACE_RE = re.compile(r"\{(\w+)\}")
SAMPLE_LINE = "2026-01-01 00:00:00|||203.0.113.5|||40000|||192.0.2.10|||80|||x|||GET / HTTP/1.1\\nHost: a|||normal"

_cache = {"mtime": None, "s": None, "seen": 0.0}


# ── 설정 md ─────────────────────────────────────────────────────────────

def md_tables(path):
    """md 의 '## 제목' 별 표 → {제목: [[셀, …], …]} (머리글·구분선 제외, 셀 양끝의 ` 는 뺌)."""
    tables, section, header_seen = {}, None, False
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("## "):
            section, header_seen = s[3:].strip(), False
            tables[section] = []
            continue
        if section is None or not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue
        if not header_seen:
            header_seen = True
            continue
        tables[section].append([c[1:-1] if len(c) >= 2 and c[0] == c[-1] == "`" else c for c in cells])
    return tables


def _range(text, default):
    """'0-5' → (0, 5), '153' → (153, 153)."""
    a, _, b = text.replace(" ", "").partition("-")
    try:
        lo, hi = int(a), int(b or a)
    except ValueError:
        return default
    return (lo, hi) if lo <= hi else (hi, lo)


def settings():
    """설정 md (바뀌었을 때만 다시 읽음, 없으면 DEFAULTS) → {공통·FW 설정·웹 설정: {항목: 값}, ports, columns, responses, source}.
    수정 시각은 2초에 한 번만 본다 (Windows 에서 stat 한 번이 1~2ms 라 줄마다 보면 하루치 변환이 몇 분 늘어남)."""
    if _cache["s"] is not None and time.monotonic() - _cache["seen"] < 2:
        return _cache["s"]
    _cache["seen"] = time.monotonic()
    try:
        mtime = SETTINGS_FILE.stat().st_mtime
    except OSError:
        mtime = 0
    if _cache["mtime"] == mtime:
        return _cache["s"]
    t = md_tables(SETTINGS_FILE) if mtime else {}
    s = {sec: dict(vals) for sec, vals in DEFAULTS.items()}
    for sec in DEFAULTS:
        s[sec].update({r[0]: r[1] for r in t.get(sec, []) if len(r) >= 2 and r[0]})
    s["ports"] = {r[0]: r[1:5] for r in (t.get("FW 포트") or DEFAULT_PORTS) if len(r) >= 5}
    cols = [(r[1], r[2]) for r in t.get("FW 열", []) if len(r) >= 3]
    # (값 이름 | None, 그대로 쓸 값·틀, 파싱 이름): '{s_ip}' 처럼 이름 하나면 바로 찾고, 섞인 틀만 fill
    s["columns"] = [(m.group(1) if m else None, LITERAL.get(v, v), n)
                    for v, n in (cols or DEFAULT_COLUMNS) for m in [re.fullmatch(r"\{(\w+)\}", v)]]
    resp = []
    for r in t.get("응답") or DEFAULT_RESPONSES:
        try:
            resp.append((r[0], float(r[1]), _range(r[2], (0, 0))))
        except (ValueError, IndexError):
            continue
    s["responses"] = [x for x in resp if x[1] > 0] or [("200", 1.0, (300, 30000))]
    s["source"] = f"docs/코드참조/{SETTINGS_FILE.name}" if mtime else "코드의 일반 예시 값 (설정 md 없음)"
    _cache.update(mtime=mtime, s=s)
    return s


# ── 원본 줄 읽기 ────────────────────────────────────────────────────────

def split_log(line):
    """학습 로그 한 줄 → 필드 (0_0_run_ui.split_log 와 같음: payload 가 '|' 로 끝나도 label 은 마지막 '|||' 뒤)."""
    head = line.split("|||", 6)
    if len(head) == 7 and "|||" in head[6]:
        return head[:6] + head[6].rsplit("|||", 1)
    return line.split("|||")


def decode_payload(payload):
    """IPS payload → HTTP 메시지. 화면의 label 정합성 검사와 같은 읽기 (글자로 적은 제어문자 → 바이트, 두 글자 \\n → 줄바꿈).
    본문에 원래 글자 그대로 있던 \\n 은 줄바꿈과 구별되지 않는다 (1.4 형식의 한계)."""
    raw = CTRL_RE.sub(lambda m: chr(int(m.group(1), 16)), payload)
    return raw.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\r\n", "\n").strip()


def split_message(http):
    """HTTP 메시지 → (요청 줄, [(헤더 이름, 값)], 본문). 콜론이 없는 헤더 줄은 값 None."""
    head, _, body = http.partition("\n\n")
    lines = head.split("\n")
    headers = []
    for h in lines[1:]:
        k, colon, v = h.partition(":")
        headers.append((k.strip(), v.strip() if colon else None))
    return lines[0], headers, body


def header(headers, name):
    """헤더 값 (없으면 None, 여러 번이면 nginx 처럼 ', ' 로 이음)."""
    vals = [v for k, v in headers if k.lower() == name and v is not None]
    return ", ".join(vals) if vals else None


def event_time(text):
    """'2026-10-26 18:12:25' → datetime (형식이 다르면 None)."""
    if len(text) != 19 or text[10] != " ":
        return None
    try:
        return dt.datetime.fromisoformat(text)
    except ValueError:
        return None


def tz_offset(text):
    """'+09:00' → timedelta(hours=9)."""
    m = re.fullmatch(r"([+-])(\d{1,2}):?(\d{2})", text.strip())
    if not m:
        return dt.timedelta(hours=9)
    d = dt.timedelta(hours=int(m.group(2)), minutes=int(m.group(3)))
    return -d if m.group(1) == "-" else d


@functools.lru_cache(maxsize=1 << 16)
def is_public(ip):
    try:
        return ipaddress.ip_address(ip).is_global
    except ValueError:
        return False


def direction(ip, s):
    return s["공통"]["공인 IP 방향" if is_public(ip) else "사설 IP 방향"]


# ── 만드는 값 (샘플 지문으로만) ─────────────────────────────────────────

def _h(sk, what):
    """샘플 지문 + 용도 → 정수 (같은 샘플이면 언제나 같은 값, label 과 무관)."""
    return int.from_bytes(hashlib.sha1(f"{sk}|{what}".encode("utf-8")).digest()[:8], "big")


def response(sk, method, s):
    """만든 응답 → (응답 코드, 본문 바이트, 응답 헤더 바이트). md '## 응답' 의 비율대로, 본문 크기 범위는 로그 균등.
    HEAD 요청과 204·304 는 본문 0."""
    table = s["responses"]
    x = _h(sk, "status") / 2 ** 64 * sum(w for _, w, _ in table)
    for code, w, (lo, hi) in table:
        if x < w:
            break
        x -= w
    if method == "HEAD" or code in ("204", "304") or hi <= 0:
        size = 0
    elif lo == hi:
        size = lo
    else:
        a, b = math.log(max(lo, 1)), math.log(hi)
        size = int(round(math.exp(a + _h(sk, "size") / 2 ** 64 * (b - a))))
    return code, size, 150 + _h(sk, "head") % 251


def request_bytes(http):
    """요청 바이트 수 (요청 줄·헤더는 CRLF 줄바꿈 + 빈 줄, 본문은 그대로)."""
    head, _, body = http.partition("\n\n")
    return len(head.encode("utf-8")) + head.count("\n") + 4 + len(body.encode("utf-8"))


def ngx(value):
    """nginx 로그 기본 escape: " \\ 와 글자 코드 32 미만·126 초과 바이트(UTF-8) → \\xHH (대문자)."""
    if isinstance(value, str):
        value = value.encode("utf-8", "replace")
    return NGX_RE.sub(lambda m: f"\\x{ord(m.group()):02X}", value.decode("latin-1"))


def remote_user(headers):
    """$remote_user: Basic 인증 헤더의 사용자 이름 (nginx 는 auth_basic 설정과 상관없이 헤더를 읽는다). 없거나 틀리면 '-'."""
    scheme, _, cred = (header(headers, "authorization") or "").partition(" ")
    if scheme.lower() != "basic":
        return "-"
    try:
        user, colon, _ = base64.b64decode(cred.strip(), validate=True).partition(b":")
    except ValueError:
        return "-"
    return ngx(user) if colon and user else "-"


def fill(template, vals):
    """'{이름}' 을 값으로 (모르는 이름은 그대로)."""
    return PLACE_RE.sub(lambda m: str(vals[m.group(1)]) if m.group(1) in vals else m.group(0), template)


# ── 형식별 한 줄 ────────────────────────────────────────────────────────

def fw_view(f, start, http, sk, s):
    """FW 로그 한 줄과 파싱 결과. 세션 시작 = 학습 로그 시각, 끝 = 시작 + 세션 시간, syslog 머리 = 끝 + 전송 지연(UTC).
    국가 열은 사설 IP 면 장비의 사설 코드(B2), 파싱 결과에서는 SIEM 처럼 '-'."""
    st = s["FW 설정"]
    lo, hi = _range(st["세션 시간(초)"], (0, 5))
    dur = lo + _h(sk, "duration") % (hi - lo + 1)
    end = start + dt.timedelta(seconds=dur)
    lo, hi = _range(st["로그 전송 지연(초)"], (0, 10))
    utc = (end + dt.timedelta(seconds=lo + _h(sk, "delay") % (hi - lo + 1)) - tz_offset(s["공통"]["시간대"])
           + dt.timedelta(microseconds=_h(sk, "usec") % 1000000))
    proto, app, app_proto, app_cat = (s["ports"].get(f[4]) or s["ports"].get("기본") or DEFAULT_PORTS[0][1:])[:4]
    _, body, rhead = response(sk, http.split(" ", 1)[0], s)
    req = request_bytes(http)
    if proto == "17":                                       # UDP: 요청 한 패킷
        sp, sb, rp, rb = 1, 28 + req, 0, 0
    else:                                                   # TCP: 머리 52바이트(SYN 60), 응답 ACK 는 세그먼트 2개당 1개
        mss = int(st["MSS"]) if st["MSS"].isdigit() and int(st["MSS"]) > 0 else 1460
        qs, rs = -(-req // mss), max(1, -(-(rhead + body) // mss))
        sp, rp = 4 + qs + -(-rs // 2), 2 + rs
        sb, rb = 52 * sp + 8 + req, 52 * rp + 8 + rhead + body
    private = st["사설 IP 국가 코드"]
    vals = {"start": f"{start:%Y-%m-%d %H:%M:%S}", "end": f"{end:%Y-%m-%d %H:%M:%S}", "duration": dur,
            "device_name": st["device_name"], "rule_id": st["룰 id"], "rule_name": st["룰 이름"],
            "nat_rule_id": st["NAT 룰 id"], "nat_rule_name": st["NAT 룰 이름"] if st["NAT 룰 id"] not in ("", "0") else "",
            "s_ip": f[1], "s_port": f[2], "d_ip": f[3], "d_port": f[4],
            "protocol": proto, "app_name": app, "app_protocol": app_proto, "app_category": app_cat,
            "zone_name": st["zone"], "interface": st["interface"], "sent_packets": sp, "recv_packets": rp,
            "sent_bytes": sb, "recv_bytes": rb, "total_packets": sp + rp, "total_bytes": sb + rb,
            "tcp_flag": st["TCP 플래그"] if proto == "6" else "-",
            "s_country": st["공인 IP 국가"] if is_public(f[1]) else private,
            "d_country": st["공인 IP 국가"] if is_public(f[3]) else private}
    cols = [str(vals.get(key, "")) if key else fill(v, vals) if "{" in v else v for key, v, _ in s["columns"]]
    raw = f"{st['syslog 머리']} {utc:%Y-%m-%dT%H:%M:%S.%f}Z [{st['로그 종류']}] [{st['장비 IP']}] " + ",".join(cols)
    fields = {n: v for (_, _, n), v in zip(s["columns"], cols) if n not in ("?", "-", "") and v != ""}
    if "event_time" in fields:
        fields["event_time"] += ".000"
    for k in ("s_country", "d_country"):
        if fields.get(k) == private:
            fields[k] = "-"
    for k in ("host", "user_id", "s_country", "tcp_flag"):    # 두 예시에서 '-' 이고 열을 모르는 값 (열 표에 있으면 그 값)
        fields.setdefault(k, "-")
    fields.update(origin=st["장비 IP"], origin_name=st["장비 이름"], logtype="fw", sublog=st["sublog"],
                  action=st["action"], s_direction=direction(f[1], s), d_direction=direction(f[3], s))
    return raw, fields


def web_view(f, t, http, sk, s):
    """웹(nginx) 로그 한 줄과 파싱 결과. 시각 = 학습 로그 시각."""
    st = s["웹 설정"]
    off = int(tz_offset(s["공통"]["시간대"]).total_seconds()) // 60
    first, headers, body = split_message(http)
    code, size, _ = response(sk, first.split(" ", 1)[0], s)

    def hv(name):
        v = header(headers, name)
        return "-" if v is None else ngx(v)

    vals = {"remote_addr": f[1], "remote_user": remote_user(headers),
            "time_local": f"{t.day:02d}/{MONTHS[t.month - 1]}/{t:%Y:%H:%M:%S} "
                          f"{'-' if off < 0 else '+'}{abs(off) // 60:02d}{abs(off) % 60:02d}",
            "request": ngx(first), "status": code, "body_bytes_sent": size, "http_referer": hv("referer"),
            "http_user_agent": hv("user-agent"), "http_x_forwarded_for": hv("x-forwarded-for"),
            "request_body": ngx(body) if body else "-"}
    raw = (f"{st['syslog 머리']}{MONTHS[t.month - 1]} {t.day:2d} {t:%H:%M:%S} {st['호스트 이름']} {st['태그']}: "
           + fill(st["형식"], vals))
    req = vals["request"].split(" ")
    ver = req[-1] if len(req) > 2 and req[-1].startswith("HTTP/") else ""
    url = " ".join(req[1:-1] if ver else req[1:])
    fields = {"action": "accept", "category": "-", "d_port": st["포트"], "domain": st["도메인"],
              "event_time": f"{t:%Y-%m-%d %H:%M:%S}.000", "http_body": f"[{vals['request_body']}]",
              "http_method": req[0], "http_status": code, "http_status_string": code, "http_uri": url.split("?", 1)[0],
              "http_url": url, "http_version": ver, "http_x_forwarded_for": vals["http_x_forwarded_for"],
              "id": vals["remote_user"], "log": "web", "logtype": "weblog", "origin": st["웹 서버 IP"],
              "origin_name": st["웹 서버 이름"], "pkt_bytes": str(size), "protocol": "6", "referer": vals["http_referer"],
              "s_country": "-", "s_direction": direction(f[1], s), "s_ip": f[1], "s_port": "0", "sublog": "access",
              "user_agent": vals["http_user_agent"]}
    return raw, fields


def cond_text(c):
    v = (c.get("value") if c["type"] == "content" else c.get("pattern")) or ""
    return f"{'정규식' if c['type'] == 'regex' else 'content'} {c.get('in', '')} '{v[:40]}{'…' if len(v) > 40 else ''}'"


def web_evidence(http, rid, ctx):
    """anomalies 의 공격 근거가 웹 로그에 남는지 → (state, 사유). 남는 부분(요청 줄·UA·Referer·XFF·본문)만으로 룰 조건을
    다시 대조해 원본에서 맞던 조건이 어긋나면 'lost'. ctx = (3_0a_common, {룰 id: 룰})."""
    C, rules = ctx
    rule = rules.get(rid)
    if rule is None:
        return "check", f"룰 {rid} 가 현재 룰 목록에 없어 확인 못 함"
    first, headers, body = split_message(http)
    kept = [f"{k}: {v}" for k, v in headers if k.lower() in WEB_KEPT and v is not None]
    p0, p1 = C.split_http(http), C.split_http("\n".join([first] + kept) + "\n\n" + body)
    conds = rule.get("conditions", [])
    gone = [c for c in conds if not c.get("negated") and c.get("in", "payload") not in SAME_IN_WEB
            and C.check_condition(c, p0) is True and C.check_condition(c, p1) is False]
    if gone:
        return "lost", (f"룰 {rid} 조건 {len(conds)}개 중 {len(gone)}개가 웹 로그에 없는 헤더에만 있음: {cond_text(gone[0])}"
                        + (f" 외 {len(gone) - 1}개" if len(gone) > 1 else ""))
    return "kept", f"요청 줄·UA·Referer·XFF·본문만으로 룰 {rid} 조건 판정이 원본과 같음"


def convert(dev, line, meta, ctx=None):
    """IPS 학습 로그 한 줄 + label 메타 → 화면·CSV 한 줄
    {raw, fields[(이름, 값)…], evidence{state, text}, lost_headers, label, rule_id, payload, request}.
    evidence.state: none = FW(로그에 근거 없음), kept·lost = 웹 로그에 근거 남음·사라짐, check = 확인 못 함, na = normal.
    ctx = (3_0a_common, {룰 id: 룰}) - 없으면 웹의 공격 근거는 확인하지 않는다."""
    s = settings()
    f = split_log(line)
    t = event_time(f[0]) if len(f) == 8 else None
    ok = t is not None
    out = {"label": f[7] if ok else "", "rule_id": meta.get("id", ""), "payload": f[6] if ok else line, "request": "",
           "raw": "", "fields": [], "lost_headers": [], "evidence": {"state": "check", "text": ""}}
    if not ok:
        out["evidence"]["text"] = f"학습 로그 줄 형식이 아님 (필드 {len(f)}개)"
        return out
    http = decode_payload(f[6])
    sk = meta.get("sk") or hashlib.sha1(line.encode("utf-8")).hexdigest()[:16]
    first, headers, _ = split_message(http)
    raw, fields = (fw_view if dev == "fw" else web_view)(f, t, http, sk, s)
    out.update(request=first, raw=raw, fields=sorted(fields.items()))
    if dev == "fw":
        ev = ("none", "FW 로그에는 payload 가 없어 공격 근거를 볼 수 없음 (label 은 원본 샘플 기준)") \
            if out["label"] == "anomalies" else ("na", "")
    else:
        out["lost_headers"] = list(dict.fromkeys(k[:40] for k, _ in headers if k.lower() not in WEB_KEPT))
        if out["label"] != "anomalies":
            ev = ("na", "")
        elif ctx is None:
            ev = ("check", "룰 조건 확인 안 함")
        else:
            ev = web_evidence(http, out["rule_id"], ctx)
        if first.startswith("HTTP/"):
            ev = (ev[0], " · ".join(x for x in ("응답 메시지 샘플 (웹 접근 로그는 요청을 기록)", ev[1]) if x))
    out["evidence"] = {"state": ev[0], "text": ev[1]}
    return out


# ── CSV ────────────────────────────────────────────────────────────────

def csv_columns(dev):
    """CSV 머리와 파싱 필드 이름 목록 (필드 이름은 예시 줄을 바꿔 얻음 → convert 와 어긋나지 않음)."""
    names = [k for k, _ in convert(dev, SAMPLE_LINE, {})["fields"]]
    head = (["index", "date", "label (원본)", "원본 룰 id", f"{DEVICES[dev]} 로그"] + names
            + ["공격 근거", "공격 근거 사유"] + (["사라진 헤더"] if dev == "web" else []) + ["원본 payload"])
    return head, names


def csv_values(dev, row, index, date, names):
    fields = dict(row["fields"])
    return ([index, date, row["label"], row["rule_id"], row["raw"]] + [fields.get(n, "") for n in names]
            + [EV_TEXT.get(row["evidence"]["state"], ""), row["evidence"]["text"]]
            + ([", ".join(row["lost_headers"])] if dev == "web" else []) + [row["payload"]])


# ── 파일 저장 ───────────────────────────────────────────────────────────

def ips_files(date):
    """그 날짜의 IPS 학습 로그와 label 파일 (로그가 없으면 None)."""
    logs = sorted((DATASET_DIR / date).glob("ips_*_snort.log"))
    return (logs[0], logs[0].with_name(logs[0].stem + "_label.jsonl")) if logs else None


def out_files(date, dev):
    """저장 파일: 08_dataset/<날짜>/<fw|web>_<yymmdd>.log 와 <fw|web>_<yymmdd>_label.jsonl."""
    log = DATASET_DIR / date / f"{dev}_{date[2:].replace('-', '')}.log"
    return log, log.with_name(log.stem + "_label.jsonl")


def _lines(path):
    """파일의 줄 (화면의 줄 색인처럼 \\n 으로만 나누고 빈 줄은 뺌)."""
    return [x for x in path.read_bytes().split(b"\n") if x.strip()]


def _replace(tmp, dst, tries=10):
    """임시 파일 → 결과 파일. Windows 에서 다른 프로그램이 결과 파일을 잠깐 열고 있으면 조금 기다렸다 다시 (1.4 와 같음)."""
    for k in range(tries):
        try:
            tmp.replace(dst)
            return
        except PermissionError:
            if k == tries - 1:
                raise
            time.sleep(0.5 + 0.5 * k)


def save(date, dev, ctx=None, progress=None):
    """그 날짜 IPS 학습 로그 전체 → out_files 두 파일 (같은 줄 순서). → (로그 경로, 쓴 줄 수, 원본 줄 수).
    원본은 처음에 한 번에 읽고 닫는다 (오래 열어 두면 1.4 가 같은 날짜 로그를 다시 만들 때 바꾸기가 막힌다).
    ctx 는 convert 와 같음 (없으면 웹의 공격 근거는 'check'). progress(끝낸 줄, 전체 줄) 은 2천 줄마다."""
    src = ips_files(date)
    if src is None or not src[1].exists():
        raise ValueError(f"{date} 학습 로그가 없습니다")
    t0 = src[0].stat().st_mtime
    lines, metas = _lines(src[0]), _lines(src[1])
    if src[0].stat().st_mtime != t0 or len(lines) != len(metas):
        raise ValueError(f"{date} 학습 로그를 다시 만드는 중입니다 (로그 {len(lines):,}줄, label {len(metas):,}줄)"
                         " - 끝난 뒤에 저장하세요")
    log, lab = out_files(date, dev)
    tmps = [p.with_name(p.name + ".tmp") for p in (log, lab)]
    n = 0
    try:
        with open(tmps[0], "w", encoding="utf-8", newline="\n") as lf, \
                open(tmps[1], "w", encoding="utf-8", newline="\n") as tf:
            for k, (line, m) in enumerate(zip(lines, metas)):
                if progress and k % 2000 == 0:
                    progress(k, len(lines))
                meta = json.loads(m)
                row = convert(dev, line.decode("utf-8", errors="replace").rstrip("\r"), meta, ctx)
                if not row["raw"]:                      # 학습 로그 줄 형식이 아님 (1.4 로그에는 없음)
                    continue
                ev = row["evidence"]
                rec = {"line": n + 1, "id": row["rule_id"], "label": row["label"],
                       "attack_type": meta.get("attack_type", ""), "sk": meta.get("sk", ""), "evidence": ev["state"]}
                if dev == "web":
                    rec.update(evidence_why=ev["text"], lost_headers=row["lost_headers"])
                lf.write(row["raw"] + "\n")
                tf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        for tmp, dst in zip(tmps, (log, lab)):
            _replace(tmp, dst)
    finally:
        for tmp in tmps:
            tmp.unlink(missing_ok=True)
    if progress:
        progress(len(lines), len(lines))
    return log, n, len(lines)


# ── 명령줄 ─────────────────────────────────────────────────────────────

def load_check():
    """명령줄용: 룰 조건 대조기(3_0a_common)와 최신 룰 입력의 룰 (화면은 읽어 둔 것을 넘긴다)."""
    spec = importlib.util.spec_from_file_location("llm_common_14a", BASE_DIR / "3_0a_common.py")
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    in_dir = C.latest_input_dir()
    return C, ({r["id"]: r for r in C.read_jsonl(in_dir / "rules_all.jsonl")} if in_dir else {})


def main(argv=None):
    ap = argparse.ArgumentParser(description="IPS 신규 학습 데이터 → FW·웹 로그 형식 (확인, 파일로 저장)")
    ap.add_argument("date", help="08_dataset 날짜 폴더 (예: 2026-10-26, --save 는 all = 모든 날짜)")
    ap.add_argument("dev", choices=sorted(DEVICES))
    ap.add_argument("n", nargs="?", type=int, default=3, help="앞에서 몇 줄 보일지")
    ap.add_argument("--count", action="store_true", help="모든 줄의 공격 근거 상태 수 (웹은 룰 조건 대조)")
    ap.add_argument("--save", action="store_true",
                    help="날짜 전체를 08_dataset/<날짜>/<fw|web>_<yymmdd>.log + _label.jsonl 로 저장")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if a.save:
        dates = (sorted({p.parent.name for p in DATASET_DIR.glob("*/ips_*_snort.log")
                         if re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.parent.name)}) if a.date == "all" else [a.date])
        ctx = load_check() if a.dev == "web" else None
        for d in dates:
            t0 = time.time()
            try:
                log, n, total = save(d, a.dev, ctx)
            except ValueError as e:
                print(e)
                return 1
            print(f"{d}: {log.relative_to(BASE_DIR).as_posix()} + {log.stem}_label.jsonl {n:,}줄"
                  + (f" (형식이 아닌 {total - n:,}줄 뺌)" if total > n else "") + f" · {time.time() - t0:.0f}초")
        return 0
    logs = sorted((DATASET_DIR / a.date).glob("ips_*_snort.log"))
    if not logs:
        print(f"{a.date} IPS 학습 로그가 없습니다")
        return 1
    lab = logs[0].with_name(logs[0].stem + "_label.jsonl")
    ctx = load_check() if a.dev == "web" else None
    states, lost = Counter(), Counter()
    with open(logs[0], encoding="utf-8", errors="replace") as lf, open(lab, encoding="utf-8") as tf:
        rows = zip((x.rstrip("\r\n") for x in lf if x.strip()), (json.loads(x) for x in tf if x.strip()))
        for k, (line, meta) in enumerate(rows):
            if not a.count and k >= a.n:
                break
            row = convert(a.dev, line, meta, ctx)
            if a.count:
                states[(row["label"], row["evidence"]["state"])] += 1
                if row["evidence"]["state"] == "lost":
                    lost[meta.get("attack_type", "")] += 1
                continue
            print(row["raw"], "\n  공격 근거:", EV_TEXT.get(row["evidence"]["state"]), row["evidence"]["text"],
                  "\n  파싱 결과:", ", ".join(f"{n}={v}" for n, v in row["fields"]), "\n")
    if a.count:
        for (label, state), n in sorted(states.items()):
            print(f"{label:10} {EV_TEXT.get(state) or '-':8} {n:,}")
        if lost:
            print("근거 사라짐 - 공격 유형별:", ", ".join(f"{t} {n:,}" for t, n in lost.most_common()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
