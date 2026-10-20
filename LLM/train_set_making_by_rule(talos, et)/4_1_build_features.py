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

# 공격 흔적 키워드: 정상 요청에는 거의 나오지 않는 형태만 (디코딩한 payload 를 소문자로 보고 센다)
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
    text = decode(http).lower()
    f = {f"k_{k}": len(rx.findall(text)) for k, rx in KEYWORD_RE.items()}
    low = http.lower()
    f["k_double_enc"] = len(re.findall(r"%25(?:2e|2f|5c|3c|3e|27|22|00|3b|7c)", low))   # 기호를 이중 인코딩
    f["k_unicode_enc"] = len(re.findall(r"%u[0-9a-f]{4}", low))
    # 제어문자·NULL 바이트: 파일 업로드(multipart) 본문의 바이너리는 정상 파일 내용이므로 세지 않는다
    # (요청 줄·헤더·일반 본문에 들어간 NULL 바이트 등만 확실한 이상 신호)
    head, _, body = http.partition("\n\n")
    scope = head if "multipart/form-data" in head.lower() else http
    f["k_ctrl_bytes"] = (sum(1 for ch in scope if ord(ch) < 32 and ch not in "\n\r\t")
                         + len(re.findall(r"\\x(?:0[0-8bcef]|1[0-9a-f])|%00", scope.lower())))
    ua = next((l.split(":", 1)[1] for l in http.split("\n")[1:] if l.lower().startswith("user-agent:")), "")
    f["k_scanner_ua"] = int(bool(UA_SCANNER.search(ua)))
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
    if name.startswith("k_"):
        k = name[2:]
        return ("회피·이상 신호", EVASION[k]) if k in EVASION else ("공격 흔적", KEYWORDS[k][:120])
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
    args = ap.parse_args(argv)
    log = C.setup_logging("4_1_build_features")
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
            f = trace_features(http)
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
