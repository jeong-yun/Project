"""[1.3 LLM 입력] LLM 입력 생성기 — 통합 결과에서 IPS/WAF 학습데이터 생성에 필요한 필드만 뽑는다.

[전체 흐름]
  1. 04_integrated/ 의 가장 최근 날짜 폴더에서 http_rules.json, llm_queue.json 을 읽는다.
  2. 룰마다 학습데이터 생성에 필요한 정보만 남긴 간결한 레코드로 바꾼다.
       - 남기는 것: 식별(id, sid, rev), 룰 이름(rule_name), 라벨(attack_type, classtype, cve, mitre),
                   요청/응답 구분, HTTP method, 탐지 조건(content·정규식 → conditions), 원본 룰
       - 빼는 것  : 파일명, 내용 해시, IP/포트 변수, gid, 참조 URL, 내부 통계 등
  3. 탐지 조건이 하나도 없는 룰(content·pcre 없음)은 페이로드를 만들 수 없으므로 뺀다.
  4. 05_llm_input/<날짜>/ 에 전체본·변경분과, LLM 호출 단위로 나눈 배치 파일을 만든다.
  5. docs/코드참조/LLM_프롬프트.md 를 prompt.md 로 함께 복사한다. (LLM 에 줄 지시문)

[출력]
  05_llm_input/<날짜>/
     ├─ rules_all.jsonl       ← 통합 룰 전체 (처음 한 번 학습데이터를 만들 때)
     ├─ rules_delta.jsonl     ← 그날 신규·변경·선별 추가 룰 (매일 추가분)
     ├─ batches/all/batch_0001.jsonl ...    ← rules_all 을 N개씩 나눈 것 (LLM 1회 호출 = 파일 1개)
     ├─ batches/delta/batch_0001.jsonl ...  ← rules_delta 를 N개씩 나눈 것
     ├─ prompt.md             ← LLM 지시문 (docs/코드참조/LLM_프롬프트.md 복사본)
     └─ summary.json          ← 개수, 공격 유형·요청/응답·소스별 분포

[사용법]
    python 1_3_build_llm_input.py                   # 통합 결과가 바뀌었을 때만 생성
    python 1_3_build_llm_input.py --force           # 무조건 다시 생성
    python 1_3_build_llm_input.py --batch-size 50   # 배치 크기 변경 (기본 25)

레코드 형식 설명: docs/코드참조/LLM입력_형식.md
"""

import argparse
import collections
import datetime as dt
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INTEGRATED_DIR = BASE_DIR / "04_integrated"
OUTPUT_DIR = BASE_DIR / "05_llm_input"
PROMPT_FILE = BASE_DIR / "docs" / "코드참조" / "LLM_프롬프트.md"
LOG_DIR = BASE_DIR / "logs"

log = logging.getLogger("build_llm_input")

# 검사 위치(버퍼) 이름을 짧고 알기 쉽게
BUFFER_NAMES = {
    "http_uri": "uri", "http_raw_uri": "raw_uri", "http_header": "header", "http_raw_header": "raw_header",
    "http_cookie": "cookie", "http_raw_cookie": "raw_cookie", "http_client_body": "body",
    "http_method": "method", "http_stat_code": "status_code", "http_stat_msg": "status_msg",
    "file_data": "response_body", "base64_data": "base64_data", "raw": "payload",
}

# 공격 유형 추정: msg 를 위에서부터 비교해 처음 맞는 유형 (학습 라벨 참고용, 정답은 원본 룰)
ATTACK_TYPES = [
    ("sql_injection",     r"sql ?injection|sqli\b|sql insertion|union select|blind sql"),
    ("xss",               r"\bxss\b|cross[ -]site scripting"),
    ("xxe",               r"\bxxe\b|xml external entit"),
    ("ssrf",              r"\bssrf\b|server[ -]side request forgery"),
    ("deserialization",   r"deserializ|ysoserial|serialized object|object injection"),
    ("code_injection",    r"template injection|\bssti\b|ognl|el injection|expression language|code injection"),
    ("command_injection", r"command injection|command execution|os command|cmd injection|shell command"
                          r"|shellshock|cve-2014-6271"),
    ("file_inclusion",    r"file inclusion|file include|\brfi\b|\blfi\b|remote inclusion|code inclusion"
                          r"|remote file|local file"),
    ("path_traversal",    r"traversal|\.\./|directory climb"),
    ("file_upload",       r"upload"),
    ("webshell",          r"webshell|web shell"),
    ("auth_bypass",       r"auth(entication)? bypass|authorization bypass|unauthenticated|login bypass"
                          r"|hard-?coded (credential|password)|default (admin|credential|password|account)"),
    ("rce",               r"remote code execution|\brce\b|code execution|arbitrary code"),
    ("buffer_overflow",   r"overflow|out[- ]of[- ]bounds"),
    ("info_disclosure",   r"information (leak|disclosure)|info(rmation)? leak|disclosure|\bleak\b|file access"
                          r"|access attempt|/etc/passwd|config file|directory listing"),
    ("scanner",           r"\bscan\b|scanner|nikto|sqlmap|nmap|nessus|probe|recon"
                          r"|(?<![a-z])artex(?![a-z])|hacking[ _]agent"),   # AI 해킹 에이전트(ARTEX 등) = 자동화 공격 도구
                          # (msg 가 'AI_Agent_User-Agent' 처럼 '_' 로 이어져 \b 가 안 맞으므로 앞뒤 영문자만 확인)
    ("dos",               r"denial of service|\bdos\b|flood"),
    ("csrf",              r"\bcsrf\b|cross[ -]site request forgery"),
    ("open_redirect",     r"open redirect"),
]
_ATTACK_RES = [(name, re.compile(rx, re.I)) for name, rx in ATTACK_TYPES]


def setup_logging():
    LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(LOG_DIR / "build_llm_input.log", encoding="utf-8")
    fh.setFormatter(fmt)
    log.setLevel(logging.INFO)
    log.addHandler(fh)
    if sys.stdout is not None:  # pythonw(작업 스케줄러) 실행 시에는 콘솔이 없음
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        log.addHandler(sh)


def read_json_array_file(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path, records):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def attack_type(r):
    """msg(+ classtype)로 공격 유형을 추정한다. 못 찾으면 'other'."""
    text = r["msg"] or ""
    for name, rx in _ATTACK_RES:
        if rx.search(text):
            return name
    if r["category"] in ("SCAN",) or r["classtype"] == "attempted-recon":
        return "scanner"
    if r["classtype"] in ("attempted-dos", "denial-of-service"):
        return "dos"
    # 공격 키워드가 없는 ET INFO / POLICY 룰 = 정보성 (동적 DNS 접속, 의심 도메인, 앱 사용 등)
    if r["category"] in ("INFO", "POLICY"):
        return "informational"
    return "other"


def conditions(r):
    """content / pcre 를 LLM 이 읽기 쉬운 '탐지 조건' 목록으로 바꾼다.

    기본값(false, 없음)인 항목은 빼서 토큰을 줄인다.
      {"type": "content", "in": "uri", "value": "/gena.cgi?", "position": {"depth": 10}}
      {"type": "content", "in": "body", "value": "cmd=", "nocase": true}
      {"type": "content", "in": "header", "value": "curl", "negated": true}   ← 있으면 안 되는 값
      {"type": "regex", "in": "uri", "pattern": "service\\x3d...", "flags": "U"}  ← 정규식 (Snort pcre)
    """
    out = []
    for c in r["contents"]:
        item = {"type": "content", "in": BUFFER_NAMES.get(c["buffer"], c["buffer"]), "value": c["value"]}
        if c.get("nocase"):
            item["nocase"] = True
        if c.get("negated"):
            item["negated"] = True
        pos = {k: c[k] for k in ("offset", "depth", "distance", "within") if k in c}
        if pos:
            item["position"] = pos
        out.append(item)
    for p in r["pcres"]:
        item = {"type": "regex", "in": BUFFER_NAMES.get(p["buffer"], p["buffer"]),
                "pattern": p["pattern"], "flags": p["flags"]}
        if p.get("negated"):
            item["negated"] = True
        out.append(item)
    return out


def to_llm_record(r, change=None):
    """통합 룰 1개 → LLM 입력 레코드 1개."""
    h = r["http"]
    md = r.get("metadata", {})
    rec = {
        "id": r["key"],
        "source": r["source"],
        "sid": r["sid"],
        "rev": r["rev"],
        "rule_name": r["msg"],                                 # 룰 이름 (Snort msg)
        "category": r["category"],
        "attack_type": attack_type(r),
        "classtype": r["classtype"],
    }
    if r["cve"]:
        rec["cve"] = r["cve"]
    if md.get("mitre_technique_id"):
        rec["mitre"] = {"tactic": md.get("mitre_tactic_id", [None])[0],
                        "technique": md["mitre_technique_id"][0],
                        "technique_name": md.get("mitre_technique_name", [None])[0]}
    if md.get("signature_severity"):
        rec["severity"] = md["signature_severity"][0]
    rec["enabled"] = r["enabled"]
    rec["side"] = h["side"]                                   # request / response / unknown
    if h["methods"]:
        rec["methods"] = h["methods"]                          # 룰이 지정한 method → 이 중에서만 생성
    else:
        rec["methods"] = ["ANY"]                               # 룰이 method 를 제한하지 않음 → 어떤 method 든 생성 가능
        rec["method_hint"] = h["method_hint"]                  # 가장 자연스러운 method (body 검사 있으면 POST, 없으면 GET)
    if h["methods_excluded"]:
        rec["methods_excluded"] = h["methods_excluded"]
    rec["conditions"] = conditions(r)
    oo = r.get("other_options", {})
    if oo.get("urilen"):
        rec["urilen"] = oo["urilen"][0]                        # URI 길이 조건 (예: ">200")
    fb = [v for v in oo.get("flowbits", []) if v.startswith(("isset", "isnotset"))]
    if fb:
        rec["requires_state"] = fb                             # 앞선 다른 요청이 있어야 탐지됨
    if change:
        rec["change"] = change                                 # new / updated / selection_added
    rec["rule"] = r["raw"]                                     # 정답지 (원본 룰)
    return rec


def write_batches(folder, records, size):
    """records 를 size 개씩 나눠 batch_0001.jsonl ... 로 저장한다. 기존 배치 파일은 먼저 지운다."""
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("batch_*.jsonl"):
        old.unlink()
    n = 0
    for i in range(0, len(records), size):
        n += 1
        write_jsonl(folder / f"batch_{n:04d}.jsonl", records[i:i + size])
    return n


def publish_dir(tmp, out_dir, tries=30):
    """임시 폴더(다 쓴 것)를 완성본 이름으로 바꾼다 → 3.x 는 반쯤 쓴 입력을 볼 수 없다.
    같은 날짜 폴더를 3.x 가 읽는 중이라(파일이 열려 있어) 이름을 못 바꾸면 2초씩 기다렸다 다시 시도한다."""
    old = out_dir.with_name(f".{out_dir.name}.old{os.getpid()}")
    last = None
    for _ in range(tries):
        moved = False
        try:
            if out_dir.exists():
                out_dir.rename(old)
                moved = True
            tmp.rename(out_dir)
            shutil.rmtree(old, ignore_errors=True)
            return
        except OSError as e:
            last = e
            if moved and not out_dir.exists():          # 옛 폴더만 옮겨진 상태 → 되돌림
                try:
                    old.rename(out_dir)
                except OSError:
                    pass
            time.sleep(2)
    raise RuntimeError(f"{out_dir} 교체 실패 (다른 프로그램이 사용 중): {last}")


def clean_stale_tmp(hours=6):
    """비정상 종료로 남은 임시 폴더(.<날짜>.tmp… / .old…) 정리."""
    for d in OUTPUT_DIR.glob(".*"):
        if d.is_dir() and (".tmp" in d.name or ".old" in d.name) and time.time() - d.stat().st_mtime > hours * 3600:
            shutil.rmtree(d, ignore_errors=True)


def latest_integrated():
    dirs = sorted(d for d in INTEGRATED_DIR.glob("*") if d.is_dir() and (d / "http_rules.json").exists())
    return dirs[-1] if dirs else None


def main(argv=None):
    parser = argparse.ArgumentParser(description="통합 결과 → LLM 입력(학습데이터 생성용) 변환")
    parser.add_argument("--force", action="store_true", help="통합 결과가 그대로여도 다시 생성")
    parser.add_argument("--batch-size", type=int, default=25, help="배치 파일 1개에 넣을 룰 수 (기본 25 - LLM 1회 출력 약 2만 토큰 이내)")
    args = parser.parse_args(argv)
    setup_logging()

    src_dir = latest_integrated()
    if src_dir is None:
        log.info("04_integrated 결과 없음 - 건너뜀 (먼저 1_2_parse_rules.py 실행)")
        return 0
    integ = read_json_array_file(src_dir / "http_rules.json")
    out_dir = OUTPUT_DIR / src_dir.name

    # 통합 결과가 바뀌지 않았으면 건너뜀 (generated_at 과 batch 크기로 비교)
    prev = {}
    if (out_dir / "summary.json").exists():
        prev = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    if (not args.force and prev.get("source_generated_at") == integ["generated_at"]
            and prev.get("batch_size") == args.batch_size):
        log.info("통합 결과 변경 없음 (%s) - 건너뜀", src_dir.relative_to(BASE_DIR))
        return 0

    by_key = {r["key"]: r for r in integ["rules"]}
    queue = []
    if (src_dir / "llm_queue.json").exists():
        queue = read_json_array_file(src_dir / "llm_queue.json")["items"]

    # 전체본 / 변경분 변환 (탐지 조건이 없는 룰은 페이로드를 만들 수 없으므로 제외)
    all_recs, skipped = [], []
    for r in integ["rules"]:
        rec = to_llm_record(r)
        (all_recs if rec["conditions"] else skipped).append(rec)
    delta_recs = []
    for q in queue:
        r = by_key.get(q["key"])
        if r is None:          # 중복 제거 등으로 통합 결과에 없는 룰
            continue
        rec = to_llm_record(r, q.get("change"))
        if rec["conditions"]:
            delta_recs.append(rec)

    # 임시 폴더에 다 쓴 뒤 한 번에 바꾼다 (수집 에이전트와 학습데이터 생성이 동시에 돌아도 3.x 가 반쯤 쓴 입력을 읽지 않게)
    clean_stale_tmp()
    final_dir = out_dir
    out_dir = OUTPUT_DIR / f".{final_dir.name}.tmp{os.getpid()}"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    write_jsonl(out_dir / "rules_all.jsonl", all_recs)
    write_jsonl(out_dir / "rules_delta.jsonl", delta_recs)
    n_all = write_batches(out_dir / "batches" / "all", all_recs, args.batch_size)
    n_delta = write_batches(out_dir / "batches" / "delta", delta_recs, args.batch_size)
    if PROMPT_FILE.exists():
        shutil.copyfile(PROMPT_FILE, out_dir / "prompt.md")
    else:
        log.warning("프롬프트 파일 없음: %s", PROMPT_FILE.relative_to(BASE_DIR))

    def dist(recs, key):
        return dict(collections.Counter(r[key] for r in recs).most_common())

    summary = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(src_dir.relative_to(BASE_DIR)),
        "source_generated_at": integ["generated_at"],
        "batch_size": args.batch_size,
        "rules_all": len(all_recs), "rules_delta": len(delta_recs),
        "batches_all": n_all, "batches_delta": n_delta,
        "skipped_no_condition": [r["id"] for r in skipped],
        "all_by_source": dist(all_recs, "source"),
        "all_by_side": dist(all_recs, "side"),
        "all_by_attack_type": dist(all_recs, "attack_type"),
        "all_method_any": sum(1 for r in all_recs if r["methods"] == ["ANY"]),
        "all_with_regex": sum(1 for r in all_recs if any(c["type"] == "regex" for c in r["conditions"])),
        "all_requires_state": sum(1 for r in all_recs if r.get("requires_state")),
        "delta_by_change": dist(delta_recs, "change") if delta_recs else {},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    publish_dir(out_dir, final_dir)          # summary.json 까지 다 쓴 뒤 완성본 이름으로
    log.info("[LLM 입력] %s: 전체 %d개(배치 %d), 변경분 %d개(배치 %d), 조건 없음 제외 %d개",
             final_dir.relative_to(BASE_DIR), len(all_recs), n_all, len(delta_recs), n_delta, len(skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
