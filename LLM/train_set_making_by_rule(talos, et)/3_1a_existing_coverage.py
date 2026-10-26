"""[3.1a 기존 데이터 확인] 기존 학습 로그(06_기존 데이터)에 룰마다 탐지되는 샘플이 몇 건 있는지 센다.  (3_0 이 실행, 3.1 이 결과를 씀)

[왜]  기존 학습 데이터가 이미 충분한 룰(기본 1000건 이상)은 LLM 생성에서 빼고,
      부족한 룰(기본 50건 이하)은 기존 샘플을 예시로 주되 똑같지 않은 샘플을 만들게 하려고.

[어떻게]  기존 로그에는 어떤 룰에서 나온 샘플인지 기록이 없다 (method 는 장비 탐지 이름).
          그래서 기존 anomalies payload 를 룰 조건에 직접 대조한다 (코드 검사 local_check 와 같은 방식).
  1. 06_기존 데이터/*.log (구분자 '|||' 8필드) 에서 anomalies payload 를 모은다 (IPS·WAF 합쳐 중복 제거).
  2. 룰마다 '대표 문자열' = 가장 긴 content 값(4자 이상, method 제외) 을 정하고, 그 안에서 가장 드문 4글자로 색인한다.
  3. payload 에 그 4글자가 있는 룰만 후보로 골라, content 값이 모두 있는지 먼저 보고, 있으면 전체 조건(정규식 포함)을 검사한다.
  4. 조건을 모두 만족하면 그 룰의 기존 샘플로 센다. 예시 payload 몇 개도 저장한다.
  - 셀 수 없는 룰 (0건으로 취급 → 일반 생성). summary 에 수와 이유를 남긴다.
      · content 없이 정규식만 있는 룰
      · 코드 검사가 평가하지 못하는 옵션(byte_test, isdataat, urilen, detection_filter 등)이 있는 룰
      · 앞선 요청이 필요한 룰(flowbits isset)
      · 목적지 포트가 기존 데이터(80·8080)를 포함하지 않는 룰 (예: 443 전용)
  - 위치(depth 등)는 보지 않는 근사 대조라 실제보다 조금 많게 셀 수 있다.

[출력]  07_llm_output/<날짜>/_0_existing/
     ├─ coverage.json        ← {룰 id: {count, ips, waf, methods, examples}} (1건 이상인 룰만)
     ├─ payload_hashes.txt   ← 기존 로그의 모든 payload 지문 (3.4 가 '기존과 똑같은 샘플' 을 빼는 데 사용)
     └─ summary.json         ← 구간별 룰 수 (1000↑ / 51~999 / 1~50 / 0 / 셀 수 없음)
  06 로그와 05 룰이 바뀌지 않았으면 다시 계산하지 않는다 (--force 로 강제).

[사용법]
    python 3_1a_existing_coverage.py
    python 3_1a_existing_coverage.py --force
"""

import argparse
import collections
import datetime as dt
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from urllib.parse import unquote_plus

_spec = importlib.util.spec_from_file_location("llm_common", Path(__file__).resolve().parent / "3_0a_common.py")
C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)

EXISTING_DIR = C.BASE_DIR / "06_기존 데이터"
OUT_SUB = "_0_existing"
SEP = "|||"
MAX_EXAMPLES = 5


norm_payload, payload_hash = C.norm_payload, C.payload_hash   # 3.4 와 같은 규칙으로 지문을 만든다


_UNCHECKED_RE, dst_port_ok = C.UNCHECKED_RE, C.dst_port_ok   # 공용 (3_0a_common.py)


def countable(rule, ports=(80, 8080)):
    """기존 데이터와 대조해 셀 수 있는 룰인지 → (가능 여부, 이유)."""
    raw = rule.get("rule", "")
    m = _UNCHECKED_RE.search(raw)
    if m:
        return False, f"평가 못 하는 옵션({m.group(1)})"
    if rule.get("requires_state"):
        return False, "앞선 요청 필요(flowbits)"
    if not dst_port_ok(raw, ports):
        return False, "목적지 포트가 기존 데이터(80·8080)와 다름"
    return True, ""


def anchor(rule):
    """룰의 대표 문자열 (소문자). 없으면 None."""
    vals = [C.unescape_value(c["value"]) for c in rule["conditions"]
            if c["type"] == "content" and not c.get("negated") and c.get("in") != "method"]
    vals = [v for v in vals if len(v) >= 4]
    return max(vals, key=len).lower() if vals else None


def inputs_fingerprint(rules_file, logs):
    """입력 지문: 기존 로그는 크기·수정 시각, 룰은 '룰 id + 원본 룰' 내용 (attack_type 같은 부가 필드가 바뀌어도 다시 계산하지 않음)."""
    h = hashlib.sha1()
    with open(rules_file, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            h.update((r["id"] + "\0" + r.get("rule", "") + "\n").encode("utf-8"))
    parts = [h.hexdigest()] + [f"{p.name}:{p.stat().st_size}:{int(p.stat().st_mtime)}" for p in logs]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def main(argv=None):
    ap = argparse.ArgumentParser(description="기존 학습 로그의 룰별 샘플 수 계산")
    ap.add_argument("--date", help="05_llm_input 날짜 폴더 (기본: 가장 최근)")
    ap.add_argument("--force", action="store_true", help="입력이 그대로여도 다시 계산")
    args = ap.parse_args(argv)
    log = C.setup_logging("3_1a_existing_coverage")
    if not C.generation_guard(log, "3_1a_existing_coverage"):     # 3_0 또는 다른 생성이 돌고 있으면 (같은 파일을 쓰므로) 멈춤
        return 2

    in_dir = C.latest_input_dir(args.date)
    logs = sorted(p for p in EXISTING_DIR.glob("*.log")) if EXISTING_DIR.is_dir() else []
    if in_dir is None:
        log.error("05_llm_input 이 없습니다")
        return 1
    out = C.output_dir(in_dir.name) / OUT_SUB
    if not logs:
        log.info("[3.1a] 06_기존 데이터 에 .log 가 없음 - 기존 데이터 확인 생략")
        return 0
    rules_file = in_dir / "rules_all.jsonl"
    fp = inputs_fingerprint(rules_file, logs)
    if (out / "summary.json").exists() and not args.force:
        if json.loads((out / "summary.json").read_text(encoding="utf-8")).get("inputs") == fp:
            log.info("[3.1a] 입력 변경 없음 - 기존 계산 결과 사용 (%s)", out)
            return 0
    started = time.time()

    # 1) 기존 payload 모으기
    attack = {}                       # payload → {"ips": n, "waf": n, "methods": Counter}
    hashes = set()
    for f in logs:
        dev = "waf" if f.name.lower().startswith("waf") else "ips"
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                p = line.rstrip("\n").split(SEP)
                if len(p) != 8:
                    continue
                hashes.add(payload_hash(p[6]))
                if p[7] != "anomalies":
                    continue
                a = attack.setdefault(p[6], {"ips": 0, "waf": 0, "methods": collections.Counter()})
                a[dev] += 1
                a["methods"][p[5]] += 1
    log.info("[3.1a] 기존 로그 %d개: payload 지문 %d개, anomalies 고유 payload %d개", len(logs), len(hashes), len(attack))

    # 2) 룰 색인: 대표 문자열 안에서 '가장 드문 4글자' 로 색인 → payload 하나당 후보 룰 수를 줄인다
    rules = C.read_jsonl(rules_file)
    gram_freq = collections.Counter()
    skip_reason = collections.Counter()
    for r in rules:
        ok, why = countable(r)
        if not ok:
            r["_anchor"] = None
            r["_uncountable"] = why
            skip_reason[why.split("(")[0]] += 1
            continue
        a = anchor(r)
        r["_anchor"] = a
        if a:
            gram_freq.update({a[k:k + 4] for k in range(len(a) - 3)})
            # 빠른 사전 확인용: 부정이 아닌 content 값(소문자)
            r["_need"] = [C.unescape_value(c["value"]).lower() for c in r["conditions"]
                          if c["type"] == "content" and not c.get("negated") and c.get("in") != "method"]
    by_gram, no_anchor = collections.defaultdict(list), 0
    for i, r in enumerate(rules):
        a = r["_anchor"]
        if a is None:
            no_anchor += 1
            continue
        g = min((a[k:k + 4] for k in range(len(a) - 3)), key=lambda x: gram_freq[x])
        by_gram[g].append(i)

    # 3) payload 마다 후보 룰 → 빠른 확인(content 값이 모두 있는지) → 전체 조건 검사
    counts = collections.defaultdict(lambda: {"count": 0, "ips": 0, "waf": 0,
                                              "methods": collections.Counter(), "examples": []})
    for n, (payload, meta) in enumerate(attack.items(), 1):
        text = norm_payload(payload)
        low = text.lower()
        try:
            low2 = unquote_plus(low)
        except Exception:
            low2 = low
        grams = {low[k:k + 4] for k in range(len(low) - 3)}
        if low2 != low:
            grams |= {low2[k:k + 4] for k in range(len(low2) - 3)}
        cands = set()
        for g in grams & by_gram.keys():
            cands.update(by_gram[g])
        parts = None
        for i in cands:
            r = rules[i]
            if not all(v in low or v in low2 for v in r["_need"]):
                continue
            if parts is None:
                parts = C.split_http(text)
            ok = True
            for c in r["conditions"]:
                if C.check_condition(c, parts) is False:
                    ok = False
                    break
            if not ok:
                continue
            c = counts[r["id"]]
            c["count"] += 1
            c["ips"] += meta["ips"]
            c["waf"] += meta["waf"]
            c["methods"].update(meta["methods"])
            if len(c["examples"]) < MAX_EXAMPLES and len(text) <= 1500:
                c["examples"].append(text)
        if n % 50000 == 0:
            log.info("  ... %d / %d payload 대조 (%.0f초)", n, len(attack), time.time() - started)
    if C.REGEX_TIMEOUTS:         # 역추적이 폭발하는 정규식: 시간 제한을 넘은 조건은 판단 불가로 셈 (3_0a_common.regex_search)
        rid = {c["pattern"]: r["id"] for r in rules for c in r["conditions"] if c["type"] == "regex"}
        log.warning("  정규식 검사 시간 초과 %d번 - 그 조건은 판단 불가로 셈: %s", sum(C.REGEX_TIMEOUTS.values()),
                    ", ".join(f"{rid.get(p, '?')} {k}번" for p, k in C.REGEX_TIMEOUTS.most_common(5)))

    # 4) 저장
    out.mkdir(parents=True, exist_ok=True)
    cov = {rid: {"count": c["count"], "ips": c["ips"], "waf": c["waf"],
                 "methods": dict(c["methods"].most_common(5)), "examples": c["examples"]}
           for rid, c in counts.items()}
    (out / "coverage.json").write_text(json.dumps(cov, ensure_ascii=False), encoding="utf-8")
    (out / "payload_hashes.txt").write_text("\n".join(sorted(hashes)) + "\n", encoding="utf-8")
    _, _, run = C.load_settings()
    hi, lo = C.run_int(run, "기존 데이터 충분 기준", 1000), C.run_int(run, "기존 데이터 참고 기준", 50)
    band = collections.Counter()
    for r in rules:
        n_ = cov.get(r["id"], {}).get("count", 0)
        band["셀 수 없음(조건 평가 불가·포트 다름·대표 문자열 없음)" if not r.get("_anchor") else
             f"{hi}건 이상 (생성 제외)" if n_ >= hi else f"{lo + 1}~{hi - 1}건 (일반 생성)" if n_ > lo else
             f"1~{lo}건 (기존 예시 참고 생성)" if n_ >= 1 else "0건 (일반 생성)"] += 1
    summary = {"generated_at": dt.datetime.now().isoformat(timespec="seconds"), "inputs": fp,
               "rules_file": str(rules_file.relative_to(C.BASE_DIR)), "logs": [p.name for p in logs],
               "existing_payloads": len(hashes), "existing_attack_payloads": len(attack),
               "rules": len(rules), "rules_without_anchor": no_anchor, "rules_with_existing": len(cov),
               "uncountable_reasons": dict(skip_reason),
               "bands": dict(band), "seconds": round(time.time() - started, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("[3.1a] 룰 %d개: %s (%.0f초)", len(rules), dict(band), time.time() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
