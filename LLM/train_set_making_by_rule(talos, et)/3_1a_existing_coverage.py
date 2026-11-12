"""[3.1a 기존 데이터 확인] 기존 학습 로그(06_기존 데이터)에 룰마다 탐지되는 샘플이 몇 건 있는지 센다.  (3_0 이 실행, 3.1 이 결과를 씀)

[왜]  기존 학습 데이터가 이미 충분한 룰(기본 1000건 이상)은 LLM 생성에서 빼고,
      부족한 룰(기본 50건 이하)은 기존 샘플을 예시로 주되 똑같지 않은 샘플을 만들게 하려고.

[어떻게]  기존 로그에는 어떤 룰에서 나온 샘플인지 기록이 없다 (method 는 장비 탐지 이름).
          그래서 기존 anomalies payload 를 룰 조건에 직접 대조한다 (코드 검사 local_check 와 같은 방식).
  1. 06_기존 데이터/*.log (구분자 '|||' 8필드) 에서 anomalies payload 를 모은다 (IPS·WAF 합쳐 중복 제거).
  2. 룰마다 '대표 문자열' = 가장 긴 content 값(4자 이상, method 제외) 을 정하고, 대표 문자열이 있는 payload 만 후보로 본다.
     다시 셀 룰이 많으면(INDEX_MIN 이상) 대표 문자열 안의 '가장 드문 4글자' 로 색인해 후보를 고르고,
     적으면 룰마다 대표 문자열을 바로 찾는다 (어느 쪽이든 결과는 같다).
  3. 후보에 content 값이 모두 있는지 먼저 보고, 있으면 전체 조건(정규식 포함)을 검사한다.
  4. 조건을 모두 만족하면 그 룰의 기존 샘플로 센다. 예시 payload 몇 개도 저장한다.
  - 셀 수 없는 룰 (0건으로 취급 → 일반 생성). summary 에 수와 이유를 남긴다.
      · content 없이 정규식만 있는 룰
      · 코드 검사가 평가하지 못하는 옵션(byte_test, isdataat, urilen, detection_filter 등)이 있는 룰
      · 앞선 요청이 필요한 룰(flowbits isset)
      · 목적지 포트가 기존 데이터(80·8080)를 포함하지 않는 룰 (예: 443 전용)
  - 위치(depth 등)는 보지 않는 근사 대조라 실제보다 조금 많게 셀 수 있다.

[바뀐 룰만 다시 셈 (2026-11-09)]  예전에는 룰이 하나만 바뀌어도 모든 룰(약 2만)을 다시 대조했다 (8~35분).
  지금은 룰별 결과(0건 포함)와 06 에서 모은 payload 를 저장해 두고, 새 룰·바뀐 룰만 대조한다.
  - 룰 지문 = 룰 id + 원본 룰 + 해석된 조건(conditions·requires_state). 지문이 같으면 저장된 결과를 쓴다.
  - 06 로그(이름·크기·수정 시각)나 대조 코드(이 파일·3_0a_common.py)가 바뀌면 저장본을 모두 버리고 다시 센다.
  - 정규식 검사 시간 초과(1초, 판단 불가 → 맞음으로 셈)가 난 룰은 저장하지 않고 다음 실행에서 다시 센다.
    제한이 실제 시간이라 CPU 가 바쁘면 초과가 늘어 결과가 달라지기 때문 (11-09 시험: CPU 100% 에서 2룰).
  - 지문에 안 잡히는 변화(regex 모듈·파이썬 버전 등)가 있으면 --force 로 전부 다시 센다.
  - 저장본을 못 쓰거나(디스크 부족 등) 지우면 다음 실행이 다시 셀 뿐 결과는 같다.

[출력]  07_llm_output/<날짜>/_0_existing/
     ├─ coverage.json        ← {룰 id: {count, ips, waf, methods, examples}} (1건 이상인 룰만)
     ├─ payload_hashes.txt   ← 기존 로그의 모든 payload 지문 (3.4 가 '기존과 똑같은 샘플' 을 빼는 데 사용)
     └─ summary.json         ← 구간별 룰 수 (1000↑ / 51~999 / 1~50 / 0 / 셀 수 없음), 다시 센 룰 수·저장된 결과를 쓴 룰 수
  07_llm_output/_0_existing_cache/   (날짜 폴더와 따로 두는 저장본)
     ├─ payloads.pkl         ← 06 에서 모은 payload 지문·anomalies payload (약 100MB, 06 을 다시 읽지 않게)
     └─ rules.json           ← {룰 지문: coverage.json 과 같은 칸} (0건 포함, 마지막에 센 05 룰만)
  06 로그·05 룰·대조 코드가 모두 그대로면 아무것도 하지 않는다.

[사용법]
    python 3_1a_existing_coverage.py
    python 3_1a_existing_coverage.py --force        # 저장본을 쓰지 않고 06 읽기부터 전부 다시 계산
"""

import argparse
import collections
import datetime as dt
import hashlib
import importlib.util
import json
import os
import pickle
import sys
import time
from pathlib import Path
from urllib.parse import unquote_plus

_spec = importlib.util.spec_from_file_location("llm_common", Path(__file__).resolve().parent / "3_0a_common.py")
C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)

EXISTING_DIR = C.BASE_DIR / "06_기존 데이터"
OUT_SUB = "_0_existing"
CACHE_DIR = C.LLM_OUTPUT_DIR / "_0_existing_cache"
CODE_FILES = (Path(__file__).resolve(), Path(_spec.origin))     # 대조 코드: 내용이 바뀌면 저장본을 모두 다시 센다
SEP = "|||"
MAX_EXAMPLES = 5
INDEX_MIN = 300          # 다시 셀 룰이 이만큼 이상이면 4글자 색인(준비에 약 30초), 적으면 룰마다 대표 문자열을 바로 찾는다


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


def rule_key(r):
    """룰 지문: id + 원본 룰 + 해석된 조건 (1.2 해석이 바뀌어 조건만 달라져도 다시 센다). attack_type 같은 부가 필드는 안 본다."""
    cond = json.dumps([r.get("conditions"), r.get("requires_state")], ensure_ascii=False, sort_keys=True)
    return hashlib.sha1("\0".join((r["id"], r.get("rule", ""), cond)).encode("utf-8")).hexdigest()[:16]


def cache_stamp(logs):
    """저장본 지문 = 06 로그(이름·크기·수정 시각) + 대조 코드 내용. 다르면 저장본을 쓰지 않는다."""
    code = hashlib.sha1(b"".join(p.read_bytes() for p in CODE_FILES)).hexdigest()[:12]
    logs_fp = hashlib.sha1("|".join(f"{p.name}:{p.stat().st_size}:{int(p.stat().st_mtime)}" for p in logs).encode())
    return f"06 {logs_fp.hexdigest()[:12]} 코드 {code}"


def load_cache(name, stamp):
    """저장본 → 내용 (없거나 깨졌거나 지문이 다르면 None → 다시 센다)."""
    try:
        with open(CACHE_DIR / name, "rb") as f:
            d = pickle.load(f) if name.endswith(".pkl") else json.load(f)
    except Exception:
        return None
    return d if isinstance(d, dict) and d.get("stamp") == stamp else None


def save_cache(name, data, log):
    """저장본 쓰기 (임시 파일에 쓴 뒤 바꿔 끼움 → 중간에 멈춰도 반쯤 쓴 파일이 남지 않음).
    못 쓰면(디스크 부족 등) 경고만 남기고 결과(coverage.json 등)는 그대로 쓴다 - 다음 실행이 다시 셀 뿐."""
    tmp = CACHE_DIR / f"{name}.{os.getpid()}.tmp"
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as f:
            if name.endswith(".pkl"):       # payload 약 100MB: pickle 이 JSON 보다 읽기가 훨씬 빠름 (약 2초 / 8초)
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                f.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))
        tmp.replace(CACHE_DIR / name)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        log.warning("  저장본 %s 을 쓰지 못함 (%s) - 다음 실행에서 다시 센다", name, e)


def read_logs(logs):
    """06 로그 → (모든 payload 지문 정렬 목록, anomalies 고유 payload [(정리된 payload, ips 수, waf 수, {탐지 이름: 수})] 로그 순서)."""
    attack, hashes = {}, set()
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
    return sorted(hashes), [(norm_payload(k), a["ips"], a["waf"], dict(a["methods"])) for k, a in attack.items()]


def count_rules(rules, attack, log, started):
    """기존 anomalies payload 를 룰 조건에 대조 → ({룰 id: coverage.json 한 칸} (0건 룰 포함), 정규식 시간 초과가 난 룰 지문)."""
    counts = {r["id"]: {"count": 0, "ips": 0, "waf": 0, "methods": collections.Counter(), "examples": []} for r in rules}
    timed = set()
    for r in rules:             # 빠른 사전 확인용: 부정이 아닌 content 값(소문자). 대표 문자열도 이 안에 있다
        r["_need"] = [C.unescape_value(c["value"]).lower() for c in r["conditions"]
                      if c["type"] == "content" and not c.get("negated") and c.get("in") != "method"]
    anchors, by_gram = [r["_anchor"] for r in rules], None
    if len(rules) >= INDEX_MIN:     # 대표 문자열 안에서 '가장 드문 4글자' 로 색인 → payload 하나당 후보 룰 수를 줄인다
        gram_freq = collections.Counter()
        for a in anchors:
            gram_freq.update({a[k:k + 4] for k in range(len(a) - 3)})
        by_gram = collections.defaultdict(list)
        for i, a in enumerate(anchors):
            by_gram[min((a[k:k + 4] for k in range(len(a) - 3)), key=lambda x: gram_freq[x])].append(i)

    for n, (text, ips, waf, methods) in enumerate(attack, 1):
        low = text.lower()
        try:
            low2 = unquote_plus(low)
        except Exception:
            low2 = low
        if by_gram is None:         # 룰이 적으면: 대표 문자열이 그대로 있는 룰만 후보
            cands = [i for i, a in enumerate(anchors) if a in low or a in low2]
        else:                       # 대표 문자열이 있으면 그 4글자도 있으므로 같은 룰이 후보가 된다
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
            before = sum(C.REGEX_TIMEOUTS.values())
            failed = any(C.check_condition(c, parts) is False for c in r["conditions"])
            if sum(C.REGEX_TIMEOUTS.values()) != before:      # 1초 제한은 실제 시간이라 부하에 따라 결과가 달라진다
                timed.add(r["_key"])
            if failed:
                continue
            c = counts[r["id"]]
            c["count"] += 1
            c["ips"] += ips
            c["waf"] += waf
            c["methods"].update(methods)
            if len(c["examples"]) < MAX_EXAMPLES and len(text) <= 1500:
                c["examples"].append(text)
        if n % 50000 == 0:
            log.info("  ... %d / %d payload 대조 (%.0f초)", n, len(attack), time.time() - started)
    return {rid: {**c, "methods": dict(c["methods"].most_common(5))} for rid, c in counts.items()}, timed


def main(argv=None):
    ap = argparse.ArgumentParser(description="기존 학습 로그의 룰별 샘플 수 계산")
    ap.add_argument("--date", help="05_llm_input 날짜 폴더 (기본: 가장 최근)")
    ap.add_argument("--force", action="store_true", help="저장본을 쓰지 않고 06 읽기부터 전부 다시 계산")
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
    rules = C.read_jsonl(rules_file)
    stamp = cache_stamp(logs)
    for r in rules:
        r["_key"] = rule_key(r)
    fp = hashlib.sha1("|".join([stamp] + [r["_key"] for r in rules]).encode("utf-8")).hexdigest()[:12]
    if (out / "summary.json").exists() and not args.force:
        if json.loads((out / "summary.json").read_text(encoding="utf-8")).get("inputs") == fp:
            log.info("[3.1a] 입력 변경 없음 - 기존 계산 결과 사용 (%s)", out)
            return 0
    started = time.time()

    # 1) 기존 payload: 06·대조 코드가 그대로면 저장본, 아니면 06 로그에서 모아 저장
    saved = None if args.force else load_cache("payloads.pkl", stamp)
    if saved:
        hashes, attack = saved["hashes"], saved["attack"]
        log.info("[3.1a] 06 저장본 사용: payload 지문 %d개, anomalies 고유 payload %d개", len(hashes), len(attack))
    else:
        hashes, attack = read_logs(logs)
        save_cache("payloads.pkl", {"stamp": stamp, "hashes": hashes, "attack": attack}, log)
        log.info("[3.1a] 기존 로그 %d개: payload 지문 %d개, anomalies 고유 payload %d개", len(logs), len(hashes), len(attack))

    # 2) 룰 나누기: 셀 수 없는 룰 / 저장된 결과를 쓰는 룰 / 다시 셀 룰 (새 룰·바뀐 룰, 저장본을 못 쓰면 전부)
    done = ((None if args.force else load_cache("rules.json", stamp)) or {}).get("rules", {})
    skip_reason, todo = collections.Counter(), []
    for r in rules:
        ok, why = countable(r)
        if not ok:
            skip_reason[why.split("(")[0]] += 1
        r["_anchor"] = anchor(r) if ok else None
        if r["_anchor"] and r["_key"] not in done:
            todo.append(r)
    matchable = [r for r in rules if r["_anchor"]]
    log.info("[3.1a] 대조할 룰 %d개: 저장된 결과 %d개 · 다시 셀 룰 %d개%s", len(matchable), len(matchable) - len(todo),
             len(todo), " (--force)" if args.force else " (저장본 없음 또는 06 로그·대조 코드가 바뀜)" if not done else "")

    # 3) 다시 셀 룰만 대조
    timed = set()
    if todo:
        new, timed = count_rules(todo, attack, log, started)
        done.update((r["_key"], new[r["id"]]) for r in todo)
    if C.REGEX_TIMEOUTS:         # 역추적이 폭발하는 정규식: 시간 제한을 넘은 조건은 판단 불가로 셈 (3_0a_common.regex_search)
        rid = {c["pattern"]: r["id"] for r in todo for c in r["conditions"] if c["type"] == "regex"}
        log.warning("  정규식 검사 시간 초과 %d번 - 그 조건은 판단 불가로 셈, 그 룰 %d개는 저장하지 않고 다음 실행에서 다시 셈: %s",
                    sum(C.REGEX_TIMEOUTS.values()), len(timed),
                    ", ".join(f"{rid.get(p, '?')} {k}번" for p, k in C.REGEX_TIMEOUTS.most_common(5)))

    # 4) 저장: 룰별 결과는 지금 05 룰만 남겨 저장본으로 (시간 초과가 난 룰은 빼고), 1건 이상인 룰은 coverage.json 으로
    res = {r["_key"]: done[r["_key"]] for r in matchable}
    save_cache("rules.json", {"stamp": stamp, "rules": {k: v for k, v in res.items() if k not in timed}}, log)
    out.mkdir(parents=True, exist_ok=True)
    cov = {r["id"]: res[r["_key"]] for r in matchable if res[r["_key"]]["count"]}
    (out / "coverage.json").write_text(json.dumps(cov, ensure_ascii=False), encoding="utf-8")
    (out / "payload_hashes.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")
    _, _, run = C.load_settings()
    hi, lo = C.run_int(run, "기존 데이터 충분 기준", 1000), C.run_int(run, "기존 데이터 참고 기준", 50)
    band = collections.Counter()
    for r in rules:
        n_ = cov.get(r["id"], {}).get("count", 0)
        band["셀 수 없음(조건 평가 불가·포트 다름·대표 문자열 없음)" if not r["_anchor"] else
             f"{hi}건 이상 (생성 제외)" if n_ >= hi else f"{lo + 1}~{hi - 1}건 (일반 생성)" if n_ > lo else
             f"1~{lo}건 (기존 예시 참고 생성)" if n_ >= 1 else "0건 (일반 생성)"] += 1
    summary = {"generated_at": dt.datetime.now().isoformat(timespec="seconds"), "inputs": fp,
               "rules_file": str(rules_file.relative_to(C.BASE_DIR)), "logs": [p.name for p in logs],
               "existing_payloads": len(hashes), "existing_attack_payloads": len(attack),
               "rules": len(rules), "rules_without_anchor": len(rules) - len(matchable), "rules_with_existing": len(cov),
               "rules_recounted": len(todo), "rules_cached": len(matchable) - len(todo),
               "uncountable_reasons": dict(skip_reason),
               "bands": dict(band), "seconds": round(time.time() - started, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("[3.1a] 룰 %d개: %s - 다시 센 룰 %d개 · 저장된 결과 %d개 (%.0f초)", len(rules), dict(band), len(todo),
             len(matchable) - len(todo), time.time() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
