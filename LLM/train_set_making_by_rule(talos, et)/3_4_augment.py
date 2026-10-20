"""[3.4 데이터 증강] 검토를 통과한 샘플을 코드로 변형해 룰마다 목표 수(기본 anomalies 50, normal 10)까지 늘린다.  (3_0 이 실행)

[왜 코드로 늘리나]
  LLM 에 룰 하나당 50개를 직접 만들게 하면 비용이 10배 이상 들고, 뒤로 갈수록 거의 같은 샘플이 반복된다.
  그래서 LLM 은 공격 부분이 서로 다른 '씨앗' 샘플(기본 anomalies 10, normal 5)만 만들고,
  이 단계에서 룰 조건과 상관없는 부분만 바꿔 수를 채운다. LLM 호출 없음 (비용 0).

[바꾸는 것]  룰 조건에 걸린 헤더·위치는 건드리지 않는다
  - Host 헤더        : 기존 학습 로그의 목적지 서버 4개 중 하나 (룰이 Host 를 검사하지 않을 때만)
  - User-Agent       : 브라우저·스크립트·도구 UA 목록 (룰이 User-Agent 를 검사하지 않을 때만)
  - 부가 헤더 추가    : Accept, Accept-Language, Accept-Encoding, Connection, Cache-Control, Referer, Cookie
  - URI 끝 파라미터   : 공격과 무관한 파라미터를 끝에 덧붙임 (예: &page=3)
  - 폼 본문 끝 파라미터: application/x-www-form-urlencoded 본문 끝에 덧붙이고 Content-Length 를 맞춤
  변형마다 1~3개를 무작위로 적용하고, 코드 검사(local_check)를 다시 통과한 것만 남긴다 (룰 기준 라벨).
    - anomalies 변형: 룰 조건을 모두 만족해야 함 / normal 변형: 룰 조건을 하나 이상 어겨야 함
    - 룰로 판단할 수 없는 변형(정규식 해석 불가)은 '판정 불가 처리'가 drop 이면 뺀다
  normal 이 목표보다 적으면: 공격 씨앗의 User-Agent 를 일반 사용자 브라우저 UA 로 바꾼 것 중
    룰이 탐지하지 않는 것(예: 스캐너·악성 UA 룰)을 normal 로 쓴다.

[기존 데이터 차감]  '기존 데이터 차감'이 예이면 anomalies 목표 = 목표 수 - 기존 학습 로그(06)의 이 룰 샘플 수.
  예: 기존 10건이면 40개만 만든다. 기존 50건 이상이면 증강하지 않고 LLM 씨앗만 남긴다. (1000건 이상은 3.1 에서 이미 제외)
  똑같은 HTTP 는 한 번만 남긴다. 같은 입력이면 같은 결과가 나온다 (룰 id 로 난수 고정).
  기존 학습 로그(06)와 payload 가 똑같은 샘플은 씨앗이든 증강이든 뺀다 (3.1a 의 payload_hashes.txt).

[한계]  변형은 공격 부분이 아닌 '맥락'(헤더·부가 파라미터)의 다양성이다. 공격 기법의 다양성은 LLM 씨앗 수가 결정한다.
        조건이 매우 좁은 룰은 목표 수에 못 미칠 수 있다 (summary 에 룰별 달성 수 기록).

[출력]  07_llm_output/<날짜>/<이름>.jsonl  ← 1_4_build_dataset.py 의 입력 (맨 위 폴더)
        07_llm_output/<날짜>/_4_augmented/<이름>.meta.json  ← 룰별 씨앗 수·증강 수·목표 미달 룰

[사용법]
    python 3_4_augment.py            # 검토됐지만 증강 안 된 것 전부
    python 3_4_augment.py --force    # 목표 수를 바꾼 뒤 다시
"""

import argparse
import collections
import datetime as dt
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("llm_common", Path(__file__).resolve().parent / "3_0a_common.py")
C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)

AUG_DIR = "_4_augmented"
HOSTS = ["192.168.10.30", "192.168.10.31", "192.168.20.20", "192.168.0.20"]
# UA 는 3_0a_common 의 실제 형식 생성기에서 (Chrome·Edge·Whale·Firefox·Safari·삼성 인터넷, 실제 버전 형식)
BROWSER_UAS = sorted({C.browser_ua(random.Random(k)) for k in range(80)})   # 일반 사용자 UA (normal·공격 씨앗 → normal)
USER_AGENTS = BROWSER_UAS[::3] + C.TOOL_UAS                                  # 공격 증강용: 브라우저 + 스크립트·도구
EXTRA_HEADERS = {
    "Accept": ["*/*", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "application/json"],
    "Accept-Language": ["ko-KR,ko;q=0.9,en-US;q=0.8", "en-US,en;q=0.5", "ko"],
    "Accept-Encoding": ["gzip, deflate", "gzip, deflate, br", "identity"],
    "Connection": ["close", "keep-alive"],
    "Cache-Control": ["no-cache", "max-age=0"],
    "Referer": ["http://{host}/", "http://{host}/index.php", "http://{host}/login"],
    "Cookie": ["PHPSESSID={rand}", "JSESSIONID={RAND}", "session={rand}; lang=ko"],
}
EXTRA_PARAMS = ["page", "lang", "t", "ref", "v", "sort", "cache", "_"]


# ── 룰이 검사하는 헤더 ──────────────────────────────────────────────────

guarded_headers = C.guarded_headers   # 룰 조건이 건드리는 헤더 (3_0a_common.py)


# ── HTTP 다루기 ────────────────────────────────────────────────────────

def parse(http):
    s = http.replace("\r\n", "\n")
    head, sep, body = s.partition("\n\n")
    lines = head.split("\n")
    return lines[0], lines[1:], body


def build(first, headers, body):
    names = [h.split(":", 1)[0].strip().lower() for h in headers]
    if body and "content-length" in names:
        headers = [f"Content-Length: {len(body.rstrip(chr(10)).encode('utf-8'))}" if h.lower().startswith("content-length:") else h
                   for h in headers]
    return first + "\n" + "".join(h + "\n" for h in headers) + "\n" + body


def set_header(headers, name, value):
    low = name.lower()
    for i, h in enumerate(headers):
        if h.split(":", 1)[0].strip().lower() == low:
            out = list(headers)
            out[i] = f"{name}: {value}"
            return out
    return headers + [f"{name}: {value}"]


def has_header(headers, name):
    return any(h.split(":", 1)[0].strip().lower() == name.lower() for h in headers)


# ── 변형 ────────────────────────────────────────────────────────────────

def mutate(http, guard, rng, uas=USER_AGENTS):
    """변형 1~3개를 무작위로 적용한 새 HTTP. 적용할 수 있는 변형이 없으면 None."""
    first, headers, body = parse(http)
    ops = []
    if "host" not in guard and has_header(headers, "Host"):
        ops.append("host")
    if "user-agent" not in guard:
        ops.append("ua")
    free = [h for h in EXTRA_HEADERS if h.lower() not in guard and not has_header(headers, h)]
    if free:
        ops.append("header")
    parts = first.split(" ")
    if len(parts) == 3 and not first.startswith("HTTP/"):
        ops.append("query")
    ctype = next((h.split(":", 1)[1].strip().lower() for h in headers if h.lower().startswith("content-type:")), "")
    if body and "x-www-form-urlencoded" in ctype:
        ops.append("body")
    if not ops:
        return None
    host = next((h.split(":", 1)[1].strip() for h in headers if h.lower().startswith("host:")), "192.168.20.20")
    for op in rng.sample(ops, rng.randint(1, min(3, len(ops)))):
        if op == "host":
            host = rng.choice(HOSTS)
            headers = set_header(headers, "Host", host)
        elif op == "ua":
            headers = set_header(headers, "User-Agent", rng.choice(uas))
        elif op == "header":
            for name in rng.sample(free, rng.randint(1, min(2, len(free)))):
                v = rng.choice(EXTRA_HEADERS[name]).format(
                    host=host, rand="%016x" % rng.getrandbits(64), RAND="%032X" % rng.getrandbits(128))
                headers = headers + [f"{name}: {v}"]
        elif op == "query":
            m, target, ver = first.split(" ")
            sep = "&" if "?" in target else "?"
            first = f"{m} {target}{sep}{rng.choice(EXTRA_PARAMS)}={rng.randint(1, 9999)} {ver}"
        elif op == "body":
            body = body.rstrip("\n") + f"&{rng.choice(EXTRA_PARAMS)}={rng.randint(1, 9999)}"
    return build(first, headers, body)


def normal_from_attacks(rule, out, seen, existing, need, rng, strict, cross=None):
    """공격 씨앗의 User-Agent 를 일반 브라우저 UA 로 바꿔, 룰이 탐지하지 않게 된 것을 normal 로 추가. 추가한 수.
    cross 가 있으면 다른 공격 룰에도 탐지되지 않는 것만 (예: 공격 경로·파라미터가 남아 다른 룰에 걸리면 안 됨)."""
    attacks = [s for s in out if s["label"] == "anomalies" and s.get("origin") != "augmented"]
    added = 0
    for seed in attacks:
        for ua in rng.sample(BROWSER_UAS, len(BROWSER_UAS)):
            if added >= need:
                return added
            first, headers, body = parse(seed["http"])
            if not has_header(headers, "User-Agent"):
                break
            new = build(first, set_header(headers, "User-Agent", ua), body)
            key = new.strip()
            if key in seen or C.payload_hash(key) in existing:
                continue
            chk = C.local_check({"label": "normal", "http": new}, rule)
            if chk["result"] != "ok" and (strict or chk["result"] == "fail"):
                break       # UA 를 바꿔도 탐지됨 (UA 가 조건이 아닌 룰) → 이 씨앗은 안 됨
            if cross and cross(new):
                break       # 다른 공격 룰에 탐지됨 → 보안 관점에서 정상이 아님
            seen.add(key)
            out.append({"label": "normal", "attack_type": "none", "http": new,
                        "note": f"증강: 공격 씨앗 #{attacks.index(seed)} 의 User-Agent 를 일반 브라우저로 바꿈 (룰 미탐지 확인)",
                        "origin": "augmented", "local_check": chk["result"]})
            added += 1
    return added


def augment_rule(rule, rec, target_a, target_n, existing=frozenset(), strict=True, cross=None, realistic=False):
    """씨앗 샘플(rec) → 목표 수까지 증강한 샘플 목록, 통계.
    existing = 기존 학습 로그 payload 지문 → 기존 데이터와 똑같은 샘플은 씨앗이든 증강이든 뺀다.
    cross(http) = 이 룰 말고 다른 공격 룰이 탐지하면 그 룰 id 목록 → normal 은 씨앗이든 증강이든 뺀다.
    realistic = 씨앗에도 헤더 현실화를 적용 (헤더 현실화 이전에 생성된 결과도 실제 UA·헤더로)."""
    rng = random.Random(C.stable_seed(rule["id"]))
    guard = guarded_headers(rule)
    out, seen, same_as_existing = [], set(), 0
    cross_dropped, cross_n = collections.Counter(), 0      # 다른 룰 id → normal 씨앗을 뺀 수
    for s in rec["samples"]:
        if realistic:
            new = C.realistic_sample(s, rule)
            if new["http"] != s["http"]:
                s = dict(new, local_check=C.local_check(new, rule)["result"])
        key = s["http"].replace("\r\n", "\n").strip()
        if C.payload_hash(key) in existing:
            same_as_existing += 1
            continue
        hits = cross(s["http"]) if cross and s["label"] == "normal" else []
        if hits:
            cross_dropped.update(hits)
            cross_n += 1
            continue
        if key not in seen:
            seen.add(key)
            out.append(s)
    stats = {}
    for label, target in (("anomalies", target_a), ("normal", target_n)):
        seeds = [s for s in out if s["label"] == label]
        have = len(seeds)
        tries = 0
        while seeds and have < target and tries < target * 30:
            tries += 1
            seed = seeds[tries % len(seeds)]
            # normal 은 일반 사용자 브라우저 UA 만 (curl·python 같은 도구 UA 는 정책 룰에 탐지됨)
            new = mutate(seed["http"], guard, rng, BROWSER_UAS if label == "normal" else USER_AGENTS)
            if new is None:
                continue
            key = new.strip()
            if key in seen or C.payload_hash(key) in existing:
                continue
            cand = {"label": label, "http": new}
            chk = C.local_check(cand, rule)
            # anomalies: 조건을 모두 만족(ok·unknown) / normal: 여전히 탐지되지 않음(ok)
            if chk["result"] == "fail" or (strict and chk["result"] == "unknown"):
                continue
            if label == "normal" and cross and cross(new):
                continue
            seen.add(key)
            out.append({"label": label, "attack_type": seed["attack_type"], "http": new,
                        "note": f"증강: 씨앗 #{seeds.index(seed)} 의 헤더·부가 파라미터 변형",
                        "origin": "augmented", "local_check": chk["result"]})
            have += 1
        if label == "normal" and have < target:
            have += normal_from_attacks(rule, out, seen, existing, target - have, rng, strict, cross)
        stats[label] = {"seeds": len(seeds), "total": have, "target": target}
    stats["same_as_existing_dropped"] = same_as_existing
    stats["normal_other_rule_dropped"] = {"samples": cross_n, "rules": dict(cross_dropped.most_common(5))}
    return out, stats


def make_cross_check(in_dir, run, log):
    """normal 교차 검사 함수 (http, 이 룰 id) → 이 normal 을 탐지하는 '다른 공격 룰' id 목록. 설정이 '아니오'면 None.

    '그 룰에 안 걸린다' 만으로는 정상이 아니다 (예: 공격 문자열을 다른 파라미터로 옮긴 룰 우회 공격).
    그래서 선별된 전체 HTTP 룰(05_llm_input/rules_all.jsonl)과 대조해, 다른 공격 룰이 탐지하면 normal 에서 뺀다.
    - 정보성 룰(attack_type informational, 예: Host 가 IP 인 요청)과 이름이 '... access' 인 접근 기록 룰은 공격 판정이 아니므로 무시
    - 조건을 모두 확인한 탐지(match)만 센다. byte_test 등 확인 못 한 부분이 있는 룰(possible)은 무시
    - 룰이 없는 공격도 있으므로, 4.1 의 '공격 흔적' 피처(k_: ;uname 같은 명령 연결, UNION SELECT, <script, ../../ 등)가
      하나라도 켜지면 함께 뺀다 (결과에 '흔적:k_cmd' 처럼 표시)
    """
    if run.get("normal 교차 검사", "예").strip() not in ("예", "yes", "y", "true", "1"):
        return None
    skip_types = {t.strip() for t in run.get("교차 검사 제외 attack_type", "informational").split(",") if t.strip()}
    skip_name = re.compile(run.get("교차 검사 제외 룰 이름", r"\saccess$").strip() or r"(?!)", re.I)
    rules = [r for r in C.read_jsonl(in_dir / "rules_all.jsonl")
             if r.get("attack_type") not in skip_types and not skip_name.search(r.get("rule_name", ""))]
    matcher = C.RuleMatcher(rules)
    log.info("[3.4] normal 교차 검사: 다른 공격 룰 %d개와 대조 (제외: %s, 이름 '%s')",
             len(matcher.rules), ",".join(sorted(skip_types)), skip_name.pattern)

    spec = importlib.util.spec_from_file_location("features", Path(__file__).resolve().parent / "4_1_build_features.py")
    F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)

    def cross(http, rid):
        hits = [h for h, kind in matcher.match(http, exclude={rid}) if kind == "match"]
        return hits + [f"흔적:{k}" for k, v in F.trace_features(C.norm_payload(http)).items() if v]
    return cross


def main(argv=None):
    ap = argparse.ArgumentParser(description="검토된 샘플을 코드로 증강 (LLM 호출 없음)")
    ap.add_argument("--date", help="날짜 폴더 (기본: 05_llm_input 의 가장 최근)")
    ap.add_argument("--force", action="store_true", help="이미 증강한 것도 다시 (목표 수를 바꿨을 때)")
    args = ap.parse_args(argv)
    log = C.setup_logging("3_4_augment")
    if not C.generation_guard(log, "3_4_augment"):     # 3_0 또는 다른 생성이 돌고 있으면 (같은 파일을 쓰므로) 멈춤
        return 2
    try:
        _, _, run = C.load_settings()
    except (ValueError, FileNotFoundError) as e:
        log.error("LLM 설정 읽기 실패: %s", e)
        return 1
    in_dir = C.latest_input_dir(args.date)
    if in_dir is None:
        log.error("05_llm_input 이 없습니다")
        return 1
    latest_base = C.output_dir(in_dir.name)
    # 날짜를 지정하지 않으면 모든 날짜 폴더의 '검토됐지만 증강 안 된' 결과를 처리한다 (--force 는 지정·최신 날짜만)
    bases = [latest_base] if (args.date or args.force) else C.output_dates()
    target_a = C.run_int(run, "목표 anomalies 수", 50)
    target_n = C.run_int(run, "목표 normal 수", 10)
    deduct = run.get("기존 데이터 차감", "예").strip() in ("예", "yes", "y", "true", "1")
    strict = run.get("판정 불가 처리", "drop").strip() == "drop"
    total, todo = 0, []
    for b in bases:
        rev_dir = b / C.REV_DIR
        revs = sorted(p for p in rev_dir.glob("*.jsonl") if not p.name.endswith(".rejected.jsonl")) if rev_dir.is_dir() else []
        total += len(revs)
        todo += [(b, r) for r in revs if args.force or not (b / r.name).exists()]
    log.info("[3.4] 증강 목표 룰당 anomalies %d / normal %d | 검토 결과 %d개 중 이번 %d개%s",
             target_a, target_n, total, len(todo),
             f" (날짜 폴더: {', '.join(sorted({b.name for b, _ in todo}))})" if todo else "")
    ctx = {}

    def context(b):
        """날짜 폴더의 기존 데이터 결과 (3.1a: 룰별 기존 샘플 수, payload 지문). 없으면 결과가 있는 가장 최근 폴더 것."""
        if b not in ctx:
            src = next((x for x in [b, latest_base] + C.output_dates()[::-1]
                        if (x / "_0_existing" / "coverage.json").exists()), None)
            cov_file = src / "_0_existing" / "coverage.json" if src else None
            cov = json.loads(cov_file.read_text(encoding="utf-8")) if deduct and cov_file else {}
            ctx[b] = (cov, C.load_existing_hashes(src) if src else set())
        return ctx[b]
    if todo:
        log.info("[3.4] 기존 학습 로그 payload 지문 %d개와 겹치는 샘플은 뺀다", len(context(latest_base)[1]))
    cross = make_cross_check(in_dir, run, log) if todo else None
    for base, r in todo:
        req_dir, aug_dir = base / C.REQ_DIR, base / AUG_DIR
        aug_dir.mkdir(parents=True, exist_ok=True)
        cov, existing = context(base)
        rules = {x["id"]: x for x in C.read_jsonl(req_dir / r.name)}
        final, per_rule, short = [], {}, []
        for rec in C.read_jsonl(r):
            rule = rules.get(rec["id"])
            if rule is None:
                continue
            have_before = cov.get(rec["id"], {}).get("count", 0)     # 기존 학습 로그의 이 룰 샘플 수
            rule_target = max(0, target_a - have_before)
            samples, st = augment_rule(rule, rec, rule_target, target_n, existing, strict,
                                       (lambda h, rid=rec["id"]: cross(h, rid)) if cross else None,
                                       C.run_yes(run, "헤더 현실화"))
            st["existing"] = have_before
            final.append(dict(rec, samples=samples))
            per_rule[rec["id"]] = st
            if st["anomalies"]["total"] < rule_target:
                short.append({"id": rec["id"], "anomalies": st["anomalies"]["total"]})
        C.write_jsonl(base / r.name, final)        # 1_4 입력 (맨 위 폴더)
        flat = [s for rec in final for s in rec["samples"]]
        meta = {"request": r.stem, "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
                "target": {"anomalies": target_a, "normal": target_n},
                "samples": len(flat), "anomalies": sum(s["label"] == "anomalies" for s in flat),
                "normal": sum(s["label"] == "normal" for s in flat),
                "llm": sum(s.get("origin") == "llm" for s in flat),
                "augmented": sum(s.get("origin") == "augmented" for s in flat),
                "same_as_existing_dropped": sum(v.get("same_as_existing_dropped", 0) for v in per_rule.values()),
                "normal_other_rule_dropped": sum(v["normal_other_rule_dropped"]["samples"] for v in per_rule.values()),
                "rules_below_target": short, "per_rule": per_rule}
        (aug_dir / f"{r.stem}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("  %s: 샘플 %d개 (anomalies %d, normal %d / LLM %d + 증강 %d), 목표 미달 룰 %d개",
                 r.stem, meta["samples"], meta["anomalies"], meta["normal"], meta["llm"], meta["augmented"], len(short))
    return 0


if __name__ == "__main__":
    sys.exit(main())
