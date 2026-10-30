"""[3.5 진행 현황] 현재 룰 중 학습데이터를 몇 % 만들었는지, 룰마다 어디까지 했는지(상태 플래그)를 계산한다.
(3_0 이 끝날 때 실행, `3_0_run_generation.py --status`, UI 가 사용. LLM 호출 없음, 비용 0)

[기준]  '현재 룰' = 가장 최근 05_llm_input/<날짜>/rules_all.jsonl (선별·중복 제거된 HTTP 룰)
        생성 결과는 07_llm_output 의 모든 날짜 폴더에서 찾는다. 룰 내용(원본 룰 텍스트)이 지금과 같은 결과만 인정한다.

[룰별 상태 플래그]  rule_status.csv · .jsonl 의 flag 열
  DONE       완료        생성 → 검토 → 증강까지 끝나 학습 로그(08_dataset)에 들어감
  REVIEWED   증강 대기   검토까지 끝남 (다음 3_0 실행 때 증강·로그 변환)
  GENERATED  검토 대기   생성까지 끝남 (다음 3_0 실행 때 검토)
  QUEUED     요청 대기   요청 파일에 들어 있고 아직 생성 전
  TODO       미생성      생성 대상인데 아직 요청도 없음 (3_0 을 실행하면 요청이 만들어짐)
  CHANGED    룰 변경     예전 내용으로 만들었는데 룰이 바뀜 → 다시 생성 대상
  MISSING    응답 빠짐   생성 응답에 이 룰이 없었음 → 다음 실행 때 새 요청으로 다시
  EMPTY      샘플 없음   생성했지만 검토·코드 검사에서 샘플이 모두 빠짐 → 확인 필요
  FAILED     생성 실패   응답 해석 실패가 반복돼 자동으로 건너뛰는 중 (3_2 --force 로 다시)
  EXISTING   기존 충분   기존 학습 로그에 이미 충분(기본 1000건 이상) → 생성하지 않음
  EXCLUDED   제외        정보성(informational)·HTTP 응답 방향 룰, LLM_설정.md '생성 제외 룰' → 생성 대상 아님 (reason 열에 사유)

[비율]  생성 대상 = 전체 - EXCLUDED - EXISTING
        완료율 = DONE / 생성 대상,   학습데이터 보유율 = (DONE + EXISTING) / 전체 룰

[거부 편향 경보] (2026-10-29, 위험 등록부 R01) 검토 모델이 '안전' 을 이유로 공격 샘플을 빼거나 판정을 거절하면 위험한 공격
  유형일수록 학습데이터가 모자란다. 공격 유형별로 거부 비율 = (안전 사유로 뺀 공격 샘플 + 판정 거절로 빠진 공격 샘플) ÷ 생성된
  공격 샘플 을 지금 결과(다시 만들기 반영) 전체와 가장 최근 날짜 폴더로 따로 세고, '거부 편향 경보 기준'(%, 기본 5)을 넘는 유형을
  경고한다 (생성 공격 샘플이 '거부 편향 최소 샘플'(기본 100) 이상인 유형만). progress.json 의 review_bias, 3.5 로그의 [경고] 줄.
  3.3 의 meta.json review_by_type 을 쓰고, 그 칸이 없는 예전 결과는 파일에서 한 번 세어 _progress/review_types.json 에 기억한다.

[출력]  07_llm_output/_progress/
     ├─ progress.json     ← 요약 (비율, 상태별 수, 출처·공격 유형별, 어디까지 했는지, 토큰·예상 남은 비용·시간, 거부 편향)
     ├─ rule_status.csv   ← 룰별 상태 (엑셀용, 한글 열 이름)
     └─ rule_status.jsonl ← 같은 내용 (UI·코드용)

[사용법]
    python 3_5_progress.py
    python 3_0_run_generation.py --status      # 같은 것
"""

import collections
import csv
import datetime as dt
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

_spec = importlib.util.spec_from_file_location("llm_common", Path(__file__).resolve().parent / "3_0a_common.py")
C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)
_rspec = importlib.util.spec_from_file_location("review_33", Path(__file__).resolve().parent / "3_3_review.py")
R = importlib.util.module_from_spec(_rspec); _rspec.loader.exec_module(R)        # type_counts·SAFETY_RE (검토와 같은 기준)

PROGRESS_DIR = C.LLM_OUTPUT_DIR / "_progress"
AUG_DIR = "_4_augmented"
FAIL_LIMIT = 2          # 3_2_generate.py 와 같은 값
FLAGS = {   # 플래그: (한글, 설명)
    "DONE": ("완료", "학습 로그(08)에 들어감"),
    "REVIEWED": ("증강 대기", "검토까지 끝남"),
    "GENERATED": ("검토 대기", "생성까지 끝남"),
    "QUEUED": ("요청 대기", "요청 파일에 있음, 생성 전"),
    "TODO": ("미생성", "아직 요청도 없음"),
    "CHANGED": ("룰 변경", "예전 내용으로 만듦 → 다시 생성"),
    "MISSING": ("응답 빠짐", "다음 실행 때 다시 요청"),
    "EMPTY": ("샘플 없음", "검토·코드 검사에서 모두 빠짐"),
    "FAILED": ("생성 실패", "응답 해석 실패 반복 → 건너뛰는 중"),
    "EXISTING": ("기존 충분", "기존 학습 로그에 충분"),
    "EXCLUDED": ("제외", "정보성·HTTP 응답 룰·생성 제외 룰"),
}
RANK = {"DONE": 7, "REVIEWED": 6, "GENERATED": 5, "QUEUED": 4, "FAILED": 3, "EMPTY": 2, "MISSING": 1}
REQ_RE = re.compile(r"^(?P<set>all|delta|pick)_b(?P<batch>\d+)_")


def _json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _listdir(d):
    """폴더 안 이름 → DirEntry (한 번에 읽음. Windows 는 파일을 하나씩 여는 것보다 훨씬 빠름)."""
    try:
        return {e.name: e for e in os.scandir(d)}
    except OSError:
        return {}


def review_types(rmeta, rev_dir, gen_dir, name, ent_meta, ent_rej, cache, new_cache, key):
    """요청 하나의 공격 유형별 [생성 공격, 검토 drop, 그중 안전 사유, 판정 거절]. meta.json 에 없으면 파일에서 세고 기억한다."""
    if isinstance(rmeta.get("review_by_type"), dict):
        return rmeta["review_by_type"]
    sig = [ent_meta.stat().st_mtime_ns if ent_meta else 0, ent_rej.stat().st_mtime_ns if ent_rej else 0]
    hit = cache.get(key)
    if hit and hit.get("m") == sig:
        new_cache[key] = hit
        return hit["t"]
    gen = C.read_jsonl(gen_dir / f"{name}.jsonl") if (gen_dir / f"{name}.jsonl").exists() else []
    rej = C.read_jsonl(rev_dir / f"{name}.rejected.jsonl") if ent_rej else []
    t = R.type_counts(gen, rej)
    new_cache[key] = {"m": sig, "t": t}
    return t


def bias_summary(by_type, threshold, min_n):
    """{유형: [생성 공격, 검토 drop, 안전 사유, 판정 거절]} → 유형별 비율과 경보 대상 (거부 비율 높은 순)."""
    out, alerts = {}, []
    for t, (gen, drop, safety, refused) in sorted(by_type.items()):
        pct = round(100.0 * (safety + refused) / gen, 2) if gen else 0.0
        out[t] = {"generated_attacks": gen, "dropped_by_reviewer": drop, "safety_dropped": safety, "refused": refused,
                  "refusal_pct": pct, "drop_pct": round(100.0 * (drop + refused) / gen, 2) if gen else 0.0}
        if gen >= min_n and pct > threshold:
            alerts.append(t)
    return out, sorted(alerts, key=lambda t: -out[t]["refusal_pct"])


def scan_outputs(latest_base):
    """07_llm_output 의 모든 날짜 폴더 → {룰 id: [(상태, 룰 내용 지문, 날짜, 요청, anomalies, normal)]}, 요청 통계.
    요청 파일의 (룰 id, 내용 지문)은 수정 시각 기준으로 _progress/req_index.json 에 기억해 두고 바뀐 것만 다시 읽는다."""
    entries = collections.defaultdict(list)
    req_stat = {"total": 0, "generated": 0, "reviewed": 0, "augmented": 0, "failed": 0, "pending": 0}
    usage = {"generator": collections.Counter(), "reviewer": collections.Counter()}
    seconds = {"generate": [], "review": []}
    batches = collections.defaultdict(lambda: [0, 0])      # 최신 날짜 all 세트: 배치 번호 → [요청 수, 생성 끝난 요청 수]
    last_done = None
    cache, new_cache = _json(PROGRESS_DIR / "req_index.json", {}) or {}, {}
    tcache, new_tcache = _json(PROGRESS_DIR / "review_types.json", {}) or {}, {}
    bias = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0, 0]))   # 날짜 → 유형 → 수
    for base in C.output_dates():
        req_dir, gen_dir, rev_dir, aug_dir = base / C.REQ_DIR, base / C.GEN_DIR, base / C.REV_DIR, base / AUG_DIR
        req_ls, gen_ls, rev_ls, aug_ls = _listdir(req_dir), _listdir(gen_dir), _listdir(rev_dir), _listdir(aug_dir)
        latest = base == latest_base
        # 지난 날짜 폴더는 생성된 요청만 본다 (생성 전 요청은 최신 날짜 요청으로 대체됨)
        names = [n[:-6] for n in (req_ls if latest else gen_ls) if n.endswith(".jsonl") and not n.startswith("_")]
        for name in sorted(names):
            ent = req_ls.get(f"{name}.jsonl")
            if ent is None:
                continue
            key, mtime = f"{base.name}/{name}", ent.stat().st_mtime_ns
            hit = cache.get(key)
            if hit and hit.get("m") == mtime:
                rules = hit["r"]
            else:
                rules = [[r["id"], C.rule_hash(r)] for r in C.read_jsonl(Path(ent.path))]
            new_cache[key] = {"m": mtime, "r": rules}
            gmeta = _json(gen_dir / f"{name}.meta.json") if f"{name}.meta.json" in gen_ls else None
            rmeta = _json(rev_dir / f"{name}.meta.json") if f"{name}.meta.json" in rev_ls else None
            ameta = _json(aug_dir / f"{name}.meta.json") if f"{name}.meta.json" in aug_ls else None
            generated = f"{name}.jsonl" in gen_ls
            fails = ((_json(gen_dir / f"{name}.fail.json", {}) or {}).get("fails", 0)
                     if f"{name}.fail.json" in gen_ls else 0)
            missing = set((gmeta or {}).get("rules_missing", []))
            per_rule = (ameta or {}).get("per_rule", {})
            if latest:
                req_stat["total"] += 1
                req_stat["generated"] += generated
                req_stat["reviewed"] += rmeta is not None
                req_stat["augmented"] += ameta is not None
                req_stat["failed"] += (not generated and fails >= FAIL_LIMIT)
                req_stat["pending"] += (not generated and fails < FAIL_LIMIT)
                m = REQ_RE.match(name)
                if m and not generated and fails < FAIL_LIMIT:          # 세트별 남은 요청 (화면의 예상 계산용)
                    req_stat.setdefault("pending_by_set", {}).setdefault(m.group("set"), 0)
                    req_stat["pending_by_set"][m.group("set")] += 1
                if m and m.group("set") == "all":
                    b = batches[int(m.group("batch"))]
                    b[0] += 1
                    b[1] += generated or fails >= FAIL_LIMIT
            for role, meta in (("generator", gmeta), ("reviewer", rmeta)):
                u = ((meta or {}).get("usage") or {}).get(role) or {}
                usage[role].update({k: v for k, v in u.items() if isinstance(v, (int, float))})
            if gmeta and gmeta.get("seconds") is not None:
                seconds["generate"].append(gmeta["seconds"])
            if rmeta and rmeta.get("seconds") is not None:
                seconds["review"].append(rmeta["seconds"])
            if rmeta is not None:          # 거부 편향 (R01): 공격 유형별 검토 결과
                tc = review_types(rmeta, rev_dir, gen_dir, name, rev_ls.get(f"{name}.meta.json"),
                                  rev_ls.get(f"{name}.rejected.jsonl"), tcache, new_tcache, key)
                for t, v in tc.items():
                    acc = bias[base.name][t]
                    for k in range(4):
                        acc[k] += v[k]
            if ameta and (last_done is None or ameta.get("finished_at", "") > last_done["at"]):
                last_done = {"request": name, "date": base.name, "at": ameta.get("finished_at", "")}
            for rid, h in rules:
                if ameta is not None:
                    st = per_rule.get(rid)
                    total_a = (st or {}).get("anomalies", {}).get("total", 0)
                    total_n = (st or {}).get("normal", {}).get("total", 0)
                    flag = "MISSING" if rid in missing else ("DONE" if total_a + total_n > 0 else "EMPTY")
                    entries[rid].append((flag, h, base.name, name, total_a, total_n))
                    continue
                if rmeta is not None:
                    flag = "MISSING" if rid in missing else "REVIEWED"
                elif generated:
                    flag = "MISSING" if rid in missing else "GENERATED"
                else:
                    flag = "FAILED" if fails >= FAIL_LIMIT else "QUEUED"
                entries[rid].append((flag, h, base.name, name, 0, 0))
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = PROGRESS_DIR / f"req_index.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(new_cache, separators=(",", ":")), encoding="utf-8")
    tmp.replace(PROGRESS_DIR / "req_index.json")
    ttmp = PROGRESS_DIR / f"review_types.json.{os.getpid()}.tmp"
    ttmp.write_text(json.dumps(new_tcache, separators=(",", ":")), encoding="utf-8")
    ttmp.replace(PROGRESS_DIR / "review_types.json")
    # '어디까지': 앞에서부터 빠짐없이 생성이 끝난 배치 수, 가장 뒤의 생성 배치
    contiguous = 0
    for b in sorted(batches):
        if batches[b][1] < batches[b][0]:
            break
        contiguous = b
    touched = max((b for b, (n, g) in batches.items() if g), default=0)
    frontier = {"batches_total": max(batches, default=0), "contiguous_done_batch": contiguous,
                "last_touched_batch": touched}
    return entries, req_stat, usage, seconds, frontier, last_done, bias


def review_bias(bias, run):
    """날짜별 유형별 수 → progress.json 의 review_bias (전체 + 가장 최근 날짜 폴더) 와 경보 목록."""
    try:
        threshold = float(run.get("거부 편향 경보 기준", 5))
    except ValueError:
        threshold = 5.0
    min_n = C.run_int(run, "거부 편향 최소 샘플", 100)
    total = collections.defaultdict(lambda: [0, 0, 0, 0])
    for d in bias.values():
        for t, v in d.items():
            for k in range(4):
                total[t][k] += v[k]
    latest = max(bias, default="")
    all_types, all_alerts = bias_summary(total, threshold, min_n)
    last_types, last_alerts = bias_summary(bias.get(latest, {}), threshold, min_n)
    return {"threshold_pct": threshold, "min_samples": min_n,
            "definition": "(안전 사유로 뺀 공격 샘플 + 판정 거절로 빠진 공격 샘플) / 생성된 공격 샘플, 지금 결과(다시 만들기 반영)",
            "all": {"by_type": all_types, "alerts": all_alerts},
            "latest_date": latest, "latest": {"by_type": last_types, "alerts": last_alerts}}


def compute(log=None):
    started = time.time()
    in_dir = C.latest_input_dir()
    if in_dir is None:
        raise RuntimeError("05_llm_input 에 완성된 입력이 없습니다 (먼저 1_0_run_daily.py)")
    _, _, run = C.load_settings()
    skip = {t.strip() for t in run.get("생성 제외 attack_type", "").split(",") if t.strip() not in ("", "-")}
    sides = {s.strip() for s in run.get("생성 대상 side", "request").split(",") if s.strip()}
    excl = C.excluded_rules()          # LLM_설정.md '생성 제외 룰' (다시 만들어도 코드 검사를 넘을 수 없는 룰, 2026-10-30)
    hi = C.run_int(run, "기존 데이터 충분 기준", 1000)
    latest_base = C.output_dir(in_dir.name)
    cov_src = next((b for b in [latest_base] + C.output_dates()[::-1]
                    if (b / "_0_existing" / "coverage.json").exists()), None)
    cov = _json(cov_src / "_0_existing" / "coverage.json", {}) if cov_src else {}

    entries, req_stat, usage, seconds, frontier, last_done, bias = scan_outputs(latest_base)
    rb = review_bias(bias, run)
    rows, flags = [], collections.Counter()
    by_source = collections.defaultdict(collections.Counter)
    by_type = collections.defaultdict(collections.Counter)
    samples = collections.Counter()
    for r in C.read_jsonl(in_dir / "rules_all.jsonl"):
        rid, h = r["id"], C.rule_hash(r)
        existing = cov.get(rid, {}).get("count", 0)
        row = {"id": rid, "source": rid.split(":")[0], "attack_type": r.get("attack_type", ""),
               "side": r.get("side", ""), "rule_name": r.get("rule_name", ""), "existing": existing,
               "date": "", "request": "", "anomalies": 0, "normal": 0, "reason": ""}
        if r.get("attack_type") in skip or r.get("side", "request") not in sides:
            flag = "EXCLUDED"
            row["reason"] = (f"생성 제외 유형 ({r.get('attack_type')})" if r.get("attack_type") in skip
                             else f"생성 대상 방향 아님 ({r.get('side') or '?'})")
        elif rid in excl:
            flag = "EXCLUDED"
            row["reason"] = "생성 제외 룰: " + excl[rid]
            same = [e for e in entries.get(rid, []) if e[1] == h]
            if same:          # 이미 만든 샘플은 그대로 둔다 (어디 있는지만 보여 주고, 샘플 합계에는 넣지 않음)
                best = max(same, key=lambda e: (RANK[e[0]], e[2], e[3]))
                row.update(date=best[2], request=best[3], anomalies=best[4], normal=best[5])
        elif existing >= hi:
            flag = "EXISTING"
        else:
            same = [e for e in entries.get(rid, []) if e[1] == h]
            if same:
                best = max(same, key=lambda e: (RANK[e[0]], e[2], e[3]))     # 가장 많이 진행된 것, 같으면 최근
                flag = best[0]
                row.update(date=best[2], request=best[3], anomalies=best[4], normal=best[5])
            elif entries.get(rid):
                flag = "CHANGED"
            else:
                flag = "TODO"
        row["flag"], row["status"] = flag, FLAGS[flag][0]
        rows.append(row)
        flags[flag] += 1
        by_source[row["source"]][flag] += 1
        by_type[row["attack_type"]][flag] += 1
        if flag == "DONE":
            samples["anomalies"] += row["anomalies"]
            samples["normal"] += row["normal"]

    def pct(a, b):
        return round(100.0 * a / b, 2) if b else 0.0

    total = len(rows)
    target = total - flags["EXCLUDED"] - flags["EXISTING"]

    def group(counter):
        t = sum(counter.values()) - counter["EXCLUDED"] - counter["EXISTING"]
        return {"rules": sum(counter.values()), "target": t, "done": counter["DONE"], "pct": pct(counter["DONE"], t)}

    gen_n, rev_n = len(seconds["generate"]), len(seconds["review"])
    avg = {"generate_seconds": round(sum(seconds["generate"]) / gen_n, 1) if gen_n else None,
           "review_seconds": round(sum(seconds["review"]) / rev_n, 1) if rev_n else None,
           "generator_tokens": round((usage["generator"]["input_tokens"] + usage["generator"]["output_tokens"]) / gen_n)
           if gen_n else None,
           "reviewer_tokens": round((usage["reviewer"]["input_tokens"] + usage["reviewer"]["output_tokens"]) / rev_n)
           if rev_n else None,
           # 화면의 '요청 수 → 예상 시간·토큰' 계산용: 모델별 입력/출력 (재시도·실패 호출까지 포함한 실제 평균)
           "generator_input_tokens": round(usage["generator"]["input_tokens"] / gen_n) if gen_n else None,
           "generator_output_tokens": round(usage["generator"]["output_tokens"] / gen_n) if gen_n else None,
           "reviewer_input_tokens": round(usage["reviewer"]["input_tokens"] / rev_n) if rev_n else None,
           "reviewer_output_tokens": round(usage["reviewer"]["output_tokens"] / rev_n) if rev_n else None,
           "generated_requests": gen_n, "reviewed_requests": rev_n}
    remaining = req_stat["pending"] + flags["TODO"] // 3           # 요청 대기 + (요청이 아직 없는 룰 / 호출당 3개)
    # 요청 1개가 걸리는 실제 시간(평균)을 동시 처리 수로 나눈다. 생성·검토를 함께 돌리면 더 느린 쪽이 전체 속도를 정한다
    gw, rw, overlap = C.parallel_settings(run)
    g_s, r_s = (avg["generate_seconds"] or 0) / gw, (avg["review_seconds"] or 0) / rw
    per_request = max(g_s, r_s) if overlap else g_s + r_s
    est = {"requests": remaining,
           "hours": round(remaining * per_request / 3600, 1) if avg["generate_seconds"] else None,
           "parallel": {"generate": gw, "review": rw, "overlap": overlap},
           "generator_tokens": remaining * avg["generator_tokens"] if avg["generator_tokens"] else None,
           "reviewer_tokens": remaining * avg["reviewer_tokens"] if avg["reviewer_tokens"] else None}
    summary = {
        "computed_at": dt.datetime.now().isoformat(timespec="seconds"),
        "input_date": in_dir.name, "output_dates": [b.name for b in C.output_dates()],
        "rules": total, "target": target, "done": flags["DONE"],
        "pct_done_of_target": pct(flags["DONE"], target),
        "pct_with_data_of_all": pct(flags["DONE"] + flags["EXISTING"], total),
        "flags": {k: flags[k] for k in FLAGS}, "flag_names": {k: v[0] for k, v in FLAGS.items()},
        "by_source": {k: group(v) for k, v in sorted(by_source.items())},
        "by_attack_type": dict(sorted(((k, group(v)) for k, v in by_type.items()), key=lambda kv: -kv[1]["target"])),
        "samples": dict(samples), "requests": req_stat, "frontier": frontier, "last_done": last_done,
        "usage": {k: dict(v) for k, v in usage.items()}, "average_per_request": avg, "estimate_remaining": est,
        "review_bias": rb, "seconds": round(time.time() - started, 1),
    }
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    C.write_jsonl(PROGRESS_DIR / "rule_status.jsonl", rows)
    tmp = PROGRESS_DIR / f"rule_status.csv.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["룰 id", "플래그", "상태", "출처", "공격 유형", "방향", "룰 이름", "기존 샘플 수",
                    "날짜 폴더", "요청", "anomalies", "normal", "제외 사유"])
        for r in rows:
            w.writerow([r["id"], r["flag"], r["status"], r["source"], r["attack_type"], r["side"], r["rule_name"],
                        r["existing"], r["date"], r["request"], r["anomalies"], r["normal"], r.get("reason", "")])
    tmp.replace(PROGRESS_DIR / "rule_status.csv")
    ptmp = PROGRESS_DIR / f"progress.json.{os.getpid()}.tmp"
    ptmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    ptmp.replace(PROGRESS_DIR / "progress.json")
    if log:
        log.info("[3.5] 현재 룰 %d개 (입력 %s) | 생성 대상 %d개 중 완료 %d개 (%.2f%%) | 학습데이터 보유 %.2f%% (완료 + 기존 충분)",
                 total, in_dir.name, target, flags["DONE"], summary["pct_done_of_target"],
                 summary["pct_with_data_of_all"])
        log.info("[3.5] 상태: %s", ", ".join(f"{FLAGS[k][0]} {flags[k]}" for k in FLAGS if flags[k]))
        fr = frontier
        log.info("[3.5] 어디까지: 배치 %d/%d 까지 빠짐없이 생성 (가장 뒤 배치 %d), 요청 %d개 중 생성 %d · 실패 %d · 대기 %d",
                 fr["contiguous_done_batch"], fr["batches_total"], fr["last_touched_batch"], req_stat["total"],
                 req_stat["generated"], req_stat["failed"], req_stat["pending"])
        if est["hours"] is not None:
            log.info("[3.5] 남은 요청 약 %d개 → 약 %.0f시간 (동시 생성 %d·검토 %d%s), 토큰 약 생성 %s · 검토 %s",
                     remaining, est["hours"], gw, rw, ", 생성·검토 함께" if overlap else "",
                     f"{est['generator_tokens'] or 0:,}", f"{est['reviewer_tokens'] or 0:,}")
        for scope, name in (("all", "전체"), ("latest", f"최근 {rb['latest_date']}")):
            types = rb[scope]["by_type"]
            if rb[scope]["alerts"]:
                log.warning("[3.5] [경고] 거부 편향 (%s): 검토 모델이 안전 사유로 빼거나 거절한 공격 샘플이 %s%% 를 넘는 유형 - %s "
                            "(화면 2 '검토에서 뺀 샘플' 에서 확인, 다시 만들기 고려)", name, f"{rb['threshold_pct']:g}",
                            ", ".join(f"{t} {types[t]['refusal_pct']:.1f}% ({types[t]['safety_dropped'] + types[t]['refused']}"
                                      f"/{types[t]['generated_attacks']})" for t in rb[scope]["alerts"]))
            elif types:
                top = max(types, key=lambda t: types[t]["refusal_pct"])
                log.info("[3.5] 거부 편향 (%s): 기준 %s%% 넘는 유형 없음 (가장 높은 %s %.1f%%)", name,
                         f"{rb['threshold_pct']:g}", top, types[top]["refusal_pct"])
        log.info("[3.5] → %s (%.1f초)", PROGRESS_DIR, summary["seconds"])
    return summary


def main():
    log = C.setup_logging("3_5_progress")
    try:
        compute(log)
    except RuntimeError as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
