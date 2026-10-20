"""[3.3 데이터 검토] 생성된 샘플을 검토 모델(기본 ChatGPT gpt-5)로 검토해 pass / fix / drop 을 반영한다.  (3_0 이 실행)

[전체 흐름]  _2_generated/<이름>.jsonl 하나마다 (이미 _3_reviewed 에 결과가 있으면 건너뜀)
  1. 검토 지시문(LLM_검토_프롬프트.md) + 룰 + 생성 샘플(번호 i) + 코드 검사 결과 → 검토 모델 호출
  2. 판정 반영: drop = 빼기, fix = 고친 HTTP 로 바꾸기, pass = 그대로. 공격 유형(attack_type)도 확정.
  3. 남은 샘플을 코드 검사(룰 조건 대조)로 한 번 더 확인 → 룰 기준 라벨만 남긴다
     anomalies = 룰 조건을 모두 만족(탐지), normal = 조건을 하나 이상 어김(미탐지).
     어긋나면 '로컬 검사 실패 처리', 룰로 판단할 수 없으면 '판정 불가 처리' (둘 다 기본 drop = 뺀다)
  4. _3_reviewed/<이름>.jsonl (+ .rejected.jsonl: 빠진 샘플과 이유, .meta.json, raw/<이름>.txt)

  검토 모델을 'none' 으로 하면 호출 없이 코드 검사만 해서 넘긴다.

[속도] 요청 여러 개를 동시에 검토한다 (LLM_설정.md '검토 동시 요청 수', 기본 4).
       --follow <생성 PID>: 생성(3.2)과 함께 돌면서 생성된 요청부터 바로 검토한다 (3_0 이 '생성·검토 동시 진행' 때 사용).

[사용법]
    python 3_3_review.py                       # 생성됐지만 검토 안 된 것 전부
    python 3_3_review.py --max-requests 15     # 15개만 (오래된 대기분부터)
    python 3_3_review.py --reviewer openai:gpt-5
    python 3_3_review.py --reviewer none       # 검토 생략 (코드 검사만)
"""

import argparse
import datetime as dt
import importlib.util
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_spec = importlib.util.spec_from_file_location("llm_common", Path(__file__).resolve().parent / "3_0a_common.py")
C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)


def merge(rule, gen, review, policy, rejected, unknown_policy="drop", realistic=True):
    """생성 결과 + 검토 판정 + 코드 재검사 → 최종 레코드 (남은 샘플이 없으면 None).
    realistic: 검토 모델이 고친 HTTP 에도 헤더 현실화(실제 UA·기본 헤더·Content-Length)를 다시 적용."""
    rid = rule["id"]
    atype = (review or {}).get("attack_type") or gen.get("attack_type") or rule["attack_type"]
    if atype not in C.ATTACK_TYPES:
        atype = rule["attack_type"]
    verdicts = {v.get("i"): v for v in (review or {}).get("reviews", []) if isinstance(v, dict)}
    final = []
    for i, s in enumerate(gen.get("samples", [])):
        ref = {"id": rid, "i": i}
        if not C.valid_sample(s):
            rejected.append(dict(ref, stage="형식", reason="label/http 오류")); continue
        v = verdicts.get(i, {"verdict": "pass" if review is None else "missing"})
        verdict = v.get("verdict")
        if verdict == "drop":
            rejected.append(dict(ref, stage="검토", reason=v.get("reason", "drop"))); continue
        s = dict(s)
        if verdict == "fix" and isinstance(v.get("http"), str) and v["http"].strip():
            s["http"], s["note"] = C.expand_repeats(v["http"]), v.get("note", s.get("note", ""))
            s = C.repair_sample(s, rule)
        if realistic:
            s = C.realistic_sample(s, rule)
        chk = C.local_check(s, rule)
        # 룰 기준 라벨만 남긴다: anomalies = 룰이 탐지, normal = 룰이 탐지 안 함. 어긋나면(fail) 뺀다
        if chk["result"] == "fail" and policy == "drop":
            rejected.append(dict(ref, stage="코드 검사", reason="; ".join(chk["reasons"]))); continue
        if chk["result"] == "unknown" and unknown_policy == "drop":   # 룰로 확인할 수 없는 샘플
            rejected.append(dict(ref, stage="코드 검사", reason="판정 불가: " + "; ".join(chk["reasons"]))); continue
        final.append({"label": s["label"], "attack_type": atype if s["label"] == "anomalies" else "none",
                      "http": s["http"], "note": s.get("note", ""), "origin": "llm",
                      "review": verdict, "local_check": chk["result"]})
    if not final:
        return None
    out = {"id": rid, "attack_type": atype, "samples": final}
    note = (review or {}).get("attack_type_note") or gen.get("attack_type_note")
    if note:
        out["attack_type_note"] = note
    return out


def review(name, rules, generated, cfg, providers, run, prompt, rev_dir, log):
    by_id = {g["id"]: g for g in generated}
    usage, verdicts = {}, {}
    started = time.time()
    if cfg is not None:
        items = []
        for r in rules:
            g = by_id.get(r["id"])
            if not g:
                continue
            samples = [dict({k: v for k, v in s.items() if k != "local_check"}, i=i)
                       for i, s in enumerate(g["samples"]) if isinstance(s, dict)]
            checks = [dict(s.get("local_check", {"result": "fail", "reasons": ["형식 오류"]}), i=i)
                      for i, s in enumerate(g["samples"]) if isinstance(s, dict)]
            items.append({"rule": r, "generated": {"attack_type": g["attack_type"], "samples": samples},
                          "local_check": checks})
        if items:
            user = (f"다음은 검토할 룰 {len(items)}개다 (JSON Lines). 지시문의 출력 형식대로 룰마다 JSON 한 줄씩만 출력하라.\n\n"
                    + "\n".join(json.dumps(i, ensure_ascii=False) for i in items))
            ids = {i["rule"]["id"] for i in items}
            for attempt in range(1, C.run_int(run, "재시도 횟수", 3) + 1):
                text, u = C.call_llm(cfg, providers, prompt, user, run, log)
                C.add_usage(usage, "reviewer", u)
                (rev_dir / "raw").mkdir(parents=True, exist_ok=True)
                (rev_dir / "raw" / f"{name}.try{attempt}.txt").write_text(text, encoding="utf-8")   # 시도마다 원문 보관
                verdicts = {o["id"]: o for o in C.parse_json_lines(text) if isinstance(o, dict) and o.get("id") in ids}
                if verdicts:
                    break
                log.warning("  %s: 검토 응답에서 JSON 을 찾지 못함 (%d번째) - 다시 요청", name, attempt)
            if not verdicts:
                raise C.LLMError(f"{name}: 검토 응답을 해석하지 못함 (다음 실행 때 다시 시도)")

    policy = run.get("로컬 검사 실패 처리", "drop")
    unknown_policy = run.get("판정 불가 처리", "drop")
    rejected, final = [], []
    for r in rules:
        g = by_id.get(r["id"])
        if not g:
            continue
        rec = merge(r, g, verdicts.get(r["id"]) if cfg is not None else None, policy, rejected, unknown_policy,
                    C.run_yes(run, "헤더 현실화"))
        if rec:
            final.append(rec)
    C.write_jsonl(rev_dir / f"{name}.jsonl", final)
    C.write_jsonl(rev_dir / f"{name}.rejected.jsonl", rejected)
    flat = [s for r in final for s in r["samples"]]
    meta = {"request": name, "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
            "seconds": round(time.time() - started, 1),
            "reviewer": f"{cfg['provider']}:{cfg['model']}" if cfg else None,
            "rules": len(by_id), "rules_kept": len(final),
            "samples": len(flat), "anomalies": sum(s["label"] == "anomalies" for s in flat),
            "normal": sum(s["label"] == "normal" for s in flat),
            "fixed": sum(s["review"] == "fix" for s in flat), "rejected": len(rejected), "usage": usage}
    (rev_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("  %s: 샘플 %d개 남음 (anomalies %d, normal %d, 수정 %d), 제외 %d개, 토큰 %s", name, meta["samples"],
             meta["anomalies"], meta["normal"], meta["fixed"], len(rejected), usage.get("reviewer"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="검토 모델로 생성 샘플 검토")
    ap.add_argument("--date", help="날짜 폴더 (기본: 05_llm_input 의 가장 최근)")
    ap.add_argument("--max-requests", type=int, default=0, help="처리할 요청 수 (기본 0 = 생성됐지만 검토 안 된 것 전부)")
    ap.add_argument("--reviewer", help="검토 모델 'provider:model' 또는 'none' (LLM_설정.md 보다 우선)")
    ap.add_argument("--force", action="store_true", help="이미 검토한 것도 다시")
    ap.add_argument("--follow", type=int, metavar="PID",
                    help="이 생성(3.2) 프로세스가 끝날 때까지 새로 생성되는 결과를 바로 검토 (3_0 이 사용)")
    ap.add_argument("--dry-run", action="store_true", help="호출하지 않고 무엇을 할지만 출력")
    args = ap.parse_args(argv)
    log = C.setup_logging("3_3_review")
    C.load_env()

    try:
        roles, providers, run = C.load_settings()
    except (ValueError, FileNotFoundError) as e:
        log.error("LLM 설정 읽기 실패: %s", e)
        return 1
    cfg = C.role_config(roles, "검토", args.reviewer)
    effort = run.get("검토 추론 강도", "").strip()
    if cfg is not None and cfg.get("provider") == "openai" and effort not in ("", "-"):
        cfg["reasoning_effort"] = effort                  # gpt-5 등: low 면 빠르지만 판단이 얕아질 수 있음
    in_dir = C.latest_input_dir(args.date)
    if in_dir is None:
        log.error("05_llm_input 이 없습니다")
        return 1

    def scan():
        """(생성 결과 전체, 검토 안 된 것 [(날짜 폴더, 파일)]).
        날짜를 지정하지 않으면 모든 날짜 폴더의 '생성됐지만 검토 안 된' 결과 (입력 날짜가 바뀌어도 지난 날짜 폴더의
        검토 대기분이 남지 않게). --force 는 지정·최신 날짜만."""
        bases = [C.output_dir(in_dir.name)] if (args.date or args.force) else C.output_dates()
        gens, todo = [], []
        for base in bases:
            gen_dir, rev_dir = base / C.GEN_DIR, base / C.REV_DIR
            found = sorted(p for p in gen_dir.glob("*.jsonl")) if gen_dir.is_dir() else []
            gens += found
            todo += [(base, g) for g in found if args.force or not (rev_dir / g.name).exists()]
        return gens, todo

    gens, todo = scan()
    cap = args.max_requests if args.max_requests > 0 else 0        # 0 = 제한 없음
    follow = bool(args.follow) and C.RT.pid_alive(args.follow) and not args.dry_run
    workers = C.parallel_settings(run)[1]
    who = f"{cfg['provider']}:{cfg['model']}" if cfg else "없음 (코드 검사만)"
    if follow:   # 대상 수는 생성 결과에 따라 정해짐 (화면은 생성 수로 추정)
        log.info("[3.3] 검토 %s | 생성과 함께 진행: 대기 %d개 + 새로 생성되는 것%s (동시 최대 %d개씩, 속도 제한에 맞춰 자동 조절)", who, len(todo),
                 f", 최대 {cap}개" if cap else "", workers)
    else:
        todo = todo[:cap] if cap else todo
        log.info("[3.3] 검토 %s | 생성 결과 %d개 중 이번 %d개 (동시 최대 %d개씩, 속도 제한에 맞춰 자동 조절)%s%s", who, len(gens), len(todo), workers,
                 f" (날짜 폴더: {', '.join(sorted({b.name for b, _ in todo}))})" if todo else "",
                 " (dry-run: 호출 안 함)" if args.dry_run else "")
    if args.dry_run or not (todo or follow):
        return 0

    if not C.generation_guard(log, "3_3_review"):
        return 2
    prompt = C.REV_PROMPT_FILE.read_text(encoding="utf-8")
    if cfg is not None:
        cfg["throttle"] = C.Throttle(workers, log, "검토")   # 동시 요청 수: 4개부터 늘려 최대까지, 429 면 줄임
    fatal, stopped = threading.Event(), threading.Event()

    def one(base, g):
        """요청 하나 검토 (스레드에서). 중지 요청·설정 오류 뒤에는 새로 시작하지 않는다 (이미 보낸 요청은 끝까지)."""
        if fatal.is_set():
            return
        if C.RT.stop_requested("generation"):
            stopped.set()
            return
        req_dir, rev_dir = base / C.REQ_DIR, base / C.REV_DIR
        rev_dir.mkdir(parents=True, exist_ok=True)
        try:
            review(g.stem, C.read_jsonl(req_dir / g.name), C.read_jsonl(g), cfg, providers, run, prompt, rev_dir, log)
        except C.LLMError as e:
            log.error("  %s 실패: %s", g.stem, e)
            if "가 없습니다" in str(e) or "HTTP 401" in str(e) or "HTTP 403" in str(e) or "HTTP 404" in str(e):
                if not fatal.is_set():
                    fatal.set()
                    log.error("설정·키·모델 이름 문제라 나머지도 중단합니다")
        except Exception:                      # 예상 못 한 오류도 다른 요청은 계속
            log.exception("  %s 실패 (예상 못 한 오류)", g.stem)

    submitted = set()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rev") as ex:
        def submit(items):
            for base, g in items:
                if cap and len(submitted) >= cap:
                    return
                if g not in submitted:
                    submitted.add(g)
                    ex.submit(one, base, g)

        submit(todo)                           # 이전 실행의 검토 대기분부터
        while follow:                          # 생성이 끝날 때까지 새 생성 결과를 바로 검토
            if fatal.is_set() or (cap and len(submitted) >= cap):
                break
            if C.RT.stop_requested("generation"):
                stopped.set()
                break
            alive = C.RT.pid_alive(args.follow)
            submit(scan()[1])
            if not alive:                      # 생성이 끝난 뒤 한 번 더 찾았으므로 끝
                break
            time.sleep(3)
    # with 를 나오면 보낸 검토가 모두 끝난 것
    if stopped.is_set():
        log.info("[3.3] 중지 요청 - 여기서 멈춤 (남은 검토는 다음 실행 때)")
    return 1 if fatal.is_set() else 0


if __name__ == "__main__":
    sys.exit(main())
