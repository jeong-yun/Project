"""[3.3 데이터 검토] 생성된 샘플을 검토 모델(LLM_설정.md, 2026-10-26 부터 Claude CLI sonnet)로 검토해 pass / fix / drop 을 반영한다.  (3_0 이 실행)

[전체 흐름]  _2_generated/<이름>.jsonl 하나마다 (이미 _3_reviewed 에 결과가 있으면 건너뜀)
  1. 검토 지시문(LLM_검토_프롬프트.md) + 룰 + 생성 샘플(번호 i) + 코드 검사 결과 → 검토 모델 호출
  2. 판정 반영: drop = 빼기, fix = 고친 HTTP 로 바꾸기, pass = 그대로. 공격 유형(attack_type)도 확정.
  3. 남은 샘플을 코드 검사(룰 조건 대조)로 한 번 더 확인 → 룰 기준 라벨만 남긴다
     anomalies = 룰 조건을 모두 만족(탐지), normal = 조건을 하나 이상 어김(미탐지).
     어긋나면 '로컬 검사 실패 처리', 룰로 판단할 수 없으면 '판정 불가 처리' (둘 다 기본 drop = 뺀다)
  4. _3_reviewed/<이름>.jsonl (+ .rejected.jsonl: 빠진 샘플과 이유, .meta.json, raw/<이름>.txt)

  검토 모델을 'none' 으로 하면 호출 없이 코드 검사만 해서 넘긴다.

[대체 검토] (2026-10-26) 검토 모델이 응답 전체를 거절하거나, '안전하게 고칠 수 없다' 며 샘플을 뺀 룰만
  LLM_설정.md '검토 대체 모델'(예: openai:gpt-5)로 다시 검토한다. gpt-6-luna 는 명령 주입·RCE·인증 우회·오버플로·역직렬화 등의
  공격 샘플을 '악용 가능한 요청을 보존·보완할 수 없다' 며 13% 쯤 뺐고(gpt-5 는 같은 요청 30개에서 0개), 거절하면 같은 모델로
  다시 보내도 대개 또 거절해 토큰만 썼다 → 거절하면 바로 대체 모델로. 룰 조건상 만들 수 없어 뺀 것은 그대로 둔다.
  meta.json 의 fallback(대체 모델·룰·이유), usage 의 reviewer = 두 모델 합계 (reviewer_primary / reviewer_fallback 따로).
  원문은 raw/<이름>.try<n>.txt (처음 모델), raw/<이름>.fb.try<n>.txt (대체 모델).
  대체 모델이 처음 검토 모델과 같으면 보통 실행에서는 쓰지 않는다 (거절하면 같은 모델로 다시 보냄). --redo-safety 에서만 쓴다.
  OpenAI 크레딧이 떨어져 2026-10-26 에 검토·대체 검토를 claude_cli:sonnet 으로 바꿨다 (같은 요청 10개에서 gpt-5 와 남은 샘플이 같음).

[속도] 요청 여러 개를 동시에 검토한다 (LLM_설정.md '검토 동시 요청 수', 기본 4).
       --follow <생성 PID>: 생성(3.2)과 함께 돌면서 생성된 요청부터 바로 검토한다 (3_0 이 '생성·검토 동시 진행' 때 사용).

[사용법]
    python 3_3_review.py                       # 생성됐지만 검토 안 된 것 전부
    python 3_3_review.py --max-requests 15     # 15개만 (오래된 대기분부터)
    python 3_3_review.py --reviewer claude_cli:opus
    python 3_3_review.py --reviewer none       # 검토 생략 (코드 검사만)
    python 3_3_review.py --redo-safety         # 이미 검토한 것 중 거절·안전 사유로 뺀 룰만 대체 모델로 다시 검토 → 3.4·1.4·3.5
                                               # (처음 모델의 판정은 raw 원문에서 다시 읽음. 바뀐 결과는 _replaced 로 옮겨 둠)
"""

import argparse
import datetime as dt
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_spec = importlib.util.spec_from_file_location("llm_common", Path(__file__).resolve().parent / "3_0a_common.py")
C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)

# 검토 모델이 '안전' 을 이유로 샘플을 뺐거나 거절한 문장 (이 룰은 대체 모델로 다시 검토). 2026-10-26 gpt-6-luna 의 삭제 이유
# 8,679개를 보고 정함 (gpt-5 가 뺀 이유 49개에는 하나도 안 걸림). 잘못 걸려도 대체 모델로 한 번 더 검토할 뿐 샘플을 빼지 않는다.
SAFETY_RE = re.compile(r"안전|지원할 수 없|도울 수 없|도와드릴 수 없|도움을 드릴 수 없|제공할 수 없|보존하거나|보완할 수 없|"
                       r"실행 가능한|구체화|악용|무기화|만들 수 없|재현.{0,12}(수 없|어렵)|"
                       r"can.?t (help|assist)|cannot (help|assist|provide)|unable to (help|assist)|I.?m sorry", re.I)
AUG_DIR = "_4_augmented"                       # 3_4_augment.py 와 같음


def is_refusal(text):
    """판정 JSON 이 없는 설명 글(거절)인지. 빈 응답·깨진 JSON 은 거절이 아님 (같은 모델로 다시 요청)."""
    s = (text or "").strip()
    return bool(s) and not s.startswith(("{", "["))


def safety_dropped(review):
    """판정 중 '안전' 사유로 뺀 샘플이 있는지."""
    return any(isinstance(v, dict) and v.get("verdict") == "drop" and SAFETY_RE.search(str(v.get("reason", "")))
               for v in (review or {}).get("reviews", []))


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


def review_items(rules, by_id):
    """검토 모델에 보낼 룰별 묶음: 룰 + 생성 샘플(번호 i) + 코드 검사 결과."""
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
    return items


def ask(name, tag, items, cfg, providers, run, prompt, rev_dir, log, usage, key, give_up_on_refusal=False):
    """검토 모델 호출 (판정 JSON 이 없으면 다시) → ({룰 id: 판정}, 마지막 응답이 거절이었는지).
    give_up_on_refusal: 거절하면 같은 모델로 다시 보내지 않음 (대체 모델이 있을 때). 원문은 raw/<이름>.<tag><n>.txt"""
    user = (f"다음은 검토할 룰 {len(items)}개다 (JSON Lines). 지시문의 출력 형식대로 룰마다 JSON 한 줄씩만 출력하라.\n\n"
            + "\n".join(json.dumps(i, ensure_ascii=False) for i in items))
    ids = {i["rule"]["id"] for i in items}
    refused = False
    for attempt in range(1, C.run_int(run, "재시도 횟수", 3) + 1):
        text, u = C.call_llm(cfg, providers, prompt, user, run, log)
        C.add_usage(usage, key, u)
        (rev_dir / "raw").mkdir(parents=True, exist_ok=True)
        (rev_dir / "raw" / f"{name}.{tag}{attempt}.txt").write_text(text, encoding="utf-8")   # 시도마다 원문 보관
        verdicts = {o["id"]: o for o in C.parse_json_lines(text) if isinstance(o, dict) and o.get("id") in ids}
        if verdicts:
            return verdicts, False
        refused = is_refusal(text)
        stop = refused and give_up_on_refusal
        log.warning("  %s: 검토 응답에서 JSON 을 찾지 못함 (%d번째)%s - %s", name, attempt,
                    f" - 거절: {' '.join(text.split())[:90]}" if refused else "", "대체 모델로" if stop else "다시 요청")
        if stop:
            break
    return {}, refused


def _total(usage):
    out = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
    for k in ("reviewer_primary", "reviewer_fallback"):
        for f in out:
            out[f] += (usage.get(k) or {}).get(f, 0)
    return out


def review(name, rules, generated, cfg, providers, run, prompt, rev_dir, log, primary=None, fb_given=None, extra=None):
    """요청 하나 검토 → _3_reviewed/<이름>.jsonl·.rejected.jsonl·.meta.json.
    primary: (--redo-safety) 처음 모델에서 이미 받은 판정 {룰 id: 판정} (다시 묻지 않음), 그때의 토큰은 extra["usage"]
    fb_given: (--redo-safety) 대체 모델로 미리 받은 판정 (있으면 그 룰은 대체 모델을 다시 부르지 않음)"""
    by_id = {g["id"]: g for g in generated}
    usage, verdicts, fallback = {}, {}, None
    for k, src in (("reviewer_primary", "usage"), ("reviewer_fallback", "fb_usage")):
        if extra and extra.get(src):
            usage[k] = dict(extra[src])
    started = time.time()
    if cfg is not None:
        items = review_items(rules, by_id)
        if items:
            fb = cfg.get("fallback")
            if primary is None:
                verdicts, _ = ask(name, "try", items, cfg, providers, run, prompt, rev_dir, log, usage, "reviewer_primary",
                                  give_up_on_refusal=fb is not None)
            else:
                verdicts = dict(primary)
            # 대체 검토: 처음 모델이 판정을 주지 않았거나(거절) '안전' 사유로 뺀 샘플이 있는 룰만
            need = [i for i in items if i["rule"]["id"] not in verdicts or safety_dropped(verdicts[i["rule"]["id"]])]
            if fb is not None and need:
                why = {i["rule"]["id"]: ("응답 거절" if i["rule"]["id"] not in verdicts else "안전 사유로 뺀 샘플") for i in need}
                got = {k: v for k, v in (fb_given or {}).items() if k in why}
                rest = [i for i in need if i["rule"]["id"] not in got]
                if rest:
                    v2, _ = ask(name, "fb.try", rest, fb, providers, run, prompt, rev_dir, log, usage, "reviewer_fallback")
                    got.update(v2)
                verdicts.update(got)
                fallback = {"model": f"{fb['provider']}:{fb['model']}", "rules": why, "answered": sorted(got)}
            if not verdicts:
                raise C.LLMError(f"{name}: 검토 응답을 해석하지 못함 (다음 실행 때 다시 시도)")
        usage["reviewer"] = _total(usage)

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
    if fallback:
        meta["fallback"] = fallback
    if extra and extra.get("redo"):
        meta["redo"] = extra["redo"]
    (rev_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    fb_txt = ""
    if fallback:
        kinds = sorted(set(fallback["rules"].values()))
        fb_txt = (f" · 대체 검토 {fallback['model']} 룰 {len(fallback['answered'])}/{len(fallback['rules'])}개"
                  f" ({', '.join(kinds)})")
    # 토큰은 두 모델 합계 (화면·진행 현황이 이 줄의 첫 토큰 값을 읽음)
    log.info("  %s: 샘플 %d개 남음 (anomalies %d, normal %d, 수정 %d), 제외 %d개, 토큰 %s%s", name, meta["samples"],
             meta["anomalies"], meta["normal"], meta["fixed"], len(rejected), usage.get("reviewer"), fb_txt)


def _first_verdicts(raws, ids):
    """처음 모델의 원문(try1, try2 …)에서 실제로 쓴 판정 = 처음으로 해석된 응답의 판정."""
    for raw in sorted(raws, key=lambda p: int(re.search(r"\.try(\d+)\.txt$", p.name).group(1))):
        got = {o["id"]: o for o in C.parse_json_lines(raw.read_text(encoding="utf-8", errors="replace"))
               if isinstance(o, dict) and o.get("id") in ids}
        if got:
            return got
    return {}


def redo_candidates(bases, fb_model):
    """이미 검토한 요청 중 처음 모델이 거절했거나 '안전' 사유로 뺀 룰이 있는 것 → [(날짜 폴더, 이름, 옛 meta, 처음 판정)]."""
    out = []
    for base in bases:
        rv = base / C.REV_DIR
        for m in sorted(rv.glob("*.meta.json")) if rv.is_dir() else []:
            try:
                meta = json.loads(m.read_text(encoding="utf-8"))
            except ValueError:
                continue
            name = meta.get("request") or m.name[:-len(".meta.json")]
            if meta.get("fallback") or meta.get("reviewer") in (None, fb_model):
                continue
            gen_p, req_p = base / C.GEN_DIR / f"{name}.jsonl", base / C.REQ_DIR / f"{name}.jsonl"
            if not (gen_p.exists() and req_p.exists()):
                continue
            ids = {g["id"] for g in C.read_jsonl(gen_p)}
            prim = _first_verdicts(sorted((rv / "raw").glob(f"{name}.try*.txt")), ids)
            if any(rid not in prim or safety_dropped(prim[rid]) for rid in ids):
                out.append((base, name, meta, prim))
    return out


def set_aside(base, name):
    """다시 검토하는 요청의 옛 결과를 _replaced 로 옮긴다 (검토 결과 + 증강 결과 → 3.4 가 다시 증강, 1.4 가 다시 변환)."""
    moves = [(base / C.REV_DIR / f"{name}{suf}", base / C.REV_DIR / "_replaced") for suf in (".jsonl", ".rejected.jsonl", ".meta.json")]
    moves += [(base / f"{name}.jsonl", base / AUG_DIR / "_replaced"), (base / AUG_DIR / f"{name}.meta.json", base / AUG_DIR / "_replaced")]
    for src, dst in moves:
        if src.exists():
            dst.mkdir(parents=True, exist_ok=True)
            src.replace(dst / src.name)


def redo_safety(args, cfg, providers, run, log, workers):
    """--redo-safety: 거절·안전 사유로 뺀 룰만 대체 모델로 다시 검토 → 3.4 증강 → 1.4 학습 로그 → 3.5 진행 현황."""
    fb = (cfg or {}).get("fallback")
    if fb is None:
        log.error("LLM_설정.md 의 '검토 대체 모델' 이 없습니다 (예: claude_cli:sonnet)")
        return 1
    fb_model = f"{fb['provider']}:{fb['model']}"
    bases = [C.output_dir(args.date)] if args.date else C.output_dates()
    todo = redo_candidates(bases, fb_model)
    if args.max_requests > 0:
        todo = todo[:args.max_requests]
    log.info("[3.3] 검토 %s | 다시 검토: 처음 검토 모델이 거절했거나 안전 사유로 뺀 룰이 있는 요청 %d개 중 이번 %d개 (동시 최대 %d개씩, 속도 제한에 맞춰 자동 조절)",
             fb_model, len(todo), len(todo), workers)
    if args.dry_run or not todo:
        return 0
    if not C.generation_guard(log, "3_3_review --redo-safety"):
        return 2
    prompt = C.REV_PROMPT_FILE.read_text(encoding="utf-8")
    fb["throttle"] = C.Throttle(workers, log, "검토 대체")
    reuse = Path(args.reuse_raw) if args.reuse_raw else None
    done, failed, fatal = [], [], threading.Event()

    def one(item):
        if fatal.is_set():                          # 로그인·사용 한도·크레딧 문제 뒤에는 새로 시작하지 않음 (다음에 다시)
            return
        base, name, old, prim = item
        rules = C.read_jsonl(base / C.REQ_DIR / f"{name}.jsonl")
        gens = C.read_jsonl(base / C.GEN_DIR / f"{name}.jsonl")
        rev_dir = base / C.REV_DIR
        given, fb_usage = None, None
        if reuse and (reuse / base.name / "raw" / f"{name}.try1.txt").exists():      # 미리 대체 모델로 검토한 원문 (시험)
            src = reuse / base.name / "raw" / f"{name}.try1.txt"
            given = {o["id"]: o for o in C.parse_json_lines(src.read_text(encoding="utf-8")) if isinstance(o, dict) and o.get("id")}
            (rev_dir / "raw").mkdir(parents=True, exist_ok=True)
            (rev_dir / "raw" / f"{name}.fb.reuse.txt").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            m2 = reuse / base.name / f"{name}.meta.json"
            if m2.exists():
                fb_usage = (json.loads(m2.read_text(encoding="utf-8")).get("usage") or {}).get("reviewer")
        try:
            # 대체 모델 판정을 먼저 받아 두고(실패하면 옛 결과 그대로), 옛 결과를 옮긴 뒤 새로 쓴다
            by_id = {g["id"]: g for g in gens}
            need = [i for i in review_items(rules, by_id) if i["rule"]["id"] not in prim or safety_dropped(prim[i["rule"]["id"]])]
            usage = {}
            rest = [i for i in need if i["rule"]["id"] not in (given or {})]
            if rest:
                v2, _ = ask(name, "fb.try", rest, fb, providers, run, prompt, rev_dir, log, usage, "reviewer_fallback")
                if not v2:
                    raise C.LLMError(f"{name}: 대체 모델 응답을 해석하지 못함")
                given = dict(given or {}, **v2)
                fb_usage = usage["reviewer_fallback"] if not fb_usage else {
                    k: fb_usage.get(k, 0) + usage["reviewer_fallback"].get(k, 0) for k in ("calls", "input_tokens", "output_tokens")}
            set_aside(base, name)
            # 처음 판정은 그때의 검토 모델(옛 meta) 것이므로 meta 의 reviewer 도 그 모델로 남긴다
            p_prov, _, p_model = str(old.get("reviewer") or f"{cfg['provider']}:{cfg['model']}").partition(":")
            review(name, rules, gens, dict(cfg, provider=p_prov, model=p_model), providers, run, prompt, rev_dir, log,
                   primary=prim, fb_given=given,
                   extra={"usage": ((old.get("usage") or {}).get("reviewer_primary") or (old.get("usage") or {}).get("reviewer")),
                          "fb_usage": fb_usage, "redo": {"at": dt.datetime.now().isoformat(timespec="seconds"),
                                                         "before": {k: old.get(k) for k in ("samples", "rejected", "fixed", "finished_at")}}})
            done.append(name)
        except C.LLMError as e:
            failed.append(name)
            log.error("  %s 다시 검토 실패 (옛 결과 그대로): %s", name, e)
            if C.is_fatal(e) and not fatal.is_set():
                fatal.set()
                log.error("로그인·사용 한도·크레딧 문제라 나머지도 중단합니다 (다음 --redo-safety 때 이어서)")
        except Exception:
            failed.append(name)
            log.exception("  %s 다시 검토 실패 (예상 못 한 오류, 옛 결과 그대로)", name)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="redo") as ex:
        list(ex.map(one, todo))
    left = len(todo) - len(done) - len(failed)
    log.info("[3.3] 다시 검토 끝: %d개 바뀜, %d개 실패(옛 결과 그대로)%s", len(done), len(failed),
             f", {left}개는 시작하지 않음 (다음에 다시)" if left else "")
    if not done:
        return 1 if failed else 0
    # 바뀐 요청을 다시 증강·변환 (이 프로세스가 잡은 생성 잠금 아래에서, 3_0 처럼)
    env = dict(os.environ, **{C.LOCK_OWNER_ENV: "generation", "PYTHONIOENCODING": "utf-8"})
    for script in ("3_4_augment.py", "1_4_build_dataset.py", "3_5_progress.py"):
        log.info("[3.3] 다시 검토 후 %s", script)
        code = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / script)], env=env,
                              cwd=str(Path(__file__).resolve().parent)).returncode
        if code != 0:
            log.error("[3.3] %s 실패 (코드 %d) - 로그 확인", script, code)
            return 1
    return 1 if failed else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="검토 모델로 생성 샘플 검토")
    ap.add_argument("--date", help="날짜 폴더 (기본: 05_llm_input 의 가장 최근)")
    ap.add_argument("--max-requests", type=int, default=0, help="처리할 요청 수 (기본 0 = 생성됐지만 검토 안 된 것 전부)")
    ap.add_argument("--reviewer", help="검토 모델 'provider:model' 또는 'none' (LLM_설정.md 보다 우선)")
    ap.add_argument("--force", action="store_true", help="이미 검토한 것도 다시")
    ap.add_argument("--follow", type=int, metavar="PID",
                    help="이 생성(3.2) 프로세스가 끝날 때까지 새로 생성되는 결과를 바로 검토 (3_0 이 사용)")
    ap.add_argument("--dry-run", action="store_true", help="호출하지 않고 무엇을 할지만 출력")
    ap.add_argument("--redo-safety", action="store_true",
                    help="이미 검토한 것 중 거절·안전 사유로 뺀 룰만 '검토 대체 모델'로 다시 검토 → 3.4·1.4·3.5 (옛 결과는 _replaced)")
    ap.add_argument("--reuse-raw", metavar="DIR", help="(--redo-safety) 대체 모델로 미리 검토한 원문 폴더 <날짜>/raw/<이름>.try1.txt")
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
    fb_name = run.get("검토 대체 모델", "").strip()       # 거절·안전 사유로 뺀 룰만 이 모델로 다시 검토 (비우면 안 함)
    if cfg is not None and fb_name not in ("", "-", "none"):
        fb = C.role_config(roles, "검토", fb_name)
        # 보통 실행은 처음 검토 모델과 같으면 쓰지 않는다 (같은 모델로 다시 보냄). --redo-safety 는 처음 판정이
        # 예전 검토 모델의 원문이라 지금 검토 모델과 같아도 쓴다
        if fb is not None and (args.redo_safety or f"{fb['provider']}:{fb['model']}" != f"{cfg['provider']}:{cfg['model']}"):
            if fb.get("provider") == "openai" and effort not in ("", "-"):
                fb["reasoning_effort"] = effort
            cfg["fallback"] = fb
    if args.redo_safety:
        return redo_safety(args, cfg, providers, run, log, C.parallel_settings(run)[1])
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
        if cfg.get("fallback"):
            cfg["fallback"]["throttle"] = C.Throttle(workers, log, "검토 대체")
            log.info("[3.3] 대체 검토 %s:%s (처음 검토 모델이 거절하거나 안전 사유로 뺀 룰만)",
                     cfg["fallback"]["provider"], cfg["fallback"]["model"])
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
            if C.is_fatal(e):                  # 설정·키·모델 이름·크레딧·Claude 로그인·구독 사용 한도
                if not fatal.is_set():
                    fatal.set()
                    log.error("설정·키·모델 이름·크레딧·사용 한도 문제라 나머지도 중단합니다 (다음 실행 때 이어서)")
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
