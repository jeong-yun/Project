"""[3.3 데이터 검토] 생성된 샘플을 검토 모델(LLM_설정.md '## 모델' 의 '검토' 행)로 검토해 pass / fix / drop 을 반영한다.  (3_0 이 실행)
  모델 이름은 이 주석에 적지 않는다 - 지금 값은 LLM_설정.md, 실제로 쓴 값은 meta.json 의 provenance 를 본다.

[전체 흐름]  _2_generated/<이름>.jsonl 하나마다 (이미 _3_reviewed 에 결과가 있으면 건너뜀)
  1. 검토 지시문(LLM_검토_프롬프트.md) + 룰 + 생성 샘플(번호 i) + 코드 검사 결과 → 검토 모델 호출
  2. 판정 반영: drop = 빼기, fix = 고친 HTTP 로 바꾸기, pass = 그대로. 공격 유형(attack_type)도 확정.
  3. 남은 샘플을 코드 검사(룰 조건 대조)로 한 번 더 확인 → 룰 기준 라벨만 남긴다
     anomalies = 룰 조건을 모두 만족(탐지), normal = 조건을 하나 이상 어김(미탐지).
     어긋나면 '로컬 검사 실패 처리', 룰로 판단할 수 없으면 '판정 불가 처리' (둘 다 기본 drop = 뺀다)
  4. _3_reviewed/<이름>.jsonl (+ .rejected.jsonl: 빠진 샘플과 이유, .meta.json, raw/<이름>.txt)
     meta.json 에는 생성 이력(provenance: 생성·검토·대체 모델, temperature·max_tokens·추론 강도, 지시문 sha256)과
     공격 유형별 검토 결과(review_by_type: 생성 공격 샘플·검토 drop·그중 안전 사유·판정 거절 → 3.5 거부 편향 경보)를 남긴다.

  검토 모델을 'none' 으로 하면 호출 없이 코드 검사만 해서 넘긴다.

[거절·대체 검토] (2026-10-26, 2026-10-27 바꿈) LLM_설정.md '검토 대체 모델' 은 화면 2 '검토에서 뺀 샘플' 의
  '다시 만들기'(--redo-safety)를 누를 때만 쓴다. 보통 실행에서는 검토 모델과 달라도 쓰지 않는다 (2026-10-27 사용자 요청:
  검토 gpt-6-luna · 대체 Claude 로 두면 실행 중에 Claude 가 바로 돌아 구독 사용량을 씀).
  - 검토 모델이 응답을 거절하거나 일부 룰의 판정을 빼면, 판정이 빠진 룰만 같은 검토 모델로 '재시도 횟수'(3번)까지 다시 묻는다.
    그래도 판정이 없는 룰의 샘플은 단계 '검토 거절' 로 .rejected.jsonl 에 남는다 (meta.json 의 refused) → 화면 '검토에서 뺀 샘플'
    (응답 거절)·다시 만들기 대상. 예전에는 판정이 없는 룰을 검토 없이 통과시켰다. 판정을 하나도 읽지 못했고 거절도 아니면
    (깨진 응답) 지금처럼 다음 실행 때 다시 검토한다. 시간 초과·네트워크 오류도 다음 실행 때.
  - '안전하게 고칠 수 없다' 며 뺀 샘플은 그대로 빠지고(안전상 이유) 다시 만들기 대상이 된다. 룰 조건상 만들 수 없어 뺀 것은 그대로.
  - 다시 만들기: 거절·안전 사유로 뺀 룰만 대체 모델로 다시 검토 → 3.4·1.4·3.5. meta.json 의 fallback(대체 모델·룰·이유),
    usage 의 reviewer = 두 모델 합계 (reviewer_primary / reviewer_fallback 따로). 원문은 raw/<이름>.try<n>.txt (처음 모델),
    raw/<이름>.fb.try<n>.txt (대체 모델).
  근거 (2026-10-26): gpt-6-luna 는 명령 주입·RCE·인증 우회·오버플로·역직렬화 등의 공격 샘플을 '악용 가능한 요청을 보존·보완할 수
  없다' 며 많이 뺐다 (gpt-5 는 같은 요청 30개에서 0개). 비율은 분모에 따라 다르다 → docs/ai_governance/04_측정_지표.md 4.1
  '안전 거부' 표 (2026-10-29 다시 셈).

[속도] 요청 여러 개를 동시에 검토한다 (LLM_설정.md '검토 동시 요청 수', 기본 4).
       --follow <생성 PID>: 생성(3.2)과 함께 돌면서 생성된 요청부터 바로 검토한다 (3_0 이 '생성·검토 동시 진행' 때 사용).

[사용법]
    python 3_3_review.py                       # 생성됐지만 검토 안 된 것 전부
    python 3_3_review.py --max-requests 15     # 15개만 (오래된 대기분부터)
    python 3_3_review.py --reviewer claude_cli:opus
    python 3_3_review.py --reviewer none       # 검토 생략 (코드 검사만)
    python 3_3_review.py --redo-safety         # 이미 검토한 것 중 거절·안전 사유로 뺀 룰만 대체 모델로 다시 검토 → 3.4·1.4·3.5
                                               # (화면 2 '다시 만들기'. 처음 모델의 판정은 raw 원문에서 다시 읽음. 바뀐 결과는 _replaced 로)
"""

import argparse
import datetime as dt
import hashlib
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
# 2026-10-27: 화면 '검토에서 뺀 샘플' 의 '안전상 이유' 문구(0_0_run_ui.py EXCL_SAFETY)도 합쳤다 - 화면은 안전상 이유인데
# 다시 만들기 대상이 아닌 344개가 있었음 ("도움을 줄 수 없어 제외", "cannot safely validate", "무해하게 만드는 수정은 어렵" 등).
# 두 곳을 바꿀 때는 함께 맞춘다.
SAFETY_RE = re.compile(r"안전|지원할 수 없|도울 수 없|도와드릴 수 없|도움을 드릴 수 없|제공할 수 없|보존하거나|보완할 수 없|"
                       r"실행 가능한|구체화|악용|무기화|만들 수 없|재현.{0,12}(수 없|어렵)|"
                       r"can.?t (help|assist)|cannot (help|assist|provide)|unable to (help|assist)|I.?m sorry|"
                       # 화면 '안전상 이유' 문구 (0_0_run_ui.py EXCL_SAFETY)
                       r"위험|익스플로잇|무해|피해 없|재현 가능|운영 가능|보존(하거나|할 수 없|해 줄 수 없)|"
                       r"재현(하거나|할 수 없|용)|재구성할 수 없|보정(하지 않|할 수 없)|보강할 수 없|검증·정제할 수 없|"
                       r"도움(이 될 수|을 줄 수 없|을 드릴 수 없)|승인하거나|공격 (실행|코드|입력을 만들|요청이라|요청으로 (바꿔야|수정할 수 없))|"
                       r"페이로드를 재작성할 수 없|실행 페이로드|명령 (실행|주입|치환) (요청이라|페이로드|기능|공격)|콜백을 유도|다운로드 후 실행|"
                       r"서비스 중단|설정 손실|DoS 트리거|(노출|권한 상승)을 유발|열거 기능을 유지|하드코(딩|드)된 (자격|인증)|파괴적 명령|"
                       r"exploit|weapon|safety|safely|unsafe|harm|can(not|'t) (assist|help|provide|preserve|keep|convert)|refus", re.I)
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
        rule = {k: v for k, v in r.items() if k != "_mask"}       # _mask: 3.1 의 마스킹 기록 (로컬 전용, 보내지 않음)
        items.append({"rule": rule, "generated": {"attack_type": g["attack_type"], "samples": samples},
                      "local_check": checks})
    return items


def ask(name, tag, items, cfg, providers, run, prompt, rev_dir, log, usage, key):
    """검토 모델 호출 → ({룰 id: 판정}, 마지막 거절 글 앞부분 ('' = 거절 없음)).
    판정이 빠진 룰(거절·응답 중단)만 같은 모델로 '재시도 횟수' 까지 다시 묻는다 (2026-10-27~, 예전에는 판정이 하나라도 읽히면
    끝내 빠진 룰은 검토 없이 통과됐다). 원문은 시도마다 raw/<이름>.<tag><n>.txt"""
    got, refusal, todo = {}, "", list(items)
    tries = C.run_int(run, "재시도 횟수", 3)
    for attempt in range(1, tries + 1):
        user = (f"다음은 검토할 룰 {len(todo)}개다 (JSON Lines). 지시문의 출력 형식대로 룰마다 JSON 한 줄씩만 출력하라.\n\n"
                + "\n".join(json.dumps(i, ensure_ascii=False) for i in todo))
        ids = {i["rule"]["id"] for i in todo}
        text, u = C.call_llm(cfg, providers, prompt, user, run, log)
        C.add_usage(usage, key, u)
        (rev_dir / "raw").mkdir(parents=True, exist_ok=True)
        (rev_dir / "raw" / f"{name}.{tag}{attempt}.txt").write_text(text, encoding="utf-8")   # 시도마다 원문 보관
        new = {o["id"]: o for o in C.parse_json_lines(text) if isinstance(o, dict) and o.get("id") in ids}
        got.update(new)
        todo = [i for i in todo if i["rule"]["id"] not in got]
        if not todo:
            return got, refusal
        if is_refusal(text):
            refusal = " ".join(text.split())[:200]
        nxt = ("빠진 룰만 다시 요청" if attempt < tries else "다음 실행 때 다시" if not got and not refusal
               else "판정이 없는 룰은 '검토에서 뺀 샘플'(응답 거절, 다시 만들기 대상)로")
        log.warning("  %s: %s (%d번째)%s - %s", name,
                    "검토 응답에서 JSON 을 찾지 못함" if not new else f"검토 응답에 룰 {len(todo)}개의 판정이 없음", attempt,
                    f" - 거절: {' '.join(text.split())[:90]}" if is_refusal(text) else "", nxt)
    return got, refusal


def type_counts(generated, rejected):
    """공격 유형별 [생성 공격 샘플, 검토 모델이 뺀 공격 샘플, 그중 안전 사유(SAFETY_RE), 판정 거절로 빠진 공격 샘플].
    유형은 생성 결과의 attack_type (화면 '검토에서 뺀 샘플' 과 같음). 3.5 가 모아 유형별 거부 비율이 기준을 넘으면 경고 (R01)."""
    by_id = {g.get("id"): g for g in generated if isinstance(g, dict)}
    out = {}
    for g in by_id.values():
        c = out.setdefault(g.get("attack_type") or "other", [0, 0, 0, 0])
        c[0] += sum(1 for s in g.get("samples", []) if isinstance(s, dict) and s.get("label") == "anomalies")
    for x in rejected:
        g = by_id.get(x.get("id")) or {}
        smp, i = g.get("samples") or [], x.get("i")
        s = smp[i] if isinstance(i, int) and 0 <= i < len(smp) else None
        if not isinstance(s, dict) or s.get("label") != "anomalies":
            continue
        c = out.setdefault(g.get("attack_type") or "other", [0, 0, 0, 0])
        if x.get("stage") == "검토":
            c[1] += 1
            c[2] += bool(SAFETY_RE.search(str(x.get("reason", ""))))
        elif x.get("stage") == "검토 거절":
            c[3] += 1
    return out


def provenance(cfg, prompt, rev_dir, name):
    """meta.json 의 생성 이력: 생성 모델(3.2 meta), 검토 모델·설정, 대체 모델(다시 만들기 때), 검토 지시문 지문.
    3_2_generate.provenance 와 같은 모양에 generator·substitute 를 더함."""
    try:
        gen = json.loads((rev_dir.parent / C.GEN_DIR / f"{name}.meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        gen = {}
    fb = (cfg or {}).get("fallback")
    return {"generator": gen.get("generator"), "generator_prompt_sha256": (gen.get("provenance") or {}).get("prompt_sha256"),
            "model": f"{cfg['provider']}:{cfg['model']}" if cfg else None,
            "substitute": f"{fb['provider']}:{fb['model']}" if fb else None,
            "temperature": (cfg or {}).get("temperature"), "max_tokens": (cfg or {}).get("max_tokens"),
            "effort": (cfg or {}).get("effort") or (cfg or {}).get("reasoning_effort"),
            "prompt_file": C.REV_PROMPT_FILE.name if cfg else None,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest() if cfg and prompt else None}


def _total(usage):
    out = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
    for k in ("reviewer_primary", "reviewer_fallback"):
        for f in out:
            out[f] += (usage.get(k) or {}).get(f, 0)
    return out


def review(name, rules, generated, cfg, providers, run, prompt, rev_dir, log, primary=None, fb_given=None, fb_last=None,
           extra=None):
    """요청 하나 검토 → _3_reviewed/<이름>.jsonl·.rejected.jsonl·.meta.json.
    primary: (--redo-safety) 처음 모델에서 이미 받은 판정 {룰 id: 판정} (다시 묻지 않음), 그때의 토큰은 extra["usage"]
    fb_given: (--redo-safety) 대체 모델로 미리 받은 판정 (있으면 그 룰은 대체 모델을 다시 부르지 않음)
    fb_last: (--redo-safety) 대체 모델에 이미 '재시도 횟수' 만큼 물었음 (None 이 아니면 다시 묻지 않고, 판정이 없는 룰은
             '검토 거절' 로 남김 → 다음 다시 만들기 대상도 아님) - 값은 대체 모델의 마지막 거절 글"""
    by_id = {g["id"]: g for g in generated}
    usage, verdicts, fallback, refused = {}, {}, None, None     # refused: 판정을 받지 못한 룰 (거절·응답 중단)
    for k, src in (("reviewer_primary", "usage"), ("reviewer_fallback", "fb_usage")):
        if extra and extra.get(src):
            usage[k] = dict(extra[src])
    started = time.time()
    if cfg is not None:
        items = review_items(rules, by_id)
        if items:
            fb = cfg.get("fallback")                        # '다시 만들기'(--redo-safety)에서만 있음
            last = ""
            if primary is None:
                verdicts, last = ask(name, "try", items, cfg, providers, run, prompt, rev_dir, log, usage, "reviewer_primary")
            else:
                verdicts = dict(primary)
            # 대체 검토: 처음 모델이 판정을 주지 않았거나(거절) '안전' 사유로 뺀 샘플이 있는 룰만
            need = [i for i in items if i["rule"]["id"] not in verdicts or safety_dropped(verdicts[i["rule"]["id"]])]
            if fb is not None and need:
                why = {i["rule"]["id"]: ("응답 거절" if i["rule"]["id"] not in verdicts else "안전 사유로 뺀 샘플") for i in need}
                got = {k: v for k, v in (fb_given or {}).items() if k in why}
                rest = [i for i in need if i["rule"]["id"] not in got]
                if rest and fb_last is None:
                    v2, last2 = ask(name, "fb.try", rest, fb, providers, run, prompt, rev_dir, log, usage, "reviewer_fallback")
                    got.update(v2)
                    last = last2 or last
                elif fb_last:
                    last = fb_last
                verdicts.update(got)
                fallback = {"model": f"{fb['provider']}:{fb['model']}", "rules": why, "answered": sorted(got)}
            missing = [i["rule"]["id"] for i in items if i["rule"]["id"] not in verdicts]
            if missing and not verdicts and not last:       # 판정을 하나도 못 읽었고 거절도 아님 (깨진 응답) → 다음 실행 때
                raise C.LLMError(f"{name}: 검토 응답을 해석하지 못함 (다음 실행 때 다시 시도)")
            if missing:                                     # 거절·응답 중단 → '검토 거절' (화면 '검토에서 뺀 샘플', 다시 만들기 대상)
                refused = {"rules": missing, "text": last}
        usage["reviewer"] = _total(usage)

    policy = run.get("로컬 검사 실패 처리", "drop")
    unknown_policy = run.get("판정 불가 처리", "drop")
    rejected, final = [], []
    for r in rules:
        g = by_id.get(r["id"])
        if not g:
            continue
        if refused and r["id"] in refused["rules"]:         # 판정을 받지 못한 룰: 샘플을 모두 '검토 거절' 로 (통과시키지 않음)
            why = (f"검토 모델이 이 룰을 판정하지 않음 (거절·응답 중단, {C.run_int(run, '재시도 횟수', 3)}번 요청)"
                   + (f": {refused['text']}" if refused["text"] else ""))
            rejected += [{"id": r["id"], "i": i, "stage": "검토 거절", "reason": why} for i in range(len(g.get("samples", [])))]
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
            "fixed": sum(s["review"] == "fix" for s in flat), "rejected": len(rejected), "usage": usage,
            "provenance": provenance(cfg, prompt, rev_dir, name), "review_by_type": type_counts(generated, rejected)}
    if fallback:
        meta["fallback"] = fallback
    if refused:
        meta["refused"] = refused
    if extra and extra.get("redo"):
        meta["redo"] = extra["redo"]
    (rev_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    fb_txt = ""
    if fallback:
        kinds = sorted(set(fallback["rules"].values()))
        fb_txt = (f" · 대체 검토 {fallback['model']} 룰 {len(fallback['answered'])}/{len(fallback['rules'])}개"
                  f" ({', '.join(kinds)})")
    if refused:
        fb_txt += f" · 응답 거절 룰 {len(refused['rules'])}개 → 검토에서 뺀 샘플"
    # 토큰은 두 모델 합계 (화면·진행 현황이 이 줄의 첫 토큰 값을 읽음)
    log.info("  %s: 샘플 %d개 남음 (anomalies %d, normal %d, 수정 %d), 제외 %d개, 토큰 %s%s", name, meta["samples"],
             meta["anomalies"], meta["normal"], meta["fixed"], len(rejected), usage.get("reviewer"), fb_txt)


def _first_verdicts(raws, ids):
    """처음 모델의 원문(try1, try2 …)에서 실제로 쓴 판정 = 룰마다 처음으로 해석된 판정 (빠진 룰만 다시 물으므로 여러 응답에
    나뉠 수 있음. 2026-10-27 전 실행은 처음으로 해석된 응답 하나뿐이라 결과가 같다)."""
    got = {}
    for raw in sorted(raws, key=lambda p: int(re.search(r"\.try(\d+)\.txt$", p.name).group(1))):
        for o in C.parse_json_lines(raw.read_text(encoding="utf-8", errors="replace")):
            if isinstance(o, dict) and o.get("id") in ids:
                got.setdefault(o["id"], o)
    return got


def _fallback_verdicts(rv, name, ids):
    """예전 대체 검토에서 받은 대체 모델 판정 (raw/<이름>.fb.try<n>.txt·.fb2.try<n>.txt…, 미리 받아 다시 쓴 것은 .fb.reuse.txt)."""
    got = {}
    for raw in sorted((rv / "raw").glob(f"{name}.fb*.txt")):
        for o in C.parse_json_lines(raw.read_text(encoding="utf-8", errors="replace")):
            if isinstance(o, dict) and o.get("id") in ids:
                got.setdefault(o["id"], o)
    return got


def redo_candidates(bases, fb_model):
    """이미 검토한 요청 중 처음 모델이 거절했거나 '안전' 사유로 뺀 룰이 있고, 그 룰을 아직 대체 모델에 보내지 않은 것
    → [(날짜 폴더, 이름, 옛 meta, 처음 판정)]. 대체 모델에 이미 보낸 룰은 다시 보내지 않는다 (대체 모델도 거절·제외한 샘플은
    제외로 남음). 2026-10-27~: 대체 검토를 한 요청도 그때 보내지 않은 룰(안전 문구를 넓혀 새로 걸린 룰 등)이 있으면 대상."""
    out = []
    for base in bases:
        rv = base / C.REV_DIR
        for m in sorted(rv.glob("*.meta.json")) if rv.is_dir() else []:
            try:
                meta = json.loads(m.read_text(encoding="utf-8"))
            except ValueError:
                continue
            name = meta.get("request") or m.name[:-len(".meta.json")]
            if meta.get("reviewer") in (None, fb_model):
                continue
            gen_p, req_p = base / C.GEN_DIR / f"{name}.jsonl", base / C.REQ_DIR / f"{name}.jsonl"
            if not (gen_p.exists() and req_p.exists()):
                continue
            ids = {g["id"] for g in C.read_jsonl(gen_p)}
            prim = _first_verdicts(sorted((rv / "raw").glob(f"{name}.try*.txt")), ids)
            sent = set((meta.get("fallback") or {}).get("rules") or {})        # 예전 대체 검토에서 보낸 룰
            if any((rid not in prim or safety_dropped(prim[rid])) and rid not in sent for rid in ids):
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
        # 예전 대체 검토에서 이미 보낸 룰: 그때 받은 판정은 그대로 다시 쓰고 다시 묻지 않는다 (그때 거절한 룰은 '검토 거절' 로 남음)
        sent = set((old.get("fallback") or {}).get("rules") or {})
        given = _fallback_verdicts(rev_dir, name, sent) if sent else None
        fb_usage = (old.get("usage") or {}).get("reviewer_fallback") if sent else None
        if reuse and (reuse / base.name / "raw" / f"{name}.try1.txt").exists():      # 미리 대체 모델로 검토한 원문 (시험)
            src = reuse / base.name / "raw" / f"{name}.try1.txt"
            given = dict(given or {}, **{o["id"]: o for o in C.parse_json_lines(src.read_text(encoding="utf-8"))
                                         if isinstance(o, dict) and o.get("id")})
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
            rest = [i for i in need if i["rule"]["id"] not in (given or {}) and i["rule"]["id"] not in sent]
            last2 = ""
            if rest:
                # 원문 파일 이름: 처음 다시 만들기는 fb.try<n>, 그 뒤는 fb2.try<n>… (예전 대체 모델 원문을 덮어쓰지 않게)
                tag = f"fb{len(list((rev_dir / 'raw').glob(f'{name}.fb*.try1.txt'))) + 1}.try" if sent else "fb.try"
                v2, last2 = ask(name, tag, rest, fb, providers, run, prompt, rev_dir, log, usage, "reviewer_fallback")
                if not v2 and not last2:                # 깨진 응답(거절 아님) → 옛 결과 그대로, 다음 '다시 만들기' 때 다시
                    raise C.LLMError(f"{name}: 대체 모델 응답을 해석하지 못함")
                # 대체 모델도 거절한 룰은 '검토 거절' 로 남기고 다시 만들기 대상에서도 뺀다 (계속 안 만들어지는 샘플은 제외)
                given = dict(given or {}, **v2)
                fb_usage = usage["reviewer_fallback"] if not fb_usage else {
                    k: fb_usage.get(k, 0) + usage["reviewer_fallback"].get(k, 0) for k in ("calls", "input_tokens", "output_tokens")}
            set_aside(base, name)
            # 처음 판정은 그때의 검토 모델(옛 meta) 것이므로 meta 의 reviewer 도 그 모델로 남긴다
            p_prov, _, p_model = str(old.get("reviewer") or f"{cfg['provider']}:{cfg['model']}").partition(":")
            review(name, rules, gens, dict(cfg, provider=p_prov, model=p_model), providers, run, prompt, rev_dir, log,
                   primary=prim, fb_given=given, fb_last=last2,
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
    # 추론 강도: openai 는 reasoning_effort, claude_cli 는 --effort (검토 대체 모델도 같은 값, C.set_effort).
    # 낮추면 빠르고 토큰이 적지만 판단이 얕아질 수 있음
    effort = run.get("검토 추론 강도", "").strip()
    C.set_effort(cfg, effort, log)
    fb_name = run.get("검토 대체 모델", "").strip()       # 거절·안전 사유로 뺀 룰을 다시 검토하는 모델 (비우면 안 함)
    # 검토 대체 모델은 '다시 만들기'(--redo-safety)에서만 쓴다 (2026-10-27~: 보통 실행에서는 검토 모델과 달라도 쓰지 않음).
    # 처음 판정은 예전 검토 모델의 원문이라 지금 검토 모델과 같아도 쓴다
    if args.redo_safety and cfg is not None and fb_name not in ("", "-", "none"):
        fb = C.role_config(roles, "검토", fb_name)
        if fb is not None:
            C.set_effort(fb, effort, log)
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
    who = f"{cfg['provider']}:{cfg['model']}{C.effort_label(cfg)}" if cfg else "없음 (코드 검사만)"
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
