"""[3.2 데이터 생성] 요청 파일마다 생성 모델(기본 UpStage Solar)을 호출해 룰별 anomalies/normal 샘플을 만든다.  (3_0 이 실행)

[전체 흐름]  _1_requests/<이름>.jsonl 하나마다 (이미 _2_generated 에 결과가 있으면 건너뜀)
  순서: 이름순 = 배치 순서 (ET → Talos → 사용자 룰). --custom-first 면 사용자 룰(custom)이 든 요청부터
  1. 지시문(LLM_프롬프트.md, 샘플 수는 LLM_설정.md 값으로 채움) + 룰들 → 생성 모델 호출
     룰마다 정규식 조건에 실제로 맞는 짧은 예시(regex_examples, 코드가 만들고 확인)를 붙인다 (LLM_설정.md '정규식 예시')
  2. 응답에서 룰별 JSON 을 꺼낸다. JSON 이 없거나, 응답이 출력 한도에서 잘려(반복 루프) 빠진 룰이 있으면
     **그 룰만** '재시도 횟수' 안에서 다시 요청한다 (반복을 {{REPEAT:문자열:횟수}} 로 쓰라는 주의를 붙여서)
     재시도까지 끝나도 끝까지 읽힌 룰이 하나도 없으면(요청 실패), 응답이 룰 객체 중간에서 끊긴 룰 중 끝까지 온
     공격 샘플이 요청 수의 절반 이상인 룰은 그 샘플만 쓴다 (meta 의 rules_salvaged. 2026-10-30~)
  3. 긴 반복 표기를 펼치고, 샘플마다 코드 검사(local_check) 결과를 붙인다 → 다음 단계 검토 모델이 참고
  4. _2_generated/<이름>.jsonl (+ .meta.json: 토큰 사용량·빠진 룰, raw/<이름>.txt: 응답 원문)
     meta.json 에는 생성 이력(provenance: 모델·temperature·max_tokens·추론 강도·지시문 sha256)과 참고 예시를 보낸 수
     (existing_examples_sent)·마스킹으로 바꾼 값 수(masked_values)도 남긴다. 원래 값은 남기지 않는다.

[속도] 요청 여러 개를 동시에 보낸다 (LLM_설정.md '생성 동시 요청 수', 기본 3).
[비용 안전장치] 기본은 요청 1개만 처리 (--max-requests 0 = 남은 것 전부). --dry-run 은 호출하지 않는다.

[사용법]
    python 3_2_generate.py --dry-run
    python 3_2_generate.py                          # 요청 1개
    python 3_2_generate.py --max-requests 20
    python 3_2_generate.py --max-requests 70 --custom-first   # 사용자 룰이 든 요청부터
    python 3_2_generate.py --generator upstage:solar-pro2
    python 3_2_generate.py --generator mock         # 키 없이 흐름 시험
    python 3_2_generate.py --reparse pick_b0000_a8bcae_c8 --dry-run   # 해석 실패한 요청: 저장된 원문을 지금 해석기로 (호출 없음)
"""

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_spec = importlib.util.spec_from_file_location("llm_common", Path(__file__).resolve().parent / "3_0a_common.py")
C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)


def with_examples(rule):
    """정규식 조건마다 코드가 만든 '그 정규식에 실제로 맞는 예시'를 붙인다 (생성 모델이 정규식을 잘못 읽지 않게)."""
    ex = C.rule_regex_examples(rule)
    return dict(rule, regex_examples=ex) if ex else rule


def user_message(rules, n_a, n_n, hint="", examples=True):
    # 샘플 수를 요청 메시지에 다시 못박는다 (지시문의 출력 예시 개수를 그대로 따라 하는 모델이 있음)
    rules = [{k: v for k, v in r.items() if k != "_mask"} for r in rules]     # _mask: 3.1 의 마스킹 기록 (로컬 전용)
    if examples:
        rules = [with_examples(r) for r in rules]
    return ((hint + "\n\n" if hint else "") +
            f"다음은 룰 {len(rules)}개다 (JSON Lines).\n"
            f"- 룰마다 반드시 anomalies {n_a}개, normal {n_n}개 (룰당 총 {n_a + n_n}개)를 만든다. "
            f"지시문의 출력 예시는 형식만 보여 주는 것이며 개수는 이 수를 따른다.\n"
            f"- anomalies {n_a}개는 서로 다른 공격 변형이어야 한다 (인코딩·파라미터·경로·값을 바꿔서).\n"
            f"- 교과서 예제처럼 짧고 공격만 눈에 띄는 요청은 쓰지 않는다. 실제 공격 트래픽처럼 정상 파라미터·헤더 사이에 공격을 넣고, "
            f"User-Agent 는 잘리지 않은 실제 문자열을 쓴다 (지시문 생성 규칙 3·8번).\n"
            f"- 같은 문자열을 20번 넘게 이어 써야 하면(버퍼 오버플로의 AAAA…, 긴 ../ 나열, %00 반복) 직접 쓰지 말고 "
            f"{{{{REPEAT:문자열:횟수}}}} 로 쓴다 (지시문 생성 규칙 11번).\n"
            + (f"- regex_examples 는 코드가 만든, 그 정규식에 실제로 맞는 최소 형태다. 흔한 형태가 정규식에 안 맞는 경우가 많으니"
               f"(예: `<.+\\/script` 는 보통의 `</script>` 와 불일치) anomalies 의 정규식 부분은 이 예시의 구조를 끝 글자까지 따르고 "
               f"(note 가 있으면 따른다), 그대로 복사하지 말고 실제 공격 안에 넣는다.\n" if examples else "") +
            f"- 출력은 룰마다 JSON 객체 하나씩. 설명 문장 없이.\n\n"
            + "\n".join(json.dumps(r, ensure_ascii=False) for r in rules))


def provenance(cfg, prompt, prompt_file):
    """meta.json 의 생성 이력: 실제로 부른 모델·설정과 지시문 지문 (지시문이 바뀌면 지문이 바뀜, 재현·감사용).
    3_3_review.py 의 provenance 와 같은 모양."""
    return {"model": f"{cfg['provider']}:{cfg['model']}", "temperature": cfg.get("temperature"),
            "max_tokens": cfg.get("max_tokens"), "effort": cfg.get("effort") or cfg.get("reasoning_effort"),
            "prompt_file": prompt_file.name, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()}


def parse_rules(text, todo, got):
    """응답 하나에서 아직 없는 룰(todo)의 객체를 got 에 더한다 → 남은 룰."""
    want = {r["id"] for r in todo}
    got.update({o["id"]: o for o in C.parse_json_lines(text)
                if isinstance(o, dict) and o.get("id") in want and isinstance(o.get("samples"), list)})
    return [r for r in todo if r["id"] not in got]


def salvage(rules, got, texts, run):
    """재시도까지 모두 끝나도 끝까지 읽힌 룰이 하나도 없을 때(요청 실패가 될 때)만: 응답이 룰 객체 중간에서 끊겼으면
    끝까지 온 샘플만 꺼내 got 에 넣는다. 끝까지 온 공격 샘플이 요청 수의 절반 이상인 룰만 (여러 응답 중 가장 많은 것).
    solar-pro4 가 본문의 \\x00·%00 나열 중 응답을 끊는 경우 (2026-10-30). → 꺼낸 룰 id 목록"""
    if got:
        return []
    need = max(1, C.run_int(run, "생성 anomalies 수", 10) // 2)
    n_a = lambda c: sum(1 for s in c["samples"] if isinstance(s, dict) and s.get("label") == "anomalies")
    out = []
    for r in rules:
        cands = [c for c in (C.salvage_rule(t, r["id"]) for t in texts) if c and n_a(c) >= need]
        if cands:
            got[r["id"]] = max(cands, key=lambda c: (n_a(c), len(c["samples"])))
            out.append(r["id"])
    return out


def generate(name, rules, cfg, providers, run, prompt, gen_dir, log):
    n_a, n_n = C.run_int(run, "생성 anomalies 수", 10), C.run_int(run, "생성 normal 수", 5)
    tries = C.run_int(run, "재시도 횟수", 3)
    examples = C.run_yes(run, "정규식 예시")             # LLM_설정.md (기본 예)
    usage, got, hint, cuts, texts = {}, {}, "", 0, []
    todo = list(rules)
    started = time.time()
    for attempt in range(1, tries + 1):
        text, u = C.call_llm(cfg, providers, prompt, user_message(todo, n_a, n_n, hint, examples), run, log)
        C.add_usage(usage, "generator", u)
        (gen_dir / "raw").mkdir(parents=True, exist_ok=True)
        (gen_dir / "raw" / f"{name}.try{attempt}.txt").write_text(text, encoding="utf-8")   # 시도마다 원문 보관
        texts.append(text)
        todo = parse_rules(text, todo, got)
        if not todo:
            break
        unit = C.repeat_tail(text)               # 반복 루프로 출력 한도까지 간 응답이면 반복 단위
        cut = unit is not None or u.get("finish_reason") == "length"
        if got and not cut:
            break      # 응답은 끝까지 왔는데 몇 룰만 빠뜨림 → 빠진 룰은 다음 실행 때 새 요청으로 (3.1 이 rules_missing 을 다시 넣음)
        if cut:
            cuts += 1
            hint = ((f"[주의] 직전 응답이 '{unit}' 를 출력 한도까지 반복하다 잘렸다. " if unit else
                     "[주의] 직전 응답이 출력 한도에서 잘렸다. ") +
                    "같은 문자열을 20번 넘게 이어 써야 하면 {{REPEAT:문자열:횟수}} 로 쓰고, 직접 길게 반복하지 않는다.")
        if attempt < tries:
            why = f" - '{unit}' 반복 루프" if unit else (" - 출력 한도에서 잘림" if cut else "")
            if not got:
                log.warning("  %s: 응답에서 JSON 을 찾지 못함 (%d번째)%s - 다시 요청", name, attempt, why)
            else:
                log.warning("  %s: 룰 %d개가 잘려 빠짐 (%d번째)%s - 그 룰만 다시 요청", name, len(todo), attempt, why)
    write_result(name, rules, got, texts, run, gen_dir, log,
                 {"seconds": round(time.time() - started, 1), "generator": f"{cfg['provider']}:{cfg['model']}",
                  "cut_responses": cuts, "regex_examples": examples, "usage": usage,
                  "provenance": provenance(cfg, prompt, C.GEN_PROMPT_FILE)})


def write_result(name, rules, got, texts, run, gen_dir, log, info):
    """해석한 룰 객체 → 샘플 정리·코드 검사 → _2_generated/<이름>.jsonl + .meta.json (생성과 --reparse 공통).
    info: meta 에 넣을 seconds·generator·cut_responses·regex_examples·usage·provenance (+ reparsed)."""
    realistic = C.run_yes(run, "헤더 현실화")
    salvaged = salvage(rules, got, texts, run)
    if salvaged:
        log.warning("  %s: 응답이 모두 중간에서 끊김 - 끝까지 온 샘플만 쓰는 룰 %d개 (%s)", name, len(salvaged), ", ".join(salvaged))
    if not got:   # 결과 파일을 만들지 않아야 다음 실행 때 다시 시도된다
        raise C.LLMError(f"{name}: 생성 응답을 해석하지 못함 (raw/{name}.try*.txt 확인, 다음 실행 때 다시 시도)")
    out, missing = [], []
    for r in rules:
        g = got.get(r["id"])
        if not g:
            missing.append(r["id"])
            continue
        samples = []
        for s in g["samples"]:
            if isinstance(s, dict) and isinstance(s.get("http"), str):
                s = dict(s, http=C.expand_repeats(s["http"]))      # {{REPEAT:A:2000}} → 실제 반복
            if C.valid_sample(s):
                s = C.repair_sample(s, r)           # URI 의 공백 등으로 깨진 요청 줄 → %20
                if realistic:                     # 잘린 UA·빠진 기본 헤더·틀린 Content-Length → 실제 값
                    s = C.realistic_sample(s, r)
                # 큰 반복 (…){100,}: 파트·빈 줄 반복은 3.3 merge 가 코드로 채우므로 채운 샘플로 검사 (검토 모델이 파트 수로 빼지 않게)
                big = C.repeat_parts(s, r)
                chk = C.local_check(big, r)
                if big is not s:
                    chk = dict(chk, note="반복 부분(파트·빈 줄)은 코드가 기준 수 이상으로 늘림 (3.3) - 반복 수 때문에 drop·fix 하지 않음")
                s = dict(s, local_check=chk)
            samples.append(s)
        out.append({"id": r["id"], "attack_type": g.get("attack_type") or r["attack_type"],
                    "attack_type_note": g.get("attack_type_note"), "samples": samples})
    C.write_jsonl(gen_dir / f"{name}.jsonl", out)
    flat = [s for o in out for s in o["samples"] if isinstance(s, dict)]
    meta = {"request": name, "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
            "seconds": info["seconds"], "generator": info["generator"],
            "rules": len(rules), "rules_done": len(out), "rules_missing": missing, "rules_salvaged": salvaged,
            "cut_responses": info["cut_responses"], "regex_examples": info["regex_examples"],
            "samples": len(flat), "local_check": {k: sum(1 for s in flat if s.get("local_check", {}).get("result") == k)
                                                  for k in ("ok", "fail", "unknown")},
            "usage": info["usage"], "provenance": info["provenance"]}
    if info.get("reparsed"):
        meta["reparsed"] = info["reparsed"]
    # 참고 예시: 보낸 수와 마스킹으로 바꾼 값 수만 (원래 값은 남기지 않음). masked_values 가 None 이면 마스킹 기록이 없는
    # 요청(마스킹 꺼짐, 또는 마스킹 전에 만든 요청 파일)
    sent = sum(len(r.get("existing_examples") or []) for r in rules)
    infos = [r["_mask"] for r in rules if isinstance(r.get("_mask"), dict)]
    meta["existing_examples_sent"] = sent
    meta["masked_values"] = (sum(i.get("values", 0) for i in infos)
                             if infos and all(i.get("on") for i in infos) else (0 if not sent else None))
    (gen_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("  %s: 룰 %d/%d, 샘플 %d개 (코드 검사 ok %d / fail %d), 토큰 %s", name, len(out), len(rules),
             len(flat), meta["local_check"]["ok"], meta["local_check"]["fail"], info["usage"].get("generator"))


def reparse(names, req_dir, gen_dir, cfg, run, dry_run, log):
    """--reparse: LLM 을 부르지 않고 저장된 응답 원문(raw/<이름>.try*.txt)을 지금 해석기로 다시 읽어 결과를 만든다
    (해석 실패한 요청용, 토큰 0). 시도 순서대로 룰마다 처음 읽힌 객체 (생성 때와 같음), 그 뒤 끊긴 룰의 샘플 꺼내기.
    이미 결과가 있는 요청은 건너뛴다. meta 의 generator·provenance 는 지금 설정 값 (원문을 만든 모델과 같은지 확인)."""
    prompt = C.fill_prompt(C.GEN_PROMPT_FILE.read_text(encoding="utf-8"), run)
    code = 0
    for name in names:
        req = req_dir / f"{name}.jsonl"
        raws = sorted((gen_dir / "raw").glob(f"{name}.try*.txt"), key=lambda p: int(p.name.rsplit(".try", 1)[1][:-4]))
        if not req.exists() or not raws:
            log.error("  %s: 요청 파일 또는 응답 원문(raw/%s.try*.txt)이 없음", name, name)
            code = 1
            continue
        if (gen_dir / req.name).exists():
            log.info("  %s: 이미 결과가 있어 건너뜀", name)
            continue
        rules = C.read_jsonl(req)
        texts = [p.read_text(encoding="utf-8") for p in raws]
        got, todo = {}, list(rules)
        for text in texts:
            todo = parse_rules(text, todo, got)
        if dry_run:
            cut = salvage(rules, dict(got), texts, run)
            log.info("  %s: 원문 %d개 → 룰 %d/%d개 읽힘%s (dry-run: 쓰지 않음)", name, len(texts), len(got), len(rules),
                     f", 끊긴 룰에서 샘플 꺼냄 {len(cut)}개" if cut else "")
            continue
        try:
            write_result(name, rules, got, texts, run, gen_dir, log,
                         {"seconds": None, "generator": f"{cfg['provider']}:{cfg['model']}",      # 시간·토큰 평균에서 빠지게
                          "cut_responses": sum(C.repeat_tail(t) is not None for t in texts),
                          "regex_examples": C.run_yes(run, "정규식 예시"), "usage": {},
                          "provenance": provenance(cfg, prompt, C.GEN_PROMPT_FILE),
                          "reparsed": {"at": dt.datetime.now().isoformat(timespec="seconds"), "raw": [p.name for p in raws],
                                       "raw_saved_at": dt.datetime.fromtimestamp(max(p.stat().st_mtime for p in raws))
                                       .isoformat(timespec="seconds"),
                                       "note": "저장된 응답 원문을 다시 해석 (LLM 호출 없음, 원래 호출 토큰은 기록 없음)"}})
        except C.LLMError as e:
            log.error("  %s: %s", name, e)
            code = 1
    return code


FAIL_LIMIT = 2      # 응답 해석 실패(재시도까지 모두 실패)가 이 횟수만큼 쌓인 요청은 자동으로 건너뜀


def fail_count(gen_dir, name):
    """요청의 응답 해석 실패 횟수 (_2_generated/<이름>.fail.json)."""
    try:
        return json.loads((gen_dir / f"{name}.fail.json").read_text(encoding="utf-8")).get("fails", 0)
    except (OSError, ValueError):
        return 0


def add_fail(gen_dir, name, error):
    gen_dir.mkdir(parents=True, exist_ok=True)
    n = fail_count(gen_dir, name) + 1
    (gen_dir / f"{name}.fail.json").write_text(json.dumps(
        {"request": name, "fails": n, "last_error": error[:300], "at": dt.datetime.now().isoformat(timespec="seconds")},
        ensure_ascii=False), encoding="utf-8")


def custom_first(todo, log):
    """사용자 룰(id 가 custom: 으로 시작)이 든 요청을 앞으로 옮긴다 (--custom-first). 기본 순서(이름순)는
    배치 순서(ET → Talos → 사용자 룰)라 사용자 룰이 맨 뒤에 온다. 앞·뒤 무리 안의 순서는 그대로."""
    n = {r: sum(str(x.get("id", "")).startswith("custom:") for x in C.read_jsonl(r)) for r in todo}
    first = [r for r in todo if n[r]]
    log.info("[3.2] 순서: 사용자 룰 먼저 - 남은 요청 %d개 중 사용자 룰이 든 요청 %d개 (룰 %d개)를 앞으로%s", len(todo),
             len(first), sum(n.values()), "" if first else " (없어서 기본 순서 그대로)")
    return first + [r for r in todo if not n[r]]


def main(argv=None):
    ap = argparse.ArgumentParser(description="생성 모델로 샘플 생성")
    ap.add_argument("--date", help="날짜 폴더 (기본: 05_llm_input 의 가장 최근)")
    ap.add_argument("--set", choices=["delta", "all", "pick", "any"], default="any",
                    help="이 종류의 요청만 (기본 전부, pick = 3_1 --rules 로 만든 지정 룰 요청)")
    ap.add_argument("--max-requests", type=int, default=1, help="처리할 요청 수 (기본 1, 0 = 남은 것 전부)")
    ap.add_argument("--requests", help="특정 요청 이름만 (쉼표 구분, 예: all_b0001_c1)")
    ap.add_argument("--generator", help="생성 모델 'provider:model' (LLM_설정.md 보다 우선)")
    ap.add_argument("--count-pending", action="store_true",
                    help="검토 대기 중인 생성 결과도 --max-requests 에 포함 (3_0 이 사용)")
    ap.add_argument("--pending-all", action="store_true",
                    help="검토 대기분을 모든 날짜 폴더에서 센다 (3.3 이 모든 날짜를 처리할 때, 3_0 이 사용)")
    ap.add_argument("--custom-first", action="store_true",
                    help="사용자 룰(custom)이 든 요청부터 (기본: 이름순 = 배치 순서 ET → Talos → 사용자 룰)")
    ap.add_argument("--force", action="store_true", help="이미 생성한 요청·실패가 반복된 요청도 다시")
    ap.add_argument("--reparse", metavar="NAMES",
                    help="LLM 을 부르지 않고 저장된 응답 원문(raw/<이름>.try*.txt)을 지금 해석기로 다시 읽어 결과를 만든다 "
                         "(쉼표 구분 요청 이름, 응답 해석에 실패한 요청용, 이미 결과가 있으면 건너뜀)")
    ap.add_argument("--dry-run", action="store_true", help="호출하지 않고 무엇을 할지만 출력")
    args = ap.parse_args(argv)
    log = C.setup_logging("3_2_generate")
    C.load_env()

    try:
        roles, providers, run = C.load_settings()
    except (ValueError, FileNotFoundError) as e:
        log.error("LLM 설정 읽기 실패: %s", e)
        return 1
    cfg = C.role_config(roles, "생성", args.generator)
    C.set_effort(cfg, run.get("생성 추론 강도", ""), log)   # openai: reasoning_effort, claude_cli: --effort (비우면 기본값)
    in_dir = C.latest_input_dir(args.date)
    if cfg is None or in_dir is None:
        log.error("생성 모델 설정 또는 05_llm_input 이 없습니다")
        return 1
    base = C.output_dir(in_dir.name)
    req_dir, gen_dir = base / C.REQ_DIR, base / C.GEN_DIR
    if args.reparse:
        names = [x.strip() for x in args.reparse.split(",") if x.strip()]
        log.info("[3.2] 저장된 응답 원문 다시 해석 %d개 (LLM 호출 없음)%s", len(names), " (dry-run: 쓰지 않음)" if args.dry_run else "")
        if not args.dry_run and not C.generation_guard(log, "3_2_generate --reparse"):
            return 2
        return reparse(names, req_dir, gen_dir, cfg, run, args.dry_run, log)
    reqs = sorted(req_dir.glob("*.jsonl")) if req_dir.is_dir() else []
    if args.set != "any":
        reqs = [r for r in reqs if r.stem.startswith(args.set + "_")]
    if args.requests:
        want = {x.strip() for x in args.requests.split(",")}
        todo = [r for r in reqs if r.stem in want]
    else:
        todo = [r for r in reqs if args.force or not (gen_dir / r.name).exists()]
        # 응답 해석 실패가 FAIL_LIMIT 번 반복된 요청은 건너뛴다 (매 실행마다 같은 요청에 비용을 쓰지 않게, --force 로 다시)
        failed = [r for r in todo if not args.force and fail_count(gen_dir, r.stem) >= FAIL_LIMIT]
        if failed:
            log.info("[3.2] 응답 해석 실패가 %d번 반복된 요청 %d개는 건너뜀 (%s 등, 다시 하려면 --force)",
                     FAIL_LIMIT, len(failed), failed[0].stem)
            todo = [r for r in todo if r not in failed]
        if args.custom_first:
            todo = custom_first(todo, log)
        if args.max_requests > 0:
            limit = args.max_requests
            if args.count_pending:   # 3_0 이 부를 때: 검토 못 한 생성 결과가 있으면 그만큼 새로 만들지 않는다
                # 3.3 이 모든 날짜 폴더의 검토 대기분을 처리할 때는(--pending-all) 같은 범위로 센다
                bases = C.output_dates() if args.pending_all else [base]
                pending = [g for b in bases for g in (b / C.GEN_DIR).glob("*.jsonl")
                           if not (b / C.REV_DIR / g.name).exists()]
                if pending:
                    log.info("[3.2] 검토 대기 중인 생성 결과 %d개 - 이번 생성 수에서 뺌", len(pending))
                limit = max(0, limit - len(pending))
            todo = todo[:limit]
    done = sum((gen_dir / r.name).exists() for r in reqs)
    workers = C.parallel_settings(run)[0]
    log.info("[3.2] 생성 %s:%s%s | 요청 %d개 중 완료 %d개 → 이번 %d개 (동시 최대 %d개씩, 속도 제한에 맞춰 자동 조절)%s", cfg["provider"],
             cfg["model"], C.effort_label(cfg), len(reqs), done, len(todo), workers, " (dry-run: 호출 안 함)" if args.dry_run else "")
    if todo and (args.dry_run or args.custom_first):        # 어떤 요청부터 하는지 (건수 확인·사용자 룰 먼저)
        log.info("[3.2] 처리 순서: %s%s", ", ".join(r.stem for r in todo[:3]),
                 f" … {todo[-1].stem} (모두 {len(todo)}개)" if len(todo) > 3 else "")
    if args.dry_run or not todo:
        return 0
    if not C.generation_guard(log, "3_2_generate"):
        return 2

    prompt = C.fill_prompt(C.GEN_PROMPT_FILE.read_text(encoding="utf-8"), run)
    cfg["throttle"] = C.Throttle(workers, log, "생성")       # 동시 요청 수: 4개부터 늘려 최대까지, 429 면 줄임
    gen_dir.mkdir(parents=True, exist_ok=True)
    fatal, stopped = threading.Event(), threading.Event()

    def one(r):
        """요청 하나 (스레드에서). 중지 요청·설정 오류 뒤에는 새로 시작하지 않는다 (이미 보낸 요청은 끝까지)."""
        if fatal.is_set():
            return
        if C.RT.stop_requested("generation"):
            stopped.set()
            return
        try:
            generate(r.stem, C.read_jsonl(r), cfg, providers, run, prompt, gen_dir, log)
        except C.LLMError as e:
            log.error("  %s 실패: %s", r.stem, e)
            if "해석하지 못함" in str(e):
                add_fail(gen_dir, r.stem, str(e))
            if C.is_fatal(e):                  # 설정·키·모델 이름·크레딧·Claude 로그인·구독 사용 한도
                if not fatal.is_set():
                    fatal.set()
                    log.error("설정·키·모델 이름·크레딧·사용 한도 문제라 나머지도 중단합니다 (다음 실행 때 이어서)")
        except Exception:                      # 예상 못 한 오류도 다른 요청은 계속
            log.exception("  %s 실패 (예상 못 한 오류)", r.stem)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gen") as ex:
        list(ex.map(one, todo))                # 순서대로 시작 (배치 순서, --custom-first 면 사용자 룰 먼저), 동시에 workers 개
    if stopped.is_set():
        log.info("[3.2] 중지 요청 - 여기서 멈춤 (남은 요청은 다음 실행 때)")
    return 1 if fatal.is_set() else 0


if __name__ == "__main__":
    sys.exit(main())
