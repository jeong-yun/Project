"""[3.2 데이터 생성] 요청 파일마다 생성 모델(기본 UpStage Solar)을 호출해 룰별 anomalies/normal 샘플을 만든다.  (3_0 이 실행)

[전체 흐름]  _1_requests/<이름>.jsonl 하나마다 (이미 _2_generated 에 결과가 있으면 건너뜀)
  1. 지시문(LLM_프롬프트.md, 샘플 수는 LLM_설정.md 값으로 채움) + 룰들 → 생성 모델 호출
     룰마다 정규식 조건에 실제로 맞는 짧은 예시(regex_examples, 코드가 만들고 확인)를 붙인다 (LLM_설정.md '정규식 예시')
  2. 응답에서 룰별 JSON 을 꺼낸다. JSON 이 없거나, 응답이 출력 한도에서 잘려(반복 루프) 빠진 룰이 있으면
     **그 룰만** '재시도 횟수' 안에서 다시 요청한다 (반복을 {{REPEAT:문자열:횟수}} 로 쓰라는 주의를 붙여서)
  3. 긴 반복 표기를 펼치고, 샘플마다 코드 검사(local_check) 결과를 붙인다 → 다음 단계 검토 모델이 참고
  4. _2_generated/<이름>.jsonl (+ .meta.json: 토큰 사용량·빠진 룰, raw/<이름>.txt: 응답 원문)

[속도] 요청 여러 개를 동시에 보낸다 (LLM_설정.md '생성 동시 요청 수', 기본 3).
[비용 안전장치] 기본은 요청 1개만 처리 (--max-requests 0 = 남은 것 전부). --dry-run 은 호출하지 않는다.

[사용법]
    python 3_2_generate.py --dry-run
    python 3_2_generate.py                          # 요청 1개
    python 3_2_generate.py --max-requests 20
    python 3_2_generate.py --generator upstage:solar-pro2
    python 3_2_generate.py --generator mock         # 키 없이 흐름 시험
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


def with_examples(rule):
    """정규식 조건마다 코드가 만든 '그 정규식에 실제로 맞는 예시'를 붙인다 (생성 모델이 정규식을 잘못 읽지 않게)."""
    ex = C.rule_regex_examples(rule)
    return dict(rule, regex_examples=ex) if ex else rule


def user_message(rules, n_a, n_n, hint="", examples=True):
    # 샘플 수를 요청 메시지에 다시 못박는다 (지시문의 출력 예시 개수를 그대로 따라 하는 모델이 있음)
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
               f"(예: `<.+\\/script` 는 보통의 `</script>` 와 불일치) anomalies 의 정규식 부분은 이 예시의 구조를 따르고, "
               f"그대로 복사하지 말고 실제 공격 안에 넣는다.\n" if examples else "") +
            f"- 출력은 룰마다 JSON 객체 하나씩. 설명 문장 없이.\n\n"
            + "\n".join(json.dumps(r, ensure_ascii=False) for r in rules))


def generate(name, rules, cfg, providers, run, prompt, gen_dir, log):
    n_a, n_n = C.run_int(run, "생성 anomalies 수", 10), C.run_int(run, "생성 normal 수", 5)
    realistic = C.run_yes(run, "헤더 현실화")
    tries = C.run_int(run, "재시도 횟수", 3)
    examples = C.run_yes(run, "정규식 예시")             # LLM_설정.md (기본 예)
    usage, got, hint, cuts = {}, {}, "", 0
    todo = list(rules)
    started = time.time()
    for attempt in range(1, tries + 1):
        text, u = C.call_llm(cfg, providers, prompt, user_message(todo, n_a, n_n, hint, examples), run, log)
        C.add_usage(usage, "generator", u)
        (gen_dir / "raw").mkdir(parents=True, exist_ok=True)
        (gen_dir / "raw" / f"{name}.try{attempt}.txt").write_text(text, encoding="utf-8")   # 시도마다 원문 보관
        want = {r["id"] for r in todo}
        got.update({o["id"]: o for o in C.parse_json_lines(text)
                    if isinstance(o, dict) and o.get("id") in want and isinstance(o.get("samples"), list)})
        todo = [r for r in todo if r["id"] not in got]
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
                s = dict(s, local_check=C.local_check(s, r))
            samples.append(s)
        out.append({"id": r["id"], "attack_type": g.get("attack_type") or r["attack_type"],
                    "attack_type_note": g.get("attack_type_note"), "samples": samples})
    C.write_jsonl(gen_dir / f"{name}.jsonl", out)
    flat = [s for o in out for s in o["samples"] if isinstance(s, dict)]
    meta = {"request": name, "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
            "seconds": round(time.time() - started, 1), "generator": f"{cfg['provider']}:{cfg['model']}",
            "rules": len(rules), "rules_done": len(out), "rules_missing": missing, "cut_responses": cuts, "regex_examples": examples,
            "samples": len(flat), "local_check": {k: sum(1 for s in flat if s.get("local_check", {}).get("result") == k)
                                                  for k in ("ok", "fail", "unknown")},
            "usage": usage}
    (gen_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("  %s: 룰 %d/%d, 샘플 %d개 (코드 검사 ok %d / fail %d), 토큰 %s", name, len(out), len(rules),
             len(flat), meta["local_check"]["ok"], meta["local_check"]["fail"], usage.get("generator"))


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


def main(argv=None):
    ap = argparse.ArgumentParser(description="생성 모델로 샘플 생성")
    ap.add_argument("--date", help="날짜 폴더 (기본: 05_llm_input 의 가장 최근)")
    ap.add_argument("--set", choices=["delta", "all", "any"], default="any", help="이 종류의 요청만 (기본 전부)")
    ap.add_argument("--max-requests", type=int, default=1, help="처리할 요청 수 (기본 1, 0 = 남은 것 전부)")
    ap.add_argument("--requests", help="특정 요청 이름만 (쉼표 구분, 예: all_b0001_c1)")
    ap.add_argument("--generator", help="생성 모델 'provider:model' (LLM_설정.md 보다 우선)")
    ap.add_argument("--count-pending", action="store_true",
                    help="검토 대기 중인 생성 결과도 --max-requests 에 포함 (3_0 이 사용)")
    ap.add_argument("--pending-all", action="store_true",
                    help="검토 대기분을 모든 날짜 폴더에서 센다 (3.3 이 모든 날짜를 처리할 때, 3_0 이 사용)")
    ap.add_argument("--force", action="store_true", help="이미 생성한 요청·실패가 반복된 요청도 다시")
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
    in_dir = C.latest_input_dir(args.date)
    if cfg is None or in_dir is None:
        log.error("생성 모델 설정 또는 05_llm_input 이 없습니다")
        return 1
    base = C.output_dir(in_dir.name)
    req_dir, gen_dir = base / C.REQ_DIR, base / C.GEN_DIR
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
    log.info("[3.2] 생성 %s:%s | 요청 %d개 중 완료 %d개 → 이번 %d개 (동시 최대 %d개씩, 속도 제한에 맞춰 자동 조절)%s", cfg["provider"], cfg["model"],
             len(reqs), done, len(todo), workers, " (dry-run: 호출 안 함)" if args.dry_run else "")
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
        list(ex.map(one, todo))                # 순서대로 시작 (배치 순서 유지), 동시에 workers 개
    if stopped.is_set():
        log.info("[3.2] 중지 요청 - 여기서 멈춤 (남은 요청은 다음 실행 때)")
    return 1 if fatal.is_set() else 0


if __name__ == "__main__":
    sys.exit(main())
