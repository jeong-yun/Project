"""[3.1 LLM 입력 준비] 05_llm_input 의 배치에서 생성할 룰만 골라 'LLM 호출 1회 = 요청 파일 1개' 로 나눈다.  (3_0 이 실행)

[전체 흐름]
  1. 05_llm_input/<날짜>/batches/<delta|all>/batch_NNNN.jsonl 을 읽는다.
  2. 생성하지 않을 룰을 뺀다 (LLM_설정.md):
       - '생성 제외 attack_type' (기본 informational — 데이터셋에서 어차피 제외)
       - '## 생성 제외 룰' 표의 룰 (다시 만들어도 코드 검사를 넘을 수 없는 룰: 성립하지 않는 정규식·HTTP 아님 등, 2026-10-30)
       - '생성 대상 side' 가 아닌 룰 (기본 request 만 — 기존 학습 로그에 응답 payload 없음)
       - 기존 학습 데이터가 충분한 룰: 3.1a 결과에서 기존 샘플이 '기존 데이터 충분 기준'(기본 1000) 이상
  3. 기존 샘플이 1 ~ '기존 데이터 참고 기준'(기본 50) 건인 룰에는 기존 payload 예시(existing_examples)를 붙인다.
     → 생성 모델이 비슷한 유형으로, 하지만 똑같지 않게 만들도록 (지시문 LLM_프롬프트.md)
     예시는 외부 LLM 에 전달되므로 붙이기 전에 민감값(Host·쿠키·인증값·IP·이메일·토큰 등)을 자리표시 값으로 바꾼다
     ('참고 예시 마스킹', 기본 예, 3_0b_mask.py). 마스킹 오류나 룰 조건이 깨진 예시는 보내지 않는다 (원래 값을 보내는 길 없음).
     요청 파일의 룰마다 _mask(바꾼 개수·뺀 예시 수)를 남기고, 이 칸은 LLM 에 보내지 않는다 (3.2·3.3 이 뺌).
  4.'호출당 룰 수' 씩 나눠 07_llm_output/<날짜>/_1_requests/<set>_bNNNN_<지문>_cK.jsonl 로 저장한다.
     - 이미 생성(3.2)까지 끝난 룰은 다시 넣지 않는다.
     - 입력 날짜가 바뀌어도(1_0 이 새 05_llm_input/<날짜> 를 만들어도) 다른 날짜 폴더에서 생성한 룰은
       룰 내용(원본 룰 텍스트)이 같으면 다시 넣지 않는다. 내용이 바뀐 룰(rev 변경 등)만 다시 만든다.
     - 조건(설정·기존 데이터 결과)이 바뀌면 아직 생성 안 된 요청 파일만 다시 만든다.
     LLM 은 호출하지 않는다 (비용 0).

[사용법]
    python 3_1_prepare_requests.py                 # 변경분(delta)
    python 3_1_prepare_requests.py --set all       # 전체
    python 3_1_prepare_requests.py --dry-run       # 파일을 쓰지 않고 건수·마스킹 수만 로그에
    python 3_1_prepare_requests.py --rules et:1:2033312,talos:1:39925 --dry-run
    python 3_1_prepare_requests.py --rules @목록.txt  # 지정 룰만 (한 줄에 id 하나, # 뒤는 설명)

[지정 룰 다시 만들기] (--rules, 2026-10-30) 생성까지 끝난(완료) 룰도 새 요청 pick_b0000_<지문>_cK.jsonl 로 만든다.
  - 공격 샘플이 모자란 룰을 바뀐 지시문·코드 검사로 다시 만들 때. 생성은 3_2 --set pick (또는 3_0 --set pick --rules …).
  - 생성 제외 유형·방향, 기존 데이터 충분 룰은 그대로 뺀다. 아직 생성 전인 pick 요청에 이미 있는 룰은 다시 넣지 않는다.
  - 새 결과는 지금 입력 날짜 폴더에 쌓이고, 예전 결과(정상 샘플 등)도 그대로 학습 로그에 남는다.
"""

import argparse
import collections
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("llm_common", Path(__file__).resolve().parent / "3_0a_common.py")
C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)
_mspec = importlib.util.spec_from_file_location("mask_examples", Path(__file__).resolve().parent / "3_0b_mask.py")
M = importlib.util.module_from_spec(_mspec); _mspec.loader.exec_module(M)

EXISTING_SUB = "_0_existing"


def main(argv=None):
    ap = argparse.ArgumentParser(description="LLM 요청 파일 준비 (호출 없음)")
    ap.add_argument("--set", choices=["delta", "all", "pick"], default="delta",
                    help="배치 종류 (기본 delta = 변경분, pick = --rules 로 지정한 룰)")
    ap.add_argument("--rules", help="지정 룰만 (완료 상태여도) 새 요청으로: 'id,id,…' 또는 '@파일' (한 줄에 id 하나)")
    ap.add_argument("--date", help="05_llm_input 날짜 폴더 (기본: 가장 최근)")
    ap.add_argument("--force", action="store_true", help="아직 생성 안 된 요청 파일을 모두 다시 만든다")
    ap.add_argument("--dry-run", action="store_true", help="파일을 쓰거나 지우지 않고 건수·마스킹 수만 로그에")
    args = ap.parse_args(argv)
    log = C.setup_logging("3_1_prepare_requests")
    wanted = []
    if args.rules:
        args.set = "pick"
        text = Path(args.rules[1:]).read_text(encoding="utf-8") if args.rules.startswith("@") else args.rules
        wanted = list(dict.fromkeys(x for line in text.split("\n")
                                    for x in re.split(r"[,\s]+", line.split("#")[0]) if x))
    elif args.set == "pick":
        log.error("--set pick 은 --rules 와 함께 씁니다")
        return 1
    if not C.generation_guard(log, "3_1_prepare_requests"):     # 3_0 또는 다른 생성이 돌고 있으면 (같은 파일을 쓰므로) 멈춤
        return 2

    try:
        _, _, run = C.load_settings()
    except (ValueError, FileNotFoundError) as e:
        log.error("LLM 설정 읽기 실패: %s", e)
        return 1
    in_dir = C.latest_input_dir(args.date)
    if in_dir is None:
        log.error("05_llm_input 에 배치가 없습니다 (먼저 1_0_run_daily.py 실행)")
        return 1
    base = C.output_dir(in_dir.name)
    req_dir, gen_dir = base / C.REQ_DIR, base / C.GEN_DIR
    req_dir.mkdir(parents=True, exist_ok=True)

    skip = {t.strip() for t in run.get("생성 제외 attack_type", "").split(",") if t.strip() not in ("", "-")}
    sides = {s.strip() for s in run.get("생성 대상 side", "request").split(",") if s.strip()}
    n = C.run_int(run, "호출당 룰 수", 3)
    hi = C.run_int(run, "기존 데이터 충분 기준", 1000)
    lo = C.run_int(run, "기존 데이터 참고 기준", 50)
    n_ex = C.run_int(run, "참고 예시 수", 3)
    mask = C.run_yes(run, "참고 예시 마스킹")       # 기본 예 (설정에 줄이 없어도 켬)
    excl = C.excluded_rules()                         # LLM_설정.md '생성 제외 룰' (다시 만들어도 코드 검사를 넘을 수 없는 룰)

    # 기존 학습 데이터 결과 (3.1a). 없으면 모든 룰을 '기존 0건' 으로 본다
    cov_file = base / EXISTING_SUB / "coverage.json"
    cov = json.loads(cov_file.read_text(encoding="utf-8")) if cov_file.exists() else {}
    if not cov_file.exists():
        log.warning("[3.1] 기존 데이터 결과(%s) 없음 - 기존 데이터 기준 없이 준비 (3_1a_existing_coverage.py 먼저 실행 권장)",
                    cov_file.relative_to(C.BASE_DIR))

    # 조건 지문: 바뀌면 아직 생성 안 된 요청 파일을 다시 만든다
    # 마스킹 설정·규칙 버전도 넣는다: 켜거나 규칙을 바꾸면 예전 예시가 든 생성 전 요청을 다시 만든다
    cond = json.dumps([sorted(skip), sorted(sides), n, hi, lo, n_ex,
                       hashlib.sha1(cov_file.read_bytes()).hexdigest() if cov_file.exists() else ""]
                      + ([f"mask{M.VERSION}"] if mask else []) + ([sorted(wanted)] if wanted else [])
                      + ([sorted(excl)] if excl else []))
    fp = hashlib.sha1(cond.encode()).hexdigest()[:6]
    manifest_file = req_dir / f"_manifest_{args.set}.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else {}
    pending = [p for p in req_dir.glob(f"{args.set}_*.jsonl") if not (gen_dir / p.name).exists()]
    gone = set()           # 다시 만들 생성 전 요청 (dry-run 은 지우지 않고 없는 것으로만 본다)
    if args.force or manifest.get("fingerprint") != fp:
        gone = set(pending)
        for p in pending if not args.dry_run else []:          # 생성 전 요청만 지움 (생성된 결과는 그대로 둔다)
            p.unlink()
        if pending:
            log.info("[3.1] 조건 변경 - 생성 전 요청 %d개를 다시 만든다%s", len(pending), " (dry-run: 지우지 않음)" if args.dry_run else "")
        pending = []

    # 다시 넣지 않을 룰: 이 묶음의 요청 파일(대기 중 포함) + 다른 묶음(all/delta)에서 이미 생성까지 끝난 룰
    # 단, 생성 응답에서 빠진 룰(.meta.json 의 rules_missing)은 다시 새 요청에 넣는다
    assigned, retry = set(), set()
    for p in req_dir.glob("*.jsonl"):
        if p in gone:
            continue
        if wanted:          # 지정 룰: 생성까지 끝난 룰도 다시. 아직 생성 전인 pick 요청에 있는 룰만 건너뛴다
            if p.name.startswith("pick_") and not (gen_dir / p.name).exists():
                assigned |= {r["id"] for r in C.read_jsonl(p)}
            continue
        if p.name.startswith(args.set + "_") or (gen_dir / p.name).exists():
            ids = {r["id"] for r in C.read_jsonl(p)}
            meta = gen_dir / f"{p.stem}.meta.json"
            if meta.exists():
                miss = set(json.loads(meta.read_text(encoding="utf-8")).get("rules_missing", []))
                ids -= miss
                retry |= miss
            assigned |= ids
    if retry - assigned:
        log.info("[3.1] 생성 응답에서 빠졌던 룰 %d개를 다시 요청에 넣는다", len(retry - assigned))
    # 다른 날짜 폴더에서 이미 생성한 룰: 내용이 같으면 건너뛴다 (입력 날짜가 바뀌어도 이어서 진행)
    prior = C.generated_rules(exclude=base) if not wanted else {}
    if prior:
        log.info("[3.1] 다른 날짜 폴더에서 생성한 룰 %d개 - 룰 내용이 같으면 건너뜀 (%s)", len(prior),
                 ", ".join(sorted({d for _, d in prior.values()})))

    stat = collections.Counter()
    mstat, mkinds = collections.Counter(), collections.Counter()     # 마스킹: 예시 수·뺀 수, 종류별 바꾼 값 수 (원래 값은 없음)
    made = 0
    chars = 0
    if wanted:              # 지정 룰: 배치 대신 rules_all 에서 (준 순서대로)
        allr = {r["id"]: r for r in C.read_jsonl(in_dir / "rules_all.jsonl")}
        lost = [x for x in wanted if x not in allr]
        if lost:
            log.warning("[3.1] 지정 룰 %d개는 지금 입력(rules_all)에 없음: %s", len(lost), ", ".join(lost[:10]))
        sources = [("0000", [allr[x] for x in wanted if x in allr])]
    else:
        sources = ((bf.stem.split("_")[1], C.read_jsonl(bf))
                   for bf in sorted((in_dir / "batches" / args.set).glob("batch_*.jsonl")))
    for bno, recs in sources:
        chosen = []
        for r in recs:
            stat["전체"] += 1
            if r["attack_type"] in skip or r.get("side", "request") not in sides:
                stat["제외: 유형·방향"] += 1
                continue
            if r["id"] in excl:
                stat["제외: 생성 제외 룰"] += 1
                continue
            ex = cov.get(r["id"], {})
            cnt = ex.get("count", 0)
            if cnt >= hi:
                stat[f"제외: 기존 {hi}건 이상"] += 1
                continue
            if r["id"] in assigned:
                stat["이미 요청에 있음"] += 1
                continue
            p = prior.get(r["id"])
            if p and p[0] == C.rule_hash(r):
                stat["이미 생성됨 (다른 날짜)"] += 1
                continue
            if p:
                stat["룰 변경 → 다시 생성"] += 1
            if 1 <= cnt <= lo:
                if mask:          # 보내기 전에 민감값을 바꾼다. 오류·룰 조건이 깨진 예시는 빼고 다음 후보로 채움
                    exs, st = M.safe_examples(C, ex.get("examples", []), r, n_ex)
                    mstat.update({k: st[k] for k in ("in", "sent", "drop_error", "drop_rule")})
                    mkinds.update(st["masked"])
                    info = {"on": True, "values": sum(st["masked"].values()), "dropped": st["drop_error"] + st["drop_rule"]}
                else:
                    exs = ex.get("examples", [])[:n_ex]
                    mstat["sent"] += len(exs)
                    info = {"on": False}
                r = dict(r, existing_count=cnt, _mask=info)
                if exs:           # 예시가 모두 빠지면 예시 없이 일반 생성 (기존 0건 룰과 같은 흐름)
                    r["existing_examples"] = exs
                stat[f"생성: 기존 1~{lo}건 (예시 참고)"] += 1
            elif cnt > lo:
                stat[f"생성: 기존 {lo + 1}~{hi - 1}건"] += 1
            else:
                stat["생성: 기존 0건"] += 1
            chosen.append(r)
        prefix = f"{args.set}_b{bno}_{fp}_c"
        # 같은 배치에 이미 요청 파일이 있으면(빠진 룰 재요청 등) 그 다음 번호부터 - 기존 요청을 덮어쓰지 않게
        used = [int(p.stem[len(prefix):]) for p in req_dir.glob(prefix + "*.jsonl") if p.stem[len(prefix):].isdigit()]
        start = max(used, default=0) + 1
        for k in range(0, len(chosen), n):
            chunk = chosen[k:k + n]
            if not args.dry_run:
                C.write_jsonl(req_dir / f"{prefix}{start + k // n}.jsonl", chunk)
            chars += sum(len(json.dumps(r, ensure_ascii=False)) for r in chunk)
            made += 1

    masking = {"enabled": mask, "rule_version": M.VERSION if mask else None,
               "examples_seen": mstat["in"], "existing_examples_sent": mstat["sent"],
               "dropped_mask_error": mstat["drop_error"], "dropped_rule_broken": mstat["drop_rule"],
               "masked_values": sum(mkinds.values()), "masked_by_kind": {k: mkinds[k] for k in M.KINDS if mkinds[k]}}
    prompt_chars = len(C.GEN_PROMPT_FILE.read_text(encoding="utf-8"))
    log.info("[3.1] %s/batches/%s: %s", in_dir.name, args.set, dict(stat))
    if mask:
        log.info("[3.1] 참고 예시 마스킹: 보낼 예시 %d개, 바꾼 값 %d개 %s, 뺀 예시 %d개 (오류 %d · 룰 조건 깨짐 %d)",
                 mstat["sent"], masking["masked_values"], masking["masked_by_kind"],
                 mstat["drop_error"] + mstat["drop_rule"], mstat["drop_error"], mstat["drop_rule"])
    else:
        log.warning("[3.1] 참고 예시 마스킹 꺼짐 - 기존 예시 %d개를 그대로 보냄 (LLM_설정.md '참고 예시 마스킹')", mstat["sent"])
    if args.dry_run:
        est = ""
        try:        # 3.5 가 센 요청당 실제 평균(시간·토큰)으로 어림 (화면의 '요청 수 → 예상' 과 같은 계산)
            avg = json.loads((C.LLM_OUTPUT_DIR / "_progress" / "progress.json").read_text(encoding="utf-8"))["average_per_request"]
            gw, rw, overlap = C.parallel_settings(run)
            g_s, r_s = (avg["generate_seconds"] or 0) / gw, (avg["review_seconds"] or 0) / rw
            est = (f", 예상 약 {made * (max(g_s, r_s) if overlap else g_s + r_s) / 60:.0f}분 (동시 {gw}·{rw}),"
                   f" 토큰 생성 약 {made * avg['generator_tokens']:,} · 검토 약 {made * avg['reviewer_tokens']:,}")
        except (OSError, ValueError, KeyError, TypeError):
            pass
        log.info("[3.1] dry-run: 요청 파일·manifest 를 쓰지 않음 (만들 요청 %d개, 생성 입력 약 %d 토큰%s)", made,
                 (chars + prompt_chars * made) // 3, est)
        return 0
    manifest = {"fingerprint": fp, "set": args.set, "input": in_dir.name, "stats": dict(stat), "masking": masking,
                "requests_total": len(list(req_dir.glob(f"{args.set}_*.jsonl")))}
    manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("[3.1] 새 요청 %d개 (전체 요청 %d개, 생성 입력 약 %d 토큰). 요청 파일: %s", made,
             manifest["requests_total"], (chars + prompt_chars * made) // 3, req_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
