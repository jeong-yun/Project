"""단계별 평가 하네스 실행기 (CLI).

사용법 (26-Agent-Office 폴더에서):
    python -m evals.run_eval evals/cases/general.yaml            # 모든 케이스
    python -m evals.run_eval evals/cases/general.yaml --case hbf_definition
    python -m evals.run_eval evals/cases/general.yaml --no-web   # 웹 검색 끄고

실제 모델·웹 검색을 호출하므로 API 비용이 든다. 결과는 화면 표와 data/evals/<시각>.json 으로 저장된다.
사람 결정 단계(interrupt)는 쓰지 않으므로 교훈은 저장되지 않는다 (평가가 학습 데이터를 오염시키지 않게).
"""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import (  # noqa: E402
    BASE_DIR,
    load_env,
    load_models,
    load_ranks,
    load_settings,
    load_team,
    load_verification,
    team_org,
)
from core.harness.evaluation import RunTrace, evaluate  # noqa: E402
from core.workflow import build_office_graph, stream_office  # noqa: E402

import yaml  # noqa: E402


def run_case(case: dict, team_id: str, use_web: bool) -> RunTrace:
    ranks, models, settings = load_ranks(), load_models(), load_settings()
    team = load_team(team_id)
    org = team_org(team, ranks)
    graph = build_office_graph(org, ranks, team, models, settings, load_verification())  # storage·checkpointer 없음
    root = case.get("start", org.id)
    web, steps = None, []
    for kind, payload in stream_office(graph, org, case["question"], root, use_web=case.get("use_web", use_web)):
        if kind == "research":
            web = payload
        elif kind == "step":
            steps.append(payload)
            print(f"   · {payload.node_id:<14} {payload.kind:<9} {payload.model.label} {payload.elapsed_sec}s")
    return RunTrace(root_id=root, web=web, steps=steps)


def main() -> None:
    parser = argparse.ArgumentParser(description="단계별 평가 하네스")
    parser.add_argument("cases", help="평가 케이스 YAML 경로")
    parser.add_argument("--case", help="특정 케이스 id 만 실행")
    parser.add_argument("--no-web", action="store_true", help="웹 검색 끄기")
    args = parser.parse_args()

    load_env()
    spec = yaml.safe_load(Path(args.cases).read_text(encoding="utf-8"))
    cases = [c for c in spec["cases"] if not args.case or c["id"] == args.case]
    summary = []
    for case in cases:
        print(f"\n▶ {case['id']}: {case['question']}")
        trace = run_case(case, spec.get("team", "general"), use_web=not args.no_web)
        report = evaluate(case, trace)
        for c in report.checks:
            print(f"   {'✅' if c.passed else '❌'} [{c.step}] {c.name} — {c.detail}")
        first = report.first_failure
        print(
            f"   ⇒ 단계(PRM) 통과율 {report.prm_score:.0%} · 최종(ORM) {'통과' if report.orm_passed else '실패'}"
            + (f" · 처음 실패한 단계: {first.step}{'/' + first.node if first.node else ''}" if first else "")
        )
        summary.append(
            {
                "case": case["id"],
                "prm_score": report.prm_score,
                "orm_passed": report.orm_passed,
                "first_failure": asdict(first) if first else None,
                "checks": [asdict(c) for c in report.checks],
                "final_answer": trace.final.content if trace.final else "",
            }
        )

    out_dir = BASE_DIR / "data" / "evals"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {out.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
