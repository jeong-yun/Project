"""단계별 평가 하네스 (Step-wise Evaluation Harness).

최종 답변만 채점(ORM: Outcome Reward Model 방식)하면 '어디서' 틀렸는지 알 수 없다.
그래서 파이프라인의 각 단계(검색 → 분배 → 초안 → 검토 → 통합 → 최종)마다 미리 정한
유닛 테스트를 돌려(PRM: Process Reward Model 방식) 정확히 어느 노드에서 실패했는지 추적한다.

평가 케이스(YAML) 예시 — evals/cases/general.yaml 참고
    - id: hbf_definition
      question: HBF에 대해 알려줘
      checks:
        - step: research              # 검색 단계: 검색어·결과 제목에서 검사
          expect: {contains_any: ["High Bandwidth Flash"]}
        - step: draft                 # 초안 단계 (여러 개면 match: any|all)
          node: staff_draft           # 특정 에이전트만 (선택)
          expect: {contains_any: ["플래시", "Flash"]}
        - step: final                 # 최종 답변
          expect: {not_contains: ["HBM의 다른 이름"], harness_pass: true}

expect 에 쓸 수 있는 검사
    contains_all / contains_any / not_contains : 문자열 포함 여부 (대소문자 무시)
    regex       : 정규식 일치
    min_score   : 검토·통합 단계의 하위 결과 점수(1~10) 하한
    critic_min  : 비평가 점수(1~5) 하한
    harness_pass: 결정론적 하네스 검증 통과 여부
    assigned    : 분배(plan) 단계에서 업무를 맡겨야 하는 하위 에이전트 id 목록
"""

import re
from dataclasses import dataclass, field

from core.schemas import AgentAnswer
from core.search import WebContext

# 파이프라인 순서 — '처음 실패한 단계'를 찾을 때 이 순서를 따른다
STEP_ORDER = ["research", "plan", "draft", "review", "integrate", "final"]


@dataclass
class RunTrace:
    """평가 대상 — 한 번 실행한 결과 전체."""

    root_id: str
    web: WebContext | None
    steps: list[AgentAnswer]

    @property
    def final(self) -> AgentAnswer | None:
        """supervisor 가 마지막으로 올린 결과 (= 최종 답변)."""
        mine = [s for s in self.steps if s.node_id == self.root_id and s.kind != "plan"]
        return mine[-1] if mine else None


@dataclass
class CheckResult:
    name: str
    step: str
    passed: bool
    detail: str
    node: str | None = None


@dataclass
class CaseReport:
    case_id: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def process_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if c.step != "final"]

    @property
    def outcome_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if c.step == "final"]

    @property
    def prm_score(self) -> float:
        """단계(과정) 검사 통과율."""
        checks = self.process_checks
        return sum(c.passed for c in checks) / len(checks) if checks else 1.0

    @property
    def orm_passed(self) -> bool:
        """최종 결과 검사 통과 여부."""
        return all(c.passed for c in self.outcome_checks)

    @property
    def first_failure(self) -> CheckResult | None:
        failed = [c for c in self.checks if not c.passed]
        return min(failed, key=lambda c: STEP_ORDER.index(c.step)) if failed else None


def _texts_for(step: str, trace: RunTrace, node: str | None) -> list[tuple[str, AgentAnswer | None]]:
    """검사 대상 (텍스트, 단계 결과) 목록."""
    if step == "research":
        web = trace.web
        if web is None:
            return [("", None)]
        return [("\n".join([*web.queries, *(r.title + " " + r.snippet for r in web.results)]), None)]
    if step == "final":
        final = trace.final
        return [(final.content if final else "", final)]
    picked = [s for s in trace.steps if s.kind == step and (node is None or s.node_id == node)]
    return [(s.content, s) for s in picked]


def _check_one(text: str, answer: AgentAnswer | None, expect: dict) -> tuple[bool, str]:
    """하나의 대상에 expect 의 모든 조건을 적용한다. (통과 여부, 실패 이유)"""
    low = text.lower()
    if "contains_all" in expect:
        missing = [k for k in expect["contains_all"] if k.lower() not in low]
        if missing:
            return False, f"다음 문구가 없음: {missing}"
    if "contains_any" in expect and not any(k.lower() in low for k in expect["contains_any"]):
        return False, f"다음 중 아무것도 없음: {expect['contains_any']}"
    if "not_contains" in expect:
        found = [k for k in expect["not_contains"] if k.lower() in low]
        if found:
            return False, f"들어가면 안 되는 문구가 있음: {found}"
    if "regex" in expect and not re.search(expect["regex"], text, flags=re.IGNORECASE | re.DOTALL):
        return False, f"정규식 불일치: {expect['regex']}"
    if "min_score" in expect:
        score = answer.review.score if answer and answer.review else None
        if score is None or score < expect["min_score"]:
            return False, f"검토 점수 {score} < {expect['min_score']}"
    if "critic_min" in expect:
        score = answer.critic.score if answer and answer.critic else None
        if score is None or score < expect["critic_min"]:
            return False, f"비평 점수 {score} < {expect['critic_min']}"
    if "harness_pass" in expect:
        passed = all(h.passed for h in answer.harness) if answer else False
        if passed != expect["harness_pass"]:
            logs = "; ".join(h.log for h in (answer.harness if answer else []) if not h.passed)
            return False, f"하네스 결과 {passed} (기대 {expect['harness_pass']}) {logs}"
    if "assigned" in expect:
        chosen = {a.child_id for a in answer.delegation.assignments} if answer and answer.delegation else set()
        missing = set(expect["assigned"]) - chosen
        if missing:
            return False, f"업무 분배에서 빠진 하위 에이전트: {sorted(missing)}"
    return True, "통과"


def evaluate(case: dict, trace: RunTrace) -> CaseReport:
    """케이스의 모든 검사를 실행해 보고서를 만든다."""
    report = CaseReport(case_id=case.get("id", "case"))
    for i, check in enumerate(case.get("checks", []), 1):
        step, node = check["step"], check.get("node")
        name = check.get("name") or f"{i}. {step}{'/' + node if node else ''}"
        targets = _texts_for(step, trace, node)
        if not targets:
            report.checks.append(CheckResult(name, step, False, "해당 단계가 실행되지 않음", node))
            continue
        outcomes = [_check_one(text, answer, check.get("expect", {})) for text, answer in targets]
        if check.get("match", "any") == "all":
            passed = all(ok for ok, _ in outcomes)
        else:
            passed = any(ok for ok, _ in outcomes)
        detail = "통과" if passed else " | ".join(reason for ok, reason in outcomes if not ok)
        report.checks.append(CheckResult(name, step, passed, detail, node))
    return report
