"""결정론적 검증기 · 단계별 평가 하네스 테스트."""

from core.harness import REGISTRY, harness_feedback, run_validators
from core.harness.evaluation import RunTrace, evaluate
from core.schemas import AgentAnswer, Assignment, Delegation, HarnessResult, ModelSpec
from core.search import SearchResult, WebContext
from tests.fakes import make_review

SPEC = ModelSpec(provider="openai", model="x")


def test_python_syntax_validator():
    v = REGISTRY["python_syntax"]
    assert v.check("코드 없음", {}).passed
    assert v.check("```python\nprint(1)\n```", {}).passed
    bad = v.check("```python\ndef f(:\n    pass\n```", {})
    assert not bad.passed and bad.score == 0.0 and "1번째 줄" in bad.log


def test_sql_sandbox_compiles_against_schema(tmp_path, monkeypatch):
    import core.harness.validators as validators

    monkeypatch.setattr(validators, "BASE_DIR", tmp_path)
    (tmp_path / "schema.sql").write_text("CREATE TABLE logs(ip TEXT, ts INTEGER);", encoding="utf-8")
    v, opts = REGISTRY["sql_sandbox"], {"sql_schema": "schema.sql", "sql_execute": True}
    assert v.check("```sql\nSELECT ip, COUNT(*) FROM logs GROUP BY ip;\n```", opts).passed
    assert "no such column" in v.check("```sql\nSELECT nope FROM logs;\n```", opts).log
    assert not v.check("```sql\nSELEC ip FROM logs;\n```", opts).passed


def test_sql_sandbox_timeout(tmp_path, monkeypatch):
    import core.harness.validators as validators

    monkeypatch.setattr(validators, "BASE_DIR", tmp_path)
    slow = "```sql\nWITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT COUNT(*) FROM c;\n```"
    result = REGISTRY["sql_sandbox"].check(slow, {"sql_execute": True, "sql_timeout_sec": 0.2})
    assert not result.passed and "타임아웃" in result.log


def test_json_keys_validator_enforces_step_keys():
    v, opts = REGISTRY["json_keys"], {"required_keys": ["extract_payload", "analyze_anomaly_pattern", "final_decision"]}
    good = '```json\n{"extract_payload": "a", "analyze_anomaly_pattern": "b", "final_decision": "c"}\n```'
    assert v.check(good, opts).passed
    assert "final_decision" in v.check('```json\n{"extract_payload": "a", "analyze_anomaly_pattern": "b"}\n```', opts).log
    assert not v.check("JSON 없이 답함", opts).passed


def test_harness_feedback_contains_error_log():
    text = harness_feedback(run_validators(["python_syntax"], "```python\nx = (\n```", {}))
    assert "0점" in text and "python_syntax" in text and "다시 작성" in text


# ─── 단계별 평가 (PRM vs ORM) ──────────────────────────────────────────


def trace(final_text="HBF 는 High Bandwidth Flash [1]", draft_text="고대역폭 플래시"):
    steps = [
        AgentAnswer(node_id="lead", rank_id="team_lead", kind="plan", content="",
                    delegation=Delegation(analysis="", assignments=[Assignment(child_id="gm_research", subtask="")]), model=SPEC),
        AgentAnswer(node_id="staff_draft", rank_id="staff", kind="draft", content=draft_text, model=SPEC,
                    harness=[HarnessResult(validator="python_syntax", passed=True, score=1.0)]),
        AgentAnswer(node_id="lead", rank_id="team_lead", kind="review", content=final_text,
                    review=make_review(score=8), model=SPEC),
    ]
    web = WebContext(queries=["HBF High Bandwidth Flash"], results=[SearchResult(title="t", url="u", snippet="s")])
    return RunTrace(root_id="lead", web=web, steps=steps)


CASE = {
    "id": "hbf",
    "checks": [
        {"step": "research", "expect": {"contains_any": ["High Bandwidth Flash"]}},
        {"step": "plan", "node": "lead", "expect": {"assigned": ["gm_research"]}},
        {"step": "draft", "expect": {"contains_any": ["플래시"], "harness_pass": True}},
        {"step": "review", "expect": {"min_score": 7}},
        {"step": "final", "expect": {"not_contains": ["HBM의 다른 이름"], "regex": r"\[\d+\]"}},
    ],
}


def test_evaluation_all_pass():
    report = evaluate(CASE, trace())
    assert report.prm_score == 1.0 and report.orm_passed and report.first_failure is None


def test_evaluation_pinpoints_failing_step():
    report = evaluate(CASE, trace(final_text="HBF는 HBM의 다른 이름이다", draft_text="HBM 설명"))
    assert not report.orm_passed
    assert report.first_failure.step == "draft"  # 최종보다 앞선 '초안' 단계에서 이미 틀렸다
    assert report.prm_score == 0.75
