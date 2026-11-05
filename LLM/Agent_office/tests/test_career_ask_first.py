"""❓ 질문 먼저 (core/ask_first.py · ask_first_ui.py) — 팀 설정(ask_first)이 있는 팀만 시작 전에 1~3개 묻는다.

지금은 👔 포트폴리오·자소서 팀만 켠다. 다른 팀은 지금과 똑같이 바로 시작해야 한다.
문서·그림은 코드로 만든 가짜(가명)뿐이다. 모델 호출·네트워크 없음.
"""

from pathlib import Path

from core import ask_first
from core.config import list_teams, load_team
from core.tasks import build_request
from tests.test_career_figures import figure_docx

ROOT = Path(__file__).resolve().parents[1]
EXTRA = "Word 는 기존 기술경력서 포맷과 유사하게, md 는 경력서.md 와 유사하게"


def update_inputs(tmp_path, extra: str = EXTRA, **more) -> dict:
    new = tmp_path / "new.md"
    new.write_text("최신 요약 (가명)\n탐지 모델 고도화 (2026-06-01 ~ 2026-10-31)\n", encoding="utf-8")
    task = next(t for t in load_team("career").tasks if t.id == "portfolio_update")
    values = {"portfolio": figure_docx(tmp_path), "portfolio_name": "기술경력서.docx", "docs": [str(new)], "docs_names": ["new.md"],
              "docs_name": "new.md", "fmt_docx": True, "fmt_md": True, **more}
    return build_request(task, values, extra).tool_inputs


def test_only_career_team_asks_first_and_only_on_portfolio_tasks():
    teams = {t.id: t for t in list_teams()}
    assert [tid for tid, t in teams.items() if t.ask_first is not None] == ["career"]  # 다른 팀은 지금처럼 바로 시작
    career = teams["career"]
    assert career.ask_first.tasks == ["portfolio", "portfolio_update"] and career.ask_first.max_questions == 3
    assert not ask_first.enabled(career, "cover_letter") and not ask_first.enabled(career, "jd_recommend")
    for tid, team in teams.items():
        if tid != "career":
            assert ask_first.questions_for(team, "anything", {"extra_request": EXTRA}) == []


def test_career_questions_follow_the_request_and_files(tmp_path):
    career = load_team("career")
    inputs = update_inputs(tmp_path)
    items = ask_first.questions_for(career, "portfolio_update", inputs)
    assert [i.id for i in items] == ["missing_ref", "split_formats", "figures"]  # 최대 3개 (합치기는 4번째라 빠짐)
    assert "'경력서.md'" in items[0].question and items[0].meta["files"] == ["경력서.md"]
    assert "2개" in items[2].question
    assert all(len(i.options) >= 2 for i in items)

    plain = ask_first.questions_for(career, "portfolio_update", update_inputs(tmp_path, extra=""))
    assert [i.id for i in plain] == ["figures", "merge"]  # 형식 지정이 없으면 형식 질문도 없다
    assert ask_first.questions_for(career, "cover_letter", {**inputs, "career_kind": "cover_letter"}) == []


def test_uploaded_reference_is_not_asked_again(tmp_path):
    ref = tmp_path / "경력서.md"
    ref.write_text("# 경력서 (가명)\n## 회사\n", encoding="utf-8")
    inputs = update_inputs(tmp_path, format_refs=[str(ref)], format_refs_names=["경력서.md"], format_refs_name="경력서.md")
    ids = [i.id for i in ask_first.questions_for(load_team("career"), "portfolio_update", inputs)]
    assert "missing_ref" not in ids and ids[:2] == ["split_formats", "figures"]


def test_answers_go_into_question_brief_and_tool_content(tmp_path):
    from core.tools import career_docs

    task = next(t for t in load_team("career").tasks if t.id == "portfolio_update")
    inputs = update_inputs(tmp_path)
    req = build_request(task, {}, "")  # 브리프 모양만 빌려 쓴다
    request = {"question": "q", "brief": req.brief, "tool_inputs": inputs}
    answers = [{"id": "merge", "question": "합칠까요?", "answer": "합치지 않는다 — 원래 항목 그대로"}, {"id": "x", "question": "빈 답", "answer": ""}]
    out = ask_first.apply_answers(request, answers)
    assert out["question"].startswith("q\n" + ask_first.HEADER) and "- 합칠까요? → 합치지 않는다 — 원래 항목 그대로" in out["question"]
    assert "빈 답" not in out["question"] and out["brief"].constraints[-1].startswith("먼저 답한 결정: 합칠까요?")
    assert out["tool_inputs"]["ask_answers"] == answers[:1] and request["tool_inputs"].get("ask_answers") is None  # 원래 요청은 그대로
    res = career_docs.run(out["question"], out["tool_inputs"], None)
    assert res.content.startswith("### ✅ 먼저 답한 질문") and "**합치지 않는다 — 원래 항목 그대로**" in res.content
    assert ask_first.apply_answers(request, []) is request


# ─── 화면 (AppTest) ──────────────────────────────────────────────────────
# AppTest.from_function 은 함수 본문을 임시 파일로 써서 실행한다 (Windows cp949 로 쓰고 utf-8 로 읽음) → 본문에는 ASCII 만.


def ask_page(root: str, team_id: str, task_id: str, request: dict) -> None:
    import sys as _sys

    if root not in _sys.path:
        _sys.path.insert(0, root)
    import streamlit as st

    import ask_first_ui
    from core.config import load_team as _load

    first = not st.session_state.get("_sent")
    st.session_state["_sent"] = True
    out = ask_first_ui.step(request if first else {}, _load(team_id), task_id)
    if out:
        st.session_state["result"] = out


def test_ask_box_waits_for_answers_then_starts_with_them(tmp_path):
    from streamlit.testing.v1 import AppTest

    request = {"question": "q", "brief": None, "tool_inputs": update_inputs(tmp_path)}
    at = AppTest.from_function(ask_page, kwargs={"root": str(ROOT), "team_id": "career", "task_id": "portfolio_update",
                                                 "request": request}, default_timeout=30).run()
    assert not at.exception, at.exception
    assert "result" not in at.session_state  # 답하기 전에는 업무를 시작하지 않는다
    assert len(at.radio) == 3 and at.radio[0].value.startswith("지금 포트폴리오(원본) 형식")  # 첫 답(원문 유지)이 기본
    at.radio[1].set_value(at.radio[1].options[1])
    at = at.button[[b.label for b in at.button].index("✅ 이 답으로 시작")].click().run()
    assert not at.exception, at.exception
    result = at.session_state["result"]
    assert "→ 한 가지 형식으로 둘 다 쓴다" in result["question"]
    assert [a["id"] for a in result["tool_inputs"]["ask_answers"]] == ["missing_ref", "split_formats", "figures"]


def test_ask_box_skip_starts_unchanged_and_other_team_never_waits(tmp_path):
    from streamlit.testing.v1 import AppTest

    request = {"question": "q", "brief": None, "tool_inputs": update_inputs(tmp_path)}
    at = AppTest.from_function(ask_page, kwargs={"root": str(ROOT), "team_id": "career", "task_id": "portfolio_update",
                                                 "request": request}, default_timeout=30).run()
    at = at.button[[b.label for b in at.button].index("⏭ 묻지 않고 시작")].click().run()
    assert not at.exception, at.exception
    assert at.session_state["result"]["question"] == "q" and "ask_answers" not in at.session_state["result"]["tool_inputs"]

    other_team = next(t for t in list_teams() if t.id != "career" and t.tasks)
    other, task_id = other_team.id, other_team.tasks[0].id
    at = AppTest.from_function(ask_page, kwargs={"root": str(ROOT), "team_id": other, "task_id": task_id,
                                                 "request": {"question": "q2", "tool_inputs": {"extra_request": EXTRA}}},
                               default_timeout=30).run()
    assert not at.exception, at.exception
    assert at.session_state["result"]["question"] == "q2" and len(at.radio) == 0  # 묻지 않고 바로
