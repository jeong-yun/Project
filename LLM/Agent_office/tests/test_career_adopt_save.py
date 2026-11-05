"""👔 채택 후 저장 — '이 안으로 채택'하면 results/career/ 에 문서가 저장되고, 저장 경로·실패가 화면에 분명히 보이는지.

2026-11-05 문의: "JD 추천 후 채택하면 저장이 안 되고 최종 문서 다운로드만 된다."
확인한 원인: Word 만들기(python-docx)가 PDF·옛 문서 글에 섞인 제어 문자(\\x02 · \\x0c · \\x0b …)에서 예외로 멈추면
 ① 결과 상자는 '문서를 만들지 못했습니다' 경고 + 최종 결과 .md 받기만 남고 ② 채택 저장(save_adopted)은 통째로 실패해
 그 오류가 '참고 기록' 탭에만 조용히 남았다. → 제어 문자 정리 · 문서가 실패해도 기록(.md)은 저장 · 저장 경로/실패를 화면에 표시.
가짜 모델·가짜 데이터(가명)만. 저장 폴더는 tmp_path.
"""

import time
from pathlib import Path

import pytest

from core import doc_export
from core.exporter import export_adopted, list_results
from core.schemas import HumanDecision
from tests.test_career_recommend_update import FACT, JD_A, JD_B, SRC, recommend_answer, run_fake

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.mark.parametrize("ch", ["\x02", "\x0c", "\x0b", "\x00", "￾"])
def test_control_characters_do_not_break_word_or_hangul(ch):
    import io
    import zipfile
    from xml.etree import ElementTree

    import docx

    md = f"# 제목{ch}\n| 요구사항 | 근거 |\n|---|---|\n| 로그{ch}분석 | 문서{ch}원문 |\n\n본문 {ch} 끝"
    d = docx.Document(io.BytesIO(doc_export.to_docx(md, f"JD{ch}추천")))
    assert "로그" in d.tables[0].cell(1, 0).text
    section = zipfile.ZipFile(io.BytesIO(doc_export.to_hwpx(md, "t"))).read("Contents/section0.xml")
    ElementTree.fromstring(section)  # 한글 파일 XML 이 깨지지 않는다


def test_read_text_drops_control_characters():
    from core.doc_text import read_bytes

    doc = read_bytes("jd.txt", "보안\x02 로그\x0c분석\x0b경험".encode("utf-8"))
    assert doc.text == "보안 로그\n분석\n경험"


def test_jd_recommend_adopt_saves_files(tmp_path, ranks, models, use_script):
    import career_ui
    from tests.fakes import make_review

    reviews = [make_review("revise", 8, recommend_answer()), make_review("approve", 9), make_review("approve", 9)]
    events, _ = run_fake("jd_recommend", {"profile_text": SRC, "jd": JD_A + "\n---\n" + JD_B, "fmt_docx": True},
                         reviews, ranks, models, use_script)
    tools = next(p for k, p in events if k == "tools")
    final = next(s for k, s in events if k == "step" and s.node_id == "lead")
    run = {"run_id": "r1", "question": "JD 추천", "tools": tools, "notes": [],
           "career_inputs": {"career_kind": "jd_recommend", "fmt_md": "예", "fmt_docx": True}}
    decision = HumanDecision(action="adopt", adopted_node_id="lead", final_answer=final.content.strip(), score=4)
    paths = export_adopted(run, decision, "career", "팀", {}, tmp_path, extra_files=career_ui.adopted_files(run, decision))
    assert sorted(p.suffix for p in paths) == [".docx", ".md", ".md"] and len(list_results(tmp_path, "career")) == 3


# ─── 실제 화면 (AppTest + 가짜 모델) ──────────────────────────────────────


@pytest.fixture
def career_app(tmp_path, monkeypatch, use_script):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    from core import call_monitor
    from tests.fakes import Script, make_review

    use_script(Script(text=FACT, ReviewResult=[make_review("revise", 8, recommend_answer())] + [make_review("approve", 9)] * 3))
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))
    call_monitor.reset()
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=90)
    at.query_params["team"] = "career"
    at.query_params["view"] = "team"
    at.session_state["office_view"] = "chart"
    yield at
    st.cache_resource.clear()
    call_monitor.reset()


def adopt_jd_recommend(at):
    at.run()
    assert not at.exception, at.exception
    at = at.selectbox(key="dept_task_career").set_value("career:jd_recommend").run()
    at.text_area(key="tf_career_jd_recommend_profile_text").input(SRC)
    at.text_area(key="tf_career_jd_recommend_jd").input(JD_A + "\n---\n" + JD_B)
    at = next(b for b in at.button if b.label == "맡기기").click().run()
    assert not at.exception, at.exception
    end = time.time() + 60
    while time.time() < end and not any(b.label == "✅ 이 안으로 채택" for b in at.button):
        time.sleep(0.2)
        at = at.run()
    at = at.run()  # 업무가 끝난 것을 알아챈 화면 새로 그리기(watch_run_state)가 지나간 뒤에 누른다
    at = next(b for b in at.button if b.label == "✅ 이 안으로 채택").click().run()
    assert not at.exception, at.exception
    return at


def test_app_jd_recommend_adopt_saves_and_shows_where(career_app, tmp_path):
    at = adopt_jd_recommend(career_app)
    saved = sorted(p.name for p in (tmp_path / "results" / "career").glob("*"))
    assert len(saved) == 3 and any(n.endswith("_JD 추천.docx") for n in saved)
    assert any("💾 결과 파일 3개" in i.value and "career" in i.value for i in at.info)
    assert sum("📁 저장됨" in c.value for c in at.caption) == 3
    assert not at.warning


def test_app_adopt_still_saves_record_and_warns_when_documents_fail(career_app, tmp_path, monkeypatch):
    import career_ui

    def broken(run, decision):
        raise ValueError("All strings must be XML compatible")

    monkeypatch.setattr(career_ui, "adopted_files", broken)
    at = adopt_jd_recommend(career_app)
    saved = list((tmp_path / "results" / "career").glob("*"))
    assert [p.suffix for p in saved] == [".md"]  # 채택 기록은 남는다
    assert any("Word·한글 문서를 만들지 못해 기록(.md)만 저장" in w.value for w in at.warning)
    assert any("💾 결과 파일 1개" in i.value for i in at.info)
