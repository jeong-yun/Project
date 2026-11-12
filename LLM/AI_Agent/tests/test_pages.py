"""화면이 키 없이 열리는지 (streamlit.testing.v1.AppTest), 그리고 가짜 모델로 한 번 돌려 보기."""

import json

import pytest
from streamlit.testing.v1 import AppTest

import common
from fakes import fake_chat

PAGES = ["app.py", "pages/1_english_word_maker.py", "pages/2_ai_safety_eval.py"]


def open_page(rel: str) -> AppTest:
    return AppTest.from_file(str(common.BASE_DIR / rel), default_timeout=120).run()


def all_text(at: AppTest) -> str:
    parts = []
    for kind in ("title", "header", "subheader", "markdown", "caption", "info", "warning", "error", "success"):
        parts += [el.value for el in getattr(at, kind)]
    return "\n".join(str(p) for p in parts)


@pytest.mark.parametrize("rel", PAGES)
def test_page_opens_without_key(rel):
    at = open_page(rel)
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.title) == 1


def test_word_page_shows_saved_sample_word():
    at = open_page("pages/1_english_word_maker.py")
    table = at.dataframe[0].value
    assert list(table["Word"]) == ["answer"]
    assert json.loads(table["Senses_Ko"][0])  # 한글 뜻 칸이 비어 있지 않다


def test_safety_dashboard_tells_call_count_before_running():
    at = open_page("pages/2_ai_safety_eval.py")
    text = all_text(at)
    assert "공격 프롬프트 9개" in text and "호출 18번" in text


def test_safety_dashboard_without_key_explains_what_to_do():
    at = open_page("pages/2_ai_safety_eval.py")
    at.button[0].click().run()
    assert not at.exception
    assert "OPENAI_API_KEY" in all_text(at)


def test_safety_dashboard_runs_with_fake_models(monkeypatch):
    from ai_safety import judge

    verdict = json.dumps({"verdict": "PASS", "severity": "none", "reason": "거부함"}, ensure_ascii=False)
    fakes = iter([fake_chat("도와드릴 수 없습니다."), fake_chat(verdict)])  # 대상, 심판 순서로 만든다
    monkeypatch.setattr(judge, "make_llm", lambda *a, **k: next(fakes))
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key")

    at = open_page("pages/2_ai_safety_eval.py")
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.metric[0].value == "100.0%"
    assert len(at.dataframe[-1].value) == 9
