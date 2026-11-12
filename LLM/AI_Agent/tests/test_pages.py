"""화면이 키 없이 열리는지 (streamlit.testing.v1.AppTest), 그리고 가짜 모델로 한 번 돌려 보기."""

import json

import pytest
from streamlit.testing.v1 import AppTest

import common
from fakes import fake_chat

PAGES = ["app.py", "pages/1_english_word_maker.py", "pages/2_ai_safety_eval.py", "pages/3_rag_docs_qa.py"]


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


# ---------------- 문서 Q&A (RAG) ----------------

RAG_PAGE = "pages/3_rag_docs_qa.py"


def test_rag_page_tells_what_is_sent_to_the_model():
    text = all_text(open_page(RAG_PAGE))
    assert "전송됩니다" in text and "회사 문서" in text
    assert "문서 7개" in text  # 기본 문서 묶음을 읽었다


def test_rag_page_guides_when_default_folder_is_missing(monkeypatch, tmp_path):
    from rag import documents

    monkeypatch.setattr(documents, "DEFAULT_DOCS_DIR", tmp_path / "없는폴더")
    at = open_page(RAG_PAGE)
    assert not at.exception
    assert "직접 올리기" in " ".join(w.value for w in at.warning)
    assert at.chat_input[0].disabled


def test_rag_page_without_key_still_shows_found_chunks():
    at = open_page(RAG_PAGE)
    at.chat_input[0].set_value("인공지능 기본법은 언제 시행되나?").run()
    assert not at.exception
    assert "OPENAI_API_KEY" in at.error[0].value
    found = at.dataframe[0].value
    assert list(found["번호"]) == [1, 2, 3]
    assert found["파일"][0].startswith("06_")


def test_rag_page_answers_with_fake_model_and_checks_citations(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key")
    replies = iter(["2026-01-22 에 시행됩니다 [1].", "2030-12-31 에 시행됩니다 [9]."])
    monkeypatch.setattr(common, "make_llm", lambda *a, **k: fake_chat(next(replies)))

    at = open_page(RAG_PAGE)
    at.chat_input[0].set_value("인공지능 기본법은 언제 시행되나?").run()
    assert not at.exception, [e.value for e in at.exception]
    assert "2026-01-22 에 시행됩니다 [1]." in all_text(at)
    assert not at.warning  # 출처 검사 통과

    at.chat_input[0].set_value("인공지능 기본법은 언제 시행되나?").run()  # 이번에는 지어낸 답
    warning = " ".join(w.value for w in at.warning)
    assert "없는 출처 번호" in warning and "2030" in warning


def test_rag_page_says_not_in_docs_without_calling_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key")

    def must_not_be_called(*a, **k):
        raise AssertionError("찾은 조각이 없으면 모델을 만들지 않아야 한다")

    monkeypatch.setattr(common, "make_llm", must_not_be_called)
    at = open_page(RAG_PAGE)
    at.chat_input[0].set_value("zzz qqq").run()
    assert not at.exception
    assert "문서에 없습니다" in all_text(at)
