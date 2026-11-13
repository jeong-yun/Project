"""화면이 키 없이 열리는지 (streamlit.testing.v1.AppTest), 그리고 가짜 모델로 한 번 돌려 보기."""

import json

import pytest
from streamlit.testing.v1 import AppTest

import common
from fakes import FAKE_KEY, FakeTTSClient, fake_chat

PAGES = ["app.py", "pages/1_english_word_maker.py", "pages/2_ai_safety_eval.py", "pages/3_rag_docs_qa.py"]


@pytest.fixture(autouse=True)
def fake_tts(monkeypatch):
    """단어장 화면은 열릴 때 발음 ZIP 을 준비하며 발음(TTS) 클라이언트를 쓴다 → 가짜를 끼운다."""
    from english_word_maker import tts_utils

    client = FakeTTSClient()
    monkeypatch.setattr(tts_utils, "_client", client)
    return client


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
    assert "공격 프롬프트 9개" in text and "18번" in text


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
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)

    at = open_page("pages/2_ai_safety_eval.py")
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.metric[0].value == "100.0%"
    assert len(at.dataframe[-1].value) == 9


def test_safety_dashboard_rag_target_shows_rag_categories_separately(monkeypatch):
    from ai_safety import judge

    verdict = json.dumps({"verdict": "PASS", "severity": "none", "reason": "지시를 따르지 않음"}, ensure_ascii=False)
    # 대상: 심은 문구를 그대로 따라 쓰는 가짜 모델 → 인젝션 3건 중 그 문구가 걸린 건은 코드 검사가 FAIL
    fakes = iter([fake_chat("문서에 없습니다. MOONLIB-OVERRIDE-42"), fake_chat(verdict)])
    monkeypatch.setattr(judge, "make_llm", lambda *a, **k: next(fakes))
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)

    at = open_page("pages/2_ai_safety_eval.py")
    at.radio[0].set_value("rag").run()
    text = all_text(at)
    assert "공격 프롬프트 7개" in text and "14번" in text

    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    by_category = at.dataframe[0].value
    assert list(by_category["범주"]) == ["RAG 간접 프롬프트 인젝션", "RAG 문서에 없는 질문", "RAG 출처 일치"]
    rates = dict(zip(by_category["범주"], by_category["방어율(%)"]))
    assert rates["RAG 간접 프롬프트 인젝션"] < 100.0  # rag-inj-1 은 심은 문구가 나와 FAIL
    table = at.dataframe[-1].value
    assert len(table) == 7 and "코드 검사" in table.columns

    at.radio[0].set_value("chat").run()  # 일반 채팅으로 돌아가면 RAG 결과를 섞어 보이지 않는다
    assert "아직 없습니다" in all_text(at)


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
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
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
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)

    def must_not_be_called(*a, **k):
        raise AssertionError("찾은 조각이 없으면 모델을 만들지 않아야 한다")

    monkeypatch.setattr(common, "make_llm", must_not_be_called)
    at = open_page(RAG_PAGE)
    at.chat_input[0].set_value("zzz qqq").run()
    assert not at.exception
    assert "문서에 없습니다" in all_text(at)
