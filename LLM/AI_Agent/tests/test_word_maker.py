"""단어장: 답변 파싱, 저장된 예시 단어장 형식, 내려받기, 발음 모듈."""

import importlib
import json
import sys

import pytest

import common
from english_word_maker import export_doc
from english_word_maker.parsing import parse_yaml_style_answer
from fakes import FakeTTSClient

ANSWER = """1) 단어(Word): answer
2) 분야(Domain): 일반
3) 품사(Part of Speech): n, v
4) 발음 기호(IPA): /ˈænsər/
5) 한글 의미(Senses_Kor):
  n) 대답
  v) 대답하다
6) 영어 의미(Senses_En):
  n) a reply to a question
  v) to reply to a question
7) 난이도(CEFR/TOEIC 등): A1
8) 빈출도(Frequency): High
9) 어원(Etymology): 고대 영어 andswaru
10) 예시(Examples_Raw):
  {"noun": "I got an answer.", "verb": "Please answer me."}
11) 유의어(Synonyms): reply, response
12) 반의어(Antonyms): question
13) 콜로케이션(Collocations): correct answer
14) 파생어(Derivatives): answerable
15) 태그(Tags): 기본어휘
16) 유사단어(Confusable): anger[SP], ensure[PR]
17) 문의 횟수(Lookup Count): 0
"""


def test_parse_answer_into_fields():
    got = parse_yaml_style_answer(ANSWER)
    assert got["word"] == "answer"
    assert got["senses_ko"] == {"noun": "대답", "verb": "대답하다"}
    assert got["senses_en"]["verb"] == "to reply to a question"
    assert got["examples"] == {"noun": "I got an answer.", "verb": "Please answer me."}
    assert got["synonyms"] == ["reply", "response"]
    assert got["similar_words"] == ["anger[SP]", "ensure[PR]"]


def test_saved_sample_vocab_uses_current_field_names():
    """저장소에 든 예시 단어장은 지금 화면이 읽는 항목 이름(뜻·예문이 품사별 dict)이어야 한다."""
    vocab = json.loads((common.BASE_DIR / "English_word" / "vocab.json").read_text(encoding="utf-8"))
    assert vocab
    for key, entry in vocab.items():
        assert "meaning" not in entry and "senses" not in entry, key
        for name in ("senses_ko", "senses_en", "examples"):
            assert isinstance(entry.get(name), dict) and entry[name], f"{key}.{name}"


def test_word_and_excel_export():
    vocab = {"answer": {**parse_yaml_style_answer(ANSWER), "lookup_count": 1}}
    assert export_doc.get_word_bytes_from_vocab(vocab).getvalue()[:2] == b"PK"


def test_pdf_export_with_font_found_or_clear_error(monkeypatch):
    vocab = {"answer": {**parse_yaml_style_answer(ANSWER), "lookup_count": 1}}
    if export_doc.find_korean_font():
        assert export_doc.get_pdf_bytes_from_vocab(vocab).getvalue()[:4] == b"%PDF"

    monkeypatch.setattr(export_doc, "KOREAN_FONT_CANDIDATES", ("/없는/글꼴.ttf",))
    with pytest.raises(export_doc.KoreanFontNotFound) as err:
        export_doc.get_pdf_bytes_from_vocab(vocab)
    message = str(err.value)
    assert "한글 글꼴" in message and "Excel" in message  # 무엇이 문제이고, 대신 무엇을 하면 되는지


def test_tts_module_imports_without_key_and_makes_client_lazily(monkeypatch):
    """키가 없어도 불러와지고, 폴더·클라이언트는 처음 쓸 때 만든다. 한 번 만든 소리는 다시 부르지 않는다.

    (키가 없어 클라이언트를 못 만들 때 빈 소리를 돌려주는 것은 tests/test_no_real_calls.py 에서 본다.)
    """
    sys.modules.pop("english_word_maker.tts_utils", None)
    tts = importlib.import_module("english_word_maker.tts_utils")  # 여기서 OpenAI() 를 만들면 막음 장치에 걸린다
    assert tts._client is None
    assert not (common.VOCAB_DIR / "audio").exists()

    client = FakeTTSClient()
    monkeypatch.setattr(tts, "_client", client)
    assert tts.tts_cached("hello.mp3", "hello") == b"mp3"
    assert tts.tts_cached("hello.mp3", "hello") == b"mp3"  # 두 번째는 저장된 파일
    assert client.calls == 1
    assert (common.VOCAB_DIR / "audio" / "hello.mp3").exists()
