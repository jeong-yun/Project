"""테스트가 실제 키·실제 모델을 쓰지 못하는지 — tests/conftest.py 의 막음 장치를 확인한다.

여기서 쓰는 키 값은 모두 가짜다. 실제 `.env` 는 읽지도 만들지도 않는다 (임시 폴더의 가짜 `.env` 만 쓴다).
"""

import os

import pytest
from dotenv.main import load_dotenv as load_dotenv_bound_early  # 막기 전에 미리 잡아 둔 원래 함수

from fakes import FAKE_KEY


def test_env_file_is_never_read(tmp_path):
    """load_dotenv 를 어떤 이름으로 부르든 `.env` 의 값이 환경 변수로 들어오지 않는다."""
    import dotenv

    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-envfile-fake\nLANGCHAIN_API_KEY=ls-envfile-fake\n", encoding="utf-8")

    dotenv.load_dotenv(env_file)
    dotenv.load_dotenv(env_file, override=True)
    load_dotenv_bound_early(env_file, override=True)  # 다른 모듈이 미리 import 해 둔 경우

    assert "OPENAI_API_KEY" not in os.environ
    assert "LANGCHAIN_API_KEY" not in os.environ


def test_provider_keys_are_removed_before_each_test():
    for name in ("OPENAI_API_KEY", "LANGCHAIN_API_KEY", "LANGSMITH_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        assert name not in os.environ
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"


def test_making_a_real_model_fails_even_with_a_key(monkeypatch, no_real_calls):
    """키가 있어도(가짜 값) 진짜 채팅 모델·임베딩·OpenAI 클라이언트(TTS)를 만들면 바로 실패한다."""
    import langchain_openai
    import openai

    import common

    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    makers = [
        lambda: common.make_llm("gpt-4.1"),
        lambda: langchain_openai.ChatOpenAI(model="gpt-4.1"),
        lambda: langchain_openai.OpenAIEmbeddings(),
        lambda: openai.OpenAI(),
        lambda: openai.AsyncOpenAI(),
    ]
    for make in makers:
        with pytest.raises(AssertionError, match="가짜"):
            make()

    assert len(no_real_calls) == len(makers)
    no_real_calls.clear()  # 일부러 낸 것이므로 지운다 (안 지우면 이 테스트가 끝날 때 실패한다)


def test_swallowed_attempt_is_still_recorded(no_real_calls):
    """앱 코드가 오류를 삼켜도(try/except) 진짜 모델을 만들려 한 흔적은 남아, 그 테스트는 끝날 때 실패한다."""
    from english_word_maker import tts_utils

    assert tts_utils.tts_bytes("hello") == b""  # tts_bytes 는 오류를 삼키고 빈 소리를 돌려준다
    assert len(no_real_calls) == 1 and "OpenAI" in no_real_calls[0]
    no_real_calls.clear()


def test_network_is_blocked():
    import httpx
    import requests

    with pytest.raises(AssertionError, match="바깥"):
        httpx.get("http://127.0.0.1:9/")
    with pytest.raises(AssertionError, match="바깥"):
        requests.get("http://127.0.0.1:9/")
