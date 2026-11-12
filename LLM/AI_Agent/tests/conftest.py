"""테스트 공통 준비물 — 실제 모델·임베딩·발음(TTS)·기록 서버를 절대 부르지 않게 막는다.

- 환경 변수의 키를 지우고, 화면이 부르는 load_dotenv() 가 .env 를 읽지 못하게 한다.
- 그래도 누가 바깥으로 요청을 보내면 바로 실패하게 httpx 전송을 막는다 (OpenAI·LangSmith 모두 httpx 를 쓴다).
- 단어장 저장 폴더를 임시 폴더로 바꿔, 테스트가 저장소의 vocab.json 을 고쳐 쓰지 않게 한다.
"""

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KEY_NAMES = (
    "OPENAI_API_KEY",
    "LANGCHAIN_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_TRACING",
    "LANGCHAIN_ENDPOINT",
)


def _no_network(*args, **kwargs):
    raise AssertionError("테스트가 바깥으로 요청을 보내려 했습니다 — 가짜 모델만 써야 합니다.")


@pytest.fixture(autouse=True)
def no_real_calls(monkeypatch, tmp_path):
    import dotenv
    import httpx

    for name in KEY_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(httpx.Client, "send", _no_network)
    monkeypatch.setattr(httpx.AsyncClient, "send", _no_network)

    import common

    vocab_dir = tmp_path / "English_word"
    vocab_dir.mkdir()
    shutil.copy(common.VOCAB_DIR / "vocab.json", vocab_dir / "vocab.json")
    monkeypatch.setattr(common, "VOCAB_DIR", vocab_dir)
    yield

