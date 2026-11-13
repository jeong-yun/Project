"""테스트 공통 준비물 — 실제 모델·임베딩·발음(TTS)·기록 서버를 절대 부르지 않게 막는다.

화면(app.py·pages/)은 열릴 때 load_dotenv() 를 부른다. 그대로 두면 사용자의 작업 폴더에 있는 `.env` 의
실제 키가 테스트 프로세스에 들어와, 가짜 모델을 끼우지 않은 테스트가 진짜 모델을 부를 수 있다. 그래서 네 겹으로 막는다.

1. 제공처 키 환경 변수를 지운다.
2. `.env` 를 읽지 못하게 한다 — load_dotenv 를 어떤 이름으로 부르든 환경 변수로 들어오지 않는다.
3. 진짜 모델(ChatOpenAI·임베딩·OpenAI 클라이언트)을 만들면 바로 실패한다. 앱 코드가 그 오류를 삼켜도
   흔적을 적어 두었다가 테스트가 끝날 때 실패시킨다. 모델이 필요한 테스트는 가짜를 끼워야 한다 (tests/fakes.py).
4. 그래도 바깥으로 나가는 요청(httpx·requests)은 바로 실패한다.

그리고 단어장 저장 폴더를 임시 폴더로 바꿔, 테스트가 저장소의 vocab.json 을 고쳐 쓰지 않게 한다.
"""

import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fakes import FAKE_KEY  # noqa: E402

KEY_NAMES = (
    "OPENAI_API_KEY",
    "OPENAI_ORG_ID",
    "OPENAI_BASE_URL",
    "AZURE_OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "UPSTAGE_API_KEY",
    "COHERE_API_KEY",
    "HUGGINGFACEHUB_API_TOKEN",
    "LANGCHAIN_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_ENDPOINT",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_TRACING",
)


def _no_network(*args, **kwargs):
    raise AssertionError("테스트가 바깥으로 요청을 보내려 했습니다 — 가짜 모델만 써야 합니다.")


@pytest.fixture(autouse=True)
def no_real_calls(monkeypatch, tmp_path):
    """모든 테스트에 자동으로 걸린다. 돌려주는 값 = 진짜 모델을 만들려 한 흔적 목록 (비어 있어야 한다)."""
    import dotenv
    import dotenv.main
    import httpx
    import langchain_openai
    import openai
    import requests

    # 1) 키 지우기 + 호출 기록(LangSmith) 끄기
    for name in KEY_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

    # 2) .env 읽기 막기 — 화면이 부르는 dotenv.load_dotenv 와, 다른 모듈이 미리 잡아 둔 함수 둘 다
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(dotenv.main.DotEnv, "set_as_environment_variables", lambda self: False)

    # 3) 진짜 모델 만들기 막기
    attempts: list[str] = []

    def forbid(cls):
        def refuse(self, *args, **kwargs):
            attempts.append(cls.__name__)
            raise AssertionError(f"테스트가 진짜 {cls.__name__} 을(를) 만들려 했습니다 — 가짜 모델을 끼워야 합니다 (tests/fakes.py).")

        monkeypatch.setattr(cls, "__init__", refuse)

    for cls in (langchain_openai.ChatOpenAI, langchain_openai.OpenAIEmbeddings, openai.OpenAI, openai.AsyncOpenAI):
        forbid(cls)

    # 4) 바깥 요청 막기
    monkeypatch.setattr(httpx.Client, "send", _no_network)
    monkeypatch.setattr(httpx.AsyncClient, "send", _no_network)
    monkeypatch.setattr(requests.Session, "send", _no_network)

    # 단어장 저장 폴더 → 임시 폴더
    import common

    vocab_dir = tmp_path / "English_word"
    vocab_dir.mkdir()
    shutil.copy(common.VOCAB_DIR / "vocab.json", vocab_dir / "vocab.json")
    monkeypatch.setattr(common, "VOCAB_DIR", vocab_dir)

    yield attempts

    # 테스트가 끝난 뒤: 진짜 모델을 만들려 한 흔적이 없고, 실제 키가 환경 변수에 들어와 있지 않아야 한다
    assert not attempts, f"테스트 도중 진짜 모델을 만들려 했습니다: {attempts} — 가짜 모델을 끼워 주세요."
    leaked = [name for name in KEY_NAMES if os.environ.get(name) not in (None, FAKE_KEY)]
    assert not leaked, f"테스트 도중 환경 변수에 키가 들어왔습니다: {leaked} (.env 를 읽었을 수 있습니다)"
