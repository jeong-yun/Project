"""tests/conftest.py 의 안전장치 — 테스트가 어떤 경우에도 `.env` 의 실제 키·진짜 모델을 쓰지 못하게 막는지 확인한다.

배경(2026-11-13): 앱 화면 테스트(AppTest 로 app.py 실행)가 돌면 app.py 의 load_env() 가 `.env` 를 읽어
`set_keys()` 로 지운 키까지 되살렸고, 그 키가 다음 테스트로 넘어가 가짜 모델을 끼우지 않은 테스트가 진짜 모델을 불렀다.
"""

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core import config, llm_factory
from core.schemas import ModelSpec
from tests.conftest import BLOCKED_MODEL, SECRET_ENV

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")
FAKE = "fake-not-a-real-key"


def keys_in_env() -> list[str]:
    return [name for name in SECRET_ENV if os.environ.get(name)]


def run_inner(tmp_path: Path, source: str, **env: str) -> str:
    """임시 테스트 파일을 이 폴더의 conftest 와 함께 새 pytest 프로세스로 돌리고 출력을 돌려준다 (GITHUB_ACTIONS 는 넘긴 것만)."""
    path = tmp_path / "test_inner.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    full_env = {k: v for k, v in os.environ.items() if k != "GITHUB_ACTIONS"} | {"PYTHONIOENCODING": "utf-8"} | env
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "tests.conftest", "-p", "no:cacheprovider", "--rootdir", str(tmp_path), str(path)],
        cwd=ROOT, env=full_env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
    )
    return proc.stdout + proc.stderr


# ─── `.env` 를 읽지 않는다 ───────────────────────────────────────────


def test_load_env_reads_no_dotenv_file_in_tests(tmp_path, monkeypatch):
    (tmp_path / "office").mkdir()
    (tmp_path / "office" / ".env").write_text(f"OPENAI_API_KEY={FAKE}\n", encoding="utf-8")
    (tmp_path / ".env").write_text(f"GOOGLE_API_KEY={FAKE}\n", encoding="utf-8")  # 상위 폴더 .env 도
    monkeypatch.setattr(config, "BASE_DIR", tmp_path / "office")
    config.load_env()
    assert keys_in_env() == []


def test_app_screen_leaves_no_dotenv_keys(no_real_keys, set_keys):
    """앱 화면을 열어도(app.py 의 load_env) 키가 환경에 올라오지 않는다 — `.env` 가 있는 폴더에서도, 없는 폴더에서도."""
    set_keys()  # 화면 테스트들이 하는 것처럼 키를 지운 뒤 앱을 연다
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception, at.exception
    assert keys_in_env() == []
    # 막은 자리가 app.py 가 실제로 지나는 자리인지 — `.env` 가 없는 폴더(GitHub 검사)에서도 이 확인은 빈말이 아니다
    assert str(config.BASE_DIR / ".env") in no_real_keys


def test_provider_keys_are_cleared_at_test_start_and_set_keys_still_works(set_keys):
    assert keys_in_env() == []
    set_keys("google", "freellm")
    assert sorted(keys_in_env()) == ["FREELLMAPI_API_KEY", "GOOGLE_API_KEY"] and os.environ["GOOGLE_API_KEY"] == "test-key"


# ─── 가짜를 끼우지 않으면 모델 생성이 실패한다 ───────────────────────


def test_model_creation_is_blocked_without_fake(models, set_keys, no_real_models):
    set_keys("openai")
    with pytest.raises(RuntimeError, match=BLOCKED_MODEL):
        llm_factory.create_chat_model(ModelSpec(provider="openai", model="gpt-5.4-mini"), models)
    assert no_real_models == ["openai · gpt-5.4-mini"]
    no_real_models.clear()  # 이 테스트는 막히는 것을 본 것이므로 끝날 때의 실패 처리는 끈다


def test_swallowed_block_still_fails_the_test(tmp_path):
    """앱 코드는 모델 오류를 삼키고 다음 후보·낱말 비교로 넘어간다 → 삼켜도 테스트는 끝날 때 실패해야 한다."""
    out = run_inner(tmp_path, """
        from core import llm_factory
        from core.schemas import ModelSpec


        def test_swallows(models, set_keys):
            set_keys("openai")
            try:
                llm_factory.create_chat_model(ModelSpec(provider="openai", model="gpt-5.4-mini"), models)
            except RuntimeError:
                pass


        def test_fake_model_is_fine(models, use_script):
            from tests.fakes import Script

            use_script(Script(text="ok"))
            assert llm_factory.create_chat_model(ModelSpec(provider="openai", model="gpt-5.4-mini"), models).invoke("q").content == "ok"


        def test_allowed_object(models, set_keys, allow_model_objects):
            set_keys("openai")
            assert llm_factory.create_chat_model(ModelSpec(provider="openai", model="gpt-5.4-mini"), models).model_name == "gpt-5.4-mini"
        """)
    assert re.search(r"3 passed.* 1 error", out), out  # 본문은 셋 다 통과, 삼킨 테스트는 끝날 때(teardown) 오류
    assert "ERROR at teardown of test_swallows" in out and BLOCKED_MODEL in out and "openai · gpt-5.4-mini" in out
