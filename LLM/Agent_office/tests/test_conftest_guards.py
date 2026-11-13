"""tests/conftest.py 의 안전장치 — 테스트가 어떤 경우에도 `.env` 의 실제 키·진짜 모델을 쓰지 못하게 막는지 확인한다.

배경(2026-11-13): 앱 화면 테스트(AppTest 로 app.py 실행)가 돌면 app.py 의 load_env() 가 `.env` 를 읽어
`set_keys()` 로 지운 키까지 되살렸고, 그 키가 다음 테스트로 넘어가 가짜 모델을 끼우지 않은 테스트가 진짜 모델을 불렀다.

맨 아래: GitHub 검사(GITHUB_ACTIONS)에서 실패한 테스트 이름을 주석(::error)으로 찍는 것 — 로그인 없이 실패 원인을 알 수 있게.
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
    assert "::error" not in out  # GitHub 검사가 아니면(GITHUB_ACTIONS 없음) 주석 줄을 찍지 않는다


# ─── GitHub 검사 실패 때 테스트 이름이 주석으로 보인다 ───────────────


def test_github_run_prints_failed_test_names_as_annotations(tmp_path):
    out = run_inner(tmp_path, """
        import pytest


        def test_ok():
            pass


        def test_wrong_number():
            print("captured output must not hide the annotation")
            assert 1 + 1 == 3, "sum is wrong\\nsecond line stays out"


        @pytest.fixture
        def broken():
            raise ValueError("setup broke with key " + "x9" * 20)


        def test_setup_error(broken):
            pass
        """, GITHUB_ACTIONS="true")
    lines = [line for line in out.splitlines() if line.startswith("::error ")]
    assert len(lines) == 2, out
    assert lines[0].startswith("::error title=pytest 실패::") and "test_inner.py::test_wrong_number — AssertionError: sum is wrong" in lines[0]
    assert "second line" not in lines[0]
    assert "test_inner.py::test_setup_error — ValueError: setup broke with key ***" in lines[1]  # 긴 영숫자 덩어리(키일 수 있는 값)는 가린다
    assert "test_ok" not in "".join(lines)


def test_github_annotations_are_capped_and_escaped():
    from types import SimpleNamespace

    from tests.conftest import GITHUB_ERROR_LIMIT, github_error_lines

    def report(i: int, message: str = "AssertionError: x"):
        return SimpleNamespace(nodeid=f"tests/test_a.py::test_{i}", longrepr=SimpleNamespace(reprcrash=SimpleNamespace(message=message)))

    lines = github_error_lines([report(i) for i in range(25)])
    assert len(lines) == GITHUB_ERROR_LIMIT == 10  # 한 단계에 보이는 오류 주석 한도
    assert lines[8].endswith("tests/test_a.py::test_8 — AssertionError: x") and lines[9].startswith("::error title=pytest 실패::외 16개")
    (line,) = github_error_lines([report(0, "ValueError: 50% done\nnext line " + "y" * 300)])
    assert line == "::error title=pytest 실패::tests/test_a.py::test_0 — ValueError: 50%25 done"  # 첫 줄만, % 는 GitHub 규칙대로
    (line,) = github_error_lines([SimpleNamespace(nodeid="tests/test_b.py", longrepr="ImportError: no module\nmore")])  # 모으기 오류
    assert line.endswith("tests/test_b.py — ImportError: no module")
    (line,) = github_error_lines([report(1, "RuntimeError: " + "가" * 500)])
    assert len(line.split(" — ", 1)[1]) == 200  # 오류 첫 줄은 200자까지
