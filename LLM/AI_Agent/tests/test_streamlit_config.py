"""`.streamlit/config.toml` 이 실제로 적용되는지 확인한다 — 새 프로세스에서 Streamlit 이 읽은 값으로 본다.

파일이 빠지거나, 값이 바뀌거나, 새 Streamlit 에서 옵션이 없어지면 실패한다. 모델을 부르지 않는다.
"""

import subprocess
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent


def _streamlit_reads(expression: str) -> str:
    """이 폴더(AI_Agent)에서 Streamlit 이 실제로 읽은 설정으로 expression 을 계산한 결과."""
    done = subprocess.run([sys.executable, "-c", f"from streamlit import config; print({expression})"],
                          cwd=APP_DIR, capture_output=True, text=True, timeout=120)
    return done.stdout.strip() or done.stderr


def test_streamlit_config_opens_app_only_on_this_pc():
    """앱을 이 PC 안쪽 주소(127.0.0.1)에서만 연다.

    앱에는 로그인이 없어서, 같은 와이파이·랜의 다른 기기가 들어오면 내 OpenAI 키로 모델·발음(TTS)을 부를 수 있다.
    Streamlit 기본값은 모든 네트워크 주소에서 여는 것이다 (2026-11-28 보안 점검).
    """
    assert _streamlit_reads("config.get_option('server.address')") == "127.0.0.1"


def test_streamlit_config_accepts_only_this_pc_host_names():
    """주소창 이름이 localhost·127.0.0.1 일 때만 화면 연결(웹소켓)을 받는다.

    이 PC 안쪽 주소에서만 열어도, 브라우저로 연 다른 사이트가 자기 이름을 127.0.0.1 로 다시 가리키게 하면(DNS 리바인딩)
    그 이름으로 이 앱에 붙을 수 있다. 설정값과, Streamlit 이 그 값으로 실제로 거르는지를 함께 본다 (포트는 보지 않는다).
    """
    allowed = "__import__('importlib').import_module('streamlit.web.server.starlette.starlette_websocket')._is_origin_allowed"
    assert _streamlit_reads(
        "(sorted(config.get_option('server.allowedHosts')), "
        f"{allowed}('http://localhost:8501', 'localhost:8501'), "
        f"{allowed}('http://127.0.0.1:8600', '127.0.0.1:8600'), "
        f"{allowed}('http://rebind.example:8501', 'rebind.example:8501'))"
    ) == "(['127.0.0.1', 'localhost'], True, True, False)"
