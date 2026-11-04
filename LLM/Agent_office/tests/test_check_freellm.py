"""scripts/check_freellm.py — 가짜 FreeLLMAPI 서버(내 PC 안의 임시 HTTP 서버)로 켜짐·키·모델 점검을 확인한다."""

import importlib.util
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_freellm.py"
GOOD_KEY = "freellmapi-test-good-key-1234"


def _load():
    spec = importlib.util.spec_from_file_location("check_freellm", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_freellm = _load()


class FakeFreeLLM(BaseHTTPRequestHandler):
    models = ["gpt-oss-20b", "gemini-3.5-flash"]

    def log_message(self, *args):  # 테스트 출력 조용히
        pass

    def _send(self, status: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _authorized(self) -> bool:
        return self.headers.get("Authorization") == f"Bearer {GOOD_KEY}"

    def do_GET(self):
        if self.path == "/api/ping":
            self._send(200, {"ok": True})
        elif self.path == "/v1/models":
            if not self._authorized():
                self._send(401, {"error": {"message": "invalid api key"}})
            else:
                self._send(200, {"object": "list", "data": [{"id": m, "object": "model"} for m in self.models]})
        else:
            self._send(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if self.path != "/v1/chat/completions" or not self._authorized():
            self._send(401, {})
        elif body.get("model") == "busy-model":
            self._send(429, {"error": {"message": "rate limited"}})
        elif body.get("model") not in self.models:
            self._send(404, {"error": {"message": "model not found"}})
        else:
            self._send(200, {"choices": [{"message": {"role": "assistant", "content": "ok"}}]})


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeFreeLLM)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()
    srv.server_close()


def _marks(lines):
    return [mark for mark, _ in lines]


def _text(lines):
    return "\n".join(text for _, text in lines)


def test_all_good_and_key_never_printed(server):
    lines = check_freellm.check(server, GOOD_KEY, ["gpt-oss-20b", "gemini-3.5-flash"])
    assert _marks(lines) == ["✅"] * 4
    assert "모델 2개" in _text(lines) and "모두 있음" in _text(lines)
    assert GOOD_KEY not in _text(lines)


def test_wrong_key_is_reported_without_echoing_it(server):
    lines = check_freellm.check(server, "freellmapi-wrong-key", [])
    assert lines[-1][0] == "❌" and "통합 키가 맞지 않습니다 (401)" in lines[-1][1]
    assert "freellmapi-wrong-key" not in _text(lines)


def test_empty_key(server):
    lines = check_freellm.check(server, "", [])
    assert lines[-1][0] == "❌" and "FREELLMAPI_API_KEY" in lines[-1][1]


def test_missing_configured_model_is_a_warning(server):
    lines = check_freellm.check(server, GOOD_KEY, ["gpt-oss-20b", "gemini-2.5-pro"])
    assert lines[-1][0] == "⚠️" and "gemini-2.5-pro" in lines[-1][1]


def test_chat_ok_429_and_404(server):
    assert check_freellm.check(server, GOOD_KEY, [], chat_model="gpt-oss-20b")[-1][0] == "✅"
    busy = check_freellm.check(server, GOOD_KEY, [], chat_model="busy-model")[-1]
    assert busy[0] == "⚠️" and "429" in busy[1]
    gone = check_freellm.check(server, GOOD_KEY, [], chat_model="nope")[-1]
    assert gone[0] == "❌" and "404" in gone[1]


def test_server_off():
    with socket.socket() as s:  # 비어 있는 포트를 하나 받아 닫는다 → 그 주소엔 아무도 없다
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    lines = check_freellm.check(f"http://127.0.0.1:{port}/v1", GOOD_KEY, [], timeout=1.0)
    assert len(lines) == 1 and lines[0][0] == "❌" and "docker compose up -d" in lines[0][1]


def test_configured_free_models_reads_real_models_yaml():
    names = check_freellm.configured_free_models(SCRIPT.parent.parent / "config" / "models.yaml")
    assert names and len(names) == len(set(names))
    assert all(":" not in n for n in names)


def test_configured_free_models_both_forms(tmp_path):
    f = tmp_path / "models.yaml"
    f.write_text(
        "model_tiers:\n  - [freellm:a, openai:x]\n"
        "ranks:\n  r:\n    fallbacks:\n      - { provider: freellm, model: b }\n      - { provider: freellm, model: a }\n",
        encoding="utf-8",
    )
    assert check_freellm.configured_free_models(f) == ["a", "b"]
    assert check_freellm.configured_free_models(tmp_path / "none.yaml") == []
