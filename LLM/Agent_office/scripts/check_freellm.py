"""FreeLLMAPI(무료 모델 서버) 연결 점검 — 서버가 켜져 있는지, 통합 키가 맞는지, 설정한 무료 모델이 있는지 한 번에 본다.

실행 (Agent_office 폴더에서):
    python scripts/check_freellm.py                 # 켜짐·키·모델 목록 점검 (무료 한도를 쓰지 않음)
    python scripts/check_freellm.py --chat gpt-oss-20b   # 위 점검 + 그 모델에 짧은 질문 1회 (한도 1회 사용)

주소·키는 .env 의 FREELLMAPI_BASE_URL(기본 http://127.0.0.1:3001/v1)·FREELLMAPI_API_KEY 를 읽는다.
키 값은 화면에 절대 찍지 않는다 (있다/없다, 맞다/틀리다만).
설치 방법: docs/FREELLM_설치.md
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_BASE_URL = "http://127.0.0.1:3001/v1"
GUIDE = "docs/FREELLM_설치.md"
OK, WARN, FAIL = "✅", "⚠️", "❌"


def configured_free_models(models_yaml: Path) -> list[str]:
    """config/models.yaml 에서 freellm 모델 이름을 모두 찾는다 (직급 후보·강도 줄 모두, 순서 유지·중복 제거)."""
    try:
        data = yaml.safe_load(models_yaml.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return []
    found: list[str] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            if node.get("provider") == "freellm" and node.get("model"):
                found.append(str(node["model"]))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str) and node.startswith("freellm:"):
            found.append(node.split(":", 1)[1])

    walk(data)
    return list(dict.fromkeys(found))


def _request(url: str, key: str = "", body: dict | None = None, timeout: float = 5.0) -> tuple[int, dict | None]:
    """(HTTP 상태, JSON 본문 또는 None). 서버에 닿지 못하면 OSError 를 그대로 올린다."""
    headers = {"Accept": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status, raw = resp.status, resp.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    try:
        return status, json.loads(raw.decode("utf-8") or "null")
    except (ValueError, UnicodeDecodeError):
        return status, None


def check(base_url: str, key: str, wanted: list[str], chat_model: str = "", timeout: float = 5.0) -> list[tuple[str, str]]:
    """점검 결과 [(표시, 한국어 설명)]. 표시는 ✅ / ⚠️ / ❌. 키 값은 결과에 넣지 않는다."""
    lines: list[tuple[str, str]] = []
    base = base_url.rstrip("/")
    root = base[: -len("/v1")] if base.endswith("/v1") else base

    # 1) 서버가 켜져 있나 — /api/ping 은 키 없이 답한다 (docker-compose 의 healthcheck 와 같은 주소)
    try:
        status, _ = _request(root + "/api/ping", timeout=timeout)
    except OSError as e:
        lines.append((FAIL, f"서버에 닿지 못함 ({root}) — {type(e).__name__}. "
                            "Docker Desktop 을 켜고 freellmapi 폴더에서 `docker compose up -d` 를 실행하세요."))
        return lines
    if status != 200:
        lines.append((WARN, f"서버는 켜져 있지만 /api/ping 이 {status} — 아직 시작 중일 수 있습니다. 30초 뒤 다시 확인하세요."))
    else:
        lines.append((OK, f"서버 켜짐 ({root})"))

    # 2) 통합 키
    if not key:
        lines.append((FAIL, "FREELLMAPI_API_KEY 가 비어 있습니다 — 대시보드 Keys 페이지 위쪽의 통합 키(freellmapi-…)를 "
                            "Agent_office/.env 에 넣으세요."))
        return lines
    try:
        status, data = _request(base + "/models", key=key, timeout=timeout)
    except OSError as e:
        lines.append((FAIL, f"/v1/models 호출 실패 — {type(e).__name__}"))
        return lines
    if status in (401, 403):
        lines.append((FAIL, f"통합 키가 맞지 않습니다 ({status}). 대시보드 Keys 페이지의 키를 다시 복사하세요 "
                            "(다른 PC 의 키는 이 PC 서버에서 쓸 수 없습니다)."))
        return lines
    if status != 200 or not isinstance(data, dict):
        lines.append((FAIL, f"/v1/models 응답이 이상합니다 (HTTP {status})."))
        return lines
    ids = [str(m.get("id")) for m in data.get("data") or [] if isinstance(m, dict)]
    lines.append((OK, f"통합 키 맞음 (키 길이 {len(key)}자)"))

    # 3) 쓸 수 있는 모델
    if not ids:
        lines.append((FAIL, "쓸 수 있는 모델이 0개입니다 — 대시보드 Keys 페이지에서 Google AI Studio·Groq 무료 키를 등록하세요."))
        return lines
    lines.append((OK, f"쓸 수 있는 모델 {len(ids)}개"))
    if wanted:
        have = set(ids)
        missing = [m for m in wanted if m not in have]
        if missing:
            lines.append((WARN, f"config/models.yaml 의 무료 모델 {len(wanted)}개 중 {len(missing)}개가 서버 목록에 없음: "
                                + ", ".join(missing)
                                + " — 해당 공급자 키를 등록했는지 보거나, 모델이 없어졌다면(404) 다른 모델로 바꾸세요."))
        else:
            lines.append((OK, f"config/models.yaml 의 무료 모델 {len(wanted)}개 모두 있음"))

    # 4) (선택) 실제 질문 1회 — 무료 한도를 1회 쓴다
    if chat_model:
        body = {"model": chat_model, "messages": [{"role": "user", "content": "Reply with: ok"}], "max_tokens": 16}
        try:
            status, data = _request(base + "/chat/completions", key=key, body=body, timeout=max(timeout, 60))
        except OSError as e:
            lines.append((FAIL, f"{chat_model} 질문 실패 — {type(e).__name__} (느린 모델이면 시간 초과일 수 있음)"))
            return lines
        if status == 200:
            lines.append((OK, f"{chat_model} 응답 받음"))
        elif status == 429:
            lines.append((WARN, f"{chat_model}: 무료 한도 초과(429) — 분당/하루 한도가 풀릴 때까지 기다리거나 다른 모델로 시험하세요."))
        elif status == 404:
            lines.append((FAIL, f"{chat_model}: 모델 없음(404) — 위 모델 목록에 있는 이름으로 시험하세요."))
        else:
            lines.append((FAIL, f"{chat_model}: HTTP {status}"))
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FreeLLMAPI 연결 점검 (키 값은 출력하지 않음)")
    parser.add_argument("--chat", default="", metavar="MODEL", help="이 모델에 짧은 질문 1회 (무료 한도 1회 사용)")
    parser.add_argument("--timeout", type=float, default=5.0, help="요청 대기 초 (기본 5)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from core.config import CONFIG_DIR, load_env

    load_env()
    base_url = os.getenv("FREELLMAPI_BASE_URL") or DEFAULT_BASE_URL
    key = os.getenv("FREELLMAPI_API_KEY") or ""
    print(f"FreeLLMAPI 점검 — 주소 {base_url}")
    lines = check(base_url, key, configured_free_models(CONFIG_DIR / "models.yaml"), args.chat, args.timeout)
    for mark, text in lines:
        print(f"{mark} {text}")
    failed = any(mark == FAIL for mark, _ in lines)
    if failed:
        print(f"\n설치·문제 해결: {GUIDE}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
