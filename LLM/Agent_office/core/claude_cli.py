"""Claude 구독으로 모델 호출 — provider: claude_cli (Claude Code CLI `claude -p` 를 LangChain ChatModel 로 감싼다).

왜 필요한가
  Anthropic API 키는 크레딧이 없어 꺼 두었다(disabled_providers). 이 PC 에 로그인된 Claude 구독(Pro/Max)은
  API 크레딧이 아니라 구독 사용 한도로 돌아가므로, 팀장처럼 호출 수가 적은 직급에 가장 좋은 Claude 를 쓸 수 있다.
  구독 한도는 사람이 직접 쓰는 Claude 대화·Claude Code 와 함께 쓰므로 팀장에만 배정하고, 한도에 걸리면 기존 대안 모델로 넘어간다.

어떻게 부르나 (호출 1회 = `claude -p` 프로세스 1개)
  - 시스템 메시지 → --system-prompt-file (Claude Code 기본 프롬프트를 통째로 바꿈), 대화 → 표준 입력(stdin)
  - 도구 없이 '답만' 하게: --tools "" · --safe-mode(CLAUDE.md·hooks·플러그인·MCP·스킬 끔) · --strict-mcp-config ·
    --permission-mode dontAsk · --permission-prompts none · --disable-slash-commands · --no-session-persistence,
    작업 폴더는 매번 새로 만든 빈 임시 폴더
  - --bare 는 쓰지 않는다 (bare 모드는 구독 로그인을 읽지 않고 ANTHROPIC_API_KEY 만 쓴다)
  - 자식 프로세스 환경에서 ANTHROPIC_API_KEY 등을 지운다 — .env 의 API 키(크레딧 없음)를 CLI 가 쓰지 않게
  - 초안 스트리밍: --output-format stream-json --include-partial-messages 의 text_delta
  - 구조화 출력: --json-schema → 결과의 structured_output (with_structured_output 이 {raw, parsed, parsing_error} 반환)

실패 처리 (llm_factory 의 재시도·대안 모델 로직이 아래 속성을 읽는다)
  - CLI 없음·로그인 안 됨 → ClaudeCLIUnavailable (상태 캐시를 '사용 불가'로 → 다음 호출부터 후보에서 빠짐)
  - 구독 사용 한도 → ClaudeCLIUsageLimit (같은 모델 재시도 없이 대안 모델로, LIMIT_COOLDOWN_SEC 동안 쉼)
  - 과부하·서버 오류 → 일시적 오류 (잠깐 기다렸다 재시도)
  - 시간 초과 → ClaudeCLITimeout (Opus 를 처음부터 다시 기다리지 않고 바로 대안 모델로)
  - 안전 필터 거부 → ClaudeCLIRefused (재시도 없이 대안 모델로, 그 팀에서만 30분 건너뜀 — 프롬프트 내용 문제)

사용량
  결과의 usage(입력·캐시·출력 토큰)를 usage_metadata 로 돌려준다 → usage.UsageCallback 이 provider claude_cli 로 기록,
  가격표(free_providers)에서 비용 0 = 메인 화면의 '구독'. CLI 가 주는 total_cost_usd 는 'API 였다면' 추정치일 뿐이라
  response_metadata 에만 남긴다.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, ConfigDict

PROVIDER = "claude_cli"
ENV_PATH = "CLAUDE_CLI_PATH"  # claude 실행 파일 경로 (선택 — 비우면 PATH·기본 설치 위치에서 찾는다)
DEFAULT_TIMEOUT_SEC = 600  # Opus 가 긴 통합 답변을 쓰는 시간 + CLI 시작(약 5~7초)
STATUS_TTL = 300.0  # 초 — 설치·로그인 확인 결과 캐시
LIMIT_COOLDOWN_SEC = 1800  # 구독 한도는 몇 시간 단위로 풀리므로 30분 동안 후보에서 뺀다 (그 뒤 한 번 다시 시도)

# 결과 메시지의 error 값 (docs: system/api_retry 의 error 분류와 같음)
UNAVAILABLE_ERRORS = {"authentication_failed", "oauth_org_not_allowed", "account_on_hold", "cloud_credential_error"}
LIMIT_ERRORS = {"rate_limit", "billing_error"}
TRANSIENT_ERRORS = {"overloaded", "server_error"}
UNAVAILABLE_TEXT = ("not logged in", "/login", "invalid api key", "oauth token has expired")
LIMIT_TEXT = ("usage limit", "hit your limit", "limit reached", "rate limit", "rate_limit", "out of extra usage")
# 모델 안전 필터가 요청 내용을 거부 (2026-10-27 실측: "Opus 5.5's safeguards flagged this message (…/legal/aup)")
REFUSAL_TEXT = ("safeguards flagged", "legal/aup", "usage policy")
REFUSAL_COOLDOWN_SEC = 1800  # 같은 팀 프롬프트는 다시 보내도 또 거부되므로 30분 동안 그 팀에서만 건너뛴다


class ClaudeCLIError(RuntimeError):
    """claude -p 호출 실패. transient/quota_exhausted 를 llm_factory 가 읽는다 (None = 메시지로 판단)."""

    transient: bool | None = None
    quota_exhausted = False
    cooldown_sec: float | None = None


class ClaudeCLIUnavailable(ClaudeCLIError):
    """CLI 가 없거나 로그인이 안 됨 — 이 provider 를 쓸 수 없음."""

    transient = False


class ClaudeCLIUsageLimit(ClaudeCLIError):
    """구독 사용 한도 — 기다려도 몇 시간은 안 풀린다."""

    transient = False
    quota_exhausted = True
    cooldown_sec = LIMIT_COOLDOWN_SEC


class ClaudeCLITimeout(ClaudeCLIError):
    transient = False


class ClaudeCLIRefused(ClaudeCLIError):
    """모델 안전 필터가 이 요청(프롬프트 내용)을 거부 — 같은 프롬프트로는 다시 해도 같다.

    프롬프트(팀 스킬)에 달린 문제라 구독 전체가 아니라 '그 팀'에서만 잠시 건너뛴다 (llm_factory.note_failure).
    """

    transient = False
    refused = True
    cooldown_sec = REFUSAL_COOLDOWN_SEC


# ─── 실행 파일·로그인 상태 ────────────────────────────────────────────


def find_executable() -> str | None:
    """claude 실행 파일 — CLAUDE_CLI_PATH > PATH > ~/.local/bin > Claude 데스크톱 앱이 함께 설치한 CLI(가장 최근 버전)."""
    override = os.getenv(ENV_PATH)
    if override:
        return override if Path(override).is_file() else shutil.which(override)
    found = shutil.which("claude")
    if found:
        return found
    home = Path.home()
    for path in (home / ".local" / "bin" / "claude.exe", home / ".local" / "bin" / "claude"):
        if path.is_file():
            return str(path)
    local = os.getenv("LOCALAPPDATA")
    if local:
        bundled = sorted(Path(local).glob("Claude-Data/*/claude-code/*/*/claude.exe"), key=lambda p: p.stat().st_mtime)
        if bundled:
            return str(bundled[-1])
    return None


def child_env() -> dict[str, str]:
    """자식 CLI 의 환경변수 — API 키·다른 Claude Code 세션의 설정이 섞이지 않게 지운다.

    - ANTHROPIC_*  : .env 의 API 키(크레딧 없음)를 쓰면 구독이 아니라 API 로 청구된다
    - CLAUDECODE·CLAUDE_CODE_* : 이 앱을 Claude Code 안에서 켰을 때 부모 세션 설정(예: CLAUDE_CODE_SIMPLE=bare 모드)이 넘어온다
      (CLAUDE_CODE_OAUTH_TOKEN 은 `claude setup-token` 으로 만든 구독 토큰이므로 남긴다)
    """
    def drop(name: str) -> bool:
        name = name.upper()
        return (
            name.startswith("ANTHROPIC_")
            or name in ("CLAUDECODE", "CLAUDE_PID")
            or name.startswith("CLAUDE_AGENT_SDK")
            or (name.startswith("CLAUDE_CODE_") and name != "CLAUDE_CODE_OAUTH_TOKEN")
        )

    return {k: v for k, v in os.environ.items() if not drop(k)}


def _popen_kwargs() -> dict:
    kwargs: dict[str, Any] = {"env": child_env(), "encoding": "utf-8", "errors": "replace"}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW  # Streamlit 에서 부를 때 콘솔 창이 뜨지 않게
    return kwargs


_status_cache: dict[str, tuple[float, str]] = {}  # 실행 파일 → (확인 시각, 상태)
_status_lock = threading.Lock()


_refreshing: set[str] = set()  # 백그라운드에서 다시 확인 중인 실행 파일


def _check_login(exe: str) -> str:
    try:
        proc = subprocess.run([exe, "auth", "status", "--json"], capture_output=True, timeout=30, **_popen_kwargs())
        info = json.loads(proc.stdout or "{}")
        return "ok" if info.get("loggedIn") else "logged_out"
    except FileNotFoundError:
        return "not_installed"
    except (OSError, ValueError, subprocess.SubprocessError):
        return "logged_out"


def _refresh(exe: str) -> None:
    status = _check_login(exe)
    with _status_lock:
        _status_cache[exe] = (time.monotonic(), status)
        _refreshing.discard(exe)


def cli_status() -> str:
    """ok | not_installed | logged_out — `claude auth status` 결과를 5분 캐시한다.

    확인에 2~7초 걸리므로 처음 한 번만 기다리고, 캐시가 오래되면 이전 값을 돌려주며 백그라운드에서 다시 확인한다
    (사이드바가 매번 다시 그려질 때 화면이 멈추지 않게).
    """
    exe = find_executable()
    if not exe:
        return "not_installed"
    with _status_lock:
        cached = _status_cache.get(exe)
        if cached and time.monotonic() - cached[0] < STATUS_TTL:
            return cached[1]
        if cached:
            if exe not in _refreshing:
                _refreshing.add(exe)
                threading.Thread(target=_refresh, args=(exe,), daemon=True).start()
            return cached[1]
    _refresh(exe)
    return _status_cache[exe][1]


def mark_unavailable(status: str = "logged_out") -> None:
    """호출 중 로그인 오류가 나면 캐시를 바로 '사용 불가'로 → 다음 호출부터 후보에서 빠진다."""
    exe = find_executable()
    if exe:
        with _status_lock:
            _status_cache[exe] = (time.monotonic(), status)


def is_available() -> bool:
    return cli_status() == "ok"


# ─── 메시지 → 프롬프트 ───────────────────────────────────────────────


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


def to_prompt(messages: list[BaseMessage]) -> tuple[str, str]:
    """(시스템 프롬프트, 표준 입력으로 보낼 프롬프트).

    질문 하나뿐이면 그대로 보낸다. 이전 답변·피드백이 이어지는 대화(자가 교정·하네스 재시도)는
    순서대로 표시해 하나의 프롬프트로 만든다 (claude -p 는 한 번에 사용자 메시지 하나만 받는다).
    """
    system = "\n\n".join(_text(m.content) for m in messages if isinstance(m, SystemMessage))
    turns = [m for m in messages if not isinstance(m, SystemMessage)]
    if len(turns) == 1 and isinstance(turns[0], HumanMessage):
        return system, _text(turns[0].content)
    parts = []
    for i, m in enumerate(turns):
        if isinstance(m, AIMessage):
            parts.append(f"[나의 이전 답변]\n{_text(m.content)}")
        else:
            label = "[지금 요청]" if i == len(turns) - 1 else "[사용자]"
            parts.append(f"{label}\n{_text(m.content)}")
    return system, "\n\n".join(parts)


# ─── 결과 해석 ──────────────────────────────────────────────────────


def usage_metadata(usage: dict | None) -> dict | None:
    """CLI 의 usage → LangChain usage_metadata (입력 = 일반 + 캐시 생성 + 캐시 읽기)."""
    if not usage:
        return None
    plain = int(usage.get("input_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    cache_write = int(usage.get("cache_creation_input_tokens") or 0)
    out = int(usage.get("output_tokens") or 0)
    total_in = plain + cache_read + cache_write
    return {
        "input_tokens": total_in, "output_tokens": out, "total_tokens": total_in + out,
        "input_token_details": {"cache_read": cache_read, "cache_creation": cache_write},
    }


def error_for(text: str, kind: str | None = None) -> ClaudeCLIError:
    """CLI 가 알려 준 실패(오류 분류 kind·메시지) → 알맞은 예외."""
    lowered = (text or "").lower()
    message = f"Claude CLI: {text or kind or '알 수 없는 오류'}" + (f" ({kind})" if kind else "")
    if kind in UNAVAILABLE_ERRORS or any(t in lowered for t in UNAVAILABLE_TEXT):
        mark_unavailable()
        return ClaudeCLIUnavailable(message)
    if any(t in lowered for t in REFUSAL_TEXT):
        return ClaudeCLIRefused(message)  # 거부 안내문에 'rate limit' 같은 낱말이 섞여도 한도로 오해하지 않게 먼저 본다
    if kind in LIMIT_ERRORS or any(t in lowered for t in LIMIT_TEXT):
        return ClaudeCLIUsageLimit(message)
    error = ClaudeCLIError(message)
    if kind in TRANSIENT_ERRORS:
        error.transient = True
    elif kind:
        error.transient = False  # model_not_found·invalid_request 등 — 다시 해도 같다
    return error


def check_result(result: dict, api_error: str | None = None) -> None:
    """result 메시지가 실패면 예외를 던진다."""
    if result.get("is_error") or result.get("subtype") not in (None, "success"):
        raise error_for(str(result.get("result") or result.get("subtype") or ""), api_error)


def response_metadata(result: dict, model: str) -> dict:
    return {
        "model_name": model, "provider": PROVIDER, "session_id": result.get("session_id"),
        "duration_ms": result.get("duration_ms"), "num_turns": result.get("num_turns"),
        # API 로 불렀다면 얼마였을지 CLI 가 추정한 값 — 구독이라 실제 청구는 없다 (참고용)
        "api_equivalent_cost_usd": result.get("total_cost_usd"),
    }


def parse_json_output(stdout: str) -> dict:
    """--output-format json 출력(마지막 줄이 결과 JSON)."""
    for line in reversed((stdout or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
            except ValueError:
                continue
            if isinstance(data, dict) and data.get("type") == "result":
                return data
    raise ClaudeCLIError(f"Claude CLI: 결과 JSON 을 찾지 못함 — {(stdout or '').strip()[:300]}")


def stream_event_text(event: dict) -> str:
    """stream-json 한 줄에서 본문 텍스트 조각 (text_delta) 만 꺼낸다."""
    if event.get("type") != "stream_event" or event.get("parent_tool_use_id"):
        return ""
    inner = event.get("event") or {}
    delta = inner.get("delta") or {}
    if inner.get("type") == "content_block_delta" and delta.get("type") == "text_delta":
        return delta.get("text") or ""
    return ""


# ─── 프로세스 실행 ───────────────────────────────────────────────────


class _Run:
    """claude -p 프로세스 하나 — 빈 임시 폴더에서 실행, 표준 입력으로 프롬프트, 시간 초과면 강제 종료."""

    def __init__(self, exe: str, model: str, system: str, prompt: str, *, stream: bool,
                 schema: dict | None = None, effort: str | None = None, timeout: float = DEFAULT_TIMEOUT_SEC):
        self.timeout = timeout
        self.timed_out = False
        self._dir = tempfile.TemporaryDirectory(prefix="agent_office_claude_")
        cmd = [
            exe, "-p", "--model", model,
            "--output-format", "stream-json" if stream else "json",
            "--safe-mode", "--tools", "", "--strict-mcp-config",
            "--permission-mode", "dontAsk", "--permission-prompts", "none",
            "--disable-slash-commands", "--no-session-persistence",
        ]
        if stream:
            cmd += ["--verbose", "--include-partial-messages"]
        if system:
            # 명령줄 길이 한도(Windows 32K)를 넘지 않게 파일로 넘긴다 — 스킬·교훈·루브릭이 붙으면 길다
            path = Path(self._dir.name) / "system_prompt.txt"
            path.write_text(system, encoding="utf-8")
            cmd += ["--system-prompt-file", str(path)]
        if schema is not None:
            cmd += ["--json-schema", json.dumps(schema, ensure_ascii=False)]
        if effort:
            cmd += ["--effort", effort]
        try:
            self.proc = subprocess.Popen(cmd, cwd=self._dir.name, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, **_popen_kwargs())
        except FileNotFoundError as e:
            self._dir.cleanup()
            mark_unavailable("not_installed")
            raise ClaudeCLIUnavailable(f"Claude CLI 실행 파일이 없음: {exe}") from e
        self._stderr: list[str] = []
        self._threads = [threading.Thread(target=self._drain_stderr, daemon=True),
                         threading.Thread(target=self._write_stdin, args=(prompt,), daemon=True)]
        for t in self._threads:
            t.start()
        self._timer = threading.Timer(timeout, self._kill)
        self._timer.daemon = True
        self._timer.start()

    def _write_stdin(self, prompt: str) -> None:
        try:
            self.proc.stdin.write(prompt)
            self.proc.stdin.close()
        except OSError:
            pass  # 프로세스가 먼저 끝남 (실패는 종료 코드·결과로 알린다)

    def _drain_stderr(self) -> None:
        for line in self.proc.stderr:
            self._stderr.append(line)

    def _kill(self) -> None:
        self.timed_out = True
        self.proc.kill()

    def lines(self) -> Iterator[str]:
        yield from self.proc.stdout

    def finish(self) -> tuple[int, str]:
        """(종료 코드, 표준 오류) — 시간 초과였으면 ClaudeCLITimeout."""
        code = self.proc.wait()
        self._timer.cancel()
        for t in self._threads:
            t.join(timeout=2)
        if self.timed_out:
            raise ClaudeCLITimeout(f"Claude CLI 가 {self.timeout:.0f}초 안에 끝나지 않아 중단함")
        return code, "".join(self._stderr).strip()

    def close(self) -> None:
        self._timer.cancel()
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        for f in (self.proc.stdout, self.proc.stderr):
            try:
                f.close()
            except OSError:
                pass
        self._dir.cleanup()


# ─── LangChain ChatModel ─────────────────────────────────────────────


class ClaudeCLIChatModel(BaseChatModel):
    """Claude 구독(claude -p)으로 답하는 ChatModel. 도구 없이 텍스트(또는 구조화 JSON)만 돌려준다."""

    model: str = "claude-opus-5-5"
    timeout: float = DEFAULT_TIMEOUT_SEC
    effort: str | None = None  # low | medium | high | xhigh | max (None = CLI 기본값)
    executable: str | None = None  # None = find_executable()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return PROVIDER

    @property
    def _identifying_params(self) -> dict:
        return {"model": self.model, "effort": self.effort}

    def _start(self, messages: list[BaseMessage], *, stream: bool, schema: dict | None = None) -> _Run:
        exe = self.executable or find_executable()
        if not exe:
            raise ClaudeCLIUnavailable("Claude CLI(claude) 가 설치되어 있지 않음")
        system, prompt = to_prompt(messages)
        return _Run(exe, self.model, system, prompt, stream=stream, schema=schema, effort=self.effort, timeout=self.timeout)

    def run_json(self, messages: list[BaseMessage], schema: dict | None = None) -> dict:
        """--output-format json 으로 한 번 호출하고 성공한 result 메시지를 돌려준다."""
        run = self._start(messages, stream=False, schema=schema)
        try:
            stdout = "".join(run.lines())
            code, stderr = run.finish()
        finally:
            run.close()
        try:
            result = parse_json_output(stdout)
        except ClaudeCLIError:
            if code != 0:
                raise error_for(stderr or stdout.strip()[:300]) from None
            raise
        check_result(result)
        return result

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        result = self.run_json(messages)
        message = AIMessage(content=str(result.get("result") or ""), usage_metadata=usage_metadata(result.get("usage")),
                            response_metadata=response_metadata(result, self.model))
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(self, messages: list[BaseMessage], stop: list[str] | None = None,
                run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        # on_llm_new_token 은 BaseChatModel.stream 이 조각마다 부른다 (여기서 또 부르면 두 번 나간다)
        run = self._start(messages, stream=True)
        result: dict | None = None
        api_error: str | None = None
        streamed = False
        try:
            for line in run.lines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("type") == "assistant" and event.get("error"):
                    api_error = event["error"]  # 예: authentication_failed, rate_limit (본문은 result 에 다시 온다)
                elif event.get("type") == "result":
                    result = event
                text = stream_event_text(event)
                if text:
                    streamed = True
                    yield ChatGenerationChunk(message=AIMessageChunk(content=text))
            code, stderr = run.finish()
        finally:
            run.close()
        if result is None:
            raise error_for(stderr or f"결과 없이 종료됨 (종료 코드 {code})")
        check_result(result, api_error)
        tail = "" if streamed else str(result.get("result") or "")  # 부분 메시지가 없었으면 결과 전문을 한 번에
        yield ChatGenerationChunk(message=AIMessageChunk(
            content=tail, usage_metadata=usage_metadata(result.get("usage")),
            response_metadata=response_metadata(result, self.model),
        ))

    def with_structured_output(self, schema, *, include_raw: bool = False, **kwargs: Any):
        """--json-schema 로 호출 → {raw, parsed, parsing_error} (include_raw=False 면 parsed 만).

        schema 는 pydantic 모델 또는 JSON Schema dict. llm_factory.invoke_structured 의 다른 provider 와 같은 모양으로 돌려준다.
        """
        is_model = isinstance(schema, type) and issubclass(schema, BaseModel)
        json_schema = schema.model_json_schema() if is_model else dict(schema)

        def _invoke(messages, config=None):
            messages = _as_messages(messages)
            # 콜백(사용량 기록)이 붙도록 invoke 경로를 쓰지 않고 직접 부르되, 기록은 아래에서 한 번 한다
            result = self._invoke_with_callbacks(messages, json_schema, config)
            data = result.get("structured_output")
            if data is None:
                data = _json_from_text(str(result.get("result") or ""))
            raw = AIMessage(content=json.dumps(data, ensure_ascii=False) if data is not None else str(result.get("result") or ""),
                            usage_metadata=usage_metadata(result.get("usage")),
                            response_metadata=response_metadata(result, self.model))
            parsed, error = None, None
            try:
                if data is None:
                    raise ValueError("structured_output 이 없음 (모델이 형식에 맞는 답을 내지 못함)")
                parsed = schema.model_validate(data) if is_model else data
            except Exception as e:  # noqa: BLE001 — 다른 provider 처럼 parsing_error 로 돌려준다
                error = e
            if include_raw:
                return {"raw": raw, "parsed": parsed, "parsing_error": error}
            if error is not None:
                raise error
            return parsed

        return RunnableLambda(_invoke)

    def _invoke_with_callbacks(self, messages: list[BaseMessage], json_schema: dict, config) -> dict:
        """구조화 호출도 on_chat_model_start/on_llm_end 콜백(usage.UsageCallback)을 거치게 한다."""
        from langchain_core.callbacks import CallbackManager

        manager = CallbackManager.configure(inheritable_callbacks=(config or {}).get("callbacks"),
                                            local_callbacks=self.callbacks)
        (run_manager,) = manager.on_chat_model_start({"name": PROVIDER}, [messages], invocation_params=self._identifying_params)
        try:
            result = self.run_json(messages, json_schema)
        except BaseException as e:
            run_manager.on_llm_error(e)
            raise
        from langchain_core.outputs import LLMResult

        message = AIMessage(content="", usage_metadata=usage_metadata(result.get("usage")))
        run_manager.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))
        return result


def _as_messages(value) -> list[BaseMessage]:
    if isinstance(value, str):
        return [HumanMessage(value)]
    if hasattr(value, "to_messages"):
        return value.to_messages()
    return list(value)


def _json_from_text(text: str):
    """structured_output 이 없을 때 본문에서 JSON 을 찾는다 (```json 코드 블록 포함)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except ValueError:
        return None
