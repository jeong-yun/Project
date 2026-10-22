"""Claude 구독 provider (claude_cli) — 실제 claude CLI 대신 가짜 프로세스(subprocess.Popen 바꿔치기)로 확인한다."""

import io
import json
import threading
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

import home_ui
from core import claude_cli, llm_factory, usage
from core.agent import AnswerStream
from core.claude_cli import ClaudeCLIChatModel, ClaudeCLITimeout, ClaudeCLIUnavailable, ClaudeCLIUsageLimit
from core.schemas import ModelSpec, RankModels
from tests.fakes import FakeOfficeModel, Script

USAGE = {"input_tokens": 10, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 1000, "output_tokens": 20}


def result_event(text="답", **extra) -> dict:
    return {"type": "result", "subtype": "success", "is_error": False, "result": text, "session_id": "s1",
            "total_cost_usd": 0.0123, "usage": USAGE, "duration_ms": 1500, "num_turns": 1, **extra}


def delta(text: str) -> dict:
    return {"type": "stream_event", "parent_tool_use_id": None,
            "event": {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}}


def auth_failed() -> list[dict]:
    """실측(2.1.304, 로그인 안 됨)과 같은 모양."""
    return [
        {"type": "assistant", "error": "authentication_failed", "is_api_error_message": True,
         "message": {"content": [{"type": "text", "text": "Not logged in · Please run /login"}]}},
        {"type": "result", "subtype": "success", "is_error": True, "result": "Not logged in · Please run /login",
         "usage": {"input_tokens": 0, "output_tokens": 0}, "terminal_reason": "api_error"},
    ]


def limit_hit() -> list[dict]:
    return [
        {"type": "assistant", "error": "rate_limit", "message": {"content": [{"type": "text", "text": "limit"}]}},
        {"type": "result", "subtype": "success", "is_error": True, "result": "You've hit your limit · resets 5pm"},
    ]


class FakeProcess:
    """subprocess.Popen 대역 — 명령·작업 폴더·환경·표준 입력을 기록하고 정해진 줄을 출력한다."""

    def __init__(self, cmd, events=None, code=0, err_text="", hang=False, **kwargs):
        self.cmd, self.cwd, self.env = cmd, kwargs.get("cwd"), kwargs.get("env")
        self.cwd_files = sorted(p.name for p in Path(self.cwd).iterdir()) if self.cwd else []
        system_file = cmd[cmd.index("--system-prompt-file") + 1] if "--system-prompt-file" in cmd else None
        self.system = Path(system_file).read_text(encoding="utf-8") if system_file else None
        self.stdin = io.StringIO()
        self.stdin.close = lambda: None  # 내용을 테스트에서 읽을 수 있게 닫지 않는다
        lines = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in (events or []))
        self._killed = threading.Event()
        self._hang = hang
        self.stdout = self._hanging() if hang else io.StringIO(lines)
        self.stderr = io.StringIO(err_text)
        self.returncode = code

    def _hanging(self):
        self._killed.wait(5)
        return
        yield  # noqa — 제너레이터로 만들기 위함

    def kill(self):
        self._killed.set()

    def wait(self, timeout=None):
        if self._hang:
            self._killed.wait(5)
            self.returncode = -9
        return self.returncode

    def poll(self):
        return self.returncode

    @property
    def prompt(self) -> str:
        return self.stdin.getvalue()


@pytest.fixture
def fake_cli(monkeypatch):
    """가짜 claude 를 설치한다. fake_cli(events, ...) → 이후 호출은 그 출력을 낸다. calls 에 FakeProcess 가 쌓인다."""
    calls: list[FakeProcess] = []
    plan: dict = {"events": [result_event()], "code": 0, "stderr": "", "hang": False}

    def popen(cmd, **kwargs):
        proc = FakeProcess(cmd, plan["events"], plan["code"], plan["stderr"], plan["hang"], **kwargs)
        calls.append(proc)
        return proc

    monkeypatch.setattr(claude_cli, "find_executable", lambda: "C:/fake/claude.exe")
    monkeypatch.setattr(claude_cli.subprocess, "Popen", popen)

    def _set(events=None, code=0, stderr="", hang=False):
        plan.update(events=events if events is not None else [result_event()], code=code, stderr=stderr, hang=hang)
        return calls

    _set.calls = calls
    return _set


@pytest.fixture
def cli_logged_in(monkeypatch):
    monkeypatch.setattr(claude_cli, "cli_status", lambda: "ok")


# ─── 메시지 변환·명령 ────────────────────────────────────────────────


def test_messages_become_system_file_and_stdin_prompt():
    system, prompt = claude_cli.to_prompt([SystemMessage("너는 팀장"), HumanMessage("질문")])
    assert (system, prompt) == ("너는 팀장", "질문")

    # 자가 교정: 이전 답변 + 피드백 → 순서대로 한 프롬프트
    system, prompt = claude_cli.to_prompt(
        [SystemMessage("A"), SystemMessage("B"), HumanMessage("질문"), AIMessage("초안"), HumanMessage("고쳐라")]
    )
    assert system == "A\n\nB"
    assert prompt.index("[사용자]\n질문") < prompt.index("[나의 이전 답변]\n초안") < prompt.index("[지금 요청]\n고쳐라")


def test_command_answers_only_without_tools_or_api_key(fake_cli, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-no-credit")
    monkeypatch.setenv("CLAUDE_CODE_SIMPLE", "1")  # 부모 Claude Code 세션의 bare 모드 설정
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sub-token")
    calls = fake_cli()
    ClaudeCLIChatModel(model="claude-opus-5-5").invoke([SystemMessage("시스템 지시"), HumanMessage("안녕")])

    proc = calls[0]
    cmd = proc.cmd
    assert cmd[:2] == ["C:/fake/claude.exe", "-p"] and "--bare" not in cmd  # bare 는 구독 로그인을 안 쓴다
    assert cmd[cmd.index("--model") + 1] == "claude-opus-5-5"
    assert cmd[cmd.index("--tools") + 1] == ""  # 도구 모두 끔
    for flag in ("--safe-mode", "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"):
        assert flag in cmd
    assert cmd[cmd.index("--permission-mode") + 1] == "dontAsk"
    assert cmd[cmd.index("--permission-prompts") + 1] == "none"
    assert proc.system == "시스템 지시" and proc.prompt == "안녕"
    assert proc.cwd_files == ["system_prompt.txt"]  # 빈 임시 폴더 (CLAUDE.md·.mcp.json 없음)
    assert not Path(proc.cwd).exists()  # 끝나면 지운다
    assert "ANTHROPIC_API_KEY" not in proc.env and "CLAUDE_CODE_SIMPLE" not in proc.env
    assert proc.env["CLAUDE_CODE_OAUTH_TOKEN"] == "sub-token"


# ─── 결과 해석 ──────────────────────────────────────────────────────


def test_json_result_and_usage(fake_cli):
    fake_cli([{"type": "system", "subtype": "init"}, result_event("안녕하세요")])
    msg = ClaudeCLIChatModel().invoke("hi")
    assert msg.content == "안녕하세요"
    assert msg.usage_metadata["input_tokens"] == 1110  # 일반 + 캐시 생성 + 캐시 읽기
    assert msg.usage_metadata["output_tokens"] == 20
    assert msg.response_metadata["api_equivalent_cost_usd"] == 0.0123  # 참고용 추정치


def test_stream_json_yields_text_deltas(fake_cli):
    calls = fake_cli([
        {"type": "system", "subtype": "init"},
        {"type": "stream_event", "event": {"type": "message_start"}},
        delta("안녕"), delta("하세요"),
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "안녕하세요"}]}},
        result_event("안녕하세요"),
    ])
    chunks = list(ClaudeCLIChatModel().stream("hi"))
    assert [c.content for c in chunks] == ["안녕", "하세요", ""]
    full = chunks[0] + chunks[1] + chunks[2]
    assert full.content == "안녕하세요" and full.usage_metadata["output_tokens"] == 20
    cmd = calls[0].cmd
    assert cmd[cmd.index("--output-format") + 1] == "stream-json" and "--include-partial-messages" in cmd


def test_stream_without_partial_messages_returns_whole_result(fake_cli):
    fake_cli([result_event("전문")])
    assert "".join(c.content for c in ClaudeCLIChatModel().stream("hi")) == "전문"


class Plan(BaseModel):
    analysis: str
    picks: list[str]


def test_structured_output_uses_json_schema(fake_cli):
    calls = fake_cli([result_event("", structured_output={"analysis": "분석", "picks": ["a", "b"]})])
    out = ClaudeCLIChatModel().with_structured_output(Plan, include_raw=True).invoke([HumanMessage("나눠라")])
    assert out["parsed"] == Plan(analysis="분석", picks=["a", "b"]) and out["parsing_error"] is None
    assert out["raw"].usage_metadata["output_tokens"] == 20
    cmd = calls[0].cmd
    assert json.loads(cmd[cmd.index("--json-schema") + 1])["required"] == ["analysis", "picks"]
    assert cmd[cmd.index("--output-format") + 1] == "json"


def test_structured_output_falls_back_to_json_text_and_reports_parse_errors(fake_cli):
    fake_cli([result_event('```json\n{"analysis": "x", "picks": []}\n```')])
    assert ClaudeCLIChatModel().with_structured_output(Plan).invoke("q") == Plan(analysis="x", picks=[])

    fake_cli([result_event("형식 없이 답함")])
    out = ClaudeCLIChatModel().with_structured_output(Plan, include_raw=True).invoke("q")
    assert out["parsed"] is None and out["parsing_error"] is not None


# ─── 오류 분류 ──────────────────────────────────────────────────────


def test_not_logged_in_is_unavailable_and_cached(fake_cli):
    fake_cli(auth_failed(), code=1)
    with pytest.raises(ClaudeCLIUnavailable) as err:
        list(ClaudeCLIChatModel().stream("hi"))
    assert not llm_factory.is_transient(err.value)  # 기다려도 소용없음 → 바로 대안 모델
    assert claude_cli.cli_status() == "logged_out"  # 다음 호출부터 후보에서 빠진다


def test_usage_limit_cools_down_and_is_not_retried(fake_cli):
    fake_cli(limit_hit(), code=1)
    with pytest.raises(ClaudeCLIUsageLimit) as err:
        ClaudeCLIChatModel().invoke("hi")
    assert llm_factory.is_quota_exhausted(err.value) and not llm_factory.is_transient(err.value)
    assert llm_factory.retry_delay(err.value, 0) is None


def test_overload_is_transient_and_bad_flags_are_permanent(fake_cli):
    assert llm_factory.is_transient(claude_cli.error_for("Overloaded", "overloaded"))
    assert not llm_factory.is_transient(claude_cli.error_for("no such model", "model_not_found"))

    fake_cli([], code=1, stderr="error: unknown option '--foo'")
    with pytest.raises(claude_cli.ClaudeCLIError, match="unknown option"):
        ClaudeCLIChatModel().invoke("hi")


def test_timeout_kills_the_process(fake_cli):
    fake_cli(hang=True)
    with pytest.raises(ClaudeCLITimeout) as err:
        ClaudeCLIChatModel(timeout=0.2).invoke("hi")
    assert not llm_factory.is_transient(err.value)  # Opus 를 처음부터 다시 기다리지 않는다


def test_cli_status_reads_auth_status(monkeypatch):
    monkeypatch.setattr(claude_cli, "find_executable", lambda: "C:/fake/claude.exe")
    answers = iter(['{"loggedIn": true, "authMethod": "claude.ai"}'])
    runs = []

    def run(cmd, **kwargs):
        runs.append(cmd)
        return type("R", (), {"stdout": next(answers), "returncode": 0})()

    monkeypatch.setattr(claude_cli.subprocess, "run", run)
    assert claude_cli.cli_status() == "ok"
    assert claude_cli.cli_status() == "ok" and len(runs) == 1  # 캐시
    assert runs[0][1:] == ["auth", "status", "--json"]

    claude_cli._status_cache.clear()
    answers = iter(['{"loggedIn": false}'])
    assert claude_cli.cli_status() == "logged_out"

    # 캐시가 오래되면 이전 값을 바로 돌려주고 백그라운드에서 다시 확인 (화면이 멈추지 않게)
    exe = "C:/fake/claude.exe"
    claude_cli._status_cache[exe] = (claude_cli.time.monotonic() - claude_cli.STATUS_TTL - 1, "logged_out")
    answers = iter(['{"loggedIn": true}'])
    assert claude_cli.cli_status() == "logged_out"
    for _ in range(100):
        if claude_cli._status_cache[exe][1] == "ok":
            break
        claude_cli.time.sleep(0.02)
    assert claude_cli.cli_status() == "ok"

    monkeypatch.setattr(claude_cli, "find_executable", lambda: None)
    assert claude_cli.cli_status() == "not_installed"


# ─── 공급자 연결 ────────────────────────────────────────────────────


def test_team_lead_prefers_subscription_then_existing_chain(models, set_keys, monkeypatch):
    lead = models.ranks["team_lead"]
    assert (lead.primary.provider, lead.primary.model) == ("claude_cli", "claude-opus-5-5")
    assert [(s.provider, s.model) for s in lead.fallback[:2]] == [("openai", "gpt-5.5"), ("openai", "gpt-5.4")]
    opus_api = next(s for s in lead.fallback if s.provider == "anthropic")
    assert opus_api.temperature is None  # 현재 Claude 는 temperature 를 주면 400

    set_keys("openai")
    spec, _ = llm_factory.available_candidates(lead, models)[0]
    assert spec.provider == "openai"  # CLI 없음 → 기존 순서
    assert llm_factory.provider_status(models)["claude_cli"] == "not_installed"

    monkeypatch.setattr(claude_cli, "cli_status", lambda: "ok")
    spec, is_fallback = llm_factory.available_candidates(lead, models)[0]
    assert (spec.provider, is_fallback) == ("claude_cli", False)
    assert llm_factory.provider_status(models)["claude_cli"] == "ok"
    assert llm_factory.provider_status(llm_factory.free_only(models))["claude_cli"] == "disabled"
    assert llm_factory.concurrency_limit("claude_cli") == 1


def test_create_chat_model_reads_settings(models):
    llm = llm_factory.create_chat_model(ModelSpec(provider="claude_cli", model="claude-opus-5-5"), models)
    assert isinstance(llm, ClaudeCLIChatModel)
    assert llm.model == "claude-opus-5-5" and llm.timeout == 600 and llm.effort is None
    assert llm_factory.structured_output_kwargs(ModelSpec(provider="claude_cli", model="x")) == {}


CHAIN = RankModels(primary=ModelSpec(provider="claude_cli", model="claude-opus-5-5"),
                   fallback=[ModelSpec(provider="openai", model="gpt-5.5")])


def _fallback_to_fake(monkeypatch, script):
    real = llm_factory.create_chat_model

    def create(spec, models):
        return real(spec, models) if spec.provider == "claude_cli" else FakeOfficeModel(script=script)

    monkeypatch.setattr(llm_factory, "create_chat_model", create)


def test_usage_limit_falls_back_and_cools_down(models, set_keys, fake_cli, cli_logged_in, monkeypatch):
    set_keys("openai")
    script = Script(Plan=Plan(analysis="대안", picks=["x"]))
    _fallback_to_fake(monkeypatch, script)
    calls = fake_cli(limit_hit(), code=1)

    call = llm_factory.invoke_structured(CHAIN, models, Plan, [HumanMessage("q")], "팀장")
    assert call.spec.provider == "openai" and call.is_fallback
    assert len(calls) == 1  # 구독 한도는 같은 모델로 다시 하지 않는다
    assert "ClaudeCLIUsageLimit" in call.errors[0]
    assert llm_factory.cooling_down(CHAIN.primary)
    remaining = llm_factory._cooldown["claude_cli:claude-opus-5-5"] - llm_factory.time.monotonic()
    assert remaining > llm_factory.QUOTA_COOLDOWN_SEC  # 구독 한도는 30분

    spec, _ = llm_factory.available_candidates(CHAIN, models)[0]
    assert spec.provider == "openai"  # 쉬는 동안은 대안 모델부터


def test_unavailable_cli_falls_back_in_draft_stream(models, set_keys, fake_cli, monkeypatch):
    set_keys("openai")
    script = Script(text="대안 초안")
    _fallback_to_fake(monkeypatch, script)
    fake_cli(auth_failed(), code=1)
    monkeypatch.setattr(claude_cli, "cli_status", lambda: "ok")  # 상태 확인은 통과했지만 호출 중 로그인 만료

    agent = type("A", (), {})()
    agent.models, agent.name = models, "팀장"
    agent.node = type("N", (), {"id": "lead"})()
    agent.rank = type("R", (), {"id": "team_lead"})()
    agent.candidates = lambda: llm_factory.require_candidates(CHAIN, models, "팀장")
    stream = AnswerStream(agent, [HumanMessage("q")])
    assert "".join(stream) == "대안 초안"
    assert stream.result.model.provider == "openai" and "ClaudeCLIUnavailable" in stream.result.errors[0]


def test_usage_is_recorded_as_subscription(models, fake_cli):
    events = []
    usage.set_sink(events.append)
    llm = llm_factory.create_chat_model(ModelSpec(provider="claude_cli", model="claude-opus-5-5"), models)

    fake_cli([delta("답"), result_event("답")])
    list(llm.stream("hi"))
    fake_cli([result_event("", structured_output={"analysis": "a", "picks": []})])
    llm.with_structured_output(Plan, include_raw=True).invoke("q")

    assert len(events) == 2
    for e in events:
        assert (e.provider, e.model, e.key_id) == ("claude_cli", "claude-opus-5-5", "")
        assert (e.input_tokens, e.output_tokens, e.cached_tokens) == (1110, 20, 1000)
        assert e.cost_usd == 0.0  # 구독 — 청구 없음

    rows = [{"provider": "claude_cli", "model": "claude-opus-5-5", "input_tokens": 1_000_000, "output_tokens": 0, "calls": 1}]
    assert home_ui.billing("claude_cli") == "구독 (청구 없음)" and home_ui.billing("openai") == "API 종량제"
    assert home_ui.api_equivalent(rows) == pytest.approx(4.0)  # 참고용 — Opus 5.5 API 가격으로 환산
