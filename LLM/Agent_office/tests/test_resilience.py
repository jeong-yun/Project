"""모델 오류 대응 테스트 — 503(요청 폭주)·429·연결 끊김 재시도, 작성 도중 끊김 복구, 실무자 실패 시 계속 진행.

실제 사례: Gemini 가 초안을 쓰는 도중 '503 This model is currently experiencing high demand' 로 끊겨
예전에는 실행 전체가 멈췄다 → 이제는 같은 모델로 잠깐 기다렸다 다시, 안 되면 대안 모델로 처음부터 다시 쓴다.
"""

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.runnables import RunnableLambda

from core import llm_factory
from core.agent import OrgAgent
from core.config import subtree
from core.llm_factory import invoke_structured, is_transient
from core.schemas import ModelSpec, RankModels
from core.workflow import build_office_graph, stream_office
from tests.fakes import Script, make_review
from tests.test_workflow import BRIEF, NO_CRITIC, SETTINGS


class ServiceUnavailable(Exception):
    """google.api_core.exceptions.ServiceUnavailable 흉내."""


OVERLOAD = ServiceUnavailable("503 This model is currently experiencing high demand. Please try again later.")
NOT_FOUND = Exception("404 models/old-model is not found")


class ScriptedStreamModel:
    """plan 의 항목 하나 = 호출 한 번. ('ok', 텍스트) 성공 / ('fail', 예외) 바로 실패 / ('cut', 텍스트, 예외) 쓰다 끊김."""

    def __init__(self, plan: list, calls: list, spec: ModelSpec):
        self.plan, self.calls, self.spec = plan, calls, spec

    def stream(self, messages, config=None):
        self.calls.append(self.spec.provider)
        step = self.plan.pop(0)
        if step[0] == "fail":
            raise step[1]
        for ch in step[1]:
            yield AIMessageChunk(content=ch)
        if step[0] == "cut":
            raise step[2]


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    waits = []
    monkeypatch.setattr(llm_factory, "sleep", lambda s: waits.append(s))
    return waits


@pytest.fixture
def staff(org, ranks, team, models, set_keys):
    set_keys("google", "openai")  # 사원: google(1순위) → openai(대안)
    node = subtree(org, "staff_draft")
    return OrgAgent(node, next(r for r in ranks if r.id == node.rank), team, models)


def install(monkeypatch, plans: dict[str, list]):
    calls: list = []
    monkeypatch.setattr(
        llm_factory, "create_chat_model", lambda spec, m: ScriptedStreamModel(plans[spec.provider], calls, spec)
    )
    return calls


# ─── 오류 분류 ────────────────────────────────────────────────────────


def test_transient_errors_are_detected():
    assert is_transient(OVERLOAD)
    assert is_transient(Exception("429 Too Many Requests"))
    assert is_transient(ConnectionError("connection reset"))
    assert not is_transient(NOT_FOUND)
    assert not is_transient(Exception("401 invalid api key"))
    # 무료 할당량 0 (Gemini Pro) 은 기다려도 안 되므로 재시도하지 않는다
    assert not is_transient(Exception("429 ResourceExhausted quota exceeded free_tier limit: 0"))


# ─── 초안 스트리밍 ────────────────────────────────────────────────────


def test_cut_mid_stream_restarts_cleanly_on_same_model(staff, monkeypatch, no_wait):
    calls = install(monkeypatch, {"google": [("cut", "반쯤 쓴", OVERLOAD), ("ok", "완성된 초안")], "openai": []})
    restarts = []
    stream = staff.stream_draft(staff.draft_messages("질문", "업무"), on_restart=lambda: restarts.append(1))
    list(stream)

    assert stream.result.content == "완성된 초안"  # 반쯤 쓴 내용은 버려짐
    assert stream.result.model.provider == "google" and not stream.result.is_fallback
    assert restarts == [1] and no_wait == [llm_factory.RETRY_DELAYS[0]]
    assert "작성 도중 끊김" in stream.result.errors[0]
    assert calls == ["google", "google"]


def test_persistent_overload_moves_to_backup_model(staff, monkeypatch, no_wait):
    calls = install(monkeypatch, {"google": [("fail", OVERLOAD)] * 3, "openai": [("ok", "대안 모델 초안")]})
    result = staff.draft(staff.draft_messages("질문", "업무"))
    assert result.content == "대안 모델 초안" and result.is_fallback
    assert calls == ["google"] * 3 + ["openai"]  # 1회 + 재시도 2회 → 대안
    assert no_wait == list(llm_factory.RETRY_DELAYS)


def test_permanent_error_skips_retry(staff, monkeypatch, no_wait):
    calls = install(monkeypatch, {"google": [("fail", NOT_FOUND)], "openai": [("ok", "초안")]})
    staff.draft(staff.draft_messages("질문", "업무"))
    assert calls == ["google", "openai"] and no_wait == []  # 404 는 기다리지 않고 바로 대안


# ─── 구조화 출력 (분배·검토·비평 등) ───────────────────────────────────


def test_structured_call_retries_transient_then_succeeds(models, set_keys, monkeypatch, no_wait):
    set_keys("google", "openai")
    attempts = []

    class Flaky:
        def with_structured_output(self, schema, **kw):
            def run(messages):
                attempts.append(1)
                if len(attempts) == 1:
                    raise OVERLOAD
                return {"parsed": make_review(), "raw": AIMessage(""), "parsing_error": None}
            return RunnableLambda(run)

    monkeypatch.setattr(llm_factory, "create_chat_model", lambda spec, m: Flaky())
    rm = RankModels(primary=ModelSpec(provider="google", model="g"), fallback=[ModelSpec(provider="openai", model="o")])
    call = invoke_structured(rm, models, type(make_review()), [], "테스트")
    assert call.spec.provider == "google" and not call.is_fallback and len(attempts) == 2


# ─── 워크플로우: 실무자 한 명이 끝내 실패해도 팀은 계속 ───────────────────


def test_failed_drafter_does_not_stop_the_team(org, ranks, team, models, set_keys, monkeypatch):
    set_keys("google", "openai")
    script = Script(ReviewResult=make_review("approve", 7))

    class Model:
        def __init__(self, spec):
            self.spec = spec

        def stream(self, messages, config=None):
            raise OVERLOAD  # 모든 모델 초안 실패

        def with_structured_output(self, schema, **kw):
            return RunnableLambda(lambda m: {"parsed": script.next(schema.__name__, m), "raw": AIMessage(""), "parsing_error": None})

    monkeypatch.setattr(llm_factory, "create_chat_model", lambda spec, m: Model(spec))
    graph = build_office_graph(org, ranks, team, models, SETTINGS, NO_CRITIC)
    events = list(stream_office(graph, org, "질문", "am_action", brief=BRIEF))

    steps = [p for k, p in events if k == "step"]
    assert [(s.node_id, s.kind) for s in steps] == [("staff_case", "draft"), ("am_action", "review")]
    assert "모델 오류로 초안을 작성하지 못했습니다" in steps[0].content
    assert steps[0].errors  # 실패 사유가 검증 기록에 남는다
