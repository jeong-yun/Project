"""테스트용 모의 LLM — API 키·비용 없이 워크플로우 전체를 돌려 보기 위한 가짜 모델.

Script 에 '어떤 출력 형식이면 무엇을 돌려줄지'를 적어 두면 FakeOfficeModel 이 그대로 답한다.
    script = Script(text="초안", ReviewResult=make_review(), Delegation=make_delegation(...))
값이 리스트면 호출할 때마다 앞에서부터 하나씩 꺼낸다 (마지막 값은 계속 재사용) → 재시도·자가 교정 시나리오 테스트용.
값이 함수면 messages 를 받아 출력을 만든다 → 에이전트마다 다른 답을 주고 싶을 때.
"""

from typing import Any

from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.runnables import RunnableLambda

from core.schemas import Assignment, CriticResult, Delegation, ReviewResult

# 가짜 모델을 붙잡아 두는 문(threading.Event)의 대기 한도 — `gate.wait(HOLD_SEC)`.
# 문은 테스트가 열고, 준비 함수(fixture)가 끝날 때도 연다. 이 값은 그조차 못 했을 때 스레드가 영영 남지 않게 하는 안전망일 뿐이다.
# 전에는 20초였는데, 문을 닫아 둔 사이에 화면을 두세 번 다시 그리는 테스트는 바쁜 PC 에서 그 20초를 넘겨
# 문이 저절로 열리고, '진행 중' 화면을 확인하기 전에 일이 끝나 버렸다 (2026-11-12 — today_worked.md).
HOLD_SEC = 600


class Script:
    def __init__(self, **outputs: Any):
        self.outputs = outputs  # 키: "text" 또는 스키마 클래스 이름 (ReviewResult, Delegation, …)
        self.calls: list[tuple[str, list]] = []  # (키, messages) 호출 기록

    def next(self, key: str, messages: list) -> Any:
        self.calls.append((key, messages))
        if key not in self.outputs:
            raise ValueError(f"no fake output for {key}")
        value = self.outputs[key]
        if isinstance(value, list):
            value = value.pop(0) if len(value) > 1 else value[0]
        if callable(value) and not isinstance(value, type):
            value = value(messages)
        return value

    def prompts(self, key: str) -> list[str]:
        """해당 종류 호출들의 전체 프롬프트 텍스트 (검증용)."""
        return ["\n".join(str(m.content) for m in msgs) for k, msgs in self.calls if k == key]


class FakeOfficeModel(FakeListChatModel):
    script: Any = None
    responses: list = ["unused"]

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        text = self.script.next("text", messages)
        for ch in text:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=ch))
            if run_manager:
                run_manager.on_llm_new_token(ch, chunk=chunk)  # LangGraph messages 스트림으로 전달
            yield chunk

    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        return self.script.next("text", messages)

    def with_structured_output(self, schema, *, include_raw=False, **kwargs):
        def _invoke(messages):
            parsed = self.script.next(schema.__name__, messages)
            return {"parsed": parsed, "raw": AIMessage(""), "parsing_error": None}

        return RunnableLambda(_invoke)


_silent: dict[tuple[float, int], bytes] = {}


def silent_mp3(seconds: float = 0.3, sr: int = 24000) -> bytes:
    """짧은 무음 MP3 바이트 — 가짜 TTS 의 답 (pedalboard 내장 LAME, 한 번 만들어 재사용)."""
    if (seconds, sr) not in _silent:
        import io

        import numpy as np
        from pedalboard.io import AudioFile

        buf = io.BytesIO()
        with AudioFile(buf, "w", sr, 1, format="mp3") as f:
            f.write(np.zeros((1, int(sr * seconds)), dtype=np.float32))
        _silent[(seconds, sr)] = buf.getvalue()
    return _silent[(seconds, sr)]


def make_review(verdict="revise", score=7, answer="보완된 답변", comment="근거를 더 보강하세요"):
    return ReviewResult(
        analysis="근거가 부족하다.", issues=["근거 부족"] if verdict == "revise" else [], score=score,
        verdict=verdict, revised_answer=answer, comment=comment,
    )


def make_delegation(*child_ids: str):
    return Delegation(analysis="분석", assignments=[Assignment(child_id=c, subtask=f"{c} 업무") for c in child_ids])


def make_critic(score=5, leaps=None, violations=None, severity=None):
    """severity 를 안 주면 지적이 있을 때 minor, 없을 때 none."""
    severity = severity or ("minor" if leaps or violations else "none")
    return CriticResult(
        analysis="논리 점검", logical_leaps=leaps or [], rule_violations=violations or [], severity=severity, score=score
    )


def prompt_lessons(storage, team_id: str, agent_id: str, limit: int = 5) -> list[str]:
    """에이전트에게 들어갈 켜 둔 교훈 중 최근 limit 개 (오래된 것 → 최신) — 앱과 같은 길(feedback.active_lessons)로 읽는다.
    나뉜 팀은 옛 팀 교훈도 함께 물려받는다."""
    from core import feedback

    return [r["lesson"] for r in reversed(feedback.active_lessons(storage, team_id).get(agent_id, [])[:limit])]
