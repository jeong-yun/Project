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
