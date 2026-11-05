"""질문 먼저 — 업무를 시작하기 전에 결과 모양을 크게 바꾸는 판단(형식·분류·그림·합치기)을 사람에게 1~3개 묻는다.

팀 설정으로 켠다 (team.yaml 의 ask_first — 지금은 👔 포트폴리오·자소서 팀만). 설정이 없는 팀은 지금과 똑같이 바로 시작한다.

  questions_for : 팀 툴이 코드로 만든 질문 (모델 호출 없음 — 개인 문서를 다른 모델로 보내지 않는다). 없으면 묻지 않는다.
  apply_answers : 사람이 고른 답을 업무 요청에 넣는다 — 질문 글 끝 '[먼저 답한 질문]' + 브리프 제약 + 툴 입력 ask_answers.

질문을 만드는 함수는 툴 모듈에 둔다 (MAKERS — 툴 이름 → '모듈:함수'). 함수는 툴 입력(dict) → [AskItem].
화면은 ask_first_ui.py (app.py 가 업무 양식의 '맡기기' 뒤, 팀 확인 단계 다음에 부른다).
"""

import importlib
from dataclasses import asdict, dataclass, field

from core.schemas import Team

MAKERS = {"career_docs": "core.tools.career_docs:ask_first_questions"}
HEADER = "[먼저 답한 질문 — 사람이 정한 답이다. 이 답을 따른다]"


@dataclass
class AskItem:
    id: str  # 질문 종류 (예: missing_ref · split_formats · figures · merge)
    question: str
    options: list[str]  # 고를 답 (첫 번째 = 원문을 바꾸지 않는 쪽 · 권장)
    why: str = ""  # 왜 묻는지 (화면 한 줄)
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def enabled(team: Team, task_id: str) -> bool:
    cfg = team.ask_first
    return cfg is not None and (not cfg.tasks or task_id in cfg.tasks)


def _maker(path: str):
    module, func = path.split(":")
    return getattr(importlib.import_module(module), func)


def questions_for(team: Team, task_id: str, tool_inputs: dict) -> list[AskItem]:
    """이 업무에서 시작 전에 물을 질문 (최대 max_questions). 설정이 없거나 질문이 없으면 빈 목록."""
    if not enabled(team, task_id):
        return []
    items: list[AskItem] = []
    for tool in team.tools:
        if tool in MAKERS:
            items += _maker(MAKERS[tool])(tool_inputs)
    return items[: team.ask_first.max_questions]


def answer_lines(answers: list[dict]) -> list[str]:
    return [f"- {a['question']} → {a['answer']}" for a in answers if str(a.get("answer") or "").strip()]


def apply_answers(request: dict, answers: list[dict]) -> dict:
    """[{question, answer}] → 업무 요청 (질문 글 · 브리프 제약 · 툴 입력). 답이 없으면 그대로."""
    lines = answer_lines(answers)
    if not lines:
        return request
    out = dict(request)
    out["question"] = f"{request.get('question', '')}\n{HEADER}\n" + "\n".join(lines)
    brief = request.get("brief")
    if brief is not None:
        out["brief"] = brief.model_copy(update={"constraints": [*brief.constraints, *(f"먼저 답한 결정: {ln[2:]}" for ln in lines)]})
    out["tool_inputs"] = {**(request.get("tool_inputs") or {}), "ask_answers": [a for a in answers if str(a.get("answer") or "").strip()]}
    return out
