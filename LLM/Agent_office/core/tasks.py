"""팀 업무 양식(TaskTemplate) → 질문 + 최종 상태 브리프 + 툴 입력 만들기.

team.yaml 의 tasks 에 업무를 정의하면 화면에 입력 양식이 자동으로 생긴다.
사용자가 칸을 채우면 이 모듈이
  - question / goal / success_criteria … 의 {key} 를 입력값으로 바꾸고
  - 값이 비어 있는 {key} 가 들어간 줄은 통째로 빼고 (선택 입력 처리)
  - 날짜 범위(start_date~end_date)가 있으면 nights·days 같은 파생 값을 계산해
완성된 업무 요청(TaskRequest)을 만든다. 브리프가 이미 완성돼 있으므로 '요청 정리' LLM 호출이 필요 없다.

예) 템플릿 "{stock} 을 {quantity}주 보유 중이야"  + 입력 {stock: "삼성전자", quantity: 10}
    → "삼성전자 을 10주 보유 중이야"
"""

import re
import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from core.brief import DesiredState
from core.schemas import TaskTemplate

PLACEHOLDER = re.compile(r"\{(\w+)\}")


class TaskInputError(ValueError):
    """필수 입력 누락·잘못된 날짜 등 사용자 입력 오류 (화면에 그대로 보여 준다)."""


@dataclass
class TaskRequest:
    question: str
    brief: DesiredState
    tool_inputs: dict  # 데이터 툴에 넘길 입력값 (예: stock, quantity, start_date)
    validators: list[str] = field(default_factory=list)  # 이 업무에서 추가로 돌릴 하네스 검증기
    harness_options: dict = field(default_factory=dict)


def _is_empty(value) -> bool:
    # 꺼진 체크박스(False)도 '비어 있음' — 그 값이 들어간 줄은 빠진다
    return value is None or value is False or (isinstance(value, str) and not value.strip())


def _fmt(value) -> str:
    """숫자는 천 단위 쉼표, 정수로 떨어지는 실수는 정수로 (10.0 → 10). 켜진 체크박스는 '예'."""
    if value is True:
        return "예"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int):
        return f"{value:,}"
    return str(value).strip()


def render(template: str, values: dict) -> str:
    """{key} 치환. 비어 있는 값이 들어간 줄은 뺀다."""
    lines = []
    for line in template.splitlines():
        keys = PLACEHOLDER.findall(line)
        if any(_is_empty(values.get(k)) for k in keys):
            continue
        lines.append(PLACEHOLDER.sub(lambda m: _fmt(values[m.group(1)]), line))
    return "\n".join(lines).strip()


def render_list(items: list[str], values: dict) -> list[str]:
    return [text for text in (render(item, values) for item in items) if text]


def store_uploads(values: dict, folder: Path) -> dict:
    """업로드한 파일 객체(name·getvalue 를 가진 것, 예: Streamlit UploadedFile)를 folder 에 저장하고 경로로 바꾼다.

    값은 저장한 경로(문자열), {key}_name 에는 원래 파일 이름이 들어간다 → 질문 템플릿에 {key}_name 으로 쓸 수 있고,
    데이터 툴은 경로로 파일을 읽는다 (워크플로우 상태·체크포인트에는 파일 내용이 아니라 경로만 들어간다).
    """
    out = dict(values)
    for key, value in values.items():
        if hasattr(value, "getvalue") and hasattr(value, "name"):
            folder.mkdir(parents=True, exist_ok=True)
            safe = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", Path(value.name).name)[-80:]
            path = folder / f"{uuid.uuid4().hex[:8]}_{safe}"
            path.write_bytes(value.getvalue())
            out[key] = str(path)
            out[f"{key}_name"] = value.name
    return out


def derive_values(values: dict) -> dict:
    """입력값에서 계산할 수 있는 값을 추가한다 (LLM 이 아니라 코드가 계산 → 틀리지 않음)."""
    derived = dict(values)
    start, end = values.get("start_date"), values.get("end_date")
    if start and end:
        start_d, end_d = date.fromisoformat(str(start)), date.fromisoformat(str(end))
        if end_d < start_d:
            raise TaskInputError("도착일이 출발일보다 빠릅니다.")
        derived["days"] = (end_d - start_d).days + 1
        derived["nights"] = derived["days"] - 1
    return derived


def build_request(task: TaskTemplate, values: dict, extra_request: str = "") -> TaskRequest:
    """입력값 검증 → 템플릿 치환 → TaskRequest."""
    missing = [f.label for f in task.fields if f.required and _is_empty(values.get(f.key))]
    if missing:
        raise TaskInputError(f"필수 입력이 비어 있습니다: {', '.join(missing)}")
    full = derive_values({f.key: values.get(f.key) for f in task.fields} | values)

    question = render(task.question, full)
    if extra_request.strip():
        question += f"\n추가 요청: {extra_request.strip()}"
    context = f"업무 유형: {task.title}"
    if task.instructions.strip():
        context += "\n\n[업무 수행 지침]\n" + render(task.instructions, full)
    brief = DesiredState(
        goal=render(task.goal, full),
        context=context,
        success_criteria=render_list(task.success_criteria, full),
        constraints=render_list(task.constraints, full) + ([f"추가 요청 반영: {extra_request.strip()}"] if extra_request.strip() else []),
        output_format=render(task.output_format, full),
        assumptions=[],
    )
    # 문자열 옵션의 {key} 도 입력값으로 바꾼다 (예: expected_count: "{count}")
    harness_options = {k: render(v, full) if isinstance(v, str) else v for k, v in task.harness_options.items()}
    if "days" in full:
        harness_options.setdefault("trip_days", full["days"])
        harness_options.setdefault("trip_start", str(full["start_date"]))
    return TaskRequest(
        question=question,
        brief=brief,
        tool_inputs={k: v for k, v in full.items() if not _is_empty(v)},
        validators=list(task.validators),
        harness_options=harness_options,
    )
