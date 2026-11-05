"""🎓 과목 툴 — 업무 입력의 course_id 로 과목을 읽어 근거 자료로 준다 (교육부 두 팀 공용, 외부 호출 없음).

근거 자료 = [선생님 프로필 · 학습자 수준과 그 수준의 규칙 · 커리큘럼과 진도 · 최근 기록 (+ 2·3단계 extras 글)].
팀장 페르소나(team.yaml)는 "근거 자료의 선생님 프로필대로 가르친다"고만 적고, 실제 선생님은 이 글로 들어간다
→ 과목마다 다른 선생님을 팀 설정을 바꾸지 않고 쓸 수 있다 (core/agent.py 의 페르소나는 팀 단위라서).

harness_options: level · kind · course_id · language (+ 선생님 만들기면 topic) → 검증기 level_fit 등이 쓴다.
2·3단계 업무는 team.yaml 업무에 course_id 칸만 두면 이 툴 하나로 과목 문맥을 받는다.

입력 (core/tasks.build_request 가 만든 tool_inputs)
  edu_kind  : career | hobby  (team.yaml 업무의 fixed)
  edu_task  : create | quiz | ask | (2·3단계 업무 id)  (fixed)
  course_id : 과목 id (교육부 화면 '📚 내 과목'에서 고르면 자동으로 채워진다)
  topic · level · goal · weekly_hours · language : 선생님 만들기 입력
"""

from core import learning
from core.schemas import ModelsConfig, ToolResult


def run(question: str, inputs: dict, models: ModelsConfig) -> ToolResult | None:
    kind = inputs.get("edu_kind") if inputs.get("edu_kind") in learning.KINDS else ""
    course_id = str(inputs.get("course_id") or "").strip()
    if course_id:
        course = learning.LearningStore().get(course_id)
        if course is None:
            return ToolResult(
                tool="learning_course", title="🎓 과목 없음",
                content=(f"과목 `{course_id}` 를 찾지 못했다. 학습자에게 교육부 화면 '📚 내 과목'에서 과목을 고르거나 "
                         "'🎓 선생님 만들기'로 과목을 먼저 만들라고 안내한다. 과목 정보를 지어내지 않는다."),
                notes=[f"과목 {course_id} 없음 — '📚 내 과목'에서 고르세요"],
                harness_options={"kind": kind, "course_id": course_id},
            )
        return ToolResult(
            tool="learning_course",
            title=f"🎓 과목 — {course.topic} · {course.tutor.title} · {course.level_name} · 진도 {course.progress}%",
            content=learning.course_context(course),
            notes=[] if not kind or kind == course.kind else [f"이 과목은 {learning.KINDS[course.kind]} 과목입니다 (다른 팀 업무로 맡김)"],
            harness_options={"level": course.level, "kind": course.kind, "course_id": course.id, "language": course.language},
        )
    if inputs.get("edu_task") == "create" or inputs.get("topic"):
        level = learning.parse_level(inputs.get("level"))
        language = bool(inputs.get("language"))
        lines = [
            f"- 배울 분야: {inputs.get('topic') or '(없음)'} · 목적: {learning.KINDS.get(kind, '(팀이 정함)')}",
            f"- 지금 수준: **{learning.level_label(level, language)}** — 첫 커리큘럼과 첫 말투는 이 수준에 맞춘다",
            f"- 이 수준의 공통 기준: {learning.level_of(level).rule}",
            "",
            "### 수준 표 (level_rules 의 키는 아래 수준 id 를 그대로 쓴다)",
            learning.level_table(),
        ]
        return ToolResult(
            tool="learning_course", title=f"🎓 새 과목 — {inputs.get('topic') or ''} · {learning.level_label(level, language)}",
            content="\n".join(lines),
            harness_options={"level": level, "kind": kind, "language": language, "topic": str(inputs.get("topic") or "")},
        )
    return None
