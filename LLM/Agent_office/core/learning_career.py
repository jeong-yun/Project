"""🧭 직무 전환 교육팀(edu_career) 전용 — 코드 계산 · 검증기 · 과목 문맥 글 (2단계, 화면 그리기 없음).

core/learning.py 의 load_extensions() 가 이 모듈을 자동으로 읽는다 (공용 파일은 고치지 않는다 — docs/PROPOSAL_교육부.md '2·3단계 세션용 안내').
화면은 edu_career_ui.py, 업무는 teams/edu_career/team.yaml 의 tasks 뒤 4개.

  🗺 합격 로드맵·커리큘럼 (roadmap)      extras["roadmap"]  = {target_role, deadline, weekly_hours, competencies, steps, certs, portfolio, at}
                                         채택하면 남은(안 끝낸) 커리큘럼을 로드맵 단계로 바꾼다 (끝낸 단원은 그대로 — learning.replace_remaining)
  🛠 개인 프로젝트 설계 (project_plan)    extras["projects"] = [{id, title, problem, deliverable, tech, difficulty, goal, status, portfolio_line,
                                                                milestones: [{id, title, due, done, blocked, memo, done_at}], at}]
  📌 진행 점검 (project_check)           지금 마일스톤 상태(완료·막힘·메모)를 근거로 다음 할 일 · 막힌 곳 해결책 · 일정 조정 → 채택하면 마일스톤·기한을 고친다
  🎯 JD 추천·지원 준비 (jd_match)        extras["jd_links"] = [{kind: jd|advice|search, title, url, note, at, …}]
                                         JD 요구사항·1차 적합도는 커리어부 코드(core/career.py extract_requirements · coverage · score_jds)를
                                         그대로 불러 '학습 기록'에 대고 계산한다 (커리어부 파일은 고치지 않는다)

검증기 (모두 결정론적 · 외부 호출 없음 — 최종 답변에만: team.yaml harness.final_only)
  roadmap_plan    : roadmap JSON — 주차·단원(제목) 형식, 주차마다 시간 합 ≤ 주당 학습 시간 (코드가 0.1시간 단위 정수로 더한다)
  project_plan    : projects JSON 1~3개 — 문제·결과물·기술·난이도(지금 수준 ~ 한 단계 위)·마일스톤(제목·기한)
  milestone_order : 마일스톤 기한이 앞에서부터 늦어지는 순서인지 ('N주차' 또는 'YYYY-MM-DD', 한 프로젝트 안에서는 한 가지 형식)
                    — 프로젝트 설계 답변이면 그 계획을, 진행 점검 답변이면 '과목에 적용한 뒤'의 일정을 본다
  progress_check  : 진행 점검 JSON — 프로젝트·마일스톤 id 가 과목에 있는지, 다음 할 일이 있는지
  jd_advice       : JD 추천 JSON — 직무명·키워드·신입/경력·부족한 요건·커리큘럼에 더할 단원, 검색어에 개인 정보가 없는지
                    (커리어부 검색어 검사 core/career_search.query_problems 재사용)
"""

from __future__ import annotations

import json
import re
from datetime import date

from core import learning
from core.harness.base import Validator, register

# core.career · core.career_search(커리어부 코드 — 고치지 않고 불러 쓴다)는 함수 안에서 import 한다:
# core.governance → core.harness → (이 모듈) → core.career → core.governance 순환 import 를 피하려고.

# ─── 공통 ───────────────────────────────────────────────────────────

WEEK_RX = re.compile(r"^\s*(\d{1,3})\s*주\s*차?\s*$")
DATE_RX = re.compile(r"^\s*(\d{4})-(\d{2})-(\d{2})\s*$")
PROJECT_STATUSES = ("계획", "진행 중", "완료", "보류")
SENIORITY = ("신입", "경력", "상관없음")


def due_key(due: object) -> tuple[str, object] | None:
    """기한 글 → 비교할 값. '3주차'·'3주' → ('week', 3), '2026-12-01' → ('date', date). 모르는 형식이면 None."""
    s = str(due or "")
    m = WEEK_RX.match(s)
    if m:
        return "week", int(m.group(1))
    m = DATE_RX.match(s)
    if m:
        try:
            return "date", date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


def tenths(value: object) -> int | None:
    """시간 → 0.1시간 단위 정수 (부동소수점 비교를 피한다). 숫자가 아니거나 음수면 None."""
    try:
        n = round(float(str(value).strip()) * 10)
    except (TypeError, ValueError):
        return None
    return n if n >= 0 else None


def hours_text(t: int) -> str:
    """0.1시간 단위 정수 → '4' / '4.5'."""
    return str(t // 10) if t % 10 == 0 else f"{t // 10}.{t % 10}"


def _course(options: dict) -> learning.Course | None:
    cid = str(options.get("course_id") or "")
    return learning.LearningStore().get(cid) if learning.valid_course_id(cid) else None


def _list(value: object) -> list:
    return value if isinstance(value, list) else ([] if value in (None, "") else [value])


def _strs(value: object) -> list[str]:
    return [str(x).strip() for x in _list(value) if str(x).strip()]


def last_week(course: learning.Course) -> int:
    return max((u.week for u in course.curriculum), default=0)


def append_units(course: learning.Course, drafts: list[dict]) -> list[learning.Unit]:
    """커리큘럼 맨 뒤에 단원을 더한다 (id 는 이어서, 주차는 마지막 단원 다음 주부터)."""
    used = [int(u.id[1:]) for u in course.curriculum if u.id[1:].isdigit()]
    units = learning.units_from_drafts(drafts, course.level, start_index=max(used, default=0) + 1,
                                       start_week=last_week(course) + 1)
    course.curriculum = [*course.curriculum, *units]
    return units


# ─── 🗺 합격 로드맵·커리큘럼 ─────────────────────────────────────────────

ROADMAP_JSON_HELP = (
    '{"roadmap": {"target_role": "목표 직무", "deadline": "24주", "weekly_hours": 5, '
    '"competencies": [{"name": "역량", "need": "합격에 필요한 수준", "now": "지금", "gap": "채울 것"}], '
    '"steps": [{"week": 1, "title": "단원 제목", "competency": "역량", "hours": 3, "goals": ["마치면 할 수 있는 것"], '
    '"link": "자격증·포트폴리오 연결 (선택)"}], "certs": ["자격증 (선택)"], "portfolio": ["포트폴리오로 남길 것"]}}'
)


def week_hours(steps: list) -> dict[int, int]:
    """주차 → 그 주 단계 시간 합 (0.1시간 단위 정수)."""
    out: dict[int, int] = {}
    for s in steps:
        if isinstance(s, dict):
            w = learning._int(s.get("week"), 0)
            out[w] = out.get(w, 0) + (tenths(s.get("hours")) or 0)
    return dict(sorted(out.items()))


def roadmap_problems(data: object, weekly_hours: object = None) -> list[str]:
    """로드맵 JSON 의 형식 문제 (빈 목록이면 통과). weekly_hours = 주당 학습 시간 (0·없음이면 시간 합 검사를 건너뛴다)."""
    road = data.get("roadmap") if isinstance(data, dict) else None
    if not isinstance(road, dict):
        return ['로드맵 JSON 이 없다 — ```json 블록에 {"roadmap": {...}} 형식으로 쓴다']
    problems = []
    if not str(road.get("target_role") or "").strip():
        problems.append("roadmap.target_role(목표 직무)이 비어 있다")
    steps = road.get("steps")
    if not isinstance(steps, list) or not steps:
        return [*problems, "roadmap.steps(주차별 단원)가 비어 있다"]
    prev = 0
    for i, s in enumerate(steps, 1):
        if not isinstance(s, dict):
            problems.append(f"{i}번째 단계가 객체가 아니다")
            continue
        week = learning._int(s.get("week"), 0)
        if week < 1 or str(s.get("week")).strip() in ("", "None"):
            problems.append(f"{i}번째 단계: week(주차)는 1 이상의 정수")
        elif week < prev:
            problems.append(f"{i}번째 단계: 주차가 앞 단계({prev}주차)보다 앞선다 ({week}주차) — 주차 순서대로 쓴다")
        prev = max(prev, week)
        if not str(s.get("title") or "").strip():
            problems.append(f"{i}번째 단계: title(단원 제목)이 비어 있다")
        if tenths(s.get("hours")) is None:
            problems.append(f"{i}번째 단계: hours(이 단계 학습 시간)는 0 이상의 숫자")
        if not _strs(s.get("goals")):
            problems.append(f"{i}번째 단계: goals(마치면 할 수 있는 것)가 비어 있다")
    limit = tenths(weekly_hours)
    if limit:
        over = [f"{w}주차 {hours_text(h)}시간" for w, h in week_hours(steps).items() if h > limit]
        if over:
            problems.append(f"주당 학습 시간 {hours_text(limit)}시간을 넘는 주가 있다: {', '.join(over[:6])} — "
                            "그 주 단계 시간(hours) 합을 줄이거나 다음 주로 나눈다 (합은 코드가 계산)")
    return problems


def roadmap_from_answer(text: str) -> dict:
    """채택한 답변 → 저장할 로드맵 (형식이 틀리면 ValueError)."""
    data = learning.find_json(text, "roadmap")
    problems = roadmap_problems(data)
    if problems:
        raise ValueError(" / ".join(problems[:4]))
    road = data["roadmap"]
    steps = [{"week": learning._int(s.get("week")), "title": str(s["title"]).strip(), "competency": str(s.get("competency") or ""),
              "hours": hours_text(tenths(s.get("hours")) or 0), "goals": _strs(s.get("goals")), "link": str(s.get("link") or "")}
             for s in road["steps"] if isinstance(s, dict)]
    comps = [{k: str(c.get(k) or "") for k in ("name", "need", "now", "gap")} for c in _list(road.get("competencies")) if isinstance(c, dict)]
    return {"target_role": str(road.get("target_role") or "").strip(), "deadline": str(road.get("deadline") or ""),
            "weekly_hours": str(road.get("weekly_hours") or ""), "competencies": comps, "steps": steps,
            "certs": _strs(road.get("certs")), "portfolio": _strs(road.get("portfolio"))}


def roadmap_drafts(road: dict) -> list[dict]:
    """로드맵 단계 → 커리큘럼 단원 초안 (자격증·포트폴리오 연결은 목표 끝에 붙인다)."""
    out = []
    for s in road.get("steps") or []:
        goals = list(s.get("goals") or [])
        if s.get("link"):
            goals.append(f"🔗 {s['link']}")
        out.append({"week": s.get("week") or 1, "title": s["title"], "goals": goals})
    return out


def apply_roadmap(course: learning.Course, road: dict, now: str | None = None) -> list[str]:
    """채택 → extras['roadmap'] 저장 + 남은(안 끝낸) 커리큘럼을 로드맵 단계로 교체. 바뀐(지워진) 단원 제목을 돌려준다."""
    course.extras["roadmap"] = {**road, "at": now or learning.now_text()}
    return learning.replace_remaining(course, roadmap_drafts(road), course.level)


def roadmap_text(course: learning.Course) -> str:
    """extras['roadmap'] → 근거 자료 글."""
    road = course.extras.get("roadmap") or {}
    if not road:
        return ""
    lines = [f"- 목표 직무: {road.get('target_role') or '-'} · 기간: {road.get('deadline') or '-'}"
             + (f" · 주당 {road['weekly_hours']}시간" if road.get("weekly_hours") else "")]
    for c in road.get("competencies") or []:
        lines.append(f"- 역량 {c.get('name')}: 필요 {c.get('need') or '-'} / 지금 {c.get('now') or '-'} → 채울 것 {c.get('gap') or '-'}")
    if road.get("certs"):
        lines.append(f"- 자격증: {', '.join(road['certs'])}")
    if road.get("portfolio"):
        lines.append(f"- 포트폴리오로 남길 것: {', '.join(road['portfolio'])}")
    lines.append(f"- 단계 {len(road.get('steps') or [])}개 (커리큘럼 표에 단원으로 들어가 있다)")
    return "\n".join(lines)


learning.register_context("roadmap", roadmap_text)


# ─── 🛠 개인 프로젝트 ───────────────────────────────────────────────────

PROJECT_JSON_HELP = (
    '{"projects": [{"title": "프로젝트 이름", "problem": "풀 문제", "deliverable": "결과물 (저장소·데모·보고서)", '
    '"tech": ["사용 기술"], "difficulty": "elementary", "goal": "채용에서 보여 줄 것", "portfolio_line": "포트폴리오에 쓸 문장 초안", '
    '"milestones": [{"id": "M1", "title": "끝났는지 확인할 수 있는 결과물", "due": "2주차"}]}]}'
)


def milestone_order_problems(milestones: list, label: str = "") -> list[str]:
    """마일스톤 기한 순서 — 앞에서부터 같거나 늦어져야 한다. 한 프로젝트 안에서는 'N주차' 또는 날짜 한 가지 형식."""
    problems, keys = [], []
    for i, m in enumerate(milestones, 1):
        if not isinstance(m, dict):
            continue
        k = due_key(m.get("due"))
        name = f"{label}{m.get('id') or f'{i}번째 마일스톤'}"
        if k is None:
            problems.append(f"{name}: 기한 '{m.get('due')}' — 'N주차' 또는 'YYYY-MM-DD' 로 쓴다")
            continue
        keys.append((name, k))
    kinds = {k[0] for _, k in keys}
    if len(kinds) > 1:
        return [*problems, f"{label or '프로젝트'}: 기한에 'N주차'와 날짜가 섞여 있다 — 한 가지 형식으로 쓴다"]
    for (pn, pk), (n, k) in zip(keys, keys[1:]):
        if k[1] < pk[1]:
            problems.append(f"{n}: 기한 {k[1]}{'주차' if k[0] == 'week' else ''} 이 앞 마일스톤 {pn} 의 기한 "
                            f"{pk[1]}{'주차' if pk[0] == 'week' else ''} 보다 빠르다 — 순서대로 늦어지게 고친다")
    return problems


def project_problems(data: object, level: str | None = None) -> list[str]:
    """프로젝트 계획 JSON 의 형식 문제. level = 과목 수준 → 난이도는 지금 수준 ~ 한 단계 위."""
    projects = data.get("projects") if isinstance(data, dict) else None
    if not isinstance(projects, list) or not projects:
        return ['프로젝트 JSON 이 없다 — ```json 블록에 {"projects": [...]} 형식으로 쓴다']
    problems = []
    if len(projects) > 3:
        problems.append(f"프로젝트가 {len(projects)}개다 — 1~3개만 설계한다")
    allowed = None
    if level in learning.LEVEL_IDS:
        allowed = {level, *([learning.next_level(level)] if learning.next_level(level) else [])}
    for i, p in enumerate(projects, 1):
        if not isinstance(p, dict):
            problems.append(f"{i}번째 프로젝트가 객체가 아니다")
            continue
        name = f"P{i}"
        for key, what in (("title", "이름"), ("problem", "풀 문제"), ("deliverable", "결과물"), ("portfolio_line", "포트폴리오 문장")):
            if not str(p.get(key) or "").strip():
                problems.append(f"{name}: {key}({what})가 비어 있다")
        if not _strs(p.get("tech")):
            problems.append(f"{name}: tech(사용 기술) 목록이 비어 있다")
        diff = str(p.get("difficulty") or "")
        if diff not in learning.LEVEL_IDS:
            problems.append(f"{name}: difficulty 는 수준 id ({', '.join(learning.LEVEL_IDS)}) 중 하나")
        elif allowed and diff not in allowed:
            names = " 또는 ".join(learning.level_of(x).name for x in learning.LEVEL_IDS if x in allowed)
            problems.append(f"{name}: 난이도 {learning.level_of(diff).name} — 지금 수준에서 한 단계 위까지({names})로 맞춘다")
        ms = p.get("milestones")
        if not isinstance(ms, list) or not ms:
            problems.append(f"{name}: milestones(마일스톤)가 비어 있다")
            continue
        for j, m in enumerate(ms, 1):
            if not isinstance(m, dict) or not str(m.get("title") or "").strip():
                problems.append(f"{name}-M{j}: 마일스톤 title(확인할 수 있는 결과물)이 비어 있다")
            elif not str(m.get("due") or "").strip():
                problems.append(f"{name}-M{j}: 마일스톤 due(기한)가 비어 있다")
    return problems


def all_order_problems(projects: list) -> list[str]:
    out = []
    for i, p in enumerate(projects, 1):
        if isinstance(p, dict) and isinstance(p.get("milestones"), list):
            out += milestone_order_problems(p["milestones"], f"{p.get('id') or f'P{i}'}-")
    return out


def _next_id(prefix: str, used: list[str]) -> str:
    nums = [int(x[len(prefix):]) for x in used if x.startswith(prefix) and x[len(prefix):].isdigit()]
    return f"{prefix}{max(nums, default=0) + 1}"


def normalize_milestone(m: dict, used: list[str]) -> dict:
    mid = str(m.get("id") or "").strip()
    if not mid or mid in used:
        mid = _next_id("M", used)
    used.append(mid)
    return {"id": mid, "title": str(m.get("title") or "").strip(), "due": str(m.get("due") or "").strip(),
            "done": bool(m.get("done")), "blocked": bool(m.get("blocked")), "memo": str(m.get("memo") or ""), "done_at": ""}


def projects_from_answer(text: str, level: str | None = None) -> list[dict]:
    """채택한 답변 → 저장할 프로젝트 목록 (id 는 저장할 때 매긴다). 형식이 틀리면 ValueError."""
    data = learning.find_json(text, "projects")
    problems = project_problems(data, level) + (all_order_problems(data["projects"]) if isinstance(data, dict) else [])
    if problems:
        raise ValueError(" / ".join(problems[:4]))
    out = []
    for p in data["projects"]:
        used: list[str] = []
        out.append({"title": str(p["title"]).strip(), "problem": str(p.get("problem") or ""), "deliverable": str(p.get("deliverable") or ""),
                    "tech": _strs(p.get("tech")), "difficulty": str(p.get("difficulty") or ""), "goal": str(p.get("goal") or ""),
                    "portfolio_line": str(p.get("portfolio_line") or ""), "status": "계획",
                    "milestones": [normalize_milestone(m, used) for m in p["milestones"] if isinstance(m, dict)]})
    return out


def add_projects(course: learning.Course, projects: list[dict], now: str | None = None) -> list[dict]:
    """과목에 프로젝트를 더한다 (이미 있는 프로젝트는 그대로, 새 id 는 P 다음 번호)."""
    have = course.extras.setdefault("projects", [])
    added = []
    for p in projects:
        pid = _next_id("P", [str(x.get("id") or "") for x in have])
        item = {**p, "id": pid, "at": now or learning.now_text()}
        have.append(item)
        added.append(item)
    return added


def find_project(course: learning.Course, pid: str) -> dict:
    p = next((x for x in course.extras.get("projects") or [] if x.get("id") == pid), None)
    if p is None:
        raise KeyError(f"과목 {course.id} 에 프로젝트 {pid} 가 없습니다")
    return p


def find_milestone(project: dict, mid: str) -> dict:
    m = next((x for x in project.get("milestones") or [] if x.get("id") == mid), None)
    if m is None:
        raise KeyError(f"프로젝트 {project.get('id')} 에 마일스톤 {mid} 가 없습니다")
    return m


def auto_status(project: dict) -> str:
    """마일스톤으로 정하는 상태 — 모두 끝 = 완료, 하나라도 끝·막힘 = 진행 중, 아니면 그대로 (보류는 사람이 고른 그대로)."""
    ms = project.get("milestones") or []
    status = project.get("status") or "계획"
    if status == "보류":
        return status
    if ms and all(m.get("done") for m in ms):
        return "완료"
    if any(m.get("done") or m.get("blocked") for m in ms):
        return "진행 중"
    return "계획" if status == "완료" else status


def set_milestone(course: learning.Course, pid: str, mid: str, *, done: bool | None = None, blocked: bool | None = None,
                  memo: str | None = None, now: str | None = None) -> dict:
    """마일스톤 체크 — 완료 · 막힘 · 메모 (화면 체크칸이 이 함수만 쓴다). 프로젝트 상태도 함께 맞춘다."""
    p = find_project(course, pid)
    m = find_milestone(p, mid)
    if done is not None:
        m["done"], m["done_at"] = bool(done), (now or learning.now_text()) if done else ""
        if done:
            m["blocked"] = False
    if blocked is not None:
        m["blocked"] = bool(blocked)
    if memo is not None:
        m["memo"] = " ".join(str(memo).split())[:300]
    p["status"] = auto_status(p)
    return m


def set_project_status(course: learning.Course, pid: str, status: str) -> None:
    if status not in PROJECT_STATUSES:
        raise ValueError(f"상태는 {', '.join(PROJECT_STATUSES)} 중 하나")
    find_project(course, pid)["status"] = status


def remove_project(course: learning.Course, pid: str) -> None:
    find_project(course, pid)
    course.extras["projects"] = [p for p in course.extras.get("projects") or [] if p.get("id") != pid]


def next_milestone(project: dict) -> dict | None:
    return next((m for m in project.get("milestones") or [] if not m.get("done")), None)


def project_pct(project: dict) -> int:
    ms = project.get("milestones") or []
    return sum(bool(m.get("done")) for m in ms) * 100 // len(ms) if ms else 0


def projects_text(course: learning.Course) -> str:
    """extras['projects'] → 근거 자료 글 (진행 점검이 이 표를 근거로 다음 할 일을 정한다)."""
    rows = []
    for p in course.extras.get("projects") or []:
        nxt = next_milestone(p)
        rows.append(f"#### {p.get('id')} · {p.get('title')} — {p.get('status') or '계획'} · 진행 {project_pct(p)}%"
                    + (f" · 다음: {nxt['id']} {nxt['title']}" if nxt else " · 모든 마일스톤 끝"))
        rows.append(f"- 문제: {p.get('problem') or '-'} · 결과물: {p.get('deliverable') or '-'} · 기술: {', '.join(p.get('tech') or []) or '-'}"
                    + (f" · 난이도: {learning.level_of(p['difficulty']).name}" if p.get("difficulty") in learning.LEVEL_IDS else ""))
        rows += ["| 마일스톤 | 기한 | 상태 | 메모 |", "|---|---|---|---|"]
        for m in p.get("milestones") or []:
            state = "✅ 완료" if m.get("done") else ("⛔ 막힘" if m.get("blocked") else "")
            rows.append(f"| {m.get('id')} {m.get('title')} | {m.get('due') or '-'} | {state} | {m.get('memo') or ''} |")
    return "\n".join(rows)


learning.register_context("projects", projects_text)


# ─── 📌 진행 점검 ──────────────────────────────────────────────────────

CHECK_JSON_HELP = (
    '{"check": {"project": "P1", "summary": "지금 상태 한 줄", "next": ["다음 할 일"], '
    '"unblock": [{"milestone": "M2", "how": "막힌 곳 해결책"}], '
    '"schedule": [{"id": "M3", "due": "6주차", "title": "(바꿀 때만) 새 제목"}], "add": [{"title": "새 마일스톤", "due": "7주차"}]}}'
)


def check_problems(data: object, course: learning.Course | None) -> list[str]:
    """진행 점검 JSON 형식 — 프로젝트·마일스톤 id 가 과목에 있는지 (course 가 있을 때), 다음 할 일이 있는지."""
    chk = data.get("check") if isinstance(data, dict) else None
    if not isinstance(chk, dict):
        return ['진행 점검 JSON 이 없다 — ```json 블록에 {"check": {...}} 형식으로 쓴다']
    problems = []
    if not _strs(chk.get("next")):
        problems.append("check.next(다음 할 일)가 비어 있다")
    pid = str(chk.get("project") or "")
    project = None
    if course is not None:
        projects = course.extras.get("projects") or []
        if not projects:
            return [*problems, "이 과목에는 아직 프로젝트가 없다 — 먼저 '🛠 개인 프로젝트 설계'를 채택한다"]
        project = next((p for p in projects if p.get("id") == pid), None)
        if project is None:
            return [*problems, f"check.project '{pid}' 가 과목에 없다 — 근거 자료의 프로젝트 id ({', '.join(p['id'] for p in projects)}) 중 하나"]
    have = {m.get("id") for m in (project or {}).get("milestones") or []}
    for key in ("unblock", "schedule"):
        for i, x in enumerate(_list(chk.get(key)), 1):
            mid = str((x or {}).get("milestone" if key == "unblock" else "id") or "") if isinstance(x, dict) else ""
            if not mid:
                problems.append(f"check.{key} {i}번째: 마일스톤 id 가 없다")
            elif project is not None and mid not in have:
                problems.append(f"check.{key} {i}번째: 마일스톤 {mid} 가 {pid} 에 없다 (새 마일스톤은 add 에)")
    for i, x in enumerate(_list(chk.get("add")), 1):
        if not isinstance(x, dict) or not str(x.get("title") or "").strip() or not str(x.get("due") or "").strip():
            problems.append(f"check.add {i}번째: 새 마일스톤은 title·due 가 필요하다")
    return problems


def apply_check(course: learning.Course, chk: dict) -> dict:
    """진행 점검 채택 → 기한·제목 고치기 + 새 마일스톤 더하기 (끝낸 마일스톤의 완료 표시는 그대로). 바뀐 내용을 돌려준다."""
    p = find_project(course, str(chk.get("project") or ""))
    changed, added = [], []
    for x in _list(chk.get("schedule")):
        if not isinstance(x, dict):
            continue
        m = find_milestone(p, str(x.get("id") or ""))
        if str(x.get("due") or "").strip() and x["due"] != m.get("due"):
            changed.append(f"{m['id']} {m.get('due')} → {x['due']}")
            m["due"] = str(x["due"]).strip()
        if str(x.get("title") or "").strip() and x["title"] != m.get("title"):
            m["title"] = str(x["title"]).strip()
    used = [str(m.get("id")) for m in p.get("milestones") or []]
    for x in _list(chk.get("add")):
        if isinstance(x, dict) and str(x.get("title") or "").strip():
            m = normalize_milestone({"title": x["title"], "due": x.get("due")}, used)
            p.setdefault("milestones", []).append(m)
            added.append(m["id"])
    # 순서는 그대로 둔다 (새 마일스톤은 맨 뒤) — 기한이 거꾸로 되면 검증기 milestone_order 가 잡는다
    p["status"] = auto_status(p)
    p["last_check"] = {"summary": str(chk.get("summary") or ""), "next": _strs(chk.get("next")), "at": learning.now_text()}
    return {"project": p["id"], "changed": changed, "added": added}


# ─── 🎯 JD 추천·지원 준비 ───────────────────────────────────────────────

JD_JSON_HELP = (
    '{"jd_advice": {"roles": [{"title": "직무명 (신입)", "why": "지금 수준·프로젝트로 맞는 이유", "keywords": ["JD 키워드"]}], '
    '"seniority": "신입", "keywords": ["검색·지원에 쓸 키워드"], "gaps": [{"need": "부족한 요건", "how": "채우는 방법"}], '
    '"add_units": [{"title": "커리큘럼에 더할 단원", "goals": ["마치면 할 수 있는 것"]}], '
    '"queries": ["직무명 + 일반 기술 용어 검색어"]}}'
)
MAX_JD_LINKS = 60  # 과목 하나에 남기는 JD 기록 수 (오래된 것부터 지운다)


def learning_record(course: learning.Course) -> str:
    """'학습 기록' 글 — JD 요구사항 1차 대응·커리어부로 넘길 요약에 쓴다. **끝낸 것만** 넣는다:
    끝낸 단원 · 끝낸 마일스톤 · 완료한 프로젝트의 기술·포트폴리오 문장 (계획만 한 프로젝트의 기술이 '근거 있음'이 되지 않게)."""
    lines = [f"[학습 기록] {course.topic} · 지금 수준 {course.level_name}"]
    for u in course.curriculum:
        if u.done:
            lines.append(f"- 끝낸 단원: {u.title} — {' / '.join(u.goals)}")
    for p in course.extras.get("projects") or []:
        done = [m["title"] for m in p.get("milestones") or [] if m.get("done")]
        if p.get("status") == "완료":
            lines.append(f"- 완료한 개인 프로젝트 {p.get('title')}: {p.get('problem') or ''} · 기술 {', '.join(p.get('tech') or [])}")
            if p.get("portfolio_line"):
                lines.append(f"  - {p['portfolio_line']}")
        elif done:
            lines.append(f"- 진행 중인 개인 프로젝트 {p.get('title')}")
        if done:
            lines.append(f"  - 끝낸 마일스톤: {' / '.join(done)}")
    return "\n".join(lines)


def jd_entries(pairs: list[tuple[str, str]], course: learning.Course, now: str | None = None) -> list[dict]:
    """JD [(이름, 글)] → 과목에 남길 JD 기록. 요구사항·1차 대응·점수는 커리어부 코드가 '학습 기록'에 대고 계산한다
    (모델 판정 없이 키워드만 — 정식 점수는 커리어부 '🎯 업무(JD) 추천'에서 경력 문서로)."""
    from core import career

    posts = career.job_posts(pairs)
    source = learning_record(course)
    fits = []
    for jd in posts:
        cover = career.coverage(jd.reqs, source)
        fits.append({"jd": jd.id, "coverage": [{"id": c.req.id, "status": c.status, "quote": ""} for c in cover]})
    answer = "```json\n" + json.dumps({"jd_fit": fits}, ensure_ascii=False) + "\n```"
    scores = career.score_jds(answer, [j.as_option() for j in posts])  # source 없이 — 인용 대조는 하지 않는다 (키워드 판정)
    stamp = now or learning.now_text()
    out = []
    for jd, score in zip(posts, scores):
        gaps = [r["text"] for r in score.rows if r["status"] == "공백"]
        out.append({"kind": "jd", "title": jd.title, "url": "", "at": stamp, "name": jd.name,
                    "note": (f"학습 기록 기준 1차 적합도 {score.score:g}점 (키워드 · 코드 계산)" if score.score is not None
                             else "요구사항을 찾지 못함"),
                    "score": score.score, "basis": score.basis,
                    "reqs": [{"id": r["id"].split("-", 1)[-1], "section": career.SECTION_LABEL.get(r["section"], "기타"),
                              "text": r["text"], "status": r["status"]} for r in score.rows],
                    "gaps": gaps[:10], "text": jd.text[:4000]})
    return out


def add_links(course: learning.Course, entries: list[dict]) -> None:
    course.extras["jd_links"] = [*(course.extras.get("jd_links") or []), *entries][-MAX_JD_LINKS:]


def private_text(course: learning.Course) -> str:
    """검색어 검사에 쓰는 '내 글' — 목표(회사 이름이 있을 수 있다)·질문·프로젝트·붙여 넣은 JD."""
    parts = [course.goal, learning_record(course), *(q.question for q in course.qa_log)]
    parts += [str(x.get("text") or "") for x in course.extras.get("jd_links") or []]
    return "\n".join(parts)


def target_role(course: learning.Course) -> str:
    return str((course.extras.get("roadmap") or {}).get("target_role") or course.topic).strip()


def query_problems(query: str, course: learning.Course | None) -> list[str]:
    """커리어부 검색어 검사 그대로 (숫자·이메일·주소·따옴표·내 글에 나온 이름 같은 말)."""
    from core import career_search

    return career_search.query_problems(query, private_text(course) if course else "", target_role(course) if course else "")


def advice_problems(data: object, course: learning.Course | None) -> list[str]:
    adv = data.get("jd_advice") if isinstance(data, dict) else None
    if not isinstance(adv, dict):
        return ['JD 추천 JSON 이 없다 — ```json 블록에 {"jd_advice": {...}} 형식으로 쓴다']
    problems = []
    roles = [r for r in _list(adv.get("roles")) if isinstance(r, dict) and str(r.get("title") or "").strip()]
    if not roles:
        problems.append("jd_advice.roles(추천 직무명)가 비어 있다")
    if not _strs(adv.get("keywords")):
        problems.append("jd_advice.keywords(키워드)가 비어 있다")
    if str(adv.get("seniority") or "") not in SENIORITY:
        problems.append(f"jd_advice.seniority 는 {' / '.join(SENIORITY)} 중 하나")
    if not isinstance(adv.get("gaps"), list):
        problems.append("jd_advice.gaps(부족한 요건) 목록이 없다 (없으면 빈 목록)")
    for i, u in enumerate(_list(adv.get("add_units")), 1):
        if not isinstance(u, dict) or not str(u.get("title") or "").strip():
            problems.append(f"jd_advice.add_units {i}번째: 단원 title 이 비어 있다")
    for q in _strs(adv.get("queries")):
        why = query_problems(q, course)
        if why:
            problems.append(f"검색어 '{q}': {' / '.join(why)} — 직무명·일반 기술 용어만 쓴다")
    return problems


def advice_from_answer(text: str, course: learning.Course | None) -> dict:
    data = learning.find_json(text, "jd_advice")
    problems = advice_problems(data, course)
    if problems:
        raise ValueError(" / ".join(problems[:4]))
    return data["jd_advice"]


def apply_advice(course: learning.Course, adv: dict, pairs: list[tuple[str, str]], now: str | None = None) -> dict:
    """JD 추천 채택 → 붙여 넣은 JD 기록(코드 계산) + 추천 기록 + 부족한 요건 단원을 커리큘럼 맨 뒤에."""
    stamp = now or learning.now_text()
    jds = jd_entries(pairs, course, stamp) if pairs else []
    roles = [r for r in _list(adv.get("roles")) if isinstance(r, dict)]
    advice = {"kind": "advice", "url": "", "at": stamp,
              "title": " · ".join(str(r.get("title")) for r in roles)[:120],
              "note": f"{adv.get('seniority')} · 키워드 {', '.join(_strs(adv.get('keywords'))[:8])}",
              "roles": [{"title": str(r.get("title") or ""), "why": str(r.get("why") or ""), "keywords": _strs(r.get("keywords"))} for r in roles],
              "seniority": str(adv.get("seniority") or ""), "keywords": _strs(adv.get("keywords")),
              "gaps": [{"need": str(g.get("need") or ""), "how": str(g.get("how") or "")} for g in _list(adv.get("gaps")) if isinstance(g, dict)],
              "queries": _strs(adv.get("queries"))}
    units = append_units(course, [u for u in _list(adv.get("add_units")) if isinstance(u, dict)])
    add_links(course, [*jds, advice])
    return {"jds": jds, "units": [u.id for u in units]}


def jd_links_text(course: learning.Course) -> str:
    """extras['jd_links'] → 근거 자료 글 (최근 JD 3개의 요구사항·1차 대응, 최근 추천, 승인 검색 결과 수)."""
    links = course.extras.get("jd_links") or []
    if not links:
        return ""
    lines = []
    for j in [x for x in links if x.get("kind") == "jd"][-3:]:
        lines.append(f"#### JD: {j.get('title')} — {j.get('note')}")
        lines += ["| 요구사항 | 구분 | 1차 대응 (학습 기록 · 키워드) |", "|---|---|---|"]
        lines += [f"| {r['text'][:80]} | {r['section']} | {r['status']} |" for r in j.get("reqs") or []]
    adv = next((x for x in reversed(links) if x.get("kind") == "advice"), None)
    if adv:
        lines.append(f"- 지난 추천 ({adv['at'][:10]}): {adv.get('title')} · {adv.get('note')}")
        lines += [f"  - 부족: {g['need']} → {g['how']}" for g in adv.get("gaps") or []]
    found = [x for x in links if x.get("kind") == "search"]
    if found:
        lines.append(f"- 승인한 검색 결과 {len(found)}건 (예: {found[-1].get('title')})")
    return "\n".join(lines)


learning.register_context("jd_links", jd_links_text)


def draft_queries(course: learning.Course, max_queries: int = 3) -> list[str]:
    """검색어 초안 — 지난 추천의 검색어 중 검사를 통과한 것, 없으면 커리어부 규칙(직무명 + 일반 기술 용어 + 연도)."""
    adv = next((x for x in reversed(course.extras.get("jd_links") or []) if x.get("kind") == "advice"), None)
    queries = [q for q in (adv or {}).get("queries") or [] if not query_problems(q, course)]
    if not queries:
        from core import career, career_search

        terms = career.tech_terms(learning_record(course))
        queries = [q for q in career_search.fallback_queries(target_role(course), terms, max_queries) if not query_problems(q, course)]
    return queries[:max_queries]


def career_handoff(course: learning.Course, jd_text: str = "") -> dict:
    """커리어부 '🎯 업무(JD) 추천' 양식에 채울 값 — JD · 희망 방향 · 학습 기록 요약(끝낸 것만, 계획은 넣지 않는다)."""
    links = course.extras.get("jd_links") or []
    if not jd_text:
        jds = [x for x in links if x.get("kind") == "jd" and x.get("text")][-3:]
        jd_text = "\n---\n".join(x["text"] for x in jds)
    adv = next((x for x in reversed(links) if x.get("kind") == "advice"), None)
    direction = f"{target_role(course)}로 직무 전환" + (f" — {adv['title']}" if adv and adv.get("title") else "")
    return {"jd": jd_text, "direction": direction[:150], "profile_text": learning_record(course)}


# ─── 검증기 ─────────────────────────────────────────────────────────


@register
class RoadmapPlanValidator(Validator):
    name = "roadmap_plan"
    description = "합격 로드맵 JSON — 주차·단원 형식, 주차마다 시간 합 ≤ 주당 학습 시간 (교육부 직무 전환 · 코드 계산)"

    def check(self, answer, options):
        data = learning.find_json(answer, "roadmap")
        limit = options.get("weekly_hours")
        if not tenths(limit):
            course = _course(options)
            limit = course.weekly_hours if course else None
        problems = roadmap_problems(data, limit)
        if problems:
            return self.fail("로드맵 형식 오류: " + " / ".join(problems[:6]) + f"\n형식 예: ```json\n{ROADMAP_JSON_HELP}\n```")
        hours = week_hours(data["roadmap"]["steps"])
        peak = max(hours.values(), default=0)
        tail = f" · 가장 많은 주 {hours_text(peak)}시간 ≤ 주당 {hours_text(tenths(limit))}시간" if tenths(limit) else " · 주당 시간 정보 없음(합 검사 생략)"
        return self.ok(f"로드맵 {len(data['roadmap']['steps'])}단계 · {len(hours)}주 형식 통과{tail}")


@register
class ProjectPlanValidator(Validator):
    name = "project_plan"
    description = "개인 프로젝트 계획 JSON 1~3개 — 문제·결과물·기술·난이도(지금 수준 ~ 한 단계 위)·마일스톤 (교육부 직무 전환)"

    def check(self, answer, options):
        data = learning.find_json(answer, "projects")
        problems = project_problems(data, options.get("level"))
        if problems:
            return self.fail("프로젝트 계획 형식 오류: " + " / ".join(problems[:6]) + f"\n형식 예: ```json\n{PROJECT_JSON_HELP}\n```")
        n = len(data["projects"])
        return self.ok(f"프로젝트 {n}개 · 마일스톤 {sum(len(p['milestones']) for p in data['projects'])}개 형식 통과")


@register
class MilestoneOrderValidator(Validator):
    name = "milestone_order"
    description = "마일스톤 기한이 앞에서부터 늦어지는 순서인지 — 'N주차' 또는 'YYYY-MM-DD' (교육부 직무 전환)"

    def check(self, answer, options):
        data = learning.find_json(answer, "projects")
        if data and isinstance(data.get("projects"), list):
            problems = all_order_problems(data["projects"])
        else:
            chk = learning.find_json(answer, "check")
            course = _course(options)
            if not chk or course is None:
                return self.ok("검사할 마일스톤 일정 없음")
            try:
                trial = course.model_copy(deep=True)
                result = apply_check(trial, chk["check"])
            except (KeyError, TypeError, AttributeError):
                return self.ok("점검 JSON 의 프로젝트를 과목에서 찾지 못함 — progress_check 가 알린다")
            problems = milestone_order_problems(find_project(trial, result["project"])["milestones"], f"{result['project']}-")
        if problems:
            return self.fail("마일스톤 기한 순서 오류: " + " / ".join(problems[:6]))
        return self.ok("마일스톤 기한 순서 통과")


@register
class ProgressCheckValidator(Validator):
    name = "progress_check"
    description = "진행 점검 JSON — 프로젝트·마일스톤 id 가 과목에 있는지, 다음 할 일 (교육부 직무 전환)"

    def check(self, answer, options):
        problems = check_problems(learning.find_json(answer, "check"), _course(options))
        if problems:
            return self.fail("진행 점검 형식 오류: " + " / ".join(problems[:6]) + f"\n형식 예: ```json\n{CHECK_JSON_HELP}\n```")
        return self.ok("진행 점검 형식 통과")


@register
class JdAdviceValidator(Validator):
    name = "jd_advice"
    description = "JD 추천 JSON — 직무명·키워드·신입/경력·부족한 요건·더할 단원, 검색어에 개인 정보 없음 (교육부 직무 전환)"

    def check(self, answer, options):
        problems = advice_problems(learning.find_json(answer, "jd_advice"), _course(options))
        if problems:
            return self.fail("JD 추천 형식 오류: " + " / ".join(problems[:6]) + f"\n형식 예: ```json\n{JD_JSON_HELP}\n```")
        return self.ok("JD 추천 형식 통과 (검색어 개인 정보 검사 통과)")
