"""🏢 빌딩 — 부서·층 계산, 팀 요약 카드, 화면용 스냅샷, 🧭 부서장 배정 (화면 그리기 없음).

  config/departments.yaml : 부서 → 팀 (이름·이모지·색·설명). 적지 않은 팀은 '기타' 부서로.
  층 규칙 (plan_floors)   : 1층 = 로비. 2층부터 부서 순서대로 팀 방을 채우고, 한 층에 rooms_per_floor 개까지.
                            부서는 되도록 한 층에 모은다 — 지금 층에 남은 칸이 모자라면 다음 층에서 시작.
  화면 (building_ui.py + office_component/building.js)
    파이썬은 배치(타일 좌표)와 방마다 상태·요약 카드만 보낸다. 걷기·깜빡임·낮밤은 브라우저가 그린다.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from core.config import CONFIG_DIR, load_yaml
from core.office import DEFAULT_COLOR, RANK_COLORS
from core.schemas import OrgNode, Rank, Team

DEPARTMENTS_FILE = CONFIG_DIR / "departments.yaml"
DEFAULT_PER_FLOOR = 4
OTHER = {"id": "etc", "name": "기타", "emoji": "📦", "color": "#64748B", "description": "아직 부서를 정하지 않은 팀"}
HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")

# 배치 (타일 단위, 1타일 = 16px) — 가로: 벽 | 층 번호판 | 팀 방 × rooms_per_floor | 엘리베이터 | 벽
PLATE_W, ROOM_W, SHAFT_W = 6, 10, 3
FLOOR_H = 5  # 한 층 높이 (맨 위 1줄은 부서 띠)
SKY_H, ROOF_H, GROUND_H = 3, 2, 2

TOOL_LABELS = {
    "web_search": "🌐 웹 검색", "travel_calendar": "📅 여행 달력", "stock_data": "📈 주가·재무 데이터",
    "floorplan_input": "📐 첨부 도면 읽기", "governance_evidence": "🧾 증적 모으기", "iac_scan": "🔎 IaC 규칙 점검",
}
FEATURE_LABELS = {
    "route_map": "🗺 경로 지도", "music_track": "🎵 곡 만들기·편집", "concept_catalog": "📚 개념 카탈로그",
    "floorplan_render": "🏠 평면도·3D",
}
STATUS_TEXT = {"running": "● 일하는 중", "waiting": "● 결정 기다리는 중", "error": "● 오류로 멈춤", "idle": "○ 쉬는 중"}
DECISION_TEXT = {"adopted": "채택", "rejected": "반려", "awaiting_human": "결정 대기", "reviewing": "진행 중", "error": "오류"}


@dataclass
class Department:
    id: str
    name: str
    emoji: str
    color: str
    description: str
    teams: list[str] = field(default_factory=list)


# ─── 부서·층 ─────────────────────────────────────────────────────────


def load_config(path: Path = DEPARTMENTS_FILE) -> dict:
    """departments.yaml — 없으면 빈 설정 (모든 팀이 '기타')."""
    return load_yaml(path) if Path(path).exists() else {}


def per_floor(cfg: dict) -> int:
    return max(1, int(cfg.get("rooms_per_floor") or DEFAULT_PER_FLOOR))


def _dept(raw: dict, teams: list[str]) -> Department:
    color = str(raw.get("color") or OTHER["color"])
    return Department(id=str(raw["id"]), name=str(raw.get("name") or raw["id"]), emoji=str(raw.get("emoji") or "🏢"),
                      color=color if HEX.match(color) else OTHER["color"], description=str(raw.get("description") or ""),
                      teams=teams)


def departments(team_ids: list[str], cfg: dict) -> list[Department]:
    """설정 순서대로 부서 목록. 없는 팀 id 는 건너뛰고(팀 폴더를 지운 경우), 두 번 적은 팀은 처음 부서에만,
    어느 부서에도 없는 팀은 맨 끝 '기타' 부서에 모은다 → 새 팀은 늘 빌딩에 보인다."""
    known, placed, out = set(team_ids), set(), []
    for raw in cfg.get("departments") or []:
        teams = [t for t in raw.get("teams") or [] if t in known and t not in placed]
        placed.update(teams)
        if teams:
            out.append(_dept(raw, teams))
    rest = [t for t in team_ids if t not in placed]
    if rest:
        out.append(_dept({**OTHER, **(cfg.get("other") or {})}, rest))
    return out


def plan_floors(depts: list[Department], rooms_per_floor: int) -> list[list[tuple[str, str]]]:
    """[층마다 [(부서 id, 팀 id) …]] — 0번이 2층 (1층은 로비). 팀이 없으면 []."""
    floors: list[list[tuple[str, str]]] = [[]]
    for d in depts:
        rooms = [(d.id, t) for t in d.teams]
        if floors[-1] and len(floors[-1]) + len(rooms) > rooms_per_floor:
            floors.append([])  # 부서를 쪼개지 않게 새 층에서 시작
        for room in rooms:
            if len(floors[-1]) >= rooms_per_floor:
                floors.append([])  # 한 층보다 큰 부서는 다음 층으로 이어서
            floors[-1].append(room)
    return [f for f in floors if f]


def build_layout(floors: list[list[tuple[str, str]]], rooms_per_floor: int) -> dict:
    """층 목록 → 빌딩 배치 (타일 좌표). 위에서부터: 하늘 · 옥상 간판 · 맨 위층 … 2층 · 1층 로비 · 길."""
    n = len(floors)
    rooms_x = 1 + PLATE_W
    shaft_x = rooms_x + rooms_per_floor * ROOM_W
    W = shaft_x + SHAFT_W + 1
    top = SKY_H + ROOF_H
    lobby_y = top + n * FLOOR_H
    H = lobby_y + FLOOR_H + GROUND_H
    out_floors = []
    for i, rooms in enumerate(floors):
        y = top + (n - 1 - i) * FLOOR_H
        slots, banners = [], []
        for slot in range(rooms_per_floor):
            x = rooms_x + slot * ROOM_W
            if slot < len(rooms):
                dept, team = rooms[slot]
                slots.append({"team": team, "dept": dept, "x": x, "y": y + 1, "w": ROOM_W, "h": FLOOR_H - 1})
                if banners and banners[-1]["dept"] == dept:
                    banners[-1]["w"] += ROOM_W
                else:
                    banners.append({"dept": dept, "x": x, "y": y, "w": ROOM_W, "h": 1})
            else:  # 빈 사무실 — 팀이 늘면 여기부터 찬다
                slots.append({"team": "", "dept": "", "x": x, "y": y + 1, "w": ROOM_W, "h": FLOOR_H - 1})
        out_floors.append({"n": i + 2, "y": y, "rooms": slots, "banners": banners})
    span = rooms_per_floor * ROOM_W
    q = span // 4
    spots = [  # 로비: 누르면 메인 아래 섹션이 바뀐다 (building_ui.SECTIONS)
        {"id": "brief", "kind": "reception", "label": "📰 안내 데스크", "x": rooms_x, "w": q},
        {"id": "user", "kind": "entrance", "label": "👤 사용자", "x": rooms_x + q, "w": q},
        {"id": "autonomy", "kind": "approval", "label": "🤖 승인 데스크", "x": rooms_x + 2 * q, "w": q},
        {"id": "usage", "kind": "office", "label": "💰 관리실", "x": rooms_x + 3 * q, "w": span - 3 * q},
    ]
    return {
        "key": f"bld:{n}x{rooms_per_floor}:" + ",".join(r[1] for f in floors for r in f),
        "w": W, "h": H, "floors": out_floors, "plate": {"x": 1, "w": PLATE_W},
        "shaft": {"x": shaft_x, "w": SHAFT_W, "y": top, "h": lobby_y + FLOOR_H - top},
        "roof": {"y": SKY_H, "h": ROOF_H}, "lobby": {"y": lobby_y, "h": FLOOR_H, "spots": spots},
        "ground": {"y": lobby_y + FLOOR_H, "h": GROUND_H},
    }


# ─── 팀 요약 카드 (team.yaml 에서) ─────────────────────────────────────


def member_line(org: OrgNode, ranks: list[Rank]) -> tuple[str, int]:
    """('팀장 1 · 부장 2 · 대리 2 · 사원 2', 7) — 높은 직급부터."""
    nodes = org.walk()
    order = {r.id: i for i, r in enumerate(ranks)}
    names = {r.id: r.name for r in ranks}
    counts: dict[str, int] = {}
    for node in nodes:
        counts[node.rank] = counts.get(node.rank, 0) + 1
    parts = [f"{names.get(r, r)} {c}" for r, c in sorted(counts.items(), key=lambda kv: -order.get(kv[0], -1))]
    return " · ".join(parts), len(nodes)


def team_card(team: Team, org: OrgNode, ranks: list[Rank]) -> dict:
    """마우스를 올리면 보이는 팀 요약 — 성격(설명)·팀장·직급별 인원·주요 업무·툴/기능."""
    rank_names = {r.id: r.name for r in ranks}
    members, headcount = member_line(org, ranks)
    lead = org.name or rank_names.get(org.rank, org.rank)
    return {
        "description": team.description,
        "lead": f"{lead} — {org.specialty}" if org.specialty else lead,
        "members": members, "headcount": headcount,
        "tasks": [t.title for t in team.tasks] or ["💬 자유 질문"],
        "tools": [TOOL_LABELS.get(t, t) for t in team.tools] + [FEATURE_LABELS.get(f, f) for f in team.features],
    }


def team_status(team_id: str, open_runs: list[dict]) -> tuple[str, int]:
    """(상태, 일하는 팀원 수) — running / waiting / error / idle. open_runs 는 결정 전 실행들 (app.py 의 runs)."""
    mine = [r for r in open_runs if r["team_id"] == team_id]
    working = sum(1 for r in mine if not r["done"] for label in r["activity"].values() if not label.startswith("⏳"))
    if any(not r["done"] for r in mine):
        return "running", working
    if any(r.get("error") for r in mine):
        return "error", 0
    return ("waiting", 0) if mine else ("idle", 0)


def recent_by_team(rows: list[dict], team_for) -> dict[str, dict]:
    """팀별 가장 최근 업무 한 줄 — rows 는 최신 순 (storage.list_runs). team_for = 옛 팀 id → 지금 팀 id."""
    out: dict[str, dict] = {}
    for row in rows:
        tid = team_for(row["team_id"], row.get("question") or "")
        if tid in out:
            continue
        text = " ".join((row.get("question") or "").split())
        when = (row.get("decided_at") or row.get("created_at") or "")[5:16].replace("T", " ")
        out[tid] = {"text": text[:40] + ("…" if len(text) > 40 else ""), "when": when,
                    "status": DECISION_TEXT.get(row.get("status"), row.get("status") or "")}
    return out


def team_costs(usage: dict[str, dict], team_for) -> tuple[dict[str, float], dict[str, float]]:
    """({팀 id: 자기 비용}, {이어 받은 팀 id: 옛 팀 비용}) — 옛 팀 몫은 층 합계에만 쓴다 (같은 부서라 같은 층)."""
    own: dict[str, float] = {}
    legacy: dict[str, float] = {}
    for tid, row in usage.items():
        heir = team_for(tid, "")
        target = own if heir == tid else legacy
        target[heir] = target.get(heir, 0.0) + float(row.get("cost_usd") or 0.0)
    return own, legacy


def money(value: float) -> str:
    return "$0" if value <= 0 else f"${value:,.2f}" if value >= 1 else f"${value:.3f}"


# ─── 화면용 스냅샷 ───────────────────────────────────────────────────


def snapshot(teams: list[Team], orgs: dict[str, OrgNode], ranks: list[Rank], cfg: dict, *, open_runs: list[dict],
             recent: dict[str, dict], costs: dict[str, float] | None, cost_days: int = 7,
             legacy_costs: dict[str, float] | None = None) -> dict:
    """화면에 보낼 전체 상태. costs = 팀 자기 id 의 비용 (None 이면 비용 미터를 그리지 않는다).
    legacy_costs = 옛 팀 id 의 비용을 이어 받은 팀에 붙인 것 — 층 합계에만 더한다 (어느 팀 몫인지 알 수 없어서)."""
    by_id = {t.id: t for t in teams}
    depts = departments([t.id for t in teams], cfg)
    rpf = per_floor(cfg)
    floors = plan_floors(depts, rpf)
    layout = build_layout(floors, rpf)
    dept_of = {t: d for d in depts for t in d.teams}
    rooms = {}
    for tid, team in by_id.items():
        org, d = orgs[tid], dept_of[tid]
        status, working = team_status(tid, open_runs)
        card = team_card(team, org, ranks)
        rank_of = {n.id: n.rank for n in org.walk()}
        people = [org, *org.children][:4]  # 방에 앉는 사람: 팀장 + 바로 아래 칸 (최대 4명)
        rooms[tid] = {
            "id": tid, "name": team.name, "dept": d.id, "deptName": f"{d.emoji} {d.name}", "color": d.color,
            "status": status, "statusText": STATUS_TEXT[status] + (f" (팀원 {working}명)" if working else ""),
            "working": working, "recent": recent.get(tid), "card": card,
            "cost": money(costs.get(tid, 0.0)) if costs is not None else "",
            "people": [{"id": n.id, "color": RANK_COLORS.get(rank_of[n.id], DEFAULT_COLOR)} for n in people],
        }
    floor_costs = []
    for f in layout["floors"]:
        total = (sum(costs.get(r["team"], 0.0) + (legacy_costs or {}).get(r["team"], 0.0) for r in f["rooms"] if r["team"])
                 if costs is not None else None)
        floor_costs.append(money(total) if total is not None else "")
    return {
        "kind": "building", "layout": layout, "rooms": rooms, "floorCosts": floor_costs, "costDays": cost_days,
        "depts": {d.id: {"name": d.name, "emoji": d.emoji, "color": d.color, "description": d.description,
                         "teams": [by_id[t].name for t in d.teams]} for d in depts},
        "live": any(r["status"] == "running" for r in rooms.values()),
    }


def floor_of(team_id: str, teams: list[Team], cfg: dict) -> int:
    """이 팀이 있는 층 번호 (2층부터)."""
    for i, f in enumerate(plan_floors(departments([t.id for t in teams], cfg), per_floor(cfg))):
        if any(t == team_id for _, t in f):
            return i + 2
    raise KeyError(team_id)


# ─── 🧭 부서장 배정: 부서에 맡긴 요청을 알맞은 팀으로 ────────────────────────


class TeamPick(BaseModel):
    """부서장의 배정 결과."""

    analysis: str = Field(description="먼저 요청의 목적과 필요한 전문 분야를 1~2문장으로 분석")
    team_id: str = Field(description="맡길 팀 id (목록에 있는 id 그대로)")
    reason: str = Field(description="그 팀을 고른 이유 한 문장")


def _bigrams(text: str) -> set[str]:
    words = re.findall(r"[0-9A-Za-z가-힣]+", text.lower())
    return {w[i:i + 2] for w in words for i in range(max(1, len(w) - 1))}


def keyword_pick(text: str, candidates: list[Team]) -> tuple[str, str]:
    """모델 없이 낱말(두 글자 묶음)이 가장 많이 겹치는 팀 — 동점이면 부서에 먼저 적힌 팀."""
    want = _bigrams(text)
    best, best_score = candidates[0], -1
    for team in candidates:
        corpus = " ".join([team.name, team.description, *(f"{t.title} {t.description}" for t in team.tasks)])
        score = len(want & _bigrams(corpus))
        if score > best_score:
            best, best_score = team, score
    return best.id, f"요청과 겹치는 낱말이 가장 많은 팀 ({best_score}개)"


def route_messages(text: str, dept: Department, candidates: list[Team]):
    from langchain_core.messages import HumanMessage, SystemMessage  # 무거운 import 는 실제로 부를 때만

    lines = [f"- {t.id}: {t.name} — {t.description} (업무: {', '.join(x.title for x in t.tasks) or '자유 질문'})"
             for t in candidates]
    system = (f"당신은 '{dept.name}' 부서장입니다. 부서 소개: {dept.description}\n"
              "사람의 요청을 읽고 아래 팀 중 가장 알맞은 팀 하나를 고르세요. 요청을 직접 처리하지 마세요.\n\n"
              "[팀 목록]\n" + "\n".join(lines))
    return [SystemMessage(content=system), HumanMessage(content=text)]


def route_request(text: str, dept: Department, candidates: list[Team], models) -> dict:
    """{team_id, reason, how: one|llm|keyword, model, note} — 팀이 하나면 바로, 아니면 보조 모델 1회, 실패하면 낱말 비교."""
    if len(candidates) == 1:
        return {"team_id": candidates[0].id, "reason": "이 부서의 팀은 하나뿐입니다.", "how": "one", "model": "", "note": ""}
    from core import llm_factory, usage  # 순환·무거운 import 방지

    note = ""
    try:
        with usage.scope(purpose="부서장 배정"):
            call = llm_factory.invoke_structured(models.utility, models, TeamPick, route_messages(text, dept, candidates),
                                                 f"{dept.name} 부서장")
        pick: TeamPick = call.parsed
        if pick.team_id in {t.id for t in candidates}:
            return {"team_id": pick.team_id, "reason": pick.reason, "how": "llm", "model": call.spec.label, "note": ""}
        note = f"모델이 목록에 없는 팀({pick.team_id})을 골라 낱말 비교로 정했습니다."
    except Exception as e:  # noqa: BLE001 — 키 없음·한도 초과 등: 배정은 낱말 비교로라도 끝낸다
        note = f"보조 모델을 쓰지 못해 낱말 비교로 정했습니다 ({type(e).__name__})."
    team_id, reason = keyword_pick(text, candidates)
    return {"team_id": team_id, "reason": reason, "how": "keyword", "model": "", "note": note}

