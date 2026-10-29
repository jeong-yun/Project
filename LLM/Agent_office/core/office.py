"""🏢 픽셀 사무실 · 🖥️ 커맨드 센터 — 진행 상태(run dict) → 화면용 스냅샷(JSON) · 한 줄 보고 (화면 그리기 없음).

픽셀 사무실 (office_ui.py + office_component/index.html)
  파이썬은 1초마다 '지금 상태' 스냅샷만 보낸다 — 방 배치, 에이전트마다 말풍선·가야 할 자리, 최근 쪽지(✉️) 이벤트.
  걷기·쪽지 날리기 같은 움직임은 브라우저(캔버스)가 스스로 그린다 → 파이썬 쪽 다시 그리기 비용이 거의 없다.

  배치 (타일 단위, 1타일 = 16px)
    위쪽 줄 : 팀장실 · 회의실(📋 분배·🧩 통합 때 모이는 탁자) · 자료실(🌐 자료 조사 때 가는 책장) · (남으면) 휴게실
    아래    : 업무 공간 — 팀장 바로 아래 칸(부장 등)마다 '책상 묶음' 하나. 묶음 맨 위가 그 관리자, 아래 줄에 하위 팀원
    맨 아래 : 출입문 = 👤 사용자 (질문이 들어오고 최종 결과가 나가는 곳)

커맨드 센터 (command_center_ui.py)
  '다들 뭐 해?' 같은 상태 질문은 LLM 없이 run 상태로 바로 한 줄씩 답한다 (rundown).
"""

from core.live import KIND_LABELS, RETRY_LABELS, STAGES, WORKING_LABELS, stage_of
from core.schemas import OrgNode

TILE = 16
USER = "__user__"  # 쪽지 이벤트에서 '사용자(출입문)'를 가리키는 id

# 직급별 셔츠 색 (스프라이트) — 없는 직급은 회색
RANK_COLORS = {
    "team_lead": "#C0392B", "gm": "#2E6FD1", "deputy_gm": "#8E44AD", "assistant_manager": "#17A589", "staff": "#E67E22",
}
DEFAULT_COLOR = "#7F8C8D"

LABEL_KIND = {label: kind for kind, label in WORKING_LABELS.items()}
# 이 일을 하는 동안 가 있는 곳 (나머지는 자기 책상)
KIND_PLACE = {"brief": "meeting", "plan": "meeting", "integrate": "meeting", "research": "library", "tools": "library"}

# ─── 배치 ─────────────────────────────────────────────────────────────
LEAD_W, MEET_W, LIB_W = 9, 13, 9  # 위쪽 방 안쪽 너비 (타일)
TOP_Y, TOP_BOTTOM = 2, 7  # 위쪽 방 안쪽 첫 줄 · 마지막 줄 (0~1 은 바깥 벽, 8 은 방 아래 벽)
CELL_W, CELL_H = 5, 4  # 책상 한 칸
MAX_COLS = 3  # 묶음 한 줄에 놓는 하위 팀원 책상 수
FLOOR_Y = 10  # 업무 공간 첫 책상 묶음 시작 줄
MAX_W = 48


def _desk(cx: int, cy: int) -> tuple[tuple[int, int], dict]:
    """책상 한 칸 (cx, cy 는 칸 왼쪽 위) → (앉는 자리, 책상 가구). 사람은 책상 뒤(위)에 앉아 아래를 본다."""
    return (cx + 2, cy + 1), {"type": "desk", "x": cx + 1, "y": cy + 2, "w": 3, "h": 1}


def _groups(org: OrgNode) -> list[list[OrgNode]]:
    """팀장 바로 아래 칸마다 [관리자, 하위 팀원들…] — 책상 묶음 하나씩."""
    return [list(child.walk()) for child in org.children]


def build_layout(org: OrgNode) -> dict:
    """조직도 → 사무실 배치. 같은 조직도면 늘 같은 결과 (화면은 key 가 바뀔 때만 배경을 다시 그린다)."""
    seats: dict[str, list[int]] = {}
    furniture: list[dict] = []

    # 업무 공간 — 책상 묶음을 왼쪽부터 놓고, 너무 넓으면 다음 줄로
    pods, x, y, row_h, floor_w = [], 2, FLOOR_Y, 0, 0
    for group in _groups(org):
        staff = group[1:]
        cols = max(1, min(MAX_COLS, len(staff)))
        rows = -(-len(staff) // cols) if staff else 0
        w, h = cols * CELL_W, (1 + rows) * CELL_H
        if x > 2 and x + w > MAX_W - 2:
            x, y, row_h = 2, y + row_h + 1, 0
        pods.append((group, x, y, cols, w))
        floor_w = max(floor_w, x + w)
        x, row_h = x + w + 1, max(row_h, h)
    floor_bottom = y + row_h if pods else FLOOR_Y + 3

    top_w = 1 + LEAD_W + 1 + MEET_W + 1 + LIB_W + 1
    W = max(top_w, floor_w + 2)
    H = floor_bottom + 3
    shift = max(0, (W - floor_w - 2) // 2)  # 책상 묶음들을 가운데로
    for group, px, py, cols, w in pods:
        px += shift
        seat, desk = _desk(px + (w - CELL_W) // 2, py)  # 관리자는 묶음 맨 위 가운데
        seats[group[0].id] = seat
        furniture.append(desk)
        for i, node in enumerate(group[1:]):
            seat, desk = _desk(px + (i % cols) * CELL_W, py + (1 + i // cols) * CELL_H)
            seats[node.id] = seat
            furniture.append(desk)

    # 위쪽 방들
    lead_x, meet_x, lib_x = 1, 1 + LEAD_W + 1, 1 + LEAD_W + 1 + MEET_W + 1
    rooms = [
        {"kind": "lead", "label": "팀장실", "x": lead_x, "y": TOP_Y, "w": LEAD_W, "h": TOP_BOTTOM - TOP_Y + 1},
        {"kind": "meeting", "label": "회의실", "x": meet_x, "y": TOP_Y, "w": MEET_W, "h": TOP_BOTTOM - TOP_Y + 1},
        {"kind": "library", "label": "자료실", "x": lib_x, "y": TOP_Y, "w": LIB_W, "h": TOP_BOTTOM - TOP_Y + 1},
    ]
    extra_x = lib_x + LIB_W + 1
    if W - 1 - extra_x >= 5:
        rooms.append({"kind": "lounge", "label": "휴게실", "x": extra_x, "y": TOP_Y, "w": W - 1 - extra_x,
                      "h": TOP_BOTTOM - TOP_Y + 1})
    else:
        rooms[2]["w"] = W - 1 - lib_x  # 남는 폭은 자료실이 쓴다
    rooms.append({"kind": "floor", "label": "업무 공간", "x": 1, "y": TOP_BOTTOM + 2, "w": W - 2, "h": H - TOP_BOTTOM - 4})

    # 벽 [x, y, w, h] — 문은 벽에 낸 빈칸
    walls = [[0, 0, W, TOP_Y], [0, 0, 1, H], [W - 1, 0, 1, H]]
    door_x = W // 2 - 1
    walls += [[0, H - 1, door_x, 1], [door_x + 2, H - 1, W - door_x - 2, 1]]
    for room in rooms[:-1]:
        right = room["x"] + room["w"]
        if right < W - 1:
            walls.append([right, TOP_Y, 1, room["h"]])
        door = room["x"] + room["w"] // 2 - 1  # 방마다 아래 벽 가운데에 2칸 문
        walls += [[room["x"] - 1, TOP_BOTTOM + 1, door - room["x"] + 1, 1],
                  [door + 2, TOP_BOTTOM + 1, right - door - 2 + 1, 1]]

    # 팀장 책상 (팀장실 가운데) · 회의 탁자 · 책장 · 장식
    lead_room, meet, lib = rooms[0], rooms[1], rooms[2]
    seats[org.id] = [lead_room["x"] + 4, TOP_Y + 2]
    furniture.append({"type": "desk", "x": lead_room["x"] + 3, "y": TOP_Y + 3, "w": 3, "h": 1, "lead": True})
    furniture += [
        {"type": "rug", "x": lead_room["x"] + 2, "y": TOP_Y + 1, "w": 5, "h": 4},
        {"type": "plant", "x": lead_room["x"], "y": TOP_Y, "w": 1, "h": 1},
        {"type": "shelf", "x": lead_room["x"] + LEAD_W - 2, "y": TOP_Y, "w": 2, "h": 1},
        {"type": "table", "x": meet["x"] + 3, "y": TOP_Y + 2, "w": MEET_W - 6, "h": 2},
        {"type": "board", "x": meet["x"] + 4, "y": TOP_Y - 1, "w": 5, "h": 1},
        {"type": "plant", "x": meet["x"] + MEET_W - 1, "y": TOP_Y, "w": 1, "h": 1},
        {"type": "shelf", "x": lib["x"], "y": TOP_Y, "w": lib["w"], "h": 1},
        {"type": "plant", "x": W - 2, "y": H - 2, "w": 1, "h": 1},
        {"type": "plant", "x": 1, "y": H - 2, "w": 1, "h": 1},
    ]
    if rooms[3]["kind"] == "lounge":
        lounge = rooms[3]
        furniture += [{"type": "cooler", "x": lounge["x"], "y": TOP_Y, "w": 1, "h": 1},
                      {"type": "sofa", "x": lounge["x"] + 1, "y": TOP_BOTTOM - 1, "w": min(3, lounge["w"] - 2), "h": 1}]

    # 회의 탁자 둘레 (위쪽 줄은 아래를, 아래쪽 줄은 위를 본다) · 책장 앞
    t = furniture[[f["type"] for f in furniture].index("table")]
    # 앞쪽 자리는 3칸씩 띄워 이름표가 겹치지 않게, 자리가 모자라면 그 사이 칸을 쓴다
    spread = [i for i in range(0, t["w"], 3)] + [i for i in range(t["w"]) if i % 3]
    meeting = [[t["x"] + i, y, face] for i in spread for y, face in ((t["y"] - 1, "down"), (t["y"] + t["h"], "up"))]
    lib_cols = [i for i in range(0, lib["w"], 3)] + [i for i in range(lib["w"]) if i % 3]
    library = [[lib["x"] + i, TOP_Y + r, "up"] for i in lib_cols for r in (2, 4)]
    return {
        "key": f"{org.id}:{W}x{H}:{len(seats)}", "w": W, "h": H, "rooms": rooms, "walls": walls, "furniture": furniture,
        "seats": seats, "spots": {"meeting": meeting, "library": library}, "door": [door_x, H - 1],
    }


# ─── 지금 상태 ─────────────────────────────────────────────────────────
def agent_state(node_id: str, run: dict | None, active_ids: set[str] | None, blocked: dict[str, str]) -> dict:
    """에이전트 한 명의 {tone, bubble, kind} — 말풍선 문구와 어디로 갈지(kind) 정하기.

    tone: working / wait / done / revise / idle / blocked / error / out(이번 업무 제외)
    """
    if run:
        label = run["activity"].get(node_id)
        if label and not label.startswith("⏳"):  # '⏳ … 기다리는 중'도 activity 에 들어오지만 일하는 중은 아니다
            return {"tone": "working", "bubble": label, "kind": LABEL_KIND.get(label, "draft")}
        if run.get("error") and node_id == run["root_id"]:
            return {"tone": "error", "bubble": "⚠️ 오류로 멈춤", "kind": ""}
        state = run["statuses"].get(node_id)
        if state:
            return {"tone": state[0], "bubble": state[1], "kind": ""}
    if node_id in blocked:
        return {"tone": "blocked", "bubble": "⛔ 고른 모델 꺼짐", "kind": ""}
    if active_ids is not None and node_id not in active_ids:
        return {"tone": "out", "bubble": "이번 업무 제외" if run else "", "kind": ""}
    return {"tone": "idle", "bubble": "대기" if not run else "💤 쉬는 중", "kind": ""}


def envelope_events(run: dict | None, parents: dict[str, str], limit: int = 16) -> list[dict]:
    """✉️ 쪽지 이벤트 — 질문(사용자→팀장), 업무 분배(상사→팀원), 결과 보고(팀원→상사, 맨 위는 →사용자).

    id 는 단계 번호로 만들어 늘 같다 → 화면은 처음 보는 id 만 날려 보낸다. 최근 limit 개만.
    """
    if not run:
        return []
    rid = run["run_id"][:8]  # 같은 화면에서 다음 업무를 맡겨도 id 가 겹치지 않게
    events = [{"id": f"{rid}:q", "from": USER, "to": run["root_id"], "kind": "ask"}]
    for i, s in enumerate(run["steps"]):
        if s.kind == "plan":
            for a in s.delegation.assignments if s.delegation else []:
                if parents.get(a.child_id) != s.node_id:
                    continue  # 하위가 아닌 id 는 워크플로우도 버린다
                events.append({"id": f"{rid}:{i}:{a.child_id}", "from": s.node_id, "to": a.child_id, "kind": "delegate"})
        else:
            to = USER if s.node_id == run["root_id"] else parents.get(s.node_id)
            if to:
                events.append({"id": f"{rid}:{i}:up", "from": s.node_id, "to": to, "kind": "report"})
    return events[-limit:]


def snapshot(layout: dict, org: OrgNode, run: dict | None, *, names: dict[str, str], ranks: dict[str, tuple[str, str]],
             specialties: dict[str, str], parents: dict[str, str], active_ids: set[str] | None,
             blocked: dict[str, str], focus: str | None) -> dict:
    """화면에 보낼 전체 상태. ranks = {에이전트 id: (직급 id, 직급 이름)}, blocked = {에이전트 id: 꺼진 이유}."""
    agents, seats, spots = [], layout["seats"], layout["spots"]
    for i, node in enumerate(org.walk()):
        st = agent_state(node.id, run, active_ids, blocked)
        place = KIND_PLACE.get(st["kind"], "desk")
        if place == "desk":
            target, face = seats[node.id], "down"
        else:
            spot = spots[place][i % len(spots[place])]  # 조직도 순서로 자리를 나눠 겹치지 않게
            target, face = spot[:2], spot[2]
        rank_id, rank_name = ranks[node.id]
        agents.append({
            "id": node.id, "name": names[node.id], "rank": rank_name, "lead": node.id == org.id,
            "color": RANK_COLORS.get(rank_id, DEFAULT_COLOR), "desk": seats[node.id], "target": list(target),
            "face": face, "place": place, "tone": st["tone"], "bubble": st["bubble"],
            "about": specialties.get(node.id, ""), "dim": active_ids is not None and node.id not in active_ids,
            "note": blocked.get(node.id, ""),
        })
    return {
        "layout": layout, "agents": agents, "events": envelope_events(run, parents), "focus": focus or "",
        "running": bool(run and not run["done"]),
        "stage": STAGES[min(stage_of(run), len(STAGES) - 1)] if run else "",
    }


# ─── 커맨드 센터: 한 줄 보고 ───────────────────────────────────────────
STATUS_WORDS = ("뭐 해", "뭐해", "뭐하", "뭐 하", "상태", "진행 상황", "진행상황", "어디까지", "현황", "다들", "보고해",
                "status", "what are you up to")
NEXT_AFTER = {
    "brief": "자료 조사·업무 분배", "tools": "업무 분배", "research": "업무 분배", "plan": "팀원 결과 기다리기",
    "critic": "비평 반영 후 보고",
}


def is_status_question(text: str) -> bool:
    """'다들 뭐 해?' 같은 짧은 상태 질문인지 — 맞으면 LLM 없이 바로 답한다. 긴 문장은 업무로 본다."""
    t = " ".join(text.lower().split())
    if t in ("/status", "/s", "/st"):
        return True
    return len(t) <= 25 and any(w in t for w in STATUS_WORDS)


def _report_to(node_id: str, run: dict, parents: dict[str, str], names: dict[str, str]) -> str:
    if node_id == run["root_id"]:
        return "사용자에게 최종 보고"
    parent = parents.get(node_id)
    return f"{names.get(parent, parent)}에게 보고" if parent else "보고"


def agent_line(node_id: str, run: dict | None, *, names: dict[str, str], parents: dict[str, str],
               active_ids: set[str] | None, blocked: dict[str, str]) -> tuple[str, str, str]:
    """(아이콘, 지금, 다음) — 한 줄 보고의 재료."""
    st = agent_state(node_id, run, active_ids, blocked)
    tone, now = st["tone"], st["bubble"]
    if tone == "working":
        kind = st["kind"]
        retry = now in RETRY_LABELS.values()
        nxt = NEXT_AFTER.get(kind) or ("다시 써서 " + _report_to(node_id, run, parents, names) if retry
                                       else _report_to(node_id, run, parents, names))
        return "▶", now, nxt
    if tone == "error":
        return "⚠", now, "오류 확인 후 다시 맡기기"
    if tone == "blocked":
        return "⛔", f"고른 모델 꺼짐 ({blocked[node_id]})", "모델을 바꾸거나 켜기 — 맡으면 대안 모델 또는 실패"
    if tone == "out":
        return "·", "이번 업무 제외", "—"
    if tone == "idle":
        return "z", "쉬는 중", "새 업무 기다리기"
    if run and run["status"] == "awaiting_human" and node_id == run["root_id"]:
        return "?", "사용자 결정 대기", "채택·반려를 기다림"
    if now.startswith("⏳ 하위"):
        return "…", "팀원 결과 기다리는 중", "받은 결과 검토·통합"
    if now.startswith("⏳"):
        return "…", "차례 기다리는 중", "맡은 업무 작성"
    if run and run["done"]:
        return "✓", f"끝남 ({now})", "없음"
    return "✓", now, "상사 검토 기다리기"


def rundown(run: dict | None, org: OrgNode, *, names: dict[str, str], parents: dict[str, str],
            active_ids: set[str] | None, blocked: dict[str, str]) -> list[str]:
    """'다들 뭐 해?' → 첫 줄 요약 + 에이전트마다 한 줄 (지금 하는 일 → 다음 할 일). LLM 호출 없음."""
    members = list(org.walk())
    lines, counts = [], {"▶": 0, "…": 0, "✓": 0}
    for node in members:
        icon, now, nxt = agent_line(node.id, run, names=names, parents=parents, active_ids=active_ids, blocked=blocked)
        counts[icon] = counts.get(icon, 0) + 1
        lines.append(f"{icon} {names[node.id]} — {now}" + (f" → 다음: {nxt}" if nxt not in ("—", "없음") else ""))
    if not run:
        head = f"지금 맡은 업무가 없어요. 팀원 {len(members)}명 모두 쉬는 중입니다."
    else:
        stage = STAGES[min(stage_of(run), len(STAGES) - 1)]
        state = ("진행 중" if not run["done"] else "오류로 멈춤" if run.get("error")
                 else "결정 대기" if run["status"] == "awaiting_human" else "끝남")
        head = (f"팀원 {len(members)}명에게 물어봤어요 ({state} · 지금 단계 {stage}) — "
                f"일하는 중 {counts['▶']} · 기다리는 중 {counts['…']} · 끝남 {counts['✓']}")
    return [head, *lines]


def step_line(step, names: dict[str, str]) -> str:
    """활동 기록 한 줄 — '📝 초안 · 사원·초안 · 85점 · 12초'."""
    score = f" · {step.review.score}점" if step.review else ""
    return f"{KIND_LABELS.get(step.kind, step.kind)} · {names.get(step.node_id, step.node_id)}{score} · {step.elapsed_sec:.0f}초"
