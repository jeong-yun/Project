"""🖥️ 커맨드 센터 — 팀장에게 한 줄 남기고, 지금 사무실 상태를 탭으로 본다 (레트로 화면).

탭 (고른 탭만 그린다 → 1초마다 다시 그려도 가볍다. 데이터는 모두 기존 것을 다시 쓴다)
  ▸ 터미널   팀장에게 한 줄. '다들 뭐 해?' 같은 상태 질문은 LLM 없이 바로 한 줄씩 보고 (core/office.rundown),
             그 밖의 말은 평소처럼 업무로 맡긴다 (app.py 가 pop_request 로 받아 자유 질문으로 시작)
  📈 모니터 · ✓ 업무 · 🔔 결정 대기 · ⏱ 자동 실행 · 📚 기억 · ◆ 조직도 · 📜 활동 · ⌘ 명령어 · 🛠 팀원
"""

import html
import time
from datetime import datetime

import streamlit as st

import office_ui
from core import autonomy, office
from core.live import STAGES, stage_of

TABS = {
    "terminal": "▸ 터미널", "monitor": "📈 모니터", "tasks": "✓ 업무", "ask": "🔔 결정 대기", "triggers": "⏱ 자동 실행",
    "memory": "📚 기억", "graph": "◆ 조직도", "activity": "📜 활동", "commands": "⌘ 명령어", "workers": "🛠 팀원",
}
COMMANDS = [
    ("다들 뭐 해?", "팀원마다 지금 하는 일 → 다음 할 일을 한 줄씩 (모델 호출 없음, 바로 답함)"),
    ("/status", "'다들 뭐 해?'와 같음"),
    ("/help", "이 명령어 목록"),
    ("/clear", "터미널 기록 지우기"),
    ("그 밖의 한 줄", "팀장에게 업무로 맡김 → 팀장이 팀원들에게 나눠 주고 결과를 모아 올림"),
]
MAX_LOG = 80
CSS = """<style>
.ao-cc-head{display:flex;flex-wrap:wrap;align-items:center;gap:8px 10px;background:#fbf3dc;color:#2b2b2b;
  border:2px solid #2b2b2b;border-radius:6px;padding:6px 10px;box-shadow:3px 3px 0 #2b2b2b;margin-bottom:6px;
  font-family:Consolas,'D2Coding','Malgun Gothic',monospace}
.ao-cc-title{font-weight:800;letter-spacing:2px}
.ao-cc-pill{border:1.5px solid #2b2b2b;padding:0 6px;font-size:.78rem;border-radius:3px;background:#fff}
.ao-cc-sub{font-size:.8rem;color:#5b5240}
.ao-cc-term{background:#fbf3dc;color:#2b2b2b;border:2px solid #2b2b2b;border-radius:6px;padding:8px 10px;
  font-family:Consolas,'D2Coding','Malgun Gothic',monospace;font-size:.8rem;line-height:1.5;
  max-height:330px;overflow:auto;display:flex;flex-direction:column-reverse}
.ao-cc-term.ao-cc-fixed{height:330px}
.ao-cc-me{background:#2b2b2b;color:#fbf3dc;padding:1px 8px;margin:8px 0 4px}
.ao-cc-lead{margin:2px 0}
.ao-cc-line{padding-left:16px;white-space:pre-wrap}
.ao-cc-dim{color:#8a7f66}
.ao-cc-foot{border-top:1px dashed #8a7f66;margin-top:8px;padding-top:4px;color:#8a7f66;font-size:.74rem}
.ao-cc-now{background:#f08a24;color:#fff;padding:0 5px;border-radius:3px}
[class*="st-key-cc_tab_"] [role="radiogroup"]{gap:4px 6px}
[class*="st-key-cc_tab_"] [role="radiogroup"] label{border:1.5px solid #2b2b2b;border-radius:3px;padding:1px 8px 1px 6px;
  background:#fbf3dc;color:#2b2b2b;margin:0;font-family:Consolas,'D2Coding','Malgun Gothic',monospace}
[class*="st-key-cc_tab_"] [role="radiogroup"] label p{color:#2b2b2b;font-size:.8rem}
[class*="st-key-cc_tab_"] [role="radiogroup"] label > div:first-child{display:none}
[class*="st-key-cc_tab_"] [role="radiogroup"] label:has(input:checked){background:#f0b45a;box-shadow:2px 2px 0 #2b2b2b}
</style>"""


def _log_key(team_id: str) -> str:
    return f"cc_log_{team_id}"


def _pending_key(team_id: str) -> str:
    return f"cc_pending_{team_id}"


def _tab_key(team_id: str) -> str:
    return f"cc_tab_{team_id}"


def _quick_key(team_id: str) -> str:
    return f"cc_quick_{team_id}"


def _log(team_id: str, kind: str, text: str) -> None:
    log = st.session_state.setdefault(_log_key(team_id), [])
    log.append((kind, text))
    del log[:-MAX_LOG]


def pop_request(team_id: str, use_web: bool) -> dict:
    """터미널에서 업무로 맡긴 한 줄 → 워크플로우 요청 (자유 질문, 최종 상태는 AI 가 정리). 없으면 빈 dict."""
    text = st.session_state.pop(_pending_key(team_id), None)
    if not text:
        return {}
    return {"question": text, "brief": None, "auto_brief": True, "use_web": use_web, "tool_inputs": {},
            "extra_validators": [], "harness_options": {}}


def lead_state(run: dict | None) -> str:
    if not run:
        return "■ 쉬는 중"
    if not run["done"]:
        return "▶ 진행 중"
    if run.get("error"):
        return "⚠ 오류"
    return "🔔 결정 대기" if run["status"] == "awaiting_human" else "✓ 끝남"


def _box(lines: list[str], fixed: bool = False) -> str:
    """콘솔 상자 HTML — lines 는 이미 이스케이프한 HTML 조각. 아래(최신)가 먼저 보인다."""
    return f"<div class='ao-cc-term{' ao-cc-fixed' if fixed else ''}'><div>{''.join(lines)}</div></div>"


def _line(text: str, cls: str = "ao-cc-line") -> str:
    return f"<div class='{cls}'>{html.escape(text)}</div>"


def answer(ctx, run: dict | None, target: str, active_ids: set[str] | None, text: str) -> bool:
    """터미널에 들어온 한 줄을 처리한다. 업무로 맡겼으면 True (화면 전체를 다시 그려 시작해야 함)."""
    tid = ctx.team_id
    text = " ".join(text.split())
    if not text:
        return False
    lead = ctx.names[target]
    if text.lower() in ("/clear", "지우기"):
        st.session_state[_log_key(tid)] = []
        return False
    _log(tid, "me", text)
    if text.lower() in ("/help", "도움말", "명령어"):
        _log(tid, "lead", "쓸 수 있는 명령어:")
        for cmd, desc in COMMANDS:
            _log(tid, "line", f"{cmd} — {desc}")
        return False
    if office.is_status_question(text):
        started = time.perf_counter()
        head, *lines = office.rundown(run, office_ui_subtree(ctx, target), names=ctx.names, parents=ctx.parents,
                                      active_ids=active_ids, blocked=ctx.blocked)
        _log(tid, "lead", head)
        for line in lines:
            _log(tid, "line", line)
        _log(tid, "dim", f"✻ {time.perf_counter() - started:.2f}초 · 모델 호출 없이 지금 상태로 답함")
        return False
    if run and not run["done"]:
        _log(tid, "lead", "지금 진행 중인 업무가 있어요. 끝난 뒤에 다시 맡겨 주세요. (상태는 '다들 뭐 해?')")
        return False
    if run and run["status"] == "awaiting_human" and not run.get("error"):
        _log(tid, "lead", "결정을 기다리는 결과가 있어요. 먼저 아래에서 채택·반려해 주세요 "
                          "(새 업무를 맡기면 그 결과가 지워집니다 — 그래도 맡기려면 아래 '업무 맡기기'를 쓰세요).")
        return False
    st.session_state[_pending_key(tid)] = text
    _log(tid, "lead", f"알겠습니다 — {lead}이(가) 팀원들에게 나눠 맡기고 결과를 모아 보고할게요.")
    return True


def office_ui_subtree(ctx, target: str):
    """문의 대상(target)을 꼭대기로 하는 조직도 — 한 줄 보고에 넣을 팀원."""
    return next(n for n in ctx.org.walk() if n.id == target)


# ─── 탭 ───────────────────────────────────────────────────────────────
def _terminal(ctx, run, target, active_ids) -> None:
    tid = ctx.team_id
    log_area = st.container()  # 기록은 입력칸 위에 — 방금 보낸 줄까지 이번 실행에서 바로 보이도록 입력을 먼저 처리한다
    with st.form(f"cc_form_{tid}", clear_on_submit=True, border=False):
        c_in, c_go = st.columns([3, 1])
        text = c_in.text_input("팀장에게 한 줄", key=f"cc_text_{tid}", label_visibility="collapsed",
                               placeholder=f"{ctx.names[target]}에게 한 줄 (예: 다들 뭐 해?)")
        sent = c_go.form_submit_button("보내기", use_container_width=True)
    text = st.session_state.pop(_quick_key(tid), None) or (text if sent else "")
    if text and answer(ctx, run, target, active_ids, text):
        st.rerun()  # 업무 시작은 화면 전체에서 (app.py 의 '제출 → 백그라운드 실행')
    log = st.session_state.get(_log_key(tid), [])
    parts = []
    if not log:
        parts.append(_line(f"{ctx.names[target]}에게 한 줄 남기세요. '다들 뭐 해?' → 팀원마다 한 줄 보고, "
                           "그 밖의 말 → 업무로 맡김. (/help)", "ao-cc-dim"))
    cls = {"me": "ao-cc-me", "lead": "ao-cc-lead", "line": "ao-cc-line", "dim": "ao-cc-dim"}
    for kind, text in log:
        parts.append(_line(("› " if kind == "me" else "● " if kind == "lead" else "") + text, cls[kind]))
    tokens = sum((s.input_tokens or 0) + (s.output_tokens or 0) for s in run["steps"]) if run else 0
    parts.append(f"<div class='ao-cc-foot'>이번 업무 토큰 {tokens / 1000:,.1f}천 · 끝난 단계 "
                 f"{len(run['steps']) if run else 0}개 · /help 명령어</div>")
    log_area.markdown(_box(parts, fixed=True), unsafe_allow_html=True)


def _monitor(ctx, run, target, active_ids) -> None:
    if not run:
        st.markdown(_box([_line("진행 중인 업무가 없습니다.", "ao-cc-dim")]), unsafe_allow_html=True)
        return
    now = stage_of(run)
    stages = " › ".join(f"[{s}]" if i == now else ("✓" + s if i < now else s) for i, s in enumerate(STAGES))
    elapsed = int(time.time() - run["started_at"])
    parts = [_line(f"단계  {stages}", "ao-cc-lead"), _line(f"경과  {elapsed // 60}분 {elapsed % 60}초 · {lead_state(run)}", "ao-cc-dim")]
    for node in office_ui_subtree(ctx, target).walk():
        icon, now_text, nxt = office.agent_line(node.id, run, names=ctx.names, parents=ctx.parents,
                                                active_ids=active_ids, blocked=ctx.blocked)
        drafting = len(run["drafting"].get(node.id, ""))
        parts.append(_line(f"{icon} {ctx.names[node.id]:<10} {now_text}" + (f" · 초안 {drafting}자" if drafting else "")))
    st.markdown(_box(parts), unsafe_allow_html=True)


def _when(ts: float | str) -> str:
    if isinstance(ts, str):
        return ts[5:16].replace("T", " ")
    return datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")


def _tasks(ctx, run, target, active_ids) -> None:
    open_runs = [r for r in ctx.runs.values() if r["status"] not in ("adopted", "rejected")]
    parts = [_line(f"열린 업무 {len(open_runs)}개 (모든 팀)", "ao-cc-lead")]
    for r in sorted(open_runs, key=lambda r: r["started_at"], reverse=True):
        parts.append(_line(f"{lead_state(r)} · {r['team_id']} · {_when(r['started_at'])} · {r['question'][:40]}"))
    recent = ctx.storage.list_runs(ctx.team_id, limit=8)
    parts.append(_line(f"이 팀 최근 업무 {len(recent)}개", "ao-cc-lead"))
    mark = {"adopted": "✅", "rejected": "❌", "awaiting_human": "🔔", "reviewing": "⏳"}
    for r in recent:
        parts.append(_line(f"{mark.get(r['status'], '·')} {_when(r['created_at'])} · {r['question'][:44]}"))
    st.markdown(_box(parts), unsafe_allow_html=True)


def _ask(ctx, run, target, active_ids) -> None:
    waiting = [r for r in ctx.runs.values() if r["done"] and r["status"] == "awaiting_human" and not r.get("error")]
    if not waiting:
        st.markdown(_box([_line("사람 결정을 기다리는 업무가 없습니다.", "ao-cc-dim")]), unsafe_allow_html=True)
        return
    st.markdown(_box([_line(f"🔔 결정을 기다리는 업무 {len(waiting)}개 — 눌러서 결정하러 가기", "ao-cc-lead")]),
                unsafe_allow_html=True)
    for r in sorted(waiting, key=lambda r: r["started_at"], reverse=True):
        if st.button(f"🔔 {r['team_id']} · {r['question'][:40]}", key=f"cc_ask_{r['run_id']}", on_click=ctx.go_to_run,
                     args=(r,), use_container_width=True):
            st.rerun()  # 팀·문의 대상이 바뀌므로 화면 전체를 다시


def _triggers(ctx, run, target, active_ids) -> None:
    cfg = autonomy.load_config(ctx.settings)
    audit = autonomy.AuditLog(ctx.data_dir / cfg.audit_file)
    now = time.time()
    last = audit.last_runs()
    parts = [
        _line(f"자율 실행 {'켜짐' if cfg.enabled else '꺼짐 (settings.yaml autonomy.enabled)'} · 하루 예산 ${cfg.daily_budget_usd:g}"
              f" · 하루 최대 {cfg.max_runs_per_day}회 · 오늘 {audit.runs_since(autonomy.day_start(now))}회 실행", "ao-cc-lead"),
    ]
    for job in cfg.jobs:
        prev = last.get(job.id)
        nxt = "바로 (첫 실행)" if prev is None else _when(prev + job.every_hours * 3600)
        parts.append(_line(f"⏱ {job.id} · {job.team} · {job.every_hours:g}시간마다 · {job.level} · "
                           f"마지막 {_when(prev) if prev else '없음'} · 다음 {nxt}"))
    if not cfg.jobs:
        parts.append(_line("정해 둔 자동 작업이 없습니다 (settings.yaml autonomy.jobs). 읽기 전용 화면입니다.", "ao-cc-dim"))
    st.markdown(_box(parts), unsafe_allow_html=True)


def _memory(ctx, run, target, active_ids) -> None:
    lessons = [row for row in ctx.storage.list_lessons(ctx.team_id) if row.get("active", 1)]
    by_agent: dict[str, list[dict]] = {}
    for row in lessons:
        by_agent.setdefault(row["agent_id"], []).append(row)
    parts = [_line(f"교훈 {len(lessons)}개 — 사람의 채택·반려에서 배운 것 (다음 업무 프롬프트에 들어감)", "ao-cc-lead")]
    for node in ctx.org.walk():
        rows = by_agent.get(node.id, [])
        parts.append(_line(f"📚 {ctx.names[node.id]} · {len(rows)}개"))
        for row in rows[:2]:
            parts.append(_line(f"   {'👍' if row.get('kind') == 'praise' else '🔧'} {row['lesson'][:70]}", "ao-cc-dim"))
    st.markdown(_box(parts), unsafe_allow_html=True)


def _graph(ctx, run, target, active_ids) -> None:
    depth = {ctx.org.id: 0}
    parts = []
    for node in ctx.org.walk():
        for child in node.children:
            depth[child.id] = depth[node.id] + 1
        state = office.agent_state(node.id, run, active_ids, ctx.blocked)
        parts.append(_line("   " * depth[node.id] + ("└ " if depth[node.id] else "") +
                           f"{ctx.names[node.id]} ({ctx.ranks[node.id][1]})" + (f" — {state['bubble']}" if state["bubble"] else "")))
    st.markdown(_box(parts), unsafe_allow_html=True)
    st.button("🗂 그림 조직도로 보기", key=f"cc_graph_{ctx.team_id}", on_click=office_ui.show_chart)


def _activity(ctx, run, target, active_ids) -> None:
    parts = []
    if run:
        parts.append(_line(f"이번 업무 — {run['question'][:50]}", "ao-cc-lead"))
        for step in run["steps"][-15:]:
            parts.append(_line(office.step_line(step, ctx.names)))
        for nid, label in list(run["activity"].items()):
            parts.append(_line(f"▶ {ctx.names.get(nid, nid)} · {label}"))
    cfg = autonomy.load_config(ctx.settings)
    entries = autonomy.AuditLog(ctx.data_dir / cfg.audit_file).entries()[-8:]
    parts.append(_line(f"감사 로그 (자율 실행) 최근 {len(entries)}줄", "ao-cc-lead"))
    for rec in entries:
        parts.append(_line(f"{_when(rec['ts'])} · {rec['event']} · {rec.get('job', '')}", "ao-cc-dim"))
    st.markdown(_box(parts), unsafe_allow_html=True)


def _commands(ctx, run, target, active_ids) -> None:
    st.markdown(_box([_line(f"{cmd} — {desc}") for cmd, desc in COMMANDS]), unsafe_allow_html=True)

    def quick(text: str) -> None:
        st.session_state[_quick_key(ctx.team_id)] = text
        st.session_state[_tab_key(ctx.team_id)] = "terminal"

    st.button("▸ 다들 뭐 해?", key=f"cc_quick_btn_{ctx.team_id}", on_click=quick, args=("다들 뭐 해?",))


def _workers(ctx, run, target, active_ids) -> None:
    used = ctx.storage.usage_by_agent(ctx.team_id, since=time.time() - 7 * 86400)
    lessons = ctx.storage.lesson_counts(ctx.team_id)
    parts = [_line("팀원 · 지금 쓸 모델 · 최근 7일 사용량 (모델은 아래 '🎛 팀원 모델 선택'에서 바꿈)", "ao-cc-lead")]
    for node in ctx.org.walk():
        rows = used.get(node.id, [])
        calls = sum(r["calls"] or 0 for r in rows)
        tokens = sum(r["tokens"] or 0 for r in rows)
        cost = sum(r["cost_usd"] or 0 for r in rows)
        parts.append(_line(f"🛠 {ctx.names[node.id]} ({ctx.ranks[node.id][1]}) · {ctx.models.get(node.id, '?')}"
                           + (f" · ⛔ 꺼짐: {ctx.blocked[node.id]}" if node.id in ctx.blocked else "")))
        parts.append(_line(f"   {calls}회 · {tokens / 1000:,.1f}천 토큰 · ${cost:.3f} · 📚 {lessons.get(node.id, 0)}", "ao-cc-dim"))
    st.markdown(_box(parts), unsafe_allow_html=True)


RENDER = {
    "terminal": _terminal, "monitor": _monitor, "tasks": _tasks, "ask": _ask, "triggers": _triggers,
    "memory": _memory, "graph": _graph, "activity": _activity, "commands": _commands, "workers": _workers,
}


def render(ctx, run: dict | None, target: str, active_ids: set[str] | None) -> None:
    """커맨드 센터 전체 — 머리줄 + 탭 고르기 + 고른 탭 하나."""
    lead = ctx.names[target]
    st.markdown(
        CSS + "<div class='ao-cc-head'><span class='ao-cc-title'>🖥 COMMAND CENTER</span>"
        f"<span class='ao-cc-pill'>{html.escape(lead_state(run))}</span>"
        f"<span class='ao-cc-sub'>{html.escape(lead)}이(가) 사무실을 지휘 중 · {html.escape(ctx.team_name)}</span></div>",
        unsafe_allow_html=True,
    )
    key = _tab_key(ctx.team_id)
    st.session_state.setdefault(key, "terminal")
    # (segmented_control 은 AppTest 1.43 에서 다룰 수 없어 가로 radio 를 탭 버튼처럼 꾸몄다 — CSS 의 st-key-cc_tab_)
    tab = st.radio("커맨드 센터 탭", list(TABS), format_func=TABS.get, key=key, horizontal=True,
                   label_visibility="collapsed")
    RENDER[tab](ctx, run, target, active_ids)
