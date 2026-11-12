"""진행 확인·개입 화면 — ⏱ 호출 모니터, ⏭ 다른 모델로, ⛔ 건너뛰기·⏹ 중단, ✏️ 고쳐 다시, 🔀 단계마다 확인.

app.py 는 이 파일의 함수만 부른다 (화면 조각). 기록·끊기 로직은 core/call_monitor.py, 모델 규칙은 core/switch_rules.py,
단계 확인은 core/step_mode.py, 체크포인트에서 다시 하기는 core/workflow.restart_from.

  render_summary(ctx, run)    : 진행 중 한 줄 요약 (+ ⚠️ 응답 없음 경고)        — 1초마다 다시 그리는 영역(st.fragment)
  render_monitor(ctx, run)    : 진행 중·조작 / 시간표 / 전체 호출 / 요약 / 개입 기록 탭 — 같은 영역
  render_step_check(ctx, run) : 🔀 멈춘 곳의 선택 화면 → 사람의 답(dict) 또는 None — 바깥 영역 (입력 중 다시 그리지 않음)
  render_restart(ctx, run)    : ✏️ 질문·브리프 고쳐 다시 → 요청(dict) 또는 None    — 바깥 영역
  render_attempts(ctx, run)   : 🆚 예전 시도와 지금 비교
  render_history(data, names) : 🕘 지난 업무 창의 ⏱ 호출 기록
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import streamlit as st

from core import call_monitor, followup, llm_factory, progress, step_mode, switch_rules
from core.brief import DesiredState, empty_state
from core.schemas import ModelsConfig, OrgNode, Team
from core.workflow import RESTART_POINTS


@dataclass
class Ctx:
    team_id: str
    team: Team
    org: OrgNode
    base_models: ModelsConfig  # 원래 설정 (팀원 규칙 판단용)
    names: dict[str, str]
    label: Callable[[str], str] = str  # 모델 키 → 화면 이름 ('모델 · 제공처 · 비용')
    stall_sec: float = call_monitor.STALL_SEC
    member_choices: dict[str, str] = field(default_factory=dict)
    progress: Callable[[dict], "progress.Estimate | None"] | None = None  # 📊 진행률·남은 시간 (app.run_progress)


def step_key(team_id: str) -> str:
    return f"step_mode_{team_id}"


def step_mode_toggle(team_id: str) -> bool:
    return st.toggle(
        "🔀 단계마다 확인", key=step_key(team_id),
        help="켜면 ① 브리프 뒤(팀장이 내놓은 방향 A·B·C 중 고르기) ② 팀장 업무 분배 뒤 ③ 초안 뒤(통합 전)에 멈추고 "
             "내 선택을 기다립니다. 끄면(기본) 지금처럼 끝까지 한 번에 진행합니다.",
    )


def run_models(ctx: Ctx, run: dict) -> ModelsConfig:
    """이번 실행이 쓰는 설정 — 맡길 때 '🆓 무료 모델만'이 켜져 있었으면 유료 provider 를 끈 설정."""
    return llm_factory.free_only(ctx.base_models) if run.get("free_only") else ctx.base_models


def _fmt(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 60}분 {sec % 60:02d}초" if sec >= 60 else f"{sec}초"


# ─── 1초마다 다시 그리는 영역 ─────────────────────────────────────────
def progress_text(est: "progress.Estimate") -> str:
    more = est.worst - est.total
    steps = f"남은 단계 {est.remaining}개" + (f" (재시도·비평 반려 시 최대 {est.remaining + more}개)" if more > 0 else "")
    what = "💬 추가 질문 " if est.phase == "followup" else ""
    return f"📊 {what}{est.percent}% · {steps} · {progress.eta_text(est.remaining_sec)} — 추정: {est.basis}"


def render_progress(est: "progress.Estimate") -> None:
    """진행 막대 하나 — 완료 % · 남은 단계 · 예상 남은 시간 (추정이라고 밝힘)."""
    st.progress(est.percent / 100, text=progress_text(est))


def _estimate(ctx: Ctx, run: dict):
    if ctx.progress is None:
        return None
    try:
        return ctx.progress(run)
    except Exception:  # noqa: BLE001 — 추정 실패가 모니터를 막지 않게
        return None


def render_summary(ctx: Ctx, run: dict) -> None:
    recs = call_monitor.calls(run["run_id"])
    if not recs:
        return
    now = time.time()
    st.caption("⏱ " + call_monitor.headline(recs, now, ctx.stall_sec))
    stalled = [r for r in recs if r.status in call_monitor.ACTIVE
               and any(f.startswith("⚠️") for f in call_monitor.flags(r, now, ctx.stall_sec))]
    if stalled:
        lines = [f"- **{r.who}** · {call_monitor.KIND_LABELS.get(r.kind, r.kind)} · `{r.provider}:{r.model}` — "
                 f"{_fmt(now - (r.last_activity or r.started_at or now))}째 응답 없음" for r in stalled[:5]]
        st.warning("⚠️ 응답이 없는 모델 호출이 있습니다. 아래 **⏱ 호출 모니터**에서 다른 모델로 바꾸거나 건너뛸 수 있습니다.\n"
                   + "\n".join(lines))


def _rows_table(records, now, stall_sec, height=None):
    rows = call_monitor.rows(records, now=now, stall_sec=stall_sec)
    for row in rows:
        row.pop("id", None)
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True, height=height)


def render_monitor(ctx: Ctx, run: dict) -> None:
    """⏱ 호출 모니터 — 켜 둔 동안 1초마다 경과 시간이 늘어난다."""
    if not st.toggle("⏱ 호출 모니터 — 어느 모델 호출이 느린지 보고, 끊거나 다른 모델로 바꾸기", key=f"mon_open_{ctx.team_id}"):
        return
    rid = run["run_id"]
    recs = call_monitor.calls(rid)
    now = time.time()
    t_live, t_gantt, t_all, t_sum, t_log = st.tabs(["▶ 진행 중·조작", "📊 시간표", "📋 전체 호출", "🧮 요약", "🧑 개입 기록"])
    with t_live:
        _controls(ctx, run, recs, now)
    with t_gantt:
        est = _estimate(ctx, run)
        if est is not None:
            render_progress(est)
            left = progress.remaining_rows(est, ctx.names)
            if left:
                with st.expander(f"⏳ 남은 단계 {len(left)}개 · 단계마다 예상 시간 (추정)"):
                    st.dataframe(left, use_container_width=True, hide_index=True)
                    st.caption("예상 시간 = 같은 팀 지난 실행의 같은 단계·모델 걸린 시간 중앙값 (없으면 같은 종류 → 기본값). "
                               "형제 팀원은 동시에 일하므로 남은 시간은 단계 시간의 합이 아니라 가장 긴 줄기로 계산합니다.")
        elif run["done"]:
            st.caption("📊 이 실행은 끝났거나 멈춰 있습니다 — 아래는 지금까지의 호출입니다.")
        st.caption("가로 막대 = 호출 하나. 빗금 = 동시 호출 자리·재시도 대기, 색 = 모델이 답하는 시간 "
                   "(파랑 응답 기다림 · 하늘 받는 중 · 초록 끝 · 빨강 실패/응답 없음 · 보라 사람이 끊음). 막대에 마우스를 올리면 자세히.")
        st.markdown(call_monitor.gantt_html(recs, now, stall_sec=ctx.stall_sec), unsafe_allow_html=True)
    with t_all:
        _rows_table(recs, now, ctx.stall_sec, height=320)
        if not recs:
            st.caption("아직 모델 호출이 없습니다.")
    with t_sum:
        _summary(recs, now)
    with t_log:
        _interventions(call_monitor.interventions(rid))


def _controls(ctx: Ctx, run: dict, recs, now) -> None:
    rid = run["run_id"]
    live = [r for r in recs if r.status in call_monitor.ACTIVE]
    if run["done"]:
        st.caption("진행 중인 실행이 없습니다 (끝났거나 멈춰 있음). 기록은 다른 탭에서 볼 수 있습니다.")
    elif not live:
        st.caption("지금 진행 중인 모델 호출이 없습니다 (데이터 조회·검증 계산 중이거나 다음 호출을 준비 중).")
    if live:
        _switch_form(ctx, run, live, now)
    if run["done"]:
        return
    ctl = call_monitor.control(rid)
    if run.get("status") == "followup_running":  # 💬 추가 질문 — 👔 부장 검토 건너뛰기 · 중단
        c1, c2 = st.columns(2)
        if c1.button("⛔ 부장 검토 건너뛰기" if "critic" not in ctl.skips else "⛔ 부장 검토 건너뜀", key=f"skip_head_{rid}",
                     disabled="critic" in ctl.skips, use_container_width=True,
                     help="수정본에 대한 부장 검토와 그에 따른 재수정을 건너뜁니다 (진행 중인 검토 호출도 끊음). 수정본은 그대로 나옵니다."):
            n = call_monitor.skip(rid, "critic")
            st.toast(f"⛔ 부장 검토를 건너뜁니다 (끊은 호출 {n}개)")
        if c2.button("⏹ 추가 질문 중단", key=f"stop_followup_{rid}", disabled=ctl.stopped, use_container_width=True,
                     help="진행 중인 호출을 끊습니다. 팀장이 답하는 중이면 결과는 그대로 두고 결정 대기로 돌아가고, "
                          "부장이 검토하는 중이면 이미 쓴 수정본을 검토 없이 냅니다."):
            n = call_monitor.stop(rid, "💬 추가 질문 중단")
            st.toast(f"⏹ 추가 질문을 중단합니다 (끊은 호출 {n}개)")
        return
    c1, c2, c3 = st.columns(3)
    if c1.button("⛔ 비평 건너뛰기" if "critic" not in ctl.skips else "⛔ 비평 건너뜀", key=f"skip_critic_{rid}",
                 disabled="critic" in ctl.skips, use_container_width=True,
                 help="이번 실행의 비평가 채점·자가 교정을 건너뜁니다 (진행 중인 비평 호출도 끊음)."):
        n = call_monitor.skip(rid, "critic")
        st.toast(f"⛔ 비평을 건너뜁니다 (끊은 호출 {n}개)")
    if c2.button("⛔ 하네스 재시도 건너뛰기" if "harness" not in ctl.skips else "⛔ 하네스 건너뜀", key=f"skip_harness_{rid}",
                 disabled="harness" in ctl.skips, use_container_width=True,
                 help="하네스(결정론적 검증) 실패 시의 재작성 반복을 그만두고 지금 답변으로 진행합니다."):
        n = call_monitor.skip(rid, "harness")
        st.toast(f"⛔ 하네스 재시도를 건너뜁니다 (끊은 호출 {n}개)")
    if c3.button("⏹ 지금까지 결과로 마무리", key=f"stop_{rid}", disabled=ctl.stopped, use_container_width=True,
                 help="새 호출을 막고 진행 중인 호출을 끊습니다. 끝난 결과는 모아 '부분 결과'로 결정 화면에 올립니다."):
        n = call_monitor.stop(rid)
        st.toast(f"⏹ 중단합니다 (끊은 호출 {n}개) — 끝난 결과로 마무리")


SWITCH_PICK = "sw_pick_{}"


def _pick(rid: str, call_id: str) -> None:
    st.session_state[SWITCH_PICK.format(rid)] = call_id


def _switch_form(ctx: Ctx, run: dict, live, now: float) -> None:
    """진행 중인 호출마다 한 줄 + '⏭ 바꾸기' 버튼 → 고른 호출의 모델 고르기·끊고 다시.

    호출 고르기는 드롭다운 대신 줄마다 버튼으로 한다 — 1초마다 다시 그려도 눌린 것이 풀리지 않는다.
    고른 호출이 끝나면 다른 호출로 저절로 옮기지 않는다 (누르려던 버튼이 엉뚱한 호출을 끊지 않게).
    """
    rid = run["run_id"]
    by_id = {r.id: r for r in live}
    picked = st.session_state.get(SWITCH_PICK.format(rid))
    for r in sorted(live, key=lambda r: r.queued_at):
        flags = call_monitor.flags(r, now, ctx.stall_sec)
        c_text, c_btn = st.columns([5, 1])
        c_text.markdown(
            f"{'👉 **' if r.id == picked else ''}{call_monitor.STATUS_LABELS.get(r.status, r.status)} · **{r.who}** · "
            f"{call_monitor.KIND_LABELS.get(r.kind, r.kind)} · `{r.provider}:{r.model}` · {_fmt(call_monitor.elapsed(r, now))}"
            + ("**" if r.id == picked else "") + "".join(f" · {f}" for f in flags)
        )
        c_btn.button("⏭ 바꾸기", key=f"sw_pick_{r.id}", on_click=_pick, args=(rid, r.id), use_container_width=True,
                     type="primary" if r.id == picked else "secondary")
    if picked is None:
        st.caption("느린 호출 줄의 **⏭ 바꾸기**를 누르면 다른 모델로 다시 하게 할 수 있습니다.")
        return
    if picked not in by_id:
        st.caption(f"고른 호출({picked})은 끝났습니다. 바꿀 호출을 다시 고르세요.")
        return
    rec = by_id[picked]
    st.markdown(f"**⏭ {rec.who}** · {call_monitor.KIND_LABELS.get(rec.kind, rec.kind)} · `{rec.provider}:{rec.model}` 를 다른 모델로")
    options = switch_rules.options(rec, ctx.org, ctx.team, ctx.base_models, run_models(ctx, run),
                                   run.get("member_choices", ctx.member_choices))
    allowed = [o.key for o in options if o.allowed]
    c_model, c_rest = st.columns([3, 2])
    if not allowed:
        c_model.warning("바꿀 수 있는 모델이 없습니다 — 아래 이유를 보세요.")
        choice = None
    else:
        choice = c_model.selectbox("다른 모델", allowed, key=f"sw_model_{rec.id}", format_func=ctx.label)
    rest = c_rest.checkbox(f"남은 업무에도 이 모델 ({rec.who})", key=f"sw_rest_{rec.id}",
                           help="이번 실행에서 같은 담당의 다음 호출도 이 모델을 먼저 씁니다 (다음 업무에는 영향 없음).")
    if c_rest.button("⏭ 이 호출 끊고 이 모델로 다시", key=f"sw_go_{rec.id}", type="primary", disabled=choice is None):
        node = next((n for n in ctx.org.walk() if n.id == rec.agent_id), None)
        spec = switch_rules.spec_for(choice, ctx.base_models, ctx.team, node.rank if node else "")
        if call_monitor.switch(rec.id, spec, for_rest=rest):
            st.toast(f"⏭ {rec.who}: {rec.provider}:{rec.model} → {choice} 로 다시 합니다")
        else:
            st.toast("그 호출은 방금 끝났습니다" + (" — 남은 업무부터 이 모델을 씁니다" if rest else ""))
    blocked = [o for o in options if not o.allowed]
    if blocked:
        with st.expander(f"🚫 고를 수 없는 모델 {len(blocked)}개와 이유"):
            st.dataframe([{"모델": ctx.label(o.key), "이유": o.reason} for o in blocked], use_container_width=True, hide_index=True)
    if rec.provider == "claude_cli":
        st.caption("claude_cli(구독)는 끊으면 claude 프로세스를 바로 종료합니다.")
    else:
        st.caption("끊긴 호출의 답은 버립니다. 스트리밍 초안은 바로 멈추지만, 한 번에 받는 호출(검토·비평 등)은 "
                   "서버에 보낸 요청이 끝날 때까지 뒤에서 기다렸다 버리므로 그 토큰은 쓰일 수 있습니다.")


def _summary(recs, now) -> None:
    if not recs:
        st.caption("아직 모델 호출이 없습니다.")
        return
    s = call_monitor.summary(recs, now)
    c = [*st.columns(2), *st.columns(2)]  # 좁은 화면에서도 숫자가 잘리지 않게 2×2
    c[0].metric("모델 호출", f"{s['calls']}번")
    c[1].metric("걸린 시간(처음~끝)", _fmt(s["wall_sec"]))
    c[2].metric("모델 시간 합", _fmt(s["model_sec"]))
    c[3].metric("기다린 시간 합", _fmt(s["wait_sec"]), help="동시 호출 자리 대기 + 재시도 전 대기")
    st.caption("동시에 여러 호출이 진행되므로 '모델 시간 합'은 걸린 시간보다 길 수 있습니다.")
    l, r = st.columns(2)
    l.markdown("**단계별**")
    l.dataframe([{"단계": k, **v} for k, v in s["by_kind"].items()], use_container_width=True, hide_index=True)
    r.markdown("**모델별**")
    r.dataframe([{"모델": k, **v} for k, v in s["by_model"].items()], use_container_width=True, hide_index=True)
    st.markdown("**가장 느린 호출**")
    st.dataframe(s["slowest"], use_container_width=True, hide_index=True)


def _interventions(entries: list[dict]) -> None:
    if not entries:
        st.caption("이번 실행에서 사람이 개입한 기록이 없습니다.")
        return
    for e in entries:
        when = time.strftime("%H:%M:%S", time.localtime(e.get("ts", 0)))
        st.markdown(f"- `{when}` 👤 **{e.get('who', '사용자')}** · {e.get('what', '')} — {e.get('detail', '')}")


# ─── 바깥 영역 (입력 중에 다시 그리지 않음) ──────────────────────────
def _brief_fields(prefix: str, brief: DesiredState | None) -> DesiredState:
    brief = brief or empty_state()
    goal = st.text_input("🎯 최종 목표", value=brief.goal, key=f"{prefix}_goal")
    criteria = st.text_area("✅ 완료 기준 (한 줄에 하나)", value="\n".join(brief.success_criteria), key=f"{prefix}_crit", height=90)
    constraints = st.text_area("🚧 제약 (한 줄에 하나)", value="\n".join(brief.constraints), key=f"{prefix}_cons", height=70)
    fmt = st.text_input("📄 결과 형식", value=brief.output_format, key=f"{prefix}_fmt")

    def lines(text: str) -> list[str]:
        return [x.strip() for x in text.splitlines() if x.strip()]

    return brief.model_copy(update={"goal": goal.strip(), "success_criteria": lines(criteria), "constraints": lines(constraints),
                                    "output_format": fmt.strip()})


NO_DIRECTION, OWN_DIRECTION = "__none__", "__own__"


def render_step_check(ctx: Ctx, run: dict) -> dict | None:
    """🔀 단계마다 확인 — 멈춘 곳의 선택 화면. '▶ 이대로 진행'을 누르면 사람의 답(dict)."""
    cp = run.get("checkpoint") or {}
    point = cp.get("point")
    rid = run["run_id"]
    st.markdown(f"#### 🔀 단계 확인 — {step_mode.POINTS.get(point, point)}")
    st.caption("단계마다 확인을 켠 업무입니다. 확인하고 고친 뒤 '▶ 이대로 진행'을 누르면 다음 단계로 갑니다. "
               "고칠 것이 없으면 그대로 누르면 됩니다.")
    with st.form(f"step_check_{rid}_{point}_{len(call_monitor.interventions(rid))}"):
        answer: dict = {}
        if point == "brief":
            opts = cp.get("options") or []
            choices = [o["title"] for o in opts] + [OWN_DIRECTION, NO_DIRECTION]
            labels = {o["title"]: f"{o['title']} — {o['summary']}" for o in opts}
            labels.update({OWN_DIRECTION: "✍️ 직접 쓰기", NO_DIRECTION: "방향 지정 안 함"})
            st.markdown("**팀장이 제안한 진행 방향** — 하나를 고르거나 직접 쓰세요")
            pick = st.radio("진행 방향", choices, format_func=labels.get, label_visibility="collapsed")
            own = st.text_input("직접 쓰는 방향 (✍️ 직접 쓰기를 골랐을 때)", key=f"own_dir_{rid}")
            st.markdown("**브리프** (모든 팀원이 받는 업무 지시)")
            brief = _brief_fields(f"sc_{rid}", DesiredState.model_validate(cp["brief"]) if cp.get("brief") else None)
            go = st.form_submit_button("▶ 이대로 진행", type="primary")
            if go:
                direction = own.strip() if pick == OWN_DIRECTION else ("" if pick == NO_DIRECTION else labels.get(pick, pick))
                answer = {"brief": brief.model_dump(), "direction": direction}
        elif point == "plan":
            if cp.get("analysis"):
                st.caption(f"팀장의 분석: {cp['analysis']}")
            given = {a["child_id"]: a["subtask"] for a in cp.get("assignments") or []}
            rows = []
            for cid in cp.get("children") or list(given):
                use = st.checkbox(f"**{ctx.names.get(cid, cid)}** 에게 맡기기", value=cid in given, key=f"sc_use_{rid}_{cid}")
                text = st.text_area("업무 지시", value=given.get(cid, ""), key=f"sc_task_{rid}_{cid}", height=80,
                                    label_visibility="collapsed", placeholder="이 팀원에게 줄 업무 (맡기려면 위를 체크)")
                rows.append((cid, use, text))
            go = st.form_submit_button("▶ 이대로 진행", type="primary")
            if go:
                answer = {"assignments": [{"child_id": c, "subtask": t} for c, use, t in rows if use and t.strip()]}
        else:
            edits = {}
            for d in cp.get("drafts") or []:
                edits[d["node_id"]] = st.text_area(f"📝 {ctx.names.get(d['node_id'], d['node_id'])} 초안", value=d["content"],
                                                   key=f"sc_draft_{rid}_{d['node_id']}", height=200)
            note = st.text_input("통합할 때 참고할 지시 (선택)", key=f"sc_note_{rid}", placeholder="예: 성과 숫자 위주로, 표로 정리")
            go = st.form_submit_button("▶ 이대로 진행 (통합 시작)", type="primary")
            if go:
                answer = {"edits": edits, "note": note}
    return answer if go else None


def render_restart(ctx: Ctx, run: dict) -> dict | None:
    """✏️ 질문·브리프 고쳐 다시 — 체크포인트에서 다시 시작할 요청. 지금 시도는 '🆚 이전 시도'로 남는다."""
    rid = run["run_id"]
    with st.expander("✏️ 질문·브리프 고쳐 다시 (체크포인트에서 다시 시작)"):
        st.caption("지금 시도는 지우지 않고 '🆚 이전 시도'로 남겨 비교할 수 있습니다. 진행 중이면 먼저 멈춘 뒤 다시 시작합니다.")
        if run.get("followups"):
            st.caption("💬 지금까지의 수정 대화와 앞선 수정본은 지금 결과를 고친 것이라, 다시 하면 새 결과의 '채택할 안'에서 빠집니다. "
                       "마지막 수정본은 '🆚 이전 시도'에서 다시 볼 수 있습니다.")
        has_brief = run.get("brief") is not None
        points = ["plan", "brief"] if has_brief else ["brief"]
        with st.form(f"restart_{rid}_{len(run.get('attempts') or [])}"):
            point = st.radio("어디서부터 다시", points, format_func=RESTART_POINTS.get, horizontal=True)
            question = st.text_area("질문", value=run["question"], key=f"rs_q_{rid}", height=100)
            redraft = st.checkbox("처음부터 다시일 때: 브리프는 AI 가 새 질문으로 다시 정리 (끄면 아래 브리프를 그대로 씀)", value=True)
            brief = _brief_fields(f"rs_{rid}", run.get("brief"))
            step = st.checkbox("🔀 이번에는 단계마다 확인", value=bool(run.get("step_mode")))
            go = st.form_submit_button("✏️ 이대로 다시 하기", type="primary")
        if not go:
            return None
        if point == "plan" and not brief.goal:
            st.warning("업무 분배부터 다시 하려면 최종 목표가 필요합니다.")
            return None
        return {
            "point": point, "question": question.strip() or run["question"], "step_mode": step,
            "brief": brief if point == "plan" or not redraft else empty_state(),
        }


def render_attempts(ctx: Ctx, run: dict, final_text: str | None) -> None:
    attempts = run.get("attempts") or []
    if not attempts:
        return
    with st.expander(f"🆚 이전 시도 {len(attempts)}개와 비교"):
        idx = st.selectbox("비교할 이전 시도", list(range(len(attempts)))[::-1],
                           format_func=lambda i: f"{attempts[i]['round']}번째 시도 — {attempts[i]['why']}", key=f"att_pick_{run['run_id']}")
        old = attempts[idx]
        revised = next((t for t in reversed(old.get("followups") or []) if t.revised), None)  # 그 시도에서 받은 마지막 수정본
        left, right = st.columns(2)
        for col, title, q, brief, text in (
            (left, f"🕘 {old['round']}번째 시도" + (f" · ✏️ 수정본 v{revised.version}" if revised else ""), old["question"],
             old.get("brief"), revised.content if revised else old["steps"][-1].content if old["steps"] else "(끝난 단계 없음)"),
            (right, f"▶ 지금 ({len(attempts) + 1}번째)", run["question"], run.get("brief"), final_text or "(아직 진행 중)"),
        ):
            with col:
                st.markdown(f"**{title}**")
                st.caption(f"Q. {q[:200]}")
                if brief is not None:
                    st.caption(f"🎯 {brief.goal[:200]}")
                with st.container(height=320, border=True):
                    st.markdown(text)


def render_history(data: dict | None, names: dict[str, str]) -> None:
    """🕘 지난 업무 창 — 저장된 ⏱ 호출 기록·개입 기록·이전 시도 수."""
    if not data:
        return
    recs, entries = call_monitor.load(data)
    if not recs and not entries:
        return
    with st.expander(f"⏱ 호출 기록 ({len(recs)}번) · 개입 {len(entries)}번"):
        end = max((r.ended_at or r.queued_at) for r in recs) if recs else time.time()
        st.caption(call_monitor.headline(recs, end))
        _summary(recs, end)
        st.markdown(call_monitor.gantt_html(recs, end), unsafe_allow_html=True)
        _interventions(entries)
        if data.get("attempts"):
            st.caption(f"✏️ 고쳐 다시 한 횟수: {len(data['attempts'])}번")


def serialize_attempts(run: dict) -> list[dict]:
    out = []
    for a in run.get("attempts") or []:
        brief = a.get("brief")
        out.append({"round": a["round"], "question": a["question"], "why": a["why"], "ended_at": a["ended_at"],
                    "brief": brief.model_dump() if brief is not None else None, "steps": [s.model_dump() for s in a["steps"]],
                    # 그 시도의 💬 수정 대화 — 고쳐 다시 하면 run_followups 에서는 비우므로 디스크에는 여기에만 남는다
                    "followups": json.loads(followup.dump_turns(a.get("followups") or []))})
    return out
