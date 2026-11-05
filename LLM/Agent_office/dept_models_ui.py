"""🎛 부서별 모델 선택 — 부서 → 👔 부장(⚖️ 비평가 · 🧭 부서장) 모델 → 그 부서의 팀 → 팀원 모델.

어디서 여나
  - 🏠 메인: 빌딩 아래 '🎛 부서별 모델 선택' (빌딩의 부서 띠를 누르면 그 부서로 펼쳐진다)
  - 🏢 팀 화면: '🎛 부서별 모델 선택 — <부서>' (지금 팀이 속한 부서, 팀원 표는 지금 팀부터)
로직은 core/dept_heads.py(부장·비평가) · core/member_models.py(팀원 상하 관계 규칙), 저장은 데이터 폴더의
department_models.json · member_models.json (config/*.yaml 은 안 고침). 바꾼 선택은 다음에 맡기는 업무부터.

화면을 가볍게: 부장 칸과 팀 요약 한 줄씩은 늘 그리고, 팀원 표는 고른 팀 하나만 그린다 (PC 가 자주 CPU 100%).
"""

import time
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

import models_ui
from core import building, dept_heads, llm_factory, member_models, model_caps, model_catalog
from core.config import team_org
from core.schemas import ModelsConfig, OrgNode, Rank, Team

USAGE_DAYS = 7
NOTE_KEY = "member_model_note"  # {팀 id: 알림} — 고른 곳에만 보인다
HEAD_NOTE_KEY = "dept_head_note"


@dataclass
class Ctx:
    teams: list[Team]
    ranks: list[Rank]
    models: ModelsConfig  # 규칙 판단용 ('무료만' 토글과 상관없는 원래 설정 + 찾은·직접 입력 모델)
    merged: model_catalog.Merged
    member_store: member_models.MemberModelStore
    dept_store: dept_heads.DeptModelStore
    storage: object
    status_icons: dict
    data_dir: Path
    runtime_models: ModelsConfig | None = None  # 지금 실제로 쓰는 설정 ('무료만'이면 유료가 꺼진 사본) — '꺼짐' 표시용
    running: bool = False

    @property
    def runtime(self) -> ModelsConfig:
        return self.runtime_models or self.models


def departments(ctx: Ctx) -> list[building.Department]:
    return building.departments([t.id for t in ctx.teams], building.load_config())


def _team(ctx: Ctx, team_id: str) -> Team:
    return next(t for t in ctx.teams if t.id == team_id)


def _legacy(team: Team) -> str | None:
    return team.legacy.team if team.legacy else None


def _label(ctx: Ctx, key: str, states: dict) -> str:
    return models_ui.option_label(key, states, ctx.status_icons, ctx.merged)


def _short(ctx: Ctx, key: str) -> str:
    return models_ui.model_name(key, ctx.merged)


def team_chains(ctx: Ctx, team: Team, org: OrgNode | None = None):
    org = org or team_org(team, ctx.ranks)
    choices = ctx.member_store.load(team.id, _legacy(team))
    return org, choices, member_models.team_rank_models(org, ctx.models, team, choices)


# ─── 👔 부장 ──────────────────────────────────────────────────────────


def _on_head_change(ctx: Ctx, dept_id: str, widget: str) -> None:
    key = st.session_state.get(widget) or ""
    ctx.dept_store.set(dept_id, key)
    st.session_state[HEAD_NOTE_KEY] = (dept_id, f"👔 부장 모델을 {'기본으로 되돌렸' if not key else _short(ctx, key) + ' 로 바꿨'}습니다 — "
                                                "다음에 맡기는 업무의 비평가·🧭 팀 배정부터 씁니다.")


def render_head(ctx: Ctx, dept: building.Department, where: str) -> None:
    head = dept_heads.head_of(dept)
    states = llm_factory.provider_status(ctx.models)
    choice = dept_heads.valid_choice(ctx.dept_store.get(dept.id), ctx.models)
    base, source = dept_heads.head_chain(ctx.models)  # 고르지 않았을 때 실제로 쓰는 모델
    base_key = member_models.model_key(base.primary)
    default = f"기본 = {_label(ctx, base_key, states)} ({dept_heads.SOURCE_LABELS[source]})"
    widget = f"dept_head_model_{where}_{dept.id}"
    st.session_state[widget] = choice  # 저장값이 기준 (다른 화면에서 바꿔도 맞게)
    c1, c2 = st.columns([3, 5])
    c1.markdown(f"**{head.label}**  \n⚖️ 비평가 · 🧭 부서장")
    # 비평·팀 고르기를 못 하는 모델·제공처에서 사라진 모델은 숨긴다 (이유는 아래 'ⓘ 숨긴 모델', core/model_caps.py)
    shown, hidden = model_caps.selectable_models(model_caps.TEAM_WORK, member_models.catalog(ctx.models))
    options = ["", *shown]
    if choice and choice not in options:  # 저장값이 사라졌거나 못 하는 모델 — 지우지 않고 그대로 보여 주고 경고
        options.append(choice)
    c2.selectbox(f"{head.name} 모델", options, key=widget, label_visibility="collapsed",
                 format_func=lambda k: default if not k else _label(ctx, k, states),
                 on_change=_on_head_change, args=(ctx, dept.id, widget))
    if choice and choice not in shown:
        c2.markdown(f":orange[⚠️ **고른 모델 {member_models.split_key(choice)[1]}**] — 지금 쓸 수 없는 모델입니다 (사라짐·능력 부족). "
                    "비평은 기본 순서의 다음 모델로 합니다.")
    models_ui.hidden_note(hidden, ctx.merged)
    for w in head_lead_warnings(ctx, dept, choice or base_key):
        c2.markdown(f":orange[⚠️ {w}]")
    st.caption(f"관점: {head.persona}  \n부장은 팀 조직도 밖에서 이 부서 **모든 팀의 최종 답변을 비평**하고, 🧭 부서장으로 요청을 알맞은 팀에 보냅니다. "
               "부장은 맨 위라 **기본은 팀장과 같거나 더 강한 모델**입니다. 더 약한 모델도 고를 수 있지만 ⚠️ 로 알립니다. "
               "비공개 팀은 부장 모델을 고르지 않으면 팀장 모델로 비평합니다 (문서가 팀 밖 제공처로 가지 않게).")
    if member_models.split_key(choice or base_key)[0] == "claude_cli":
        st.caption("💳 Claude 구독(claude_cli): 청구는 없지만 **구독 사용 한도**를 씁니다 (내 Claude 대화·Claude Code 와 공유, 동시 1개). "
                   "업무마다 비평 1번 + 불합격이면 자가 교정·다시 비평이 더해지고, 팀이 둘 이상인 부서는 🧭 팀 배정마다 1번 더 부릅니다. "
                   "한도가 걱정되면 gpt-5.4 같은 API 모델을 고르세요.")
    note = st.session_state.get(HEAD_NOTE_KEY)
    if note and note[0] == dept.id:
        st.session_state.pop(HEAD_NOTE_KEY, None)
        st.info(note[1])


def head_lead_warnings(ctx: Ctx, dept: building.Department, head_key: str) -> list[str]:
    """부장(맨 위)이 이 부서 어느 팀장보다 약하면 팀마다 한 줄 (비공개 팀은 고르지 않았으면 팀장 모델로 비평하므로 뺀다)."""
    out = []
    chosen = bool(dept_heads.valid_choice(ctx.dept_store.get(dept.id), ctx.models))
    for tid in dept.teams:
        team = _team(ctx, tid)
        if team.private and not chosen:
            continue
        org, _, chains = team_chains(ctx, team)
        w = member_models.head_warning(ctx.models, head_key, member_models.model_key(chains[org.id].primary), f"{team.name} 팀장")
        if w:
            out.append(w)
    return out


def render_team_summaries(ctx: Ctx, dept: building.Department) -> None:
    """팀마다 한 줄 — 팀장 모델 · 이 팀의 비평가 모델(출처) · 경고."""
    for tid in dept.teams:
        team = _team(ctx, tid)
        org, choices, chains = team_chains(ctx, team)
        plan = dept_heads.critic_plan(ctx.models, team, chains, org.id, dept_heads.head_of(dept), ctx.dept_store.get(dept.id))
        lead = member_models.model_key(chains[org.id].primary)
        critic = member_models.model_key(plan.chain.primary)
        lock = " 🔒" if team.private else ""
        picked = f" · 팀원 {len(choices)}명 직접 선택" if choices else ""
        st.markdown(f"- **{team.name}**{lock} — 팀장 {_short(ctx, lead)} · ⚖️ 비평 **{_short(ctx, critic)}** "
                    f"<small>({plan.source_label})</small>{picked}", unsafe_allow_html=True)
        for w in plan.warnings:
            st.caption(f"&emsp;⚠️ {w}")


# ─── 팀원 ─────────────────────────────────────────────────────────────


def member_widget_key(team_id: str, node_id: str) -> str:
    return f"member_model_{team_id}_{node_id}"


def _name_of(node: OrgNode, ranks_by_id: dict) -> str:
    return node.name or ranks_by_id[node.rank].name


def _on_member_change(ctx: Ctx, team: Team, org: OrgNode, node_id: str) -> None:
    """선택을 저장하고, 규칙을 어기게 된 하위를 낮춘 뒤 무엇이 바뀌었는지 알림으로 남긴다 (위젯 콜백)."""
    new_key = st.session_state.get(member_widget_key(team.id, node_id)) or ""
    legacy = _legacy(team)
    choices, changes = member_models.apply_choice(org, ctx.models, team, ctx.member_store.load(team.id, legacy), node_id, new_key)
    ctx.member_store.save(team.id, choices, legacy)
    notes = st.session_state.setdefault(NOTE_KEY, {})
    if changes:
        ranks_by_id = {r.id: r for r in ctx.ranks}
        nodes = {n.id: n for n in org.walk()}
        lines = [f"{_name_of(nodes[c.node_id], ranks_by_id)}: {member_models.split_key(c.before)[1]} → "
                 f"{member_models.split_key(c.after)[1]}" for c in changes]
        notes[team.id] = "🔽 상하 관계 규칙에 따라 아래 팀원의 모델이 함께 바뀌었습니다 — " + " · ".join(lines)
    else:
        notes.pop(team.id, None)


def _reset_members(ctx: Ctx, team: Team) -> None:
    ctx.member_store.save(team.id, {}, _legacy(team))
    st.session_state.setdefault(NOTE_KEY, {})[team.id] = "모든 팀원을 기본(직급 설정)으로 되돌렸습니다."


def usage_text(rows: list[dict]) -> str:
    if not rows:
        return "최근 사용 없음"
    calls = sum(r["calls"] or 0 for r in rows)
    tokens = sum(r["tokens"] or 0 for r in rows)
    top = rows[0]["model"] + (f" 외 {len(rows) - 1}개" if len(rows) > 1 else "")
    return f"{calls}회 · {tokens / 1000:,.1f}천 토큰 · {top}"


def chosen_off(plan: member_models.MemberPlan, runtime_states: dict) -> tuple[str, str] | None:
    """(모델, 이유) — 사람이 고른(또는 규칙으로 낮춘) 모델을 지금 쓸 수 없을 때. 기본 팀원이거나 쓸 수 있으면 None."""
    if not (plan.choice or plan.limited):
        return None
    provider, model = member_models.split_key(plan.effective)
    reason = llm_factory.STATUS_REASONS.get(runtime_states.get(provider, "no_key"), "")
    return (model, reason) if reason else None


def render_members(ctx: Ctx, team: Team) -> None:
    """팀원 표 — 조직도 순서로 한 줄에 한 명 (모델 · 최근 사용량 · 고른 모델 꺼짐)."""
    org = team_org(team, ctx.ranks)
    ranks_by_id = {r.id: r for r in ctx.ranks}
    choices = ctx.member_store.load(team.id, _legacy(team))
    # 저장된 선택 중 사라졌거나 팀 업무를 못 하는 모델 — 저장값은 그대로, 이번엔 대체·기본으로 (경고 표시, core/model_caps.py)
    _, issues = model_caps.check_choices(choices, model_caps.TEAM_WORK)
    issue_of = {i.node_id: i for i in issues}
    hidden_all: dict[str, model_caps.Hidden] = {}
    st.caption(
        ("지금: 모두 기본(직급 설정). " if not choices else f"지금: {len(choices)}명 직접 선택. ")
        + "**기본** = config/models.yaml 의 직급 모델. 규칙: 팀원은 상사보다 강한 모델을 쓸 수 없고, 상사의 모델을 **직접 고르면** "
        "그 아래 팀원은 **더 낮은** 모델만 쓸 수 있습니다. 팀장이 기본이면(Opus 5.5) 부장도 Opus 까지 고를 수 있습니다. "
        "상사를 바꿔 규칙에 걸린 팀원은 자동으로 낮춰집니다. 모델이 안 되면 직급의 대안 모델로 넘어가되, 상사 규칙을 넘는 대안은 쓰지 않습니다."
    )
    note = (st.session_state.get(NOTE_KEY) or {}).pop(team.id, None)
    if note:
        st.info(note)
    plans = member_models.plan(org, ctx.models, team, choices)
    runtime_plans = member_models.plan(org, ctx.runtime, team, choices) if ctx.runtime_models else plans
    states = llm_factory.provider_status(ctx.models)
    runtime_states = llm_factory.provider_status(ctx.runtime) if ctx.runtime_models else states
    used = ctx.storage.usage_by_agent(team.id, since=time.time() - USAGE_DAYS * 86400)
    depth = {org.id: 0}
    for node in org.walk():
        for child in node.children:
            depth[child.id] = depth[node.id] + 1
    head = st.columns([3, 5, 3])
    head[0].caption("팀원")
    head[1].caption("모델 · 제공처 · 비용 (구독 / 유료 API / 무료)")
    head[2].caption(f"최근 {USAGE_DAYS}일 사용량")
    for node in org.walk():
        p = plans[node.id]
        # 팀 업무를 못 하는 모델·제공처에서 사라진 모델은 선택지에서 숨긴다 (이유는 아래 'ⓘ 숨긴 모델')
        shown, hidden = model_caps.selectable_models(model_caps.TEAM_WORK, member_models.allowed_keys(ctx.models, node, p))
        hidden_all.update({h.key: h for h in hidden})
        options = ["", *shown]
        # '기본'을 고르면 쓸 모델 (본인 선택만 뺀 결과 — 상사 규칙에 걸리면 자동으로 낮춘 모델)
        as_default = p if not p.choice else member_models.plan(
            org, ctx.models, team, {k: v for k, v in choices.items() if k != node.id})[node.id]
        default_label = ("🔽 기본(자동 제한) · " if as_default.limited else "기본 · ") + _label(ctx, as_default.effective, states)
        key = member_widget_key(team.id, node.id)
        st.session_state[key] = p.choice if p.choice in options else ""  # 저장값이 기준 (자동으로 낮춘 값 반영)
        c_name, c_pick, c_use = st.columns([3, 5, 3])
        rank = ranks_by_id[node.rank]
        name = _name_of(node, ranks_by_id)
        c_name.markdown("&emsp;" * depth[node.id] + f"{rank.emoji} **{name}**" + ("" if name.startswith(rank.name) else f" · {rank.name}"))
        c_pick.selectbox(
            f"{name} 모델", options, key=key, label_visibility="collapsed",
            format_func=lambda k, d=default_label: d if not k else _label(ctx, k, states),
            on_change=_on_member_change, args=(ctx, team, org, node.id),
        )
        c_use.caption(usage_text(used.get(node.id, [])))
        off = chosen_off(runtime_plans[node.id], runtime_states)
        if off:
            c_use.markdown(f":red[⛔ **고른 모델 꺼짐**] — {off[1]}")  # 무료 서버면 'Docker Desktop 실행 후 FreeLLMAPI 켜기'까지
        issue = issue_of.get(node.id)
        if issue:  # 저장된 선택이 사라졌거나 못 하는 모델 — 저장값은 그대로, 이번엔 대체·기본으로
            use = member_models.split_key(issue.use)[1] if issue.use else "기본(직급의 다음 후보)"
            c_use.markdown(f":orange[⚠️ **고른 모델 {member_models.split_key(issue.key)[1]}**] — {issue.reasons[0]} → {use} 사용")
    models_ui.hidden_note(list(hidden_all.values()), ctx.merged)
    if choices:
        st.button("↩️ 모두 기본으로", key=f"member_reset_{team.id}", on_click=_reset_members, args=(ctx, team))


# ─── 부서 하나 전체 ────────────────────────────────────────────────────


def render_dept(ctx: Ctx, dept: building.Department, where: str, team_id: str | None = None) -> None:
    """부서 → 부장 → 팀 요약 → 팀원 표(고른 팀 하나). team_id = 처음에 펼칠 팀 (팀 화면 = 지금 팀, 메인 = 없음)."""
    if ctx.running:
        st.info("진행 중인 업무는 시작할 때의 모델을 그대로 씁니다. 바꾼 선택은 다음에 맡기는 업무부터 적용됩니다.")
    render_head(ctx, dept, where)
    st.markdown(f"**{dept.emoji} {dept.name}의 팀** — 팀장 모델과 이 팀을 비평할 모델")
    render_team_summaries(ctx, dept)
    pick_key, anchor = f"dept_team_pick_{where}_{dept.id}", f"dept_team_anchor_{where}_{dept.id}"
    options = ["", *dept.teams] if team_id is None else list(dept.teams)
    if st.session_state.get(anchor) != team_id or st.session_state.get(pick_key) not in options:
        st.session_state[pick_key] = team_id if team_id in options else options[0]  # 팀을 옮기면 그 팀부터
        st.session_state[anchor] = team_id
    names = {t.id: t.name for t in ctx.teams}
    picked = st.radio("팀원 모델을 볼 팀", options, key=pick_key, horizontal=True,
                      format_func=lambda t: names.get(t, t) if t else "— 접기 —")
    if picked:
        st.markdown(f"**🎛 {names[picked]} 팀원 모델**")
        render_members(ctx, _team(ctx, picked))
    st.caption("🆕 = 최근 각 사 목록에 새로 나온 모델 · ✍️ = 직접 입력한 모델 · ⚠️ = 목록에서 사라진 모델. "
               "목록에 없는 모델은 '➕ 직접 입력'으로 넣습니다.")


def render_home(ctx: Ctx, pick_key: str) -> None:
    """🏠 메인 — 부서를 고르면 그 부서의 부장·팀·팀원 (빌딩의 부서 띠를 누르면 그 부서로)."""
    depts = departments(ctx)
    ids = [d.id for d in depts]
    if st.session_state.get(pick_key) not in ids:
        st.session_state[pick_key] = ids[0]
    dept_id = st.selectbox("부서", ids, key=pick_key,
                           format_func=lambda i: next(f"{d.emoji} {d.name} ({len(d.teams)}팀)" for d in depts if d.id == i))
    render_dept(ctx, next(d for d in depts if d.id == dept_id), "home")
