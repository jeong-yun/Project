"""26-Agent-Office 메인 화면 (Streamlit).

실행: streamlit run app.py

화면 구성 (위 → 아래)
  ① 조직도   : 👤 사용자 ↔ supervisor(팀장) → 전문가들 → 하위 전문가 (첨부 그림의 Multi-Agent Team 구조)
               각 칸에 지금 하는 일(📋 분배 중 / ✍️ 작성 중 / 🔍 검토 중 …)이 실시간으로 표시된다
  ② 문의 대상 : 기본은 팀장. 다른 칸을 고르면 그 에이전트가 supervisor 가 되어 그 아래 팀만 일한다
  ③ 질문      : 질문 + '원하는 최종 상태'(목표·완료 기준·제약·형식). 예시를 고르면 자동으로 채워진다
  ④ 타임라인  : 업무 브리프 → 웹 검색 자료 → 분배·초안·검토·통합 단계가 끝나는 대로 표시
  ⑤ 최종 결정 : 사람이 채택(수정 가능)/반려 → 참여한 에이전트마다 교훈 저장
               결정 없이 버린 결과는 저장소에서 삭제된다 (피드백이 있는 결과만 보관)

화면 로직만 여기 있고, 실제 일은 core/ 에서 한다.
  core/workflow.py  LangGraph 워크플로우 (브리프 → 검색 → 분배·초안·검토·통합 → 사람 결정 → 피드백)
  core/agent.py     에이전트 한 명의 LLM 호출 (분배·초안·검토)
"""

import base64
import html
import os
import uuid
from pathlib import Path

import streamlit as st
from pydantic import ValidationError

from core import llm_factory
from core.agent import NoAvailableModelError
from core.brief import EXAMPLES, DesiredState, empty_state
from core.config import (
    BASE_DIR,
    ConfigError,
    list_teams,
    load_env,
    load_models,
    load_ranks,
    load_settings,
    load_verification,
    parent_map,
    rank_models,
    resolve_avatar,
    subtree,
    team_org,
    validate_config,
)
from core.harness import REGISTRY as HARNESS_REGISTRY
from core.schemas import AgentAnswer, HumanDecision, OrgNode
from core.search import WebContext
from core.storage import Storage
from core.workflow import WEB_SEARCH_TOOL, build_office_graph, make_checkpointer, resume_with_decision, stream_office

load_env()
st.set_page_config(page_title="Agent Office", page_icon="🏢", layout="wide")

# ════════════════════════════════════════════════════════════════════
# 설정 읽기
# ════════════════════════════════════════════════════════════════════
try:
    ranks = load_ranks()
    models = load_models()
    settings = load_settings()
    verification_cfg = load_verification()
    teams = list_teams()
    validate_config(ranks, models, teams)
except (ConfigError, ValidationError, FileNotFoundError) as e:
    st.error(f"설정 파일 오류 — config/ 또는 teams/ 의 YAML 을 확인하세요.\n\n{e}")
    st.stop()

ranks_by_id = {r.id: r for r in ranks}
PROVIDER_STATUS = {"ok": ("✅", "사용 가능"), "disabled": ("⛔", "비활성 (models.yaml)"), "no_key": ("⚪", "키 없음")}
# 단계 종류 → 화면 표시
KIND_LABELS = {"plan": "📋 업무 분배", "draft": "📝 초안", "review": "🔍 검토", "integrate": "🧩 통합"}
WORKING_LABELS = {
    "brief": "🎯 요청 정리 중", "research": "🌐 자료 조사 중", "plan": "📋 업무 분배 중", "draft": "✍️ 작성 중",
    "review": "🔍 검토 중", "integrate": "🧩 통합 중", "critic": "⚖️ 비평 받는 중",
}
RETRY_LABELS = {"critic": "🔁 자가 교정 중", "harness": "🧪 오류 수정 중"}
# AGENT_OFFICE_DATA_DIR 환경변수로 저장 위치를 바꿀 수 있다 (테스트·여러 사용자 분리용)
DATA_DIR = Path(os.getenv("AGENT_OFFICE_DATA_DIR") or BASE_DIR / settings.get("storage", {}).get("data_dir", "data"))


# ════════════════════════════════════════════════════════════════════
# 저장소 (서버가 켜져 있는 동안 하나만 만들어 재사용: st.cache_resource)
# ════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_checkpointer():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return make_checkpointer(DATA_DIR / "checkpoints.db")


def delete_thread(run_id: str) -> None:
    """LangGraph 체크포인트(대기 상태)도 함께 지운다."""
    try:
        get_checkpointer().delete_thread(run_id)
    except Exception:  # noqa: BLE001 — 체크포인터 종류에 따라 미지원일 수 있음
        pass


@st.cache_resource
def get_storage() -> Storage:
    storage = Storage(DATA_DIR / settings.get("storage", {}).get("db_file", "runs.db"))
    # 앱 시작 시: 사람 결정 없이 남은 오래된 실행(피드백 없는 불필요 기록) 정리
    hours = settings.get("storage", {}).get("purge_undecided_after_hours", 1)
    for run_id in storage.purge_undecided(hours):
        delete_thread(run_id)
    return storage


storage = get_storage()


def discard_run(run: dict) -> None:
    """피드백 없이 버리는 실행 — 저장소와 체크포인트에서 삭제."""
    storage.delete_run(run["run_id"])
    delete_thread(run["run_id"])


# ════════════════════════════════════════════════════════════════════
# 사이드바: 팀 / 모델 상태 / 검증 설정
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 팀 선택")
    team_ids = [t.id for t in teams]
    default_team = settings.get("default_team")
    team_id = st.selectbox(
        "팀",
        team_ids,
        index=team_ids.index(default_team) if default_team in team_ids else 0,
        format_func=lambda tid: next(t.name for t in teams if t.id == tid),
        label_visibility="collapsed",
    )
    team = next(t for t in teams if t.id == team_id)
    st.caption(team.description)

    st.markdown("### 모델 provider 상태")
    for provider, status in llm_factory.provider_status(models).items():
        icon, text = PROVIDER_STATUS[status]
        st.markdown(f"{icon} {provider} — {text}")
    st.caption("사용할 수 없는 provider 는 config/models.yaml 의 대안(fallback) 모델로 대체됩니다.")

    st.markdown("### 검증 장치")
    critic = verification_cfg.get("critic", {})
    st.caption(
        f"⚖️ 비평가: {'켜짐' if critic.get('enabled', True) else '꺼짐'} "
        f"({'최종 통합만' if critic.get('scope', 'final') == 'final' else '모든 검토'}, 자가 교정 최대 {critic.get('max_self_corrections', 1)}회)"
    )
    validators = team.harness.validators
    st.caption(
        "🧪 하네스: " + (", ".join(f"{v}({HARNESS_REGISTRY[v].description})" for v in validators) if validators else "없음")
    )
    skills_dir = BASE_DIR / "teams" / team_id / "skills"
    skill_files = sorted(p.stem for p in skills_dir.glob("*.md")) if skills_dir.exists() else []
    st.caption("📚 팀 스킬: " + (", ".join(skill_files) if skill_files else "없음"))

org = team_org(team, ranks)
nodes = {n.id: n for n in org.walk()}
parents = parent_map(org)


def name_of(node_id: str) -> str:
    node = nodes[node_id]
    return node.name or ranks_by_id[node.rank].name


# ════════════════════════════════════════════════════════════════════
# ① 조직도 (HTML/CSS 트리) — 상태가 바뀔 때마다 다시 그린다
# ════════════════════════════════════════════════════════════════════
CHART_CSS = """
<style>
.ao-org{display:flex;align-items:flex-start;gap:14px;overflow-x:auto;padding:16px 8px 8px;
        border:2px dashed rgba(127,127,127,.45);border-radius:14px}
.ao-user{flex:0 0 auto;text-align:center;margin-top:18px}
.ao-user .ao-face{font-size:44px;line-height:1}
.ao-arrow{flex:0 0 auto;font-size:26px;margin-top:34px;opacity:.6}
.ao-tree{flex:1 0 auto}
.ao-tree ul{position:relative;display:flex;justify-content:center;margin:0;padding:18px 0 0 0}
.ao-tree>ul{padding-top:0}
.ao-tree li{list-style:none;position:relative;text-align:center;padding:18px 5px 0 5px}
.ao-tree li::before,.ao-tree li::after{content:'';position:absolute;top:0;right:50%;width:50%;height:18px;
        border-top:2px solid rgba(127,127,127,.6)}
.ao-tree li::after{right:auto;left:50%;border-left:2px solid rgba(127,127,127,.6)}
.ao-tree li:only-child::before,.ao-tree li:only-child::after{display:none}
.ao-tree li:only-child{padding-top:0}
.ao-tree li:first-child::before,.ao-tree li:last-child::after{border:0 none}
.ao-tree li:last-child::before{border-right:2px solid rgba(127,127,127,.6);border-radius:0 6px 0 0}
.ao-tree li:first-child::after{border-radius:6px 0 0 0}
.ao-tree ul ul::before{content:'';position:absolute;top:0;left:50%;height:18px;border-left:2px solid rgba(127,127,127,.6)}
.ao-node{display:inline-block;min-width:104px;max-width:140px;padding:8px 6px;border-radius:12px;
         border:2px solid rgba(127,127,127,.35);background:rgba(127,127,127,.06)}
.ao-node img{width:52px;height:52px;border-radius:50%}
.ao-node .ao-emoji{font-size:40px;line-height:52px}
.ao-node .ao-name{font-weight:700;font-size:.9rem;margin-top:2px}
.ao-node .ao-meta{font-size:.72rem;opacity:.7}
.ao-node.ao-selected{border-color:#4C9AFF;box-shadow:0 0 0 3px rgba(76,154,255,.25)}
.ao-node.ao-dim{opacity:.35}
.ao-node.ao-st-working{border-color:#FFAB00;animation:ao-pulse 1.2s ease-in-out infinite}
.ao-badge{display:inline-block;margin-top:4px;padding:2px 8px;border-radius:999px;font-size:.72rem;font-weight:600}
.ao-b-working{background:#FFF0B3;color:#7A5200}
.ao-b-done{background:#E3FCEF;color:#006644}
.ao-b-revise{background:#DEEBFF;color:#0747A6}
.ao-b-wait{background:rgba(127,127,127,.18)}
.ao-dots i{display:inline-block;width:5px;height:5px;margin-left:3px;border-radius:50%;background:currentColor;
           animation:ao-blink 1.4s infinite both}
.ao-dots i:nth-child(2){animation-delay:.2s}.ao-dots i:nth-child(3){animation-delay:.4s}
@keyframes ao-pulse{0%,100%{box-shadow:0 0 0 0 rgba(255,171,0,.5)}50%{box-shadow:0 0 0 6px rgba(255,171,0,0)}}
@keyframes ao-blink{0%,80%,100%{opacity:.2}40%{opacity:1}}
</style>
"""
DOTS = "<span class='ao-dots'><i></i><i></i><i></i></span>"


@st.cache_data
def image_data_uri(path: str, mtime: float) -> str:
    """이미지를 HTML 에 바로 넣을 수 있는 data URI 로 (파일이 바뀌면 mtime 이 달라져 다시 읽음)."""
    ext = Path(path).suffix.lstrip(".").replace("jpg", "jpeg")
    return f"data:image/{ext};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def avatar_path(node_id: str, working: bool = False) -> Path | None:
    node = nodes[node_id]
    return resolve_avatar(team_id, ranks_by_id[node.rank], working, node)


def avatar_for_chat(node_id: str, working: bool = False) -> str:
    path = avatar_path(node_id, working)
    return str(path) if path else ranks_by_id[nodes[node_id].rank].emoji


def node_html(node: OrgNode, statuses: dict, selected: str, active_ids: set | None, lessons: dict) -> str:
    state, label = statuses.get(node.id, (None, ""))
    path = avatar_path(node.id, working=state == "working")
    img = (
        f"<img src='{image_data_uri(str(path), path.stat().st_mtime)}'/>"
        if path
        else f"<div class='ao-emoji'>{ranks_by_id[node.rank].emoji}</div>"
    )
    classes = ["ao-node"]
    if node.id == selected:
        classes.append("ao-selected")
    if active_ids is not None and node.id not in active_ids:
        classes.append("ao-dim")  # 이번 질문에 참여하지 않는 칸은 흐리게
    if state == "working":
        classes.append("ao-st-working")
    candidates = llm_factory.available_candidates(rank_models(node.rank, models, team), models)
    model = candidates[0][0].model if candidates else "모델 없음"
    badge = (
        f"<div><span class='ao-badge ao-b-{state}'>{html.escape(label)}{DOTS if state == 'working' else ''}</span></div>"
        if state
        else ""
    )
    return (
        f"<div class='{' '.join(classes)}' title='{html.escape(node.specialty)}'>{img}"
        f"<div class='ao-name'>{html.escape(name_of(node.id))}</div>"
        f"<div class='ao-meta'>{html.escape(model)} · 📚{lessons.get(node.id, 0)}</div>{badge}</div>"
    )


def tree_html(node: OrgNode, *args) -> str:
    children = "".join(tree_html(c, *args) for c in node.children)
    return f"<li>{node_html(node, *args)}{f'<ul>{children}</ul>' if children else ''}</li>"


def render_chart(slot, statuses: dict, selected: str, active_ids: set | None = None) -> None:
    lessons = storage.lesson_counts(team_id)
    slot.markdown(
        CHART_CSS
        + "<div class='ao-org'><div class='ao-user'><div class='ao-face'>👤</div><b>사용자</b></div>"
        + "<div class='ao-arrow'>⟷</div>"
        + f"<div class='ao-tree'><ul>{tree_html(org, statuses, selected, active_ids, lessons)}</ul></div></div>",
        unsafe_allow_html=True,
    )


def step_status(step: AgentAnswer) -> tuple[str, str]:
    """완료된 단계 → 조직도 배지."""
    if step.kind == "plan":
        return "wait", "⏳ 하위 결과 대기"
    if step.kind == "draft":
        return "done", "📝 초안 완료"
    if step.review and step.review.verdict == "approve":
        return "done", f"✅ 승인 {step.review.score}점"
    return "revise", f"{'🧩 통합' if step.kind == 'integrate' else '✏️ 보완'} {step.review.score if step.review else ''}점"


def statuses_from_run(run: dict) -> dict:
    statuses: dict = {}
    for step in run["steps"]:
        statuses[step.node_id] = step_status(step)
        if step.delegation:
            for a in step.delegation.assignments:
                statuses.setdefault(a.child_id, ("wait", "⏳ 대기"))
    if run["status"] == "awaiting_human":
        statuses[run["root_id"]] = ("wait", "👤 결정 대기")
    return statuses


# ════════════════════════════════════════════════════════════════════
# ④ 타임라인 렌더링
# ════════════════════════════════════════════════════════════════════
def render_brief(brief: DesiredState | None) -> None:
    if brief is None:
        return
    with st.container(border=True):
        st.markdown(f"**🎯 업무 브리프 (최종 상태 정의)** — {brief.goal}")
        if brief.success_criteria:
            st.markdown("**완료 기준**\n" + "\n".join(f"- {c}" for c in brief.success_criteria))
        extra = []
        if brief.constraints:
            extra.append("제약: " + " · ".join(brief.constraints))
        if brief.output_format:
            extra.append("형식: " + brief.output_format)
        if brief.assumptions:
            extra.append("가정: " + " · ".join(brief.assumptions))
        for line in extra:
            st.caption(line)


def render_web(web: WebContext | None) -> None:
    if web is None:
        return
    title = f"🌐 웹 검색 자료 {len(web.results)}건" if web.results else "🌐 웹 검색 — 사용한 자료 없음"
    with st.expander(title, expanded=False):
        if web.queries:
            st.caption("검색어: " + " · ".join(web.queries))
        for i, r in enumerate(web.results, 1):
            st.markdown(f"**[{i}] [{r.title}]({r.url})**  \n{r.snippet[:300]}")
        for note in web.notes:
            st.caption(f"ℹ️ {note[:300]}")


def render_meta(step: AgentAnswer) -> None:
    tokens = (
        f" · 토큰 {step.input_tokens or 0:,} → {step.output_tokens or 0:,}"
        if step.input_tokens is not None or step.output_tokens is not None
        else ""
    )
    fallback = " · ⚠️ 대안 모델" if step.is_fallback else ""
    attempts = f" · 작성 {step.attempts}회" if step.attempts > 1 else ""
    st.caption(f"{step.model.label} · {step.elapsed_sec}초{tokens}{fallback}{attempts}")


def render_verification(step: AgentAnswer) -> None:
    """비평가·하네스·자가 교정 기록."""
    if not (step.critic or step.harness or step.verification_log or step.errors):
        return
    marks = []
    if step.critic:
        ok = not step.critic.logical_leaps and not step.critic.rule_violations
        marks.append(f"⚖️ 비평 {step.critic.score}/5 {'통과' if ok else '지적'}")
    if step.harness:
        marks.append("🧪 하네스 " + ("통과" if all(h.passed for h in step.harness) else "실패"))
    with st.expander("검증 기록 · " + (" · ".join(marks) if marks else "상세")):
        if step.critic:
            st.markdown(f"**비평가 분석**: {step.critic.analysis}")
            for x in [*step.critic.logical_leaps, *step.critic.rule_violations]:
                st.markdown(f"- {x}")
        for h in step.harness:
            st.markdown(f"- 🧪 `{h.validator}` {'✅' if h.passed else '❌'} {h.log}")
        for log in step.verification_log:
            st.markdown(f"- 🔁 {log}")
        for err in step.errors:
            st.text(f"건너뛴 모델: {err[:300]}")


def render_step(step: AgentAnswer, root_id: str) -> None:
    """타임라인의 한 단계. root_id = 이번 실행의 supervisor (그 결과는 사용자에게 보고된다)."""
    if step.kind == "plan":
        route = ""
    elif step.node_id == root_id:
        route = " → 👤 사용자에게 보고"
    else:
        route = f" → {name_of(parents[step.node_id])}에게 보고"
    with st.chat_message("assistant", avatar=avatar_for_chat(step.node_id)):
        st.markdown(f"**{name_of(step.node_id)}** · {KIND_LABELS[step.kind]}{route}")
        if step.kind == "plan" and step.delegation:
            st.caption(step.delegation.analysis)
            st.markdown(step.content)
        elif step.review:
            r = step.review
            verdict = "✅ 승인" if r.verdict == "approve" else ("🧩 통합" if step.kind == "integrate" else "✏️ 보완")
            st.markdown(f"{verdict} · 하위 결과 {r.score}/10점")
            st.caption(f"분석: {r.analysis}")
            if r.issues:
                st.markdown("\n".join(f"- {issue}" for issue in r.issues))
            if r.comment:
                st.markdown(f"> {r.comment}")
            if r.verdict == "revise" or step.kind == "integrate":
                with st.expander(f"{name_of(step.node_id)} 결과 보기"):
                    st.markdown(step.content)
        else:
            st.markdown(step.content)
        render_meta(step)
        render_verification(step)


def final_answer(run: dict) -> AgentAnswer | None:
    mine = [s for s in run["steps"] if s.node_id == run["root_id"] and s.kind != "plan"]
    return mine[-1] if mine else None


def render_final(run: dict) -> None:
    final = final_answer(run)
    if not final:
        return
    st.markdown(f"#### 📌 최종 결과 — {name_of(final.node_id)}")
    with st.container(border=True):
        st.markdown(final.content)
    if team.disclaimer:
        st.caption(f"ℹ️ {team.disclaimer}")


# ════════════════════════════════════════════════════════════════════
# ⑤ 사람의 최종 결정
# ════════════════════════════════════════════════════════════════════
def adoptable_steps(run: dict) -> list[AgentAnswer]:
    """채택 후보 = 에이전트별 마지막 결과 (분배 단계 제외). 최종 결과(supervisor)를 맨 앞에."""
    last = {s.node_id: s for s in run["steps"] if s.kind != "plan"}
    final = final_answer(run)
    others = [s for nid, s in last.items() if not final or nid != final.node_id]
    return ([final] if final else []) + others


def submit_decision(run: dict, decision: HumanDecision) -> None:
    """멈춰 있던 그래프를 사람의 결정으로 재개하고, 생성된 교훈을 기록한다."""
    lessons, notes = [], []
    with st.spinner("결정을 반영하고 에이전트별 피드백을 정리하는 중…"):
        for kind, payload in resume_with_decision(make_graph(), org, decision, run["run_id"]):
            if kind == "lessons":
                lessons = payload
            elif kind == "note":
                notes.append(payload)
    storage.save_decision(run["run_id"], decision)  # 피드백이 있는 결과만 저장소에 남는다
    run.update(status="adopted" if decision.action == "adopt" else "rejected",
               decision=decision, lessons=lessons, notes=notes)


def render_decision(run: dict, run_key: tuple) -> None:
    st.markdown("#### 👤 최종 결정")
    if run["status"] == "awaiting_human":
        candidates = adoptable_steps(run)
        st.caption(
            "채택할 안을 고르고 필요하면 직접 수정하세요. 결정 내용은 참여한 모든 에이전트에게 교훈으로 피드백됩니다. "
            "결정하지 않고 '결과 버리기'를 누르거나 새 질문을 하면 이 결과는 저장되지 않습니다."
        )
        # 선택이 바뀌면 아래 최종 답변 입력란이 해당 안으로 채워지도록 form 밖에 둔다
        idx = st.radio(
            "채택할 안",
            list(range(len(candidates))),
            format_func=lambda i: f"{name_of(candidates[i].node_id)} {KIND_LABELS[candidates[i].kind]}"
            + (" (최종)" if i == 0 else ""),
            horizontal=True,
            key=f"pick_{run['run_id']}",
        )
        chosen = candidates[idx]
        with st.form(f"decide_{run['run_id']}"):
            final = st.text_area("최종 답변 (필요하면 직접 수정)", value=chosen.content, height=260,
                                 key=f"final_{run['run_id']}_{idx}")
            score = st.slider("만족도", 1, 5, 4)
            comment = st.text_input("코멘트 (에이전트들에게 전달할 피드백, 선택)")
            c1, c2, c3 = st.columns(3)
            adopt = c1.form_submit_button("✅ 이 안으로 채택", type="primary", use_container_width=True)
            reject = c2.form_submit_button("❌ 전체 반려", use_container_width=True)
            discard = c3.form_submit_button("🗑 결과 버리기 (저장 안 함)", use_container_width=True)
        if discard:
            discard_run(run)
            del st.session_state["runs"][run_key]
            st.rerun()
        if adopt or reject:
            decision = HumanDecision(
                action="adopt" if adopt else "reject",
                adopted_node_id=chosen.node_id if adopt else None,
                final_answer=final.strip() if adopt else "",
                edited=bool(adopt and final.strip() != chosen.content.strip()),
                score=score,
                comment=comment.strip(),
            )
            try:
                submit_decision(run, decision)
            except Exception as e:  # noqa: BLE001
                st.error(f"결정 반영 중 오류: {type(e).__name__}: {e}")
                return
            st.rerun()
        return

    decision: HumanDecision = run["decision"]
    if decision.action == "adopt":
        edited = " · ✍️ 직접 수정" if decision.edited else ""
        st.success(f"✅ {name_of(decision.adopted_node_id)} 안 채택{edited} · 만족도 {decision.score}/5 · 저장됨")
        with st.container(border=True):
            st.markdown(decision.final_answer)
    else:
        st.warning(f"❌ 전체 반려 · 만족도 {decision.score}/5 · 저장됨")
    if decision.comment:
        st.caption(f"코멘트: {decision.comment}")

    st.markdown("##### 📚 에이전트별 피드백 (다음 업무부터 프롬프트에 반영)")
    for lesson in run["lessons"]:
        icon = "👍" if lesson.kind == "praise" else "🔧"
        st.markdown(f"- **{name_of(lesson.agent_id)}** {icon} {lesson.lesson}")
    if run["notes"]:
        with st.expander("피드백 생성 참고"):
            for note in run["notes"]:
                st.text(note[:500])


def render_run(run: dict, run_key: tuple) -> None:
    """저장된 실행 결과를 다시 그린다 (버튼 클릭 등으로 화면이 다시 그려질 때)."""
    st.markdown(f"**Q.** {run['question']}")
    render_brief(run.get("brief"))
    render_web(run.get("web"))
    for step in run["steps"]:
        render_step(step, run["root_id"])
    if run["error"]:
        st.error(run["error"])
        return
    if run["steps"]:
        render_final(run)
        render_decision(run, run_key)


def make_graph():
    return build_office_graph(
        org, ranks, team, models, settings, verification_cfg, storage=storage, checkpointer=get_checkpointer()
    )


# ════════════════════════════════════════════════════════════════════
# 본문
# ════════════════════════════════════════════════════════════════════
st.title("🏢 Agent Office")
st.caption(f"현재 팀: **{team.name}** · 사용자는 맨 위 에이전트(supervisor)에게 일을 맡기고, supervisor 가 전문가들에게 나눠 줍니다.")

st.session_state.setdefault("runs", {})  # {(team_id, 문의 대상 id): run dict}

chart_slot = st.empty()

# ② 문의 대상 — 기본은 supervisor(조직도 맨 위)
target = st.radio(
    "문의 대상 (고른 에이전트가 supervisor 가 되어 그 아래 팀이 일합니다)",
    options=[n.id for n in org.walk()],
    index=0,  # 조직도 맨 위(supervisor)
    format_func=name_of,
    horizontal=True,
    key=f"target_{team_id}",
)
active_ids = {n.id for n in subtree(org, target).walk()}
run_key = (team_id, target)
current = st.session_state["runs"].get(run_key)
render_chart(chart_slot, statuses_from_run(current) if current else {}, target, active_ids)

with st.expander("👥 에이전트 상세 (모델 · 전문 분야 · 누적 교훈)"):
    lesson_counts = storage.lesson_counts(team_id)
    rows = []
    for n in org.walk():
        cands = llm_factory.available_candidates(rank_models(n.rank, models, team), models)
        spec, fb = cands[0] if cands else (None, False)
        rows.append({
            "에이전트": name_of(n.id), "직급": ranks_by_id[n.rank].name, "전문 분야": n.specialty,
            "모델": (spec.label + (" (대안)" if fb else "")) if spec else "없음",
            "하위": ", ".join(name_of(c.id) for c in n.children) or "-", "교훈": lesson_counts.get(n.id, 0),
        })
    st.dataframe(rows, hide_index=True, use_container_width=True)

# ③ 질문 — 예시 선택 / 좋은 질문 쓰는 법 / 최종 상태 정의
st.divider()
st.markdown(f"#### 💬 {name_of(target)} 에게 맡기기")


def apply_example() -> None:
    """예시를 고르면 질문·최종 상태 입력란을 채운다 (selectbox 의 on_change 콜백)."""
    title = st.session_state["example_pick"]
    ex = next((e for e in EXAMPLES if e["title"] == title), None)
    if ex is None:
        return
    st.session_state["q_text"] = ex["question"]
    st.session_state["b_goal"] = ex["goal"]
    st.session_state["b_criteria"] = "\n".join(ex["success_criteria"])
    st.session_state["b_constraints"] = "\n".join(ex["constraints"])
    st.session_state["b_format"] = ex["output_format"]


c1, c2 = st.columns([1, 1])
c1.selectbox(
    "💡 질문 예시 (고르면 아래가 채워집니다)",
    ["(직접 입력)", *[e["title"] for e in EXAMPLES]],
    key="example_pick",
    on_change=apply_example,
)
with c2.expander("📝 좋은 질문 쓰는 법 — '과정'이 아니라 '최종 상태'를 정의하기"):
    st.markdown(
        """
- ❌ 과정 지시: "1) HBF 검색하고 2) 정리한 다음 3) 표로 만들어줘"
- ✅ 최종 상태: "**목표** 팀 회의에서 5분 안에 공유할 HBF 설명 자료 / **완료 기준** 정식 명칭·정의, HBM 과 비교 표, 사실마다 출처 / **제약** 비전공자용, A4 1장"
- 과정(누가 무엇을 조사할지)은 supervisor 가 스스로 나눕니다. 사람은 **무엇이 되면 끝인지**만 정하면 됩니다.
- 완료 기준은 검토자·비평가가 '충족/미충족'을 판단하는 채점 기준이 됩니다 → 검증 가능한 문장으로 쓰세요.
- 결과는 그대로 믿지 말고, 최종 결정 전에 '취약점이 없는지' 타임라인의 검토·비평 기록을 확인하세요.
"""
    )

web_available = WEB_SEARCH_TOOL in team.tools
with st.form("ask_form", clear_on_submit=False):
    question = st.text_area("질문", placeholder="무엇을 원하는지 한두 문장으로", height=90, key="q_text")
    with st.expander("🎯 원하는 최종 상태 (선택 — 비워 두면 AI 가 질문에서 정리합니다)", expanded=False):
        goal = st.text_input("최종 목표 — 결과물과 용도", key="b_goal", placeholder="예: 팀 회의에서 공유할 1장짜리 설명 자료")
        criteria = st.text_area("완료 기준 — 한 줄에 하나", key="b_criteria", height=90,
                                placeholder="예: 정식 명칭과 정의를 쓴다\n예: 사실마다 출처 번호를 단다")
        constraints = st.text_area("제약 — 한 줄에 하나", key="b_constraints", height=68, placeholder="예: 비전공자용, A4 1장")
        output_format = st.text_input("결과 형식", key="b_format", placeholder="예: 요약 → 비교 표 → 핵심 3개 → 출처")
    o1, o2 = st.columns(2)
    use_web = o1.checkbox("🌐 웹 검색 사용", value=web_available, disabled=not web_available,
                          help="답변 전에 검색어를 만들어 웹을 검색하고, 결과를 모든 에이전트가 근거로 사용합니다.")
    auto_brief = o2.checkbox("🤖 빈 항목은 AI 가 정리", value=True,
                             help="최종 목표·완료 기준이 비어 있으면 보조 모델이 질문에서 추론해 채웁니다.")
    submitted = st.form_submit_button("맡기기", type="primary")


def lines(text: str) -> list[str]:
    return [x.strip() for x in text.splitlines() if x.strip()]


# ④ 실행 — 이벤트가 올 때마다 조직도·타임라인을 갱신한다
if submitted and not question.strip():
    st.warning("질문을 입력하세요.")
elif submitted:
    # 같은 대상에 대한 직전 결과가 결정 없이 남아 있으면 → 피드백 없는 불필요 기록이므로 삭제
    previous = st.session_state["runs"].get(run_key)
    if previous and previous["status"] not in ("adopted", "rejected"):
        discard_run(previous)

    brief_input = DesiredState(
        goal=goal.strip(), context="", success_criteria=lines(criteria), constraints=lines(constraints),
        output_format=output_format.strip(), assumptions=[],
    )
    run = {
        "run_id": uuid.uuid4().hex, "question": question.strip(), "root_id": target, "steps": [],
        "status": "reviewing", "error": None, "decision": None, "lessons": [], "notes": [], "web": None, "brief": None,
    }
    st.session_state["runs"][run_key] = run
    storage.create_run(run["run_id"], team_id, target, run["question"])

    statuses: dict = {target: ("working", "🎯 요청 정리 중")}
    render_chart(chart_slot, statuses, target, active_ids)
    brief_slot, web_slot = st.empty(), st.empty()
    timeline = st.container()
    with timeline:
        draft_slot = st.empty()  # 실무자가 쓰는 중인 초안 미리보기
    progress = st.empty()
    drafting = {"node": None, "text": ""}

    def show_working(node_id: str, text: str) -> None:
        with progress.container(), st.chat_message("assistant", avatar=avatar_for_chat(node_id, working=True)):
            st.markdown(f"<span class='ao-badge ao-b-working'>{html.escape(text)}{DOTS}</span>", unsafe_allow_html=True)

    show_working(target, f"{name_of(target)} — 요청을 '최종 상태'로 정리하는 중")
    try:
        events = stream_office(
            make_graph(), org, run["question"], target, thread_id=run["run_id"],
            use_web=bool(use_web and web_available),
            brief=None if brief_input.is_empty() and not brief_input.constraints and not brief_input.output_format else brief_input,
            auto_brief=auto_brief,
        )
        for kind, payload in events:
            if kind == "start":
                nid = payload["node"]
                statuses[nid] = ("working", WORKING_LABELS.get(payload["kind"], "작업 중"))
                show_working(nid, f"{name_of(nid)} — {WORKING_LABELS.get(payload['kind'], '작업 중')}")
            elif kind == "delegated":
                statuses[payload["node"]] = ("wait", "⏳ 하위 결과 대기")
                for cid in payload["children"]:
                    statuses[cid] = ("wait", "⏳ 대기")
            elif kind == "retry":
                nid = payload["node"]
                statuses[nid] = ("working", RETRY_LABELS[payload["reason"]])
                show_working(nid, f"{name_of(nid)} — {RETRY_LABELS[payload['reason']]}")
                drafting.update(node=None, text="")
            elif kind == "brief":
                run["brief"] = payload
                with brief_slot.container():
                    render_brief(payload)
            elif kind == "research":
                run["web"] = payload
                with web_slot.container():
                    render_web(payload)
            elif kind == "note":
                run["notes"].append(payload)
            elif kind == "token":
                nid, text = payload
                if drafting["node"] != nid:
                    drafting.update(node=nid, text="")
                    progress.empty()
                drafting["text"] += text
                with draft_slot.container(), st.chat_message("assistant", avatar=avatar_for_chat(nid, working=True)):
                    st.markdown(f"**{name_of(nid)}** · ✍️ 작성 중\n\n{drafting['text']}▌")
                continue  # 토큰마다 조직도를 다시 그리지 않는다
            elif kind == "interrupt":
                run["status"] = "awaiting_human"
                statuses[target] = ("wait", "👤 결정 대기")
            elif kind == "step":
                run["steps"].append(payload)
                storage.save_steps(run["run_id"], run["steps"], "reviewing")
                statuses[payload.node_id] = step_status(payload)
                if payload.delegation:
                    for a in payload.delegation.assignments:
                        statuses[a.child_id] = ("wait", "⏳ 대기")
                draft_slot.empty()
                drafting.update(node=None, text="")
                with timeline:
                    render_step(payload, target)
            render_chart(chart_slot, statuses, target, active_ids)
        progress.empty()
        storage.save_steps(run["run_id"], run["steps"], run["status"])
        render_final(run)
        render_decision(run, run_key)
    except Exception as e:  # noqa: BLE001 — 모델 전부 실패, 스트리밍 도중 실패 등
        progress.empty()
        run["status"] = "error"
        run["error"] = str(e) if isinstance(e, NoAvailableModelError) else f"진행 중 오류: {type(e).__name__}: {e}"
        discard_run(run)  # 결정할 수 없는 실패 결과는 저장하지 않는다
        st.error(run["error"])
elif current:
    render_run(current, run_key)
