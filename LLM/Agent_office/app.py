"""Agent Office 메인 화면 (Streamlit).

실행: streamlit run app.py

화면 구성 (위 → 아래)
  ① 문의 대상  : 기본은 팀장(supervisor). 다른 칸을 고르면 그 에이전트가 supervisor 가 되어 그 아래 팀만 일한다
  ② 실시간 영역 (작업 중에는 1초마다 자동으로 다시 그림 — 작업은 백그라운드에서 계속된다)
     - 조직도   : 👤 사용자 ↔ supervisor → 전문가들. 칸을 **클릭**하면 그 에이전트를 아래 패널에서 본다
     - 에이전트 패널 : 고른 에이전트의 상태·맡은 업무·작성 중인 초안·결과 (스크롤 상자), 🖼 이미지 바꾸기
     - 탭       : 📜 전체 진행(모든 단계, 스크롤) / 🎯 브리프·근거 자료(데이터·웹 검색)
  ③ 최종 결과·결정 : 사람이 채택(수정 가능)/반려 → 참여한 에이전트마다 교훈 저장 (결정 없이 버리면 삭제)
  ④ 업무 맡기기 : 팀 업무 양식(예: 주가 전망 — 종목·기간 입력칸) 또는 자유 질문 + '원하는 최종 상태'

화면 로직만 여기 있고, 실제 일은 core/ 에서 한다.
  core/live.py      백그라운드 실행 + 진행 상태(run dict)
  core/workflow.py  LangGraph 워크플로우 (브리프 → 자료 → 분배·초안·검토·통합 → 사람 결정 → 피드백)
  core/agent.py     에이전트 한 명의 LLM 호출 (분배·초안·검토)
"""

import base64
import html
import os
import time
import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from pydantic import ValidationError
from st_click_detector import click_detector

import concept_ui
from core import concept, llm_factory, stock_calc
from core.avatars import AvatarError, has_custom_avatar, reset_avatar, save_avatar
from core.brief import EXAMPLES, DesiredState
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
from core.live import KIND_LABELS, new_run, start_run
from core.schemas import AgentAnswer, HumanDecision, OrgNode, TaskField, TaskTemplate, ToolResult
from core.search import WebContext
from core.storage import Storage
from core.tasks import TaskInputError, build_request
from core.exporter import export_adopted, list_results
from core.route import CATEGORIES, RouteResult, build_route, map_html
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
PROVIDER_STATUS = {
    "ok": ("✅", "사용 가능"), "disabled": ("⛔", "비활성"), "no_key": ("⚪", "키 없음"),
    "offline": ("🔌", "서버 꺼짐 (FreeLLMAPI Docker 를 켜세요)"),
}
REFRESH_SEC = 1.0  # 작업 중 화면 자동 갱신 간격
# AGENT_OFFICE_DATA_DIR 환경변수로 저장 위치를 바꿀 수 있다 (테스트·여러 사용자 분리용)
DATA_DIR = Path(os.getenv("AGENT_OFFICE_DATA_DIR") or BASE_DIR / settings.get("storage", {}).get("data_dir", "data"))
# 채택한 결과 파일 저장 폴더 (results/<팀>/). AGENT_OFFICE_RESULTS_DIR 로 바꿀 수 있다
RESULTS_DIR = Path(os.getenv("AGENT_OFFICE_RESULTS_DIR") or BASE_DIR / settings.get("storage", {}).get("results_dir", "results"))
ROUTE_MAP = "route_map"  # team.yaml features 에 있으면 여행 경로 지도를 만든다


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
    st.caption("사용할 수 없는 provider 는 config/models.yaml 의 대안(fallback) 모델로 대체됩니다. "
               "유료 키가 모두 안 되면 마지막에 freellm(무료 모델)으로 넘어갑니다.")
    # 유료 키를 아끼거나 키가 없을 때: 유료 provider 를 모두 끄고 FreeLLMAPI 무료 모델만 쓴다
    if st.toggle("🆓 무료 모델만 사용", key="free_only", help="FreeLLMAPI(무료 티어) 모델만 사용합니다. 무료 모델은 하루 호출 한도가 작습니다."):
        models = llm_factory.free_only(models)
        if llm_factory.provider_status(models).get("freellm") != "ok":
            st.warning("FreeLLMAPI 가 꺼져 있거나 키가 없습니다. README 의 '무료 모델' 절을 확인하세요.")

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

    # 개념 편집 팀: 카탈로그(GitHub CS 저장소) 위치·git 상태
    is_concept = concept_ui.CONCEPT_FEATURE in team.features
    concept_ctx = concept_ui.sidebar(settings) if is_concept else None

    # 채택해서 저장한 결과 파일 — 내려받기
    saved = list_results(RESULTS_DIR, team_id, limit=10)
    st.markdown(f"### 📂 저장된 결과 ({len(saved)})")
    st.caption(f"채택한 결과는 `{RESULTS_DIR.name}/{team_id}/` 폴더에 저장됩니다.")
    for path in saved:
        st.download_button(
            ("🗺 " if path.suffix == ".html" else "📄 ") + path.stem[:34], data=path.read_bytes(), file_name=path.name,
            mime="text/html" if path.suffix == ".html" else "text/markdown", key=f"dl_{path.name}", use_container_width=True,
        )

org = team_org(team, ranks)
nodes = {n.id: n for n in org.walk()}
parents = parent_map(org)


def name_of(node_id: str) -> str:
    node = nodes[node_id]
    return node.name or ranks_by_id[node.rank].name


def make_graph():
    return build_office_graph(
        org, ranks, team, models, settings, verification_cfg, storage=storage, checkpointer=get_checkpointer()
    )


# ════════════════════════════════════════════════════════════════════
# 이미지
# ════════════════════════════════════════════════════════════════════
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


# ════════════════════════════════════════════════════════════════════
# 조직도 (HTML/CSS 트리) — 칸을 클릭하면 그 에이전트를 아래 패널에서 본다
#   click_detector: HTML 안의 <a id="..."> 를 누르면 그 id 를 돌려주는 작은 컴포넌트 (st-click-detector)
#   iframe 안에 그려지므로 색은 밝은/어두운 테마 모두에서 보이도록 카드에 직접 지정한다
# ════════════════════════════════════════════════════════════════════
CHART_CSS = """
<style>
body{margin:0;font-family:'Source Sans Pro','Malgun Gothic',sans-serif}
.ao-org{display:flex;align-items:flex-start;gap:14px;overflow-x:auto;padding:14px 8px 8px;
        border:2px dashed rgba(127,127,127,.45);border-radius:14px}
.ao-user{flex:0 0 auto;text-align:center;margin-top:14px;background:#fff;color:#172B4D;border-radius:12px;padding:6px 10px}
.ao-user .ao-face{font-size:40px;line-height:1}
.ao-arrow{flex:0 0 auto;font-size:26px;margin-top:34px;color:#8993A4}
.ao-tree{flex:1 0 auto}
.ao-tree ul{position:relative;display:flex;justify-content:center;margin:0;padding:18px 0 0 0}
.ao-tree>ul{padding-top:0}
.ao-tree li{list-style:none;position:relative;text-align:center;padding:18px 5px 0 5px}
.ao-tree li::before,.ao-tree li::after{content:'';position:absolute;top:0;right:50%;width:50%;height:18px;border-top:2px solid #A5ADBA}
.ao-tree li::after{right:auto;left:50%;border-left:2px solid #A5ADBA}
.ao-tree li:only-child::before,.ao-tree li:only-child::after{display:none}
.ao-tree li:only-child{padding-top:0}
.ao-tree li:first-child::before,.ao-tree li:last-child::after{border:0 none}
.ao-tree li:last-child::before{border-right:2px solid #A5ADBA;border-radius:0 6px 0 0}
.ao-tree li:first-child::after{border-radius:6px 0 0 0}
.ao-tree ul ul::before{content:'';position:absolute;top:0;left:50%;height:18px;border-left:2px solid #A5ADBA}
a.ao-link{text-decoration:none;color:inherit}
.ao-node{display:inline-block;min-width:104px;max-width:140px;padding:8px 6px;border-radius:12px;cursor:pointer;
         border:2px solid #DFE1E6;background:#fff;color:#172B4D;transition:transform .1s}
.ao-node:hover{transform:translateY(-2px);border-color:#4C9AFF}
.ao-node img{width:52px;height:52px;border-radius:50%;object-fit:cover}
.ao-node .ao-emoji{font-size:40px;line-height:52px}
.ao-node .ao-name{font-weight:700;font-size:.88rem;margin-top:2px}
.ao-node .ao-meta{font-size:.7rem;color:#6B778C}
.ao-node.ao-focus{border-color:#4C9AFF;box-shadow:0 0 0 3px rgba(76,154,255,.35)}
.ao-node.ao-dim{opacity:.4}
.ao-node.ao-st-working{border-color:#FFAB00;animation:ao-pulse 1.2s ease-in-out infinite}
.ao-badge{display:inline-block;margin-top:4px;padding:2px 8px;border-radius:999px;font-size:.7rem;font-weight:600}
.ao-b-working{background:#FFF0B3;color:#7A5200}
.ao-b-done{background:#E3FCEF;color:#006644}
.ao-b-revise{background:#DEEBFF;color:#0747A6}
.ao-b-wait{background:#EBECF0;color:#42526E}
.ao-dots i{display:inline-block;width:5px;height:5px;margin-left:3px;border-radius:50%;background:currentColor;
           animation:ao-blink 1.4s infinite both}
.ao-dots i:nth-child(2){animation-delay:.2s}.ao-dots i:nth-child(3){animation-delay:.4s}
.ao-hint{font-size:.75rem;color:#8993A4;margin:4px 0 0 4px}
@keyframes ao-pulse{0%,100%{box-shadow:0 0 0 0 rgba(255,171,0,.5)}50%{box-shadow:0 0 0 6px rgba(255,171,0,0)}}
@keyframes ao-blink{0%,80%,100%{opacity:.2}40%{opacity:1}}
</style>
"""
DOTS = "<span class='ao-dots'><i></i><i></i><i></i></span>"
NODE_LINK = "node:"  # 조직도 칸 링크 id 접두어


def node_html(node: OrgNode, statuses: dict, focus: str, active_ids: set | None, lessons: dict) -> str:
    state, label = statuses.get(node.id, (None, ""))
    path = avatar_path(node.id, working=state == "working")
    img = (
        f"<img src='{image_data_uri(str(path), path.stat().st_mtime)}'/>"
        if path
        else f"<div class='ao-emoji'>{ranks_by_id[node.rank].emoji}</div>"
    )
    classes = ["ao-node"]
    if node.id == focus:
        classes.append("ao-focus")
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
        f"<a class='ao-link' href='#' id='{NODE_LINK}{node.id}'>"
        f"<div class='{' '.join(classes)}' title='{html.escape(node.specialty)}'>{img}"
        f"<div class='ao-name'>{html.escape(name_of(node.id))}</div>"
        f"<div class='ao-meta'>{html.escape(model)} · 📚{lessons.get(node.id, 0)}</div>{badge}</div></a>"
    )


def tree_html(node: OrgNode, *args) -> str:
    children = "".join(tree_html(c, *args) for c in node.children)
    return f"<li>{node_html(node, *args)}{f'<ul>{children}</ul>' if children else ''}</li>"


def chart_html(statuses: dict, focus: str, active_ids: set | None) -> str:
    lessons = storage.lesson_counts(team_id)
    return (
        CHART_CSS
        + "<div class='ao-org'><div class='ao-user'><div class='ao-face'>👤</div><b>사용자</b></div>"
        + "<div class='ao-arrow'>⟷</div>"
        + f"<div class='ao-tree'><ul>{tree_html(org, statuses, focus, active_ids, lessons)}</ul></div></div>"
        + "<div class='ao-hint'>👆 칸을 누르면 그 에이전트가 한 일을 아래에서 볼 수 있습니다</div>"
    )


# ════════════════════════════════════════════════════════════════════
# 타임라인·자료 렌더링
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


def render_tools(results: list[ToolResult]) -> None:
    """데이터 툴 결과 (시장 데이터·여행 달력 등) — 에이전트들이 근거로 쓴 사실 자료."""
    for r in results:
        with st.expander(r.title, expanded=False):
            if r.content:
                st.markdown(r.content)
            for note in r.notes:
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


def route_of(step: AgentAnswer, root_id: str) -> str:
    if step.kind == "plan":
        return ""
    if step.node_id == root_id:
        return " → 👤 사용자에게 보고"
    return f" → {name_of(parents[step.node_id])}에게 보고"


def render_step_body(step: AgentAnswer, expand_result: bool = False) -> None:
    """한 단계의 내용 (분배 / 초안 / 검토·통합)."""
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
            with st.expander(f"{name_of(step.node_id)} 결과 보기", expanded=expand_result):
                st.markdown(step.content)
    else:
        st.markdown(step.content)
    render_meta(step)
    render_verification(step)


def render_step(step: AgentAnswer, root_id: str) -> None:
    """전체 진행 타임라인의 한 단계."""
    with st.chat_message("assistant", avatar=avatar_for_chat(step.node_id)):
        st.markdown(f"**{name_of(step.node_id)}** · {KIND_LABELS[step.kind]}{route_of(step, root_id)}")
        render_step_body(step)


def final_answer(run: dict) -> AgentAnswer | None:
    mine = [s for s in run["steps"] if s.node_id == run["root_id"] and s.kind != "plan"]
    return mine[-1] if mine else None


def assigned_task(run: dict, node_id: str) -> str | None:
    """이 에이전트가 위에서 받은 업무 지시 (supervisor 면 브리프의 목표)."""
    if node_id == run["root_id"]:
        return run["brief"].goal if run.get("brief") else None
    for s in run["steps"]:
        if s.delegation:
            for a in s.delegation.assignments:
                if a.child_id == node_id:
                    return a.subtask
    parent = parents.get(node_id)
    if parent and any(s.node_id == parent for s in run["steps"]):
        return f"{name_of(parent)}의 업무를 그대로 넘겨받음"
    return None


# ════════════════════════════════════════════════════════════════════
# 에이전트 패널 — 조직도에서 고른 에이전트가 한 일 + 이미지 바꾸기
# ════════════════════════════════════════════════════════════════════
def render_avatar_editor(node_id: str) -> None:
    with st.expander("🖼 이 에이전트 이미지 바꾸기"):
        st.caption(
            "png·jpg·webp, 5MB 이하. 올린 그림은 가운데를 정사각형으로 잘라 저장합니다. '작업 중' 그림을 따로 올리지 않으면 "
            "말풍선(…)을 붙여 자동으로 만듭니다. 저장 위치: "
            f"`teams/{team_id}/avatars/{node_id}.png`"
        )
        c1, c2 = st.columns(2)
        normal_now, working_now = avatar_path(node_id), avatar_path(node_id, working=True)
        if normal_now:
            c1.image(str(normal_now), width=72, caption="현재 · 평소")
        if working_now:
            c2.image(str(working_now), width=72, caption="현재 · 작업 중")
        exts = settings.get("avatar", {}).get("allowed_ext", ["png", "jpg", "jpeg", "webp"])
        normal = st.file_uploader("평소 이미지", type=exts, key=f"up_{team_id}_{node_id}")
        working = st.file_uploader("작업 중 이미지 (선택)", type=exts, key=f"upw_{team_id}_{node_id}")
        b1, b2 = st.columns(2)
        if b1.button("💾 저장", key=f"save_av_{team_id}_{node_id}", disabled=normal is None, use_container_width=True):
            try:
                save_avatar(team_id, node_id, normal.getvalue(), working.getvalue() if working else None)
            except AvatarError as e:
                st.warning(str(e))
            else:
                st.rerun()
        if b2.button("↩ 기본 이미지로", key=f"reset_av_{team_id}_{node_id}",
                     disabled=not has_custom_avatar(team_id, node_id), use_container_width=True):
            reset_avatar(team_id, node_id)
            st.rerun()


def render_route_tab(run: dict) -> None:
    """🗺 여행 지도 탭 — 일차 선택 · 움직이는 경로 지도 · 방문지 표 (core/route.py)."""
    if not run["done"]:
        st.caption("최종 계획서가 나오면 지도를 만듭니다.")
        return
    if run.get("post_busy"):
        st.info("🗺 지도 만드는 중… 방문지를 뽑고 OpenStreetMap 에서 좌표를 확인하고 있습니다 (약 30초~1분).")
        return
    route: RouteResult | None = run.get("route")
    if route is None:
        if run.get("route_error"):
            st.warning(f"지도를 만들지 못했습니다: {run['route_error'][:200]}")
        final = final_answer(run)
        if final and st.button("🗺 지도 만들기", key=f"mkmap_{run['run_id']}"):
            with st.spinner("지도 만드는 중…"):
                run["route"] = build_route(final.content, models, cache_path=DATA_DIR / "geocode_cache.json")
            st.rerun(scope="fragment")
        return

    days = sorted({s.day for s in route.itinerary.stops})
    day = st.radio("보기", [0, *days], format_func=lambda d: "전체 일정" if d == 0 else f"{d}일차",
                   horizontal=True, key=f"mapday_{run['run_id']}")
    st.caption("▶ 버튼을 누르면 시간 순서대로 이동 경로가 그려집니다 · 점선은 이동 방향으로 흐릅니다 · "
               "마커 숫자 = 일차-순서 · 마커를 누르면 메모가 보입니다")
    components.html(map_html(route, day or None, run.get("trip_start")), height=540)
    rows = [
        {"일차": s.day, "순서": s.order, "시간": s.time, "종류": f"{CATEGORIES[s.category][0]} {CATEGORIES[s.category][1]}",
         "장소": s.name, "메모": s.note, "위치": "확인됨" if route.verified.get(f"{s.day}-{s.order}") else "추정"}
        for s in route.itinerary.stops if not day or s.day == day
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)
    if route.notes:
        with st.expander("지도 생성 참고"):
            for note in route.notes:
                st.text(note[:300])


def render_agent_panel(node_id: str, run: dict | None) -> None:
    node = nodes[node_id]
    rank = ranks_by_id[node.rank]
    with st.container(border=True):
        c_img, c_info = st.columns([1, 6])
        working = bool(run and node_id in run["activity"])
        path = avatar_path(node_id, working=working)
        if path:
            c_img.image(str(path), width=72)
        else:
            c_img.markdown(f"<div style='font-size:48px'>{rank.emoji}</div>", unsafe_allow_html=True)
        cands = llm_factory.available_candidates(rank_models(node.rank, models, team), models)
        model = cands[0][0].label + (" (대안)" if cands and cands[0][1] else "") if cands else "모델 없음"
        state = run["statuses"].get(node_id) if run else None
        c_info.markdown(f"#### {name_of(node_id)}" + (f" · {state[1]}" if state else ""))
        c_info.caption(f"{rank.name} · {node.specialty or rank.role} · {model} · 📚 교훈 {storage.lesson_counts(team_id).get(node_id, 0)}개")

        if run:
            task_text = assigned_task(run, node_id)
            if task_text:
                st.markdown(f"**📥 맡은 업무** — {task_text}")
            if node_id in run["activity"]:
                st.markdown(f"<span class='ao-badge ao-b-working'>{html.escape(run['activity'][node_id])}{DOTS}</span>",
                            unsafe_allow_html=True)
            my_steps = [s for s in run["steps"] if s.node_id == node_id]
            drafting = run["drafting"]["node"] == node_id
            if my_steps or drafting:
                # 자세한 내용은 고정 높이 상자 안에서 스크롤
                with st.container(height=380):
                    for s in my_steps:
                        st.markdown(f"**{KIND_LABELS[s.kind]}**{route_of(s, run['root_id'])}")
                        render_step_body(s, expand_result=True)
                        st.divider()
                    if drafting:
                        st.markdown("**✍️ 지금 작성 중인 초안**")
                        st.markdown(run["drafting"]["text"] + "▌")
            elif node_id not in {n.id for n in subtree(org, run["root_id"]).walk()}:
                st.caption("이번 업무의 문의 대상 아래에 있지 않은 에이전트입니다.")
            elif not run["done"]:
                st.caption("아직 차례가 오지 않았습니다.")
            else:
                st.caption("이번 업무에서는 일을 맡지 않았습니다 (supervisor 가 필요한 전문가만 골라 맡김).")
        render_avatar_editor(node_id)


# ════════════════════════════════════════════════════════════════════
# 사람의 최종 결정
# ════════════════════════════════════════════════════════════════════
def adoptable_steps(run: dict) -> list[AgentAnswer]:
    """채택 후보 = 에이전트별 마지막 결과 (분배 단계 제외). 최종 결과(supervisor)를 맨 앞에."""
    last = {s.node_id: s for s in run["steps"] if s.kind != "plan"}
    final = final_answer(run)
    others = [s for nid, s in last.items() if not final or nid != final.node_id]
    return ([final] if final else []) + others


def check_options(run: dict) -> dict:
    """검산 옵션 = 데이터 툴이 계산한 값(현재가·보유 수량 등) + 업무 양식 옵션(stock_checks 등)."""
    options: dict = {}
    for tool in run.get("tools") or []:
        options.update(tool.harness_options)
    return {**options, **(run.get("harness_options") or {})}


def calc_report(run: dict, answer: str) -> str:
    """주식 업무의 코드 검산표 (점수·평균가·손익비·평단). 검산 대상이 없으면 빈 문자열."""
    options = check_options(run)
    return stock_calc.report(answer, options) if options.get("stock_checks") else ""


def save_adopted(run: dict, decision: HumanDecision) -> list[str]:
    """채택한 결과를 results/<팀>/ 에 파일로 저장 (여행 팀은 지도 .html 도). 실패해도 결정은 유지된다."""
    try:
        html_text = None
        if ROUTE_MAP in team.features:
            route = run.get("route")
            # 사람이 고쳤거나 최종안이 아닌 다른 안을 채택했으면, 채택한 내용으로 지도를 다시 만든다
            if decision.edited or decision.adopted_node_id != run["root_id"] or route is None:
                with st.spinner("채택한 계획으로 지도를 다시 만드는 중…"):
                    route = build_route(decision.final_answer, models, cache_path=DATA_DIR / "geocode_cache.json")
                run["route"] = route
            html_text = map_html(route, trip_start=run.get("trip_start")) if route else None
        names = {nid: name_of(nid) for nid in nodes}
        appendix = calc_report(run, decision.final_answer)  # 채택한(수정한) 내용 기준으로 다시 검산
        paths = export_adopted(run, decision, team_id, team.name, names, RESULTS_DIR, team.disclaimer, html_text, appendix)
        return [str(p) for p in paths]
    except Exception as e:  # noqa: BLE001
        run["notes"].append(f"결과 파일 저장 실패: {type(e).__name__}: {e}")
        return []


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
    if decision.action == "adopt":
        run["saved_files"] = save_adopted(run, decision)
    run.update(status="adopted" if decision.action == "adopt" else "rejected",
               decision=decision, lessons=lessons, notes=run["notes"] + notes)


def render_final_and_decision(run: dict, run_key: tuple) -> None:
    final = final_answer(run)
    if final:
        st.markdown(f"#### 📌 최종 결과 — {name_of(final.node_id)}")
        if is_concept:
            concept_ui.result_tabs(final.content, run)  # 최종본·검증 로그·GitHub 요약·적용 계획
        else:
            with st.container(height=420, border=True):
                st.markdown(final.content)
        report = calc_report(run, final.content)
        if report:
            # LLM 이 쓴 숫자를 코드로 다시 계산한 결과 — 판단 전에 꼭 확인할 값
            with st.expander("🧮 코드 검산 — 점수·평균 매수가·손익비·평단 (LLM 이 아닌 코드 계산)", expanded=True):
                st.markdown(report)
        if team.disclaimer:
            st.caption(f"ℹ️ {team.disclaimer}")

    st.markdown("#### 👤 최종 결정")
    if run["status"] == "awaiting_human":
        candidates = adoptable_steps(run)
        st.caption(
            "채택할 안을 고르고 필요하면 직접 수정하세요. 결정 내용은 참여한 모든 에이전트에게 교훈으로 피드백됩니다. "
            "결정하지 않고 '결과 버리기'를 누르거나 새 업무를 맡기면 이 결과는 저장되지 않습니다."
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
            if is_concept:  # 개념 편집 팀: 카탈로그에 들어갈 최종본과 적용 계획을 나눠서 고친다
                concept_entry, concept_plan = concept_ui.decision_fields(chosen.content, f"{run['run_id']}_{idx}")
                edited = chosen.content
            else:
                edited = st.text_area("최종 답변 (필요하면 직접 수정)", value=chosen.content, height=260,
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
        plan = None
        if adopt and is_concept:
            plan, problem = concept_ui.check_adoptable(concept_entry, concept_plan)
            if problem:
                st.error(problem)
                return
            edited = concept.replace_final(chosen.content, concept_entry)  # 채택 기록에도 실제 반영본이 남게
        if adopt or reject:
            decision = HumanDecision(
                action="adopt" if adopt else "reject",
                adopted_node_id=chosen.node_id if adopt else None,
                final_answer=edited.strip() if adopt else "",
                edited=bool(adopt and edited.strip() != chosen.content.strip()),
                score=score,
                comment=comment.strip(),
            )
            try:
                submit_decision(run, decision)
            except Exception as e:  # noqa: BLE001
                st.error(f"결정 반영 중 오류: {type(e).__name__}: {e}")
                return
            if adopt and is_concept:
                # 📚 채택한 최종본을 카탈로그 파일에 반영 (로컬만 — 커밋·푸시는 아래 버튼)
                try:
                    if concept_ctx is None:
                        raise ValueError("카탈로그 폴더를 찾지 못했습니다 (사이드바 경로 확인)")
                    concept_ui.apply_adopted(concept_ctx, run, concept_entry, plan)
                except (ValueError, KeyError) as e:
                    run["concept_error"] = f"카탈로그 반영 실패: {e} — 파일은 바뀌지 않았습니다."
            st.rerun()
        return

    decision: HumanDecision = run["decision"]
    if decision.action == "adopt":
        edited_mark = " · ✍️ 직접 수정" if decision.edited else ""
        st.success(f"✅ {name_of(decision.adopted_node_id)} 안 채택{edited_mark} · 만족도 {decision.score}/5 · 저장됨")
        if is_concept:
            concept_ui.adopted_section(concept_ctx, run)  # 📚 카탈로그 반영 결과 → 💾 커밋 → 🚀 푸시
        else:
            with st.container(height=300, border=True):
                st.markdown(decision.final_answer)
    else:
        st.warning(f"❌ 전체 반려 · 만족도 {decision.score}/5 · 저장됨")
    if decision.comment:
        st.caption(f"코멘트: {decision.comment}")
    for file in run.get("saved_files") or []:
        path = Path(file)
        c1, c2 = st.columns([4, 1])
        c1.caption(f"📁 저장됨: `{path.relative_to(BASE_DIR) if path.is_relative_to(BASE_DIR) else path}`")
        if path.exists():
            c2.download_button("⬇ 받기", data=path.read_bytes(), file_name=path.name, key=f"dl_now_{path.name}",
                               mime="text/html" if path.suffix == ".html" else "text/markdown", use_container_width=True)
    st.markdown("##### 📚 에이전트별 피드백 (다음 업무부터 프롬프트에 반영)")
    for lesson in run["lessons"]:
        icon = "👍" if lesson.kind == "praise" else "🔧"
        st.markdown(f"- **{name_of(lesson.agent_id)}** {icon} {lesson.lesson}")


# ════════════════════════════════════════════════════════════════════
# 본문 ① 문의 대상
# ════════════════════════════════════════════════════════════════════
st.title("🏢 Agent Office")
st.caption(f"현재 팀: **{team.name}** · 사용자는 맨 위 에이전트(supervisor)에게 일을 맡기고, supervisor 가 전문가들에게 나눠 줍니다.")

st.session_state.setdefault("runs", {})  # {(team_id, 문의 대상 id): run dict}

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
FOCUS_KEY, CLICK_KEY, PENDING_FOCUS_KEY = f"focus_{team_id}", f"last_click_{team_id}", f"pending_focus_{team_id}"
# 기본은 '아무 에이전트도 고르지 않음' — 조직도와 질문만 보이고, 칸을 누른 에이전트의 내용만 펼친다
NO_FOCUS = "__none__"
st.session_state.setdefault(FOCUS_KEY, NO_FOCUS)
# 위젯이 그려진 뒤에는 그 값을 바꿀 수 없으므로, 제출 시 정한 포커스는 다음 실행 맨 앞에서 반영한다
if PENDING_FOCUS_KEY in st.session_state:
    st.session_state[FOCUS_KEY] = st.session_state.pop(PENDING_FOCUS_KEY)
# 에이전트 패널의 상태 배지 스타일 (조직도는 iframe 안이라 CSS 를 따로 가진다)
st.markdown(
    "<style>.ao-badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:.85rem;font-weight:600}"
    ".ao-b-working{background:#FFF0B3;color:#7A5200;animation:ao-pulse 1.2s ease-in-out infinite}"
    ".ao-dots i{display:inline-block;width:5px;height:5px;margin-left:3px;border-radius:50%;background:currentColor;"
    "animation:ao-blink 1.4s infinite both}.ao-dots i:nth-child(2){animation-delay:.2s}.ao-dots i:nth-child(3){animation-delay:.4s}"
    "@keyframes ao-pulse{0%,100%{opacity:1}50%{opacity:.55}}@keyframes ao-blink{0%,80%,100%{opacity:.2}40%{opacity:1}}</style>",
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════════════
# 본문 ② 실시간 영역 — 작업 중에는 1초마다 이 부분만 다시 그린다 (st.fragment)
# ════════════════════════════════════════════════════════════════════
def render_live() -> None:
    run = st.session_state["runs"].get(run_key)

    # 조직도 — 칸을 누르면 그 에이전트로 포커스 이동
    clicked = click_detector(chart_html(run["statuses"] if run else {}, st.session_state[FOCUS_KEY], active_ids),
                             key=f"chart_{team_id}")
    if clicked and clicked.startswith(NODE_LINK) and clicked != st.session_state.get(CLICK_KEY):
        st.session_state[CLICK_KEY] = clicked
        st.session_state[FOCUS_KEY] = clicked[len(NODE_LINK):]
        st.rerun(scope="fragment")  # 조직도의 파란 테두리(선택 표시)도 바로 옮기도록 이 영역만 다시 그림

    # 선택지 이름은 고정(에이전트 이름)으로 둔다 — 진행 상태를 이름에 넣으면 1초마다 선택지가 바뀌어 선택이 풀릴 수 있다
    focus = st.selectbox("🔎 에이전트 보기 (조직도의 칸을 눌러도 됩니다)", [NO_FOCUS, *[n.id for n in org.walk()]],
                         format_func=lambda n: "— 고르지 않음 (질문만 보기) —" if n == NO_FOCUS else name_of(n), key=FOCUS_KEY)

    if run:
        elapsed = int(time.time() - run["started_at"])
        progress = f"⏳ 진행 중 ({elapsed}초)" if not run["done"] else "✅ 완료"
        done_steps = len([s for s in run["steps"] if s.kind != "plan"])
        # 질문 요약 — 개념 편집 팀은 질문에 초안·관련 항목이 길게 들어가므로 개념명과 판정만 보여 준다
        info = run.get("concept")
        if info:
            how = "✏️ 수정 — " + info["target"] if info.get("target") else "🆕 추가"
            question = f"**{info['title']}** · {how}"
        else:
            question = run["question"] if len(run["question"]) <= 300 else run["question"][:300] + "…"
        with st.container(border=True):
            st.markdown(f"**Q.** {question}")
            st.caption(f"{progress} · 완료된 단계 {done_steps}개 — 조직도에서 에이전트를 누르면 그 에이전트가 한 일과 지금 작성 중인 내용을 볼 수 있습니다.")

    if focus != NO_FOCUS:
        render_agent_panel(focus, run)

    if run:
        has_map = ROUTE_MAP in team.features
        # 전체 타임라인은 원할 때만 펼친다 (기본은 조직도·질문만 — 실시간으로 늘어나는 글이 화면을 채우지 않게)
        if not st.toggle("📜 전체 진행·근거 자료 보기", key=f"show_all_{team_id}"):
            return
        tabs = st.tabs(["📜 전체 진행", "🎯 브리프·근거 자료", *(["🗺 여행 지도"] if has_map else [])])
        tab_all, tab_ctx = tabs[0], tabs[1]
        if has_map:
            with tabs[2]:
                render_route_tab(run)
        with tab_all, st.container(height=560):
            st.markdown(f"**Q.** {run['question']}")
            for step in run["steps"]:
                render_step(step, run["root_id"])
            if run["drafting"]["node"]:
                nid = run["drafting"]["node"]
                with st.chat_message("assistant", avatar=avatar_for_chat(nid, working=True)):
                    st.markdown(f"**{name_of(nid)}** · ✍️ 작성 중\n\n{run['drafting']['text']}▌")
        with tab_ctx, st.container(height=560):
            render_brief(run.get("brief"))
            render_tools(run.get("tools") or [])
            render_web(run.get("web"))
            if run["notes"]:
                with st.expander("참고 기록"):
                    for note in run["notes"]:
                        st.text(note[:500])


def watch_run_state() -> None:
    """작업이 끝나거나(→ '최종 결정' 표시) 지도 만들기가 끝나면(→ 자동 갱신 멈춤) 화면 전체를 다시 그린다."""
    run = st.session_state["runs"].get(run_key)
    if run:
        state_key = f"live_state_{run['run_id']}"
        state_now = (run["done"], run.get("post_busy", False))
        previous_state = st.session_state.get(state_key)
        if previous_state != state_now:
            st.session_state[state_key] = state_now
            if previous_state is not None:
                st.rerun()


current = st.session_state["runs"].get(run_key)
running = bool(current and (not current["done"] or current.get("post_busy")))  # 지도 만드는 중에도 자동 갱신


def live_area() -> None:
    render_live()
    watch_run_state()  # 전체 진행을 접어 둔 상태(render_live 가 일찍 끝남)에서도 완료를 감지해야 한다


st.fragment(run_every=REFRESH_SEC if running else None)(live_area)()


# ════════════════════════════════════════════════════════════════════
# 본문 ③ 최종 결과·결정 (작업이 끝난 뒤)
# ════════════════════════════════════════════════════════════════════
if current and current["done"]:
    if current["error"]:
        st.error(current["error"])
    elif current["steps"]:
        render_final_and_decision(current, run_key)


# ════════════════════════════════════════════════════════════════════
# 본문 ④ 업무 맡기기 — 팀 업무 양식 또는 자유 질문
# ════════════════════════════════════════════════════════════════════
st.divider()
FREE_FORM = "free"
web_available = WEB_SEARCH_TOOL in team.tools
request: dict = {}  # 이번 제출에서 워크플로우로 넘길 값 (업무 양식이든 자유 질문이든 여기에 모은다)


def render_field(field: TaskField, key: str):
    """업무 양식의 입력칸 하나를 그린다. 값이 없으면 None 또는 빈 문자열."""
    label = field.label + (" *" if field.required else "")
    if field.type == "select":
        index = field.options.index(field.default) if field.default in field.options else 0
        return st.selectbox(label, field.options, index=index, key=key, help=field.help or None)
    if field.type == "number":
        value = float(field.default) if field.default is not None else None
        return st.number_input(label, value=value, min_value=0.0, step=1.0, key=key,
                               placeholder=field.placeholder or None, help=field.help or None)
    if field.type == "date":
        picked = st.date_input(label, value=None, key=key, format="YYYY-MM-DD", help=field.help or None)
        return picked.isoformat() if picked else None
    if field.type == "textarea":
        return st.text_area(label, value=str(field.default or ""), key=key, height=80,
                            placeholder=field.placeholder, help=field.help or None)
    return st.text_input(label, value=str(field.default or ""), key=key, placeholder=field.placeholder, help=field.help or None)


st.markdown(f"#### 💬 {name_of(target)} 에게 업무 맡기기")
# (안쪽에 '원하는 최종 상태' 접기 상자가 있어 바깥은 접기 상자가 아닌 테두리 상자로 둔다 — Streamlit 은 접기 상자 중첩 불가)
with st.container(border=True):
    task_ids = [t.id for t in team.tasks]
    mode = (
        st.selectbox(
            "📋 업무 선택",
            [*task_ids, FREE_FORM],
            format_func=lambda tid: "💬 자유 질문" if tid == FREE_FORM else next(t.title for t in team.tasks if t.id == tid),
            key=f"task_pick_{team_id}",
        )
        if team.tasks
        else FREE_FORM
    )
    task: TaskTemplate | None = next((t for t in team.tasks if t.id == mode), None)

    if task and is_concept:
        # ── 개념 편집 팀: ✍️ 작성(추가·수정 자동 판정) / 🗑 삭제 (concept_ui.py) ──
        st.caption(task.description)
        if concept_ctx is None:
            st.warning("사이드바의 '📁 개념 카탈로그' 폴더를 먼저 맞춰 주세요.")
        else:
            request = concept_ui.input_form(concept_ctx, task.id, team_id, web_available, running,
                                            settings.get("concept", {}).get("related_limit", 6)) or {}
    elif task:
        # ── 팀 업무 양식: 입력칸 → 질문·최종 상태 브리프 자동 생성 (core/tasks.py) ──
        st.caption(task.description)
        with st.form(f"task_form_{team_id}_{task.id}", clear_on_submit=False):
            values = {}
            cols = st.columns(2)
            for i, field in enumerate(task.fields):
                with cols[i % 2] if field.type != "textarea" else st.container():
                    values[field.key] = render_field(field, key=f"tf_{team_id}_{task.id}_{field.key}")
            extra = st.text_area("추가 요청 (선택)", key=f"tf_{team_id}_{task.id}__extra", height=68,
                                 placeholder="예: 배당도 함께 고려해줘 / 아이가 걷기 힘들어해")
            use_web = st.checkbox("🌐 웹 검색 사용 (최신 뉴스·자료)", value=web_available, disabled=not web_available)
            submitted = st.form_submit_button("맡기기", type="primary")
        if submitted:
            try:
                built = build_request(task, values, extra)
            except TaskInputError as e:
                st.warning(str(e))
            else:
                request = {
                    "question": built.question, "brief": built.brief, "auto_brief": False,
                    "use_web": bool(use_web and web_available), "tool_inputs": built.tool_inputs,
                    "extra_validators": built.validators, "harness_options": built.harness_options,
                }
    else:
        # ── 자유 질문 ──
        def apply_example() -> None:
            """예시를 고르면 질문·최종 상태 입력란을 채운다 (selectbox 의 on_change 콜백)."""
            ex = next((e for e in EXAMPLES if e["title"] == st.session_state["example_pick"]), None)
            if ex is None:
                return
            st.session_state["q_text"] = ex["question"]
            st.session_state["b_goal"] = ex["goal"]
            st.session_state["b_criteria"] = "\n".join(ex["success_criteria"])
            st.session_state["b_constraints"] = "\n".join(ex["constraints"])
            st.session_state["b_format"] = ex["output_format"]

        c1, c2 = st.columns([1, 1])
        c1.selectbox("💡 질문 예시 (고르면 아래가 채워집니다)", ["(직접 입력)", *[e["title"] for e in EXAMPLES]],
                     key="example_pick", on_change=apply_example)
        with c2.popover("📝 좋은 질문 쓰는 법"):
            st.markdown(
                """
- ❌ 과정 지시: "1) HBF 검색하고 2) 정리한 다음 3) 표로 만들어줘"
- ✅ 최종 상태: "**목표** 팀 회의에서 5분 안에 공유할 HBF 설명 자료 / **완료 기준** 정식 명칭·정의, HBM 과 비교 표, 사실마다 출처 / **제약** 비전공자용, A4 1장"
- 과정(누가 무엇을 조사할지)은 supervisor 가 스스로 나눕니다. 사람은 **무엇이 되면 끝인지**만 정하면 됩니다.
- 완료 기준은 검토자·비평가의 채점 기준이 됩니다 → 검증 가능한 문장으로 쓰세요.
- 결과는 그대로 믿지 말고, 최종 결정 전에 검토·비평 기록을 확인하세요.
"""
            )
        with st.form("ask_form", clear_on_submit=False):
            question = st.text_area("질문", placeholder="무엇을 원하는지 한두 문장으로", height=90, key="q_text")
            with st.expander("🎯 원하는 최종 상태 (선택 — 비워 두면 AI 가 질문에서 정리합니다)", expanded=False):
                goal = st.text_input("최종 목표 — 결과물과 용도", key="b_goal", placeholder="예: 팀 회의에서 공유할 1장짜리 설명 자료")
                criteria = st.text_area("완료 기준 — 한 줄에 하나", key="b_criteria", height=90,
                                        placeholder="예: 정식 명칭과 정의를 쓴다\n예: 사실마다 출처 번호를 단다")
                constraints = st.text_area("제약 — 한 줄에 하나", key="b_constraints", height=68, placeholder="예: 비전공자용, A4 1장")
                output_format = st.text_input("결과 형식", key="b_format", placeholder="예: 요약 → 비교 표 → 핵심 3개 → 출처")
            o1, o2 = st.columns(2)
            use_web = o1.checkbox("🌐 웹 검색 사용", value=web_available, disabled=not web_available)
            auto_brief = o2.checkbox("🤖 빈 항목은 AI 가 정리", value=True)
            submitted = st.form_submit_button("맡기기", type="primary")

        def lines(text: str) -> list[str]:
            return [x.strip() for x in text.splitlines() if x.strip()]

        if submitted and not question.strip():
            st.warning("질문을 입력하세요.")
        elif submitted:
            brief_input = DesiredState(
                goal=goal.strip(), context="", success_criteria=lines(criteria), constraints=lines(constraints),
                output_format=output_format.strip(), assumptions=[],
            )
            has_brief = not brief_input.is_empty() or brief_input.constraints or brief_input.output_format
            request = {
                "question": question.strip(), "brief": brief_input if has_brief else None, "auto_brief": auto_brief,
                "use_web": bool(use_web and web_available), "tool_inputs": {}, "extra_validators": [], "harness_options": {},
            }


# ════════════════════════════════════════════════════════════════════
# 제출 → 백그라운드 실행 시작 (core/live.py)
# ════════════════════════════════════════════════════════════════════
if request:
    if running:
        st.warning("진행 중인 업무가 있습니다. 끝난 뒤에 새 업무를 맡겨 주세요.")
    else:
        # 같은 대상에 대한 직전 결과가 결정 없이 남아 있으면 → 피드백 없는 불필요 기록이므로 삭제
        if current and current["status"] not in ("adopted", "rejected"):
            discard_run(current)
        run = new_run(uuid.uuid4().hex, request["question"], target)
        run["trip_start"] = request["tool_inputs"].get("start_date")  # 지도 재생 시각용
        run["harness_options"] = request["harness_options"]  # 화면 검산표용 (주식 업무의 stock_checks 등)
        run["concept"] = request.get("concept")  # 개념 편집 팀: 자동 판정(추가/수정)·대상 항목
        # 지도 기능이 있는 팀은 처음부터 '후처리 예정'으로 표시 — 워크플로우가 끝난 뒤 지도까지 다 만들 때까지 화면이 자동 갱신된다
        run["post_busy"] = ROUTE_MAP in team.features
        st.session_state["runs"][run_key] = run
        st.session_state[PENDING_FOCUS_KEY] = NO_FOCUS  # 새 업무는 조직도·질문만 보이는 상태로 시작 (에이전트를 누르면 펼침)
        storage.create_run(run["run_id"], team_id, target, run["question"])
        graph = make_graph()

        def events():
            return stream_office(
                graph, org, run["question"], target, thread_id=run["run_id"],
                use_web=request["use_web"], brief=request["brief"], auto_brief=request["auto_brief"],
                tool_inputs=request["tool_inputs"], extra_validators=request["extra_validators"],
                harness_options=request["harness_options"],
            )

        def on_finish(r: dict) -> None:
            try:
                if r["status"] == "error":
                    discard_run(r)  # 결정할 수 없는 실패 결과는 저장하지 않는다
                    return
                storage.save_steps(r["run_id"], r["steps"], r["status"])
                final = final_answer(r)
                if r["post_busy"] and final:
                    # 여행 팀: 최종 계획서로 경로 지도 만들기 (사람은 그동안 결과를 보고 결정할 수 있다)
                    r["activity"][r["root_id"]] = "🗺 지도 만드는 중"
                    try:
                        r["route"] = build_route(final.content, models, cache_path=DATA_DIR / "geocode_cache.json")
                    except Exception as e:  # noqa: BLE001 — 지도 실패가 결과를 막지 않게
                        r["route_error"] = f"{type(e).__name__}: {e}"
            finally:
                r["activity"].pop(r["root_id"], None)
                r["post_busy"] = False  # 화면 자동 갱신을 멈춰도 된다는 신호

        start_run(run, events, on_step=lambda r: storage.save_steps(r["run_id"], r["steps"], "reviewing"),
                  on_finish=on_finish)
        st.rerun()
