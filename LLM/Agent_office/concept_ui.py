"""개념 편집 팀 전용 화면 조각 — 메인 화면(app.py)이 이 팀을 고르면 불러 쓴다.

메인 화면의 조직도·에이전트 패널·이미지 편집·타임라인·최종 결정은 다른 팀과 똑같이 쓰고,
개념 편집 팀에만 필요한 부분만 여기서 그린다.
  - 사이드바: 카탈로그(GitHub jeong-yun/CS 의 Term/Every_day) 위치·git 상태
  - 업무 입력: ✍️ 작성(기존에 있으면 수정 · 없으면 추가 — 자동 판정) / 🗑 삭제
  - 최종 결과: 최종본·검증 로그·GitHub 요약·적용 계획 탭, 형식 검사
  - 채택 후: 📚 카탈로그 반영 → 💾 커밋 → 🚀 푸시
"""

import json
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from core import catalog as cat
from core import concept
from core.config import BASE_DIR

CONCEPT_FEATURE = "concept_catalog"  # team.yaml features 에 있으면 이 화면 조각을 쓴다
TASK_WRITE, TASK_DELETE = "write", "delete"


@dataclass
class CatalogContext:
    root: Path  # .../CS/Term/Every_day
    repo: Path | None  # git 저장소 최상위 (없으면 커밋·푸시 불가)
    subdir: str  # 저장소 안의 상대 경로 (git 명령 범위)
    catalog: cat.Catalog

    def label(self, name: str) -> str:
        entry = next((e for e in self.catalog.entries if e.name == name), None)
        return f"[{entry.category}] {name}" if entry else name


# ─── 사이드바 ───────────────────────────────────────────────────────────


def sidebar(settings: dict) -> CatalogContext | None:
    st.markdown("### 📁 개념 카탈로그")
    default_dir = (BASE_DIR / settings.get("concept", {}).get("catalog_dir", "../../../CS/Term/Every_day")).resolve()
    root = Path(st.text_input("Term/Every_day 폴더", value=str(default_dir), key="concept_root",
                              help="GitHub jeong-yun/CS 저장소를 clone 한 폴더 안의 Term/Every_day"))
    if not root.exists():
        st.error("폴더가 없습니다. `git clone https://github.com/jeong-yun/CS.git` 후 경로를 맞춰 주세요.")
        return None
    repo = cat.repo_root(root)
    subdir = str(root.relative_to(repo)).replace("\\", "/") if repo else ""
    catalog = cat.Catalog.load(root)
    counts = " · ".join(f"{k} {len(catalog.by_category(k))}" for k in cat.CATEGORY_FILES)
    st.caption(f"항목: {counts}")
    if repo is None:
        st.warning("git 저장소가 아닙니다 — 반영은 되지만 커밋·푸시는 할 수 없습니다.")
    else:
        status = cat.git_status(repo, subdir)
        st.caption("커밋 안 된 변경 없음" if not status else f"커밋 안 된 변경:\n```\n{status}\n```")
        if st.button("⬇️ 최신 받기 (git pull)", use_container_width=True, key="concept_pull"):
            ok, log = cat.git_pull(repo)
            (st.success if ok else st.error)(log[-500:] or "완료")
    return CatalogContext(root, repo, subdir, catalog)


# ─── 업무 입력 ──────────────────────────────────────────────────────────


def input_form(ctx: CatalogContext, task_id: str, team_id: str, web_available: bool, busy: bool,
               related_limit: int = 6) -> dict | None:
    """작성/삭제 입력. 작성은 '맡기기'를 누르면 워크플로우 요청(dict)을 돌려주고, 삭제는 여기서 바로 처리한다."""
    names = [e.name for e in ctx.catalog.entries]
    if task_id == TASK_DELETE:
        target_name = st.selectbox("삭제할 항목", names, format_func=ctx.label, key=f"cd_target_{team_id}")
        target = next(e for e in ctx.catalog.entries if e.name == target_name)
        with st.container(height=240, border=True):
            st.text(target.text[:6000])
        refs = cat.mentions(ctx.catalog, target)
        if refs:
            st.warning("다른 항목이 이 개념을 언급합니다 — 삭제 후 해당 문장을 확인하세요:\n"
                       + "\n".join(f"- [{e.category}] {e.name}" for e in refs[:15]))
        if st.button("🗑 승인 — old_term 으로 이관", type="primary", disabled=busy, key=f"cd_go_{team_id}"):
            changes = cat.apply_plan(ctx.catalog, concept.delete_plan(target), "")
            st.session_state["concept_delete_result"] = {"title": target.name, "changes": changes}
            st.rerun()
        result = st.session_state.get("concept_delete_result")
        if result:
            st.success(f"🗑 「{result['title']}」 을 old_term 으로 옮겼습니다.")
            render_changes(result["changes"], [])
            git_panel(ctx, f"del_{result['title']}", f"Term: {result['title']} 삭제")
        return None

    # ── ✍️ 작성: 추가/수정은 고르지 않는다 — 개념명으로 기존 항목을 찾아 자동 판정 ──
    name = st.text_input("개념명(한/영) *", key=f"cw_name_{team_id}",
                         placeholder="예: 그룹 상대적 정책 최적화(Group Relative Policy Optimization, GRPO)")
    draft = st.text_area("초안", height=220, key=f"cw_draft_{team_id}",
                         placeholder="정의·작동 원리·장단점·참고 자료 등 초안을 붙여 넣으세요 (비워도 개념명으로 작성)")
    target, sub_title, action = None, "", "add"
    if name.strip():
        matches = cat.find_existing(ctx.catalog, name)
        ADD = "__add__"

        def option_label(n: str) -> str:
            if n == ADD:
                return "🆕 새 항목으로 추가"
            m = next(m for m in matches if m.entry.name == n)
            return f"✏️ 수정 — {ctx.label(n)}" + (f" (7.변형 소절 '{m.detail[:40]}')" if m.kind == "하위" else "")

        if matches:
            m0 = matches[0]
            st.info(f"기존에 있는 개념입니다 → **수정**으로 진행합니다: {ctx.label(m0.entry.name)}"
                    + (f"  \n(다른 항목의 7.변형 소절 '{m0.detail}' 로 편입되어 있어 그 부모 항목을 수정합니다)" if m0.kind == "하위" else ""))
        else:
            st.success("카탈로그에 없는 개념입니다 → **추가**로 진행합니다. (검토 중 팀이 기존 개념의 하위라고 판단하면 편입으로 바꿉니다)")
        picked = st.selectbox("판정 (필요하면 바꾸기)", [m.entry.name for m in matches] + [ADD], format_func=option_label,
                              key=f"cw_pick_{team_id}_{hash(name)}")
        if picked != ADD:
            hit = next(m for m in matches if m.entry.name == picked)
            target, sub_title, action = hit.entry, (hit.detail if hit.kind == "하위" else ""), "update"
    suggested = [e.name for e, _ in cat.find_related(ctx.catalog, name, draft, related_limit) if not target or e.name != target.name]
    related_names = st.multiselect("함께 비교할 기존 항목 (자동 추천 — 추가·삭제 가능)", names, default=suggested,
                                   format_func=ctx.label, key=f"cw_related_{team_id}_{hash(name)}")
    use_web = st.checkbox("🌐 웹 검색으로 원전 확인", value=web_available, disabled=not web_available, key=f"cw_web_{team_id}")
    if not st.button("맡기기", type="primary", disabled=busy or not name.strip(), key=f"cw_go_{team_id}"):
        return None
    related = [e for e in ctx.catalog.entries if e.name in related_names]
    question, brief = concept.build_concept_request(action, name.strip(), draft, related, target, sub_title=sub_title)
    return {
        "question": question, "brief": brief, "auto_brief": False, "use_web": bool(use_web and web_available),
        "tool_inputs": {}, "extra_validators": [], "harness_options": {},
        "concept": {"action": action, "title": name.strip(), "target": target.name if target else ""},
    }


# ─── 최종 결과 · 결정 · 반영 ─────────────────────────────────────────────


def result_tabs(content: str, run: dict) -> None:
    """최종 결과를 탭으로 — 최종본(카탈로그에 들어갈 글) / 검증 로그 / GitHub 요약 / 적용 계획."""
    res = concept.parse_result(content)
    info = run.get("concept") or {}
    plan = res.plan or {}
    decided = {"add": "🆕 추가", "update": "✏️ 수정", "delete": "🗑 삭제"}.get(plan.get("action"), "판정 없음")
    target = plan.get("target") or info.get("target")
    st.caption(f"편집 팀 판정: **{decided}**" + (f" — 대상 「{target}」" if target and plan.get("action") == "update" else "")
               + (f" · 통합으로 빠질 항목: {', '.join(plan['absorb'])}" if plan.get("absorb") else ""))
    problems = concept.check_answer(content)
    if problems:
        st.warning("형식 검사에서 남은 문제 (채택 전 아래에서 직접 고칠 수 있습니다):\n" + "\n".join(f"- {p}" for p in problems))
    tabs = st.tabs(["📄 최종본", "🧠 검증 로그", "🐙 GitHub 요약", "🧾 적용 계획"])
    with tabs[0], st.container(height=420):
        st.text(res.final_entry or "(최종본 없음 — <final> 블록이 없습니다)")
    with tabs[1], st.container(height=420):
        if res.change_notes:  # 최종본에서 떼어 낸 🔧수정·➕추가 표시와 이유 — 확인용 (카탈로그에는 들어가지 않음)
            notes = "\n".join(f"- {n}" for n in res.change_notes)
            st.markdown(f"**🔧 초안 대비 변경 요약** (최종본에는 넣지 않음)\n{notes}")
        st.markdown(res.thought or "(검증 로그 없음)")
    with tabs[2], st.container(height=420):
        st.markdown(res.summary_md or "(요약 없음)")
    with tabs[3], st.container(height=420):
        st.json(plan)
        for e in plan.get("edits") or []:
            st.caption(f"✏️ 다른 항목 수정 — {e.get('entry')}: {e.get('reason', '')}")


def decision_fields(content: str, key: str) -> tuple[str, str]:
    """채택 폼 안의 입력칸 — 최종본과 적용 계획을 고칠 수 있게 나눠 보여 준다."""
    res = concept.parse_result(content)
    entry = st.text_area("최종본 (카탈로그에 들어갈 글 — 필요하면 수정)", value=res.final_entry, height=300, key=f"ce_{key}")
    plan = st.text_area("적용 계획 JSON (필요하면 수정)", value=json.dumps(res.plan, ensure_ascii=False, indent=2),
                        height=180, key=f"cp_{key}")
    return entry, plan


def check_adoptable(entry: str, plan_text: str) -> tuple[dict | None, str]:
    """채택 전에 반영 가능한지 확인. (계획, 오류 메시지)"""
    try:
        plan = json.loads(plan_text)
    except json.JSONDecodeError as e:
        return None, f"적용 계획 JSON 오류: {e}"
    if not entry.strip() and plan.get("action") != "delete":
        return None, "최종본이 비어 있습니다 — 최종 통합안(<final> 이 있는 안)을 고르세요."
    return plan, ""


def apply_adopted(ctx: CatalogContext, run: dict, entry: str, plan: dict) -> None:
    """채택한 결과를 카탈로그 파일에 반영한다 (로컬만 — 커밋·푸시는 버튼으로)."""
    info = run.get("concept") or {}
    fresh = cat.Catalog.load(ctx.root)
    # 팀이 '수정'으로 판정했는데 대상 이름을 빠뜨렸거나 못 찾는 이름이면, 화면에서 판정한 기존 항목을 쓴다
    if plan.get("action") == "update" and info.get("target") and not fresh.find(str(plan.get("target", ""))):
        plan["target"] = info["target"]
    run["concept_changes"] = cat.apply_plan(fresh, plan, entry)
    run["concept_edit_logs"] = concept.apply_edits(cat.Catalog.load(ctx.root), plan.get("edits") or [])
    run["concept_commit"] = f"Term: {plan.get('title') or info.get('title', '')} {concept.ACTIONS.get(plan.get('action'), '정리')}"


def render_changes(changes: list, edit_logs: list[str]) -> None:
    for ch in changes:
        st.markdown(f"- `{ch.path}` **{ch.action}** — {ch.detail}")
    for log in edit_logs:
        st.markdown(f"- {log}")


def git_panel(ctx: CatalogContext, key: str, message: str) -> None:
    """변경 내용 확인 → 💾 커밋 → 🚀 푸시 (버튼을 눌러야만 실행)."""
    if ctx.repo is None:
        st.caption("git 저장소가 아니라 커밋·푸시는 할 수 없습니다.")
        return
    with st.expander("변경 내용 (git diff)"):
        st.code(cat.git_diff(ctx.repo, ctx.subdir)[:30000] or "(차이 없음 — 이미 커밋했거나 새 파일만 있음)", language="diff")
    msg = st.text_input("커밋 메시지", value=message, key=f"msg_{key}")
    pending = cat.git_status(ctx.repo, ctx.subdir)
    st.caption("📝 아직 커밋하지 않은 변경이 있습니다 — '🚀 커밋 후 푸시'를 누르면 커밋과 GitHub 업로드를 한 번에 합니다."
               if pending else "커밋하지 않은 변경 없음")
    c1, c2 = st.columns(2)
    if c1.button("💾 커밋만", use_container_width=True, key=f"commit_{key}", disabled=not pending):
        ok, log = cat.git_commit(ctx.repo, ctx.subdir, msg)
        (st.success if ok else st.error)(log[-800:])
    if c2.button("🚀 커밋 후 GitHub 에 푸시", type="primary", use_container_width=True, key=f"push_{key}"):
        if pending:  # 커밋 안 된 변경이 있으면 먼저 커밋 — 커밋 없이 푸시하면 아무것도 올라가지 않는다
            ok, log = cat.git_commit(ctx.repo, ctx.subdir, msg)
            if not ok:
                st.error("커밋 실패: " + log[-800:])
                return
        ok, log = cat.git_push(ctx.repo)
        if not ok:
            st.error("푸시 실패: " + log[-800:])
        elif "up-to-date" in log.lower() or "up to date" in log.lower():
            st.warning("올릴 새 커밋이 없습니다 (GitHub 와 이미 같음).")
        else:
            st.success("GitHub 에 올렸습니다.\n\n" + log[-500:])


def adopted_section(ctx: CatalogContext | None, run: dict) -> None:
    """채택 후 — 카탈로그 반영 결과 + 커밋·푸시."""
    st.markdown("##### 📚 카탈로그 반영")
    if run.get("concept_error"):
        st.error(run["concept_error"])
    if ctx is None:
        st.warning("카탈로그 폴더를 찾지 못해 반영하지 못했습니다 (사이드바의 경로 확인).")
        return
    render_changes(run.get("concept_changes") or [], run.get("concept_edit_logs") or [])
    if run.get("concept_changes"):
        git_panel(ctx, run["run_id"], run.get("concept_commit") or "Term: 개념 정리")
