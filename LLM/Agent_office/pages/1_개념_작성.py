"""📚 개념 작성 — IT 개념 카탈로그(GitHub jeong-yun/CS 의 Term/Every_day)에 항목을 추가·수정·삭제한다.

흐름
  ① 작업 입력   ✍️ 작성(개념명·초안) 또는 🗑 삭제 — 추가/수정은 고르지 않는다: 개념명으로 기존 항목을 찾아
                (독립 항목이거나 다른 항목의 7.변형 소절에 있으면) 수정, 없으면 추가. 관련 기존 항목도 코드로 찾아 함께 보낸다
  ② 팀 검토     개념 편집 팀(편집장 → 수석에디터 → 팩트체크·계보분석)이 'IT 개념 프롬프트' 규칙으로 검증·작성
                (형식은 하네스 concept_entry 가 코드로 검사: <thought>·<final>·개조식·절대 표현·참고문헌 형식·적용 계획)
  ③ 사람 승인   최종본·적용 계획을 확인(필요하면 수정) → 승인하면 카탈로그 파일에 반영
                - 통합(병합·편입)으로 빠지는 기존 항목은 old_term.txt 로 이관, 0. Summary.md 요약 갱신, 기초 용어 추가
  ④ 커밋·푸시   변경 내용(diff)을 보고 커밋 → 'GitHub 에 푸시' 버튼을 눌러야만 올라간다
삭제는 LLM 없이 바로 미리보기 → 승인 → old_term 이관.
"""

import json
import uuid
from pathlib import Path

import streamlit as st

from core import catalog as cat
from core import concept, llm_factory
from core.config import BASE_DIR, load_env, load_models, load_ranks, load_settings, load_team, load_verification, team_org
from core.live import KIND_LABELS, new_run, start_run
from core.schemas import HumanDecision
from core.storage import Storage
from core.workflow import build_office_graph, make_checkpointer, resume_with_decision, stream_office

load_env()
st.set_page_config(page_title="개념 작성", page_icon="📚", layout="wide")

settings = load_settings()
ranks, verification_cfg = load_ranks(), load_verification()
team = load_team("concept")
org = team_org(team, ranks)
names = {n.id: n.name or n.id for n in org.walk()}
DATA_DIR = BASE_DIR / settings.get("storage", {}).get("data_dir", "data")
concept_cfg = settings.get("concept", {})
RUN_KEY = "concept_run"


@st.cache_resource
def get_checkpointer():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return make_checkpointer(DATA_DIR / "concept_checkpoints.db")


@st.cache_resource
def get_storage() -> Storage:
    return Storage(DATA_DIR / settings.get("storage", {}).get("db_file", "runs.db"))


storage = get_storage()


def make_graph(models):
    return build_office_graph(org, ranks, team, models, settings, verification_cfg, storage=storage, checkpointer=get_checkpointer())


# ════════════════════════════════════════════════════════════════════
# 사이드바: 카탈로그 위치 · git 상태 · 모델
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📁 카탈로그 위치")
    default_dir = (BASE_DIR / concept_cfg.get("catalog_dir", "../../../CS/Term/Every_day")).resolve()
    root = Path(st.text_input("Term/Every_day 폴더", value=str(default_dir), help="GitHub jeong-yun/CS 저장소를 clone 한 폴더 안의 Term/Every_day"))
    repo = cat.repo_root(root) if root.exists() else None
    subdir = str(root.relative_to(repo)).replace("\\", "/") if repo else ""
    if not root.exists():
        st.error("폴더가 없습니다. 먼저 `git clone https://github.com/jeong-yun/CS.git` 를 하세요.")
    elif repo is None:
        st.warning("git 저장소가 아닙니다 — 반영은 되지만 커밋·푸시는 할 수 없습니다.")
    else:
        st.caption(f"저장소: `{repo}`")
        status = cat.git_status(repo, subdir)
        st.caption("변경 없음" if not status else f"커밋 안 된 변경:\n```\n{status}\n```")
        if st.button("⬇️ 최신 받기 (git pull)", use_container_width=True):
            ok, log = cat.git_pull(repo)
            (st.success if ok else st.error)(log[-500:] or "완료")

    st.markdown("### 🤖 모델")
    models = load_models()
    if st.toggle("🆓 무료 모델만 사용", key="free_only", help="FreeLLMAPI(무료 티어) 모델만 사용합니다."):
        models = llm_factory.free_only(models)
    for provider, state in llm_factory.provider_status(models).items():
        st.caption(f"{'✅' if state == 'ok' else '⚪'} {provider} — {state}")


# ════════════════════════════════════════════════════════════════════
# 본문
# ════════════════════════════════════════════════════════════════════
st.title("📚 개념 작성")
st.caption(
    "초안을 넣으면 개념 편집 팀이 원전 기준으로 검증하고 기존 개념과 비교해 추가·수정·통합을 판정합니다. "
    "사람이 승인하면 카탈로그 파일에 반영하고, 커밋·푸시는 버튼을 눌렀을 때만 합니다."
)

if not root.exists():
    st.stop()
catalog = cat.Catalog.load(root)
counts = {k: len(catalog.by_category(k)) for k in cat.CATEGORY_FILES}
st.caption(f"현재 카탈로그: Cloud {counts['Cloud']} · AI {counts['AI']} · Security {counts['Security']} 항목")
label_of = {e.name: f"[{e.category}] {e.name}" for e in catalog.entries}

run: dict | None = st.session_state.get(RUN_KEY)
busy = run is not None and not run.get("done")

# ── ① 작업 입력 ───────────────────────────────────────────────────────
st.markdown("#### ① 작업 입력")
with st.container(border=True):
    # 추가·수정은 고르지 않는다 — 개념명으로 기존 항목을 찾아 있으면 '수정', 없으면 '추가' (core.catalog.find_existing)
    mode = st.radio("작업", ["write", "delete"], horizontal=True, key="c_mode",
                    format_func=lambda m: "✍️ 작성 (기존에 있으면 수정 · 없으면 추가)" if m == "write" else "🗑 삭제")
    target, sub_title, action = None, "", "add"

    if mode == "delete":
        target_name = st.selectbox("삭제할 항목", [e.name for e in catalog.entries], format_func=lambda n: label_of[n], key="c_target")
        target = next(e for e in catalog.entries if e.name == target_name)
        st.markdown("**삭제 미리보기** — 승인하면 이 항목을 `old_term.txt` 로 옮기고 요약에서 뺍니다.")
        with st.expander(target.header, expanded=False):
            st.text(target.body[:4000])
        refs = cat.mentions(catalog, target)
        if refs:
            st.warning("다른 항목이 이 개념을 언급합니다 — 삭제 후 해당 문장을 직접 확인하세요:\n" + "\n".join(f"- [{e.category}] {e.name}" for e in refs[:15]))
        if st.button("🗑 승인 — old_term 으로 이관", type="primary", disabled=busy):
            changes = cat.apply_plan(catalog, concept.delete_plan(target), "")
            st.session_state[RUN_KEY] = {"run_id": uuid.uuid4().hex, "done": True, "status": "adopted", "applied": changes,
                                         "title": target.name, "action": "delete", "steps": [], "statuses": {}, "drafting": {}}
            st.rerun()
    else:
        name = st.text_input("개념명(한/영)", placeholder="예: 그룹 상대적 정책 최적화(Group Relative Policy Optimization, GRPO)", key="c_name")
        draft = st.text_area("초안", height=260, key="c_draft", placeholder="정의·작동 원리·장단점·참고 자료 등 초안을 붙여 넣으세요 (비워도 개념명으로 작성)")
        if name.strip():
            matches = cat.find_existing(catalog, name)
            ADD = "__add__"

            def option_label(n: str) -> str:
                if n == ADD:
                    return "🆕 새 항목으로 추가"
                m = next(m for m in matches if m.entry.name == n)
                return f"✏️ 수정 — {label_of[n]}" + (f" (7.변형 소절 '{m.detail[:40]}')" if m.kind == "하위" else "")

            if matches:
                m0 = matches[0]
                st.info(f"기존에 있는 개념입니다 → **수정**으로 진행합니다: {label_of[m0.entry.name]}"
                        + (f"  \n(다른 항목의 7.변형 소절 '{m0.detail}' 로 편입되어 있어 그 부모 항목을 수정합니다)" if m0.kind == "하위" else ""))
            else:
                st.success("카탈로그에 없는 개념입니다 → **추가**로 진행합니다. (검토 중 팀이 기존 개념의 하위라고 판단하면 편입으로 바꿉니다)")
            picked = st.selectbox("판정 (필요하면 바꾸기)", [m.entry.name for m in matches] + [ADD], format_func=option_label,
                                  key=f"c_pick_{hash(name)}")
            if picked != ADD:
                hit = next(m for m in matches if m.entry.name == picked)
                target, sub_title, action = hit.entry, (hit.detail if hit.kind == "하위" else ""), "update"
        suggested = [e.name for e, _ in cat.find_related(catalog, name, draft, concept_cfg.get("related_limit", 6)) if not target or e.name != target.name]
        related_names = st.multiselect("함께 비교할 기존 항목 (자동 추천 — 추가·삭제 가능)", [e.name for e in catalog.entries],
                                       default=suggested, format_func=lambda n: label_of[n], key=f"c_related_{hash(name)}")
        use_web = st.checkbox("🌐 웹 검색으로 원전 확인", value=True, key="c_web")
        if st.button("🔍 검토 시작", type="primary", disabled=busy or not name.strip()):
            related = [e for e in catalog.entries if e.name in related_names]
            question, brief = concept.build_concept_request(action, name.strip(), draft, related, target, sub_title=sub_title)
            new = new_run(uuid.uuid4().hex, question, org.id)
            new.update(action=action, title=name.strip(), target=target.name if target else "", catalog_root=str(root))
            graph = make_graph(models)
            storage.create_run(new["run_id"], team.id, org.id, question)

            def events():
                return stream_office(graph, org, question, org.id, thread_id=new["run_id"], use_web=use_web, brief=brief, auto_brief=False)

            def on_finish(r: dict) -> None:
                if r["status"] != "error":
                    storage.save_steps(r["run_id"], r["steps"], r["status"])

            st.session_state[RUN_KEY] = new
            start_run(new, events, on_finish=on_finish)
            st.rerun()


# ── ② 팀 검토 진행 ────────────────────────────────────────────────────
def final_step(r: dict):
    mine = [s for s in r["steps"] if s.node_id == org.id and s.kind != "plan"]
    return mine[-1] if mine else None


@st.fragment(run_every=1.0 if busy else None)
def render_progress() -> None:
    r = st.session_state.get(RUN_KEY)
    if not r or not r.get("steps") and not r.get("statuses"):
        return
    st.markdown("#### ② 팀 검토")
    with st.container(border=True):
        cols = st.columns(len(names))
        for col, (nid, label) in zip(cols, names.items()):
            state = r["statuses"].get(nid, ("idle", "대기"))
            col.markdown(f"**{label}**  \n{state[1] if state[0] != 'idle' else '⚪ 대기'}")
        drafting = r.get("drafting") or {}
        if drafting.get("node"):
            st.caption(f"✍️ {names.get(drafting['node'], drafting['node'])} 작성 중…")
            st.code(drafting["text"][-1500:], language=None)
        for s in r["steps"]:
            verdict = f" · {s.review.score}/10" if s.review else ""
            with st.expander(f"{names.get(s.node_id, s.node_id)} · {KIND_LABELS.get(s.kind, s.kind)}{verdict} · {s.model.label} · {s.elapsed_sec}초"):
                if s.review:
                    st.caption(s.review.analysis)
                st.markdown(s.content[:20000])
                for h in s.harness:
                    st.markdown(f"- 🧪 `{h.validator}` {'✅' if h.passed else '❌'} {h.log[:600]}")
                for log in s.verification_log:
                    st.caption(f"🔁 {log}")
        if r.get("error"):
            st.error(r["error"])
    if busy and r.get("done"):
        st.rerun()  # 끝나면 전체를 다시 그려 결정 화면을 보여 준다


render_progress()

run = st.session_state.get(RUN_KEY)
if not run or not run.get("done"):
    st.stop()

# ── ③ 사람 승인 ────────────────────────────────────────────────────────
if run.get("status") == "awaiting_human":
    final = final_step(run)
    if final is None:
        st.error("최종 결과가 없습니다.")
        st.stop()
    result = concept.parse_result(final.content)
    problems = concept.check_answer(final.content)
    st.markdown("#### ③ 사람 승인")
    if problems:
        st.warning("형식 검사에서 남은 문제가 있습니다 (승인 전 직접 고칠 수 있습니다):\n" + "\n".join(f"- {p}" for p in problems))
    tabs = st.tabs(["📄 최종본", "🧠 검증 로그", "🐙 GitHub 요약", "🧾 적용 계획"])
    with tabs[0]:
        st.text(result.final_entry or "(최종본 없음)")
    with tabs[1]:
        st.markdown(result.thought or "(검증 로그 없음)")
    with tabs[2]:
        st.markdown(result.summary_md or "(요약 없음)")
    with tabs[3]:
        st.json(result.plan or {})
        plan = result.plan or {}
        if plan.get("absorb"):
            st.info("통합으로 빠질 기존 항목 (old_term 으로 이관): " + ", ".join(plan["absorb"]))
        for e in plan.get("edits") or []:
            st.caption(f"✏️ {e.get('entry')}: {e.get('reason', '')}")

    with st.form(f"decide_{run['run_id']}"):
        edited_entry = st.text_area("최종본 (필요하면 수정)", value=result.final_entry, height=360)
        edited_plan = st.text_area("적용 계획 JSON (필요하면 수정)", value=json.dumps(result.plan, ensure_ascii=False, indent=2), height=220)
        score = st.slider("만족도", 1, 5, 4)
        comment = st.text_input("코멘트 (편집 팀에게 줄 피드백, 선택)")
        c1, c2 = st.columns(2)
        approve = c1.form_submit_button("✅ 승인 — 카탈로그에 반영", type="primary", use_container_width=True)
        reject = c2.form_submit_button("❌ 반려", use_container_width=True)

    if approve or reject:
        try:
            plan = json.loads(edited_plan) if approve else {}
        except json.JSONDecodeError as e:
            st.error(f"적용 계획 JSON 오류: {e}")
            st.stop()
        decision = HumanDecision(
            action="adopt" if approve else "reject",
            adopted_node_id=final.node_id if approve else None,
            final_answer=final.content.replace(result.final_entry, edited_entry) if approve else "",
            edited=approve and edited_entry.strip() != result.final_entry.strip(),
            score=score, comment=comment,
        )
        with st.spinner("결정을 반영하고 편집 팀에 피드백을 남기는 중…"):
            for kind, payload in resume_with_decision(make_graph(models), org, decision, run["run_id"]):
                if kind == "lessons":
                    run["lessons"] = payload
            storage.save_decision(run["run_id"], decision)
        if approve:
            # 팀이 '수정'으로 판정했는데 대상 이름을 빠뜨렸거나 못 찾는 이름이면, 코드가 찾은 기존 항목을 쓴다
            if plan.get("action") == "update" and run.get("target") and not cat.Catalog.load(root).find(str(plan.get("target", ""))):
                plan["target"] = run["target"]
            run["action"] = plan.get("action", run.get("action"))  # 커밋 메시지의 추가/수정 표시
            try:
                fresh = cat.Catalog.load(root)
                run["applied"] = cat.apply_plan(fresh, plan, edited_entry)
                run["edit_logs"] = concept.apply_edits(cat.Catalog.load(root), plan.get("edits") or [])
                run["status"] = "adopted"
            except (ValueError, KeyError) as e:
                st.error(f"반영 실패: {e} — 적용 계획을 고쳐 다시 승인하세요.")
                st.stop()
        else:
            run["status"] = "rejected"
        st.rerun()

elif run.get("status") == "rejected":
    st.info("반려했습니다. 코멘트는 편집 팀의 다음 작업에 교훈으로 반영됩니다.")

elif run.get("status") == "error":
    st.error(run.get("error") or "실패")

# ── ④ 반영 결과 · 커밋 · 푸시 ──────────────────────────────────────────
if run.get("status") == "adopted":
    st.markdown("#### ④ 반영 결과")
    for ch in run.get("applied") or []:
        st.markdown(f"- `{ch.path}` **{ch.action}** — {ch.detail}")
    for log in run.get("edit_logs") or []:
        st.markdown(f"- {log}")
    if repo:
        diff = cat.git_diff(repo, subdir)
        with st.expander("변경 내용 (git diff)", expanded=False):
            st.code(diff[:30000] or "(차이 없음 — 이미 커밋했거나 새 파일만 있음)", language="diff")
        verb = {"add": "추가", "update": "수정", "delete": "삭제"}.get(run.get("action"), "정리")
        message = st.text_input("커밋 메시지", value=f"Term: {run.get('title', '')} {verb}", key=f"msg_{run['run_id']}")
        c1, c2 = st.columns(2)
        if c1.button("💾 커밋", use_container_width=True):
            ok, log = cat.git_commit(repo, subdir, message)
            (st.success if ok else st.error)(log[-800:])
        if c2.button("🚀 GitHub 에 푸시", type="primary", use_container_width=True):
            ok, log = cat.git_push(repo)
            (st.success if ok else st.error)(log[-800:] or "푸시 완료")
    if st.button("🆕 새 작업"):
        st.session_state.pop(RUN_KEY, None)
        st.rerun()
