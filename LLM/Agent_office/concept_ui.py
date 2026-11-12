"""개념 편집 팀 전용 화면 조각 — 메인 화면(app.py)이 이 팀을 고르면 불러 쓴다.

메인 화면의 조직도·에이전트 패널·이미지 편집·타임라인·최종 결정은 다른 팀과 똑같이 쓰고,
개념 편집 팀에만 필요한 부분만 여기서 그린다.
  - 사이드바: 카탈로그(GitHub jeong-yun/CS 의 Term/Every_day) 위치·git 상태
  - 업무 입력: ✍️ 작성(기존에 있으면 수정 · 없으면 추가 — 자동 판정) / 🗑 삭제
  - 최종 결과: 최종본·검증 로그·GitHub 요약·적용 계획 탭, 형식 검사
    · 기존 항목 수정이면 맨 앞에 '✏️ 변경 제안' 탭 — 변경마다 고치기 전 → 고친 뒤와 근거를 보고 반영할지 고른다 (원본은 그대로,
      고른 변경만 코드가 끼워 넣는다. 근거 글은 화면에만 — 카탈로그·커밋 메시지에는 들어가지 않는다)
  - 📜 적용된 프롬프트: 입력칸 아래·결과 위 — 스킬(IT 개념 프롬프트) 판·전문, 보낸 업무 브리프, 에이전트별 시스템 프롬프트
  - 채택 후: 📚 카탈로그 반영 → 💾 커밋 → 🚀 푸시 (GitHub 에 이 PC 에 없는 커밋이 있으면 먼저 자동 rebase, 충돌하면 되돌리고 중단)
"""

import json
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

import prompt_ui
from core import catalog as cat
from core import concept
from core.config import BASE_DIR, load_team
from core.memo import SigCache, file_sig

CONCEPT_FEATURE = "concept_catalog"  # team.yaml features 에 있으면 이 화면 조각을 쓴다
TASK_WRITE, TASK_DELETE = "write", "delete"

# ─── 캐시: 화면을 다시 그릴 때마다 git 명령(한 번에 0.3~1초)·카탈로그 파싱을 되풀이하지 않게 ──────────────
# 카탈로그 파일·.git 의 HEAD·index 등이 바뀌면 바로 다시 읽고, 바뀐 게 없어도 GIT_TTL_SEC 마다 새로 읽는다
# (터미널에서 푸시한 것·'n시간 전' 표시). 커밋·푸시·pull 버튼을 누르면 즉시 비운다.
GIT_TTL_SEC = 30
_cache = SigCache(maxsize=64)
GIT_FILES = ("HEAD", "index", "FETCH_HEAD", "ORIG_HEAD", "packed-refs", "logs/HEAD")
# GitHub 에서 받아 오기(fetch)는 네트워크라 더 드물게 — FETCH_TTL_SEC 마다 한 번. 받아 오면 FETCH_HEAD 가 바뀌어
# 위의 git 상태 캐시도 자동으로 다시 읽는다. 커밋·푸시 버튼(refresh_git)으로는 비우지 않는다 (푸시·최신 받기는 직접 fetch).
FETCH_TTL_SEC = 120
_fetch_cache = SigCache(maxsize=8)


def refresh_git() -> None:
    _cache.clear()


def fetch_remote(repo: Path) -> tuple[bool, str]:
    """사이드바용 git fetch (FETCH_TTL_SEC 안에서는 다시 하지 않음). 실패해도(오프라인) 화면은 그대로 그린다."""
    return _fetch_cache.get(("fetch", str(repo)), (int(time.time() // FETCH_TTL_SEC),),
                            lambda: cat.git_fetch(repo, quiet=True))


def _catalog_files(root: Path) -> list[Path]:
    try:
        return sorted(root.iterdir())
    except OSError:
        return []


def _git_sig(repo: Path, root: Path) -> tuple | None:
    sig = file_sig(root, *_catalog_files(root), *(repo / ".git" / name for name in GIT_FILES))
    return None if sig is None else (*sig, int(time.time() // GIT_TTL_SEC))


def load_catalog(root: Path) -> cat.Catalog:
    """카탈로그 (파일이 그대로면 직전에 읽은 것). 항목 목록은 복사해서 돌려줘 고쳐 써도 캐시가 바뀌지 않는다."""
    loaded = _cache.get(("catalog", str(root)), file_sig(root, *_catalog_files(root)), lambda: cat.Catalog.load(root))
    return cat.Catalog(loaded.root, list(loaded.entries))


def repo_root(root: Path) -> Path | None:
    return _cache.get(("root", str(root)), (int(time.time() // GIT_TTL_SEC),), lambda: cat.repo_root(root))


def repo_state(repo: Path, subdir: str, root: Path) -> tuple[cat.RepoInfo, str]:
    """(GitHub 정보, git status) — 사이드바용."""
    return _cache.get(("state", str(repo), subdir), _git_sig(repo, root),
                      lambda: (cat.repo_info(repo, subdir), cat.git_status(repo, subdir)))


def git_diff_status(ctx: "CatalogContext") -> tuple[str, str]:
    """(git diff, git status) — 채택 후 커밋 패널용."""
    return _cache.get(("diff", str(ctx.repo), ctx.subdir), _git_sig(ctx.repo, ctx.root),
                      lambda: (cat.git_diff(ctx.repo, ctx.subdir), cat.git_status(ctx.repo, ctx.subdir)))


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
    repo = repo_root(root)
    subdir = str(root.relative_to(repo)).replace("\\", "/") if repo else ""
    catalog = load_catalog(root)
    counts = " · ".join(f"{k} {len(catalog.by_category(k))}" for k in cat.CATEGORY_FILES)
    if catalog.legacy:  # AI 를 나누기 전 카탈로그 — 옛 AI 파일 항목도 보이게
        counts += f" · AI(나누기 전) {len(catalog.by_category('AI'))}"
    st.caption(f"항목: {counts}")
    ctx = CatalogContext(root, repo, subdir, catalog)
    if repo is None:
        st.warning("git 저장소가 아닙니다 — 반영은 되지만 커밋·푸시는 할 수 없습니다.")
    else:
        fetched, _ = fetch_remote(repo)
        info, status = repo_state(repo, subdir, root)
        link = info.tree(subdir)
        if link:
            st.markdown(f"🐙 [GitHub 에서 보기]({link})")
        if info.last_commit:
            st.caption(f"마지막 커밋: {info.last_commit}")
        st.caption("커밋 안 된 변경 없음" if not status else f"커밋 안 된 변경:\n```\n{status}\n```")
        if status:
            # git_panel 안에 'git diff' 접기 상자가 있어 접기 상자 안에 넣을 수 없다 (Streamlit 은 겹쳐 접기를 막음)
            with st.container(border=True):
                st.markdown("**📝 커밋 안 된 변경 올리기**")
                st.caption("앱을 다시 켜서 채택 직후의 커밋 상자가 사라졌어도 여기서 커밋하고 GitHub 에 올릴 수 있습니다.")
                git_panel(ctx, "sidebar", cat.pending_commit_message(git_diff_status(ctx)[0]))
        if info.behind > 0:
            # GitHub 웹에서 직접 고친 것 등 — 먼저 받아 두면 푸시가 거절되지 않는다 (푸시할 때도 자동으로 합침)
            st.warning(f"⬇️ GitHub 에 이 PC 에 없는 커밋 {info.behind}개 — '⬇️ 최신 받기'를 먼저 눌러 주세요")
        elif not fetched:
            st.caption("GitHub 최신 여부 확인 실패 (오프라인?) — 마지막으로 받아 온 정보 기준")
        if info.unpushed > 0:
            # 커밋만 하고 푸시하지 않은 채 남은 것 — GitHub 에는 아직 없다
            st.warning(f"⬆️ GitHub 에 아직 올리지 않은 커밋 {info.unpushed}개")
            if st.button("🚀 지금 푸시", use_container_width=True, key="concept_push_pending"):
                _rerun_with(push_messages(repo, subdir, root))
        if st.button("⬇️ 최신 받기 (git pull --rebase)", use_container_width=True, key="concept_pull"):
            sync = cat.git_sync(repo, root)
            refresh_git()
            _rerun_with(sync_messages(sync) or [("success", "이미 최신입니다.")])
        show_messages(st.session_state.pop(SIDEBAR_MSGS, []))
    return ctx


SIDEBAR_MSGS = "concept_sidebar_msgs"


def _rerun_with(msgs: list[tuple[str, str]]) -> None:
    """사이드바 버튼 결과 — 위에 이미 그린 '커밋 N개'·항목 수가 옛 값이라, 결과를 넘겨 두고 화면을 새로 그린다."""
    st.session_state[SIDEBAR_MSGS] = msgs
    st.rerun()


def show_messages(msgs: list[tuple[str, str]]) -> None:
    """[(종류, 내용)] — 종류: success·info·warning·error, log(접힌 git 출력)."""
    for kind, text in msgs:
        if kind == "log":
            with st.expander("git 출력"):
                st.code(text)
        else:
            getattr(st, kind)(text)


def sync_messages(sync: cat.SyncResult) -> list[tuple[str, str]]:
    """git_sync 결과 — 합쳤으면 몇 개 합쳤는지, 실패하면 이유와 할 일 (새로 받은 게 없으면 빈 목록)."""
    if not sync.ok:
        return [("error", sync.message)] + ([("log", sync.log[-3000:])] if sync.log else [])
    return [("info", "🔄 " + sync.message)] if sync.integrated else []


def push_messages(repo: Path, subdir: str, root: Path) -> list[tuple[str, str]]:
    """🚀 푸시 — GitHub 가 앞서 있으면 먼저 자동으로 합친(rebase) 뒤 올린다. 충돌하면 되돌리고 올리지 않는다."""
    sync, ok, log = cat.git_sync_push(repo, root)
    refresh_git()
    msgs = sync_messages(sync)
    if not sync.ok:
        return msgs
    if not ok:
        return msgs + [("error", "푸시 실패: " + log[-800:])]
    if "up-to-date" in log.lower() or "up to date" in log.lower():
        return msgs + [("warning", "올릴 새 커밋이 없습니다 (GitHub 와 이미 같음).")]
    link = cat.repo_info(repo, subdir).commit(cat.head_sha(repo))
    return msgs + [("success", "GitHub 에 올렸습니다." + (f" → [올린 커밋 보기]({link})" if link else "") + "\n\n" + log[-500:])]


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
            refresh_git()
            st.session_state["concept_delete_result"] = {"title": target.name, "changes": changes}
            st.rerun()
        result = st.session_state.get("concept_delete_result")
        if result:
            st.success(f"🗑 「{result['title']}」 을 old_term 으로 옮겼습니다.")
            render_changes(result["changes"], [])
            git_panel(ctx, f"del_{result['title']}", f"Term: {result['title']} 삭제")
        return None

    # ── ✍️ 작성: 추가/수정은 고르지 않는다 — 개념명으로 기존 항목을 찾아 자동 판정 ──
    ai_panel(ctx)
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
                    + (f"  \n(다른 항목의 7.변형 소절 '{m0.detail}' 로 편입되어 있어 그 부모 항목을 수정합니다)" if m0.kind == "하위" else "")
                    + "  \n원본 항목은 그대로 두고, 근거가 있거나 형식에 어긋난 곳만 변경 제안으로 받습니다. "
                      "결과 화면에서 변경마다 근거를 읽고 반영할지 고르면 됩니다.")
            status = concept.app_status(m0.entry) if m0.entry.category == "AI-App" else None
            if status:
                st.caption(f"🧩 Agent Office 적용 현황: {STATUS_ICON.get(status.get('status'), '❔')} — {status.get('note', '')}")
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
    related = [e for e in ctx.catalog.entries if e.name in related_names]
    # 요청은 LLM 없이 코드로 만든다 → 맡기기 전에도 보낼 브리프를 그대로 보여 줄 수 있다
    question, brief = (concept.build_concept_request(action, name.strip(), draft, related, target, sub_title=sub_title)
                       if name.strip() else ("", None))
    applied_prompt(question, brief, concept.used_prompt_version(question), key=f"in_{team_id}")
    if not st.button("맡기기", type="primary", disabled=busy or not name.strip(), key=f"cw_go_{team_id}"):
        return None
    return {
        "question": question, "brief": brief, "auto_brief": False, "use_web": bool(use_web and web_available),
        "tool_inputs": {}, "extra_validators": [],
        # 수정이면 원본 전문(변경 제안을 여기에 적용)과 제목 줄 날짜(기존 이력 + 오늘 수정일)를 하네스·결과 화면이 쓴다
        "harness_options": ({"concept_header": concept.expected_header(target), "concept_original": target.text}
                            if action == "update" and target else {}),
        "concept": {"action": action, "title": name.strip(), "target": target.name if target else "",
                    "prompt": concept.prompt_version()["hash"]},
    }


# ─── 🧩 AI 개념 — LLM 앱 적용 가능 / 불가 (두 파일) ───────────────────────

STATUS_ORDER = {"없음": 0, "일부": 1, "적용": 2}  # 앱 성능을 올릴 여지가 큰 것(없음·일부)을 위로
STATUS_ICON = {"적용": "✅ 적용", "일부": "🟡 일부", "없음": "⬜ 없음"}
UNCHECKED = "❔ 확인 안 함"


def app_rows(entries: list[cat.Entry]) -> list[dict]:
    """AI-App 항목 표 — Agent Office 적용 현황(적용/일부/없음 + 관련 파일), 성능을 올릴 여지가 큰 것부터."""
    status_rows = concept.status_rows()  # 표 파일은 한 번만 확인 (Windows 에서 파일 확인 1회 약 1ms)
    rows = []
    for e in entries:
        s = concept.app_status(e, status_rows) or {}
        rows.append({"항목": e.name, "Agent Office": STATUS_ICON.get(s.get("status"), UNCHECKED),
                     "관련 파일": ", ".join(s.get("files") or []), "현황 설명": s.get("note", "")})
    rows.sort(key=lambda r: STATUS_ORDER.get(r["Agent Office"].split()[-1], -1))
    return rows


def ai_panel(ctx: CatalogContext) -> None:
    """🧩 AI 항목을 'LLM 앱에 적용 가능 / 불가' 두 파일로 나눠 보기 (접힘). 앱 적용 가능 쪽에는 Agent Office 적용 현황."""
    app, model = ctx.catalog.by_category("AI-App"), ctx.catalog.by_category("AI-Model")
    with st.expander(f"🧩 AI 개념 — LLM 앱에 적용할 수 있나? (가능 {len(app)} · 불가 {len(model)})", expanded=False):
        if ctx.catalog.legacy:
            st.warning("이 카탈로그는 AI 를 두 파일로 나누기 전 모양입니다 (옛 AI 파일이 있음). CS 저장소에서 나눈 판을 받으면 "
                       "여기서 앱 적용 가능·불가로 나눠 볼 수 있습니다. 지금도 읽기·쓰기는 됩니다.")
        st.caption("LLM 을 API 로 부르는 앱(Agent Office 같은 앱)이 모델을 바꾸지 않고 코드만으로 적용할 수 있는가로 나눈 두 파일입니다. "
                   "개념 편집 팀이 새 AI 항목을 쓸 때 어느 파일에 넣을지 함께 정합니다.")
        t1, t2 = st.tabs([f"✅ 앱 적용 가능 ({len(app)})", f"🧱 앱 적용 불가 — 모델·학습·GPU ({len(model)})"])
        with t1:
            rows = app_rows(app)
            counts = {k: sum(1 for r in rows if r["Agent Office"] == v) for k, v in {**STATUS_ICON, "?": UNCHECKED}.items()}
            st.caption("Agent Office 적용 현황 (코드를 직접 확인한 것 — `teams/concept/ai_app_status.yaml`): "
                       + " · ".join(f"{STATUS_ICON[s]} {counts[s]}" for s in STATUS_ICON)
                       + (f" · {UNCHECKED} {counts['?']}" if counts["?"] else "")
                       + " — '없음·일부'가 앱 성능을 더 올릴 수 있는 후보입니다.")
            st.dataframe(rows, hide_index=True, use_container_width=True)
        with t2:
            st.caption("모델 구조·서빙·GPU 커널은 그 기술을 쓴 모델·서빙을 고르는 것으로만 효과가 있고, "
                       "가중치 학습·모델 내부 분석·연구는 API 로 부르는 앱에서 적용할 수 없습니다.")
            st.dataframe([{"항목": e.name, "태그": e.tag} for e in model], hide_index=True, use_container_width=True)


# ─── 📜 적용된 프롬프트 ─────────────────────────────────────────────────


def applied_prompt(question: str, brief, used: str = "", key: str = "") -> None:
    """📜 적용된 프롬프트 (접힘) — 에이전트가 실제로 받는 것: 스킬 전문(판 표시) · 업무 브리프 · 에이전트별 시스템 프롬프트.

    used: 지난 결과를 만든 프롬프트 판(해시) — 지금 스킬 파일과 다르면 알려 준다.
    """
    ver = concept.prompt_version()
    with st.expander(f"📜 적용된 프롬프트 — IT 개념 프롬프트 v{used or ver['hash']}", expanded=False):
        if used and used != ver["hash"]:
            st.warning(f"이 결과는 프롬프트 **v{used}** 로 만들었습니다. 지금 스킬 파일은 **v{ver['hash']}** 이라 아래 스킬 전문과 "
                       "다를 수 있습니다 (그때 내용은 git 기록의 teams/concept/skills/ 에서 확인).")
        elif question and not used:
            st.caption("이 요청에는 프롬프트 판 기록이 없습니다 (판 기록 기능 이전에 만든 결과). 아래는 지금 스킬 파일입니다.")
        st.caption(f"지금 스킬: {ver['label']} — `teams/concept/skills/it_concept_prompt.md`")
        st.caption("에이전트가 받는 것 = 시스템 프롬프트(페르소나·직급·전문 분야·공통 규칙 + 스킬 전문 + 지난 피드백 교훈) "
                   "+ 업무 브리프 + 맡은 업무·근거 자료(실행 중에 정해짐)")
        tabs = st.tabs(["📘 IT 개념 프롬프트 (스킬)", "📋 업무 브리프·제약", "👥 에이전트별 역할"])
        with tabs[0]:
            st.code(ver["text"], language="markdown", wrap_lines=True, height=420)
        with tabs[1]:
            if brief is None:
                st.caption("개념명을 넣으면 보낼 업무 브리프(요청·완료 기준·제약·결과 형식)가 여기 나옵니다.")
            else:
                st.code(brief.as_prompt(question), language="markdown", wrap_lines=True, height=420)
        with tabs[2]:  # 부서 화면 '📜 프롬프트·📚 교훈'과 같은 보기 — 실행 때와 같은 코드로 만든 프롬프트 (교훈 포함)
            try:
                team = load_team(concept.TEAM_ID)
            except Exception as e:  # noqa: BLE001 — 설정을 못 읽어도 화면은 그린다
                st.caption(f"에이전트 정보를 읽지 못했습니다: {e}")
                return
            prompt_ui.member_prompts(team, key=f"concept_{key}")


# ─── 최종 결과 · 결정 · 반영 ─────────────────────────────────────────────


def change_options(run: dict) -> dict:
    """형식 검사·결과 나누기에 넘길 옵션. 기존 항목 수정이면 원본 전문(concept_original)이 들어 있다.
    앱을 다시 켜 되살린 실행은 옵션이 비어 있어 요청 글의 <original> 에서 원본을 되찾는다."""
    opts = dict(run.get("harness_options") or {})
    if not opts.get("concept_original"):
        original = concept.original_of(run.get("question") or "")
        if original:
            opts["concept_original"] = original
    return opts


def read_result(content: str, run: dict) -> concept.ConceptResult:
    """결과 나누기 — 채택한 뒤에는 사람이 확정한 최종본(<final>)을 그대로 읽는다."""
    return concept.parse_result(content, change_options(run), adopted=run.get("status") == "adopted")


def change_key(run: dict, content: str, i: int) -> str:
    """변경 제안 체크 상자의 키 — 안(수정본 v1·v2 …)마다 따로 기억한다."""
    return f"cc_{run.get('run_id', '')}_{zlib.crc32(content.encode('utf-8')):08x}_{i}"


def picked_changes(res: concept.ConceptResult, run: dict, content: str) -> list[bool]:
    """변경마다 반영할지 — 화면의 체크 상태. 아직 화면에 그리지 않은 안은 기본값(적용할 수 있고 근거가 있는 변경)."""
    return [c.usable and bool(st.session_state.get(change_key(run, content, i), c.default_on)) for i, c in enumerate(res.changes)]


def change_list(res: concept.ConceptResult, run: dict, content: str, decided: bool) -> list[bool]:
    """✏️ 변경 제안 — 변경마다 종류 · 고치기 전 → 고친 뒤 · 근거를 보여 주고 반영할지 고르게 한다. 돌려주는 값 = 변경마다 고름 여부."""
    if not res.changes:
        st.info("팀이 고칠 곳을 찾지 못했습니다 — 원본 그대로입니다. 고칠 점이 있으면 아래 '💬 결과에 대해 묻기 · 수정 요청'에 적어 주세요.")
        return []
    if decided:
        st.caption("팀이 냈던 변경 제안입니다 (참고용). 채택할 때 고른 변경만 반영됐고, 실제로 바뀐 곳은 아래 '변경 내용 (git diff)'에서 볼 수 있습니다.")
    else:
        st.caption("원본 항목은 그대로 두고, 여기서 체크한 변경만 반영합니다. 근거를 읽고 반영하지 않을 변경은 체크를 푸세요 → "
                   "'📄 최종본'에서 결과 확인 → 아래 '👤 최종 결정'에서 채택. "
                   "근거 글은 이 화면과 이 PC 의 실행 기록에만 남고 GitHub 에는 올라가지 않습니다.")
    picked = []
    for i, c in enumerate(res.changes):
        with st.container(border=True):
            label = f"변경 {i + 1} · {concept.CHANGE_KINDS[c.kind]}"  # '1. …' 로 쓰면 화면이 번호 목록으로 읽어 번호가 사라진다
            if decided:
                st.markdown(f"**{label}**")
                on = False
            else:
                on = st.checkbox(f"{label} — 반영", value=c.default_on, disabled=not c.usable, key=change_key(run, content, i))
            if c.problem:
                st.error(f"적용할 수 없습니다 — {c.problem}. 이 부분은 원본 그대로 둡니다. 필요하면 아래 '수정 요청'으로 다시 받아 보세요.")
            elif c.no_evidence:
                st.warning(f"근거 없음 — {c.no_evidence}. 기본으로 반영하지 않습니다 (그래도 반영하려면 체크하세요).")
            before, after = st.columns(2)
            before.caption("고치기 전 (원본)")
            before.code(c.find or "(비어 있음)", language=None, wrap_lines=True)
            after.caption("고친 뒤")
            after.code(c.replace or "(삭제)", language=None, wrap_lines=True)
            st.markdown("**근거** — " + (c.evidence or "(없음)") + (f"  \n**출처** — {c.source}" if c.source else ""))
            picked.append(bool(on) and c.usable)
    return picked


def result_tabs(content: str, run: dict) -> None:
    """최종 결과를 탭으로 — (기존 항목 수정이면 변경 제안 /) 최종본(카탈로그에 들어갈 글) / 검증 로그 / GitHub 요약 / 적용 계획."""
    opts = change_options(run)
    res = read_result(content, run)
    info = run.get("concept") or {}
    plan = res.plan or {}
    decided = {"add": "🆕 추가", "update": "✏️ 수정", "delete": "🗑 삭제"}.get(plan.get("action"), "판정 없음")
    target = plan.get("target") or info.get("target")
    st.caption(f"편집 팀 판정: **{decided}**" + (f" — 대상 「{target}」" if target and plan.get("action") == "update" else "")
               + (f" · 통합으로 빠질 항목: {', '.join(plan['absorb'])}" if plan.get("absorb") else "")
               + (f" · 넣을 곳: **{plan['category']}**" if plan.get("category") else ""))
    applied_prompt(run.get("question", ""), run.get("brief"), concept.used_prompt_version(run.get("question", "")),
                   key=f"res_{run.get('run_id', '')}")
    changing = res.changes is not None  # 기존 항목 수정 — 원본 + 변경 제안
    decided = run.get("status") in ("adopted", "rejected")
    # 채택한 수정 기록에는 사람이 확정한 <final> 이 들어 있다 — 팀이 쓴 것이 아니므로 '수정은 <final> 을 쓰지 않는다'로 다시 따지지 않는다
    problems = [] if changing and run.get("status") == "adopted" else concept.check_answer(content, opts)
    if problems:
        st.warning("형식 검사에서 남은 문제 (채택 전 아래에서 직접 고칠 수 있습니다):\n" + "\n".join(f"- {p}" for p in problems))
    tabs = st.tabs(([f"✏️ 변경 제안 ({len(res.changes)})"] if changing else []) + ["📄 최종본", "🧠 검증 로그", "🐙 GitHub 요약", "🧾 적용 계획"])
    final_entry = res.final_entry
    if changing:
        with tabs[0], st.container(height=420):
            picked = change_list(res, run, content, decided)
        tabs = tabs[1:]
        if not decided:
            final_entry = concept.assemble(res.original, res.changes, picked, opts.get("concept_header", ""))
    with tabs[0], st.container(height=420):
        if changing and not decided:
            st.caption(f"원본에 고른 변경 {sum(picked)}개를 반영한 글입니다. 변경 제안에 없는 곳은 원본과 글자 그대로 같고, "
                       "제목 줄의 수정일은 코드가 붙였습니다." + ("" if any(picked) else " 고른 변경이 없어 이대로 채택하면 수정일만 바뀝니다."))
            left = concept.check_entry(final_entry, opts.get("concept_header", ""))
            if left:  # 원본에 원래 있던 형식 문제 — 하네스가 막지는 않지만 알려 준다
                st.caption(f"원본에 원래 있던 형식 문제 {len(left)}건이 남아 있습니다 (이번 변경 제안으로는 고쳐지지 않음): "
                           + " / ".join(x[:40] for x in left[:4]) + (" …" if len(left) > 4 else ""))
        st.text(final_entry or "최종본이 없습니다 — 아래 '채택할 안'에서 편집장·부장의 안을 고르거나, 다시 맡겨 주세요.")
    with tabs[1], st.container(height=420):
        if res.change_notes:  # 최종본에서 떼어 낸 🔧수정·➕추가 표시와 이유 — 확인용 (카탈로그에는 들어가지 않음)
            notes = "\n".join(f"- {n}" for n in res.change_notes)
            st.markdown(f"**🔧 초안 대비 변경 요약** (최종본에는 넣지 않음)\n{notes}")
        st.markdown(res.thought or "(검증 로그 없음)")
    with tabs[2], st.container(height=420):
        st.markdown(res.summary_md or "(요약 없음)")
    with tabs[3], st.container(height=420):
        st.json({k: v for k, v in plan.items() if k != "changes"} if changing else plan)  # 변경 제안은 첫 탭에서 본다
        for e in plan.get("edits") or []:
            st.caption(f"✏️ 다른 항목 수정 — {e.get('entry')}: {e.get('reason', '')}")


def decision_fields(content: str, key: str, run: dict | None = None) -> tuple[str, str]:
    """채택 폼 안의 입력칸 — 최종본과 적용 계획을 고칠 수 있게 나눠 보여 준다.

    기존 항목 수정이면 최종본 = 원본 + 위 '✏️ 변경 제안'에서 고른 변경 (고름이 바뀌면 입력칸을 새 글로 다시 채운다).
    """
    run = run or {}
    res = read_result(content, run)
    entry_value, plan = res.final_entry, res.plan
    if res.changes is not None:
        picked = picked_changes(res, run, content)
        entry_value = concept.assemble(res.original, res.changes, picked, change_options(run).get("concept_header", ""))
        key += "_" + "".join("1" if on else "0" for on in picked)
        plan = {k: v for k, v in plan.items() if k != "changes"}  # 변경 제안·근거는 위에서 고른다 (카탈로그 반영에는 쓰이지 않음)
        shown = not res.changes or change_key(run, content, 0) in st.session_state
        st.caption((f"위 '✏️ 변경 제안'에서 고른 변경 {sum(picked)}개" if shown else
                    f"이 안의 변경 제안 가운데 근거가 있는 {sum(picked)}개(기본값 — 변경을 하나씩 고르려면 가장 최근 안을 고르세요)")
                   + "를 원본에 반영한 글입니다. 여기서 직접 고치면 고친 글이 그대로 들어갑니다 "
                     "(변경을 다시 고르면 직접 고친 내용은 지워집니다).")
    entry = st.text_area("최종본 (카탈로그에 들어갈 글 — 필요하면 수정)", value=entry_value, height=300, key=f"ce_{key}")
    plan = st.text_area("적용 계획 JSON (필요하면 수정)", value=json.dumps(plan, ensure_ascii=False, indent=2),
                        height=180, key=f"cp_{key}")
    return entry, plan


def human_edited(content: str, entry: str, run: dict) -> bool:
    """채택 기록의 '✍️ 직접 수정' 표시. 기존 항목 수정은 코드가 내놓은 기본 최종본과 다를 때 (변경을 빼거나 더 켰거나 직접 고침)."""
    res = read_result(content, run)
    if res.changes is None:
        return concept.replace_final(content, entry).strip() != content.strip()
    return entry.strip() != res.final_entry.strip()


def followup_original(run: dict) -> str:
    """💬 수정 요청에 붙일 원본 전문. 후속 대화는 요청 글을 앞 4천 자만 보여 줘(core/followup.py) 긴 항목은 원본이 잘리는데,
    변경 제안의 find 는 원본에서 글자 그대로 복사해야 한다 → 근거 자료 맨 앞에 통째로 싣는다. 추가·예전 결과면 ''."""
    original = change_options(run).get("concept_original")
    if not original:
        return ""
    return ("## 기존 카탈로그 항목 — 수정 대상 원본 전문 (변경 제안 changes 의 find 는 여기서 글자 그대로 복사한다)\n"
            f"<{concept.ORIGINAL_TAG}>\n{original}\n</{concept.ORIGINAL_TAG}>")


def check_adoptable(entry: str, plan_text: str) -> tuple[dict | None, str]:
    """채택 전에 반영 가능한지 확인. (계획, 오류 메시지)"""
    try:
        plan = json.loads(plan_text)
    except json.JSONDecodeError as e:
        return None, f"적용 계획 JSON 오류: {e}"
    if not isinstance(plan, dict):
        return None, "적용 계획 JSON 은 { … } 객체여야 합니다."
    plan, _ = concept.normalize_plan(plan)  # 사람이 고친 계획도 absorb·edits 모양을 맞춘다 (모양 문제는 결과 화면에서 이미 알림)
    if not entry.strip() and plan.get("action") != "delete":
        return None, "최종본이 비어 있습니다 — '채택할 안'에서 최종 통합안(편집장·부장의 안)을 고르세요."
    if plan.get("action") == "add" and plan.get("category") in cat.LEGACY_FILES:  # 옛 결과 — 새 항목은 넣을 AI 파일을 정해야 한다
        return None, ("AI 는 두 파일로 나뉘었습니다 — 적용 계획 JSON 의 category 를 \"AI-App\"(LLM 앱에서 바로 적용) 또는 "
                      "\"AI-Model\"(앱 적용 불가 — 모델·학습·GPU) 로 고친 뒤 채택해 주세요.")
    return plan, ""


def apply_adopted(ctx: CatalogContext, run: dict, entry: str, plan: dict) -> None:
    """채택한 결과를 카탈로그 파일에 반영한다 (로컬만 — 커밋·푸시는 버튼으로)."""
    info = run.get("concept") or {}
    fresh = cat.Catalog.load(ctx.root)
    # 팀이 '수정'으로 판정했는데 대상 이름을 빠뜨렸거나 못 찾는 이름이면, 화면에서 판정한 기존 항목을 쓴다
    if plan.get("action") == "update" and info.get("target") and not fresh.find(str(plan.get("target", ""))):
        plan["target"] = info["target"]
    original = change_options(run).get("concept_original")
    if original and plan.get("action") == "update":
        # 기존 항목 수정(원본 + 변경 제안) — 대상은 언제나 그 원본 항목이고, 원본이 그 사이 바뀌었으면 덮어쓰지 않는다
        name = cat.parse_header(original.partition("\n")[0])[1]
        target = fresh.find(name)
        if target is None or target.text != original:
            raise ValueError("카탈로그의 원본 항목이 이 결과를 만든 뒤에 바뀌었습니다 (다른 채택이나 GitHub 에서 받은 변경). "
                             "바뀐 내용을 덮어쓰지 않도록 멈췄습니다. 같은 개념을 다시 맡겨 주세요")
        plan["target"] = name
    run["concept_changes"] = cat.apply_plan(fresh, plan, entry)
    run["concept_edit_logs"] = concept.apply_edits(cat.Catalog.load(ctx.root), plan.get("edits") or [])
    run["concept_commit"] = f"Term: {plan.get('title') or info.get('title', '')} {concept.ACTIONS.get(plan.get('action'), '정리')}"
    refresh_git()


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
    diff, pending = git_diff_status(ctx)
    with st.expander("변경 내용 (git diff)"):
        st.code(diff[:30000] or "(차이 없음 — 이미 커밋했거나 새 파일만 있음)", language="diff")
    msg = st.text_input("커밋 메시지", value=message, key=f"msg_{key}")
    st.caption("📝 아직 커밋하지 않은 변경이 있습니다 — '🚀 커밋 후 푸시'를 누르면 커밋과 GitHub 업로드를 한 번에 합니다."
               if pending else "커밋하지 않은 변경 없음")
    c1, c2 = st.columns(2)
    if c1.button("💾 커밋만", use_container_width=True, key=f"commit_{key}", disabled=not pending):
        ok, log = cat.git_commit(ctx.repo, ctx.subdir, msg)
        refresh_git()
        (st.success if ok else st.error)(log[-800:])
    if c2.button("🚀 커밋 후 GitHub 에 푸시", type="primary", use_container_width=True, key=f"push_{key}"):
        if pending:  # 커밋 안 된 변경이 있으면 먼저 커밋 — 커밋 없이 푸시하면 아무것도 올라가지 않는다
            ok, log = cat.git_commit(ctx.repo, ctx.subdir, msg)
            refresh_git()
            if not ok:
                st.error("커밋 실패: " + log[-800:])
                return
        show_messages(push_messages(ctx.repo, ctx.subdir, ctx.root))


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
