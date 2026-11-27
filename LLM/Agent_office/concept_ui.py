"""개념 편집 팀 전용 화면 조각 — 메인 화면(app.py)이 이 팀을 고르면 불러 쓴다.

메인 화면의 조직도·에이전트 패널·이미지 편집·타임라인·최종 결정은 다른 팀과 똑같이 쓰고,
개념 편집 팀에만 필요한 부분만 여기서 그린다.
  - 사이드바: 카탈로그(GitHub jeong-yun/CS 의 Term/Every_day) 위치·git 상태
  - 업무 입력: ✍️ 작성(기존에 있으면 수정 · 없으면 추가 — 자동 판정) / 🗑 삭제 / 🎧 개념 듣기(카탈로그 개념 하나나 직접 적은 글 → 듣기용 MP3)
  - 최종 결과: 최종본·검증 로그·GitHub 요약·적용 계획 탭, 형식 검사
    · 기존 항목 수정이면 맨 앞에 '✏️ 변경 제안' 탭 — 변경마다 고치기 전 → 고친 뒤와 근거를 보고 반영할지 고른다 (원본은 그대로,
      고른 변경만 코드가 끼워 넣는다. 근거 글은 화면에만 — 카탈로그·커밋 메시지에는 들어가지 않는다)
  - 🎧 듣기용 MP3: 최종본 → 읽기용 대본 → 고른 음성 모델(Gemini·OpenAI TTS, core/listen.py) — 버튼을 눌렀을 때만, 만든 파일은 결과 폴더에 저장해 재사용
  - 📜 적용된 프롬프트: 입력칸 아래·결과 위 — 스킬(IT 개념 프롬프트) 판·전문, 보낸 업무 브리프, 에이전트별 시스템 프롬프트
  - 채택 후: 📚 카탈로그 반영 → 💾 커밋 → 🚀 푸시 (GitHub 에 이 PC 에 없는 커밋이 있으면 먼저 자동 rebase, 충돌하면 되돌리고 중단)
"""

import json
import time
import zlib
from dataclasses import dataclass, replace
from pathlib import Path

import streamlit as st

import prompt_ui
from core import catalog as cat
from core import concept, listen, llm_factory, music, usage
from core.config import BASE_DIR, load_team
from core.memo import SigCache, file_sig

CONCEPT_FEATURE = "concept_catalog"  # team.yaml features 에 있으면 이 화면 조각을 쓴다
TASK_WRITE, TASK_DELETE = "write", "delete"
TASK_LISTEN = "listen"  # 🎧 개념 듣기 — 팀에 맡기지 않고 바로 (catalog_listen)

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
    """(git diff, git status) — 사이드바 커밋 상자와 본문 안내(commit_hint)용."""
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
                st.caption("채택·삭제로 바뀐 카탈로그 파일을 여기서 커밋하고 GitHub 에 올립니다 (앱을 다시 켠 뒤에도 그대로 남아 있습니다).")
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
            commit_hint(ctx)
        return None

    # ── ✍️ 작성: 추가/수정은 고르지 않는다 — 개념명으로 기존 항목을 찾아 자동 판정 ──
    ai_panel(ctx)
    name = st.text_input("개념명(한/영) *", key=f"cw_name_{team_id}",
                         placeholder="예: 그룹 상대적 정책 최적화(Group Relative Policy Optimization, GRPO)")
    draft = st.text_area("초안", height=220, key=f"cw_draft_{team_id}",
                         placeholder="정의·작동 원리·장단점·참고 자료 등 초안을 붙여 넣으세요 (비워도 개념명으로 작성)")
    target, sub_title, action, merging = None, "", "add", False
    if name.strip():
        matches = cat.find_existing(ctx.catalog, name)
        ADD = "__add__"
        # 결과 화면의 '원본 기반으로 다시 받기'(merge_notice)가 남긴 표시 — 팀이 이 개념을 그 기존 항목에 편입하기로 판정했다
        merge = st.session_state.get(f"cw_merge_{team_id}") or {}
        into = ctx.catalog.find(merge["target"]) if merge.get("name") == name.strip() and not matches else None

        def option_label(n: str) -> str:
            if n == ADD:
                return "🆕 새 항목으로 추가"
            if into and n == into.name:
                return f"✏️ 편입 — {ctx.label(n)} 에 넣기 (원본 기반 수정)"
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
        elif into:
            st.info(f"팀이 기존 항목에 편입하기로 판정한 개념입니다 → 그 항목을 **원본 기반 수정**으로 진행합니다: {ctx.label(into.name)}"
                    "  \n원본 항목은 그대로 두고, 편입에 필요한 곳만 변경 제안으로 받습니다. 결과 화면에서 변경마다 근거를 읽고 반영할지 고르면 됩니다.")
        else:
            st.success("카탈로그에 없는 개념입니다 → **추가**로 진행합니다. (검토 중 팀이 기존 개념의 하위라고 판단하면 편입으로 바꿉니다)")
        picked = st.selectbox("판정 (필요하면 바꾸기)", [m.entry.name for m in matches] + ([into.name] if into else []) + [ADD],
                              format_func=option_label, key=f"cw_pick_{team_id}_{hash(name)}")
        if into and picked == into.name:
            target, action, merging = into, "update", True
        elif picked != ADD:
            hit = next(m for m in matches if m.entry.name == picked)
            target, sub_title, action = hit.entry, (hit.detail if hit.kind == "하위" else ""), "update"
    suggested = [e.name for e, _ in cat.find_related(ctx.catalog, name, draft, related_limit) if not target or e.name != target.name]
    related_names = st.multiselect("함께 비교할 기존 항목 (자동 추천 — 추가·삭제 가능)", names, default=suggested,
                                   format_func=ctx.label, key=f"cw_related_{team_id}_{hash(name)}")
    use_web = st.checkbox("🌐 웹 검색으로 원전 확인", value=web_available, disabled=not web_available, key=f"cw_web_{team_id}")
    related = [e for e in ctx.catalog.entries if e.name in related_names]
    # 요청은 LLM 없이 코드로 만든다 → 맡기기 전에도 보낼 브리프를 그대로 보여 줄 수 있다
    question, brief = (concept.build_concept_request(action, name.strip(), draft, related, target, sub_title=sub_title, merge=merging)
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
        st.caption("팀이 냈던 변경 제안입니다 (참고용). 채택할 때 고른 변경만 반영됐고, 아직 커밋하지 않았다면 실제로 바뀐 곳을 왼쪽 사이드바의 '변경 내용 (git diff)'에서 볼 수 있습니다.")
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


def merge_notice(run: dict, target_name: str) -> None:
    """'추가'로 맡겼는데 팀이 기존 항목 편입(수정)으로 판정한 결과 — 그 항목을 원본 기반 수정으로 다시 맡기는 길을 안내한다.

    추가 요청에는 그 항목의 원본 전문이 없어(관련 항목은 발췌만) 팀이 항목 전체를 <final> 에 다시 썼다 → 변경마다 근거를 보고 고를 수 없다.
    버튼은 위 '✍️ 개념 작성' 입력칸만 채운다 (맡기기는 사람이 누른다 — 팀이 한 번 더 일한다). 대상을 못 찾으면 전처럼 <final> 방식뿐이라고 알린다.
    """
    root = st.session_state.get("concept_root") or ""
    target = load_catalog(Path(root)).find(target_name) if root and target_name and Path(root).is_dir() else None
    if target is None:
        st.warning(f"팀이 이 개념을 기존 항목 「{target_name or '이름 없음'}」 에 편입(수정)하기로 판정했지만, 그 항목을 카탈로그에서 찾지 못했습니다. "
                   "그래서 원본 기반(변경마다 고르기)으로 다시 받을 수 없습니다 — 채택하려면 아래 적용 계획 JSON 의 target 을 실제 항목 이름으로 "
                   "고쳐야 하고, 채택하면 그 항목 전체가 '📄 최종본' 글로 바뀝니다.")
        return
    name, draft = concept.request_inputs(run.get("question", ""))
    st.info(f"팀이 이 개념을 기존 항목 「{target.name}」 에 편입(수정)하기로 판정했습니다. 지금 '📄 최종본'은 그 항목 전체를 다시 쓴 글이라 "
            "무엇이 왜 바뀌는지 변경마다 골라 반영할 수 없습니다 (이대로 채택하면 항목 전체가 이 글로 바뀝니다).  \n"
            "아래 버튼을 누르면 위 '✍️ 개념 작성' 입력칸이 '기존 항목에 편입 — 원본 기반 수정'으로 채워집니다. 거기서 **맡기기**를 누르면 "
            "원본은 그대로 두고 변경 제안(원문 → 바꿀 글 + 근거)으로 다시 받습니다 (팀이 한 번 더 일합니다).")
    st.button("✏️ 원본 기반으로 다시 받기 — 입력칸 채우기", key=f"cm_{run.get('run_id', '')}", on_click=_prefill_merge,
              args=(name or (run.get("concept") or {}).get("title", ""), draft, target.name))


def _prefill_merge(name: str, draft: str, target_name: str) -> None:
    """버튼 콜백 — 화면을 그리기 전에 실행되므로 '✍️ 개념 작성' 입력칸(input_form)의 값을 바꿀 수 있다."""
    team_id = concept.TEAM_ID
    st.session_state[f"cw_name_{team_id}"] = name
    st.session_state[f"cw_draft_{team_id}"] = draft
    st.session_state[f"cw_merge_{team_id}"] = {"name": name, "target": target_name}  # input_form 이 '편입 — 원본 기반 수정'을 고르게 한다


def result_tabs(content: str, run: dict) -> tuple[str, str]:
    """최종 결과를 탭으로 — (기존 항목 수정이면 변경 제안 /) 최종본(카탈로그에 들어갈 글) / 검증 로그 / GitHub 요약 / 적용 계획.
    돌려주는 값 = (지금 보이는 최종본, 적용 계획의 범주 — 없으면 '') — 🎧 듣기용 MP3 의 본문과 저장 폴더."""
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
    asked_add = info.get("action") == "add" or "- 작업(코드 자동 판정): 추가 (add)" in run.get("question", "")
    if asked_add and plan.get("action") == "update" and not changing and not decided:
        merge_notice(run, str(plan.get("target") or ""))
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
        if res.outside:  # 검증 로그 태그 앞(또는 최종본 앞)에 쓴 글 — 수정 요청에 답한 판정 표 등. 어느 탭에도 안 나오던 글이다
            st.markdown("**📝 검증 로그 태그 밖에 쓴 글** (팀이 답하며 덧붙인 글 — 이 화면에만 보이고 카탈로그·GitHub 에는 들어가지 않음)")
            st.markdown(res.outside)
            st.divider()
        st.markdown(res.thought or "(검증 로그 없음)")
    with tabs[2], st.container(height=420):
        st.markdown(res.summary_md or "(요약 없음)")
    with tabs[3], st.container(height=420):
        st.json({k: v for k, v in plan.items() if k != "changes"} if changing else plan)  # 변경 제안은 첫 탭에서 본다
        for e in plan.get("edits") or []:
            st.caption(f"✏️ 다른 항목 수정 — {e.get('entry')}: {e.get('reason', '')}")
    return final_entry, str(plan.get("category") or "")


def _show_script(saved: listen.ScriptResult) -> None:
    with st.container(height=260):
        for s in saved.script.sections:
            st.markdown(f"**{s.heading}**  \n{s.text}")


def _tts_label(model: str, models) -> str:
    """음성 모델 고르기 상자의 한 줄 — 설명 · 모델 이름 · 분당 요금(추정) · 종료 예정 · 키 없음."""
    spec = listen.TTS_MODELS[model]
    now = listen.estimate(listen.CHARS_PER_MIN, model)[1]
    later = listen.next_price(model)
    price = f"분당 약 ${now:.3f}" + (f" → {later[0]} 부터 ${later[1]:.3f}" if later and later[1] != now else "")
    parts = [f"{spec.note} — {model}", f"{price} (추정)"]
    if spec.shutdown:
        parts.append(f"{spec.shutdown} 종료 예정")
    if not llm_factory.is_available(spec.provider, models):
        parts.append("키 없음")
    return " · ".join(parts)


def _pick_tts(models) -> str:
    """🔊 음성 모델 고르기 — 처음에는 키가 있는 첫 모델(추천 순서). 고른 값은 앱을 켜 둔 동안 모든 업무에 이어진다."""
    choices = listen.tts_choices()
    if st.session_state.get("listen_tts_model") not in choices:  # 앱을 켜 둔 사이 종료일이 지난 모델
        st.session_state.pop("listen_tts_model", None)
    ready = [m for m in choices if llm_factory.is_available(listen.TTS_MODELS[m].provider, models)]
    return st.selectbox("🔊 음성 모델", choices, index=choices.index(ready[0]) if ready else 0, key="listen_tts_model",
                        format_func=lambda m: _tts_label(m, models),
                        help="음성을 만들 때 쓸 모델입니다. 요금은 대본 길이로 셈한 추정이고, 실제 청구는 각 회사 사용량 화면에서 확인합니다.")


def _made_with(model: str) -> str:
    spec = listen.TTS_MODELS.get(model)
    return f"{listen.provider_name(model)} {model} ({spec.voice} 목소리)" if spec else model


def listen_panel(entry: str, run: dict, folder: Path, models, on_calls, category: str = "") -> None:
    """🎧 듣기용 MP3 — 음성 모델 고르기 → ① 대본 쓰기(보조 모델) → 예상 길이·요금(추정) 확인 → ② 음성 만들기(고른 TTS) → 듣기·받기.

    모델·TTS 는 버튼을 눌렀을 때만 부른다. 대본(.json)·음성(.mp3)은 folder(팀 폴더)/<category 범주>/ 에 저장해
    다시 그릴 때·앱을 다시 켤 때 재사용한다 (범주 폴더로 나누기 전 팀 폴더 바로 아래 파일도 찾는다 — listen.paths).
    만든 음성의 모델은 대본 .json 의 audio_model 에 남긴다. '다른 음성 모델로 다시 만들기'는 대본을 두고 음성만 지운다.
    on_calls(run): 호출을 마친 뒤 ⏱ 호출 기록을 저장하는 함수 (app.save_monitor).
    """
    if not entry.strip():
        return
    rid = run.get("run_id", "")
    level_key = f"listen_level_{rid}"
    level = st.session_state.get(level_key, listen.DEFAULT_LEVEL)  # 아래 말투 고르기 칸의 값 — 그리기 전에 파일 이름을 정한다
    json_path, mp3_path = (listen.level_path(p, level) for p in listen.paths(folder, rid, entry, category))
    saved = listen.load_script(json_path)
    direct = rid in (listen.CATALOG_RUN, listen.TEXT_RUN)  # 🎧 개념 듣기 업무 — 업무 결과가 아니라 고른 개념·적은 글
    with st.expander("🎧 듣기용 MP3 — 들으며 이해하기", expanded=saved is not None or direct):
        st.radio("대본 말투", list(listen.LEVELS), format_func=listen.LEVELS.get, key=level_key, horizontal=True,
                 help="쉽게: 초등학생도 알아듣게 짧은 문장과 일상 비유로, 어려운 말은 처음 나올 때 풀어 말합니다. 보통: 강의 말투. "
                      "두 말투는 따로 저장되어 바꿔 가며 들을 수 있습니다 (말투를 고르기 전에 만든 MP3 는 '보통'에 있습니다).")
        error = ""
        if mp3_path.exists():
            data = mp3_path.read_bytes()
            st.audio(data, format="audio/mpeg")
            st.caption(f"{music.fmt_time(music.audio_duration(mp3_path))} · 대본 {listen.script_chars(saved.script) if saved else '?'}자 · "
                       f"{_made_with((saved.audio_model if saved else '') or listen.OLD_TTS)} · AI 로 합성한 음성 · "
                       f"`{mp3_path.parent.relative_to(folder.parent).as_posix()}/` 폴더에 저장 — 앱을 다시 켜도 "
                       + {listen.CATALOG_RUN: "같은 개념을 고르면 여기서", listen.TEXT_RUN: "같은 글을 다시 넣으면 여기서"}.get(rid, "🕘 지난 업무에서")
                       + " 들을 수 있습니다.")
            c1, c2, c3 = st.columns(3)
            c1.download_button("⬇ MP3 받기", data=data, file_name=mp3_path.name, mime="audio/mpeg",
                               key=f"listen_dl_{rid}", use_container_width=True)
            if c2.button("🔁 다른 음성 모델로 다시 만들기", key=f"listen_revoice_{rid}", use_container_width=True,
                         help="대본은 그대로 두고 음성만 지웁니다 — 모델을 고른 뒤 '음성 만들기'를 누르면 됩니다."):
                mp3_path.unlink(missing_ok=True)
                st.rerun()
            if c3.button("📝 대본부터 다시 만들기", key=f"listen_reset_{rid}", use_container_width=True):
                mp3_path.unlink(missing_ok=True)
                json_path.unlink(missing_ok=True)
                st.rerun()
            if saved:
                _show_script(saved)
            return
        model = _pick_tts(models)
        if saved is None:
            minutes, usd = listen.estimate(listen.MAX_SCRIPT_CHARS, model)
            st.caption(f"{'적은 글로' if rid == listen.TEXT_RUN else '카탈로그 본문으로'} "
                       f"{listen.MIN_SCRIPT_CHARS}~{listen.MAX_SCRIPT_CHARS}자(약 3~{minutes:.0f}분) 읽기용 대본을 먼저 쓰고, "
                       f"예상 요금을 확인한 뒤 음성을 만듭니다. 대본은 보조 모델 1~2번 호출, 음성은 많아야 약 ${usd:.2f} "
                       f"(추정 — {listen.provider_name(model)} {model} 기준).")
            if st.button("🎧 듣기용 MP3 만들기 (1/2 대본 쓰기)", key=f"listen_script_{rid}", type="primary"):
                try:
                    with st.spinner("대본 쓰는 중… (보통 10~40초)"), usage.scope(team_id=run.get("team_id", ""), run_id=rid,
                                                                             purpose="듣기용 MP3 대본"):
                        saved = listen.make_script(entry, models, level)
                    listen.save_script(json_path, saved)
                except Exception as e:  # noqa: BLE001 — 모델 오류는 제각각, 화면에 이유만
                    error = f"대본을 쓰지 못했습니다 ({type(e).__name__}: {str(e)[:200]}). 잠시 뒤 다시 눌러 주세요."
                on_calls(run)
            if saved is None:
                if error:
                    st.error(error)
                return
        minutes, usd = listen.estimate(listen.script_chars(saved.script), model)
        st.caption(f"대본 {listen.script_chars(saved.script)}자(공백 제외) · 절 {len(saved.script.sections)}개 · 예상 길이 약 {minutes:.1f}분 · "
                   f"예상 요금 약 ${usd:.3f} (추정 — 실제 청구는 {listen.provider_name(model)} 사용량 화면에서 확인)")
        if saved.warnings:
            st.warning("코드 검사에서 남은 점 — 본문에 없는 내용일 수 있으니 아래 대본을 읽어 보고, 이상하면 '대본 다시 쓰기'를 눌러 주세요:\n"
                       + "\n".join(f"- {w}" for w in saved.warnings))
        _show_script(saved)
        c1, c2 = st.columns(2)
        if c1.button(f"🔊 음성 만들기 (2/2 · 약 ${usd:.3f} 추정)", key=f"listen_tts_{rid}", type="primary", use_container_width=True):
            try:
                with st.spinner("음성 만드는 중… (조각마다 몇 초씩, 보통 1분 안)"), usage.scope(
                        team_id=run.get("team_id", ""), run_id=rid, purpose="듣기용 MP3 음성"):
                    listen.make_audio(saved.script, mp3_path, models, model)
                listen.save_script(json_path, replace(saved, audio_model=model))  # 어느 모델로 만든 음성인지 (화면에 보인다)
            except listen.ListenError as e:
                error = str(e)
            except Exception as e:  # noqa: BLE001 — 파일 쓰기 등 TTS 밖의 오류도 화면에 이유만
                error = f"음성 파일을 만들지 못했습니다 ({type(e).__name__}: {str(e)[:200]}). 잠시 뒤 다시 눌러 주세요."
            on_calls(run)
            if not error:
                st.rerun()
        if c2.button("📝 대본 다시 쓰기", key=f"listen_rewrite_{rid}", use_container_width=True):
            json_path.unlink(missing_ok=True)
            st.rerun()
        if error:
            st.error(error)


def history_listen(folder: Path, run_id: str) -> None:
    """🕘 지난 업무 창 — 이 업무로 만든 듣기용 MP3 (없으면 아무것도 그리지 않는다)."""
    path = listen.saved_audio(folder, run_id)
    if path:
        data = path.read_bytes()
        st.markdown("##### 🎧 듣기용 MP3")
        st.audio(data, format="audio/mpeg")
        st.download_button("⬇ MP3 받기", data=data, file_name=path.name, mime="audio/mpeg",
                           key=f"hist_listen_{run_id}")


LISTEN_SOURCES = {"catalog": "📚 카탈로그 개념", "text": "✍️ 직접 적은 글"}


def catalog_listen(ctx: CatalogContext | None, team_id: str, folder: Path, models) -> None:
    """🎧 개념 듣기 — 카탈로그 개념 하나나 직접 적은 글로 듣기용 MP3 (결과 화면과 같은 listen_panel: 말투 → 대본 → 예상 요금 확인 → 음성).

    한 번에 하나씩, 버튼을 눌렀을 때만 만든다 (여러 개를 한꺼번에 만드는 기능은 요금 때문에 두지 않는다).
    파일 이름의 업무 번호 자리는 listen.CATALOG_RUN · listen.TEXT_RUN — 같은 개념을 고르거나 같은 글을 다시 넣으면 저장한 대본·MP3 를
    그대로 쓰고, 본문이 바뀌면(채택·직접 수정·글 고침) 지문이 달라져 새로 만든다. 카탈로그 개념은 항목의 범주 폴더에,
    적은 글은 listen.OWN_TEXT 폴더에 저장한다. 업무가 아니라 ⏱ 호출 기록은 저장하지 않고 🏠 사용량에만 남는다.
    ctx 가 None(카탈로그 폴더가 없음)이어도 적은 글은 된다 — 카탈로그 개념 쪽에서만 폴더를 맞추라고 알린다.
    """
    # Streamlit 은 이번 실행에서 그리지 않은 위젯의 값을 지운다 → 카탈로그 ↔ 적은 글을 오가면 적던 글·말투·고른 개념이 사라진다.
    # 그리기 전에 값을 다시 넣어 두면 지우지 않는다 (Streamlit 문서의 '위젯 정리 막기' 방법 — 1.54~1.66 모두 같음)
    for key in (f"cl_text_{team_id}", f"cl_target_{team_id}", f"listen_level_{listen.TEXT_RUN}", f"listen_level_{listen.CATALOG_RUN}"):
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]
    source = st.radio("무엇으로 만들까요", list(LISTEN_SOURCES), format_func=LISTEN_SOURCES.get, key=f"cl_source_{team_id}",
                      horizontal=True)
    if source == "text":
        text = st.text_area("들을 글 — 개념 설명이나 공부할 내용을 붙여 넣으세요 (첫 줄이 파일 이름이 됩니다)", key=f"cl_text_{team_id}",
                            height=220, placeholder="예: 다른 곳에서 정리한 개념 설명, 수업 메모, 책에서 옮겨 적은 글").strip()
        if not text:
            st.caption("글을 넣으면 말투(쉽게·보통)를 고르고 대본 → 음성 순서로 만들 수 있습니다. 넣기만 해서는 모델을 부르지 않습니다.")
            return
        if len(text) > listen.ENTRY_LIMIT:
            st.warning(f"글이 {len(text):,}자로 깁니다. 대본은 앞 {listen.ENTRY_LIMIT:,}자만 보고 씁니다 — 나머지는 나눠서 따로 만들어 주세요.")
        listen_panel(text, {"run_id": listen.TEXT_RUN, "team_id": team_id}, folder, models, lambda run: None, listen.OWN_TEXT)
        return
    if ctx is None:
        st.warning("사이드바의 '📁 개념 카탈로그' 폴더를 먼저 맞춰 주세요. ✍️ 직접 적은 글은 카탈로그 없이도 만들 수 있습니다.")
        return
    entries = ctx.catalog.entries
    if not entries:
        st.info("카탈로그에 항목이 없습니다. 사이드바 '📁 개념 카탈로그'의 폴더 경로를 확인해 주세요.")
        return
    name = st.selectbox("들을 개념 (카탈로그 항목 — 이름을 쳐서 찾을 수 있습니다)", [e.name for e in entries],
                        format_func=ctx.label, key=f"cl_target_{team_id}")
    entry = next(e for e in entries if e.name == name)
    listen_panel(entry.text, {"run_id": listen.CATALOG_RUN, "team_id": team_id}, folder, models, lambda run: None,
                 entry.category)  # 카탈로그 항목은 범주를 안다 → results/<팀>/<범주>/


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
    """변경 내용 확인 → 💾 커밋 → 🚀 푸시 (버튼을 눌러야만 실행). 사이드바의 '📝 커밋 안 된 변경 올리기' 상자 하나에서만 쓴다."""
    diff, pending = git_diff_status(ctx)
    with st.expander("변경 내용 (git diff)"):
        st.code(diff[:30000] or "(차이 없음 — 이미 커밋했거나 새 파일만 있음)", language="diff")
    st.text_input("커밋 메시지", value=message, key=f"msg_{key}")  # 값은 버튼 콜백(_commit_clicked)이 key 로 읽는다
    st.caption("📝 아직 커밋하지 않은 변경이 있습니다 — '🚀 커밋 후 푸시'를 누르면 커밋과 GitHub 업로드를 한 번에 합니다."
               if pending else "커밋하지 않은 변경 없음")
    c1, c2 = st.columns(2)
    c1.button("💾 커밋만", use_container_width=True, key=f"commit_{key}", disabled=not pending,
              on_click=_commit_clicked, args=(ctx, key, False))
    c2.button("🚀 커밋 후 GitHub 에 푸시", type="primary", use_container_width=True, key=f"push_{key}",
              on_click=_commit_clicked, args=(ctx, key, True))


def _commit_clicked(ctx: CatalogContext, key: str, push: bool) -> None:
    """커밋 버튼 콜백 — 화면을 그리기 전에 실행되므로 사이드바의 '커밋 안 된 변경' 목록과 본문 안내가 커밋 뒤 상태로 그려진다.

    버튼을 그린 뒤에 커밋하면 옛 목록·상자가 화면에 남아 다시 누르게 되고, st.rerun() 으로 새로 그리면 그 아래 본문 입력칸의 선택이 사라진다.
    결과 글은 SIDEBAR_MSGS 에 넘겨 사이드바 맨 아래에 보여 준다.
    """
    msgs: list[tuple[str, str]] = []
    if cat.git_status(ctx.repo, ctx.subdir):  # 커밋 안 된 변경이 있으면 먼저 커밋 — 커밋 없이 푸시하면 아무것도 올라가지 않는다
        ok, log = cat.git_commit(ctx.repo, ctx.subdir, st.session_state.get(f"msg_{key}", ""))
        refresh_git()
        msgs = [("success", log[-800:])] if ok else [("error", "커밋 실패: " + log[-800:])]
        push = push and ok
    if push:
        msgs = push_messages(ctx.repo, ctx.subdir, ctx.root)
    st.session_state[SIDEBAR_MSGS] = msgs


def commit_hint(ctx: CatalogContext) -> None:
    """반영 뒤 안내 — 커밋·푸시 상자는 사이드바 한 곳에만 둔다 (2026-11-12: 채택 직후 본문과 사이드바에 상자가 두 개 보여 헷갈렸다).

    사이드바 상자는 커밋 안 된 변경이 있는 동안 어느 화면에서나, 앱을 다시 켠 뒤에도 보인다. 커밋만 한 뒤의 '🚀 지금 푸시'도 사이드바에 있다."""
    if ctx.repo is None:
        st.caption("git 저장소가 아니라 커밋·푸시는 할 수 없습니다.")
    elif git_diff_status(ctx)[1]:
        st.info("⬅️ 커밋하고 GitHub 에 올리려면 왼쪽 사이드바의 '📝 커밋 안 된 변경 올리기'를 쓰세요. "
                "(사이드바가 접혀 있으면 화면 왼쪽 위의 화살표를 눌러 펼칩니다)")
    else:
        st.caption("커밋 안 된 변경 없음 — 이미 커밋했습니다. GitHub 에 올렸는지는 왼쪽 사이드바에서 볼 수 있습니다.")


def adopted_section(ctx: CatalogContext | None, run: dict) -> None:
    """채택 후 — 카탈로그 반영 결과 + 어디서 커밋·푸시하는지 안내 (상자는 사이드바에)."""
    st.markdown("##### 📚 카탈로그 반영")
    if run.get("concept_error"):
        st.error(run["concept_error"])
    if ctx is None:
        st.warning("카탈로그 폴더를 찾지 못해 반영하지 못했습니다 (사이드바의 경로 확인).")
        return
    render_changes(run.get("concept_changes") or [], run.get("concept_edit_logs") or [])
    if run.get("concept_changes"):
        commit_hint(ctx)
