"""📜 프롬프트·📚 교훈 — 부서 화면에서 팀원마다 실제로 받는 시스템 프롬프트를 보고, 교훈을 관리·정리한다.

보이는 프롬프트 = 실행 때 보내는 프롬프트: 실행(core/workflow.py)과 같은 core/feedback.team_prompts 로 만든다
(페르소나·직급·전문 분야·업무 규칙·스킬 + 글자 한도 안에서 실제로 들어가는 교훈). 한도를 넘어 빠진 교훈도 보여 준다.
교훈은 사용자가 버튼을 누를 때만 바뀐다 — 고치기 · 끄기/켜기 · 지우기(한 번 더 확인) · 정리 제안 반영(지우지 않고 끔).

app.py 가 다시 그릴 때마다 configure() 로 저장소·설정을 넘긴다. 개념 편집 팀 '📜 적용된 프롬프트'(concept_ui.py)도
member_prompts() 를 같이 쓴다.
"""

from dataclasses import dataclass, field

import streamlit as st

from core import verification
from core.config import load_models, load_ranks, team_org
from core.feedback import (
    DEFAULT_LESSON_CHARS, AgentPrompt, PromptLessons, active_lessons, apply_cleanup, lesson_origin, orphan_lessons, propose_cleanup,
    split_lesson, suggest_owner, team_lessons, team_prompts, undo_merge,
)
from core.schemas import CleanupProposal, ModelsConfig, Rank, ReviewResult, Team

RELATION = {"overlap": "겹침", "conflict": "충돌"}
DROP_ORDER = ("한도를 넘으면 규칙 문장(교훈 한 문장)은 끝까지 남기고, 세부 코멘트(사람 코멘트 원문)부터 뺍니다. "
              "빼는 순서: ① 이번 질문과 관련이 낮은 것 ② 같은 피드백을 받은 횟수가 적은 것 ③ 오래된 것.")
KIND = {"praise": "👍 유지할 점", "improve": "🔧 고칠 점"}
PRIVATE_NOTE = "'(사람 코멘트: …)' 뒤의 글은 공개 저장소 knowledge/ 파일에 올라가지 않습니다 (개인 정보 보호)."


@dataclass
class Ctx:
    storage: object | None
    ranks: list[Rank]
    models: ModelsConfig
    budget: int = DEFAULT_LESSON_CHARS
    verification_cfg: dict = field(default_factory=dict)


_ctx: Ctx | None = None


def configure(ctx: Ctx) -> None:
    global _ctx
    _ctx = ctx


def _context() -> Ctx:
    """app.py 밖(개념 팀 결과 화면 단독 테스트 등)에서는 설정만 읽고 저장소 없이 — 교훈 없이 보인다."""
    return _ctx or Ctx(storage=None, ranks=load_ranks(), models=load_models())


def _names(team: Team, ranks: list[Rank]) -> dict[str, str]:
    by_id = {r.id: r for r in ranks}
    return {n.id: n.name or by_id[n.rank].name for n in team_org(team, ranks).walk()}


def _flash(key: str, text: str) -> None:
    st.session_state[f"prompt_msg_{key}"] = text


def _show_flash(key: str) -> None:
    text = st.session_state.pop(f"prompt_msg_{key}", None)
    if text:
        st.success(text)


# ─── 글자 한도 경고 (부서 화면 맨 위) ─────────────────────────────────────


def _dropped(pick: PromptLessons) -> str:
    """빠진 것 한 줄 — '빠진 교훈 N개' (규칙 문장까지 통째로) · '세부 코멘트만 빠진 교훈 N개'."""
    parts = [f"빠진 교훈 {len(pick.left_out)}개" if pick.left_out else "",
             f"세부 코멘트만 빠진 교훈 {len(pick.cut)}개" if pick.cut else ""]
    return " · ".join(p for p in parts if p)


def budget_warning(teams: list[Team], where: str) -> None:
    """부서 화면 맨 위 — 프롬프트에 들어가지 못하는 교훈이 있으면 알린다 (글자 한도 초과 · 지금 없는 팀원 id)."""
    ctx = _context()
    if ctx.storage is None:
        return
    over = [f"{team.name} {_names(team, ctx.ranks).get(agent, agent)}({_dropped(pick)})"
            for team in teams for agent, pick in team_lessons(ctx.storage, team.id, ctx.budget).items() if _dropped(pick)]
    if over:
        st.warning(f"⚠️ 교훈이 많아 글자 한도({ctx.budget:,}자)를 넘었습니다: {', '.join(over)}  \n{DROP_ORDER}  \n"
                   f"{where}에서 무엇이 빠지는지 보고, 필요 없는 교훈은 꺼 주세요. "
                   "같은 피드백이 다시 오면 결정 때 자동으로 하나로 합쳐집니다.")
    lost = [f"{t.name} {sum(map(len, o.values()))}개" for t in teams if (o := orphan_lessons(ctx.storage, t, ctx.ranks))]
    if lost:
        st.warning(f"⚠️ 지금 조직도에 없는 팀원 이름으로 남은 교훈이 있어 어느 프롬프트에도 들어가지 않습니다: {', '.join(lost)}  \n"
                   "아래 '📜 프롬프트·📚 교훈 보기'를 켜면 그 팀 탭 맨 위에서 받을 팀원을 골라 옮길 수 있습니다.")


# ─── 팀원별 프롬프트 보기 ──────────────────────────────────────────────


def _lesson_note(ap: AgentPrompt) -> str:
    pick = ap.lessons
    n = len(pick.kept)
    return f"교훈 {n}개 · {pick.chars:,}자 (한도 {pick.budget:,}자)" if n else "교훈 없음"


def member_prompts(team: Team, key: str, manage: bool = False) -> None:
    """팀원마다 실행 때 받는 시스템 프롬프트 — 표 + 고른 팀원의 프롬프트 전문 (+ manage 면 교훈 관리)."""
    ctx = _context()
    prompts = team_prompts(team, ctx.ranks, ctx.models, ctx.storage, ctx.budget)
    st.dataframe(
        [{"팀원": a.name, "직급": a.rank, "전문 분야": a.specialty, "프롬프트에 넣는 교훈": _lesson_note(a),
          "한도 넘어 빠진 것": _dropped(a.lessons)} for a in prompts],
        hide_index=True, use_container_width=True,
    )
    pick = st.selectbox("팀원 고르기", range(len(prompts)), format_func=lambda i: f"{prompts[i].name} · {prompts[i].rank}",
                        key=f"prompt_agent_{key}")
    ap = prompts[pick]
    if ctx.storage is None:
        st.caption("저장소에 연결되지 않아 교훈 없이 보입니다. 실행 때는 이 끝에 지난 피드백 교훈이 붙습니다.")
    elif ap.lessons.left_out or ap.lessons.cut:
        lines = [f"⚠️ {ap.name}의 교훈이 글자 한도({ap.lessons.budget:,}자)를 넘었습니다."]
        if ap.lessons.left_out:
            lines.append(f"교훈 {len(ap.lessons.left_out)}개가 글자 한도를 넘어 규칙 문장까지 아래 프롬프트에 들어가지 않습니다: "
                         + ", ".join(f"#{r['id']}" for r in ap.lessons.left_out))
        if ap.lessons.cut:
            lines.append(f"교훈 {len(ap.lessons.cut)}개는 규칙 문장만 들어가고 세부 코멘트가 빠집니다: " + ", ".join(
                f"#{r['id']}({r['created_at'][:10]}, {len(r['detail']):,}자)" for r in ap.lessons.cut))
        lines += [DROP_ORDER, "이 화면은 질문이 없어 반복 횟수·최신 순으로 고릅니다. 실제 업무에서는 그 질문과 관련 높은 세부 "
                  "코멘트가 먼저 들어가므로 빠지는 것이 다를 수 있습니다. '📚 교훈 관리'에서 필요 없는 교훈을 꺼 주세요."]
        st.warning("  \n".join(lines))
    if not manage:
        st.code(ap.prompt, language="markdown", wrap_lines=True, height=360)
        return
    view = st.radio("보기", ["📜 시스템 프롬프트", "📚 교훈 관리"], horizontal=True, key=f"prompt_view_{key}",
                    label_visibility="collapsed")
    if view == "📜 시스템 프롬프트":
        st.code(ap.prompt, language="markdown", wrap_lines=True, height=420)
        st.caption("실제 업무에서는 이 시스템 프롬프트 뒤에 업무 브리프·맡은 업무·근거 자료가 더해집니다.")
    else:
        render_lessons(team, ap, key)


# ─── 📚 교훈 관리 (팀원 한 명) ─────────────────────────────────────────


def _set_active(lesson_id: int, active: bool, key: str) -> None:
    _context().storage.update_lesson(lesson_id, active=active)
    _flash(key, f"교훈 #{lesson_id}을(를) {'켰습니다' if active else '껐습니다 — 다음 업무부터 프롬프트에 들어가지 않습니다'}.")


def _save_text(lesson_id: int, widget: str, key: str) -> None:
    text = (st.session_state.get(widget) or "").strip()
    if not text:
        _flash(key, "빈 교훈은 저장하지 않았습니다. 필요 없으면 '끄기'를 눌러 주세요.")
        return
    _context().storage.update_lesson(lesson_id, lesson=text)
    _flash(key, f"교훈 #{lesson_id}을(를) 고쳤습니다. 다음 업무부터 고친 문장이 들어갑니다.")


def _confirm_delete(lesson_id: int, ask: bool) -> None:
    """🗑 지우기 → 한 번 더 묻는 상태 / 취소 → 묻지 않는 상태."""
    st.session_state[f"lesson_del_{lesson_id}"] = ask


def _delete(lesson_id: int, key: str) -> None:
    _context().storage.delete_lesson(lesson_id)
    st.session_state.pop(f"lesson_del_{lesson_id}", None)
    _flash(key, f"교훈 #{lesson_id}을(를) 지웠습니다.")


def _ask_cleanup(team: Team, agent_id: str, store: str) -> None:
    ctx = _context()
    rows = active_lessons(ctx.storage, team.id).get(agent_id, [])
    proposals, notes = propose_cleanup(team, {agent_id: rows}, _names(team, ctx.ranks), ctx.models)
    st.session_state[store] = proposals
    st.session_state[f"{store}_notes"] = notes


def render_lessons(team: Team, ap: AgentPrompt, key: str) -> None:
    """팀원 한 명의 교훈 — 정리 제안 받기 · 문장 고치기 · 끄기/켜기 · 지우기(확인)."""
    ctx = _context()
    if ctx.storage is None:
        st.caption("저장소에 연결되지 않아 교훈을 볼 수 없습니다.")
        return
    _show_flash(key)
    rows = ctx.storage.list_lessons(team.id, ap.node_id)  # 최신 먼저 (꺼진 것 포함)
    kept = {r["id"] for r in ap.lessons.kept}
    cut = {r["id"] for r in ap.lessons.cut}
    st.caption(f"켜 둔 교훈 {sum(bool(r['active']) for r in rows)}개 · 꺼 둔 교훈 {sum(not r['active'] for r in rows)}개 · "
               f"프롬프트에 넣는 교훈 {len(kept)}개 ({ap.lessons.chars:,}자 / 한도 {ap.lessons.budget:,}자)")
    store = f"cleanup_{team.id}_{ap.node_id}"
    st.button("🧹 정리 제안 받기", key=f"ask_{store}", on_click=_ask_cleanup, args=(team, ap.node_id, store),
              disabled=sum(bool(r["active"]) for r in rows) < 2,
              help="보조 모델이 겹치거나 서로 어긋나는 교훈을 찾아 합친 문장을 제안합니다. 누르기만 해서는 아무것도 바뀌지 않습니다.")
    if store in st.session_state:
        for note in st.session_state.get(f"{store}_notes") or []:
            st.caption(f"참고: {note}")
        if not render_cleanup(team, st.session_state[store], store):
            st.caption("겹치거나 어긋나는 교훈을 찾지 못했습니다.")
    for row in rows:
        lid = row["id"]
        state = ("✂ 규칙 문장만 들어감 (세부 코멘트는 한도 넘어 빠짐)" if lid in cut else "✅ 프롬프트에 들어감" if lid in kept
                 else "⚠️ 한도 넘어 빠짐") if row["active"] else "⏸ 꺼 둠"
        hits = f" · 🔁 같은 피드백 {row['hits']}회" if row.get("hits", 1) > 1 else ""
        with st.container(border=True):
            st.caption(f"#{lid} · {row['created_at'][:10]} · {KIND.get(row['kind'], row['kind'])} · {state}{hits} · {len(row['lesson']):,}자")
            widget = f"lesson_text_{lid}"
            st.text_area("교훈 문장", value=row["lesson"], key=widget, height=110 if len(row["lesson"]) > 200 else 68,
                         label_visibility="collapsed")
            c1, c2, c3 = st.columns(3)
            c1.button("💾 고친 문장 저장", key=f"lesson_save_{lid}", on_click=_save_text, args=(lid, widget, key),
                      use_container_width=True)
            c2.button("⏸ 끄기" if row["active"] else "▶ 켜기", key=f"lesson_toggle_{lid}", on_click=_set_active,
                      args=(lid, not row["active"], key), use_container_width=True,
                      help="끈 교훈은 프롬프트에 들어가지 않지만 지워지지 않아 다시 켤 수 있습니다.")
            if st.session_state.get(f"lesson_del_{lid}"):
                st.warning("정말 지울까요? 지우면 되돌릴 수 없습니다. 다시 쓸 수도 있다면 '끄기'를 권합니다.")
                d1, d2 = st.columns(2)
                d1.button("🗑 지우기 확인", key=f"lesson_del_ok_{lid}", on_click=_delete, args=(lid, key), type="primary",
                          use_container_width=True)
                d2.button("취소", key=f"lesson_del_no_{lid}", on_click=_confirm_delete, args=(lid, False),
                          use_container_width=True)
            else:
                c3.button("🗑 지우기", key=f"lesson_del_btn_{lid}", on_click=_confirm_delete, args=(lid, True),
                          use_container_width=True)
    st.caption(PRIVATE_NOTE)


# ─── 🧹 정리 제안 (결정 직후 · 교훈 관리) ─────────────────────────────────


def _proposal_key(p: CleanupProposal) -> str:
    return f"{p.team_id}_{p.agent_id}_{'-'.join(map(str, p.ids))}"


def _apply(p: CleanupProposal, choice: str, widget: str, key: str, keep: int | None = None) -> None:
    text = st.session_state.get(widget, "") if choice == "merge" else ""
    _flash(key, apply_cleanup(_context().storage, p, choice, text, keep))
    st.session_state.setdefault("cleanup_done", set()).add(_proposal_key(p))


def _origin_lines(row: dict) -> None:
    """충돌하는 교훈 한 개 — 교훈 문장 + 그 교훈이 나온 원래 피드백 (결정·날짜·실행·질문·코멘트)."""
    o = lesson_origin(_context().storage, row)
    rule, _ = split_lesson(row["lesson"])
    st.markdown(f"**#{row['id']}** {KIND.get(row['kind'], row['kind'])} — {rule}")
    decided = o["action"] + (f" (만족도 {o['score']}/5)" if o["action"] and o["score"] else "")
    question = f"질문: {o['question'][:80]}{'…' if len(o['question']) > 80 else ''}" if o["question"] else ""
    run = f"실행 `{o['run_id'][:8]}`" if o["run_id"] else ""
    st.caption("원래 피드백: " + " · ".join(x for x in (o["date"], decided, run, question) if x))
    if o["comment"]:
        with st.expander("그때 남긴 코멘트 보기"):
            st.text(o["comment"])


def render_cleanup(team: Team, proposals: list[CleanupProposal], key: str) -> int:
    """정리 제안마다 원래 교훈 · 합친 문장(고칠 수 있음) · 버튼. 버튼을 누르기 전에는 아무것도 바꾸지 않는다.

    겹침: [합친 문장으로 바꾸기 / 옛 교훈만 끄기 / 둘 다 유지]
    충돌: 교훈마다 원래 피드백(결정·날짜·실행·코멘트)을 보여 주고 [#번호만 남기기 … / 합친 문장으로 바꾸기 / 둘 다 유지]
    이미 처리했거나 그사이 꺼진 교훈이 낀 제안은 숨긴다. 보인 제안 수를 돌려준다.
    """
    ctx = _context()
    if ctx.storage is None or not proposals:
        return 0
    _show_flash(key)
    done = st.session_state.get("cleanup_done", set())
    names = _names(team, ctx.ranks)
    active = {r["id"]: r for rows in active_lessons(ctx.storage, team.id).values() for r in rows}
    shown = 0
    for p in proposals:
        pk = _proposal_key(p)
        if pk in done or not all(i in active for i in p.ids):
            continue
        shown += 1
        conflict = p.relation == "conflict"
        widget = f"cleanup_text_{key}_{pk}"
        with st.container(border=True):
            new = f" — 이번에 생긴 교훈 #{p.new_id} 포함" if p.new_id else ""
            st.markdown(f"**{'⚖️' if conflict else '🧹'} {names.get(p.agent_id, p.agent_id)}** · {RELATION[p.relation]} · "
                        f"교훈 {len(p.ids)}개{new}")
            st.caption(f"이유: {p.reason or '-'} · 찾은 방법: {'보조 모델' if p.by == 'model' else '문장 비교'}")
            if conflict:  # 어느 쪽이 맞는지는 사람만 안다 — 원래 피드백을 나란히 보여 주고 고르게 한다
                for i in p.ids:
                    _origin_lines(active[i])
                st.caption("따를 교훈을 골라 주세요. 고르지 않은 교훈은 지우지 않고 꺼 둡니다 (📚 교훈 관리에서 다시 켤 수 있음).")
                cols = st.columns(len(p.ids))
                for col, i in zip(cols, p.ids):
                    col.button(f"#{i}만 남기기{' (새 교훈)' if i == p.new_id else ''}", key=f"cleanup_only_{key}_{pk}_{i}",
                               on_click=_apply, args=(p, "only", widget, key, i), use_container_width=True)
                with st.expander("두 교훈을 한 문장으로 합치기 (직접 고쳐서)"):
                    st.text_area("합친 문장 (고쳐도 됩니다)", value=p.merged, key=widget, height=120)
                    st.button("🔗 합친 문장으로 바꾸기", key=f"cleanup_merge_{key}_{pk}", on_click=_apply,
                              args=(p, "merge", widget, key), use_container_width=True)
                st.button("둘 다 유지", key=f"cleanup_keep_{key}_{pk}", on_click=_apply, args=(p, "keep", widget, key))
                continue
            with st.expander("원래 교훈 보기"):
                for i in p.ids:
                    r = active[i]
                    st.markdown(f"- **#{i}** ({r['created_at'][:10]}{' · 새 교훈' if i == p.new_id else ''}) {r['lesson']}")
            st.text_area("합친 문장 (고쳐도 됩니다)", value=p.merged, key=widget, height=120)
            st.caption(f"'합친 문장으로 바꾸기'를 누르면 이 문장이 새 교훈이 되고, 원래 교훈은 지우지 않고 꺼 둡니다. {PRIVATE_NOTE}")
            cols = st.columns(3 if p.new_id else 2)
            cols[0].button("🔗 합친 문장으로 바꾸기", key=f"cleanup_merge_{key}_{pk}", on_click=_apply,
                           args=(p, "merge", widget, key), type="primary", use_container_width=True)
            if p.new_id:
                cols[1].button("⏸ 옛 교훈만 끄기", key=f"cleanup_off_{key}_{pk}", on_click=_apply, args=(p, "off_old", widget, key),
                               use_container_width=True, help="새 교훈은 그대로 두고, 겹치는 옛 교훈만 끕니다.")
            cols[-1].button("둘 다 유지", key=f"cleanup_keep_{key}_{pk}", on_click=_apply, args=(p, "keep", widget, key),
                            use_container_width=True)
    return shown


# ─── 🔁 결정 직후 자동으로 합친 교훈 (알림 + 되돌리기) ──────────────────────


def _undo(p: CleanupProposal, key: str) -> None:
    _flash(key, undo_merge(_context().storage, p))
    st.session_state.setdefault("merge_undone", set()).add(_proposal_key(p))


def render_merged(team: Team, merged: list[CleanupProposal], key: str) -> None:
    """같은 피드백이 다시 와서 자동으로 합친 교훈 — 무엇을 합쳤는지 알리고 되돌릴 수 있게 한다."""
    ctx = _context()
    if ctx.storage is None or not merged:
        return
    st.markdown("##### 🔁 같은 피드백이 다시 와서 기존 교훈과 합쳤습니다")
    _show_flash(key)
    names = _names(team, ctx.ranks)
    undone = st.session_state.get("merge_undone", set())
    for p in merged:
        rows = {r["id"]: r for r in ctx.storage.list_lessons(team.id, p.agent_id)}
        with st.container(border=True):
            parts = [f"#{i}{'(이번)' if i == p.new_id else ''}" for i in p.ids]
            st.markdown(f"**{names.get(p.agent_id, p.agent_id)}** · 교훈 {' · '.join(parts)} → **#{p.merged_id}** "
                        f"(같은 피드백 {p.hits}회 — 프롬프트 맨 앞에 '[중요 — 피드백 {p.hits}회 반복]'으로 들어갑니다)")
            st.caption(f"합친 이유: {p.reason or '-'} · 원래 교훈은 지우지 않고 꺼 두었습니다. 사람 코멘트 원문은 합친 교훈에 그대로 남습니다.")
            with st.expander("합친 문장 · 원래 교훈 보기"):
                st.markdown(f"**합친 문장** {p.merged}")
                for i in p.ids:
                    if i in rows:
                        st.markdown(f"- #{i} ({rows[i]['created_at'][:10]}) {rows[i]['lesson']}")
            if _proposal_key(p) not in undone:
                st.button("↩ 되돌리기", key=f"merge_undo_{key}_{_proposal_key(p)}", on_click=_undo, args=(p, key),
                          help="원래 교훈을 다시 켜고 합친 교훈을 끕니다. 어느 것도 지우지 않습니다.")


# ─── 지금 없는 팀원 id 로 남은 교훈 → 받을 팀원 고르기 ──────────────────────


def _move(rows: list[dict], widget: str, key: str) -> None:
    target = st.session_state[widget]
    for r in rows:
        _context().storage.update_lesson(r["id"], agent_id=target)
    _flash(key, f"교훈 {len(rows)}개를 옮겼습니다. 다음 업무부터 그 팀원의 프롬프트에 들어갑니다.")


def render_orphans(team: Team, key: str) -> None:
    """조직도가 바뀌어 지금 없는 팀원 id 로 남은 교훈 — 받을 팀원을 사용자가 골라야 옮긴다 (그전에는 바꾸지 않음)."""
    ctx = _context()
    if ctx.storage is None:
        return
    _show_flash(key)
    orphans = orphan_lessons(ctx.storage, team, ctx.ranks)
    if not orphans:
        return
    names = _names(team, ctx.ranks)
    members = list(names)
    st.warning(f"⚠️ 지금 조직도에 없는 팀원 이름으로 남은 교훈 {sum(map(len, orphans.values()))}개 — 어느 프롬프트에도 들어가지 않습니다. "
               "받을 팀원을 고르고 '옮기기'를 눌러 주세요 (누르기 전에는 바뀌지 않습니다). 필요 없으면 옮긴 뒤 끄면 됩니다.")
    for old, rows in orphans.items():
        with st.container(border=True):
            st.markdown(f"**옛 이름 `{old}`** · 교훈 {len(rows)}개")
            for r in rows:
                st.caption(f"#{r['id']} · {r['created_at'][:10]} · {KIND.get(r['kind'], r['kind'])} · {r['lesson']}")
            widget = f"orphan_to_{team.id}_{old}"
            c1, c2 = st.columns([3, 1])
            c1.selectbox("받을 팀원", members, index=members.index(suggest_owner(old, team, ctx.ranks)),
                         format_func=lambda n: names[n], key=widget)
            c2.button("➡ 옮기기", key=f"orphan_move_{team.id}_{old}", on_click=_move, args=(rows, widget, key),
                      use_container_width=True)


# ─── 부서 전체 (부장 + 팀마다) ─────────────────────────────────────────


def critic_prompt(team: Team, head) -> str:
    """부장이 이 팀 결과를 비평할 때의 시스템 프롬프트 — 실행과 같은 verification.build_critic_messages 로 만든다."""
    rules = verification.critic_settings(_context().verification_cfg, team.verification).get("rules")
    empty = ReviewResult(analysis="", issues=[], score=0, verdict="approve", revised_answer="", comment="")
    return str(verification.build_critic_messages("", "", empty, [], rules, "", head.critic_persona())[0].content)


def render_dept(dept, head, teams: list[Team], key: str) -> None:
    """부서 화면 '📜 프롬프트·📚 교훈' — 👔 부장의 비평 관점 + 팀마다 팀원 프롬프트·교훈 관리."""
    st.caption("팀원마다 다음 업무에서 실제로 받는 시스템 프롬프트입니다 (실행과 같은 코드로 만듦). "
               f"교훈은 팀원마다 {_context().budget:,}자 안에서 모두 들어가고, 넘치면 오래된 것부터 빠집니다.")
    tabs = st.tabs([f"👔 {head.name}", *(f"{t.name}" for t in teams)])
    with tabs[0]:
        st.markdown(f"**비평·팀 배정 관점** — {head.persona}")
        st.caption("부장은 팀의 최종 답변(검토 과정의 논리)을 채점하는 비평가입니다. 부장에게는 교훈이 들어가지 않습니다. "
                   "관점은 config/departments.yaml 의 head.persona 에서 바꿉니다.")
        pick = st.selectbox("어느 팀을 비평할 때", range(len(teams)), format_func=lambda i: teams[i].name, key=f"critic_team_{key}")
        st.code(critic_prompt(teams[pick], head), language="markdown", wrap_lines=True, height=300)
        st.caption("근거 자료(웹 검색·데이터 툴)가 있는 업무에서는 '근거 자료와 대조' 규칙 두 줄이 더해집니다.")
    for tab, team in zip(tabs[1:], teams):
        with tab:
            render_orphans(team, key=f"{key}_{team.id}_orphans")
            member_prompts(team, key=f"{key}_{team.id}", manage=True)
