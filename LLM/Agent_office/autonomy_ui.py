"""🤖 자율 실행 상태 패널 (🏠 메인 화면) + 👤 승인 대기열 (여기와 🖥️ 커맨드 센터 ⏱·🔔 탭이 같이 쓴다).

켜짐 여부 · 작업별 마지막·다음 실행 · 오늘 비용 대 예산 · 감사 로그 해시 체인 검증 · 사람 승인 대기열을 보여 준다.
켜고 끄기·작업 추가는 config/settings.yaml 의 autonomy 에서 한다 (화면에서 바꾸지 않음).
'조용한 실패'(몇 주째 안 돌았는데 모름)를 막으려고 만든 화면이다 (설계 문서 docs/AI_전용_업무_설계.md 6장).

이 화면이 쓰는 것은 승인 대기열의 결정뿐이다 — [✅ 승인] → 확인 → 감사 로그에 approval_granted + 실행 결과,
[❌ 기각] → approval_rejected. 무엇을 실행하는지는 core/approvals.py 의 실행기 목록이 정한다 (설계 문서 5.5장).
"""

import re
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from core import approvals, autonomy
from core.storage import Storage

DECISION_TEXT = {"granted": "✅ 승인", "rejected": "❌ 기각", "expired": "⌛ 만료"}
OUTCOME_TEXT = {"action_executed": "실행됨", "action_failed": "⚠️ 실행 실패", "action_not_executed": "기록만 (사람이 직접)"}
ACTION_ICON = {"save_proposal": "📝", "create_branch_with_patch": "🌿"}
PATCH_PREVIEW_CHARS = 6000
STATUS_TEXT = {"run_end": "✅ 끝남", "run_error": "⚠️ 오류", "skipped_budget": "⏸ 예산 초과로 건너뜀", "skipped_runs": "⏸ 실행 수 상한으로 건너뜀"}


def _when(ts: float | None) -> str:
    return f"{datetime.fromtimestamp(ts):%Y-%m-%d %H:%M}" if ts else "-"


def _result_text(rec: dict | None) -> str:
    if not rec:
        return "-"
    text = STATUS_TEXT.get(rec["event"], rec["event"])
    if rec["event"] == "run_error":
        return f"{text}: {str(rec.get('error', ''))[:80]}"
    parts = [text]
    if rec.get("harness_failed"):
        parts.append("미검증(" + ", ".join(rec["harness_failed"]) + ")")
    if "fabricated_ids" in rec:
        parts.append(f"지어낸 번호 {rec['fabricated_ids']}건")
    if "quote_mismatches" in rec:
        parts.append(f"인용 불일치 {rec['quote_mismatches']}건")
    if rec.get("cost_usd") is not None:
        parts.append(f"${float(rec['cost_usd']):.4f}")
    if rec.get("seconds") is not None:
        parts.append(f"{float(rec['seconds']):.0f}초")
    return " · ".join(parts)


def render(settings: dict, data_dir: Path, storage: Storage, now: float | None = None, expanded: bool = False) -> None:
    cfg = autonomy.load_config(settings)
    audit = autonomy.AuditLog(Path(data_dir) / cfg.audit_file)
    now = time.time() if now is None else now
    ok, bad_line = audit.verify()
    expire_due(cfg, audit, now, ok)
    queue = autonomy.approval_queue(audit)
    today = autonomy.day_start(now)
    spent = sum(r["cost_usd"] or 0.0 for r in storage.usage_summary(since=today))
    runs_today = audit.runs_since(today)

    state = "켜짐" if cfg.enabled else "꺼짐"
    log_text = "감사 로그 온전" if ok else f"⚠️ 감사 로그 {bad_line}번째 줄부터 어긋남"
    title = f"🤖 자율 실행 · {state} · 작업 {len(cfg.jobs)}개 · 승인 대기 {len(queue)}건 · {log_text}"
    # 마지막 요청을 결정해 대기열이 비어도 결과 알림을 볼 수 있게 펼친 채로 둔다
    with st.expander(title, expanded=expanded or bool(queue) or not ok or _flash_key("home") in st.session_state):
        st.caption("켜고 끄기·작업은 `config/settings.yaml` 의 `autonomy`, 실행은 `scripts/run_autonomous.py` "
                   "(작업 스케줄러가 1시간마다 부르면 때가 된 작업만). 화면에서 하는 일은 승인 대기열의 승인·기각뿐입니다. "
                   "설계: `docs/AI_전용_업무_설계.md`")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("상태", state)
        c2.metric("오늘 추정 비용 / 예산", f"${spent:.4f}", f"예산 ${cfg.daily_budget_usd:g}", delta_color="off",
                  help="오늘(현지 날짜) 모든 LLM 호출 합계 — 이 값이 예산 이상이면 새 자율 실행을 시작하지 않습니다.")
        c3.metric("오늘 자율 실행", f"{runs_today}회", f"상한 {cfg.max_runs_per_day}회", delta_color="off")
        c4.metric("감사 로그", "온전" if ok else "어긋남", f"{len(audit.entries())}줄" if ok else f"{bad_line}번째 줄",
                  delta_color="off" if ok else "inverse")
        if not ok:
            st.error(f"감사 로그(해시 체인)가 {bad_line}번째 줄부터 맞지 않습니다 — 누가 고치거나 지웠을 수 있습니다. "
                     "이 상태에서는 자율 실행 스크립트가 실행을 멈춥니다.")

        plan = autonomy.schedule(cfg, audit, now)
        st.markdown("##### 작업")
        if plan:
            st.dataframe(
                [{"작업": p["job"], "팀/업무": f"{p['team']}/{p['task'] or '자유 질문'}", "수준": p["level"],
                  "주기": f"{p['every_hours']:g}시간", "마지막 실행": _when(p["last"]),
                  "다음 실행": ("지금 (때가 됨)" if p["due"] else _when(p["next"])) if cfg.enabled else "꺼짐",
                  "마지막 결과": _result_text(p["last_result"])} for p in plan],
                hide_index=True, use_container_width=True,
            )
        else:
            st.caption("등록된 작업이 없습니다. `config/settings.yaml` 의 `autonomy.jobs` 주석에 🛡️ 보안 거버넌스 팀 주 1회 예시가 있습니다.")

        render_queue(cfg, audit, "home", now=now)


# ─── 👤 승인 대기열 ──────────────────────────────────────────────────────────


_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$])")


def _plain(text) -> str:
    """모델·요청이 낸 글을 마크다운이 아니라 글자 그대로 — 링크·이미지(밖으로 요청이 나감)·$수식·표가 끼어들지 않게."""
    return _MD_SPECIAL.sub(r"\\\1", str(text))


def _flash_key(prefix: str) -> str:
    return f"{prefix}_ap_flash"


def expire_due(cfg: autonomy.AutonomyConfig, audit: autonomy.AuditLog, now: float, ok: bool) -> None:
    """기한이 지난 요청이 있을 때만 만료로 닫는다 (실행 없음) — 커맨드 센터가 1초마다 다시 그려도 잠금을 잡지 않게."""
    limit = cfg.approvals.expiry_days * 86400
    if ok and any(now - q["ts"] >= limit for q in approvals.pending(audit)):
        approvals.expire_stale(audit, cfg.approvals, now)


def _decide(prefix: str, cfg: autonomy.AutonomyConfig, audit_path: str, rid: str, decision: str) -> None:
    """버튼 콜백 — 결정을 감사 로그에 남기고 결과를 다음 화면에 한 번 보여 준다."""
    st.session_state.pop(f"{prefix}_ap_confirm_{rid}", None)
    comment = st.session_state.get(f"{prefix}_ap_comment_{rid}", "")
    try:
        out = approvals.decide(autonomy.AuditLog(audit_path), cfg.approvals, rid, decision, comment, via="ui")
    except (approvals.ApprovalError, TimeoutError) as e:
        st.session_state[_flash_key(prefix)] = ("error", f"결정을 기록하지 않았습니다: {_plain(e)}", [])
        return
    res = out["outcome"] or {}
    if out["status"] == "already":
        msg = f"`{rid}` 은(는) 이미 결정된 요청입니다 — 다시 실행하지 않았습니다."
    elif out["status"] == "expired":
        msg = f"`{rid}` 은(는) {cfg.approvals.expiry_days:g}일이 지나 만료되었습니다 — 실행하지 않았습니다."
    elif out["status"] == "rejected":
        msg = f"❌ `{rid}` 기각 — 실행하지 않습니다."
    else:
        msg = f"✅ `{rid}` 승인 — {OUTCOME_TEXT.get(res.get('event'), '결과 없음')}"
        if res.get("error"):
            msg += f": {_plain(res['error'])}"
    kind = "error" if res.get("event") == "action_failed" else "success"
    st.session_state[_flash_key(prefix)] = (kind, msg, res.get("manual_steps") or [])


def _ask_confirm(prefix: str, rid: str, on: bool) -> None:
    st.session_state[f"{prefix}_ap_confirm_{rid}"] = on


def _preview(audit: autonomy.AuditLog, req: dict) -> None:
    """요청 내용 미리보기 — 패치는 diff 로, 그 밖의 내용은 JSON 으로."""
    payload, problem = approvals.load_payload(audit, req)
    if problem:
        st.warning(f"내용 확인 실패: {_plain(problem)}")
    if payload is None:
        return
    try:
        prop = approvals.Proposal(**payload)
    except ValueError:
        st.json(payload, expanded=False)
        return
    st.markdown(f"**{_plain(prop.title)}** · 대상 {_plain(prop.target_file)}")
    if prop.rationale:
        st.caption(f"근거: {_plain(prop.rationale[:400])}")
    if prop.findings:
        st.caption("고치려는 탐지: " + " · ".join(_plain(f) for f in prop.findings[:8]))
    if prop.rescan:
        r = prop.rescan
        st.caption(f"재검사 ({_plain(r.scanner)}): {'✅ 통과' if r.passed else '⚠️ 실패'} · 전 {len(r.before)}건 → 후 {len(r.after)}건"
                   + (f" (남음: {_plain(', '.join(r.after[:5]))})" if r.after else ""))
    if prop.patch:
        text = prop.patch
        if len(text) > PATCH_PREVIEW_CHARS:
            text = text[:PATCH_PREVIEW_CHARS] + "\n… (잘림 — 전체는 승인 뒤 proposals 폴더)"
        st.code(text, language="diff", height=min(60 + 20 * text.count("\n"), 360))
    elif prop.new_content:
        st.code(prop.new_content[:PATCH_PREVIEW_CHARS], language=None, height=240)


def _item(prefix: str, cfg: autonomy.AutonomyConfig, audit: autonomy.AuditLog, entries: list[dict], req: dict) -> None:
    rid, action = req["id"], req.get("action", "")
    run_cost, total = approvals.costs(entries, req)
    icon = ACTION_ICON.get(action, "⛔" if action in autonomy.IRREVERSIBLE_ACTIONS else "❔")
    with st.container(border=True):
        team_task = f" ({req['team']}/{req.get('task') or '자유 질문'})" if req.get("team") else ""
        st.markdown(f"**{icon} {_plain(action)}** · 작업 {_plain(req.get('job', '') + team_task)} · "
                    f"요청 {_when(req['ts'])} · id `{rid}`")
        if req.get("summary"):
            st.markdown(_plain(req["summary"]))
        st.caption(f"왜 승인이 필요한가: {_plain(approvals.why(req))}")
        st.caption(f"승인하면: {_plain(approvals.on_approve(action))}")
        run_text = "(아직 끝나지 않음)" if run_cost is None else _plain(f"${run_cost:.4f}")
        st.caption(f"비용 — 이 요청을 낸 실행 {run_text} · 이 작업의 자율 실행 합계 {_plain(f'${total:.4f}')}")
        _preview(audit, req)
        st.text_input("메모 (선택)", key=f"{prefix}_ap_comment_{rid}", placeholder="승인·기각 이유 — 감사 로그에 남습니다")
        args = (prefix, cfg, str(audit.path), rid)
        if st.session_state.get(f"{prefix}_ap_confirm_{rid}"):
            st.warning(f"정말 승인할까요? {_plain(approvals.on_approve(action))}")
            c1, c2 = st.columns(2)
            c1.button("확인: 승인", key=f"{prefix}_ap_confirm_{rid}_go", type="primary", on_click=_decide,
                      args=(*args, "grant"), use_container_width=True)
            c2.button("취소", key=f"{prefix}_ap_cancel_{rid}", on_click=_ask_confirm, args=(prefix, rid, False),
                      use_container_width=True)
        else:
            c1, c2 = st.columns(2)
            c1.button("✅ 승인", key=f"{prefix}_ap_grant_{rid}", on_click=_ask_confirm, args=(prefix, rid, True),
                      use_container_width=True)
            c2.button("❌ 기각", key=f"{prefix}_ap_reject_{rid}", on_click=_decide, args=(*args, "reject"),
                      use_container_width=True)


def _outcome_text(h: dict) -> str:
    res = h["outcome"]
    if h["decision"] != "granted":
        return "실행 없음"
    if not res:
        return "결과 기록 없음 (중단됨)"
    text = OUTCOME_TEXT.get(res["event"], res["event"])
    if res.get("error"):
        return f"{text}: {res['error'][:80]}"
    result = res.get("result") or {}
    where = result.get("branch") or result.get("export_dir")
    return f"{text} · {where}" if where else text


def render_queue(cfg: autonomy.AutonomyConfig, audit: autonomy.AuditLog, prefix: str, now: float | None = None,
                 show_history: bool = True) -> None:
    """👤 승인 대기열 — 요청마다 무엇·왜·미리보기·비용 + [✅ 승인]→확인 / [❌ 기각], 아래에 결정 이력.

    prefix: 위젯 key 앞머리 — 같은 요청이 🏠 패널과 커맨드 센터 탭에 함께 그려져도 key 가 겹치지 않게.
    """
    now = time.time() if now is None else now
    ok, bad_line = audit.verify()
    expire_due(cfg, audit, now, ok)
    entries = audit.entries()
    queue = approvals.pending(audit, entries)

    st.markdown(f"##### 👤 승인 대기열 ({len(queue)}건)")
    st.caption("자율 수준 밖의 행동은 실행하지 않고 여기에 남깁니다 (기본 거부). 승인해도 실행하는 것은 로컬·되돌릴 수 있는 "
               "행동(제안 저장, 새 로컬 브랜치에 커밋)뿐이고, push·merge·삭제·외부 전송·설정 적용·배포는 기록만 남기고 "
               f"사람이 할 명령을 보여 줍니다. {cfg.approvals.expiry_days:g}일 동안 결정하지 않으면 만료됩니다.")
    flash = st.session_state.pop(_flash_key(prefix), None)
    if flash:
        kind, msg, steps = flash
        (st.error if kind == "error" else st.success)(msg)
        if steps:
            st.markdown("사람이 직접 할 일 (앱은 실행하지 않음):")
            st.code("\n".join(steps), language="bash")
    if not ok:
        st.error(f"감사 로그가 {bad_line}번째 줄부터 어긋나 승인·기각을 받지 않습니다.")
    elif not queue:
        st.caption("결정을 기다리는 요청이 없습니다.")
    for req in queue[:20] if ok else []:
        _item(prefix, cfg, audit, entries, req)
    if ok and len(queue) > 20:
        st.caption(f"… 외 {len(queue) - 20}건 — `python scripts/run_autonomous.py --approvals`")

    hist = approvals.history(audit, entries)
    if show_history and hist:
        st.markdown(f"###### 결정 이력 ({len(hist)}건)")
        st.dataframe([{"결정 시각": _when(h["decided"]), "요청 id": h["id"], "행동": h["action"], "작업": h["job"],
                       "결정": DECISION_TEXT[h["decision"]], "결과": _outcome_text(h), "메모": h["comment"],
                       "경로": {"ui": "화면", "cli": "CLI"}.get(h["via"], h["via"] or "자동"),
                       "다음 할 일 (사람)": " / ".join((h["outcome"] or {}).get("manual_steps") or [])}
                      for h in hist[:30]], hide_index=True, use_container_width=True)
