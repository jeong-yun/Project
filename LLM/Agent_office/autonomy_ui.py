"""🤖 자율 실행 상태 패널 (🏠 메인 화면) — 읽기 전용.

켜짐 여부 · 작업별 마지막·다음 실행 · 오늘 비용 대 예산 · 감사 로그 해시 체인 검증 · 사람 승인 대기열을 보여 준다.
감사 로그(data/autonomy/audit.jsonl)와 사용량 DB 만 읽는다. 켜고 끄기·작업 추가는 config/settings.yaml 의 autonomy 에서 한다.
'조용한 실패'(몇 주째 안 돌았는데 모름)를 막으려고 만든 화면이다 (설계 문서 docs/AI_전용_업무_설계.md 6장).
"""

import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from core import autonomy
from core.storage import Storage

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


def render(settings: dict, data_dir: Path, storage: Storage, now: float | None = None) -> None:
    cfg = autonomy.load_config(settings)
    audit = autonomy.AuditLog(Path(data_dir) / cfg.audit_file)
    now = time.time() if now is None else now
    ok, bad_line = audit.verify()
    queue = autonomy.approval_queue(audit)
    today = autonomy.day_start(now)
    spent = sum(r["cost_usd"] or 0.0 for r in storage.usage_summary(since=today))
    runs_today = audit.runs_since(today)

    state = "켜짐" if cfg.enabled else "꺼짐"
    log_text = "감사 로그 온전" if ok else f"⚠️ 감사 로그 {bad_line}번째 줄부터 어긋남"
    title = f"🤖 자율 실행 · {state} · 작업 {len(cfg.jobs)}개 · 승인 대기 {len(queue)}건 · {log_text}"
    with st.expander(title, expanded=bool(queue) or not ok):
        st.caption("읽기 전용 — 켜고 끄기·작업은 `config/settings.yaml` 의 `autonomy`, 실행은 `scripts/run_autonomous.py` "
                   "(작업 스케줄러가 1시간마다 부르면 때가 된 작업만). 설계: `docs/AI_전용_업무_설계.md`")
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

        st.markdown(f"##### 👤 승인 대기열 ({len(queue)}건)")
        st.caption("자율 수준(observe·propose) 밖의 행동은 실행하지 않고 여기에만 남깁니다 (기본 거부). 승인·기각 처리는 다음 단계입니다.")
        if queue:
            st.dataframe([{"요청 시각": _when(q["ts"]), "작업": q["job"], "행동": q.get("action", ""), "내용": q.get("summary", "")}
                          for q in queue[:50]], hide_index=True, use_container_width=True)
