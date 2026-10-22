"""🏠 메인 화면 — 모델별 토큰 사용량·추정 비용, API 키별 내역(이름 붙이기), 이전 키, 가격표.

데이터는 core/usage.py 가 모델 호출마다 기록한 data/runs.db 의 usage·api_keys 테이블에서 읽는다.
API 키 원문은 저장하지 않고 지문(sha256 앞 12자리)과 '앞 3자…뒤 4자'만 쓴다 → 사람이 키마다 이름을 붙여 구분한다.
"""

import os
from collections import defaultdict
from datetime import datetime

import streamlit as st

from core import usage
from core.storage import Storage

PERIODS = {"오늘": 1, "7일": 7, "30일": 30, "전체": None}  # 최근 며칠
PROVIDER_NAMES = {"openai": "OpenAI", "google": "Google", "anthropic": "Anthropic", "upstage": "Upstage",
                  "freellm": "무료(FreeLLMAPI)"}


def period_start(label: str) -> float | None:
    days = PERIODS[label]
    if days is None:
        return None
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    return today - (days - 1) * 86400


def current_keys(models) -> dict[str, tuple[str, str]]:
    """지금 .env 에 있는 키 — {키 id: (provider, 가린 키)}."""
    out = {}
    for provider, env in models.providers.items():
        key = os.getenv(env)
        if key:
            key_id, masked = usage.key_fingerprint(key)
            out[key_id] = (provider, masked)
    return out


def key_label(key: dict | None, key_id: str, provider: str = "") -> str:
    if not key_id:
        return f"{PROVIDER_NAMES.get(provider, provider)} (키 없음)"
    if key and key.get("label"):
        return key["label"]
    masked = key["masked"] if key else key_id[:6]
    return f"{PROVIDER_NAMES.get(key['provider'] if key else provider, provider)} 키 {masked}"


def money(value: float | None) -> str:
    if value is None:
        return "-"
    return f"${value:,.4f}" if value < 1 else f"${value:,.2f}"


def fmt_day(ts: float | None) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else "-"


def render(storage: Storage, models, open_runs: list[dict], team_names: dict, go_to_team) -> None:
    st.title("🏠 Agent Office")
    st.caption("모델별 토큰 사용량과 추정 비용, API 키별 사용 내역을 봅니다. 팀 업무는 왼쪽 '화면'에서 🏢 팀 업무를 고르세요.")

    _team_shortcuts(open_runs, team_names, go_to_team)

    now_keys = current_keys(models)
    for key_id, (provider, masked) in now_keys.items():  # 아직 호출 전이어도 이름을 미리 붙일 수 있게
        storage.remember_key(key_id, provider, masked)
    keys = {k["key_id"]: k for k in storage.api_keys()}

    period = st.radio("기간", list(PERIODS), index=2, horizontal=True, key="home_period")
    rows = storage.usage_summary(period_start(period))

    _totals(rows)
    _by_model(rows, keys, period, storage)
    _current_keys(storage, keys, now_keys, rows)
    _previous_keys(storage, keys, now_keys)
    _pricing(storage, rows)


# ─── 팀 바로가기 ──────────────────────────────────────────────────────────


def _team_shortcuts(open_runs: list[dict], team_names: dict, go_to_team) -> None:
    st.markdown("#### 🏢 팀")
    cols = st.columns(min(len(team_names), 6) or 1)
    for i, (tid, name) in enumerate(team_names.items()):
        busy = sum(1 for r in open_runs if r["team_id"] == tid)
        cols[i % len(cols)].button(name + (f" · 🔔{busy}" if busy else ""), key=f"home_team_{tid}",
                                   on_click=go_to_team, args=(tid,), use_container_width=True)


# ─── 요약 ────────────────────────────────────────────────────────────────


def _totals(rows: list[dict]) -> None:
    cost = sum(r["cost_usd"] or 0 for r in rows)
    unpriced = sum(r["unpriced_calls"] for r in rows)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("추정 비용", money(cost))
    c2.metric("호출", f"{sum(r['calls'] for r in rows):,}")
    c3.metric("입력 토큰", f"{sum(r['input_tokens'] for r in rows):,}")
    c4.metric("출력 토큰", f"{sum(r['output_tokens'] for r in rows):,}")
    st.caption("추정 비용 = 토큰 × 가격표(아래 ⚙️). 무료 등급·배치 할인·세금은 빠져 있어 실제 청구액과 다를 수 있습니다 — "
               "실제 결제 금액은 각 사 콘솔(OpenAI Usage · Google Cloud Billing · Anthropic Console)에서 확인하세요.")
    if unpriced:
        st.warning(f"가격표에 없는 모델 호출 {unpriced:,}건은 비용에 빠져 있습니다 — 아래 ⚙️ 가격표에서 가격을 넣으면 다시 계산합니다.")


def _by_model(rows: list[dict], keys: dict, period: str, storage: Storage) -> None:
    st.markdown(f"#### 📊 모델별 사용량 ({period})")
    if not rows:
        st.info("아직 기록이 없습니다. 팀에 업무를 맡기면 모델 호출마다 토큰·비용이 여기에 쌓입니다 (이 기능을 켠 뒤부터 기록).")
        return
    by_model: dict[tuple[str, str], dict] = defaultdict(lambda: {"calls": 0, "in": 0, "out": 0, "cost": 0.0, "unpriced": 0})
    for r in rows:
        m = by_model[(r["provider"], r["model"])]
        m["calls"] += r["calls"]
        m["in"] += r["input_tokens"]
        m["out"] += r["output_tokens"]
        m["cost"] += r["cost_usd"] or 0
        m["unpriced"] += r["unpriced_calls"]
    table = [
        {"모델": model, "회사": PROVIDER_NAMES.get(provider, provider), "호출": m["calls"], "입력 토큰": m["in"],
         "출력 토큰": m["out"], "추정 비용($)": round(m["cost"], 4) if m["unpriced"] < m["calls"] else None}
        for (provider, model), m in sorted(by_model.items(), key=lambda kv: -kv[1]["cost"])
    ]
    c1, c2 = st.columns([3, 2])
    c1.dataframe(table, hide_index=True, use_container_width=True)
    chart = {row["모델"]: row["추정 비용($)"] or 0 for row in table}
    if any(chart.values()):
        c2.bar_chart(chart, horizontal=True, y_label="", x_label="추정 비용($)")
    daily = storage.usage_daily(period_start(period))
    if len(daily) > 1:
        st.caption("날짜별 추정 비용")
        st.bar_chart({d["day"]: d["cost_usd"] or 0 for d in daily}, x_label="", y_label="$")


# ─── API 키 ──────────────────────────────────────────────────────────────


def _key_totals(storage: Storage, key_id: str) -> dict:
    rows = storage.usage_summary(None, key_id=key_id)
    top = sorted(rows, key=lambda r: -(r["cost_usd"] or 0) - r["calls"] * 1e-9)
    return {
        "calls": sum(r["calls"] for r in rows),
        "tokens": sum(r["input_tokens"] + r["output_tokens"] for r in rows),
        "cost": sum(r["cost_usd"] or 0 for r in rows),
        "models": ", ".join(f"{r['model']}({r['calls']})" for r in top[:3]) + (" …" if len(top) > 3 else ""),
        "first": min((r["first_ts"] for r in rows), default=None),
        "last": max((r["last_ts"] for r in rows), default=None),
    }


def _name_editor(storage: Storage, table: list[dict], ids: list[str], key: str) -> None:
    """이름 칸만 고칠 수 있는 표 + 저장 버튼."""
    edited = st.data_editor(
        table, hide_index=True, use_container_width=True, key=f"editor_{key}",
        disabled=[c for c in table[0] if c != "이름"],
        column_config={"이름": st.column_config.TextColumn("이름 ✏️", help="이 키가 무엇인지 알아볼 이름 (예: 회사 OpenAI 키)")},
    )
    if st.button("💾 이름 저장", key=f"save_{key}"):
        changed = 0
        for key_id, before, after in zip(ids, table, edited):
            if (after.get("이름") or "").strip() != (before.get("이름") or "").strip():
                storage.set_key_label(key_id, after.get("이름") or "")
                changed += 1
        st.success(f"이름 {changed}개를 저장했습니다." if changed else "바뀐 이름이 없습니다.")
        if changed:
            st.rerun()


def _current_keys(storage: Storage, keys: dict, now_keys: dict, rows: list[dict]) -> None:
    st.markdown("#### 🔑 지금 쓰는 API 키")
    st.caption("화면에서는 키 원문을 볼 수 없으므로 '이름'을 붙여 구분하세요. 키 원문은 저장하지 않습니다(지문·앞 3자·뒤 4자만).")
    if not now_keys:
        st.info(".env 에 API 키가 없습니다.")
        return
    period_cost = defaultdict(float)
    for r in rows:
        period_cost[r["key_id"]] += r["cost_usd"] or 0
    ids, table = [], []
    for key_id, (provider, masked) in now_keys.items():
        info = keys.get(key_id)
        total = _key_totals(storage, key_id)
        ids.append(key_id)
        table.append({
            "이름": (info or {}).get("label", ""), "회사": PROVIDER_NAMES.get(provider, provider), "키": masked,
            "선택 기간 비용($)": round(period_cost.get(key_id, 0.0), 4), "누적 비용($)": round(total["cost"], 4),
            "누적 호출": total["calls"], "처음 사용": fmt_day((info or {}).get("first_seen")),
        })
    _name_editor(storage, table, ids, "current")


def _previous_keys(storage: Storage, keys: dict, now_keys: dict) -> None:
    previous = [k for k in keys.values() if k["key_id"] not in now_keys and k["key_id"]]
    st.markdown(f"#### 🗂 이전 API 키 ({len(previous)})")
    if not previous:
        st.caption(".env 의 키를 바꾸면 예전 키의 사용 내역이 여기에 간단히 남습니다.")
        return
    st.caption("지금 .env 에 없는 키 — 그 키로 쓴 모델과 비용 요약입니다. 이름을 붙여 두면 나중에 어떤 키였는지 알 수 있습니다.")
    ids, table = [], []
    for k in previous:
        total = _key_totals(storage, k["key_id"])
        ids.append(k["key_id"])
        table.append({
            "이름": k.get("label", ""), "회사": PROVIDER_NAMES.get(k["provider"], k["provider"]), "키": k["masked"],
            "사용 기간": f"{fmt_day(total['first'] or k['first_seen'])} ~ {fmt_day(total['last'] or k['last_seen'])}",
            "누적 비용($)": round(total["cost"], 4), "누적 호출": total["calls"], "토큰": total["tokens"],
            "주요 모델": total["models"] or "-",
        })
    _name_editor(storage, table, ids, "previous")


# ─── 가격표 ──────────────────────────────────────────────────────────────


def _pricing(storage: Storage, rows: list[dict]) -> None:
    pricing = usage.load_pricing()
    overrides = storage.price_overrides()
    used = {r["model"] for r in rows if r["provider"] not in pricing["free_providers"]}
    models = sorted(set(pricing["models"]) | used | set(overrides))
    missing = sorted(m for m in used if not usage.price_for(m) and m not in pricing["per_call"])
    with st.expander("⚙️ 가격표 (USD / 100만 토큰)" + (f" — 가격 없는 모델 {len(missing)}개" if missing else ""), expanded=bool(missing)):
        st.caption("기본값은 config/pricing.yaml (2026-10-22 확인, 출처: 공식/제3자). 여기서 고치면 그 값이 우선합니다. "
                   "곡 단위 모델(Lyria)·무료 모델은 파일에서 관리합니다.")
        table = []
        for m in models:
            price = usage.price_for(m) or {}
            base = pricing["models"].get(m, {})
            table.append({
                "모델": m, "입력": price.get("input"), "출력": price.get("output"), "캐시 입력": price.get("cached_input"),
                "출처": "화면에서 고침" if m in overrides else {"official": "공식", "third_party": "제3자"}.get(base.get("source"), "-"),
            })
        edited = st.data_editor(
            table, hide_index=True, use_container_width=True, key="editor_pricing", disabled=["모델", "출처"],
            column_config={c: st.column_config.NumberColumn(c, min_value=0.0, format="%.3f") for c in ("입력", "출력", "캐시 입력")},
        )
        c1, c2 = st.columns(2)
        if c1.button("💾 가격 저장", key="save_pricing"):
            changed = 0
            for before, after in zip(table, edited):
                if (after["입력"], after["출력"], after["캐시 입력"]) != (before["입력"], before["출력"], before["캐시 입력"]):
                    if after["입력"] is None or after["출력"] is None:
                        continue  # 입력·출력 가격은 둘 다 있어야 한다
                    storage.set_price_override(after["모델"], float(after["입력"]), float(after["출력"]),
                                               float(after["캐시 입력"]) if after["캐시 입력"] is not None else None)
                    changed += 1
            fixed = storage.reprice_unpriced(usage.estimate_cost)
            st.success(f"가격 {changed}개 저장" + (f" · 비용이 비어 있던 기록 {fixed}건 다시 계산" if fixed else ""))
            if changed or fixed:
                st.rerun()
        if overrides and c2.button("↩ 화면에서 고친 가격 모두 되돌리기", key="reset_pricing"):
            for m in overrides:
                storage.clear_price_override(m)
            st.rerun()
        st.caption("※ 가격을 고쳐도 이미 계산된 기록의 비용은 바뀌지 않습니다 (기록 당시 가격). 가격이 비어 있던 기록만 다시 계산합니다.")
