"""🏠 메인 화면 — 📰 AI 브리핑·💡 개선 아이디어 카드·🔥 주목 저장소, 모델별 토큰 사용량·추정 비용,
API 키별 내역(이름 붙이기), 이전 키, 가격표, 🤖 자율 실행 상태(autonomy_ui.py).

AI 브리핑·카드는 core/feeds.py 가 모아 data/feeds/ 에 저장한 것을 읽는다 (수집은 버튼이나 TTL 이 지났을 때 백그라운드로).

데이터는 core/usage.py 가 모델 호출마다 기록한 data/runs.db 의 usage·api_keys 테이블에서 읽는다.
API 키 원문은 저장하지 않고 지문(sha256 앞 12자리)과 '앞 3자…뒤 4자'만 쓴다 → 사람이 키마다 이름을 붙여 구분한다.
"""

import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

import autonomy_ui
import board_ui
from core import feeds, usage
from core.storage import Storage

PERIODS = {"오늘": 1, "7일": 7, "30일": 30, "전체": None}  # 최근 며칠
PROVIDER_NAMES = {"openai": "OpenAI", "google": "Google", "anthropic": "Anthropic", "upstage": "Upstage",
                  "freellm": "무료(FreeLLMAPI)", "claude_cli": "구독(Claude)"}
SUBSCRIPTION = "claude_cli"  # Claude 구독(claude -p) — 토큰은 기록하지만 청구 없음, 한도는 사람이 쓰는 Claude 와 공유


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
        if provider == SUBSCRIPTION:
            continue  # API 키가 아니라 구독 로그인 (CLAUDE_CLI_PATH 는 실행 파일 경로)
        key = os.getenv(env)
        if key:
            key_id, masked = usage.key_fingerprint(key)
            out[key_id] = (provider, masked)
    return out


def billing(provider: str) -> str:
    """모델별 표의 '요금' 칸 — 구독·무료·API 종량제를 구분한다."""
    if provider == SUBSCRIPTION:
        return "구독 (청구 없음)"
    if provider in usage.load_pricing()["free_providers"]:
        return "무료"
    return "API 종량제"


def api_equivalent(rows: list[dict]) -> float:
    """구독 호출을 Anthropic API 로 했다면 얼마였을지 (가격표 기준, 참고용)."""
    return sum(usage.estimate_cost("anthropic", r["model"], r["input_tokens"], r["output_tokens"], calls=r["calls"]) or 0
               for r in rows if r["provider"] == SUBSCRIPTION)


def money(value: float | None) -> str:
    if value is None:
        return "-"
    return f"${value:,.4f}" if value < 1 else f"${value:,.2f}"


def fmt_day(ts: float | None) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else "-"


def render(storage: Storage, models, open_runs: list[dict], team_names: dict, go_to_team,
           settings: dict | None = None, data_dir: Path | None = None) -> None:
    st.title("🏠 Agent Office")
    st.caption("AI 소식·이 앱 개선 아이디어, 모델별 토큰 사용량과 추정 비용, API 키별 사용 내역을 봅니다. "
               "팀 업무는 왼쪽 '화면'에서 🏢 팀 업무를 고르세요.")

    _team_shortcuts(open_runs, team_names, go_to_team)
    if settings is not None and data_dir is not None:
        autonomy_ui.render(settings, Path(data_dir), storage)  # 🤖 자율 실행 상태·승인 대기열 (읽기 전용)
        _feeds(settings, models, Path(data_dir), list(team_names.values()))

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


# ─── 📰 AI 브리핑 · 🔥 주목 저장소 · 💡 아이디어 카드 (core/feeds.py) ─────────────────


def _md(text: str) -> str:
    """모델·외부 글을 마크다운에 넣을 때 — 링크 문법·수식($)이 먹지 않게."""
    return (text or "").replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]").replace("$", "\\$")


def _ago(ts: float | None) -> str:
    if not ts:
        return "-"
    sec = max(time.time() - ts, 0)
    rel = f"{int(sec // 60)}분 전" if sec < 3600 else f"{int(sec // 3600)}시간 전" if sec < 86400 else f"{int(sec // 86400)}일 전"
    return f"{datetime.fromtimestamp(ts):%Y-%m-%d %H:%M} ({rel})"


def _link_md(link: dict) -> str:
    host = (urlparse(link["url"]).hostname or "").removeprefix("www.")
    label = feeds.SOURCE_LABELS.get(link["source"], link["source"]).split(" ", 1)[-1]
    return f"[{label} · {_md(host)}]({link['url'].replace(')', '%29').replace(' ', '%20')})"


def _feeds(settings: dict, models, data_dir: Path, teams: list[str]) -> None:
    cfg = feeds.feed_settings(settings)
    snap = feeds.load_snapshot(data_dir)
    running = feeds.collecting(data_dir)
    # TTL 이 지났으면 화면을 열 때 한 번 백그라운드로 (같은 브라우저 세션에서는 한 번만 — 실패해도 되풀이하지 않게)
    if (not running and cfg["auto_collect"] and feeds.is_stale(snap, cfg["ttl_hours"])
            and not st.session_state.get("feeds_auto_started")):
        st.session_state["feeds_auto_started"] = True
        if feeds.start_collect(settings, models, data_dir, teams):
            running = time.time()

    head, button = st.columns([5, 1])
    head.markdown("#### 📰 오늘의 AI 브리핑")
    if button.button("🔄 지금 수집", key="feeds_collect", disabled=bool(running), use_container_width=True,
                     help="뉴스·GitHub·Hacker News(+키가 있는 Reddit·네이버·Threads)를 모아 보조 모델 1회로 요약합니다."):
        feeds.start_collect(settings, models, data_dir, teams)
        st.rerun()
    _feed_status(snap, running, data_dir, cfg)
    # 전체 화면 보드와 같은 '오늘의 한 줄'·갱신 시각·서비스 상태·지수 띠 + 보드 열기 버튼 (board_ui.py)
    board_ui.headline_strip(snap, models, settings, data_dir)
    if not snap or not snap.get("items"):
        if not running:
            st.info("아직 모은 소식이 없습니다. '🔄 지금 수집'을 누르면 1~2분 안에 채워집니다 (보조 모델 1회 비용).")
        return

    left, right = st.columns([3, 2])
    with left:
        if snap.get("briefing"):
            for point in snap["briefing"]:
                badge = f" ⚠️ _주장 — {_md(', '.join(point['hype']))}_" if point.get("hype") else ""
                st.markdown(f"- {_md(point['text'])}{badge}  \n  {' · '.join(_link_md(l) for l in point['links'])}")
        else:
            errors = (snap.get("llm") or {}).get("errors") or []
            st.warning("요약을 만들지 못했습니다 — 아래 '모은 글'에서 원문을 보세요." + (f"\n\n{errors[-1][:300]}" if errors else ""))
    with right:
        _top_repos(snap)
    _collected_items(snap)
    _idea_cards(snap, data_dir)


def _feed_status(snap: dict | None, running: float | None, data_dir: Path, cfg: dict) -> None:
    if running:
        _wait_for_collect(data_dir)
    if err := feeds.last_error(data_dir):
        st.warning(f"지난 수집이 실패했습니다: {err}")
    if not snap:
        return
    if snap.get("attempt_error") and (snap.get("attempted_at") or 0) > (snap.get("collected_at") or 0):
        st.warning(f"{_ago(snap['attempted_at'])} 수집: {snap['attempt_error']}")
    if not snap.get("collected_at"):
        return
    llm = snap.get("llm") or {}
    cost = llm.get("cost_usd")
    cost_text = ("비용 미상 (가격표에 없는 모델)" if cost is None and llm.get("model")
                 else "요약 실패 (비용 없음)" if cost is None else money(cost))
    tokens = f", 입력 {llm.get('input_tokens') or 0:,}·출력 {llm.get('output_tokens') or 0:,} 토큰" if llm.get("model") else ""
    next_auto = (f" · 다음 자동 수집: {_ago(snap['collected_at'] + cfg['ttl_hours'] * 3600).split(' (')[0]} 이후"
                 if cfg["auto_collect"] else " · 자동 수집 꺼짐")
    st.caption(f"마지막 수집 {_ago(snap['collected_at'])} · 글 {len(snap.get('items') or [])}건 · "
               f"이번 수집 비용 {cost_text}" + (f" ({llm['model']}{tokens})" if llm.get("model") else "") + next_auto)


@st.fragment(run_every=2)
def _wait_for_collect(data_dir: Path) -> None:
    """수집 스레드가 끝나면 화면 전체를 다시 그린다."""
    started = feeds.collecting(data_dir)
    if not started:
        st.rerun()
    st.info(f"⏳ 소식을 모으는 중… ({int(time.time() - started)}초) — 끝나면 자동으로 새로 고칩니다. 다른 화면을 봐도 계속 모읍니다.")


def _top_repos(snap: dict) -> None:
    st.markdown("##### 🔥 주목 저장소")
    repos = sorted((i for i in snap["items"] if i["source"] == "github"), key=lambda i: -(i.get("score") or 0))[:6]
    if not repos:
        status = (snap.get("sources") or {}).get("github") or {}
        st.caption("GitHub 결과 없음" + (f" — {status.get('note')}" if status.get("note") else ""))
        return
    for r in repos:
        lic = r["extra"].get("license", "없음")
        lic = "⚠️ 라이선스 없음" if lic == "없음" else lic
        stars = r.get("score") or 0
        st.markdown(f"**[{_md(r['title'])}]({r['url']})** ★{stars / 1000:.1f}k" if stars >= 1000 else
                    f"**[{_md(r['title'])}]({r['url']})** ★{stars}")
        st.caption(f"push {r['extra'].get('pushed_at', '-')} · {lic} · {r['extra'].get('language') or '-'}"
                   + (f" — {r['snippet'][:90]}" if r.get("snippet") else ""))


def _collected_items(snap: dict) -> None:
    items = snap["items"]
    hype = sum(1 for i in items if i.get("hype"))
    with st.expander(f"🗞 모은 글 {len(items)}건" + (f" · ⚠️ 과장·광고 의심 {hype}건" if hype else "") + " · 소스 상태"):
        st.caption("인터넷에서 모은 글입니다 — 사실 확인 전의 자료이고, ⚠️ 표시는 '주장'(수익·강의 판매·제휴 링크 등)입니다.")
        rows = [{"소스": feeds.SOURCE_LABELS.get(s, s), "상태": {"ok": "✅", "error": "❌", "skipped": "⚪ 건너뜀",
                                                                "off": "⛔ 끔"}.get(v["state"], v["state"]),
                 "건수": v["count"], "메모": v.get("note", "")} for s, v in (snap.get("sources") or {}).items()]
        if rows:
            st.dataframe(rows, hide_index=True, use_container_width=True)
        st.dataframe(
            [{"소스": feeds.SOURCE_LABELS.get(i["source"], i["source"]), "제목": i["title"], "날짜": i.get("published") or "",
              "점수": i.get("score"), "⚠️": ", ".join(i.get("hype") or []), "링크": i["url"]} for i in items],
            hide_index=True, use_container_width=True,
            column_config={"링크": st.column_config.LinkColumn("링크", display_text="열기")},
        )


def _idea_cards(snap: dict, data_dir: Path) -> None:
    st.markdown("#### 💡 개선 아이디어 카드")
    cards = feeds.all_cards(snap, feeds.card_states(data_dir))
    if not cards:
        st.caption("이번 수집에서 이 앱과 관련 있는 아이디어가 나오지 않았습니다.")
        return
    c1, c2 = st.columns([3, 2])
    kinds = c1.multiselect("종류", list(feeds.KINDS), default=list(feeds.KINDS), format_func=feeds.KINDS.get, key="cards_kind")
    states = {"": "🆕 새 카드", **feeds.STATES, "all": "전체"}
    state = c2.radio("상태", list(states), format_func=states.get, horizontal=True, key="cards_state")
    shown = [c for c in cards if c["kind"] in kinds and (state == "all" or c["state"] == state)]
    counts = {s: sum(1 for c in cards if c["state"] == s) for s in states if s != "all"}
    st.caption(" · ".join(f"{states[s]} {n}" for s, n in counts.items()) + " — ⭐ 관심 카드만 작업 요청으로 내보낼 수 있습니다.")
    cols = st.columns(2)
    for n, card in enumerate(shown):
        with cols[n % 2].container(border=True):
            _card(card, data_dir)
    if not shown:
        st.caption("조건에 맞는 카드가 없습니다.")

    request = feeds.work_request(cards)
    approved = counts.get("interested", 0)
    with st.expander(f"📤 작업 요청 만들기 (⭐ 관심 카드 {approved}개)"):
        if request:
            st.caption("오른쪽 위 복사 버튼으로 복사해 Claude 에게 붙여 넣으면 별도 worktree 에서 구현합니다.")
            st.code(request, language="markdown")
        else:
            st.caption("카드에서 ⭐ 관심을 누른 것만 작업 요청으로 내보낼 수 있습니다.")


def _card(card: dict, data_dir: Path) -> None:
    head = f"{feeds.KINDS.get(card['kind'], card['kind'])} · 작업 {feeds.EFFORTS.get(card['effort'], card['effort'])}"
    if card["state"]:
        head += f" · {feeds.STATES[card['state']]}"
    if not card.get("new"):
        head += " · 지난 수집"
    st.caption(head)
    st.markdown(f"**{_md(card['title'])}**" + ("  ⚠️ **과장·광고 의심**" if card.get("hype") else ""))
    if card.get("hype"):
        st.markdown(f"📢 _주장(확인 안 됨)_: {_md(card['summary'])}")
        st.caption("⚠️ " + _md(", ".join(card.get("hype_reasons") or [])))
    else:
        st.markdown(_md(card["summary"]))
    st.caption(f"관련성 {'●' * card['relevance_score']}{'○' * (5 - card['relevance_score'])} {_md(card['relevance'])}")
    if card.get("files"):
        st.caption("고칠 파일: " + ", ".join(f"`{f}`" for f in card["files"]))
    if card.get("done"):
        st.caption(f"✔ 이미 한 작업과 비슷: {_md(card['done'])} (today_worked.md)")
    st.caption("출처: " + " · ".join(_link_md(l) for l in card["links"]))
    plain = {k: v for k, v in card.items() if k not in ("state", "new")}
    buttons = st.columns(4)
    for col, (state, label) in zip(buttons, feeds.STATES.items()):
        col.button(label, key=f"card_{state}_{card['id']}", disabled=card["state"] == state, use_container_width=True,
                   on_click=feeds.set_card_state, args=(data_dir, plain, state))
    if card["state"]:
        buttons[3].button("↩", key=f"card_reset_{card['id']}", help="상태 지우기", use_container_width=True,
                          on_click=feeds.set_card_state, args=(data_dir, plain, None))


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
    sub_calls = sum(r["calls"] for r in rows if r["provider"] == SUBSCRIPTION)
    if sub_calls:
        st.caption(f"🎫 Claude 구독 호출 {sub_calls:,}건은 비용 0 (구독, 청구 없음) — API 로 했다면 약 {money(api_equivalent(rows))} (참고용). "
                   "구독 한도는 내가 쓰는 Claude 대화·Claude Code 와 함께 줄어듭니다.")
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
        {"모델": model, "회사": PROVIDER_NAMES.get(provider, provider), "요금": billing(provider), "호출": m["calls"],
         "입력 토큰": m["in"], "출력 토큰": m["out"],
         "추정 비용($)": round(m["cost"], 4) if m["unpriced"] < m["calls"] else None}
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
