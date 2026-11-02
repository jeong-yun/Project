"""모델 목록 화면 — 🔄 목록 새로 고침 · ➕ 모델 직접 입력(🧪 연결 시험) · 강도(급) 고르기.

사이드바 '🧩 모델 목록 · 직접 입력'과 '🎛 팀원 모델 선택' 상자 안에서 같은 화면을 쓴다 (where 로 위젯 키를 나눔).
로직은 core/model_catalog.py, 저장은 데이터 폴더의 model_catalog.json · custom_models.json (config/*.yaml 은 안 고침).
"""

import time
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from core import member_models, model_catalog, usage
from core.schemas import ModelsConfig

PROVIDER_NAMES = {"claude_cli": "Claude CLI", "anthropic": "Anthropic", "openai": "OpenAI", "google": "Google",
                  "upstage": "Upstage", "freellm": "FreeLLM"}
NOTE_KEY = "model_catalog_note"
REFRESH_WAIT_SEC = 15  # 🔄 버튼: 이만큼만 기다리고, 더 걸리면 백그라운드로 계속 받는다


@dataclass
class Ctx:
    models: ModelsConfig  # 지금 쓰는 설정 ('무료 모델만'이면 유료가 꺼진 사본) — 연결 시험에 쓴다
    yaml_models: ModelsConfig  # config/models.yaml 그대로 — 목록 받기·강도 기준
    merged: model_catalog.Merged
    data_dir: Path


def model_name(key: str, merged: model_catalog.Merged) -> str:
    provider, model = member_models.split_key(key)
    info = merged.info.get(key)
    return info.display if info and info.display else model


def badges(key: str, merged: model_catalog.Merged) -> str:
    """🆕 새 모델 · ✍️ 직접 입력 · ⚠️ 목록에서 사라짐."""
    info = merged.info.get(key)
    if not info:
        return ""
    marks = [m for m, on in (("🆕", info.new), ("✍️", info.source == "custom"), ("⚠️", info.missing)) if on]
    return (" " + "".join(marks)) if marks else ""


def option_label(key: str, states: dict[str, str], status_icons: dict, merged: model_catalog.Merged) -> str:
    """선택지 한 줄 — '모델 · 제공처 · 비용 (구독/유료 API/무료)' + 표시(🆕 ✍️ ⚠️) + 꺼진 provider 아이콘."""
    provider, _ = member_models.split_key(key)
    icon = "" if states.get(provider) == "ok" else f" {status_icons.get(states.get(provider), ('⚪', ''))[0]}"
    return (f"{model_name(key, merged)} · {PROVIDER_NAMES.get(provider, provider)} · {member_models.cost_label(provider)}"
            f"{badges(key, merged)}{icon}")


def _note(where: str, text: str) -> None:
    st.session_state[f"{NOTE_KEY}_{where}"] = text


def _ago(ts: float) -> str:
    sec = max(0, time.time() - ts)
    if sec < 3600:
        return f"{int(sec // 60)}분 전"
    if sec < 86400:
        return f"{int(sec // 3600)}시간 전"
    return f"{int(sec // 86400)}일 전"


def _tier_label(row_key: str, ctx: Ctx) -> str:
    provider, model = member_models.split_key(row_key)
    return f"{model} ({PROVIDER_NAMES.get(provider, provider)})"


def _row_of(key: str, ctx: Ctx) -> int | None:
    return next((i for i, row in enumerate(ctx.merged.models.model_tiers) if key in row), None)


def _guess_label(key: str, ctx: Ctx) -> str:
    rows = model_catalog.tier_rows(ctx.yaml_models)
    row = model_catalog.guess_tier(key, ctx.yaml_models.model_tiers)
    return _tier_label(rows[row], ctx) if rows else "-"


def _cost_text(cost: float | None, provider: str) -> str:
    if provider in usage.load_pricing()["free_providers"]:
        return "0 (구독·무료)"
    return usage.UNPRICED if cost is None else f"${cost:.6f}"


def _status(ctx: Ctx) -> None:
    data = model_catalog.load_catalog(ctx.data_dir)
    providers = data.get("providers") or {}
    refreshed = float(data.get("refreshed_at") or 0)
    head = f"마지막 확인 {_ago(refreshed)}" if refreshed else "아직 목록을 받은 적 없음 (yaml 의 모델만)"
    if model_catalog.refreshing():
        head += " · ⏳ 지금 받는 중"
    st.caption(f"{head} · 하루에 한 번 앱을 켤 때 자동으로 새로 받습니다.")
    lines = []
    for provider in ctx.yaml_models.providers:
        entry = providers.get(provider)
        if not entry:
            continue
        found = entry.get("models") or {}
        new = sum(1 for k, i in ctx.merged.info.items() if i.new and k.startswith(provider + ":"))
        text = f"{PROVIDER_NAMES.get(provider, provider)} {len(found)}개"
        if new:
            text += f" · 🆕 {new}"
        if entry.get("error"):
            text += f" · ❌ 이번엔 못 받음 ({entry['error'][:60]}) — 저장된 목록 사용"
        lines.append(text)
    if lines:
        st.caption(" / ".join(lines))
    new_keys = [k for k, i in ctx.merged.info.items() if i.new]
    if new_keys:
        st.markdown("🆕 **새로 나온 모델**: " + ", ".join(f"`{member_models.split_key(k)[1]}`" for k in new_keys[:12])
                    + (f" 외 {len(new_keys) - 12}개" if len(new_keys) > 12 else ""))
    for key, info in ctx.merged.info.items():
        if info.missing:
            provider, model = member_models.split_key(key)
            hint = f" → 대신 `{member_models.split_key(info.suggest)[1]}` 를 고려해 보세요" if info.suggest else ""
            st.warning(f"⚠️ `{model}` ({PROVIDER_NAMES.get(provider, provider)}) 가 목록에서 사라졌습니다{hint}. "
                       "자동으로 바꾸지 않습니다 — 바꾸려면 config/models.yaml 또는 '🎛 팀원 모델 선택'에서.")


def _refresh(ctx: Ctx, where: str) -> None:
    if st.button("🔄 모델 목록 새로 고침", key=f"catalog_refresh_{where}",
                 help="각 사 모델 목록 API 를 지금 다시 부릅니다 (키가 있고 켜진 provider 만)."):
        thread = model_catalog.start_refresh(ctx.yaml_models, ctx.data_dir, force=True)
        with st.spinner("각 사 모델 목록을 받는 중…"):
            if thread:
                thread.join(timeout=REFRESH_WAIT_SEC)
        _note(where, "목록을 새로 받았습니다." if not model_catalog.refreshing()
              else "아직 받는 중입니다 — 잠시 뒤 화면을 다시 열면 보입니다.")
        st.rerun()


def _custom_form(ctx: Ctx, where: str) -> None:
    st.markdown("**➕ 모델 직접 입력** — 목록에 없는 모델 이름을 그대로 넣습니다 (예: `gpt-6-luna`)")
    store = model_catalog.CustomStore(ctx.data_dir)
    c1, c2 = st.columns(2)
    provider = c1.selectbox("제공처", list(ctx.yaml_models.providers), key=f"custom_provider_{where}",
                            format_func=lambda p: PROVIDER_NAMES.get(p, p))
    hidden = sorted(member_models.split_key(k)[1] for k, i in ctx.merged.info.items()
                    if not i.selectable and k.startswith(provider + ":"))
    picked = ""
    if hidden:  # 선택지에 바로 안 넣은 찾은 모델 (FreeLLMAPI 수백 개) — 여기서 골라 넣는다 (입력칸에서 글자를 쳐서 찾기)
        picked = st.selectbox(f"찾은 모델에서 고르기 ({len(hidden)}개)", ["", *hidden], key=f"custom_found_{where}",
                              format_func=lambda m: m or "— 직접 입력 —")
    name = c2.text_input("모델 이름", key=f"custom_name_{where}", placeholder="API 에 보내는 이름 그대로").strip() or picked
    display = st.text_input("표시 이름 (선택)", key=f"custom_display_{where}", placeholder="예: GPT-6 Luna")
    rows = model_catalog.tier_rows(ctx.yaml_models)
    key = f"{provider}:{name}" if name else ""
    tier = st.selectbox(
        "이 모델은", ["", *rows], key=f"custom_tier_{where}",
        format_func=lambda r: (f"🤖 자동 추정 — {_guess_label(key, ctx)} 와 같은 급" if key else "🤖 자동 추정")
        if not r else f"{_tier_label(r, ctx)} 와 같은 급",
        help="상하 관계 규칙(부하는 상사보다 약한 모델)에 쓰는 강도입니다. 모르면 자동 추정 그대로 두세요.",
    )
    if name:
        st.caption(f"🧪 연결 시험 예상 비용: {_cost_text(model_catalog.estimate_test_cost(provider, name), provider)} "
                   f"(짧은 호출 2번, 토큰 약 {sum(model_catalog.TEST_TOKENS)}개)")
    b1, b2 = st.columns(2)
    test_key = f"custom_test_{where}"
    if b1.button("🧪 연결 시험", key=f"custom_test_btn_{where}", disabled=not name, use_container_width=True):
        with st.spinner(f"{name} 에 짧게 물어보는 중…"):
            st.session_state[test_key] = (key, model_catalog.connection_test(provider, name, ctx.models))

    def add() -> None:  # 버튼 콜백 — 입력칸을 비우려면 위젯을 다시 그리기 전에 해야 한다
        try:
            added = store.add(provider, name, display, ctx.yaml_models)
        except model_catalog.CustomModelError as e:
            _note(where, f"❌ {e}")
            return
        store.set_tier(added, tier or None)
        for field in ("name", "display", "found", "tier"):
            st.session_state.pop(f"custom_{field}_{where}", None)
        _note(where, f"✍️ {name} 를 추가했습니다 — '🎛 팀원 모델 선택'에서 고를 수 있습니다.")

    b2.button("💾 추가", key=f"custom_add_{where}", disabled=not name, use_container_width=True, type="primary", on_click=add)
    tested = st.session_state.get(test_key)
    if tested and tested[0] == key:
        result: model_catalog.TestResult = tested[1]
        spent = (f"실제 토큰 입력 {result.input_tokens}·출력 {result.output_tokens} · 비용 {_cost_text(result.cost_usd, provider)}"
                 f" · {result.elapsed_sec}초")
        if result.ok:
            st.success(f"✅ 연결됨 — 답: “{result.reply[:60]}” · 구조화 출력 "
                       f"{'✅' if result.structured_ok else '시험 안 함'} · {spent}")
        elif result.text_ok:
            st.warning(f"⚠️ 텍스트는 되지만 구조화 출력이 안 됩니다 (팀 업무에는 구조화 출력이 필요) — {result.error} · {spent}")
        else:
            st.error(f"❌ 연결 실패 — {result.error} · {spent}")


def _tier_picker(ctx: Ctx, where: str) -> None:
    extra = [k for k, i in ctx.merged.info.items() if i.source != "yaml" and i.selectable]
    if not extra:
        return
    st.markdown("**📐 모델 강도(급)** — 찾은 모델·직접 입력한 모델이 어느 모델과 같은 급인지 (yaml 모델은 models.yaml 이 정함)")
    store = model_catalog.CustomStore(ctx.data_dir)
    user = store.tiers()
    rows = model_catalog.tier_rows(ctx.yaml_models)
    pick = st.selectbox("모델", extra, key=f"tier_model_{where}",
                        format_func=lambda k: f"{model_name(k, ctx.merged)} · {PROVIDER_NAMES.get(member_models.split_key(k)[0])}"
                        f"{badges(k, ctx.merged)}")
    widget = f"tier_pick_{where}_{pick}"
    st.session_state[widget] = user.get(pick, "") if user.get(pick) in rows else ""

    def save() -> None:
        store.set_tier(pick, st.session_state.get(widget) or None)
        _note(where, f"📐 {member_models.split_key(pick)[1]} 의 급을 바꿨습니다.")

    st.selectbox("이 모델은", ["", *rows], key=widget, on_change=save,
                 format_func=lambda r: f"🤖 자동 추정 — {_guess_label(pick, ctx)} 와 같은 급" if not r
                 else f"{_tier_label(r, ctx)} 와 같은 급")
    custom = {k for k in store.keys()}
    if pick in custom and st.button("🗑 직접 입력한 이 모델 지우기", key=f"custom_remove_{where}"):
        store.remove(pick)
        _note(where, f"🗑 {member_models.split_key(pick)[1]} 를 지웠습니다 (고른 팀원은 기본으로 돌아갑니다).")
        st.rerun()


def render(ctx: Ctx, where: str) -> None:
    """목록 상태 · 🔄 · ➕ 직접 입력 · 📐 급 고르기."""
    note = st.session_state.pop(f"{NOTE_KEY}_{where}", None)  # 누른 곳(사이드바·🎛 상자)에만 알림
    if note:
        st.info(note)
    _refresh(ctx, where)
    _status(ctx)
    st.divider()
    _custom_form(ctx, where)
    st.divider()
    _tier_picker(ctx, where)
