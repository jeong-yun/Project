"""모델 목록 화면 — 🔄 목록 새로 고침 · ➕ 모델 직접 입력(🧪 연결 시험) · 강도(급) 고르기.

사이드바 '🧩 모델 목록 · 직접 입력'과 '🎛 팀원 모델 선택' 상자 안에서 같은 화면을 쓴다 (where 로 위젯 키를 나눔).
로직은 core/model_catalog.py, 저장은 데이터 폴더의 model_catalog.json · custom_models.json (config/*.yaml 은 안 고침).
"""

import time
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from core import member_models, model_caps, model_catalog, usage
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
    """🆕 새 모델 · ✍️ 직접 입력 · ⚠️ 목록에서 사라짐 · 🔤 텍스트 전용(사진 못 읽음)."""
    info = merged.info.get(key)
    text_only = model_caps.caps_for(key).text_only
    if not info:
        return " 🔤" if text_only else ""
    marks = [m for m, on in (("🆕", info.new), ("✍️", info.source == "custom"), ("⚠️", info.missing),
                             ("🔤", text_only)) if on]
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


def _status(ctx: Ctx, where: str = "side") -> None:
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
    _removed(ctx, where)


def _removed(ctx: Ctx, where: str) -> None:
    """제공처 목록에서 사라진 모델 — 선택지에서 숨기고 실행 때 건너뛴다 (core/model_caps). 추천 대체 모델로 바꾸기 버튼."""
    store = model_caps.CapsStore(ctx.data_dir)
    gone = {r.key: r for r in model_caps.removed_defaults(ctx.yaml_models, data_dir=ctx.data_dir)}
    for key, info in ctx.merged.info.items():
        if not info.missing:
            continue
        provider, model = member_models.split_key(key)
        name = PROVIDER_NAMES.get(provider, provider)
        r = gone.get(key)
        if r is None:  # 목록 받기가 실패했거나 오래됨 → 숨기지 않고 알리기만
            hint = f" → 대신 `{member_models.split_key(info.suggest)[1]}` 를 고려해 보세요" if info.suggest else ""
            st.warning(f"⚠️ `{model}` ({name}) 가 목록에 없습니다{hint}. 목록 확인이 오래됐거나 실패해서 숨기지는 않았습니다.")
            continue
        if r.replacement:
            st.info(f"🔁 `{model}` ({name}) 는 제공처 목록에서 사라져 **대체 모델 `{member_models.split_key(r.replacement)[1]}`** "
                    "로 실행합니다 (config/models.yaml 은 그대로).")
            if st.button("↩️ 대체 취소", key=f"replace_undo_{where}_{key}"):
                store.set_replacement(key, None)
                _note(where, f"`{model}` 대체를 취소했습니다 — 이 모델은 계속 건너뜁니다.")
                st.rerun()
            continue
        hint = f" 추천 대체: `{member_models.split_key(r.suggest)[1]}`." if r.suggest else ""
        st.warning(f"⚠️ `{model}` ({name}) 가 제공처 목록에서 사라졌습니다 — 선택지에서 숨기고, 실행 때는 건너뛰고 다음 후보를 씁니다."
                   f"{hint} (config/models.yaml 은 자동으로 바꾸지 않음)")
        if r.suggest and st.button(f"🔁 추천 대체 모델로 바꾸기 ({member_models.split_key(r.suggest)[1]})",
                                   key=f"replace_{where}_{key}",
                                   help="데이터 폴더 model_caps.json 에 기록합니다. 이 모델이 쓰이던 자리(직급 기본·대안)에 대체 모델을 씁니다."):
            store.set_replacement(key, r.suggest)
            _note(where, f"🔁 `{model}` → `{member_models.split_key(r.suggest)[1]}` 로 바꿔 실행합니다.")
            st.rerun()


def removed_notice(ctx: Ctx) -> None:
    """사라진 기본 모델이 있으면 한 번만(세션당) ⚠️ 알림 — 자세한 내용·대체 버튼은 '🧩 모델 목록'."""
    gone = [r for r in model_caps.removed_defaults(ctx.yaml_models, data_dir=ctx.data_dir) if not r.replacement]
    keys = tuple(sorted(r.key for r in gone))
    if keys and st.session_state.get("removed_notice_shown") != keys:
        st.session_state["removed_notice_shown"] = keys
        names = ", ".join(member_models.split_key(k)[1] for k in keys)
        st.toast(f"제공처에서 사라진 모델 {len(keys)}개 ({names}) — 건너뛰고 다음 후보를 씁니다. "
                 "'🧩 모델 목록'에서 추천 대체 모델로 바꿀 수 있습니다.", icon="⚠️")


def hidden_note(hidden: list, merged: model_catalog.Merged | None = None, title: str = "") -> None:
    """'ⓘ 숨긴 모델 N개 (이유)' — 선택지에서 뺀 모델과 이유 (model_caps.Hidden 목록).
    expander 안에서도 쓸 수 있게 popover 로 그린다 (Streamlit 은 expander 를 겹칠 수 없다)."""
    if not hidden:
        return
    with st.popover(title or f"ⓘ 숨긴 모델 {len(hidden)}개 (이유)"):
        for h in hidden:
            provider, model = member_models.split_key(h.key)
            name = model_name(h.key, merged) if merged else model
            st.markdown(f"- **{name}** · {PROVIDER_NAMES.get(provider, provider)} — " + "; ".join(h.reasons))


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
    with_vision = st.checkbox("🖼 사진 읽기도 시험 (작은 그림 1장, 호출 1번 더)", key=f"custom_vision_{where}",
                              help="글자 하나가 든 작은 그림을 보내 맞히는지 봅니다. 결과는 사진을 읽는 일의 선택지에 반영됩니다.")
    if b1.button("🧪 연결 시험", key=f"custom_test_btn_{where}", disabled=not name, use_container_width=True):
        with st.spinner(f"{name} 에 짧게 물어보는 중…"):
            result = model_catalog.connection_test(provider, name, ctx.models)
            probe = model_caps.vision_probe(provider, name, ctx.models) if with_vision and result.text_ok else None
            st.session_state[test_key] = (key, result, probe)

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
        if len(tested) > 2 and tested[2] is not None:
            _probe_line(tested[2])
        spent = (f"실제 토큰 입력 {result.input_tokens}·출력 {result.output_tokens} · 비용 {_cost_text(result.cost_usd, provider)}"
                 f" · {result.elapsed_sec}초")
        if result.ok:
            st.success(f"✅ 연결됨 — 답: “{result.reply[:60]}” · 구조화 출력 "
                       f"{'✅' if result.structured_ok else '시험 안 함'} · {spent}")
        elif result.text_ok:
            st.warning(f"⚠️ 텍스트는 되지만 구조화 출력이 안 됩니다 (팀 업무에는 구조화 출력이 필요) — {result.error} · {spent}")
        else:
            st.error(f"❌ 연결 실패 — {result.error} · {spent}")


def _probe_line(probe: model_caps.ProbeResult) -> None:
    if probe.ok:
        st.success(f"🖼 사진 읽기 ✅ — 그림의 글자 {probe.expected} 를 맞힘 (답 “{probe.reply[:20]}”, {probe.elapsed_sec}초)")
    elif probe.ok is False:
        st.warning(f"🖼 사진 읽기 ❌ — 텍스트 전용으로 기록했습니다 (글자 {probe.expected}, 답 “{probe.reply[:20]}”"
                   f"{', ' + probe.error[:120] if probe.error else ''}). 사진을 읽는 일의 선택지에서 숨깁니다.")
    else:
        st.info(f"🖼 사진 시험을 끝내지 못했습니다 — {probe.error[:160]} (기록 안 함)")


def _caps_panel(ctx: Ctx, where: str) -> None:
    """👁 모델 능력 — 모델마다 사진 읽기·형식 출력·문맥과 그 근거, 🧪 사진 시험, 사람이 지정."""
    st.markdown("**👁 모델 능력** — 사진을 못 읽는 모델은 사진을 읽는 일(도면 그림·JD 이미지 등)의 선택지·대안에서 빠집니다.")
    keys = list(dict.fromkeys([k for row in ctx.merged.models.model_tiers for k in row]))
    if not keys:
        return
    shown, hidden = model_caps.selectable_models(model_caps.IMAGE_READING, keys, data_dir=ctx.data_dir)
    st.caption(f"사진 읽기에 쓸 수 있는 모델 {len(shown)}개: " + ", ".join(member_models.split_key(k)[1] for k in shown[:14])
               + (f" 외 {len(shown) - 14}개" if len(shown) > 14 else ""))
    hidden_note(hidden, ctx.merged, f"ⓘ 사진 읽기에서 숨긴 모델 {len(hidden)}개 (이유)")
    pick = st.selectbox("모델", keys, key=f"caps_model_{where}",
                        format_func=lambda k: f"{model_name(k, ctx.merged)} · {PROVIDER_NAMES.get(member_models.split_key(k)[0])}"
                        f"{badges(k, ctx.merged)}")
    caps = model_caps.caps_for(pick, ctx.data_dir)
    rows = []
    for field, label in (("vision", "👁 사진 읽기"), ("structured_output", "📝 형식(구조화) 출력"), ("context", "📏 문맥")):
        value = getattr(caps, field)
        text = (f"약 {value:,} 토큰" if field == "context" and value else
                "된다" if value is True else "안 된다" if value is False else "모름")
        rows.append(f"- {label}: **{text}**" + (f" · 근거: {caps.why(field)}" if field in caps.source else ""))
    st.markdown("\n".join(rows))
    store = model_caps.CapsStore(ctx.data_dir)
    widget = f"caps_user_{where}_{pick}"
    user = (store.read()["user"].get(pick) or {}).get("vision")
    st.session_state[widget] = "" if user is None else ("yes" if user else "no")

    def save() -> None:
        value = st.session_state.get(widget)
        store.set_user(pick, "vision", None if not value else value == "yes")
        _note(where, f"👁 {member_models.split_key(pick)[1]} 의 사진 읽기를 "
              + ("자동 판단으로 되돌렸습니다." if not value else f"'{'된다' if value == 'yes' else '안 된다'}'로 지정했습니다."))

    c1, c2 = st.columns(2)
    c1.selectbox("사진 읽기 (사람이 지정)", ["", "yes", "no"], key=widget, on_change=save,
                 format_func=lambda v: {"": "🤖 자동 판단", "yes": "된다", "no": "안 된다"}[v])
    provider, model = member_models.split_key(pick)
    probe_key = f"caps_probe_{where}"
    if c2.button("🧪 사진 시험", key=f"caps_probe_btn_{where}", use_container_width=True,
                 help="글자 하나가 든 작은 그림(약 1KB)을 보내 맞히는지 봅니다. 호출 1번 (사용량에 '사진 시험'으로 기록)."):
        with st.spinner(f"{model} 에 작은 그림을 보내는 중…"):
            st.session_state[probe_key] = (pick, model_caps.vision_probe(provider, model, ctx.models, ctx.data_dir))
    probed = st.session_state.get(probe_key)
    if probed and probed[0] == pick:
        _probe_line(probed[1])


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
    _status(ctx, where)
    st.divider()
    _custom_form(ctx, where)
    st.divider()
    _tier_picker(ctx, where)
    st.divider()
    _caps_panel(ctx, where)
