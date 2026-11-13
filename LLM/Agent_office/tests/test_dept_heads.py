"""👔 부서 부장 = 비평가·🧭 부서장 — 모델 정하는 순서, 저장, 무료만, 비공개 팀, 강도 경고, 팀 배정, 워크플로우,
큰 입력 줄이기, Gemini 시간 제한, 커리어 표 번호, 화면(AppTest)."""

import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from core import building, dept_heads, llm_factory, member_models, verification
from core.config import BASE_DIR, load_team, team_org
from core.member_models import team_rank_models
from core.schemas import ModelSpec, RankModels
from core.workflow import build_office_graph, stream_office
from tests.fakes import FakeOfficeModel, Script, make_critic, make_review
from tests.test_workflow import BRIEF, SETTINGS, full_script

OPUS = "claude_cli:claude-opus-5-5"
CRITIC_ON = {"critic": {"enabled": True, "scope": "final", "min_score": 4, "max_self_corrections": 0}}


def keys(chain: RankModels) -> list[str]:
    return [member_models.model_key(s) for s in chain.candidates]


def dept(dept_id: str, team_ids: list[str]) -> building.Department:
    return next(d for d in building.departments(team_ids, building.load_config()) if d.id == dept_id)


def chains_of(team, ranks, models, choices=None):
    org = team_org(team, ranks)
    return org, team_rank_models(org, models, team, choices or {})


# ─── 비평가 모델 정하는 순서 ────────────────────────────────────────────


def test_critic_resolution_order(org, team, models):
    chains = team_rank_models(org, models, team, {})
    head = dept_heads.head_of(dept("research", ["general", "concept"]))
    # 기본 설정: models.yaml critic = 부장 기본 (팀장과 같은 Opus 5.5 구독이 맨 앞)
    plan = dept_heads.critic_plan(models, team, chains, org.id, head)
    assert plan.source == "critic" and keys(plan.chain)[0] == OPUS
    # 4) critic 을 지우면 utility (예전 동작)
    models = models.model_copy(update={"critic": None})
    plan = dept_heads.critic_plan(models, team, chains, org.id, head)
    assert plan.source == "utility" and plan.chain.primary == models.utility.primary
    # 3) models.yaml critic 이 있으면 그것
    with_critic = models.model_copy(update={"critic": RankModels(primary=ModelSpec(provider="openai", model="gpt-5.4"))})
    plan = dept_heads.critic_plan(with_critic, team, chains, org.id, head)
    assert plan.source == "critic" and keys(plan.chain)[0] == "openai:gpt-5.4"
    # 1) 부장 모델을 고르면 그것이 먼저
    plan = dept_heads.critic_plan(with_critic, team, chains, org.id, head, "openai:gpt-5.5")
    assert plan.source == "chosen" and keys(plan.chain)[0] == "openai:gpt-5.5" and "openai:gpt-5.4" in keys(plan.chain)
    assert plan.who == "🔬 리서치·지식부 부장"
    # 지금 고를 수 없는 모델이 저장돼 있으면(지운 직접 입력 모델) 무시
    assert dept_heads.critic_plan(models, team, chains, org.id, head, "openai:gone-model").source == "utility"


def test_chosen_fallbacks_never_climb_above_choice(models):
    tiers = member_models.tier_map(models)
    chain = dept_heads.chosen_chain("google:gemini-3.5-flash", models)
    assert keys(chain)[0] == "google:gemini-3.5-flash"
    assert "openai:gpt-5.4-mini" not in keys(chain)  # utility 의 대안이지만 고른 모델보다 강함 → 빠짐
    assert all(tiers[k] >= tiers["google:gemini-3.5-flash"] for k in keys(chain))
    assert "openai:gpt-5.4-mini" in keys(dept_heads.chosen_chain("openai:gpt-5.5", models))  # 더 약하면 남는다


def test_department_head_store_roundtrip(tmp_path):
    store = dept_heads.DeptModelStore(tmp_path)
    assert store.get("security") == ""
    store.set("security", "openai:gpt-5.4")
    store.set("career", OPUS)
    assert dept_heads.DeptModelStore(tmp_path).load_all() == {"security": "openai:gpt-5.4", "career": OPUS}
    store.set("security", "")  # 기본으로 되돌림
    assert json.loads((tmp_path / "department_models.json").read_text(encoding="utf-8")) == {"career": OPUS}
    (tmp_path / "department_models.json").write_text("{깨짐", encoding="utf-8")
    assert store.load_all() == {}


def test_head_identity_comes_from_departments_yaml():
    head = dept_heads.head_of(dept("security", ["security_governance", "infra_security"]))
    assert head.name == "보안부 부장" and "통제 번호" in head.persona and head.label == "🛡️ 보안부 부장"
    assert head.critic_persona().startswith("너는 보안부 부장이다.")
    bare = building.Department(id="x", name="실험부", emoji="🧪", color="#000000", description="실험한다.")
    assert dept_heads.head_of(bare).name == "실험부 부장" and "실험한다." in dept_heads.head_of(bare).persona


# ─── 무료만 · 비공개 팀 · 강도 경고 ──────────────────────────────────────


def test_free_only_mode_gives_free_critic(org, team, models, set_keys, monkeypatch):
    set_keys("openai", "google", "freellm")
    monkeypatch.setattr(llm_factory, "freellm_reachable", lambda: True)
    free = llm_factory.free_only(models)
    chains = team_rank_models(org, models, team, {})
    for choice in ("", "openai:gpt-5.4"):  # 기본이든 유료 부장이든 '무료만'이면 무료 모델로 비평
        plan = dept_heads.critic_plan(models, team, chains, org.id, None, choice)
        usable = llm_factory.available_candidates(plan.chain, free)
        assert usable and all(spec.provider == "freellm" for spec, _ in usable)
    # 무료 모델을 부장으로 고르면 평소에도 대안은 무료만 (유료로 넘어가지 않음)
    plan = dept_heads.critic_plan(models, team, chains, org.id, None, "freellm:gemini-3.6-flash")
    assert {s.provider for s in plan.chain.candidates} == {"freellm"}
    # critic 목록에 무료가 없어도 utility 의 무료 모델을 끝에 붙인다
    paid_only = models.model_copy(update={"critic": RankModels(primary=ModelSpec(provider="openai", model="gpt-5.4"))})
    chain, _ = dept_heads.head_chain(paid_only)
    assert any(s.provider == "freellm" for s in chain.fallback)


def test_private_team_defaults_to_team_lead_model_and_warns_on_other_provider(ranks, models):
    career = load_team("career")
    assert career.private
    org, chains = chains_of(career, ranks, models)
    head = dept_heads.head_of(dept("career", ["career"]))
    plan = dept_heads.critic_plan(models, career, chains, org.id, head)
    assert plan.source == "private" and plan.chain == chains[org.id] and not plan.warnings
    members = {c.primary.provider for c in chains.values()}
    outside = next(k for k in member_models.catalog(models) if member_models.split_key(k)[0] not in members)
    plan = dept_heads.critic_plan(models, career, chains, org.id, head, outside)
    assert plan.source == "chosen" and any("비공개 팀" in w for w in plan.warnings)
    # 일반 팀은 제공처가 달라도 경고하지 않는다
    general = load_team("general")
    g_org, g_chains = chains_of(general, ranks, models)
    assert not any("비공개" in w for w in dept_heads.critic_plan(models, general, g_chains, g_org.id, None, outside).warnings)


def test_tier_warning_when_critic_is_weaker_than_lead(org, team, models):
    chains = team_rank_models(org, models, team, {})  # 팀장 기본 = Opus 5.5 (가장 강함)
    weak = dept_heads.critic_plan(models, team, chains, org.id, None, "google:gemini-3.5-flash")
    assert any("팀장 모델" in w and "약합니다" in w for w in weak.warnings)
    assert not dept_heads.critic_plan(models, team, chains, org.id, None, OPUS).warnings
    # 팀장을 약한 모델로 고르면 같은 부장 모델이라도 경고가 사라진다
    low_lead = team_rank_models(org, models, team, {org.id: "google:gemini-3.5-flash-lite"})
    assert not dept_heads.critic_plan(models, team, low_lead, org.id, None, "google:gemini-3.5-flash").warnings


# ─── 🧭 부서장(팀 배정)도 부장 모델 ─────────────────────────────────────


def test_router_uses_department_head_model(use_script, monkeypatch):
    dom, abr = load_team("travel_domestic"), load_team("travel_abroad")
    travel = dept("travel", [dom.id, abr.id])
    script = use_script(Script(TeamPick=building.TeamPick(analysis="해외", team_id="travel_abroad", reason="오사카")))
    used: list[str] = []

    def fake(spec, models):
        used.append(member_models.model_key(spec))
        return FakeOfficeModel(script=script)

    monkeypatch.setattr(llm_factory, "create_chat_model", fake)
    models = building_models()
    picked = building.route_request("오사카 3박 4일", travel, [dom, abr], models, choice="openai:gpt-5.4")
    assert picked["how"] == "llm" and picked["model"] == "openai · gpt-5.4" and used == ["openai:gpt-5.4"]
    assert "여행부 부장" in script.prompts("TeamPick")[0] and "운영 시간" in script.prompts("TeamPick")[0]  # 부장 관점
    used.clear()
    building.route_request("오사카", travel, [dom, abr], models)  # 고르지 않으면 부장 기본(models.yaml critic)
    # claude_cli 는 테스트 PC 에 없으므로(꺼짐) 부장 기본의 다음 후보 — 팀 배정은 모델 1번 (낱말 비교는 실패할 때만)
    assert len(used) == 1 and used[0] in keys(models.critic) and used[0] != member_models.model_key(models.utility.primary)


def building_models():
    from core.config import load_models

    return load_models()


# ─── 워크플로우: 부장이 비평하고, 누가 어느 모델로 비평했는지 남긴다 ─────────────


def test_workflow_critic_is_department_head_with_chosen_model(org, ranks, team, models, use_script, monkeypatch):
    script = use_script(full_script(CriticResult=make_critic(5)))
    used: list[tuple[str, str]] = []

    def fake(spec, models):
        used.append(member_models.model_key(spec))
        return FakeOfficeModel(script=script)

    monkeypatch.setattr(llm_factory, "create_chat_model", fake)
    head = dept_heads.head_of(dept("research", ["general", "concept"]))
    graph = build_office_graph(org, ranks, team, models, SETTINGS, CRITIC_ON, dept_head=head, head_choice="openai:gpt-5.4")
    steps = [p for k, p in stream_office(graph, org, "질문", "am_action", brief=BRIEF) if k == "step"]
    final = steps[-1]
    assert final.critic.score == 5 and final.critic_model.label == "openai · gpt-5.4"
    assert final.critic_by == "🔬 리서치·지식부 부장"
    critic_prompt = script.prompts("CriticResult")[0]
    assert critic_prompt.startswith("너는 리서치·지식부 부장이다.") and "원전" in critic_prompt


def test_workflow_without_head_uses_default_head_chain(org, ranks, team, models, use_script):
    use_script(full_script(CriticResult=make_critic(5)))
    graph = build_office_graph(org, ranks, team, models, SETTINGS, CRITIC_ON)
    final = [p for k, p in stream_office(graph, org, "질문", "am_action", brief=BRIEF) if k == "step"][-1]
    assert final.critic_model in models.critic.candidates and final.critic_by == "⚖️ 비평가"
    # critic 을 지운 설정이면 예전처럼 utility
    no_critic = models.model_copy(update={"critic": None})
    use_script(full_script(CriticResult=make_critic(5)))
    graph = build_office_graph(org, ranks, team, no_critic, SETTINGS, CRITIC_ON)
    final = [p for k, p in stream_office(graph, org, "질문", "am_action", brief=BRIEF) if k == "step"][-1]
    assert final.critic_model == models.utility.primary


def test_large_input_is_trimmed_before_the_critic(org, ranks, team, models, use_script):
    big = "가나다라마바사 " * 12000  # 약 9.6만 자 — 4만 토큰짜리 통합본 흉내
    script = use_script(Script(text=big, ReviewResult=make_review(answer=big), CriticResult=make_critic(5)))
    graph = build_office_graph(org, ranks, team, models, SETTINGS, CRITIC_ON)
    list(stream_office(graph, org, "질문", "am_action", brief=BRIEF))
    prompt = script.prompts("CriticResult")[0]
    cfg = verification.DEFAULT_CRITIC
    assert len(prompt) < cfg["input_chars"] + cfg["answer_chars"] + cfg["evidence_chars"] + 4000
    assert "비평 입력 한도" in prompt  # 수정 답변은 앞·뒤를 남기고 가운데를 줄임
    assert "검토 대상가 길어" in prompt  # 하위 결과도 몫대로 줄임 (안내 문구)


def test_trim_middle_keeps_head_and_tail():
    text = "앞" * 1000 + "중" * 5000 + "끝" * 1000
    out = verification.trim_middle(text, 1500)
    assert out.startswith("앞") and out.endswith("끝") and "5,500자 생략" in out and len(out) < 1600
    assert verification.trim_middle("짧다", 1500) == "짧다" and verification.trim_middle(text, 0) == text


# ─── Gemini 시간 제한 ─────────────────────────────────────────────────


def test_google_model_gets_request_timeout_from_settings(set_keys, models, monkeypatch, allow_model_objects):
    import core.config

    set_keys("google")
    monkeypatch.setattr(core.config, "load_settings", lambda: {"google": {"timeout_sec": 42}})
    llm = llm_factory.create_chat_model(ModelSpec(provider="google", model="gemini-3.5-flash"), models)
    assert isinstance(llm.client, llm_factory._TimedClient) and llm.client._timeout == 42
    monkeypatch.setattr(core.config, "load_settings", lambda: {"google": {"timeout_sec": 0}})
    assert llm_factory.google_timeout() is None  # 0 = 제한 없음


def test_timed_client_turns_deadline_into_transient_timeout():
    from google.api_core.exceptions import DeadlineExceeded

    seen = {}

    class Inner:
        def generate_content(self, **kw):
            seen.update(kw)
            raise DeadlineExceeded("Deadline Exceeded")

    client = llm_factory._TimedClient(Inner(), 180)
    with pytest.raises(llm_factory.GeminiTimeoutError) as err:
        client.generate_content(request="r")
    assert seen["timeout"] == 180
    assert llm_factory.is_transient(err.value) and llm_factory.is_timeout(err.value)
    # 라이브러리 재시도 대상(GoogleAPIError)이 아니다 → 라이브러리 안에서 또 180초를 기다리지 않는다
    from google.api_core.exceptions import GoogleAPIError

    assert not isinstance(err.value, GoogleAPIError)


def test_timeout_retries_same_model_once_then_falls_back(models, set_keys, monkeypatch):
    set_keys("google", "openai")
    waits, calls = [], []
    monkeypatch.setattr(llm_factory, "sleep", lambda s: waits.append(s))

    class Model:
        def __init__(self, spec):
            self.spec = spec

        def with_structured_output(self, schema, **kw):
            def run(messages):
                calls.append(self.spec.provider)
                if self.spec.provider == "google":
                    raise llm_factory.GeminiTimeoutError("Gemini 응답 시간 초과 (180초)")
                return {"parsed": make_review(), "raw": AIMessage(""), "parsing_error": None}
            return RunnableLambda(run)

    monkeypatch.setattr(llm_factory, "create_chat_model", lambda spec, m: Model(spec))
    rm = RankModels(primary=ModelSpec(provider="google", model="g"), fallback=[ModelSpec(provider="openai", model="o")])
    call = llm_factory.invoke_structured(rm, models, type(make_review()), [], "비평가")
    assert calls == ["google", "google", "openai"] and call.is_fallback  # 같은 모델 1번 더 → 대안
    assert waits == [llm_factory.RETRY_DELAYS[0]]
    assert llm_factory.failure_reason(call.errors[0]) == "시간 초과"


# ─── 커리어: 표·목록의 순서 번호는 숫자 검사에서 뺀다 ───────────────────────


def test_career_table_row_numbers_are_not_unsourced():
    from core.career import unsourced_numbers

    body = (
        "| 번호 | 프로젝트 | 성과 |\n|---|---|---|\n"
        "| 13 | IPS 고도화 | 탐지율 47% 개선 |\n"
        "| 14 | LLM 보안 | 오탐 20건 감소 |\n\n"
        "13. IPS 고도화 프로젝트를 이끌었다.\n"
        "14) 오탐을 줄였다.\n"
        "- 15. 세 번째 항목\n"
        "그 결과 처리 시간이 35분 줄었다.\n"
    )
    found = unsourced_numbers(body, {"47%"})
    assert any(f.startswith("20건") for f in found)  # 표의 다른 칸 숫자는 그대로 검사
    assert any(f.startswith("35분") or f.startswith("35") for f in found)  # 문장 안 숫자도 그대로
    assert not any(f.startswith(("13", "14", "15")) for f in found)  # 줄 번호는 사실이 아니다


# ─── 화면 (AppTest) ─────────────────────────────────────────────────


@pytest.fixture
def app_env(tmp_path, monkeypatch, set_keys):
    import streamlit as st

    set_keys()
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))
    st.cache_resource.clear()
    yield tmp_path
    st.cache_resource.clear()


def _app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=90)


def test_team_screen_groups_models_by_department(app_env):
    at = _app()
    at.query_params["team"] = "general"
    at.query_params["view"] = "team"
    at.run()
    assert not at.exception, at.exception
    assert any(e.label == "🎛 부서별 모델 선택 — 🔬 리서치·지식부" for e in at.expander)
    head = at.selectbox(key="dept_head_model_team_research")
    assert head.value == "" and head.format_func("").startswith("기본 = claude-opus-5-5")
    assert at.radio(key="dept_team_pick_team_research").value == "general"  # 지금 팀부터
    at = head.set_value("openai:gpt-5.4").run()
    assert not at.exception, at.exception
    saved = json.loads((app_env / "data" / "department_models.json").read_text(encoding="utf-8"))
    assert saved == {"research": "openai:gpt-5.4"}
    summary = [m.value for m in at.markdown if m.value.startswith("- **") and "⚖️ 비평" in m.value]
    assert len(summary) == 2 and all("**gpt-5.4**" in s for s in summary)  # 부서의 두 팀 모두 이 부장이 비평
    assert any("팀장 모델" in c.value and "약합니다" in c.value for c in at.caption)  # 팀장(Opus)보다 약함
    # 같은 부서의 다른 팀 팀원 표로 바꿔 보기
    at = at.radio(key="dept_team_pick_team_research").set_value("concept").run()
    assert not at.exception, at.exception
    concept_lead = team_org(load_team("concept"), _ranks()).id
    assert any(s.key == f"member_model_concept_{concept_lead}" for s in at.selectbox)
    assert not any(s.key and s.key.startswith("member_model_general_") for s in at.selectbox)  # 한 팀만 그린다


def _ranks():
    from core.config import load_ranks

    return load_ranks()


def test_home_department_models_and_band_opens_department_screen(app_env):
    at = _app()
    at.run()
    assert not at.exception, at.exception
    box = "🎛 부서별 모델 선택 — 👔 부장(비평가) · 팀원"
    assert any(e.label == box for e in at.expander)
    assert not any(s.key and s.key.startswith("member_model_") for s in at.selectbox)  # 팀을 고르기 전에는 팀원 표 없음
    at = at.selectbox(key="dept_models_pick").set_value("security").run()  # 메인 '🎛 부서별 모델 선택'에서 부서 고르기
    assert not at.exception, at.exception
    head = at.selectbox(key="dept_head_model_home_security")
    at = head.set_value(OPUS).run()
    assert not at.exception, at.exception
    saved = json.loads((app_env / "data" / "department_models.json").read_text(encoding="utf-8"))
    assert saved == {"security": OPUS}
    at = at.radio(key="dept_team_pick_home_security").set_value("infra_security").run()
    assert not at.exception, at.exception
    assert any(s.key and s.key.startswith("member_model_infra_security_") for s in at.selectbox)
    # 빌딩의 부서 띠를 누르면 → 🏢 보안부 업무 화면 (부서장 창구 · 부서 사무실 · 같은 '🎛 부서별 모델 선택')
    at.session_state["building_nav"] = {"kind": "dept", "id": "security"}
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["view"] == "dept" and at.session_state["dept_pick"] == "security"
    assert any(e.label == "🎛 부서별 모델 선택 — 🛡️ 보안부" for e in at.expander)
    assert at.selectbox(key="dept_head_model_team_security").value == OPUS
