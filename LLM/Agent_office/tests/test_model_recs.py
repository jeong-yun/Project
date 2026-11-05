"""⭐ 역할별 추천 모델 (config/model_recommendations.yaml · core/model_recs.py)

적용 순서(고른 모델 > 추천 > 직급 기본), 상하 규칙, 모델 능력(그림 역할), 비공개 팀에서 무료 모델 제외, '🆓 무료 모델만',
사람 선택 유지, 되돌리기(한 명·부서 전체), 예상 비용, 진행률 추정, 화면(AppTest). 진짜 모델은 부르지 않는다.
"""

import json
from pathlib import Path

import pytest

from core import building, dept_heads, live, llm_factory, member_models, model_recs, progress
from core.config import CONFIG_DIR, list_teams, load_models, load_ranks, rank_models, team_org
from core.member_models import plan

BASE_DIR = Path(__file__).resolve().parent.parent
OPUS = "claude_cli:claude-opus-5-5"
FREE = "freellm"


@pytest.fixture
def recs_on():
    """실제 config/model_recommendations.yaml 을 켠다 (conftest 가 기본으로 끔)."""
    model_recs.configure(CONFIG_DIR / model_recs.FILE_NAME)
    yield
    model_recs.configure(None)


@pytest.fixture
def custom_recs(tmp_path):
    """테스트용 추천 파일을 쓰고 켠다. 사용: custom_recs({...})"""
    import yaml

    def _use(data: dict) -> Path:
        path = tmp_path / "recs.yaml"
        path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        model_recs.configure(path)
        return path

    yield _use
    model_recs.configure(None)


@pytest.fixture
def teams():
    return list_teams()


def _team(teams, team_id):
    return next(t for t in teams if t.id == team_id)


def _depts(teams):
    return building.departments([t.id for t in teams], building.load_config())


def tier(models, key):
    return member_models.tier_map(models).get(key, 999)


def keys(chain):
    return [member_models.model_key(s) for s in chain.candidates]


# ─── 설정 파일 점검 ──────────────────────────────────────────────────


def test_shipped_recommendations_pass_every_check(recs_on, ranks, models, teams):
    """추천 파일이 설정·규칙과 맞는다 — 없는 팀·사람·모델, 그림 역할, 상하 규칙(부서장 ≥ 팀장 ≥ …) 위반이 없음."""
    assert model_recs.check(models, teams, ranks, _depts(teams)) == []


def test_every_person_in_every_team_has_a_recommendation_with_reason(recs_on, ranks, models, teams):
    recs = model_recs.load()
    for t in teams:
        if t.id not in recs.teams:
            continue
        for node in team_org(t, ranks).walk():
            rec = model_recs.member_rec(t, node, models)
            assert rec is not None and rec.why, f"{t.id}.{node.id}"
    assert set(recs.heads) == {d.id for d in _depts(teams) if d.id != "etc"}


def test_check_reports_broken_entries(custom_recs, ranks, models, teams):
    custom_recs({
        "roles": {"r": {"chain": ["openai:gpt-5.4"]}},
        "heads": {"nowhere": {"role": "r", "why": "x"}},
        "teams": {
            "general": {
                "lead": {"chain": ["openai:gpt-5.4-mini"], "why": "약한 팀장"},
                "gm_research": {"chain": ["openai:gpt-5.5"], "why": "팀장보다 강함"},
                "ghost": {"role": "r", "why": "x"},
                "staff_draft": {"role": "없는역할", "chain": ["openai:not-a-model", "openai:gpt-5.4-nano"], "why": "x"},
            },
            "design": {"am_program": {"chain": [f"{FREE}:gpt-oss-120b"], "needs": ["image"], "why": "그림"}},
            "career": {"staff_facts": {"chain": [f"{FREE}:gpt-oss-20b", "openai:gpt-5.4-nano"], "why": "x"}},
            "no_team": {"lead": {"role": "r", "why": "x"}},
        },
    })
    text = "\n".join(model_recs.check(models, teams, ranks, _depts(teams)))
    assert "heads.nowhere: 없는 부서" in text
    assert "teams.no_team: 없는 팀" in text
    assert "teams.general.ghost: 조직도에 없는 사람" in text
    assert "없는 역할 '없는역할'" in text
    assert "openai:not-a-model 가 models.yaml model_tiers 에 없습니다" in text
    assert "상하 규칙 위반" in text and "gpt-5.5" in text  # 부장이 팀장보다 강함
    assert "그림을 읽는 역할인데 맨 앞 모델 gpt-oss-120b" in text
    assert "🔒 비공개 팀인데 freellm:gpt-oss-20b" in text


# ─── 적용 순서: 고른 모델 > 추천 > 직급 기본 ─────────────────────────


def test_order_choice_then_recommendation_then_rank(recs_on, ranks, models, teams, org, team):
    rec = model_recs.member_rec(team, next(n for n in org.walk() if n.id == "dgm_research"), models)
    p = plan(org, models, team, {})
    assert p["dgm_research"].source == "rec" and p["dgm_research"].effective == member_models.model_key(rec.chain.primary)
    assert p["dgm_research"].recommended == "google:gemini-3.8-flash" and p["dgm_research"].why
    # 사람이 고르면 그것
    p = plan(org, models, team, {"dgm_research": "openai:gpt-5.4-mini"})
    assert p["dgm_research"].effective == "openai:gpt-5.4-mini" and p["dgm_research"].choice
    # 추천 파일이 꺼지면 직급 기본 (예전과 같음)
    model_recs.configure(None)
    p = plan(org, models, team, {})
    assert p["dgm_research"].source == "rank"
    assert p["dgm_research"].effective == member_models.model_key(rank_models("deputy_gm", models, team).primary)


def test_team_without_recommendation_keeps_rank_defaults(custom_recs, ranks, models, org, team):
    custom_recs({"teams": {"stock": {"lead": {"chain": ["openai:gpt-5.4"], "why": "x"}}}})
    chains = member_models.team_rank_models(org, models, team, {})
    assert all(chains[n.id] == rank_models(n.rank, models, team) for n in org.walk())


def test_recommended_chain_is_used_at_runtime(recs_on, ranks, models, org, team):
    chains = member_models.team_rank_models(org, models, team, {})
    assert keys(chains["staff_draft"])[0] == "google:gemini-3.5-flash-lite"
    assert keys(chains["am_action"])[0] == "openai:gpt-5.4-nano"  # 사원·대리끼리 Google·OpenAI 를 나눠 Gemini 한도 분산
    assert keys(chains[org.id])[0] == OPUS


# ─── 상하 규칙 ───────────────────────────────────────────────────────


def test_recommendations_respect_hierarchy_in_every_team(recs_on, ranks, models, teams):
    for t in teams:
        org = team_org(t, ranks)
        plans = plan(org, models, t, {})
        for node in org.walk():
            for child in node.children:
                assert tier(models, plans[child.id].effective) >= tier(models, plans[node.id].effective), (t.id, child.id)


def test_head_is_at_least_as_strong_as_every_lead(recs_on, ranks, models, teams):
    by_id = {t.id: t for t in teams}
    for d in _depts(teams):
        if d.id == "etc":
            continue
        chain, source = dept_heads.head_chain(models, "", d.id, dept_heads.dept_private(d, by_id))
        assert source == "recommended"
        for tid in d.teams:
            org = team_org(by_id[tid], ranks)
            lead = plan(org, models, by_id[tid], {})[org.id].effective
            assert tier(models, member_models.model_key(chain.primary)) <= tier(models, lead), (d.id, tid)


def test_recommendation_stronger_than_superior_is_capped(custom_recs, ranks, models, org, team):
    custom_recs({"teams": {"general": {
        "lead": {"chain": ["openai:gpt-5.4-mini"], "why": "약한 팀장"},
        "gm_research": {"chain": ["openai:gpt-5.5", "google:gemini-3.8-flash", "google:gemini-3.5-flash"], "why": "강함"},
    }}})
    p = plan(org, models, team, {})
    gm = p["gm_research"]
    assert gm.limited and gm.effective == "google:gemini-3.5-flash"  # 추천 목록에서 팀장보다 강하지 않은 첫 모델
    chain = member_models.member_rank_models(gm, models, team)
    assert all(tier(models, k) >= tier(models, "openai:gpt-5.4-mini") for k in keys(chain))  # 대안도 상사를 넘지 않음


def test_unavailable_superior_lowers_recommended_subordinate(custom_recs, ranks, models, org, team):
    """팀장 추천(구독 Opus)이 지금 안 되면 팀장은 다음 후보 → 그 아래 추천도 그 모델을 넘지 않게 낮춘다."""
    custom_recs({"teams": {"general": {
        "lead": {"chain": [OPUS, "openai:gpt-5.4-mini"], "why": "팀장"},
        "gm_research": {"chain": ["openai:gpt-5.5", "openai:gpt-5.4-nano"], "why": "부장"},
    }}})
    p = plan(org, models, team, {}, {"claude_cli": "not_installed"})
    assert p[org.id].effective == "openai:gpt-5.4-mini"
    assert p["gm_research"].limited and p["gm_research"].effective == "openai:gpt-5.4-nano"
    p = plan(org, models, team, {}, {"claude_cli": "ok"})  # 구독이 되면 그대로
    assert p[org.id].effective == OPUS and not p["gm_research"].limited


def test_chosen_superior_caps_recommended_subordinates(recs_on, ranks, models, org, team):
    p = plan(org, models, team, {"gm_research": "openai:gpt-5.4-mini"})
    assert tier(models, p["dgm_research"].effective) > tier(models, "openai:gpt-5.4-mini")  # 직접 고른 상사보다 '낮은' 줄
    assert p["dgm_research"].recommended == "google:gemini-3.8-flash"  # 추천 자체는 그대로 (표시용)


# ─── 모델 능력 (그림 역할) ───────────────────────────────────────────


def test_image_roles_start_with_a_model_that_reads_images(recs_on, ranks, models, teams):
    from core import model_caps

    found = 0
    for t in teams:
        for node in team_org(t, ranks).walk():
            rec = model_recs.member_rec(t, node, models)
            if rec and rec.needs_image:
                found += 1
                assert model_caps.can(rec.keys[0], model_caps.IMAGE_READING), (t.id, node.id)
    assert found >= 2  # 설계 방 프로그램 · 커리어 사실 추출


def test_model_that_cannot_do_team_work_is_skipped(custom_recs, ranks, models, org, team, monkeypatch):
    from core import model_caps

    real = model_caps.can
    monkeypatch.setattr(model_caps, "can", lambda key, needs, *a, **k: key != "openai:gpt-5.4" and real(key, needs, *a, **k))
    custom_recs({"teams": {"general": {"gm_research": {"chain": ["openai:gpt-5.4", "google:gemini-3.8-flash"], "why": "x"}}}})
    rec = model_recs.member_rec(team, next(n for n in org.walk() if n.id == "gm_research"), models)
    assert rec.keys == ["google:gemini-3.8-flash"] and any("gpt-5.4" in d for d in rec.dropped)


# ─── 비공개 팀 · 무료만 ──────────────────────────────────────────────


def test_private_teams_never_get_free_models_by_default(recs_on, ranks, models, teams):
    for t in teams:
        if not t.private:
            continue
        for node in team_org(t, ranks).walk():
            rec = model_recs.member_rec(t, node, models)
            assert rec is not None and all(not k.startswith(FREE + ":") for k in rec.keys), (t.id, node.id)
    career = _team(teams, "career")
    org = team_org(career, ranks)
    chains = member_models.team_rank_models(org, models, career, {})
    head = dept_heads.head_of(next(d for d in _depts(teams) if d.id == "career"))
    critic = dept_heads.critic_plan(models, career, chains, org.id, head)
    assert critic.source == "recommended" and all(s.provider != FREE for s in critic.chain.candidates)
    assert not critic.warnings  # 비평가 제공처도 팀원 제공처 안


def test_free_only_mode_uses_free_models_and_private_falls_back_with_warning(recs_on, ranks, models, teams):
    free = llm_factory.free_only(models)
    general = _team(teams, "general")
    org = team_org(general, ranks)
    assert all(p.effective.startswith(FREE + ":") for p in plan(org, free, general, {}).values())
    career = _team(teams, "career")
    corg = team_org(career, ranks)
    assert all(p.source == "rank" for p in plan(corg, free, career, {}).values())  # 추천이 모두 꺼져 직급 기본으로
    assert model_recs.private_free_fallback(career, corg, free)
    assert not model_recs.private_free_fallback(career, corg, models)
    chain, source = dept_heads.head_chain(free, "", "research")
    usable = [s for s in chain.candidates if s.provider not in free.disabled_providers]
    assert source == "recommended" and usable and all(s.provider == FREE for s in usable)


# ─── 사람 선택 유지 · 저장 ───────────────────────────────────────────


def test_saved_choices_are_kept_and_win(recs_on, ranks, models, org, team, tmp_path):
    store = member_models.MemberModelStore(tmp_path)
    store.save("general", {"staff_draft": "openai:gpt-5.4-nano"})
    before = store.path.read_text(encoding="utf-8")
    p = plan(org, models, team, store.load("general"))
    assert p["staff_draft"].effective == "openai:gpt-5.4-nano"  # 추천(gemini-3.5-flash-lite)보다 사람 선택
    assert p["staff_draft"].recommended == "google:gemini-3.5-flash-lite"
    assert store.path.read_text(encoding="utf-8") == before  # 추천은 저장 파일을 건드리지 않는다


def test_reverting_one_person_only_clears_that_choice(recs_on, ranks, models, org, team):
    choices = {"staff_draft": "openai:gpt-5.4-nano", "am_fact": "google:gemini-3.5-flash-lite"}
    new, _ = member_models.apply_choice(org, models, team, choices, "staff_draft", "")
    assert new == {"am_fact": "google:gemini-3.5-flash-lite"}
    assert plan(org, models, team, new)["staff_draft"].source == "rec"


# ─── 예상 비용 · 진행률 ───────────────────────────────────────────────


def test_estimate_counts_subscription_calls_and_cost(recs_on, ranks, models, org, team):
    plans = plan(org, models, team, {})
    est = model_recs.estimate_team(org, {n: p.effective for n, p in plans.items()}, OPUS)
    assert est.subscription_calls == 2 + 1  # 팀장(분배 + 통합) + 부서장 비평
    assert est.paid_calls == 4 * 2 + 3 * 1  # 관리자 4명(부장 2·차장·하위 있는 대리) × 2 + 초안 3명 × 1
    assert est.usd > 0 and not est.unknown
    text = model_recs.estimate_text(est)
    assert text.startswith("1회 예상 $") and "구독 호출 3회" in text and "원)" in text
    # 지난 기록이 있으면 그 평균 토큰을 쓴다 — 토큰이 늘면 비용도 는다
    hist = {n.id: [{"calls": 1, "tokens": 400_000}] for n in org.walk()}
    assert model_recs.estimate_team(org, {n: p.effective for n, p in plans.items()}, OPUS, hist).usd > est.usd


def test_progress_estimate_works_with_recommended_models(recs_on, ranks, models, org, team):
    """진행률·남은 시간은 모델별 기록이 없으면 같은 단계의 기록·기본값으로 — 기본 모델이 바뀌어도 깨지지 않는다."""
    plans = plan(org, models, team, {})
    p = progress.Plan(org=org, root_id=org.id, node_models={n: x.effective for n, x in plans.items()},
                      critic_model=OPUS, utility_model="google:gemini-3.5-flash")
    run = live.new_run("r1", "질문", org.id, "general")
    past = [{"kind": "draft", "provider": "openai", "model": "gpt-5.4", "status": "done", "started_at": 0, "ended_at": s}
            for s in (30, 50)]  # 예전 기본 모델의 기록만 있음
    est = progress.estimate(p, run, [], past)
    drafts = [u for u in est.units if u.kind == "draft"]
    assert drafts and all(u.est_sec == 40 and u.basis == "kind" for u in drafts)  # 같은 단계 기록의 중앙값
    assert est.remaining_sec > 0 and est.total == len(est.units)


# ─── 화면 (AppTest) ──────────────────────────────────────────────────


@pytest.fixture
def app_env(tmp_path, monkeypatch, set_keys, recs_on):
    import streamlit as st

    set_keys()
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))
    st.cache_resource.clear()
    yield tmp_path
    st.cache_resource.clear()


def _app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=90)
    at.query_params["team"] = "general"
    at.query_params["view"] = "team"
    return at


def test_screen_shows_recommendation_and_reverts(app_env):
    data = app_env / "data"
    data.mkdir(parents=True)
    (data / "member_models.json").write_text(json.dumps({
        "general": {"staff_draft": "openai:gpt-5.4-nano"},
        "stock": {"lead": "openai:gpt-5.5"},
    }), encoding="utf-8")
    (data / "department_models.json").write_text(json.dumps({"research": "openai:gpt-5.5", "invest": "openai:gpt-5.4"}),
                                                 encoding="utf-8")
    at = _app().run()
    assert not at.exception, at.exception
    assert not any("추천 모델 점검" in e.label for e in at.expander)  # 점검 경고 없음
    gm = at.selectbox(key="member_model_general_gm_research")
    assert gm.value == "" and gm.format_func("").startswith("⭐ 추천: gpt-5.4")
    assert "사실관계 검토" in gm.format_func("")  # 이유 한 줄
    head = at.selectbox(key="dept_head_model_team_research")
    assert head.value == "openai:gpt-5.5" and head.format_func("").startswith("⭐ 추천: claude-opus-5-5")
    assert any(m.value.startswith("✋ 추천과 다름") for m in at.markdown)  # 사원·초안
    assert any("직접 고름" in m.value for m in at.markdown)  # 부장
    assert any(c.value.startswith("✋ 이 부서에 ⭐ 추천과 다르게 고른 사람이 2명") for c in at.caption)
    assert any("💰 1회 예상 \\$" in c.value and "구독 호출" in c.value for c in at.caption)  # $ 는 수식이 안 되게 \$
    assert any("⭐" in m.value and "팀장" in m.value for m in at.markdown if m.value.startswith("- **"))

    at = at.button(key="member_rec_general_staff_draft").click().run()  # 한 명 되돌리기
    assert not at.exception, at.exception
    saved = json.loads((data / "member_models.json").read_text(encoding="utf-8"))
    assert "general" not in saved and saved["stock"] == {"lead": "openai:gpt-5.5"}

    at = at.button(key="dept_rec_all_team_research").click().run()  # 부서 전체 되돌리기 (부장)
    assert not at.exception, at.exception
    assert json.loads((data / "department_models.json").read_text(encoding="utf-8")) == {"invest": "openai:gpt-5.4"}
    assert json.loads((data / "member_models.json").read_text(encoding="utf-8")) == {"stock": {"lead": "openai:gpt-5.5"}}
    assert at.selectbox(key="dept_head_model_team_research").value == ""
