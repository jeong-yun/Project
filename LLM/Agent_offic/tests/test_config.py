"""설정 파일·조직도·아바타·스킬 테스트."""

import pytest

from core.config import (
    ConfigError,
    default_org,
    list_teams,
    load_skills,
    load_team,
    parent_map,
    rank_models,
    resolve_avatar,
    subtree,
    validate_config,
)
from core.schemas import OrgNode, Rank, Team


def test_shipped_config_is_valid(ranks, models):
    validate_config(ranks, models, list_teams())
    assert [r.name for r in ranks] == ["사원", "대리", "차장", "부장", "팀장"]


def test_template_team_is_hidden_but_valid(ranks, models):
    assert "_template" not in [t.id for t in list_teams()]
    validate_config(ranks, models, [load_team("_template")])


def test_general_team_org_is_supervisor_tree(org):
    assert org.id == "lead" and [c.id for c in org.children] == ["gm_research", "gm_action"]
    assert parent_map(org)["staff_draft"] == "dgm_research"
    assert [n.id for n in subtree(org, "gm_action").walk()] == ["gm_action", "am_action", "staff_case"]


def test_default_org_is_rank_chain(ranks):
    org = default_org(ranks)
    assert [n.id for n in org.walk()] == ["team_lead", "gm", "deputy_gm", "assistant_manager", "staff"]


def test_duplicate_node_id_and_unknown_validator_rejected(ranks, models, team):
    dup = Team(**{**team.model_dump(), "org": OrgNode(id="a", rank="gm", children=[OrgNode(id="a", rank="staff")])})
    with pytest.raises(ConfigError, match="중복"):
        validate_config(ranks, models, [dup])
    bad = Team(**{**team.model_dump(), "harness": {"validators": ["nope"]}})
    with pytest.raises(ConfigError, match="검증기"):
        validate_config(ranks, models, [bad])


def test_missing_rank_model_is_rejected(ranks, models):
    models.ranks.pop("gm")
    with pytest.raises(ConfigError, match="gm"):
        validate_config(ranks, models, [])


def test_team_override_takes_priority(models, team):
    override = models.ranks["team_lead"]
    team.model_overrides["staff"] = override
    assert rank_models("staff", models, team) is override


def test_avatar_resolution(ranks, team, org):
    staff = next(r for r in ranks if r.id == "staff")
    node = subtree(org, "staff_draft")
    assert resolve_avatar(team.id, staff, node=node).name == "staff.png"
    assert resolve_avatar(team.id, staff, working=True, node=node).name == "staff_working.png"
    missing = Rank(id="nobody", name="없음", role="-", avatar="assets/avatars/nobody.png")
    assert resolve_avatar(team.id, missing) is None


def test_skills_are_loaded_from_team_folder():
    skills = load_skills("general")
    assert "적대적 리뷰어" in skills and "## 스킬: answer_style" in skills
    assert load_skills("_template") == ""
