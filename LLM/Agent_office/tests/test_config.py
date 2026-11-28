"""설정 파일·조직도·아바타·스킬 테스트."""

import subprocess
import sys
from pathlib import Path

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


def _streamlit_reads(expression: str) -> str:
    """이 폴더(Agent_office)에서 Streamlit 이 실제로 읽은 설정으로 expression 을 계산한 결과 — 새 프로세스로 확인한다.

    파일이 빠지거나, 값이 바뀌거나, 새 Streamlit 에서 옵션이 없어지면 이것을 쓰는 테스트가 실패한다."""
    done = subprocess.run([sys.executable, "-c", f"from streamlit import config; print({expression})"],
                          cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True, timeout=120)
    return done.stdout.strip() or done.stderr


def test_streamlit_config_never_sends_big_screen_parts_as_hash_only():
    """`.streamlit/config.toml` — 큰 화면 조각을 '해시만' 보내는 기능을 끈다 (브라우저 'Connection error … 404' 방지).

    웹소켓으로 보낼 수 있는 가장 큰 메시지(server.maxMessageSize, MB)보다 기준이 크면 해시만 보내는 일이 없다.
    """
    assert _streamlit_reads("config.get_option('global.minCachedMessageSize') >= "
                            "config.get_option('server.maxMessageSize') * 1e6") == "True"


def test_streamlit_config_sends_no_usage_stats():
    """`.streamlit/config.toml` — Streamlit 사용 통계를 끈다 (브라우저 'Bad message format — Tried to use SessionInfo
    before it was initialized' 방지).

    켜져 있으면 서버가 화면 실행이 끝날 때마다 '통계용 메시지'(pageProfile)를 보낸다. Streamlit 1.43 의 화면 코드는
    연결이 끊겼다 붙은 직후 이 메시지를 받으면 오류 창을 띄운다 (2026-11-12 재현: 켬 9/10 → 끔 0/10).
    """
    assert _streamlit_reads("config.get_option('browser.gatherUsageStats')") == "False"


def test_streamlit_config_does_not_scan_module_files_for_new_sessions():
    """`.streamlit/config.toml` — 코드 파일 감시를 끈다 (새 브라우저 세션마다 서버가 몇 초씩 멈추는 것을 막는다).

    켜져 있으면 Streamlit 은 새 세션의 첫 화면 실행이 끝날 때 불러온 모듈 약 1,900개의 파일을 이벤트 루프 위에서 훑고,
    그동안 모든 화면이 멈춘다 (2026-11-12 측정은 today_worked.md). 꺼 두면 코드를 고친 뒤 앱을 다시 켜야 반영된다.
    """
    assert _streamlit_reads("config.get_option('server.fileWatcherType')") == "none"


def test_streamlit_config_opens_app_only_on_this_pc():
    """`.streamlit/config.toml` — 앱을 이 PC 안쪽 주소(127.0.0.1)에서만 연다.

    앱에는 로그인이 없어서, 같은 와이파이·랜의 다른 기기가 들어오면 내 API 키로 업무를 돌리고 지난 기록·올린 파일을 볼 수 있다.
    Streamlit 기본값은 모든 네트워크 주소에서 여는 것이다 (방화벽이 막아 주더라도 '허용'을 한 번 누르면 열린다 — 2026-11-23).
    """
    assert _streamlit_reads("config.get_option('server.address')") == "127.0.0.1"


def test_streamlit_config_accepts_only_this_pc_host_names():
    """`.streamlit/config.toml` — 주소창 이름이 localhost·127.0.0.1 일 때만 화면 연결(웹소켓)을 받는다.

    앱에는 로그인이 없다. 이 PC 안쪽 주소에서만 열어도, 브라우저로 연 다른 사이트가 자기 이름을 127.0.0.1 로 다시 가리키게
    하면(DNS 리바인딩) 그 이름으로 이 앱에 붙을 수 있다. Streamlit 은 허용 이름을 정하지 않으면 어떤 이름이든 받는다
    (2026-11-28 보안 점검). 설정값과, Streamlit 이 그 값으로 실제로 거르는지를 함께 본다 (포트는 보지 않는다).
    """
    allowed = "__import__('importlib').import_module('streamlit.web.server.starlette.starlette_websocket')._is_origin_allowed"
    assert _streamlit_reads(
        "(sorted(config.get_option('server.allowedHosts')), "
        f"{allowed}('http://localhost:8501', 'localhost:8501'), "
        f"{allowed}('http://127.0.0.1:8600', '127.0.0.1:8600'), "
        f"{allowed}('http://rebind.example:8501', 'rebind.example:8501'))"
    ) == "(['127.0.0.1', 'localhost'], True, True, False)"
