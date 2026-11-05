"""바뀐 파일 → 테스트 고르기 규칙 (select_tests.py)."""

import select_tests as st

SOURCES = {
    "tests/test_feeds.py": "from core import feeds\nfrom tests.fakes import Script\n",
    "tests/test_home.py": "import pytest\n\ndef test_x():\n    from core.feeds import collect\n",
    "tests/test_config.py": "from core.config import load_models\n# models.yaml 을 읽는다\n",
    "tests/test_teams.py": "from core.config import list_teams\n",
    "tests/test_edu_hobby.py": "from streamlit.testing.v1 import AppTest\nimport edu_hobby_ui\n",
    "tests/test_education.py": "from core import learning\n",
    "tests/test_tools.py": "from core.tools import run_tools\n",
    "tests/test_unrelated.py": "import json\n# core.feedsX 는 다른 모듈\nfrom core.feedsx import a\n",
}


def test_split_changed_separates_agent_office():
    ao, other = st.split_changed(["LLM/Agent_office/core/feeds.py", "GAME/Soccer/x.py", "README.md"])
    assert ao == ["core/feeds.py"]
    assert other == ["GAME/Soccer/x.py", "README.md"]


def test_module_name():
    assert st.module_name("core/feeds.py") == "core.feeds"
    assert st.module_name("core/tools/__init__.py") == "core.tools"
    assert st.module_name("edu_hobby_ui.py") == "edu_hobby_ui"
    assert st.module_name("README.md") is None
    assert st.module_name("scripts/my-script.py") is None  # import 할 수 없는 이름


def test_imported_modules_includes_from_import_names_and_lazy_imports():
    names = st.imported_modules(SOURCES["tests/test_home.py"])
    assert "core.feeds" in names and "pytest" in names
    assert {"core", "core.feeds"} <= st.imported_modules("from core import feeds")
    assert st.imported_modules("def (:") == set()  # 문법 오류는 빈 집합


def test_module_change_picks_name_pair_first_and_importers_second():
    plan = st.plan_agent_office(["core/feeds.py"], SOURCES)
    assert plan.first == {"tests/test_feeds.py": "core/feeds.py 의 이름 짝"}
    assert set(plan.second) == {"tests/test_home.py"}  # test_feeds 는 1순위에 이미 있음
    assert "tests/test_unrelated.py" not in plan.second  # core.feedsx 는 다른 모듈


def test_ui_file_pairs_with_and_without_ui_suffix():
    plan = st.plan_agent_office(["edu_hobby_ui.py"], SOURCES)
    assert "tests/test_edu_hobby.py" in plan.first


def test_package_init_selects_importers_of_package():
    plan = st.plan_agent_office(["core/tools/__init__.py"], SOURCES)
    assert "tests/test_tools.py" in plan.first  # 이름 짝 (폴더 이름)
    assert plan.second == {}


def test_changed_test_file_is_first():
    plan = st.plan_agent_office(["tests/test_home.py"], SOURCES)
    assert plan.first == {"tests/test_home.py": "바뀐 테스트 파일"}


def test_deleted_test_file_is_ignored():
    plan = st.plan_agent_office(["tests/test_gone.py"], SOURCES)
    assert plan.first == {} and plan.smoke_reasons == []


def test_broad_files_add_smoke():
    plan = st.plan_agent_office(["app.py", "requirements.txt", "tests/fixtures/a.json"], SOURCES)
    assert len(plan.smoke_reasons) == 3
    assert plan.first == {}


def test_fakes_change_adds_its_importers_as_second():
    plan = st.plan_agent_office(["tests/fakes.py"], SOURCES)
    assert plan.smoke_reasons
    assert set(plan.second) == {"tests/test_feeds.py"}


def test_config_file_picks_test_config_and_mentions():
    plan = st.plan_agent_office(["config/models.yaml"], SOURCES)
    assert plan.first == {"tests/test_config.py": "설정 파일 config/models.yaml"}
    assert plan.second == {}  # 본문에 나오는 곳이 test_config 하나뿐 → 이미 1순위


def test_team_config_picks_test_teams_and_team_test():
    plan = st.plan_agent_office(["teams/edu_hobby/team.yaml", "teams/general/prompts/a.md"], SOURCES)
    assert set(plan.first) == {"tests/test_teams.py", "tests/test_edu_hobby.py"}


def test_docs_select_nothing():
    plan = st.plan_agent_office(["README.md", "docs/a.md", "knowledge/x.md"], SOURCES)
    assert plan.first == {} and plan.second == {} and not plan.smoke_reasons
    assert plan.no_test == ["README.md", "docs/a.md", "knowledge/x.md"]


def test_finalize_includes_second_under_cap():
    plan = st.plan_agent_office(["core/feeds.py"], SOURCES)
    out = st.finalize(plan, smoke=[], max_files=8)
    assert set(out.run) == {"tests/test_feeds.py", "tests/test_home.py"}
    assert out.left_to_ci == []


def test_finalize_leaves_second_to_ci_over_cap_but_keeps_first():
    plan = st.plan_agent_office(["core/feeds.py"], SOURCES)
    out = st.finalize(plan, smoke=[], max_files=1)
    assert set(out.run) == {"tests/test_feeds.py"}
    assert out.left_to_ci == ["tests/test_home.py"]
    assert set(st.finalize(plan, smoke=[], max_files=1, run_all=True).run) == {"tests/test_feeds.py", "tests/test_home.py"}


def test_finalize_smoke_counts_files_and_skips_duplicates():
    plan = st.plan_agent_office(["app.py", "tests/test_config.py"], SOURCES)
    out = st.finalize(plan, smoke=["tests/test_config.py::test_a", "tests/test_teams.py"], max_files=8)
    assert list(out.run) == ["tests/test_config.py", "tests/test_teams.py"]  # 파일 전체가 이미 있으면 node 는 안 더함
    assert out.run["tests/test_teams.py"].startswith("대표 묶음")


def test_other_project_nearest_tests_folder():
    dirs = {"GAME/Soccer", "LLM/train_set_making_by_rule(talos, et)", "tools/prepush"}

    def has_tests(d):
        return d in dirs

    assert st.other_project_of("GAME/Soccer/src/a.py", has_tests) == "GAME/Soccer"
    assert st.other_project_of("GAME/Soccer/tests/test_a.py", has_tests) == "GAME/Soccer"
    assert st.other_project_of("LLM/train_set_making_by_rule(talos, et)/3_0b_mask.py", has_tests) == \
        "LLM/train_set_making_by_rule(talos, et)"
    assert st.other_project_of("Titanic/a.ipynb", has_tests) is None
    assert st.other_project_of("README.md", has_tests) is None  # 저장소 루트는 보지 않음
    assert st.other_project_of(".githooks/pre-push", has_tests) == "tools/prepush"


def test_plan_other_groups_by_project():
    out = st.plan_other(["GAME/Soccer/a.py", "GAME/Soccer/b.py", "PROJECTS.md"], lambda d: d == "GAME/Soccer")
    assert out == {"GAME/Soccer": ["GAME/Soccer/a.py", "GAME/Soccer/b.py"]}


def test_read_smoke_skips_comments():
    assert st.read_smoke("# 설명\n\ntests/test_a.py\n  tests/test_b.py::t  \n") == ["tests/test_a.py", "tests/test_b.py::t"]


def test_smoke_list_points_at_real_tests():
    """smoke_tests.txt 에 적힌 테스트 파일이 실제로 있는지 (파일 이름이 바뀌면 여기서 알려 준다)."""
    from pathlib import Path

    here = Path(st.__file__).resolve().parent
    ao = here.parent.parent / st.AO_DIR
    smoke = st.read_smoke((here / "smoke_tests.txt").read_text(encoding="utf-8"))
    assert smoke
    for node in smoke:
        assert (ao / node.split("::", 1)[0]).is_file(), node
