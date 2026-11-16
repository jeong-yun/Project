"""되풀이된 교훈 → 고정 규칙(팀 업무 지침 · 부서장 검토 기준 · 코드 검사 후보)으로 올리기.

  - 같은 피드백을 PROMOTE_HITS 번 이상 받은 켜 둔 교훈만 제안한다
  - 사람이 '⬆ 올리기'를 누르기 전에는 팀 파일이 바뀌지 않는다 (모델로 다듬기도 글 상자만 바꾼다)
  - 올린 규칙은 다음 업무 프롬프트(📜 프롬프트 보기와 같은 team_prompts)에 글자 한도와 상관없이 항상 들어가고, 원래 교훈은 꺼진다
  - 비공개 팀은 개인 정보 확인 없이는 공개 파일에 쓰지 않는다 · 사람 코멘트 원문은 어느 팀이든 공개 파일에 넣지 않는다
  - 새 검증기 concept_pending (개념 팀이 24번 되풀이한 지적을 코드로)

팀 설정 파일은 저장소에 커밋되는 파일이라, 테스트는 teams/ 를 임시 폴더에 복사해 그 사본만 바꾼다. 모델은 가짜(tests/fakes.py)만.
"""

import shutil

import pytest
import yaml
from streamlit.testing.v1 import AppTest

from core import config
from core.config import load_models, load_ranks, load_team
from core.feedback import (
    PROMOTE_HITS, polish_rule, promote_lesson, promotion_candidates, team_lessons, team_prompts,
)
from core.harness import REGISTRY
from core.schemas import Lesson, RuleDraft
from core.storage import Storage
from tests.fakes import Script

RULE = "카탈로그 필수 절 구조를 유지한다"
SECRET = "홍길동 회사 메모"  # 사람 코멘트 원문 — 공개 파일에 들어가면 안 된다


@pytest.fixture
def teams(tmp_path, monkeypatch):
    """teams/ 사본 — 저장소의 팀 파일은 건드리지 않는다 (아바타 그림은 빼고 복사)."""
    copy = tmp_path / "teams"
    shutil.copytree(config.TEAMS_DIR, copy, ignore=shutil.ignore_patterns("avatars"))
    monkeypatch.setattr(config, "TEAMS_DIR", copy)
    return copy


@pytest.fixture
def storage(tmp_path):
    return Storage(tmp_path / "runs.db")


def add(storage, team_id, agent, text, hits=1):
    return storage.add_lessons(team_id, None, [Lesson(agent_id=agent, kind="improve", lesson=text)], hits=hits)[0]


def files(folder):
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


def test_candidates_need_repeats_and_active(storage):
    once, twice = add(storage, "concept", "lead", "한 번"), add(storage, "concept", "lead", "두 번", hits=PROMOTE_HITS - 1)
    many, off = add(storage, "concept", "lead", "여러 번", hits=24), add(storage, "concept", "lead", "꺼 둔 것", hits=24)
    storage.update_lesson(off, active=False)
    assert [r["id"] for r in promotion_candidates(storage.list_lessons("concept"))] == [many]
    assert {once, twice} & {r["id"] for r in promotion_candidates(storage.list_lessons("concept"))} == set()


def test_skill_rule_reaches_every_prompt_only_after_confirm(teams, storage, use_script):
    lid = add(storage, "concept", "am_fact", f"{RULE} (사람 코멘트: {SECRET})", hits=24)
    before = files(teams)
    script = use_script(Script(RuleDraft=RuleDraft(rule=f"{RULE}.\n(사람 코멘트: {SECRET})")))
    text, notes = polish_rule(load_team("concept"), storage.list_lessons("concept")[0], "skill", load_models())
    assert text == f"{RULE}." and notes == []  # 다듬은 문장에서도 사람 코멘트 원문은 뗀다
    assert SECRET not in script.prompts("RuleDraft")[0]  # 모델에도 규칙 문장만 보낸다
    assert files(teams) == before and storage.list_lessons("concept")[0]["active"] == 1  # 다듬기만으로는 아무것도 안 바뀐다

    msg = promote_lesson(storage, load_team("concept"), lid, "skill", text)
    assert not msg.startswith("⚠️") and "teams/concept/skills/promoted_rules.md" in msg and "커밋" in msg
    skill = (teams / "concept" / "skills" / "promoted_rules.md").read_text(encoding="utf-8")
    assert f"- {RULE}." in skill and SECRET not in skill
    [row] = storage.list_lessons("concept")
    assert row["active"] == 0 and row["promoted_to"] == "skill"  # 지우지 않고 끔 + 올린 곳 기록

    team = load_team("concept")
    prompts = team_prompts(team, load_ranks(), load_models(), storage, 10)  # 글자 한도 10자 — 교훈이었다면 빠졌을 크기
    assert all(f"- {RULE}." in p.prompt for p in prompts)  # 맡은 절만 받는 실무자(skill_sections)까지 모두
    assert team_lessons(storage, "concept", 20000) == {}  # 원래 교훈은 프롬프트에 없다


def test_critic_rule_goes_to_team_yaml_and_head_prompt(teams, storage):
    import prompt_ui

    lid = add(storage, "general", "lead", "출처 없는 수치를 쓰지 않는다", hits=5)
    msg = promote_lesson(storage, load_team("general"), lid, "critic", "출처 없는 수치가 있으면 '출처 미확인' 위반이다")
    assert not msg.startswith("⚠️") and "team.yaml" in msg
    team = load_team("general")  # 일반 팀은 verification 절이 없었다 → 새로 만든다
    assert team.verification["critic"]["rules"] == ["출처 없는 수치가 있으면 '출처 미확인' 위반이다"]

    class Head:  # 부서장 — 비평 관점 한 줄만 있으면 된다
        def critic_persona(self):
            return "너는 리서치·지식부 부서장이다."

    assert "'출처 미확인' 위반이다" in prompt_ui.critic_prompt(team, Head())  # 📜 화면 = 실행과 같은 비평 프롬프트 함수
    assert storage.list_lessons("general")[0]["active"] == 0


def test_critic_rule_keeps_comments_and_other_rules(teams, storage):
    before = (teams / "concept" / "team.yaml").read_text(encoding="utf-8")
    lid = add(storage, "concept", "lead", "판정 표 근거", hits=3)
    promote_lesson(storage, load_team("concept"), lid, "critic", '근거 없는 "판정"은 위반이다')
    after = (teams / "concept" / "team.yaml").read_text(encoding="utf-8")
    assert len(after.splitlines()) == len(before.splitlines()) + 1  # 딱 한 줄만 늘었다 (주석·순서 그대로)
    rules = yaml.safe_load(after)["verification"]["critic"]["rules"]
    assert rules[:-1] == yaml.safe_load(before)["verification"]["critic"]["rules"] and rules[-1] == '근거 없는 "판정"은 위반이다'


def test_private_team_needs_confirm_before_public_file(teams, storage):
    team = load_team("career")
    assert team.private
    lid = add(storage, "career", "lead", f"경력 숫자는 문서에 있는 것만 쓴다 (사람 코멘트: {SECRET})", hits=4)
    before = files(teams)
    msg = promote_lesson(storage, team, lid, "skill", f"경력 숫자는 문서에 있는 것만 쓴다 (사람 코멘트: {SECRET})")
    assert msg.startswith("⚠️") and "비공개" in msg
    assert files(teams) == before and storage.list_lessons("career")[0]["active"] == 1

    msg = promote_lesson(storage, team, lid, "skill", f"경력 숫자는 문서에 있는 것만 쓴다 (사람 코멘트: {SECRET})", private_ok=True)
    assert not msg.startswith("⚠️")
    skill = (teams / "career" / "skills" / "promoted_rules.md").read_text(encoding="utf-8")
    assert "경력 숫자는 문서에 있는 것만 쓴다" in skill and SECRET not in skill  # 개인 정보 거르기(knowledge/ 와 같은 방식)


def test_code_candidate_only_marks(teams, storage):
    lid = add(storage, "concept", "gm_editor", RULE, hits=24)
    before = files(teams)
    msg = promote_lesson(storage, load_team("concept"), lid, "code")
    assert "코드 검사 후보" in msg and files(teams) == before
    [row] = storage.list_lessons("concept")
    assert row["active"] == 1 and row["promoted_to"] == "code"  # 교훈은 켠 채로 — 아직 규칙이 없으니까
    assert promotion_candidates([row])  # 지침·검토 기준으로도 올릴 수 있게 계속 제안
    promote_lesson(storage, load_team("concept"), lid, "skill", RULE)
    assert storage.list_lessons("concept")[0]["promoted_to"] == "code skill"


def test_polish_falls_back_without_model(storage, use_script):
    use_script(Script())  # 다듬기 모델 출력 없음 → 호출 실패
    add(storage, "concept", "lead", f"{RULE} (사람 코멘트: {SECRET})", hits=3)
    text, notes = polish_rule(load_team("concept"), storage.list_lessons("concept")[0], "critic", load_models())
    assert text == RULE and notes and "그대로" in notes[0]


def test_old_db_gets_promoted_column(tmp_path):
    import sqlite3

    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:  # 올린 곳 칸이 없던 예전 DB
        conn.execute("CREATE TABLE lessons (id INTEGER PRIMARY KEY AUTOINCREMENT, team_id TEXT NOT NULL, agent_id TEXT NOT NULL,"
                     " run_id TEXT, kind TEXT NOT NULL, lesson TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1,"
                     " created_at TEXT NOT NULL)")
        conn.execute("INSERT INTO lessons (team_id, agent_id, kind, lesson, created_at) VALUES ('concept','lead','improve','옛 교훈','2026-10-01')")
    [row] = Storage(db).list_lessons("concept")
    assert row["promoted_to"] is None and row["lesson"] == "옛 교훈"


# ─── 📚 교훈 관리 화면 ───────────────────────────────────────────────────


def _page(db, team_id, agent):
    import prompt_ui
    from core.config import load_models, load_ranks, load_team
    from core.storage import Storage

    prompt_ui.configure(prompt_ui.Ctx(storage=Storage(db), ranks=load_ranks(), models=load_models()))
    team = load_team(team_id)
    prompt_ui.render_lessons(team, next(p for p in prompt_ui.team_prompts(
        team, prompt_ui._ctx.ranks, prompt_ui._ctx.models, prompt_ui._ctx.storage, 20000) if p.node_id == agent), "m")


def test_screen_asks_then_saves_only_on_click(teams, tmp_path, use_script):
    db = tmp_path / "runs.db"
    lid = add(Storage(db), "concept", "am_fact", f"{RULE} (사람 코멘트: {SECRET})", hits=24)
    use_script(Script(RuleDraft=RuleDraft(rule="필수 절 구조(1~6절·참고 문헌)를 그대로 둔다")))
    before = files(teams)
    at = AppTest.from_function(_page, args=(db, "concept", "am_fact"), default_timeout=30).run()
    assert not at.exception, at.exception
    assert any("24번 되풀이됐습니다" in m.value for m in at.markdown)
    assert at.text_area(key=f"promote_text_{lid}").value == RULE  # 첫 문장은 사람 코멘트를 뗀 규칙 문장

    at = at.button(key=f"promote_polish_{lid}").click().run()
    assert at.text_area(key=f"promote_text_{lid}").value == "필수 절 구조(1~6절·참고 문헌)를 그대로 둔다"
    assert files(teams) == before and Storage(db).list_lessons("concept")[0]["active"] == 1  # 다듬기는 저장하지 않는다

    at.text_area(key=f"promote_text_{lid}").set_value("필수 절 구조를 그대로 둔다 (사람이 고침)")
    at = at.button(key=f"promote_go_{lid}").click().run()
    assert not at.exception, at.exception
    assert "필수 절 구조를 그대로 둔다 (사람이 고침)" in (teams / "concept" / "skills" / "promoted_rules.md").read_text(encoding="utf-8")
    assert any("커밋" in s.value for s in at.success)
    assert Storage(db).list_lessons("concept")[0]["active"] == 0
    assert not any("되풀이됐습니다" in m.value for m in at.markdown)  # 꺼진 교훈은 더 제안하지 않는다


def test_screen_warns_private_team_and_needs_checkbox(teams, tmp_path):
    db = tmp_path / "runs.db"
    lid = add(Storage(db), "career", "lead", "경력 숫자는 문서에 있는 것만 쓴다", hits=3)
    before = files(teams)
    at = AppTest.from_function(_page, args=(db, "career", "lead"), default_timeout=30).run()
    assert not at.exception, at.exception
    assert any("비공개 팀" in w.value for w in at.warning)
    at = at.button(key=f"promote_go_{lid}").click().run()  # 체크하지 않고 누름 → 바꾸지 않음
    assert files(teams) == before and any("바꾸지 않았습니다" in w.value for w in at.warning)
    at = at.checkbox(key=f"promote_ok_{lid}").check().run()
    at = at.button(key=f"promote_go_{lid}").click().run()
    assert (teams / "career" / "skills" / "promoted_rules.md").exists()


def test_no_promotion_box_for_lessons_given_once(teams, tmp_path):
    db = tmp_path / "runs.db"
    add(Storage(db), "concept", "lead", "한 번 받은 지적")
    at = AppTest.from_function(_page, args=(db, "concept", "lead"), default_timeout=30).run()
    assert not at.exception, at.exception
    assert not any("되풀이" in m.value for m in at.markdown)


# ─── 새 검증기 concept_pending ───────────────────────────────────────────


def test_concept_pending_flags_marks_left_in_final():
    from tests.test_concept import GOOD_ENTRY, PLAN, answer_with

    check = REGISTRY["concept_pending"].check
    assert check(answer_with(GOOD_ENTRY, PLAN), {}).passed
    assert check(answer_with(GOOD_ENTRY + "\n - 후속 반박은 확인되지 않음", PLAN), {}).passed  # 스킬이 쓰라는 문구는 괜찮다
    bad = check(answer_with(GOOD_ENTRY + "\n - PENDING: 2025 벤치마크 수치", PLAN), {})
    assert not bad.passed and "<review_log>" in bad.log
    pending_in_log = answer_with(GOOD_ENTRY, PLAN).replace("</review_log>", "PENDING: 원전 미확인 수치\n</review_log>")
    assert check(pending_in_log, {}).passed  # 검증 로그에 모아 두는 것은 맞는 자리


def test_concept_pending_checks_change_proposals():
    from tests.test_concept_changes import FACT, OPTIONS, answer

    check = REGISTRY["concept_pending"].check
    assert check(answer([FACT]), OPTIONS).passed
    assert not check(answer([{**FACT, "replace": FACT["replace"] + " (미확인 후보)"}]), OPTIONS).passed


def test_concept_team_runs_the_pending_check():
    team = load_team("concept")
    assert "concept_pending" in team.harness.validators and "concept_pending" in team.harness.final_only
