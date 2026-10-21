"""실시간 진행 상태(core/live.py) · 백그라운드 실행 · 에이전트 이미지 바꾸기(core/avatars.py) 테스트."""

import io

import pytest
from PIL import Image

from core import avatars
from core.config import resolve_avatar, subtree
from core.live import apply_event, consume, new_run, start_run
from core.llm_factory import NoAvailableModelError
from core.schemas import AgentAnswer, ModelSpec
from core.workflow import build_office_graph, stream_office
from tests.fakes import Script, make_delegation, make_review
from tests.test_workflow import BRIEF, NO_CRITIC, SETTINGS

SPEC = ModelSpec(provider="openai", model="x")


def step(node_id, kind="draft", review=None):
    return AgentAnswer(node_id=node_id, rank_id="staff", kind=kind, content="내용", review=review, model=SPEC)


# ─── 진행 상태 반영 ────────────────────────────────────────────────────


def test_apply_event_tracks_status_activity_and_draft():
    run = new_run("r1", "질문", "lead")
    assert run["statuses"]["lead"][0] == "working"

    apply_event(run, "delegated", {"node": "lead", "children": ["a", "b"]})
    assert run["statuses"]["lead"] == ("wait", "⏳ 하위 결과 대기") and run["statuses"]["a"][0] == "wait"

    apply_event(run, "start", {"node": "a", "kind": "draft"})
    apply_event(run, "token", ("a", "안녕"))
    apply_event(run, "token", ("a", "하세요"))
    assert run["activity"]["a"] == "✍️ 작성 중" and run["drafting"] == {"a": "안녕하세요"}

    apply_event(run, "step", step("a"))
    assert run["statuses"]["a"] == ("done", "📝 초안 완료")
    assert "a" not in run["activity"] and "a" not in run["drafting"]  # 완료되면 작성 중 표시 정리

    apply_event(run, "step", step("lead", "integrate", make_review("revise", 8)))
    apply_event(run, "interrupt", None)
    assert run["status"] == "awaiting_human" and run["statuses"]["lead"] == ("wait", "👤 결정 대기")


def test_consume_marks_errors_without_raising():
    run = new_run("r1", "질문", "lead")

    def broken():
        yield "start", {"node": "lead", "kind": "plan"}
        raise NoAvailableModelError("모델 없음")

    consume(run, broken())
    assert run["done"] and run["status"] == "error" and "모델 없음" in run["error"]
    assert run["activity"] == {}  # 끝나면 진행 표시 정리


def test_background_run_completes_and_reports_steps(org, ranks, team, models, use_script):
    use_script(Script(text="초안", ReviewResult=make_review("approve", 9), Delegation=make_delegation("staff_case")))
    graph = build_office_graph(org, ranks, team, models, SETTINGS, NO_CRITIC)
    run = new_run("bg", "질문", "gm_action")
    saved = []
    thread = start_run(run, lambda: stream_office(graph, org, "질문", "gm_action", brief=BRIEF),
                       on_step=lambda r: saved.append(len(r["steps"])), on_finish=lambda r: saved.append("finish"))
    thread.join(timeout=30)

    assert run["done"] and run["status"] == "reviewing" and run["error"] is None
    assert [s.node_id for s in run["steps"]] == ["staff_case", "am_action", "gm_action"]
    assert saved == [1, 2, 3, "finish"]  # 단계마다 저장 콜백 → 끝나면 마무리 콜백


# ─── 에이전트 이미지 ────────────────────────────────────────────────────


@pytest.fixture
def teams_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(avatars, "TEAMS_DIR", tmp_path)
    import core.config as config

    monkeypatch.setattr(config, "TEAMS_DIR", tmp_path)
    return tmp_path


def png_bytes(size=(300, 200), color="red") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_save_avatar_crops_square_and_auto_makes_working(teams_dir):
    base, working = avatars.save_avatar("general", "lead", png_bytes((300, 200)))
    assert Image.open(base).size == (avatars.SIZE, avatars.SIZE)  # 가로로 긴 그림 → 가운데 정사각형
    assert working.name == "lead_working.png" and working.exists()
    # 자동 '작업 중' 이미지에는 말풍선이 그려져 평소 이미지와 다르다
    assert Image.open(base).tobytes() != Image.open(working).tobytes()
    assert avatars.has_custom_avatar("general", "lead")


def test_custom_avatar_takes_priority_and_reset_restores_default(ranks, org, teams_dir):
    # (org 를 먼저 읽은 뒤 teams 폴더를 임시 폴더로 바꾼다)
    lead = subtree(org, "lead")
    team_lead = next(r for r in ranks if r.id == "team_lead")
    default = resolve_avatar("general", team_lead, node=lead)

    avatars.save_avatar("general", "lead", png_bytes(), working_data=png_bytes(color="blue"))
    assert resolve_avatar("general", team_lead, node=lead).name == "lead.png"
    assert resolve_avatar("general", team_lead, working=True, node=lead).name == "lead_working.png"

    avatars.reset_avatar("general", "lead")
    assert resolve_avatar("general", team_lead, node=lead) == default
    assert not avatars.has_custom_avatar("general", "lead")


def test_invalid_image_is_rejected(teams_dir):
    with pytest.raises(avatars.AvatarError, match="읽을 수 없습니다"):
        avatars.save_avatar("general", "lead", b"not an image")
