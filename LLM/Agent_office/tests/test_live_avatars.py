"""실시간 진행 상태(core/live.py) · 백그라운드 실행 · 에이전트 이미지 바꾸기(core/avatars.py) 테스트."""

import base64
import io
import os
import subprocess
import time

import pytest
from PIL import Image

from core import avatars
from core.config import BASE_DIR, resolve_avatar, subtree
from core.live import STAGES, apply_event, consume, new_run, restore_run, stage_of, start_run
from core.storage import Storage
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


def test_stage_of_follows_furthest_stage():
    run = new_run("r1", "질문", "lead")
    assert STAGES[stage_of(run)] == "🎯 요청 정리"
    apply_event(run, "brief", object())
    apply_event(run, "start", {"node": "lead", "kind": "research"})
    assert STAGES[stage_of(run)] == "🌐 자료 수집"
    apply_event(run, "start", {"node": "a", "kind": "draft"})
    apply_event(run, "start", {"node": "b", "kind": "plan"})  # 동시에 여러 단계 → 가장 뒤 단계
    assert STAGES[stage_of(run)] == "✍️ 작성"
    apply_event(run, "step", step("a"))
    apply_event(run, "start", {"node": "lead", "kind": "critic"})
    assert STAGES[stage_of(run)] == "⚖️ 검증"
    apply_event(run, "interrupt", None)
    assert STAGES[stage_of(run)] == "👤 결정"
    run["status"] = "adopted"
    assert stage_of(run) == len(STAGES)


def test_restore_run_rebuilds_waiting_run_from_storage(tmp_path):
    storage = Storage(tmp_path / "runs.db")
    storage.create_run("r9", "stock", "lead", "삼성전자 전망")
    storage.save_steps("r9", [step("a"), step("lead", "integrate", make_review("revise", 8))], "awaiting_human")
    run = restore_run(storage.get_run("r9"))
    assert run["team_id"] == "stock" and run["root_id"] == "lead" and run["done"] and run["restored"]
    assert run["status"] == "awaiting_human" and len(run["steps"]) == 2
    assert run["statuses"]["lead"] == ("wait", "👤 결정 대기") and run["statuses"]["a"][0] == "done"
    assert stage_of(run) == len(STAGES) - 1


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


# ─── 🏢 부서 사무실 캐릭터 얼굴 (올린 전용 이미지를 작게 줄인 것) ───────────────────


def face_of(team_id: str, node_id: str) -> str:
    return avatars.office_faces(team_id, [node_id]).get(node_id, "")


def face_image(url: str) -> Image.Image:
    assert url.startswith("data:image/png;base64,")
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))


def test_office_face_only_for_agent_with_own_image(teams_dir):
    assert face_of("general", "lead") == ""  # 올린 그림이 없으면 얼굴도 없다 (도트 머리 그대로)
    avatars.save_avatar("general", "lead", png_bytes((300, 200)))
    assert face_image(face_of("general", "lead")).size == (avatars.FACE_SIZE, avatars.FACE_SIZE)
    assert face_of("general", "gm_action") == ""  # 다른 에이전트는 그대로
    # 팀 직급 이미지(teams/<팀>/avatars/<직급id>.png)는 얼굴로 쓰지 않는다 — 그 직급 모두가 같은 얼굴이 된다
    (teams_dir / "general" / "avatars" / "staff.png").write_bytes(png_bytes())
    assert face_of("general", "staff_case") == ""


def test_office_face_is_made_once_and_follows_change_and_reset(teams_dir, monkeypatch):
    base, _ = avatars.save_avatar("general", "lead", png_bytes(color="red"))
    old = time.time() - 60
    os.utime(base, (old, old))  # 방금 바뀐 파일은 캐시하지 않으므로(core/memo.py) 예전 파일처럼 만든다
    made, real = [], avatars.make_face
    monkeypatch.setattr(avatars, "make_face", lambda path: made.append(path) or real(path))
    red = face_of("general", "lead")
    assert face_of("general", "lead") == red and len(made) == 1  # 화면을 다시 그려도 한 번만 줄인다

    avatars.save_avatar("general", "lead", png_bytes(color="blue"))  # 그림을 바꾸면 얼굴도 바뀐다
    blue = face_of("general", "lead")
    assert blue and blue != red
    avatars.reset_avatar("general", "lead")  # '기본 이미지로' → 얼굴이 사라진다
    assert face_of("general", "lead") == ""


def test_office_face_stays_small_and_broken_file_gives_no_face(teams_dir):
    noisy = io.BytesIO()  # 색이 제각각인 그림 = PNG 로 가장 안 줄어드는 경우
    Image.frombytes("RGB", (500, 400), os.urandom(500 * 400 * 3)).save(noisy, format="PNG")
    avatars.save_avatar("general", "lead", noisy.getvalue())
    assert len(face_of("general", "lead")) <= 4000  # 글자 수 (48×48·64색이면 3,000 안팎이 끝)

    (teams_dir / "general" / "avatars" / "gm_action.png").write_bytes(b"not an image")  # 손으로 넣은 깨진 파일
    assert face_of("general", "gm_action") == ""


def test_uploaded_avatars_are_git_ignored():
    """올린 사진(증명사진 등)이 공개 저장소에 올라가지 않게 — 모든 팀의 avatars 폴더는 git 이 무시한다."""
    for path in ("teams/general/avatars/lead.png", "teams/career/avatars/lead_working.png", "teams/new_team/avatars/me.jpg"):
        out = subprocess.run(["git", "check-ignore", "-q", path], cwd=BASE_DIR, capture_output=True)
        assert out.returncode == 0, f"{path} 가 git 에 올라갈 수 있습니다"
    out = subprocess.run(["git", "check-ignore", "-q", "assets/avatars/staff.png"], cwd=BASE_DIR, capture_output=True)
    assert out.returncode == 1  # 공용 기본 이미지(글자 원)는 그대로 저장소에 둔다
