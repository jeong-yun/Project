"""음악 팀 테스트 — 팀 설정, 블록 추출, 하네스(music_prompt), Lyria 생성 흐름, 채택 저장(.mp3), 화면 smoke.

Lyria·Gemini 호출은 모두 가짜 함수로 바꾼다 (네트워크·비용 없음). 음원은 사인파로 만든 MP3 를 쓴다.
"""

import base64
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core import audio_edit, music
from core.brief import DesiredState
from core.config import list_teams, load_team, team_org, validate_config
from core.exporter import MIME, export_adopted, list_results
from core.harness import REGISTRY
from core.live import new_run
from core.schemas import HumanDecision
from core.tasks import build_request
from core.workflow import build_office_graph, stream_office
from tests.fakes import Script, make_delegation, make_review

NO_CRITIC = {"critic": {"enabled": False}}
CFG = music.music_settings({"music": {"grade": False}})

SONG_ANSWER = """퇴근길을 위로하는 30초 시티팝.

### 🎤 가사
```lyrics
[Verse]
오늘도 수고했어 나의 하루야
붉은 노을 따라 천천히 걸어가
[Chorus]
괜찮아 괜찮아 내일은 또 오니까
불빛 속에 너를 안아 줄게
```

### 🎼 Lyria 프롬프트
```lyria
A warm Korean city-pop song, 102 BPM, key of D major, soft female vocal, slap bass, electric piano.
[0:00 - 0:08] Intro: electric piano chords and light drums
[0:08 - 0:18] Verse: gentle vocal over groovy bass
[0:18 - 0:30] Chorus: full band, bright synth pad
```
"""

INSTR_ANSWER = """```lyria
Instrumental only, no vocals. A calm lo-fi hip hop track, 80 BPM, key of A minor, dusty piano, vinyl crackle.
[0:00 - 0:20] Intro: piano and crackle
[0:20 - 0:50] Main groove: boom-bap drums enter
[0:50 - 1:00] Outro: drums drop out, piano fades
```"""


def sine_mp3_bytes(tmp_path: Path, seconds: float = 3.0, name: str = "sine.mp3") -> bytes:
    sr = 44100
    t = np.arange(int(sr * seconds)) / sr
    x = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    path = tmp_path / name
    audio_edit.save_audio(path, np.stack([x, x]), sr)
    return path.read_bytes()


def check(answer: str, **options):
    return REGISTRY["music_prompt"].check(answer, options)


# ─── 팀 설정 · 업무 양식 ──────────────────────────────────────────────


def test_music_team_is_valid(ranks, models):
    teams = list_teams()
    validate_config(ranks, models, teams)
    team = load_team("music")
    assert team.name == "🎵 음악 팀" and "music_track" in team.features
    assert [t.title for t in team.tasks] == ["🎤 노래 만들기", "🎼 반주 만들기"]
    assert [n.id for n in team_org(team, ranks).walk()] == ["lead", "gm_review", "lyricist", "composer"]
    assert "music_prompt" in team.harness.final_only


def test_song_task_builds_brief_and_harness_options():
    task = load_team("music").tasks[0]
    req = build_request(task, {"theme": "퇴근길 위로", "language": "한국어", "length": "2분", "genre": "시티팝",
                               "mood": "", "vocal": "여성 보컬", "lyrics": "내가 쓴 가사 한 줄\n두 번째 줄"})
    assert req.harness_options == {"music_kind": "song", "music_length": "2분"}
    assert "분위기" not in req.question  # 빈 입력 줄은 빠진다
    assert "내가 쓴 가사 한 줄\n두 번째 줄" in req.question
    assert "```lyrics" in req.brief.context and "가사 언어는 한국어" in req.brief.context
    instr = build_request(load_team("music").tasks[1], {"theme": "브이로그", "length": "1분"})
    assert instr.harness_options["music_kind"] == "instrumental"


# ─── 블록 추출 · 길이 · 음절 ──────────────────────────────────────────


def test_extract_blocks_and_parse_length():
    blocks = music.extract_blocks(SONG_ANSWER)
    assert blocks.lyrics.startswith("[Verse]") and blocks.prompt.startswith("A warm Korean city-pop")
    assert music.extract_blocks("블록 없음") == music.MusicBlocks("", "")
    assert [music.parse_length(x) for x in ("30초 (클립)", "1분 30초", "2분", "2:30", "90", 45, None, "길게")] == [
        30, 90, 120, 150, 90, 45, None, None]
    assert music.fmt_time(150) == "2:30"


def test_count_syllables_korean_and_english():
    assert music.count_syllables("오늘도 수고했어") == 7
    assert music.count_syllables("hello beautiful love") == 2 + 3 + 1
    assert music.lyric_lines("[Verse]\n가사\n\n[Chorus]\n후렴") == ["가사", "후렴"]


# ─── 하네스 music_prompt ─────────────────────────────────────────────


def test_good_song_and_instrumental_pass():
    assert check(SONG_ANSWER, music_kind="song", music_length="30초 (클립)").passed
    assert check(INSTR_ANSWER, music_kind="instrumental", music_length="1분").passed
    assert check("작사 팁: 후렴을 반복하세요").passed  # 자유 질문(블록 없음)은 검사하지 않는다


def test_missing_blocks_fail_for_music_tasks():
    result = check("가사만 있음", music_kind="song", music_length="1분")
    assert not result.passed and "```lyria" in result.log


def test_instrumental_needs_phrase_and_no_lyrics():
    no_phrase = INSTR_ANSWER.replace("Instrumental only, no vocals. ", "")
    assert "Instrumental only, no vocals." in check(no_phrase, music_kind="instrumental", music_length="1분").log
    with_lyrics = INSTR_ANSWER + "\n```lyrics\n[Verse]\n노래 가사\n```"
    assert "가사 블록이 있다" in check(with_lyrics, music_kind="instrumental", music_length="1분").log
    # 노래 요청에 'Instrumental only' 가 있으면 모순
    contradiction = SONG_ANSWER.replace("A warm Korean", "Instrumental only, no vocals. A warm Korean")
    assert "Instrumental only" in check(contradiction, music_kind="song", music_length="30초").log


def test_song_lyrics_need_tags_and_reasonable_amount():
    untagged = SONG_ANSWER.replace("[Verse]\n", "").replace("[Chorus]\n", "")
    assert "섹션 태그가 없다" in check(untagged, music_kind="song", music_length="30초").log
    long_lyrics = SONG_ANSWER.replace("[Chorus]\n", "[Chorus]\n" + "라라라 라라라 노래해\n" * 20)
    assert "너무 길다" in check(long_lyrics, music_kind="song", music_length="30초").log
    long_line = SONG_ANSWER.replace("불빛 속에 너를 안아 줄게", "가" * 40)
    assert "부르기 어렵다" in check(long_line, music_kind="song", music_length="30초").log
    # 3분 곡에 4줄은 너무 짧다 (시간표도 3분에 못 미침)
    log = check(SONG_ANSWER, music_kind="song", music_length="3분").log
    assert "너무 짧다" in log and "채우지 못한다" in log
    # 1분 이상 노래는 후렴 섹션이 있어야 한다
    no_chorus = SONG_ANSWER.replace("[Chorus]", "[Verse 2]")
    assert "[Chorus]" in check(no_chorus, music_kind="song", music_length="1분").log


def test_artist_names_are_rejected():
    styled = SONG_ANSWER.replace("A warm Korean city-pop song", "A city-pop song in the style of Taylor Swift")
    assert "Taylor Swift" in check(styled, music_kind="song", music_length="30초").log
    korean = SONG_ANSWER.replace("붉은 노을 따라", "아이유의 목소리로")
    assert "아이유" in check(korean, music_kind="song", music_length="30초").log
    assert music.find_artists("bright build-up, Daft Punk-ish") == ["Daft Punk"]
    assert music.find_artists("a quiet piano piece") == []


def test_timestamps_must_fit_length():
    assert "시간표 구간이 없다" in check(SONG_ANSWER.replace("[0:", "(0:"), music_kind="song", music_length="30초").log
    overlap = SONG_ANSWER.replace("[0:18 - 0:30]", "[0:12 - 0:30]")
    assert "겹친다" in check(overlap, music_kind="song", music_length="30초").log
    too_long = SONG_ANSWER.replace("[0:18 - 0:30]", "[0:18 - 0:50]")
    assert "넘는다" in check(too_long, music_kind="song", music_length="30초").log
    gap = SONG_ANSWER.replace("[0:18 - 0:30]", "[0:25 - 0:30]")
    assert "비어 있다" in check(gap, music_kind="song", music_length="30초").log


def test_music_validator_runs_only_on_final_answer(ranks, models, use_script):
    team = load_team("music")
    org = team_org(team, ranks)
    use_script(Script(text=SONG_ANSWER, ReviewResult=make_review("approve", 9, answer=SONG_ANSWER),
                      Delegation=make_delegation("lyricist", "composer")))
    req = build_request(team.tasks[0], {"theme": "퇴근길", "length": "30초 (클립)"})
    graph = build_office_graph(org, ranks, team, models, {}, NO_CRITIC)
    events = list(stream_office(graph, org, req.question, "gm_review", brief=req.brief, auto_brief=False,
                                harness_options=req.harness_options))
    steps = [p for k, p in events if k == "step"]
    leaves = [s for s in steps if s.node_id in ("lyricist", "composer")]
    assert leaves and all(not s.harness for s in leaves)  # 실무자는 일부만 쓰므로 전체 형식 검사 제외
    final = [s for s in steps if s.node_id == "gm_review"][-1]
    assert [(h.validator, h.passed) for h in final.harness] == [("music_prompt", True)]


# ─── Lyria 생성 흐름 (가짜 Lyria · 가짜 채점) ──────────────────────────


@pytest.fixture
def fake_lyria(tmp_path):
    calls = []
    audio = sine_mp3_bytes(tmp_path, 3.0)

    def lyria(model, prompt):
        calls.append((model, prompt))
        return music.LyriaOutput(audio=audio, text="ok")

    lyria.calls = calls
    return lyria


def grades(*scores, vocals=True):
    seq = list(scores)

    def grader(model, audio, prompt):
        grader.prompts.append(prompt)
        score = seq.pop(0) if len(seq) > 1 else seq[0]
        return music.TrackGrade(heard="시티팝, 여성 보컬", vocals_present=vocals, matches_brief=score >= 3,
                                issues=[] if score >= 3 else ["분위기가 다름"], score=score)

    grader.prompts = []
    return grader


def test_make_track_clip_song(tmp_path, fake_lyria):
    track = music.make_track(SONG_ANSWER, tmp_path / "gen1", CFG, kind="song", length="30초 (클립)", lyria=fake_lyria)
    model, sent = fake_lyria.calls[0]
    assert model == "lyria-3-clip-preview" and len(fake_lyria.calls) == 1
    assert "Create a" not in sent  # 클립은 항상 30초 → 길이 지시 없음
    assert sent.startswith("A warm Korean city-pop") and "Lyrics:\n[Verse]\n오늘도 수고했어" in sent
    assert track.best.path.read_bytes()[:3] in (b"ID3", b"\xff\xfb", b"\xff\xf3") or track.best.path.stat().st_size > 1000
    assert abs(track.best.duration - 3.0) < 0.1 and track.cost_usd == 0.04
    assert any("다릅니다" in n for n in track.notes)  # 3초짜리 가짜 음원 → 30초 요청과 다르다는 안내


def test_make_track_song_model_adds_length(tmp_path, fake_lyria):
    track = music.make_track(INSTR_ANSWER.replace("Instrumental only, no vocals. ", ""), tmp_path, CFG,
                             kind="instrumental", length="2분", lyria=fake_lyria)
    model, sent = fake_lyria.calls[0]
    assert model == "lyria-3.5" and track.cost_usd == 0.08
    assert sent.startswith("Create a 2:00 long instrumental track.")
    assert "Instrumental only, no vocals." in sent and "Lyrics:" not in sent  # 반주 문구는 코드가 보충


def test_poor_grade_regenerates_once_and_keeps_best(tmp_path, fake_lyria):
    cfg = music.music_settings({"music": {"grade": True, "max_regenerations": 1}})
    grader = grades(2, 4)
    track = music.make_track(SONG_ANSWER, tmp_path, cfg, kind="song", length="2분", lyria=fake_lyria, grader=grader)
    assert len(fake_lyria.calls) == 2 and [t.grade.score for t in track.takes] == [2, 4]
    assert track.chosen == 1 and track.cost_usd == 0.16
    assert any("다시 생성" in n for n in track.notes)
    assert "보컬이 있는 노래" in grader.prompts[0] and "퇴근길" in grader.prompts[0]


def test_regeneration_cap_is_configurable(tmp_path, fake_lyria):
    always_bad = grades(1)
    cfg = music.music_settings({"music": {"grade": True, "max_regenerations": 1}})
    track = music.make_track(SONG_ANSWER, tmp_path / "a", cfg, kind="song", length="2분", lyria=fake_lyria, grader=always_bad)
    assert len(track.takes) == 2 and track.chosen == 0  # 점수가 같으면 먼저 만든 것
    fake_lyria.calls.clear()
    cfg0 = music.music_settings({"music": {"grade": True, "max_regenerations": 0}})
    music.make_track(SONG_ANSWER, tmp_path / "b", cfg0, kind="song", length="2분", lyria=fake_lyria, grader=always_bad)
    assert len(fake_lyria.calls) == 1


def test_instrumental_with_vocals_is_graded_down(tmp_path, fake_lyria):
    cfg = music.music_settings({"music": {"grade": True, "max_regenerations": 0}})
    track = music.make_track(INSTR_ANSWER, tmp_path, cfg, kind="instrumental", length="1분", lyria=fake_lyria,
                             grader=grades(5, vocals=True))
    assert track.best.grade.score == 2 and "보컬이 들림" in track.best.grade.issues[-1]


def test_grader_failure_keeps_track(tmp_path, fake_lyria):
    def broken(model, audio, prompt):
        raise RuntimeError("quota")

    cfg = music.music_settings({"music": {"grade": True}})
    track = music.make_track(SONG_ANSWER, tmp_path, cfg, kind="song", length="30초", lyria=fake_lyria, grader=broken)
    assert len(track.takes) == 1 and track.best.grade is None and "채점 실패" in track.notes[0]


def test_make_track_without_prompt_raises(tmp_path, fake_lyria):
    with pytest.raises(music.MusicError):
        music.make_track("가사만 있고 프롬프트 없음", tmp_path, CFG, lyria=fake_lyria)
    assert not fake_lyria.calls


def test_call_lyria_decodes_base64_from_interactions(monkeypatch, tmp_path):
    raw = sine_mp3_bytes(tmp_path, 1.0)
    seen = {}

    def create(model, input):
        seen.update(model=model, input=input)
        return SimpleNamespace(output_audio=SimpleNamespace(data=base64.b64encode(raw).decode()), output_text="[Verse] ...")

    monkeypatch.setattr(music, "_genai_client", lambda timeout_sec: SimpleNamespace(interactions=SimpleNamespace(create=create)))
    out = music.call_lyria("lyria-3.5", "prompt")
    assert out.audio == raw and out.text == "[Verse] ..." and seen == {"model": "lyria-3.5", "input": "prompt"}

    blocked = SimpleNamespace(interactions=SimpleNamespace(create=lambda model, input: SimpleNamespace(output_audio=None, output_text="blocked")))
    monkeypatch.setattr(music, "_genai_client", lambda timeout_sec: blocked)
    with pytest.raises(music.MusicError, match="blocked"):
        music.call_lyria("lyria-3.5", "in the style of someone")


def test_call_gemini_audio_sends_audio_part(monkeypatch):
    grade = music.TrackGrade(heard="피아노", vocals_present=False, matches_brief=True, issues=[], score=4)
    seen = {}

    def generate_content(model, contents, config):
        seen.update(model=model, contents=contents, config=config)
        return SimpleNamespace(parsed=grade, text="")

    monkeypatch.setattr(music, "_genai_client", lambda timeout_sec: SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
    assert music.call_gemini_audio("gemini-x", b"mp3", "채점해") is grade
    part = seen["contents"][0]
    assert part.inline_data.mime_type == "audio/mp3" and part.inline_data.data == b"mp3" and seen["contents"][1] == "채점해"
    assert seen["config"].response_schema is music.TrackGrade


def test_missing_key_is_clear_error(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(music.MusicError, match="GOOGLE_API_KEY"):
        music._genai_client()


def test_recover_track_reloads_existing_takes(tmp_path):
    folder = tmp_path / "gen1"
    folder.mkdir()
    (folder / "take1.mp3").write_bytes(sine_mp3_bytes(tmp_path, 2.0))
    (folder / "take2.mp3").write_bytes(sine_mp3_bytes(tmp_path, 1.0))
    track = music.recover_track(folder, SONG_ANSWER, CFG, "song", "30초")
    assert len(track.takes) == 2 and track.best.path.name == "take2.mp3" and track.model == "lyria-3-clip-preview"
    assert music.recover_track(tmp_path / "none", SONG_ANSWER, CFG) is None


# ─── 채택 저장 (.md 옆에 .mp3) ────────────────────────────────────────


def test_export_adopted_saves_mp3_next_to_markdown(tmp_path):
    src = tmp_path / "v2.mp3"
    src.write_bytes(sine_mp3_bytes(tmp_path, 1.0))
    run = new_run("r1", "퇴근길 노래 만들어줘", "lead")
    run["brief"] = DesiredState(goal="30초 노래", context="", success_criteria=[], constraints=[], output_format="", assumptions=[])
    decision = HumanDecision(action="adopt", adopted_node_id="lead", final_answer=SONG_ANSWER, edited=False, score=5, comment="")
    paths = export_adopted(run, decision, "music", "🎵 음악 팀", {"lead": "팀장·음악PD"}, tmp_path / "results",
                           appendix="- 모델: lyria-3.5", audio_path=src, appendix_title="음원 (Gemini Lyria)")
    md, mp3 = paths
    assert mp3.suffix == ".mp3" and mp3.stem == md.stem and mp3.read_bytes() == src.read_bytes()
    assert "## 음원 (Gemini Lyria)" in md.read_text(encoding="utf-8")
    assert {p.suffix for p in list_results(tmp_path / "results", "music")} == {".md", ".mp3"}
    assert MIME[".mp3"] == "audio/mpeg"


# ─── 화면 smoke (Streamlit AppTest) ───────────────────────────────────


def test_app_renders_music_team(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=60)
    at.session_state["team_pick"] = "music"
    at.run()
    assert not at.exception
    assert at.query_params["team"] == ["music"] or at.query_params["team"] == "music"
    pick = next(s for s in at.selectbox if s.label == "📋 업무 선택")
    assert pick.value == "song"
    labels = {w.label for w in at.text_input} | {w.label for w in at.text_area} | {w.label for w in at.selectbox}
    assert {"주제·용도 *", "가사 언어", "길이", "다듬을 가사 (선택)"} <= labels
    pick.select("instrumental").run()
    assert not at.exception
    assert "원하는 악기 (선택)" in {w.label for w in at.text_input}


def test_app_music_section_recovers_track_and_edits(tmp_path, monkeypatch):
    """서버를 다시 켜 되살린 '결정 대기' 음악 업무 → 만든 곡을 다시 생성 없이 불러와 듣기·편집·되돌리기."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    from core.schemas import AgentAnswer, ModelSpec
    from core.storage import Storage

    data = tmp_path / "data"
    storage = Storage(data / "runs.db")
    storage.create_run("m1", "music", "lead", "퇴근길 노래 만들어줘")
    final = AgentAnswer(node_id="lead", rank_id="team_lead", kind="integrate", content=SONG_ANSWER,
                        model=ModelSpec(provider="openai", model="fake"))
    storage.save_steps("m1", [final], "awaiting_human")
    take = data / "music" / "m1" / "gen1" / "take1.mp3"
    take.parent.mkdir(parents=True)
    take.write_bytes(sine_mp3_bytes(tmp_path, 4.0))

    trim = audio_edit.EditPlan(analysis="", needs_regeneration=False, regeneration_reason="", summary="2초로 자르기", steps=[
        audio_edit.EditStep(op="trim", start_sec=0, end_sec=2, seconds=0, amount=0, bass_db=0, treble_db=0, position="none")])
    monkeypatch.setattr(audio_edit, "plan_edit", lambda request, duration, models: (trim, []))
    monkeypatch.setattr(music, "call_lyria", lambda *a: pytest.fail("되살린 업무에서 Lyria 를 다시 부르면 안 된다 (유료)"))
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(data))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))
    st.cache_resource.clear()  # 다른 테스트가 만든 저장소·진행 중 업무 캐시를 비운다
    try:
        at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=90)
        at.session_state["team_pick"] = "music"
        at.run()
        assert not at.exception
        assert any("다시 불러왔습니다" in c.value for c in at.caption)
        assert next(b for b in at.button if b.label == "↩ 되돌리기").disabled

        next(t for t in at.text_input if t.label == "편집 요청").input("2초로 잘라줘")
        next(b for b in at.button if b.label == "✂️ 편집 적용").click().run()
        assert not at.exception
        assert any("v2 만듦" in s.value and "길이" in s.value for s in at.success)
        assert (take.parent / "v2.mp3").exists()
        assert any("**v2**" in c.value for c in at.caption)  # 버전 줄: v1 → **v2**

        next(b for b in at.button if b.label == "↩ 되돌리기").click().run()
        assert any("**v1** → v2" in c.value for c in at.caption)
    finally:
        st.cache_resource.clear()


# ─── 실제 SDK 호출 경로 (가짜 google-genai Client) ─────────────────────────


class FakeGenaiClient:
    """google-genai Client 흉내 — 실제 SDK 처럼 Client 가 사라지면(__del__) 연결을 닫는다."""

    def __init__(self, audio: bytes, log: list):
        self.closed, self.log = False, log
        client = self

        class Interactions:
            def create(self, model, input):
                if client.closed:
                    raise RuntimeError("Cannot send a request, as the client has been closed.")
                log.append(("lyria", model))
                return type("I", (), {"output_audio": type("A", (), {"data": base64.b64encode(audio).decode()})(),
                                      "output_text": ""})()

        class Models:
            def generate_content(self, model, contents, config):
                if client.closed:
                    raise RuntimeError("Cannot send a request, as the client has been closed.")
                log.append(("grade", model))
                grade = music.TrackGrade(heard="국악 타악 반주", vocals_present=False, matches_brief=True, issues=[], score=4)
                return type("R", (), {"parsed": grade, "text": grade.model_dump_json()})()

        self._interactions, self._models = Interactions(), Models()

    @property
    def interactions(self):
        # 실제 SDK 처럼 하위 객체는 Client 를 붙잡지 않는다 → Client 를 변수에 잡아 두지 않으면 곧바로 닫힘
        return self._interactions

    @property
    def models(self):
        return self._models

    def __del__(self):
        self.closed = True


def test_lyria_and_grade_keep_client_alive(tmp_path, monkeypatch):
    """회귀: `_genai_client().interactions.create(...)` 처럼 임시 Client 를 쓰면 요청 전에 닫혀 'client has been closed'."""
    log: list = []
    audio = sine_mp3_bytes(tmp_path, 1.0)
    monkeypatch.setattr(music, "_genai_client", lambda timeout_sec: FakeGenaiClient(audio, log))
    out = music.call_lyria("lyria-3.5", "Instrumental only, no vocals.")
    assert out.audio == audio
    grade = music.call_gemini_audio("gemini-3.5-flash", audio, "채점")
    assert grade.score == 4 and log == [("lyria", "lyria-3.5"), ("grade", "gemini-3.5-flash")]


def test_app_music_button_generates_in_background(tmp_path, monkeypatch):
    """'🎵 곡 만들기' → 백그라운드에서 생성(화면은 멈추지 않음) → 끝나면 플레이어. 실제 SDK 경로를 가짜 Client 로."""
    import time

    import streamlit as st
    from streamlit.testing.v1 import AppTest

    from core.schemas import AgentAnswer, ModelSpec
    from core.storage import Storage

    data = tmp_path / "data"
    storage = Storage(data / "runs.db")
    storage.create_run("m2", "music", "lead", "러닝용 반주")
    final = AgentAnswer(node_id="lead", rank_id="team_lead", kind="integrate", content=SONG_ANSWER,
                        model=ModelSpec(provider="openai", model="fake"))
    storage.save_steps("m2", [final], "awaiting_human")
    log: list = []
    monkeypatch.setattr(music, "_genai_client", lambda timeout_sec: FakeGenaiClient(sine_mp3_bytes(tmp_path, 2.0), log))
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(data))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))
    st.cache_resource.clear()
    try:
        at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=90)
        at.session_state["team_pick"] = "music"
        at.run()
        button = next(b for b in at.button if b.label.startswith("🎵 곡 만들기"))
        button.click().run()
        assert not at.exception
        deadline = time.time() + 60
        while not list((data / "music" / "m2").glob("gen1/take1.mp3")) and time.time() < deadline:
            time.sleep(0.2)
        time.sleep(0.5)  # 스레드가 post_busy 를 내릴 시간
        at.run()
        assert not at.exception
        assert ("lyria", "lyria-3.5") in log
        assert not [w for w in at.warning if "만들지 못했습니다" in w.value]
        assert any("생성 1회" in c.value for c in at.caption)  # 플레이어 아래 줄: 모델·예상 비용·생성 횟수
    finally:
        st.cache_resource.clear()
