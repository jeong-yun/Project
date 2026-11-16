"""🎤 노래 음색 변환 팀 (teams/voice) · 🗣 일상 말 음성 변환 팀 (teams/voice_speech) — core/voice.py · core/tools/voice_svc.py ·
검증기 voice_* · 화면(AppTest) 포함. 2026-11-16 에 예전 '🎙 음성 합성 팀'을 둘로 나눴다 (노래 팀이 옛 id voice 를 그대로 씀).

실제 모델·Colab·외부 다운로드 호출 없음. 소리는 numpy 로 만든 사인파·무음·꽉 찬 사각파만 쓴다 (사람 목소리 없음).
"""

import importlib.util
import inspect
import io
import json
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from core import building, scope, voice
from core.config import BASE_DIR, list_teams, load_models, load_ranks, load_team, team_org, validate_config
from core.harness import REGISTRY
from core.tasks import TaskInputError, build_request
from core.tools import TOOLS
from tests.fakes import Script, make_delegation, make_review

APP = str(BASE_DIR / "app.py")
ROOT = str(BASE_DIR)
SR = 16000
PLAN_OK = {"source": voice.SOURCES[0], "consent": True, "singer": "myvoice", "songs": 8, "minutes": 40,
           "gpu": "없음", "where": "내 PC (A안)", "goal": "내 목소리로 좋아하는 발라드 커버 만들기"}


def wav_bytes(samples: np.ndarray, sr: int = SR) -> bytes:
    """[-1, 1] 소리 → 16비트 PCM wav (preprocess.py 가 쓰는 형식)."""
    data = np.clip(samples, -1, 1)
    if data.ndim == 1:
        data = data[:, None]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(data.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((data * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def sine(sec: float, amp: float = 0.3, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sec * sr)) / sr
    return amp * np.sin(2 * np.pi * 220 * t)


def make_zip(path: Path, files: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return path


@pytest.fixture
def dataset(tmp_path):
    square = np.where(sine(3.0, amp=1.0) >= 0, 1.0, -1.0)  # 최댓값에 닿는 소리 → 클리핑 의심
    return make_zip(tmp_path / "dataset_raw.zip", {
        "dataset_raw/myvoice/a.wav": wav_bytes(sine(5.0)),
        "dataset_raw/myvoice/b.wav": wav_bytes(sine(10.0)),
        "dataset_raw/myvoice/c.wav": wav_bytes(sine(1.0)),  # 2초 미만
        "dataset_raw/myvoice/d.wav": wav_bytes(sine(20.0)),  # 15초 초과
        "dataset_raw/myvoice/e.wav": wav_bytes(np.zeros(4 * SR)),  # 무음
        "dataset_raw/myvoice/f.wav": wav_bytes(square),
        "dataset_raw/readme.txt": b"memo",
    })


# ─── 팀 · 부서 ──────────────────────────────────────────────────────


def test_team_loads_in_creative_department_third_room():
    teams = list_teams()
    validate_config(load_ranks(), load_models(), teams)
    team = load_team("voice")
    assert team.name == "🎤 노래 음색 변환 팀" and "노래 음색 변환" in team.description
    assert [t.id for t in team.tasks] == ["plan", "check"] and "voice_svc" in team.tools
    assert {"voice_guard"} <= set(team.harness.validators) and {"voice_plan", "voice_data"} <= set(team.harness.final_only)
    names = [n.name for n in team_org(team, load_ranks()).walk()]
    assert names[0].startswith("팀장") and any("권리" in n for n in names) and len(names) == 4
    cfg = building.load_config()
    creative = next(d for d in building.departments([t.id for t in teams], cfg) if d.id == "creative")
    assert creative.teams == ["music", "design", "voice", "voice_speech"]  # 한 층 네 칸 — 노래 팀은 세 번째, 말 팀은 네 번째 방
    floor = next(f for f in building.plan_floors(building.departments([t.id for t in teams], cfg), building.per_floor(cfg))
                 if ("creative", "voice") in f)
    assert [t for _, t in floor] == ["music", "design", "voice", "voice_speech"]
    head = next(d for d in cfg["departments"] if d["id"] == "creative")["head"]["persona"]
    assert "노래 음색 변환 팀" in head and "일상 말 음성 변환 팀" in head and "동의" in head
    assert TOOLS["voice_svc"] and {"voice_guard", "voice_plan", "voice_data"} <= REGISTRY.keys()


# ─── 책임 있는 사용 — 양식 ─────────────────────────────────────────────


def test_plan_form_needs_consent_source_and_safe_singer():
    voice.check_form("plan", PLAN_OK)  # 통과
    with pytest.raises(TaskInputError, match="목소리 사용 확인"):
        voice.check_form("plan", {**PLAN_OK, "consent": False})
    with pytest.raises(TaskInputError, match="목소리 출처"):
        voice.check_form("plan", {**PLAN_OK, "source": "유명 가수 음원"})
    with pytest.raises(TaskInputError, match="영문·숫자"):
        voice.check_form("plan", {**PLAN_OK, "singer": "../나"})
    with pytest.raises(TaskInputError, match="도움을 드리지 않습니다"):
        voice.check_form("plan", {**PLAN_OK, "singer": "IU"})
    with pytest.raises(TaskInputError, match="속이거나"):
        voice.check_form("plan", {**PLAN_OK, "goal": "엄마 목소리로 전화해서 속이려고"})
    with pytest.raises(TaskInputError, match="동의를 받지 않은"):
        voice.check_form("check", {}, extra="친구 목소리를 동의 없이 몰래 학습")
    # 양식 자체도 확인 체크를 필수로 받는다 (기존 방식)
    task = next(t for t in load_team("voice").tasks if t.id == "plan")
    with pytest.raises(TaskInputError, match="필수 입력"):
        build_request(task, {**PLAN_OK, "consent": False})


def test_misuse_words_do_not_catch_ordinary_requests():
    for ok in ("내 목소리로 아이유 노래 커버", "속이 시원한 고음으로 바꾸고 싶어", "동의를 받은 친구 목소리", "변환한 목소리가 쇳소리가 나요"):
        assert voice.misuse(ok) == "", ok
    assert voice.misuse("아이유 목소리로 바꿔줘") and voice.misuse("BTS 의 음색을 복제")
    assert voice.misuse("음성 인증 우회용") and voice.misuse("보이스피싱")


# ─── 맞춤 파일 ──────────────────────────────────────────────────────


def test_settings_come_from_rules_not_the_model():
    assert voice.plan_settings(45)["epochs"] == 400 and voice.plan_settings(45)["batch_size"] == 16
    assert voice.plan_settings(45, oom=True)["batch_size"] == 8
    assert "부족" in voice.plan_settings(5)["notes"][0] and voice.plan_settings(5)["epochs"] == 400  # 근거 없는 구간은 기본값
    assert "모름" in voice.plan_settings(None)["notes"][0]
    assert [voice.verdict(m) for m in (9.9, 10, 29.9, 30, 90)] == ["부족", "주의", "주의", "충분", "충분"]
    assert voice.kit_from_inputs({**PLAN_OK, "consent": False}) is None  # 툴도 확인 없이는 만들지 않는다
    kit = voice.kit_from_inputs({**PLAN_OK, "where": "Colab (B안)", "gpu": "있음", "oom": True})
    assert (kit["mode"], kit["fast"], kit["batch_size"], kit["singer"]) == ("B", False, 8, "myvoice")
    assert voice.kit_from_inputs(PLAN_OK)["fast"] is True  # GPU 없는 PC 의 A안 → 빠른 모델


def test_notebook_keeps_cells_and_changes_only_the_values():
    original = json.loads((voice.ASSET_DIR / voice.NOTEBOOK).read_text(encoding="utf-8"))
    kit = voice.kit_from_inputs({**PLAN_OK, "singer": "jiyun_01", "oom": True})
    nb = json.loads(voice.build_notebook(kit))
    assert len(nb["cells"]) == len(original["cells"]) and nb["metadata"] == original["metadata"]
    assert [c["cell_type"] for c in nb["cells"]] == [c["cell_type"] for c in original["cells"]]
    changed = [(i, a, b) for i, (c0, c1) in enumerate(zip(original["cells"], nb["cells"]))
               for a, b in zip(c0["source"], c1["source"]) if a != b]
    # 화자 이름 줄 · batch_size 줄 · 변환 셀의 설명·명령 줄만 (epochs 는 규칙상 기본값 400 그대로라 원본과 같다)
    assert [(i, a[:12]) for i, a, _ in changed] == [(8, "SINGER = 'si"), (11, "cfg['train']"), (15, "# -t: 키 조절(반"),
                                                    (15, '!svc infer "')]
    code = "".join(nb["cells"][8]["source"] + nb["cells"][11]["source"])
    assert "SINGER = 'jiyun_01'" in code and "['epochs'] = 400 " in code and "['batch_size'] = 8 " in code
    infer = changed[3][2]  # 노래 변환: 음높이 자동 예측을 끈다 (원본 셀은 옵션이 없어 fork 기본값 '켜짐'으로 돌았다)
    assert infer == changed[3][1].replace(" -fm crepe -t 0 ", " -fm crepe -na -t 0 ")
    assert "-na: 음높이 자동 예측 끔" in changed[2][2] and "12 = 한 옥타브" in changed[2][2]
    assert "AI 로 합성한 음성" in "".join(nb["cells"][0]["source"])  # 첫 설명 셀에 표시 안내 (뒤에 덧붙임)
    for c in nb["cells"]:  # 코드 셀이 파이썬으로 읽힌다 (! · % 줄은 빼고)
        if c["cell_type"] == "code":
            compile("".join(x for x in c["source"] if not x.lstrip().startswith(("!", "%"))), "cell", "exec")


def test_kit_zip_has_custom_notebook_and_original_scripts():
    kit = voice.kit_from_inputs(PLAN_OK)
    with zipfile.ZipFile(io.BytesIO(voice.kit_zip(kit))) as zf:
        assert set(zf.namelist()) == {voice.NOTEBOOK, "README.md", "preprocess.py", "run.bat"}
        assert zf.read("preprocess.py") == (voice.ASSET_DIR / "preprocess.py").read_bytes()
        readme = zf.read("README.md").decode("utf-8")
    assert "--singer myvoice --fast" in readme and "AI 로 합성한 음성" in readme and "epochs 400" in readme
    assert "재 보지 않음" in readme  # 학습 시간은 지어내지 않는다
    evidence = voice.plan_evidence(kit)
    assert "| epochs | 400 |" in evidence and "dataset_raw/myvoice/" in evidence


def test_slice_rules_are_read_from_the_original_script():
    spec = importlib.util.spec_from_file_location("svc_preprocess", voice.ASSET_DIR / "preprocess.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # 맨 위는 import 만 — 설치·실행은 main() 안에서만 한다
    params = inspect.signature(module.slice_wav).parameters
    assert voice.slice_defaults() == {k: params[k].default for k in ("threshold_db", "min_keep", "max_len")}


# ─── 데이터 점검 ─────────────────────────────────────────────────────


def test_dataset_numbers_are_computed(dataset):
    stats = voice.inspect_zip(dataset)
    assert stats["clips"] == 6 and stats["minutes"] == pytest.approx(0.7) and stats["verdict"] == "부족"
    row = stats["singers"][0]
    assert row["name"] == "myvoice" and row["clips"] == 6
    assert (row["min_sec"], row["median_sec"], row["max_sec"]) == (1.0, 4.5, 20.0)
    assert (row["short"], row["long"], row["silent"], row["clipped"]) == (1, 1, 1, 1)
    assert row["sample_rates"] == [SR] and row["channels"] == [1]
    assert stats["issues"] == [] and stats["ignored"] == 1 and stats["skipped"] == 0
    table = voice.dataset_table(stats)
    assert "| myvoice | 6개 | 0.7분 | 1.0 · 4.5 · 20.0초 | 1개 | 1개 | 1개 | 1개 | 16000Hz | 1 | 부족 |" in table
    assert "폴더 구조: 맞음" in table


def test_dataset_structure_problems_are_reported(tmp_path):
    flat = make_zip(tmp_path / "flat.zip", {"myvoice/a.wav": wav_bytes(sine(3)), "dataset_raw/b.wav": wav_bytes(sine(3)),
                                            "dataset_raw/x/y.wav": wav_bytes(sine(3, sr=44100), sr=44100),
                                            "dataset_raw/x/broken.wav": b"RIFF0000"})
    stats = voice.inspect_zip(flat)
    issues = " ".join(stats["issues"])
    assert "`dataset_raw` 가 아니다" in issues and "화자 폴더 없이" in issues and "샘플레이트가 섞여" in issues
    assert stats["skipped"] == 1 and stats["clips"] == 3


@pytest.mark.parametrize("name", ["../evil.wav", "dataset_raw/../../evil.wav", "/abs.wav", "C:/win.wav"])
def test_zip_paths_outside_the_folder_are_refused(tmp_path, name):
    bad = make_zip(tmp_path / "bad.zip", {"dataset_raw/me/ok.wav": wav_bytes(sine(3)), name: wav_bytes(sine(3))})
    with pytest.raises(voice.VoiceError, match="폴더 밖"):
        voice.inspect_zip(bad)
    with pytest.raises(TaskInputError, match="폴더 밖"):  # 맡기기 전에 막는다
        voice.check_form("check", {"dataset": str(bad)})


def test_zip_size_limits_and_broken_zip(tmp_path, dataset, monkeypatch):
    monkeypatch.setattr(voice, "MAX_TOTAL_BYTES", 100_000)  # 풀린 크기 합 한도
    with pytest.raises(voice.VoiceError, match="압축을 풀면 너무 큽니다"):
        voice.inspect_zip(dataset)
    monkeypatch.setattr(voice, "MAX_ZIP_BYTES", 100)
    with pytest.raises(voice.VoiceError, match="너무 큽니다"):
        voice.inspect_zip(dataset)
    (tmp_path / "x.zip").write_bytes(b"not a zip")
    monkeypatch.setattr(voice, "MAX_ZIP_BYTES", 10**9)
    with pytest.raises(voice.VoiceError, match="열 수 없습니다"):
        voice.inspect_zip(tmp_path / "x.zip")


# ─── 툴 ─────────────────────────────────────────────────────────────


def test_tool_gives_settings_or_numbers_only(dataset):
    run = TOOLS["voice_svc"]
    plan = run("q", {**PLAN_OK, "voice_task": "plan"}, None)
    assert plan.harness_options["voice_kit"]["epochs"] == 400 and "코드가 정한 학습 설정" in plan.content
    no = run("q", {**PLAN_OK, "consent": False, "voice_task": "plan"}, None)
    assert no.harness_options["voice_blocked"] and "voice_kit" not in no.harness_options
    data = run("q", {"dataset": str(dataset), "dataset_name": "dataset_raw.zip", "voice_task": "check"}, None)
    assert data.harness_options["voice_dataset"]["clips"] == 6 and "모델에 보내지 않았다" in data.content
    assert "RIFF" not in data.content  # 음원 내용은 근거 자료에 들어가지 않는다
    blocked = run("아이유 목소리로 노래 바꾸는 법", {}, None)
    assert blocked.harness_options["voice_blocked"] and "범위 밖" in blocked.content
    assert run("Colab 세션이 끊겼어요", {}, None) is None  # 자유 질문은 스킬 글로 답한다


# ─── 하네스 ─────────────────────────────────────────────────────────

GOOD_PLAN = """목소리 출처: 내 목소리 — 학습·변환에 써도 된다고 확인함.

## 실행 순서
1. 전처리: `python preprocess.py --input songs --singer myvoice --fast` → `dataset_raw/myvoice/`
2. `dataset_raw.zip` 을 Drive `MyDrive/svc_project/` 에 업로드
3. 학습: epochs 400 · batch_size 16 으로 `svc train`
4. 변환: `svc infer cover.wav -fm crepe -na -t 0` 로 음색 변환 (음높이 자동 예측 끔), `-t` 로 키 조절 → 반주와 합치기

결과물을 남에게 보이거나 올릴 때는 'AI 로 합성한 음성'이라고 표시하세요.
"""
SONG_INFER_LINE = "4. 변환: `svc infer cover.wav -fm crepe -na -t 0` 로 음색 변환 (음높이 자동 예측 끔), `-t` 로 키 조절 → 반주와 합치기"


def check(name, answer, **options):
    return REGISTRY[name].check(answer, options)


def test_voice_plan_validator():
    kit = voice.kit_from_inputs(PLAN_OK)
    assert check("voice_plan", GOOD_PLAN, voice_kit=kit).passed
    cases = {
        "출처": GOOD_PLAN.replace("목소리 출처: 내 목소리 — 학습·변환에 써도 된다고 확인함.", ""),
        "표시": GOOD_PLAN.replace("결과물을 남에게 보이거나 올릴 때는 'AI 로 합성한 음성'이라고 표시하세요.", ""),
        "단계": GOOD_PLAN.replace(SONG_INFER_LINE, "4. 반주와 합치기 (`-na`)"),
        "epochs": GOOD_PLAN.replace("epochs 400", "epochs 800"),
        "batch_size": GOOD_PLAN + "\n```python\ncfg['train']['batch_size'] = 32\n```",
        "화자": GOOD_PLAN.replace("--singer myvoice", "--singer other"),
    }
    for want, answer in cases.items():
        result = check("voice_plan", answer, voice_kit=kit)
        assert not result.passed and want in result.log, (want, result.log)
    assert check("voice_plan", "아무 글", voice_blocked="x").passed  # 범위 밖이면 voice_guard 가 본다


def test_song_plan_turns_auto_f0_off():
    """노래: 음높이 자동 예측 끔(-na) — 원조 so-vits-svc README '노래에는 켜지 마라', fork 의 기본값은 켜짐."""
    kit = voice.kit_from_inputs(PLAN_OK)
    no_na = check("voice_plan", GOOD_PLAN.replace(" -na", "").replace(" (음높이 자동 예측 끔)", ""), voice_kit=kit)
    assert not no_na.passed and "-na" in no_na.log
    for on in (GOOD_PLAN + "\n쇳소리가 나면 `svc infer cover.wav -a -t 0` 로 다시 해 보세요.",
               GOOD_PLAN + "\n`--auto-predict-f0` 를 붙여 보세요.", GOOD_PLAN + "\n음높이 자동 예측을 켜세요."):
        result = check("voice_plan", on, voice_kit=kit)
        assert not result.passed and "켜라고" in result.log, on
    for fine in (GOOD_PLAN + "\n음높이 자동 예측(`-a`)은 말 변환용이라 켜지 않습니다.",  # 켜지 말라는 설명은 괜찮다
                 GOOD_PLAN.replace("-na", "--no-auto-predict-f0")):
        assert check("voice_plan", fine, voice_kit=kit).passed, fine
    # 맞춤 안내·근거 자료에 끄는 이유와 키 바꾸는 법 (지어낸 값 없이: 기본 0, 12 = 한 옥타브)
    readme = voice.kit_readme(kit)
    evidence = voice.plan_evidence(kit)
    for text in (readme, evidence):
        assert "-na" in text and "`-t 0` 을 반음 수로" in text and "12 = 한 옥타브" in text
        assert all(s in text for s in voice.SONG_SOURCES)
    assert f"`svc infer {voice.INFER_SONG}`" in evidence


def test_voice_guard_validator():
    assert check("voice_guard", GOOD_PLAN).passed
    assert not check("voice_guard", "아이유의 목소리로 학습하려면 음원을 모으세요").passed
    assert check("voice_guard", "아이유 노래를 내 목소리로 부른 커버").passed  # 곡은 괜찮다 — 목소리가 대상일 때만
    refusal = "이 팀은 남을 속이는 용도에는 도움을 드리지 않습니다. 내 목소리로 다시 맡겨 주세요."
    assert check("voice_guard", refusal, voice_blocked="속이는 용도").passed
    leaked = check("voice_guard", refusal + "\n`svc train` 을 실행하세요", voice_blocked="속이는 용도")
    assert not leaked.passed and "실행 방법" in leaked.log


def test_voice_data_validator_catches_changed_numbers(dataset):
    stats = voice.inspect_zip(dataset)
    good = "점검 결과 총 6개 클립, 0.7분입니다. 분량 판정은 부족 — 음원을 더 모으세요."
    assert check("voice_data", good, voice_dataset=stats).passed
    for bad in (good.replace("6개", "60개"), good.replace("0.7분", "35.0분"), good.replace("부족", "넉넉")):
        assert not check("voice_data", bad, voice_dataset=stats).passed, bad
    assert not check("voice_data", good + " 전체 30분 분량이면 충분합니다.", voice_dataset=stats).passed
    assert not check("voice_data", "총 120개 클립으로 보입니다", voice_dataset=None).passed  # 점검 못 했는데 숫자를 지어냄
    assert check("voice_data", "zip 을 열지 못했습니다. 다시 묶어 올려 주세요.", voice_dataset=None).passed


# ─── 범위 확인 · 부서장 배정 ─────────────────────────────────────────


def no_router(*args, **kwargs):
    raise AssertionError("낱말 확인으로 끝나야 합니다")


def test_scope_keeps_music_and_voice_apart():
    teams = list_teams()
    by = {t.id: t for t in teams}
    # 말 팀과 같이 쓰는 낱말(내 목소리·Colab·쇳소리 등)만 있는 질문은 이제 두 팀이 비겨 부서장 배정으로 간다 → 노래 낱말을 넣어 묻는다
    for q in ("커버 보컬을 변환했더니 쇳소리가 나요", "노래 커버용 내 목소리 모델 학습 데이터는 몇 분이나 필요해?",
              "노래 음색 변환 Colab 학습 세션이 끊겼어요"):
        d = scope.check(q, by["voice"], teams, None, route=no_router)
        assert d["in_scope"] and d["stage"] == "keyword", q
        assert not scope.heuristic(q, by["music"], teams), q  # 음악 팀은 낱말로 바로 받지 않는다
        assert not scope.heuristic(q, by["voice_speech"], teams), q  # 말 팀도
    for q in ("친구 생일에 들려줄 신나는 축하 노래 가사 써줘", "보컬 있는 발라드 노래 만들어줘"):
        assert scope.check(q, by["music"], teams, None, route=no_router)["in_scope"], q
        assert not scope.heuristic(q, by["voice"], teams), q


def test_creative_head_routes_four_teams(set_keys, use_script):
    teams = {t.id: t for t in list_teams()}
    dept = next(d for d in building.departments(list(teams), building.load_config()) if d.id == "creative")
    cands = [teams[t] for t in dept.teams]
    models = load_models()
    set_keys()  # 키 없음 → 낱말 비교로도 네 팀이 갈린다
    for text, want in (("내 목소리 모델로 노래 음색 바꾸기 — 보컬 분리하고 Colab 학습", "voice"),
                       ("낭독 녹음을 내 목소리로 바꾸는 말 음성 변환 학습 계획", "voice_speech"),
                       ("회사 송년회 오프닝 노래 가사와 반주 만들기", "music"),
                       ("30평 아파트 방 세 개 평면도 그려줘", "design")):
        assert building.route_request(text, dept, cands, models)["team_id"] == want, text
    script = use_script(Script(TeamPick=building.TeamPick(analysis="음색", team_id="voice", reason="보컬 음색 변환")))
    assert building.route_request("이 노래를 내 목소리로", dept, cands, models)["team_id"] == "voice"
    prompt = script.prompts("TeamPick")[0]
    assert "노래 음색 변환 팀" in prompt and "일상 말 음성 변환 팀" in prompt and "음악 팀" in prompt and "다루지 않음" in prompt


# ─── 가짜 모델로 학습 계획 끝까지 ────────────────────────────────────


def test_fake_model_plan_retries_wrong_epochs(ranks, models, use_script):
    from core.workflow import build_office_graph, stream_office

    team = load_team("voice")
    bad = GOOD_PLAN.replace("epochs 400", "epochs 1000")
    script = use_script(Script(text=[bad, GOOD_PLAN], Delegation=make_delegation("train_eng"),
                               ReviewResult=make_review("approve", 9, answer="")))
    task = next(t for t in team.tasks if t.id == "plan")
    req = build_request(task, PLAN_OK)
    org = team_org(team, ranks)
    graph = build_office_graph(org, ranks, team, models, {}, {"critic": {"enabled": False}})
    # 실무자 한 명에게 바로 맡긴다 (이 사람이 최종 답변자 → final_only 검증기도 돈다)
    events = list(stream_office(graph, org, req.question, "train_eng", brief=req.brief, auto_brief=False,
                                tool_inputs=req.tool_inputs, extra_validators=req.validators, harness_options=req.harness_options))
    tools = next(p for k, p in events if k == "tools")
    assert tools[0].tool == "voice_svc" and tools[0].harness_options["voice_kit"]["epochs"] == 400
    assert "코드가 정한 학습 설정" in script.prompts("text")[0]
    final = [p for k, p in events if k == "step" and p.node_id == "train_eng"][-1]
    assert {h.validator for h in final.harness} == {"voice_guard", "voice_plan"} and all(h.passed for h in final.harness)
    assert any("epochs" in p and "자동 검증(하네스)" in p for p in script.prompts("text"))  # 틀린 값이 피드백으로


# ─── 화면 (AppTest) ─────────────────────────────────────────────────


@pytest.fixture
def app(tmp_path, monkeypatch, set_keys):
    set_keys()
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["team"] = "voice"
    at.query_params["view"] = "team"
    return at


def test_voice_forms_open_and_consent_is_required(app):
    at = app.run()
    assert not at.exception, at.exception
    picker = at.selectbox(key="dept_task_creative")
    assert [o for o in picker.options if o.startswith("🎤 노래 음색 변환 팀 › ")][:2] == [
        "🎤 노래 음색 변환 팀 › 📋 학습 계획 만들기", "🎤 노래 음색 변환 팀 › 🔍 데이터 점검"]
    at = picker.set_value("voice:plan").run()
    assert not at.exception, at.exception
    assert any(c.label.startswith("이 목소리를 학습·변환에 써도 된다는 것을 확인했습니다") for c in at.checkbox)
    at.text_input(key="tf_voice_plan_singer").set_value("myvoice")
    at = at.button[[b.label for b in at.button].index("맡기기")].click().run()
    assert not at.exception, at.exception
    assert any("목소리 사용 확인" in w.value for w in at.warning)

    at = at.selectbox(key="dept_task_creative").set_value("voice:check").run()
    assert not at.exception, at.exception
    assert len(at.get("file_uploader")) == 1


def kit_page(root: str) -> None:
    import sys as _sys

    if root not in _sys.path:
        _sys.path.insert(0, root)
    import voice_ui
    from core import voice as _v
    from core.schemas import ToolResult

    kit = _v.kit_from_inputs({"source": _v.SOURCES[0], "consent": True, "singer": "myvoice", "minutes": 40})
    voice_ui.preview({"run_id": "r1", "tools": [ToolResult(tool="voice_svc", title="t", content="c",
                                                           harness_options={"voice_kit": kit})]})


def test_kit_download_button_shows():
    at = AppTest.from_function(kit_page, kwargs={"root": ROOT}, default_timeout=30).run()
    assert not at.exception, at.exception
    buttons = at.get("download_button")
    assert len(buttons) == 1 and "맞춤 실행 파일" in buttons[0].proto.label


# ═══ 🗣 일상 말 음성 변환 팀 (teams/voice_speech — 같은 도구·같은 코드, kind=speech) ═══════════════

SPEECH_OK = {"source": voice.SOURCES[0], "consent": True, "singer": "myvoice", "minutes": 40, "gpu": "없음",
             "where": "배경 음악 없는 녹음 — Colab 에서 자르기만 (C안)", "goal": "내 낭독을 오디오북 내레이션으로", "voice_kind": "speech"}


def test_speech_team_loads_next_to_song_team_with_same_tool():
    teams = list_teams()
    validate_config(load_ranks(), load_models(), teams)
    team = load_team("voice_speech")
    assert team.name == "🗣 일상 말 음성 변환 팀" and "말하는 녹음" in team.description
    assert team.tools == ["voice_svc"] and team.features == ["voice_kit"]  # 노래 팀과 같은 툴·같은 화면 코드
    assert {"voice_guard"} <= set(team.harness.validators) and {"voice_plan", "voice_data"} <= set(team.harness.final_only)
    assert {t.id: t.fixed for t in team.tasks} == {"plan": {"voice_task": "plan", "voice_kind": "speech"},
                                                   "check": {"voice_task": "check", "voice_kind": "speech"}}
    assert {t.id: t.fixed["voice_kind"] for t in load_team("voice").tasks} == {"plan": "song", "check": "song"}
    rejects = " ".join(team.scope.rejects)
    assert "노래 음색 변환 팀" in rejects and "가짜 녹취" in rejects and "본인 인증" in rejects and "TTS" in rejects
    assert "일상 말 음성 변환 팀" in " ".join(load_team("voice").scope.rejects)  # 노래 팀은 말 요청을 말 팀으로 안내
    cfg = building.load_config()
    assert building.floor_of("voice_speech", teams, cfg) == building.floor_of("voice", teams, cfg)


def test_old_voice_records_stay_with_the_song_team(tmp_path):
    """노래 팀이 옛 id `voice` 를 그대로 쓴다 → 옛 업무·교훈은 옮기거나 이어 붙이지 않아도 노래 팀에 보이고, 말 팀에는 섞이지 않는다."""
    from core.schemas import HumanDecision, Lesson
    from core.storage import Storage

    teams = list_teams()
    links = {t.id: (t.legacy.team, t.legacy.match) for t in teams if t.legacy}  # app.py 가 만드는 것과 같은 연결
    assert "voice" not in {old for old, _ in links.values()} and "voice_speech" not in links
    storage = Storage(tmp_path / "runs.db")
    storage.create_run("r_old", "voice", "lead", "노래 음색 변환(SVC) 학습 계획을 세워줘.\n화자 이름: myvoice")
    storage.save_decision("r_old", HumanDecision(action="adopt", adopted_node_id="lead", final_answer="…", score=5))
    storage.add_lessons("voice", "r_old", [Lesson(agent_id="lead", kind="improve", lesson="epochs 는 코드 값 그대로")])
    storage.set_team_links(links)
    assert [r["id"] for r in storage.list_runs("voice")] == ["r_old"] and storage.list_runs("voice_speech") == []
    assert storage.team_for("voice", "아무 질문") == "voice"
    assert storage.recent_lessons("voice", "lead", 5) == ["epochs 는 코드 값 그대로"]
    assert storage.recent_lessons("voice_speech", "lead", 5) == []


def test_speech_kit_modes_and_borrowed_settings():
    kit = voice.kit_from_inputs(SPEECH_OK)
    assert (kit["kind"], kit["mode"], kit["fast"], kit["epochs"], kit["batch_size"], kit["songs"]) == ("speech", "C", False, 400, 16, None)
    assert any("확인 못 함" in n for n in kit["notes"]) and kit["notes"][0].startswith("말 녹음 약 40분")
    a = voice.kit_from_inputs({**SPEECH_OK, "where": "배경 음악이 섞인 녹음 — 내 PC 에서 분리 (A안)"})
    b = voice.kit_from_inputs({**SPEECH_OK, "where": "배경 음악이 섞인 녹음 — Colab 에서 분리 (B안)", "oom": True})
    assert (a["mode"], a["fast"], b["mode"], b["batch_size"]) == ("A", True, "B", 8)
    assert voice.kit_from_inputs({**SPEECH_OK, "consent": False}) is None
    # 노래 팀 양식에는 C안이 없다 — 말 종류가 아니면 'C안' 글자가 있어도 C안으로 보지 않는다
    assert voice.kit_from_inputs({**PLAN_OK, "where": "(C안)"})["mode"] == "A"
    task = next(t for t in load_team("voice_speech").tasks if t.id == "plan")
    options = next(f for f in task.fields if f.key == "where").options
    assert [voice.kit_from_inputs({**SPEECH_OK, "where": o})["mode"] for o in options] == ["C", "A", "B"]


def _code(book: dict) -> list[str]:
    return ["".join(c["source"]) for c in book["cells"] if c["cell_type"] == "code"]


def test_speech_notebook_turns_on_auto_f0_and_adds_one_split_cell():
    original = json.loads((voice.ASSET_DIR / voice.NOTEBOOK).read_text(encoding="utf-8"))
    song = json.loads(voice.build_notebook(voice.kit_from_inputs(PLAN_OK)))
    nb = json.loads(voice.build_notebook(voice.kit_from_inputs(SPEECH_OK)))
    song_infer = next(c for c in _code(song) if "!svc infer" in c)
    speech_infer = next(c for c in _code(nb) if "!svc infer" in c)
    assert " -a " not in song_infer and " -fm crepe -na -t 0 " in song_infer and "cover.wav" in song_infer  # 노래는 끈다
    assert '"{WORK}/speech.wav"' in speech_infer and " -fm crepe -a -t 0 " in speech_infer and "speech_converted.wav" in speech_infer
    assert "cover" not in next(c for c in _code(nb) if "Audio(" in c)
    assert len(nb["cells"]) == len(original["cells"]) + 1  # C안: 자르기 셀 하나만 더함
    split = [c for c in _code(nb) if "svc pre-split" in c]
    assert len(split) == 1 and "speech.zip" in split[0] and "/content/speech_raw/myvoice" in split[0]
    order = [c.split("\n")[0][:6] for c in _code(nb)]
    assert order.index("# 4-C)") == order.index("# 4) [") + 1  # A안 셀 바로 뒤
    head = "".join(nb["cells"][0]["source"])
    assert "4-C" in head and "AI 로 합성한 음성" in head and "일상 말 음성 변환 팀" in head
    a = json.loads(voice.build_notebook(voice.kit_from_inputs({**SPEECH_OK, "where": "배경 음악이 섞인 녹음 — 내 PC 에서 분리 (A안)"})))
    assert len(a["cells"]) == len(original["cells"]) and not any("svc pre-split" in c for c in _code(a))
    for c in nb["cells"]:  # 코드 셀이 파이썬으로 읽힌다 (! · % 줄은 빼고)
        if c["cell_type"] == "code":
            compile("".join(x for x in c["source"] if not x.lstrip().startswith(("!", "%"))), "cell", "exec")


def test_speech_kit_zip_and_evidence_cite_sources_and_unknowns():
    kit = voice.kit_from_inputs(SPEECH_OK)
    with zipfile.ZipFile(io.BytesIO(voice.kit_zip(kit))) as zf:
        assert set(zf.namelist()) == {voice.NOTEBOOK, "README.md", "preprocess.py", "run.bat"}
        readme = zf.read("README.md").decode("utf-8")
    assert "speech.zip" in readme and "`-a`" in readme and "확인 못 한 것" in readme and "AI 로 합성한 음성" in readme
    assert "cover.wav" not in readme and "반주" not in readme
    evidence = voice.plan_evidence(kit)
    assert "| 종류 | 말 음성 변환" in evidence and "-fm crepe -a -t 0" in evidence and "pre-split" in evidence
    assert all(s in evidence for s in voice.SPEECH_SOURCES + voice.SPEECH_UNKNOWN)
    assert "preprocess.py --input" not in evidence  # C안은 내 PC 전처리를 쓰지 않는다
    song = voice.plan_evidence(voice.kit_from_inputs(PLAN_OK))
    assert "| 종류 | 노래 음색 변환 |" in song and "확인 못 한 것" not in song


@pytest.mark.parametrize("bad", [
    "엄마인 척 전화해서 돈 부쳐 달라고 하려고", "상사 목소리로 대신 전화 걸기", "아빠 행세하는 음성 메시지",
    "내 목소리로 은행 ARS 본인 확인 통과하기", "목소리로 본인 인증 통과", "성문 인증 뚫는 법", "voice ID 우회",
    "친구가 그런 말을 한 것처럼 가짜 녹취 만들기", "통화 녹음을 조작해서 증거로 제출", "딥페이크 음성 만들기",
    "상대가 모르게 다른 사람 목소리로 바꿔서", "진짜 본인인 줄 알게", "대통령 목소리로 연설 만들기", "유명 유튜버 음성 복제",
    "보이스 피싱", "몰래 녹음한 동료 목소리로 학습"])
def test_speech_misuse_refused_before_and_inside_the_team(bad):
    assert voice.misuse(bad), bad
    with pytest.raises(TaskInputError, match="도움을 드리지 않습니다"):
        voice.check_form("plan", SPEECH_OK, extra=bad)
    blocked = TOOLS["voice_svc"](bad, {}, None)  # 자유 질문·부서장 배정으로 들어와도
    assert blocked.harness_options["voice_blocked"] and "범위 밖" in blocked.content


def test_speech_misuse_does_not_catch_ordinary_requests():
    for ok in ("내 낭독 녹음을 오디오북 내레이션으로", "강의 녹음 목소리를 또렷하게", "동의를 받은 동생 목소리로 동화 낭독",
               "변환한 말소리가 끊겨요", "속도가 빠른 대화 녹음도 되나요", "녹음 파일이 10분 밖에 없어요",
               "자연스럽게 진짜 사람 목소리처럼 들리게 하고 싶어", "전화 받을 때 쓸 내 안내 멘트 녹음"):
        assert voice.misuse(ok) == "", ok


GOOD_SPEECH = """목소리 출처: 내 목소리 — 학습·변환에 써도 된다고 확인함.

## 실행 순서
1. 녹음을 `speech.zip` 으로 묶어 Drive `MyDrive/svc_project/` 에 업로드
2. 4-C 셀에서 `svc pre-split` 으로 자르기 → `dataset_raw/myvoice/`
3. 학습: epochs 400 · batch_size 16 으로 `svc train`
4. 변환: `svc infer speech.wav -fm crepe -a -t 0` — 음높이 자동 예측을 켠다

결과물을 남에게 들려줄 때는 'AI 로 합성한 음성'이라고 표시하세요. 다른 사람인 척 쓰지 않습니다.
"""


def test_speech_plan_validator_needs_auto_f0():
    kit = voice.kit_from_inputs(SPEECH_OK)
    assert check("voice_plan", GOOD_SPEECH, voice_kit=kit).passed
    no_a = check("voice_plan", GOOD_SPEECH.replace(" -a", "").replace(" — 음높이 자동 예측을 켠다", ""), voice_kit=kit)
    assert not no_a.passed and "자동 예측" in no_a.log
    off = check("voice_plan", GOOD_SPEECH + "\n쇳소리가 나면 `-na` 로 끄세요.", voice_kit=kit)
    assert not off.passed and "-na" in off.log
    assert not check("voice_plan", GOOD_SPEECH.replace("epochs 400", "epochs 900"), voice_kit=kit).passed
    # 노래 계획에는 -a 를 요구하지 않는다 (노래 팀 동작은 그대로)
    assert check("voice_plan", GOOD_PLAN, voice_kit=voice.kit_from_inputs(PLAN_OK)).passed


def test_speech_data_check_uses_same_counts_and_says_borrowed_rule(dataset):
    data = TOOLS["voice_svc"]("q", {"dataset": str(dataset), "dataset_name": "dataset_raw.zip", "voice_task": "check",
                                    "voice_kind": "speech"}, None)
    stats = data.harness_options["voice_dataset"]
    assert stats["kind"] == "speech" and stats["clips"] == 6 and stats["verdict"] == "부족"  # 숫자는 노래와 같은 계산
    assert "말에 맞는 분량은 공식 문서에서 확인 못 함" in data.content
    assert "확인 못 함" not in voice.dataset_table(voice.inspect_zip(dataset))  # 노래 표는 그대로


def test_scope_keeps_song_and_speech_apart():
    teams = list_teams()
    by = {t.id: t for t in teams}
    for q in ("낭독 녹음으로 내 목소리 모델을 학습하려면 몇 분 필요해?", "오디오북 내레이션 말 음성 변환할 때 음높이 자동 예측 켜야 해?",
              "강의 녹음 파일을 svc pre-split 으로 자르면 돼?"):
        d = scope.check(q, by["voice_speech"], teams, None, route=no_router)
        assert d["in_scope"] and d["stage"] == "keyword", q
        assert not scope.heuristic(q, by["voice"], teams), q
    q = "노래 커버 보컬 음색 변환 키 조절은?"
    assert scope.heuristic(q, by["voice"], teams) and not scope.heuristic(q, by["voice_speech"], teams)


def test_speech_form_opens_and_needs_consent(app):
    at = app.run()
    assert not at.exception, at.exception
    picker = at.selectbox(key="dept_task_creative")
    assert [o for o in picker.options if o.startswith("🗣 일상 말 음성 변환 팀 › ")][:2] == [
        "🗣 일상 말 음성 변환 팀 › 📋 말 변환 학습 계획 만들기", "🗣 일상 말 음성 변환 팀 › 🔍 데이터 점검"]
    at = picker.set_value("voice_speech:plan").run()
    assert not at.exception, at.exception
    at.text_input(key="tf_voice_speech_plan_singer").set_value("myvoice")
    at = at.button[[b.label for b in at.button].index("맡기기")].click().run()
    assert not at.exception, at.exception
    assert any("목소리 사용 확인" in w.value for w in at.warning)


def speech_kit_page(root: str, where: str) -> None:
    import sys as _sys

    if root not in _sys.path:
        _sys.path.insert(0, root)
    import voice_ui
    from core import voice as _v
    from core.schemas import ToolResult

    kit = _v.kit_from_inputs({"source": _v.SOURCES[0], "consent": True, "singer": "myvoice", "minutes": 40, "voice_kind": "speech",
                              "where": where})
    voice_ui.preview({"run_id": "r1", "tools": [ToolResult(tool="voice_svc", title="t", content="c",
                                                           harness_options={"voice_kit": kit})]})


def test_speech_kit_download_uses_same_screen_code():
    # 함수 본문은 Windows 기본 인코딩으로 임시 파일에 저장된다 → 한글 값은 본문에 쓰지 않고 kwargs 로 넘긴다
    at = AppTest.from_function(speech_kit_page, kwargs={"root": ROOT, "where": SPEECH_OK["where"]}, default_timeout=30).run()
    assert not at.exception, at.exception
    buttons = at.get("download_button")
    assert len(buttons) == 1 and "맞춤 실행 파일" in buttons[0].proto.label
    assert any("음높이 자동 예측" in c.value for c in at.caption)
