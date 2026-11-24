"""🎧 듣기용 MP3 — 대본 말투 '쉽게(초등학생 눈높이) / 보통' 고르기와, 카탈로그에 이미 있는 개념으로 만들기.

core/listen.py (make_script level · level_path) · concept_ui (listen_panel 의 말투 고르기 · catalog_listen) · app.py (🎧 개념 듣기 업무).
실제 모델·실제 TTS 는 부르지 않는다 (use_script · fake_tts — tests/conftest.py)."""

from pathlib import Path

import pytest

from core import catalog as cat
from core import listen
from tests.fakes import Script, silent_mp3
from tests.test_listen import ENTRY, GOOD, script

RID = "run12345678"


# ─── 대본: 쉽게 / 보통 ─────────────────────────────────────────────


def test_easy_script_asks_for_child_level_words(use_script, models):
    s = use_script(Script(ListenScript=GOOD))
    listen.make_script(ENTRY, models, level="easy")
    prompt = s.prompts("ListenScript")[0]
    assert "초등학생" in prompt  # 듣는 사람
    assert "짧" in prompt and "풀어" in prompt  # 짧은 문장 · 어려운 말은 처음 나올 때 풀어 말하기
    assert "실제로는" in prompt  # 비유 뒤에 진짜 내용으로 이어 — 비유가 사실처럼 들리지 않게
    assert "타일 단위" in prompt and "약어" in prompt and "본문에 없는 숫자" in prompt  # 본문·공통 규칙은 그대로


def test_normal_script_keeps_lecture_rules(use_script, models):
    s = use_script(Script(ListenScript=GOOD))
    listen.make_script(ENTRY, models, level="normal")
    prompt = s.prompts("ListenScript")[0]
    assert "초등학생" not in prompt and "강의" in prompt


def test_default_level_is_easy(use_script, models):
    s = use_script(Script(ListenScript=GOOD))
    listen.make_script(ENTRY, models)
    assert "초등학생" in s.prompts("ListenScript")[0]


def test_easy_rewrite_keeps_code_check(use_script, models):
    s = use_script(Script(ListenScript=[script("2019년에 Google 이 만들었어요."), GOOD]))
    result = listen.make_script(ENTRY, models, level="easy")
    retry = s.prompts("ListenScript")[1]
    assert "2019" in retry and "Google" in retry  # 본문에 없는 숫자·영문 용어 검사는 쉽게 말투에도 그대로
    assert "초등학생" in retry  # 다시 쓸 때도 쉽게
    assert result.calls == 2 and result.warnings == [] and result.script == GOOD


# ─── 저장 파일: 쉽게는 이름 끝에 표시, 보통은 예전 이름 그대로 ─────────────


def test_easy_files_have_own_names_and_normal_keeps_old_names(tmp_path):
    json_path, mp3_path = listen.paths(tmp_path, RID, ENTRY)
    assert listen.level_path(mp3_path, "normal") == mp3_path  # 말투를 고르기 전에 만든 MP3 = 보통
    easy_json, easy_mp3 = listen.level_path(json_path, "easy"), listen.level_path(mp3_path, "easy")
    assert easy_mp3 != mp3_path and easy_mp3.suffix == ".mp3" and easy_mp3.parent == mp3_path.parent
    assert easy_json.suffix == ".json" and easy_json.stem == easy_mp3.stem
    easy_mp3.parent.mkdir(parents=True, exist_ok=True)  # 범주 폴더(여기서는 미분류/)는 저장할 때 만든다
    easy_mp3.write_bytes(b"x")
    assert listen.saved_audio(tmp_path, RID) == easy_mp3  # 🕘 지난 업무 창도 쉽게 MP3 를 찾는다


# ─── 화면: 말투 고르기 (concept_ui.listen_panel) ─────────────────────


def _panel(entry, folder, saved):
    import concept_ui
    from core.config import load_models

    # AppTest.from_function 은 함수 몸통만 실행한다 — 모듈 상수(RID)를 못 보므로 값을 그대로 쓴다
    concept_ui.listen_panel(entry, {"run_id": "run12345678"}, folder, load_models(), lambda run: saved.append(run["run_id"]))


def _app(page, **kwargs):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(page, kwargs=kwargs, default_timeout=30).run()
    assert not at.exception, at.exception
    return at


def _click(at, words: str):
    at = next(b for b in at.button if words in b.label).click().run()
    assert not at.exception, at.exception
    return at


def test_panel_writes_easy_script_by_default(fake_tts, use_script, tmp_path):
    s = use_script(Script(ListenScript=GOOD))
    at = _app(_panel, entry=ENTRY, folder=tmp_path, saved=[])
    assert at.radio(key=f"listen_level_{RID}").value == "easy"
    at = _click(at, "듣기용 MP3 만들기")
    assert len(s.calls) == 1 and "초등학생" in s.prompts("ListenScript")[0]
    at = _click(at, "음성 만들기")
    assert fake_tts and len(at.get("audio")) == 1
    _, mp3_path = listen.paths(tmp_path, RID, ENTRY)
    assert listen.level_path(mp3_path, "easy").exists() and not mp3_path.exists()


def test_old_mp3_stays_under_normal_and_easy_can_be_made_beside_it(fake_tts, use_script, tmp_path):
    json_path, mp3_path = listen.paths(tmp_path, RID, ENTRY)  # 말투를 고르기 전에 만든 MP3 (보통)
    listen.save_script(json_path, listen.ScriptResult(GOOD, [], 1))
    mp3_path.write_bytes(silent_mp3())
    s = use_script(Script(ListenScript=[GOOD, GOOD]))
    at = _app(_panel, entry=ENTRY, folder=tmp_path, saved=[])
    assert not at.get("audio") and any("듣기용 MP3 만들기" in b.label for b in at.button)  # 기본 '쉽게'는 아직 없음
    at = at.radio(key=f"listen_level_{RID}").set_value("normal").run()
    assert len(at.get("audio")) == 1 and s.calls == []  # 예전 MP3 는 '보통'에서 그대로 들린다
    at = at.radio(key=f"listen_level_{RID}").set_value("easy").run()
    at = _click(at, "듣기용 MP3 만들기")
    at = _click(at, "음성 만들기")
    assert listen.level_path(mp3_path, "easy").exists() and mp3_path.exists()  # 쉽게는 옆에 새로 — 예전 것은 그대로


def test_remake_from_script_deletes_only_that_level(fake_tts, use_script, tmp_path):
    json_path, mp3_path = listen.paths(tmp_path, RID, ENTRY)
    for level in ("normal", "easy"):
        listen.save_script(listen.level_path(json_path, level), listen.ScriptResult(GOOD, [], 1))
        listen.level_path(mp3_path, level).write_bytes(silent_mp3())
    s = use_script(Script(ListenScript=GOOD))
    at = _app(_panel, entry=ENTRY, folder=tmp_path, saved=[])
    assert len(at.get("audio")) == 1
    at = _click(at, "대본부터 다시 만들기")  # 이미 만든 MP3 → 새 지시문으로 다시
    assert not listen.level_path(mp3_path, "easy").exists() and mp3_path.exists()  # 보통 MP3 는 남는다
    at = _click(at, "듣기용 MP3 만들기")
    assert len(s.calls) == 1 and "초등학생" in s.prompts("ListenScript")[0]


# ─── 카탈로그의 예전 개념 → MP3 (concept_ui.catalog_listen) ─────────────

OTHER = """보정된 테스트 타임 스케일링(Calibrated Test-Time Scaling, CaTS) (작성일: 26.10.01)
1. 정의 및 개념
 - 추론할 때 계산을 더 써서 여러 답 가운데 하나를 고르는 방법임."""


@pytest.fixture
def catalog_root(tmp_path):
    root = tmp_path / "CS" / "Term" / "Every_day"
    root.mkdir(parents=True)
    (root / cat.CATEGORY_FILES["AI-Model"]).write_text(f"{ENTRY}\n\n{OTHER}\n", encoding="utf-8")
    return root


def _catalog_page(root, folder):
    from pathlib import Path  # AppTest.from_function 은 함수 몸통만 실행한다 — 쓰는 것은 안에서 불러온다

    import concept_ui
    from core.config import load_models

    ctx = concept_ui.CatalogContext(Path(root), None, "", concept_ui.load_catalog(Path(root)))
    concept_ui.catalog_listen(ctx, "concept", Path(folder), load_models())


def test_catalog_concept_becomes_mp3_one_at_a_time_on_click(fake_tts, use_script, catalog_root, tmp_path):
    entries = cat.Catalog.load(catalog_root).entries
    first, second = (e.name for e in entries)
    folder = tmp_path / "results" / "concept"
    s = use_script(Script(ListenScript=GOOD))
    at = _app(_catalog_page, root=str(catalog_root), folder=str(folder))
    assert not at.multiselect  # 여러 개를 한 번에 만드는 기능은 없다 (요금)
    assert fake_tts == [] and s.calls == []  # 고르기·그리기만으로는 부르지 않는다
    at = at.selectbox(key="cl_target_concept").set_value(first).run()
    at = _click(at, "듣기용 MP3 만들기")
    prompt = s.prompts("ListenScript")[0]
    assert "타일 단위" in prompt and "초등학생" in prompt  # 고른 항목 본문으로, 쉽게
    assert fake_tts == [] and any("$" in c.value and "추정" in c.value for c in at.caption)  # 음성 전에 예상 요금
    at = _click(at, "음성 만들기")
    assert len(at.get("audio")) == 1
    made = list(folder.rglob("*.mp3"))
    assert len(made) == 1 and listen.CATALOG_RUN in made[0].name  # 업무 번호 대신 'catalog'
    assert entries[0].category in listen.CATEGORY_FOLDERS and made[0].parent == folder / entries[0].category  # 항목의 범주 폴더 (미분류 아님)
    n_tts = len(fake_tts)
    at = at.selectbox(key="cl_target_concept").set_value(second).run()  # 다른 개념 — 저절로 만들지 않는다
    assert not at.get("audio") and len(fake_tts) == n_tts and len(s.calls) == 1
    at = at.selectbox(key="cl_target_concept").set_value(first).run()  # 다시 고르면 저장한 MP3 를 그대로
    assert len(at.get("audio")) == 1 and len(fake_tts) == n_tts and len(s.calls) == 1


def test_empty_catalog_says_what_to_do(tmp_path):
    root = tmp_path / "빈 카탈로그"
    root.mkdir()
    at = _app(_catalog_page, root=str(root), folder=str(tmp_path / "results"))
    assert at.info and not at.button


def test_listen_task_in_app_opens_catalog_picker(catalog_root, tmp_path, monkeypatch, set_keys):
    from streamlit.testing.v1 import AppTest

    set_keys()
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=90)
    at.query_params["team"] = "concept"
    at.query_params["view"] = "team"
    at = at.run()
    at = at.text_input(key="concept_root").set_value(str(catalog_root)).run()
    at = at.selectbox(key="dept_task_research").set_value("concept:listen").run()  # 부서 업무 목록 → 🎧 개념 듣기
    assert not at.exception, at.exception
    assert at.selectbox(key="cl_target_concept").options  # 카탈로그 항목에서 고른다
    assert any("듣기용 MP3 만들기" in b.label for b in at.button)
    assert not any(b.label == "맡기기" for b in at.button)  # 팀에 맡기지 않고 바로 만든다
