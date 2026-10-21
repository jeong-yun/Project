"""화면 연기 테스트 (Streamlit AppTest) — 📐 설계 팀을 열었을 때 업무 양식(첨부·체크박스 칸 포함)이 그려지는지.

실제 모델 호출 없음 (업무를 맡기지 않는다). 데이터·결과 폴더는 임시 폴더를 쓴다.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def app(tmp_path, monkeypatch, set_keys):
    set_keys()  # API 키 없음
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["team"] = "design"
    return at


def test_design_team_page_renders_task_forms(app):
    at = app.run()
    assert not at.exception, at.exception
    picker = at.selectbox(key="task_pick_design")
    assert picker.options[:3] == ["📐 평면도 만들기", "✏️ 도면 수정·층 추가", "🧊 3D 만들기"]
    assert any(c.label == "🧊 3D 모델도 만들기" for c in at.checkbox)
    assert any(t.label.startswith("어떤 집·건물인가요") for t in at.text_area)

    at = picker.set_value("edit").run()
    assert not at.exception, at.exception
    uploaders = at.get("file_uploader")
    assert len(uploaders) == 1
    assert any(c.label == "첨부 대신 최근 저장한 설계안 사용" for c in at.checkbox)

    at = at.selectbox(key="task_pick_design").set_value("model3d").run()
    assert not at.exception, at.exception
    assert next(c for c in at.checkbox if c.label == "🧊 3D 모델 만들기").value is True


def test_design_form_requires_change_text(app):
    at = app.run()
    at = at.selectbox(key="task_pick_design").set_value("edit").run()
    at = at.button[[b.label for b in at.button].index("맡기기")].click().run()
    assert not at.exception, at.exception
    assert any("필수 입력이 비어 있습니다: 바꿀 내용" in w.value for w in at.warning)
