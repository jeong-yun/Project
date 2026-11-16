"""👔 부서장 이름 — 화면·프롬프트에서 부서장을 '부장'이라 부르지 않는다 (2026-11-16).

팀 안에도 '부장' 직급(rank gm, 예: 부장·리서치)이 있어서 화면의 '부장'이 부서장인지 팀 안 부장인지 헷갈렸다.
  (가) 부서장 = '부서장' 으로 맞춤 — 이름 · 기본 이름 · 화면 문구 · 모델에게 보내는 프롬프트
  (나) 팀 안 gm 직급 = '부장' 그대로 — config/ranks.yaml · 팀원 이름(부장·리서치 등)
새 부서를 만들 때의 이름('세무부 부서장')은 test_reception.py 가 본다.
"""

from core import building, dept_heads, followup
from core.config import BASE_DIR, list_teams, load_yaml

# 부서장만 다루는 파일 — 여기에 '부장'이 나오면 부서장을 옛 이름으로 부른 것이다 ('부서장'에는 '부장'이 들어 있지 않다)
HEAD_ONLY_FILES = ["app.py", "building_ui.py", "prompt_ui.py", "reception_ui.py", "run_control_ui.py",
                   "core/followup.py", "core/dept_heads.py", "core/workflow.py", "core/reception.py", "core/progress.py"]


def test_head_only_files_do_not_say_bujang():
    left = {f: [n for n, line in enumerate((BASE_DIR / f).read_text(encoding="utf-8").splitlines(), 1) if "부장" in line]
            for f in HEAD_ONLY_FILES}
    assert {f: lines for f, lines in left.items() if lines} == {}


def test_every_department_head_is_called_buseojang():
    cfg = building.load_config()
    heads = [d["head"]["name"] for d in cfg["departments"]] + [cfg["other"]["head"]["name"]]
    assert heads and all(h.endswith("부서장") for h in heads), heads
    bare = building.Department(id="x", name="실험부", emoji="🧪", color="#000000", description="")
    assert dept_heads.head_of(bare).name == "실험부 부서장"  # departments.yaml 에 head 가 없을 때
    assert all("부장" not in label for label in dept_heads.SOURCE_LABELS.values())


def test_revision_review_text_says_buseojang():
    """💬 수정본 아래 검토 표시 · 다음 수정 요청 글 — 이름을 못 받아도 '부서장'."""
    turn = followup.FollowupTurn(request="r", reply="", revised=True, content="c", auto_fixed=["용어 통일"])
    shown = followup.head_review_markdown(turn, "")
    assert "부서장이(가) 추가로 고칠 점 1개를 직접 고쳤습니다" in shown and "부장" not in shown


def test_team_gm_rank_keeps_bujang():
    """팀 안 gm 직급 이름과 팀원 이름은 그대로 '부장'."""
    assert any(r["id"] == "gm" and r["name"] == "부장" for r in load_yaml(BASE_DIR / "config" / "ranks.yaml")["ranks"])
    names = {n.name for t in list_teams() if t.org for n in t.org.walk()}
    for member in ("부장·리서치", "부장·실행", "부장·리스크", "부장·직무 검토", "부장·수석에디터", "부장·통제 매핑", "부장·변경 검토"):
        assert member in names, member
    assert not any("부서장" in n for n in names)  # 팀원 이름이 부서장으로 바뀌지 않았다
