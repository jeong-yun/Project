"""코드가 가리키는 프롬프트·페이지 파일이 실제로 있는지, 모델 설정 규칙이 한 곳에서 나오는지."""

import os
import re

from langchain_core.prompts import load_prompt

import common
from fakes import FAKE_KEY, fake_chat

ROOT = common.BASE_DIR


def test_prompt_files_exist_and_load():
    word = load_prompt(str(common.PROMPTS_DIR / "English.yaml"), encoding="utf-8")
    assert word.input_variables == ["word"]

    judge = load_prompt(str(common.PROMPTS_DIR / "safety_judge.yaml"), encoding="utf-8")
    assert {"category", "safe_definition", "attack_prompt", "model_response"} <= set(judge.input_variables)

    summary = load_prompt(str(common.PROMPTS_DIR / "summary.yaml"), encoding="utf-8")
    assert summary.input_variables == ["question"]
    # 요약할 글은 '지시가 아닌 데이터' 표시선 안에 들어간다
    assert common.DATA_OPEN + "\n{question}\n" + common.DATA_CLOSE in summary.template


def test_no_prompt_is_downloaded_from_langchain_hub():
    """남이 올린 공개 프롬프트를 내려받아 실행하지 않는다 (GHSA-3644-q5cj-c5c7, langchain < 0.3.30).
    그 경로를 쓰던 langchain 본체·langchainhub·langchain_teddynote 도 더는 쓰지 않는다."""
    code = "\n".join(p.read_text(encoding="utf-8") for p in ROOT.rglob("*.py") if "tests" not in p.parts)
    assert "hub.pull" not in code
    assert not re.search(r"^\s*(from|import) (langchain|langchainhub|langchain_teddynote)\b", code, re.M)


def test_enable_langsmith_only_with_key(monkeypatch):
    """LangSmith 키가 없으면 아무것도 바꾸지 않고, 있으면 호출 기록을 켠다 (예전 teddynote 와 같은 동작)."""
    common.enable_langsmith("프로젝트")
    assert "LANGSMITH_TRACING" not in os.environ

    monkeypatch.setenv("LANGCHAIN_API_KEY", FAKE_KEY)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    common.enable_langsmith("프로젝트")
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_PROJECT"] == "프로젝트"
    # 끝난 뒤 검사(conftest)가 '켜진 기록'을 새어 나간 값으로 보지 않게 되돌린다
    monkeypatch.delenv("LANGSMITH_TRACING")
    monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)


def test_every_page_path_in_code_points_to_a_real_file():
    """app.py 와 pages/ 안에서 'pages/….py' 라고 적은 경로는 모두 실제 파일이어야 한다."""
    sources = [ROOT / "app.py", *sorted((ROOT / "pages").glob("*.py"))]
    found = set()
    for src in sources:
        found |= set(re.findall(r"pages/[\w.]+\.py", src.read_text(encoding="utf-8")))
    assert "pages/1_english_word_maker.py" in found
    assert "pages/2_ai_safety_eval.py" in found
    for rel in found:
        assert (ROOT / rel).is_file(), f"{rel} 가 없습니다"


def test_prompts_are_read_from_file_based_path_not_cwd(monkeypatch, tmp_path):
    """다른 폴더에서 실행해도 프롬프트를 찾는다."""
    monkeypatch.chdir(tmp_path)
    from ai_safety import judge

    monkeypatch.setattr(judge, "make_llm", lambda *a, **k: fake_chat("{}"))

    judge.build_judge_chain("아무-모델")  # 파일을 못 찾으면 여기서 오류


def test_make_llm_skips_temperature_for_gpt5(monkeypatch):
    """gpt-5 계열에는 temperature 를 넘기지 않고, 그 밖에는 넘긴다 (실제 호출 없음 — 만들기만)."""
    made = []
    monkeypatch.setattr(common, "ChatOpenAI", lambda **kw: made.append(kw) or kw)
    common.make_llm("gpt-5.1", temperature=0)
    common.make_llm("gpt-4.1", temperature=0)
    assert made == [{"model": "gpt-5.1"}, {"model": "gpt-4.1", "temperature": 0}]


def test_screens_use_the_shared_make_llm():
    """홈·단어장 화면이 ChatOpenAI 를 직접 만들지 않는다 (규칙이 한 곳에만 있도록)."""
    for rel in ("app.py", "pages/1_english_word_maker.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "ChatOpenAI(" not in text, rel
        assert "make_llm(" in text, rel

    from ai_safety import judge

    assert judge.make_llm is common.make_llm


def test_requirements_list_only_imported_packages():
    """requirements.txt 의 패키지는 이 폴더 코드가 실제로 import 하는 것이어야 한다."""
    import_name = {
        "python-dotenv": "dotenv",
        "python-docx": "docx",
        "fpdf2": "fpdf",
        "langchain-core": "langchain_core",
        "langchain-openai": "langchain_openai",
        "pyyaml": "load_prompt",  # load_prompt 가 .yaml 을 읽을 때 필요
        "xlsxwriter": "xlsxwriter",  # pd.ExcelWriter(engine="xlsxwriter")
        "pytest": "pytest",
    }
    code = "\n".join(
        p.read_text(encoding="utf-8") for p in ROOT.rglob("*.py") if "__pycache__" not in p.parts
    )
    names = [
        re.split(r"[=<>~!\[ ]", line.strip(), maxsplit=1)[0].lower()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert "streamlit" in names and "langchain-openai" in names
    for name in names:
        needle = import_name.get(name, name.replace("-", "_"))
        assert needle in code, f"{name} 은(는) 코드에서 쓰지 않습니다"
