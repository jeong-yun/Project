"""👔 포트폴리오·자소서 팀 — 그림 JD·요약 문서 (png·jpg·webp · 스캔 PDF) → 비전 모델 글자 옮기기 → 맡기기 전 확인 단계.

가짜 비전 모델(llm_factory.invoke_structured 바꿔치기) — 네트워크·비용 없음. 그림은 테스트 안에서 PIL 로 만든 가상 공고다.
"""

import io
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from core import career, llm_factory, member_models, vision
from core.config import load_team, team_org
from core.harness import REGISTRY
from core.tools import career_docs

ROOT = Path(__file__).resolve().parent.parent
JD_TEXT = "[자격 요건]\n- 보안 로그 분석 경험 3년 이상\n[우대 사항]\n- 침해사고 대응 경험"


def image_bytes(fmt: str = "PNG", size=(400, 300), text: str = "가상 채용 공고") -> bytes:
    image = Image.new("RGB", size, "white")
    ImageDraw.Draw(image).text((10, 10), text, fill="black")
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def fake_vision(monkeypatch, set_keys):
    """그림 1장(조각 묶음) = 호출 1번. 돌려주는 글은 hint(파일 이름·쪽)에 따라 texts 에서 고른다."""
    set_keys("google")
    calls: list[dict] = []
    texts: dict[str, str] = {}

    def fake(rank_models, models, schema, messages, who):
        prompt = messages[1].content[0]["text"]
        images = [c for c in messages[1].content if c.get("type") == "image_url"]
        key = next((k for k in texts if k in prompt), "")
        calls.append({"prompt": prompt, "images": len(images), "model": rank_models.primary.label, "schema": schema})
        parsed = vision.Transcript(text=texts.get(key, JD_TEXT), unclear=["3년 이상 (흐림)"] if key == "" else [])
        return llm_factory.StructuredCall(parsed=parsed, spec=rank_models.primary, is_fallback=False, elapsed_sec=0.1,
                                          input_tokens=1000, output_tokens=200)

    monkeypatch.setattr(llm_factory, "invoke_structured", fake)
    return calls, texts


def career_chains(models) -> list:
    team = load_team("career")
    from core.config import load_ranks

    org = team_org(team, load_ranks())
    plans = member_models.team_rank_models(org, models, team, {})
    return [plans[n.id] for n in reversed(list(org.walk()))]


# ─── 그림 준비 · 모델 고르기 ───────────────────────────────────────────────


def test_prepare_image_downscales_wide_and_tiles_tall_screenshots():
    wide = vision.prepare_image(image_bytes(size=(3200, 1000)))
    assert len(wide) == 1 and Image.open(io.BytesIO(wide[0])).size == (1600, 500)
    tall = vision.prepare_image(image_bytes(size=(1000, 5000)))
    sizes = [Image.open(io.BytesIO(t)).size for t in tall]
    assert len(tall) == 3 and all(w == 1000 and h <= vision.TILE_HEIGHT for w, h in sizes)
    assert sum(h for _, h in sizes) - vision.TILE_OVERLAP * (len(tall) - 1) == 5000  # 겹친 만큼만 더 길다
    huge = vision.prepare_image(image_bytes(size=(1000, 60000)))
    assert len(huge) <= vision.MAX_TILES
    rgba = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
    buf = io.BytesIO()
    rgba.save(buf, format="WEBP")
    assert Image.open(io.BytesIO(vision.prepare_image(buf.getvalue())[0])).getpixel((5, 5)) == (255, 255, 255)  # 투명 → 흰색


def test_reader_models_follow_member_chains_free_only_and_disabled(set_keys, models):
    import career_ui

    set_keys("google", "openai")
    pool = career_ui.reader_pool(career_chains(models))
    usable = vision.usable_vision_models(models, pool)
    assert usable and usable[0].model == "gemini-3.5-flash-lite"  # 사원(사실 추출)의 가벼운 모델이 먼저
    assert all(s.provider in vision.VISION_PROVIDERS for s in usable)
    assert all(s.provider != "anthropic" for s in usable)  # models.yaml disabled_providers
    # 🆓 무료 모델만 사용 → 유료 비전 모델로 넘어가지 않는다
    free = llm_factory.free_only(models)
    assert vision.usable_vision_models(free, pool) == []
    assert "쓸 수 없습니다" in career_ui.no_model_reason(pool, free)
    # 🎛 팀원이 무료 모델만 쓰게 골라진 경우 (대안도 무료만) → 그림 모델 후보가 없다
    team = load_team("career")
    from core.config import load_ranks

    org = team_org(team, load_ranks())
    choices = {n.id: "freellm:gpt-oss-20b" for n in org.walk()}
    choices["lead"] = "freellm:gemini-3.7-flash"
    chains = member_models.team_rank_models(org, models, team, choices)
    locked = career_ui.reader_pool([chains[n.id] for n in reversed(list(org.walk()))])
    assert vision.usable_vision_models(models, locked) == []
    assert "무료" in career_ui.no_model_reason(locked, models)


# ─── 파일 → 글자 ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("suffix,fmt", [(".png", "PNG"), (".jpg", "JPEG"), (".webp", "WEBP")])
def test_image_jd_is_transcribed_by_vision_model(fake_vision, models, tmp_path, suffix, fmt):
    calls, _ = fake_vision
    path = tmp_path / f"ab12cd34_공고{suffix}"
    path.write_bytes(image_bytes(fmt))
    assert career_docs.needs_vision(str(path))
    items = career_docs.vision_items({"jd_file": [str(path)], "jd_file_names": [f"공고{suffix}"]})
    assert list(items) == ["jd"] and items["jd"][0].name == f"공고{suffix}"
    rank = vision.vision_models(models, career_ui_pool(models))
    out = career_docs.transcribe_file(str(path), f"공고{suffix}", rank, models)
    assert out.text == JD_TEXT and "비전 모델" in out.how and len(out.reads) == 1
    assert calls[0]["images"] == 1 and calls[0]["schema"] is vision.Transcript
    assert out.cost == pytest.approx((1000 * 0.30 + 200 * 2.50) / 1_000_000)  # pricing.yaml gemini-3.5-flash-lite


def career_ui_pool(models):
    import career_ui

    return career_ui.reader_pool(career_chains(models))


def test_several_images_are_joined_in_upload_order(fake_vision, models, tmp_path):
    calls, texts = fake_vision
    texts.update({"1.png": "[주요 업무]\n- 로그 분석", "2.jpg": "[자격 요건]\n- 경험 3년 이상", "3.png": "[우대 사항]\n- 자격증"})
    paths = []
    for name, fmt in (("1.png", "PNG"), ("2.jpg", "JPEG"), ("3.png", "PNG")):
        p = tmp_path / f"x_{name}"
        p.write_bytes(image_bytes(fmt))
        paths.append(str(p))
    items = career_docs.vision_items({"jd_file": paths, "jd_file_names": ["1.png", "2.jpg", "3.png"]})
    rank = vision.vision_models(models, career_ui_pool(models))
    parts = [career_docs.transcribe_file(i.path, i.name, rank, models) for i in items["jd"]]
    joined = career_docs.join_transcripts(parts)
    assert joined.index("[1.png]") < joined.index("[주요 업무]") < joined.index("[2.jpg]") < joined.index("[자격 요건]") \
        < joined.index("[3.png]") < joined.index("[우대 사항]")
    assert len(calls) == 3


def test_tall_screenshot_goes_in_one_call_as_ordered_pieces(fake_vision, models, tmp_path):
    calls, _ = fake_vision
    path = tmp_path / "long.png"
    path.write_bytes(image_bytes(size=(800, 4500)))
    out = career_docs.transcribe_file(str(path), "long.png", vision.vision_models(models, career_ui_pool(models)), models)
    assert len(calls) == 1 and calls[0]["images"] == 3 and "조각 3개" in calls[0]["prompt"] and "3조각" in out.how


def test_image_only_pdf_pages_go_through_vision(fake_vision, models, tmp_path):
    from tests.test_career import make_pdf

    calls, texts = fake_vision
    texts["2쪽"] = "스캔한 2쪽 글"
    scanned = tmp_path / "scan.pdf"
    Image.new("RGB", (300, 400), "white").save(scanned, format="PDF")  # 글자 층 없는 그림 PDF
    assert career_docs.needs_vision(str(scanned))
    out = career_docs.transcribe_file(str(scanned), "scan.pdf", vision.vision_models(models, career_ui_pool(models)), models)
    assert out.text == JD_TEXT and "글자 층 없는 1쪽" in out.how and len(calls) == 1

    # 글자 층 쪽 + 스캔 쪽이 섞인 PDF — 글자 층은 코드로, 스캔 쪽만 비전 모델로 (쪽 순서 그대로)
    import pypdfium2 as pdfium

    text_pdf = pdfium.PdfDocument(make_pdf(["Python log analysis 3 years"]))
    image_pdf = pdfium.PdfDocument(scanned.read_bytes())
    text_pdf.import_pages(image_pdf)
    mixed = tmp_path / "mixed.pdf"
    with open(mixed, "wb") as f:
        text_pdf.save(f)
    out = career_docs.transcribe_file(str(mixed), "mixed.pdf", vision.vision_models(models, career_ui_pool(models)), models)
    assert out.text.index("Python log analysis") < out.text.index("스캔한 2쪽 글") and len(calls) == 2

    text_only = tmp_path / "text.pdf"
    text_only.write_bytes(make_pdf(["Python log analysis"]))
    assert not career_docs.needs_vision(str(text_only))  # 글자 층 PDF 는 원래대로 코드가 읽는다 (확인 단계 없음)
    assert career_docs.vision_items({"jd_file": [str(text_only)], "docs": [str(text_only)]}) == {}


def test_no_vision_model_raises_friendly_error(set_keys, models, tmp_path):
    set_keys()  # 키 없음
    import career_ui

    pool = career_ui_pool(models)
    assert vision.usable_vision_models(models, pool) == []
    assert "API 키 없음" in career_ui.no_model_reason(pool, models)
    path = tmp_path / "a.png"
    path.write_bytes(image_bytes())
    with pytest.raises(llm_factory.NoAvailableModelError):
        career_docs.transcribe_file(str(path), "a.png", vision.vision_models(models, pool), models)


# ─── 확인한 글 = JD·사실의 출처 ─────────────────────────────────────────────


def test_confirmed_text_is_what_the_tool_and_validators_see(tmp_path):
    jd_png = tmp_path / "jd.png"
    jd_png.write_bytes(image_bytes())
    doc_png = tmp_path / "ab12cd34_경력.png"
    doc_png.write_bytes(image_bytes())
    raw_doc = "로그 분류 시간을 180분에서 46분으로 줄였다."  # 비전 모델이 잘못 읽은 글 (45 → 46)
    fixed_doc = "로그 분류 시간을 180분에서 45분으로 줄였다."  # 사람이 원본과 대조해 고친 글
    inputs = {"jd_file": [str(jd_png)], "jd_file_names": ["jd.png"], "docs": [str(doc_png)], "docs_names": ["경력.png"],
              "questions": "1. 경험 (700자)", "career_kind": "cover_letter"}

    # 확인 전: 그림은 읽지 않고 확인 단계가 필요하다고만 알린다 (지어낸 근거가 들어가지 않게)
    before = career_docs.run("질문", inputs, None)
    assert before.harness_options["career_source"] == "" and sum("이 내용으로 진행" in n for n in before.notes) == 2

    import career_ui

    items = career_docs.vision_items(inputs)
    assert set(items) == {"jd", "docs"}
    request = {"question": "q", "tool_inputs": inputs}
    texts = {"jd": JD_TEXT.replace("3년", "5년"), "docs": [fixed_doc]}
    confirmed = career_ui.apply_confirmed(request, items, texts, "그림은 비전 모델(gemini-3.5-flash-lite)이 옮기고 사람이 확인")
    assert confirmed["tool_inputs"]["docs_confirmed"] == {str(doc_png): fixed_doc}
    assert career_docs.vision_items(confirmed["tool_inputs"]) == {}  # 확인한 뒤에는 다시 묻지 않는다

    res = career_docs.run("질문", confirmed["tool_inputs"], None)
    opts = res.harness_options
    assert opts["career_source"] == fixed_doc and raw_doc not in res.content
    assert "5년 이상" in opts["career_reference"] and "3년 이상" not in opts["career_reference"]
    assert "| 경력.png | 그림(확인한 글) |" in res.content and "🖼 그림은 비전 모델" in res.content
    assert opts["career_requirements"]  # 확인한 JD 에서 요구사항을 뽑았다

    evidence = '\n\n```json\n{"evidence": [{"claim": "단축", "quote": "로그 분류 시간을 180분에서 45분으로 줄였다."}], "jd_coverage": [], "needs_check": []}\n```'
    good = "## 문항 1. 경험\n분류 시간을 180분에서 45분으로 줄였고, 5년 이상 경험 요건에 맞춰 설명합니다." + evidence
    bad = "## 문항 1. 경험\n분류 시간을 180분에서 46분으로 줄였습니다." + evidence
    options = {k: opts[k] for k in ("career_source", "career_reference")}
    assert REGISTRY["career_facts"].check(good, options).passed
    result = REGISTRY["career_facts"].check(bad, options)
    assert not result.passed and "46분" in result.log  # 잘못 읽은 숫자는 확인한 글에 없어 걸린다
    assert career.check_facts(good, opts["career_source"], opts["career_reference"]).ok


# ─── 화면: 확인 단계 ──────────────────────────────────────────────────────


# AppTest.from_function 은 이 함수 본문을 임시 파일로 써서 실행한다 — Windows 기본 인코딩(cp949)으로 쓰고 utf-8 로 읽어
# 한글이 있으면 조용히 아무것도 안 그린다 → 본문에는 ASCII 만. 처음 한 번만 '맡기기'를 누른 것처럼 request 를 넘긴다.
def confirm_page(root: str, request: dict, models_cfg, chains) -> None:
    import sys as _sys

    if root not in _sys.path:
        _sys.path.insert(0, root)
    import streamlit as st

    import career_ui

    first = not st.session_state.get("_sent")
    st.session_state["_sent"] = True
    out = career_ui.confirm_step(request if first else {}, "career", "cover_letter", models_cfg, chains)
    if out:
        st.session_state["result"] = out


@pytest.mark.parametrize("fmt,suffix", [("PNG", ".png"), ("JPEG", ".jpg")])
def test_confirm_box_shows_transcript_model_cost_and_returns_edited_text(fake_vision, models, tmp_path, fmt, suffix):
    from streamlit.testing.v1 import AppTest

    calls, _ = fake_vision
    path = tmp_path / f"ab12cd34_공고{suffix}"
    path.write_bytes(image_bytes(fmt))
    request = {"question": "q", "tool_inputs": {"jd_file": [str(path)], "jd_file_names": [f"공고{suffix}"], "questions": "1. x (700자)"}}
    at = AppTest.from_function(confirm_page, kwargs={"root": str(ROOT), "request": request, "models_cfg": models,
                                                     "chains": career_chains(models)}, default_timeout=30).run()
    assert not at.exception, at.exception
    assert len(calls) == 1 and "result" not in at.session_state  # 확인 전에는 업무를 시작하지 않는다
    box = next(t for t in at.text_area if t.label.startswith("JD 글"))
    assert box.value == JD_TEXT
    info = " ".join(i.value for i in at.info)
    assert "gemini-3.5-flash-lite" in info and "추정 비용 $0.0008" in info  # 어느 모델이 읽었는지 · 비용
    assert any("고른 모델의 회사" in c.value for c in at.caption)  # 그림이 외부로 전송된다는 안내
    assert any("흐림" in c.value for c in at.caption)

    box.set_value(JD_TEXT.replace("3년", "4년"))
    at = at.button[[b.label for b in at.button].index("✅ 이 내용으로 진행")].click().run()
    assert not at.exception, at.exception
    result = at.session_state["result"]
    assert result["tool_inputs"]["jd_confirmed"] == JD_TEXT.replace("3년", "4년") and result["question"] == "q"
    assert "gemini-3.5-flash-lite" in result["tool_inputs"]["vision_summary"]
    assert len(calls) == 1  # 진행할 때 다시 읽지 않는다
    res = career_docs.run("q", result["tool_inputs"], None)
    assert "4년 이상" in res.harness_options["career_reference"]


def test_confirm_box_without_vision_model_shows_korean_message(set_keys, models, tmp_path):
    from streamlit.testing.v1 import AppTest

    set_keys()
    path = tmp_path / "jd.png"
    path.write_bytes(image_bytes())
    request = {"question": "q", "tool_inputs": {"jd_file": [str(path)]}}
    at = AppTest.from_function(confirm_page, kwargs={"root": str(ROOT), "request": request, "models_cfg": models,
                                                     "chains": career_chains(models)}, default_timeout=30).run()
    assert not at.exception, at.exception
    assert any(vision.NO_VISION_MODEL in e.value for e in at.error)
    assert "result" not in at.session_state
    at = at.button[[b.label for b in at.button].index("✖ 취소")].click().run()
    assert "career_vision_career_cover_letter" not in at.session_state  # 취소 → 붙잡아 둔 업무를 버린다
    assert not at.run().error


def test_text_only_request_passes_straight_through(models, tmp_path):
    import career_ui

    md = tmp_path / "jd.md"
    md.write_text(JD_TEXT, encoding="utf-8")
    request = {"question": "q", "tool_inputs": {"jd_file": [str(md)]}}
    assert career_ui.confirm_step(request, "career", "cover_letter", models, []) is request


def test_team_yaml_accepts_images_for_jd_and_docs():
    team = load_team("career")
    for task in team.tasks:
        fields = {f.key: f for f in task.fields}
        for key in ("jd_file", "docs"):
            assert {"png", "jpg", "jpeg", "webp", "pdf", "docx"} <= set(fields[key].options)
            assert fields[key].multiple
        assert "전송" in fields["jd_file"].help and "확인" in fields["docs"].help

