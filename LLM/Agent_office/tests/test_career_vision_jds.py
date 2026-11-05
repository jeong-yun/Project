"""👔 그림 JD — 한 번만 읽기(캐시) · 확인 단계에서 JD 묶음 바꾸기.

2026-11-05 문의 ① '그림으로 JD 를 넣으면 한 번만 읽지 않고 계속 읽는다' ② '긴 JD 는 캡처 여러 장, 짧은 JD 여러 개는 캡처 한 장'.
그림은 코드로 만든 가짜(가상 채용 공고 글자), 비전 모델은 가짜 (tests/test_career_images.py 의 fake_vision).
"""

import pytest

import career_ui
from core import llm_factory
from tests.test_career_images import career_chains, fake_vision, image_bytes  # noqa: F401 — fixture



def two_images(tmp_path) -> list:
    out = []
    for i in (1, 2):
        path = tmp_path / f"ab12cd3{i}_공고{i}.png"
        path.write_bytes(image_bytes(text=f"가상 채용 공고 {i}"))
        out.append(career_ui.career_docs.VisionItem("jds", str(path), f"공고{i}.png"))
    return out


# ─── B2: JD 묶음 바꾸기 ──────────────────────────────────────────────────────

JD_X = "[가 회사 보안 분석가 (가명)]\n[자격 요건]\n- 로그 분석 경험"
JD_Y = "[나 회사 데이터 엔지니어 (가명)]\n[자격 요건]\n- 파이프라인 운영 경험"
JD_Z = "[다 회사 클라우드 보안 (가명)]\n[자격 요건]\n- IAM 정책 관리 경험"


def test_read_jds_merges_same_group_and_splits_on_dash_lines():
    from core.tools import career_docs

    inputs = {"career_kind": "jd_recommend", "jd": JD_Z, "jd_file": ["a.png", "b.png", "c.png"],
              "jd_file_names": ["공고1.png", "공고2.png", "공고3.png"],
              "jds_confirmed": {"a.png": JD_X, "b.png": "- 우대: 침해사고 대응", "c.png": JD_Y + "\n---\n" + JD_Z},
              "jds_group": {"a.png": "G1", "b.png": "G1", "c.png": "G3"}}
    pairs, _ = career_docs.read_jds(inputs)
    assert [n for n, _ in pairs] == ["붙여 넣은 JD", "공고1.png + 공고2.png", "공고3.png (1/2)", "공고3.png (2/2)"]
    assert pairs[1][1] == JD_X + "\n\n- 우대: 침해사고 대응"  # 캡처 두 장 → JD 하나 (올린 순서)
    preview = career_docs.jd_list_preview(inputs)
    assert "| J2 | 가 회사 보안 분석가 (가명) | 공고1.png + 공고2.png |" in preview and "| J4 |" in preview
    no_groups = {k: v for k, v in inputs.items() if k != "jds_group"}
    assert len(career_docs.read_jds(no_groups)[0]) == 5  # 묶음을 고르지 않으면 지금처럼 파일 하나 = JD 하나 (+ '---' 나누기)


def test_confirm_box_lets_user_regroup_jds_and_shows_final_list(fake_vision, models, tmp_path):
    from streamlit.testing.v1 import AppTest

    from core.tools import career_docs
    from tests.test_career_images import ROOT

    calls, texts = fake_vision
    texts.update({"공고1": JD_X, "공고2": "- 우대: 침해사고 대응", "공고3": JD_Y + "\n---\n" + JD_Z})
    paths = []
    for i in (1, 2, 3):
        path = tmp_path / f"ab12cd3{i}_공고{i}.png"
        path.write_bytes(image_bytes(text=f"가상 채용 공고 {i}"))
        paths.append(str(path))
    request = {"question": "q", "tool_inputs": {"career_kind": "jd_recommend", "jd_file": paths,
                                                "jd_file_names": ["공고1.png", "공고2.png", "공고3.png"]}}
    at = AppTest.from_function(confirm_page, kwargs={"root": str(ROOT), "request": request, "models_cfg": models,
                                                     "chains": career_chains(models)}, default_timeout=30).run()
    assert not at.exception, at.exception
    table = next(m.value for m in at.markdown if "최종 JD 목록" in m.value)
    assert "| J1 | 가 회사 보안 분석가 (가명) | 공고1.png |" in table and "| J4 |" in table  # 처음엔 파일마다 + '---' 나누기
    at.selectbox(key=next(s.key for s in at.selectbox if s.label.startswith("공고2.png"))).set_value(1)
    at = at.run()
    table = next(m.value for m in at.markdown if "최종 JD 목록" in m.value)
    assert "| J1 | 가 회사 보안 분석가 (가명) | 공고1.png + 공고2.png |" in table
    assert "| J2 | 나 회사 데이터 엔지니어 (가명) | 공고3.png (1/2) |" in table and "| J3 | 다 회사" in table and "| J4 |" not in table
    at = at.button[[b.label for b in at.button].index("✅ 이 내용으로 진행")].click().run()
    assert not at.exception, at.exception
    res = career_docs.run("q", at.session_state["result"]["tool_inputs"], None)
    jds = res.harness_options["career_jds"]
    assert [(j["id"], j["name"]) for j in jds] == [("J1", "공고1.png + 공고2.png"), ("J2", "공고3.png (1/2)"), ("J3", "공고3.png (2/2)")]
    assert len(calls) == 3


# ─── B1: 한 번만 읽기 ────────────────────────────────────────────────────────


def test_interrupted_reading_resumes_without_rereading_finished_images(fake_vision, models, tmp_path):
    calls, _ = fake_vision
    items = {"jds": two_images(tmp_path)}
    pool = career_ui.reader_pool(career_chains(models))
    spec = career_ui.vision.usable_vision_models(models, pool)[0]
    real = llm_factory.invoke_structured

    class Interrupted(Exception):
        """Streamlit 이 화면을 다시 그리려고 실행을 끊은 것과 같은 상황."""

    def stop_on_second(*args, **kwargs):
        if len(calls) == 1:
            calls.append({"interrupted": True})
            raise Interrupted
        return real(*args, **kwargs)

    cache: dict = {}
    llm_factory.invoke_structured = stop_on_second
    try:
        with pytest.raises(Interrupted):
            career_ui.read_items(items, spec, pool, models, cache=cache)
    finally:
        llm_factory.invoke_structured = real
    assert len(cache) == 1  # 첫 그림은 끝나는 즉시 남는다
    seen = []
    reads = career_ui.read_items(items, spec, pool, models, cache=cache, progress=lambda i, n, name: seen.append((i, n, name)))
    assert [c.get("interrupted") for c in calls] == [None, True, None]  # 다시 실행해도 첫 그림은 읽지 않는다 (두 번째만)
    assert [f.name for f in reads["jds"]] == ["공고1.png", "공고2.png"] and seen == [(1, 2, "공고1.png"), (2, 2, "공고2.png")]
    career_ui.read_items(items, spec, pool, models, cache=cache)
    assert len(calls) == 3  # 같은 그림 · 같은 모델은 다시 부르지 않는다
    other = next((s for s in career_ui.vision.usable_vision_models(models, pool) if s.label != spec.label), None)
    if other is not None:
        career_ui.read_items(items, other, pool, models, cache=cache)
        assert len(calls) == 5  # 다른 모델을 고르면 새로 읽는다


def confirm_page(root: str, request: dict, models_cfg, chains) -> None:
    import sys as _sys

    if root not in _sys.path:
        _sys.path.insert(0, root)
    import streamlit as st

    import career_ui as _cu

    first = not st.session_state.get("_sent")
    st.session_state["_sent"] = True
    out = _cu.confirm_step(request if first else {}, "career", "jd_recommend", models_cfg, chains)
    if out:
        st.session_state["result"] = out


def test_confirm_box_reads_each_image_once_across_reruns_and_reread_reads_again(fake_vision, models, tmp_path):
    from streamlit.testing.v1 import AppTest

    from tests.test_career_images import ROOT

    calls, _ = fake_vision
    paths = [i.path for i in two_images(tmp_path)]
    request = {"question": "q", "tool_inputs": {"career_kind": "jd_recommend", "jd_file": paths, "jd_file_names": ["공고1.png", "공고2.png"]}}
    at = AppTest.from_function(confirm_page, kwargs={"root": str(ROOT), "request": request, "models_cfg": models,
                                                     "chains": career_chains(models)}, default_timeout=30).run()
    assert not at.exception, at.exception
    assert len(calls) == 2
    for _ in range(3):  # 글을 고치거나 다른 것을 눌러 화면이 다시 그려져도
        at = at.run()
    assert len(calls) == 2
    at = at.button[[b.label for b in at.button].index("🔁 고른 모델로 다시 읽기")].click().run()
    assert not at.exception, at.exception
    assert len(calls) == 4  # '다시 읽기'만 새로 읽는다
