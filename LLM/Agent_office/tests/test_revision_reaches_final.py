"""💬 수정 요청으로 만든 수정본이 최종본까지 가는지 (2026-11-10).

문의: "결과 수정 후 최종본에 반영이 안 되는 것 같아."
확인한 것 ① 수정본을 채택하면 저장(결정 기록·결과 파일·개념 카탈로그)은 수정본으로 되는데, 화면 위쪽 '📌 최종 결과'
 (개념 편집 팀은 '📄 최종본' 탭)와 '⬇ 최종 결과 받기'는 수정 전 원래 결과로 되돌아갔다 → 고침.
확인한 것 ② 👔 부장 검토는 긴 수정본의 가운데를 잘린 채로 받았다 (개념 편집 팀은 그 가운데가 최종본) → 고침:
 <final> 최종본부터 끝까지는 통째로 넣고 그 앞(검증 로그)만 줄인다 (core/verification.py trim_keep_final).
가짜 모델만 쓴다. 화면 준비는 tests/test_progress_followup.py 의 것을 그대로 쓴다.
"""

from core import followup, verification
from tests.fakes import Script, make_critic, make_delegation, make_review
from tests.test_followup import edit, head_review, judge
from tests.test_progress_followup import _open, app_env, wait_for  # noqa: F401 — app_env 는 pytest 준비 함수


def _result_box(at) -> tuple[str, str]:
    """('📌 최종 결과' 제목 줄, 그 바로 아래 결과 글)."""
    texts = [m.value for m in at.markdown]
    i = next(i for i, t in enumerate(texts) if "📌 최종 결과" in t)
    return texts[i], texts[i + 1]


def test_adopted_revision_stays_in_final_result_box(app_env, use_script):
    gate, draft_gate = app_env
    gate.set()
    draft_gate.set()

    def delegation(messages):
        text = messages[-1].content
        return make_delegation("gm_research", "gm_action") if "gm_research" in text else make_delegation("am_fact", "staff_draft")

    use_script(Script(
        text="초안 [1]", Delegation=delegation, ReviewResult=make_review(verdict="approve", score=9, answer="보완안"),
        CriticResult=make_critic(5), FollowupResult=[edit("수정본 하나")], RevisionCritic=[judge(5)]))
    at = _open().run()
    at = at.selectbox(key="dept_task_research").set_value("general:free").run()
    at.text_area(key="q_text").input("양자컴퓨팅 요약")
    at = next(b for b in at.button if b.label == "맡기기").click().run()
    assert not at.exception, at.exception
    at = wait_for(lambda: (lambda a: a if any("💬 결과에 대해 묻기" in m.value for m in a.markdown) else None)(_open().run()), 60)
    original = _result_box(at)[1]
    assert original and original != "수정본 하나"
    at.text_area(key=next(t.key for t in at.text_area if t.placeholder and "3번 근거" in t.placeholder)).input("정의를 줄여줘")
    at = next(b for b in at.button if b.label == "💬 보내기").click().run()
    assert not at.exception, at.exception
    at = wait_for(lambda: (lambda a: a if any("수정본 v1" in m.value for m in a.markdown) else None)(_open().run()), 30)
    assert _result_box(at) == (_result_box(at)[0], "수정본 하나") and "수정본 v1" in _result_box(at)[0]  # 채택 전: 최신 수정본
    at = next(b for b in at.button if b.label == "✅ 이 안으로 채택").click().run()  # 기본 선택 = 최신 수정본
    assert not at.exception, at.exception
    at = _open().run()
    assert any("안 채택" in s.value for s in at.success)
    title, shown = _result_box(at)
    assert shown == "수정본 하나"  # 채택한 뒤에도 채택한 수정본이 보인다 (원래 결과로 되돌아가지 않는다)
    assert "채택" in title


def test_head_sees_the_changed_part_of_a_long_revision(org, team, models):
    """개념 편집 팀 수정본 모양(검증 로그 → 최종본 → 요약)이 한도(answer_chars)보다 길 때, 고친 문장이 부장 프롬프트에 있어야 한다.

    2026-11-10 실제 채택본 31,524자: 한도 24,000자라 16,000~23,524자가 빠졌고, 빠진 곳은 모두 최종본 안(최종본의 55%)이었다.
    """
    head = head_review(org, team, models)
    limit = head.cfg["answer_chars"]

    def answer(changed: str) -> str:  # 로그 ≈ 한도의 1/2 · 최종본 ≈ 1/2 (고친 문장은 그 가운데) · 요약 ≈ 1/5
        body = "본문 한 줄\n" * (limit // 28)
        return ("<review_log>\n" + "검증 로그 한 줄\n" * (limit // 20) + "</review_log>\n<final>\n" + body + changed + "\n" + body
                + "</final>\n" + "요약 한 줄\n" * (limit // 35))

    before, after = answer("세션 기반(stateful) 프로토콜이다"), answer("세션 ID 사용 여부는 서버가 정한다")
    assert len(after) > limit
    out = followup.FollowupResult(analysis="수정", revised=True, reply="고쳤습니다", changes=["2절 세션 설명"], revised_answer=after)
    prompt = followup.revision_messages(head, "MCP 정리", before, "2절의 stateful 단정을 고쳐줘", out)[-1].content
    assert "세션 ID 사용 여부는 서버가 정한다" in prompt  # 요청이 반영된 곳을 부장이 볼 수 있어야 한다
    # 최종본부터 끝까지(최종본·요약)는 수정본·수정 전 결과 모두 한 글자도 빠지지 않는다 — 줄어드는 것은 그 앞의 검증 로그
    assert after[after.index("<final>"):].strip() in prompt and before[before.index("<final>"):] in prompt
    assert prompt.count("비평 입력 한도") == 2 and prompt.count("검증 로그 한 줄") < 2 * (limit // 20)
    assert len(prompt) < len(before) + len(after)  # 통째로 넣은 것보다는 짧다


def test_trim_keep_final_only_shortens_what_comes_before_final():
    log, rest = "로그 한 줄\n" * 300, "<final>\n" + "본문 한 줄\n" * 200 + "</final>\n요약"  # 2,100자 + 1,419자
    text = log + rest
    assert verification.trim_keep_final(text, 5000) == text == verification.trim_keep_final(text, 0)  # 한도 안·한도 없음 = 그대로
    cut = verification.trim_keep_final(text, 2000)  # 최종본부터 끝까지 넣고 남은 몫만큼만 로그를 남긴다
    assert cut.endswith(rest) and cut.startswith("로그 한 줄") and "비평 입력 한도" in cut and len(cut) < 2100
    gone = verification.trim_keep_final(text, 1000)  # 최종본만으로 한도를 넘으면 로그를 통째로 빼고 최종본은 그대로
    assert gone.endswith(rest) and "로그 한 줄" not in gone and "2,100자 생략" in gone
    assert verification.trim_keep_final(rest, 1000) == rest  # 앞에 줄일 것이 없으면 그대로
    plain = "가" * 3000  # <final> 이 없는 글(다른 팀)은 전과 같이 가운데를 줄인다
    assert verification.trim_keep_final(plain, 1500) == verification.trim_middle(plain, 1500) != plain
