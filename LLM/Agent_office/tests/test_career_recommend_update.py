"""👔 포트폴리오·자소서 팀 — 🎯 업무(JD) 추천 · 🔄 포트폴리오 최신화.

  JD 추천      : 모델은 요구사항마다 근거 있음/부분/공백만 판정 → 점수·순위는 코드가 계산, 순위가 다르면 하네스가 다시 쓰게 한다
  포트폴리오 최신화 : 바뀐 점(이전 → 이후 · 이유 · 출처) 검사, 외부 검색은 사람이 승인한 '일반 낱말' 검색어만 (개인 정보가 든 검색어는 막는다)

가짜 모델·가짜 검색 엔진만 쓴다 (네트워크·비용 없음). 인물·회사는 모두 가상이다.
"""

import json
from pathlib import Path

import pytest

from core import career, career_search, search
from core.config import load_team, today
from core.harness import REGISTRY
from core.tasks import build_request
from core.tools import career_docs

ROOT = Path(__file__).resolve().parent.parent

SRC = "보안 로그 분석 자동화를 Python 으로 만들었다.\n보안관제팀에서 3년 근무했다.\n로그 수집 파이프라인을 개선했다."
JD_A = "[가 회사 보안 분석가]\n[자격 요건]\n- Python 로그 분석 경험\n- 보안 관제 경험\n[우대 사항]\n- 침해사고 대응 경험"
JD_B = "[나 회사 데이터 엔지니어]\n[자격 요건]\n- Spark 대용량 처리 경험\n[주요 업무]\n- 로그 수집 파이프라인 운영\n[우대 사항]\n- Kafka 경험"
Q1 = "보안 로그 분석 자동화를 Python 으로 만들었다."
Q2 = "보안관제팀에서 3년 근무했다."
Q3 = "로그 수집 파이프라인을 개선했다."


def check(name: str, answer: str, **options):
    return REGISTRY[name].check(answer, options)


def jds_option() -> list[dict]:
    return [j.as_option() for j in career.job_posts([("a.md", JD_A), ("b.md", JD_B)])]


def recommend_answer(ranking=("J1", "J2"), best="J1", j1_r2=("근거 있음", Q2), body="## 한눈에 보기\nJ1 이 더 맞습니다.") -> str:
    fits = [
        {"jd": "J1", "title": "가 회사", "summary": "보안 분석 경험이 바로 이어지는 자리",
         "coverage": [{"id": "J1-R1", "status": "근거 있음", "quote": Q1},
                      {"id": "J1-R2", "status": j1_r2[0], "quote": j1_r2[1]},
                      {"id": "J1-R3", "status": "공백", "quote": ""}],
         "strengths": ["Python 로그 분석"], "gaps": ["침해사고 대응"], "fill_first": ["침해사고 대응 사례 정리"]},
        {"jd": "J2", "title": "나 회사", "summary": "데이터 엔지니어링 전환 자리",
         "coverage": [{"id": "J2-R1", "status": "공백", "quote": ""},
                      {"id": "J2-R2", "status": "부분", "quote": Q3},
                      {"id": "J2-R3", "status": "공백", "quote": ""}],
         "strengths": ["파이프라인 개선"], "gaps": ["Spark", "Kafka"], "fill_first": ["Spark 작은 프로젝트"]},
    ]
    data = {"jd_fit": fits, "ranking": list(ranking), "best": {"jd": best, "why": "필수 요건 둘 다 근거 있음"},
            "evidence": [{"claim": "자동화", "quote": Q1}], "needs_check": []}
    return body + "\n\n```json\n" + json.dumps(data, ensure_ascii=False) + "\n```"


# ─── 🎯 JD 추천 — 점수는 코드가 ─────────────────────────────────────────────


def test_split_pasted_jds_and_number_requirements_per_jd():
    assert career.split_jds(JD_A + "\n---\n" + JD_B + "\n\n=====\n") == [JD_A, JD_B]
    assert career.split_jds(JD_A) == [JD_A]
    jds = jds_option()
    assert [j["id"] for j in jds] == ["J1", "J2"] and jds[0]["title"] == "가 회사 보안 분석가"
    assert [(r["id"], r["section"]) for r in jds[0]["reqs"]] == [("J1-R1", "자격"), ("J1-R2", "자격"), ("J1-R3", "우대")]
    assert [(r["id"], r["section"]) for r in jds[1]["reqs"]] == [("J2-R1", "자격"), ("J2-R2", "업무"), ("J2-R3", "우대")]


def test_score_is_computed_by_code_with_required_weight():
    scores = {s.id: s for s in career.score_jds(recommend_answer(), jds_option(), SRC)}
    # J1: 필수 2개 근거 있음(1×2 + 1×2) + 우대 공백(0×1) = 4 / 5 → 80
    assert scores["J1"].score == 80.0 and (scores["J1"].earned, scores["J1"].total) == (4.0, 5.0)
    # J2: 필수 공백(0×2) + 업무 부분(0.5×1) + 우대 공백(0×1) = 0.5 / 4 → 12.5
    assert scores["J2"].score == 12.5
    assert "필수 2개(근거 2·부분 0·공백 0) ×2" in scores["J1"].basis and "→ 4/5 × 100 = 80" in scores["J1"].basis
    assert [s.id for s in career.ranked(list(scores.values()))] == ["J1", "J2"]
    assert career.ranking_text(list(scores.values())) == "J1 80점 > J2 12.5점"


def test_unverified_quote_counts_zero_so_generous_labels_cannot_raise_the_score():
    answer = recommend_answer(j1_r2=("근거 있음", "보안관제팀에서 10년 근무했다."))  # 문서에 없는 인용
    j1 = career.score_jds(answer, jds_option(), SRC)[0]
    assert j1.score == 40.0 and j1.unverified == ["J1-R2"] and "인용 불일치로 0 처리: J1-R2" in j1.basis
    facts = REGISTRY["career_facts"].check(answer, {"career_source": SRC, "career_reference": JD_A})
    assert not facts.passed and "10년" in facts.log  # career_facts 도 jd_fit 인용을 글자 그대로 대조한다
    assert career.score_jds(answer, jds_option())[0].score == 80.0  # 원문 없이 부르면 판정 그대로 (화면 복원용)


def test_jd_recommend_validator_requires_every_jd_and_code_ranking():
    opts = {"career_jds": jds_option(), "career_source": SRC}
    ok = check("jd_recommend", recommend_answer(), **opts)
    assert ok.passed and "J1 80점 > J2 12.5점" in ok.log, ok.log

    wrong = check("jd_recommend", recommend_answer(ranking=("J2", "J1"), best="J2"), **opts)
    assert not wrong.passed
    assert "코드 계산: J1 80점 > J2 12.5점" in wrong.log and "best J2 는 코드 계산 1위가 아니다" in wrong.log

    data = json.loads(recommend_answer().split("```json\n")[1].split("\n```")[0])
    data["jd_fit"] = data["jd_fit"][:1]
    data["jd_fit"][0]["coverage"] = data["jd_fit"][0]["coverage"][:2]
    data["jd_fit"][0]["summary"] = ""
    missing = check("jd_recommend", "본문\n```json\n" + json.dumps(data, ensure_ascii=False) + "\n```", **opts)
    assert not missing.passed
    assert all(s in missing.log for s in ("판정하지 않은 JD: J2", "J1 에서 판정하지 않은 요구사항: J1-R3", "J1 한 줄 요약(summary)이 없다"))
    assert not check("jd_recommend", "JSON 없음", **opts).passed
    assert check("jd_recommend", "아무 글", career_jds=[]).passed  # JD 가 없으면 검사하지 않는다


def test_ties_may_be_listed_in_either_order():
    tie = recommend_answer(j1_r2=("공백", ""))  # J1: 2/5 = 40 · J2: 12.5 → 순서 J1 > J2 그대로
    assert check("jd_recommend", tie, career_jds=jds_option(), career_source=SRC).passed
    same = [{"id": "J1", "title": "x", "reqs": []}, {"id": "J2", "title": "y", "reqs": []}]
    answer = recommend_answer(ranking=("J2", "J1"), best="J2")
    assert career.check_recommend(answer, same, SRC) == []  # 둘 다 점수 없음 → 순서 자유


def test_recommend_document_puts_code_scores_first():
    md = career.render_document(recommend_answer(), "jd_recommend", title="JD 추천", jds=jds_option(), source=SRC)
    assert md.index("## 추천 순위 (점수는 코드가 계산)") < md.index("## 설명 (작성: 팀)")
    assert "| 1 | J1. 가 회사 보안 분석가 | 보안 분석 경험이 바로 이어지는 자리 | **80점** |" in md
    assert "**가장 맞는 JD: J1. 가 회사 보안 분석가** (80점) — 필수 요건 둘 다 근거 있음" in md
    assert "| J2-R2 | 업무 | 로그 수집 파이프라인 운영 | 부분 | 로그 수집 파이프라인을 개선했다. |" in md
    assert "**먼저 채울 것**\n- Spark 작은 프로젝트" in md
    bad = career.render_document(recommend_answer(ranking=("J2", "J1"), best="J2"), "jd_recommend", jds=jds_option(), source=SRC)
    assert "모델이 고른 추천(J2)이 코드 계산 1위(J1)와 다릅니다" in bad


def test_tool_reads_many_jds_from_paste_files_and_sample(tmp_path):
    jd_file = tmp_path / "ab12cd34_공고C.md"
    jd_file.write_text("[다 회사 AI 거버넌스]\n[자격 요건]\n- ISMS-P 인증 대응 경험", encoding="utf-8")
    inputs = {"career_kind": "jd_recommend", "profile_text": SRC, "jd": JD_A + "\n---\n" + JD_B,
              "jd_file": [str(jd_file)], "jd_file_names": ["공고C.md"], "direction": "AI 거버넌스"}
    res = career_docs.run("q", inputs, None)
    jds = res.harness_options["career_jds"]
    assert [(j["id"], j["name"]) for j in jds] == [("J1", "붙여 넣은 JD 1"), ("J2", "붙여 넣은 JD 2"), ("J3", "공고C.md")]
    assert "JD 3개" in res.title and "| 붙여 넣은 요약 | 글 |" in res.content
    assert "### 🎯 J3. 다 회사 AI 거버넌스 (공고C.md) — 요구사항 1개" in res.content and "J3-R1" in res.content
    assert "점수 계산 규칙" in res.content and res.harness_options["career_requirements"] == []  # jd_coverage 는 이 업무에서 쉰다
    assert res.harness_options["career_source"] == SRC and "AI 거버넌스" in res.harness_options["career_reference"]
    sample = career_docs.run("q", {"career_kind": "jd_recommend", "use_sample": True}, None)
    titles = [j["title"] for j in sample.harness_options["career_jds"]]
    assert len(titles) == 3 and all("가상 회사" in t for t in titles) and "김샘플" in sample.harness_options["career_source"]


def test_image_jds_are_confirmed_one_file_at_a_time(tmp_path):
    import career_ui
    from tests.test_career_images import image_bytes

    a, b, c = tmp_path / "a.png", tmp_path / "b.md", tmp_path / "c.png"
    a.write_bytes(image_bytes())
    b.write_text(JD_B, encoding="utf-8")
    c.write_bytes(image_bytes())
    inputs = {"career_kind": "jd_recommend", "profile_text": SRC, "jd_file": [str(a), str(b), str(c)], "jd_file_names": ["a.png", "b.md", "c.png"]}
    items = career_docs.vision_items(inputs)
    assert list(items) == ["jds"] and [i.name for i in items["jds"]] == ["a.png", "c.png"]  # 글 파일 b.md 는 코드가 읽는다
    before = career_docs.run("q", inputs, None)
    assert [j["name"] for j in before.harness_options["career_jds"]] == ["b.md"]  # 확인 전 그림은 쓰지 않는다
    confirmed = career_ui.apply_confirmed({"question": "q", "tool_inputs": inputs}, items, {"jds": [JD_A, "[다 회사]\n[자격 요건]\n- SQL 경험"]}, "확인함")
    assert career_docs.vision_items(confirmed["tool_inputs"]) == {}
    after = career_docs.run("q", confirmed["tool_inputs"], None)
    assert [j["name"] for j in after.harness_options["career_jds"]] == ["a.png", "b.md", "c.png"]  # 올린 순서 그대로
    assert after.harness_options["career_jds"][0]["title"] == "가 회사 보안 분석가"


# ─── 🔄 포트폴리오 최신화 — 바뀐 점 · 변경 목록 ───────────────────────────────

OLD = (ROOT / "teams" / "career" / "samples" / "sample_portfolio_old.md").read_text(encoding="utf-8")
NEW = (ROOT / "teams" / "career" / "samples" / "sample_profile.md").read_text(encoding="utf-8")
URL = "https://example.com/trends"
CARD = "- **문제**: {p}\n- **역할**: {r}\n- **기술**: {t}\n- **결과**: {o}\n- **배운 점**: {l}"
UPDATED = ("# 포트폴리오 — 보안 관리자\n## 요약\n가상 인물 김샘플. 보안 로그 분석과 클라우드 로그 수집을 다룹니다.\n"
           "## 핵심 역량\n| 영역 | 기술·도구 | 근거 |\n|---|---|---|\n| 로그 분석 | Python, pandas, Elasticsearch | 프로젝트 1 |\n"
           "| 클라우드 | AWS CloudTrail, S3, Airflow | 프로젝트 2 |\n## 프로젝트\n### 1. 보안 로그 분석 자동화 (2022.04 ~ 2022.12)\n"
           + CARD.format(p="관제 요원이 매일 아침 방화벽·웹 로그를 수작업으로 분류했다.", r="4명 팀에서 분석 스크립트 설계와 개발을 맡았다.",
                         t="Python, pandas, Elasticsearch", o="일일 분류 시간을 180분에서 45분으로 줄였다.", l="규칙을 코드로 남긴다.")
           + "\n### 2. 클라우드 로그 수집 파이프라인 개선 (2025.02 ~ 2025.06)\n"
           + CARD.format(p="클라우드 로그 수집 누락.", r="[확인 필요]", t="AWS CloudTrail, S3, Python, Airflow",
                         o="수집 누락을 하루 평균 2.3%에서 0.4%로 줄였다.", l="수집 단계부터 데이터 품질 지표를 둔다 [W1].")
           + "\n## 직무 강점\n- 로그 분석 자동화 (프로젝트 1)")


def update_answer(changes: list[dict], body: str = UPDATED) -> str:
    data = {"changes": changes, "evidence": [{"claim": "새 프로젝트", "quote": "수집 누락을 하루 평균 2.3%에서 0.4%로 줄였다."}],
            "jd_coverage": [], "needs_check": ["프로젝트 2 역할"]}
    return body + "\n\n```json\n" + json.dumps(data, ensure_ascii=False) + "\n```"


GOOD_CHANGES = [
    {"part": "프로젝트 > 클라우드 로그 수집", "before": "", "after": "수집 누락을 하루 평균 2.3%에서 0.4%로 줄였다.",
     "why": "최신 문서의 새 프로젝트", "source": "수집 누락을 하루 평균 2.3%에서 0.4%로 줄였다.", "kind": "문서"},
    {"part": "직무 강점", "before": "반복 업무 자동화 (프로젝트 1)", "after": "로그 분석 자동화 (프로젝트 1)",
     "why": "직무 이름에 맞춤", "source": "일일 분류 시간을 180분에서 45분으로 줄였다.", "kind": "문서"},
    {"part": "프로젝트 2 배운 점", "before": "", "after": "수집 단계부터 데이터 품질 지표를 둔다",
     "why": "요즘 공고에서 쓰는 데이터 품질 용어", "source": URL, "kind": "외부"},
]


def update_options(web_count: int = 1) -> dict:
    return {"career_old_portfolio": OLD, "career_source": OLD + "\n\n" + NEW, "career_web_urls": [URL], "career_web_count": web_count,
            "career_reference": "보안 관리자"}


def test_diff_candidates_find_new_projects_numbers_and_terms():
    diff = career.diff_candidates(OLD, NEW)
    assert any("클라우드 로그 수집 파이프라인 개선" in p for p in diff["projects"])
    assert not any("보안 로그 분석 자동화" in p for p in diff["projects"])  # 이미 포트폴리오에 있다
    assert any("2.3%에서 0.4%" in n for n in diff["numbers"]) and not any("180분에서 45분" in n for n in diff["numbers"])
    assert {"Airflow", "CloudTrail", "ISMS-P"} <= set(diff["terms"]) and "Python" not in diff["terms"]


def test_portfolio_changes_validator():
    opts = update_options()
    ok = check("portfolio_changes", update_answer(GOOD_CHANGES), **opts)
    assert ok.passed and "3건 (문서 2 · 외부 1)" in ok.log, ok.log
    assert check("career_facts", update_answer(GOOD_CHANGES), career_source=opts["career_source"]).passed
    assert check("portfolio_format", update_answer(GOOD_CHANGES)).passed

    bad = [dict(GOOD_CHANGES[0], before="예전에 없던 문장"),
           dict(GOOD_CHANGES[1], source="일일 분류 시간을 180분에서 30분으로 줄였다."),
           dict(GOOD_CHANGES[2], source="https://not-approved.example"),
           dict(GOOD_CHANGES[2], after="업계 평균 오탐을 37% 줄였다", source=URL)]
    res = check("portfolio_changes", update_answer(bad, UPDATED.replace("[W1]", "[W3]")), **opts)
    assert not res.passed
    for part in ("변경 1(프로젝트 > 클라우드 로그 수집): before 가 지금 포트폴리오에 글자 그대로 없다",
                 "변경 2(직무 강점): source 인용이 내 문서에 글자 그대로 없다",
                 "변경 3(프로젝트 2 배운 점): source 는 승인한 외부 자료의 URL",
                 "변경 4(프로젝트 2 배운 점): after 에 내 문서에 없는 숫자 37%",
                 "[W3] 은 외부 자료 목록에 없다"):
        assert part in res.log, res.log
    none = check("portfolio_changes", "본문\n```json\n{\"evidence\": []}\n```", **opts)
    assert not none.passed and "changes(바뀐 점 목록)가 없다" in none.log
    assert check("portfolio_changes", "본문\n```json\n{\"no_changes\": \"최신 문서와 같음\"}\n```", **opts).passed
    no_web = check("portfolio_changes", update_answer(GOOD_CHANGES), **{**opts, "career_web_urls": [], "career_web_count": 0})
    assert not no_web.passed and "외부 자료가 없으므로" in no_web.log


def test_update_document_has_change_table_and_sources():
    web = [{"title": "데이터 품질 트렌드", "url": URL, "snippet": "요약"}]
    md = career.render_document(update_answer(GOOD_CHANGES), "portfolio_update", title="포트폴리오(최신화)", web=web)
    assert md.index("## 프로젝트") < md.index("# 부록 — 검토용") < md.index("## 바뀐 점 (이전 → 이후)")
    assert "| 1 | 프로젝트 > 클라우드 로그 수집 | (새로 추가) | 수집 누락을" in md
    assert f"| 🌐 {URL} |" in md and f"- [W1] 데이터 품질 트렌드 — {URL}" in md


def test_tool_update_reads_old_portfolio_latest_docs_and_approved_web(tmp_path):
    old = tmp_path / "ab12cd34_예전.md"
    old.write_text(OLD, encoding="utf-8")
    new = tmp_path / "ab12cd34_최신.md"
    new.write_text(NEW, encoding="utf-8")
    inputs = {"career_kind": "portfolio_update", "portfolio": str(old), "portfolio_name": "예전.md", "docs": [str(new)],
              "docs_names": ["최신.md"], "role": "보안 관리자", "web_on": True, "web_approved": True,
              "web_queries": ["보안 관리자 채용 트렌드 2026"], "web_results": [{"title": "트렌드", "url": URL, "snippet": "요약"}],
              "fetched": [{"url": "https://github.com/o/r", "raw": "https://raw.githubusercontent.com/o/r/HEAD/README.md",
                           "kind": "github", "use": "ref", "text": "# 공개 프로젝트\n- RAG 데모"}]}
    res = career_docs.run("q", inputs, None)
    opts = res.harness_options
    assert "포트폴리오 1개 · 문서 1개 · 외부 자료 1건" in res.title
    assert "### 📁 지금 포트폴리오 — 예전.md" in res.content and "[W1] 트렌드 (https://example.com/trends)" in res.content
    assert "사람이 승인한 검색어: 보안 관리자 채용 트렌드 2026" in res.content and "클라우드 로그 수집 파이프라인 개선" in res.content
    assert "### 🌐 GitHub 글 (외부 자료" in res.content and "RAG 데모" not in opts["career_source"]  # 외부 자료는 사실 출처가 아니다
    assert opts["career_old_portfolio"] == OLD.strip() and opts["career_web_count"] == 1
    assert URL in opts["career_web_urls"] and "https://github.com/o/r" in opts["career_web_urls"]
    assert "180분에서 45분" in opts["career_source"] and "2.3%에서 0.4%" in opts["career_source"]
    mine = career_docs.run("q", {**inputs, "github_as_source": True}, None)
    assert "RAG 데모" in mine.harness_options["career_source"] and "| GitHub: https://github.com/o/r | md |" in mine.content
    sample = career_docs.run("q", {"career_kind": "portfolio_update", "use_sample": True}, None)
    assert "2023년 판" in sample.harness_options["career_old_portfolio"] and sample.harness_options["career_web_urls"] == []


# ─── 외부 검색어 — 개인 정보가 나가지 않게 ─────────────────────────────────────


def test_query_check_blocks_names_companies_projects_numbers_and_quotes():
    year = today().year
    role = "보안 데이터 분석가"
    for safe in (f"{role} 채용 트렌드 {year}", "security data analyst skills", "ISO 27001 NIST CSF 2.0 최신 동향",
                 "보안 로그 분석 자동화 Python Elasticsearch 트렌드", "Isolation Forest 이상탐지 보안 적용 사례"):
        assert career_search.query_problems(safe, NEW, role) == [], safe
    blocked = {
        "가나다시큐리티 보안관제팀 채용": "가나다시큐리티",  # 회사·부서 이름
        "김샘플 포트폴리오": "김샘플",  # 사람 이름
        "이상 접속 탐지 모델 시범 적용": "시범",  # 프로젝트 이름의 낱말
        "SIEM 로그 분석 180분 단축": "숫자 180",  # 문서의 숫자
        f"보안 트렌드 {year - 5}": f"숫자 {year - 5}",  # 올해 ±1 이 아닌 연도
        "\"일일 분류 시간\"": "따옴표",
        "kim@example.com 보안": "이메일",
        "https://github.com/o/r 보안": "주소",
    }
    for query, reason in blocked.items():
        problems = career_search.query_problems(query, NEW, role)
        assert problems and any(reason in p for p in problems), (query, problems)


def test_draft_queries_use_only_role_and_generic_terms(monkeypatch, models):
    seen = []

    def fake_plan(question, models, max_queries, hint=""):
        seen.append(question)
        return search.SearchPlan(need_search=True, queries=["보안 관리자 채용 트렌드", "가나다시큐리티 채용 공고"]), []

    monkeypatch.setattr(search, "plan_queries", fake_plan)
    queries, notes = career_search.draft_queries("보안 관리자", NEW, models, {"web_search": {"max_queries": 3}})
    assert queries == ["보안 관리자 채용 트렌드"] and any("개인 정보 검사에 걸려" in n for n in notes)
    assert "김샘플" not in seen[0] and "가나다" not in seen[0] and "180" not in seen[0]  # 모델에 문서 원문을 넘기지 않는다
    assert "Python" in seen[0] and "보안 관리자" in seen[0]

    def broken(*args, **kwargs):
        raise RuntimeError("모델 없음")

    monkeypatch.setattr(search, "plan_queries", broken)
    queries, notes = career_search.draft_queries("보안 관리자", NEW, models, {"web_search": {"max_queries": 3}})
    assert len(queries) == 3 and all(not career_search.query_problems(q, NEW, "보안 관리자") for q in queries)
    assert queries[0] == f"보안 관리자 채용 요구 역량 {today().year}" and "코드가 만든 검색어" in notes[0]

    def no_model(question, models, max_queries, hint=""):  # 실제 plan_queries 의 '모델 없음' 모양 — 질문 글을 검색어로 돌려준다
        return search.SearchPlan(need_search=True, queries=[question[:200]]), ["검색어 생성 실패 → 질문으로 검색: 키 없음"]

    monkeypatch.setattr(search, "plan_queries", no_model)
    queries, notes = career_search.draft_queries("보안 관리자", NEW, models, {"web_search": {"max_queries": 3}})
    assert queries[0] == f"보안 관리자 채용 요구 역량 {today().year}"
    assert notes == ["보조 모델을 쓸 수 없어 코드가 만든 검색어를 씁니다 (직무 이름 + 일반 기술 용어 + 연도)."]


def test_search_queries_sends_only_the_given_queries(no_network_search):
    context = search.search_queries(["보안 관리자 채용 트렌드"], {"web_search": {"max_total_results": 5}})
    assert no_network_search == ["보안 관리자 채용 트렌드"] and context.queries == ["보안 관리자 채용 트렌드"]
    assert [r.url for r in context.results][:1] == ["https://example.com/보안 관리자 채용 트렌드"]
    assert search.search_queries([" "], {}).results == [] and no_network_search == ["보안 관리자 채용 트렌드"]


def test_github_address_rules():
    # 주소 규칙 전체는 tests/test_career_web_read.py — 여기는 처음부터 있던 세 모양과 '네트워크에 나가기 전에 막는다'만
    def raw(url):
        return career_search.target(url).fetch

    assert raw("https://github.com/o/r") == "https://raw.githubusercontent.com/o/r/HEAD/README.md"
    assert raw("https://github.com/o/r/blob/main/docs/PROJECTS.md") == "https://raw.githubusercontent.com/o/r/main/docs/PROJECTS.md"
    assert raw("https://raw.githubusercontent.com/o/r/main/README.md") == "https://raw.githubusercontent.com/o/r/main/README.md"
    for bad in ("http://evil.example/x.md", "https://github.com/o", "https://github.com/o/r/blob/main/../x.md",
                "https://user@github.com/o/r", "https://github.com/o/r/blob/main/logo.png"):
        with pytest.raises(ValueError):  # FetchProblem — 네트워크에 나가기 전에 막는다
            career_search.target(bad)


# ─── 팀 설정 · 가짜 모델로 끝까지 ─────────────────────────────────────────────


def test_team_yaml_defines_both_tasks_without_web_search_tool():
    team = load_team("career")
    assert team.private and "web_search" not in team.tools
    tasks = {t.id: t for t in team.tasks}
    rec, upd = tasks["jd_recommend"], tasks["portfolio_update"]
    assert rec.validators == ["jd_recommend"] and upd.validators == ["portfolio_format", "portfolio_changes", "career_style"]
    assert {"jd_recommend", "portfolio_changes"} <= set(team.harness.final_only)
    assert team.harness.also_at["jd_recommend"] == ["am_writer", "gm_role"]
    web = next(f for f in upd.fields if f.key == "web_on")
    assert web.type == "checkbox" and web.default is False  # 기본 꺼짐
    assert next(f for f in upd.fields if f.key == "portfolio").multiple is True  # 여러 판 (Word판·md판)
    assert any("JD" in x and "추천" in x for x in team.scope.accepts) and any("최신화" in x for x in team.scope.accepts)
    assert 2 <= len(team.free_examples) <= 4
    req = build_request(rec, {"use_sample": True, "direction": "AI 거버넌스"})
    assert req.tool_inputs["career_kind"] == "jd_recommend" and req.validators == ["jd_recommend"]
    for name in ("sample_jd_b.md", "sample_jd_c.md", "sample_portfolio_old.md"):
        assert "가상" in (ROOT / "teams" / "career" / "samples" / name).read_text(encoding="utf-8")


FACT = ("## 사실 카드\n| # | 경험 | 원문 인용 |\n|---|---|---|\n| 1 | 자동화 | " + Q1 + " |\n\n```json\n"
        + json.dumps({"evidence": [{"claim": "자동화", "quote": Q1}], "jd_coverage": [], "needs_check": []}, ensure_ascii=False) + "\n```")


def run_fake(task_id: str, values: dict, reviews: list, ranks, models, use_script, tool_inputs: dict | None = None):
    from core.config import team_org
    from core.workflow import build_office_graph, stream_office
    from tests.fakes import Script

    team = load_team("career")
    script = use_script(Script(text=FACT, ReviewResult=reviews))
    req = build_request(next(t for t in team.tasks if t.id == task_id), values)
    org = team_org(team, ranks)
    graph = build_office_graph(org, ranks, team, models, {}, {"critic": {"enabled": False}})
    events = list(stream_office(graph, org, req.question, "lead", brief=req.brief, auto_brief=False,
                                tool_inputs={**req.tool_inputs, **(tool_inputs or {})}, extra_validators=req.validators,
                                harness_options=req.harness_options))
    return events, script


def test_fake_model_jd_recommend_is_sent_back_when_ranking_disagrees_with_code(ranks, models, use_script):
    from tests.fakes import make_review

    bad = recommend_answer(ranking=("J2", "J1"), best="J2")
    good = recommend_answer()
    reviews = [make_review("revise", 8, bad), make_review("revise", 8, good), make_review("approve", 9), make_review("approve", 9)]
    events, script = run_fake("jd_recommend", {"profile_text": SRC, "jd": JD_A + "\n---\n" + JD_B}, reviews, ranks, models, use_script)
    tools = next(p for k, p in events if k == "tools")
    assert [j["id"] for j in tools[0].harness_options["career_jds"]] == ["J1", "J2"]
    feedback = next(p for p in script.prompts("ReviewResult") if "자동 검증(하네스)" in p)
    assert "코드 계산: J1 80점 > J2 12.5점" in feedback  # 코드가 계산한 순위가 다시 쓰라는 피드백으로
    steps = [p for k, p in events if k == "step"]
    staff = next(s for s in steps if s.node_id == "staff_facts")
    assert "jd_recommend" not in {h.validator for h in staff.harness}  # 사원은 사실 카드만
    final = next(s for s in steps if s.node_id == "lead")
    assert {"career_facts", "jd_recommend"} <= {h.validator for h in final.harness}
    assert all(h.passed for h in final.harness), [h.log for h in final.harness if not h.passed]

    import career_ui

    run = {"run_id": "r1", "tools": tools, "career_inputs": {"career_kind": "jd_recommend", "fmt_md": "예", "fmt_docx": True}}
    out = career_ui.build(run, final.content)
    assert set(out["files"]) == {"JD 추천.md", "JD 추천.docx"} and out["facts"].ok
    assert [(s.id, s.score) for s in out["ranking"]] == [("J1", 80.0), ("J2", 12.5)]
    assert "## 추천 순위 (점수는 코드가 계산)" in out["markdown"]
    assert career_ui.build({"run_id": "x", "question": "q"}, final.content)["kind"] == "jd_recommend"  # 되살린 업무도 종류를 안다


def test_fake_model_portfolio_update_with_approved_web_source(ranks, models, use_script):
    from tests.fakes import make_review

    bad = update_answer([dict(GOOD_CHANGES[2], source="https://not-approved.example")])
    good = update_answer(GOOD_CHANGES)
    reviews = [make_review("revise", 8, bad), make_review("revise", 8, good), make_review("approve", 9), make_review("approve", 9)]
    web = {"web_on": True, "web_approved": True, "web_queries": ["보안 관리자 채용 트렌드"],
           "web_results": [{"title": "데이터 품질 트렌드", "url": URL, "snippet": "요약"}]}
    events, script = run_fake("portfolio_update", {"use_sample": True, "role": "보안 관리자", "fmt_docx": True, "fmt_md": True},
                              reviews, ranks, models, use_script, tool_inputs=web)
    feedback = next(p for p in script.prompts("ReviewResult") if "자동 검증(하네스)" in p)
    assert "승인한 외부 자료의 URL" in feedback
    final = next(s for k, s in events if k == "step" and s.node_id == "lead")
    assert {"career_facts", "portfolio_format", "portfolio_changes"} <= {h.validator for h in final.harness}
    assert all(h.passed for h in final.harness), [h.log for h in final.harness if not h.passed]

    import career_ui

    tools = next(p for k, p in events if k == "tools")
    run = {"run_id": "r2", "tools": tools, "career_inputs": {"career_kind": "portfolio_update", "fmt_md": True, "fmt_docx": True, **web}}
    out = career_ui.build(run, final.content)
    assert set(out["files"]) == {"포트폴리오(최신화).md", "포트폴리오(최신화).docx"}
    assert len(out["changes"]) == 3 and out["change_problems"] == [] and out["facts"].ok
    assert f"- [W1] 데이터 품질 트렌드 — {URL}" in out["markdown"]


# ─── 화면 (AppTest) ──────────────────────────────────────────────────────


@pytest.fixture
def app(tmp_path, monkeypatch, set_keys):
    from streamlit.testing.v1 import AppTest

    set_keys()
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.query_params["team"] = "career"
    at.query_params["view"] = "team"
    return at


def test_career_page_shows_both_new_forms(app):
    at = app.run()
    assert not at.exception, at.exception
    picker = at.selectbox(key="dept_task_career")  # 🏢 부서 업무 목록 (팀별로 묶임)
    assert [o.split(" › ")[1] for o in picker.options[1:5]] == ["✍️ 자소서 작성", "📁 포트폴리오 작성", "🎯 업무(JD) 추천",
                                                                 "🔄 포트폴리오 최신화"]

    at = picker.set_value("career:jd_recommend").run()
    assert not at.exception, at.exception
    assert any("코드가 판정으로 계산" in c.value for c in at.caption)
    assert {t.label for t in at.text_area} >= {"또는 요약 문서 붙여 넣기", "JD 붙여 넣기 (여러 개는 '---' 만 있는 줄로 나눔)"}
    assert len(at.get("file_uploader")) == 2

    at = at.selectbox(key="dept_task_career").set_value("career:portfolio_update").run()
    assert not at.exception, at.exception
    web = next(c for c in at.checkbox if c.label.startswith("🌐 외부 자료로"))
    assert web.value is False  # 기본 꺼짐
    assert any(t.label.startswith("참고 주소 — GitHub·웹") for t in at.text_area)
    assert len(at.get("file_uploader")) == 4  # 지금 포트폴리오 + 형식 참고 파일 + 최신 요약 문서 + JD 파일


# AppTest.from_function 은 함수 본문을 임시 파일로 써서 실행한다 — Windows 기본 인코딩 문제로 본문에는 ASCII 만 쓴다.
def web_page(root: str, request: dict, models_cfg) -> None:
    import sys as _sys

    if root not in _sys.path:
        _sys.path.insert(0, root)
    import streamlit as st

    import career_ui

    first = not st.session_state.get("_sent")
    st.session_state["_sent"] = True
    out = career_ui.confirm_step(request if first else {}, "career", "portfolio_update", models_cfg, [])
    if out:
        st.session_state["result"] = out


def update_request(**extra) -> dict:
    task = next(t for t in load_team("career").tasks if t.id == "portfolio_update")
    req = build_request(task, {"use_sample": True, "role": "보안 관리자", **extra})
    return {"question": req.question, "tool_inputs": req.tool_inputs}


def test_web_box_shows_queries_and_searches_only_after_approval(monkeypatch, models, no_network_search):
    from streamlit.testing.v1 import AppTest

    def fake_plan(question, models, max_queries, hint=""):
        return search.SearchPlan(need_search=True, queries=["보안 관리자 채용 트렌드", "가나다시큐리티 채용 공고"]), []

    monkeypatch.setattr(search, "plan_queries", fake_plan)
    at = AppTest.from_function(web_page, kwargs={"root": str(ROOT), "request": update_request(web_on=True), "models_cfg": models},
                               default_timeout=30).run()
    assert not at.exception, at.exception
    assert no_network_search == [] and "result" not in at.session_state  # 승인 전에는 아무것도 보내지 않는다
    assert any("검색어는 검색 엔진(DuckDuckGo)으로 나갑니다" in c.value and "승인하세요" in c.value for c in at.caption)
    box = next(t for t in at.text_area if t.label.startswith("보낼 검색어"))
    assert box.value == "보안 관리자 채용 트렌드"  # 개인 정보가 든 모델 검색어는 초안에서 빠진다
    assert any("개인 정보 검사에 걸려" in c.value for c in at.caption)

    box.set_value("보안 관리자 채용 트렌드\n김샘플 보안 관리자 포트폴리오\nSIEM 최신 동향")
    at = at.run()
    shown = " ".join(m.value for m in at.markdown)
    assert "⛔ `김샘플 보안 관리자 포트폴리오`" in shown and "→ **보내지 않음**" in shown and "✅ `SIEM 최신 동향`" in shown
    assert no_network_search == []

    at = at.button[[b.label for b in at.button].index("🔍 승인 — 검색하고 진행")].click().run()
    assert not at.exception, at.exception
    assert sorted(no_network_search) == ["SIEM 최신 동향", "보안 관리자 채용 트렌드"]  # 막힌 검색어는 보내지 않았다
    inputs = at.session_state["result"]["tool_inputs"]
    assert inputs["web_approved"] and inputs["web_queries"] == ["보안 관리자 채용 트렌드", "SIEM 최신 동향"]
    assert any("1개는 보내지 않았습니다" in n for n in inputs["web_notes"])
    res = career_docs.run("q", inputs, None)
    assert "[W1]" in res.content and res.harness_options["career_web_count"] == len(inputs["web_results"]) > 0


def test_web_box_skip_and_off_by_default(models, no_network_search, set_keys):
    from streamlit.testing.v1 import AppTest

    import career_ui

    set_keys()  # 키 없음 → 검색어 초안은 보조 모델 없이 코드가 만든다 (이 테스트는 '외부 자료 없이 진행'만 본다)
    plain = update_request()
    assert career_ui.confirm_step(plain, "career", "portfolio_update", models, []) is plain  # 끄면 상자 없이 바로 시작
    at = AppTest.from_function(web_page, kwargs={"root": str(ROOT), "request": update_request(web_on=True), "models_cfg": models},
                               default_timeout=30).run()
    at = at.button[[b.label for b in at.button].index("🚫 외부 자료 없이 진행")].click().run()
    assert not at.exception, at.exception
    inputs = at.session_state["result"]["tool_inputs"]
    assert inputs["web_approved"] and inputs["web_results"] == [] and no_network_search == []


def test_web_box_rejects_address_it_cannot_read(models, no_network_search):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(web_page, kwargs={"root": str(ROOT), "request": update_request(ref_urls="http://evil.example/x.md"),
                                                 "models_cfg": models}, default_timeout=30).run()
    assert not at.exception, at.exception
    assert any("⛔ `http://evil.example/x.md`" in m.value and "https 주소만 읽습니다" in m.value for m in at.markdown)
    assert "▶ 진행" in [b.label for b in at.button]


def preview_page(root: str, run: dict, text: str) -> None:
    import sys as _sys

    if root not in _sys.path:
        _sys.path.insert(0, root)
    import career_ui

    career_ui.preview(run, text)


def test_preview_shows_code_ranking_and_change_list(tmp_path):
    from streamlit.testing.v1 import AppTest

    rec = career_docs.run("q", {"career_kind": "jd_recommend", "profile_text": SRC, "jd": JD_A + "\n---\n" + JD_B}, None)
    run = {"run_id": "abcd1234", "tools": [rec], "career_inputs": {"career_kind": "jd_recommend", "fmt_md": "예"}}
    at = AppTest.from_function(preview_page, kwargs={"root": str(ROOT), "run": run, "text": recommend_answer()}, default_timeout=30).run()
    assert not at.exception, at.exception
    shown = " ".join(m.value for m in at.markdown)
    assert "추천 순위 (점수는 코드가 판정으로 계산" in shown and "| 1 | J1. 가 회사 보안 분석가 | **80점** |" in shown
    assert any("가장 맞는 JD: **J1. 가 회사 보안 분석가** — 80점" in s.value for s in at.success)

    upd = career_docs.run("q", {"career_kind": "portfolio_update", "use_sample": True, "web_results": [{"title": "t", "url": URL}]}, None)
    run = {"run_id": "efgh5678", "tools": [upd], "career_inputs": {"career_kind": "portfolio_update", "fmt_md": True}}
    at = AppTest.from_function(preview_page, kwargs={"root": str(ROOT), "run": run, "text": update_answer(GOOD_CHANGES)},
                               default_timeout=30).run()
    assert not at.exception, at.exception
    shown = " ".join(m.value for m in at.markdown)
    assert "바뀐 점 3건** (내 문서 2 · 외부 자료 1)" in shown and "🌐 **프로젝트 2 배운 점**" in shown
    assert not at.error  # 변경 목록 검사 통과
