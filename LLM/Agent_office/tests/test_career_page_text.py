"""웹 페이지 → 글 (core.career_search.page_text): 메뉴·바닥글·광고·쿠키 안내 줄은 빼고 공고 본문은 남긴다.

표본 HTML(tests/fixtures/job_pages)은 실제 사이트를 복사하지 않고 흔한 모양을 흉내 내 직접 만들었다.
목록은 '- ' 줄로 남아야 한다 — JD 요구사항을 코드가 그 줄에서 뽑는다.
"""

from pathlib import Path

import pytest

from core import career_search as cs

PAGES = Path(__file__).parent / "fixtures" / "job_pages"
TERMS = "근무 조건: 정규직, 수습 3개월, 주 5일 근무이며 연봉은 경력과 면접 결과에 따라 협의합니다. 지원서는 채용 담당자가 직접 읽고 일주일 안에 결과를 알려 드립니다."  # 모든 표본에 있는 본문 문단

# 표본 → (남아야 하는 본문 줄, 빠져야 하는 잡음 글). 목록 항목은 '- ' 로 시작하는 줄 그대로.
EXPECTED = {
    "semantic.html": (
        [TERMS, "보안 관제 엔지니어 (경력 3년 이상)", "예시회사 보안관제센터에서 함께 일할 엔지니어를 찾습니다.", "주요 업무",
         "- SIEM 이벤트 분석과 대응", "- 탐지 룰 작성과 튜닝", "자격 요건", "- 보안 관제 실무 경력 3년 이상",
         "- 네트워크·리눅스 기본 지식", "- Python 으로 반복 업무 자동화 경험", "우대 사항", "- 정보보안기사 자격증",
         "- 침해사고 대응 보고서 작성 경험"],
        ["쿠키를 사용합니다", "모두 동의", "예시채용 홈", "채용 공고 찾기", "기업 정보", "로그인", "이력서 무료 첨삭",
         "다른 공고", "클라우드 보안 담당자", "개인정보보호 담당자", "이용약관", "모든 권리 보유", "추적 코드"],
    ),
    "div_soup.html": (
        [TERMS, "데이터 엔지니어 모집 (신입·경력)", "예시회사 데이터플랫폼팀이 데이터 엔지니어를 모집합니다.", "지원 자격",
         "- SQL 과 Python 사용 경험", "- 데이터 파이프라인 구축 또는 운영 경험", "우대 사항", "- Airflow 사용 경험",
         "- 클라우드 데이터 웨어하우스 운영 경험"],
        ["채용정보", "인재풀", "고객센터", "전체 공고", "IT·개발", "디자인", "프리미엄 공고", "최근 본 공고", "맨 위로",
         "광고문의", "사업자등록번호"],
    ),
    "article_header.html": (
        [TERMS, "백엔드 개발자 (Go)", "예시회사 · 서울 · 정규직", "이런 일을 합니다", "- 결제 API 설계와 개발",
         "- 서비스 장애 원인 분석", "이런 분을 찾습니다", "- Go 또는 Java 로 서버 개발 3년 이상",
         "- 관계형 데이터베이스 설계 경험", "마감일: 채용 시 마감"],
        ["- 홈", "- 개발", "공유하기", "링크 복사", "비슷한 공고", "프론트엔드 개발자", "DevOps 엔지니어", "고객센터"],
    ),
    "role_attrs.html": (
        [TERMS, "클라우드 보안 담당자", "예시회사의 클라우드 보안을 맡을 분을 찾습니다.", "필수 요건", "- AWS 보안 설정 실무 경험",
         "- IAM 권한 설계 경험", "우대 요건", "- ISMS-P 인증 대응 경험"],
        ["예시잡", "검색", "커뮤니티", "마이페이지", "스크랩한 공고", "많이 본 공고", "이용약관"],
    ),
    "sloppy.html": (
        [TERMS, "정보보호 담당자 채용", "예시회사가 정보보호 담당자를 찾습니다.", "근무지: 판교", "담당 업무",
         "- 정보보호 관리체계 운영", "- 보안 정책·지침 작성과 교육", "자격 요건", "- 정보보호 업무 경력 2년 이상",
         "- 보안 관련 법령 이해"],
        ["- 채용", "- 기업", "로그인", "추천 공고", "보안 컨설턴트", "모의해킹 담당자", "쿠키 사용", "개인정보처리방침",
         "고객센터"],
    ),
}


def score(text: str, body: list[str], noise: list[str]) -> tuple[int, int]:
    """(남은 본문 줄 수, 남은 잡음 수). 본문은 줄이 정확히 같아야, 잡음은 글 어디에든 있으면 남은 것으로 센다."""
    lines = set(text.splitlines())
    return sum(b in lines for b in body), sum(n in text for n in noise)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_page_text_keeps_body_and_drops_menus_footers(name):
    body, noise = EXPECTED[name]
    text = cs.page_text((PAGES / name).read_text(encoding="utf-8"))
    lines = set(text.splitlines())
    assert [b for b in body if b not in lines] == [], text
    assert [n for n in noise if n in text] == [], text


def test_page_text_falls_back_when_filter_leaves_too_little():
    # 본문이 통째로 메뉴처럼 보이는 상자 안에 있으면, 거른 결과 대신 전체 글을 돌려준다 (못 읽는 것보다 잡음 섞인 편이 낫다)
    html = ("<html><body><div class='menu-layout'><h1>품질 담당자</h1><ul>"
            + "".join(f"<li>품질 관리 업무 {i}번 항목을 맡습니다</li>" for i in range(20)) + "</ul></div></body></html>")
    text = cs.page_text(html)
    assert "품질 담당자" in text and "- 품질 관리 업무 0번 항목을 맡습니다" in text
