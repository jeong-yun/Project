"""📺 전체 화면 보드 (board_ui.py) — 데이터 변환, 안전한 HTML, 화면 연기 테스트."""

import json
from pathlib import Path

import board_ui
from core.config import load_models, load_settings

SNAP = {
    "collected_at": 1_700_000_000,
    "briefing": [
        {"text": "기업 82%가 AI 에이전트 관리에 자신 있다고 답했지만 충돌 해결 준비는 26%였습니다.",
         "links": [{"title": "기업 82% 'AI 에이전트 관리 자신'", "url": "https://www.newstap.co.kr/a", "source": "news", "date": "2026-10-21"},
                   {"title": "같은 소식", "url": "https://b.example/x", "source": "news", "date": "2026-10-21"}],
         "hype": []},
        {"text": "근거 없는 폭등 예측 " * 20, "links": [], "hype": ["수익 주장"]},
    ],
    # 실제 data/feeds/latest.json 의 글 모양 (뉴스 날짜는 published, 저장소는 extra 에 라이선스·언어)
    "items": [
        {"source": "news", "title": "뉴스 원문", "url": "https://n.example/1", "snippet": "원문 요약",
         "published": "2026-10-22", "author": "", "score": 0, "extra": {}},
        {"source": "github", "title": "acme/agent-kit", "url": "https://github.com/acme/agent-kit", "snippet": "kit",
         "published": "2026-09-01", "author": "acme", "score": 900,
         "extra": {"license": "MIT", "pushed_at": "2026-10-21", "language": "Python", "topics": []}},
        {"source": "github", "title": "acme/mcp-box", "url": "https://github.com/acme/mcp-box", "snippet": "box",
         "published": "2026-09-02", "author": "acme", "score": 1500,
         "extra": {"license": "MIT", "pushed_at": "2026-10-20", "language": "Go", "topics": []}},
    ],
}


def test_news_cards_from_briefing():
    cards = board_ui.news_cards(SNAP)
    assert len(cards) == 2
    first, second = cards
    assert first["src"] == "newstap.co.kr" and first["count"] == "2개 출처" and first["url"] == "https://www.newstap.co.kr/a"
    assert first["date"] == "2026-10-21" and not first["hype"]
    assert second["hype"] and len(second["sum"]) <= board_ui.SUMMARY_MAX and second["sum"].endswith("…")  # 3줄까지만


def test_news_cards_fall_back_to_raw_news_without_briefing():
    cards = board_ui.news_cards({"items": SNAP["items"]})
    assert [c["title"] for c in cards] == ["뉴스 원문"] and cards[0]["cat"] == "AI 뉴스"
    assert cards[0]["date"] == "2026-10-22"  # 원문 글의 날짜는 published
    assert board_ui.news_cards(None) == []


def test_keywords_prefer_top_repos_then_news_keywords():
    assert board_ui.keywords(SNAP, {}) == ["#mcp-box", "#agent-kit"]
    assert board_ui.keywords({"items": []}, {"feeds": {"news": {"keywords": ["AI 에이전트", "LLM"]}}}) == ["#AI에이전트", "#LLM"]


def test_tips_file_loads():
    tips = board_ui.load_tips()
    assert tips and all(t["tool"] and t["cmd"] and t["url"].startswith("https://") for t in tips)


def test_board_html_escapes_script_breakout():
    evil = {**SNAP, "briefing": [{"text": "</script><script>alert(1)</script>", "links": [], "hype": []}]}
    data = board_ui.board_data(evil, load_models(), load_settings())
    page = board_ui.board_html(data)
    payload = page.split("var D = ", 1)[1].split(";\n", 1)[0]
    assert "</script>" not in payload  # 데이터 속 글자가 스크립트를 끝내지 못한다
    assert json.loads(payload.replace("<\\/", "</"))["headline"].startswith("</script>")


def test_board_data_without_snapshot():
    data = board_ui.board_data(None, load_models(), load_settings())
    assert data["news"] == [] and "지금 수집" in data["headline"] and data["collected_at"] is None
    assert {l["name"] for l in data["lights"]} <= set(board_ui.STATUS_LABELS.values())
    assert data["market"] == []


def test_rotation_setting_reaches_template():
    assert board_ui.rotation({}) == (board_ui.ROTATE_SEC, board_ui.TIP_ROTATE_SEC)
    assert board_ui.rotation({"board": {"rotate_sec": 1, "tip_rotate_sec": "x"}}) == (3, board_ui.TIP_ROTATE_SEC)  # 3초 밑·잘못된 값은 막는다
    assert board_ui.rotation(load_settings()) == (8, 20)
    data = board_ui.board_data(SNAP, load_models(), {"board": {"rotate_sec": 12}})
    page = board_ui.board_html(data)
    payload = json.loads(page.split("var D = ", 1)[1].split(";\n", 1)[0].replace("<\\/", "</"))
    assert payload["rotate_sec"] == 12 and "NEWS_MS = (D.rotate_sec || 8) * 1000" in page
    assert 'bar.style.transition = "width " + NEWS_MS' in page  # 진행 막대도 같은 간격


def test_market_cells_reach_template():
    quotes = [{"name": "코스피", "ticker": "^KS11", "price": 2612.45, "change_pct": 0.52, "date": "2026-10-24"},
              {"name": "원/달러", "ticker": "KRW=X", "price": None, "change_pct": None, "date": ""}]
    data = board_ui.board_data(SNAP, load_models(), load_settings(), quotes)
    assert [(c["value"], c["change"], c["dir"], c["date"]) for c in data["market"]] == [
        ("2,612.45", "+0.52%", "up", "10/24"), ("—", "", "flat", "")]
    html = board_ui.market_strip_html(data["market"][:1])
    assert "2,612.45" in html and "▲ +0.52%" in html and "10/24" in html


def _app(tmp_path, monkeypatch, view):
    from streamlit.testing.v1 import AppTest

    data = tmp_path / "data"
    (data / "feeds").mkdir(parents=True)
    (data / "feeds" / "latest.json").write_text(json.dumps(SNAP, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(data))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))
    settings = load_settings()
    settings.setdefault("feeds", {})["auto_collect"] = False
    monkeypatch.setattr("core.config.load_settings", lambda: settings)  # 화면을 열어도 실제 수집을 시작하지 않게
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=120)
    at.query_params["view"] = view
    return at


def test_board_page_renders_without_sidebar(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch, board_ui.BOARD_VIEW)
    at.run()
    assert not at.exception
    assert not at.sidebar.radio  # 보드는 사이드바(화면 선택) 없이
    # 시세는 못 받아도 (테스트는 네트워크 금지) 보드는 그려지고 그 칸만 '—'
    page = at.get("iframe")[0].proto.srcdoc
    assert '"name": "코스피", "value": "—"' in page and '"rotate_sec": 8' in page and '"market_pending": false' in page


def test_home_shows_headline_and_board_link(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch, "home")
    at.run()
    assert not at.exception
    assert any("오늘의 한 줄" in m.value and "에이전트 관리" in m.value for m in at.markdown)
