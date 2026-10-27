"""📈 지수·환율 띠 (core/market.py) — 일봉 읽기, 실패 → '—', 캐시 기간, 화면 글자. 네트워크 없이 가짜 일봉으로."""

import json
import threading
import time

import pandas as pd
import pytest

from core import market

ITEMS = [{"name": "코스피", "ticker": "^KS11"}, {"name": "원/달러", "ticker": "KRW=X"}]
SETTINGS = {"board": {"market": ITEMS, "market_cache_min": 10, "market_wait_sec": 5}}


def history(*closes, last_day="2026-10-24"):
    """yfinance Ticker.history 와 같은 모양 (날짜 인덱스 + Close 열). 마지막 줄이 last_day."""
    days = pd.date_range(end=last_day, periods=len(closes), freq="D", tz="Asia/Seoul")
    return pd.DataFrame({"Close": list(closes)}, index=days)


@pytest.fixture
def fake_yahoo(monkeypatch):
    """티커 → 일봉(또는 예외). 불린 티커는 calls 에 쌓인다."""
    table: dict = {}
    calls: list[str] = []

    def fetch(ticker, timeout=10):
        calls.append(ticker)
        value = table[ticker]
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(market, "fetch_history", fetch)
    return table, calls


def test_parse_quote_last_close_change_and_date():
    q = market.parse_quote("코스피", "^KS11", history(2500.0, 2512.5))
    assert q == {"name": "코스피", "ticker": "^KS11", "price": 2512.5, "change_pct": 0.5, "date": "2026-10-24"}
    # 휴일·주말: 빈 값(NaN)은 건너뛰고 마지막 거래일 종가와 그 날짜
    closed = history(2500.0, 2512.5, float("nan"))
    assert market.parse_quote("코스피", "^KS11", closed)["date"] == "2026-10-23"
    assert market.parse_quote("코스피", "^KS11", history(2500.0))["change_pct"] is None  # 하루치뿐이면 등락 없음
    assert market.parse_quote("코스피", "^KS11", pd.DataFrame()) is None


def test_failed_ticker_shows_dash_and_others_still_work(tmp_path, fake_yahoo):
    table, _ = fake_yahoo
    table.update({"^KS11": history(2500.0, 2475.0), "KRW=X": RuntimeError("Yahoo 응답 없음")})
    quotes = market.quotes(SETTINGS, tmp_path)
    assert [q["name"] for q in quotes] == ["코스피", "원/달러"]  # 설정 순서 그대로
    kospi, won = (market.view(q) for q in quotes)
    assert kospi == {"name": "코스피", "value": "2,475.00", "change": "-1.00%", "dir": "down", "date": "10/24",
                     "title": "^KS11 · 2026-10-24 기준 (yfinance)"}
    assert won["value"] == "—" and won["change"] == "" and won["date"] == ""


def test_failure_keeps_previous_value(tmp_path, fake_yahoo):
    table, _ = fake_yahoo
    table.update({"^KS11": history(2500.0, 2525.0), "KRW=X": history(1400.0, 1393.0)})
    market.refresh(ITEMS, tmp_path, now=time.time() - 3600)
    table["KRW=X"] = RuntimeError("일시 오류")
    cache = market.refresh(ITEMS, tmp_path)
    assert cache["quotes"][1]["price"] == 1393.0  # 이번에 실패한 칸은 지난번 값 (날짜가 함께 보여 오래된 값인 걸 안다)


def test_cache_ttl(tmp_path, fake_yahoo):
    table, calls = fake_yahoo
    table.update({"^KS11": history(1.0, 2.0), "KRW=X": history(3.0, 4.0)})
    market.quotes(SETTINGS, tmp_path)
    market.quotes(SETTINGS, tmp_path)
    assert sorted(calls) == ["KRW=X", "^KS11"]  # 10분 안에 다시 열어도 한 번만 묻는다

    cache = json.loads(market.cache_path(tmp_path).read_text(encoding="utf-8"))
    assert market.is_fresh(cache, ITEMS, 10, now=cache["fetched_at"] + 9 * 60)
    assert not market.is_fresh(cache, ITEMS, 10, now=cache["fetched_at"] + 11 * 60)
    assert not market.is_fresh(cache, ITEMS[:1], 10, now=cache["fetched_at"])  # 설정의 종목이 바뀌면 새로 받는다

    cache["fetched_at"] -= 11 * 60
    market.cache_path(tmp_path).write_text(json.dumps(cache), encoding="utf-8")
    market.quotes(SETTINGS, tmp_path)
    assert len(calls) == 4  # 기간이 지나면 다시 묻는다


def test_slow_yahoo_does_not_block_screen(tmp_path, monkeypatch):
    """화면은 wait 초만 기다리고 '—'로 그린다. 받는 중이면 pending → 보드가 1분 뒤 다시 그린다."""
    release = threading.Event()

    def slow(ticker, timeout=10):
        release.wait(5)
        return history(10.0, 11.0)

    monkeypatch.setattr(market, "fetch_history", slow)
    start = time.time()
    quotes = market.quotes({"board": {"market": ITEMS, "market_wait_sec": 0.2}}, tmp_path)
    assert time.time() - start < 2
    assert all(q["price"] is None for q in quotes) and market.pending(tmp_path)
    release.set()
    for _ in range(50):
        if not market.pending(tmp_path):
            break
        time.sleep(0.1)
    assert [q["price"] for q in market.quotes({"board": {"market": ITEMS}}, tmp_path, wait=0)] == [11.0, 11.0]


def test_settings_defaults_and_empty_list(tmp_path):
    assert [m["name"] for m in market.market_settings({})["items"]] == ["코스피", "코스닥", "S&P 500", "나스닥", "다우", "원/달러"]
    assert market.quotes({"board": {"market": []}}, tmp_path) == []  # 비우면 띠를 그리지 않는다


def test_real_settings_list_the_six_indices():
    from core.config import load_settings

    cfg = market.market_settings(load_settings())
    assert [m["ticker"] for m in cfg["items"]] == ["^KS11", "^KQ11", "^GSPC", "^IXIC", "^DJI", "KRW=X"]
    assert 10 <= cfg["cache_min"] <= 15
