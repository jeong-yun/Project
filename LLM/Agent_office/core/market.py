"""📺 보드·🏠 메인 위쪽의 지수·환율 띠 — 코스피 · 코스닥 · S&P 500 · 나스닥 · 다우 · 원/달러.

숫자는 LLM 이 아니라 yfinance(Yahoo Finance, 키 불필요)로 가져온다 — 지어내지 않게.
  - 종목: settings.yaml board.market (없으면 DEFAULT_MARKET). P_test 뉴스 브리핑(news_ui.py)의 지수 표와 같은 6개
  - 값: 최근 10일 일봉의 마지막 종가와 전날 대비 등락률. 장이 닫혔거나 주말·휴일이면 마지막 종가와 그 날짜가 그대로 나온다
  - 캐시: data/market/quotes.json — board.market_cache_min(기본 10분) 안이면 Yahoo 에 다시 묻지 않는다
    (보드는 10분마다 스스로 새로 고치고, 화면은 수시로 다시 그려지므로)
  - 새로 받기는 백그라운드 스레드. 화면은 최대 board.market_wait_sec(기본 4초)만 기다리고, 늦으면 있던 값으로 그린다
  - 한 종목이 실패하면 그 칸만 이전 값, 이전 값도 없으면 '—'. 띠가 비어도 보드는 그대로 돈다
"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_MARKET = [
    {"name": "코스피", "ticker": "^KS11"},
    {"name": "코스닥", "ticker": "^KQ11"},
    {"name": "S&P 500", "ticker": "^GSPC"},
    {"name": "나스닥", "ticker": "^IXIC"},
    {"name": "다우", "ticker": "^DJI"},
    {"name": "원/달러", "ticker": "KRW=X"},
]
CACHE_MIN, WAIT_SEC = 10, 4
_lock = threading.Lock()
_jobs: dict[str, threading.Thread] = {}  # 캐시 파일 → 새로 받는 중인 스레드


def market_settings(settings: dict) -> dict:
    cfg = (settings or {}).get("board") or {}
    items = cfg.get("market")
    return {
        "items": [m for m in (DEFAULT_MARKET if items is None else items) if m.get("name") and m.get("ticker")],
        "cache_min": float(cfg.get("market_cache_min", CACHE_MIN)),
        "wait_sec": float(cfg.get("market_wait_sec", WAIT_SEC)),
    }


# ─── 시세 한 종목 (테스트에서 fetch_history 를 가짜로 바꾼다) ─────────────────


def fetch_history(ticker: str, timeout: float = 10):
    """최근 10일 일봉 (pandas DataFrame). 휴일이 길어도 직전 거래일 두 개가 들어오게 10일."""
    import yfinance as yf

    return yf.Ticker(ticker).history(period="10d", timeout=timeout)


def blank(name: str, ticker: str) -> dict:
    return {"name": name, "ticker": ticker, "price": None, "change_pct": None, "date": ""}


def parse_quote(name: str, ticker: str, history) -> dict | None:
    """일봉 → 마지막 종가·전날 대비 %·그 날짜. 값이 없으면 None."""
    try:
        closes = history["Close"].dropna()
    except (KeyError, TypeError, AttributeError):
        return None
    if closes.empty:
        return None
    last = float(closes.iloc[-1])
    prev = float(closes.iloc[-2]) if len(closes) > 1 else None
    change = round((last / prev - 1) * 100, 2) if prev else None
    return {"name": name, "ticker": ticker, "price": round(last, 2), "change_pct": change,
            "date": str(closes.index[-1].date())}


def fetch_quote(name: str, ticker: str, timeout: float = 10) -> dict | None:
    """실패하면 None (네트워크·티커 오류는 그 칸만 '—')."""
    try:
        return parse_quote(name, ticker, fetch_history(ticker, timeout))
    except Exception:  # noqa: BLE001 — yfinance 는 네트워크·형식 오류를 여러 예외로 낸다
        return None


# ─── 캐시 + 백그라운드 새로 받기 ──────────────────────────────────────────


def cache_path(data_dir: Path) -> Path:
    return Path(data_dir) / "market" / "quotes.json"


def load_cache(data_dir: Path) -> dict:
    try:
        return json.loads(cache_path(data_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def refresh(items: list[dict], data_dir: Path, timeout: float = 10, now: float | None = None) -> dict:
    """6종목을 한꺼번에 받아 캐시에 쓴다. 실패한 종목은 이전 값, 이전 값도 없으면 빈 칸('—')."""
    previous = {q["ticker"]: q for q in load_cache(data_dir).get("quotes") or []}
    with ThreadPoolExecutor(max_workers=max(1, len(items))) as pool:
        got = list(pool.map(lambda m: fetch_quote(m["name"], m["ticker"], timeout), items))
    quotes = [q or {**previous.get(m["ticker"], blank(m["name"], m["ticker"])), "name": m["name"]}
              for m, q in zip(items, got)]
    cache = {"fetched_at": now or time.time(), "quotes": quotes}
    path = cache_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return cache


def is_fresh(cache: dict, items: list[dict], cache_min: float, now: float | None = None) -> bool:
    """캐시가 cache_min 안에 받은 것이고 종목 목록도 설정과 같으면 True."""
    if not cache.get("fetched_at"):
        return False
    if [q.get("ticker") for q in cache.get("quotes") or []] != [m["ticker"] for m in items]:
        return False
    return (now or time.time()) - cache["fetched_at"] < cache_min * 60


def _start(items: list[dict], data_dir: Path, timeout: float) -> threading.Thread:
    """이미 받는 중이면 그 스레드를, 아니면 새 스레드를 돌려준다 (같은 폴더에 동시에 두 번 묻지 않게)."""
    key = str(cache_path(data_dir))
    with _lock:
        job = _jobs.get(key)
        if job and job.is_alive():
            return job

        def work():
            try:
                refresh(items, data_dir, timeout)
            except OSError:  # 캐시를 못 써도 화면은 이전 값('—')으로 그린다
                pass
            finally:
                with _lock:
                    _jobs.pop(key, None)

        job = threading.Thread(target=work, name="market-refresh", daemon=True)
        _jobs[key] = job
        job.start()
        return job


def pending(data_dir: Path) -> bool:
    """아직 받는 중이면 True — 보드는 이때 10분을 기다리지 않고 1분 뒤에 새로 고쳐 숫자를 채운다
    (앱을 켠 뒤 첫 시세는 yfinance 불러오기까지 수십 초 걸릴 수 있다)."""
    with _lock:
        job = _jobs.get(str(cache_path(data_dir)))
        return bool(job and job.is_alive())


def quotes(settings: dict, data_dir: Path, wait: float | None = None) -> list[dict]:
    """설정 순서대로 종목마다 {name, ticker, price, change_pct, date}. 값이 없으면 price=None ('—').
    캐시가 오래됐으면 백그라운드로 새로 받고 최대 wait 초(None 이면 설정값) 기다린다. 0 이면 기다리지 않는다."""
    cfg = market_settings(settings)
    items = cfg["items"]
    if not items:
        return []
    cache = load_cache(data_dir)
    if not is_fresh(cache, items, cfg["cache_min"]):
        wait = cfg["wait_sec"] if wait is None else wait
        job = _start(items, data_dir, timeout=max(wait, 5) * 2)
        if wait > 0:
            job.join(wait)
            cache = load_cache(data_dir)
    have = {q.get("ticker"): q for q in cache.get("quotes") or []}
    return [{**have.get(m["ticker"], blank(m["name"], m["ticker"])), "name": m["name"]} for m in items]


# ─── 화면용 글자 ───────────────────────────────────────────────────────


def view(q: dict) -> dict:
    """보드·메인 띠에 그대로 쓰는 글자: 값 '2,612.45' · 등락 '+0.52%' · 방향 up/down/flat · 날짜 '10/24'."""
    price, change = q.get("price"), q.get("change_pct")
    date = q.get("date") or ""
    return {
        "name": q.get("name", ""),
        "value": f"{price:,.2f}" if price is not None else "—",
        "change": f"{change:+.2f}%" if change is not None else "",
        "dir": "flat" if not change else "up" if change > 0 else "down",
        "date": f"{int(date[5:7])}/{int(date[8:10])}" if len(date) >= 10 else "",
        "title": f"{q.get('ticker', '')} · {date} 기준 (yfinance)" if date else f"{q.get('ticker', '')} · 시세를 받지 못함",
    }
