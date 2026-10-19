"""주가 데이터 툴 (yfinance — Yahoo Finance, API 키 불필요).

흐름
  1) 종목 찾기: 업무 입력(stock, holdings)과 질문에서 종목을 찾아 Yahoo 티커로 바꾼다.
     - 6자리 숫자 코드면 코드로 바로 (005930 → 005930.KS, 없으면 .KQ)
     - 이름이면 보조 모델이 티커를 추정 (삼성전자 → 005930.KS) → 실제 조회로 검증, 실패 시 .KS/.KQ 를 바꿔 재시도
  2) 1년 일봉을 받아 기준일 종가·기간별 수익률·52주 고저·이동평균·변동성을 '코드로' 계산
  3) 보유 수량·매입가가 있으면 평가금액·평가손익·수익률도 코드로 계산
     → LLM 은 숫자를 계산하지 않고 이 값을 인용만 한다 (계산 환각 방지)

주의: Yahoo 데이터는 지연·누락이 있을 수 있어 결과에 '기준일'과 출처를 항상 함께 적는다.
"""

import math
import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core import llm_factory
from core.config import date_label, today
from core.llm_factory import NoAvailableModelError
from core.schemas import ModelsConfig, ToolResult

CODE6 = re.compile(r"^\d{6}$")


class TickerItem(BaseModel):
    query: str = Field(description="사용자가 쓴 종목 이름 또는 코드 그대로")
    name: str = Field(description="정식 종목명")
    ticker: str = Field(
        description="Yahoo Finance 티커. 한국 코스피는 6자리코드.KS, 코스닥은 6자리코드.KQ, 미국은 심볼 (예: 005930.KS, 247540.KQ, AAPL)"
    )
    quantity: float = Field(description="보유 수량이 적혀 있으면 그 수, 없으면 0")
    buy_price: float = Field(description="평균 매입가가 적혀 있으면 그 값(숫자만), 없으면 0")


class TickerResolution(BaseModel):
    items: list[TickerItem] = Field(description="입력에 나온 종목 목록. 종목이 없으면 빈 목록")


# ─── 외부 데이터 (테스트에서 가짜로 바꿀 수 있게 함수로 분리) ───────────────────


def fetch_history(ticker: str):
    """1년 일봉 (pandas DataFrame). 종목이 없으면 빈 DataFrame."""
    import yfinance as yf

    return yf.Ticker(ticker).history(period="1y", auto_adjust=False)


def fetch_info(ticker: str) -> dict:
    """회사명·통화·밸류에이션 (실패하면 빈 dict — 없어도 진행)."""
    import yfinance as yf

    try:
        return yf.Ticker(ticker).info or {}
    except Exception:  # noqa: BLE001
        return {}


# ─── 1) 종목 찾기 ──────────────────────────────────────────────────


def resolve(question: str, inputs: dict, models: ModelsConfig) -> tuple[list[TickerItem], list[str]]:
    stock = str(inputs.get("stock", "")).strip()
    holdings = str(inputs.get("holdings", "")).strip()
    qty, buy = float(inputs.get("quantity") or 0), float(inputs.get("buy_price") or 0)

    if stock and CODE6.match(stock) and not holdings:  # 6자리 코드 → LLM 없이
        return [TickerItem(query=stock, name=stock, ticker=f"{stock}.KS", quantity=qty, buy_price=buy)], []
    if not (stock or holdings):
        return [], []  # 종목이 없는 업무 (예: 분야별 추천) → 조회 안 함

    text = "\n".join(x for x in [f"종목: {stock}" if stock else "", f"보유 종목: {holdings}" if holdings else ""] if x)
    messages = [
        SystemMessage(
            "너는 증권 데이터 담당이다. 입력에 나온 종목마다 Yahoo Finance 티커를 찾는다. "
            "한국 코스피는 '6자리코드.KS', 코스닥은 '6자리코드.KQ', 미국 주식은 심볼을 쓴다. "
            "보유 수량·평균 매입가가 적혀 있으면 숫자만 뽑고, 없으면 0 으로 둔다."
        ),
        HumanMessage(text),
    ]
    try:
        call = llm_factory.invoke_structured(models.utility, models, TickerResolution, messages, "종목 찾기")
    except NoAvailableModelError as e:
        return [], [f"종목 찾기 실패: {e}"]
    items = call.parsed.items
    if stock and len(items) == 1:  # 단일 종목 업무는 입력 칸의 수량·매입가를 우선
        items[0].quantity = qty or items[0].quantity
        items[0].buy_price = buy or items[0].buy_price
    return items, call.errors


def _load(item: TickerItem):
    """티커로 조회, 비어 있으면 .KS ↔ .KQ 를 바꿔 한 번 더."""
    history = fetch_history(item.ticker)
    if len(history) == 0 and item.ticker.endswith((".KS", ".KQ")):
        alt = item.ticker[:-3] + (".KQ" if item.ticker.endswith(".KS") else ".KS")
        alt_history = fetch_history(alt)
        if len(alt_history):
            item.ticker, history = alt, alt_history
    return history


# ─── 2·3) 지표·평가손익 계산 ─────────────────────────────────────────


def _pct(a: float, b: float) -> str:
    return f"{(a / b - 1) * 100:+.1f}%" if b else "-"


def _num(x: float, currency: str) -> str:
    return f"{x:,.0f} {currency}" if currency in ("KRW", "JPY") else f"{x:,.2f} {currency}"


def summarize(item: TickerItem, history, info: dict) -> str:
    close = history["Close"].astype(float)
    last, last_date = float(close.iloc[-1]), history.index[-1].date()
    currency = info.get("currency") or ("KRW" if item.ticker.endswith((".KS", ".KQ")) else "USD")

    def ret(n: int) -> str:
        return _pct(last, float(close.iloc[-n - 1])) if len(close) > n else "-"

    def ma(n: int) -> str:
        if len(close) < n:
            return "-"
        value = float(close.tail(n).mean())
        return f"{_num(value, currency)} (현재가가 {'위' if last >= value else '아래'})"

    vol = close.pct_change().tail(20).std()
    vol_text = f"{vol * math.sqrt(252) * 100:.0f}%" if vol == vol else "-"  # NaN 방지
    name = info.get("longName") or info.get("shortName") or item.name
    rows = [
        ("기준일 종가", f"{_num(last, currency)} ({last_date})"),
        ("수익률 1일 / 1주 / 1개월", f"{ret(1)} / {ret(5)} / {ret(21)}"),
        ("수익률 3개월 / 6개월 / 1년", f"{ret(63)} / {ret(126)} / {_pct(last, float(close.iloc[0]))}"),
        ("52주 최고 / 최저", f"{_num(float(close.max()), currency)} / {_num(float(close.min()), currency)}"),
        ("이동평균 20일", ma(20)),
        ("이동평균 60일", ma(60)),
        ("이동평균 120일", ma(120)),
        ("20일 변동성(연율)", vol_text),
        ("20일 평균 거래량", f"{float(history['Volume'].tail(20).mean()):,.0f}"),
    ]
    valuation = [
        ("시가총액", info.get("marketCap"), lambda v: _num(v, currency)),
        ("PER(최근 12개월)", info.get("trailingPE"), lambda v: f"{v:.1f}"),
        ("PER(예상)", info.get("forwardPE"), lambda v: f"{v:.1f}"),
        ("PBR", info.get("priceToBook"), lambda v: f"{v:.2f}"),
        ("배당수익률", info.get("dividendYield"), lambda v: f"{v:.2f}%"),
    ]
    rows += [(label, fmt(v)) for label, v, fmt in valuation if isinstance(v, (int, float))]
    table = "\n".join(f"| {k} | {v} |" for k, v in rows)
    text = f"### {name} ({item.ticker})\n| 항목 | 값 |\n|---|---|\n{table}"

    if item.quantity > 0 and item.buy_price > 0:
        cost, value = item.quantity * item.buy_price, item.quantity * last
        text += (
            f"\n\n**보유 평가 (코드 계산)**: {item.quantity:,.0f}주 × 평균 매입가 {_num(item.buy_price, currency)}"
            f" = 매입금액 {_num(cost, currency)} → 평가금액 {_num(value, currency)}, "
            f"평가손익 {_num(value - cost, currency)} ({_pct(value, cost)})"
        )
    return text


def run(question: str, inputs: dict, models: ModelsConfig) -> ToolResult | None:
    items, notes = resolve(question, inputs, models)
    if not items and not notes:
        return None
    sections = []
    for item in items:
        history = _load(item)
        if len(history) == 0:
            notes.append(f"'{item.query}' 시세를 찾지 못함 (티커 {item.ticker}) — 종목명·코드를 확인하세요")
            continue
        sections.append(summarize(item, history, fetch_info(item.ticker)))
    header = (
        f"출처: Yahoo Finance 일별 시세 (조회일 {date_label(today())}). 지연·누락이 있을 수 있으며 "
        "아래 숫자는 코드로 계산한 값이므로 그대로 인용한다 (직접 다시 계산하지 않는다)."
    )
    content = header + "\n\n" + "\n\n".join(sections) if sections else ""
    return ToolResult(tool="stock_data", title=f"📈 시장 데이터 {len(sections)}종목", content=content, notes=notes)
