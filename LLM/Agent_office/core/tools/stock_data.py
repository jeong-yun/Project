"""주가 데이터 툴 (yfinance — Yahoo Finance, API 키 불필요).

흐름
  1) 종목 찾기: 업무 입력(stock, holdings)과 질문에서 종목을 찾아 Yahoo 티커로 바꾼다.
     - 6자리 숫자 코드면 코드로 바로 (005930 → 005930.KS, 없으면 .KQ)
     - 이름이면 보조 모델이 티커를 추정 (삼성전자 → 005930.KS) → 실제 조회로 검증, 실패 시 .KS/.KQ 를 바꿔 재시도
  2) 1년 일봉을 받아 기준일 종가·기간별 수익률·52주 고저·이동평균·변동성을 '코드로' 계산
  3) 종목 식별(시장·업종), 재무 지표(EPS·ROE·이익률·부채비율·배당성향 등), 애널리스트 컨센서스(목표가)
     - 컨센서스 평균이 현재가와 너무 멀거나 최고·최저 차이가 크면 '극단값 주의' 표시 (평균만 믿지 않게)
  4) 보유 수량·매입가가 있으면 평가금액·평가손익·수익률, 비중 한도 안의 추가 매수 가능 금액,
     추가 매수(물타기·불타기) 후 평균 단가 시뮬레이션도 코드로 계산
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

# Yahoo 거래소 코드 → 시장 이름
EXCHANGES = {"KSC": "KOSPI", "KOE": "KOSDAQ", "NMS": "NASDAQ", "NGM": "NASDAQ", "NCM": "NASDAQ", "NYQ": "NYSE", "ASE": "NYSE American"}
# 컨센서스 평균 목표가가 현재가와 이 비율 이상 떨어져 있으면 '극단값 주의'
CONSENSUS_GAP_WARN = 0.6
# 보유 기간별 손절 참고폭 (현재가 대비 %) — 주식 매매 업무의 기준과 같다
STOP_REFERENCE = [("단기", (3, 5)), ("중기", (8, 10)), ("장기", (15,))]


class TickerItem(BaseModel):
    query: str = Field(description="사용자가 쓴 종목 이름 또는 코드 그대로")
    name: str = Field(description="정식 종목명")
    ticker: str = Field(
        description="Yahoo Finance 티커. 한국 코스피는 6자리코드.KS, 코스닥은 6자리코드.KQ, 미국은 심볼 (예: 005930.KS, 247540.KQ, AAPL)"
    )
    quantity: float = Field(description="보유 수량이 적혀 있으면 그 수, 없으면 0")
    buy_price: float = Field(description="평균 매입가가 적혀 있으면 그 값(숫자만), 없으면 0")


class Holding(TickerItem):
    """TickerItem + 업무 입력칸에서만 받는 비중 정보 (LLM 스키마에는 넣지 않는다)."""

    weight: float = 0  # 전체 투자 예산 중 이 종목 비중(%) — 모르면 0
    max_weight: float = 0  # 이 종목 최대 허용 비중(%)


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
    # 보유 종목(포트폴리오) 또는 추천 후보 종목 목록
    holdings = str(inputs.get("holdings") or inputs.get("candidates") or "").strip()
    qty, buy = float(inputs.get("quantity") or 0), float(inputs.get("buy_price") or 0)

    extra = {"weight": float(inputs.get("weight") or 0), "max_weight": float(inputs.get("max_weight") or 0)}

    if stock and CODE6.match(stock) and not holdings:  # 6자리 코드 → LLM 없이
        return [Holding(query=stock, name=stock, ticker=f"{stock}.KS", quantity=qty, buy_price=buy, **extra)], []
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
    items = [Holding(**i.model_dump()) for i in call.parsed.items]
    if stock and len(items) == 1:  # 단일 종목 업무는 입력 칸의 수량·매입가·비중을 우선
        items[0].quantity = qty or items[0].quantity
        items[0].buy_price = buy or items[0].buy_price
        items[0].weight, items[0].max_weight = extra["weight"], extra["max_weight"]
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
    market = EXCHANGES.get(info.get("exchange", ""), info.get("exchange") or "확인 불가")
    sector = " / ".join(x for x in (info.get("sector"), info.get("industry")) if x) or "확인 불가"
    text = (
        f"### {name} ({item.ticker})\n"
        f"[종목 식별] {name} / {item.ticker.split('.')[0]} / {market} / 업종 {sector} (기준일 {last_date})\n\n"
        f"| 항목 | 값 |\n|---|---|\n{table}"
    )
    text += fundamentals(info, currency)
    text += consensus(info, last, currency)
    text += stop_reference(last, currency)
    if item.quantity > 0 and item.buy_price > 0:
        text += position_block(item, last, currency)
    return text


def _ratio(v: float) -> str:
    return f"{v * 100:.1f}%"


def fundamentals(info: dict, currency: str) -> str:
    """재무 지표 (Yahoo 집계값). 확정 공시가 아니므로 DART 등으로 교차 확인하라고 적는다."""
    items = [
        ("EPS(최근 12개월)", info.get("trailingEps"), lambda v: _num(v, currency)),
        ("EPS(예상)", info.get("forwardEps"), lambda v: _num(v, currency)),
        ("ROE", info.get("returnOnEquity"), _ratio),
        ("영업이익률", info.get("operatingMargins"), _ratio),
        ("순이익률", info.get("profitMargins"), _ratio),
        ("매출 성장률(전년 동기)", info.get("revenueGrowth"), _ratio),
        ("이익 성장률(전년 동기)", info.get("earningsGrowth"), _ratio),
        ("부채비율(부채/자본)", info.get("debtToEquity"), lambda v: f"{v:.1f}%"),
        ("유동비율", info.get("currentRatio"), lambda v: f"{v * 100:.0f}%"),
        ("배당성향", info.get("payoutRatio"), _ratio),
        ("베타(시장 대비 변동성)", info.get("beta"), lambda v: f"{v:.2f}"),
    ]
    rows = [f"| {label} | {fmt(v)} |" for label, v, fmt in items if isinstance(v, (int, float))]
    missing = [label for label, v, _ in items if not isinstance(v, (int, float))]
    if not rows:
        return "\n\n**재무 지표**: 확인 불가 (Yahoo 에 값 없음 — DART·네이버 금융에서 확인 필요)"
    text = "\n\n**재무 지표 (Yahoo 집계값 · 확정 공시 아님 → DART 로 교차 확인)**\n| 항목 | 값 |\n|---|---|\n" + "\n".join(rows)
    if missing:
        text += f"\n확인 불가: {', '.join(missing)}"
    return text


def consensus(info: dict, last: float, currency: str) -> str:
    """애널리스트 컨센서스 + 극단값 주의 (평균 목표가만 믿지 않도록)."""
    mean, high, low = (info.get(k) for k in ("targetMeanPrice", "targetHighPrice", "targetLowPrice"))
    if not isinstance(mean, (int, float)) or not mean:
        return "\n\n**애널리스트 컨센서스**: 확인 불가"
    has_range = isinstance(high, (int, float)) and isinstance(low, (int, float)) and low > 0
    count = info.get("numberOfAnalystOpinions")
    lines = [
        f"\n\n**애널리스트 컨센서스 (전망치 — 확정 사실 아님)**: 평균 목표가 {_num(mean, currency)} (현재가 대비 {_pct(mean, last)})"
        + (f", 최고 {_num(high, currency)} / 최저 {_num(low, currency)}" if has_range else "")
        + (f", 애널리스트 {count}명" if count else "")
        + f", 투자의견 {info.get('recommendationKey') or '-'}"
    ]
    gap = abs(mean / last - 1)
    if gap >= CONSENSUS_GAP_WARN:
        lines.append(
            f"⚠️ 극단값 주의: 평균 목표가가 현재가와 {gap * 100:.0f}% 차이 — 오래된 목표가·데이터 오류일 수 있으므로 "
            "최근 증권사 리포트로 다시 확인하기 전에는 근거로 쓰지 않는다"
        )
    if has_range and high / low >= 2:
        lines.append(f"⚠️ 목표가 편차 큼 (최고/최저 {high / low:.1f}배) — 평균값만으로 판단하지 않는다")
    return "\n".join(lines)


def stop_reference(last: float, currency: str) -> str:
    """보유 기간별 손절 참고 가격 (현재가 기준) — 매매 계획의 손절폭이 기준에 맞는지 비교용."""
    parts = []
    for horizon, pcts in STOP_REFERENCE:
        prices = " ~ ".join(f"-{p}% {_num(last * (1 - p / 100), currency)}" for p in pcts)
        parts.append(f"{horizon} {prices}" + (" 이하" if len(pcts) == 1 else ""))
    return "\n\n**보유 기간별 손절 참고가 (현재가 기준, 코드 계산)**: " + " / ".join(parts)


def max_add_amount(value: float, weight_pct: float, max_weight_pct: float) -> float | None:
    """현재 비중·최대 허용 비중으로 '더 살 수 있는 금액'. 비중을 모르면 None."""
    if weight_pct <= 0 or max_weight_pct <= 0:
        return None
    total_budget = value / (weight_pct / 100)  # 이 종목 평가금액 ÷ 비중 = 전체 투자 예산
    return max(total_budget * max_weight_pct / 100 - value, 0.0)


def position_block(item: "Holding", last: float, currency: str) -> str:
    """보유 평가 + 추가 매수 한도 + 평균 단가 시뮬레이션 (모두 코드 계산)."""
    cost, value = item.quantity * item.buy_price, item.quantity * last
    text = (
        f"\n\n**보유 평가 (코드 계산)**: {item.quantity:,.0f}주 × 평균 매입가 {_num(item.buy_price, currency)}"
        f" = 매입금액 {_num(cost, currency)} → 평가금액 {_num(value, currency)}, "
        f"평가손익 {_num(value - cost, currency)} ({_pct(value, cost)})"
    )
    if last < item.buy_price:
        text += f"\n- 본전까지 필요한 상승률: {_pct(item.buy_price, last)}"
    weight, max_weight = getattr(item, "weight", 0), getattr(item, "max_weight", 0)
    limit = max_add_amount(value, weight, max_weight)
    if limit is not None:
        text += (
            f"\n- 비중: 현재 {weight:g}% → 최대 허용 {max_weight:g}% 기준 추가 매수 가능 금액 "
            f"{_num(limit, currency)} (현재가로 약 {int(limit // last):,}주)"
        )
    elif weight <= 0:
        text += "\n- 비중: 미정 (전체 예산 대비 비중을 모르므로 추가 매수 한도는 보수적으로 제시해야 한다)"
    rows = []
    for ratio in (0.25, 0.5, 1.0):
        add = max(round(item.quantity * ratio), 1)
        new_avg = (cost + add * last) / (item.quantity + add)
        rows.append(
            f"| 현재가로 {add:,}주 추가 (보유의 {ratio:.0%}) | {_num(add * last, currency)} | {_num(new_avg, currency)} | {_pct(new_avg, item.buy_price)} |"
        )
    kind = "물타기 — 손실 중" if last < item.buy_price else "불타기 — 수익 중"
    text += (
        f"\n\n**추가 매수 후 평균 단가 시뮬레이션 ({kind}, 코드 계산)**\n"
        "| 경우 | 추가 금액 | 새 평균 단가 | 평단 변화 |\n|---|---|---|---|\n" + "\n".join(rows)
    )
    return text


def run(question: str, inputs: dict, models: ModelsConfig) -> ToolResult | None:
    items, notes = resolve(question, inputs, models)
    if not items and not notes:
        return None
    sections, options = [], {}
    for item in items:
        history = _load(item)
        if len(history) == 0:
            notes.append(f"'{item.query}' 시세를 찾지 못함 (티커 {item.ticker}) — 종목명·코드를 확인하세요")
            continue
        sections.append(summarize(item, history, fetch_info(item.ticker)))
        if len(items) == 1:  # 단일 종목 업무 → 하네스 검산(stock_plan)에 현재가·보유 정보를 넘긴다
            last = float(history["Close"].iloc[-1])
            options = {"current_price": last, "quantity": item.quantity, "buy_price": item.buy_price}
            limit = max_add_amount(item.quantity * last, getattr(item, "weight", 0), getattr(item, "max_weight", 0))
            if limit is not None:
                options["max_add_amount"] = limit
    header = (
        f"출처: Yahoo Finance 일별 시세 (조회일 {date_label(today())}). 지연·누락이 있을 수 있으며 "
        "아래 숫자는 코드로 계산한 값이므로 그대로 인용한다 (직접 다시 계산하지 않는다). "
        f"인용할 때는 [Yahoo Finance, {today().isoformat()}] 처럼 출처와 기준일을 붙인다."
    )
    content = header + "\n\n" + "\n\n".join(sections) if sections else ""
    return ToolResult(
        tool="stock_data", title=f"📈 시장 데이터 {len(sections)}종목", content=content, notes=notes, harness_options=options
    )
