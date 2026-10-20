"""주식 업무의 '숫자 검산' — LLM 이 쓴 점수표·매매 계획을 코드로 다시 계산해 맞는지 확인한다.

왜 필요한가?
  가중 평균 점수, 분할 매수 평균가, 손익비, 물타기 후 평단가는 '계산'이다.
  LLM 은 계산을 자주 틀리고, 틀린 숫자로 매수/보유/매도를 정하면 판단 자체가 틀어진다.
  그래서 답변 끝의 ```json 블록(기계가 읽는 요약)을 코드로 다시 계산해
    - 하네스: 틀리면 '정답 값'을 알려 주며 다시 쓰게 하고 (validators.StockPlanValidator)
    - 화면  : 사람이 볼 수 있게 검산표를 따로 보여 준다 (report)

업무별 JSON 블록 (team.yaml 의 업무 양식이 이 형식을 요구한다)
  recommend    : {"candidates": [{"ticker", "extract_fundamentals", "analyze_risk", "analyze_opportunity",
                                  "final_decision": {"score", "one_line_rationale", "key_assumption"}}]}
  scorecard    : {"scorecard": {"valuation": 4, ..., "risk": null}, "composite": 4.35, "verdict": "매수"}
  trade_plan   : {"trade_plan": {"horizon", "tranches": [{"price", "weight"}], "take_profit": [1차, 2차],
                                 "stop_loss", "risk_reward"}}
  position_plan: {"position_plan": {"action", "add_orders": [{"price", "quantity"}], "new_average",
                                    "take_profit_half", "take_profit_full", "stop_loss"}}
"""

import json
from dataclasses import dataclass, field

from core.harness.base import extract_code_blocks

# ─── 판단(7개 항목 루브릭) ──────────────────────────────────────────

SCORE_ITEMS = {  # key: (표시 이름, 가중치 %)
    "valuation": ("① 밸류에이션", 20),
    "earnings": ("② 실적·수익성", 20),
    "financial": ("③ 재무 건전성", 15),
    "growth": ("④ 성장성", 15),
    "industry": ("⑤ 산업·경쟁력", 10),
    "momentum": ("⑥ 수급·모멘텀·심리", 10),
    "risk": ("⑦ 리스크", 10),
}
BUY_LINE, SELL_LINE = 3.75, 2.50  # 3.75 이상 매수 / 2.50~3.74 보유 / 2.50 미만 매도
MAX_MISSING = 2  # '데이터 부족' 항목이 3개 이상이면 점수와 무관하게 판단 보류
HOLD_OFF = "판단 보류"

# ─── 매매 계획 ─────────────────────────────────────────────────────

# 보유 기간별 손절폭(평균 매수가 대비 %) — 권장 범위와, 이 밖이면 다시 쓰게 하는 허용 범위
STOP_BANDS = {
    "단기": {"recommended": "3~5%", "allowed": (2.0, 7.0)},
    "중기": {"recommended": "8~10%", "allowed": (6.0, 13.0)},
    "장기": {"recommended": "15% 이상", "allowed": (12.0, 60.0)},
}
POSITION_ACTIONS = ("보유", "추가매수", "분할매도", "전량매도", "비중축소")


@dataclass
class CheckResult:
    """한 종류의 JSON 블록 검산 결과."""

    kind: str
    found: bool = False
    problems: list[str] = field(default_factory=list)  # 틀린 곳 (하네스가 다시 쓰게 함)
    computed: dict = field(default_factory=dict)  # 코드가 계산한 값 (화면 검산표)
    warnings: list[str] = field(default_factory=list)  # 틀린 건 아니지만 사람이 볼 점


# ─── JSON 블록 찾기 ─────────────────────────────────────────────────


def json_blocks(answer: str) -> list:
    """답변 속 ```json 블록을 파싱한다 (파싱 안 되는 블록은 건너뜀)."""
    out = []
    for src in extract_code_blocks(answer, "json"):
        try:
            out.append(json.loads(src))
        except json.JSONDecodeError:
            continue
    return out


def find_block(answer: str, key: str) -> dict | None:
    """key 를 가진 첫 JSON 객체."""
    return next((b for b in json_blocks(answer) if isinstance(b, dict) and key in b), None)


def _num(x) -> float | None:
    """숫자 또는 '72,000' 같은 문자열 → float. 아니면 None."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.replace(",", "").replace("원", "").replace("$", "").strip())
        except ValueError:
            return None
    return None


def _price(x: float) -> str:
    return f"{x:,.0f}" if x >= 1000 else f"{x:,.2f}"


# ─── 계산 함수 (화면·테스트에서도 그대로 사용) ──────────────────────────


def composite_score(scores: dict) -> tuple[float | None, list[str]]:
    """가중 평균 종합 점수. 데이터 부족(null) 항목은 빼고 남은 가중치로 다시 나눈다."""
    missing = [k for k in SCORE_ITEMS if _num(scores.get(k)) is None]
    used = {k: _num(scores[k]) for k in SCORE_ITEMS if k not in missing}
    total_w = sum(SCORE_ITEMS[k][1] for k in used)
    if not total_w:
        return None, missing
    return round(sum(v * SCORE_ITEMS[k][1] for k, v in used.items()) / total_w, 2), missing


def verdict_for(composite: float | None, missing: list[str]) -> str:
    if composite is None or len(missing) > MAX_MISSING:
        return HOLD_OFF
    if composite >= BUY_LINE:
        return "매수"
    return "보유" if composite >= SELL_LINE else "매도"


def average_price(orders: list[tuple[float, float]]) -> float:
    """(가격, 수량 또는 비중) 목록의 가중 평균가."""
    total = sum(q for _, q in orders)
    return sum(p * q for p, q in orders) / total if total else 0.0


def risk_reward(avg: float, take_profit: float, stop_loss: float) -> float | None:
    """손익비 = (1차 목표가 - 평균 매수가) / (평균 매수가 - 손절가)."""
    risk = avg - stop_loss
    return round((take_profit - avg) / risk, 2) if risk > 0 else None


def horizon_key(text: str) -> str | None:
    return next((k for k in STOP_BANDS if k in str(text)), None)


def stop_band_pct(horizon: str) -> tuple[float, float] | None:
    """보유 기간별 권장 손절폭(%) 하한·상한 (장기는 상한 없음)."""
    key = horizon_key(horizon)
    return {"단기": (3.0, 5.0), "중기": (8.0, 10.0), "장기": (15.0, 0.0)}.get(key) if key else None


# ─── 검산 (업무별) ─────────────────────────────────────────────────


def check_recommend(answer: str, options: dict) -> CheckResult:
    res = CheckResult("recommend")
    data = find_block(answer, "candidates")
    if data is None:
        res.problems.append('"candidates" 목록을 가진 ```json 블록이 없다 (Stage 1 스키마로 종목별 평가를 출력하라)')
        return res
    res.found = True
    candidates = data.get("candidates") or []
    expected = int(_num(options.get("expected_count")) or 0)
    if not candidates:
        res.problems.append("candidates 가 비어 있다")
    elif expected and len(candidates) != expected:
        res.problems.append(f"요청한 종목 수는 {expected}개인데 candidates 는 {len(candidates)}개다")
    rows = []
    for i, c in enumerate(candidates, 1):
        if not isinstance(c, dict):
            res.problems.append(f"candidates[{i}] 가 객체가 아니다")
            continue
        name = c.get("ticker") or f"{i}번째"
        for key in ("extract_fundamentals", "analyze_risk", "analyze_opportunity"):
            if not str(c.get(key) or "").strip():
                res.problems.append(f"{name}: {key} 가 비어 있다 (점수 전에 분석을 먼저 써야 한다)")
        decision = c.get("final_decision") or {}
        score = _num(decision.get("score"))
        if score is None or score != int(score) or not 1 <= score <= 5:
            res.problems.append(f"{name}: final_decision.score 는 1~5 정수여야 한다 (현재 {decision.get('score')!r})")
        for key in ("one_line_rationale", "key_assumption"):
            if not str(decision.get(key) or "").strip():
                res.problems.append(f"{name}: final_decision.{key} 가 비어 있다")
        rows.append({"ticker": name, "score": score, "rationale": decision.get("one_line_rationale", ""),
                     "assumption": decision.get("key_assumption", "")})
    res.computed = {"candidates": sorted(rows, key=lambda r: -(r["score"] or 0))}
    return res


def check_scorecard(answer: str, options: dict) -> CheckResult:
    res = CheckResult("scorecard")
    data = find_block(answer, "scorecard")
    if data is None:
        res.problems.append('"scorecard" 를 가진 ```json 블록이 없다 (7개 항목 점수·composite·verdict 를 출력하라)')
        return res
    res.found = True
    scores = data.get("scorecard") or {}
    unknown = [k for k in scores if k not in SCORE_ITEMS]
    absent = [k for k in SCORE_ITEMS if k not in scores]
    if unknown or absent:
        res.problems.append(f"scorecard 키는 {list(SCORE_ITEMS)} 7개여야 한다 (없음: {absent}, 모르는 키: {unknown})")
    for k in SCORE_ITEMS:
        v = scores.get(k)
        if v is not None and not (_num(v) is not None and _num(v) == int(_num(v)) and 1 <= _num(v) <= 5):
            res.problems.append(f"{SCORE_ITEMS[k][0]} 점수는 1~5 정수 또는 null(데이터 부족)이어야 한다 (현재 {v!r})")
    if res.problems:
        return res

    composite, missing = composite_score(scores)
    verdict = verdict_for(composite, missing)
    formula = " + ".join(
        f"({int(_num(scores[k]))}×{w / 100:.2f})" for k, (_, w) in SCORE_ITEMS.items() if k not in missing
    )
    if missing:
        used_w = sum(w for k, (_, w) in SCORE_ITEMS.items() if k not in missing)
        formula = f"[{formula}] ÷ {used_w / 100:.2f}"
    res.computed = {
        "scores": {k: _num(scores.get(k)) for k in SCORE_ITEMS},
        "composite": composite, "verdict": verdict, "missing": missing, "formula": formula,
    }
    stated = _num(data.get("composite"))
    if composite is not None and (stated is None or abs(stated - composite) > 0.01):
        res.problems.append(f"종합 점수 계산이 틀렸다: 적은 값 {data.get('composite')!r}, 코드 계산 {formula} = {composite:.2f}")
    if str(data.get("verdict", "")).strip() != verdict:
        reason = (
            f"데이터 부족 항목 {len(missing)}개(3개 이상)" if len(missing) > MAX_MISSING
            else f"종합 {composite:.2f}점 (3.75 이상 매수 / 2.50~3.74 보유 / 2.50 미만 매도)"
        )
        res.problems.append(f"결론이 결정 밴드와 다르다: 적은 결론 {data.get('verdict')!r}, 규칙상 '{verdict}' — {reason}")
    if missing:
        res.warnings.append("데이터 부족: " + ", ".join(SCORE_ITEMS[k][0] for k in missing))
    return res


def check_trade_plan(answer: str, options: dict) -> CheckResult:
    res = CheckResult("trade_plan")
    data = find_block(answer, "trade_plan")
    if data is None:
        res.problems.append('"trade_plan" 을 가진 ```json 블록이 없다 (분할 매수·목표가·손절가·손익비를 출력하라)')
        return res
    res.found = True
    plan = data.get("trade_plan") or {}
    horizon = str(plan.get("horizon", ""))
    tranches = [(_num(t.get("price")), _num(t.get("weight"))) for t in plan.get("tranches") or [] if isinstance(t, dict)]
    tps = [_num(x) for x in plan.get("take_profit") or []]
    sl = _num(plan.get("stop_loss"))

    if not horizon_key(horizon):
        res.problems.append(f"horizon 은 단기/중기/장기 중 하나여야 한다 (현재 {horizon!r})")
    if len(tranches) != 3 or any(p is None or p <= 0 or w is None or w <= 0 for p, w in tranches):
        res.problems.append("tranches 는 1·2·3차 3개이고, 각각 양수 price 와 weight(%)가 있어야 한다")
    if len(tps) != 2 or any(t is None or t <= 0 for t in tps):
        res.problems.append("take_profit 은 [1차 목표가, 2차 목표가] 두 개의 양수여야 한다")
    if sl is None or sl <= 0:
        res.problems.append("stop_loss 는 양수 가격이어야 한다")
    if res.problems:
        return res

    weight_sum = sum(w for _, w in tranches)
    if abs(weight_sum - 100) > 0.5:
        res.problems.append(f"분할 매수 비중 합이 100%가 아니다 (현재 {weight_sum:g}%)")
    avg = average_price(tranches)
    tp1, tp2 = tps
    rr = risk_reward(avg, tp1, sl)
    stop_pct = (avg - sl) / avg * 100
    current = _num(options.get("current_price"))
    res.computed = {
        "horizon": horizon, "tranches": tranches, "weight_sum": weight_sum, "average": avg,
        "take_profit": tps, "tp_pct": [(t / avg - 1) * 100 for t in tps], "stop_loss": sl,
        "stop_pct": stop_pct, "risk_reward": rr, "current_price": current,
    }
    if sl >= min(p for p, _ in tranches):
        res.problems.append(f"손절가 {_price(sl)} 가 가장 낮은 매수가 {_price(min(p for p, _ in tranches))} 이상이다 (손절가는 모든 매수가보다 낮아야 한다)")
    if tp1 <= avg:
        res.problems.append(f"1차 목표가 {_price(tp1)} 가 평균 매수가 {_price(avg)} 이하다")
    if tp2 < tp1:
        res.problems.append(f"2차 목표가 {_price(tp2)} 가 1차 목표가 {_price(tp1)} 보다 낮다")
    key = horizon_key(horizon)
    lo, hi = STOP_BANDS[key]["allowed"]
    if not lo <= stop_pct <= hi:
        res.problems.append(
            f"{key} 손절폭은 평균 매수가 대비 {STOP_BANDS[key]['recommended']} 내외여야 하는데 "
            f"현재 {stop_pct:.1f}% (평균 매수가 {_price(avg)}, 손절가 {_price(sl)})"
        )
    stated_rr = _num(plan.get("risk_reward"))
    if rr is not None and (stated_rr is None or abs(stated_rr - rr) > max(0.05, rr * 0.05)):
        res.problems.append(
            f"손익비 계산이 틀렸다: 적은 값 {plan.get('risk_reward')!r}, 코드 계산 "
            f"({_price(tp1)} - {_price(avg)}) ÷ ({_price(avg)} - {_price(sl)}) = {rr:.2f}"
        )
    if rr is not None and rr < 1.5:
        res.warnings.append(f"손익비 {rr:.2f} — 1.5 미만이면 기대 수익 대비 감수할 손실이 큰 편입니다")
    if current:
        far = [p for p, _ in tranches if abs(p / current - 1) > 0.3]
        if far:
            res.warnings.append(f"현재가 {_price(current)} 와 30% 이상 떨어진 매수가가 있습니다: {', '.join(map(_price, far))}")
    return res


def check_position_plan(answer: str, options: dict) -> CheckResult:
    res = CheckResult("position_plan")
    data = find_block(answer, "position_plan")
    if data is None:
        res.problems.append('"position_plan" 을 가진 ```json 블록이 없다 (행동·추가 매수·익절가·손절가를 출력하라)')
        return res
    res.found = True
    plan = data.get("position_plan") or {}
    action = str(plan.get("action", "")).replace(" ", "")
    orders = [(_num(o.get("price")), _num(o.get("quantity"))) for o in plan.get("add_orders") or [] if isinstance(o, dict)]
    half, full, sl = (_num(plan.get(k)) for k in ("take_profit_half", "take_profit_full", "stop_loss"))
    qty, avg0 = _num(options.get("quantity")) or 0, _num(options.get("buy_price")) or 0
    current = _num(options.get("current_price"))
    max_add = _num(options.get("max_add_amount"))

    if action not in POSITION_ACTIONS:
        res.problems.append(f"action 은 {list(POSITION_ACTIONS)} 중 하나여야 한다 (현재 {plan.get('action')!r})")
    if any(p is None or p <= 0 or q is None or q <= 0 for p, q in orders):
        res.problems.append("add_orders 의 각 항목은 양수 price 와 quantity(주)가 있어야 한다")
    for label, v in (("take_profit_half", half), ("take_profit_full", full), ("stop_loss", sl)):
        if v is None or v <= 0:
            res.problems.append(f"{label} 는 양수 가격이어야 한다")
    if res.problems:
        return res

    if half > full:
        res.problems.append(f"50% 익절가 {_price(half)} 가 전량 익절가 {_price(full)} 보다 높다")
    if current and sl >= current:
        res.problems.append(f"기계적 손절가 {_price(sl)} 가 현재가 {_price(current)} 이상이다 (지지선 이탈 가격은 현재가보다 낮아야 한다)")
    add_qty = sum(q for _, q in orders)
    add_amount = sum(p * q for p, q in orders)
    new_avg = average_price([(avg0, qty), *orders]) if qty and avg0 else None
    res.computed = {
        "action": action, "orders": orders, "add_qty": add_qty, "add_amount": add_amount,
        "quantity": qty, "buy_price": avg0, "new_quantity": qty + add_qty, "new_average": new_avg,
        "take_profit_half": half, "take_profit_full": full, "stop_loss": sl, "current_price": current,
        "max_add_amount": max_add,
    }
    if orders and new_avg:
        stated = _num(plan.get("new_average"))
        if stated is None or abs(stated / new_avg - 1) > 0.005:
            res.problems.append(
                f"추가 매수 후 평균 단가가 틀렸다: 적은 값 {plan.get('new_average')!r}, 코드 계산 "
                f"({_price(avg0)}×{qty:g} + 추가 {add_amount:,.0f}) ÷ {qty + add_qty:g}주 = {_price(new_avg)}"
            )
    if orders and max_add is not None and add_amount > max_add * 1.01:
        res.problems.append(f"추가 매수 금액 {add_amount:,.0f} 이 허용 한도 {max_add:,.0f} 를 넘는다 (최대 비중 제한)")
    if orders and current and avg0 and current < avg0:
        high = [p for p, _ in orders if p > current * 1.02]
        if high:
            res.warnings.append(f"손실 구간인데 현재가({_price(current)})보다 높은 물타기 가격이 있습니다: {', '.join(map(_price, high))}")
    if current and avg0:
        res.computed["pnl_pct"] = (current / avg0 - 1) * 100
        res.computed["stop_pnl_pct"] = (sl / (new_avg or avg0) - 1) * 100
    return res


CHECKS = {
    "recommend": check_recommend,
    "scorecard": check_scorecard,
    "trade_plan": check_trade_plan,
    "position_plan": check_position_plan,
}


def run_checks(answer: str, options: dict) -> list[CheckResult]:
    """options['stock_checks'] 에 적힌 검산만 실행한다."""
    return [CHECKS[name](answer, options) for name in options.get("stock_checks", []) if name in CHECKS]


# ─── 화면용 검산표 ─────────────────────────────────────────────────


def report(answer: str, options: dict) -> str:
    """사람이 볼 검산표 (마크다운). 검산할 것이 없으면 빈 문자열."""
    parts = []
    for res in run_checks(answer, options):
        c = res.computed
        if res.kind == "recommend" and c:
            rows = "\n".join(
                f"| {r['ticker']} | {'⭐' * int(r['score'] or 0)} {r['score'] or '-'} | {r['rationale']} | {r['assumption']} |"
                for r in c["candidates"]
            )
            parts.append("**종목별 투자 매력도 (점수순)**\n\n| 종목 | 점수 | 핵심 근거 | 틀릴 수 있는 전제 |\n|---|---|---|---|\n" + rows)
        elif res.kind == "scorecard" and c:
            rows = "\n".join(
                f"| {label} | {w}% | {'데이터 부족' if c['scores'][k] is None else int(c['scores'][k])} |"
                for k, (label, w) in SCORE_ITEMS.items()
            )
            comp = "-" if c["composite"] is None else f"{c['composite']:.2f}"
            parts.append(
                f"**7개 항목 가중 점수 → 결론: {c['verdict']}** (종합 {comp} / 5.00)\n\n"
                f"| 항목 | 가중치 | 점수 |\n|---|---|---|\n{rows}\n\n"
                f"계산: {c['formula']} = {comp}  ·  밴드: 3.75↑ 매수 / 2.50~3.74 보유 / 2.50↓ 매도 / 데이터 부족 3개↑ 보류"
            )
        elif res.kind == "trade_plan" and c:
            rows = "\n".join(
                f"| {i}차 | {_price(p)} | {w:g}% |" + (f" {(p / c['current_price'] - 1) * 100:+.1f}% |" if c["current_price"] else " - |")
                for i, (p, w) in enumerate(c["tranches"], 1)
            )
            band = stop_band_pct(c["horizon"])
            band_txt = "" if not band else (f" (권장 {band[0]:g}~{band[1]:g}%)" if band[1] else f" (권장 {band[0]:g}% 이상)")
            rr = "-" if c["risk_reward"] is None else f"{c['risk_reward']:.2f}"
            parts.append(
                f"**매매 계획 검산 — {c['horizon']}**\n\n| 분할 | 매수가 | 비중 | 현재가 대비 |\n|---|---|---|---|\n{rows}\n\n"
                f"- 평균 매수가(비중 가중): **{_price(c['average'])}** · 비중 합 {c['weight_sum']:g}%\n"
                f"- 1차 목표가 {_price(c['take_profit'][0])} ({c['tp_pct'][0]:+.1f}%) · 2차 목표가 {_price(c['take_profit'][1])} ({c['tp_pct'][1]:+.1f}%)\n"
                f"- 손절가 {_price(c['stop_loss'])} (평균 매수가 대비 -{c['stop_pct']:.1f}%{band_txt})\n"
                f"- 손익비 **{rr}** = (1차 목표가 − 평균 매수가) ÷ (평균 매수가 − 손절가)"
            )
        elif res.kind == "position_plan" and c:
            lines = [f"**보유 포지션 계획 검산 — 권고: {c['action']}**", ""]
            if c.get("pnl_pct") is not None:
                lines.append(f"- 현재 손익률: {c['pnl_pct']:+.1f}% (평균 단가 {_price(c['buy_price'])} → 현재가 {_price(c['current_price'])})")
            if c["orders"]:
                orders = ", ".join(f"{_price(p)}×{q:g}주" for p, q in c["orders"])
                lines.append(f"- 추가 매수: {orders} = {c['add_amount']:,.0f}")
                if c["new_average"]:
                    lines.append(f"- 추가 매수 후: {c['new_quantity']:g}주, 평균 단가 **{_price(c['new_average'])}**")
                if c["max_add_amount"] is not None:
                    lines.append(f"- 추가 매수 허용 한도(최대 비중 기준): {c['max_add_amount']:,.0f}")
            lines.append(f"- 50% 익절 {_price(c['take_profit_half'])} · 전량 익절 {_price(c['take_profit_full'])}")
            stop = f"- 기계적 손절 {_price(c['stop_loss'])}"
            if c.get("stop_pnl_pct") is not None:
                stop += f" (도달 시 평균 단가 대비 {c['stop_pnl_pct']:+.1f}%)"
            lines.append(stop)
            parts.append("\n".join(lines))
        if not res.found:
            parts.append(f"⚠️ {res.kind}: 기계가 읽을 JSON 요약이 없어 검산하지 못했습니다.")
        parts += [f"❌ {p}" for p in res.problems if res.found]
        parts += [f"⚠️ {w}" for w in res.warnings]
    return "\n\n".join(parts)
