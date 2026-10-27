"""📺 전체 화면 보드 (TV·키오스크용) — P_test 'AI 브리핑 보드' 시안을 이 앱의 실제 데이터로 그린다.

시안의 설계 원칙 (멀리서 1초만 보는 화면)
  - 맨 위: 시계·날짜(화면이 살아 있다는 신호) · 갱신 시각(오래되면 주황) · 서비스 상태 불빛
    그 아래 지수·환율 띠 (코스피·코스닥·S&P 500·나스닥·다우·원/달러 — P_test 뉴스 브리핑의 지수 표와 같은 6개)
  - '오늘의 한 줄'을 가장 크게 (60px) — 대부분 1초만 본다
  - 뉴스 카드는 몇 초마다 넘김 (settings.yaml board.rotate_sec, 기본 8초): 제목 2줄·요약 3줄까지,
    출처·발행일·AI 요약 표시·원문 QR (장식보다 출처가 먼저). 아래 진행 막대·'다음 뉴스' 목록이 같이 넘어간다
  - CLI 팁은 20초마다 하나: 명령 하나 + 설명 한 줄 (표를 통째로 띄우지 않는다)
  - 아래: 흐르는 자막 대신 고정된 '다음 뉴스' 목록 + 키워드

데이터
  - 뉴스: core/feeds.py 가 모은 data/feeds/latest.json 의 AI 브리핑 (없으면 모은 뉴스 원문)
  - 지수·환율: core/market.py (yfinance, 10분 캐시 — 실패한 칸은 '—')
  - 서비스 상태: llm_factory.provider_status (실제 연결 상태)
  - CLI 팁: config/cli_tips.yaml
  - 주소 ?view=board 로 연다 (사이드바·메뉴 없이 화면 전체). 10분마다 스스로 새로 고쳐 새 소식을 읽는다.
"""

import html as html_lib
import json
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
import streamlit.components.v1 as components

from core import feeds, llm_factory, market
from core.config import CONFIG_DIR, load_yaml

BOARD_VIEW = "board"
TIPS_FILE = CONFIG_DIR / "cli_tips.yaml"
RELOAD_MIN = 10  # 보드가 스스로 새로 고치는 간격 (분)
ROTATE_SEC, TIP_ROTATE_SEC = 8, 20  # settings.yaml board.rotate_sec · tip_rotate_sec 기본값
# 상태 불빛에 보여 줄 provider (이 앱이 실제로 쓰는 순서)
STATUS_LABELS = {"claude_cli": "Claude 구독", "openai": "OpenAI", "google": "Gemini", "freellm": "무료 모델"}
TITLE_MAX, SUMMARY_MAX = 60, 120  # 시안 기준: 제목 2줄·요약 3줄(약 75~120자)
MARKET_COLOR = {"up": "#1F845A", "down": "#D64545", "flat": "inherit"}  # 🏠 메인 띠 (보드는 템플릿 CSS)


def _clip(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _domain(url: str) -> str:
    host = urlparse(url or "").netloc
    return host[4:] if host.startswith("www.") else host


def rotation(settings: dict) -> tuple[int, int]:
    """(뉴스, CLI 팁) 넘김 간격(초). 너무 짧으면 읽을 수 없어 3초 밑으로는 내리지 않는다."""
    cfg = (settings or {}).get("board") or {}

    def sec(key: str, default: int) -> int:
        try:
            return max(3, int(cfg.get(key, default)))
        except (TypeError, ValueError):
            return default

    return sec("rotate_sec", ROTATE_SEC), sec("tip_rotate_sec", TIP_ROTATE_SEC)


def load_tips(path: Path = TIPS_FILE) -> list[dict]:
    try:
        return [t for t in (load_yaml(path) or {}).get("tips") or [] if t.get("tool") and t.get("cmd")]
    except (OSError, ValueError):
        return []


def news_cards(snap: dict | None) -> list[dict]:
    """보드의 뉴스 카드 — AI 브리핑 한 줄마다 1장 (없으면 모은 뉴스 원문)."""
    if not snap:
        return []
    cards = []
    for point in snap.get("briefing") or []:
        links = point.get("links") or []
        first = links[0] if links else {}
        cards.append({
            "cat": "AI 브리핑",
            "title": _clip(first.get("title") or point.get("text", ""), TITLE_MAX),
            "sum": _clip(point.get("text", ""), SUMMARY_MAX),
            "src": _domain(first.get("url", "")) or feeds.SOURCE_LABELS.get(first.get("source", ""), ""),
            "date": first.get("date", ""),
            "count": f"{len(links)}개 출처" if len(links) > 1 else "1개 출처" if links else "",
            "hype": bool(point.get("hype")),
            "url": first.get("url", ""),
        })
    if not cards:  # 요약이 없으면 뉴스 원문 몇 개라도
        for item in [i for i in snap.get("items") or [] if i.get("source") == "news"][:6]:
            cards.append({"cat": "AI 뉴스", "title": _clip(item.get("title", ""), TITLE_MAX),
                          "sum": _clip(item.get("snippet", ""), SUMMARY_MAX), "src": _domain(item.get("url", "")),
                          "date": item.get("published") or item.get("date", ""), "count": "", "hype": False,
                          "url": item.get("url", "")})
    return cards


def keywords(snap: dict | None, settings: dict) -> list[str]:
    """아래 키워드 — 주목 저장소 이름이 있으면 그것, 없으면 뉴스 검색어."""
    repos = sorted((i for i in (snap or {}).get("items") or [] if i.get("source") == "github"),
                   key=lambda i: -(i.get("score") or 0))
    names = [r["title"].split("/")[-1] for r in repos[:3] if r.get("title")]
    if not names:
        names = [k.replace(" ", "") for k in feeds.feed_settings(settings)["news"].get("keywords", [])[:3]]
    return ["#" + n for n in names]


def status_lights(models) -> list[dict]:
    states = llm_factory.provider_status(models)
    return [{"name": label, "state": "ok" if states.get(p) == "ok" else "off" if states.get(p) == "disabled" else "warn"}
            for p, label in STATUS_LABELS.items() if p in states]


def board_data(snap: dict | None, models, settings: dict, quotes: list[dict] | None = None) -> dict:
    cfg = feeds.feed_settings(settings)
    cards = news_cards(snap)
    rotate, tip_rotate = rotation(settings)
    return {
        "headline": (cards[0]["title"] or cards[0]["sum"]) if cards else "아직 모은 소식이 없습니다 — 🏠 메인에서 '🔄 지금 수집'을 눌러 주세요.",
        "news": cards,
        "tips": load_tips(),
        "keywords": keywords(snap, settings),
        "lights": status_lights(models),
        "market": [market.view(q) for q in quotes or []],
        "market_pending": False,
        "collected_at": (snap or {}).get("collected_at"),
        "stale_hours": cfg["ttl_hours"],
        "reload_min": RELOAD_MIN,
        "rotate_sec": rotate,
        "tip_rotate_sec": tip_rotate,
    }


# ─── 화면 ────────────────────────────────────────────────────────────────


def render_board_page(models, settings: dict, data_dir: Path) -> None:
    """?view=board — Streamlit 메뉴·사이드바를 숨기고 보드만 화면 가득."""
    st.markdown(
        "<style>[data-testid='stSidebar'],[data-testid='stHeader'],[data-testid='stToolbar'],"
        "[data-testid='collapsedControl'],footer{display:none!important}"
        ".block-container,[data-testid='stMainBlockContainer']{padding:0!important;max-width:100%!important}"
        "[data-testid='stAppViewContainer'],.stApp{background:#0d1420}"
        "iframe{position:fixed!important;inset:0;display:block;width:100vw!important;height:100vh!important;border:0;z-index:999}</style>",
        unsafe_allow_html=True,
    )
    data_dir = Path(data_dir)
    data = board_data(feeds.load_snapshot(data_dir), models, settings, market.quotes(settings, data_dir))
    data["market_pending"] = market.pending(data_dir)  # 첫 시세가 늦으면 1분 뒤 다시 그려 채운다
    components.html(board_html(data), height=1080, scrolling=False)


def market_strip_html(cells: list[dict]) -> str:
    """🏠 메인용 작은 지수·환율 띠 (보드 맨 위 띠와 같은 값)."""
    parts = []
    for c in cells:
        arrow = {"up": "▲ ", "down": "▼ "}.get(c["dir"], "")
        change = (f" <span style='color:{MARKET_COLOR[c['dir']]};font-weight:600'>{arrow}{html_lib.escape(c['change'])}</span>"
                  if c["change"] else "")
        date = f" <span style='opacity:.6;font-size:.8em'>{html_lib.escape(c['date'])}</span>" if c["date"] else ""
        parts.append(f"<span style='white-space:nowrap' title='{html_lib.escape(c['title'], quote=True)}'>"
                     f"<span style='opacity:.7'>{html_lib.escape(c['name'])}</span> "
                     f"<b style='font-variant-numeric:tabular-nums'>{html_lib.escape(c['value'])}</b>{change}{date}</span>")
    return "<div style='display:flex;flex-wrap:wrap;gap:4px 22px;font-size:.88rem;margin-top:8px'>" + "".join(parts) + "</div>"


def headline_strip(snap: dict | None, models, settings: dict, data_dir: Path | None = None) -> None:
    """🏠 메인 위쪽 — 보드와 같은 '오늘의 한 줄' + 갱신 시각 + 서비스 상태 (+ 지수·환율 띠, 캐시에 있을 때만)."""
    # 메인은 시세를 기다리지 않는다 (wait=0) — 캐시가 오래됐으면 백그라운드로 받고, 다음에 다시 그릴 때 보인다
    quotes = market.quotes(settings, data_dir, wait=0) if data_dir else []
    data = board_data(snap, models, settings, quotes)
    age = ""
    stale = True
    if data["collected_at"]:
        hours = (datetime.now().timestamp() - data["collected_at"]) / 3600
        stale = hours >= data["stale_hours"]
        age = f"{int(hours * 60)}분 전 갱신" if hours < 1 else f"{hours:.0f}시간 전 갱신"
    color = {"ok": "#1F845A", "warn": "#E2741E", "off": "#8993A4"}
    lights = " ".join(f"<span style='white-space:nowrap'><span style='color:{color[l['state']]}'>●</span> "
                      f"{html_lib.escape(l['name'])}</span>" for l in data["lights"])
    fresh = (f"<span style='border:2px solid {'#E2741E' if stale else 'rgba(127,127,127,.35)'};border-radius:999px;"
             f"padding:2px 12px;color:{'#E2741E' if stale else 'inherit'}'>{age or '수집 전'}"
             f"{' · 확인 필요' if stale and age else ''}</span>")
    cells = [c for c in data["market"] if c["value"] != "—"]
    st.markdown(
        "<div style='border-top:2px solid rgba(127,127,127,.3);border-bottom:2px solid rgba(127,127,127,.3);"
        "padding:14px 2px;margin:6px 0 10px'>"
        "<div style='display:flex;flex-wrap:wrap;gap:10px 22px;align-items:center;font-size:.9rem;opacity:.9'>"
        f"<b style='color:#C98A12;letter-spacing:1px'>오늘의 한 줄</b>{fresh}<span>{lights}</span></div>"
        f"<div style='font-size:1.65rem;font-weight:700;line-height:1.35;margin-top:8px'>"
        f"{html_lib.escape(data['headline'])}</div>"
        f"{market_strip_html(cells) if cells else ''}</div>",
        unsafe_allow_html=True,
    )
    st.link_button("📺 전체 화면 보드 열기 (새 탭 · F11 로 전체 화면)", f"?view={BOARD_VIEW}")


def board_html(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return BOARD_TEMPLATE.replace("__DATA__", payload)


# 시안(1920×1080 무대를 창 크기에 맞춰 줄임)을 그대로 옮기고, 예시 데이터를 __DATA__ 로 바꿨다.
# 시안과 다른 곳 (실제 데이터·지수 띠를 넣으니 시안 크기로는 글이 잘려서 — 잘리지 않을 만큼만 바꿈)
#   (시안을 1920×1080 으로 그리면 시안 자체도 뉴스 제목이 반쯤 잘리고 요약·CLI 팁 아래쪽이 안 보인다)
#   - 맨 위 줄 아래에 지수·환율 띠 (사용자 요청). 그만큼 시계 줄 112→88px, 칸 사이 28→18px, 위아래 여백을 줄임
#   - 맨 위 브랜드는 'AI 브리핑'만 (서비스 불빛이 4개라 부제 '뉴스 요약과 CLI 활용'이 들어갈 자리가 없다)
#   - '오늘의 한 줄'은 1줄 (시안은 2줄 — 2줄이면 뉴스 카드 요약이 들어갈 자리가 없다)
#   - 뉴스 카드의 QR 은 오른쪽 칸 (시안은 출처 줄 오른쪽 — 그 줄이 QR 높이만큼 커져 제목·요약이 잘린다)
#   - 글자 크기를 조금 줄임: 뉴스 제목 66→58, 요약 44→38, CLI 팁 이름 40→34·명령 38→28·설명 32→28·설치 22→20, 팁 QR 120→90
BOARD_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap">
<style>
:root{--bg:#0d1420;--panel:#142033;--panel-2:#1a2940;--line:#26354e;--fg:#e9eef5;--muted:#93a3b8;--accent:#f2b544;
  --ok:#4fc08d;--warn:#f08a3c;--off:#5b6b82;--up:#4fc08d;--down:#f0625a;
  --font-body:"IBM Plex Sans KR","Malgun Gothic","Apple SD Gothic Neo",sans-serif;--font-mono:"JetBrains Mono",Consolas,"D2Coding",monospace}
html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font-family:var(--font-body);overflow:hidden}
.screen{position:fixed;inset:0;background:var(--bg)}
.stage{position:absolute;left:50%;top:50%;width:1920px;height:1080px;transform:translate(-50%,-50%) scale(var(--s,.5));
  display:grid;grid-template-rows:auto auto 1fr auto;gap:18px;padding:32px 64px 36px;box-sizing:border-box}
/* 맨 위: 시계 줄 + 지수·환율 띠 */
.head{display:grid;grid-template-rows:88px auto;gap:12px;min-width:0}
.top{display:grid;grid-template-columns:auto minmax(0,1fr) auto;align-items:center;gap:40px}
.clock{display:flex;align-items:baseline;gap:24px}
.clock time{font:700 76px/1 var(--font-mono);font-variant-numeric:tabular-nums;letter-spacing:-1px}
.clock span{font-size:34px;color:var(--muted);font-weight:500;white-space:nowrap}
.brand{font-size:34px;font-weight:600;color:var(--muted);letter-spacing:.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.brand b{color:var(--fg);font-weight:700}
.status{display:flex;align-items:center;gap:28px;font-size:28px;color:var(--muted)}
.fresh{display:inline-flex;align-items:center;gap:12px;padding:10px 20px;border-radius:999px;border:2px solid var(--line);color:var(--fg);font-weight:500;white-space:nowrap}
.fresh.stale{border-color:var(--warn);color:var(--warn)}
.lights{display:flex;gap:20px}.light{display:inline-flex;align-items:center;gap:10px;white-space:nowrap}
.dot{width:18px;height:18px;border-radius:50%;background:var(--ok);flex:none}.dot.warn{background:var(--warn)}.dot.off{background:var(--off)}
.market{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));background:var(--panel);border-radius:14px;overflow:hidden}
.market:empty{display:none}
.mk{display:grid;gap:2px;padding:8px 20px;border-left:2px solid var(--bg);min-width:0}
.mk .row{display:flex;align-items:baseline;justify-content:space-between;gap:10px;min-width:0}
.mk:first-child{border-left:0}
.mk .nm{font-size:24px;font-weight:600;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.mk .chg{font-size:24px;font-weight:700;white-space:nowrap;font-variant-numeric:tabular-nums}
.mk .val{font:700 30px/1.15 var(--font-mono);font-variant-numeric:tabular-nums;white-space:nowrap}
.mk .dt{font:500 20px var(--font-mono);color:var(--muted)}
.mk.up .chg{color:var(--up)}.mk.down .chg{color:var(--down)}.mk.flat .chg{color:var(--muted)}
.mk.none .val{color:var(--muted)}
/* 오늘의 한 줄 */
.brief{display:grid;grid-template-columns:220px minmax(0,1fr);gap:32px;align-items:center;padding:14px 0;border-top:2px solid var(--line);border-bottom:2px solid var(--line)}
.brief .label{font-size:30px;font-weight:700;color:var(--accent);letter-spacing:2px}
.brief p{margin:0;font-size:60px;line-height:1.3;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
/* 뉴스 카드 + CLI 팁 */
.main{display:grid;grid-template-columns:minmax(0,1fr) 640px;gap:40px;min-height:0}
.card,.tip{background:var(--panel);border-radius:20px;padding:30px 48px;box-sizing:border-box;min-height:0;min-width:0;position:relative;overflow:hidden}
.card{display:grid;grid-template-columns:minmax(0,1fr) auto;grid-template-rows:auto auto auto 1fr;gap:14px 36px;align-content:start}
.card .eyebrow{grid-column:1/-1}.card #n-qrbox{grid-column:2;grid-row:2/5;align-self:end}
.card .meta{grid-column:1;align-self:end}
.eyebrow{display:flex;align-items:center;justify-content:space-between;font-size:28px;color:var(--muted);font-weight:500;letter-spacing:1px}
.pager{display:flex;gap:12px}.pager i{width:16px;height:16px;border-radius:50%;background:var(--line)}.pager i.on{background:var(--fg)}
.card h2{margin:0;font-size:58px;line-height:1.22;font-weight:700;text-wrap:balance;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.card .sum{margin:0;font-size:38px;line-height:1.45;color:#cfd8e4;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
.chips{display:flex;flex-wrap:wrap;gap:14px;align-items:center;font-size:28px}
.chip{padding:8px 18px;border-radius:10px;background:var(--panel-2);color:var(--fg);font-weight:600}
.chip.ai{background:transparent;border:2px solid var(--accent);color:var(--accent)}
.chip.hype{background:transparent;border:2px solid var(--warn);color:var(--warn)}
.chip.plain{background:transparent;padding-inline:0;color:var(--muted);font-weight:500}
.qr{display:grid;justify-items:center;gap:8px;flex:none}
.qr .code{width:150px;height:150px;background:var(--fg);padding:10px;border-radius:10px;box-sizing:content-box;display:grid;place-items:center;color:var(--bg);font:700 24px var(--font-mono)}
.qr .code img,.qr .code canvas{width:150px;height:150px;display:block}.qr span{font-size:22px;color:var(--muted)}
.progress{position:absolute;left:0;bottom:0;height:8px;background:var(--accent);width:0}
.fade{transition:opacity .4s ease}.fade.out{opacity:0}
.tip{display:grid;grid-template-rows:auto auto auto auto 1fr;gap:14px;background:var(--panel-2);padding-inline:40px}
.tip .tool{font-size:34px;line-height:1.2;font-weight:700}
.cmd{font:700 28px/1.35 var(--font-mono);background:var(--bg);color:var(--accent);padding:14px 22px;border-radius:12px;word-break:break-all}
.cmd::before{content:"> ";color:var(--muted)}
.tip .why{margin:0;font-size:28px;line-height:1.45;color:#cfd8e4;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.tip .foot{display:flex;align-items:flex-end;justify-content:space-between;gap:20px;align-self:end}
.tip dl{margin:0;display:grid;grid-template-columns:auto 1fr;gap:4px 16px;font-size:20px;min-width:0}
.tip dt{color:var(--muted);white-space:nowrap}.tip dd{margin:0;font-family:var(--font-mono);word-break:break-all}
.tip .qr .code,.tip .qr .code img,.tip .qr .code canvas{width:90px;height:90px}.tip .qr span{font-size:20px}
/* 아래: 다음 뉴스 + 키워드 */
.next{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:40px;align-items:center;border-top:2px solid var(--line);padding-top:16px}
.next ol{list-style:none;margin:0;padding:0;display:flex;gap:40px;font-size:30px;min-width:0}
.next li{flex:1 1 0;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0}
.next li b{color:var(--fg);font-weight:600;margin-right:12px}
.next .label{font-size:26px;color:var(--accent);font-weight:700;letter-spacing:2px;margin-bottom:12px}
.kw{display:flex;gap:14px;font-size:28px}
.kw span{padding:8px 18px;border:2px solid var(--line);border-radius:999px;color:var(--fg);max-width:230px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#full{position:fixed;right:12px;top:10px;z-index:9;font:600 13px var(--font-body);color:var(--muted);background:var(--panel-2);
  border:1px solid var(--line);border-radius:8px;padding:6px 12px;cursor:pointer;opacity:.5}
#full:hover{opacity:1}
@media (prefers-reduced-motion:reduce){.fade{transition:none}.progress{transition:none!important}}
</style></head><body>
<button id="full" type="button" title="전체 화면 (F11 도 됩니다)">⛶ 전체 화면</button>
<div class="screen" id="screen"><div class="stage" id="stage">
  <header class="head">
    <div class="top">
      <div class="clock"><time id="time">--:--</time><span id="date"></span></div>
      <div class="brand"><b>AI 브리핑</b></div>
      <div class="status"><span class="fresh" id="fresh"><span class="dot" id="fresh-dot"></span><span id="fresh-text">수집 전</span></span>
        <div class="lights" id="lights"></div></div>
    </div>
    <div class="market" id="market"></div>
  </header>
  <section class="brief"><div class="label">오늘의 한 줄</div><p id="headline"></p></section>
  <div class="main">
    <article class="card" id="card">
      <div class="eyebrow"><span id="n-cat"></span><span class="pager" id="n-pager"></span></div>
      <h2 class="fade" id="n-title"></h2>
      <p class="sum fade" id="n-sum"></p>
      <div class="meta fade" id="n-meta"><div class="chips" id="n-chips"></div></div>
      <div class="qr fade" id="n-qrbox"><div class="code" id="n-qr">QR</div><span>원문 보기</span></div>
      <div class="progress" id="n-progress"></div>
    </article>
    <aside class="tip" id="tip">
      <div class="eyebrow"><span>오늘의 CLI 팁</span><span class="pager" id="t-pager"></span></div>
      <div class="tool fade" id="t-tool"></div><div class="cmd fade" id="t-cmd"></div><p class="why fade" id="t-why"></p>
      <div class="foot fade" id="t-foot"><dl><dt>설치</dt><dd id="t-install"></dd><dt>지침 파일</dt><dd id="t-file"></dd></dl>
        <div class="qr"><div class="code" id="t-qr">QR</div><span>공식 문서</span></div></div>
    </aside>
  </div>
  <footer class="next"><div style="min-width:0"><div class="label">다음 뉴스</div><ol id="next"></ol></div><div class="kw" id="kw"></div></footer>
</div></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js"></script>
<script>
(function(){
  var D = __DATA__;
  var NEWS = D.news || [], TIPS = D.tips || [], NEWS_MS = (D.rotate_sec || 8) * 1000, TIP_MS = (D.tip_rotate_sec || 20) * 1000;
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var $ = function(id){ return document.getElementById(id); };
  function txt(el, s){ el.textContent = s || ""; }
  function el(tag, cls, s){ var e = document.createElement(tag); if (cls) e.className = cls; if (s) e.textContent = s; return e; }
  var stage = $("stage");
  function fit(){ stage.style.setProperty("--s", Math.min(window.innerWidth / 1920, window.innerHeight / 1080)); }
  window.addEventListener("resize", fit); fit();

  var days = ["일","월","화","수","목","금","토"];
  function tick(){
    var d = new Date();
    txt($("time"), String(d.getHours()).padStart(2,"0") + ":" + String(d.getMinutes()).padStart(2,"0"));
    txt($("date"), (d.getMonth()+1) + "월 " + d.getDate() + "일 " + days[d.getDay()] + "요일");
    if (D.collected_at) {
      var mins = Math.floor((Date.now() / 1000 - D.collected_at) / 60), stale = mins >= D.stale_hours * 60;
      txt($("fresh-text"), (mins < 60 ? mins + "분 전 갱신" : Math.floor(mins / 60) + "시간 전 갱신") + (stale ? " · 확인 필요" : ""));
      $("fresh").classList.toggle("stale", stale); $("fresh-dot").className = "dot" + (stale ? " warn" : "");
    } else { $("fresh").classList.add("stale"); $("fresh-dot").className = "dot warn"; }
  }
  tick(); setInterval(tick, 10000);

  (D.lights || []).forEach(function(l){
    var s = el("span", "light");
    s.appendChild(el("span", "dot" + (l.state === "ok" ? "" : " " + l.state))); s.appendChild(document.createTextNode(l.name));
    $("lights").appendChild(s);
  });
  // 지수·환율 띠: 이름 · 등락률 / 값 · 기준 날짜 (값이 없으면 '—')
  (D.market || []).forEach(function(m){
    var c = el("div", "mk " + m.dir + (m.value === "—" ? " none" : "")); c.title = m.title;
    var r1 = el("div", "row"), r2 = el("div", "row");
    r1.appendChild(el("span", "nm", m.name));
    r1.appendChild(el("span", "chg", m.change ? (m.dir === "up" ? "▲ " : m.dir === "down" ? "▼ " : "") + m.change : ""));
    r2.appendChild(el("span", "val", m.value)); r2.appendChild(el("span", "dt", m.date));
    c.appendChild(r1); c.appendChild(r2);
    $("market").appendChild(c);
  });
  txt($("headline"), D.headline);
  (D.keywords || []).forEach(function(k){ $("kw").appendChild(el("span", "", k)); });

  function qr(box, url, size){
    box.innerHTML = "";
    if (!url) { box.textContent = "—"; return; }
    try { new QRCode(box, {text: url, width: size, height: size, colorDark: "#0d1420", colorLight: "#e9eef5", correctLevel: QRCode.CorrectLevel.M}); }
    catch (e) { box.textContent = "QR"; }
  }
  function pager(box, n, on){ box.innerHTML = ""; for (var k = 0; k < n; k++) box.appendChild(el("i", k === on ? "on" : "")); }
  function swap(ids, fill){
    var els = ids.map($);
    if (reduce) { fill(); return; }
    els.forEach(function(e){ e.classList.add("out"); });
    setTimeout(function(){ fill(); els.forEach(function(e){ e.classList.remove("out"); }); }, 400);
  }

  var ni = 0;
  function showNews(){
    if (!NEWS.length) { txt($("n-cat"), "AI 브리핑"); txt($("n-title"), "아직 모은 소식이 없습니다"); txt($("n-sum"), "🏠 메인 화면에서 '🔄 지금 수집'을 누르면 1~2분 안에 채워집니다."); return; }
    var n = NEWS[ni], idx = ni;
    swap(["n-title","n-sum","n-meta","n-qrbox"], function(){
      txt($("n-cat"), n.cat + " · " + (idx + 1) + " / " + NEWS.length);
      txt($("n-title"), n.title); txt($("n-sum"), n.sum);
      var chips = $("n-chips"); chips.innerHTML = "";
      if (n.src) chips.appendChild(el("span", "chip", n.src));
      if (n.date) chips.appendChild(el("span", "chip plain", "발행 " + n.date));
      chips.appendChild(el("span", "chip ai", "AI 요약"));
      if (n.hype) chips.appendChild(el("span", "chip hype", "⚠️ 주장 — 확인 필요"));
      if (n.count) chips.appendChild(el("span", "chip plain", n.count));
      qr($("n-qr"), n.url, 150); pager($("n-pager"), NEWS.length, idx);
      var list = $("next"); list.innerHTML = "";
      for (var k = 1; k < Math.min(NEWS.length, 4); k++){
        var m = NEWS[(idx + k) % NEWS.length], li = el("li");
        li.appendChild(el("b", "", m.src)); li.appendChild(document.createTextNode(m.title)); list.appendChild(li);
      }
    });
    var bar = $("n-progress"); bar.style.transition = "none"; bar.style.width = "0";
    if (!reduce && NEWS.length > 1) { void bar.offsetWidth; bar.style.transition = "width " + NEWS_MS + "ms linear"; bar.style.width = "100%"; }
    ni = (ni + 1) % NEWS.length;
  }
  var ti = 0;
  function showTip(){
    if (!TIPS.length) { $("tip").style.display = "none"; return; }
    var t = TIPS[ti], idx = ti;
    swap(["t-tool","t-cmd","t-why","t-foot"], function(){
      txt($("t-tool"), t.tool); txt($("t-cmd"), t.cmd); txt($("t-why"), t.why);
      txt($("t-install"), t.install); txt($("t-file"), t.file); qr($("t-qr"), t.url, 90); pager($("t-pager"), TIPS.length, idx);
    });
    ti = (ti + 1) % TIPS.length;
  }
  showNews(); if (NEWS.length > 1) setInterval(showNews, NEWS_MS);
  showTip(); if (TIPS.length > 1) setInterval(showTip, TIP_MS);

  // 전체 화면: 보드가 들어 있는 페이지(부모 창) 전체를 키운다. 막히면 F11 안내
  $("full").addEventListener("click", function(){
    var doc = (window.parent && window.parent.document) || document, root = doc.documentElement;
    if (doc.fullscreenElement) { doc.exitFullscreen().catch(function(){}); return; }
    (root.requestFullscreen ? root.requestFullscreen() : Promise.reject()).catch(function(){ alert("F11 키를 눌러 전체 화면으로 보세요."); });
  });
  // 새 소식을 읽도록 주기적으로 새로 고침 (부모 페이지 = Streamlit 이 최신 데이터로 다시 그림).
  // 지수·환율을 아직 받는 중이었으면 1분 뒤에 한 번 더 (앱을 막 켰을 때 첫 시세가 늦다)
  var reloadMs = D.market_pending ? 60000 : (D.reload_min || 10) * 60000;
  setTimeout(function(){ try { window.parent.location.reload(); } catch (e) { location.reload(); } }, reloadMs);
})();
</script></body></html>"""
