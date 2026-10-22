"""📺 전체 화면 보드 (TV·키오스크용) — P_test 'AI 브리핑 보드' 시안을 이 앱의 실제 데이터로 그린다.

시안의 설계 원칙 (멀리서 1초만 보는 화면)
  - 맨 위: 시계·날짜(화면이 살아 있다는 신호) · 갱신 시각(오래되면 주황) · 서비스 상태 불빛
  - '오늘의 한 줄'을 가장 크게 (60px) — 대부분 1초만 본다
  - 뉴스 카드는 15초마다 넘김: 제목 2줄·요약 3줄까지, 출처·발행일·AI 요약 표시·원문 QR (장식보다 출처가 먼저)
  - CLI 팁은 20초마다 하나: 명령 하나 + 설명 한 줄 (표를 통째로 띄우지 않는다)
  - 아래: 흐르는 자막 대신 고정된 '다음 뉴스' 목록 + 키워드

데이터
  - 뉴스: core/feeds.py 가 모은 data/feeds/latest.json 의 AI 브리핑 (없으면 모은 뉴스 원문)
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

from core import feeds, llm_factory
from core.config import CONFIG_DIR, load_yaml

BOARD_VIEW = "board"
TIPS_FILE = CONFIG_DIR / "cli_tips.yaml"
RELOAD_MIN = 10  # 보드가 스스로 새로 고치는 간격 (분)
# 상태 불빛에 보여 줄 provider (이 앱이 실제로 쓰는 순서)
STATUS_LABELS = {"claude_cli": "Claude 구독", "openai": "OpenAI", "google": "Gemini", "freellm": "무료 모델"}
TITLE_MAX, SUMMARY_MAX = 60, 120  # 시안 기준: 제목 2줄·요약 3줄(약 75~120자)


def _clip(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _domain(url: str) -> str:
    host = urlparse(url or "").netloc
    return host[4:] if host.startswith("www.") else host


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


def board_data(snap: dict | None, models, settings: dict) -> dict:
    cfg = feeds.feed_settings(settings)
    cards = news_cards(snap)
    return {
        "headline": (cards[0]["title"] or cards[0]["sum"]) if cards else "아직 모은 소식이 없습니다 — 🏠 메인에서 '🔄 지금 수집'을 눌러 주세요.",
        "news": cards,
        "tips": load_tips(),
        "keywords": keywords(snap, settings),
        "lights": status_lights(models),
        "collected_at": (snap or {}).get("collected_at"),
        "stale_hours": cfg["ttl_hours"],
        "reload_min": RELOAD_MIN,
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
    data = board_data(feeds.load_snapshot(Path(data_dir)), models, settings)
    components.html(board_html(data), height=1080, scrolling=False)


def headline_strip(snap: dict | None, models, settings: dict) -> None:
    """🏠 메인 위쪽 — 보드와 같은 '오늘의 한 줄' + 갱신 시각 + 서비스 상태 (멀리서도 1초에 읽히게)."""
    data = board_data(snap, models, settings)
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
    st.markdown(
        "<div style='border-top:2px solid rgba(127,127,127,.3);border-bottom:2px solid rgba(127,127,127,.3);"
        "padding:14px 2px;margin:6px 0 10px'>"
        "<div style='display:flex;flex-wrap:wrap;gap:10px 22px;align-items:center;font-size:.9rem;opacity:.9'>"
        f"<b style='color:#C98A12;letter-spacing:1px'>오늘의 한 줄</b>{fresh}<span>{lights}</span></div>"
        f"<div style='font-size:1.65rem;font-weight:700;line-height:1.35;margin-top:8px'>"
        f"{html_lib.escape(data['headline'])}</div></div>",
        unsafe_allow_html=True,
    )
    st.link_button("📺 전체 화면 보드 열기 (새 탭 · F11 로 전체 화면)", f"?view={BOARD_VIEW}")


def board_html(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return BOARD_TEMPLATE.replace("__DATA__", payload)


# 시안(1920×1080 무대를 창 크기에 맞춰 줄임)을 그대로 옮기고, 예시 데이터를 __DATA__ 로 바꿨다.
BOARD_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap">
<style>
:root{--bg:#0d1420;--panel:#142033;--panel-2:#1a2940;--line:#26354e;--fg:#e9eef5;--muted:#93a3b8;--accent:#f2b544;
  --ok:#4fc08d;--warn:#f08a3c;--off:#5b6b82;
  --font-body:"IBM Plex Sans KR","Malgun Gothic","Apple SD Gothic Neo",sans-serif;--font-mono:"JetBrains Mono",Consolas,"D2Coding",monospace}
html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font-family:var(--font-body);overflow:hidden}
.screen{position:fixed;inset:0;background:var(--bg)}
.stage{position:absolute;left:50%;top:50%;width:1920px;height:1080px;transform:translate(-50%,-50%) scale(var(--s,.5));
  display:grid;grid-template-rows:112px auto 1fr 150px;gap:28px;padding:40px 64px 44px;box-sizing:border-box}
.top{display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:40px}
.clock{display:flex;align-items:baseline;gap:24px}
.clock time{font:700 76px/1 var(--font-mono);font-variant-numeric:tabular-nums;letter-spacing:-1px}
.clock span{font-size:34px;color:var(--muted);font-weight:500}
.brand{font-size:34px;font-weight:600;color:var(--muted)}.brand b{color:var(--fg);font-weight:700}
.status{display:flex;align-items:center;gap:28px;font-size:28px;color:var(--muted)}
.fresh{display:inline-flex;align-items:center;gap:12px;padding:10px 20px;border-radius:999px;border:2px solid var(--line);color:var(--fg);font-weight:500}
.fresh.stale{border-color:var(--warn);color:var(--warn)}
.lights{display:flex;gap:20px}.light{display:inline-flex;align-items:center;gap:10px}
.dot{width:18px;height:18px;border-radius:50%;background:var(--ok);flex:none}.dot.warn{background:var(--warn)}.dot.off{background:var(--off)}
.brief{display:grid;grid-template-columns:220px 1fr;gap:32px;align-items:center;padding:28px 0;border-top:2px solid var(--line);border-bottom:2px solid var(--line)}
.brief .label{font-size:30px;font-weight:700;color:var(--accent);letter-spacing:2px}
.brief p{margin:0;font-size:58px;line-height:1.25;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.main{display:grid;grid-template-columns:1fr 640px;gap:40px;min-height:0}
.card,.tip{background:var(--panel);border-radius:20px;padding:34px 44px;box-sizing:border-box;min-height:0;min-width:0;position:relative;overflow:hidden}
.card{display:grid;grid-template-columns:1fr auto;grid-template-rows:auto auto 1fr auto;gap:16px 36px}.card .eyebrow{grid-column:1/-1}.card #n-qrbox{grid-column:2;grid-row:2/5;align-self:end}
.eyebrow{display:flex;align-items:center;justify-content:space-between;font-size:28px;color:var(--muted);font-weight:500;letter-spacing:1px}
.pager{display:flex;gap:12px}.pager i{width:16px;height:16px;border-radius:50%;background:var(--line)}.pager i.on{background:var(--fg)}
.card h2{margin:0;font-size:56px;line-height:1.22;font-weight:700;text-wrap:balance;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.card .sum{margin:0;font-size:38px;line-height:1.45;color:#cfd8e4;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
.meta{display:flex;align-items:flex-end;justify-content:space-between;gap:32px}
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
.tip{display:grid;grid-template-rows:auto auto auto auto 1fr;gap:14px;background:var(--panel-2)}.tip .foot{align-self:end}
.tip .tool{font-size:32px;font-weight:700}
.cmd{font:700 26px/1.35 var(--font-mono);background:var(--bg);color:var(--accent);padding:22px 26px;border-radius:12px;word-break:break-all}
.cmd::before{content:"> ";color:var(--muted)}
.tip .why{margin:0;font-size:28px;line-height:1.45;color:#cfd8e4;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.tip .foot{display:flex;align-items:flex-end;justify-content:space-between;gap:20px}
.tip dl{margin:0;display:grid;grid-template-columns:auto 1fr;gap:4px 14px;font-size:19px;min-width:0}
.tip dt{color:var(--muted)}.tip dd{margin:0;font-family:var(--font-mono);word-break:break-all}
.tip .qr .code,.tip .qr .code img,.tip .qr .code canvas{width:84px;height:84px}.tip .qr span{font-size:18px}
.next{display:grid;grid-template-columns:1fr auto;gap:40px;align-items:center;border-top:2px solid var(--line);padding-top:24px}
.next ol{list-style:none;margin:0;padding:0;display:flex;gap:40px;font-size:30px;min-width:0}
.next li{color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0}
.next li b{color:var(--fg);font-weight:600;margin-right:12px}
.next .label{font-size:26px;color:var(--accent);font-weight:700;letter-spacing:2px;margin-bottom:12px}
.kw{display:flex;gap:14px;font-size:28px}.kw span{padding:8px 18px;border:2px solid var(--line);border-radius:999px;color:var(--fg)}
#full{position:fixed;right:12px;top:10px;z-index:9;font:600 13px var(--font-body);color:var(--muted);background:var(--panel-2);
  border:1px solid var(--line);border-radius:8px;padding:6px 12px;cursor:pointer;opacity:.5}
#full:hover{opacity:1}
@media (prefers-reduced-motion:reduce){.fade{transition:none}.progress{transition:none!important}}
</style></head><body>
<button id="full" type="button" title="전체 화면 (F11 도 됩니다)">⛶ 전체 화면</button>
<div class="screen" id="screen"><div class="stage" id="stage">
  <header class="top">
    <div class="clock"><time id="time">--:--</time><span id="date"></span></div>
    <div class="brand"><b>AI 브리핑</b></div>
    <div class="status"><span class="fresh" id="fresh"><span class="dot" id="fresh-dot"></span><span id="fresh-text">수집 전</span></span>
      <div class="lights" id="lights"></div></div>
  </header>
  <section class="brief"><div class="label">오늘의 한 줄</div><p id="headline"></p></section>
  <div class="main">
    <article class="card" id="card">
      <div class="eyebrow"><span id="n-cat"></span><span class="pager" id="n-pager"></span></div>
      <h2 class="fade" id="n-title"></h2>
      <div class="qr fade" id="n-qrbox"><div class="code" id="n-qr">QR</div><span>원문 보기</span></div>
      <p class="sum fade" id="n-sum"></p>
      <div class="chips fade" id="n-chips"></div>
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
  var NEWS = D.news || [], TIPS = D.tips || [], NEWS_MS = 15000, TIP_MS = 20000;
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var $ = function(id){ return document.getElementById(id); };
  function txt(el, s){ el.textContent = s || ""; }
  var screen = $("screen"), stage = $("stage");
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
    var s = document.createElement("span"); s.className = "light";
    var dot = document.createElement("span"); dot.className = "dot" + (l.state === "ok" ? "" : " " + l.state);
    s.appendChild(dot); s.appendChild(document.createTextNode(l.name)); $("lights").appendChild(s);
  });
  txt($("headline"), D.headline);
  (D.keywords || []).forEach(function(k){ var s = document.createElement("span"); s.textContent = k; $("kw").appendChild(s); });

  function qr(el, url, size){
    el.innerHTML = "";
    if (!url) { el.textContent = "—"; return; }
    try { new QRCode(el, {text: url, width: size, height: size, colorDark: "#0d1420", colorLight: "#e9eef5", correctLevel: QRCode.CorrectLevel.M}); }
    catch (e) { el.textContent = "QR"; }
  }
  function pager(el, n, on){ el.innerHTML = ""; for (var k = 0; k < n; k++){ var i = document.createElement("i"); if (k === on) i.className = "on"; el.appendChild(i); } }
  function chip(cls, s){ var c = document.createElement("span"); c.className = "chip" + (cls ? " " + cls : ""); c.textContent = s; return c; }
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
    swap(["n-title","n-sum","n-chips","n-qrbox"], function(){
      txt($("n-cat"), n.cat + " · " + (idx + 1) + " / " + NEWS.length);
      txt($("n-title"), n.title); txt($("n-sum"), n.sum);
      var chips = $("n-chips"); chips.innerHTML = "";
      if (n.src) chips.appendChild(chip("", n.src));
      if (n.date) chips.appendChild(chip("plain", "발행 " + n.date));
      chips.appendChild(chip("ai", "AI 요약"));
      if (n.hype) chips.appendChild(chip("hype", "⚠️ 주장 — 확인 필요"));
      if (n.count) chips.appendChild(chip("plain", n.count));
      qr($("n-qr"), n.url, 150); pager($("n-pager"), NEWS.length, idx);
      var list = $("next"); list.innerHTML = "";
      for (var k = 1; k < Math.min(NEWS.length, 4); k++){
        var m = NEWS[(idx + k) % NEWS.length], li = document.createElement("li"), b = document.createElement("b");
        b.textContent = m.src; li.appendChild(b); li.appendChild(document.createTextNode(m.title)); list.appendChild(li);
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
      txt($("t-install"), t.install); txt($("t-file"), t.file); qr($("t-qr"), t.url, 84); pager($("t-pager"), TIPS.length, idx);
    });
    ti = (ti + 1) % TIPS.length;
  }
  showNews(); if (NEWS.length > 1) setInterval(showNews, NEWS_MS);
  showTip(); if (TIPS.length > 1) setInterval(showTip, TIP_MS);

  // 전체 화면: 보드가 들어 있는 페이지(부모 창) 전체를 키운다. 막히면 F11 안내
  $("full").addEventListener("click", function(){
    var doc = (window.parent && window.parent.document) || document, el = doc.documentElement;
    if (doc.fullscreenElement) { doc.exitFullscreen().catch(function(){}); return; }
    (el.requestFullscreen ? el.requestFullscreen() : Promise.reject()).catch(function(){ alert("F11 키를 눌러 전체 화면으로 보세요."); });
  });
  // 새 소식을 읽도록 주기적으로 새로 고침 (부모 페이지 = Streamlit 이 최신 데이터로 다시 그림)
  setTimeout(function(){ try { window.parent.location.reload(); } catch (e) { location.reload(); } }, (D.reload_min || 10) * 60000);
})();
</script></body></html>"""
