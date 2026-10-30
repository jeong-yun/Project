// 🏙 픽셀 빌딩 (🏠 메인 화면) — 스냅샷: core/building.py · 공용 그림: pixel.js · 불러오기: index.html
//  - 층마다 부서 띠 + 팀 방. 방 안의 작은 팀원·모니터·상태등은 실행 상태(일하는 중/결정 대기/오류/쉬는 중)를 보여 준다.
//  - 방에 마우스를 올리면 팀 요약 카드, 누르면 그 팀 화면. 부서 띠를 누르면 🧭 부서장, 로비를 누르면 아래 섹션.
//  - 낮·밤은 이 컴퓨터 시각으로 브라우저가 정한다 (파이썬 호출 없음). CPU 를 아끼려고 초당 12번까지만 그린다.
"use strict";
const cv = document.getElementById("c"), g = cv.getContext("2d");
const tip = document.getElementById("tip");
const world = document.createElement("canvas"), w = world.getContext("2d");
let L = null, S = null, bg = null, bgKey = "", scale = 1, lastH = 0, hover = null, hoverKey = "";
const FPS_MS = 1000 / 12;
let lastDraw = 0;
const lift = { y: null, target: null, wait: 0 };   // 엘리베이터 칸 (세계 px)

// ── 시간대 ───────────────────────────────────────────────────────
function phase() {
  const h = new Date().getHours();
  return h >= 7 && h < 17 ? "day" : (h >= 17 && h < 19) || (h >= 5 && h < 7) ? "dusk" : "night";
}
const SKY = { day: ["#8fd3f4", "#b6e4f8", "#d9f2fb"], dusk: ["#3b2f63", "#a4508b", "#f6a46b"], night: ["#0b1026", "#141b3d", "#1f2a52"] };
const LAMP = { running: "#3ddc84", waiting: "#ff9f1c", error: "#ff3b30", idle: "#4a4f5c" };

function mix(a, b, t) {   // 두 색 섞기 (#rrggbb)
  const pa = parseInt(a.slice(1), 16), pb = parseInt(b.slice(1), 16);
  const ch = (s) => Math.round(((pa >> s) & 255) * (1 - t) + ((pb >> s) & 255) * t);
  return "#" + [16, 8, 0].map((s) => ch(s).toString(16).padStart(2, "0")).join("");
}

// ── 배경 (배치·시간대가 바뀔 때만) ────────────────────────────────────
function drawBackground(ph) {
  bg = document.createElement("canvas");
  bg.width = L.w * T; bg.height = L.h * T;
  const b = bg.getContext("2d"), W = bg.width;
  const skyH = (L.roof.y + L.roof.h) * T;
  const bands = SKY[ph];
  for (let i = 0; i < 3; i++) rect(b, 0, (skyH * i) / 3, W, skyH / 3 + 1, bands[i]);
  if (ph === "night") {
    for (let i = 0; i < 40; i++) { const k = hash("star" + i); rect(b, k % W, (k >> 8) % (skyH - 8), 1 + (k % 3 === 0), 1, "#fffbe0"); }
    rect(b, W - 70, 8, 12, 12, "#f4f1c9"); rect(b, W - 66, 8, 8, 8, bands[0]);   // 초승달
  } else if (ph === "day") {
    rect(b, 40, 10, 14, 14, "#ffe066"); rect(b, 42, 12, 10, 10, "#fff3a3");
  } else {
    rect(b, 30, skyH - 22, 18, 18, "#ffb347");
  }
  // 옥상: 난간 · 물탱크 · 간판 기둥 · 안테나 기둥
  const roofTop = L.roof.y * T, top = skyH;
  rect(b, T, top - 6, W - 2 * T, 6, "#6d6475"); rect(b, T, top - 6, W - 2 * T, 1, "#8d8497");
  rect(b, 3 * T, roofTop + 4, 26, 18, "#7b8a99"); rect(b, 3 * T, roofTop + 4, 26, 3, "#9aa9b7"); rect(b, 3 * T + 4, roofTop + 22, 3, 4, "#555"); rect(b, 3 * T + 19, roofTop + 22, 3, 4, "#555");
  const signW = 16 * T, signX = (W - signW) / 2;
  rect(b, signX + 10, roofTop + 14, 3, top - roofTop - 18, "#3b3547"); rect(b, signX + signW - 13, roofTop + 14, 3, top - roofTop - 18, "#3b3547");
  rect(b, signX, roofTop - 14, signW, 28, "#2b2738"); rect(b, signX + 2, roofTop - 12, signW - 4, 24, ph === "day" ? "#3d3552" : "#4b2f6e");
  const ax = (L.shaft.x + 1) * T + 6;
  rect(b, ax, roofTop - 18, 2, top - roofTop + 12, "#4a445a"); rect(b, ax - 5, roofTop - 8, 12, 2, "#4a445a");
  // 바깥벽 (벽돌)
  const y0 = top, y1 = L.ground.y * T;
  for (const x of [0, (L.w - 1) * T]) {
    rect(b, x, y0, T, y1 - y0, "#8a5a44");
    for (let y = y0; y < y1; y += 6) rect(b, x + ((y / 6) % 2 ? 0 : 8), y, 1, 6, "#6e4534"), rect(b, x, y, T, 1, "#6e4534");
  }
  // 층마다: 층 번호판 · 부서 띠 · 방 · 바닥판
  for (const f of L.floors) {
    const fy = f.y * T, fh = 5 * T;
    rect(b, L.plate.x * T, fy, L.plate.w * T, fh, "#2d3142"); rect(b, L.plate.x * T, fy, L.plate.w * T, 2, "#3d4258");
    for (const r of f.rooms) drawRoomShell(b, r, ph);
    for (const bn of f.banners) {
      const col = S.depts[bn.dept] ? S.depts[bn.dept].color : "#64748b";
      rect(b, bn.x * T, fy, bn.w * T, T, col); rect(b, bn.x * T, fy + T - 3, bn.w * T, 3, shade(col, -0.35)); rect(b, bn.x * T, fy, bn.w * T, 1, shade(col, 0.3));
    }
    rect(b, T, fy + fh - 3, (L.w - 2) * T, 3, "#2a2433");
  }
  drawLobby(b, ph);
  // 엘리베이터 통로
  const sx = L.shaft.x * T, sw = L.shaft.w * T;
  rect(b, sx, L.shaft.y * T, sw, L.shaft.h * T, "#232031");
  rect(b, sx + sw / 2 - 1, L.shaft.y * T, 1, L.shaft.h * T, "#4b4560"); rect(b, sx + 4, L.shaft.y * T, 1, L.shaft.h * T, "#35304a"); rect(b, sx + sw - 5, L.shaft.y * T, 1, L.shaft.h * T, "#35304a");
  for (const f of [...L.floors.map((f) => f.y), L.lobby.y]) { rect(b, sx, f * T + 5 * T - 3, sw, 3, "#2a2433"); rect(b, sx + 6, f * T + 6, sw - 12, 5, "#141220"); }
  // 길 · 인도
  const gy = L.ground.y * T;
  rect(b, 0, gy, W, 10, "#b7b2a8"); for (let x = 0; x < W; x += 24) rect(b, x, gy, 1, 10, "#9d978c");
  rect(b, 0, gy + 10, W, L.ground.h * T - 10, "#3a3a44");
  for (let x = 8; x < W; x += 40) rect(b, x, gy + 20, 18, 2, "#e8d36a");
  for (const x of [T + 2, W - 2 * T - 2]) plant(b, x, gy - 14);
}

function drawRoomShell(b, r, ph) {
  const x0 = r.x * T, y0 = r.y * T, ww = r.w * T, hh = r.h * T;
  const room = r.team && S.rooms[r.team];
  if (!room) {   // 빈 사무실
    rect(b, x0, y0, ww, hh, "#3a3546"); rect(b, x0, y0 + hh - 8, ww, 8, "#4a4456");
    for (let i = 0; i < ww; i += 10) rect(b, x0 + i, y0 + hh - 8, 1, 8, "#3d3849");
    rect(b, x0 + 14, y0 + hh - 20, 14, 12, "#9c7a52"); rect(b, x0 + 14, y0 + hh - 20, 14, 2, "#b8946a"); rect(b, x0 + 30, y0 + hh - 16, 10, 8, "#8a6a46");
    rect(b, x0, y0, 3, hh, "#2a2433");
    return;
  }
  const wall = mix(room.color, "#efe3cc", 0.78), dark = mix(room.color, "#3a3040", 0.55);
  rect(b, x0, y0, ww, hh, wall);
  rect(b, x0, y0 + hh - 22, ww, 1, shade(wall, -0.12));                    // 허리 몰딩
  rect(b, x0, y0 + hh - 8, ww, 8, "#a87a4f");                              // 나무 바닥
  for (let i = 0; i < ww; i += 12) rect(b, x0 + i, y0 + hh - 8, 1, 8, "#8e6640");
  // 창문 (밤에는 일하는 방만 불이 켜진다 — 앞쪽 frame 에서 덧그림)
  const wx = x0 + ww - 48, wy = y0 + 9;
  rect(b, wx, wy, 30, 20, dark); rect(b, wx + 2, wy + 2, 26, 16, SKY[ph][2]); rect(b, wx + 14, wy + 2, 2, 16, dark); rect(b, wx + 2, wy + 9, 26, 1, dark);
  if (ph !== "day") rect(b, wx + 2, wy + 2, 26, 16, "rgba(10,12,30,.45)");
  rect(b, x0, y0, 3, hh, "#2a2433");                                        // 방 사이 벽
  plant(b, x0 + ww - 18, y0 + hh - 24);
}

function drawLobby(b, ph) {
  const y0 = L.lobby.y * T, hh = L.lobby.h * T, x0 = L.plate.x * T, x1 = L.shaft.x * T;
  rect(b, x0, y0, L.plate.w * T, hh, "#2d3142");
  rect(b, x0 + L.plate.w * T, y0, x1 - x0 - L.plate.w * T, hh, "#e9e4d8");
  for (let x = x0 + L.plate.w * T; x < x1; x += 16) for (let k = 0; k < 2; k++) rect(b, x + k * 8, y0 + hh - 12 + k * 6, 8, 6, (x / 16 + k) % 2 ? "#cfc8b6" : "#f4f0e6");
  rect(b, x0 + L.plate.w * T, y0, x1 - x0 - L.plate.w * T, 3, "#c9c0ab");
  for (const s of L.lobby.spots) {
    const sx = s.x * T, sw = s.w * T, fy = y0 + hh - 12;
    rect(b, sx, y0, 2, hh - 12, "#d8d0bd");
    if (s.kind === "reception") {
      rect(b, sx + 14, fy - 18, sw - 28, 18, "#7a4e2d"); rect(b, sx + 14, fy - 18, sw - 28, 3, "#a26a3c");
      rect(b, sx + 22, fy - 28, 12, 9, "#2b2b35"); rect(b, sx + 23, fy - 27, 10, 7, "#9bd8ff");
      for (let i = 0; i < 4; i++) rect(b, sx + sw - 40 + i * 6, fy - 24, 5, 6, ["#f4f1e8", "#fde68a", "#f4f1e8", "#bfdbfe"][i]);   // 신문
    } else if (s.kind === "entrance") {
      const dx = sx + sw / 2 - 22;
      rect(b, dx - 3, y0 + 14, 50, hh - 26, "#4a445a"); rect(b, dx, y0 + 17, 21, hh - 29, "#a9d8ee"); rect(b, dx + 23, y0 + 17, 21, hh - 29, "#a9d8ee");
      rect(b, dx + 2, y0 + 19, 4, hh - 33, "rgba(255,255,255,.5)"); rect(b, dx + 25, y0 + 19, 4, hh - 33, "rgba(255,255,255,.5)");
      rect(b, dx - 6, fy, 56, 4, "#7d2e2e");
    } else if (s.kind === "approval") {
      rect(b, sx + 14, fy - 18, sw - 28, 18, "#5b6b7d"); rect(b, sx + 14, fy - 18, sw - 28, 3, "#7d8fa3");
      for (let i = 0; i < 4; i++) rect(b, sx + 22, fy - 22 - i * 2, 14, 2, i % 2 ? "#f4f1e8" : "#e5e0d0");
      rect(b, sx + sw - 38, fy - 26, 6, 8, "#7a4e2d"); rect(b, sx + sw - 40, fy - 20, 10, 3, "#c0392b");   // 도장
    } else if (s.kind === "office") {
      rect(b, sx + 14, fy - 30, 22, 30, "#6b7280"); rect(b, sx + 16, fy - 28, 18, 26, "#9ca3af"); rect(b, sx + 23, fy - 18, 4, 4, "#374151");   // 금고
      const cx = sx + 46;
      rect(b, cx, y0 + 12, 46, 30, "#f8fafc"); rect(b, cx, y0 + 12, 46, 2, "#cbd5e1");
      [10, 16, 12, 22].forEach((hgt, i) => rect(b, cx + 6 + i * 10, y0 + 40 - hgt, 6, hgt, ["#60a5fa", "#34d399", "#fbbf24", "#f87171"][i]));
    }
  }
}

// ── 스냅샷 받기 ─────────────────────────────────────────────────
function onSnapshot(s) {
  if (!s || s.kind !== "building") return;
  S = s;
  const ph = phase();
  if (!L || s.layout.key !== L.key || bgKey !== s.layout.key + ph + JSON.stringify(s.depts)) {
    L = s.layout; world.width = L.w * T; world.height = L.h * T;
    bgKey = L.key + ph + JSON.stringify(s.depts);
    drawBackground(ph);
  }
  resize();
  if (hover) showTip(hover, null);   // 상태가 바뀌면 열린 카드도 새로
}
AO.onSnapshot = onSnapshot;

function resize() {
  if (!L) return;
  const cssW = Math.max(320, document.getElementById("wrap").clientWidth);
  scale = cssW / (L.w * T);
  const cssH = Math.round(L.h * T * scale), dpr = window.devicePixelRatio || 1;
  if (cv.width !== Math.round(cssW * dpr) || cv.height !== Math.round(cssH * dpr)) {
    cv.width = Math.round(cssW * dpr); cv.height = Math.round(cssH * dpr); cv.style.height = cssH + "px";
  }
  if (cssH !== lastH) { lastH = cssH; AO.send("streamlit:setFrameHeight", { height: cssH + 2 }); }
}
window.addEventListener("resize", resize);

// ── 움직이는 것들 ────────────────────────────────────────────────
function drawRoomLife(r, now, ph) {
  const room = S.rooms[r.team];
  const x0 = r.x * T, y0 = r.y * T, ww = r.w * T, hh = r.h * T, floorY = y0 + hh - 8;
  const busy = room.status === "running";
  if (ph !== "day" && (busy || room.status === "waiting")) rect(w, x0 + ww - 46, y0 + 11, 26, 16, "rgba(255,214,102,.75)");   // 밤에 켜진 창
  const n = room.people.length, span = (ww - 60) / n;
  room.people.forEach((p, i) => {
    const px = x0 + 14 + span * (i + 0.5);
    const bob = busy && Math.floor(now / 300 + i) % 2 ? -1 : 0;
    drawPerson(w, px, floorY - 8, 1, room.id + p.id, p.color, { seated: true, bob, alpha: room.status === "idle" && ph === "night" ? 0.55 : 1 });
    if (i === 0) rect(w, px - 1, floorY - 17 + bob, 2, 3, "#f5c518");   // 팀장: 금색 넥타이 (몸통 첫 줄 아래)
    rect(w, px - 11, floorY - 12, 22, 4, "#c8975b"); rect(w, px - 11, floorY - 8, 22, 8, "#9a6b3c"); rect(w, px - 9, floorY - 1, 2, 1, "#3d2a18");   // 책상
    const on = busy && i < Math.max(1, room.working);
    rect(w, px + 4, floorY - 19, 8, 7, "#2b2b35");
    rect(w, px + 5, floorY - 18, 6, 5, on ? ["#7fd1ff", "#8fe3c0", "#9bd8ff"][Math.floor(now / 180 + i) % 3] : "#3a4a5a");
  });
  // 상태등
  const lit = room.status === "idle" || room.status === "error" || Math.floor(now / (room.status === "running" ? 350 : 700)) % 2 === 0;
  rect(w, x0 + ww - 12, y0 + 4, 7, 7, "#1d1a26");
  rect(w, x0 + ww - 11, y0 + 5, 5, 5, lit ? LAMP[room.status] : shade(LAMP[room.status], -0.6));
}

function drawLift(now) {
  const stops = [...L.floors.map((f) => f.y), L.lobby.y].map((y) => y * T + 5 * T - 3 - 22);
  const live = Object.values(S.rooms).filter((r) => r.status === "running").map((r) => r.id);
  if (lift.y === null) lift.y = stops[stops.length - 1];
  if (lift.target === null || (Math.abs(lift.y - lift.target) < 0.5 && now > lift.wait)) {
    let next = stops[hash(String(Math.floor(now / 4000))) % stops.length];
    if (live.length) {   // 일하는 팀이 있으면 그 층과 로비를 오간다 (쪽지 배달)
      const f = L.floors.find((fl) => fl.rooms.some((r) => r.team === live[Math.floor(now / 8000) % live.length]));
      next = Math.abs(lift.y - stops[stops.length - 1]) < 1 && f ? f.y * T + 5 * T - 25 : stops[stops.length - 1];
    }
    if (Math.abs(lift.y - next) < 0.5) lift.wait = now + 1500; else lift.target = next;
  }
  if (lift.target !== null) {
    const d = lift.target - lift.y, step = Math.sign(d) * Math.min(Math.abs(d), 0.9);
    lift.y += step;
    if (Math.abs(lift.y - lift.target) < 0.5) { lift.y = lift.target; lift.target = null; lift.wait = now + 1600; }
  }
  const sx = L.shaft.x * T + 7, sw = L.shaft.w * T - 14, y = Math.round(lift.y);
  rect(w, sx, y, sw, 22, "#c9a227"); rect(w, sx + 2, y + 2, sw - 4, 18, "#e8d07a"); rect(w, sx + sw / 2 - 1, y + 2, 2, 18, "#b38f1e");
}

function drawLobbyLife(now) {
  const y0 = L.lobby.y * T, fy = y0 + L.lobby.h * T - 12;
  for (const s of L.lobby.spots) {
    const sx = s.x * T, sw = s.w * T;
    if (s.kind === "reception") drawPerson(w, sx + sw / 2, fy - 12, 1, "reception", "#2563eb", { seated: true });
    else if (s.kind === "entrance") {
      const step = Math.floor(now / 600) % 2;
      drawPerson(w, sx + sw / 2 + (step ? 1 : 0), fy + 2, 2, "__user__", "#f97316", { frame: 0 });
    } else if (s.kind === "approval") {   // 🤖 자율 실행 로봇
      const rx = sx + sw / 2 + 6, ry = fy - 34, blink = Math.floor(now / 1500) % 5 === 0;
      rect(w, rx, ry, 14, 11, "#9aa5b1"); rect(w, rx + 1, ry + 1, 12, 9, "#cbd5e1");
      rect(w, rx + 3, ry + 4, 3, blink ? 1 : 3, "#0ea5e9"); rect(w, rx + 8, ry + 4, 3, blink ? 1 : 3, "#0ea5e9");
      rect(w, rx + 6, ry - 4, 2, 4, "#64748b"); rect(w, rx + 5, ry - 6, 4, 2, Math.floor(now / 500) % 2 ? "#ef4444" : "#7f1d1d");
      rect(w, rx + 2, ry + 11, 10, 6, "#64748b");
    } else if (s.kind === "office") drawPerson(w, sx + sw - 30, fy, 1, "accountant", "#16a34a", {});
  }
}

function drawClouds(now, ph) {
  if (ph === "night") return;
  const W = L.w * T;
  for (let i = 0; i < 4; i++) {
    const k = hash("cloud" + i), y = 6 + (k % 22), x = ((k % W) + now / (90 + i * 40)) % (W + 60) - 50;
    const c = ph === "day" ? "rgba(255,255,255,.9)" : "rgba(255,210,230,.7)";
    rect(w, x, y + 4, 34, 8, c); rect(w, x + 8, y, 16, 6, c); rect(w, x + 4, y + 2, 26, 6, c);
  }
}

function frame(now) {
  requestAnimationFrame(frame);
  if (!L || !S || now - lastDraw < FPS_MS) return;
  lastDraw = now;
  const ph = phase();
  if (bgKey !== L.key + ph + JSON.stringify(S.depts)) { bgKey = L.key + ph + JSON.stringify(S.depts); drawBackground(ph); }
  w.imageSmoothingEnabled = false;
  w.drawImage(bg, 0, 0);
  drawClouds(now, ph);
  if (Math.floor(now / 600) % 2) rect(w, (L.shaft.x + 1) * T + 5, L.roof.y * T - 21, 4, 4, "#ff3b30");   // 안테나 불빛
  for (const f of L.floors) for (const r of f.rooms) if (r.team && S.rooms[r.team]) drawRoomLife(r, now, ph);
  drawLobbyLife(now);
  drawLift(now);
  const dpr = window.devicePixelRatio || 1;
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.imageSmoothingEnabled = false;
  g.drawImage(world, 0, 0, cv.width, cv.height);
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  drawOverlay(now);
}
requestAnimationFrame(frame);

// ── 글자 (화면 해상도로 선명하게) ──────────────────────────────────────
const FONT = '"Malgun Gothic", "Apple SD Gothic Neo", "Segoe UI Emoji", sans-serif';
function label(text, x, y, { size, bold = true, fg = "#fff", bgc = null, align = "left", pad = 4 }) {
  g.font = `${bold ? "bold " : ""}${size}px ${FONT}`; g.textAlign = align; g.textBaseline = "middle";
  const tw = g.measureText(text).width;
  if (bgc) {
    const left = align === "center" ? x - tw / 2 : x;
    g.fillStyle = bgc; rrect(g, left - pad, y - size * 0.72, tw + pad * 2, size * 1.44, 4); g.fill();
  }
  g.fillStyle = fg; g.fillText(text, x, y);
  return tw;
}

function drawOverlay(now) {
  const s = scale, fs = Math.max(10, Math.min(14, 6.6 * s));
  const W = L.w * T;
  label("🏢 AGENT OFFICE", (W / 2) * s, (L.roof.y * T) * s, { size: fs + 4, fg: Math.floor(now / 900) % 6 ? "#ffe9a8" : "#fff7d6", align: "center" });
  L.floors.forEach((f, i) => {
    const cx = (L.plate.x + L.plate.w / 2) * T * s, fy = f.y * T * s;
    label(`${f.n}F`, cx, fy + 2.1 * T * s, { size: fs + 6, align: "center", fg: "#f3ead7" });
    const cost = S.floorCosts[i];
    if (cost) label(`💰 ${cost}`, cx, fy + 3.6 * T * s, { size: fs - 2, align: "center", fg: "#9be7b4", bold: false });
    for (const bn of f.banners) {
      const d = S.depts[bn.dept];
      if (d) label(`${d.emoji} ${d.name}`, (bn.x * T + 6) * s, (f.y * T + 7) * s, { size: fs - 1 });
    }
    for (const r of f.rooms) {
      const x = (r.x * T + 7) * s, y = (r.y * T + 11) * s;
      if (!r.team) { label("빈 사무실", x, y, { size: fs - 2, fg: "#cfc8dd", bold: false }); continue; }
      const room = S.rooms[r.team];
      const tw = label(clip(room.name, 11), x, y, { size: fs - 1, fg: "#2a2030", bgc: "rgba(255,250,238,.88)", pad: 3 });
      if (room.status === "waiting" || room.status === "error") {   // 이름표 오른쪽 (창문 아래 줄로 내려가지 않게 자리 확인)
        const text = room.status === "waiting" ? "❗ 결정 대기" : "⚠ 오류", right = (r.x * T + r.w * T - 14) * s;
        const bob = Math.floor(now / 400) % 2 ? -1 : 0;
        g.font = `bold ${fs - 2}px ${FONT}`;
        const bw = g.measureText(text).width, fits = x + tw + 12 + bw < right;
        label(text, fits ? x + tw + 12 : x + 3, (fits ? y : y + fs * 1.6) + bob,
              { size: fs - 2, fg: "#fff", bgc: room.status === "waiting" ? "rgba(234,128,0,.95)" : "rgba(200,30,30,.95)", pad: 3 });
      }
    }
  });
  const ly = L.lobby.y * T;
  label("1F", (L.plate.x + L.plate.w / 2) * T * s, (ly + 2.1 * T) * s, { size: fs + 6, align: "center", fg: "#f3ead7" });
  label("로비", (L.plate.x + L.plate.w / 2) * T * s, (ly + 3.6 * T) * s, { size: fs - 2, align: "center", fg: "#c9c3d9", bold: false });
  for (const sp of L.lobby.spots) label(sp.label, (sp.x * T + sp.w * T / 2) * s, (ly + 8) * s, { size: fs - 2, align: "center", fg: "#3a2a14", bgc: "rgba(255,248,230,.9)", pad: 3 });
  if (S.live) label("● LIVE", (W - T * 1.3) * s, 10 * s, { size: fs - 1, align: "right", bgc: "rgba(200,40,40,.9)" });
  if (hover) {   // 올린 곳 테두리
    const b = hover.box;
    g.strokeStyle = "#ffd166"; g.lineWidth = 2.5; g.strokeRect(b[0] * s + 1, b[1] * s + 1, b[2] * s - 2, b[3] * s - 2);
  }
}

// ── 마우스 ──────────────────────────────────────────────────────
function hitAt(ev) {
  const r = cv.getBoundingClientRect(), mx = (ev.clientX - r.left) / scale, my = (ev.clientY - r.top) / scale;
  const inside = (x, y, ww, hh) => mx >= x && mx < x + ww && my >= y && my < y + hh;
  for (const f of L.floors) {
    for (const bn of f.banners) if (inside(bn.x * T, bn.y * T, bn.w * T, T)) return { kind: "dept", id: bn.dept, box: [bn.x * T, bn.y * T, bn.w * T, T] };
    for (const rm of f.rooms) if (inside(rm.x * T, rm.y * T, rm.w * T, rm.h * T))
      return rm.team ? { kind: "team", id: rm.team, box: [rm.x * T, rm.y * T, rm.w * T, rm.h * T] } : { kind: "vacant", id: "", box: [rm.x * T, rm.y * T, rm.w * T, rm.h * T] };
    if (inside(L.plate.x * T, f.y * T, L.plate.w * T, 5 * T)) return { kind: "floor", id: String(f.n), floor: f, box: [L.plate.x * T, f.y * T, L.plate.w * T, 5 * T] };
  }
  for (const sp of L.lobby.spots) if (sp.id !== "user" && inside(sp.x * T, L.lobby.y * T, sp.w * T, L.lobby.h * T))
    return { kind: "section", id: sp.id, label: sp.label, box: [sp.x * T, L.lobby.y * T, sp.w * T, L.lobby.h * T] };
  return null;
}

function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }
function row(parent, key, value) { const r = el("div", "row"); r.appendChild(el("span", "k", key)); r.appendChild(document.createTextNode(value)); parent.appendChild(r); return r; }
function chips(parent, key, items) {
  const r = el("div", "row chips"); r.appendChild(el("span", "k", key));
  for (const it of items) r.appendChild(el("span", "", it));
  parent.appendChild(r);
}

function fillTip(h) {
  tip.innerHTML = ""; tip.className = "card";
  const hd = el("div", "hd"), bd = el("div", "bd");
  tip.appendChild(hd); tip.appendChild(bd);
  if (h.kind === "team") {
    const room = S.rooms[h.id], c = room.card;
    hd.style.background = room.color;
    hd.appendChild(document.createTextNode(room.name)); hd.appendChild(el("small", "", room.deptName));
    bd.appendChild(el("div", "", c.description));
    row(bd, "팀장", c.lead);
    row(bd, "팀원", `${c.headcount}명 — ${c.members}`);
    chips(bd, "주요 업무", c.tasks);
    if (c.tools.length) chips(bd, "툴·기능", c.tools);
    const st = row(bd, "지금", ""); st.appendChild(el("span", "st", room.statusText)); st.lastChild.style.color = shade(LAMP[room.status], -0.3);
    if (room.recent) row(bd, "최근", `${room.recent.when} · ${room.recent.status} · ${room.recent.text}`);
    if (room.cost) row(bd, `최근 ${S.costDays}일`, `추정 비용 ${room.cost}`);
    bd.appendChild(el("div", "foot", "누르면 이 팀 화면으로 갑니다"));
  } else if (h.kind === "dept") {
    const d = S.depts[h.id];
    hd.style.background = d.color; hd.textContent = `${d.emoji} ${d.name}`;
    bd.appendChild(el("div", "", d.description));
    chips(bd, "팀", d.teams);
    bd.appendChild(el("div", "foot", "누르면 🧭 부서장에게 요청할 수 있습니다 (알맞은 팀을 골라 줌)"));
  } else if (h.kind === "floor") {
    hd.style.background = "#2d3142"; hd.textContent = `${h.floor.n}층`;
    const names = h.floor.banners.map((b) => S.depts[b.dept] ? `${S.depts[b.dept].emoji} ${S.depts[b.dept].name}` : b.dept);
    chips(bd, "부서", names);
    const i = L.floors.indexOf(h.floor);
    if (S.floorCosts[i]) row(bd, `최근 ${S.costDays}일`, `이 층 추정 비용 ${S.floorCosts[i]}`);
    const free = h.floor.rooms.filter((r) => !r.team).length;
    bd.appendChild(el("div", "foot", free ? `빈 사무실 ${free}칸 — 팀이 늘면 여기부터 찹니다` : "이 층은 꽉 찼습니다 — 팀이 늘면 위에 새 층이 생깁니다"));
  } else if (h.kind === "vacant") {
    hd.style.background = "#4a4456"; hd.textContent = "빈 사무실";
    bd.appendChild(el("div", "", "teams/ 에 새 팀을 만들면 config/departments.yaml 의 부서 순서대로 이 칸부터 채워집니다. 층이 꽉 차면 위에 새 층이 생깁니다."));
  } else if (h.kind === "section") {
    hd.style.background = "#7a5c2e"; hd.textContent = h.label;
    const what = { brief: "📰 오늘의 AI 브리핑·아이디어 카드·전체 화면 보드", autonomy: "🤖 자율 실행 상태와 승인 대기열", usage: "💰 모델별 토큰·추정 비용, API 키별 내역, 가격표" }[h.id];
    bd.appendChild(el("div", "", what));
    bd.appendChild(el("div", "foot", "누르면 빌딩 아래에 이 내용을 엽니다"));
  }
}

function showTip(h, ev) {
  const key = h.kind + ":" + h.id + ":" + (h.kind === "team" ? S.rooms[h.id].statusText : "");
  if (key !== hoverKey) { hoverKey = key; fillTip(h); }
  tip.style.display = "block";
  if (!ev) return;
  const r = cv.getBoundingClientRect();
  let x = ev.clientX - r.left + 16, y = ev.clientY - r.top + 14;
  if (x + tip.offsetWidth > r.width) x = ev.clientX - r.left - tip.offsetWidth - 12;
  if (y + tip.offsetHeight > r.height) y = Math.max(0, r.height - tip.offsetHeight - 4);
  tip.style.left = Math.max(0, x) + "px"; tip.style.top = y + "px";
}

cv.addEventListener("mousemove", (ev) => {
  if (!L) return;
  const h = hitAt(ev);
  cv.style.cursor = h && h.kind !== "vacant" && h.kind !== "floor" ? "pointer" : "default";
  hover = h;
  if (!h) { tip.style.display = "none"; hoverKey = ""; return; }
  showTip(h, ev);
});
cv.addEventListener("mouseleave", () => { hover = null; hoverKey = ""; tip.style.display = "none"; });
cv.addEventListener("click", (ev) => {
  const h = L && hitAt(ev);
  if (h && (h.kind === "team" || h.kind === "dept" || h.kind === "section"))
    AO.send("streamlit:setComponentValue", { value: { kind: h.kind, id: h.id, t: Date.now() }, dataType: "json" });
});
