// 🎨 공용 픽셀 그림 — 🏢 픽셀 사무실(office.js)과 🏙 빌딩(building.js)이 함께 쓴다.
// 외부 이미지·라이브러리 없이 모든 픽셀을 코드로 그린다 (ASSETS.md).
// 주의: 일반 <script> 라 여기 const 이름은 office.js·building.js 와 겹치면 안 된다.
"use strict";
const T = 16;                                   // 타일 크기 (세계 좌표 px)

function hash(s) { let h = 2166136261; for (const c of s) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619); } return h >>> 0; }
function rect(c, x, y, ww, hh, col) { c.fillStyle = col; c.fillRect(Math.round(x), Math.round(y), ww, hh); }
function shade(hex, f) {
  const n = parseInt(hex.slice(1), 16), c = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => Math.max(0, Math.min(255, Math.round(v * (1 + f)))));
  return "#" + c.map((v) => v.toString(16).padStart(2, "0")).join("");
}
function clip(text, n) { const arr = [...text]; return arr.length > n ? arr.slice(0, n).join("") + "…" : text; }
function rrect(c, x, y, ww, hh, r) {
  c.beginPath(); c.moveTo(x + r, y); c.arcTo(x + ww, y, x + ww, y + hh, r); c.arcTo(x + ww, y + hh, x, y + hh, r);
  c.arcTo(x, y + hh, x, y, r); c.arcTo(x, y, x + ww, y, r); c.closePath();
}

// ── 캐릭터 (10×17 칸) — h 머리 · s 피부 · e 눈 · c 셔츠(직급 색) · d 셔츠 그늘 · w 깃 · p 바지 · k 신발 ──
const SKIN = ["#f1c27d", "#e0ac69", "#c68642", "#ffdbac", "#a8724a"];
const HAIR = ["#2c1b10", "#4a3020", "#1b1b1b", "#7a4a1e", "#b8860b", "#5a5a5a", "#a0522d", "#3d2b56"];
const PANTS = ["#2f3542", "#3b3b58", "#4b4b4b", "#2d3e50"];
const FRONT_HEAD = {
  0: ["..hhhhhh..", ".hhhhhhhh.", ".hhhhhhhh.", ".ssssssss.", ".ssessess.", ".ssssssss.", "..ssssss.."],
  1: ["..hhhhhh..", ".hhhhhhhh.", "hhhhhhhhhh", "hhssssssh.", "hssessessh", "h.ssssss.h", "..ssssss.."],
  2: ["...hhhh...", "..hhhhhh..", ".hhhhhhhh.", ".hssssssh.", ".ssessess.", ".ssssssss.", "..ssssss.."],
};
const BACK_HEAD = ["..hhhhhh..", ".hhhhhhhh.", ".hhhhhhhh.", ".hhhhhhhh.", ".hhhhhhhh.", ".shhhhhhs.", "..ssssss.."];
const BODY = ["..cwwwwc..", ".cccccccc.", "cccccccccc", "cdccccccdc", "cdccccccdc", "s.cccccc.s"];
const LEGS = [["..pppppp..", "..pp..pp..", "..pp..pp..", "..kk..kk.."],
              ["..pppppp..", "..pp...pp.", ".pp....pp.", ".kk....kk."],
              ["..pppppp..", ".pp...pp..", ".pp....pp.", ".kk....kk."]];

// 같은 id 면 늘 같은 머리·피부·바지 색
function charPalette(id, color) {
  const h = hash(id);
  return { h: HAIR[h % HAIR.length], s: SKIN[(h >> 3) % SKIN.length], e: "#1b1b1b", c: color, d: shade(color, -0.25),
           w: "#f4f4f4", p: PANTS[(h >> 6) % PANTS.length], k: "#222", t: "#7b1e1e", g: "#111" };
}

// 사람 한 명 (앞모습) — (x, y) 는 발 가운데, px = 한 칸 크기. seated 면 다리 없이 (책상 뒤)
function drawPerson(c, x, y, px, id, color, { seated = false, frame = 0, bob = 0, alpha = 1 } = {}) {
  const pal = charPalette(id, color);
  const rows = seated ? [...FRONT_HEAD[hash(id) % 3], ...BODY] : [...FRONT_HEAD[hash(id) % 3], ...BODY, ...LEGS[frame]];
  const left = Math.round(x) - 5 * px, top = Math.round(y) - 17 * px + bob;
  c.globalAlpha = alpha;
  rows.forEach((row, r) => {
    for (let i = 0; i < 10; i++) {
      const ch = row[i];
      if (ch === ".") continue;
      c.fillStyle = pal[ch] || pal.c;
      c.fillRect(left + i * px, top + r * px, px, px);
    }
  });
  c.globalAlpha = 1;
}

// 화분 (16×16) — (x0, y0) 는 왼쪽 위
function plant(c, x0, y0) {
  rect(c, x0 + 4, y0 + 9, 8, 6, "#9c5a32"); rect(c, x0 + 4, y0 + 9, 8, 1, "#b8703f");
  for (const [dx, dy, s] of [[3, 2, 5], [8, 1, 5], [5, -3, 6], [1, -1, 4], [10, 4, 4], [6, 4, 4]]) rect(c, x0 + dx, y0 + dy, s, s, dy < 0 ? "#3f9b4a" : "#2f7d3a");
}
