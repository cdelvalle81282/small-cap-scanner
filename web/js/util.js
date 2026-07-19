// formatting + tiny helpers shared across views

export const $ = (sel, root = document) => root.querySelector(sel);
export const el = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstElementChild; };

export const pct = (v, dp = 0) => v == null ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(dp)}%`;
export const money = (v) => v == null ? "—" : `$${v.toFixed(2)}`;
export const cls = (v) => v == null ? "" : (v >= 0 ? "pos" : "neg");

export function cap(v) {
  if (v == null) return "—";
  if (v >= 1e9) return `$${(v / 1e9).toFixed(1)}B`;
  return `$${(v / 1e6).toFixed(0)}M`;
}
export function dvol(v) {
  if (v == null) return "—";
  if (v >= 1e9) return `$${(v / 1e9).toFixed(1)}B`;
  if (v >= 1e6) return `$${(v / 1e6).toFixed(1)}M`;
  return `$${(v / 1e3).toFixed(0)}K`;
}
export function crossLabel(iso) {
  try { return new Date(iso + "T00:00").toLocaleDateString("en-US", { month: "short", day: "2-digit" }); }
  catch { return iso || ""; }
}

// quality-score colour ramp (cyan intensity — never green/red so it can't be
// confused with bullish/bearish direction)
export function qColor(q) {
  return q >= 80 ? "#22d3ee" : q >= 65 ? "#38bdf8" : q >= 50 ? "#64a7c9" : "#5a708f";
}
export function qSegs(q) {
  let s = "";
  for (let i = 0; i < 10; i++) {
    const on = (i + 1) * 10 <= q + 5;
    s += `<i style="background:${on ? qColor(q) : "var(--bg-3)"}"></i>`;
  }
  return s;
}

// sparkline from an array of closes, direction-coloured, with an endpoint dot
export function sparkline(closes, dir, w, h) {
  if (!closes || closes.length < 2) return "";
  const col = dir === "bullish" ? "#10b981" : "#f43f5e";
  const mn = Math.min(...closes), mx = Math.max(...closes), rng = (mx - mn) || 1, pad = 3;
  const pt = (v, i) => [pad + (i / (closes.length - 1)) * (w - 2 * pad), pad + (1 - (v - mn) / rng) * (h - 2 * pad)];
  const line = closes.map((v, i) => { const [x, y] = pt(v, i); return `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`; }).join(" ");
  const [ex, ey] = pt(closes[closes.length - 1], closes.length - 1);
  const gid = "g" + Math.random().toString(36).slice(2, 8);
  return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
    <defs><linearGradient id="${gid}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="${col}" stop-opacity=".25"/><stop offset="1" stop-color="${col}" stop-opacity="0"/></linearGradient></defs>
    <path d="${line} L ${w - pad} ${h - pad} L ${pad} ${h - pad} Z" fill="url(#${gid})"/>
    <path d="${line}" fill="none" stroke="${col}" stroke-width="1.5" stroke-linejoin="round"/>
    <circle cx="${ex.toFixed(1)}" cy="${ey.toFixed(1)}" r="2.4" fill="${col}"/></svg>`;
}

// minimal, safe markdown → html for the AI analysis text
export function mdLite(s) {
  if (!s) return "";
  const esc = s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return esc
    .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
    .replace(/^### (.+)$/gm, "<b>$1</b>")
    .replace(/\n/g, "<br>");
}
