import { api } from "../api.js";
import { state as appState } from "../state.js";
import { $, pct, money, cls, crossLabel } from "../util.js";

const state = {
  f: { ma: "20/50", eps: 10, window: 30, direction: "both", start: "2022-01-01" },
  data: null,
  sort: { key: "cross", dir: -1 },  // default: most recent trigger first
};
const HZ = [15, 30, 60, 90];

// column accessors for the sortable track-record table
const SORT = {
  ticker: (s) => s.ticker,
  dir: (s) => s.signal_type,
  eps: (s) => Math.abs(s.eps_change_pct || 0),
  cross: (s) => s.cross_date || "",
  entry: (s) => s.entry_price,
  now: (s) => (s.current_return == null ? null : s.current_return),
  15: (s) => s.forward_returns?.[15],
  30: (s) => s.forward_returns?.[30],
  60: (s) => s.forward_returns?.[60],
  90: (s) => s.forward_returns?.[90],
};

export async function render(root, ctx) {
  ctx.setCrumb("Triggered");
  root.innerHTML = `
    <div class="panel" style="margin-bottom:16px">
      <div class="phead"><h3>Triggered Track Record · how every trigger has performed</h3></div>
      <div style="display:flex;gap:12px;align-items:end;padding:12px 14px;flex-wrap:wrap">
        <div><span class="lbl">MA</span><div class="seg" id="ma" style="width:150px">
          <button data-v="20/50">20 / 50</button><button data-v="50/200">50 / 200</button></div></div>
        <div><span class="lbl">Min EPS Δ</span><input class="inp" id="eps" value="${state.f.eps}" style="width:80px"></div>
        <div><span class="lbl">Direction</span><div class="seg" id="direction" style="width:170px">
          <button data-v="both">Both</button><button data-v="bullish">Bull</button><button data-v="bearish">Bear</button></div></div>
        <div><span class="lbl">Since</span><input class="inp" id="start" value="${state.f.start}" style="width:120px"></div>
        <button class="run" id="run" style="margin:0;width:auto;padding:9px 20px">↗ Compute</button>
      </div>
    </div>
    <div id="body"><div class="loading">Set filters and Compute…</div></div>`;

  for (const id of ["ma", "direction"]) {
    const seg = $("#" + id, root);
    seg.querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.v === state.f[id]));
    seg.addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; state.f[id] = b.dataset.v; seg.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b)); });
  }
  $("#run", root).addEventListener("click", () => run(root, ctx));
  if (state.data) paint(root, ctx); else run(root, ctx);
}

async function run(root, ctx) {
  state.f.eps = $("#eps", root).value; state.f.start = $("#start", root).value;
  $("#body", root).innerHTML = `<div class="loading">Computing track record across every trigger — this runs a full historical scan, ~10–25s…</div>`;
  try { state.data = await api.performance(state.f); paint(root, ctx); }
  catch (e) { $("#body", root).innerHTML = `<div class="loading">Error: ${e.message}</div>`; }
}

function meterColor(win) { return win >= 55 ? "var(--green)" : win >= 45 ? "var(--cyan)" : "var(--red)"; }

function aggCard(label, s) {
  const win = s ? s.win_rate : null;
  return `<div class="hcard"><div class="h">${label}</div>
    <div class="win" style="color:${win == null ? "var(--t3)" : meterColor(win)}">${win == null ? "—" : win + "%"}</div>
    <div class="row"><span>win rate · n=${(s && s.sample) || 0}</span></div>
    <div class="meter"><i style="width:${win || 0}%;background:${meterColor(win || 0)}"></i></div>
    <div class="row"><span>avg return</span><b class="${cls(s && s.avg_return)}">${pct(s && s.avg_return, 2)}</b></div>
    <div class="row"><span>median</span><b class="${cls(s && s.median_return)}">${pct(s && s.median_return, 2)}</b></div></div>`;
}

function sortedRows() {
  const { key, dir } = state.sort;
  const acc = SORT[key] || SORT.cross;
  return [...state.data.signals].sort((a, b) => {
    const va = acc(a), vb = acc(b);
    if (va == null && vb == null) return 0;
    if (va == null) return 1;    // nulls always last
    if (vb == null) return -1;
    return va < vb ? -dir : va > vb ? dir : 0;
  });
}

function th(key, label, right = true) {
  const on = state.sort.key === key;
  const arrow = on ? `<span class="arrow">${state.sort.dir < 0 ? "▼" : "▲"}</span>` : "";
  return `<th class="sortable ${right ? "" : "l"}" data-k="${key}">${label}${arrow}</th>`;
}

function paint(root, ctx) {
  const d = state.data, bh = d.summary.by_horizon;
  const cards = aggCard("Now", d.summary.current) + HZ.map(h => aggCard(`+${h} days`, bh[h])).join("");

  $("#body", root).innerHTML = `
    <div class="ft-cards" style="grid-template-columns:repeat(5,1fr)">${cards}</div>
    <div class="panel"><div class="phead"><h3>${d.summary.total_signals} triggers · sortable track record</h3>
      <span class="lbl">click a column to sort · · = not enough forward data yet</span></div>
      <div style="overflow-x:auto;max-height:60vh"><table class="ft-tbl">
        <thead><tr>
          ${th("ticker", "Ticker", false)}${th("dir", "Dir", false)}${th("eps", "EPS Δ")}
          ${th("cross", "Cross")}${th("entry", "Entry")}${th("now", "Now")}
          ${HZ.map(h => th(String(h), `+${h}d`)).join("")}</tr></thead>
        <tbody id="ft-body"></tbody></table></div></div>`;

  $(".ft-tbl thead", root).onclick = e => {
    const h = e.target.closest("th.sortable"); if (!h) return;
    const k = h.dataset.k;
    if (state.sort.key === k) state.sort.dir *= -1;
    else state.sort = { key: k, dir: k === "ticker" ? 1 : -1 };
    paint(root, ctx);
  };
  paintRows(root, ctx);
}

function paintRows(root, ctx) {
  const rows = sortedRows();
  const cell = (v) => v == null ? `<td class="pending">·</td>` : `<td class="${cls(v)}">${pct(v, 1)}</td>`;
  $("#ft-body", root).innerHTML = rows.map((s, i) => {
    const bull = s.signal_type === "bullish";
    const fr = s.forward_returns || {};
    const held = s.days_held != null ? `<span style="color:var(--t3)"> · ${s.days_held}d</span>` : "";
    const now = s.current_return == null ? `<td class="pending">·</td>`
      : `<td class="${cls(s.current_return)}">${pct(s.current_return, 1)}${held}</td>`;
    return `<tr data-i="${i}" style="cursor:pointer"><td class="l"><b>${s.ticker}</b></td>
      <td class="l"><span class="dir ${bull ? "bull" : "bear"}">${bull ? "▲" : "▼"}</span></td>
      <td class="${cls(s.eps_change_pct)}">${pct(s.eps_change_pct)}</td>
      <td>${crossLabel(s.cross_date)}</td><td>${money(s.entry_price)}</td>
      ${now}${HZ.map(h => cell(fr[h])).join("")}</tr>`;
  }).join("");

  $("#ft-body", root).onclick = e => {
    const tr = e.target.closest("tr"); if (!tr) return;
    const s = rows[+tr.dataset.i];
    appState.signal = {
      ticker: s.ticker, signal_type: s.signal_type, eps_change_pct: s.eps_change_pct,
      trend_change_date: s.cross_date,
    };
    ctx.navigate("ticker", { sym: s.ticker, cross: s.cross_date, dir: s.signal_type });
  };
}
