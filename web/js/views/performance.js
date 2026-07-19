import { api } from "../api.js";
import { $, pct, money, cls, crossLabel } from "../util.js";

const state = {
  f: { ma: "20/50", eps: 10, window: 30, direction: "both", start: "2022-01-01" },
  data: null,
};
const HZ = [15, 30, 60, 90];

export async function render(root, ctx) {
  ctx.setCrumb("Follow-Through");
  root.innerHTML = `
    <div class="panel" style="margin-bottom:16px">
      <div class="phead"><h3>Signal Follow-Through · realized return after the trigger</h3></div>
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
  $("#run", root).addEventListener("click", () => run(root));
  if (state.data) paint(root); else run(root);
}

async function run(root) {
  state.f.eps = $("#eps", root).value; state.f.start = $("#start", root).value;
  $("#body", root).innerHTML = `<div class="loading">Computing follow-through across every trigger — this runs a full historical scan, ~10–25s…</div>`;
  try { state.data = await api.performance(state.f); paint(root); }
  catch (e) { $("#body", root).innerHTML = `<div class="loading">Error: ${e.message}</div>`; }
}

function meterColor(win) { return win >= 55 ? "var(--green)" : win >= 45 ? "var(--cyan)" : "var(--red)"; }

function paint(root) {
  const d = state.data, bh = d.summary.by_horizon;
  const cards = HZ.map(h => {
    const s = bh[h] || {};
    const win = s.win_rate;
    return `<div class="hcard"><div class="h">+${h} days</div>
      <div class="win" style="color:${win == null ? "var(--t3)" : meterColor(win)}">${win == null ? "—" : win + "%"}</div>
      <div class="row"><span>win rate · n=${s.sample || 0}</span></div>
      <div class="meter"><i style="width:${win || 0}%;background:${meterColor(win || 0)}"></i></div>
      <div class="row"><span>avg return</span><b class="${cls(s.avg_return)}">${pct(s.avg_return, 2)}</b></div>
      <div class="row"><span>median</span><b class="${cls(s.median_return)}">${pct(s.median_return, 2)}</b></div></div>`;
  }).join("");

  const rows = d.signals.map(s => {
    const bull = s.signal_type === "bullish";
    const fr = s.forward_returns || {};
    const cell = (v) => v == null ? `<td class="pending">·</td>` : `<td class="${cls(v)}">${pct(v, 1)}</td>`;
    return `<tr><td class="l"><b>${s.ticker}</b></td>
      <td class="l"><span class="dir ${bull ? "bull" : "bear"}">${bull ? "▲" : "▼"}</span></td>
      <td class="${cls(s.eps_change_pct)}">${pct(s.eps_change_pct)}</td>
      <td>${crossLabel(s.cross_date)}</td><td>${money(s.entry_price)}</td>
      ${HZ.map(h => cell(fr[h])).join("")}</tr>`;
  }).join("");

  $("#body", root).innerHTML = `
    <div class="ft-cards">${cards}</div>
    <div class="panel"><div class="phead"><h3>${d.summary.total_signals} triggers · realized returns</h3>
      <span class="lbl">· = not enough forward data yet</span></div>
      <div style="overflow-x:auto;max-height:60vh"><table class="ft-tbl">
        <thead><tr><th class="l">Ticker</th><th class="l">Dir</th><th>EPS Δ</th><th>Cross</th><th>Entry</th>
          ${HZ.map(h => `<th>+${h}d</th>`).join("")}</tr></thead>
        <tbody>${rows}</tbody></table></div></div>`;
}
