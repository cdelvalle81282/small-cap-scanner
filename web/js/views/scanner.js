import { api } from "../api.js";
import { state } from "../state.js";
import { $, pct, money, cls, cap, dvol, crossLabel, qColor, qSegs, sparkline } from "../util.js";

const st = {
  f: { min_price: 1, max_price: 50, min_cap: "50M", max_cap: "10B", ma: "20/50", eps: 10, window: 30, direction: "both", recency: 30 },
  sort: "score", data: null, sel: 0, rows: [],
};

const SORTS = [["score", "Quality"], ["cross", "Cross date"], ["eps", "EPS Δ"], ["rvol", "RVOL"], ["dvol", "$ Vol"]];
const sortKey = {
  score: (r) => r.score, cross: (r) => r.trend_change_date,
  eps: (r) => Math.abs(r.eps_change_pct || 0), rvol: (r) => r.rvol || 0, dvol: (r) => r.avg_dollar_vol || 0,
};
const PRESET_KEY = "scanPresets";
const loadPresets = () => { try { return JSON.parse(localStorage.getItem(PRESET_KEY)) || {}; } catch { return {}; } };

export async function render(root, ctx) {
  ctx.setCrumb("Scanner");
  root.innerHTML = `
  <div class="scan-grid">
    <div class="panel rail">
      <div class="phead"><h3>Scan Parameters</h3></div>
      <div class="field"><span class="lbl">Presets</span><div style="display:flex;gap:6px">
        <select class="inp" id="preset" style="flex:1"></select><span class="chip" id="savePreset" style="cursor:pointer">Save</span></div></div>
      <div class="field"><span class="lbl">Price Range ($)</span><div class="row2">
        <input class="inp" id="min_price" value="${st.f.min_price}"><input class="inp" id="max_price" value="${st.f.max_price}"></div></div>
      <div class="field"><span class="lbl">Market Cap</span><div class="row2">
        <input class="inp" id="min_cap" value="${st.f.min_cap}"><input class="inp" id="max_cap" value="${st.f.max_cap}"></div></div>
      <div class="field"><span class="lbl">MA Crossover</span><div class="seg" id="ma">
        <button data-v="20/50">20 / 50</button><button data-v="50/200">50 / 200</button></div></div>
      <div class="field"><span class="lbl">Min EPS Δ (%)</span><input class="inp" id="eps" value="${st.f.eps}"></div>
      <div class="field"><span class="lbl">Direction</span><div class="seg" id="direction">
        <button data-v="both">Both</button><button data-v="bullish">Bull</button><button data-v="bearish">Bear</button></div></div>
      <div class="field"><span class="lbl">Trend Window (d)</span><input class="inp" id="window" value="${st.f.window}"></div>
      <div class="field"><span class="lbl">Max Days Since Cross</span><input class="inp" id="recency" value="${st.f.recency}"></div>
      <button class="run" id="run">▸ Run Scanner</button>
    </div>

    <div class="panel">
      <div class="sig-head">
        <div class="count"><b id="cnt">—</b> signals</div>
        <div class="split" id="split"></div>
        <div class="sorts" id="sorts">${SORTS.map(([k, l]) => `<span class="chip ${k === st.sort ? "on" : ""}" data-k="${k}">${l}</span>`).join("")}</div>
      </div>
      <div class="tbl">
        <div class="thead"><div>Signal</div><div>Ticker</div><div>90d</div><div class="r">EPS Δ</div><div class="r">RVOL</div><div>Quality</div><div class="r">Price · cap</div><div class="r">Cross</div></div>
        <div id="rows"><div class="loading">Run the scanner…</div></div>
      </div>
    </div>

    <div class="panel focus" id="focus"><div class="loading">Select a signal</div></div>
  </div>`;

  for (const id of ["ma", "direction"]) {
    const seg = $("#" + id, root);
    seg.querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.v === st.f[id]));
    seg.addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; st.f[id] = b.dataset.v; seg.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b)); });
  }
  $("#run", root).addEventListener("click", () => runScan(root, ctx));
  $("#sorts", root).addEventListener("click", e => { const c = e.target.closest(".chip"); if (!c) return; st.sort = c.dataset.k; $("#sorts", root).querySelectorAll(".chip").forEach(x => x.classList.toggle("on", x === c)); paint(root, ctx); });

  refreshPresets(root);
  $("#savePreset", root).onclick = () => {
    readForm(root);
    const name = prompt("Preset name?"); if (!name) return;
    const all = loadPresets(); all[name] = { ...st.f };
    localStorage.setItem(PRESET_KEY, JSON.stringify(all)); refreshPresets(root, name);
  };
  $("#preset", root).onchange = e => {
    const all = loadPresets(); const p = all[e.target.value]; if (!p) return;
    Object.assign(st.f, p); applyForm(root); runScan(root, ctx);
  };

  if (st.data) paint(root, ctx); else runScan(root, ctx);
}

function refreshPresets(root, selected = "") {
  const all = loadPresets();
  $("#preset", root).innerHTML = `<option value="">— presets —</option>` + Object.keys(all).map(n => `<option ${n === selected ? "selected" : ""}>${n}</option>`).join("");
}
function readForm(root) {
  for (const id of ["min_price", "max_price", "min_cap", "max_cap", "eps", "window", "recency"]) st.f[id] = $("#" + id, root).value;
}
function applyForm(root) {
  for (const id of ["min_price", "max_price", "min_cap", "max_cap", "eps", "window", "recency"]) $("#" + id, root).value = st.f[id];
  for (const id of ["ma", "direction"]) $("#" + id, root).querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.v === st.f[id]));
}

async function runScan(root, ctx) {
  readForm(root);
  const btn = $("#run", root); btn.disabled = true; btn.textContent = "▸ Scanning…";
  $("#rows", root).innerHTML = `<div class="loading">Scanning universe…</div>`;
  try { st.data = await api.signals(st.f); st.sel = 0; paint(root, ctx); }
  catch (e) { $("#rows", root).innerHTML = `<div class="loading">Error: ${e.message}</div>`; }
  finally { btn.disabled = false; btn.textContent = "▸ Run Scanner"; }
}

function paint(root, ctx) {
  const d = st.data; if (!d) return;
  const rows = [...d.signals].sort((a, b) => { const va = sortKey[st.sort](a), vb = sortKey[st.sort](b); return va < vb ? 1 : va > vb ? -1 : 0; });
  st.rows = rows;
  $("#cnt", root).textContent = d.count;
  const bp = d.count ? Math.round(d.bullish / d.count * 100) : 0;
  $("#split", root).innerHTML = `<span class="pos">${d.bullish} bull</span>
    <div class="splitbar"><i style="width:${bp}%;background:var(--green)"></i><i style="width:${100 - bp}%;background:var(--red)"></i></div>
    <span class="neg">${d.bearish} bear</span>`;
  $("#rows", root).innerHTML = rows.length ? rows.map((r, i) => rowHTML(r, i)).join("") : `<div class="loading">No signals match — widen the filters.</div>`;
  const rc = $("#rows", root);
  rc.onclick = e => {
    const t = e.target.closest(".trow"); if (!t) return;
    const i = +t.dataset.i;
    if (e.target.closest(".tk")) { openDetail(ctx, st.rows[i]); return; }
    select(root, ctx, i);
  };
  rc.onmouseover = e => { const t = e.target.closest(".trow"); if (t) select(root, ctx, +t.dataset.i); };
  if (rows.length) select(root, ctx, Math.min(st.sel, rows.length - 1));
  else $("#focus", root).innerHTML = `<div class="loading">No selection</div>`;
}

function rowHTML(r, i) {
  const bull = r.signal_type === "bullish";
  return `<div class="trow" data-i="${i}">
    <div><span class="dir ${bull ? "bull" : "bear"}">${bull ? "▲ BULL" : "▼ BEAR"}</span></div>
    <div class="tk"><b>${r.ticker}</b><span class="co">${r.name || ""}</span></div>
    <div class="spark">${sparkline(r.spark, r.signal_type, 120, 34)}</div>
    <div class="eps ${cls(r.eps_change_pct)}">${pct(r.eps_change_pct)}</div>
    <div class="rvol"><b>${r.rvol != null ? r.rvol.toFixed(1) + "×" : "—"}</b><div class="bar"><i style="width:${Math.min(100, (r.rvol || 0) / 4 * 100)}%"></i></div></div>
    <div class="q"><div class="qtop"><span class="qv" style="color:${qColor(r.score)}">${r.score}</span><span class="qt">/100</span></div><div class="qbar">${qSegs(r.score)}</div></div>
    <div class="r"><div class="price">${money(r.latest_close)}</div><div class="sub2">${cap(r.market_cap)} · ${dvol(r.avg_dollar_vol)}</div></div>
    <div class="ago">${r.days_ago}d<div class="u">${crossLabel(r.trend_change_date)}</div></div>
  </div>`;
}

function openDetail(ctx, r) {
  if (!r) return;
  state.signal = r;
  ctx.navigate("ticker", { sym: r.ticker, cross: r.trend_change_date, dir: r.signal_type });
}

const FLAB = { eps: "EPS surprise", rvol: "Relative volume", trend: "Trend alignment", timing: "Cross timing" };
function select(root, ctx, i) {
  st.sel = i;
  root.querySelectorAll(".trow").forEach(t => t.classList.toggle("sel", +t.dataset.i === i));
  const r = st.rows[i]; const bull = r.signal_type === "bullish";
  const fval = { eps: pct(r.eps_change_pct), rvol: r.rvol != null ? r.rvol.toFixed(1) + "×" : "—", trend: "SMA200", timing: `${r.days_between}d after` };
  $("#focus", root).innerHTML = `
    <div class="fx-tk"><div><b>${r.ticker}</b><div class="co">${r.name || ""}</div>
      <span class="dir ${bull ? "bull" : "bear"}" style="margin-top:5px">${bull ? "▲ Bullish" : "▼ Bearish"} ${r.fast_ma}/${r.slow_ma}</span></div>
      <div><div class="fx-p">${money(r.latest_close)}</div><div class="sub2 ${cls(r.eps_change_pct)}" style="text-align:right">EPS ${pct(r.eps_change_pct)}</div></div></div>
    <div class="fx-chart">${sparkline(r.spark, r.signal_type, 290, 110)}</div>
    <div class="why"><div class="lbl">Why this scores<span class="qbig" style="color:${qColor(r.score)}">${r.score}</span></div>
      ${Object.entries(FLAB).map(([k, lab]) => `<div class="factor"><div class="fr"><span>${lab}</span><b>${fval[k]}</b></div>
        <div class="fbar"><i style="width:${r.factors[k]}%"></i></div></div>`).join("")}</div>
    <div class="fx-actions"><button class="btn" id="watch">☆ Watch 30d</button><button class="btn solid" id="open">Open detail →</button></div>`;
  $("#open", root).onclick = () => openDetail(ctx, r);
  $("#watch", root).onclick = async e => {
    const b = e.target; b.disabled = true; b.textContent = "…";
    try {
      await api.watchAdd({ ticker: r.ticker, signal_type: r.signal_type, eps_change_pct: r.eps_change_pct,
        eps_date: r.eps_change_date || "", signal_date: r.trend_change_date, fast_ma: r.fast_ma, slow_ma: r.slow_ma });
      b.textContent = "✓ Watching";
    } catch { b.textContent = "✕ failed"; b.disabled = false; }
  };
}
