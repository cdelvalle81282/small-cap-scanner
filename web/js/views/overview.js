import { api } from "../api.js";
import { $, pct, money, cls, qColor, crossLabel } from "../util.js";

export async function render(root, ctx) {
  ctx.setCrumb("Overview");
  root.innerHTML = `<div class="loading">Loading market snapshot…</div>`;
  let meta, sig;
  try { [meta, sig] = await Promise.all([api.meta(), api.signals({ recency: 30, sort: "score" })]); }
  catch (e) { root.innerHTML = `<div class="loading">Error: ${e.message}</div>`; return; }

  const fresh = meta.is_fresh;
  const top = sig.signals.slice(0, 6);
  const avgScore = sig.signals.length ? Math.round(sig.signals.reduce((a, s) => a + s.score, 0) / sig.signals.length) : 0;

  root.innerHTML = `
    <div class="ov-grid">
      <div class="stat"><div class="k">New Signals · 30d</div><div class="v" style="color:var(--cyan)">${sig.count}</div>
        <div class="d">${sig.bullish} bull · ${sig.bearish} bear</div></div>
      <div class="stat"><div class="k">Avg Quality</div><div class="v" style="color:${qColor(avgScore)}">${avgScore}</div>
        <div class="d">across current signals</div></div>
      <div class="stat"><div class="k">Universe</div><div class="v">${meta.universe_size.toLocaleString()}</div>
        <div class="d">$1–50 · $50M–10B</div></div>
      <div class="stat"><div class="k">Data</div>
        <div class="v" style="color:${fresh ? "var(--green)" : "var(--amber)"}">${fresh ? "Fresh" : "Stale"}</div>
        <div class="d">prices ${meta.latest_price_date || "—"}${meta.days_stale != null ? ` · ${meta.days_stale}d ago` : ""}</div></div>
    </div>

    <div class="panel">
      <div class="phead"><h3>Top signals by quality</h3><span class="lbl" id="goscan" style="cursor:pointer;color:var(--cyan)">Open scanner →</span></div>
      <div class="tbl"><div class="thead" style="grid-template-columns:64px 1.4fr 90px 70px 76px 54px">
        <div>Signal</div><div>Ticker</div><div class="r">EPS Δ</div><div>Quality</div><div class="r">Price</div><div class="r">Cross</div></div>
        <div id="toprows"></div></div>
    </div>`;

  $("#toprows", root).innerHTML = top.map(r => {
    const bull = r.signal_type === "bullish";
    return `<div class="trow" data-sym="${r.ticker}" data-cross="${r.trend_change_date}" data-dir="${r.signal_type}" style="grid-template-columns:64px 1.4fr 90px 70px 76px 54px">
      <div><span class="dir ${bull ? "bull" : "bear"}">${bull ? "▲ BULL" : "▼ BEAR"}</span></div>
      <div class="tk"><b>${r.ticker}</b><span class="co">${r.name || ""}</span></div>
      <div class="eps ${cls(r.eps_change_pct)}">${pct(r.eps_change_pct)}</div>
      <div class="q"><div class="qtop"><span class="qv" style="color:${qColor(r.score)}">${r.score}</span><span class="qt">/100</span></div></div>
      <div class="r"><div class="price">${money(r.latest_close)}</div></div>
      <div class="ago">${crossLabel(r.trend_change_date)}</div></div>`;
  }).join("") || `<div class="loading">No recent signals</div>`;

  $("#goscan", root).onclick = () => ctx.navigate("scanner");
  $("#toprows", root).onclick = e => {
    const t = e.target.closest(".trow"); if (!t) return;
    ctx.navigate("ticker", { sym: t.dataset.sym, cross: t.dataset.cross, dir: t.dataset.dir });
  };
}
