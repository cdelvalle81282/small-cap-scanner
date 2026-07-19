import { api } from "../api.js";
import { state } from "../state.js";
import { $, pct, money, cls, qColor, crossLabel } from "../util.js";

// matches the server-side pre-warmed default so this hits the cache
const DEFAULT_FILTER = { min_price: 1, max_price: 50, min_cap: "50M", max_cap: "10B", recency: 30, sort: "score" };

export async function render(root, ctx) {
  ctx.setCrumb("Overview");
  root.innerHTML = `
    <div class="ov-grid">
      <div class="stat"><div class="k">New Signals · 30d</div><div class="v" id="k-count" style="color:var(--cyan)">…</div><div class="d" id="k-split">scanning…</div></div>
      <div class="stat"><div class="k">Avg Quality</div><div class="v" id="k-q">…</div><div class="d">across current signals</div></div>
      <div class="stat"><div class="k">Universe</div><div class="v" id="k-uni">…</div><div class="d">$1–50 · $50M–10B</div></div>
      <div class="stat"><div class="k">Data</div><div class="v" id="k-data">…</div><div class="d" id="k-datad">—</div></div>
    </div>
    <div class="panel">
      <div class="phead"><h3>Top signals by quality</h3><span class="lbl" id="goscan" style="cursor:pointer;color:var(--cyan)">Open scanner →</span></div>
      <div class="tbl"><div class="thead" style="grid-template-columns:64px 1.4fr 90px 70px 76px 54px">
        <div>Signal</div><div>Ticker</div><div class="r">EPS Δ</div><div>Quality</div><div class="r">Price</div><div class="r">Cross</div></div>
        <div id="toprows"><div class="loading">Loading signals…</div></div></div>
    </div>`;
  $("#goscan", root).onclick = () => ctx.navigate("scanner");

  // 1) meta is fast — fill freshness + universe immediately
  api.meta().then(m => {
    const fresh = m.is_fresh;
    $("#k-uni", root).textContent = m.universe_size.toLocaleString();
    $("#k-data", root).innerHTML = `<span style="color:${fresh ? "var(--green)" : "var(--amber)"}">${fresh ? "Fresh" : "Stale"}</span>`;
    $("#k-datad", root).textContent = `prices ${m.latest_price_date || "—"}${m.days_stale != null ? ` · ${m.days_stale}d ago` : ""}`;
  }).catch(() => {});

  // 2) signals may take a moment on a cold cache — fill when ready
  try {
    const sig = await api.signals(DEFAULT_FILTER);
    const avgScore = sig.signals.length ? Math.round(sig.signals.reduce((a, s) => a + s.score, 0) / sig.signals.length) : 0;
    $("#k-count", root).textContent = sig.count;
    $("#k-split", root).textContent = `${sig.bullish} bull · ${sig.bearish} bear`;
    $("#k-q", root).innerHTML = `<span style="color:${qColor(avgScore)}">${avgScore}</span>`;

    const top = sig.signals.slice(0, 6);
    $("#toprows", root).innerHTML = top.map((r, i) => {
      const bull = r.signal_type === "bullish";
      return `<div class="trow" data-i="${i}" style="grid-template-columns:64px 1.4fr 90px 70px 76px 54px">
        <div><span class="dir ${bull ? "bull" : "bear"}">${bull ? "▲ BULL" : "▼ BEAR"}</span></div>
        <div class="tk"><b>${r.ticker}</b><span class="co">${r.name || ""}</span></div>
        <div class="eps ${cls(r.eps_change_pct)}">${pct(r.eps_change_pct)}</div>
        <div class="q"><div class="qtop"><span class="qv" style="color:${qColor(r.score)}">${r.score}</span><span class="qt">/100</span></div></div>
        <div class="r"><div class="price">${money(r.latest_close)}</div></div>
        <div class="ago">${crossLabel(r.trend_change_date)}</div></div>`;
    }).join("") || `<div class="loading">No recent signals</div>`;

    $("#toprows", root).onclick = e => {
      const t = e.target.closest(".trow"); if (!t) return;
      const r = top[+t.dataset.i]; state.signal = r;
      ctx.navigate("ticker", { sym: r.ticker, cross: r.trend_change_date, dir: r.signal_type });
    };
  } catch (e) {
    $("#toprows", root).innerHTML = `<div class="loading">Error: ${e.message}</div>`;
    $("#k-count", root).textContent = "—"; $("#k-split", root).textContent = "error";
  }
}
