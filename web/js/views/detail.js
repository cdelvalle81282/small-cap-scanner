import { api } from "../api.js";
import { state } from "../state.js";
import { $, pct, money, cls, mdLite, crossLabel } from "../util.js";

const HZ = [15, 30, 60, 90];
let chartRef = { candles: null, lines: [] };

async function waitForCharts(ms = 3000) {
  const t0 = Date.now();
  while (!window.LightweightCharts && Date.now() - t0 < ms) await new Promise(r => setTimeout(r, 50));
  return !!window.LightweightCharts;
}

function aiBody(ai, sig, sym) {
  if (ai && ai.text) return `<div class="ai-text">${mdLite(ai.text)}</div>`;
  return `<div style="color:var(--t3);font-family:var(--mono);font-size:12px">No saved AI analysis${sig ? " yet — click Analyze." : ` for ${sym}.`}</div>`;
}

export async function render(root, ctx, params) {
  const sym = (params.sym || "").toUpperCase();
  ctx.setCrumb("Detail · " + sym);
  root.innerHTML = `<span class="dt-back" id="back">← Back</span><div class="loading">Loading ${sym}…</div>`;
  $("#back", root).onclick = () => ctx.navigate("scanner");

  let d;
  try { d = await api.ticker(sym, { cross_date: params.cross, direction: params.dir || "bullish" }); }
  catch (e) { root.innerHTML = `<div class="loading">Error: ${e.message}</div>`; return; }

  const stock = d.stock || {};
  const last = d.prices.length ? d.prices[d.prices.length - 1].c : null;
  const ff = d.follow_through && d.follow_through.forward_returns;
  const ai = d.ai_analysis;
  const sig = (state.signal && state.signal.ticker === sym) ? state.signal : null;
  const showFT = !!ff;

  const ftPanel = showFT ? `
    <div class="panel"><div class="phead"><h3>Follow-through · realized return</h3></div>
      <div class="ft-cards" style="grid-template-columns:repeat(4,1fr);gap:10px;padding:14px">
        ${HZ.map(h => { const v = ff[h];
          return `<div class="hcard" style="padding:16px 10px"><div class="h">+${h}d</div>
            <div class="win" style="font-size:26px;color:${v == null ? "var(--t3)" : (v >= 0 ? "var(--green)" : "var(--red)")}">${v == null ? "·" : pct(v, 1)}</div></div>`;
        }).join("")}
      </div></div>` : "";

  root.innerHTML = `
    <span class="dt-back" id="back">← Back</span>
    <div class="dt-head"><h1>${sym}</h1><span class="co">${stock.name || ""}</span>
      <span class="price" style="margin-left:auto;font-size:22px">${money(last)}</span>
      ${sig ? `<span class="chip" id="watch" style="cursor:pointer">☆ Watch 30d</span>` : ""}</div>

    <div class="panel" style="margin-bottom:14px">
      <div class="phead"><h3>Price &amp; annotations</h3>
        <span class="legend"><i style="background:#f59e0b"></i>SMA20 <i style="background:#22d3ee"></i>SMA50 <i style="background:#ef4444"></i>SMA200 <i style="background:#a78bfa"></i>EPS <i style="background:#c084fc"></i>cross</span></div>
      <div id="chart"></div>
      <div id="levels-note"></div>
    </div>

    <div class="two">
      ${ftPanel}
      <div class="panel" ${showFT ? "" : 'style="grid-column:1 / -1"'}>
        <div class="phead"><h3>AI chart analysis</h3>
          ${sig ? `<span class="chip on" id="analyze" style="cursor:pointer">${ai ? "↻ Re-run" : "✦ Analyze"}</span>` : (ai ? `<span class="lbl">saved ${ai.analysis_date || ""}</span>` : "")}</div>
        <div style="padding:14px 16px" id="ai-body">${aiBody(ai, sig, sym)}</div>
      </div>
    </div>`;
  $("#back", root).onclick = () => ctx.navigate("scanner");

  const az = $("#analyze", root);
  if (az) az.onclick = async () => {
    const b = $("#ai-body", root); az.textContent = "…";
    b.innerHTML = `<div class="loading">Analyzing chart with Claude Sonnet 5 — ~20s…</div>`;
    try {
      const res = await api.analyze({ ticker: sym, signal_type: sig.signal_type, eps_change_pct: sig.eps_change_pct,
        eps_change_date: sig.eps_change_date || "", trend_change_date: sig.trend_change_date,
        fast_ma: sig.fast_ma, slow_ma: sig.slow_ma, days_between: sig.days_between });
      if (res.error) { b.innerHTML = `<div style="color:var(--red);font-family:var(--mono);font-size:12px">Error: ${res.error}</div>`; }
      else { b.innerHTML = `<div class="ai-text">${mdLite(res.text)}</div>`; applyLevels(root, res.levels || [], res.trend_break_price); }
    } catch (e) { b.innerHTML = `<div style="color:var(--red)">Error: ${e.message}</div>`; }
    az.textContent = "↻ Re-run";
  };
  const wb = $("#watch", root);
  if (wb) wb.onclick = async () => {
    wb.textContent = "…";
    try {
      await api.watchAdd({ ticker: sym, signal_type: sig.signal_type, eps_change_pct: sig.eps_change_pct,
        eps_date: sig.eps_change_date || "", signal_date: sig.trend_change_date, fast_ma: sig.fast_ma, slow_ma: sig.slow_ma });
      wb.textContent = "✓ Watching";
    } catch { wb.textContent = "✕ failed"; }
  };

  if (await waitForCharts() && d.prices.length) {
    drawChart($("#chart", root), d.prices, params.cross, d.earnings || []);
    if (ai && ai.levels && ai.levels.length) applyLevels(root, ai.levels, ai.trend_break_price);
  } else if (d.prices.length) {
    $("#chart", root).innerHTML = `<div class="loading">Chart library unavailable</div>`;
  }
}

function sma(prices, period) {
  const out = []; let sum = 0;
  for (let i = 0; i < prices.length; i++) {
    sum += prices[i].c;
    if (i >= period) sum -= prices[i - period].c;
    if (i >= period - 1) out.push({ time: prices[i].date, value: +(sum / period).toFixed(4) });
  }
  return out;
}

// nearest trading-day bar on or before a target date (EPS often lands off-session)
function snapToBar(dates, target) {
  if (!dates.length) return target;
  if (target <= dates[0]) return dates[0];
  if (target >= dates[dates.length - 1]) return dates[dates.length - 1];
  let lo = 0, hi = dates.length - 1, ans = dates[0];
  while (lo <= hi) { const m = (lo + hi) >> 1; if (dates[m] <= target) { ans = dates[m]; lo = m + 1; } else hi = m - 1; }
  return ans;
}

function drawChart(container, prices, cross, earnings) {
  const LC = window.LightweightCharts;
  container.innerHTML = "";
  const chart = LC.createChart(container, {
    height: 360, autoSize: true,
    layout: { background: { color: "transparent" }, textColor: "#93a4bd", fontFamily: "ui-monospace, monospace", fontSize: 11 },
    grid: { vertLines: { color: "rgba(255,255,255,.03)" }, horzLines: { color: "rgba(255,255,255,.05)" } },
    rightPriceScale: { borderColor: "#1d2c44" }, timeScale: { borderColor: "#1d2c44" }, crosshair: { mode: 0 },
  });
  const candles = chart.addCandlestickSeries({ upColor: "#10b981", downColor: "#f43f5e", borderVisible: false, wickUpColor: "#10b981", wickDownColor: "#f43f5e" });
  candles.setData(prices.map(p => ({ time: p.date, open: p.o, high: p.h, low: p.l, close: p.c })));

  const vol = chart.addHistogramSeries({ priceScaleId: "", priceFormat: { type: "volume" } });
  chart.priceScale("").applyOptions({ scaleMargins: { top: 0.85, bottom: 0 } });
  vol.setData(prices.map(p => ({ time: p.date, value: p.v, color: p.c >= p.o ? "rgba(16,185,129,.35)" : "rgba(244,63,94,.35)" })));

  chart.addLineSeries({ color: "#f59e0b", lineWidth: 1, priceLineVisible: false, lastValueVisible: false }).setData(sma(prices, 20));
  chart.addLineSeries({ color: "#22d3ee", lineWidth: 1, priceLineVisible: false, lastValueVisible: false }).setData(sma(prices, 50));
  chart.addLineSeries({ color: "#ef4444", lineWidth: 2, priceLineVisible: false, lastValueVisible: false }).setData(sma(prices, 200));

  // markers: every earnings report (▲ beat / ▼ miss) + the MA crossover
  const dates = prices.map(p => p.date);
  const markers = [];
  for (const e of earnings) {
    const rd = (e.report_date || "").slice(0, 10);
    if (!rd || rd < dates[0] || rd > dates[dates.length - 1]) continue;
    const chg = e.eps_change_pct;
    markers.push({ time: snapToBar(dates, rd), position: "belowBar", color: "#a78bfa", shape: "circle",
      text: "EPS " + (chg == null ? "" : (chg >= 0 ? "+" : "") + Math.round(chg) + "%") });
  }
  if (cross) markers.push({ time: snapToBar(dates, cross), position: "aboveBar", color: "#c084fc", shape: "arrowDown", text: "MA cross" });
  markers.sort((a, b) => a.time < b.time ? -1 : a.time > b.time ? 1 : 0);
  candles.setMarkers(markers);

  chart.timeScale().fitContent();
  chartRef = { candles, lines: [] };
}

// draw AI support/resistance + trend-break as horizontal price lines (and always
// show them as text under the chart, so they're visible even if a line fails)
function applyLevels(root, levels, trendBreak) {
  const c = chartRef.candles;
  const LS = (window.LightweightCharts && window.LightweightCharts.LineStyle)
    ? window.LightweightCharts.LineStyle.Dashed : 2;
  if (c) { chartRef.lines.forEach(l => { try { c.removePriceLine(l); } catch { /* */ } }); chartRef.lines = []; }

  const addLine = (price, color, width, title) => {
    const p = Number(price);
    if (!isFinite(p) || !c) return;
    try { chartRef.lines.push(c.createPriceLine({ price: p, color, lineWidth: width, lineStyle: LS, axisLabelVisible: true, title })); }
    catch (e) { /* keep going — text note still shows the level */ }
  };

  const note = [];
  for (const lv of (levels || [])) {
    if (lv.price == null) continue;
    const res = (lv.type || "").toLowerCase().startsWith("res");
    addLine(lv.price, res ? "#f59e0b" : "#10b981", 1, `${res ? "R " : "S "}${lv.price}`);
    note.push(`<span style="color:${res ? "var(--amber)" : "var(--green)"}">${res ? "R" : "S"} $${lv.price}</span>`);
  }
  if (trendBreak != null) {
    addLine(trendBreak, "#c084fc", 2, `break ${trendBreak}`);
    note.push(`<span style="color:#c084fc">break $${trendBreak}</span>`);
  }
  const el = $("#levels-note", root);
  if (el) el.innerHTML = note.length ? `AI levels: ${note.join(" · ")}` : "";
}
