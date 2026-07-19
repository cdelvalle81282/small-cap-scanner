import { api } from "../api.js";
import { $, pct, money, cls, mdLite, crossLabel } from "../util.js";

const HZ = [15, 30, 60, 90];

async function waitForCharts(ms = 3000) {
  const t0 = Date.now();
  while (!window.LightweightCharts && Date.now() - t0 < ms) await new Promise(r => setTimeout(r, 50));
  return !!window.LightweightCharts;
}

export async function render(root, ctx, params) {
  const sym = (params.sym || "").toUpperCase();
  ctx.setCrumb("Detail · " + sym);
  root.innerHTML = `<span class="dt-back" id="back">← Back to scanner</span><div class="loading">Loading ${sym}…</div>`;
  $("#back", root).onclick = () => ctx.navigate("scanner");

  let d;
  try { d = await api.ticker(sym, { cross_date: params.cross, direction: params.dir || "bullish" }); }
  catch (e) { root.innerHTML = `<div class="loading">Error: ${e.message}</div>`; return; }

  const stock = d.stock || {};
  const last = d.prices.length ? d.prices[d.prices.length - 1].c : null;
  const ff = d.follow_through && d.follow_through.forward_returns;
  const ai = d.ai_analysis;

  root.innerHTML = `
    <span class="dt-back" id="back">← Back to scanner</span>
    <div class="dt-head"><h1>${sym}</h1><span class="co">${stock.name || ""}</span>
      <span class="price" style="margin-left:auto;font-size:20px">${money(last)}</span></div>

    <div class="two" style="margin-bottom:14px">
      <div class="panel" style="grid-column:1 / -1"><div class="phead"><h3>Price · SMA overlay</h3>
        ${params.cross ? `<span class="lbl">signal cross ${crossLabel(params.cross)}</span>` : ""}</div>
        <div id="chart"></div></div>
    </div>

    <div class="two">
      <div class="panel"><div class="phead"><h3>Follow-through</h3></div>
        <div class="ft-cards" style="grid-template-columns:repeat(4,1fr);gap:10px;padding:12px">
          ${HZ.map(h => {
            const v = ff ? ff[h] : null;
            return `<div class="hcard" style="padding:12px 10px"><div class="h">+${h}d</div>
              <div class="win" style="font-size:22px;color:${v == null ? "var(--t3)" : (v >= 0 ? "var(--green)" : "var(--red)")}">${v == null ? "·" : pct(v, 1)}</div></div>`;
          }).join("")}
        </div>
        ${ff ? "" : `<div style="padding:0 14px 12px;color:var(--t3);font-family:var(--mono);font-size:11px">Open from a signal to see its realized returns.</div>`}
      </div>
      <div class="panel"><div class="phead"><h3>AI chart analysis</h3>${ai ? `<span class="lbl">saved ${ai.analysis_date || ""}</span>` : ""}</div>
        <div style="padding:12px 15px">${ai && ai.text ? `<div class="ai-text">${mdLite(ai.text)}</div>` : `<div style="color:var(--t3);font-family:var(--mono);font-size:11px">No saved AI analysis for ${sym}. (Live "Analyze with AI" wires in a later phase.)</div>`}</div>
      </div>
    </div>`;
  $("#back", root).onclick = () => ctx.navigate("scanner");

  // chart
  if (await waitForCharts() && d.prices.length) {
    drawChart($("#chart", root), d.prices, params.cross);
  } else if (d.prices.length) {
    $("#chart", root).innerHTML = `<div class="loading">Chart library unavailable</div>`;
  }
}

function sma(prices, period, key = "c") {
  const out = []; let sum = 0;
  for (let i = 0; i < prices.length; i++) {
    sum += prices[i][key];
    if (i >= period) sum -= prices[i - period][key];
    out.push(i >= period - 1 ? { time: prices[i].date, value: +(sum / period).toFixed(4) } : null);
  }
  return out.filter(Boolean);
}

function drawChart(container, prices, cross) {
  const LC = window.LightweightCharts;
  const chart = LC.createChart(container, {
    height: 340, autoSize: true,
    layout: { background: { color: "transparent" }, textColor: "#93a4bd", fontFamily: "ui-monospace, monospace", fontSize: 10 },
    grid: { vertLines: { color: "rgba(255,255,255,.03)" }, horzLines: { color: "rgba(255,255,255,.05)" } },
    rightPriceScale: { borderColor: "#1d2c44" },
    timeScale: { borderColor: "#1d2c44" },
    crosshair: { mode: 0 },
  });
  const candles = chart.addCandlestickSeries({
    upColor: "#10b981", downColor: "#f43f5e", borderVisible: false, wickUpColor: "#10b981", wickDownColor: "#f43f5e",
  });
  candles.setData(prices.map(p => ({ time: p.date, open: p.o, high: p.h, low: p.l, close: p.c })));

  const vol = chart.addHistogramSeries({ priceScaleId: "", priceFormat: { type: "volume" } });
  chart.priceScale("").applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
  vol.setData(prices.map(p => ({ time: p.date, value: p.v, color: p.c >= p.o ? "rgba(16,185,129,.4)" : "rgba(244,63,94,.4)" })));

  chart.addLineSeries({ color: "#f59e0b", lineWidth: 1, priceLineVisible: false, lastValueVisible: false }).setData(sma(prices, 20));
  chart.addLineSeries({ color: "#22d3ee", lineWidth: 1, priceLineVisible: false, lastValueVisible: false }).setData(sma(prices, 50));

  if (cross) {
    candles.setMarkers([{ time: cross, position: "aboveBar", color: "#a78bfa", shape: "arrowDown", text: "cross" }]);
  }
  chart.timeScale().fitContent();
}
