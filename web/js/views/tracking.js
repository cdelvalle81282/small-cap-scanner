import { api } from "../api.js";
import { $, pct, money, cls, crossLabel } from "../util.js";

let tab = "watchlist";

export async function render(root, ctx) {
  ctx.setCrumb("Tracking");
  root.innerHTML = `
    <div class="panel">
      <div class="phead" style="gap:0">
        <div class="sorts" id="tabs">
          <span class="chip ${tab === "watchlist" ? "on" : ""}" data-t="watchlist">Watchlist</span>
          <span class="chip ${tab === "trades" ? "on" : ""}" data-t="trades">Trades</span>
          <span class="chip ${tab === "alerts" ? "on" : ""}" data-t="alerts">Alerts</span>
        </div>
      </div>
      <div id="tab-body"><div class="loading">Loading…</div></div>
    </div>`;
  $("#tabs", root).onclick = e => { const c = e.target.closest(".chip"); if (!c) return; tab = c.dataset.t; render(root, ctx); };
  paint(root, ctx);
}

async function paint(root, ctx) {
  const body = $("#tab-body", root);
  try {
    if (tab === "watchlist") return renderWatch(body);
    if (tab === "trades") return renderTrades(body);
    return renderAlerts(body);
  } catch (e) { body.innerHTML = `<div class="loading">Error: ${e.message}</div>`; }
}

async function renderWatch(body) {
  const { watchlist } = await api.watchlist();
  if (!watchlist.length) { body.innerHTML = `<div class="loading">Watchlist empty — add signals from the scanner.</div>`; return; }
  body.innerHTML = `<div style="overflow-x:auto"><table class="ft-tbl">
    <thead><tr><th class="l">Ticker</th><th class="l">Signal</th><th>EPS Δ</th><th>Added</th><th>Expires</th><th>Break level</th><th></th></tr></thead>
    <tbody>${watchlist.map(w => {
      const bull = w.signal_type === "bullish";
      return `<tr><td class="l"><b>${w.ticker}</b></td>
        <td class="l"><span class="dir ${bull ? "bull" : "bear"}">${bull ? "▲" : "▼"} ${w.fast_ma || ""}/${w.slow_ma || ""}</span></td>
        <td class="${cls(w.eps_change_pct)}">${pct(w.eps_change_pct)}</td>
        <td>${crossLabel(w.added_date)}</td><td>${crossLabel(w.expiry_date)}</td>
        <td>${w.trend_break_price != null ? money(w.trend_break_price) : "—"}</td>
        <td><span class="chip" data-rm="${w.id}" style="cursor:pointer">Remove</span></td></tr>`;
    }).join("")}</tbody></table></div>`;
  body.onclick = async e => { const r = e.target.closest("[data-rm]"); if (!r) return; await api.watchRemove(+r.dataset.rm); renderWatch(body); };
}

async function renderTrades(body) {
  const { trades } = await api.trades();
  const pnl = t => t.status === "closed" && t.exit_price && t.entry_price
    ? (t.direction === "short" ? (t.entry_price - t.exit_price) : (t.exit_price - t.entry_price)) / t.entry_price * 100 : null;
  body.innerHTML = `
    <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:end;padding:12px 14px;border-bottom:1px solid var(--border)">
      <div><span class="lbl">Ticker</span><input class="inp" id="t_tk" style="width:80px"></div>
      <div><span class="lbl">Dir</span><div class="seg" id="t_dir" style="width:120px"><button class="on" data-v="long">Long</button><button data-v="short">Short</button></div></div>
      <div><span class="lbl">Entry $</span><input class="inp" id="t_px" style="width:80px"></div>
      <div><span class="lbl">Shares</span><input class="inp" id="t_sh" style="width:80px"></div>
      <div><span class="lbl">Stop</span><input class="inp" id="t_stop" style="width:70px"></div>
      <div><span class="lbl">Target</span><input class="inp" id="t_tgt" style="width:70px"></div>
      <button class="run" id="t_add" style="margin:0;width:auto;padding:8px 16px">+ Log trade</button>
    </div>
    <div style="overflow-x:auto"><table class="ft-tbl">
      <thead><tr><th class="l">Ticker</th><th class="l">Dir</th><th>Entry</th><th>Shares</th><th class="l">Status</th><th>Exit</th><th>P&L</th><th></th></tr></thead>
      <tbody>${trades.map(t => {
        const p = pnl(t);
        return `<tr><td class="l"><b>${t.ticker}</b></td><td class="l">${t.direction}</td>
          <td>${money(t.entry_price)}</td><td>${t.shares ?? "—"}</td><td class="l">${t.status}</td>
          <td>${t.exit_price != null ? money(t.exit_price) : "—"}</td>
          <td class="${cls(p)}">${p == null ? "—" : pct(p, 1)}</td>
          <td>${t.status === "open" ? `<span class="chip" data-close="${t.id}" style="cursor:pointer">Close</span> ` : ""}<span class="chip" data-del="${t.id}" style="cursor:pointer">✕</span></td></tr>`;
      }).join("") || `<tr><td colspan="8" style="text-align:center;color:var(--t3);padding:20px">No trades logged</td></tr>`}</tbody></table></div>`;

  const seg = $("#t_dir", body); let dir = "long";
  seg.onclick = e => { const b = e.target.closest("button"); if (!b) return; dir = b.dataset.v; seg.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b)); };
  $("#t_add", body).onclick = async () => {
    const tk = $("#t_tk", body).value.trim(); const px = parseFloat($("#t_px", body).value);
    if (!tk || !px) return;
    await api.tradeAdd({ ticker: tk, direction: dir, entry_date: new Date().toISOString().slice(0, 10), entry_price: px,
      shares: parseFloat($("#t_sh", body).value) || null, stop_price: parseFloat($("#t_stop", body).value) || null,
      target_price: parseFloat($("#t_tgt", body).value) || null });
    renderTrades(body);
  };
  body.querySelectorAll("[data-close]").forEach(el => el.onclick = async () => {
    const xp = parseFloat(prompt("Exit price?")); if (!xp) return;
    await api.tradeClose(+el.dataset.close, { exit_date: new Date().toISOString().slice(0, 10), exit_price: xp });
    renderTrades(body);
  });
  body.querySelectorAll("[data-del]").forEach(el => el.onclick = async () => {
    if (!confirm("Delete this trade?")) return; await api.tradeDelete(+el.dataset.del); renderTrades(body);
  });
}

async function renderAlerts(body) {
  const { signal_alerts, price_alerts } = await api.alerts();
  const sa = signal_alerts.map(a => {
    const bull = a.signal_type === "bullish";
    return `<tr><td class="l"><b>${a.ticker}</b></td><td class="l"><span class="dir ${bull ? "bull" : "bear"}">${bull ? "▲" : "▼"}</span></td>
      <td class="${cls(a.eps_change_pct)}">${pct(a.eps_change_pct)}</td><td>${crossLabel(a.cross_date)}</td><td>${crossLabel(a.alert_date)}</td></tr>`;
  }).join("");
  body.innerHTML = `
    ${price_alerts.length ? `<div style="padding:12px 14px;border-bottom:1px solid var(--border)"><div class="lbl" style="margin-bottom:8px">Price-level breaches</div>
      ${price_alerts.slice(0, 10).map(p => `<div class="mono" style="font-size:12px;color:var(--t2);padding:2px 0">${p.ticker} — ${p.level_label || p.level_type} @ ${money(p.level_price)} · ${crossLabel(p.alert_date)}</div>`).join("")}</div>` : ""}
    <div style="overflow-x:auto"><table class="ft-tbl">
      <thead><tr><th class="l">Ticker</th><th class="l">Dir</th><th>EPS Δ</th><th>Cross</th><th>Alerted</th></tr></thead>
      <tbody>${sa || `<tr><td colspan="5" style="text-align:center;color:var(--t3);padding:20px">No recent signal alerts</td></tr>`}</tbody></table></div>`;
}
