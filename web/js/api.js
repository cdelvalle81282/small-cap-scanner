// thin API client over the FastAPI backend

async function get(path, params) {
  const url = new URL(path, location.origin);
  if (params) for (const [k, v] of Object.entries(params)) if (v != null && v !== "") url.searchParams.set(k, v);
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  return r.json();
}
async function send(path, method, body) {
  const r = await fetch(path, { method, headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  return r.json();
}

export const api = {
  meta: () => get("/api/meta"),
  signals: (f) => get("/api/signals", f),
  performance: (f) => get("/api/performance", f),
  ticker: (sym, f) => get(`/api/ticker/${sym}`, f),
  news: (sym) => get(`/api/news/${sym}`),
  analyze: (body) => send("/api/analyze", "POST", body),
  watchlist: () => get("/api/watchlist"),
  watchAdd: (body) => send("/api/watchlist", "POST", body),
  watchRemove: (id) => send(`/api/watchlist/${id}`, "DELETE"),
  trades: () => get("/api/trades"),
  tradeAdd: (body) => send("/api/trades", "POST", body),
  tradeClose: (id, body) => send(`/api/trades/${id}/close`, "POST", body),
  tradeDelete: (id) => send(`/api/trades/${id}`, "DELETE"),
  alerts: () => get("/api/alerts"),
};
