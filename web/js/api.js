// thin API client over the FastAPI backend

async function get(path, params) {
  const url = new URL(path, location.origin);
  if (params) for (const [k, v] of Object.entries(params)) if (v != null && v !== "") url.searchParams.set(k, v);
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  return r.json();
}

export const api = {
  meta: () => get("/api/meta"),
  signals: (f) => get("/api/signals", f),
  performance: (f) => get("/api/performance", f),
  ticker: (sym, f) => get(`/api/ticker/${sym}`, f),
};
