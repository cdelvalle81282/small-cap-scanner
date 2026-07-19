import { api } from "./api.js";
import { $ } from "./util.js";
import * as overview from "./views/overview.js";
import * as scanner from "./views/scanner.js";
import * as performance from "./views/performance.js";
import * as detail from "./views/detail.js";

const ROUTES = { overview, scanner, performance, ticker: detail };
const view = $("#view");

const ctx = {
  setCrumb: (t) => { $("#crumb").textContent = t; },
  navigate(route, params = {}) {
    let hash = "#/" + route;
    if (route === "ticker") {
      hash += "/" + params.sym + "?" + new URLSearchParams({ cross: params.cross || "", dir: params.dir || "bullish" });
    }
    if (location.hash === hash) route_render(); else location.hash = hash;
  },
};

function parseHash() {
  const raw = location.hash.replace(/^#\/?/, "") || "overview";
  const [path, query] = raw.split("?");
  const parts = path.split("/");
  const route = parts[0] || "overview";
  const params = Object.fromEntries(new URLSearchParams(query || ""));
  if (route === "ticker") params.sym = parts[1];
  return { route, params };
}

async function route_render() {
  const { route, params } = parseHash();
  const mod = ROUTES[route] || overview;
  document.querySelectorAll("#nav a").forEach(a => a.classList.toggle("on", a.dataset.route === route));
  view.innerHTML = `<div class="loading">Loading…</div>`;
  try { await mod.render(view, ctx, params); }
  catch (e) { view.innerHTML = `<div class="loading">Error: ${e.message}</div>`; }
}

// ── shell: nav, freshness, kpis, clock ───────────────────────
document.querySelectorAll("#nav a").forEach(a =>
  a.addEventListener("click", () => ctx.navigate(a.dataset.route)));
window.addEventListener("hashchange", route_render);

async function loadShell() {
  try {
    const m = await api.meta();
    const fresh = m.is_fresh;
    $("#fresh").innerHTML = `<div class="pulse ${fresh ? "" : "stale"}"></div>
      <div class="meta"><div class="l">prices ${m.latest_price_date || "—"}</div>
      <div class="s">${fresh ? "fresh" : "stale"} · ${m.universe_size.toLocaleString()} in universe</div></div>`;
    $("#kpis").innerHTML = `
      <div class="kpi accent"><div class="k">Universe</div><div class="v">${m.universe_size.toLocaleString()}</div></div>
      <div class="kpi"><div class="k">Data</div><div class="v" style="color:${fresh ? "var(--green)" : "var(--amber)"};font-size:14px">${fresh ? "FRESH" : "STALE"}</div></div>`;
  } catch { /* shell degrades gracefully */ }
}

function clock() {
  try {
    const t = new Date().toLocaleTimeString("en-US", { timeZone: "America/New_York", hour12: false });
    $("#clock").innerHTML = `${t} <b>ET</b>`;
  } catch { /* ignore */ }
}

loadShell();
clock(); setInterval(clock, 1000);
route_render();
