"""FastAPI backend for the Small Cap Scanner web app.

JSON over the existing SQLite + core/ modules. Runs on its own port next to the
Streamlit app during the migration; nginx flips to it once the frontend is ready.

Run locally:  uvicorn api.main:app --reload --port 8600
"""
from datetime import date
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import DB_PATH, ScannerConfig
from core.database import Database
from core.performance import DEFAULT_HORIZONS, follow_through, ticker_follow_through
from core.scanner import Scanner
from core.scoring import score_signal

app = FastAPI(title="Small Cap Scanner API", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_db() -> Database:
    db = Database(DB_PATH)
    db.initialize()
    return db


# ── param helpers ────────────────────────────────────────────
def _cap(v: str | None) -> float | None:
    if not v:
        return None
    s = v.strip().upper().replace(",", "")
    try:
        if s.endswith("B"):
            return float(s[:-1]) * 1e9
        if s.endswith("M"):
            return float(s[:-1]) * 1e6
        return float(s)
    except ValueError:
        return None


def _build_config(
    min_price, max_price, min_cap, max_cap, ma, eps, window, direction,
) -> ScannerConfig:
    fast, slow = (20, 50) if ma == "20/50" else (50, 200)
    return ScannerConfig(
        min_price=min_price if min_price is not None else 0.0,
        max_price=max_price if max_price is not None else 1e6,
        min_market_cap=_cap(min_cap) or 0.0,
        max_market_cap=_cap(max_cap) or 1e14,
        ma_crossover_pairs=[(fast, slow)],
        eps_change_threshold=float(eps),
        trend_window_days=int(window),
        direction=direction,
    )


def _days_ago(iso: str) -> int | None:
    try:
        return (date.today() - date.fromisoformat(iso)).days
    except Exception:
        return None


# ── endpoints ────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/meta")
def meta():
    db = get_db()
    latest = db.get_latest_price_date()
    stale = _days_ago(latest) if latest else None
    universe = db.get_stock_universe(1.0, 50.0, 50_000_000, 10_000_000_000)
    return {
        "latest_price_date": latest,
        "days_stale": stale,
        "is_fresh": stale is not None and stale <= 4,  # tolerate a weekend
        "universe_size": len(universe),
    }


@app.get("/api/signals")
def signals(
    min_price: float | None = None, max_price: float | None = None,
    min_cap: str | None = None, max_cap: str | None = None,
    ma: str = "20/50", eps: float = 10.0, window: int = 30,
    direction: str = "both", recency: int = 30, sort: str = "score",
):
    db = get_db()
    cfg = _build_config(min_price, max_price, min_cap, max_cap, ma, eps, window, direction)
    as_of = date.today().isoformat()
    raw = Scanner(db, cfg).scan(as_of)

    # most-recent signal per ticker, then recency filter on the crossover date
    latest: dict[str, dict] = {}
    for s in raw:
        cd = s.get("trend_change_date")
        if not cd:
            continue
        if s["ticker"] not in latest or cd > latest[s["ticker"]]["trend_change_date"]:
            latest[s["ticker"]] = s
    rows = [s for s in latest.values() if (_days_ago(s["trend_change_date"]) or 999) <= recency]

    tickers = [s["ticker"] for s in rows]
    enrich = db.get_signal_enrichment(tickers)
    sparks = db.get_recent_closes(tickers, 90)

    out = []
    for s in rows:
        e = enrich.get(s["ticker"], {})
        avgv = e.get("avg_volume")
        latv = e.get("latest_volume")
        rvol = round(latv / avgv, 2) if avgv and latv else None
        q = score_signal(
            eps_change_pct=s.get("eps_change_pct"), rvol=rvol,
            trend_aligned=None, days_between=s.get("days_between"),
            trend_window=cfg.trend_window_days,
        )
        out.append({
            "ticker": s["ticker"], "name": e.get("name"),
            "signal_type": s["signal_type"], "fast_ma": s["fast_ma"], "slow_ma": s["slow_ma"],
            "eps_change_pct": s.get("eps_change_pct"), "eps_change_date": s.get("eps_change_date"),
            "trend_change_date": s["trend_change_date"], "days_between": s.get("days_between"),
            "days_ago": _days_ago(s["trend_change_date"]),
            "market_cap": e.get("market_cap"), "latest_close": e.get("latest_close"),
            "avg_dollar_vol": e.get("avg_dollar_vol"), "avg_volume": avgv, "rvol": rvol,
            "score": q["score"], "factors": q["factors"],
            "spark": sparks.get(s["ticker"], []),
        })

    keys = {
        "score": lambda r: r["score"], "cross": lambda r: r["trend_change_date"],
        "eps": lambda r: abs(r.get("eps_change_pct") or 0), "rvol": lambda r: r.get("rvol") or 0,
        "dvol": lambda r: r.get("avg_dollar_vol") or 0,
    }
    out.sort(key=keys.get(sort, keys["score"]), reverse=True)

    bull = sum(1 for r in out if r["signal_type"] == "bullish")
    return {"count": len(out), "bullish": bull, "bearish": len(out) - bull, "signals": out}


@app.get("/api/performance")
def performance(
    min_price: float | None = None, max_price: float | None = None,
    min_cap: str | None = None, max_cap: str | None = None,
    ma: str = "20/50", eps: float = 10.0, window: int = 30,
    direction: str = "both", start: str = "2022-01-01",
):
    db = get_db()
    cfg = _build_config(min_price, max_price, min_cap, max_cap, ma, eps, window, direction)
    res = follow_through(db, cfg, horizons=DEFAULT_HORIZONS, start_date=start)
    # slim per-signal payload for the table
    rows = [{
        "ticker": s["ticker"], "signal_type": s["signal_type"],
        "eps_change_pct": s.get("eps_change_pct"), "cross_date": s["trend_change_date"],
        "entry_price": s.get("entry_price"),
        "forward_returns": s.get("forward_returns", {}),
    } for s in res["signals"]]
    return {"summary": res["summary"], "horizons": list(DEFAULT_HORIZONS), "signals": rows}


@app.get("/api/ticker/{sym}")
def ticker(sym: str, cross_date: str | None = None, direction: str = "bullish"):
    db = get_db()
    sym = sym.upper()
    prices = db.get_daily_prices(sym, "2020-01-01", date.today().isoformat())
    series = [
        {"date": r["date"], "o": r["open"], "h": r["high"], "l": r["low"],
         "c": r["close"], "v": r["volume"]}
        for r in prices
    ]
    ff = ticker_follow_through(db, sym, cross_date, direction) if cross_date else None
    return {
        "ticker": sym,
        "stock": db.get_stock(sym),
        "prices": series,
        "earnings": db.get_earnings(sym),
        "ai_analysis": db.get_ai_analysis(sym),
        "follow_through": ff,
    }


# ── static frontend (mounted last so /api/* wins) ────────────
_WEB = Path(__file__).parent.parent / "web"
if _WEB.exists():
    app.mount("/", StaticFiles(directory=str(_WEB), html=True), name="web")
