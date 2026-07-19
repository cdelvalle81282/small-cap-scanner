"""FastAPI backend for the Small Cap Scanner web app.

JSON over the existing SQLite + core/ modules. Runs on its own port next to the
Streamlit app during the migration; nginx flips to it once the frontend is ready.

Run locally:  uvicorn api.main:app --reload --port 8600
"""
import json
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import DB_PATH, ScannerConfig
from core.chart_analyzer import analyze_chart, build_signal_chart
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


# ── watchlist ────────────────────────────────────────────────
class WatchReq(BaseModel):
    ticker: str
    signal_type: str
    eps_change_pct: float | None = None
    eps_date: str = ""
    signal_date: str = ""
    fast_ma: int | None = None
    slow_ma: int | None = None


@app.get("/api/watchlist")
def watchlist():
    return {"watchlist": get_db().get_watchlist(active_only=True)}


@app.post("/api/watchlist")
def watch_add(r: WatchReq):
    db = get_db()
    try:
        exp = (date.fromisoformat(r.eps_date) + timedelta(days=30)).isoformat() if r.eps_date \
            else (date.today() + timedelta(days=30)).isoformat()
    except Exception:
        exp = (date.today() + timedelta(days=30)).isoformat()
    ai = db.get_ai_analysis(r.ticker.upper()) or {}
    wid = db.add_to_watchlist({
        "ticker": r.ticker.upper(), "signal_type": r.signal_type, "eps_change_pct": r.eps_change_pct,
        "eps_date": r.eps_date, "signal_date": r.signal_date, "fast_ma": r.fast_ma, "slow_ma": r.slow_ma,
        "levels_json": json.dumps(ai.get("levels", [])), "trend_break_price": ai.get("trend_break_price"),
        "trend_break_condition": ai.get("trend_break_condition", ""), "ai_analysis": ai.get("text", ""),
        "expiry_date": exp, "added_date": date.today().isoformat(),
    })
    return {"id": wid, "expiry_date": exp}


@app.delete("/api/watchlist/{wid}")
def watch_remove(wid: int):
    get_db().remove_from_watchlist(wid)
    return {"ok": True}


# ── trades ───────────────────────────────────────────────────
class TradeReq(BaseModel):
    ticker: str
    direction: str
    entry_date: str
    entry_price: float
    shares: float | None = None
    stop_price: float | None = None
    target_price: float | None = None
    notes: str | None = None
    signal_type: str | None = None
    eps_change_pct: float | None = None
    eps_date: str | None = None


class CloseReq(BaseModel):
    exit_date: str
    exit_price: float


@app.get("/api/trades")
def trades():
    return {"trades": get_db().get_trades()}


@app.post("/api/trades")
def trade_add(r: TradeReq):
    tid = get_db().add_trade({
        "ticker": r.ticker.upper(), "direction": r.direction, "status": "open",
        "entry_date": r.entry_date, "entry_price": r.entry_price, "shares": r.shares,
        "stop_price": r.stop_price, "target_price": r.target_price, "exit_date": None, "exit_price": None,
        "notes": r.notes, "signal_type": r.signal_type, "eps_change_pct": r.eps_change_pct,
        "eps_date": r.eps_date, "added_date": date.today().isoformat(),
    })
    return {"id": tid}


@app.post("/api/trades/{tid}/close")
def trade_close(tid: int, r: CloseReq):
    get_db().close_trade(tid, r.exit_date, r.exit_price)
    return {"ok": True}


@app.delete("/api/trades/{tid}")
def trade_delete(tid: int):
    get_db().delete_trade(tid)
    return {"ok": True}


# ── alerts ───────────────────────────────────────────────────
@app.get("/api/alerts")
def alerts():
    db = get_db()
    today = date.today()
    return {
        "signal_alerts": db.get_signal_alerts((today - timedelta(days=30)).isoformat(), today.isoformat()),
        "price_alerts": db.get_all_alerts(),
    }


# ── live AI analysis ─────────────────────────────────────────
class AnalyzeReq(BaseModel):
    ticker: str
    signal_type: str
    eps_change_pct: float | None = None
    eps_change_date: str
    trend_change_date: str
    fast_ma: int
    slow_ma: int
    days_between: int | None = None


@app.post("/api/analyze")
def analyze(r: AnalyzeReq):
    db = get_db()
    sym = r.ticker.upper()
    rows = db.get_daily_prices(sym, "2020-01-01", date.today().isoformat())
    if not rows:
        return {"error": "no price data for " + sym}
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    sig = r.model_dump()
    sig["ticker"] = sym
    try:
        fig = build_signal_chart(df, sig, db.get_earnings(sym))
        result = analyze_chart(fig, sig)
    except Exception as e:
        return {"error": str(e)}
    db.save_ai_analysis(sym, result, signal_date=r.trend_change_date)
    return result


# ── static frontend (mounted last so /api/* wins) ────────────
_WEB = Path(__file__).parent.parent / "web"
if _WEB.exists():
    app.mount("/", StaticFiles(directory=str(_WEB), html=True), name="web")
