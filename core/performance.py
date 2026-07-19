"""Signal follow-through — how triggered tickers actually performed after the cross.

Answers "what happened 15 / 30 / 60 / 90 days after each trigger?" with PURE
hold-to-horizon returns: enter at the trigger-day close, hold exactly N trading
days, measure the return. Direction-aware (a bearish signal "worked" if price
fell). No stop-loss — unlike the Backtest page's strategy, this is meant to show
the raw shape of the move across horizons, so 15d and 90d actually differ.
"""
from datetime import date, timedelta
from statistics import mean

from config import ScannerConfig
from core.database import Database
from core.scanner import Scanner

DEFAULT_HORIZONS = (15, 30, 60, 90)


def _forward_returns(rows: list[dict], cross_date: str, direction: str,
                     horizons: tuple[int, ...]) -> dict | None:
    """Given a ticker's date-ordered price rows, realized returns at each horizon.

    A horizon with no data yet (a recent signal) comes back as None, not dropped —
    the UI shows it as "pending" so recent triggers still appear.
    """
    idx = next((i for i, r in enumerate(rows) if r["date"] == cross_date), None)
    if idx is None:
        return None
    entry = rows[idx].get("close")
    if not entry or entry <= 0:
        return None
    out: dict[int, float | None] = {}
    for h in horizons:
        j = idx + h
        exit_close = rows[j].get("close") if j < len(rows) else None
        if exit_close:
            raw = (exit_close - entry) / entry * 100
            out[h] = round(raw if direction == "bullish" else -raw, 2)
        else:
            out[h] = None
    return {"entry_date": cross_date, "entry_price": round(entry, 4), "forward_returns": out}


def follow_through(
    db: Database,
    scanner_config: ScannerConfig,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    start_date: str = "2018-01-01",
    end_date: str | None = None,
) -> dict:
    """Realized forward returns for every signal the given filter produces.

    Returns {"signals": [...signal + forward_returns...], "summary": {total, by_horizon}}.
    """
    end = end_date or date.today().isoformat()
    signals: list[dict] = []
    raw = Scanner(db, scanner_config).scan(end)
    with db.read_connection() as conn:
        for sig in raw:
            cd = sig.get("trend_change_date")
            if not cd or cd < start_date or cd > end:
                continue
            rows = db.get_price_history(sig["ticker"], conn=conn)
            fr = _forward_returns(rows, cd, sig["signal_type"], horizons)
            if fr is None:
                continue
            signals.append({**sig, **fr})
    return {
        "signals": signals,
        "summary": {
            "total_signals": len(signals),
            "by_horizon": summarize_horizons(signals, horizons),
        },
    }


def ticker_follow_through(
    db: Database,
    ticker: str,
    cross_date: str,
    direction: str,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> dict | None:
    """Forward returns for a single fired signal — for the ticker detail view."""
    rows = db.get_price_history(ticker)
    return _forward_returns(rows, cross_date, direction, horizons)


def relative_volume(db: Database, ticker: str, as_of: str | None = None) -> float | None:
    """Latest session volume vs the trailing 20-session average (RVOL).

    Per-ticker convenience for the detail view. The scanner/API list path should
    compute RVOL in a single batched query instead (avoid N+1) — see
    Database.get_signal_enrichment, which already returns the 20-day average.
    """
    end = as_of or date.today().isoformat()
    rows = db.get_daily_prices(ticker, "2000-01-01", end)
    vols = [r["volume"] for r in rows if r.get("volume") is not None]
    if len(vols) < 21:
        return None
    avg20 = mean(vols[-21:-1])
    if not avg20:
        return None
    return round(vols[-1] / avg20, 2)


def summarize_horizons(
    signals: list[dict], horizons: tuple[int, ...] = DEFAULT_HORIZONS
) -> dict[int, dict]:
    """Aggregate win-rate / avg / median realized return per horizon across signals."""
    out: dict[int, dict] = {}
    for h in horizons:
        rets = [
            s["forward_returns"][h]
            for s in signals
            if s.get("forward_returns") and s["forward_returns"].get(h) is not None
        ]
        if not rets:
            out[h] = {"win_rate": None, "avg_return": None, "median_return": None, "sample": 0}
            continue
        wins = sum(1 for r in rets if r > 0)
        s = sorted(rets)
        mid = len(s) // 2
        med = s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2
        out[h] = {
            "win_rate": round(wins / len(rets) * 100, 1),
            "avg_return": round(mean(rets), 2),
            "median_return": round(med, 2),
            "sample": len(rets),
        }
    return out
