"""
Exit strategy comparison for C17 signals.

Tests whether looser, trend-following exits outperform the current tight stop
(close < prev_day_low) which stops out 98.6% of trades, many within 1-2 days.

Strategies compared (all share the same entry: next-day midpoint):
  1. baseline    - close < prev_day_low (current system)
  2. atr_0.5x    - D1: prev_low stop; D2+: ATR(14) trailing at 0.5x
  3. atr_1x      - D1: prev_low stop; D2+: ATR(14) trailing at 1.0x
  4. atr_2x      - D1: prev_low stop; D2+: ATR(14) trailing at 2.0x
  5. sma5        - exit when close < 5-day SMA
  6. sma10       - exit when close < 10-day SMA
  7. sma20       - exit when close < 20-day SMA
  8. ma_5_10     - exit when 5-day SMA crosses below 10-day SMA
  9. ma_5_20     - exit when 5-day SMA crosses below 20-day SMA
 10. ma_10_20    - exit when 10-day SMA crosses below 20-day SMA
 11. time_5      - hold 5 trading days, no stop
 12. time_10     - hold 10 trading days, no stop
 13. time_15     - hold 15 trading days, no stop

Max hold for all strategies: 30 trading days.
"""

import sys
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, median

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import DB_PATH, ScannerConfig, BacktestConfig
from core.database import Database
from core.backtest import Backtester
from analysis_winners_losers import enrich_deep, filt_eps_trend as filt_c17

MAX_HOLD = 30          # trading days cap for all strategies
LOOKBACK = 120         # calendar days of history before signal for MA seeding
FORWARD = 60           # calendar days after signal to fetch


# ── Data helpers ─────────────────────────────────────────────────────────────

def get_price_df(db, ticker, signal_date):
    signal_dt = date.fromisoformat(signal_date)
    fetch_start = (signal_dt - timedelta(days=LOOKBACK)).isoformat()
    fetch_end = (signal_dt + timedelta(days=FORWARD)).isoformat()
    rows = db.get_daily_prices(ticker, fetch_start, fetch_end)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def compute_atr(df, period=14):
    d = df.copy()
    d["prev_close"] = d["close"].shift(1)
    d["tr"] = d.apply(lambda r: max(
        r["high"] - r["low"],
        abs(r["high"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
        abs(r["low"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
    ), axis=1)
    return d["tr"].rolling(period).mean()


def trade_result(entry_price, exit_price, direction):
    if direction == "bullish":
        return (exit_price - entry_price) / entry_price * 100
    return (entry_price - exit_price) / entry_price * 100


# ── Exit strategies ───────────────────────────────────────────────────────────

def sim_baseline(df, entry_idx, entry_price, direction):
    """Current system: exit if close < prev_day_low (long) or > prev_day_high (short)."""
    for offset in range(1, MAX_HOLD + 1):
        i = entry_idx + offset
        if i >= len(df):
            break
        prev = df.iloc[i - 1]
        row = df.iloc[i]
        if direction == "bullish" and row["close"] < prev["low"]:
            return {"return": trade_result(entry_price, row["close"], direction),
                    "hold": offset, "exit": "stop"}
        if direction == "bearish" and row["close"] > prev["high"]:
            return {"return": trade_result(entry_price, row["close"], direction),
                    "hold": offset, "exit": "stop"}
    last = df.iloc[min(entry_idx + MAX_HOLD, len(df) - 1)]
    return {"return": trade_result(entry_price, last["close"], direction),
            "hold": MAX_HOLD, "exit": "horizon"}


def sim_atr(df, atr_series, entry_idx, entry_price, direction, atr_mult):
    """D1: prev_low/prev_high stop. D2+: ATR trailing."""
    if direction == "bullish":
        trail_anchor = entry_price
    else:
        trail_anchor = entry_price

    for offset in range(1, MAX_HOLD + 1):
        i = entry_idx + offset
        if i >= len(df):
            break
        prev = df.iloc[i - 1]
        row = df.iloc[i]
        close = row["close"]
        atr_val = atr_series.iloc[i - 1] if i - 1 < len(atr_series) else None
        if atr_val is None or pd.isna(atr_val):
            atr_val = (df["high"].iloc[max(0, i - 14):i].mean() -
                       df["low"].iloc[max(0, i - 14):i].mean())

        if offset == 1:
            stop = (direction == "bullish" and close < prev["low"]) or \
                   (direction == "bearish" and close > prev["high"])
        else:
            if direction == "bullish":
                trail_anchor = max(trail_anchor, close)
                stop = close < trail_anchor - atr_mult * atr_val
            else:
                trail_anchor = min(trail_anchor, close)
                stop = close > trail_anchor + atr_mult * atr_val

        if stop:
            return {"return": trade_result(entry_price, close, direction),
                    "hold": offset, "exit": "stop"}

    last = df.iloc[min(entry_idx + MAX_HOLD, len(df) - 1)]
    return {"return": trade_result(entry_price, last["close"], direction),
            "hold": MAX_HOLD, "exit": "horizon"}


def sim_sma_support(df, entry_idx, entry_price, direction, period):
    """Exit when close crosses below (long) or above (short) the N-day SMA."""
    df = df.copy()
    df[f"sma{period}"] = df["close"].rolling(period).mean()

    for offset in range(1, MAX_HOLD + 1):
        i = entry_idx + offset
        if i >= len(df):
            break
        sma = df[f"sma{period}"].iloc[i]
        close = df["close"].iloc[i]
        if pd.isna(sma):
            continue
        if direction == "bullish" and close < sma:
            return {"return": trade_result(entry_price, close, direction),
                    "hold": offset, "exit": "stop"}
        if direction == "bearish" and close > sma:
            return {"return": trade_result(entry_price, close, direction),
                    "hold": offset, "exit": "stop"}

    last = df.iloc[min(entry_idx + MAX_HOLD, len(df) - 1)]
    return {"return": trade_result(entry_price, last["close"], direction),
            "hold": MAX_HOLD, "exit": "horizon"}


def sim_ma_cross(df, entry_idx, entry_price, direction, fast, slow):
    """Exit when fast MA crosses below slow MA (long) or above (short)."""
    df = df.copy()
    df["fast_ma"] = df["close"].rolling(fast).mean()
    df["slow_ma"] = df["close"].rolling(slow).mean()

    for offset in range(1, MAX_HOLD + 1):
        i = entry_idx + offset
        if i >= len(df) or i < 1:
            break
        fast_prev = df["fast_ma"].iloc[i - 1]
        slow_prev = df["slow_ma"].iloc[i - 1]
        fast_cur = df["fast_ma"].iloc[i]
        slow_cur = df["slow_ma"].iloc[i]
        if any(pd.isna(v) for v in [fast_prev, slow_prev, fast_cur, slow_cur]):
            continue

        if direction == "bullish" and fast_prev >= slow_prev and fast_cur < slow_cur:
            close = df["close"].iloc[i]
            return {"return": trade_result(entry_price, close, direction),
                    "hold": offset, "exit": "stop"}
        if direction == "bearish" and fast_prev <= slow_prev and fast_cur > slow_cur:
            close = df["close"].iloc[i]
            return {"return": trade_result(entry_price, close, direction),
                    "hold": offset, "exit": "stop"}

    last = df.iloc[min(entry_idx + MAX_HOLD, len(df) - 1)]
    return {"return": trade_result(entry_price, last["close"], direction),
            "hold": MAX_HOLD, "exit": "horizon"}


def sim_time(df, entry_idx, entry_price, direction, hold_days):
    """No stop — hold exactly N trading days."""
    i = entry_idx + hold_days
    if i >= len(df):
        i = len(df) - 1
    close = df["close"].iloc[i]
    actual_hold = i - entry_idx
    return {"return": trade_result(entry_price, close, direction),
            "hold": actual_hold, "exit": "horizon"}


# ── Summary stats ─────────────────────────────────────────────────────────────

def stats(results):
    returns = [r["return"] for r in results if r is not None]
    if not returns:
        return None
    winners = [r for r in returns if r > 0]
    losers  = [r for r in returns if r <= 0]
    n = len(returns)
    wr = len(winners) / n * 100
    avg_w = mean(winners) if winners else 0
    avg_l = mean(losers) if losers else 0
    gross_l = abs(sum(r for r in losers if r < 0))
    pf = sum(winners) / gross_l if gross_l > 0 else float("inf")
    exp = (len(winners) / n) * avg_w + (len(losers) / n) * avg_l
    avg_hold = mean(r["hold"] for r in results if r is not None)
    stop_pct = sum(1 for r in results if r and r["exit"] == "stop") / n * 100
    return {
        "n": n, "wr": wr, "pf": pf, "exp": exp,
        "avg_w": avg_w, "avg_l": avg_l, "med": median(returns),
        "avg_hold": avg_hold, "stop_pct": stop_pct,
    }


def print_table(all_stats):
    header = f"{'Strategy':<14} {'n':>4} {'WR%':>6} {'PF':>6} {'Exp%':>6} {'AvgW%':>7} {'AvgL%':>7} {'Med%':>7} {'Hold':>5} {'Stop%':>6}"
    print(header)
    print("-" * len(header))
    for name, s in all_stats.items():
        if s is None:
            print(f"{name:<14}  no data")
            continue
        pf_str = f"{s['pf']:.2f}" if s['pf'] != float("inf") else "  inf"
        print(f"{name:<14} {s['n']:>4} {s['wr']:>6.1f} {pf_str:>6} {s['exp']:>6.2f}"
              f" {s['avg_w']:>7.1f} {s['avg_l']:>7.1f} {s['med']:>7.2f}"
              f" {s['avg_hold']:>5.1f} {s['stop_pct']:>6.1f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    db = Database(DB_PATH)
    sc = ScannerConfig()
    bc = BacktestConfig(start_date="2022-01-01", end_date="2026-06-09")

    print("Running backtest and applying C17 filter...")
    bt = Backtester(db, sc, bc)
    results = bt.run()
    enriched = enrich_deep(db, results["signals"])
    c17 = filt_c17(enriched)
    print(f"C17 signals: {len(c17)}\n")

    strategies = {
        "baseline":   [],
        "atr_0.5x":   [],
        "atr_1x":     [],
        "atr_2x":     [],
        "sma5":       [],
        "sma10":      [],
        "sma20":      [],
        "ma_5_10":    [],
        "ma_5_20":    [],
        "ma_10_20":   [],
        "time_5":     [],
        "time_10":    [],
        "time_15":    [],
    }

    skipped = 0
    for sig in c17:
        ticker = sig["ticker"]
        signal_date = sig["trend_date"]
        direction = sig["signal_type"]
        entry_price = sig["entry_price"]

        df = get_price_df(db, ticker, signal_date)
        if df is None or len(df) < 30:
            skipped += 1
            continue

        # Find entry index (first trading day after signal date)
        signal_dt = pd.Timestamp(signal_date)
        after = df[df["date"] > signal_dt]
        if after.empty:
            skipped += 1
            continue
        entry_idx = after.index[0]

        # Use actual midpoint as entry price (consistent with backtester)
        entry_row = df.iloc[entry_idx]
        ep = (entry_row["high"] + entry_row["low"]) / 2
        if ep <= 0:
            skipped += 1
            continue

        atr_series = compute_atr(df)

        strategies["baseline"].append(sim_baseline(df, entry_idx, ep, direction))
        strategies["atr_0.5x"].append(sim_atr(df, atr_series, entry_idx, ep, direction, 0.5))
        strategies["atr_1x"].append(sim_atr(df, atr_series, entry_idx, ep, direction, 1.0))
        strategies["atr_2x"].append(sim_atr(df, atr_series, entry_idx, ep, direction, 2.0))
        strategies["sma5"].append(sim_sma_support(df, entry_idx, ep, direction, 5))
        strategies["sma10"].append(sim_sma_support(df, entry_idx, ep, direction, 10))
        strategies["sma20"].append(sim_sma_support(df, entry_idx, ep, direction, 20))
        strategies["ma_5_10"].append(sim_ma_cross(df, entry_idx, ep, direction, 5, 10))
        strategies["ma_5_20"].append(sim_ma_cross(df, entry_idx, ep, direction, 5, 20))
        strategies["ma_10_20"].append(sim_ma_cross(df, entry_idx, ep, direction, 10, 20))
        strategies["time_5"].append(sim_time(df, entry_idx, ep, direction, 5))
        strategies["time_10"].append(sim_time(df, entry_idx, ep, direction, 10))
        strategies["time_15"].append(sim_time(df, entry_idx, ep, direction, 15))

    if skipped:
        print(f"(Skipped {skipped} signals — insufficient price data)\n")

    all_stats = {name: stats(results) for name, results in strategies.items()}
    print_table(all_stats)
    print()

    # Highlight best by profit factor and expectancy
    best_pf = max((s for s in all_stats.items() if s[1] and s[1]["pf"] != float("inf")),
                  key=lambda x: x[1]["pf"])
    best_exp = max((s for s in all_stats.items() if s[1]),
                   key=lambda x: x[1]["exp"])
    print(f"Best profit factor : {best_pf[0]} ({best_pf[1]['pf']:.2f})")
    print(f"Best expectancy    : {best_exp[0]} ({best_exp[1]['exp']:.2f}%)")


if __name__ == "__main__":
    main()
