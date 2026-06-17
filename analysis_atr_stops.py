"""Test hybrid stop: day 1 = prev-day-low, then ATR trailing stop at various multipliers."""

import sys
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import DB_PATH, BacktestConfig, ScannerConfig
from core.backtest import Backtester
from core.database import Database
from analysis_winners_losers import enrich_deep, filt_c17


def compute_atr(prices_df, period=14):
    """Compute ATR from a DataFrame with high, low, close columns."""
    df = prices_df.copy()
    df["prev_close"] = df["close"].shift(1)
    df["tr"] = df.apply(lambda r: max(
        r["high"] - r["low"],
        abs(r["high"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
        abs(r["low"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
    ), axis=1)
    return df["tr"].rolling(period).mean()


def simulate_trade(db, ticker, signal_date, entry_price, direction, atr_mult, max_days=15):
    """Simulate a single trade with hybrid stop.

    Day 1: stop = prev_day_low (current behavior)
    Day 2+: trailing stop = highest_close - atr_mult * ATR

    If atr_mult is None, use current behavior (prev day low) for all days.
    """
    signal_dt = date.fromisoformat(signal_date)
    fetch_start = (signal_dt - timedelta(days=30)).isoformat()
    fetch_end = (signal_dt + timedelta(days=int(max_days * 1.6) + 10)).isoformat()

    rows = db.get_daily_prices(ticker, fetch_start, fetch_end)
    if not rows:
        return None

    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # Compute ATR over the pre-signal period
    atr_series = compute_atr(df)

    # Find signal date index
    signal_idx = df.index[df["date"] == signal_date].tolist()
    if not signal_idx:
        return None
    signal_idx = signal_idx[0]

    # Entry is next day after signal
    entry_idx = signal_idx + 1
    if entry_idx >= len(df):
        return None

    # Walk forward
    highest_close = entry_price
    prev_row = df.iloc[signal_idx]

    for day_num in range(1, max_days + 1):
        cur_idx = signal_idx + day_num
        if cur_idx >= len(df):
            break

        row = df.iloc[cur_idx]
        close = row["close"]

        if day_num == 1:
            # Day 1: use prev day low (current behavior) regardless of atr_mult
            prev_low = prev_row["low"]
            if direction == "bullish" and close < prev_low:
                ret = (close - entry_price) / entry_price * 100
                return {"return": ret, "hold_days": day_num, "exit_reason": "stop_d1", "exit_price": close}
        else:
            if atr_mult is not None:
                # ATR trailing stop
                atr_val = atr_series.iloc[cur_idx - 1] if cur_idx - 1 < len(atr_series) else atr_series.iloc[-1]
                if pd.isna(atr_val):
                    atr_val = df["high"].iloc[max(0, cur_idx-14):cur_idx].mean() - df["low"].iloc[max(0, cur_idx-14):cur_idx].mean()

                stop_level = highest_close - atr_mult * atr_val
                if direction == "bullish" and close < stop_level:
                    ret = (close - entry_price) / entry_price * 100
                    return {"return": ret, "hold_days": day_num, "exit_reason": f"atr_stop", "exit_price": close}
            else:
                # Keep using prev day low (original behavior)
                prev_low = df.iloc[cur_idx - 1]["low"]
                if direction == "bullish" and close < prev_low:
                    ret = (close - entry_price) / entry_price * 100
                    return {"return": ret, "hold_days": day_num, "exit_reason": "stop_pdl", "exit_price": close}

        # Update trailing high
        if close > highest_close:
            highest_close = close

        prev_row = row

    # Held to horizon
    last_idx = min(signal_idx + max_days, len(df) - 1)
    exit_price = df.iloc[last_idx]["close"]
    ret = (exit_price - entry_price) / entry_price * 100
    return {"return": ret, "hold_days": max_days, "exit_reason": "horizon", "exit_price": exit_price}


def main():
    db = Database(DB_PATH)
    db.initialize()

    # Test on multiple signal sets for robustness
    signal_sets = {}

    for tw in [30, 60]:
        sc = ScannerConfig(
            min_price=1.0, max_price=50.0, min_market_cap=50_000_000, max_market_cap=10_000_000_000,
            ma_crossover_pairs=[(20, 50)], eps_change_threshold=10.0, trend_window_days=tw, direction="both",
        )
        bc = BacktestConfig(
            start_date="2022-01-01", end_date="2026-03-19",
            forward_return_days=[5, 10, 15, 30, 60], ma_crossover_pairs=[(20, 50)],
            eps_thresholds=[10.0], trend_windows=[tw],
        )
        result = Backtester(db, sc, bc).run()
        enriched = enrich_deep(db, result["signals"])
        c17 = filt_c17(enriched)
        signal_sets[f"C17_{tw}d"] = (c17, result["signals"])

    atr_mults = [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    for set_name, (c17, raw_signals) in signal_sets.items():
        # Build filter variants
        filters = {
            "C17 base": c17,
            "C17+up>=2": [s for s in c17 if s.get("consec_up_days", 0) >= 2],
            "C17+up>=2+vol>20": [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("vol_acceleration", 0) > 20],
            "C17+up>=2+slope>0": [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("slope_5d", 0) > 0],
        }

        for filt_name, sigs in filters.items():
            if len(sigs) < 3:
                continue

            print("=" * 120)
            print(f"  {set_name} | {filt_name} | {len(sigs)} signals")
            print("=" * 120)

            print(f"\n  {'ATR Mult':>10s} {'Sigs':>5s} {'WR':>6s} {'Avg':>8s} {'AvgW':>8s} {'AvgL':>8s} "
                  f"{'PF':>8s} {'HoldAll':>8s} {'HoldW':>8s} {'HoldL':>8s} {'Horizon':>8s} {'StopD1':>8s}")
            print("  " + "-" * 110)

            for atr_mult in atr_mults:
                label = f"PDL(all)" if atr_mult is None else f"{atr_mult:.1f}x ATR"
                results = []

                for s in sigs:
                    # Find matching raw signal for entry price
                    match = None
                    for rs in raw_signals:
                        if rs["ticker"] == s["ticker"] and rs.get("trend_change_date") == s["trend_date"]:
                            match = rs
                            break
                    if not match or not match.get("entry_price"):
                        continue

                    trade = simulate_trade(
                        db, s["ticker"], s["trend_date"],
                        match["entry_price"], "bullish", atr_mult, max_days=15
                    )
                    if trade:
                        results.append(trade)

                if not results:
                    continue

                rets = [r["return"] for r in results]
                wins = [r for r in results if r["return"] > 0]
                losses = [r for r in results if r["return"] <= 0]
                wr = len(wins) / len(results) * 100
                avg = mean(rets)
                avg_w = mean([r["return"] for r in wins]) if wins else 0
                avg_l = mean([r["return"] for r in losses]) if losses else 0
                gg = sum(r["return"] for r in wins)
                gl = sum(r["return"] for r in losses)
                pf = gg / abs(gl) if gl else float("inf")
                pf_s = f"{pf:.2f}" if pf != float("inf") else "inf"

                hold_all = mean([r["hold_days"] for r in results])
                hold_w = mean([r["hold_days"] for r in wins]) if wins else 0
                hold_l = mean([r["hold_days"] for r in losses]) if losses else 0
                horizon_exits = sum(1 for r in results if r["exit_reason"] == "horizon")
                d1_stops = sum(1 for r in results if r["exit_reason"] == "stop_d1")

                print(f"  {label:>10s} {len(results):>5d} {wr:>5.0f}% {avg:>+7.1f}% {avg_w:>+7.1f}% {avg_l:>+7.1f}% "
                      f"{pf_s:>8s} {hold_all:>7.1f}d {hold_w:>7.1f}d {hold_l:>7.1f}d {horizon_exits:>7d}  {d1_stops:>7d}")

            # Show individual trades for best combo
            if filt_name == "C17+up>=2+vol>20":
                for atr_mult in atr_mults:
                    label = f"PDL(all)" if atr_mult is None else f"{atr_mult:.1f}x ATR"
                    print(f"\n  --- {label} trades ---")
                    for s in sorted(sigs, key=lambda x: x["trend_date"]):
                        match = None
                        for rs in raw_signals:
                            if rs["ticker"] == s["ticker"] and rs.get("trend_change_date") == s["trend_date"]:
                                match = rs
                                break
                        if not match or not match.get("entry_price"):
                            continue
                        trade = simulate_trade(
                            db, s["ticker"], s["trend_date"],
                            match["entry_price"], "bullish", atr_mult, max_days=15
                        )
                        if trade:
                            wl = "W" if trade["return"] > 0 else "L"
                            print(f"      {s['ticker']:7s} {s['trend_date']}  {trade['return']:>+7.1f}%  {wl}  "
                                  f"hold={trade['hold_days']:>2d}d  exit={trade['exit_reason']}")
            print()


if __name__ == "__main__":
    main()
