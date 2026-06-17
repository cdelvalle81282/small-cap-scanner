"""Test: Enter Day 2 open after green candle, but anchor stop to Day 1 close.

Compare:
  A) Stop = entry_price - X% (stop moves with entry price)
  B) Stop = day1_close - X% (stop anchored to confirmation close)

With (B), gap-up entries have tighter risk, gap-down entries have wider risk.

Also test ATR-based: stop = day1_close - 0.5x ATR vs entry - 0.5x ATR.
"""

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
from analysis_winners_losers import enrich_deep


def compute_atr(df, period=14):
    d = df.copy()
    d["prev_close"] = d["close"].shift(1)
    d["tr"] = d.apply(lambda r: max(
        r["high"] - r["low"],
        abs(r["high"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
        abs(r["low"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
    ), axis=1)
    return d["tr"].rolling(period).mean()


def simulate_anchored(
    db, ticker, signal_date,
    entry_mode="d2_open",         # "d1_close", "d2_open"
    confirm="green_candle",       # "green_candle", "close_above_prior", "none"
    stop_anchor="entry",          # "entry" or "d1_close"
    stop_type="atr",              # "atr" or "pct"
    atr_mult=0.5,                 # for ATR stop
    pct_stop=3.0,                 # for % stop
    max_days=15,
):
    signal_dt = date.fromisoformat(signal_date)
    fetch_start = (signal_dt - timedelta(days=50)).isoformat()
    fetch_end = (signal_dt + timedelta(days=int(max_days * 1.6) + 15)).isoformat()

    rows = db.get_daily_prices(ticker, fetch_start, fetch_end)
    if not rows:
        return None

    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    atr_series = compute_atr(df)

    sig_idx_list = df.index[df["date"] == signal_date].tolist()
    if not sig_idx_list:
        return None
    sig_idx = sig_idx_list[0]

    d0 = df.iloc[sig_idx]
    d1_idx = sig_idx + 1
    d2_idx = sig_idx + 2

    if d1_idx >= len(df):
        return None
    d1 = df.iloc[d1_idx]

    # Confirmation
    if confirm == "green_candle" and d1["close"] <= d1["open"]:
        return None
    if confirm == "close_above_prior" and d1["close"] <= d0["close"]:
        return None

    # Entry
    if entry_mode == "d1_close":
        entry_price = d1["close"]
        entry_idx = d1_idx
    elif entry_mode == "d2_open":
        if d2_idx >= len(df):
            return None
        entry_price = df.iloc[d2_idx]["open"]
        entry_idx = d2_idx
    else:
        return None

    if entry_price <= 0:
        return None

    d1_close = d1["close"]

    # Stop anchor point
    anchor = d1_close if stop_anchor == "d1_close" else entry_price

    # Initial ATR value at entry
    entry_atr = atr_series.iloc[entry_idx] if entry_idx < len(atr_series) else None
    if entry_atr is None or pd.isna(entry_atr):
        entry_atr = (df["high"].iloc[max(0, entry_idx-14):entry_idx].mean() -
                     df["low"].iloc[max(0, entry_idx-14):entry_idx].mean())

    # Calculate initial stop level
    if stop_type == "atr":
        initial_stop = anchor - atr_mult * entry_atr
    else:
        initial_stop = anchor * (1 - pct_stop / 100)

    # Check if we're already below stop at entry
    if entry_price < initial_stop:
        # Entered below stop — immediate exit
        return {
            "return": 0.0, "hold_days": 0, "exit_reason": "below_stop_at_entry",
            "entry_price": entry_price, "exit_price": entry_price,
            "d1_close": d1_close, "gap_pct": (entry_price - d1_close) / d1_close * 100,
            "initial_stop": initial_stop, "risk_pct": (entry_price - initial_stop) / entry_price * 100,
        }

    # Walk forward with trailing stop
    highest_close = max(entry_price, d1_close) if stop_anchor == "d1_close" else entry_price

    for day_offset in range(1, max_days + 1):
        cur_idx = entry_idx + day_offset
        if cur_idx >= len(df):
            break

        row = df.iloc[cur_idx]
        close = row["close"]

        # Current ATR
        atr_val = atr_series.iloc[cur_idx - 1] if cur_idx - 1 < len(atr_series) else entry_atr
        if pd.isna(atr_val):
            atr_val = entry_atr

        # Trailing stop
        if stop_type == "atr":
            stop_level = highest_close - atr_mult * atr_val
        else:
            stop_level = highest_close * (1 - pct_stop / 100)

        if close < stop_level:
            ret = (close - entry_price) / entry_price * 100
            return {
                "return": ret, "hold_days": day_offset, "exit_reason": "trailing_stop",
                "entry_price": entry_price, "exit_price": close,
                "d1_close": d1_close, "gap_pct": (entry_price - d1_close) / d1_close * 100,
                "initial_stop": initial_stop,
                "risk_pct": (entry_price - initial_stop) / entry_price * 100,
            }

        if close > highest_close:
            highest_close = close

    # Horizon
    last_idx = min(entry_idx + max_days, len(df) - 1)
    exit_price = df.iloc[last_idx]["close"]
    ret = (exit_price - entry_price) / entry_price * 100
    return {
        "return": ret, "hold_days": max_days, "exit_reason": "horizon",
        "entry_price": entry_price, "exit_price": exit_price,
        "d1_close": d1_close, "gap_pct": (entry_price - d1_close) / d1_close * 100,
        "initial_stop": initial_stop,
        "risk_pct": (entry_price - initial_stop) / entry_price * 100,
    }


def run_sim(db, signals, raw_signals, **kwargs):
    results = []
    for s in signals:
        match = None
        for rs in raw_signals:
            if rs["ticker"] == s["ticker"] and rs.get("trend_change_date") == s["trend_date"]:
                match = rs
                break
        if not match:
            continue
        trade = simulate_anchored(db, s["ticker"], s["trend_date"], **kwargs)
        if trade:
            trade["ticker"] = s["ticker"]
            trade["signal_date"] = s["trend_date"]
            trade["eps_pct"] = s.get("eps_change_pct", 0)
            results.append(trade)
    return results


def show(label, results, show_trades=False):
    if not results:
        print(f"  {label:85s} | n=  0")
        return
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
    hold = mean([r["hold_days"] for r in results])
    avg_risk = mean([r.get("risk_pct", 0) for r in results])

    print(f"  {label:85s} | n={len(results):>3d} | WR={wr:>5.1f}% | Avg={avg:>+7.1f}% "
          f"| AvgW={avg_w:>+6.1f}% | AvgL={avg_l:>+6.1f}% | PF={pf_s:>6s} "
          f"| Hold={hold:.1f}d | AvgRisk={avg_risk:.1f}%")

    if show_trades:
        for r in sorted(results, key=lambda x: x["signal_date"]):
            wl = "W" if r["return"] > 0 else "L"
            gap = r.get("gap_pct", 0)
            risk = r.get("risk_pct", 0)
            print(f"      {r['ticker']:7s} {r['signal_date']}  "
                  f"d1cl=${r.get('d1_close',0):>7.2f}  "
                  f"entry=${r['entry_price']:>7.2f} (gap{gap:>+5.1f}%)  "
                  f"stop=${r.get('initial_stop',0):>7.2f} (risk={risk:>4.1f}%)  "
                  f"exit=${r['exit_price']:>7.2f}  {r['return']:>+7.1f}%  {wl}  "
                  f"hold={r['hold_days']:>2d}d  {r['exit_reason']}")


def main():
    db = Database(DB_PATH)
    db.initialize()

    configs = [(10.0, 60), (20.0, 45), (20.0, 60), (25.0, 30), (25.0, 60), (30.0, 30), (30.0, 60)]

    signal_cache = {}
    for eps_thresh, tw in configs:
        sc = ScannerConfig(
            min_price=1.0, max_price=50.0, min_market_cap=50_000_000, max_market_cap=10_000_000_000,
            ma_crossover_pairs=[(20, 50)], eps_change_threshold=eps_thresh,
            trend_window_days=tw, direction="both",
        )
        bc = BacktestConfig(
            start_date="2022-01-01", end_date="2026-03-19",
            forward_return_days=[5, 10, 15, 30, 60], ma_crossover_pairs=[(20, 50)],
            eps_thresholds=[eps_thresh], trend_windows=[tw],
        )
        result = Backtester(db, sc, bc).run()
        enriched = enrich_deep(db, result["signals"])
        c17 = [s for s in enriched
               if s["signal_type"] == "bullish"
               and s["eps_change_pct"] > 0
               and s.get("avg_dollar_vol", 0) >= 500_000
               and abs(s["eps_change_pct"]) < 100
               and s["days_between"] > 10]
        signal_cache[(eps_thresh, tw)] = (c17, result["signals"])

    # ═══════════════════════════════════════════════════════════════════
    print("=" * 180)
    print("  STOP ANCHOR COMPARISON: Entry at D2 open, stop based on entry vs D1 close")
    print("=" * 180)

    for eps_thresh, tw in configs:
        c17, raw = signal_cache[(eps_thresh, tw)]
        if len(c17) < 3:
            continue

        print(f"\n{'='*180}")
        print(f"  EPS>={eps_thresh:.0f}% / {tw}d -- {len(c17)} C17 signals")
        print(f"{'='*180}")

        print("\n  --- ATR STOPS ---")
        for atr_m in [0.5, 0.75, 1.0]:
            # D1 close entry, stop from entry (current realistic baseline)
            r = run_sim(db, c17, raw, entry_mode="d1_close", confirm="green_candle",
                        stop_anchor="entry", stop_type="atr", atr_mult=atr_m)
            show(f"D1 close entry | stop from ENTRY | {atr_m:.2f}x ATR", r)

            # D2 open entry, stop from entry
            r = run_sim(db, c17, raw, entry_mode="d2_open", confirm="green_candle",
                        stop_anchor="entry", stop_type="atr", atr_mult=atr_m)
            show(f"D2 open entry  | stop from ENTRY | {atr_m:.2f}x ATR", r)

            # D2 open entry, stop from D1 close
            r = run_sim(db, c17, raw, entry_mode="d2_open", confirm="green_candle",
                        stop_anchor="d1_close", stop_type="atr", atr_mult=atr_m)
            show(f"D2 open entry  | stop from D1 CLOSE | {atr_m:.2f}x ATR", r)
            print()

        print("  --- PERCENTAGE STOPS ---")
        for pct in [2.0, 3.0, 5.0, 7.0]:
            r = run_sim(db, c17, raw, entry_mode="d1_close", confirm="green_candle",
                        stop_anchor="entry", stop_type="pct", pct_stop=pct)
            show(f"D1 close entry | stop from ENTRY | {pct:.0f}% stop", r)

            r = run_sim(db, c17, raw, entry_mode="d2_open", confirm="green_candle",
                        stop_anchor="entry", stop_type="pct", pct_stop=pct)
            show(f"D2 open entry  | stop from ENTRY | {pct:.0f}% stop", r)

            r = run_sim(db, c17, raw, entry_mode="d2_open", confirm="green_candle",
                        stop_anchor="d1_close", stop_type="pct", pct_stop=pct)
            show(f"D2 open entry  | stop from D1 CLOSE | {pct:.0f}% stop", r)
            print()

    # ═══════════════════════════════════════════════════════════════════
    # Also test with close>prior confirmation
    print(f"\n\n{'='*180}")
    print("  SAME WITH close > prior close CONFIRMATION")
    print(f"{'='*180}")

    for eps_thresh, tw in [(25.0, 30), (25.0, 60), (20.0, 60), (30.0, 60)]:
        c17, raw = signal_cache[(eps_thresh, tw)]
        if len(c17) < 3:
            continue

        print(f"\n  --- EPS>={eps_thresh:.0f}% / {tw}d ---")
        for atr_m in [0.5, 0.75]:
            r = run_sim(db, c17, raw, entry_mode="d1_close", confirm="close_above_prior",
                        stop_anchor="entry", stop_type="atr", atr_mult=atr_m)
            show(f"D1 close | close>prior | stop ENTRY | {atr_m:.2f}x ATR", r)

            r = run_sim(db, c17, raw, entry_mode="d2_open", confirm="close_above_prior",
                        stop_anchor="entry", stop_type="atr", atr_mult=atr_m)
            show(f"D2 open  | close>prior | stop ENTRY | {atr_m:.2f}x ATR", r)

            r = run_sim(db, c17, raw, entry_mode="d2_open", confirm="close_above_prior",
                        stop_anchor="d1_close", stop_type="atr", atr_mult=atr_m)
            show(f"D2 open  | close>prior | stop D1 CLOSE | {atr_m:.2f}x ATR", r)
            print()

    # ═══════════════════════════════════════════════════════════════════
    # Trade details for best configs
    print(f"\n\n{'='*180}")
    print("  INDIVIDUAL TRADES: EPS>=25%/30d")
    print(f"{'='*180}")

    c17, raw = signal_cache[(25.0, 30)]

    combos = [
        ("D1 close entry, stop from entry, 0.75x ATR",
         {"entry_mode": "d1_close", "confirm": "green_candle",
          "stop_anchor": "entry", "stop_type": "atr", "atr_mult": 0.75}),
        ("D2 open entry, stop from entry, 0.75x ATR",
         {"entry_mode": "d2_open", "confirm": "green_candle",
          "stop_anchor": "entry", "stop_type": "atr", "atr_mult": 0.75}),
        ("D2 open entry, stop from D1 close, 0.75x ATR",
         {"entry_mode": "d2_open", "confirm": "green_candle",
          "stop_anchor": "d1_close", "stop_type": "atr", "atr_mult": 0.75}),
        ("D2 open entry, stop from D1 close, 0.50x ATR",
         {"entry_mode": "d2_open", "confirm": "green_candle",
          "stop_anchor": "d1_close", "stop_type": "atr", "atr_mult": 0.5}),
        ("D2 open entry, stop from D1 close, 5% stop",
         {"entry_mode": "d2_open", "confirm": "green_candle",
          "stop_anchor": "d1_close", "stop_type": "pct", "pct_stop": 5.0}),
    ]

    for label, kwargs in combos:
        r = run_sim(db, c17, raw, **kwargs)
        print(f"\n  >> {label}")
        show(label, r, show_trades=True)

    # Also show EPS>=20%/60d
    print(f"\n\n{'='*180}")
    print("  INDIVIDUAL TRADES: EPS>=20%/60d (broader)")
    print(f"{'='*180}")

    c17, raw = signal_cache[(20.0, 60)]
    for label, kwargs in combos:
        r = run_sim(db, c17, raw, **kwargs)
        print(f"\n  >> {label}")
        show(label, r, show_trades=True)


if __name__ == "__main__":
    main()
