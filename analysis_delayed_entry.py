"""Test realistic entry timing after green candle confirmation.

Current simulation (unrealistic): enter at midpoint on Day 1, but only if Day 1 closes green.
You can't know at midpoint whether the day will close green.

Realistic alternatives:
  A) Late Day 1: enter at Day 1 close (last 30 min) after seeing the green candle forming
  B) Day 2 open: enter at Day 2 open after confirming Day 1 was green
  C) Day 2 midpoint: enter at Day 2 midpoint after confirming Day 1 was green
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
from analysis_winners_losers import enrich_deep, filt_c17


def compute_atr(df, period=14):
    d = df.copy()
    d["prev_close"] = d["close"].shift(1)
    d["tr"] = d.apply(lambda r: max(
        r["high"] - r["low"],
        abs(r["high"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
        abs(r["low"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
    ), axis=1)
    return d["tr"].rolling(period).mean()


def simulate_trade_delayed(
    db, ticker, signal_date, direction,
    entry_mode="d1_midpoint",  # "d1_midpoint", "d1_close", "d2_open", "d2_midpoint"
    require_green_candle=False,
    require_close_above_prior=False,
    atr_mult=0.5,
    max_days=15,
):
    """Simulate a trade with configurable entry timing after green candle confirmation.

    Timeline:
      Day 0 = signal day (MA crossover detected after close)
      Day 1 = first trading day after signal (current entry day)
      Day 2 = second trading day after signal

    Entry modes:
      d1_midpoint — enter at Day 1 midpoint (UNREALISTIC with green candle filter)
      d1_close    — enter at Day 1 close (realistic: last 30min)
      d2_open     — enter at Day 2 open (realistic: full confirmation)
      d2_midpoint — enter at Day 2 midpoint (realistic: full confirmation)
    """
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

    signal_idx_list = df.index[df["date"] == signal_date].tolist()
    if not signal_idx_list:
        return None
    signal_idx = signal_idx_list[0]

    # Day 1 = signal_idx + 1, Day 2 = signal_idx + 2
    d1_idx = signal_idx + 1
    d2_idx = signal_idx + 2

    if d1_idx >= len(df):
        return None

    d0_row = df.iloc[signal_idx]
    d1_row = df.iloc[d1_idx]

    # ── Green candle / close-above-prior check on Day 1 ──
    if require_green_candle and d1_row["close"] <= d1_row["open"]:
        return None
    if require_close_above_prior and d1_row["close"] <= d0_row["close"]:
        return None

    # ── Determine entry price and entry index ──
    if entry_mode == "d1_midpoint":
        entry_price = (d1_row["high"] + d1_row["low"]) / 2
        entry_idx = d1_idx
    elif entry_mode == "d1_close":
        entry_price = d1_row["close"]
        entry_idx = d1_idx
    elif entry_mode in ("d2_open", "d2_midpoint"):
        if d2_idx >= len(df):
            return None
        d2_row = df.iloc[d2_idx]
        if entry_mode == "d2_open":
            entry_price = d2_row["open"]
        else:
            entry_price = (d2_row["high"] + d2_row["low"]) / 2
        entry_idx = d2_idx
    else:
        return None

    if entry_price <= 0:
        return None

    # ── Walk forward from entry day ──
    highest_close = entry_price
    prev_walk_row = df.iloc[entry_idx - 1] if entry_idx > 0 else df.iloc[entry_idx]

    for day_offset in range(0, max_days + 1):
        cur_idx = entry_idx + day_offset
        if cur_idx >= len(df):
            break

        row = df.iloc[cur_idx]
        close = row["close"]

        # Skip the entry bar itself for stop checks (we just entered)
        if day_offset == 0:
            if close > highest_close:
                highest_close = close
            prev_walk_row = row
            continue

        # Day 1 after entry: prev day low stop
        if day_offset == 1:
            prev_low = prev_walk_row["low"]
            if direction == "bullish" and close < prev_low:
                ret = (close - entry_price) / entry_price * 100
                return {"return": ret, "hold_days": day_offset, "exit_reason": "stop_d1",
                        "entry_price": entry_price, "exit_price": close,
                        "ticker": "", "signal_date": signal_date}
        else:
            # Day 2+: ATR trailing stop
            if atr_mult is not None:
                atr_val = atr_series.iloc[cur_idx - 1] if cur_idx - 1 < len(atr_series) else None
                if atr_val is None or pd.isna(atr_val):
                    atr_val = (df["high"].iloc[max(0, cur_idx-14):cur_idx].mean() -
                               df["low"].iloc[max(0, cur_idx-14):cur_idx].mean())
                stop_level = highest_close - atr_mult * atr_val
                if direction == "bullish" and close < stop_level:
                    ret = (close - entry_price) / entry_price * 100
                    return {"return": ret, "hold_days": day_offset, "exit_reason": "atr_stop",
                            "entry_price": entry_price, "exit_price": close,
                            "ticker": "", "signal_date": signal_date}

        if close > highest_close:
            highest_close = close
        prev_walk_row = row

    # Held to horizon
    last_idx = min(entry_idx + max_days, len(df) - 1)
    exit_price = df.iloc[last_idx]["close"]
    ret = (exit_price - entry_price) / entry_price * 100
    return {"return": ret, "hold_days": max_days, "exit_reason": "horizon",
            "entry_price": entry_price, "exit_price": exit_price,
            "ticker": "", "signal_date": signal_date}


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

        trade = simulate_trade_delayed(db, s["ticker"], s["trend_date"], "bullish", **kwargs)
        if trade:
            trade["ticker"] = s["ticker"]
            trade["signal_date"] = s["trend_date"]
            trade["eps_change_pct"] = s.get("eps_change_pct", 0)
            results.append(trade)
    return results


def show(label, results, show_trades=False):
    if not results:
        print(f"  {label:70s} | n=  0")
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

    print(f"  {label:70s} | n={len(results):>3d} | WR={wr:>5.1f}% | Avg={avg:>+7.1f}% "
          f"| AvgW={avg_w:>+6.1f}% | AvgL={avg_l:>+6.1f}% | PF={pf_s:>6s} | Hold={hold:.1f}d")

    if show_trades:
        for r in sorted(results, key=lambda x: x["signal_date"]):
            wl = "W" if r["return"] > 0 else "L"
            print(f"      {r['ticker']:7s} {r['signal_date']}  entry=${r['entry_price']:>7.2f}  "
                  f"exit=${r['exit_price']:>7.2f}  {r['return']:>+7.1f}%  {wl}  "
                  f"hold={r['hold_days']:>2d}d  exit={r['exit_reason']}")


def main():
    db = Database(DB_PATH)
    db.initialize()

    # Test configs: (eps_threshold, trend_window)
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

    # ═══════════════════════════════════════════════════════════════════════════
    print("=" * 150)
    print("  DELAYED ENTRY ANALYSIS: Impact of realistic entry timing with green candle confirmation")
    print("=" * 150)
    print()
    print("  Scenarios:")
    print("    A) d1_midpoint + green candle = CURRENT (unrealistic: can't know at midpoint if day closes green)")
    print("    B) d1_close + green candle    = Enter at Day 1 close (last 30 min, you can see it's green)")
    print("    C) d2_open + green candle     = Enter at Day 2 open (full confirmation, overnight risk)")
    print("    D) d2_midpoint + green candle  = Enter at Day 2 midpoint (full confirmation)")
    print("    E) d1_midpoint NO filter       = Baseline without green candle (current production)")
    print()

    for eps_thresh, tw in configs:
        c17, raw = signal_cache[(eps_thresh, tw)]
        if len(c17) < 3:
            continue

        print(f"\n{'='*150}")
        print(f"  EPS>={eps_thresh:.0f}% / {tw}d window — {len(c17)} C17 signals")
        print(f"{'='*150}")

        scenarios = [
            ("E) BASELINE: d1_midpoint, no green filter",
             {"entry_mode": "d1_midpoint", "require_green_candle": False}),
            ("A) d1_midpoint + green candle (UNREALISTIC)",
             {"entry_mode": "d1_midpoint", "require_green_candle": True}),
            ("B) d1_close + green candle (last 30 min)",
             {"entry_mode": "d1_close", "require_green_candle": True}),
            ("C) d2_open + green candle (next day open)",
             {"entry_mode": "d2_open", "require_green_candle": True}),
            ("D) d2_midpoint + green candle (next day midpoint)",
             {"entry_mode": "d2_midpoint", "require_green_candle": True}),
        ]

        for label, kwargs in scenarios:
            results = run_sim(db, c17, raw, atr_mult=0.5, max_days=15, **kwargs)
            show(label, results)

    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*150}")
    print("  SAME TEST WITH close > prior close FILTER (alternative confirmation)")
    print(f"{'='*150}")

    for eps_thresh, tw in configs:
        c17, raw = signal_cache[(eps_thresh, tw)]
        if len(c17) < 3:
            continue

        print(f"\n  --- EPS>={eps_thresh:.0f}% / {tw}d ---")

        scenarios = [
            ("E) BASELINE: d1_midpoint, no filter",
             {"entry_mode": "d1_midpoint"}),
            ("A) d1_midpoint + close>prior (UNREALISTIC)",
             {"entry_mode": "d1_midpoint", "require_close_above_prior": True}),
            ("B) d1_close + close>prior",
             {"entry_mode": "d1_close", "require_close_above_prior": True}),
            ("C) d2_open + close>prior",
             {"entry_mode": "d2_open", "require_close_above_prior": True}),
            ("D) d2_midpoint + close>prior",
             {"entry_mode": "d2_midpoint", "require_close_above_prior": True}),
        ]

        for label, kwargs in scenarios:
            results = run_sim(db, c17, raw, atr_mult=0.5, max_days=15, **kwargs)
            show(label, results)

    # ═══════════════════════════════════════════════════════════════════════════
    # Individual trade detail for primary config
    print(f"\n\n{'='*150}")
    print("  INDIVIDUAL TRADES: EPS>=25%/30d — all entry timing scenarios")
    print(f"{'='*150}")

    c17, raw = signal_cache[(25.0, 30)]

    for label, kwargs in [
        ("B) d1_close + green candle", {"entry_mode": "d1_close", "require_green_candle": True}),
        ("C) d2_open + green candle", {"entry_mode": "d2_open", "require_green_candle": True}),
        ("D) d2_midpoint + green candle", {"entry_mode": "d2_midpoint", "require_green_candle": True}),
    ]:
        print(f"\n  >> {label}")
        results = run_sim(db, c17, raw, atr_mult=0.5, max_days=15, **kwargs)
        show(label, results, show_trades=True)


if __name__ == "__main__":
    main()
