"""
Comprehensive optimization analysis for the C17 + EPS>=25% trading system.
Tests new dimensions not yet explored:
  1. Entry price variants (open, midpoint, close, VWAP proxy)
  2. Entry day confirmation (close > open)
  3. Market regime filter (SPY above 50/200 SMA)
  4. Profit targets (1x, 1.5x, 2x, 3x ATR)
  5. Time stops (exit if not profitable after N days)
  6. ATR period variants (7, 10, 14, 20)
  7. Max hold period optimization (5, 10, 15, 20, 30)
  8. Hybrid exits (profit target + ATR trailing)
  9. EPS threshold sweep (15%, 20%, 25%, 30%, 35%)
  10. Entry day volume confirmation
  11. Gap up on entry day
  12. Relative strength vs SPY
  13. Higher close on entry day (close > prior close)
"""

import sys
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, median

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import DB_PATH, BacktestConfig, ScannerConfig
from core.backtest import Backtester
from core.database import Database
from analysis_winners_losers import enrich_deep, filt_c17


# ─── Helpers ───────────────────────────────────────────────────────────────

def compute_atr(df, period=14):
    """Compute ATR series from a DataFrame with high, low, close."""
    d = df.copy()
    d["prev_close"] = d["close"].shift(1)
    d["tr"] = d.apply(lambda r: max(
        r["high"] - r["low"],
        abs(r["high"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
        abs(r["low"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
    ), axis=1)
    return d["tr"].rolling(period).mean()


def get_spy_data(db):
    """Fetch SPY daily prices and compute SMAs for regime filter."""
    rows = db.get_daily_prices("SPY", "2020-01-01", "2027-01-01")
    if not rows:
        return None
    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["sma_50"] = df["close"].rolling(50).mean()
    df["sma_200"] = df["close"].rolling(200).mean()
    df["date_str"] = df["date"]
    return df.set_index("date_str")


def simulate_trade_v2(
    db, ticker, signal_date, direction,
    entry_type="midpoint",       # "open", "midpoint", "close", "vwap"
    atr_mult=0.5,                # ATR trailing stop multiplier (None = prev day low)
    atr_period=14,               # ATR lookback
    max_days=15,                 # max holding period
    profit_target_atr=None,      # exit at N x ATR profit (None = no target)
    time_stop_days=None,         # exit if not profitable after N days
    require_entry_confirmation=False,  # require close > open on entry day
    require_entry_above_prior=False,   # require close > prior close on entry day
    require_entry_volume=False,  # require above-avg volume on entry day
    require_gap_up=False,        # require open > prior close on entry day
):
    """
    Enhanced trade simulator with many configurable exit/entry options.
    Returns dict with return, hold_days, exit_reason, entry_price, exit_price
    or None if trade can't be taken.
    """
    signal_dt = date.fromisoformat(signal_date)
    fetch_start = (signal_dt - timedelta(days=50)).isoformat()
    fetch_end = (signal_dt + timedelta(days=int(max_days * 1.6) + 10)).isoformat()

    rows = db.get_daily_prices(ticker, fetch_start, fetch_end)
    if not rows:
        return None

    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # Compute ATR
    atr_series = compute_atr(df, atr_period)

    # Compute 20d average volume
    vol_sma20 = df["volume"].rolling(20).mean()

    # Find signal date index
    signal_idx_list = df.index[df["date"] == signal_date].tolist()
    if not signal_idx_list:
        return None
    signal_idx = signal_idx_list[0]

    # Entry is next day after signal
    entry_idx = signal_idx + 1
    if entry_idx >= len(df):
        return None

    entry_row = df.iloc[entry_idx]
    prev_row = df.iloc[signal_idx]

    # ── Entry day filters ──
    if require_entry_confirmation and entry_row["close"] <= entry_row["open"]:
        return None  # entry day didn't close green

    if require_entry_above_prior and entry_row["close"] <= prev_row["close"]:
        return None  # entry day didn't close above prior

    if require_gap_up and entry_row["open"] <= prev_row["close"]:
        return None  # no gap up

    if require_entry_volume:
        avg_vol = vol_sma20.iloc[entry_idx] if entry_idx < len(vol_sma20) else None
        if avg_vol and not pd.isna(avg_vol) and entry_row["volume"] < avg_vol:
            return None  # below average volume

    # ── Entry price ──
    if entry_type == "open":
        entry_price = entry_row["open"]
    elif entry_type == "close":
        entry_price = entry_row["close"]
    elif entry_type == "vwap":
        entry_price = (entry_row["high"] + entry_row["low"] + entry_row["close"]) / 3
    else:  # midpoint
        entry_price = (entry_row["high"] + entry_row["low"]) / 2

    if entry_price <= 0:
        return None

    # Get ATR at entry for profit target
    entry_atr = atr_series.iloc[entry_idx] if entry_idx < len(atr_series) else None
    if entry_atr is None or pd.isna(entry_atr):
        entry_atr = df["high"].iloc[max(0, entry_idx-14):entry_idx].mean() - \
                    df["low"].iloc[max(0, entry_idx-14):entry_idx].mean()

    # ── Walk forward ──
    highest_close = entry_price
    prev_walk_row = df.iloc[signal_idx]

    for day_num in range(1, max_days + 1):
        cur_idx = signal_idx + day_num
        if cur_idx >= len(df):
            break

        row = df.iloc[cur_idx]
        close = row["close"]

        # ── Check profit target ──
        if profit_target_atr is not None and direction == "bullish":
            target_price = entry_price + profit_target_atr * entry_atr
            if row["high"] >= target_price:
                # Hit profit target — use target price as exit
                ret = (target_price - entry_price) / entry_price * 100
                return {
                    "return": ret, "hold_days": day_num,
                    "exit_reason": "profit_target", "entry_price": entry_price,
                    "exit_price": target_price,
                }

        # ── Check time stop ──
        if time_stop_days is not None and day_num >= time_stop_days:
            if direction == "bullish" and close <= entry_price:
                ret = (close - entry_price) / entry_price * 100
                return {
                    "return": ret, "hold_days": day_num,
                    "exit_reason": "time_stop", "entry_price": entry_price,
                    "exit_price": close,
                }

        # ── Check stop loss ──
        if day_num == 1:
            # Day 1: prev day low stop
            prev_low = prev_walk_row["low"]
            if direction == "bullish" and close < prev_low:
                ret = (close - entry_price) / entry_price * 100
                return {
                    "return": ret, "hold_days": day_num,
                    "exit_reason": "stop_d1", "entry_price": entry_price,
                    "exit_price": close,
                }
        else:
            if atr_mult is not None:
                # ATR trailing stop
                atr_val = atr_series.iloc[cur_idx - 1] if cur_idx - 1 < len(atr_series) else entry_atr
                if pd.isna(atr_val):
                    atr_val = entry_atr
                stop_level = highest_close - atr_mult * atr_val
                if direction == "bullish" and close < stop_level:
                    ret = (close - entry_price) / entry_price * 100
                    return {
                        "return": ret, "hold_days": day_num,
                        "exit_reason": "atr_stop", "entry_price": entry_price,
                        "exit_price": close,
                    }
            else:
                # Prev day low (original)
                prev_low = df.iloc[cur_idx - 1]["low"]
                if direction == "bullish" and close < prev_low:
                    ret = (close - entry_price) / entry_price * 100
                    return {
                        "return": ret, "hold_days": day_num,
                        "exit_reason": "stop_pdl", "entry_price": entry_price,
                        "exit_price": close,
                    }

        # Update trailing high
        if close > highest_close:
            highest_close = close
        prev_walk_row = row

    # Held to max_days horizon
    last_idx = min(signal_idx + max_days, len(df) - 1)
    exit_price = df.iloc[last_idx]["close"]
    ret = (exit_price - entry_price) / entry_price * 100
    return {
        "return": ret, "hold_days": max_days,
        "exit_reason": "horizon", "entry_price": entry_price,
        "exit_price": exit_price,
    }


def run_simulation(db, signals, raw_signals, spy_df=None, spy_regime=None, **sim_kwargs):
    """Run simulation across all signals, return list of trade results."""
    results = []
    for s in signals:
        # Match to raw signal for entry price reference
        match = None
        for rs in raw_signals:
            if rs["ticker"] == s["ticker"] and rs.get("trend_change_date") == s["trend_date"]:
                match = rs
                break
        if not match:
            continue

        # Market regime filter
        if spy_df is not None and spy_regime is not None:
            trend_date = s["trend_date"]
            if trend_date in spy_df.index:
                spy_row = spy_df.loc[trend_date]
                if spy_regime == "above_50" and spy_row["close"] < spy_row["sma_50"]:
                    continue
                if spy_regime == "above_200" and spy_row["close"] < spy_row["sma_200"]:
                    continue
                if spy_regime == "above_both" and (
                    spy_row["close"] < spy_row["sma_50"] or spy_row["close"] < spy_row["sma_200"]
                ):
                    continue

        trade = simulate_trade_v2(
            db, s["ticker"], s["trend_date"], "bullish", **sim_kwargs
        )
        if trade:
            trade["ticker"] = s["ticker"]
            trade["signal_date"] = s["trend_date"]
            trade["eps_change_pct"] = s.get("eps_change_pct", 0)
            results.append(trade)

    return results


def show_results(label, results, show_trades=False):
    """Print summary metrics for a set of trade results."""
    if not results:
        print(f"  {label:65s} | n=  0")
        return {}

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

    # Exit reason breakdown
    reasons = {}
    for r in results:
        reasons[r["exit_reason"]] = reasons.get(r["exit_reason"], 0) + 1

    print(f"  {label:65s} | n={len(results):>3d} | WR={wr:>5.1f}% | Avg={avg:>+7.1f}% "
          f"| AvgW={avg_w:>+6.1f}% | AvgL={avg_l:>+6.1f}% | PF={pf_s:>6s} "
          f"| Hold={hold_all:.1f}d(W:{hold_w:.1f}/L:{hold_l:.1f})")

    if show_trades:
        for r in sorted(results, key=lambda x: x["signal_date"]):
            wl = "W" if r["return"] > 0 else "L"
            print(f"      {r['ticker']:7s} {r['signal_date']}  {r['return']:>+7.1f}%  {wl}  "
                  f"hold={r['hold_days']:>2d}d  exit={r['exit_reason']:15s}  "
                  f"entry=${r['entry_price']:.2f}  exit=${r['exit_price']:.2f}  "
                  f"EPS={r['eps_change_pct']:>+.0f}%")

    return {
        "n": len(results), "wr": wr, "avg": avg, "avg_w": avg_w, "avg_l": avg_l,
        "pf": pf, "hold": hold_all, "reasons": reasons,
    }


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    db = Database(DB_PATH)
    db.initialize()

    # Try to get SPY data for regime filter
    spy_df = get_spy_data(db)
    has_spy = spy_df is not None and len(spy_df) > 200
    if has_spy:
        print(f"SPY data loaded: {len(spy_df)} rows")
    else:
        print("WARNING: No SPY data available — skipping market regime tests")

    # ── Run backtests at multiple EPS thresholds ──
    eps_thresholds_to_test = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
    trend_windows_to_test = [30, 45, 60]

    best_signals_by_config = {}

    for eps_thresh in eps_thresholds_to_test:
        for tw in trend_windows_to_test:
            sc = ScannerConfig(
                min_price=1.0, max_price=50.0,
                min_market_cap=50_000_000, max_market_cap=10_000_000_000,
                ma_crossover_pairs=[(20, 50)],
                eps_change_threshold=eps_thresh,
                trend_window_days=tw, direction="both",
            )
            bc = BacktestConfig(
                start_date="2022-01-01", end_date="2026-03-19",
                forward_return_days=[5, 10, 15, 30, 60],
                ma_crossover_pairs=[(20, 50)],
                eps_thresholds=[eps_thresh], trend_windows=[tw],
            )
            result = Backtester(db, sc, bc).run()
            enriched = enrich_deep(db, result["signals"])

            # Apply C17-style filter (but with the given EPS threshold already applied by scanner)
            c17 = [s for s in enriched
                    if s["signal_type"] == "bullish"
                    and s["eps_change_pct"] > 0
                    and s.get("avg_dollar_vol", 0) >= 500_000
                    and abs(s["eps_change_pct"]) < 100
                    and s["days_between"] > 10]

            best_signals_by_config[(eps_thresh, tw)] = (c17, result["signals"])

    # =====================================================================
    # Use 25% EPS / 30d as our primary test set (the current best)
    # =====================================================================
    primary_c17, primary_raw = best_signals_by_config[(25.0, 30)]

    print(f"\nPrimary test set: C17 + EPS>=25% + 30d window = {len(primary_c17)} signals")

    # =======================================================================
    print("\n" + "=" * 140)
    print("  SECTION 1: EPS THRESHOLD x TREND WINDOW SWEEP (0.5x ATR stop, midpoint entry)")
    print("=" * 140)

    for tw in trend_windows_to_test:
        print(f"\n  --- Trend Window: {tw}d ---")
        for eps_thresh in eps_thresholds_to_test:
            c17, raw = best_signals_by_config[(eps_thresh, tw)]
            results = run_simulation(db, c17, raw, atr_mult=0.5, max_days=15)
            show_results(f"EPS>={eps_thresh:.0f}% / {tw}d window", results)

    # =======================================================================
    print("\n" + "=" * 140)
    print("  SECTION 2: ENTRY PRICE VARIANTS (C17+EPS>=25%, 0.5x ATR, 30d)")
    print("=" * 140)

    for entry_type in ["open", "midpoint", "close", "vwap"]:
        results = run_simulation(
            db, primary_c17, primary_raw,
            entry_type=entry_type, atr_mult=0.5, max_days=15,
        )
        show_results(f"Entry: {entry_type}", results, show_trades=(entry_type == "midpoint"))

    # =======================================================================
    print("\n" + "=" * 140)
    print("  SECTION 3: ENTRY DAY CONFIRMATION FILTERS (C17+EPS>=25%, 0.5x ATR)")
    print("=" * 140)

    for label, kwargs in [
        ("No filters (baseline)", {}),
        ("Entry day close > open (green candle)", {"require_entry_confirmation": True}),
        ("Entry day close > prior close", {"require_entry_above_prior": True}),
        ("Entry day gap up (open > prior close)", {"require_gap_up": True}),
        ("Entry day above-avg volume", {"require_entry_volume": True}),
        ("Green candle + above-avg volume", {"require_entry_confirmation": True, "require_entry_volume": True}),
        ("Close > prior + above-avg volume", {"require_entry_above_prior": True, "require_entry_volume": True}),
        ("Gap up + green candle", {"require_gap_up": True, "require_entry_confirmation": True}),
        ("Green candle + close > prior", {"require_entry_confirmation": True, "require_entry_above_prior": True}),
    ]:
        results = run_simulation(
            db, primary_c17, primary_raw,
            atr_mult=0.5, max_days=15, **kwargs,
        )
        show_results(label, results)

    # =======================================================================
    if has_spy:
        print("\n" + "=" * 140)
        print("  SECTION 4: MARKET REGIME FILTER (SPY position)")
        print("=" * 140)

        for regime_label, regime_val in [
            ("No regime filter", None),
            ("SPY above 50 SMA", "above_50"),
            ("SPY above 200 SMA", "above_200"),
            ("SPY above both 50 & 200 SMA", "above_both"),
        ]:
            results = run_simulation(
                db, primary_c17, primary_raw,
                spy_df=spy_df, spy_regime=regime_val,
                atr_mult=0.5, max_days=15,
            )
            show_results(regime_label, results)

    # =======================================================================
    print("\n" + "=" * 140)
    print("  SECTION 5: PROFIT TARGETS (ATR-based, C17+EPS>=25%, 0.5x ATR stop)")
    print("=" * 140)

    for pt in [None, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        label = f"Profit target: {'None':>5s}" if pt is None else f"Profit target: {pt:.1f}x ATR"
        results = run_simulation(
            db, primary_c17, primary_raw,
            atr_mult=0.5, max_days=15, profit_target_atr=pt,
        )
        show_results(label, results)

    # =======================================================================
    print("\n" + "=" * 140)
    print("  SECTION 6: TIME STOPS (exit if not profitable after N days)")
    print("=" * 140)

    for ts in [None, 2, 3, 5, 7, 10]:
        label = f"Time stop: {'None':>5s}" if ts is None else f"Time stop: {ts}d"
        results = run_simulation(
            db, primary_c17, primary_raw,
            atr_mult=0.5, max_days=15, time_stop_days=ts,
        )
        show_results(label, results)

    # =======================================================================
    print("\n" + "=" * 140)
    print("  SECTION 7: ATR PERIOD FOR STOP CALCULATION")
    print("=" * 140)

    for atr_p in [7, 10, 14, 20, 30]:
        results = run_simulation(
            db, primary_c17, primary_raw,
            atr_mult=0.5, atr_period=atr_p, max_days=15,
        )
        show_results(f"ATR period: {atr_p}d", results)

    # =======================================================================
    print("\n" + "=" * 140)
    print("  SECTION 8: MAX HOLDING PERIOD")
    print("=" * 140)

    for max_d in [5, 7, 10, 15, 20, 30, 45, 60]:
        results = run_simulation(
            db, primary_c17, primary_raw,
            atr_mult=0.5, max_days=max_d,
        )
        show_results(f"Max hold: {max_d}d", results)

    # =======================================================================
    print("\n" + "=" * 140)
    print("  SECTION 9: ATR STOP MULTIPLIER SWEEP (finer granularity)")
    print("=" * 140)

    for mult in [0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0]:
        results = run_simulation(
            db, primary_c17, primary_raw,
            atr_mult=mult, max_days=15,
        )
        show_results(f"ATR stop: {mult:.2f}x", results)

    # Also test no ATR (prev day low all days)
    results = run_simulation(
        db, primary_c17, primary_raw,
        atr_mult=None, max_days=15,
    )
    show_results("No ATR (prev day low always)", results)

    # =======================================================================
    print("\n" + "=" * 140)
    print("  SECTION 10: BEST COMBINATIONS (mixing top performers from above)")
    print("=" * 140)

    # Test combinations of the most promising parameters
    combos = [
        # (label, kwargs)
        ("BASELINE: midpoint, 0.5x ATR, 15d, no filters",
         {"atr_mult": 0.5, "max_days": 15}),

        ("Open entry + 0.5x ATR + 15d",
         {"entry_type": "open", "atr_mult": 0.5, "max_days": 15}),

        ("Close entry + 0.5x ATR + 15d",
         {"entry_type": "close", "atr_mult": 0.5, "max_days": 15}),

        ("Midpoint + 0.5x ATR + 20d hold",
         {"atr_mult": 0.5, "max_days": 20}),

        ("Midpoint + 0.5x ATR + 30d hold",
         {"atr_mult": 0.5, "max_days": 30}),

        ("Midpoint + 0.5x ATR + 3x ATR target + 15d",
         {"atr_mult": 0.5, "max_days": 15, "profit_target_atr": 3.0}),

        ("Midpoint + 0.5x ATR + 5x ATR target + 20d",
         {"atr_mult": 0.5, "max_days": 20, "profit_target_atr": 5.0}),

        ("Midpoint + 0.5x ATR + 3d time stop + 15d",
         {"atr_mult": 0.5, "max_days": 15, "time_stop_days": 3}),

        ("Midpoint + 0.5x ATR + 5d time stop + 15d",
         {"atr_mult": 0.5, "max_days": 15, "time_stop_days": 5}),

        ("Midpoint + 0.5x ATR + green candle + 15d",
         {"atr_mult": 0.5, "max_days": 15, "require_entry_confirmation": True}),

        ("Midpoint + 0.5x ATR + close>prior + 15d",
         {"atr_mult": 0.5, "max_days": 15, "require_entry_above_prior": True}),

        ("Open + 0.5x ATR + green candle + 20d",
         {"entry_type": "open", "atr_mult": 0.5, "max_days": 20,
          "require_entry_confirmation": True}),

        ("Open + 0.5x ATR + green candle + 3x target + 20d",
         {"entry_type": "open", "atr_mult": 0.5, "max_days": 20,
          "require_entry_confirmation": True, "profit_target_atr": 3.0}),

        ("Close + 0.5x ATR + close>prior + 20d",
         {"entry_type": "close", "atr_mult": 0.5, "max_days": 20,
          "require_entry_above_prior": True}),

        ("Midpoint + 0.4x ATR + green candle + 20d",
         {"atr_mult": 0.4, "max_days": 20,
          "require_entry_confirmation": True}),

        ("Midpoint + 0.3x ATR + green candle + 15d",
         {"atr_mult": 0.3, "max_days": 15,
          "require_entry_confirmation": True}),

        ("Midpoint + 0.5x ATR + above-avg vol + 15d",
         {"atr_mult": 0.5, "max_days": 15, "require_entry_volume": True}),

        ("Midpoint + 0.5x ATR + gap up + 20d",
         {"atr_mult": 0.5, "max_days": 20, "require_gap_up": True}),

        ("Midpoint + 0.5x ATR + green+vol + 20d",
         {"atr_mult": 0.5, "max_days": 20,
          "require_entry_confirmation": True, "require_entry_volume": True}),

        ("Midpoint + 0.5x ATR + 5d time stop + 3x target + 20d",
         {"atr_mult": 0.5, "max_days": 20,
          "time_stop_days": 5, "profit_target_atr": 3.0}),
    ]

    for label, kwargs in combos:
        results = run_simulation(db, primary_c17, primary_raw, **kwargs)
        show_results(label, results)

    # =======================================================================
    # SECTION 11: Test across multiple EPS thresholds with best exit combo
    print("\n" + "=" * 140)
    print("  SECTION 11: BEST EXIT PARAMETERS ACROSS ALL EPS/WINDOW CONFIGS")
    print("=" * 140)
    print("  (Using best exit params found above applied to each EPS/window combo)")

    # We'll test a few promising exit combos across all configs
    exit_combos = [
        ("0.5x ATR, 15d", {"atr_mult": 0.5, "max_days": 15}),
        ("0.5x ATR, 20d", {"atr_mult": 0.5, "max_days": 20}),
        ("0.5x ATR, 20d, green candle", {"atr_mult": 0.5, "max_days": 20, "require_entry_confirmation": True}),
        ("0.5x ATR, 20d, close>prior", {"atr_mult": 0.5, "max_days": 20, "require_entry_above_prior": True}),
        ("0.5x ATR, 20d, 3x target", {"atr_mult": 0.5, "max_days": 20, "profit_target_atr": 3.0}),
    ]

    for (eps_thresh, tw), (c17, raw) in sorted(best_signals_by_config.items()):
        if len(c17) < 3:
            continue
        print(f"\n  --- EPS>={eps_thresh:.0f}% / {tw}d window ({len(c17)} C17 signals) ---")
        for exit_label, exit_kwargs in exit_combos:
            results = run_simulation(db, c17, raw, **exit_kwargs)
            show_results(f"  {exit_label}", results)

    # =======================================================================
    # SECTION 12: Individual trade detail for the best configs
    print("\n" + "=" * 140)
    print("  SECTION 12: INDIVIDUAL TRADE DETAILS — TOP CONFIGURATIONS")
    print("=" * 140)

    # Show trades for a few top configs
    top_configs = [
        ("C17+EPS>=25%+30d: 0.5xATR/15d (current best)",
         primary_c17, primary_raw, {"atr_mult": 0.5, "max_days": 15}),
        ("C17+EPS>=25%+30d: 0.5xATR/20d",
         primary_c17, primary_raw, {"atr_mult": 0.5, "max_days": 20}),
        ("C17+EPS>=30%+30d: 0.5xATR/15d",
         *best_signals_by_config.get((30.0, 30), ([], [])),
         {"atr_mult": 0.5, "max_days": 15}),
        ("C17+EPS>=20%+30d: 0.5xATR/15d",
         *best_signals_by_config.get((20.0, 30), ([], [])),
         {"atr_mult": 0.5, "max_days": 15}),
    ]

    for label, c17_sigs, raw_sigs, sim_kwargs in top_configs:
        results = run_simulation(db, c17_sigs, raw_sigs, **sim_kwargs)
        print(f"\n  >> {label}")
        show_results(f"  Summary", results, show_trades=True)

    # =======================================================================
    # SECTION 13: Monte Carlo analysis on best system
    print("\n" + "=" * 140)
    print("  SECTION 13: MONTE CARLO ANALYSIS (randomize trade order, 10000 simulations)")
    print("=" * 140)

    results = run_simulation(
        db, primary_c17, primary_raw, atr_mult=0.5, max_days=15,
    )
    if len(results) >= 5:
        rets = [r["return"] for r in results]
        n_sims = 10000
        sim_wr = []
        sim_avg = []
        sim_pf = []

        np.random.seed(42)
        for _ in range(n_sims):
            sample = list(np.random.choice(rets, size=len(rets), replace=True))
            w = [r for r in sample if r > 0]
            l = [r for r in sample if r <= 0]
            sim_wr.append(len(w) / len(sample) * 100)
            sim_avg.append(mean(sample))
            gg = sum(w)
            gl = sum(r for r in l if r < 0)
            pf = gg / abs(gl) if gl else 100
            sim_pf.append(min(pf, 100))  # cap at 100 for stats

        sim_wr.sort()
        sim_avg.sort()
        sim_pf.sort()

        def pctile(arr, p):
            idx = int(len(arr) * p / 100)
            return arr[min(idx, len(arr)-1)]

        print(f"  Based on {len(rets)} actual trades, {n_sims} bootstrap simulations:")
        print(f"  Win Rate:      p5={pctile(sim_wr,5):.1f}%  p25={pctile(sim_wr,25):.1f}%  "
              f"p50={pctile(sim_wr,50):.1f}%  p75={pctile(sim_wr,75):.1f}%  p95={pctile(sim_wr,95):.1f}%")
        print(f"  Avg Return:    p5={pctile(sim_avg,5):+.1f}%  p25={pctile(sim_avg,25):+.1f}%  "
              f"p50={pctile(sim_avg,50):+.1f}%  p75={pctile(sim_avg,75):+.1f}%  p95={pctile(sim_avg,95):+.1f}%")
        print(f"  Profit Factor: p5={pctile(sim_pf,5):.2f}  p25={pctile(sim_pf,25):.2f}  "
              f"p50={pctile(sim_pf,50):.2f}  p75={pctile(sim_pf,75):.2f}  p95={pctile(sim_pf,95):.2f}")

        # Probability of losing money
        losing_sims = sum(1 for a in sim_avg if a <= 0) / n_sims * 100
        sub40_wr = sum(1 for w in sim_wr if w < 40) / n_sims * 100
        print(f"\n  P(avg return <= 0): {losing_sims:.1f}%")
        print(f"  P(win rate < 40%):  {sub40_wr:.1f}%")


if __name__ == "__main__":
    main()
