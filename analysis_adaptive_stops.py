"""
Adaptive stop analysis for the CURRENT SYSTEM's 12 trades.
D0 close entry, vol >= 1.2x, configs EPS>=25%/30d and EPS>=30%/60d combined and deduped.

Tests two adaptive stop rules:
1. Signal vol > 2x -> wider ATR trailing stop (let big-volume trades breathe)
2. RSI > 80 at exit -> override/delay the stop (stay in strong momentum moves)
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
from analysis_winners_losers import enrich_deep

# ---------------------------------------------------------------------------
# Config: the two parameter combos that produce our 12-trade universe
# ---------------------------------------------------------------------------
CONFIGS = [(25.0, 30), (30.0, 60)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_db():
    db = Database(DB_PATH)
    db.initialize()
    return db


def compute_atr(prices_df, period=14):
    """Compute ATR series from a DataFrame with high, low, close columns."""
    df = prices_df.copy()
    df["prev_close"] = df["close"].shift(1)
    df["tr"] = df.apply(lambda r: max(
        r["high"] - r["low"],
        abs(r["high"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
        abs(r["low"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
    ), axis=1)
    return df["tr"].rolling(period).mean()


def compute_rsi_series(closes_series, period=14):
    """Compute RSI as a pandas Series from a close-price Series."""
    delta = closes_series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def gather_signals(db):
    """Run both configs, apply C17-like filters + vol>=1.2x, combine & dedup."""
    all_signals = []

    for eps_thresh, trend_window in CONFIGS:
        sc = ScannerConfig(
            min_price=1.0, max_price=50.0,
            min_market_cap=50_000_000, max_market_cap=10_000_000_000,
            ma_crossover_pairs=[(20, 50)],
            eps_change_threshold=eps_thresh,
            trend_window_days=trend_window,
            direction="both",
        )
        bc = BacktestConfig(
            start_date="2022-01-01", end_date="2026-03-19",
            forward_return_days=[5, 10, 15, 30, 60],
            ma_crossover_pairs=[(20, 50)],
            eps_thresholds=[eps_thresh],
            trend_windows=[trend_window],
        )
        result = Backtester(db, sc, bc).run()
        enriched = enrich_deep(db, result["signals"])
        # C17-like filter
        filtered = [s for s in enriched
                    if s["signal_type"] == "bullish"
                    and s["eps_change_pct"] > 0
                    and s.get("avg_dollar_vol", 0) >= 500_000
                    and abs(s["eps_change_pct"]) < 100
                    and s["days_between"] > 10]
        for s in filtered:
            s["_config"] = f"EPS>={eps_thresh}/{trend_window}d"
        all_signals.extend(filtered)

    # Dedup by (ticker, trend_date)
    seen = set()
    deduped = []
    for s in all_signals:
        key = (s["ticker"], s["trend_date"])
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    return deduped


def get_signal_vol_ratio(db, ticker, signal_date):
    """Get signal day volume / 20-day average volume ratio. Returns (ratio, passes_1.2x)."""
    signal_dt = date.fromisoformat(signal_date)
    fetch_start = (signal_dt - timedelta(days=60)).isoformat()
    rows = db.get_daily_prices(ticker, fetch_start, signal_date)
    if len(rows) < 21:
        return None, False

    df = pd.DataFrame(rows)
    df["volume"] = df["volume"].astype(float)

    signal_row = df[df["date"] == signal_date]
    if signal_row.empty:
        return None, False

    signal_idx = signal_row.index[0]
    if signal_idx < 20:
        return None, False

    signal_vol = df["volume"].iloc[signal_idx]
    avg_vol_20 = df["volume"].iloc[signal_idx - 20:signal_idx].mean()

    if avg_vol_20 <= 0:
        return None, False

    ratio = signal_vol / avg_vol_20
    return ratio, ratio >= 1.2


# ---------------------------------------------------------------------------
# Core simulation: adaptive stops
# ---------------------------------------------------------------------------

def simulate_adaptive(db, ticker, signal_date,
                      base_atr_mult=0.5,
                      high_vol_atr_mult=None,  # wider ATR mult for high-vol signals
                      vol_threshold=2.0,         # signal vol ratio above which to use wider stop
                      signal_vol_ratio=1.0,      # actual signal vol ratio for this trade
                      rsi_override_level=None,   # RSI level above which to override stop
                      rsi_override_mode="skip",  # "skip" or "widen"
                      rsi_widen_mult=1.0,         # ATR mult when RSI override widens
                      max_overrides=1,            # max times RSI can override per trade (-1=unlimited)
                      max_days=15):
    """Simulate a single trade with adaptive stop rules.

    Entry: signal day (D0) close price.
    Day 1: stop = close < prev day low (unchanged from current system).
    Day 2+: trailing stop = highest_close - atr_mult * ATR(14).
      - atr_mult = high_vol_atr_mult if signal_vol_ratio > vol_threshold, else base_atr_mult
      - When close < stop_level, check RSI before exiting:
        - If RSI > rsi_override_level, DON'T exit (apply rsi_override_mode logic)
        - Track number of overrides
    """
    signal_dt = date.fromisoformat(signal_date)
    fetch_start = (signal_dt - timedelta(days=60)).isoformat()
    fetch_end = (signal_dt + timedelta(days=int(max_days * 2.5) + 10)).isoformat()

    rows = db.get_daily_prices(ticker, fetch_start, fetch_end)
    if not rows:
        return None

    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df.reset_index(drop=True)

    signal_idx_list = df.index[df["date"] == signal_date].tolist()
    if not signal_idx_list:
        return None
    signal_idx = signal_idx_list[0]

    # Entry = D0 close
    entry_price = df["close"].iloc[signal_idx]
    if entry_price <= 0:
        return None

    # Compute ATR and RSI series
    atr_series = compute_atr(df)
    rsi_series = compute_rsi_series(df["close"])

    # Determine which ATR mult to use based on signal volume
    if high_vol_atr_mult is not None and signal_vol_ratio > vol_threshold:
        active_atr_mult = high_vol_atr_mult
        using_wide_stop = True
    else:
        active_atr_mult = base_atr_mult
        using_wide_stop = False

    # Walk forward
    highest_close = entry_price
    peak_return = 0.0
    peak_day = 0
    prev_row = df.iloc[signal_idx]
    rsi_overrides_used = 0
    exit_day = None
    exit_price = None
    exit_reason = None

    for day_num in range(1, max_days + 1):
        cur_idx = signal_idx + day_num
        if cur_idx >= len(df):
            break

        row = df.iloc[cur_idx]
        close = row["close"]

        # Track peak
        ret_so_far = (close - entry_price) / entry_price * 100
        if close > highest_close:
            highest_close = close
            peak_return = (highest_close - entry_price) / entry_price * 100
            peak_day = day_num

        # Stop check
        stop_triggered = False
        if day_num == 1:
            # Day 1: always use prev day low (signal day low)
            prev_low = prev_row["low"]
            if close < prev_low:
                stop_triggered = True
                exit_reason_candidate = "stop_d1"
        else:
            # Day 2+: ATR trailing stop
            atr_val = atr_series.iloc[cur_idx - 1] if cur_idx - 1 < len(atr_series) and pd.notna(atr_series.iloc[cur_idx - 1]) else None
            if atr_val is None or pd.isna(atr_val):
                lookback = df.iloc[max(0, cur_idx - 14):cur_idx]
                atr_val = (lookback["high"] - lookback["low"]).mean()

            stop_level = highest_close - active_atr_mult * atr_val
            if close < stop_level:
                stop_triggered = True
                exit_reason_candidate = "atr_stop"

        if stop_triggered:
            # Check RSI override (only for day 2+ with atr_stop)
            rsi_override_applied = False
            if (rsi_override_level is not None
                    and day_num > 1
                    and exit_reason_candidate == "atr_stop"):
                cur_rsi = rsi_series.iloc[cur_idx] if cur_idx < len(rsi_series) else None
                if cur_rsi is not None and not pd.isna(cur_rsi) and cur_rsi > rsi_override_level:
                    can_override = (max_overrides == -1 or rsi_overrides_used < max_overrides)
                    if can_override:
                        rsi_overrides_used += 1
                        rsi_override_applied = True

                        if rsi_override_mode == "widen":
                            # Temporarily use wider stop; re-check with rsi_widen_mult
                            wide_stop = highest_close - rsi_widen_mult * atr_val
                            if close < wide_stop:
                                # Even the wider stop triggers, exit anyway
                                rsi_override_applied = False
                        # else mode == "skip": just skip the exit entirely

            if rsi_override_applied:
                # Don't exit; continue holding
                pass
            else:
                exit_day = day_num
                exit_price = close
                exit_reason = exit_reason_candidate
                break

        prev_row = row

    # If no stop triggered, exit at max_days horizon
    if exit_day is None:
        last_idx = min(signal_idx + max_days, len(df) - 1)
        exit_price = df.iloc[last_idx]["close"]
        exit_day = max_days
        exit_reason = "horizon"

    trade_return = (exit_price - entry_price) / entry_price * 100
    give_back = peak_return - trade_return

    return {
        "ticker": ticker,
        "signal_date": signal_date,
        "entry_price": round(entry_price, 4),
        "exit_price": round(exit_price, 4),
        "return": round(trade_return, 2),
        "hold_days": exit_day,
        "exit_reason": exit_reason,
        "peak_return": round(peak_return, 2),
        "peak_day": peak_day,
        "give_back": round(give_back, 2),
        "rsi_overrides_used": rsi_overrides_used,
        "using_wide_stop": using_wide_stop,
        "signal_vol_ratio": round(signal_vol_ratio, 2),
    }


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def run_scenario(db, signals, vol_ratios, label, *,
                 base_atr_mult=0.5,
                 high_vol_atr_mult=None,
                 vol_threshold=2.0,
                 rsi_override_level=None,
                 rsi_override_mode="skip",
                 rsi_widen_mult=1.0,
                 max_overrides=1,
                 max_days=15):
    """Run a scenario across all signals and return list of trade results."""
    results = []
    for s in signals:
        vr = vol_ratios.get((s["ticker"], s["trend_date"]), 1.0)

        # Determine effective max_days: if vol > threshold and we have extended hold, use it
        effective_max_days = max_days

        trade = simulate_adaptive(
            db, s["ticker"], s["trend_date"],
            base_atr_mult=base_atr_mult,
            high_vol_atr_mult=high_vol_atr_mult,
            vol_threshold=vol_threshold,
            signal_vol_ratio=vr,
            rsi_override_level=rsi_override_level,
            rsi_override_mode=rsi_override_mode,
            rsi_widen_mult=rsi_widen_mult,
            max_overrides=max_overrides,
            max_days=effective_max_days,
        )
        if trade:
            results.append(trade)
    return results


def run_scenario_extended(db, signals, vol_ratios, label, *,
                          base_atr_mult=0.5,
                          high_vol_atr_mult=None,
                          vol_threshold=2.0,
                          rsi_override_level=None,
                          rsi_override_mode="skip",
                          rsi_widen_mult=1.0,
                          max_overrides=1,
                          base_max_days=15,
                          high_vol_max_days=30):
    """Like run_scenario but with extended hold for high-vol signals."""
    results = []
    for s in signals:
        vr = vol_ratios.get((s["ticker"], s["trend_date"]), 1.0)

        if high_vol_atr_mult is not None and vr > vol_threshold:
            effective_max_days = high_vol_max_days
        else:
            effective_max_days = base_max_days

        trade = simulate_adaptive(
            db, s["ticker"], s["trend_date"],
            base_atr_mult=base_atr_mult,
            high_vol_atr_mult=high_vol_atr_mult,
            vol_threshold=vol_threshold,
            signal_vol_ratio=vr,
            rsi_override_level=rsi_override_level,
            rsi_override_mode=rsi_override_mode,
            rsi_widen_mult=rsi_widen_mult,
            max_overrides=max_overrides,
            max_days=effective_max_days,
        )
        if trade:
            results.append(trade)
    return results


def summarize(results, label):
    """Print single summary line for a scenario."""
    if not results:
        print(f"  {label}: n=0  (no results)")
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
    avg_hold = mean([r["hold_days"] for r in results])
    total_overrides = sum(r["rsi_overrides_used"] for r in results)

    print(f"  {label}: n={len(results):>2d}  WR={wr:>5.1f}%  Avg={avg:>+6.1f}%  "
          f"AvgW={avg_w:>+6.1f}%  AvgL={avg_l:>+5.1f}%  PF={pf_s:>5s}  "
          f"Hold={avg_hold:>4.1f}d  RSI_overrides={total_overrides}")

    return {
        "label": label, "results": results,
        "wr": wr, "avg": avg, "avg_w": avg_w, "avg_l": avg_l,
        "pf": pf, "avg_hold": avg_hold, "total_overrides": total_overrides,
        "n": len(results),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    db = get_db()

    print("=" * 100)
    print("  ADAPTIVE STOP ANALYSIS")
    print("  Rules: D0 close entry, vol >= 1.2x, EPS>=25%/30d + EPS>=30%/60d")
    print("         combined & deduped, baseline 0.5x ATR trailing stop, 15d max hold")
    print("=" * 100)

    # Gather signals
    print("\nGathering signals from both configs...")
    deduped = gather_signals(db)
    print(f"  Deduped C17-filtered signals: {len(deduped)}")

    # Get vol ratios and filter
    vol_ratios = {}
    vol_filtered = []
    for s in deduped:
        vr, passes = get_signal_vol_ratio(db, s["ticker"], s["trend_date"])
        if passes and vr is not None:
            vol_ratios[(s["ticker"], s["trend_date"])] = vr
            vol_filtered.append(s)

    signals = sorted(vol_filtered, key=lambda x: x["trend_date"])
    print(f"  After volume >= 1.2x filter: {len(signals)}")

    # Show the trades and their vol ratios
    print(f"\n  {'#':>3s} {'Ticker':<8s} {'Signal Date':<12s} {'EPS%':>7s} {'VolRatio':>9s}")
    print("  " + "-" * 45)
    for i, s in enumerate(signals, 1):
        vr = vol_ratios[(s["ticker"], s["trend_date"])]
        high_vol = " ***" if vr > 2.0 else ""
        print(f"  {i:>3d} {s['ticker']:<8s} {s['trend_date']:<12s} "
              f"{s['eps_change_pct']:>+6.1f}% {vr:>8.2f}x{high_vol}")

    # Count high-vol signals
    high_vol_count = sum(1 for s in signals if vol_ratios[(s["ticker"], s["trend_date"])] > 2.0)
    med_vol_count = sum(1 for s in signals if vol_ratios[(s["ticker"], s["trend_date"])] > 1.5)
    print(f"\n  Signals with vol > 2.0x: {high_vol_count}/{len(signals)}")
    print(f"  Signals with vol > 1.5x: {med_vol_count}/{len(signals)}")

    # ======================================================================
    # Run all scenarios
    # ======================================================================
    all_scenarios = {}

    # ------ SECTION 1: Baseline ------
    print("\n" + "=" * 100)
    print("  SECTION 1: BASELINE")
    print("=" * 100)
    r = run_scenario(db, signals, vol_ratios, "A",
                     base_atr_mult=0.5, max_days=15)
    all_scenarios["A"] = summarize(r, "A: Baseline (0.5x ATR, no adaptations)")

    # ------ SECTION 2: Volume-Adaptive Stop Only ------
    print("\n" + "=" * 100)
    print("  SECTION 2: VOLUME-ADAPTIVE STOP ONLY (no RSI override)")
    print("=" * 100)

    r = run_scenario(db, signals, vol_ratios, "B",
                     base_atr_mult=0.5, high_vol_atr_mult=0.75, vol_threshold=2.0, max_days=15)
    all_scenarios["B"] = summarize(r, "B: Vol>2.0x -> 0.75x ATR, else 0.5x")

    r = run_scenario(db, signals, vol_ratios, "C",
                     base_atr_mult=0.5, high_vol_atr_mult=1.0, vol_threshold=2.0, max_days=15)
    all_scenarios["C"] = summarize(r, "C: Vol>2.0x -> 1.0x ATR, else 0.5x")

    r = run_scenario(db, signals, vol_ratios, "D",
                     base_atr_mult=0.5, high_vol_atr_mult=1.5, vol_threshold=2.0, max_days=15)
    all_scenarios["D"] = summarize(r, "D: Vol>2.0x -> 1.5x ATR, else 0.5x")

    r = run_scenario(db, signals, vol_ratios, "E",
                     base_atr_mult=0.5, high_vol_atr_mult=0.75, vol_threshold=1.5, max_days=15)
    all_scenarios["E"] = summarize(r, "E: Vol>1.5x -> 0.75x ATR, else 0.5x")

    # ------ SECTION 3: RSI Override Only ------
    print("\n" + "=" * 100)
    print("  SECTION 3: RSI OVERRIDE ONLY (no volume adaptation)")
    print("=" * 100)

    r = run_scenario(db, signals, vol_ratios, "F",
                     base_atr_mult=0.5, rsi_override_level=80, rsi_override_mode="skip",
                     max_overrides=1, max_days=15)
    all_scenarios["F"] = summarize(r, "F: RSI>80 skip exit, max 1 override, 0.5x ATR")

    r = run_scenario(db, signals, vol_ratios, "G",
                     base_atr_mult=0.5, rsi_override_level=80, rsi_override_mode="skip",
                     max_overrides=2, max_days=15)
    all_scenarios["G"] = summarize(r, "G: RSI>80 skip exit, max 2 overrides, 0.5x ATR")

    r = run_scenario(db, signals, vol_ratios, "H",
                     base_atr_mult=0.5, rsi_override_level=80, rsi_override_mode="skip",
                     max_overrides=-1, max_days=15)
    all_scenarios["H"] = summarize(r, "H: RSI>80 skip exit, unlimited overrides, 0.5x ATR")

    r = run_scenario(db, signals, vol_ratios, "I",
                     base_atr_mult=0.5, rsi_override_level=75, rsi_override_mode="skip",
                     max_overrides=1, max_days=15)
    all_scenarios["I"] = summarize(r, "I: RSI>75 skip exit, max 1 override, 0.5x ATR")

    r = run_scenario(db, signals, vol_ratios, "J",
                     base_atr_mult=0.5, rsi_override_level=85, rsi_override_mode="skip",
                     max_overrides=1, max_days=15)
    all_scenarios["J"] = summarize(r, "J: RSI>85 skip exit, max 1 override, 0.5x ATR")

    # ------ SECTION 4: Combined (Vol + RSI) ------
    print("\n" + "=" * 100)
    print("  SECTION 4: COMBINED (Vol-Adaptive + RSI Override)")
    print("=" * 100)

    r = run_scenario(db, signals, vol_ratios, "K",
                     base_atr_mult=0.5, high_vol_atr_mult=0.75, vol_threshold=2.0,
                     rsi_override_level=80, rsi_override_mode="skip", max_overrides=1, max_days=15)
    all_scenarios["K"] = summarize(r, "K: Vol>2x->0.75x + RSI>80 skip(1), else 0.5x")

    r = run_scenario(db, signals, vol_ratios, "L",
                     base_atr_mult=0.5, high_vol_atr_mult=1.0, vol_threshold=2.0,
                     rsi_override_level=80, rsi_override_mode="skip", max_overrides=1, max_days=15)
    all_scenarios["L"] = summarize(r, "L: Vol>2x->1.0x + RSI>80 skip(1), else 0.5x")

    r = run_scenario(db, signals, vol_ratios, "M",
                     base_atr_mult=0.5, high_vol_atr_mult=0.75, vol_threshold=2.0,
                     rsi_override_level=80, rsi_override_mode="skip", max_overrides=2, max_days=15)
    all_scenarios["M"] = summarize(r, "M: Vol>2x->0.75x + RSI>80 skip(2), else 0.5x")

    r = run_scenario(db, signals, vol_ratios, "N",
                     base_atr_mult=0.5, high_vol_atr_mult=1.0, vol_threshold=2.0,
                     rsi_override_level=80, rsi_override_mode="skip", max_overrides=-1, max_days=15)
    all_scenarios["N"] = summarize(r, "N: Vol>2x->1.0x + RSI>80 skip(unl), else 0.5x")

    r = run_scenario(db, signals, vol_ratios, "O",
                     base_atr_mult=0.5, high_vol_atr_mult=0.75, vol_threshold=1.5,
                     rsi_override_level=80, rsi_override_mode="skip", max_overrides=1, max_days=15)
    all_scenarios["O"] = summarize(r, "O: Vol>1.5x->0.75x + RSI>80 skip(1), else 0.5x")

    # ------ SECTION 5: Extended Hold for High-Vol ------
    print("\n" + "=" * 100)
    print("  SECTION 5: EXTENDED HOLD FOR HIGH-VOL SIGNALS")
    print("=" * 100)

    r = run_scenario_extended(db, signals, vol_ratios, "P",
                              base_atr_mult=0.5, high_vol_atr_mult=0.75, vol_threshold=2.0,
                              rsi_override_level=80, rsi_override_mode="skip", max_overrides=1,
                              base_max_days=15, high_vol_max_days=30)
    all_scenarios["P"] = summarize(r, "P: Vol>2x->30d/0.75x+RSI>80(1), else 15d/0.5x")

    r = run_scenario_extended(db, signals, vol_ratios, "Q",
                              base_atr_mult=0.5, high_vol_atr_mult=1.0, vol_threshold=2.0,
                              rsi_override_level=80, rsi_override_mode="skip", max_overrides=2,
                              base_max_days=15, high_vol_max_days=30)
    all_scenarios["Q"] = summarize(r, "Q: Vol>2x->30d/1.0x+RSI>80(2), else 15d/0.5x")

    # ======================================================================
    # DETAILED TRADE-BY-TRADE COMPARISON: Top 3 vs Baseline
    # ======================================================================
    print("\n" + "=" * 100)
    print("  DETAILED TRADE-BY-TRADE COMPARISON: TOP 3 SCENARIOS vs BASELINE")
    print("=" * 100)

    # Rank scenarios by avg return (excluding baseline A)
    ranked = sorted(
        [(k, v) for k, v in all_scenarios.items() if k != "A" and v and "avg" in v],
        key=lambda x: x[1].get("avg", -999),
        reverse=True,
    )

    baseline_results = all_scenarios.get("A", {}).get("results", [])
    if not baseline_results:
        print("  No baseline results!")
        return

    # Build baseline lookup
    baseline_lookup = {}
    for r in baseline_results:
        baseline_lookup[(r["ticker"], r["signal_date"])] = r

    top_3 = ranked[:3]

    for scenario_label, scenario_data in top_3:
        scenario_results = scenario_data.get("results", [])
        if not scenario_results:
            continue

        full_label = scenario_data.get("label", scenario_label)
        print(f"\n  --- {full_label} ---")
        print(f"  {'Ticker':<8s} {'Date':<12s} {'BaseRet':>8s} {'NewRet':>8s} {'Diff':>7s} "
              f"{'BaseHold':>9s} {'NewHold':>8s} {'RSI_OR':>6s} {'WideStop':>9s} {'Change':>40s}")
        print("  " + "-" * 120)

        losers_became_winners = []
        winners_became_losers = []
        biggest_gains = []
        biggest_losses = []

        for new_r in sorted(scenario_results, key=lambda x: x["signal_date"]):
            key = (new_r["ticker"], new_r["signal_date"])
            base_r = baseline_lookup.get(key)
            if not base_r:
                continue

            base_ret = base_r["return"]
            new_ret = new_r["return"]
            diff = new_ret - base_ret
            base_hold = base_r["hold_days"]
            new_hold = new_r["hold_days"]
            rsi_or = new_r["rsi_overrides_used"]
            wide = "Y" if new_r["using_wide_stop"] else "N"

            # Determine what changed
            changes = []
            if new_r["using_wide_stop"] and not base_r.get("using_wide_stop", False):
                changes.append("wider stop (high vol)")
            if rsi_or > 0:
                changes.append(f"RSI override x{rsi_or}")
            if new_hold > base_hold:
                changes.append(f"held {new_hold - base_hold}d longer")
            elif new_hold < base_hold:
                changes.append(f"exited {base_hold - new_hold}d earlier")
            if base_r["exit_reason"] != new_r["exit_reason"]:
                changes.append(f"{base_r['exit_reason']}->{new_r['exit_reason']}")
            if not changes:
                changes.append("no change")
            change_str = "; ".join(changes)

            # Track flips
            if base_ret <= 0 and new_ret > 0:
                losers_became_winners.append((new_r["ticker"], base_ret, new_ret))
            if base_ret > 0 and new_ret <= 0:
                winners_became_losers.append((new_r["ticker"], base_ret, new_ret))
            biggest_gains.append((new_r["ticker"], diff, base_ret, new_ret))
            biggest_losses.append((new_r["ticker"], diff, base_ret, new_ret))

            print(f"  {new_r['ticker']:<8s} {new_r['signal_date']:<12s} {base_ret:>+7.1f}% "
                  f"{new_ret:>+7.1f}% {diff:>+6.1f}% "
                  f"{base_hold:>7d}d  {new_hold:>6d}d  {rsi_or:>5d}  {wide:>8s}  {change_str}")

        # Analysis
        print()
        # Losers -> Winners
        if losers_became_winners:
            print(f"  LOSERS that became WINNERS:")
            for ticker, base_ret, new_ret in losers_became_winners:
                print(f"    {ticker}: {base_ret:+.1f}% -> {new_ret:+.1f}%")
        else:
            print(f"  No losers became winners.")

        # Winners -> Losers
        if winners_became_losers:
            print(f"  WINNERS that became LOSERS:")
            for ticker, base_ret, new_ret in winners_became_losers:
                print(f"    {ticker}: {base_ret:+.1f}% -> {new_ret:+.1f}%")
        else:
            print(f"  No winners became losers.")

        # Biggest gains
        biggest_gains.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  Trades that GAINED MOST from adaptive rules:")
        for ticker, diff, base_ret, new_ret in biggest_gains[:3]:
            print(f"    {ticker}: {base_ret:+.1f}% -> {new_ret:+.1f}%  (improved by {diff:+.1f}pp)")

        # Biggest losses
        biggest_losses.sort(key=lambda x: x[1])
        print(f"\n  Trades that LOST MOST from adaptive rules:")
        for ticker, diff, base_ret, new_ret in biggest_losses[:3]:
            if diff < 0:
                print(f"    {ticker}: {base_ret:+.1f}% -> {new_ret:+.1f}%  (worsened by {diff:+.1f}pp)")
            else:
                print(f"    {ticker}: {base_ret:+.1f}% -> {new_ret:+.1f}%  (no worse)")

    # ======================================================================
    # Final summary comparison
    # ======================================================================
    print("\n" + "=" * 100)
    print("  FINAL SUMMARY: ALL SCENARIOS RANKED BY AVG RETURN")
    print("=" * 100)

    all_ranked = sorted(
        [(k, v) for k, v in all_scenarios.items() if v and "avg" in v],
        key=lambda x: x[1].get("avg", -999),
        reverse=True,
    )

    print(f"\n  {'#':>3s} {'Label':>55s} {'n':>3s} {'WR':>6s} {'Avg':>7s} {'AvgW':>7s} "
          f"{'AvgL':>6s} {'PF':>6s} {'Hold':>5s} {'RSI_OR':>6s}")
    print("  " + "-" * 110)
    for rank, (label, data) in enumerate(all_ranked, 1):
        pf = data.get("pf", 0)
        pf_s = f"{pf:.2f}" if pf != float("inf") else "inf"
        marker = " <-- BASELINE" if label == "A" else ""
        print(f"  {rank:>3d} {data.get('label', label):>55s} {data.get('n', 0):>3d} "
              f"{data.get('wr', 0):>5.1f}% {data.get('avg', 0):>+6.1f}% "
              f"{data.get('avg_w', 0):>+6.1f}% {data.get('avg_l', 0):>+5.1f}% "
              f"{pf_s:>6s} {data.get('avg_hold', 0):>4.1f}d "
              f"{data.get('total_overrides', 0):>5d}{marker}")

    print("\nDone.")


if __name__ == "__main__":
    main()
