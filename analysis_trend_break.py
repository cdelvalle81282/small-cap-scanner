"""
EPS Beat + Trend Break Analysis Script.

Thesis: When EPS beats expectations but the stock breaks below pre-earnings
support, that's a short signal (good news couldn't save it). When EPS beats
and the stock breaks above pre-earnings resistance, that's a long signal.

Uses swing-point support/resistance detection and reuses the ATR trailing
stop trade simulation from analysis_full_backtest.py.
"""

import sys
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, median
from collections import defaultdict

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import DB_PATH
from core.database import Database
from analysis_full_backtest import (
    compute_atr, get_price_data, simulate_long, simulate_short, show_stats,
)

# ─── Parameters ───────────────────────────────────────────────────────────────
MIN_EPS_SURPRISE = 0          # any positive surprise
SWING_LOOKBACK = 5            # bars each side for swing detection
PRE_EARNINGS_WINDOW = 60      # trading days before earnings to find S/R
POST_EARNINGS_WINDOW = 45     # max days after earnings to find break
ATR_MULT = 0.5                # trailing stop multiplier
MAX_HOLD = 15                 # max hold days


def find_swing_points(df, lookback=5):
    """Find swing highs and lows using a rolling window comparison."""
    highs = []
    lows = []
    for i in range(lookback, len(df) - lookback):
        window_high = df["high"].iloc[i - lookback:i + lookback + 1]
        if df["high"].iloc[i] == window_high.max():
            highs.append(i)
        window_low = df["low"].iloc[i - lookback:i + lookback + 1]
        if df["low"].iloc[i] == window_low.min():
            lows.append(i)
    return highs, lows


def find_support_resistance(df, earnings_idx, lookback=SWING_LOOKBACK,
                            pre_window=PRE_EARNINGS_WINDOW):
    """Find the most recent swing high/low in the pre-earnings window."""
    start_idx = max(0, earnings_idx - pre_window)
    pre_df = df.iloc[start_idx:earnings_idx].copy()
    if len(pre_df) < lookback * 2 + 1:
        return None, None

    high_idxs, low_idxs = find_swing_points(pre_df, lookback)

    support = None
    if low_idxs:
        # Most recent swing low — use the low price at that bar
        last_low_idx = low_idxs[-1]
        support = pre_df["low"].iloc[last_low_idx]

    resistance = None
    if high_idxs:
        # Most recent swing high — use the high price at that bar
        last_high_idx = high_idxs[-1]
        resistance = pre_df["high"].iloc[last_high_idx]

    return support, resistance


def scan_for_breaks(db, backtest_start="2021-01-01", backtest_end="2026-03-20"):
    """Scan all tickers for EPS beat + trend break signals."""
    conn = sqlite3.connect(DB_PATH)

    # Get all tickers with earnings data
    tickers = [r[0] for r in conn.execute(
        "SELECT DISTINCT ticker FROM earnings ORDER BY ticker"
    ).fetchall()]

    print(f"Scanning {len(tickers)} tickers for EPS beat + trend break signals...")

    signals = []

    for ticker in tickers:
        # Get earnings with positive surprise in our date range
        earnings = conn.execute(
            """SELECT report_date, eps_change_pct, eps_actual, eps_prior
               FROM earnings
               WHERE ticker = ? AND report_date >= ? AND report_date <= ?
                 AND eps_change_pct > ?
               ORDER BY report_date""",
            (ticker, backtest_start, backtest_end, MIN_EPS_SURPRISE),
        ).fetchall()

        if not earnings:
            continue

        for earn in earnings:
            report_date_str = earn[0]
            eps_change_pct = earn[1]
            report_dt = date.fromisoformat(report_date_str)

            # Fetch price data: pre-earnings window + post-earnings window
            fetch_start = (report_dt - timedelta(days=PRE_EARNINGS_WINDOW * 2)).isoformat()
            fetch_end = (report_dt + timedelta(days=POST_EARNINGS_WINDOW * 2)).isoformat()
            rows = db.get_daily_prices(ticker, fetch_start, fetch_end)
            if not rows:
                continue

            df = pd.DataFrame(rows)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            # Find earnings date index in price data
            earn_idx_list = df.index[df["date"] == report_date_str].tolist()
            if not earn_idx_list:
                # Earnings on non-trading day — find next trading day
                earn_idx_list = df.index[df["date"] > report_date_str].tolist()
                if not earn_idx_list:
                    continue
            earn_idx = earn_idx_list[0]

            # Need enough pre-earnings data
            if earn_idx < SWING_LOOKBACK * 2 + 1:
                continue

            # Find support and resistance
            support, resistance = find_support_resistance(
                df, earn_idx, SWING_LOOKBACK, PRE_EARNINGS_WINDOW
            )

            if support is None and resistance is None:
                continue

            # Compute 20-day avg volume before earnings for volume ratio
            vol_start = max(0, earn_idx - 20)
            avg_vol_20d = df["volume"].iloc[vol_start:earn_idx].mean()

            # Pre-earnings trend: 20 MA vs 50 MA
            trend_context = "flat"
            if earn_idx >= 50:
                ma20 = df["close"].iloc[earn_idx - 20:earn_idx].mean()
                ma50 = df["close"].iloc[earn_idx - 50:earn_idx].mean()
                if ma20 > ma50 * 1.01:
                    trend_context = "bullish"
                elif ma20 < ma50 * 0.99:
                    trend_context = "bearish"

            # Scan post-earnings for breaks
            post_end = min(earn_idx + POST_EARNINGS_WINDOW + 1, len(df))
            for j in range(earn_idx + 1, post_end):
                close = df["close"].iloc[j]
                break_date = df["date"].iloc[j]
                vol_on_break = df["volume"].iloc[j]
                vol_ratio = vol_on_break / avg_vol_20d if avg_vol_20d > 0 else 1.0
                days_to_break = j - earn_idx

                # Short signal: close breaks below support
                if support is not None and close < support:
                    signals.append({
                        "ticker": ticker,
                        "earnings_date": report_date_str,
                        "eps_change_pct": eps_change_pct,
                        "break_date": break_date,
                        "direction": "SHORT",
                        "level_type": "support",
                        "level_price": support,
                        "break_price": close,
                        "vol_ratio": vol_ratio,
                        "days_to_break": days_to_break,
                        "trend_context": trend_context,
                    })
                    break  # Take first break only

                # Long signal: close breaks above resistance
                if resistance is not None and close > resistance:
                    signals.append({
                        "ticker": ticker,
                        "earnings_date": report_date_str,
                        "eps_change_pct": eps_change_pct,
                        "break_date": break_date,
                        "direction": "LONG",
                        "level_type": "resistance",
                        "level_price": resistance,
                        "break_price": close,
                        "vol_ratio": vol_ratio,
                        "days_to_break": days_to_break,
                        "trend_context": trend_context,
                    })
                    break  # Take first break only

    conn.close()
    return signals


def simulate_trades(db, signals):
    """Run trade simulation on each signal using full_backtest logic."""
    trades = []
    skipped = 0

    for sig in signals:
        # get_price_data needs a signal date and returns df, atr, sig_idx
        df, atr_series, sig_idx = get_price_data(
            db, sig["ticker"], sig["break_date"], lookback=80, forward=40
        )
        if df is None:
            skipped += 1
            continue

        vol_ratio = sig["vol_ratio"]

        if sig["direction"] == "LONG":
            trade = simulate_long(df, atr_series, sig_idx, vol_ratio,
                                  atr_mult=ATR_MULT, max_days=MAX_HOLD)
        else:
            trade = simulate_short(df, atr_series, sig_idx, vol_ratio,
                                   atr_mult=ATR_MULT, max_days=MAX_HOLD)

        if trade is None:
            skipped += 1
            continue

        # Merge signal metadata into trade
        trade["ticker"] = sig["ticker"]
        trade["signal_date"] = sig["break_date"]
        trade["earnings_date"] = sig["earnings_date"]
        trade["eps_pct"] = sig["eps_change_pct"]
        trade["direction"] = sig["direction"]
        trade["level_type"] = sig["level_type"]
        trade["level_price"] = sig["level_price"]
        trade["break_price"] = sig["break_price"]
        trade["days_to_break"] = sig["days_to_break"]
        trade["trend_context"] = sig["trend_context"]
        trades.append(trade)

    if skipped:
        print(f"  Skipped {skipped} signals (insufficient price data)")

    return trades


def vol_bucket(ratio):
    if ratio < 0.8:
        return "<0.8x"
    elif ratio < 1.2:
        return "0.8-1.2x"
    elif ratio < 2.0:
        return "1.2-2.0x"
    else:
        return ">2.0x"


def eps_bucket(pct):
    if pct < 25:
        return "<25%"
    elif pct < 50:
        return "25-50%"
    else:
        return ">50%"


def days_bucket(days):
    if days < 5:
        return "<5d"
    elif days < 15:
        return "5-15d"
    elif days < 30:
        return "15-30d"
    else:
        return ">30d"


def quick_stats(trades):
    """Return a compact stats dict without printing."""
    if not trades:
        return {"n": 0, "wr": 0, "avg": 0, "med": 0, "pf": 0, "net": 0,
                "avg_w": 0, "avg_l": 0}
    wins = [t for t in trades if t["return"] > 0]
    losses = [t for t in trades if t["return"] <= 0]
    rets = [t["return"] for t in trades]
    wr = len(wins) / len(trades) * 100
    avg = mean(rets)
    med = median(rets)
    gg = sum(t["return"] for t in wins)
    gl = sum(t["return"] for t in losses)
    pf = gg / abs(gl) if gl else float("inf")
    avg_w = mean([t["return"] for t in wins]) if wins else 0
    avg_l = mean([t["return"] for t in losses]) if losses else 0
    return {"n": len(trades), "wr": wr, "avg": avg, "med": med, "pf": pf,
            "net": gg + gl, "avg_w": avg_w, "avg_l": avg_l,
            "wins": len(wins), "losses": len(losses)}


def print_combo_row(label, s):
    """Print one row of the combo scanner table."""
    if s["n"] == 0:
        return
    pf_s = f"{s['pf']:.2f}" if s["pf"] != float("inf") else "inf"
    print(f"  {label:55s} {s['n']:>5d} {s['wr']:>6.1f}% {s['avg']:>+7.1f}% "
          f"{s['med']:>+7.1f}% {pf_s:>7s} {s['net']:>+8.1f}% "
          f"{s['avg_w']:>+6.1f}% {s['avg_l']:>+6.1f}%")


def explore_filters(trades, long_trades, short_trades):
    """Deep exploration of promising filter combinations."""

    header = (f"  {'Filter':55s} {'N':>5s} {'WR':>7s} {'Avg':>8s} "
              f"{'Med':>8s} {'PF':>7s} {'Net':>9s} {'AvgW':>7s} {'AvgL':>7s}")
    divider = "  " + "-" * 120

    # ==================================================================
    # A) Fine-grained days-to-break analysis
    # ==================================================================
    print(f"\n\n{'=' * 120}")
    print("  DEEP DIVE A: DAYS TO BREAK — FINE-GRAINED")
    print(f"{'=' * 120}")
    print(header)
    print(divider)

    day_ranges = [
        ("1d (gap day)", 1, 1),
        ("2-3d", 2, 3),
        ("4-7d (first week)", 4, 7),
        ("8-14d (second week)", 8, 14),
        ("15-21d", 15, 21),
        ("22-30d", 22, 30),
        ("31-45d", 31, 45),
    ]
    for label, lo, hi in day_ranges:
        subset = [t for t in trades if lo <= t["days_to_break"] <= hi]
        print_combo_row(f"All — {label}", quick_stats(subset))

    print()
    for label, lo, hi in day_ranges:
        subset = [t for t in long_trades if lo <= t["days_to_break"] <= hi]
        print_combo_row(f"LONG — {label}", quick_stats(subset))

    print()
    for label, lo, hi in day_ranges:
        subset = [t for t in short_trades if lo <= t["days_to_break"] <= hi]
        print_combo_row(f"SHORT — {label}", quick_stats(subset))

    # ==================================================================
    # B) 15-30d bucket deep dive — by direction, trend, volume, EPS
    # ==================================================================
    d15_30 = [t for t in trades if 15 <= t["days_to_break"] <= 30]

    print(f"\n\n{'=' * 120}")
    print(f"  DEEP DIVE B: 15-30d BREAK DELAY — SLICED (n={len(d15_30)})")
    print(f"{'=' * 120}")
    print(header)
    print(divider)

    print_combo_row("15-30d ALL", quick_stats(d15_30))
    print_combo_row("15-30d LONG", quick_stats([t for t in d15_30 if t["direction"] == "LONG"]))
    print_combo_row("15-30d SHORT", quick_stats([t for t in d15_30 if t["direction"] == "SHORT"]))
    print()

    # by trend within 15-30d
    for ctx in ["bullish", "bearish", "flat"]:
        subset = [t for t in d15_30 if t["trend_context"] == ctx]
        print_combo_row(f"15-30d + trend={ctx}", quick_stats(subset))
        print_combo_row(f"  -> LONG + trend={ctx}",
                        quick_stats([t for t in subset if t["direction"] == "LONG"]))
        print_combo_row(f"  -> SHORT + trend={ctx}",
                        quick_stats([t for t in subset if t["direction"] == "SHORT"]))
    print()

    # by volume within 15-30d
    for vb in ["<0.8x", "0.8-1.2x", "1.2-2.0x", ">2.0x"]:
        subset = [t for t in d15_30 if vol_bucket(t["vol_ratio"]) == vb]
        print_combo_row(f"15-30d + vol={vb}", quick_stats(subset))
    print()

    # by EPS within 15-30d
    for eb in ["<25%", "25-50%", ">50%"]:
        subset = [t for t in d15_30 if eps_bucket(t["eps_pct"]) == eb]
        print_combo_row(f"15-30d + eps={eb}", quick_stats(subset))

    # ==================================================================
    # C) Bullish trend context deep dive
    # ==================================================================
    bullish = [t for t in trades if t["trend_context"] == "bullish"]

    print(f"\n\n{'=' * 120}")
    print(f"  DEEP DIVE C: BULLISH PRE-EARNINGS TREND — SLICED (n={len(bullish)})")
    print(f"{'=' * 120}")
    print(header)
    print(divider)

    print_combo_row("Bullish ALL", quick_stats(bullish))
    print_combo_row("Bullish LONG", quick_stats([t for t in bullish if t["direction"] == "LONG"]))
    print_combo_row("Bullish SHORT", quick_stats([t for t in bullish if t["direction"] == "SHORT"]))
    print()

    # bullish + days-to-break
    for label, lo, hi in day_ranges:
        subset = [t for t in bullish if lo <= t["days_to_break"] <= hi]
        print_combo_row(f"Bullish + {label}", quick_stats(subset))
    print()

    # bullish + volume
    for vb in ["<0.8x", "0.8-1.2x", "1.2-2.0x", ">2.0x"]:
        subset = [t for t in bullish if vol_bucket(t["vol_ratio"]) == vb]
        print_combo_row(f"Bullish + vol={vb}", quick_stats(subset))
    print()

    # bullish + EPS
    for eb in ["<25%", "25-50%", ">50%"]:
        subset = [t for t in bullish if eps_bucket(t["eps_pct"]) == eb]
        print_combo_row(f"Bullish + eps={eb}", quick_stats(subset))

    # ==================================================================
    # D) Bearish trend + SHORT (the core thesis)
    # ==================================================================
    bearish_short = [t for t in short_trades if t["trend_context"] == "bearish"]

    print(f"\n\n{'=' * 120}")
    print(f"  DEEP DIVE D: BEARISH TREND + SHORT — THE CORE THESIS (n={len(bearish_short)})")
    print(f"{'=' * 120}")
    print(header)
    print(divider)

    print_combo_row("Bearish SHORT ALL", quick_stats(bearish_short))
    print()

    for label, lo, hi in day_ranges:
        subset = [t for t in bearish_short if lo <= t["days_to_break"] <= hi]
        print_combo_row(f"Bearish SHORT + {label}", quick_stats(subset))
    print()

    for vb in ["<0.8x", "0.8-1.2x", "1.2-2.0x", ">2.0x"]:
        subset = [t for t in bearish_short if vol_bucket(t["vol_ratio"]) == vb]
        print_combo_row(f"Bearish SHORT + vol={vb}", quick_stats(subset))
    print()

    for eb in ["<25%", "25-50%", ">50%"]:
        subset = [t for t in bearish_short if eps_bucket(t["eps_pct"]) == eb]
        print_combo_row(f"Bearish SHORT + eps={eb}", quick_stats(subset))

    # ==================================================================
    # E) Bullish trend + LONG (expected best case for longs)
    # ==================================================================
    bullish_long = [t for t in long_trades if t["trend_context"] == "bullish"]

    print(f"\n\n{'=' * 120}")
    print(f"  DEEP DIVE E: BULLISH TREND + LONG (n={len(bullish_long)})")
    print(f"{'=' * 120}")
    print(header)
    print(divider)

    print_combo_row("Bullish LONG ALL", quick_stats(bullish_long))
    print()

    for label, lo, hi in day_ranges:
        subset = [t for t in bullish_long if lo <= t["days_to_break"] <= hi]
        print_combo_row(f"Bullish LONG + {label}", quick_stats(subset))
    print()

    for vb in ["<0.8x", "0.8-1.2x", "1.2-2.0x", ">2.0x"]:
        subset = [t for t in bullish_long if vol_bucket(t["vol_ratio"]) == vb]
        print_combo_row(f"Bullish LONG + vol={vb}", quick_stats(subset))
    print()

    for eb in ["<25%", "25-50%", ">50%"]:
        subset = [t for t in bullish_long if eps_bucket(t["eps_pct"]) == eb]
        print_combo_row(f"Bullish LONG + eps={eb}", quick_stats(subset))

    # ==================================================================
    # F) Combo scanner — exhaustive 2- and 3-filter combos, ranked by PF
    # ==================================================================
    print(f"\n\n{'=' * 120}")
    print("  DEEP DIVE F: COMBO SCANNER — ALL 2- AND 3-FILTER COMBOS (min 20 trades)")
    print(f"{'=' * 120}")

    directions = [("ALL", trades), ("LONG", long_trades), ("SHORT", short_trades)]
    trend_filters = [("any_trend", None), ("bullish", "bullish"), ("bearish", "bearish")]
    day_filters = [
        ("any_delay", None, None),
        ("<5d", 1, 4),
        ("5-14d", 5, 14),
        ("15-30d", 15, 30),
        (">30d", 31, 999),
    ]
    vol_filters = [
        ("any_vol", None, None),
        ("vol>=1.2x", 1.2, 999),
        ("vol>=2x", 2.0, 999),
        ("vol<1.2x", 0, 1.2),
    ]
    eps_filters = [
        ("any_eps", None, None),
        ("eps<25%", 0, 25),
        ("eps>=25%", 25, 999),
        ("eps>=50%", 50, 999),
    ]

    results = []
    for dir_label, dir_pool in directions:
        for trend_label, trend_val in trend_filters:
            for day_label, d_lo, d_hi in day_filters:
                for vol_label, v_lo, v_hi in vol_filters:
                    for eps_label, e_lo, e_hi in eps_filters:
                        # Skip the all-"any" combo (that's the baseline)
                        active = [x for x in [trend_label, day_label, vol_label, eps_label]
                                  if not x.startswith("any")]
                        if len(active) < 1:
                            continue

                        subset = list(dir_pool)
                        if trend_val is not None:
                            subset = [t for t in subset if t["trend_context"] == trend_val]
                        if d_lo is not None:
                            subset = [t for t in subset if d_lo <= t["days_to_break"] <= d_hi]
                        if v_lo is not None:
                            subset = [t for t in subset if v_lo <= t["vol_ratio"] < v_hi]
                        if e_lo is not None:
                            subset = [t for t in subset if e_lo <= t["eps_pct"] < e_hi]

                        if len(subset) < 20:
                            continue
                        s = quick_stats(subset)
                        combo_label = f"{dir_label} + {' + '.join(active)}"
                        results.append((combo_label, s))

    # Sort by PF descending
    results.sort(key=lambda x: x[1]["pf"] if x[1]["pf"] != float("inf") else 999, reverse=True)

    print(f"\n  TOP 40 COMBOS BY PROFIT FACTOR (min 20 trades):")
    print(header)
    print(divider)
    for label, s in results[:40]:
        print_combo_row(label, s)

    # Also show top by win rate
    results_wr = sorted(results, key=lambda x: x[1]["wr"], reverse=True)
    print(f"\n  TOP 40 COMBOS BY WIN RATE (min 20 trades):")
    print(header)
    print(divider)
    for label, s in results_wr[:40]:
        print_combo_row(label, s)

    # Also show top by avg return
    results_avg = sorted(results, key=lambda x: x[1]["avg"], reverse=True)
    print(f"\n  TOP 40 COMBOS BY AVG RETURN (min 20 trades):")
    print(header)
    print(divider)
    for label, s in results_avg[:40]:
        print_combo_row(label, s)

    # ==================================================================
    # G) Best combos — trade lists
    # ==================================================================
    # Find combos with PF >= 1.3 and n >= 20
    strong = [(l, s) for l, s in results if s["pf"] >= 1.3 and s["pf"] != float("inf") and s["n"] >= 25]
    if strong:
        print(f"\n\n{'=' * 120}")
        print(f"  STRONG COMBOS (PF >= 1.3, n >= 25): {len(strong)} found")
        print(f"{'=' * 120}")
        print(header)
        print(divider)
        for label, s in strong:
            print_combo_row(label, s)

    # ==================================================================
    # H) Year consistency check for top combos
    # ==================================================================
    print(f"\n\n{'=' * 120}")
    print("  DEEP DIVE H: YEAR-BY-YEAR FOR TOP COMBOS")
    print(f"{'=' * 120}")

    # Pick top 10 by PF for year breakdown
    top_combos_for_years = results[:10]

    for combo_label, combo_stats in top_combos_for_years:
        print(f"\n  {combo_label} (overall: n={combo_stats['n']}, "
              f"WR={combo_stats['wr']:.1f}%, PF={combo_stats['pf']:.2f})")

        # Reconstruct the subset — parse the label to figure out filters
        # Instead, store the subset. Let's redo with subsets stored.
        # For simplicity, just re-filter from trades using the label tokens.
        # This is a bit hacky but avoids storing 1000s of lists.
        subset = _reconstruct_subset(combo_label, trades, long_trades, short_trades)
        if not subset:
            print("    (could not reconstruct)")
            continue

        by_yr = defaultdict(list)
        for t in subset:
            by_yr[t["signal_date"][:4]].append(t)

        print(f"    {'Year':6s} {'N':>5s} {'WR':>7s} {'Avg':>8s} {'PF':>7s} {'Net':>8s}")
        print("    " + "-" * 50)
        for yr in sorted(by_yr.keys()):
            ys = quick_stats(by_yr[yr])
            pf_s = f"{ys['pf']:.2f}" if ys["pf"] != float("inf") else "inf"
            print(f"    {yr:6s} {ys['n']:>5d} {ys['wr']:>6.1f}% {ys['avg']:>+7.1f}% "
                  f"{pf_s:>7s} {ys['net']:>+7.1f}%")

    # ==================================================================
    # I) Regime change deep dive
    # ==================================================================
    explore_regime_change(trades, long_trades, short_trades)


def explore_regime_change(trades, long_trades, short_trades):
    """Deep regime analysis to diagnose why performance collapsed in 2025-2026."""

    if not trades:
        print("\n  No trades available for regime analysis.")
        return

    header = (f"  {'Filter':55s} {'N':>5s} {'WR':>7s} {'Avg':>8s} "
              f"{'Med':>8s} {'PF':>7s} {'Net':>9s} {'AvgW':>7s} {'AvgL':>7s}")
    divider = "  " + "-" * 120

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 1: REGIME SHIFT — BASIC STATS BY YEAR
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 120}")
    print("  REGIME SHIFT 1: BASIC STATS BY YEAR")
    print(f"{'=' * 120}")

    by_year = defaultdict(list)
    for t in trades:
        by_year[t["signal_date"][:4]].append(t)

    print(f"\n  {'Year':6s} {'N':>5s} {'Longs':>6s} {'Shorts':>6s} "
          f"{'%Long':>6s} {'AvgD2B':>7s} {'AvgVol':>7s} {'AvgEPS':>7s} "
          f"{'AvgPx':>7s} {'WR':>7s} {'PF':>7s} {'AvgRet':>8s}")
    print("  " + "-" * 105)

    for yr in sorted(by_year.keys()):
        yt = by_year[yr]
        s = quick_stats(yt)
        longs = [t for t in yt if t["direction"] == "LONG"]
        pct_long = len(longs) / len(yt) * 100 if yt else 0
        avg_d2b = mean([t["days_to_break"] for t in yt])
        avg_vol = mean([t["vol_ratio"] for t in yt])
        avg_eps = mean([t["eps_pct"] for t in yt])
        avg_px = mean([t["entry_price"] for t in yt])
        pf_s = f"{s['pf']:.2f}" if s["pf"] != float("inf") else "inf"
        print(f"  {yr:6s} {s['n']:>5d} {len(longs):>6d} {s['n'] - len(longs):>6d} "
              f"{pct_long:>5.1f}% {avg_d2b:>6.1f}d {avg_vol:>6.2f}x {avg_eps:>+6.1f}% "
              f"${avg_px:>6.2f} {s['wr']:>6.1f}% {pf_s:>7s} {s['avg']:>+7.1f}%")

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 2: REGIME SHIFT — STOP-OUT ANALYSIS BY YEAR
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 120}")
    print("  REGIME SHIFT 2: STOP-OUT ANALYSIS BY YEAR")
    print("  Hypothesis: 2025-2026 may have choppier price action causing more stops")
    print(f"{'=' * 120}")

    print(f"\n  {'Year':6s} {'N':>5s} {'%StopD1':>8s} {'%ATRStop':>9s} {'%Horizon':>9s} "
          f"{'HoldW':>6s} {'HoldL':>6s} {'AvgHold':>8s}")
    print("  " + "-" * 70)

    for yr in sorted(by_year.keys()):
        yt = by_year[yr]
        n = len(yt)
        n_d1 = sum(1 for t in yt if t["exit_reason"] == "stop_d1")
        n_atr = sum(1 for t in yt if t["exit_reason"] == "atr_stop")
        n_hor = sum(1 for t in yt if t["exit_reason"] == "horizon")
        wins = [t for t in yt if t["return"] > 0]
        losses = [t for t in yt if t["return"] <= 0]
        hold_w = mean([t["hold_days"] for t in wins]) if wins else 0.0
        hold_l = mean([t["hold_days"] for t in losses]) if losses else 0.0
        hold_all = mean([t["hold_days"] for t in yt])
        print(f"  {yr:6s} {n:>5d} {n_d1 / n * 100:>7.1f}% {n_atr / n * 100:>8.1f}% "
              f"{n_hor / n * 100:>8.1f}% {hold_w:>5.1f}d {hold_l:>5.1f}d {hold_all:>7.1f}d")

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 3: REGIME SHIFT — WINNER/LOSER CHARACTERISTICS BY YEAR
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 120}")
    print("  REGIME SHIFT 3: WINNER/LOSER CHARACTERISTICS BY YEAR")
    print("  Questions: Did losers get bigger? Did winners shrink?")
    print(f"{'=' * 120}")

    print(f"\n  {'Year':6s} {'N':>5s} {'AvgW':>8s} {'MedW':>8s} {'AvgL':>8s} "
          f"{'MedL':>8s} {'W/L Ratio':>10s}")
    print("  " + "-" * 65)

    for yr in sorted(by_year.keys()):
        yt = by_year[yr]
        wins = [t for t in yt if t["return"] > 0]
        losses = [t for t in yt if t["return"] <= 0]
        avg_w = mean([t["return"] for t in wins]) if wins else 0.0
        med_w = median([t["return"] for t in wins]) if wins else 0.0
        avg_l = mean([t["return"] for t in losses]) if losses else 0.0
        med_l = median([t["return"] for t in losses]) if losses else 0.0
        ratio = avg_w / abs(avg_l) if avg_l != 0 else float("inf")
        ratio_s = f"{ratio:.2f}" if ratio != float("inf") else "inf"
        print(f"  {yr:6s} {len(yt):>5d} {avg_w:>+7.1f}% {med_w:>+7.1f}% "
              f"{avg_l:>+7.1f}% {med_l:>+7.1f}% {ratio_s:>10s}")

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 4: REGIME SHIFT — SIGNAL QUALITY: BREAK DISTANCE BY YEAR
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 120}")
    print("  REGIME SHIFT 4: SIGNAL QUALITY — BREAK DISTANCE BY YEAR")
    print("  Metric: how far past the S/R level the break-day close was (% of level)")
    print("  Hypothesis: weaker/smaller breaks in 2025-2026 = less conviction")
    print(f"{'=' * 120}")

    print(f"\n  {'Year':6s} {'N':>5s} {'AvgBreakDist':>13s} {'MedBreakDist':>13s} "
          f"{'AvgW BreakDist':>15s} {'AvgL BreakDist':>15s}")
    print("  " + "-" * 75)

    for yr in sorted(by_year.keys()):
        yt = by_year[yr]
        dists = []
        for t in yt:
            lp = t["level_price"]
            bp = t["break_price"]
            if lp and lp != 0:
                dists.append(abs(bp - lp) / lp * 100)
            else:
                dists.append(0.0)
        # attach distances temporarily for winner/loser split
        yt_with_dist = list(zip(yt, dists))
        wins_dist = [d for t, d in yt_with_dist if t["return"] > 0]
        losses_dist = [d for t, d in yt_with_dist if t["return"] <= 0]
        avg_d = mean(dists) if dists else 0.0
        med_d = median(dists) if dists else 0.0
        avg_wd = mean(wins_dist) if wins_dist else 0.0
        avg_ld = mean(losses_dist) if losses_dist else 0.0
        print(f"  {yr:6s} {len(yt):>5d} {avg_d:>12.2f}% {med_d:>12.2f}% "
              f"{avg_wd:>14.2f}% {avg_ld:>14.2f}%")

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 5: REGIME SHIFT — DIRECTION ASYMMETRY BY YEAR
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 120}")
    print("  REGIME SHIFT 5: DIRECTION ASYMMETRY BY YEAR")
    print("  Question: which side (LONG or SHORT) collapsed?")
    print(f"{'=' * 120}")

    print(f"\n  {'Year':6s} {'Side':5s} {'N':>5s} {'WR':>7s} {'PF':>7s} {'AvgRet':>8s}")
    print("  " + "-" * 45)

    for yr in sorted(by_year.keys()):
        yt = by_year[yr]
        for side, pool in [("LONG", [t for t in yt if t["direction"] == "LONG"]),
                           ("SHORT", [t for t in yt if t["direction"] == "SHORT"])]:
            if not pool:
                print(f"  {yr:6s} {side:5s} {'—':>5s}")
                continue
            s = quick_stats(pool)
            pf_s = f"{s['pf']:.2f}" if s["pf"] != float("inf") else "inf"
            print(f"  {yr:6s} {side:5s} {s['n']:>5d} {s['wr']:>6.1f}% "
                  f"{pf_s:>7s} {s['avg']:>+7.1f}%")
        print()

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 6: REGIME SHIFT — TOP COMBOS: 2022-2024 vs 2025-2026
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 120}")
    print("  REGIME SHIFT 6: TOP COMBOS — EARLY (2022-2024) vs LATE (2025-2026)")
    print("  Same combo scanner, split by regime period (min 15 trades each)")
    print(f"{'=' * 120}")

    early_trades = [t for t in trades if t["signal_date"][:4] <= "2024"]
    late_trades = [t for t in trades if t["signal_date"][:4] >= "2025"]
    early_long = [t for t in early_trades if t["direction"] == "LONG"]
    early_short = [t for t in early_trades if t["direction"] == "SHORT"]
    late_long = [t for t in late_trades if t["direction"] == "LONG"]
    late_short = [t for t in late_trades if t["direction"] == "SHORT"]

    directions_reg = [
        ("ALL", early_trades, late_trades),
        ("LONG", early_long, late_long),
        ("SHORT", early_short, late_short),
    ]
    trend_filters = [("any_trend", None), ("bullish", "bullish"), ("bearish", "bearish")]
    day_filters = [
        ("any_delay", None, None), ("<5d", 1, 4), ("5-14d", 5, 14),
        ("15-30d", 15, 30), (">30d", 31, 999),
    ]
    vol_filters = [
        ("any_vol", None, None), ("vol>=1.2x", 1.2, 999),
        ("vol>=2x", 2.0, 999), ("vol<1.2x", 0, 1.2),
    ]
    eps_filters = [
        ("any_eps", None, None), ("eps<25%", 0, 25),
        ("eps>=25%", 25, 999), ("eps>=50%", 50, 999),
    ]

    def _run_combo_scan(pool_all, pool_long, pool_short, min_n=15):
        """Run the full combo scanner on a given pool and return ranked results."""
        dir_pools = [("ALL", pool_all), ("LONG", pool_long), ("SHORT", pool_short)]
        combos = []
        for dir_label, dir_pool in dir_pools:
            for trend_label, trend_val in trend_filters:
                for day_label, d_lo, d_hi in day_filters:
                    for vol_label, v_lo, v_hi in vol_filters:
                        for eps_label, e_lo, e_hi in eps_filters:
                            active = [x for x in [trend_label, day_label, vol_label, eps_label]
                                      if not x.startswith("any")]
                            if len(active) < 1:
                                continue
                            subset = list(dir_pool)
                            if trend_val is not None:
                                subset = [t for t in subset if t["trend_context"] == trend_val]
                            if d_lo is not None:
                                subset = [t for t in subset if d_lo <= t["days_to_break"] <= d_hi]
                            if v_lo is not None:
                                subset = [t for t in subset if v_lo <= t["vol_ratio"] < v_hi]
                            if e_lo is not None:
                                subset = [t for t in subset if e_lo <= t["eps_pct"] < e_hi]
                            if len(subset) < min_n:
                                continue
                            s = quick_stats(subset)
                            label = f"{dir_label} + {' + '.join(active)}"
                            combos.append((label, s))
        combos.sort(key=lambda x: x[1]["pf"] if x[1]["pf"] != float("inf") else 999, reverse=True)
        return combos

    early_combos = _run_combo_scan(early_trades, early_long, early_short)
    late_combos = _run_combo_scan(late_trades, late_long, late_short)

    for period_label, combos, pool in [
        (f"EARLY (2022-2024, n={len(early_trades)})", early_combos, early_trades),
        (f"LATE  (2025-2026, n={len(late_trades)})", late_combos, late_trades),
    ]:
        print(f"\n  --- TOP 20 COMBOS BY PF: {period_label} ---")
        if not pool:
            print("  No trades in this period.")
            continue
        print(header)
        print(divider)
        for lbl, s in combos[:20]:
            print_combo_row(lbl, s)

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 7: REGIME SHIFT — MONTHLY HEATMAP
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 120}")
    print("  REGIME SHIFT 7: MONTHLY HEATMAP")
    print("  Shows if the performance collapse was gradual or sudden")
    print(f"{'=' * 120}")

    by_month = defaultdict(list)
    for t in trades:
        ym = t["signal_date"][:7]  # "YYYY-MM"
        by_month[ym].append(t)

    print(f"\n  {'YearMon':9s} {'N':>5s} {'WR':>7s} {'AvgRet':>8s} {'Long':>5s} {'Short':>6s}")
    print("  " + "-" * 50)

    for ym in sorted(by_month.keys()):
        mt = by_month[ym]
        s = quick_stats(mt)
        n_long = sum(1 for t in mt if t["direction"] == "LONG")
        n_short = len(mt) - n_long
        wr_s = f"{s['wr']:.0f}%" if mt else "—"
        avg_s = f"{s['avg']:+.1f}%" if mt else "—"
        print(f"  {ym:9s} {len(mt):>5d} {wr_s:>7s} {avg_s:>8s} {n_long:>5d} {n_short:>6d}")

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 8: REGIME SHIFT — ROLLING 20-TRADE WINDOW
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 120}")
    print("  REGIME SHIFT 8: ROLLING 20-TRADE PERFORMANCE (chronological)")
    print("  Shows if performance degraded gradually or fell off a cliff")
    print(f"{'=' * 120}")

    window = 20
    sorted_trades = sorted(trades, key=lambda t: t["signal_date"])
    n_total = len(sorted_trades)

    print(f"\n  {'Window':8s} {'StartDate':12s} {'EndDate':12s} "
          f"{'N':>5s} {'WR':>7s} {'PF':>7s} {'AvgRet':>8s}")
    print("  " + "-" * 70)

    if n_total < window:
        print(f"  Not enough trades for rolling window (need {window}, have {n_total})")
    else:
        step = max(1, window // 2)  # 50% overlap — enough resolution without noise
        i = 0
        win_num = 1
        while i + window <= n_total:
            chunk = sorted_trades[i:i + window]
            s = quick_stats(chunk)
            start_dt = chunk[0]["signal_date"]
            end_dt = chunk[-1]["signal_date"]
            pf_s = f"{s['pf']:.2f}" if s["pf"] != float("inf") else "inf"
            print(f"  W{win_num:<6d} {start_dt:12s} {end_dt:12s} "
                  f"{s['n']:>5d} {s['wr']:>6.1f}% {pf_s:>7s} {s['avg']:>+7.1f}%")
            i += step
            win_num += 1


def _reconstruct_subset(label, trades, long_trades, short_trades):
    """Parse a combo label like 'LONG + bullish + 15-30d' and reconstruct the trade subset."""
    parts = [p.strip() for p in label.split("+")]
    if not parts:
        return []

    dir_part = parts[0]
    if dir_part == "LONG":
        subset = list(long_trades)
    elif dir_part == "SHORT":
        subset = list(short_trades)
    else:
        subset = list(trades)

    for p in parts[1:]:
        if p in ("bullish", "bearish", "flat"):
            subset = [t for t in subset if t["trend_context"] == p]
        elif p == "<5d":
            subset = [t for t in subset if t["days_to_break"] <= 4]
        elif p == "5-14d":
            subset = [t for t in subset if 5 <= t["days_to_break"] <= 14]
        elif p == "15-30d":
            subset = [t for t in subset if 15 <= t["days_to_break"] <= 30]
        elif p == ">30d":
            subset = [t for t in subset if t["days_to_break"] >= 31]
        elif p == "vol>=1.2x":
            subset = [t for t in subset if t["vol_ratio"] >= 1.2]
        elif p == "vol>=2x":
            subset = [t for t in subset if t["vol_ratio"] >= 2.0]
        elif p == "vol<1.2x":
            subset = [t for t in subset if t["vol_ratio"] < 1.2]
        elif p == "eps<25%":
            subset = [t for t in subset if t["eps_pct"] < 25]
        elif p == "eps>=25%":
            subset = [t for t in subset if t["eps_pct"] >= 25]
        elif p == "eps>=50%":
            subset = [t for t in subset if t["eps_pct"] >= 50]
    return subset


def main():
    db = Database(DB_PATH)
    db.initialize()

    # Data range info
    conn = sqlite3.connect(DB_PATH)
    price_range = conn.execute(
        "SELECT MIN(date), MAX(date), COUNT(DISTINCT date) FROM daily_prices"
    ).fetchone()
    eps_range = conn.execute(
        "SELECT MIN(report_date), MAX(report_date), COUNT(*) FROM earnings"
    ).fetchone()
    conn.close()

    print(f"Price data: {price_range[0]} to {price_range[1]} ({price_range[2]} dates)")
    print(f"Earnings data: {eps_range[0]} to {eps_range[1]} ({eps_range[2]} records)")

    # ── Step 1: Find signals ──────────────────────────────────────────────────
    signals = scan_for_breaks(db)
    print(f"\nFound {len(signals)} trend break signals")
    longs = [s for s in signals if s["direction"] == "LONG"]
    shorts = [s for s in signals if s["direction"] == "SHORT"]
    print(f"  Long (resistance break): {len(longs)}")
    print(f"  Short (support break): {len(shorts)}")

    # ── Step 2: Simulate trades ───────────────────────────────────────────────
    print("\nSimulating trades...")
    trades = simulate_trades(db, signals)
    trades.sort(key=lambda t: t["signal_date"])
    print(f"Simulated {len(trades)} trades")

    long_trades = [t for t in trades if t["direction"] == "LONG"]
    short_trades = [t for t in trades if t["direction"] == "SHORT"]

    # ── Section 1: Summary stats ──────────────────────────────────────────────
    print(f"\n\n{'=' * 100}")
    print("  EPS BEAT + TREND BREAK — SUMMARY")
    print(f"{'=' * 100}")

    show_stats("ALL TRADES", trades)
    show_stats("LONG ONLY (resistance break)", long_trades)
    show_stats("SHORT ONLY (support break)", short_trades)

    # ── Section 2: By volume bucket ───────────────────────────────────────────
    print(f"\n\n{'=' * 100}")
    print("  BY VOLUME ON BREAK DAY")
    print(f"{'=' * 100}")

    by_vol = defaultdict(list)
    for t in trades:
        by_vol[vol_bucket(t["vol_ratio"])].append(t)
    for bucket in ["<0.8x", "0.8-1.2x", "1.2-2.0x", ">2.0x"]:
        show_stats(f"Volume {bucket}", by_vol.get(bucket, []))

    # ── Section 3: By EPS surprise size ───────────────────────────────────────
    print(f"\n\n{'=' * 100}")
    print("  BY EPS SURPRISE SIZE")
    print(f"{'=' * 100}")

    by_eps = defaultdict(list)
    for t in trades:
        by_eps[eps_bucket(t["eps_pct"])].append(t)
    for bucket in ["<25%", "25-50%", ">50%"]:
        show_stats(f"EPS surprise {bucket}", by_eps.get(bucket, []))

    # ── Section 4: By days to break ───────────────────────────────────────────
    print(f"\n\n{'=' * 100}")
    print("  BY DAYS FROM EARNINGS TO BREAK")
    print(f"{'=' * 100}")

    by_days = defaultdict(list)
    for t in trades:
        by_days[days_bucket(t["days_to_break"])].append(t)
    for bucket in ["<5d", "5-15d", "15-30d", ">30d"]:
        show_stats(f"Days to break {bucket}", by_days.get(bucket, []))

    # ── Section 5: By pre-earnings trend ──────────────────────────────────────
    print(f"\n\n{'=' * 100}")
    print("  BY PRE-EARNINGS TREND (20 MA vs 50 MA)")
    print(f"{'=' * 100}")

    by_trend = defaultdict(list)
    for t in trades:
        by_trend[t["trend_context"]].append(t)
    for ctx in ["bullish", "bearish", "flat"]:
        show_stats(f"Pre-earnings trend: {ctx}", by_trend.get(ctx, []))

    # ── Section 6: By year ────────────────────────────────────────────────────
    print(f"\n\n{'=' * 100}")
    print("  BY YEAR")
    print(f"{'=' * 100}")

    by_year = defaultdict(list)
    for t in trades:
        by_year[t["signal_date"][:4]].append(t)

    print(f"\n  {'Year':6s} {'Trades':>7s} {'Long':>6s} {'Short':>6s} {'WR':>7s} "
          f"{'AvgRet':>8s} {'PF':>8s} {'Net':>8s}")
    print("  " + "-" * 65)

    for year in sorted(by_year.keys()):
        yr_trades = by_year[year]
        yr_longs = [t for t in yr_trades if t["direction"] == "LONG"]
        yr_shorts = [t for t in yr_trades if t["direction"] == "SHORT"]
        wins = [t for t in yr_trades if t["return"] > 0]
        wr = len(wins) / len(yr_trades) * 100
        avg = mean([t["return"] for t in yr_trades])
        gg = sum(t["return"] for t in wins)
        gl = sum(t["return"] for t in yr_trades if t["return"] <= 0)
        pf = gg / abs(gl) if gl else float("inf")
        pf_s = f"{pf:.2f}" if pf != float("inf") else "inf"
        net = gg + gl
        print(f"  {year:6s} {len(yr_trades):>7d} {len(yr_longs):>6d} {len(yr_shorts):>6d} "
              f"{wr:>6.1f}% {avg:>+7.1f}% {pf_s:>8s} {net:>+7.1f}%")

    # ── Section 7: Trade list ─────────────────────────────────────────────────
    print(f"\n\n{'=' * 100}")
    print("  TRADE LIST — CHRONOLOGICAL")
    print(f"{'=' * 100}")

    print(f"\n  {'Dir':5s} {'Ticker':7s} {'Earnings':12s} {'Break':12s} {'D2B':>4s} "
          f"{'EPS%':>6s} {'Vol':>5s} {'Trend':8s} {'Level$':>8s} "
          f"{'Entry$':>8s} {'Exit$':>8s} {'Return':>8s} {'Hold':>5s} {'Exit':10s} {'W/L':>4s}")
    print("  " + "-" * 130)

    for t in trades:
        wl = "W" if t["return"] > 0 else "L"
        print(f"  {t['direction']:5s} {t['ticker']:7s} {t['earnings_date']:12s} "
              f"{t['signal_date']:12s} {t['days_to_break']:>3d}d "
              f"{t['eps_pct']:>+5.0f}% {t['vol_ratio']:>4.1f}x {t['trend_context']:8s} "
              f"${t['level_price']:>7.2f} ${t['entry_price']:>7.2f} ${t['exit_price']:>7.2f} "
              f"{t['return']:>+7.1f}% {t['hold_days']:>4d}d {t['exit_reason']:10s} {wl:>3s}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 100}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 100}")

    if trades:
        total_ret = sum(t["return"] for t in trades)
        first = trades[0]["signal_date"]
        last = trades[-1]["signal_date"]
        print(f"\n  Signal range: {first} to {last}")
        print(f"  Total trades: {len(trades)} ({len(long_trades)} long, {len(short_trades)} short)")
        print(f"  Cumulative return (equal-weight): {total_ret:+.1f}%")

        best = max(trades, key=lambda t: t["return"])
        worst = min(trades, key=lambda t: t["return"])
        print(f"  Best trade: {best['ticker']} {best['signal_date']} "
              f"{best['direction']} {best['return']:+.1f}%")
        print(f"  Worst trade: {worst['ticker']} {worst['signal_date']} "
              f"{worst['direction']} {worst['return']:+.1f}%")

        # Max consecutive losses
        max_consec = 0
        cur_consec = 0
        for t in trades:
            if t["return"] <= 0:
                cur_consec += 1
                max_consec = max(max_consec, cur_consec)
            else:
                cur_consec = 0
        print(f"  Max consecutive losses: {max_consec}")
    else:
        print("\n  No trades found.")

    # ==================================================================
    # DEEP EXPLORATION — promising filters & combinations
    # ==================================================================
    explore_filters(trades, long_trades, short_trades)


if __name__ == "__main__":
    main()
