"""
Trigger day (Day 0) price action analysis for vol >= 1.2x signals.

Questions:
1. Does Day 0 gap up from prior close?
2. Where does Day 0 close relative to its range? (close near high = move happened early/held)
3. Is Day 0 midpoint above prior close? (if so, move visible by midday)
4. How does entering at Day 0 midpoint or Day 0 close compare to Day 1 open?
5. What's the Day 0 open-to-high move? (tells us about early-session strength)
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


CONFIGS = [
    (25.0, 30), (30.0, 60),
]


def compute_atr(df, period=14):
    d = df.copy()
    d["prev_close"] = d["close"].shift(1)
    d["tr"] = d.apply(lambda r: max(
        r["high"] - r["low"],
        abs(r["high"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
        abs(r["low"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
    ), axis=1)
    return d["tr"].rolling(period).mean()


def simulate_entry(db, ticker, signal_date, entry_mode="d1_open", vol_min=1.2, atr_mult=0.5, max_days=15):
    """
    Simulate trade with different entry points.
    entry_mode: "d0_midpoint", "d0_close", "d1_open", "d1_midpoint"
    """
    signal_dt = date.fromisoformat(signal_date)
    fetch_start = (signal_dt - timedelta(days=80)).isoformat()
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

    if sig_idx < 20:
        return None

    d0 = df.iloc[sig_idx]

    # Volume filter on signal day
    vol_20d = df["volume"].iloc[sig_idx - 20:sig_idx].mean()
    if vol_20d <= 0:
        return None
    sig_vol_ratio = d0["volume"] / vol_20d
    if sig_vol_ratio < vol_min:
        return {"filtered": True}

    # Determine entry price and index
    if entry_mode == "d0_midpoint":
        entry_price = (d0["high"] + d0["low"]) / 2
        entry_idx = sig_idx
    elif entry_mode == "d0_close":
        entry_price = d0["close"]
        entry_idx = sig_idx
    elif entry_mode == "d1_open":
        d1_idx = sig_idx + 1
        if d1_idx >= len(df):
            return None
        entry_price = df.iloc[d1_idx]["open"]
        entry_idx = d1_idx
    elif entry_mode == "d1_midpoint":
        d1_idx = sig_idx + 1
        if d1_idx >= len(df):
            return None
        d1 = df.iloc[d1_idx]
        entry_price = (d1["high"] + d1["low"]) / 2
        entry_idx = d1_idx
    else:
        return None

    if entry_price <= 0:
        return None

    # Walk forward from entry
    highest_close = entry_price
    prev_row = df.iloc[entry_idx - 1] if entry_idx > 0 else df.iloc[entry_idx]

    for day_offset in range(0, max_days + 1):
        cur_idx = entry_idx + day_offset
        if cur_idx >= len(df):
            break

        row = df.iloc[cur_idx]
        close = row["close"]

        if day_offset == 0:
            if close > highest_close:
                highest_close = close
            prev_row = row
            continue

        if day_offset == 1:
            prev_low = prev_row["low"]
            if close < prev_low:
                ret = (close - entry_price) / entry_price * 100
                return {"return": ret, "hold_days": day_offset, "exit_reason": "stop_d1",
                        "entry_price": entry_price, "exit_price": close,
                        "vol_ratio": sig_vol_ratio, "entry_mode": entry_mode}
        else:
            atr_val = atr_series.iloc[cur_idx - 1] if cur_idx - 1 < len(atr_series) else None
            if atr_val is None or pd.isna(atr_val):
                atr_val = (df["high"].iloc[max(0, cur_idx-14):cur_idx].mean() -
                           df["low"].iloc[max(0, cur_idx-14):cur_idx].mean())
            stop_level = highest_close - atr_mult * atr_val
            if close < stop_level:
                ret = (close - entry_price) / entry_price * 100
                return {"return": ret, "hold_days": day_offset, "exit_reason": "atr_stop",
                        "entry_price": entry_price, "exit_price": close,
                        "vol_ratio": sig_vol_ratio, "entry_mode": entry_mode}

        if close > highest_close:
            highest_close = close
        prev_row = row

    last_idx = min(entry_idx + max_days, len(df) - 1)
    exit_price = df.iloc[last_idx]["close"]
    ret = (exit_price - entry_price) / entry_price * 100
    return {"return": ret, "hold_days": max_days, "exit_reason": "horizon",
            "entry_price": entry_price, "exit_price": exit_price,
            "vol_ratio": sig_vol_ratio, "entry_mode": entry_mode}


def get_day0_profile(db, ticker, signal_date):
    """Get detailed Day 0 price action profile."""
    signal_dt = date.fromisoformat(signal_date)
    fetch_start = (signal_dt - timedelta(days=80)).isoformat()
    fetch_end = (signal_dt + timedelta(days=10)).isoformat()

    rows = db.get_daily_prices(ticker, fetch_start, fetch_end)
    if not rows:
        return None

    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    sig_idx_list = df.index[df["date"] == signal_date].tolist()
    if not sig_idx_list:
        return None
    sig_idx = sig_idx_list[0]

    if sig_idx < 20:
        return None

    d0 = df.iloc[sig_idx]

    # Volume
    vol_20d = df["volume"].iloc[sig_idx - 20:sig_idx].mean()
    if vol_20d <= 0:
        return None
    sig_vol_ratio = d0["volume"] / vol_20d
    if sig_vol_ratio < 1.2:
        return None

    # Prior day
    prev = df.iloc[sig_idx - 1]
    prev_close = prev["close"]

    d0_open = d0["open"]
    d0_high = d0["high"]
    d0_low = d0["low"]
    d0_close = d0["close"]
    d0_mid = (d0_high + d0_low) / 2
    d0_range = d0_high - d0_low if d0_high > d0_low else 0.001

    # Day 1 (next day)
    d1_idx = sig_idx + 1
    d1_open = df.iloc[d1_idx]["open"] if d1_idx < len(df) else None

    # Gap from prior close
    gap_pct = (d0_open - prev_close) / prev_close * 100

    # Where close sits in range (0 = at low, 1 = at high)
    close_position = (d0_close - d0_low) / d0_range

    # Open-to-high as % of range (how much upside from open)
    open_to_high_pct = (d0_high - d0_open) / d0_open * 100
    open_to_low_pct = (d0_low - d0_open) / d0_open * 100

    # Is midpoint above prior close? (move visible by midday)
    mid_above_prev = d0_mid > prev_close

    # Is open above prior close? (gap up)
    open_above_prev = d0_open > prev_close

    # Close vs open
    d0_green = d0_close > d0_open
    d0_body_pct = abs(d0_close - d0_open) / d0_range * 100

    # How much of the day's range was above the open?
    if d0_range > 0:
        pct_range_above_open = (d0_high - d0_open) / d0_range * 100
    else:
        pct_range_above_open = 50

    # D1 open vs D0 close (overnight gap)
    overnight_gap = ((d1_open - d0_close) / d0_close * 100) if d1_open else None

    # D0 midpoint vs D1 open (would D0 mid entry be better than D1 open?)
    d0_mid_vs_d1_open = ((d1_open - d0_mid) / d0_mid * 100) if d1_open else None

    # D0 close vs D1 open
    d0_close_vs_d1_open = ((d1_open - d0_close) / d0_close * 100) if d1_open else None

    return {
        "prev_close": prev_close,
        "d0_open": d0_open,
        "d0_high": d0_high,
        "d0_low": d0_low,
        "d0_close": d0_close,
        "d0_mid": d0_mid,
        "d1_open": d1_open,
        "vol_ratio": sig_vol_ratio,
        "gap_pct": gap_pct,
        "close_position": close_position,  # 0-1, where in the range
        "open_to_high_pct": open_to_high_pct,
        "open_to_low_pct": open_to_low_pct,
        "mid_above_prev": mid_above_prev,
        "open_above_prev": open_above_prev,
        "d0_green": d0_green,
        "d0_body_pct": d0_body_pct,
        "pct_range_above_open": pct_range_above_open,
        "overnight_gap": overnight_gap,
        "d0_mid_vs_d1_open": d0_mid_vs_d1_open,
        "d0_close_vs_d1_open": d0_close_vs_d1_open,
        # Return from various entry points to D0 close
        "prev_close_to_d0_close": (d0_close - prev_close) / prev_close * 100,
        "d0_open_to_d0_close": (d0_close - d0_open) / d0_open * 100,
    }


def show(label, results):
    if not results:
        print(f"  {label:55s} | n=  0")
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
    hold_w = mean([r["hold_days"] for r in wins]) if wins else 0
    hold_l = mean([r["hold_days"] for r in losses]) if losses else 0
    print(f"  {label:55s} | n={len(results):>3d} | WR={wr:>5.1f}% | Avg={avg:>+7.1f}% "
          f"| AvgW={avg_w:>+6.1f}% | AvgL={avg_l:>+6.1f}% | PF={pf_s:>6s} "
          f"| HoldW={hold_w:.1f}d HoldL={hold_l:.1f}d")


def main():
    db = Database(DB_PATH)
    db.initialize()

    print("Loading signals...")
    signal_cache = {}
    for eps_thresh, tw in CONFIGS:
        sc = ScannerConfig(
            min_price=1.0, max_price=50.0, min_market_cap=50_000_000,
            max_market_cap=10_000_000_000, ma_crossover_pairs=[(20, 50)],
            eps_change_threshold=eps_thresh, trend_window_days=tw, direction="both",
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

    # Collect unique vol >= 1.2x signals with day 0 profiles
    seen = set()
    all_signals = []

    for (eps_thresh, tw), (c17, raw) in sorted(signal_cache.items()):
        for s in c17:
            key = (s["ticker"], s["trend_date"])
            if key in seen:
                continue

            profile = get_day0_profile(db, s["ticker"], s["trend_date"])
            if not profile:
                continue

            seen.add(key)
            all_signals.append({
                "ticker": s["ticker"],
                "signal_date": s["trend_date"],
                "eps_pct": s.get("eps_change_pct", 0),
                "profile": profile,
            })

    print(f"  {len(all_signals)} unique signals with vol >= 1.2x\n")

    # ======================================================================
    print("=" * 160)
    print("  TRIGGER DAY (Day 0) PRICE ACTION: What happens on the cross day?")
    print("=" * 160)

    # Detailed table
    print(f"\n  {'Ticker':7s} {'Signal':12s} {'PrevCl':>8s} {'D0 Open':>8s} {'D0 High':>8s} "
          f"{'D0 Low':>8s} {'D0 Mid':>8s} {'D0 Close':>8s} {'D1 Open':>8s} | "
          f"{'Gap%':>6s} {'O->H%':>6s} {'O->L%':>6s} {'ClPos':>6s} {'Body%':>6s} | "
          f"{'D0mid':>6s} {'D0cl':>6s} {'D1opn':>6s} | {'OvntGap':>8s}")
    print("  " + "-" * 155)

    for s in sorted(all_signals, key=lambda x: x["signal_date"]):
        p = s["profile"]
        ovn = f"{p['overnight_gap']:>+6.1f}%" if p["overnight_gap"] is not None else "    N/A"
        d1o = f"${p['d1_open']:>7.2f}" if p["d1_open"] else "    N/A"

        # Entry prices for comparison
        d0mid_s = f"${p['d0_mid']:>5.2f}" if p['d0_mid'] else ""
        d0cl_s = f"${p['d0_close']:>5.2f}" if p['d0_close'] else ""

        print(f"  {s['ticker']:7s} {s['signal_date']:12s} ${p['prev_close']:>7.2f} ${p['d0_open']:>7.2f} "
              f"${p['d0_high']:>7.2f} ${p['d0_low']:>7.2f} ${p['d0_mid']:>7.2f} ${p['d0_close']:>7.2f} "
              f"{d1o} | "
              f"{p['gap_pct']:>+5.1f}% {p['open_to_high_pct']:>+5.1f}% {p['open_to_low_pct']:>+5.1f}% "
              f"{p['close_position']:>5.2f} {p['d0_body_pct']:>5.1f}% | "
              f"{d0mid_s} {d0cl_s} {d1o} | {ovn}")

    # ======================================================================
    print(f"\n\n{'='*120}")
    print("  SUMMARY: Day 0 price action characteristics (vol >= 1.2x signals)")
    print(f"{'='*120}")

    gaps = [s["profile"]["gap_pct"] for s in all_signals]
    o2h = [s["profile"]["open_to_high_pct"] for s in all_signals]
    o2l = [s["profile"]["open_to_low_pct"] for s in all_signals]
    close_pos = [s["profile"]["close_position"] for s in all_signals]
    bodies = [s["profile"]["d0_body_pct"] for s in all_signals]
    greens = sum(1 for s in all_signals if s["profile"]["d0_green"])
    mid_above = sum(1 for s in all_signals if s["profile"]["mid_above_prev"])
    open_above = sum(1 for s in all_signals if s["profile"]["open_above_prev"])
    prev_to_close = [s["profile"]["prev_close_to_d0_close"] for s in all_signals]
    open_to_close = [s["profile"]["d0_open_to_d0_close"] for s in all_signals]

    overnight = [s["profile"]["overnight_gap"] for s in all_signals if s["profile"]["overnight_gap"] is not None]
    mid_vs_d1 = [s["profile"]["d0_mid_vs_d1_open"] for s in all_signals if s["profile"]["d0_mid_vs_d1_open"] is not None]
    close_vs_d1 = [s["profile"]["d0_close_vs_d1_open"] for s in all_signals if s["profile"]["d0_close_vs_d1_open"] is not None]

    print(f"\n  DAY 0 GAP (open vs prior close):")
    print(f"    Avg gap: {mean(gaps):>+.2f}%  Med: {median(gaps):>+.2f}%  Min: {min(gaps):>+.2f}%  Max: {max(gaps):>+.2f}%")
    print(f"    Gapped up (>0): {sum(1 for g in gaps if g > 0)}/{len(gaps)} ({sum(1 for g in gaps if g > 0)/len(gaps)*100:.0f}%)")
    print(f"    Gapped up >1%: {sum(1 for g in gaps if g > 1)}/{len(gaps)}")
    print(f"    Gapped down: {sum(1 for g in gaps if g < 0)}/{len(gaps)}")

    print(f"\n  DAY 0 INTRADAY RANGE (from open):")
    print(f"    Open -> High:  avg {mean(o2h):>+.2f}%  med {median(o2h):>+.2f}%  (upside from open)")
    print(f"    Open -> Low:   avg {mean(o2l):>+.2f}%  med {median(o2l):>+.2f}%  (downside from open)")
    print(f"    Open -> Close: avg {mean(open_to_close):>+.2f}%  med {median(open_to_close):>+.2f}%")

    print(f"\n  DAY 0 CANDLE STRUCTURE:")
    print(f"    Green candle (close>open): {greens}/{len(all_signals)} ({greens/len(all_signals)*100:.0f}%)")
    print(f"    Close position in range:   avg {mean(close_pos):.2f}  med {median(close_pos):.2f}  (0=low, 1=high)")
    print(f"    Body as % of range:        avg {mean(bodies):.1f}%  med {median(bodies):.1f}%")

    print(f"\n  DAY 0 MIDPOINT vs PRIOR CLOSE:")
    print(f"    Midpoint above prior close: {mid_above}/{len(all_signals)} ({mid_above/len(all_signals)*100:.0f}%)")
    print(f"    Open above prior close:     {open_above}/{len(all_signals)} ({open_above/len(all_signals)*100:.0f}%)")
    print(f"    Prior close -> D0 close:    avg {mean(prev_to_close):>+.2f}%  med {median(prev_to_close):>+.2f}%")

    print(f"\n  OVERNIGHT GAP (D0 close -> D1 open):")
    print(f"    Avg: {mean(overnight):>+.2f}%  Med: {median(overnight):>+.2f}%  Min: {min(overnight):>+.2f}%  Max: {max(overnight):>+.2f}%")
    print(f"    D1 opens above D0 close: {sum(1 for g in overnight if g > 0)}/{len(overnight)}")

    print(f"\n  D0 MIDPOINT vs D1 OPEN (cost of waiting):")
    print(f"    Avg: {mean(mid_vs_d1):>+.2f}%  Med: {median(mid_vs_d1):>+.2f}%")
    print(f"    D1 open > D0 midpoint: {sum(1 for g in mid_vs_d1 if g > 0)}/{len(mid_vs_d1)} "
          f"({sum(1 for g in mid_vs_d1 if g > 0)/len(mid_vs_d1)*100:.0f}%)")
    print(f"    (positive = D1 open is MORE EXPENSIVE than D0 midpoint)")

    # ======================================================================
    print(f"\n\n{'='*120}")
    print("  ENTRY TIMING COMPARISON: D0 midpoint vs D0 close vs D1 open vs D1 midpoint")
    print("  All using vol >= 1.2x filter, 0.5x ATR stop, 15d max hold")
    print(f"{'='*120}")

    entry_modes = [
        ("D0 midpoint (buy trigger day ~midday)", "d0_midpoint"),
        ("D0 close (buy trigger day last 30min)", "d0_close"),
        ("D1 open (buy next morning)", "d1_open"),
        ("D1 midpoint (buy next day ~midday)", "d1_midpoint"),
    ]

    for label, mode in entry_modes:
        seen2 = set()
        results = []
        for (eps_thresh, tw), (c17, raw) in sorted(signal_cache.items()):
            for s in c17:
                key = (s["ticker"], s["trend_date"])
                if key in seen2:
                    continue
                trade = simulate_entry(db, s["ticker"], s["trend_date"],
                                       entry_mode=mode, vol_min=1.2)
                if not trade or trade.get("filtered"):
                    continue
                seen2.add(key)
                trade["ticker"] = s["ticker"]
                trade["signal_date"] = s["trend_date"]
                results.append(trade)

        show(label, results)

    # ======================================================================
    # Per-config comparison for the main modes
    # ======================================================================
    print(f"\n\n{'='*120}")
    print("  PER-CONFIG ENTRY COMPARISON: D0 midpoint vs D1 open")
    print(f"{'='*120}")

    for eps_thresh, tw in CONFIGS:
        c17, raw = signal_cache[(eps_thresh, tw)]
        if len(c17) < 3:
            continue

        print(f"\n  --- EPS>={eps_thresh:.0f}% / {tw}d ---")
        for label, mode in entry_modes:
            seen3 = set()
            results3 = []
            for s in c17:
                key = (s["ticker"], s["trend_date"])
                if key in seen3:
                    continue
                trade = simulate_entry(db, s["ticker"], s["trend_date"],
                                       entry_mode=mode, vol_min=1.2)
                if not trade or trade.get("filtered"):
                    continue
                seen3.add(key)
                trade["ticker"] = s["ticker"]
                trade["signal_date"] = s["trend_date"]
                results3.append(trade)
            show(label, results3)

    # ======================================================================
    # Trade-by-trade comparison of entry prices
    # ======================================================================
    print(f"\n\n{'='*120}")
    print("  TRADE-BY-TRADE: Entry price comparison (D0 mid vs D0 close vs D1 open)")
    print(f"{'='*120}")

    print(f"\n  {'Ticker':7s} {'Signal':12s} | {'D0 Mid':>8s} {'D0 Close':>8s} {'D1 Open':>8s} | "
          f"{'D0mid savings':>14s} {'D0cl savings':>13s} | {'D0 green':>9s} {'Vol':>5s}")
    print("  " + "-" * 100)

    for s in sorted(all_signals, key=lambda x: x["signal_date"]):
        p = s["profile"]
        if p["d1_open"] is None:
            continue

        d0_mid_save = (p["d1_open"] - p["d0_mid"]) / p["d0_mid"] * 100
        d0_cl_save = (p["d1_open"] - p["d0_close"]) / p["d0_close"] * 100

        print(f"  {s['ticker']:7s} {s['signal_date']:12s} | "
              f"${p['d0_mid']:>7.2f} ${p['d0_close']:>7.2f} ${p['d1_open']:>7.2f} | "
              f"{d0_mid_save:>+12.1f}%  {d0_cl_save:>+11.1f}%  | "
              f"{'YES' if p['d0_green'] else 'no':>8s} {p['vol_ratio']:.1f}x")

    # Summarize savings
    savings_mid = [(s["profile"]["d1_open"] - s["profile"]["d0_mid"]) / s["profile"]["d0_mid"] * 100
                   for s in all_signals if s["profile"]["d1_open"] is not None]
    savings_cl = [(s["profile"]["d1_open"] - s["profile"]["d0_close"]) / s["profile"]["d0_close"] * 100
                  for s in all_signals if s["profile"]["d1_open"] is not None]

    print(f"\n  D0 midpoint vs D1 open: avg savings {mean(savings_mid):>+.2f}%  "
          f"med {median(savings_mid):>+.2f}%  (positive = D0 mid is cheaper)")
    print(f"  D0 close vs D1 open:    avg savings {mean(savings_cl):>+.2f}%  "
          f"med {median(savings_cl):>+.2f}%  (positive = D0 close is cheaper)")
    print(f"  D0 mid cheaper than D1 open: {sum(1 for s in savings_mid if s > 0)}/{len(savings_mid)}")
    print(f"  D0 close cheaper than D1 open: {sum(1 for s in savings_cl if s > 0)}/{len(savings_cl)}")

    print(f"\n{'='*120}")
    print("  ANALYSIS COMPLETE")
    print(f"{'='*120}")


if __name__ == "__main__":
    main()
