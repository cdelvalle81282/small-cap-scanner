"""
Volume pattern analysis: can volume at the signal/entry predict winners vs losers?

Questions:
1. Do losers have the MA cross on average/below-average volume?
2. Do winners have above-average volume at the trigger?
3. Are there volume patterns in the days leading up to the signal that differ?
4. Does volume on entry day (Day 1) differ between W and L?
5. Volume trend (rising vs falling) in the 5-10 days before signal?
"""

import sys
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, median, stdev

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import DB_PATH, BacktestConfig, ScannerConfig
from core.backtest import Backtester
from core.database import Database
from analysis_winners_losers import enrich_deep
from analysis_optimization import simulate_trade_v2


CONFIGS = [
    (10.0, 60), (20.0, 45), (20.0, 60),
    (25.0, 30), (25.0, 60), (30.0, 30), (30.0, 60),
]


def compute_volume_profile(db, ticker, signal_date, max_days=15):
    """
    Compute detailed volume metrics around the signal date.

    Timeline:
      Day -20 to -1: pre-signal volume baseline
      Day 0: signal day (MA crossover detected)
      Day 1: entry day (next trading day)
      Day 2+: post-entry
    """
    signal_dt = date.fromisoformat(signal_date)
    fetch_start = (signal_dt - timedelta(days=120)).isoformat()
    fetch_end = (signal_dt + timedelta(days=int(max_days * 1.6) + 10)).isoformat()

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

    d1_idx = sig_idx + 1
    if d1_idx >= len(df):
        return None

    # Volume baselines
    vol_50d = df["volume"].iloc[max(0, sig_idx - 50):sig_idx].mean()
    vol_20d = df["volume"].iloc[sig_idx - 20:sig_idx].mean()
    vol_10d = df["volume"].iloc[sig_idx - 10:sig_idx].mean()
    vol_5d = df["volume"].iloc[sig_idx - 5:sig_idx].mean()

    # Signal day volume
    vol_signal = df.iloc[sig_idx]["volume"]

    # Entry day (Day 1) volume
    vol_d1 = df.iloc[d1_idx]["volume"]

    # Dollar volume baselines
    dv_20d = (df["volume"].iloc[sig_idx - 20:sig_idx] * df["close"].iloc[sig_idx - 20:sig_idx]).mean()

    # Signal day dollar volume
    dv_signal = vol_signal * df.iloc[sig_idx]["close"]
    dv_d1 = vol_d1 * df.iloc[d1_idx]["close"]

    # Volume ratios
    sig_vs_20d = vol_signal / vol_20d if vol_20d > 0 else 0
    sig_vs_50d = vol_signal / vol_50d if vol_50d > 0 else 0
    d1_vs_20d = vol_d1 / vol_20d if vol_20d > 0 else 0
    d1_vs_50d = vol_d1 / vol_50d if vol_50d > 0 else 0

    # Volume trend: is volume rising or falling into the signal?
    # Compare last 5d avg vs prior 5d avg (days -10 to -6 vs -5 to -1)
    vol_prior_5d = df["volume"].iloc[sig_idx - 10:sig_idx - 5].mean()
    vol_recent_5d = df["volume"].iloc[sig_idx - 5:sig_idx].mean()
    vol_trend_5d = (vol_recent_5d / vol_prior_5d - 1) * 100 if vol_prior_5d > 0 else 0

    # Volume trend last 3 days before signal
    vol_3d = df["volume"].iloc[sig_idx - 3:sig_idx].tolist()
    vol_rising_3d = all(vol_3d[i] > vol_3d[i-1] for i in range(1, len(vol_3d))) if len(vol_3d) == 3 else False
    vol_falling_3d = all(vol_3d[i] < vol_3d[i-1] for i in range(1, len(vol_3d))) if len(vol_3d) == 3 else False

    # Consecutive days of rising volume into signal
    consec_rising = 0
    for i in range(sig_idx - 1, max(0, sig_idx - 10), -1):
        if df.iloc[i]["volume"] > df.iloc[i - 1]["volume"]:
            consec_rising += 1
        else:
            break

    # Volume spike: is signal day > 2x 20d avg?
    sig_spike = sig_vs_20d > 2.0
    d1_spike = d1_vs_20d > 2.0

    # Price-volume relationship on signal day
    sig_close = df.iloc[sig_idx]["close"]
    sig_open = df.iloc[sig_idx]["open"]
    sig_green = sig_close > sig_open

    d1_close = df.iloc[d1_idx]["close"]
    d1_open = df.iloc[d1_idx]["open"]
    d1_green = d1_close > d1_open

    # Up-volume vs down-volume (last 10d before signal)
    up_vol = 0
    down_vol = 0
    for i in range(sig_idx - 10, sig_idx):
        if i < 0:
            continue
        if df.iloc[i]["close"] > df.iloc[i]["open"]:
            up_vol += df.iloc[i]["volume"]
        else:
            down_vol += df.iloc[i]["volume"]
    up_vol_ratio = up_vol / (up_vol + down_vol) if (up_vol + down_vol) > 0 else 0.5

    # OBV trend (last 10d)
    obv = 0
    obv_values = []
    for i in range(sig_idx - 10, sig_idx + 1):
        if i < 1:
            continue
        if df.iloc[i]["close"] > df.iloc[i - 1]["close"]:
            obv += df.iloc[i]["volume"]
        elif df.iloc[i]["close"] < df.iloc[i - 1]["close"]:
            obv -= df.iloc[i]["volume"]
        obv_values.append(obv)
    obv_rising = len(obv_values) >= 2 and obv_values[-1] > obv_values[0]

    # Volume on max price day in the 5d before signal
    max_price_idx = df["close"].iloc[sig_idx - 5:sig_idx].idxmax()
    vol_at_max = df.iloc[max_price_idx]["volume"]
    vol_at_max_ratio = vol_at_max / vol_20d if vol_20d > 0 else 0

    # Volume percentile: where does signal volume rank in last 50d?
    vol_50d_series = df["volume"].iloc[max(0, sig_idx - 50):sig_idx].values
    sig_vol_pctile = (np.sum(vol_50d_series < vol_signal) / len(vol_50d_series) * 100) if len(vol_50d_series) > 0 else 50

    d1_vol_pctile = (np.sum(vol_50d_series < vol_d1) / len(vol_50d_series) * 100) if len(vol_50d_series) > 0 else 50

    return {
        # Raw volumes
        "vol_50d_avg": vol_50d,
        "vol_20d_avg": vol_20d,
        "vol_10d_avg": vol_10d,
        "vol_5d_avg": vol_5d,
        "vol_signal": vol_signal,
        "vol_d1": vol_d1,
        # Dollar volumes
        "dv_20d_avg": dv_20d,
        "dv_signal": dv_signal,
        "dv_d1": dv_d1,
        # Ratios
        "sig_vs_20d": sig_vs_20d,
        "sig_vs_50d": sig_vs_50d,
        "d1_vs_20d": d1_vs_20d,
        "d1_vs_50d": d1_vs_50d,
        # Trend
        "vol_trend_5d_pct": vol_trend_5d,
        "vol_rising_3d": vol_rising_3d,
        "vol_falling_3d": vol_falling_3d,
        "consec_rising_days": consec_rising,
        # Spikes
        "sig_spike_2x": sig_spike,
        "d1_spike_2x": d1_spike,
        # Percentiles
        "sig_vol_pctile": sig_vol_pctile,
        "d1_vol_pctile": d1_vol_pctile,
        # Price-volume
        "sig_green": sig_green,
        "d1_green": d1_green,
        "up_vol_ratio_10d": up_vol_ratio,
        "obv_rising_10d": obv_rising,
        "vol_at_max_ratio": vol_at_max_ratio,
    }


def fmt_vol(v):
    if v >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    elif v >= 1_000:
        return f"{v/1_000:.0f}K"
    return f"{v:.0f}"


def fmt_dv(v):
    if v >= 1_000_000:
        return f"${v/1_000_000:.1f}M"
    elif v >= 1_000:
        return f"${v/1_000:.0f}K"
    return f"${v:.0f}"


def main():
    db = Database(DB_PATH)
    db.initialize()

    print("Loading signals across 7 config combos...")
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
        print(f"  EPS>={eps_thresh:.0f}% / {tw}d: {len(c17)} C17 signals")

    # Collect unique trades with volume profiles
    print("\nSimulating trades and computing volume profiles...")
    seen = set()
    winners = []
    losers = []

    for (eps_thresh, tw), (c17, raw) in sorted(signal_cache.items()):
        for s in c17:
            key = (s["ticker"], s["trend_date"])
            if key in seen:
                continue

            match = None
            for rs in raw:
                if rs["ticker"] == s["ticker"] and rs.get("trend_change_date") == s["trend_date"]:
                    match = rs
                    break
            if not match:
                continue

            trade = simulate_trade_v2(
                db, s["ticker"], s["trend_date"], "bullish",
                entry_type="midpoint", atr_mult=0.5, max_days=15,
            )
            if not trade:
                continue

            vol_profile = compute_volume_profile(db, s["ticker"], s["trend_date"])
            if not vol_profile:
                continue

            seen.add(key)

            rec = {
                "ticker": s["ticker"],
                "signal_date": s["trend_date"],
                "eps_pct": s.get("eps_change_pct", 0),
                "trade_return": trade["return"],
                "hold_days": trade["hold_days"],
                "exit_reason": trade["exit_reason"],
                "vol": vol_profile,
            }

            if trade["return"] > 0:
                winners.append(rec)
            else:
                losers.append(rec)

    print(f"  Winners: {len(winners)}, Losers: {len(losers)}")

    # ======================================================================
    # HEADER
    # ======================================================================
    print()
    print("=" * 180)
    print("  VOLUME PATTERN ANALYSIS: Do volume characteristics predict winners vs losers?")
    print("  C17 filter | midpoint entry | 0.5x ATR trailing stop | 15d max hold")
    print("=" * 180)

    # ======================================================================
    # 1. SIGNAL DAY VOLUME
    # ======================================================================
    print()
    print("-" * 120)
    print("  1. SIGNAL DAY (Day 0) VOLUME: How does the MA crossover day's volume compare?")
    print("-" * 120)

    for label, group in [("WINNERS", winners), ("LOSERS", losers)]:
        if not group:
            continue
        ratios = [g["vol"]["sig_vs_20d"] for g in group]
        pctiles = [g["vol"]["sig_vol_pctile"] for g in group]
        spikes = sum(1 for g in group if g["vol"]["sig_spike_2x"])
        greens = sum(1 for g in group if g["vol"]["sig_green"])

        print(f"\n  {label} (n={len(group)}):")
        print(f"    Signal vol / 20d avg:  avg={mean(ratios):.2f}x  med={median(ratios):.2f}x  "
              f"min={min(ratios):.2f}x  max={max(ratios):.2f}x")
        print(f"    Signal vol percentile: avg={mean(pctiles):.0f}th  med={median(pctiles):.0f}th  "
              f"min={min(pctiles):.0f}th  max={max(pctiles):.0f}th")
        print(f"    Volume spike (>2x):    {spikes}/{len(group)} ({spikes/len(group)*100:.0f}%)")
        print(f"    Signal day green:      {greens}/{len(group)} ({greens/len(group)*100:.0f}%)")

    # ======================================================================
    # 2. ENTRY DAY (Day 1) VOLUME
    # ======================================================================
    print()
    print("-" * 120)
    print("  2. ENTRY DAY (Day 1) VOLUME: Does entry day volume differ?")
    print("-" * 120)

    for label, group in [("WINNERS", winners), ("LOSERS", losers)]:
        if not group:
            continue
        ratios = [g["vol"]["d1_vs_20d"] for g in group]
        pctiles = [g["vol"]["d1_vol_pctile"] for g in group]
        spikes = sum(1 for g in group if g["vol"]["d1_spike_2x"])
        greens = sum(1 for g in group if g["vol"]["d1_green"])

        print(f"\n  {label} (n={len(group)}):")
        print(f"    D1 vol / 20d avg:    avg={mean(ratios):.2f}x  med={median(ratios):.2f}x  "
              f"min={min(ratios):.2f}x  max={max(ratios):.2f}x")
        print(f"    D1 vol percentile:   avg={mean(pctiles):.0f}th  med={median(pctiles):.0f}th  "
              f"min={min(pctiles):.0f}th  max={max(pctiles):.0f}th")
        print(f"    Volume spike (>2x):  {spikes}/{len(group)} ({spikes/len(group)*100:.0f}%)")
        print(f"    D1 green candle:     {greens}/{len(group)} ({greens/len(group)*100:.0f}%)")

    # ======================================================================
    # 3. VOLUME TREND INTO SIGNAL
    # ======================================================================
    print()
    print("-" * 120)
    print("  3. VOLUME TREND: Is volume rising or falling into the signal?")
    print("-" * 120)

    for label, group in [("WINNERS", winners), ("LOSERS", losers)]:
        if not group:
            continue
        trends = [g["vol"]["vol_trend_5d_pct"] for g in group]
        rising_3d = sum(1 for g in group if g["vol"]["vol_rising_3d"])
        falling_3d = sum(1 for g in group if g["vol"]["vol_falling_3d"])
        consec = [g["vol"]["consec_rising_days"] for g in group]

        print(f"\n  {label} (n={len(group)}):")
        print(f"    5d vol trend (recent vs prior): avg={mean(trends):>+.1f}%  med={median(trends):>+.1f}%")
        print(f"    3 consecutive rising vol days:  {rising_3d}/{len(group)} ({rising_3d/len(group)*100:.0f}%)")
        print(f"    3 consecutive falling vol days: {falling_3d}/{len(group)} ({falling_3d/len(group)*100:.0f}%)")
        print(f"    Avg consecutive rising vol days: {mean(consec):.1f}")

    # ======================================================================
    # 4. UP-VOLUME vs DOWN-VOLUME
    # ======================================================================
    print()
    print("-" * 120)
    print("  4. ACCUMULATION: Up-volume vs down-volume in 10d before signal")
    print("-" * 120)

    for label, group in [("WINNERS", winners), ("LOSERS", losers)]:
        if not group:
            continue
        up_ratios = [g["vol"]["up_vol_ratio_10d"] for g in group]
        obv_up = sum(1 for g in group if g["vol"]["obv_rising_10d"])

        print(f"\n  {label} (n={len(group)}):")
        print(f"    Up-vol / total vol (10d):  avg={mean(up_ratios):.2f}  med={median(up_ratios):.2f}  "
              f"(>0.50 = accumulation, <0.50 = distribution)")
        print(f"    OBV rising (10d):          {obv_up}/{len(group)} ({obv_up/len(group)*100:.0f}%)")

    # ======================================================================
    # 5. INDIVIDUAL TRADE DETAIL
    # ======================================================================
    print()
    print("=" * 180)
    print("  INDIVIDUAL TRADE VOLUME PROFILES")
    print("=" * 180)

    all_trades = sorted(winners + losers, key=lambda x: (x["signal_date"], x["ticker"]))

    hdr = (f"  {'Ticker':7s} {'Signal':12s} {'Ret':>8s} {'W/L':>4s} | "
           f"{'Sig/20d':>7s} {'Sig%ile':>7s} {'D1/20d':>7s} {'D1%ile':>7s} | "
           f"{'VolTrend':>9s} {'Rise3d':>7s} {'ConsRise':>9s} | "
           f"{'UpVolR':>7s} {'OBV+':>5s} | "
           f"{'SigSpk':>7s} {'D1Spk':>6s} {'SigGrn':>7s} {'D1Grn':>6s}")
    print(hdr)
    print("  " + "-" * 170)

    for t in all_trades:
        v = t["vol"]
        wl = "W" if t["trade_return"] > 0 else "L"
        print(f"  {t['ticker']:7s} {t['signal_date']:12s} {t['trade_return']:>+7.1f}% {wl:>3s} | "
              f"{v['sig_vs_20d']:>6.2f}x {v['sig_vol_pctile']:>5.0f}th {v['d1_vs_20d']:>6.2f}x {v['d1_vol_pctile']:>5.0f}th | "
              f"{v['vol_trend_5d_pct']:>+8.1f}% {'YES' if v['vol_rising_3d'] else 'no':>6s} {v['consec_rising_days']:>8d}d | "
              f"{v['up_vol_ratio_10d']:>6.2f} {'Y' if v['obv_rising_10d'] else 'N':>4s} | "
              f"{'YES' if v['sig_spike_2x'] else 'no':>6s} {'YES' if v['d1_spike_2x'] else 'no':>5s} "
              f"{'YES' if v['sig_green'] else 'no':>6s} {'YES' if v['d1_green'] else 'no':>5s}")

    # ======================================================================
    # 6. VOLUME BUCKET ANALYSIS
    # ======================================================================
    print()
    print("=" * 120)
    print("  VOLUME BUCKET ANALYSIS: Win rate by signal day volume level")
    print("=" * 120)

    all_for_buckets = winners + losers

    # Signal day volume buckets
    buckets = [
        ("Signal vol < 0.5x avg", lambda t: t["vol"]["sig_vs_20d"] < 0.5),
        ("Signal vol 0.5-1.0x avg", lambda t: 0.5 <= t["vol"]["sig_vs_20d"] < 1.0),
        ("Signal vol 1.0-1.5x avg", lambda t: 1.0 <= t["vol"]["sig_vs_20d"] < 1.5),
        ("Signal vol 1.5-2.0x avg", lambda t: 1.5 <= t["vol"]["sig_vs_20d"] < 2.0),
        ("Signal vol > 2.0x avg", lambda t: t["vol"]["sig_vs_20d"] >= 2.0),
    ]

    print(f"\n  {'Bucket':30s} {'N':>4s} {'W':>4s} {'L':>4s} {'WR%':>6s} {'Avg Ret':>9s} {'Avg W':>8s} {'Avg L':>8s}")
    print(f"  {'-'*75}")

    for label, filt in buckets:
        grp = [t for t in all_for_buckets if filt(t)]
        if not grp:
            print(f"  {label:30s} {'0':>4s}")
            continue
        w = [t for t in grp if t["trade_return"] > 0]
        l = [t for t in grp if t["trade_return"] <= 0]
        wr = len(w) / len(grp) * 100
        avg = mean([t["trade_return"] for t in grp])
        avg_w = mean([t["trade_return"] for t in w]) if w else 0
        avg_l = mean([t["trade_return"] for t in l]) if l else 0
        print(f"  {label:30s} {len(grp):>4d} {len(w):>4d} {len(l):>4d} {wr:>5.1f}% {avg:>+8.1f}% {avg_w:>+7.1f}% {avg_l:>+7.1f}%")

    # Entry day volume buckets
    print(f"\n  {'Bucket':30s} {'N':>4s} {'W':>4s} {'L':>4s} {'WR%':>6s} {'Avg Ret':>9s} {'Avg W':>8s} {'Avg L':>8s}")
    print(f"  {'-'*75}")

    d1_buckets = [
        ("D1 vol < 0.5x avg", lambda t: t["vol"]["d1_vs_20d"] < 0.5),
        ("D1 vol 0.5-1.0x avg", lambda t: 0.5 <= t["vol"]["d1_vs_20d"] < 1.0),
        ("D1 vol 1.0-1.5x avg", lambda t: 1.0 <= t["vol"]["d1_vs_20d"] < 1.5),
        ("D1 vol 1.5-2.0x avg", lambda t: 1.5 <= t["vol"]["d1_vs_20d"] < 2.0),
        ("D1 vol > 2.0x avg", lambda t: t["vol"]["d1_vs_20d"] >= 2.0),
    ]

    for label, filt in d1_buckets:
        grp = [t for t in all_for_buckets if filt(t)]
        if not grp:
            print(f"  {label:30s} {'0':>4s}")
            continue
        w = [t for t in grp if t["trade_return"] > 0]
        l = [t for t in grp if t["trade_return"] <= 0]
        wr = len(w) / len(grp) * 100
        avg = mean([t["trade_return"] for t in grp])
        avg_w = mean([t["trade_return"] for t in w]) if w else 0
        avg_l = mean([t["trade_return"] for t in l]) if l else 0
        print(f"  {label:30s} {len(grp):>4d} {len(w):>4d} {len(l):>4d} {wr:>5.1f}% {avg:>+8.1f}% {avg_w:>+7.1f}% {avg_l:>+7.1f}%")

    # ======================================================================
    # 7. COMBINED VOLUME FILTERS
    # ======================================================================
    print()
    print("=" * 120)
    print("  COMBINED VOLUME FILTERS: Testing actionable volume-based rules")
    print("=" * 120)

    filters = [
        ("Baseline (all trades)", lambda t: True),
        ("Signal vol > 1.0x 20d avg", lambda t: t["vol"]["sig_vs_20d"] >= 1.0),
        ("Signal vol > 1.5x 20d avg", lambda t: t["vol"]["sig_vs_20d"] >= 1.5),
        ("Signal vol > 2.0x 20d avg", lambda t: t["vol"]["sig_vs_20d"] >= 2.0),
        ("D1 vol > 1.0x 20d avg", lambda t: t["vol"]["d1_vs_20d"] >= 1.0),
        ("D1 vol > 1.5x 20d avg", lambda t: t["vol"]["d1_vs_20d"] >= 1.5),
        ("D1 vol > 2.0x 20d avg", lambda t: t["vol"]["d1_vs_20d"] >= 2.0),
        ("Signal OR D1 > 1.5x", lambda t: t["vol"]["sig_vs_20d"] >= 1.5 or t["vol"]["d1_vs_20d"] >= 1.5),
        ("Signal AND D1 > 1.0x", lambda t: t["vol"]["sig_vs_20d"] >= 1.0 and t["vol"]["d1_vs_20d"] >= 1.0),
        ("Vol trend rising (5d > +20%)", lambda t: t["vol"]["vol_trend_5d_pct"] > 20),
        ("Up-vol ratio > 0.55", lambda t: t["vol"]["up_vol_ratio_10d"] > 0.55),
        ("Up-vol ratio > 0.60", lambda t: t["vol"]["up_vol_ratio_10d"] > 0.60),
        ("OBV rising 10d", lambda t: t["vol"]["obv_rising_10d"]),
        ("Signal day green candle", lambda t: t["vol"]["sig_green"]),
        ("D1 green candle", lambda t: t["vol"]["d1_green"]),
        ("D1 green + D1 vol > 1.5x", lambda t: t["vol"]["d1_green"] and t["vol"]["d1_vs_20d"] >= 1.5),
        ("D1 green + sig vol > 1.5x", lambda t: t["vol"]["d1_green"] and t["vol"]["sig_vs_20d"] >= 1.5),
        ("Rising vol + OBV rising", lambda t: t["vol"]["vol_trend_5d_pct"] > 0 and t["vol"]["obv_rising_10d"]),
        ("Sig vol > 50th pctile", lambda t: t["vol"]["sig_vol_pctile"] >= 50),
        ("Sig vol > 75th pctile", lambda t: t["vol"]["sig_vol_pctile"] >= 75),
        ("Sig vol < 25th pctile", lambda t: t["vol"]["sig_vol_pctile"] < 25),
    ]

    print(f"\n  {'Filter':40s} {'N':>4s} {'W':>4s} {'L':>4s} {'WR%':>6s} {'Avg Ret':>9s} {'PF':>7s} {'Avg W':>8s} {'Avg L':>8s}")
    print(f"  {'-'*90}")

    for label, filt in filters:
        grp = [t for t in all_for_buckets if filt(t)]
        if not grp:
            print(f"  {label:40s}    0")
            continue
        w = [t for t in grp if t["trade_return"] > 0]
        l = [t for t in grp if t["trade_return"] <= 0]
        wr = len(w) / len(grp) * 100
        avg = mean([t["trade_return"] for t in grp])
        avg_w = mean([t["trade_return"] for t in w]) if w else 0
        avg_l = mean([t["trade_return"] for t in l]) if l else 0
        gg = sum(t["trade_return"] for t in w)
        gl = sum(t["trade_return"] for t in l)
        pf = gg / abs(gl) if gl else float("inf")
        pf_s = f"{pf:.2f}" if pf != float("inf") else "inf"
        print(f"  {label:40s} {len(grp):>4d} {len(w):>4d} {len(l):>4d} {wr:>5.1f}% {avg:>+8.1f}% {pf_s:>6s} {avg_w:>+7.1f}% {avg_l:>+7.1f}%")

    # ======================================================================
    # 8. BIG WINNERS vs BIG LOSERS volume comparison
    # ======================================================================
    print()
    print("=" * 120)
    print("  BIG WINNERS (>10%) vs BIG LOSERS (<-3%): Volume comparison")
    print("=" * 120)

    big_w = [t for t in winners if t["trade_return"] > 10]
    big_l = [t for t in losers if t["trade_return"] < -3]

    for label, group in [("BIG WINNERS (>10%)", big_w), ("BIG LOSERS (<-3%)", big_l)]:
        if not group:
            print(f"\n  {label}: no trades")
            continue
        print(f"\n  {label} (n={len(group)}):")
        sig_ratios = [g["vol"]["sig_vs_20d"] for g in group]
        d1_ratios = [g["vol"]["d1_vs_20d"] for g in group]
        up_ratios = [g["vol"]["up_vol_ratio_10d"] for g in group]
        trends = [g["vol"]["vol_trend_5d_pct"] for g in group]

        print(f"    Signal vol / 20d: avg={mean(sig_ratios):.2f}x  med={median(sig_ratios):.2f}x")
        print(f"    D1 vol / 20d:     avg={mean(d1_ratios):.2f}x  med={median(d1_ratios):.2f}x")
        print(f"    Up-vol ratio:     avg={mean(up_ratios):.2f}  med={median(up_ratios):.2f}")
        print(f"    Vol trend 5d:     avg={mean(trends):>+.1f}%  med={median(trends):>+.1f}%")

        for t in group:
            v = t["vol"]
            print(f"      {t['ticker']:7s} {t['signal_date']}  ret={t['trade_return']:>+7.1f}%  "
                  f"sig/20d={v['sig_vs_20d']:.2f}x  d1/20d={v['d1_vs_20d']:.2f}x  "
                  f"upvol={v['up_vol_ratio_10d']:.2f}  trend={v['vol_trend_5d_pct']:>+.0f}%  "
                  f"obv={'UP' if v['obv_rising_10d'] else 'DN'}")

    print()
    print("=" * 120)
    print("  ANALYSIS COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()
