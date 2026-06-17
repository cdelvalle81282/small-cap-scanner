"""
Detailed backtest: C17 + signal day volume >= 1.2x 20d avg + D1 open entry.
No green candle requirement. Shows avg wins, avg losses, avg hold time per config.
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
    (10.0, 60), (20.0, 45), (20.0, 60),
    (25.0, 30), (25.0, 60), (30.0, 30), (30.0, 60),
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


def simulate(db, ticker, signal_date, vol_min=1.2, atr_mult=0.5, max_days=15):
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

    # Volume filter
    vol_20d = df["volume"].iloc[sig_idx - 20:sig_idx].mean()
    if vol_20d <= 0:
        return None
    sig_vol_ratio = d0["volume"] / vol_20d
    if sig_vol_ratio < vol_min:
        return {"filtered": True, "reason": "vol_too_low", "vol_ratio": sig_vol_ratio}

    # Entry at D1 open
    d1_idx = sig_idx + 1
    if d1_idx >= len(df):
        return None
    d1 = df.iloc[d1_idx]
    entry_price = d1["open"]
    entry_idx = d1_idx

    if entry_price <= 0:
        return None

    # Signal day info
    sig_green = d0["close"] > d0["open"]
    sig_close = d0["close"]
    sig_open = d0["open"]

    # D1 info
    d1_green = d1["close"] > d1["open"]
    d1_vol_ratio = d1["volume"] / vol_20d if vol_20d > 0 else 0

    # Walk forward
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
                        "vol_ratio": sig_vol_ratio, "sig_green": sig_green,
                        "d1_green": d1_green, "d1_vol_ratio": d1_vol_ratio}
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
                        "vol_ratio": sig_vol_ratio, "sig_green": sig_green,
                        "d1_green": d1_green, "d1_vol_ratio": d1_vol_ratio}

        if close > highest_close:
            highest_close = close
        prev_row = row

    last_idx = min(entry_idx + max_days, len(df) - 1)
    exit_price = df.iloc[last_idx]["close"]
    ret = (exit_price - entry_price) / entry_price * 100
    return {"return": ret, "hold_days": max_days, "exit_reason": "horizon",
            "entry_price": entry_price, "exit_price": exit_price,
            "vol_ratio": sig_vol_ratio, "sig_green": sig_green,
            "d1_green": d1_green, "d1_vol_ratio": d1_vol_ratio}


def show_summary(label, results):
    if not results:
        print(f"  {label}: no trades")
        return
    rets = [r["return"] for r in results]
    wins = [r for r in results if r["return"] > 0]
    losses = [r for r in results if r["return"] <= 0]
    wr = len(wins) / len(results) * 100
    avg = mean(rets)
    avg_w = mean([r["return"] for r in wins]) if wins else 0
    avg_l = mean([r["return"] for r in losses]) if losses else 0
    med_w = median([r["return"] for r in wins]) if wins else 0
    med_l = median([r["return"] for r in losses]) if losses else 0
    gg = sum(r["return"] for r in wins)
    gl = sum(r["return"] for r in losses)
    pf = gg / abs(gl) if gl else float("inf")
    pf_s = f"{pf:.2f}" if pf != float("inf") else "inf"
    avg_hold = mean([r["hold_days"] for r in results])
    avg_hold_w = mean([r["hold_days"] for r in wins]) if wins else 0
    avg_hold_l = mean([r["hold_days"] for r in losses]) if losses else 0
    max_w = max([r["return"] for r in wins]) if wins else 0
    max_l = min([r["return"] for r in losses]) if losses else 0

    print(f"\n  {label}")
    print(f"  {'='*80}")
    print(f"  Trades: {len(results)}  |  Winners: {len(wins)}  |  Losers: {len(losses)}  |  Win Rate: {wr:.1f}%")
    print(f"  Avg Return: {avg:>+.1f}%  |  Profit Factor: {pf_s}")
    print(f"  ")
    print(f"  {'':15s} {'Avg':>8s} {'Median':>8s} {'Best':>8s} {'Worst':>8s} {'Avg Hold':>9s}")
    print(f"  {'-'*55}")
    print(f"  {'Winners':15s} {avg_w:>+7.1f}% {med_w:>+7.1f}% {max_w:>+7.1f}% {'':>8s} {avg_hold_w:>7.1f}d")
    print(f"  {'Losers':15s} {avg_l:>+7.1f}% {med_l:>+7.1f}% {'':>8s} {max_l:>+7.1f}% {avg_hold_l:>7.1f}d")
    print(f"  {'All':15s} {avg:>+7.1f}% {median(rets):>+7.1f}% {max_w:>+7.1f}% {max_l:>+7.1f}% {avg_hold:>7.1f}d")


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

    print()
    print("=" * 120)
    print("  BACKTEST: C17 + Signal Volume >= 1.2x 20d Avg + D1 Open Entry")
    print("  Stop: Day 1 = close < prev low | Day 2+ = trailing 0.5x ATR | Max hold 15d")
    print("=" * 120)

    # Run per config
    for eps_thresh, tw in CONFIGS:
        c17, raw = signal_cache[(eps_thresh, tw)]
        if len(c17) < 3:
            continue

        # Baseline (no vol filter, D1 open)
        seen_b = set()
        baseline = []
        for s in c17:
            key = (s["ticker"], s["trend_date"])
            if key in seen_b:
                continue
            trade = simulate(db, s["ticker"], s["trend_date"], vol_min=0.0)
            if not trade or trade.get("filtered"):
                continue
            seen_b.add(key)
            trade["ticker"] = s["ticker"]
            trade["signal_date"] = s["trend_date"]
            trade["eps_pct"] = s.get("eps_change_pct", 0)
            baseline.append(trade)

        # Vol >= 1.2x filter
        seen_v = set()
        filtered = []
        for s in c17:
            key = (s["ticker"], s["trend_date"])
            if key in seen_v:
                continue
            trade = simulate(db, s["ticker"], s["trend_date"], vol_min=1.2)
            if not trade or trade.get("filtered"):
                continue
            seen_v.add(key)
            trade["ticker"] = s["ticker"]
            trade["signal_date"] = s["trend_date"]
            trade["eps_pct"] = s.get("eps_change_pct", 0)
            filtered.append(trade)

        show_summary(f"EPS>={eps_thresh:.0f}% / {tw}d -- BASELINE (no vol filter, D1 open)", baseline)
        show_summary(f"EPS>={eps_thresh:.0f}% / {tw}d -- VOL >= 1.2x + D1 OPEN", filtered)

        # Trade detail for the filtered version
        if filtered:
            print(f"\n  Individual trades (vol >= 1.2x):")
            print(f"  {'Ticker':7s} {'Signal':12s} {'EPS%':>6s} {'SigVol':>7s} | "
                  f"{'Entry$':>8s} {'Exit$':>8s} {'Return':>8s} {'W/L':>4s} {'Hold':>5s} {'Exit':10s} | "
                  f"{'SigGrn':>7s} {'D1Grn':>6s} {'D1Vol':>6s}")
            print(f"  {'-'*110}")
            for t in sorted(filtered, key=lambda x: x["signal_date"]):
                wl = "W" if t["return"] > 0 else "L"
                print(f"  {t['ticker']:7s} {t['signal_date']:12s} {t['eps_pct']:>+5.0f}% {t['vol_ratio']:>6.2f}x | "
                      f"${t['entry_price']:>7.2f} ${t['exit_price']:>7.2f} {t['return']:>+7.1f}% {wl:>3s} "
                      f"{t['hold_days']:>4d}d {t['exit_reason']:10s} | "
                      f"{'YES' if t['sig_green'] else 'no':>6s} {'YES' if t['d1_green'] else 'no':>5s} "
                      f"{t['d1_vol_ratio']:>5.2f}x")

    # ======================================================================
    # Combined all-config view (deduped)
    # ======================================================================
    print(f"\n\n{'='*120}")
    print("  ALL-CONFIG COMBINED (deduped): Vol >= 1.2x + D1 Open")
    print(f"{'='*120}")

    seen_all = set()
    all_baseline = []
    all_filtered = []

    for (eps_thresh, tw), (c17, raw) in sorted(signal_cache.items()):
        for s in c17:
            key = (s["ticker"], s["trend_date"])
            if key in seen_all:
                continue

            # Baseline
            trade_b = simulate(db, s["ticker"], s["trend_date"], vol_min=0.0)
            # Filtered
            trade_f = simulate(db, s["ticker"], s["trend_date"], vol_min=1.2)

            if trade_b and not trade_b.get("filtered"):
                seen_all.add(key)
                trade_b["ticker"] = s["ticker"]
                trade_b["signal_date"] = s["trend_date"]
                trade_b["eps_pct"] = s.get("eps_change_pct", 0)
                all_baseline.append(trade_b)

                if trade_f and not trade_f.get("filtered"):
                    trade_f["ticker"] = s["ticker"]
                    trade_f["signal_date"] = s["trend_date"]
                    trade_f["eps_pct"] = s.get("eps_change_pct", 0)
                    all_filtered.append(trade_f)

    show_summary("ALL CONFIGS -- BASELINE (no vol filter, D1 open)", all_baseline)
    show_summary("ALL CONFIGS -- VOL >= 1.2x + D1 OPEN", all_filtered)

    # Exit reason breakdown
    if all_filtered:
        print(f"\n  Exit reason breakdown (vol >= 1.2x):")
        for reason in ["stop_d1", "atr_stop", "horizon"]:
            grp = [t for t in all_filtered if t["exit_reason"] == reason]
            if not grp:
                continue
            w = [t for t in grp if t["return"] > 0]
            l = [t for t in grp if t["return"] <= 0]
            wr = len(w) / len(grp) * 100
            avg_h = mean([t["hold_days"] for t in grp])
            avg_r = mean([t["return"] for t in grp])
            print(f"    {reason:10s}: n={len(grp):>2d}  WR={wr:>5.1f}%  Avg={avg_r:>+6.1f}%  Hold={avg_h:.1f}d")

    print()
    print("=" * 120)
    print("  ANALYSIS COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()
