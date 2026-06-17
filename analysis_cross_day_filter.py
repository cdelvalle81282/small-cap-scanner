"""
Test: signal day (cross date) must have close>open AND volume 1.2-2.0x the 20d avg.
Entry: next morning (Day 1 open).
Stop: Day 1 = close < prev day low; Day 2+ = trailing at highest_close - 0.5x ATR(14).
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


def simulate_filtered(
    db, ticker, signal_date,
    require_sig_green=False,
    vol_min_ratio=None,       # e.g. 1.2
    vol_max_ratio=None,       # e.g. 2.0
    entry_mode="d1_open",     # "d1_open", "d1_midpoint", "d1_close", "d2_open"
    atr_mult=0.5,
    max_days=15,
):
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
    d1_idx = sig_idx + 1
    if d1_idx >= len(df):
        return None
    d1 = df.iloc[d1_idx]

    # -- Signal day filters --
    if require_sig_green and d0["close"] <= d0["open"]:
        return {"filtered": True, "reason": "sig_not_green"}

    # Volume ratio check on signal day
    vol_20d = df["volume"].iloc[sig_idx - 20:sig_idx].mean()
    if vol_20d <= 0:
        return None
    sig_vol_ratio = d0["volume"] / vol_20d

    if vol_min_ratio is not None and sig_vol_ratio < vol_min_ratio:
        return {"filtered": True, "reason": "vol_too_low", "vol_ratio": sig_vol_ratio}
    if vol_max_ratio is not None and sig_vol_ratio > vol_max_ratio:
        return {"filtered": True, "reason": "vol_too_high", "vol_ratio": sig_vol_ratio}

    # -- Entry price --
    if entry_mode == "d1_open":
        entry_price = d1["open"]
        entry_idx = d1_idx
    elif entry_mode == "d1_midpoint":
        entry_price = (d1["high"] + d1["low"]) / 2
        entry_idx = d1_idx
    elif entry_mode == "d1_close":
        entry_price = d1["close"]
        entry_idx = d1_idx
    elif entry_mode == "d2_open":
        d2_idx = sig_idx + 2
        if d2_idx >= len(df):
            return None
        entry_price = df.iloc[d2_idx]["open"]
        entry_idx = d2_idx
    else:
        return None

    if entry_price <= 0:
        return None

    # -- Walk forward --
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
                        "vol_ratio": sig_vol_ratio}
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
                        "vol_ratio": sig_vol_ratio}

        if close > highest_close:
            highest_close = close
        prev_row = row

    last_idx = min(entry_idx + max_days, len(df) - 1)
    exit_price = df.iloc[last_idx]["close"]
    ret = (exit_price - entry_price) / entry_price * 100
    return {"return": ret, "hold_days": max_days, "exit_reason": "horizon",
            "entry_price": entry_price, "exit_price": exit_price,
            "vol_ratio": sig_vol_ratio}


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
        for r in sorted(results, key=lambda x: x.get("signal_date", "")):
            wl = "W" if r["return"] > 0 else "L"
            vol_s = f"vol={r.get('vol_ratio', 0):.2f}x" if "vol_ratio" in r else ""
            print(f"      {r.get('ticker',''):7s} {r.get('signal_date',''):12s}  "
                  f"entry=${r['entry_price']:>7.2f}  exit=${r['exit_price']:>7.2f}  "
                  f"{r['return']:>+7.1f}%  {wl}  hold={r['hold_days']:>2d}d  "
                  f"exit={r['exit_reason']}  {vol_s}")


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

    # ======================================================================
    print()
    print("=" * 160)
    print("  CROSS-DAY FILTER TEST")
    print("  Rule: signal day close>open + volume 1.2-2.0x 20d avg -> enter next morning (D1 open)")
    print("=" * 160)

    # Test multiple scenarios across all configs
    scenarios = [
        # label, require_sig_green, vol_min, vol_max, entry_mode
        ("A) Baseline: no filter, D1 midpoint (current)",
         False, None, None, "d1_midpoint"),
        ("B) Baseline: no filter, D1 open",
         False, None, None, "d1_open"),
        ("C) Sig green only, D1 open",
         True, None, None, "d1_open"),
        ("D) Sig green + vol >= 1.0x, D1 open",
         True, 1.0, None, "d1_open"),
        ("E) Sig green + vol >= 1.2x, D1 open",
         True, 1.2, None, "d1_open"),
        ("F) Sig green + vol 1.0-2.0x, D1 open",
         True, 1.0, 2.0, "d1_open"),
        ("G) Sig green + vol 1.2-2.0x, D1 open  <-- USER'S RULE",
         True, 1.2, 2.0, "d1_open"),
        ("H) Sig green + vol 1.2-2.0x, D1 midpoint",
         True, 1.2, 2.0, "d1_midpoint"),
        ("I) Sig green + vol 1.2-2.0x, D1 close",
         True, 1.2, 2.0, "d1_close"),
        ("J) Sig green + vol >= 1.2x (no cap), D1 open",
         True, 1.2, None, "d1_open"),
        ("K) Sig green + vol >= 1.5x, D1 open",
         True, 1.5, None, "d1_open"),
        ("L) Sig green + vol 0.8-1.5x, D1 open",
         True, 0.8, 1.5, "d1_open"),
        ("M) Vol >= 1.2x only (no green req), D1 open",
         False, 1.2, None, "d1_open"),
        ("N) Vol 1.2-2.0x only (no green req), D1 open",
         False, 1.2, 2.0, "d1_open"),
    ]

    for eps_thresh, tw in CONFIGS:
        c17, raw = signal_cache[(eps_thresh, tw)]
        if len(c17) < 5:
            continue

        print(f"\n{'='*160}")
        print(f"  EPS>={eps_thresh:.0f}% / {tw}d window -- {len(c17)} C17 signals")
        print(f"{'='*160}")

        for label, req_green, vol_min, vol_max, entry_mode in scenarios:
            seen = set()
            results = []
            filtered_reasons = {"sig_not_green": 0, "vol_too_low": 0, "vol_too_high": 0}

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

                trade = simulate_filtered(
                    db, s["ticker"], s["trend_date"],
                    require_sig_green=req_green,
                    vol_min_ratio=vol_min,
                    vol_max_ratio=vol_max,
                    entry_mode=entry_mode,
                    atr_mult=0.5, max_days=15,
                )
                if not trade:
                    continue

                seen.add(key)

                if trade.get("filtered"):
                    filtered_reasons[trade["reason"]] = filtered_reasons.get(trade["reason"], 0) + 1
                    continue

                trade["ticker"] = s["ticker"]
                trade["signal_date"] = s["trend_date"]
                trade["eps_pct"] = s.get("eps_change_pct", 0)
                results.append(trade)

            show(label, results)

    # ======================================================================
    # Detailed trades for user's specific rule across all configs
    # ======================================================================
    print(f"\n\n{'='*160}")
    print("  DETAILED TRADES: Sig green + vol 1.2-2.0x + D1 open entry (user's rule)")
    print(f"{'='*160}")

    seen = set()
    all_user_trades = []

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

            trade = simulate_filtered(
                db, s["ticker"], s["trend_date"],
                require_sig_green=True,
                vol_min_ratio=1.2,
                vol_max_ratio=2.0,
                entry_mode="d1_open",
                atr_mult=0.5, max_days=15,
            )
            if not trade:
                continue

            seen.add(key)

            if trade.get("filtered"):
                continue

            trade["ticker"] = s["ticker"]
            trade["signal_date"] = s["trend_date"]
            trade["eps_pct"] = s.get("eps_change_pct", 0)
            trade["config"] = f"EPS>={eps_thresh:.0f}%/{tw}d"
            all_user_trades.append(trade)

    all_user_trades.sort(key=lambda x: (x["signal_date"], x["ticker"]))

    print(f"\n  {'Ticker':7s} {'Signal':12s} {'EPS%':>6s} {'VolR':>6s} | "
          f"{'Entry$':>8s} {'Exit$':>8s} {'Return':>8s} {'W/L':>4s} {'Hold':>5s} {'Exit':10s}")
    print("  " + "-" * 90)

    for t in all_user_trades:
        wl = "W" if t["return"] > 0 else "L"
        print(f"  {t['ticker']:7s} {t['signal_date']:12s} {t['eps_pct']:>+5.0f}% {t.get('vol_ratio',0):>5.2f}x | "
              f"${t['entry_price']:>7.2f} ${t['exit_price']:>7.2f} {t['return']:>+7.1f}% {wl:>3s} "
              f"{t['hold_days']:>4d}d {t['exit_reason']:10s}")

    if all_user_trades:
        print()
        show("ALL CONFIGS COMBINED (user's rule)", all_user_trades)

    # ======================================================================
    # What gets filtered out? Show them too
    # ======================================================================
    print(f"\n\n{'='*160}")
    print("  WHAT GETS FILTERED OUT by sig green + vol 1.2-2.0x?")
    print(f"{'='*160}")

    seen2 = set()
    filtered_out = []
    passed = set((t["ticker"], t["signal_date"]) for t in all_user_trades)

    for (eps_thresh, tw), (c17, raw) in sorted(signal_cache.items()):
        for s in c17:
            key = (s["ticker"], s["trend_date"])
            if key in seen2:
                continue

            match = None
            for rs in raw:
                if rs["ticker"] == s["ticker"] and rs.get("trend_change_date") == s["trend_date"]:
                    match = rs
                    break
            if not match:
                continue

            # Run unfiltered baseline for comparison
            baseline = simulate_filtered(
                db, s["ticker"], s["trend_date"],
                require_sig_green=False,
                vol_min_ratio=None, vol_max_ratio=None,
                entry_mode="d1_open",
                atr_mult=0.5, max_days=15,
            )
            if not baseline or baseline.get("filtered"):
                continue

            seen2.add(key)

            if key not in passed:
                # This trade was filtered by the user's rule
                # Get filter reason
                filtered_trade = simulate_filtered(
                    db, s["ticker"], s["trend_date"],
                    require_sig_green=True,
                    vol_min_ratio=1.2, vol_max_ratio=2.0,
                    entry_mode="d1_open",
                    atr_mult=0.5, max_days=15,
                )
                reason = filtered_trade.get("reason", "unknown") if filtered_trade and filtered_trade.get("filtered") else "unknown"
                vol_r = filtered_trade.get("vol_ratio", 0) if filtered_trade else 0

                baseline["ticker"] = s["ticker"]
                baseline["signal_date"] = s["trend_date"]
                baseline["eps_pct"] = s.get("eps_change_pct", 0)
                baseline["filter_reason"] = reason
                baseline["sig_vol_ratio"] = vol_r
                filtered_out.append(baseline)

    filtered_out.sort(key=lambda x: (x["signal_date"], x["ticker"]))

    # Categorize filtered
    filtered_winners = [t for t in filtered_out if t["return"] > 0]
    filtered_losers = [t for t in filtered_out if t["return"] <= 0]

    print(f"\n  Filtered out: {len(filtered_out)} trades ({len(filtered_winners)} winners, {len(filtered_losers)} losers)")
    print(f"\n  {'Ticker':7s} {'Signal':12s} {'Reason':14s} {'VolR':>6s} | "
          f"{'Return':>8s} {'W/L':>4s} {'Hold':>5s} | Would have been...")
    print("  " + "-" * 80)

    for t in filtered_out:
        wl = "W" if t["return"] > 0 else "L"
        print(f"  {t['ticker']:7s} {t['signal_date']:12s} {t['filter_reason']:14s} {t.get('sig_vol_ratio',0):>5.2f}x | "
              f"{t['return']:>+7.1f}% {wl:>3s} {t['hold_days']:>4d}d | "
              f"{'MISSED WINNER' if t['return'] > 0 else 'avoided loser'}")

    if filtered_winners:
        print(f"\n  Missed winners avg return: {mean([t['return'] for t in filtered_winners]):>+.1f}%")
    if filtered_losers:
        print(f"  Avoided losers avg return: {mean([t['return'] for t in filtered_losers]):>+.1f}%")

    # ======================================================================
    # Also test with different ATR multipliers
    # ======================================================================
    print(f"\n\n{'='*160}")
    print("  ATR STOP SENSITIVITY: User's rule (sig green + vol 1.2-2.0x + D1 open) with different ATR stops")
    print(f"{'='*160}")

    for atr_m in [0.25, 0.5, 0.75, 1.0]:
        seen3 = set()
        results3 = []
        for (eps_thresh, tw), (c17, raw) in sorted(signal_cache.items()):
            for s in c17:
                key = (s["ticker"], s["trend_date"])
                if key in seen3:
                    continue
                match = None
                for rs in raw:
                    if rs["ticker"] == s["ticker"] and rs.get("trend_change_date") == s["trend_date"]:
                        match = rs
                        break
                if not match:
                    continue
                trade = simulate_filtered(
                    db, s["ticker"], s["trend_date"],
                    require_sig_green=True, vol_min_ratio=1.2, vol_max_ratio=2.0,
                    entry_mode="d1_open", atr_mult=atr_m, max_days=15,
                )
                if not trade or trade.get("filtered"):
                    continue
                seen3.add(key)
                trade["ticker"] = s["ticker"]
                trade["signal_date"] = s["trend_date"]
                results3.append(trade)

        show(f"ATR {atr_m:.2f}x + sig green + vol 1.2-2.0x + D1 open", results3)

    print()
    print("=" * 160)
    print("  ANALYSIS COMPLETE")
    print("=" * 160)


if __name__ == "__main__":
    main()
