"""
Apply full methodology to the other configs that had <50% win rate.
Show all trades so discretionary chart review can be applied.
Current system uses: EPS>=25%/30d + EPS>=30%/60d (combined 87.5% WR after discretionary).
Other configs with <50% WR (with vol>=1.2x, D0 close entry):
  - EPS>=10%/60d: 43.5% WR
  - EPS>=20%/45d: 35.7% WR
  - EPS>=20%/60d: 43.8% WR
  - EPS>=25%/60d: 46.2% WR
  - EPS>=30%/30d: 60.0% WR (above 50%, included for completeness)
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


# All 7 configs
ALL_CONFIGS = [
    (10.0, 60), (20.0, 45), (20.0, 60),
    (25.0, 30), (25.0, 60), (30.0, 30), (30.0, 60),
]

CURRENT_CONFIGS = [(25.0, 30), (30.0, 60)]
OTHER_CONFIGS = [c for c in ALL_CONFIGS if c not in CURRENT_CONFIGS]


def compute_atr(df, period=14):
    d = df.copy()
    d["prev_close"] = d["close"].shift(1)
    d["tr"] = d.apply(lambda r: max(
        r["high"] - r["low"],
        abs(r["high"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
        abs(r["low"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
    ), axis=1)
    return d["tr"].rolling(period).mean()


def simulate(db, ticker, signal_date, atr_mult=0.5, max_days=15):
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
    vol_20d = df["volume"].iloc[sig_idx - 20:sig_idx].mean()
    if vol_20d <= 0:
        return None
    sig_vol_ratio = d0["volume"] / vol_20d
    if sig_vol_ratio < 1.2:
        return {"filtered": True, "reason": "vol_too_low", "vol_ratio": sig_vol_ratio}

    entry_price = d0["close"]  # D0 close entry
    entry_idx = sig_idx
    if entry_price <= 0:
        return None

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
                        "entry_price": entry_price, "exit_price": close, "vol_ratio": sig_vol_ratio}
        else:
            atr_val = atr_series.iloc[cur_idx - 1] if cur_idx - 1 < len(atr_series) else None
            if atr_val is None or pd.isna(atr_val):
                atr_val = (df["high"].iloc[max(0, cur_idx - 14):cur_idx].mean() -
                           df["low"].iloc[max(0, cur_idx - 14):cur_idx].mean())
            stop_level = highest_close - atr_mult * atr_val
            if close < stop_level:
                ret = (close - entry_price) / entry_price * 100
                return {"return": ret, "hold_days": day_offset, "exit_reason": "atr_stop",
                        "entry_price": entry_price, "exit_price": close, "vol_ratio": sig_vol_ratio}
        if close > highest_close:
            highest_close = close
        prev_row = row

    last_idx = min(entry_idx + max_days, len(df) - 1)
    exit_price = df.iloc[last_idx]["close"]
    ret = (exit_price - entry_price) / entry_price * 100
    return {"return": ret, "hold_days": max_days, "exit_reason": "horizon",
            "entry_price": entry_price, "exit_price": exit_price, "vol_ratio": sig_vol_ratio}


def show_stats(label, trades):
    if not trades:
        print(f"\n  {label}: no trades")
        return
    wins = [t for t in trades if t["return"] > 0]
    losses = [t for t in trades if t["return"] <= 0]
    rets = [t["return"] for t in trades]
    wr = len(wins) / len(trades) * 100
    avg = mean(rets)
    gg = sum(t["return"] for t in wins)
    gl = sum(t["return"] for t in losses)
    pf = gg / abs(gl) if gl else float("inf")
    pf_s = f"{pf:.2f}" if pf != float("inf") else "inf"
    avg_w = mean([t["return"] for t in wins]) if wins else 0
    avg_l = mean([t["return"] for t in losses]) if losses else 0
    hold_w = mean([t["hold_days"] for t in wins]) if wins else 0
    hold_l = mean([t["hold_days"] for t in losses]) if losses else 0

    print(f"\n  {label}")
    print(f"  {'=' * 80}")
    print(f"  Trades: {len(trades)}  |  Winners: {len(wins)}  |  Losers: {len(losses)}  |  Win Rate: {wr:.1f}%")
    print(f"  Avg Return: {avg:+.1f}%  |  PF: {pf_s}  |  AvgW: {avg_w:+.1f}% ({hold_w:.1f}d)  |  AvgL: {avg_l:+.1f}% ({hold_l:.1f}d)")


def main():
    db = Database(DB_PATH)
    db.initialize()

    # First, get the CURRENT system trades (to identify overlap)
    print("Loading current system signals (EPS>=25%/30d + EPS>=30%/60d)...")
    current_signals = {}
    for eps_thresh, tw in CURRENT_CONFIGS:
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
        current_signals[(eps_thresh, tw)] = c17

    # Get current system trade keys
    current_keys = set()
    for (eps_thresh, tw), c17 in sorted(current_signals.items()):
        for s in c17:
            key = (s["ticker"], s["trend_date"])
            trade = simulate(db, s["ticker"], s["trend_date"])
            if trade and not trade.get("filtered"):
                current_keys.add(key)

    print(f"  Current system has {len(current_keys)} unique trades\n")

    # Now run each OTHER config
    print("=" * 120)
    print("  OTHER CONFIGS: Full methodology applied (vol>=1.2x, D0 close, 0.5x ATR)")
    print("  Showing all trades for discretionary chart review")
    print("=" * 120)

    for eps_thresh, tw in OTHER_CONFIGS:
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

        seen = set()
        trades = []
        vol_filtered = 0
        for s in c17:
            key = (s["ticker"], s["trend_date"])
            if key in seen:
                continue
            trade = simulate(db, s["ticker"], s["trend_date"])
            if not trade:
                continue
            if trade.get("filtered"):
                vol_filtered += 1
                continue
            seen.add(key)
            trade["ticker"] = s["ticker"]
            trade["signal_date"] = s["trend_date"]
            trade["eps_pct"] = s.get("eps_change_pct", 0)
            trade["in_current"] = key in current_keys
            trades.append(trade)

        print(f"\n\n{'=' * 120}")
        print(f"  EPS>={eps_thresh:.0f}% / {tw}d  —  {len(trades)} trades (vol filtered out: {vol_filtered})")
        print(f"{'=' * 120}")

        # Show all trades
        print(f"\n  {'Ticker':7s} {'Signal':12s} {'EPS%':>6s} {'Vol':>5s} {'Entry$':>8s} "
              f"{'Exit$':>8s} {'Return':>8s} {'Hold':>5s} {'Exit':10s} {'W/L':>4s} {'Overlap':>10s}")
        print("  " + "-" * 100)
        for t in sorted(trades, key=lambda x: x["signal_date"]):
            wl = "W" if t["return"] > 0 else "L"
            overlap = "CURRENT" if t["in_current"] else "NEW"
            print(f"  {t['ticker']:7s} {t['signal_date']:12s} {t['eps_pct']:>+5.0f}% "
                  f"{t['vol_ratio']:>4.1f}x ${t['entry_price']:>7.2f} ${t['exit_price']:>7.2f} "
                  f"{t['return']:>+7.1f}% {t['hold_days']:>4d}d {t['exit_reason']:10s} {wl:>3s}  {overlap:>10s}")

        show_stats(f"EPS>={eps_thresh:.0f}% / {tw}d — ALL TRADES", trades)

        # Show just the NEW trades (not in current system)
        new_trades = [t for t in trades if not t["in_current"]]
        if new_trades:
            show_stats(f"EPS>={eps_thresh:.0f}% / {tw}d — NEW TRADES ONLY (not in current system)", new_trades)

            print(f"\n  NEW TRADES for discretionary review:")
            print(f"  {'Ticker':7s} {'Signal':12s} {'EPS%':>6s} {'Vol':>5s} {'Entry$':>8s} "
                  f"{'Exit$':>8s} {'Return':>8s} {'Hold':>5s} {'Exit':10s} {'W/L':>4s}")
            print("  " + "-" * 90)
            for t in sorted(new_trades, key=lambda x: x["signal_date"]):
                wl = "W" if t["return"] > 0 else "L"
                print(f"  {t['ticker']:7s} {t['signal_date']:12s} {t['eps_pct']:>+5.0f}% "
                      f"{t['vol_ratio']:>4.1f}x ${t['entry_price']:>7.2f} ${t['exit_price']:>7.2f} "
                      f"{t['return']:>+7.1f}% {t['hold_days']:>4d}d {t['exit_reason']:10s} {wl:>3s}")

    # ===================================================================
    # Combined view: what if we added ALL other config trades to the current system?
    # ===================================================================
    print(f"\n\n{'=' * 120}")
    print("  COMBINED: Current system + all NEW trades from other configs (deduped)")
    print(f"{'=' * 120}")

    all_seen = set()
    all_trades = []

    # First add current system trades
    for (eps_thresh, tw), c17 in sorted(current_signals.items()):
        for s in c17:
            key = (s["ticker"], s["trend_date"])
            if key in all_seen:
                continue
            trade = simulate(db, s["ticker"], s["trend_date"])
            if not trade or trade.get("filtered"):
                continue
            all_seen.add(key)
            trade["ticker"] = s["ticker"]
            trade["signal_date"] = s["trend_date"]
            trade["eps_pct"] = s.get("eps_change_pct", 0)
            trade["source"] = "current"
            all_trades.append(trade)

    # Then add new trades from other configs
    for eps_thresh, tw in OTHER_CONFIGS:
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

        for s in c17:
            key = (s["ticker"], s["trend_date"])
            if key in all_seen:
                continue
            trade = simulate(db, s["ticker"], s["trend_date"])
            if not trade or trade.get("filtered"):
                continue
            all_seen.add(key)
            trade["ticker"] = s["ticker"]
            trade["signal_date"] = s["trend_date"]
            trade["eps_pct"] = s.get("eps_change_pct", 0)
            trade["source"] = f"EPS>={eps_thresh:.0f}%/{tw}d"
            all_trades.append(trade)

    print(f"\n  {'Ticker':7s} {'Signal':12s} {'EPS%':>6s} {'Vol':>5s} {'Entry$':>8s} "
          f"{'Exit$':>8s} {'Return':>8s} {'Hold':>5s} {'Exit':10s} {'W/L':>4s} {'Source':>20s}")
    print("  " + "-" * 110)
    for t in sorted(all_trades, key=lambda x: x["signal_date"]):
        wl = "W" if t["return"] > 0 else "L"
        print(f"  {t['ticker']:7s} {t['signal_date']:12s} {t['eps_pct']:>+5.0f}% "
              f"{t['vol_ratio']:>4.1f}x ${t['entry_price']:>7.2f} ${t['exit_price']:>7.2f} "
              f"{t['return']:>+7.1f}% {t['hold_days']:>4d}d {t['exit_reason']:10s} {wl:>3s}  {t['source']:>20s}")

    show_stats("ALL CONFIGS COMBINED (deduped)", all_trades)

    # Just the new ones
    new_only = [t for t in all_trades if t["source"] != "current"]
    if new_only:
        show_stats("NEW TRADES FROM OTHER CONFIGS ONLY", new_only)

    print(f"\n  Current system trades: {sum(1 for t in all_trades if t['source'] == 'current')}")
    print(f"  New trades from other configs: {len(new_only)}")
    print(f"  Total combined: {len(all_trades)}")

    print(f"\n{'=' * 120}")
    print("  Review the NEW trades above and apply discretionary chart filters.")
    print("  Which losers would you avoid based on price structure?")
    print(f"{'=' * 120}")


if __name__ == "__main__":
    main()
