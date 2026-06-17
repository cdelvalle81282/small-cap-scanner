"""
Compare system performance before and after applying discretionary filters.
6 of 7 losers would have been avoided by chart reading:
- NRDS: unfilled gap overhead
- U: no trend break, sideways range
- ARRY: unfilled gaps
- BILL: downtrend, no reversal confirmed
- SPCE: no trend change, still going down
- PLUG: consolidation, not a trend change
Only XERS was a legitimate loss (already trending up, closed above prior highs).
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


CONFIGS = [(25.0, 30), (30.0, 60)]


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
        return {"filtered": True}

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
    wins = [t for t in trades if t["return"] > 0]
    losses = [t for t in trades if t["return"] <= 0]
    rets = [t["return"] for t in trades]
    wr = len(wins) / len(trades) * 100
    avg = mean(rets)
    med = median(rets)
    gg = sum(t["return"] for t in wins)
    gl = sum(t["return"] for t in losses)
    pf = gg / abs(gl) if gl else float("inf")
    pf_s = f"{pf:.2f}" if pf != float("inf") else "inf"
    avg_w = mean([t["return"] for t in wins]) if wins else 0
    avg_l = mean([t["return"] for t in losses]) if losses else 0
    hold_w = mean([t["hold_days"] for t in wins]) if wins else 0
    hold_l = mean([t["hold_days"] for t in losses]) if losses else 0
    avg_hold = mean([t["hold_days"] for t in trades])

    print(f"\n  {label}")
    print(f"  {'=' * 80}")
    print(f"  Trades: {len(trades)}  |  Winners: {len(wins)}  |  Losers: {len(losses)}  |  Win Rate: {wr:.1f}%")
    print(f"  Avg Return: {avg:+.1f}%  |  Median: {med:+.1f}%  |  Profit Factor: {pf_s}")
    print(f"  Avg Winner: {avg_w:+.1f}% (hold {hold_w:.1f}d)  |  Avg Loser: {avg_l:+.1f}% (hold {hold_l:.1f}d)")
    print(f"  Avg Hold (all): {avg_hold:.1f}d")
    print(f"  Total gross gains: {gg:+.1f}%  |  Total gross losses: {gl:+.1f}%")
    return {"wr": wr, "avg": avg, "pf": pf, "avg_w": avg_w, "avg_l": avg_l, "n": len(trades),
            "wins": len(wins), "losses": len(losses), "gg": gg, "gl": gl}


def main():
    db = Database(DB_PATH)
    db.initialize()

    print("Loading signals from EPS>=25%/30d + EPS>=30%/60d...")
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
        signal_cache[(eps_thresh, tw)] = c17

    # Collect all trades
    seen = set()
    all_trades = []
    for (eps_thresh, tw), c17 in sorted(signal_cache.items()):
        for s in c17:
            key = (s["ticker"], s["trend_date"])
            if key in seen:
                continue
            trade = simulate(db, s["ticker"], s["trend_date"])
            if not trade or trade.get("filtered"):
                continue
            seen.add(key)
            trade["ticker"] = s["ticker"]
            trade["signal_date"] = s["trend_date"]
            trade["eps_pct"] = s.get("eps_change_pct", 0)
            all_trades.append(trade)

    # Discretionary avoid list
    avoid = {
        ("NRDS", "2025-05-12"),   # unfilled gap overhead
        ("U", "2025-05-13"),       # no trend break, sideways range
        ("ARRY", "2025-05-14"),    # unfilled gaps
        ("BILL", "2025-05-15"),    # downtrend, no reversal
        ("SPCE", "2025-07-25"),    # no trend change, still going down
        ("PLUG", "2026-01-22"),    # consolidation, not a trend change
    }

    print(f"\n{'=' * 100}")
    print("  ALL TRADES (before discretionary filter)")
    print(f"{'=' * 100}")
    print(f"  {'Ticker':7s} {'Signal':12s} {'EPS%':>6s} {'Vol':>5s} {'Entry$':>8s} "
          f"{'Exit$':>8s} {'Return':>8s} {'Hold':>5s} {'Exit':10s} W/L")
    print("  " + "-" * 90)
    for t in sorted(all_trades, key=lambda x: x["signal_date"]):
        wl = "W" if t["return"] > 0 else "L"
        key = (t["ticker"], t["signal_date"])
        flag = " << AVOID" if key in avoid else ""
        print(f"  {t['ticker']:7s} {t['signal_date']:12s} {t['eps_pct']:>+5.0f}% "
              f"{t['vol_ratio']:>4.1f}x ${t['entry_price']:>7.2f} ${t['exit_price']:>7.2f} "
              f"{t['return']:>+7.1f}% {t['hold_days']:>4d}d {t['exit_reason']:10s} {wl}{flag}")

    before = show_stats("BEFORE: All trades (scanner signals only)", all_trades)

    # Apply filter
    filtered = [t for t in all_trades if (t["ticker"], t["signal_date"]) not in avoid]

    print(f"\n\n{'=' * 100}")
    print("  AFTER DISCRETIONARY FILTER (6 losers removed)")
    print(f"{'=' * 100}")
    print(f"  {'Ticker':7s} {'Signal':12s} {'EPS%':>6s} {'Vol':>5s} {'Entry$':>8s} "
          f"{'Exit$':>8s} {'Return':>8s} {'Hold':>5s} {'Exit':10s} W/L")
    print("  " + "-" * 90)
    for t in sorted(filtered, key=lambda x: x["signal_date"]):
        wl = "W" if t["return"] > 0 else "L"
        print(f"  {t['ticker']:7s} {t['signal_date']:12s} {t['eps_pct']:>+5.0f}% "
              f"{t['vol_ratio']:>4.1f}x ${t['entry_price']:>7.2f} ${t['exit_price']:>7.2f} "
              f"{t['return']:>+7.1f}% {t['hold_days']:>4d}d {t['exit_reason']:10s} {wl}")

    after = show_stats("AFTER: Discretionary filter applied", filtered)

    # Comparison
    print(f"\n\n{'=' * 100}")
    print("  BEFORE vs AFTER COMPARISON")
    print(f"{'=' * 100}")
    print(f"  {'Metric':25s} {'Before':>15s} {'After':>15s} {'Change':>15s}")
    print("  " + "-" * 70)
    print(f"  {'Trades':25s} {before['n']:>15d} {after['n']:>15d} {after['n']-before['n']:>+15d}")
    print(f"  {'Winners':25s} {before['wins']:>15d} {after['wins']:>15d} {after['wins']-before['wins']:>+15d}")
    print(f"  {'Losers':25s} {before['losses']:>15d} {after['losses']:>15d} {after['losses']-before['losses']:>+15d}")
    print(f"  {'Win Rate':25s} {before['wr']:>14.1f}% {after['wr']:>14.1f}% {after['wr']-before['wr']:>+14.1f}%")
    print(f"  {'Avg Return':25s} {before['avg']:>+14.1f}% {after['avg']:>+14.1f}% {after['avg']-before['avg']:>+14.1f}%")
    pf_b = f"{before['pf']:.2f}" if before['pf'] != float('inf') else "inf"
    pf_a = f"{after['pf']:.2f}" if after['pf'] != float('inf') else "inf"
    print(f"  {'Profit Factor':25s} {pf_b:>15s} {pf_a:>15s}")
    print(f"  {'Avg Winner':25s} {before['avg_w']:>+14.1f}% {after['avg_w']:>+14.1f}%")
    print(f"  {'Avg Loser':25s} {before['avg_l']:>+14.1f}% {after['avg_l']:>+14.1f}%")
    print(f"  {'Gross Gains':25s} {before['gg']:>+14.1f}% {after['gg']:>+14.1f}%")
    print(f"  {'Gross Losses':25s} {before['gl']:>+14.1f}% {after['gl']:>+14.1f}%")
    print(f"  {'Losses Avoided':25s} {'':>15s} {before['gl']-after['gl']:>+14.1f}%")

    freq_b = before['n'] / 10
    freq_a = after['n'] / 10
    print(f"\n  Signal frequency: {freq_b:.1f}/month -> {freq_a:.1f}/month")
    print(f"  (over ~10 months of signal history)")

    print(f"\n{'=' * 100}")
    print("  DISCRETIONARY FILTER RULES (from chart review)")
    print(f"{'=' * 100}")
    print("  1. UNFILLED GAPS: If there are recent unfilled gaps near the signal,")
    print("     wait for the gap to fill before entering (NRDS, ARRY)")
    print("  2. NO TREND CHANGE: If the stock is sideways/consolidating and the")
    print("     EPS + MA cross hasn't changed the trend structure, avoid (U, BILL, PLUG)")
    print("  3. STILL TRENDING DOWN: If the stock is in a downtrend and hasn't")
    print("     broken above resistance/swing high, avoid (SPCE)")
    print("  4. CONFIRMED TREND CHANGE: Only enter when price structure confirms")
    print("     a new trend — breakout above range, close above resistance, etc.")


if __name__ == "__main__":
    main()
