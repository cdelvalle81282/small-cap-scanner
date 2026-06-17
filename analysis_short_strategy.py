"""
Can we short the failing bullish signals from the weaker configs?
If the bullish crossover + EPS signal fires but the stock drops anyway,
shorting at D0 close with a trailing stop could be profitable.

Tests:
1. All 7 configs — short every bullish signal that passes C17 filters
2. Other configs only — short NEW signals not in the current system
3. Inverse stop logic: cover if close > lowest close + 0.5x ATR (trailing up)
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


def simulate_long(db, ticker, signal_date, atr_mult=0.5, max_days=15):
    """Standard long simulation (same as before)."""
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

    entry_price = d0["close"]
    if entry_price <= 0:
        return None
    return {"entry_price": entry_price, "vol_ratio": sig_vol_ratio, "passed": True}


def simulate_short(db, ticker, signal_date, atr_mult=0.5, max_days=15):
    """
    SHORT simulation: sell at D0 close, cover using inverse stop logic.
    Day 1: cover if close > previous day's HIGH (stock breaking out)
    Day 2+: trailing cover stop: cover if close > lowest_close + ATR * mult
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
    vol_20d = df["volume"].iloc[sig_idx - 20:sig_idx].mean()
    if vol_20d <= 0:
        return None
    sig_vol_ratio = d0["volume"] / vol_20d
    if sig_vol_ratio < 1.2:
        return {"filtered": True}

    entry_price = d0["close"]  # Short at D0 close
    entry_idx = sig_idx
    if entry_price <= 0:
        return None

    lowest_close = entry_price
    prev_row = df.iloc[entry_idx - 1] if entry_idx > 0 else df.iloc[entry_idx]

    for day_offset in range(0, max_days + 1):
        cur_idx = entry_idx + day_offset
        if cur_idx >= len(df):
            break
        row = df.iloc[cur_idx]
        close = row["close"]

        if day_offset == 0:
            if close < lowest_close:
                lowest_close = close
            prev_row = row
            continue

        if day_offset == 1:
            # Day 1: cover if close > previous day's HIGH (breakout)
            prev_high = prev_row["high"]
            if close > prev_high:
                ret = (entry_price - close) / entry_price * 100  # Short P&L
                return {"return": ret, "hold_days": day_offset, "exit_reason": "stop_d1",
                        "entry_price": entry_price, "exit_price": close, "vol_ratio": sig_vol_ratio}
        else:
            # Day 2+: trailing cover stop
            atr_val = atr_series.iloc[cur_idx - 1] if cur_idx - 1 < len(atr_series) else None
            if atr_val is None or pd.isna(atr_val):
                atr_val = (df["high"].iloc[max(0, cur_idx - 14):cur_idx].mean() -
                           df["low"].iloc[max(0, cur_idx - 14):cur_idx].mean())
            stop_level = lowest_close + atr_mult * atr_val
            if close > stop_level:
                ret = (entry_price - close) / entry_price * 100
                return {"return": ret, "hold_days": day_offset, "exit_reason": "atr_stop",
                        "entry_price": entry_price, "exit_price": close, "vol_ratio": sig_vol_ratio}

        if close < lowest_close:
            lowest_close = close
        prev_row = row

    last_idx = min(entry_idx + max_days, len(df) - 1)
    exit_price = df.iloc[last_idx]["close"]
    ret = (entry_price - exit_price) / entry_price * 100
    return {"return": ret, "hold_days": max_days, "exit_reason": "horizon",
            "entry_price": entry_price, "exit_price": exit_price, "vol_ratio": sig_vol_ratio}


def show_stats(label, trades):
    if not trades:
        print(f"\n  {label}: no trades")
        return {}
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
    print(f"  {'=' * 90}")
    print(f"  Trades: {len(trades)}  |  Winners: {len(wins)}  |  Losers: {len(losses)}  |  Win Rate: {wr:.1f}%")
    print(f"  Avg Return: {avg:+.1f}%  |  Median: {med:+.1f}%  |  Profit Factor: {pf_s}")
    print(f"  Avg Winner: {avg_w:+.1f}% (hold {hold_w:.1f}d)  |  Avg Loser: {avg_l:+.1f}% (hold {hold_l:.1f}d)")
    print(f"  Avg Hold (all): {avg_hold:.1f}d")
    print(f"  Total gross gains: {gg:+.1f}%  |  Total gross losses: {gl:+.1f}%")
    return {"wr": wr, "avg": avg, "pf": pf, "n": len(trades), "wins": len(wins), "losses": len(losses),
            "gg": gg, "gl": gl, "avg_w": avg_w, "avg_l": avg_l}


def main():
    db = Database(DB_PATH)
    db.initialize()

    # ===================================================================
    # Gather all signals across all configs
    # ===================================================================
    print("Loading signals from all 7 configs...")
    all_signals = {}
    for eps_thresh, tw in ALL_CONFIGS:
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
        all_signals[(eps_thresh, tw)] = c17

    # Get current system trade keys
    current_keys = set()
    seen_current = set()
    for (eps_thresh, tw) in CURRENT_CONFIGS:
        for s in all_signals.get((eps_thresh, tw), []):
            key = (s["ticker"], s["trend_date"])
            if key in seen_current:
                continue
            check = simulate_long(db, s["ticker"], s["trend_date"])
            if check and not check.get("filtered") and check.get("passed"):
                current_keys.add(key)
            seen_current.add(key)

    print(f"  Current system: {len(current_keys)} trades")

    # ===================================================================
    # Part 1: Short ALL bullish signals from other configs (NEW trades only)
    # ===================================================================
    print(f"\n\n{'=' * 120}")
    print("  PART 1: SHORT the NEW trades from other configs")
    print("  (These are bullish signals NOT in the current long system)")
    print(f"{'=' * 120}")

    other_seen = set()
    other_short_trades = []
    for eps_thresh, tw in OTHER_CONFIGS:
        for s in all_signals.get((eps_thresh, tw), []):
            key = (s["ticker"], s["trend_date"])
            if key in other_seen or key in current_keys:
                continue
            other_seen.add(key)
            trade = simulate_short(db, s["ticker"], s["trend_date"])
            if not trade or trade.get("filtered"):
                continue
            trade["ticker"] = s["ticker"]
            trade["signal_date"] = s["trend_date"]
            trade["eps_pct"] = s.get("eps_change_pct", 0)
            trade["config"] = f"EPS>={eps_thresh:.0f}%/{tw}d"
            other_short_trades.append(trade)

    print(f"\n  {'Ticker':7s} {'Signal':12s} {'EPS%':>6s} {'Vol':>5s} {'Short@':>8s} "
          f"{'Cover@':>8s} {'Return':>8s} {'Hold':>5s} {'Exit':10s} {'W/L':>4s} {'Config':>15s}")
    print("  " + "-" * 110)
    for t in sorted(other_short_trades, key=lambda x: x["signal_date"]):
        wl = "W" if t["return"] > 0 else "L"
        print(f"  {t['ticker']:7s} {t['signal_date']:12s} {t['eps_pct']:>+5.0f}% "
              f"{t['vol_ratio']:>4.1f}x ${t['entry_price']:>7.2f} ${t['exit_price']:>7.2f} "
              f"{t['return']:>+7.1f}% {t['hold_days']:>4d}d {t['exit_reason']:10s} {wl:>3s}  {t['config']:>15s}")

    show_stats("SHORT: New trades from other configs", other_short_trades)

    # ===================================================================
    # Part 2: Short ALL bullish signals across ALL configs
    # ===================================================================
    print(f"\n\n{'=' * 120}")
    print("  PART 2: SHORT every bullish signal across ALL 7 configs (deduped)")
    print(f"{'=' * 120}")

    all_seen = set()
    all_short_trades = []
    for eps_thresh, tw in ALL_CONFIGS:
        for s in all_signals.get((eps_thresh, tw), []):
            key = (s["ticker"], s["trend_date"])
            if key in all_seen:
                continue
            all_seen.add(key)
            trade = simulate_short(db, s["ticker"], s["trend_date"])
            if not trade or trade.get("filtered"):
                continue
            trade["ticker"] = s["ticker"]
            trade["signal_date"] = s["trend_date"]
            trade["eps_pct"] = s.get("eps_change_pct", 0)
            trade["in_current"] = key in current_keys
            all_short_trades.append(trade)

    print(f"\n  {'Ticker':7s} {'Signal':12s} {'EPS%':>6s} {'Vol':>5s} {'Short@':>8s} "
          f"{'Cover@':>8s} {'Return':>8s} {'Hold':>5s} {'Exit':10s} {'W/L':>4s} {'System':>10s}")
    print("  " + "-" * 110)
    for t in sorted(all_short_trades, key=lambda x: x["signal_date"]):
        wl = "W" if t["return"] > 0 else "L"
        sys_label = "CURRENT" if t["in_current"] else "OTHER"
        print(f"  {t['ticker']:7s} {t['signal_date']:12s} {t['eps_pct']:>+5.0f}% "
              f"{t['vol_ratio']:>4.1f}x ${t['entry_price']:>7.2f} ${t['exit_price']:>7.2f} "
              f"{t['return']:>+7.1f}% {t['hold_days']:>4d}d {t['exit_reason']:10s} {wl:>3s}  {sys_label:>10s}")

    show_stats("SHORT: All configs combined", all_short_trades)

    # Current system signals shorted
    current_shorts = [t for t in all_short_trades if t["in_current"]]
    other_shorts = [t for t in all_short_trades if not t["in_current"]]
    show_stats("SHORT: Current system signals only (the ones we're LONG on)", current_shorts)
    show_stats("SHORT: Other config signals only", other_shorts)

    # ===================================================================
    # Part 3: The losers from the current LONG system — shorted
    # ===================================================================
    print(f"\n\n{'=' * 120}")
    print("  PART 3: What if we shorted the LONG losers from current system?")
    print("  (The 7 trades we lose money on going long)")
    print(f"{'=' * 120}")

    # Run long trades first to identify losers
    long_seen = set()
    long_trades = []
    for eps_thresh, tw in CURRENT_CONFIGS:
        for s in all_signals.get((eps_thresh, tw), []):
            key = (s["ticker"], s["trend_date"])
            if key in long_seen:
                continue
            long_seen.add(key)
            # Full long sim
            signal_dt = date.fromisoformat(s["trend_date"])
            fetch_start = (signal_dt - timedelta(days=80)).isoformat()
            fetch_end = (signal_dt + timedelta(days=40)).isoformat()
            rows = db.get_daily_prices(s["ticker"], fetch_start, fetch_end)
            if not rows:
                continue
            df = pd.DataFrame(rows)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            sig_idx_list = df.index[df["date"] == s["trend_date"]].tolist()
            if not sig_idx_list:
                continue
            sig_idx = sig_idx_list[0]
            if sig_idx < 20:
                continue
            d0 = df.iloc[sig_idx]
            vol_20d = df["volume"].iloc[sig_idx - 20:sig_idx].mean()
            if vol_20d <= 0:
                continue
            sig_vol_ratio = d0["volume"] / vol_20d
            if sig_vol_ratio < 1.2:
                continue

            # Get the long result
            from analysis_discretionary_filter import simulate as sim_long
            long_result = sim_long(db, s["ticker"], s["trend_date"])
            if not long_result or long_result.get("filtered"):
                continue

            long_result["ticker"] = s["ticker"]
            long_result["signal_date"] = s["trend_date"]
            long_result["eps_pct"] = s.get("eps_change_pct", 0)
            long_trades.append(long_result)

    long_losers = [t for t in long_trades if t["return"] <= 0]
    print(f"\n  Current system long losers: {len(long_losers)}")

    if long_losers:
        print(f"\n  {'Ticker':7s} {'Signal':12s} {'Long Ret':>9s} {'Short Ret':>10s} {'Short Hold':>11s} {'Short Exit':>11s}")
        print("  " + "-" * 70)
        for lt in sorted(long_losers, key=lambda x: x["signal_date"]):
            st = simulate_short(db, lt["ticker"], lt["signal_date"])
            if st and not st.get("filtered"):
                print(f"  {lt['ticker']:7s} {lt['signal_date']:12s} {lt['return']:>+8.1f}% "
                      f"{st['return']:>+9.1f}% {st['hold_days']:>10d}d {st['exit_reason']:>11s}")
            else:
                print(f"  {lt['ticker']:7s} {lt['signal_date']:12s} {lt['return']:>+8.1f}%  {'N/A':>9s}")

    # ===================================================================
    # Part 4: Combined strategy — long current system + short other configs
    # ===================================================================
    print(f"\n\n{'=' * 120}")
    print("  PART 4: COMBINED STRATEGY")
    print("  Long: current system (EPS>=25%/30d + EPS>=30%/60d)")
    print("  Short: NEW signals from other configs only")
    print(f"{'=' * 120}")

    combined = []
    # Add long trades
    for lt in long_trades:
        combined.append({
            "ticker": lt["ticker"],
            "signal_date": lt["signal_date"],
            "direction": "LONG",
            "return": lt["return"],
            "hold_days": lt["hold_days"],
            "entry_price": lt["entry_price"],
            "exit_price": lt["exit_price"],
            "vol_ratio": lt["vol_ratio"],
        })
    # Add short trades (other configs only)
    for st in other_short_trades:
        combined.append({
            "ticker": st["ticker"],
            "signal_date": st["signal_date"],
            "direction": "SHORT",
            "return": st["return"],
            "hold_days": st["hold_days"],
            "entry_price": st["entry_price"],
            "exit_price": st["exit_price"],
            "vol_ratio": st["vol_ratio"],
        })

    print(f"\n  {'Dir':5s} {'Ticker':7s} {'Signal':12s} {'Vol':>5s} {'Entry$':>8s} "
          f"{'Exit$':>8s} {'Return':>8s} {'Hold':>5s} {'W/L':>4s}")
    print("  " + "-" * 75)
    for t in sorted(combined, key=lambda x: x["signal_date"]):
        wl = "W" if t["return"] > 0 else "L"
        print(f"  {t['direction']:5s} {t['ticker']:7s} {t['signal_date']:12s} "
              f"{t['vol_ratio']:>4.1f}x ${t['entry_price']:>7.2f} ${t['exit_price']:>7.2f} "
              f"{t['return']:>+7.1f}% {t['hold_days']:>4d}d {wl:>3s}")

    show_stats("COMBINED: Long current + Short others", combined)

    longs_only = [t for t in combined if t["direction"] == "LONG"]
    shorts_only = [t for t in combined if t["direction"] == "SHORT"]
    show_stats("  Long leg only", longs_only)
    show_stats("  Short leg only", shorts_only)


if __name__ == "__main__":
    main()
