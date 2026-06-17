"""
Full system backtest over maximum available data range.
Long: EPS>=25%/30d + EPS>=30%/60d, C17 filters, vol>=1.2x, D0 close, 0.5x ATR
Short: EPS>=20%/45d + EPS>=20%/60d (NEW signals only) + discretionary rejects
"""

import sys
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, median
from collections import defaultdict

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import DB_PATH, BacktestConfig, ScannerConfig
from core.backtest import Backtester
from core.database import Database
from analysis_winners_losers import enrich_deep


LONG_CONFIGS = [(25.0, 30), (30.0, 60)]
SHORT_CONFIGS = [(20.0, 45), (20.0, 60)]
ALL_CONFIGS = LONG_CONFIGS + SHORT_CONFIGS

# Known discretionary rejects (bullish signals from long configs that fail chart review)
# These become short candidates
DISCRETIONARY_REJECTS = {
    ("NRDS", "2025-05-12"),
    ("U", "2025-05-13"),
    ("ARRY", "2025-05-14"),
    ("BILL", "2025-05-15"),
    ("SPCE", "2025-07-25"),
    ("PLUG", "2026-01-22"),
}


def compute_atr(df, period=14):
    d = df.copy()
    d["prev_close"] = d["close"].shift(1)
    d["tr"] = d.apply(lambda r: max(
        r["high"] - r["low"],
        abs(r["high"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
        abs(r["low"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
    ), axis=1)
    return d["tr"].rolling(period).mean()


def get_price_data(db, ticker, signal_date, lookback=80, forward=40):
    signal_dt = date.fromisoformat(signal_date)
    fetch_start = (signal_dt - timedelta(days=lookback)).isoformat()
    fetch_end = (signal_dt + timedelta(days=forward)).isoformat()
    rows = db.get_daily_prices(ticker, fetch_start, fetch_end)
    if not rows:
        return None, None, None
    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    atr_series = compute_atr(df)
    sig_idx_list = df.index[df["date"] == signal_date].tolist()
    if not sig_idx_list:
        return None, None, None
    sig_idx = sig_idx_list[0]
    if sig_idx < 20:
        return None, None, None
    return df, atr_series, sig_idx


def check_vol_filter(df, sig_idx, min_ratio=1.2):
    d0 = df.iloc[sig_idx]
    vol_20d = df["volume"].iloc[sig_idx - 20:sig_idx].mean()
    if vol_20d <= 0:
        return None
    ratio = d0["volume"] / vol_20d
    if ratio < min_ratio:
        return None
    return ratio


def simulate_long(df, atr_series, sig_idx, vol_ratio, atr_mult=0.5, max_days=15):
    entry_price = df.iloc[sig_idx]["close"]
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
            if close < prev_row["low"]:
                ret = (close - entry_price) / entry_price * 100
                return {"return": ret, "hold_days": day_offset, "exit_reason": "stop_d1",
                        "entry_price": entry_price, "exit_price": close, "vol_ratio": vol_ratio}
        else:
            atr_val = atr_series.iloc[cur_idx - 1] if cur_idx - 1 < len(atr_series) else None
            if atr_val is None or pd.isna(atr_val):
                atr_val = (df["high"].iloc[max(0, cur_idx - 14):cur_idx].mean() -
                           df["low"].iloc[max(0, cur_idx - 14):cur_idx].mean())
            stop_level = highest_close - atr_mult * atr_val
            if close < stop_level:
                ret = (close - entry_price) / entry_price * 100
                return {"return": ret, "hold_days": day_offset, "exit_reason": "atr_stop",
                        "entry_price": entry_price, "exit_price": close, "vol_ratio": vol_ratio}
        if close > highest_close:
            highest_close = close
        prev_row = row

    last_idx = min(entry_idx + max_days, len(df) - 1)
    exit_price = df.iloc[last_idx]["close"]
    ret = (exit_price - entry_price) / entry_price * 100
    return {"return": ret, "hold_days": max_days, "exit_reason": "horizon",
            "entry_price": entry_price, "exit_price": exit_price, "vol_ratio": vol_ratio}


def simulate_short(df, atr_series, sig_idx, vol_ratio, atr_mult=0.5, max_days=15):
    entry_price = df.iloc[sig_idx]["close"]
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
            if close > prev_row["high"]:
                ret = (entry_price - close) / entry_price * 100
                return {"return": ret, "hold_days": day_offset, "exit_reason": "stop_d1",
                        "entry_price": entry_price, "exit_price": close, "vol_ratio": vol_ratio}
        else:
            atr_val = atr_series.iloc[cur_idx - 1] if cur_idx - 1 < len(atr_series) else None
            if atr_val is None or pd.isna(atr_val):
                atr_val = (df["high"].iloc[max(0, cur_idx - 14):cur_idx].mean() -
                           df["low"].iloc[max(0, cur_idx - 14):cur_idx].mean())
            stop_level = lowest_close + atr_mult * atr_val
            if close > stop_level:
                ret = (entry_price - close) / entry_price * 100
                return {"return": ret, "hold_days": day_offset, "exit_reason": "atr_stop",
                        "entry_price": entry_price, "exit_price": close, "vol_ratio": vol_ratio}
        if close < lowest_close:
            lowest_close = close
        prev_row = row

    last_idx = min(entry_idx + max_days, len(df) - 1)
    exit_price = df.iloc[last_idx]["close"]
    ret = (entry_price - exit_price) / entry_price * 100
    return {"return": ret, "hold_days": max_days, "exit_reason": "horizon",
            "entry_price": entry_price, "exit_price": exit_price, "vol_ratio": vol_ratio}


def show_stats(label, trades, indent=2):
    prefix = " " * indent
    if not trades:
        print(f"{prefix}{label}: no trades")
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

    print(f"\n{prefix}{label}")
    print(f"{prefix}{'=' * 90}")
    print(f"{prefix}Trades: {len(trades)}  |  Winners: {len(wins)}  |  Losers: {len(losses)}  |  Win Rate: {wr:.1f}%")
    print(f"{prefix}Avg Return: {avg:+.1f}%  |  Median: {med:+.1f}%  |  Profit Factor: {pf_s}")
    print(f"{prefix}Avg Winner: {avg_w:+.1f}% (hold {hold_w:.1f}d)  |  Avg Loser: {avg_l:+.1f}% (hold {hold_l:.1f}d)")
    print(f"{prefix}Avg Hold (all): {avg_hold:.1f}d")
    print(f"{prefix}Total gross gains: {gg:+.1f}%  |  Total gross losses: {gl:+.1f}%")
    return {"wr": wr, "avg": avg, "pf": pf, "n": len(trades), "wins": len(wins), "losses": len(losses),
            "gg": gg, "gl": gl, "avg_w": avg_w, "avg_l": avg_l, "hold_w": hold_w, "hold_l": hold_l,
            "avg_hold": avg_hold}


def main():
    db = Database(DB_PATH)
    db.initialize()

    # Check data range
    print("Checking data range...")
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    price_range = conn.execute("SELECT MIN(date), MAX(date), COUNT(DISTINCT date) FROM daily_prices").fetchone()
    eps_range = conn.execute("SELECT MIN(report_date), MAX(report_date), COUNT(*) FROM earnings").fetchone()
    ticker_count = conn.execute("SELECT COUNT(DISTINCT ticker) FROM daily_prices").fetchone()[0]
    conn.close()

    print(f"  Price data: {price_range[0]} to {price_range[1]} ({price_range[2]} unique dates)")
    print(f"  Earnings data: {eps_range[0]} to {eps_range[1]} ({eps_range[2]} records)")
    print(f"  Tickers with price data: {ticker_count}")

    backtest_start = "2021-01-01"
    backtest_end = "2026-03-20"

    # ===================================================================
    # Gather ALL signals across all needed configs
    # ===================================================================
    print(f"\nLoading signals from {backtest_start} to {backtest_end}...")
    all_signals = {}
    for eps_thresh, tw in ALL_CONFIGS:
        sc = ScannerConfig(
            min_price=1.0, max_price=50.0, min_market_cap=50_000_000,
            max_market_cap=10_000_000_000, ma_crossover_pairs=[(20, 50)],
            eps_change_threshold=eps_thresh, trend_window_days=tw, direction="both",
        )
        bc = BacktestConfig(
            start_date=backtest_start, end_date=backtest_end,
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
        print(f"  EPS>={eps_thresh:.0f}%/{tw}d: {len(result['signals'])} raw -> {len(c17)} C17 bullish")

    # ===================================================================
    # Build LONG trades from long configs
    # ===================================================================
    print(f"\n{'=' * 120}")
    print("  BUILDING LONG TRADES")
    print(f"{'=' * 120}")

    long_keys = set()
    long_trades = []
    long_all_keys = set()  # All long signals (before disc filter) for short exclusion

    for eps_thresh, tw in LONG_CONFIGS:
        for s in all_signals.get((eps_thresh, tw), []):
            key = (s["ticker"], s["trend_date"])
            if key in long_all_keys:
                continue
            long_all_keys.add(key)

            df, atr_series, sig_idx = get_price_data(db, s["ticker"], s["trend_date"])
            if df is None:
                continue
            vol_ratio = check_vol_filter(df, sig_idx)
            if vol_ratio is None:
                continue

            # Check if this is a discretionary reject
            is_reject = key in DISCRETIONARY_REJECTS

            if not is_reject:
                trade = simulate_long(df, atr_series, sig_idx, vol_ratio)
                if trade:
                    trade["ticker"] = s["ticker"]
                    trade["signal_date"] = s["trend_date"]
                    trade["eps_pct"] = s.get("eps_change_pct", 0)
                    trade["direction"] = "LONG"
                    trade["source"] = "bullish"
                    long_trades.append(trade)
                    long_keys.add(key)

    print(f"  Long signals (passed vol filter): {len(long_all_keys)}")
    print(f"  Discretionary rejects: {len(DISCRETIONARY_REJECTS & long_all_keys)}")
    print(f"  Long trades (after disc filter): {len(long_trades)}")

    # ===================================================================
    # Build SHORT trades from:
    # 1. Short configs (EPS>=20%/45d + 20%/60d) — NEW signals not in long system
    # 2. Discretionary rejects from long system
    # ===================================================================
    print(f"\n{'=' * 120}")
    print("  BUILDING SHORT TRADES")
    print(f"{'=' * 120}")

    short_keys = set()
    short_trades = []

    # Source 1: Short configs (new signals only)
    short_config_count = 0
    for eps_thresh, tw in SHORT_CONFIGS:
        for s in all_signals.get((eps_thresh, tw), []):
            key = (s["ticker"], s["trend_date"])
            if key in short_keys or key in long_all_keys:
                continue
            short_keys.add(key)

            df, atr_series, sig_idx = get_price_data(db, s["ticker"], s["trend_date"])
            if df is None:
                continue
            vol_ratio = check_vol_filter(df, sig_idx)
            if vol_ratio is None:
                continue

            trade = simulate_short(df, atr_series, sig_idx, vol_ratio)
            if trade:
                trade["ticker"] = s["ticker"]
                trade["signal_date"] = s["trend_date"]
                trade["eps_pct"] = s.get("eps_change_pct", 0)
                trade["direction"] = "SHORT"
                trade["source"] = "weak_config"
                short_trades.append(trade)
                short_config_count += 1

    # Source 2: Discretionary rejects
    disc_short_count = 0
    for ticker, sig_date in DISCRETIONARY_REJECTS:
        key = (ticker, sig_date)
        if key in short_keys:
            continue

        df, atr_series, sig_idx = get_price_data(db, ticker, sig_date)
        if df is None:
            continue
        vol_ratio = check_vol_filter(df, sig_idx)
        if vol_ratio is None:
            continue

        trade = simulate_short(df, atr_series, sig_idx, vol_ratio)
        if trade:
            trade["ticker"] = ticker
            trade["signal_date"] = sig_date
            trade["eps_pct"] = 0  # We don't have EPS readily here
            trade["direction"] = "SHORT"
            trade["source"] = "disc_reject"
            short_trades.append(trade)
            short_keys.add(key)
            disc_short_count += 1

    print(f"  Short from weak configs: {short_config_count}")
    print(f"  Short from disc rejects: {disc_short_count}")
    print(f"  Total short trades: {len(short_trades)}")

    # ===================================================================
    # Combine and sort chronologically
    # ===================================================================
    all_trades = sorted(long_trades + short_trades, key=lambda x: x["signal_date"])

    print(f"\n\n{'=' * 120}")
    print("  FULL SYSTEM BACKTEST — ALL TRADES CHRONOLOGICALLY")
    print(f"{'=' * 120}")

    print(f"\n  {'Dir':5s} {'Src':12s} {'Ticker':7s} {'Signal':12s} {'EPS%':>6s} {'Vol':>5s} "
          f"{'Entry$':>8s} {'Exit$':>8s} {'Return':>8s} {'Hold':>5s} {'Exit':10s} {'W/L':>4s}")
    print("  " + "-" * 110)

    cumulative_return = 0
    equity_curve = []

    for t in all_trades:
        wl = "W" if t["return"] > 0 else "L"
        src = t["source"][:12]
        cumulative_return += t["return"]
        equity_curve.append({
            "date": t["signal_date"],
            "trade_return": t["return"],
            "cumulative": cumulative_return,
            "direction": t["direction"],
        })
        print(f"  {t['direction']:5s} {src:12s} {t['ticker']:7s} {t['signal_date']:12s} "
              f"{t['eps_pct']:>+5.0f}% {t['vol_ratio']:>4.1f}x ${t['entry_price']:>7.2f} "
              f"${t['exit_price']:>7.2f} {t['return']:>+7.1f}% {t['hold_days']:>4d}d "
              f"{t['exit_reason']:10s} {wl:>3s}")

    # ===================================================================
    # Overall stats
    # ===================================================================
    show_stats("FULL SYSTEM — ALL TRADES", all_trades)
    show_stats("LONG LEG ONLY", long_trades)
    show_stats("SHORT LEG ONLY", short_trades)

    # ===================================================================
    # By year
    # ===================================================================
    print(f"\n\n{'=' * 120}")
    print("  PERFORMANCE BY YEAR")
    print(f"{'=' * 120}")

    by_year = defaultdict(list)
    for t in all_trades:
        year = t["signal_date"][:4]
        by_year[year].append(t)

    print(f"\n  {'Year':6s} {'Trades':>7s} {'Longs':>7s} {'Shorts':>7s} {'WR':>7s} "
          f"{'AvgRet':>8s} {'PF':>8s} {'GrossG':>8s} {'GrossL':>8s} {'Net':>8s}")
    print("  " + "-" * 85)

    for year in sorted(by_year.keys()):
        trades = by_year[year]
        longs = [t for t in trades if t["direction"] == "LONG"]
        shorts = [t for t in trades if t["direction"] == "SHORT"]
        wins = [t for t in trades if t["return"] > 0]
        wr = len(wins) / len(trades) * 100 if trades else 0
        avg = mean([t["return"] for t in trades]) if trades else 0
        gg = sum(t["return"] for t in wins)
        gl = sum(t["return"] for t in trades if t["return"] <= 0)
        pf = gg / abs(gl) if gl else float("inf")
        pf_s = f"{pf:.2f}" if pf != float("inf") else "inf"
        net = gg + gl
        print(f"  {year:6s} {len(trades):>7d} {len(longs):>7d} {len(shorts):>7d} "
              f"{wr:>6.1f}% {avg:>+7.1f}% {pf_s:>8s} {gg:>+7.1f}% {gl:>+7.1f}% {net:>+7.1f}%")

    # ===================================================================
    # By quarter
    # ===================================================================
    print(f"\n\n{'=' * 120}")
    print("  PERFORMANCE BY QUARTER")
    print(f"{'=' * 120}")

    by_quarter = defaultdict(list)
    for t in all_trades:
        d = t["signal_date"]
        month = int(d[5:7])
        q = (month - 1) // 3 + 1
        by_quarter[f"{d[:4]} Q{q}"].append(t)

    print(f"\n  {'Quarter':10s} {'Trades':>7s} {'L':>3s} {'S':>3s} {'WR':>7s} "
          f"{'AvgRet':>8s} {'PF':>8s} {'Net':>8s}")
    print("  " + "-" * 60)

    for q in sorted(by_quarter.keys()):
        trades = by_quarter[q]
        longs = sum(1 for t in trades if t["direction"] == "LONG")
        shorts = sum(1 for t in trades if t["direction"] == "SHORT")
        wins = [t for t in trades if t["return"] > 0]
        wr = len(wins) / len(trades) * 100 if trades else 0
        avg = mean([t["return"] for t in trades])
        gg = sum(t["return"] for t in wins)
        gl = sum(t["return"] for t in trades if t["return"] <= 0)
        pf = gg / abs(gl) if gl else float("inf")
        pf_s = f"{pf:.2f}" if pf != float("inf") else "inf"
        net = gg + gl
        print(f"  {q:10s} {len(trades):>7d} {longs:>3d} {shorts:>3d} "
              f"{wr:>6.1f}% {avg:>+7.1f}% {pf_s:>8s} {net:>+7.1f}%")

    # ===================================================================
    # Equity curve (cumulative)
    # ===================================================================
    print(f"\n\n{'=' * 120}")
    print("  EQUITY CURVE (cumulative % return, equal-weight per trade)")
    print(f"{'=' * 120}")

    if equity_curve:
        print(f"\n  {'#':>3s} {'Date':12s} {'Dir':5s} {'Trade':>8s} {'Cumulative':>12s}")
        print("  " + "-" * 50)
        for i, e in enumerate(equity_curve, 1):
            print(f"  {i:>3d} {e['date']:12s} {e['direction']:5s} "
                  f"{e['trade_return']:>+7.1f}% {e['cumulative']:>+11.1f}%")

    # ===================================================================
    # Signal distribution over time
    # ===================================================================
    print(f"\n\n{'=' * 120}")
    print("  SIGNAL DISTRIBUTION — When did signals appear?")
    print(f"{'=' * 120}")

    by_month = defaultdict(lambda: {"long": 0, "short": 0})
    for t in all_trades:
        m = t["signal_date"][:7]
        by_month[m][t["direction"].lower()] += 1

    print(f"\n  {'Month':10s} {'Long':>6s} {'Short':>6s} {'Total':>6s}")
    print("  " + "-" * 35)
    for m in sorted(by_month.keys()):
        d = by_month[m]
        print(f"  {m:10s} {d['long']:>6d} {d['short']:>6d} {d['long']+d['short']:>6d}")

    # ===================================================================
    # Summary
    # ===================================================================
    first_signal = all_trades[0]["signal_date"] if all_trades else "N/A"
    last_signal = all_trades[-1]["signal_date"] if all_trades else "N/A"
    total_months = 0
    if all_trades:
        fd = date.fromisoformat(first_signal)
        ld = date.fromisoformat(last_signal)
        total_months = (ld.year - fd.year) * 12 + (ld.month - fd.month) + 1

    print(f"\n\n{'=' * 120}")
    print("  SUMMARY")
    print(f"{'=' * 120}")
    print(f"\n  Data range searched: {backtest_start} to {backtest_end}")
    print(f"  First signal: {first_signal}")
    print(f"  Last signal: {last_signal}")
    print(f"  Active months: {total_months}")
    print(f"  Total trades: {len(all_trades)} ({len(long_trades)} long, {len(short_trades)} short)")
    if total_months > 0:
        print(f"  Frequency: {len(all_trades)/total_months:.1f} trades/month "
              f"({len(long_trades)/total_months:.1f} long, {len(short_trades)/total_months:.1f} short)")
    if all_trades:
        total_return = sum(t["return"] for t in all_trades)
        print(f"  Cumulative return (equal-weight): {total_return:+.1f}%")
        max_dd_trade = min(all_trades, key=lambda t: t["return"])
        print(f"  Worst single trade: {max_dd_trade['ticker']} {max_dd_trade['signal_date']} "
              f"{max_dd_trade['direction']} {max_dd_trade['return']:+.1f}%")
        best_trade = max(all_trades, key=lambda t: t["return"])
        print(f"  Best single trade: {best_trade['ticker']} {best_trade['signal_date']} "
              f"{best_trade['direction']} {best_trade['return']:+.1f}%")

    # Consecutive losses
    if all_trades:
        max_consec_loss = 0
        cur_consec = 0
        for t in all_trades:
            if t["return"] <= 0:
                cur_consec += 1
                max_consec_loss = max(max_consec_loss, cur_consec)
            else:
                cur_consec = 0
        print(f"  Max consecutive losses: {max_consec_loss}")

        # Max drawdown in cumulative return
        peak = 0
        max_dd = 0
        running = 0
        for t in all_trades:
            running += t["return"]
            if running > peak:
                peak = running
            dd = peak - running
            if dd > max_dd:
                max_dd = dd
        print(f"  Max drawdown (cumulative): {max_dd:.1f}pp from peak")

    print(f"\n  NOTE: Discretionary filter can only be applied to signals we've manually reviewed.")
    print(f"  For new signals outside the known reject list, chart review is required before entry.")
    print(f"  The long system without discretionary filter runs at 58.3% WR / PF 8.93.")


if __name__ == "__main__":
    main()
