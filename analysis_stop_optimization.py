"""
Stop optimization analysis for the CURRENT SYSTEM's 12 trades.
D0 close entry, vol >= 1.2x, configs EPS>=25%/30d and EPS>=30%/60d combined and deduped.

1. Stop optimization sweep across ATR multipliers
2. Loser post-exit behavior (recovery analysis)
3. Winner trajectory before stop (peak return, give-back, post-exit continuation)
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


def get_db():
    db = Database(DB_PATH)
    db.initialize()
    return db


def gather_signals(db):
    """Run backtests for both configs, enrich, apply volume >= 1.2x filter, dedupe."""
    all_enriched = []
    seen = set()

    for eps_thresh, trend_win in CONFIGS:
        sc = ScannerConfig(
            min_price=1.0, max_price=50.0,
            min_market_cap=50_000_000, max_market_cap=10_000_000_000,
            ma_crossover_pairs=[(20, 50)],
            eps_change_threshold=eps_thresh,
            trend_window_days=trend_win,
            direction="both",
        )
        bc = BacktestConfig(
            start_date="2022-01-01", end_date="2026-03-19",
            forward_return_days=[5, 10, 15, 30, 60],
            ma_crossover_pairs=[(20, 50)],
            eps_thresholds=[eps_thresh],
            trend_windows=[trend_win],
        )
        result = Backtester(db, sc, bc).run()
        enriched = enrich_deep(db, result["signals"])

        # Apply C17-style filters: bullish, positive EPS, dollar vol >= 500K,
        # |EPS| < 100%, days_between > 10
        filtered = [
            s for s in enriched
            if s["signal_type"] == "bullish"
            and s["eps_change_pct"] > 0
            and s.get("avg_dollar_vol", 0) >= 500_000
            and abs(s["eps_change_pct"]) < 100
            and s["days_between"] > 10
        ]

        for s in filtered:
            key = (s["ticker"], s["trend_date"])
            if key not in seen:
                seen.add(key)
                all_enriched.append(s)

    return all_enriched


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


def get_price_df(db, ticker, signal_date, extra_days_after=60):
    """Fetch price data around signal date: 30 days before to extra_days_after after."""
    signal_dt = date.fromisoformat(signal_date)
    fetch_start = (signal_dt - timedelta(days=40)).isoformat()
    fetch_end = (signal_dt + timedelta(days=int(extra_days_after * 1.6) + 10)).isoformat()

    rows = db.get_daily_prices(ticker, fetch_start, fetch_end)
    if not rows:
        return None, None

    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    atr_series = compute_atr(df)

    signal_idx_list = df.index[df["date"] == signal_date].tolist()
    if not signal_idx_list:
        return None, None

    return df, signal_idx_list[0]


def check_volume_filter(db, ticker, signal_date):
    """Check if signal day volume >= 1.2x the 20-day average volume."""
    signal_dt = date.fromisoformat(signal_date)
    fetch_start = (signal_dt - timedelta(days=40)).isoformat()
    rows = db.get_daily_prices(ticker, fetch_start, signal_date)
    if len(rows) < 21:
        return False

    df = pd.DataFrame(rows)
    df["volume"] = df["volume"].astype(float)

    signal_row = df[df["date"] == signal_date]
    if signal_row.empty:
        return False

    signal_vol = signal_row.iloc[0]["volume"]
    avg_vol_20 = df["volume"].iloc[-21:-1].mean()

    if avg_vol_20 <= 0:
        return False

    return signal_vol >= 1.2 * avg_vol_20


def simulate_trade(db, ticker, signal_date, direction, atr_mult, max_days=15):
    """Simulate a single trade with D0 close entry and hybrid stop.

    Entry: signal day (D0) close price.
    Day 1: stop = close < prev day low (prev day = signal day).
    Day 2+: trailing stop = highest_close - atr_mult * ATR(14).
    If atr_mult is None, use prev day low for all days.

    Returns dict with return, hold_days, exit_reason, exit_price, exit_idx,
    entry_price, highest_close, peak_return, and day-by-day trajectory.
    """
    df, signal_idx = get_price_df(db, ticker, signal_date, extra_days_after=max_days + 50)
    if df is None:
        return None

    atr_series = compute_atr(df)

    # D0 close entry
    entry_price = df.iloc[signal_idx]["close"]
    if entry_price <= 0:
        return None

    highest_close = entry_price
    peak_return = 0.0
    peak_day = 0
    prev_row = df.iloc[signal_idx]
    trajectory = []

    for day_num in range(1, max_days + 1):
        cur_idx = signal_idx + day_num
        if cur_idx >= len(df):
            break

        row = df.iloc[cur_idx]
        close = row["close"]
        ret_so_far = (close - entry_price) / entry_price * 100

        trajectory.append({
            "day": day_num,
            "date": row["date"],
            "close": close,
            "return_pct": ret_so_far,
        })

        # Track peak
        if close > highest_close:
            highest_close = close
            peak_return = (highest_close - entry_price) / entry_price * 100
            peak_day = day_num

        # Stop check
        if day_num == 1:
            # Day 1: prev day low (signal day low)
            prev_low = prev_row["low"]
            if direction == "bullish" and close < prev_low:
                ret = (close - entry_price) / entry_price * 100
                return {
                    "return": ret, "hold_days": day_num, "exit_reason": "stop_d1",
                    "exit_price": close, "entry_price": entry_price,
                    "exit_idx": cur_idx, "highest_close": highest_close,
                    "peak_return": peak_return, "peak_day": peak_day,
                    "trajectory": trajectory,
                }
        else:
            if atr_mult is not None:
                # ATR trailing stop
                atr_val = atr_series.iloc[cur_idx - 1] if cur_idx - 1 < len(atr_series) else atr_series.iloc[-1]
                if pd.isna(atr_val):
                    atr_val = (df["high"].iloc[max(0, cur_idx - 14):cur_idx].mean()
                               - df["low"].iloc[max(0, cur_idx - 14):cur_idx].mean())

                stop_level = highest_close - atr_mult * atr_val
                if direction == "bullish" and close < stop_level:
                    ret = (close - entry_price) / entry_price * 100
                    return {
                        "return": ret, "hold_days": day_num, "exit_reason": "atr_stop",
                        "exit_price": close, "entry_price": entry_price,
                        "exit_idx": cur_idx, "highest_close": highest_close,
                        "peak_return": peak_return, "peak_day": peak_day,
                        "trajectory": trajectory,
                    }
            else:
                # Prev day low for all days
                prev_low = df.iloc[cur_idx - 1]["low"]
                if direction == "bullish" and close < prev_low:
                    ret = (close - entry_price) / entry_price * 100
                    return {
                        "return": ret, "hold_days": day_num, "exit_reason": "stop_pdl",
                        "exit_price": close, "entry_price": entry_price,
                        "exit_idx": cur_idx, "highest_close": highest_close,
                        "peak_return": peak_return, "peak_day": peak_day,
                        "trajectory": trajectory,
                    }

        prev_row = row

    # Held to max horizon
    last_idx = min(signal_idx + max_days, len(df) - 1)
    exit_price = df.iloc[last_idx]["close"]
    ret = (exit_price - entry_price) / entry_price * 100
    return {
        "return": ret, "hold_days": max_days, "exit_reason": "horizon",
        "exit_price": exit_price, "entry_price": entry_price,
        "exit_idx": last_idx, "highest_close": highest_close,
        "peak_return": peak_return, "peak_day": peak_day,
        "trajectory": trajectory,
    }


def post_exit_analysis(db, ticker, signal_date, trade_result, post_days=30):
    """After trade exit, track what happens for post_days more trading days."""
    df, signal_idx = get_price_df(db, ticker, signal_date, extra_days_after=80)
    if df is None:
        return None

    exit_idx = trade_result["exit_idx"]
    entry_price = trade_result["entry_price"]
    exit_price = trade_result["exit_price"]

    post_data = []
    max_rally_from_exit = 0.0
    max_dd_from_exit = 0.0
    recovered_above_entry = False
    recovery_day = None

    for i in range(1, post_days + 1):
        idx = exit_idx + i
        if idx >= len(df):
            break

        row = df.iloc[idx]
        close = row["close"]

        rally_from_exit = (close - exit_price) / exit_price * 100
        dd_from_exit = (exit_price - close) / exit_price * 100 if close < exit_price else 0

        if rally_from_exit > max_rally_from_exit:
            max_rally_from_exit = rally_from_exit
        if dd_from_exit > max_dd_from_exit:
            max_dd_from_exit = dd_from_exit

        if not recovered_above_entry and close > entry_price:
            recovered_above_entry = True
            recovery_day = i

        post_data.append({
            "day": i, "date": row["date"], "close": close,
            "return_from_exit_pct": rally_from_exit,
        })

    max_rally_10d = 0.0
    max_rally_30d = max_rally_from_exit
    for p in post_data:
        if p["day"] <= 10 and p["return_from_exit_pct"] > max_rally_10d:
            max_rally_10d = p["return_from_exit_pct"]

    # Categorize
    if recovered_above_entry and recovery_day is not None and recovery_day <= 10:
        category = "V-recovery"
    elif recovered_above_entry and recovery_day is not None and recovery_day <= 30:
        category = "Slow recovery"
    else:
        category = "Kept falling"

    return {
        "max_rally_10d": max_rally_10d,
        "max_rally_30d": max_rally_30d,
        "max_dd_from_exit": max_dd_from_exit,
        "recovered_above_entry": recovered_above_entry,
        "recovery_day": recovery_day,
        "category": category,
        "post_data": post_data,
    }


def main():
    db = get_db()

    print("=" * 100)
    print("  GATHERING SIGNALS: EPS>=25%/30d + EPS>=30%/60d, C17 filters, deduped")
    print("=" * 100)

    all_signals = gather_signals(db)
    print(f"  Total C17-filtered signals (before vol filter): {len(all_signals)}")

    # Apply volume >= 1.2x filter
    vol_filtered = []
    for s in all_signals:
        if check_volume_filter(db, s["ticker"], s["trend_date"]):
            vol_filtered.append(s)

    print(f"  After volume >= 1.2x filter: {len(vol_filtered)}")

    # List the trades
    print(f"\n  {'#':>3s} {'Ticker':<8s} {'Signal Date':<12s} {'EPS%':>7s} {'Signal':>8s}")
    print("  " + "-" * 45)
    for i, s in enumerate(sorted(vol_filtered, key=lambda x: x["trend_date"]), 1):
        print(f"  {i:>3d} {s['ticker']:<8s} {s['trend_date']:<12s} "
              f"{s['eps_change_pct']:>+6.1f}% {s['signal_type']:>8s}")

    signals = sorted(vol_filtered, key=lambda x: x["trend_date"])

    # ======================================================================
    # PART 1: STOP OPTIMIZATION SWEEP
    # ======================================================================
    print("\n" + "=" * 100)
    print("  PART 1: STOP OPTIMIZATION SWEEP (D0 close entry, 15-day max hold)")
    print("=" * 100)

    atr_mults = [None, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5]
    sweep_results = {}

    print(f"\n  {'Stop Type':>16s} {'WR':>6s} {'AvgRet':>8s} {'PF':>8s} {'AvgW':>8s} "
          f"{'AvgL':>8s} {'AvgHold':>8s} {'W/L':>6s} {'D1Stop':>7s} {'Horizon':>8s}")
    print("  " + "-" * 95)

    for mult in atr_mults:
        label = "PDL only" if mult is None else f"{mult:.2f}x ATR"
        results = []

        for s in signals:
            trade = simulate_trade(db, s["ticker"], s["trend_date"], "bullish", mult, max_days=15)
            if trade:
                results.append({**trade, "ticker": s["ticker"], "signal_date": s["trend_date"]})

        if not results:
            print(f"  {label:>16s}  (no results)")
            continue

        sweep_results[mult] = results

        rets = [r["return"] for r in results]
        wins = [r for r in results if r["return"] > 0]
        losses = [r for r in results if r["return"] <= 0]
        wr = len(wins) / len(results) * 100
        avg_ret = mean(rets)
        avg_w = mean([r["return"] for r in wins]) if wins else 0
        avg_l = mean([r["return"] for r in losses]) if losses else 0
        gg = sum(r["return"] for r in wins)
        gl = sum(r["return"] for r in losses)
        pf = gg / abs(gl) if gl else float("inf")
        pf_s = f"{pf:.2f}" if pf != float("inf") else "inf"
        avg_hold = mean([r["hold_days"] for r in results])
        d1_stops = sum(1 for r in results if r["exit_reason"] == "stop_d1")
        horizon_exits = sum(1 for r in results if r["exit_reason"] == "horizon")

        print(f"  {label:>16s} {wr:>5.0f}% {avg_ret:>+7.1f}% {pf_s:>8s} {avg_w:>+7.1f}% "
              f"{avg_l:>+7.1f}% {avg_hold:>7.1f}d {len(wins):>2d}/{len(losses):<2d}  "
              f"{d1_stops:>5d}   {horizon_exits:>6d}")

    # Show individual trades for the current 0.5x ATR stop
    print(f"\n  --- Individual trades at 0.5x ATR (current system) ---")
    if 0.5 in sweep_results:
        print(f"  {'#':>3s} {'Ticker':<8s} {'Date':<12s} {'Entry$':>8s} {'Exit$':>8s} "
              f"{'Return':>8s} {'W/L':>4s} {'Hold':>5s} {'Exit Reason':<12s}")
        print("  " + "-" * 80)
        for i, r in enumerate(sweep_results[0.5], 1):
            wl = "W" if r["return"] > 0 else "L"
            print(f"  {i:>3d} {r['ticker']:<8s} {r['signal_date']:<12s} ${r['entry_price']:>7.2f} "
                  f"${r['exit_price']:>7.2f} {r['return']:>+7.1f}% {wl:>4s} {r['hold_days']:>4d}d "
                  f"{r['exit_reason']:<12s}")

    # ======================================================================
    # PART 2: LOSER POST-EXIT BEHAVIOR (0.5x ATR)
    # ======================================================================
    print("\n" + "=" * 100)
    print("  PART 2: LOSER POST-EXIT BEHAVIOR (0.5x ATR stop, D0 close entry)")
    print("  What happens after we get stopped out? Does the stock recover?")
    print("=" * 100)

    if 0.5 in sweep_results:
        losers = [r for r in sweep_results[0.5] if r["return"] <= 0]

        if losers:
            print(f"\n  Found {len(losers)} loser(s) under 0.5x ATR stop\n")

            print(f"  {'#':>3s} {'Ticker':<8s} {'Date':<12s} {'TradeRet':>9s} {'HoldD':>6s} "
                  f"{'MaxRally10d':>12s} {'MaxRally30d':>12s} {'MaxDD':>8s} "
                  f"{'Recovered?':>11s} {'RecDay':>7s} {'Category':<15s}")
            print("  " + "-" * 120)

            categories = []
            for i, loser in enumerate(losers, 1):
                pa = post_exit_analysis(db, loser["ticker"], loser["signal_date"], loser)
                if pa is None:
                    print(f"  {i:>3d} {loser['ticker']:<8s} {loser['signal_date']:<12s}  (no post-exit data)")
                    continue

                categories.append(pa["category"])
                rec_str = "Yes" if pa["recovered_above_entry"] else "No"
                rec_day_str = f"{pa['recovery_day']:>5d}d" if pa["recovery_day"] else "  N/A"

                print(f"  {i:>3d} {loser['ticker']:<8s} {loser['signal_date']:<12s} "
                      f"{loser['return']:>+8.1f}% {loser['hold_days']:>5d}d "
                      f"{pa['max_rally_10d']:>+11.1f}% {pa['max_rally_30d']:>+11.1f}% "
                      f"{pa['max_dd_from_exit']:>+7.1f}% "
                      f"{rec_str:>11s} {rec_day_str:>7s} {pa['category']:<15s}")

            # Summary
            if categories:
                print(f"\n  Summary:")
                for cat in ["V-recovery", "Slow recovery", "Kept falling"]:
                    cnt = categories.count(cat)
                    pct = cnt / len(categories) * 100
                    print(f"    {cat:<15s}: {cnt:>2d} ({pct:>5.1f}%)")
        else:
            print("\n  No losers found under 0.5x ATR stop!")
    else:
        print("\n  0.5x ATR results not available")

    # Also show for other ATR levels if there are losers
    for mult_check in [None, 0.25]:
        if mult_check in sweep_results:
            label = "PDL only" if mult_check is None else f"{mult_check}x ATR"
            losers_alt = [r for r in sweep_results[mult_check] if r["return"] <= 0]
            if losers_alt:
                print(f"\n  --- Loser post-exit for {label} stop ({len(losers_alt)} losers) ---")
                print(f"  {'#':>3s} {'Ticker':<8s} {'Date':<12s} {'TradeRet':>9s} {'HoldD':>6s} "
                      f"{'MaxRally10d':>12s} {'MaxRally30d':>12s} {'MaxDD':>8s} "
                      f"{'Recovered?':>11s} {'RecDay':>7s} {'Category':<15s}")
                print("  " + "-" * 120)

                cats = []
                for i, loser in enumerate(losers_alt, 1):
                    pa = post_exit_analysis(db, loser["ticker"], loser["signal_date"], loser)
                    if pa is None:
                        continue
                    cats.append(pa["category"])
                    rec_str = "Yes" if pa["recovered_above_entry"] else "No"
                    rec_day_str = f"{pa['recovery_day']:>5d}d" if pa["recovery_day"] else "  N/A"
                    print(f"  {i:>3d} {loser['ticker']:<8s} {loser['signal_date']:<12s} "
                          f"{loser['return']:>+8.1f}% {loser['hold_days']:>5d}d "
                          f"{pa['max_rally_10d']:>+11.1f}% {pa['max_rally_30d']:>+11.1f}% "
                          f"{pa['max_dd_from_exit']:>+7.1f}% "
                          f"{rec_str:>11s} {rec_day_str:>7s} {pa['category']:<15s}")

                if cats:
                    print(f"\n  Summary ({label}):")
                    for cat in ["V-recovery", "Slow recovery", "Kept falling"]:
                        cnt = cats.count(cat)
                        pct = cnt / len(cats) * 100
                        print(f"    {cat:<15s}: {cnt:>2d} ({pct:>5.1f}%)")

    # ======================================================================
    # PART 3: WINNER TRAJECTORY BEFORE STOP
    # ======================================================================
    print("\n" + "=" * 100)
    print("  PART 3: WINNER TRAJECTORY (0.5x ATR stop, D0 close entry)")
    print("  How much do winners run before giving back? Do they keep going after exit?")
    print("=" * 100)

    if 0.5 in sweep_results:
        winners = [r for r in sweep_results[0.5] if r["return"] > 0]

        if winners:
            print(f"\n  Found {len(winners)} winner(s) under 0.5x ATR stop\n")

            print(f"  {'#':>3s} {'Ticker':<8s} {'Date':<12s} {'Entry$':>8s} {'Exit$':>8s} "
                  f"{'FinalRet':>9s} {'PeakRet':>9s} {'GiveBack':>9s} {'PeakDay':>8s} "
                  f"{'HoldD':>6s} {'Exit':>10s} {'Post10d':>8s} {'Post30d':>8s}")
            print("  " + "-" * 130)

            give_backs = []
            post_10d_rets = []
            post_30d_rets = []

            for i, winner in enumerate(winners, 1):
                give_back = winner["peak_return"] - winner["return"]
                give_backs.append(give_back)

                # Post-exit analysis for winners
                pa = post_exit_analysis(db, winner["ticker"], winner["signal_date"], winner)
                post_10d = 0.0
                post_30d = 0.0
                if pa and pa["post_data"]:
                    for p in pa["post_data"]:
                        if p["day"] == 10:
                            post_10d = p["return_from_exit_pct"]
                        if p["day"] == 30:
                            post_30d = p["return_from_exit_pct"]
                    # If we don't have exactly day 10 or 30, use closest available
                    if post_10d == 0.0 and len(pa["post_data"]) >= 10:
                        post_10d = pa["post_data"][9]["return_from_exit_pct"]
                    if post_30d == 0.0 and len(pa["post_data"]) >= 30:
                        post_30d = pa["post_data"][29]["return_from_exit_pct"]
                    # fallback: last available
                    if post_10d == 0.0 and pa["post_data"]:
                        last_available = [p for p in pa["post_data"] if p["day"] <= 10]
                        if last_available:
                            post_10d = last_available[-1]["return_from_exit_pct"]
                    if post_30d == 0.0 and pa["post_data"]:
                        last_available = [p for p in pa["post_data"] if p["day"] <= 30]
                        if last_available:
                            post_30d = last_available[-1]["return_from_exit_pct"]

                post_10d_rets.append(post_10d)
                post_30d_rets.append(post_30d)

                print(f"  {i:>3d} {winner['ticker']:<8s} {winner['signal_date']:<12s} "
                      f"${winner['entry_price']:>7.2f} ${winner['exit_price']:>7.2f} "
                      f"{winner['return']:>+8.1f}% {winner['peak_return']:>+8.1f}% "
                      f"{give_back:>+8.1f}% {winner['peak_day']:>7d}d "
                      f"{winner['hold_days']:>5d}d {winner['exit_reason']:>10s} "
                      f"{post_10d:>+7.1f}% {post_30d:>+7.1f}%")

            # Summary
            print(f"\n  Winner Trajectory Summary:")
            print(f"    Avg final return:     {mean([w['return'] for w in winners]):>+7.1f}%")
            print(f"    Avg peak return:      {mean([w['peak_return'] for w in winners]):>+7.1f}%")
            print(f"    Avg give-back:        {mean(give_backs):>+7.1f}%")
            print(f"    Avg days to peak:     {mean([w['peak_day'] for w in winners]):>7.1f}d")
            print(f"    Avg hold days:        {mean([w['hold_days'] for w in winners]):>7.1f}d")
            if post_10d_rets:
                print(f"    Avg post-exit 10d:    {mean(post_10d_rets):>+7.1f}%")
            if post_30d_rets:
                print(f"    Avg post-exit 30d:    {mean(post_30d_rets):>+7.1f}%")

            # How many keep running?
            kept_running_10d = sum(1 for r in post_10d_rets if r > 0)
            kept_running_30d = sum(1 for r in post_30d_rets if r > 0)
            print(f"\n    Stocks that kept running after exit:")
            print(f"      Within 10d: {kept_running_10d}/{len(post_10d_rets)} "
                  f"({kept_running_10d/len(post_10d_rets)*100:.0f}%)")
            print(f"      Within 30d: {kept_running_30d}/{len(post_30d_rets)} "
                  f"({kept_running_30d/len(post_30d_rets)*100:.0f}%)")
        else:
            print("\n  No winners found under 0.5x ATR stop!")

    # ======================================================================
    # CROSS-REFERENCE: Compare all multipliers side by side for each trade
    # ======================================================================
    print("\n" + "=" * 100)
    print("  CROSS-REFERENCE: Each trade's return under different stop types")
    print("=" * 100)

    mult_labels = []
    for m in atr_mults:
        if m is None:
            mult_labels.append("PDL")
        else:
            mult_labels.append(f"{m:.2f}x")

    header = f"  {'Ticker':<8s} {'Date':<12s}"
    for lbl in mult_labels:
        header += f" {lbl:>8s}"
    print(header)
    print("  " + "-" * (20 + 9 * len(mult_labels)))

    for s in signals:
        line = f"  {s['ticker']:<8s} {s['trend_date']:<12s}"
        for mult in atr_mults:
            if mult in sweep_results:
                match = next((r for r in sweep_results[mult]
                              if r["ticker"] == s["ticker"] and r["signal_date"] == s["trend_date"]), None)
                if match:
                    line += f" {match['return']:>+7.1f}%"
                else:
                    line += f" {'N/A':>8s}"
            else:
                line += f" {'N/A':>8s}"
        print(line)

    print("\n  Done.")


if __name__ == "__main__":
    main()
