"""
Post-stop analysis for WINNERS: after the trailing stop fires on a winning trade,
does the stock keep moving higher? Is the tight stop leaving money on the table?

For each C17 winner (bullish + EPS>0 + avg_dollar_vol>=$500K + |EPS|<100% + days>10),
tracks the stock behavior over 5/10/15/20/30 trading days after the exit day.
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
from analysis_optimization import simulate_trade_v2


HORIZONS = [5, 10, 15, 20, 30]

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


def get_post_exit_data(db, ticker, exit_date_str, entry_price, exit_price, max_horizon=30):
    """Fetch price data after exit and compute post-exit metrics."""
    exit_dt = date.fromisoformat(exit_date_str)
    fetch_start = (exit_dt - timedelta(days=5)).isoformat()
    fetch_end = (exit_dt + timedelta(days=int(max_horizon * 1.6) + 10)).isoformat()

    rows = db.get_daily_prices(ticker, fetch_start, fetch_end)
    if not rows:
        return None

    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    exit_idx_list = df.index[df["date"] == exit_date_str].tolist()
    if not exit_idx_list:
        return None
    exit_idx = exit_idx_list[0]

    post_start = exit_idx + 1
    if post_start >= len(df):
        return None

    post_df = df.iloc[post_start:].reset_index(drop=True)
    if len(post_df) < 5:
        return None

    result = {"exit_price": exit_price, "entry_price": entry_price}

    # Max drawdown and max rally from exit price
    max_dd = 0.0
    max_rally = 0.0
    new_high_within_5d = False

    for i in range(len(post_df)):
        low = post_df.iloc[i]["low"]
        high = post_df.iloc[i]["high"]

        dd_from_exit = (low - exit_price) / exit_price * 100
        rally_from_exit = (high - exit_price) / exit_price * 100

        if dd_from_exit < max_dd:
            max_dd = dd_from_exit
        if rally_from_exit > max_rally:
            max_rally = rally_from_exit

        if i < 5 and high > exit_price:
            new_high_within_5d = True

    result["max_drawdown_pct"] = max_dd
    result["max_rally_pct"] = max_rally
    result["new_high_within_5d"] = new_high_within_5d

    # Returns at each horizon from exit price
    for h in HORIZONS:
        if h - 1 < len(post_df):
            close_at_h = post_df.iloc[h - 1]["close"]
            result[f"ret_{h}d"] = (close_at_h - exit_price) / exit_price * 100
        else:
            result[f"ret_{h}d"] = None

    # What would the return be from ENTRY if you held to each horizon from exit?
    # i.e. total return = stop gain + post-exit movement
    for h in HORIZONS:
        if h - 1 < len(post_df):
            close_at_h = post_df.iloc[h - 1]["close"]
            result[f"hold_from_entry_{h}d"] = (close_at_h - entry_price) / entry_price * 100
        else:
            result[f"hold_from_entry_{h}d"] = None

    # Did the stock ever drop below entry price after the winning exit?
    dropped_below_entry = False
    days_to_drop = None
    for i in range(min(30, len(post_df))):
        if post_df.iloc[i]["close"] < entry_price:
            dropped_below_entry = True
            days_to_drop = i + 1
            break
    result["dropped_below_entry"] = dropped_below_entry
    result["days_to_drop_below_entry"] = days_to_drop

    # Max gain from entry if held through (no stop exit)
    max_from_entry = 0.0
    for i in range(min(30, len(post_df))):
        high = post_df.iloc[i]["high"]
        gain = (high - entry_price) / entry_price * 100
        if gain > max_from_entry:
            max_from_entry = gain
    result["max_gain_from_entry_post"] = max_from_entry

    # Lower lows analysis
    period_lows = []
    for start in range(0, min(30, len(post_df)), 5):
        end = min(start + 5, len(post_df))
        if start >= len(post_df):
            break
        period_lows.append(post_df.iloc[start:end]["low"].min())

    consecutive_lower_lows = 0
    if len(period_lows) >= 2:
        for i in range(1, len(period_lows)):
            if period_lows[i] < period_lows[i - 1]:
                consecutive_lower_lows += 1
            else:
                break
    result["consecutive_lower_low_periods"] = consecutive_lower_lows

    # Higher highs analysis
    period_highs = []
    for start in range(0, min(30, len(post_df)), 5):
        end = min(start + 5, len(post_df))
        if start >= len(post_df):
            break
        period_highs.append(post_df.iloc[start:end]["high"].max())

    consecutive_higher_highs = 0
    if len(period_highs) >= 2:
        for i in range(1, len(period_highs)):
            if period_highs[i] > period_highs[i - 1]:
                consecutive_higher_highs += 1
            else:
                break
    result["consecutive_higher_high_periods"] = consecutive_higher_highs

    return result


def categorize_winner_post(post_data, trade_return):
    """
    Categorize post-exit behavior:
    - "kept running": continued higher, 10d return from exit > +10%
    - "modest continuation": 10d return from exit +2% to +10%
    - "topped out": 10d return from exit within +/-2% (the stop nailed the top)
    - "reversed": 10d return from exit < -2% (pulled back after exit)
    - "collapsed": 10d return from exit < -10% (major reversal)
    """
    ret_10d = post_data.get("ret_10d")
    if ret_10d is None:
        return "insufficient data"
    if ret_10d > 10:
        return "kept running"
    elif ret_10d > 2:
        return "modest continuation"
    elif ret_10d >= -2:
        return "topped out"
    elif ret_10d >= -10:
        return "reversed"
    else:
        return "collapsed"


def find_exit_date(db, ticker, signal_date, trade):
    """Given the trade result, compute the actual exit date."""
    signal_dt = date.fromisoformat(signal_date)
    fetch_start = (signal_dt - timedelta(days=5)).isoformat()
    fetch_end = (signal_dt + timedelta(days=30)).isoformat()

    rows = db.get_daily_prices(ticker, fetch_start, fetch_end)
    if not rows:
        return None

    df = pd.DataFrame(rows)
    sig_idx_list = df.index[df["date"] == signal_date].tolist()
    if not sig_idx_list:
        return None
    sig_idx = sig_idx_list[0]

    # Entry is day after signal (d1_idx)
    entry_idx = sig_idx + 1
    exit_idx = entry_idx + trade["hold_days"]

    if exit_idx < len(df):
        return df.iloc[exit_idx]["date"]
    return None


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

    # Collect unique winners
    print("\nSimulating trades and collecting winners...")
    seen = set()
    all_winners = []

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
            if not trade or trade["return"] <= 0:
                continue

            seen.add(key)

            exit_date = find_exit_date(db, s["ticker"], s["trend_date"], trade)
            if not exit_date:
                continue

            all_winners.append({
                "ticker": s["ticker"],
                "signal_date": s["trend_date"],
                "eps_pct": s.get("eps_change_pct", 0),
                "entry_price": trade.get("entry_price", 0),
                "exit_price": trade.get("exit_price", 0),
                "trade_return": trade["return"],
                "hold_days": trade["hold_days"],
                "exit_reason": trade["exit_reason"],
                "exit_date": exit_date,
            })

    print(f"  Total unique winners: {len(all_winners)}")

    # Analyze post-exit behavior
    print("\nAnalyzing post-exit behavior for each winner...\n")
    analyzed = []
    for w in all_winners:
        post = get_post_exit_data(
            db, w["ticker"], w["exit_date"],
            w["entry_price"], w["exit_price"],
        )
        if post:
            w["post"] = post
            w["category"] = categorize_winner_post(post, w["trade_return"])
            analyzed.append(w)

    analyzed.sort(key=lambda x: (x["signal_date"], x["ticker"]))

    # ══════════════════════════════════════════════════════════════════════
    # DETAILED TABLE
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 200)
    print("  POST-STOP ANALYSIS FOR WINNERS: Does the stock keep moving higher after the trailing stop fires?")
    print("  Winners from C17 filter | midpoint entry | 0.5x ATR trailing stop | 15d max hold")
    print("=" * 200)

    hdr = (f"  {'Ticker':7s} {'Signal':12s} {'Exit Date':12s} {'Entry$':>8s} {'Exit$':>8s} "
           f"{'Gain%':>7s} {'Reason':10s} {'Hold':>5s} | "
           f"{'MaxDD%':>7s} {'MaxUp%':>7s} | "
           f"{'5d%':>7s} {'10d%':>7s} {'15d%':>7s} {'20d%':>7s} {'30d%':>7s} | "
           f"{'Category':20s}")
    print(hdr)
    print("  " + "-" * 190)

    for w in analyzed:
        p = w["post"]
        rets = []
        for h in HORIZONS:
            v = p.get(f"ret_{h}d")
            rets.append(f"{v:>+6.1f}%" if v is not None else "    N/A")

        print(f"  {w['ticker']:7s} {w['signal_date']:12s} {w['exit_date']:12s} "
              f"${w['entry_price']:>7.2f} ${w['exit_price']:>7.2f} "
              f"{w['trade_return']:>+6.1f}% {w['exit_reason']:10s} {w['hold_days']:>4d}d | "
              f"{p['max_drawdown_pct']:>+6.1f}% {p['max_rally_pct']:>+6.1f}% | "
              f"{rets[0]} {rets[1]} {rets[2]} {rets[3]} {rets[4]} | "
              f"{w['category']:20s}")

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print()
    print("=" * 120)
    print("  SUMMARY STATISTICS (%d winners analyzed)" % len(analyzed))
    print("=" * 120)

    # How many kept going up?
    new_high_5d = sum(1 for w in analyzed if w["post"]["new_high_within_5d"])
    print(f"\n  CONTINUED MOMENTUM:")
    print(f"    Made a new high within 5d of exit: {new_high_5d} / {len(analyzed)} ({new_high_5d/len(analyzed)*100:.1f}%)")

    # Did they drop below entry?
    dropped = sum(1 for w in analyzed if w["post"]["dropped_below_entry"])
    print(f"\n  GAVE BACK ALL GAINS:")
    print(f"    Dropped below entry price within 30d of exit: {dropped} / {len(analyzed)} ({dropped/len(analyzed)*100:.1f}%)")
    if dropped:
        drop_days = [w["post"]["days_to_drop_below_entry"] for w in analyzed if w["post"]["dropped_below_entry"]]
        print(f"    Avg days to drop below entry: {mean(drop_days):.1f}d")

    # Return distribution from exit
    print(f"\n  RETURN DISTRIBUTION FROM EXIT PRICE:")
    print(f"    {'Horizon':10s} {'Avg':>8s} {'Median':>8s} {'Min':>8s} {'Max':>8s} {'StdDev':>8s}")
    print(f"    {'-'*52}")
    for h in HORIZONS:
        vals = [w["post"][f"ret_{h}d"] for w in analyzed if w["post"].get(f"ret_{h}d") is not None]
        if vals:
            print(f"    {h}d{'':<8s} {mean(vals):>+7.1f}% {median(vals):>+7.1f}% "
                  f"{min(vals):>+7.1f}% {max(vals):>+7.1f}% {pd.Series(vals).std():>7.1f}%")

    # What did winners actually gain vs what they COULD have gained?
    print(f"\n  MONEY LEFT ON THE TABLE:")
    print(f"    {'Ticker':7s} {'Signal':12s} {'Stop Gain':>10s} {'Max if held 30d':>16s} {'Left on table':>14s}")
    print(f"    {'-'*65}")
    total_left = []
    for w in analyzed:
        p = w["post"]
        max_post = p["max_gain_from_entry_post"]
        # Total potential: trade return + max further gain from exit
        # But max_gain_from_entry_post is already from entry price
        total_potential = w["trade_return"] + max_post  # approximate
        # More accurate: max gain from entry = trade_return + max_rally from exit
        actual_max = w["trade_return"] + p["max_rally_pct"]
        left = p["max_rally_pct"]  # how much more it went up after exit
        total_left.append(left)

        ret_30 = p.get("ret_30d")
        ret_30_s = f"{ret_30:>+6.1f}%" if ret_30 is not None else "   N/A"

        print(f"    {w['ticker']:7s} {w['signal_date']:12s}  {w['trade_return']:>+7.1f}%   "
              f"exit+{p['max_rally_pct']:>+6.1f}% more     "
              f"{p['max_rally_pct']:>+6.1f}%")

    print(f"\n    Average additional upside after exit: {mean(total_left):>+.1f}%")
    print(f"    Median additional upside after exit:  {median(total_left):>+.1f}%")

    # Comparing: what if you held with NO stop at all?
    print(f"\n  HOLD FROM ENTRY (NO STOP) vs ACTUAL STOP EXIT:")
    print(f"    {'Horizon':10s} {'Stopped (avg)':>14s} {'Held (avg)':>12s} {'Diff':>8s}")
    print(f"    {'-'*48}")
    for h in HORIZONS:
        hold_vals = [w["post"][f"hold_from_entry_{h}d"] for w in analyzed
                     if w["post"].get(f"hold_from_entry_{h}d") is not None]
        stop_gain = mean([w["trade_return"] for w in analyzed])
        if hold_vals:
            hold_avg = mean(hold_vals)
            print(f"    {h}d{'':<8s} {stop_gain:>+7.1f}% lock  {hold_avg:>+7.1f}%    {hold_avg - stop_gain:>+7.1f}%")

    # ══════════════════════════════════════════════════════════════════════
    # CATEGORY BREAKDOWN
    # ══════════════════════════════════════════════════════════════════════
    print()
    print("=" * 120)
    print("  WINNER CATEGORIZATION (based on 10d return from exit price)")
    print("=" * 120)
    print("    kept running       = 10d from exit > +10%  (stop clearly too tight)")
    print("    modest continuation= 10d from exit +2% to +10%")
    print("    topped out         = 10d from exit within +/-2% (stop nailed the top)")
    print("    reversed           = 10d from exit -2% to -10% (good exit)")
    print("    collapsed          = 10d from exit < -10% (great exit)")
    print()

    categories = {}
    for w in analyzed:
        cat = w["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(w)

    cat_order = ["kept running", "modest continuation", "topped out", "reversed", "collapsed", "insufficient data"]
    for cat in cat_order:
        trades = categories.get(cat, [])
        if not trades:
            continue
        n = len(trades)
        avg_stop = mean([t["trade_return"] for t in trades])
        avg_hold = mean([t["hold_days"] for t in trades])
        ret_10_vals = [t["post"]["ret_10d"] for t in trades if t["post"].get("ret_10d") is not None]
        ret_30_vals = [t["post"]["ret_30d"] for t in trades if t["post"].get("ret_30d") is not None]
        avg_10 = mean(ret_10_vals) if ret_10_vals else 0
        avg_30 = mean(ret_30_vals) if ret_30_vals else 0
        avg_max_rally = mean([t["post"]["max_rally_pct"] for t in trades])
        avg_max_dd = mean([t["post"]["max_drawdown_pct"] for t in trades])

        print(f"    {cat:22s}: {n:>3d} / {len(analyzed):>3d} ({n/len(analyzed)*100:>5.1f}%) | "
              f"avg stop gain: {avg_stop:>+6.1f}% | avg hold: {avg_hold:.1f}d | "
              f"10d after: {avg_10:>+6.1f}% | 30d after: {avg_30:>+6.1f}% | "
              f"max rally: {avg_max_rally:>+6.1f}%")

    # Detail per category
    for cat in cat_order:
        trades = categories.get(cat, [])
        if not trades:
            continue
        print(f"\n    -- {cat.upper()} ({len(trades)} trades) --")
        for t in trades:
            p = t["post"]
            r10 = p.get("ret_10d")
            r30 = p.get("ret_30d")
            r10s = f"{r10:>+6.1f}%" if r10 is not None else "   N/A"
            r30s = f"{r30:>+6.1f}%" if r30 is not None else "   N/A"
            drop_s = f"dropped@{p['days_to_drop_below_entry']}d" if p["dropped_below_entry"] else "held above"
            print(f"       {t['ticker']:7s} {t['signal_date']}  "
                  f"stop gain={t['trade_return']:>+6.1f}% ({t['hold_days']}d) | "
                  f"DD={p['max_drawdown_pct']:>+6.1f}% rally={p['max_rally_pct']:>+6.1f}% | "
                  f"10d={r10s} 30d={r30s} | {drop_s}")

    # ══════════════════════════════════════════════════════════════════════
    # BY EXIT REASON
    # ══════════════════════════════════════════════════════════════════════
    print()
    print("=" * 120)
    print("  POST-EXIT BEHAVIOR BY EXIT REASON")
    print("=" * 120)

    for reason in ["atr_stop", "stop_d1", "horizon"]:
        trades = [w for w in analyzed if w["exit_reason"] == reason]
        if not trades:
            continue
        n = len(trades)
        avg_hold = mean([t["hold_days"] for t in trades])
        avg_stop = mean([t["trade_return"] for t in trades])
        avg_dd = mean([t["post"]["max_drawdown_pct"] for t in trades])
        avg_rally = mean([t["post"]["max_rally_pct"] for t in trades])
        ret_10 = [t["post"]["ret_10d"] for t in trades if t["post"].get("ret_10d") is not None]
        avg_10 = mean(ret_10) if ret_10 else 0
        dropped = sum(1 for t in trades if t["post"]["dropped_below_entry"])

        cats = {}
        for t in trades:
            c = t["category"]
            cats[c] = cats.get(c, 0) + 1
        cat_str = ", ".join(f"{c}:{v}" for c, v in sorted(cats.items()))

        print(f"\n  {reason} (n={n}, avg hold={avg_hold:.1f}d, avg stop gain={avg_stop:>+.1f}%):")
        print(f"    Avg further DD: {avg_dd:>+.1f}% | Avg rally: {avg_rally:>+.1f}% | "
              f"10d ret from exit: {avg_10:>+.1f}% | Dropped below entry: {dropped}/{n} ({dropped/n*100:.0f}%)")
        print(f"    Categories: {cat_str}")

    # ══════════════════════════════════════════════════════════════════════
    # BY TRADE SIZE
    # ══════════════════════════════════════════════════════════════════════
    print()
    print("=" * 120)
    print("  SMALL vs BIG WINNERS: Post-exit behavior by trade size")
    print("=" * 120)

    small = [w for w in analyzed if w["trade_return"] <= 5]
    big = [w for w in analyzed if w["trade_return"] > 5]

    for label, group in [("Small winners (<=5%)", small), ("Big winners (>5%)", big)]:
        if not group:
            continue
        n = len(group)
        avg_stop = mean([t["trade_return"] for t in group])
        avg_dd = mean([t["post"]["max_drawdown_pct"] for t in group])
        avg_rally = mean([t["post"]["max_rally_pct"] for t in group])
        ret_10 = [t["post"]["ret_10d"] for t in group if t["post"].get("ret_10d") is not None]
        ret_30 = [t["post"]["ret_30d"] for t in group if t["post"].get("ret_30d") is not None]
        avg_10 = mean(ret_10) if ret_10 else 0
        avg_30 = mean(ret_30) if ret_30 else 0
        dropped = sum(1 for t in group if t["post"]["dropped_below_entry"])
        cats = {}
        for t in group:
            c = t["category"]
            cats[c] = cats.get(c, 0) + 1
        cat_str = ", ".join(f"{c}:{v}" for c, v in sorted(cats.items()))

        print(f"\n  {label} (n={n}, avg stop gain={avg_stop:>+.1f}%):")
        print(f"    Avg further DD: {avg_dd:>+.1f}% | Avg rally: {avg_rally:>+.1f}% | "
              f"10d: {avg_10:>+.1f}% | 30d: {avg_30:>+.1f}% | "
              f"Dropped below entry: {dropped}/{n} ({dropped/n*100:.0f}%)")
        print(f"    Categories: {cat_str}")

    print()
    print("=" * 120)
    print("  ANALYSIS COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()
