"""
Post-stop-loss analysis: what happens AFTER a losing trade gets stopped out?
Do stopped-out losers keep falling, recover quickly, or start a new trend?

For each C17 loser (bullish + EPS>0 + avg_dollar_vol>=$500K + |EPS|<100% + days>10),
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


def get_post_exit_data(db, ticker, exit_date_str, entry_price, exit_price, max_horizon=30):
    """
    Fetch price data after the exit date and compute post-exit metrics.
    Returns dict with drawdown, rally, horizon returns, recovery info, etc.
    """
    exit_dt = date.fromisoformat(exit_date_str)
    # Fetch enough data: exit day + max_horizon + buffer for weekends/holidays
    fetch_start = (exit_dt - timedelta(days=5)).isoformat()
    fetch_end = (exit_dt + timedelta(days=int(max_horizon * 1.6) + 10)).isoformat()

    rows = db.get_daily_prices(ticker, fetch_start, fetch_end)
    if not rows:
        return None

    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # Find exit date index
    exit_idx_list = df.index[df["date"] == exit_date_str].tolist()
    if not exit_idx_list:
        return None
    exit_idx = exit_idx_list[0]

    # Post-exit data starts the day after exit
    post_start = exit_idx + 1
    if post_start >= len(df):
        return None

    post_df = df.iloc[post_start:].reset_index(drop=True)
    if len(post_df) < 5:
        return None

    result = {
        "exit_price": exit_price,
        "entry_price": entry_price,
    }

    # --- Max drawdown and max rally from exit price ---
    max_dd = 0.0
    max_rally = 0.0
    recovered_above_entry = False
    recovery_days = None
    new_low_within_5d = False

    for i in range(len(post_df)):
        low = post_df.iloc[i]["low"]
        high = post_df.iloc[i]["high"]
        close = post_df.iloc[i]["close"]

        dd_from_exit = (low - exit_price) / exit_price * 100
        rally_from_exit = (high - exit_price) / exit_price * 100

        if dd_from_exit < max_dd:
            max_dd = dd_from_exit

        if rally_from_exit > max_rally:
            max_rally = rally_from_exit

        # Check if close recovered above entry price
        if not recovered_above_entry and close >= entry_price:
            recovered_above_entry = True
            recovery_days = i + 1  # 1-indexed trading days after exit

        # New low within 5 trading days
        if i < 5 and low < exit_price:
            new_low_within_5d = True

    result["max_drawdown_pct"] = max_dd
    result["max_rally_pct"] = max_rally
    result["recovered_above_entry"] = recovered_above_entry
    result["recovery_days"] = recovery_days
    result["new_low_within_5d"] = new_low_within_5d

    # --- Returns at each horizon ---
    for h in HORIZONS:
        if h - 1 < len(post_df):
            close_at_h = post_df.iloc[h - 1]["close"]
            ret = (close_at_h - exit_price) / exit_price * 100
            result[f"ret_{h}d"] = ret
        else:
            result[f"ret_{h}d"] = None

    # --- Recovery within specific windows ---
    for window in [10, 20, 30]:
        found = False
        for i in range(min(window, len(post_df))):
            if post_df.iloc[i]["close"] >= entry_price:
                found = True
                break
        result[f"recovered_within_{window}d"] = found

    # --- Would holding (no stop) have been profitable? ---
    # Check at each horizon whether buy-and-hold from entry would have been positive
    # We need the price at original entry + horizon trading days
    # But we only have post-exit data; use exit_day offset
    # For hold analysis, we use entry_price vs close at horizon-from-exit
    # This approximation shows: if you held from entry without stopping, what's the return
    # at the same calendar point as exit + remaining horizon days
    for h in HORIZONS:
        if h - 1 < len(post_df):
            close_at_h = post_df.iloc[h - 1]["close"]
            hold_ret = (close_at_h - entry_price) / entry_price * 100
            result[f"hold_ret_{h}d"] = hold_ret
        else:
            result[f"hold_ret_{h}d"] = None

    # --- Lower lows analysis for new trend detection ---
    # Break post-exit into 5d periods and check if lows are consecutively lower
    period_lows = []
    for start in range(0, min(30, len(post_df)), 5):
        end = min(start + 5, len(post_df))
        if start >= len(post_df):
            break
        period_low = post_df.iloc[start:end]["low"].min()
        period_lows.append(period_low)

    consecutive_lower_lows = 0
    if len(period_lows) >= 2:
        for i in range(1, len(period_lows)):
            if period_lows[i] < period_lows[i - 1]:
                consecutive_lower_lows += 1
            else:
                break

    result["consecutive_lower_low_periods"] = consecutive_lower_lows
    result["period_lows"] = period_lows

    return result


def categorize_loser(post_data):
    """
    Categorize each loser as one of:
    - "kept falling": max further DD > 5% AND never recovers above entry in 30d
    - "V-recovery": recovers above entry within 10d
    - "sideways chop": stays within 5% of exit for 10d+
    - "new trend down": makes lower lows for 3+ consecutive 5d periods
    """
    max_dd = post_data["max_drawdown_pct"]
    recovered_within_10d = post_data.get("recovered_within_10d", False)
    recovered_within_30d = post_data.get("recovered_within_30d", False)
    max_rally = post_data["max_rally_pct"]
    consec_ll = post_data["consecutive_lower_low_periods"]

    # Check sideways: both max_dd and max_rally within 5% of exit for at least 10d
    sideways = abs(max_dd) <= 5 and abs(max_rally) <= 5

    if recovered_within_10d:
        return "V-recovery"
    elif consec_ll >= 3:
        return "new trend down"
    elif max_dd < -5 and not recovered_within_30d:
        return "kept falling"
    elif sideways:
        return "sideways chop"
    elif max_dd < -5:
        return "kept falling"
    else:
        return "sideways chop"


def main():
    db = Database(DB_PATH)
    db.initialize()

    configs = [
        (10.0, 60), (20.0, 45), (20.0, 60),
        (25.0, 30), (25.0, 60), (30.0, 30), (30.0, 60),
    ]

    # --- Phase 1: Collect signals across all configs ---
    print("Loading signals across %d config combos..." % len(configs))
    signal_cache = {}
    for eps_thresh, tw in configs:
        sc = ScannerConfig(
            min_price=1.0, max_price=50.0,
            min_market_cap=50_000_000, max_market_cap=10_000_000_000,
            ma_crossover_pairs=[(20, 50)],
            eps_change_threshold=eps_thresh,
            trend_window_days=tw, direction="both",
        )
        bc = BacktestConfig(
            start_date="2022-01-01", end_date="2026-03-19",
            forward_return_days=[5, 10, 15, 30, 60],
            ma_crossover_pairs=[(20, 50)],
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
        print("  EPS>=%.0f%% / %dd: %d C17 signals" % (eps_thresh, tw, len(c17)))

    # --- Phase 2: Simulate trades, collect losers (deduplicated) ---
    print("\nSimulating trades and collecting losers...")
    seen = set()
    all_losers = []

    for (eps_thresh, tw), (c17, raw) in sorted(signal_cache.items()):
        for s in c17:
            key = (s["ticker"], s["trend_date"])
            if key in seen:
                continue

            trade = simulate_trade_v2(
                db, s["ticker"], s["trend_date"], "bullish",
                entry_type="midpoint", atr_mult=0.5, max_days=15,
            )
            if not trade:
                continue

            seen.add(key)

            # Only keep losers (return <= 0)
            if trade["return"] > 0:
                continue

            # Compute exit date: signal_date + hold_days trading days
            signal_dt = date.fromisoformat(s["trend_date"])
            fetch_start = (signal_dt - timedelta(days=5)).isoformat()
            fetch_end = (signal_dt + timedelta(days=int(trade["hold_days"] * 1.6) + 10)).isoformat()
            rows = db.get_daily_prices(s["ticker"], fetch_start, fetch_end)
            if not rows:
                continue

            df_temp = pd.DataFrame(rows)
            sig_idx_list = df_temp.index[df_temp["date"] == s["trend_date"]].tolist()
            if not sig_idx_list:
                continue
            sig_idx = sig_idx_list[0]

            exit_idx = sig_idx + trade["hold_days"]
            if exit_idx >= len(df_temp):
                continue

            exit_date_str = df_temp.iloc[exit_idx]["date"]

            all_losers.append({
                "ticker": s["ticker"],
                "signal_date": s["trend_date"],
                "entry_price": trade["entry_price"],
                "exit_price": trade["exit_price"],
                "stop_return": trade["return"],
                "exit_reason": trade["exit_reason"],
                "hold_days": trade["hold_days"],
                "exit_date": exit_date_str,
                "eps_pct": s.get("eps_change_pct", 0),
            })

    print("  Total unique losers: %d" % len(all_losers))

    # --- Phase 3: Analyze post-exit behavior ---
    print("\nAnalyzing post-exit behavior for each loser...\n")
    analyzed = []
    for loser in all_losers:
        post = get_post_exit_data(
            db, loser["ticker"], loser["exit_date"],
            loser["entry_price"], loser["exit_price"],
            max_horizon=30,
        )
        if post is None:
            continue

        category = categorize_loser(post)

        analyzed.append({
            **loser,
            **post,
            "category": category,
        })

    if not analyzed:
        print("No losers with sufficient post-exit data found.")
        return

    # Sort by signal date
    analyzed.sort(key=lambda x: (x["signal_date"], x["ticker"]))

    # ===================================================================
    # SECTION 1: Detailed table for each loser
    # ===================================================================
    print("=" * 200)
    print("  POST-STOP-LOSS ANALYSIS: What happens after a losing trade exits?")
    print("  Losers from C17 filter | midpoint entry | 0.5x ATR trailing stop | 15d max hold")
    print("=" * 200)
    print()

    # Header line 1: trade info
    hdr1 = (
        "  %-7s %-11s %-11s %8s %8s %7s %6s %5s | "
        "%8s %8s | %7s %7s %7s %7s %7s | %-6s %5s | %-15s"
    )
    print(hdr1 % (
        "Ticker", "Signal", "Exit Date", "Entry$", "Exit$", "Stop%", "Reason", "Hold",
        "MaxDD%", "MaxUp%",
        "5d%", "10d%", "15d%", "20d%", "30d%",
        "Recov?", "Days",
        "Category",
    ))
    print("  " + "-" * 190)

    for t in analyzed:
        recov_str = "Yes" if t["recovered_above_entry"] else "No"
        recov_days_str = "%3dd" % t["recovery_days"] if t["recovery_days"] else "  --"

        rets = []
        for h in HORIZONS:
            val = t.get("ret_%dd" % h)
            rets.append("%+6.1f%%" % val if val is not None else "    N/A")

        row = (
            "  %-7s %-11s %-11s $%7.2f $%7.2f %+6.1f%% %6s %4dd | "
            "%+7.1f%% %+7.1f%% | %s %s %s %s %s | %-6s %s | %-15s"
        )
        print(row % (
            t["ticker"], t["signal_date"], t["exit_date"],
            t["entry_price"], t["exit_price"], t["stop_return"],
            t["exit_reason"], t["hold_days"],
            t["max_drawdown_pct"], t["max_rally_pct"],
            rets[0], rets[1], rets[2], rets[3], rets[4],
            recov_str, recov_days_str,
            t["category"],
        ))

    # ===================================================================
    # SECTION 2: Summary statistics
    # ===================================================================
    n = len(analyzed)
    print()
    print("=" * 130)
    print("  SUMMARY STATISTICS (%d losers analyzed)" % n)
    print("=" * 130)

    # --- What % kept falling (new low within 5d) ---
    new_low_5d = sum(1 for t in analyzed if t["new_low_within_5d"])
    print()
    print("  CONTINUED DECLINE:")
    print("    Losers that made a NEW LOW within 5d of exit: %d / %d (%.1f%%)" % (
        new_low_5d, n, new_low_5d / n * 100))

    # --- What % recovered above entry within 10d, 20d, 30d ---
    print()
    print("  RECOVERY ABOVE ENTRY PRICE:")
    for window in [10, 20, 30]:
        recovered = sum(1 for t in analyzed if t.get("recovered_within_%dd" % window, False))
        print("    Recovered above entry within %2dd: %d / %d (%.1f%%)" % (
            window, recovered, n, recovered / n * 100))

    # --- What % would have been profitable if you just held (no stop) ---
    print()
    print("  WOULD HOLDING (NO STOP) HAVE BEEN PROFITABLE?")
    print("  (Return from original entry price at each horizon AFTER exit)")
    for h in HORIZONS:
        key = "hold_ret_%dd" % h
        vals = [t[key] for t in analyzed if t[key] is not None]
        if vals:
            profitable = sum(1 for v in vals if v > 0)
            avg_ret = mean(vals)
            med_ret = median(vals)
            print("    %2dd after exit: %d / %d profitable (%.1f%%) | avg ret: %+.1f%% | median ret: %+.1f%%" % (
                h, profitable, len(vals), profitable / len(vals) * 100, avg_ret, med_ret))

    # --- Average max drawdown after exit vs average max recovery ---
    print()
    print("  MAX DRAWDOWN vs MAX RALLY AFTER EXIT:")
    dd_vals = [t["max_drawdown_pct"] for t in analyzed]
    rally_vals = [t["max_rally_pct"] for t in analyzed]
    print("    Avg max further drawdown: %+.1f%% (median: %+.1f%%)" % (mean(dd_vals), median(dd_vals)))
    print("    Avg max rally from exit:  %+.1f%% (median: %+.1f%%)" % (mean(rally_vals), median(rally_vals)))
    print("    Avg stop loss at exit:    %+.1f%%" % mean([t["stop_return"] for t in analyzed]))

    # --- Return distribution at each horizon ---
    print()
    print("  RETURN DISTRIBUTION FROM EXIT PRICE:")
    print("    %-8s %8s %8s %8s %8s %8s" % ("Horizon", "Avg", "Median", "Min", "Max", "StdDev"))
    print("    " + "-" * 50)
    for h in HORIZONS:
        key = "ret_%dd" % h
        vals = [t[key] for t in analyzed if t[key] is not None]
        if vals:
            std_val = float(np.std(vals)) if len(vals) > 1 else 0.0
            print("    %-8s %+7.1f%% %+7.1f%% %+7.1f%% %+7.1f%% %7.1f%%" % (
                "%dd" % h, mean(vals), median(vals), min(vals), max(vals), std_val))

    # --- Categorization ---
    print()
    print("  LOSER CATEGORIZATION:")
    print("    (kept falling  = max further DD > 5%% AND never recovers above entry in 30d)")
    print("    (V-recovery    = recovers above entry within 10d)")
    print("    (sideways chop = stays within 5%% of exit for 10d+)")
    print("    (new trend dn  = lower lows for 3+ consecutive 5d periods)")
    print()

    categories = {}
    for t in analyzed:
        cat = t["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(t)

    for cat in ["V-recovery", "sideways chop", "kept falling", "new trend down"]:
        group = categories.get(cat, [])
        pct = len(group) / n * 100 if n > 0 else 0
        if group:
            avg_stop = mean([t["stop_return"] for t in group])
            avg_dd = mean([t["max_drawdown_pct"] for t in group])
            avg_rally = mean([t["max_rally_pct"] for t in group])
            ret_10_vals = [t["ret_10d"] for t in group if t["ret_10d"] is not None]
            avg_10d = mean(ret_10_vals) if ret_10_vals else 0
            ret_30_vals = [t["ret_30d"] for t in group if t["ret_30d"] is not None]
            avg_30d = mean(ret_30_vals) if ret_30_vals else 0
            print("    %-15s: %3d / %3d (%5.1f%%) | avg stop: %+5.1f%% | avg DD: %+6.1f%% | "
                  "avg rally: %+6.1f%% | 10d ret: %+5.1f%% | 30d ret: %+5.1f%%" % (
                      cat, len(group), n, pct, avg_stop, avg_dd, avg_rally, avg_10d, avg_30d))
        else:
            print("    %-15s: %3d / %3d (%5.1f%%)" % (cat, 0, n, 0.0))

    # --- Category detail: show tickers ---
    print()
    print("  CATEGORY DETAIL:")
    for cat in ["V-recovery", "sideways chop", "kept falling", "new trend down"]:
        group = categories.get(cat, [])
        if group:
            print()
            print("    -- %s (%d trades) --" % (cat.upper(), len(group)))
            for t in sorted(group, key=lambda x: x["signal_date"]):
                ret_10 = t.get("ret_10d")
                ret_30 = t.get("ret_30d")
                r10_str = "%+5.1f%%" % ret_10 if ret_10 is not None else "  N/A"
                r30_str = "%+5.1f%%" % ret_30 if ret_30 is not None else "  N/A"
                recov_str = "recov@%dd" % t["recovery_days"] if t["recovery_days"] else "no recov"
                print("       %-7s %-11s stop=%+5.1f%% | DD=%+6.1f%% rally=%+6.1f%% | "
                      "10d=%s 30d=%s | %s" % (
                          t["ticker"], t["signal_date"], t["stop_return"],
                          t["max_drawdown_pct"], t["max_rally_pct"],
                          r10_str, r30_str, recov_str))

    # ===================================================================
    # SECTION 3: Should you re-enter after a stop-out?
    # ===================================================================
    print()
    print("=" * 130)
    print("  RE-ENTRY ANALYSIS: Should you buy back after being stopped out?")
    print("=" * 130)
    print()

    # For each horizon, show what % of losers had a positive return from exit price
    print("  If you bought at the EXIT PRICE (day after stop-out):")
    print("    %-8s  %8s  %10s  %8s  %8s" % ("Horizon", "Win%", "Avg Ret", "Avg Win", "Avg Loss"))
    print("    " + "-" * 52)
    for h in HORIZONS:
        key = "ret_%dd" % h
        vals = [t[key] for t in analyzed if t[key] is not None]
        if not vals:
            continue
        wins = [v for v in vals if v > 0]
        losses = [v for v in vals if v <= 0]
        wr = len(wins) / len(vals) * 100
        avg_ret = mean(vals)
        avg_w = mean(wins) if wins else 0
        avg_l = mean(losses) if losses else 0
        print("    %-8s  %7.1f%%  %+8.1f%%  %+7.1f%%  %+7.1f%%" % (
            "%dd" % h, wr, avg_ret, avg_w, avg_l))

    # Compare: if you held from entry (ignoring the stop), same horizons after exit
    print()
    print("  If you HELD FROM ENTRY (no stop loss), return at same calendar point:")
    print("    %-8s  %8s  %10s  %8s  %8s" % ("Horizon", "Win%", "Avg Ret", "Avg Win", "Avg Loss"))
    print("    " + "-" * 52)
    for h in HORIZONS:
        key = "hold_ret_%dd" % h
        vals = [t[key] for t in analyzed if t[key] is not None]
        if not vals:
            continue
        wins = [v for v in vals if v > 0]
        losses = [v for v in vals if v <= 0]
        wr = len(wins) / len(vals) * 100
        avg_ret = mean(vals)
        avg_w = mean(wins) if wins else 0
        avg_l = mean(losses) if losses else 0
        print("    %-8s  %7.1f%%  %+8.1f%%  %+7.1f%%  %+7.1f%%" % (
            "%dd" % h, wr, avg_ret, avg_w, avg_l))

    # ===================================================================
    # SECTION 4: Stop loss return vs post-exit behavior correlation
    # ===================================================================
    print()
    print("=" * 130)
    print("  STOP LOSS SEVERITY vs POST-EXIT BEHAVIOR")
    print("=" * 130)
    print()

    # Split losers by stop severity
    mild_stops = [t for t in analyzed if t["stop_return"] > -5]
    harsh_stops = [t for t in analyzed if t["stop_return"] <= -5]

    for label, group in [("Mild stop (> -5%%)", mild_stops), ("Harsh stop (<= -5%%)", harsh_stops)]:
        if not group:
            print("  %s: no trades" % label)
            continue
        gn = len(group)
        avg_dd = mean([t["max_drawdown_pct"] for t in group])
        avg_rally = mean([t["max_rally_pct"] for t in group])
        new_low = sum(1 for t in group if t["new_low_within_5d"])
        recov_10 = sum(1 for t in group if t.get("recovered_within_10d", False))
        recov_30 = sum(1 for t in group if t.get("recovered_within_30d", False))

        cat_counts = {}
        for t in group:
            c = t["category"]
            cat_counts[c] = cat_counts.get(c, 0) + 1
        cat_str = ", ".join("%s:%d" % (k, v) for k, v in sorted(cat_counts.items()))

        print("  %s (n=%d):" % (label, gn))
        print("    Avg further DD: %+.1f%% | Avg rally: %+.1f%%" % (avg_dd, avg_rally))
        print("    New low in 5d: %d/%d (%.0f%%) | Recov entry in 10d: %d/%d (%.0f%%) | "
              "Recov entry in 30d: %d/%d (%.0f%%)" % (
                  new_low, gn, new_low / gn * 100,
                  recov_10, gn, recov_10 / gn * 100,
                  recov_30, gn, recov_30 / gn * 100))
        print("    Categories: %s" % cat_str)
        print()

    # ===================================================================
    # SECTION 5: Exit reason breakdown
    # ===================================================================
    print("=" * 130)
    print("  POST-EXIT BEHAVIOR BY EXIT REASON")
    print("=" * 130)
    print()

    exit_reasons = {}
    for t in analyzed:
        r = t["exit_reason"]
        if r not in exit_reasons:
            exit_reasons[r] = []
        exit_reasons[r].append(t)

    for reason, group in sorted(exit_reasons.items()):
        gn = len(group)
        avg_stop = mean([t["stop_return"] for t in group])
        avg_dd = mean([t["max_drawdown_pct"] for t in group])
        avg_rally = mean([t["max_rally_pct"] for t in group])
        avg_hold = mean([t["hold_days"] for t in group])

        ret_10_vals = [t["ret_10d"] for t in group if t["ret_10d"] is not None]
        avg_10d = mean(ret_10_vals) if ret_10_vals else 0
        recov_30 = sum(1 for t in group if t.get("recovered_within_30d", False))

        cat_counts = {}
        for t in group:
            c = t["category"]
            cat_counts[c] = cat_counts.get(c, 0) + 1
        cat_str = ", ".join("%s:%d" % (k, v) for k, v in sorted(cat_counts.items()))

        print("  %s (n=%d, avg hold=%.1fd, avg stop=%+.1f%%):" % (reason, gn, avg_hold, avg_stop))
        print("    Avg further DD: %+.1f%% | Avg rally: %+.1f%% | "
              "10d ret from exit: %+.1f%% | Recov in 30d: %d/%d (%.0f%%)" % (
                  avg_dd, avg_rally, avg_10d, recov_30, gn, recov_30 / gn * 100))
        print("    Categories: %s" % cat_str)
        print()

    print("=" * 130)
    print("  ANALYSIS COMPLETE")
    print("=" * 130)


if __name__ == "__main__":
    main()
