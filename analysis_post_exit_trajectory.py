"""Post-exit trajectory analysis for the current system's 12 trades.

Maps the FULL post-exit path for each of the 7 winners and 5 losers:
  - Current system: D0 close entry, vol >= 1.2x, EPS>=25%/30d + EPS>=30%/60d combined & deduped,
    0.5x ATR trailing stop.
  - After the stop fires, fetches 60 trading days of price data.
  - Tracks: pullback depth, recovery time, new highs, re-entry opportunities.
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
# Configs: the two parameter sets whose trades we combine & dedupe
# ---------------------------------------------------------------------------
CONFIGS = [(25.0, 30), (30.0, 60)]


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


def simulate_atr_trade(db, ticker, signal_date, direction, atr_mult=0.5, max_days=60):
    """Simulate trade with D0 close entry, vol >= 1.2x filter, 0.5x ATR trailing stop.

    Returns dict with entry info, exit info, and the full price path during the trade.
    """
    signal_dt = date.fromisoformat(signal_date)
    fetch_start = (signal_dt - timedelta(days=60)).isoformat()
    fetch_end = (signal_dt + timedelta(days=int(max_days * 1.8) + 30)).isoformat()

    rows = db.get_daily_prices(ticker, fetch_start, fetch_end)
    if not rows:
        return None

    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # Find signal date index
    signal_idx_list = df.index[df["date"] == signal_date].tolist()
    if not signal_idx_list:
        return None
    signal_idx = signal_idx_list[0]

    # Volume filter: signal day volume >= 1.2x 20d avg
    if signal_idx >= 20:
        avg_vol_20 = df["volume"].iloc[signal_idx - 20:signal_idx].mean()
    else:
        avg_vol_20 = df["volume"].iloc[:signal_idx].mean() if signal_idx > 0 else df["volume"].iloc[signal_idx]
    signal_vol = df["volume"].iloc[signal_idx]
    if avg_vol_20 > 0 and signal_vol < 1.2 * avg_vol_20:
        return None  # filtered out

    # Entry = D0 close
    entry_price = df["close"].iloc[signal_idx]
    entry_date = signal_date

    # Compute ATR at entry
    atr_series = compute_atr(df)

    # Walk forward with trailing stop
    highest_close = entry_price
    trade_path = []  # list of {day, date, close, high, low, volume}

    exit_info = None
    for day_num in range(1, max_days + 1):
        cur_idx = signal_idx + day_num
        if cur_idx >= len(df):
            break

        row = df.iloc[cur_idx]
        close = row["close"]

        trade_path.append({
            "day": day_num,
            "date": row["date"],
            "close": close,
            "high": row["high"],
            "low": row["low"],
            "volume": row["volume"],
        })

        if close > highest_close:
            highest_close = close

        # ATR trailing stop check
        atr_val = atr_series.iloc[cur_idx] if cur_idx < len(atr_series) and pd.notna(atr_series.iloc[cur_idx]) else None
        if atr_val is None or pd.isna(atr_val):
            # fallback ATR
            lookback = df.iloc[max(0, cur_idx - 14):cur_idx]
            if len(lookback) > 0:
                atr_val = (lookback["high"] - lookback["low"]).mean()
            else:
                atr_val = 0

        if direction == "bullish":
            stop_level = highest_close - atr_mult * atr_val
            if close < stop_level and exit_info is None:
                ret = (close - entry_price) / entry_price * 100
                exit_info = {
                    "exit_date": row["date"],
                    "exit_price": close,
                    "exit_day": day_num,
                    "return_pct": ret,
                    "exit_reason": "atr_stop",
                    "peak_close": highest_close,
                }
                # Don't break - keep collecting path data for post-exit analysis

    # If no stop triggered, exit at horizon
    if exit_info is None and trade_path:
        last = trade_path[-1]
        ret = (last["close"] - entry_price) / entry_price * 100
        exit_info = {
            "exit_date": last["date"],
            "exit_price": last["close"],
            "exit_day": last["day"],
            "return_pct": ret,
            "exit_reason": "horizon",
            "peak_close": highest_close,
        }

    if exit_info is None:
        return None

    return {
        "ticker": ticker,
        "signal_date": signal_date,
        "entry_date": entry_date,
        "entry_price": entry_price,
        "direction": direction,
        "trade_path": trade_path,
        **exit_info,
    }


def get_post_exit_prices(db, ticker, exit_date, days=60):
    """Fetch up to `days` trading days of prices after exit_date."""
    exit_dt = date.fromisoformat(exit_date)
    fetch_start = exit_date  # include exit date itself
    fetch_end = (exit_dt + timedelta(days=int(days * 1.8) + 30)).isoformat()

    rows = db.get_daily_prices(ticker, fetch_start, fetch_end)
    if not rows:
        return []

    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # Find the exit date index and get rows AFTER it
    exit_idx_list = df.index[df["date"] == exit_date].tolist()
    if not exit_idx_list:
        # exit date not in data, just take everything
        return df.to_dict("records")[:days]

    exit_idx = exit_idx_list[0]
    post = df.iloc[exit_idx + 1: exit_idx + 1 + days]
    return post.to_dict("records")


def get_extended_prices(db, ticker, entry_date, total_days=120):
    """Fetch extended price data starting from entry date."""
    entry_dt = date.fromisoformat(entry_date)
    fetch_end = (entry_dt + timedelta(days=int(total_days * 1.8) + 30)).isoformat()

    rows = db.get_daily_prices(ticker, entry_date, fetch_end)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df


def simulate_reentry(db, ticker, reentry_date, reentry_price, direction, atr_mult=0.5, max_days=60):
    """Simulate a re-entry trade with same ATR trailing stop."""
    reentry_dt = date.fromisoformat(reentry_date)
    fetch_start = (reentry_dt - timedelta(days=30)).isoformat()
    fetch_end = (reentry_dt + timedelta(days=int(max_days * 1.8) + 30)).isoformat()

    rows = db.get_daily_prices(ticker, fetch_start, fetch_end)
    if not rows:
        return None

    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    atr_series = compute_atr(df)

    reentry_idx_list = df.index[df["date"] == reentry_date].tolist()
    if not reentry_idx_list:
        return None
    reentry_idx = reentry_idx_list[0]

    highest_close = reentry_price

    for day_num in range(1, max_days + 1):
        cur_idx = reentry_idx + day_num
        if cur_idx >= len(df):
            break

        row = df.iloc[cur_idx]
        close = row["close"]

        if close > highest_close:
            highest_close = close

        atr_val = atr_series.iloc[cur_idx] if cur_idx < len(atr_series) and pd.notna(atr_series.iloc[cur_idx]) else None
        if atr_val is None or pd.isna(atr_val):
            lookback = df.iloc[max(0, cur_idx - 14):cur_idx]
            atr_val = (lookback["high"] - lookback["low"]).mean() if len(lookback) > 0 else 0

        if direction == "bullish":
            stop_level = highest_close - atr_mult * atr_val
            if close < stop_level:
                ret = (close - reentry_price) / reentry_price * 100
                return {
                    "exit_date": row["date"],
                    "exit_price": close,
                    "hold_days": day_num,
                    "return_pct": ret,
                    "exit_reason": "atr_stop",
                    "peak_close": highest_close,
                }

    # Horizon exit
    last_idx = min(reentry_idx + max_days, len(df) - 1)
    if last_idx > reentry_idx:
        last_row = df.iloc[last_idx]
        ret = (last_row["close"] - reentry_price) / reentry_price * 100
        return {
            "exit_date": last_row["date"],
            "exit_price": last_row["close"],
            "hold_days": last_idx - reentry_idx,
            "return_pct": ret,
            "exit_reason": "horizon",
            "peak_close": highest_close,
        }
    return None


def gather_trades(db):
    """Run both configs, enrich, apply filters, dedupe, simulate with ATR stop."""
    all_trades = {}  # key = (ticker, signal_date)

    for eps_thresh, tw in CONFIGS:
        sc = ScannerConfig(
            min_price=1.0, max_price=50.0,
            min_market_cap=50_000_000, max_market_cap=10_000_000_000,
            ma_crossover_pairs=[(20, 50)],
            eps_change_threshold=10.0, trend_window_days=tw, direction="both",
        )
        bc = BacktestConfig(
            start_date="2022-01-01", end_date="2026-03-19",
            forward_return_days=[5, 10, 15, 30, 60],
            ma_crossover_pairs=[(20, 50)],
            eps_thresholds=[10.0], trend_windows=[tw],
        )
        result = Backtester(db, sc, bc).run()
        enriched = enrich_deep(db, result["signals"])

        # C17 filter: bullish + pos EPS + dollar vol >= $500K + |EPS| < 100% + days > 10
        c17 = [s for s in enriched
               if s["signal_type"] == "bullish"
               and s["eps_change_pct"] > 0
               and s.get("avg_dollar_vol", 0) >= 500_000
               and abs(s["eps_change_pct"]) < 100
               and s["days_between"] > 10]

        # Additional filter: EPS >= threshold for this config
        filtered = [s for s in c17 if abs(s["eps_change_pct"]) >= eps_thresh]

        for s in filtered:
            key = (s["ticker"], s["trend_date"])
            if key not in all_trades:
                all_trades[key] = s

    # Now simulate each with ATR trailing stop
    trades = []
    for (ticker, sig_date), s in all_trades.items():
        trade = simulate_atr_trade(db, ticker, sig_date, "bullish", atr_mult=0.5, max_days=60)
        if trade:
            trade["eps_change_pct"] = s.get("eps_change_pct", 0)
            trade["config"] = f"EPS>={s.get('eps_change_pct', 0):.0f}%"
            trades.append(trade)

    return trades


def analyze_post_exit(db, trade):
    """Analyze post-exit trajectory for a single trade."""
    ticker = trade["ticker"]
    entry_price = trade["entry_price"]
    exit_price = trade["exit_price"]
    exit_date = trade["exit_date"]
    exit_day = trade["exit_day"]
    peak_close = trade["peak_close"]
    is_winner = trade["return_pct"] > 0

    # Get 60 trading days after exit
    post_prices = get_post_exit_prices(db, ticker, exit_date, days=60)

    # Get extended data from entry (for full trend analysis)
    extended = get_extended_prices(db, ticker, trade["entry_date"], total_days=180)

    result = {
        "ticker": ticker,
        "entry_date": trade["entry_date"],
        "entry_price": entry_price,
        "exit_date": exit_date,
        "exit_price": exit_price,
        "exit_day": exit_day,
        "trade_return": trade["return_pct"],
        "peak_close_during_trade": peak_close,
        "is_winner": is_winner,
        "eps_change_pct": trade.get("eps_change_pct", 0),
        "exit_reason": trade["exit_reason"],
    }

    if not post_prices:
        result["post_exit_available"] = False
        return result

    result["post_exit_available"] = True

    # --- Post-exit analysis ---
    post_closes = [p["close"] for p in post_prices]
    post_dates = [p["date"] for p in post_prices]

    # Post-exit drawdown from exit price
    min_post_close = min(post_closes) if post_closes else exit_price
    max_drawdown_from_exit = (min_post_close - exit_price) / exit_price * 100
    bottom_idx = post_closes.index(min_post_close)
    days_to_bottom = bottom_idx + 1

    result["post_exit_min_close"] = min_post_close
    result["post_exit_max_dd_from_exit"] = max_drawdown_from_exit
    result["post_exit_days_to_bottom"] = days_to_bottom

    # Does it fall below entry price?
    below_entry_days = sum(1 for c in post_closes if c < entry_price)
    result["days_below_entry_post_exit"] = below_entry_days

    # Recovery: days to recover back to exit price
    recovery_day = None
    for i, c in enumerate(post_closes):
        if c >= exit_price:
            recovery_day = i + 1
            break
    result["days_to_recover_exit_price"] = recovery_day

    # Days to make new high above trade's peak close
    new_high_day = None
    for i, c in enumerate(post_closes):
        if c > peak_close:
            new_high_day = i + 1
            break
    result["days_to_new_high_above_peak"] = new_high_day

    # Price at milestones (% return from entry)
    milestones = [10, 20, 30, 45, 60]
    for m in milestones:
        if m - 1 < len(post_closes):
            price = post_closes[m - 1]
            ret = (price - entry_price) / entry_price * 100
            result[f"post_exit_{m}d_price"] = price
            result[f"post_exit_{m}d_ret_from_entry"] = ret
        else:
            result[f"post_exit_{m}d_price"] = None
            result[f"post_exit_{m}d_ret_from_entry"] = None

    # --- Full trend analysis (from entry, using extended data) ---
    if len(extended) > 0:
        entry_idx_list = extended.index[extended["date"] == trade["entry_date"]].tolist()
        if entry_idx_list:
            entry_idx = entry_idx_list[0]
            # Look at first 60 trading days from entry
            window_60 = extended.iloc[entry_idx:entry_idx + 61]
            if len(window_60) > 0:
                max_price_60d = window_60["close"].max()
                max_price_60d_idx = window_60["close"].idxmax()
                days_to_max = max_price_60d_idx - entry_idx
                total_trend_return = (max_price_60d - entry_price) / entry_price * 100

                result["max_price_60d_from_entry"] = max_price_60d
                result["days_entry_to_max_60d"] = days_to_max
                result["total_trend_return_60d"] = total_trend_return

                # Capture efficiency
                captured = trade["return_pct"]
                efficiency = (captured / total_trend_return * 100) if total_trend_return > 0 else 0
                result["capture_efficiency"] = efficiency
            else:
                result["max_price_60d_from_entry"] = None
                result["days_entry_to_max_60d"] = None
                result["total_trend_return_60d"] = None
                result["capture_efficiency"] = None

            # Ultimate high within ALL available data
            remaining = extended.iloc[entry_idx:]
            if len(remaining) > 0:
                ult_max = remaining["close"].max()
                ult_max_idx = remaining["close"].idxmax()
                ult_days = ult_max_idx - entry_idx
                ult_return = (ult_max - entry_price) / entry_price * 100

                result["ultimate_high"] = ult_max
                result["days_to_ultimate_high"] = ult_days
                result["ultimate_return"] = ult_return
            else:
                result["ultimate_high"] = None
                result["days_to_ultimate_high"] = None
                result["ultimate_return"] = None
        else:
            result["max_price_60d_from_entry"] = None
            result["days_entry_to_max_60d"] = None
            result["total_trend_return_60d"] = None
            result["capture_efficiency"] = None
            result["ultimate_high"] = None
            result["days_to_ultimate_high"] = None
            result["ultimate_return"] = None
    else:
        result["max_price_60d_from_entry"] = None

    # --- Re-entry analysis ---
    # Re-entry signal: stock closes above exit price after the pullback
    # We look for the bottom first, then the recovery above exit price
    reentry_result = None
    reentry_vol_result = None

    if post_prices and is_winner:
        # Find post-exit bottom, then look for recovery
        found_bottom = False
        for i, p in enumerate(post_prices):
            if i >= bottom_idx:
                found_bottom = True
            if found_bottom and p["close"] >= exit_price:
                # Basic re-entry: close above exit price
                if reentry_result is None:
                    reentry_result = {
                        "reentry_date": p["date"],
                        "reentry_price": p["close"],
                        "days_after_exit": i + 1,
                    }
                # Volume-confirmed re-entry
                if reentry_vol_result is None:
                    # Check volume
                    # Get 20d avg volume around this time
                    re_dt = date.fromisoformat(p["date"])
                    vol_start = (re_dt - timedelta(days=40)).isoformat()
                    vol_rows = db.get_daily_prices(ticker, vol_start, p["date"])
                    if vol_rows and len(vol_rows) > 5:
                        vol_df = pd.DataFrame(vol_rows)
                        vol_df["volume"] = vol_df["volume"].astype(float)
                        avg_vol = vol_df["volume"].iloc[:-1].tail(20).mean()
                        if avg_vol > 0 and p["volume"] >= 1.0 * avg_vol:
                            reentry_vol_result = {
                                "reentry_date": p["date"],
                                "reentry_price": p["close"],
                                "days_after_exit": i + 1,
                                "vol_confirmed": True,
                            }
                if reentry_result and reentry_vol_result:
                    break

        # Simulate re-entry trade
        for label, re_info in [("basic", reentry_result), ("vol_confirmed", reentry_vol_result)]:
            if re_info:
                re_trade = simulate_reentry(
                    db, ticker, re_info["reentry_date"], re_info["reentry_price"],
                    "bullish", atr_mult=0.5, max_days=60
                )
                if re_trade:
                    re_info[f"trade_return"] = re_trade["return_pct"]
                    re_info[f"trade_hold_days"] = re_trade["hold_days"]
                    re_info[f"trade_exit_reason"] = re_trade["exit_reason"]
                    re_info[f"trade_exit_date"] = re_trade["exit_date"]
                    re_info[f"trade_peak"] = re_trade["peak_close"]
                else:
                    re_info["trade_return"] = None

    result["reentry_basic"] = reentry_result
    result["reentry_vol"] = reentry_vol_result

    return result


def print_section_header(title):
    print("\n" + "=" * 120)
    print(f"  {title}")
    print("=" * 120)


def main():
    db = Database(DB_PATH)
    db.initialize()

    print("=" * 120)
    print("  POST-EXIT TRAJECTORY ANALYSIS")
    print("  System: D0 close entry | Vol >= 1.2x | EPS>=25%/30d + EPS>=30%/60d | 0.5x ATR trailing stop")
    print("=" * 120)

    # Gather and simulate all trades
    print("\nGathering trades from both configs and deduplicating...")
    trades = gather_trades(db)
    print(f"Total trades after dedup + vol filter + ATR stop: {len(trades)}")

    winners = [t for t in trades if t["return_pct"] > 0]
    losers = [t for t in trades if t["return_pct"] <= 0]
    print(f"Winners: {len(winners)}, Losers: {len(losers)}")

    # Show trade summary
    print(f"\n  {'Ticker':<7s} {'Signal':<11s} {'Entry$':>7s} {'Exit$':>7s} {'Ret%':>7s} {'W/L':>4s} "
          f"{'Days':>5s} {'Peak$':>7s} {'Exit Reason':<12s} {'EPS%':>6s}")
    print("  " + "-" * 95)
    for t in sorted(trades, key=lambda x: x["signal_date"]):
        wl = "W" if t["return_pct"] > 0 else "L"
        print(f"  {t['ticker']:<7s} {t['signal_date']:<11s} {t['entry_price']:>7.2f} {t['exit_price']:>7.2f} "
              f"{t['return_pct']:>+6.1f}% {wl:>4s} {t['exit_day']:>5d} {t['peak_close']:>7.2f} "
              f"{t['exit_reason']:<12s} {t['eps_change_pct']:>+5.0f}%")

    # Analyze each trade
    print("\nAnalyzing post-exit trajectories...")
    analyses = []
    for t in trades:
        a = analyze_post_exit(db, t)
        analyses.append(a)

    winner_analyses = [a for a in analyses if a["is_winner"]]
    loser_analyses = [a for a in analyses if not a["is_winner"]]

    # ======================================================================
    # SECTION 1: POST-EXIT PULLBACK (Winners)
    # ======================================================================
    print_section_header("SECTION 1: POST-EXIT PULLBACK (Winners)")
    print(f"\n  {'Ticker':<7s} {'Exit$':>7s} {'PostBot$':>8s} {'DD%':>7s} {'DaysBot':>7s} "
          f"{'BelowEntry':>10s} {'FellBelowEntry?':>15s}")
    print("  " + "-" * 75)

    for a in sorted(winner_analyses, key=lambda x: x["entry_date"]):
        if not a.get("post_exit_available"):
            continue
        fell_below = "YES" if a["days_below_entry_post_exit"] > 0 else "no"
        print(f"  {a['ticker']:<7s} {a['exit_price']:>7.2f} {a['post_exit_min_close']:>8.2f} "
              f"{a['post_exit_max_dd_from_exit']:>+6.1f}% {a['post_exit_days_to_bottom']:>7d} "
              f"{a['days_below_entry_post_exit']:>10d}d  {fell_below:>15s}")

    # Aggregate
    dd_vals = [a["post_exit_max_dd_from_exit"] for a in winner_analyses if a.get("post_exit_available")]
    bot_days = [a["post_exit_days_to_bottom"] for a in winner_analyses if a.get("post_exit_available")]
    below_entry = [a["days_below_entry_post_exit"] for a in winner_analyses if a.get("post_exit_available")]
    if dd_vals:
        print(f"\n  AVERAGES:  Drawdown from exit: {mean(dd_vals):+.1f}%  |  Days to bottom: {mean(bot_days):.1f}"
              f"  |  Days below entry: {mean(below_entry):.1f}")
        print(f"  MEDIANS:   Drawdown from exit: {median(dd_vals):+.1f}%  |  Days to bottom: {median(bot_days):.1f}"
              f"  |  Days below entry: {median(below_entry):.1f}")

    # ======================================================================
    # SECTION 2: RECOVERY (Winners)
    # ======================================================================
    print_section_header("SECTION 2: RECOVERY (Winners)")
    print(f"\n  {'Ticker':<7s} {'ExitDay':>7s} {'RecovDay':>8s} {'NewHighDay':>10s} "
          f"{'10d%':>7s} {'20d%':>7s} {'30d%':>7s} {'45d%':>7s} {'60d%':>7s}")
    print("  " + "-" * 85)

    for a in sorted(winner_analyses, key=lambda x: x["entry_date"]):
        if not a.get("post_exit_available"):
            continue
        recov = f"{a['days_to_recover_exit_price']}d" if a['days_to_recover_exit_price'] else "NEVER"
        new_hi = f"{a['days_to_new_high_above_peak']}d" if a['days_to_new_high_above_peak'] else "NEVER"
        vals = []
        for m in [10, 20, 30, 45, 60]:
            r = a.get(f"post_exit_{m}d_ret_from_entry")
            vals.append(f"{r:>+6.1f}%" if r is not None else "   N/A")
        print(f"  {a['ticker']:<7s} {a['exit_day']:>7d} {recov:>8s} {new_hi:>10s} "
              f"{'  '.join(vals)}")

    # ======================================================================
    # SECTION 3: THE FULL TREND (Winners)
    # ======================================================================
    print_section_header("SECTION 3: THE FULL TREND (Winners)")
    print(f"\n  {'Ticker':<7s} {'Entry$':>7s} {'TradeRet':>8s} {'Peak60d$':>8s} {'TotalRet60d':>11s} "
          f"{'DaysToMax':>9s} {'Captured%':>9s} {'UltHigh$':>8s} {'UltRet':>8s} {'UltDays':>7s}")
    print("  " + "-" * 105)

    for a in sorted(winner_analyses, key=lambda x: x["entry_date"]):
        max60 = a.get("max_price_60d_from_entry")
        total60 = a.get("total_trend_return_60d")
        days_max = a.get("days_entry_to_max_60d")
        eff = a.get("capture_efficiency")
        ult_hi = a.get("ultimate_high")
        ult_ret = a.get("ultimate_return")
        ult_days = a.get("days_to_ultimate_high")

        max60_s = f"{max60:>8.2f}" if max60 else "     N/A"
        total60_s = f"{total60:>+10.1f}%" if total60 is not None else "       N/A"
        days_max_s = f"{days_max:>9d}" if days_max is not None else "      N/A"
        eff_s = f"{eff:>8.1f}%" if eff is not None else "      N/A"
        ult_hi_s = f"{ult_hi:>8.2f}" if ult_hi else "     N/A"
        ult_ret_s = f"{ult_ret:>+7.1f}%" if ult_ret is not None else "     N/A"
        ult_days_s = f"{ult_days:>7d}" if ult_days is not None else "    N/A"

        print(f"  {a['ticker']:<7s} {a['entry_price']:>7.2f} {a['trade_return']:>+7.1f}% {max60_s} {total60_s} "
              f"{days_max_s} {eff_s} {ult_hi_s} {ult_ret_s} {ult_days_s}")

    # Averages
    trade_rets = [a["trade_return"] for a in winner_analyses]
    total_rets = [a["total_trend_return_60d"] for a in winner_analyses if a.get("total_trend_return_60d") is not None]
    efficiencies = [a["capture_efficiency"] for a in winner_analyses if a.get("capture_efficiency") is not None and a["capture_efficiency"] > 0]
    ult_rets = [a["ultimate_return"] for a in winner_analyses if a.get("ultimate_return") is not None]
    ult_days_list = [a["days_to_ultimate_high"] for a in winner_analyses if a.get("days_to_ultimate_high") is not None]

    if total_rets:
        print(f"\n  AVERAGES:")
        print(f"    Trade return:       {mean(trade_rets):>+.1f}%")
        print(f"    Total avail (60d):  {mean(total_rets):>+.1f}%")
        if efficiencies:
            print(f"    Capture efficiency: {mean(efficiencies):>.1f}%")
        if ult_rets:
            print(f"    Ultimate return:    {mean(ult_rets):>+.1f}%")
        if ult_days_list:
            print(f"    Days to ult high:   {mean(ult_days_list):>.1f}")

    # ======================================================================
    # SECTION 4: RE-ENTRY ANALYSIS (Winners)
    # ======================================================================
    print_section_header("SECTION 4: RE-ENTRY ANALYSIS (Winners)")
    print(f"\n  {'Ticker':<7s} {'ExitDate':<11s} {'ReEntryDate':<11s} {'Gap':>5s} "
          f"{'ReEntry$':>8s} {'ReEntRet':>8s} {'ReEntHold':>9s} {'Combined':>8s}")
    print("  " + "-" * 85)

    reentry_returns = []
    combined_returns = []
    viable_reentries = 0

    for a in sorted(winner_analyses, key=lambda x: x["entry_date"]):
        re = a.get("reentry_basic")
        if re and re.get("trade_return") is not None:
            viable_reentries += 1
            gap = re["days_after_exit"]
            combined = a["trade_return"] + re["trade_return"]
            reentry_returns.append(re["trade_return"])
            combined_returns.append(combined)
            print(f"  {a['ticker']:<7s} {a['exit_date']:<11s} {re['reentry_date']:<11s} {gap:>4d}d "
                  f"{re['reentry_price']:>8.2f} {re['trade_return']:>+7.1f}% "
                  f"{re.get('trade_hold_days', 0):>8d}d {combined:>+7.1f}%")
        else:
            reason = "no recovery" if not re else "no trade data"
            print(f"  {a['ticker']:<7s} {a['exit_date']:<11s} {'---':>11s} {'---':>5s} "
                  f"{'---':>8s} {'---':>8s} {'---':>9s} {'---':>8s}  ({reason})")

    # Volume-confirmed re-entries
    print(f"\n  Volume-confirmed re-entries (vol >= 1.0x avg):")
    print(f"  {'Ticker':<7s} {'ExitDate':<11s} {'ReEntryDate':<11s} {'Gap':>5s} "
          f"{'ReEntry$':>8s} {'ReEntRet':>8s} {'ReEntHold':>9s} {'Combined':>8s}")
    print("  " + "-" * 85)

    for a in sorted(winner_analyses, key=lambda x: x["entry_date"]):
        re = a.get("reentry_vol")
        if re and re.get("trade_return") is not None:
            gap = re["days_after_exit"]
            combined = a["trade_return"] + re["trade_return"]
            print(f"  {a['ticker']:<7s} {a['exit_date']:<11s} {re['reentry_date']:<11s} {gap:>4d}d "
                  f"{re['reentry_price']:>8.2f} {re['trade_return']:>+7.1f}% "
                  f"{re.get('trade_hold_days', 0):>8d}d {combined:>+7.1f}%")
        else:
            print(f"  {a['ticker']:<7s} {a['exit_date']:<11s} {'---':>11s} {'---':>5s} "
                  f"{'---':>8s} {'---':>8s} {'---':>9s} {'---':>8s}")

    # ======================================================================
    # SECTION 5: DO LOSERS HAVE A CONTINUATION TREND?
    # ======================================================================
    print_section_header("SECTION 5: DO LOSERS HAVE A CONTINUATION TREND?")
    print(f"\n  {'Ticker':<7s} {'Entry$':>7s} {'TradeRet':>8s} {'Max60d$':>8s} {'MaxRet60d':>9s} "
          f"{'DaysToMax':>9s} {'UltHigh$':>8s} {'UltRet':>8s} {'UltDays':>7s}")
    print("  " + "-" * 85)

    for a in sorted(loser_analyses, key=lambda x: x["entry_date"]):
        max60 = a.get("max_price_60d_from_entry")
        total60 = a.get("total_trend_return_60d")
        days_max = a.get("days_entry_to_max_60d")
        ult_hi = a.get("ultimate_high")
        ult_ret = a.get("ultimate_return")
        ult_days = a.get("days_to_ultimate_high")

        max60_s = f"{max60:>8.2f}" if max60 else "     N/A"
        total60_s = f"{total60:>+8.1f}%" if total60 is not None else "     N/A"
        days_max_s = f"{days_max:>9d}" if days_max is not None else "      N/A"
        ult_hi_s = f"{ult_hi:>8.2f}" if ult_hi else "     N/A"
        ult_ret_s = f"{ult_ret:>+7.1f}%" if ult_ret is not None else "     N/A"
        ult_days_s = f"{ult_days:>7d}" if ult_days is not None else "    N/A"

        print(f"  {a['ticker']:<7s} {a['entry_price']:>7.2f} {a['trade_return']:>+7.1f}% {max60_s} {total60_s} "
              f"{days_max_s} {ult_hi_s} {ult_ret_s} {ult_days_s}")

    # Loser stats
    loser_max_rets = [a["total_trend_return_60d"] for a in loser_analyses if a.get("total_trend_return_60d") is not None]
    loser_ult_rets = [a["ultimate_return"] for a in loser_analyses if a.get("ultimate_return") is not None]
    loser_would_win = sum(1 for r in loser_max_rets if r > 0)
    if loser_max_rets:
        print(f"\n  Of {len(loser_analyses)} losers:")
        print(f"    {loser_would_win} would have been profitable within 60d if not stopped out")
        print(f"    Avg max return within 60d: {mean(loser_max_rets):+.1f}%")
        if loser_ult_rets:
            print(f"    Avg ultimate return (all data): {mean(loser_ult_rets):+.1f}%")

    # ======================================================================
    # SECTION 6: SUMMARY TABLES
    # ======================================================================
    print_section_header("SECTION 6: SUMMARY TABLES")

    # Table 1: Winner trajectory summary
    print(f"\n  TABLE 1: WINNER TRAJECTORY SUMMARY")
    print(f"  {'Ticker':<7s} {'TrdRet':>7s} {'PeakRet':>7s} {'Ex>Bot':>7s} {'BotDay':>6s} "
          f"{'Ex>Recov':>8s} {'RecDay':>6s} {'UltHigh$':>8s} {'UltDay':>6s} {'TotAvail':>8s} {'Capt%':>6s}")
    print("  " + "-" * 95)

    for a in sorted(winner_analyses, key=lambda x: x["entry_date"]):
        peak_ret = (a["peak_close_during_trade"] - a["entry_price"]) / a["entry_price"] * 100
        ex_bot = a.get("post_exit_max_dd_from_exit", 0)
        bot_day = a.get("post_exit_days_to_bottom", 0)
        recov = a.get("days_to_recover_exit_price")
        recov_s = f"{recov:>5d}d" if recov else " NEVER"
        ult_hi = a.get("ultimate_high")
        ult_day = a.get("days_to_ultimate_high")
        total_avail = a.get("total_trend_return_60d")
        eff = a.get("capture_efficiency")

        ult_hi_s = f"{ult_hi:>8.2f}" if ult_hi else "     N/A"
        ult_day_s = f"{ult_day:>6d}" if ult_day is not None else "   N/A"
        total_s = f"{total_avail:>+7.1f}%" if total_avail is not None else "     N/A"
        eff_s = f"{eff:>5.1f}%" if eff is not None and eff > 0 else "   N/A"

        print(f"  {a['ticker']:<7s} {a['trade_return']:>+6.1f}% {peak_ret:>+6.1f}% {ex_bot:>+6.1f}% "
              f"{bot_day:>6d} {recov_s:>8s} {ult_hi_s} {ult_day_s} {total_s} {eff_s}")

    # Table 2: Winner re-entry opportunities
    print(f"\n  TABLE 2: WINNER RE-ENTRY OPPORTUNITIES")
    print(f"  {'Ticker':<7s} {'ExitDate':<11s} {'ReEntDate':<11s} {'DaysBtwn':>8s} "
          f"{'ReEnt$':>7s} {'ReEntRet':>8s} {'ReEntHold':>9s} {'Combined':>8s}")
    print("  " + "-" * 85)

    for a in sorted(winner_analyses, key=lambda x: x["entry_date"]):
        re = a.get("reentry_basic")
        if re and re.get("trade_return") is not None:
            gap = re["days_after_exit"]
            combined = a["trade_return"] + re["trade_return"]
            print(f"  {a['ticker']:<7s} {a['exit_date']:<11s} {re['reentry_date']:<11s} {gap:>7d}d "
                  f"{re['reentry_price']:>7.2f} {re['trade_return']:>+7.1f}% "
                  f"{re.get('trade_hold_days', 0):>8d}d {combined:>+7.1f}%")
        else:
            print(f"  {a['ticker']:<7s} {a['exit_date']:<11s} {'N/A':>11s} {'N/A':>8s} "
                  f"{'N/A':>7s} {'N/A':>8s} {'N/A':>9s} {'N/A':>8s}")

    # Table 3: Loser max potential
    print(f"\n  TABLE 3: LOSER MAX POTENTIAL")
    print(f"  {'Ticker':<7s} {'TrdRet':>7s} {'Max60d$':>8s} {'MaxRet60d':>9s} {'DaysToMax':>9s}")
    print("  " + "-" * 50)

    for a in sorted(loser_analyses, key=lambda x: x["entry_date"]):
        max60 = a.get("max_price_60d_from_entry")
        max_ret = a.get("total_trend_return_60d")
        days_max = a.get("days_entry_to_max_60d")
        max60_s = f"{max60:>8.2f}" if max60 else "     N/A"
        max_ret_s = f"{max_ret:>+8.1f}%" if max_ret is not None else "     N/A"
        days_s = f"{days_max:>9d}" if days_max is not None else "      N/A"
        print(f"  {a['ticker']:<7s} {a['trade_return']:>+6.1f}% {max60_s} {max_ret_s} {days_s}")

    # Table 4: Aggregate stats
    print(f"\n  TABLE 4: AGGREGATE STATISTICS")
    print("  " + "-" * 60)

    if ult_days_list:
        print(f"    Avg total trend duration (entry to ult high):  {mean(ult_days_list):>6.1f} days")
    if total_rets:
        print(f"    Avg total available return (60d):               {mean(total_rets):>+6.1f}%")
    if efficiencies:
        print(f"    Avg capture efficiency:                         {mean(efficiencies):>6.1f}%")
    if reentry_returns:
        print(f"    Avg re-entry return:                            {mean(reentry_returns):>+6.1f}%")
    if combined_returns:
        print(f"    Avg combined return (trade + re-entry):         {mean(combined_returns):>+6.1f}%")
    print(f"    Viable re-entries (of {len(winner_analyses)} winners):             {viable_reentries}")
    if trade_rets:
        print(f"    Avg original trade return (winners only):       {mean(trade_rets):>+6.1f}%")

    # Left on the table
    if total_rets and trade_rets:
        left = mean(total_rets) - mean(trade_rets)
        print(f"    Avg return LEFT ON TABLE (60d avail - trade):   {left:>+6.1f}%")

    # Compare winners vs losers max potential
    if loser_max_rets and total_rets:
        print(f"\n    Winner avg max 60d return:  {mean(total_rets):>+6.1f}%")
        print(f"    Loser avg max 60d return:   {mean(loser_max_rets):>+6.1f}%")

    # Day-by-day post-exit path for each winner
    print_section_header("APPENDIX: DAY-BY-DAY POST-EXIT PATH (Winners, % return from ENTRY)")

    for a in sorted(winner_analyses, key=lambda x: x["entry_date"]):
        if not a.get("post_exit_available"):
            continue
        ticker = a["ticker"]
        entry_price = a["entry_price"]
        exit_price = a["exit_price"]
        peak = a["peak_close_during_trade"]

        print(f"\n  {ticker} | Entry: ${entry_price:.2f} | Exit: ${exit_price:.2f} ({a['exit_date']}) | "
              f"Trade: {a['trade_return']:+.1f}% | Peak during trade: ${peak:.2f}")

        post_prices = get_post_exit_prices(db, ticker, a["exit_date"], days=60)
        if not post_prices:
            print("    No post-exit data")
            continue

        # Print in rows of 10
        for start in range(0, min(len(post_prices), 60), 10):
            chunk = post_prices[start:start + 10]
            day_labels = "    "
            price_line = "    "
            ret_line = "    "
            marker_line = "    "

            for i, p in enumerate(chunk):
                day = start + i + 1
                close = p["close"]
                ret = (close - entry_price) / entry_price * 100
                day_labels += f"{'D' + str(day):>8s}"
                price_line += f"${close:>6.2f} "
                ret_line += f"{ret:>+7.1f}%"

                # Markers
                marker = " " * 8
                if close < entry_price:
                    marker = "  <ENTRY"
                elif close > peak:
                    marker = "NEWHIGH!"
                elif close >= exit_price:
                    marker = "  >EXIT "
                marker_line += f"{marker}"

            print(f"  Days: {day_labels}")
            print(f"  Close:{price_line}")
            print(f"  Ret%: {ret_line}")
            print(f"  Mark: {marker_line}")


if __name__ == "__main__":
    main()
