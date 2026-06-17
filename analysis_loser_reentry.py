"""
Deep dive on losers from the current system to find optimal re-entry after stop-out.

Current system: D0 close entry, vol >= 1.2x, combined EPS>=25%/30d and EPS>=30%/60d,
0.5x ATR trailing stop, 15d max hold.

For each loser, after the stop fires, track 60 trading days of price data and test
multiple re-entry strategies.
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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def get_db():
    db = Database(DB_PATH)
    db.initialize()
    return db


def compute_atr(prices_df, period=14):
    df = prices_df.copy()
    df["prev_close"] = df["close"].shift(1)
    df["tr"] = df.apply(lambda r: max(
        r["high"] - r["low"],
        abs(r["high"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
        abs(r["low"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
    ), axis=1)
    return df["tr"].rolling(period).mean()


def compute_rsi(closes, period=14):
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = pd.Series(gains).rolling(period).mean()
    avg_loss = pd.Series(losses).rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def filt_c17(signals):
    """C17 = Bull + PosEPS + Liq$500K + |EPS|<100 + Days>10."""
    return [s for s in signals
            if s["signal_type"] == "bullish"
            and s["eps_change_pct"] > 0
            and s.get("avg_dollar_vol", 0) >= 500_000
            and abs(s["eps_change_pct"]) < 100
            and s["days_between"] > 10]


def get_combined_signals(db):
    """Run both configs, apply C17 + vol>=1.2x filter, combine and dedupe."""
    all_signals = []

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
        c17 = filt_c17(enriched)
        # Volume filter: signal-day volume >= 1.2x 20d avg
        filtered = [s for s in c17 if s.get("vol_trend", -100) >= 20]  # vol_trend is % above baseline
        for s in filtered:
            s["_config"] = f"EPS>={eps_thresh}%/{trend_win}d"
        all_signals.extend(filtered)

    # Dedupe by (ticker, trend_date) - keep first occurrence
    seen = set()
    deduped = []
    for s in all_signals:
        key = (s["ticker"], s["trend_date"])
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    return deduped


def simulate_atr_trail(db, ticker, signal_date, entry_price, atr_mult=0.5, max_days=15):
    """Simulate trade with D0 close entry and 0.5x ATR trailing stop.

    Entry = D0 close (signal day close).
    Stop = trailing: highest_close - atr_mult * ATR(14).
    """
    signal_dt = date.fromisoformat(signal_date)
    fetch_start = (signal_dt - timedelta(days=30)).isoformat()
    fetch_end = (signal_dt + timedelta(days=int(max_days * 1.6) + 10)).isoformat()

    rows = db.get_daily_prices(ticker, fetch_start, fetch_end)
    if not rows:
        return None

    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    atr_series = compute_atr(df)

    signal_idx = df.index[df["date"] == signal_date].tolist()
    if not signal_idx:
        return None
    signal_idx = signal_idx[0]

    # Entry is D0 close (the signal day close)
    entry_price_actual = df.iloc[signal_idx]["close"]

    # Walk forward from the day AFTER signal
    highest_close = entry_price_actual

    for day_num in range(1, max_days + 1):
        cur_idx = signal_idx + day_num
        if cur_idx >= len(df):
            break

        row = df.iloc[cur_idx]
        close = row["close"]

        # ATR trailing stop
        atr_val = atr_series.iloc[cur_idx - 1] if cur_idx - 1 < len(atr_series) else atr_series.iloc[-1]
        if pd.isna(atr_val):
            atr_val = (df["high"].iloc[max(0, cur_idx - 14):cur_idx].mean()
                       - df["low"].iloc[max(0, cur_idx - 14):cur_idx].mean())

        stop_level = highest_close - atr_mult * atr_val
        if close < stop_level:
            ret = (close - entry_price_actual) / entry_price_actual * 100
            return {
                "entry_price": entry_price_actual,
                "exit_price": close,
                "exit_date": row["date"],
                "return": ret,
                "hold_days": day_num,
                "exit_reason": "atr_stop",
                "exit_day_idx": cur_idx,
            }

        if close > highest_close:
            highest_close = close

    # Held to horizon
    last_idx = min(signal_idx + max_days, len(df) - 1)
    exit_price = df.iloc[last_idx]["close"]
    ret = (exit_price - entry_price_actual) / entry_price_actual * 100
    return {
        "entry_price": entry_price_actual,
        "exit_price": exit_price,
        "exit_date": df.iloc[last_idx]["date"],
        "return": ret,
        "hold_days": max_days,
        "exit_reason": "horizon",
        "exit_day_idx": last_idx,
    }


def simulate_reentry_trade(post_df, reentry_idx, entry_price, atr_mult=0.5, max_days=15):
    """Simulate a re-entry trade from a given index in the post-exit dataframe.

    Uses 0.5x ATR trailing stop and 15d max hold from re-entry.
    """
    if reentry_idx >= len(post_df):
        return None

    re_entry_price = post_df.iloc[reentry_idx]["close"]
    highest_close = re_entry_price

    atr_series = compute_atr(post_df)

    for day_num in range(1, max_days + 1):
        cur_idx = reentry_idx + day_num
        if cur_idx >= len(post_df):
            break

        row = post_df.iloc[cur_idx]
        close = row["close"]

        atr_val = atr_series.iloc[cur_idx - 1] if cur_idx - 1 < len(atr_series) else np.nan
        if pd.isna(atr_val):
            # Fallback
            start = max(0, cur_idx - 14)
            atr_val = (post_df["high"].iloc[start:cur_idx].mean()
                       - post_df["low"].iloc[start:cur_idx].mean())

        stop_level = highest_close - atr_mult * atr_val
        if close < stop_level:
            ret = (close - re_entry_price) / re_entry_price * 100
            peak = (highest_close - re_entry_price) / re_entry_price * 100
            return {
                "reentry_price": re_entry_price,
                "exit_price": close,
                "exit_date": row["date"],
                "return": ret,
                "hold_days": day_num,
                "exit_reason": "atr_stop",
                "peak_return": peak,
            }

        if close > highest_close:
            highest_close = close

    # Held to max_days
    last_idx = min(reentry_idx + max_days, len(post_df) - 1)
    exit_price = post_df.iloc[last_idx]["close"]
    ret = (exit_price - re_entry_price) / re_entry_price * 100
    peak = (highest_close - re_entry_price) / re_entry_price * 100
    return {
        "reentry_price": re_entry_price,
        "exit_price": exit_price,
        "exit_date": post_df.iloc[last_idx]["date"],
        "return": ret,
        "hold_days": max_days,
        "exit_reason": "horizon",
        "peak_return": peak,
    }


def get_post_exit_data(db, ticker, signal_date, exit_day_idx_offset, days_after=60):
    """Get price data starting from the day after exit, for `days_after` trading days."""
    signal_dt = date.fromisoformat(signal_date)
    fetch_start = (signal_dt - timedelta(days=30)).isoformat()
    fetch_end = (signal_dt + timedelta(days=int((exit_day_idx_offset + days_after) * 1.6) + 30)).isoformat()

    rows = db.get_daily_prices(ticker, fetch_start, fetch_end)
    if not rows:
        return None, None

    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    signal_idx = df.index[df["date"] == signal_date].tolist()
    if not signal_idx:
        return None, None
    signal_idx = signal_idx[0]

    # exit_day_idx_offset is the number of trading days after signal that exit occurred
    exit_idx = signal_idx + exit_day_idx_offset
    if exit_idx >= len(df):
        return None, None

    # Post-exit data: start from exit day (inclusive, for ATR calc continuity)
    # but the "day 1 after exit" is exit_idx + 1
    post_start = max(0, exit_idx - 20)  # include lookback for ATR
    post_df = df.iloc[post_start:].reset_index(drop=True)

    # Index of exit day in post_df
    exit_in_post = exit_idx - post_start

    return post_df, exit_in_post


# ---------------------------------------------------------------------------
# main analysis
# ---------------------------------------------------------------------------

def main():
    db = get_db()

    print("=" * 120)
    print("  LOSER RE-ENTRY ANALYSIS")
    print("  System: D0 close entry, vol>=1.2x, EPS>=25%/30d + EPS>=30%/60d combined, 0.5x ATR trail, 15d max")
    print("=" * 120)

    # Get combined signals
    print("\nGathering signals from both configs...")
    combined = get_combined_signals(db)
    print(f"Combined deduped signals: {len(combined)}")

    # Simulate each trade with the current system
    losers = []
    winners = []

    for s in combined:
        trade = simulate_atr_trail(db, s["ticker"], s["trend_date"], s["entry_price"],
                                   atr_mult=0.5, max_days=15)
        if trade is None:
            continue

        s["trade"] = trade
        if trade["return"] <= 0:
            losers.append(s)
        else:
            winners.append(s)

    print(f"Winners: {len(winners)}, Losers: {len(losers)}")
    print(f"Win rate: {len(winners)/(len(winners)+len(losers))*100:.1f}%")

    # Show losers
    print(f"\nLOSERS (the focus of this analysis):")
    print(f"  {'Ticker':<8s} {'Signal Date':<12s} {'Config':<16s} {'Entry$':>8s} {'Exit$':>8s} "
          f"{'Return':>8s} {'Hold':>5s} {'ExitReason':<12s} {'EPS%':>7s}")
    print("  " + "-" * 95)
    for s in sorted(losers, key=lambda x: x["trend_date"]):
        t = s["trade"]
        print(f"  {s['ticker']:<8s} {s['trend_date']:<12s} {s['_config']:<16s} ${t['entry_price']:>7.2f} "
              f"${t['exit_price']:>7.2f} {t['return']:>+7.2f}% {t['hold_days']:>4d}d "
              f"{t['exit_reason']:<12s} {s['eps_change_pct']:>+6.1f}%")

    if not losers:
        print("\n  No losers found! Nothing to analyze for re-entry.")
        return

    # ======================================================================
    # SECTION 1: DAY-BY-DAY POST-EXIT PATH
    # ======================================================================
    print("\n" + "=" * 120)
    print("  SECTION 1: DAY-BY-DAY POST-EXIT PATH (60 trading days after exit)")
    print("=" * 120)

    loser_post_data = {}  # ticker -> (post_df, exit_in_post, entry_price, exit_price, exit_date)

    for s in sorted(losers, key=lambda x: x["trend_date"]):
        t = s["trade"]
        ticker = s["ticker"]
        entry_price = t["entry_price"]
        exit_price = t["exit_price"]
        exit_date = t["exit_date"]
        hold_days = t["hold_days"]

        post_df, exit_in_post = get_post_exit_data(
            db, ticker, s["trend_date"], hold_days, days_after=60
        )

        if post_df is None or exit_in_post is None:
            print(f"\n  {ticker} ({s['trend_date']}): insufficient post-exit data")
            continue

        loser_post_data[f"{ticker}_{s['trend_date']}"] = {
            "post_df": post_df,
            "exit_in_post": exit_in_post,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "exit_date": exit_date,
            "ticker": ticker,
            "signal_date": s["trend_date"],
            "original_return": t["return"],
        }

        print(f"\n  {ticker}  Signal={s['trend_date']}  Entry=${entry_price:.2f}  "
              f"Exit=${exit_price:.2f} ({t['return']:+.2f}%)  ExitDate={exit_date}")
        print(f"  {'Day':>4s} {'Date':<11s} {'Close':>8s} {'vs Entry':>9s} {'vs Exit':>9s} "
              f"{'Vol/20d':>8s} {'RSI(14)':>8s} {'Green?':>7s} {'NewHigh?':>9s}")
        print("  " + "-" * 80)

        # Compute 20d avg volume from pre-exit data
        pre_exit_vols = post_df["volume"].iloc[max(0, exit_in_post - 20):exit_in_post].tolist()
        avg_vol_20 = mean(pre_exit_vols) if pre_exit_vols else 1

        # Compute RSI over the full post_df
        all_closes = post_df["close"].values
        rsi_series = compute_rsi(all_closes, 14)

        post_exit_high = exit_price  # Track new highs since exit

        day_count = 0
        for i in range(exit_in_post + 1, len(post_df)):
            day_count += 1
            if day_count > 60:
                break

            row = post_df.iloc[i]
            close = row["close"]
            vs_entry = (close - entry_price) / entry_price * 100
            vs_exit = (close - exit_price) / exit_price * 100
            vol_ratio = row["volume"] / avg_vol_20 if avg_vol_20 > 0 else 0

            rsi_val = rsi_series.iloc[i - 1] if i - 1 < len(rsi_series) and not pd.isna(rsi_series.iloc[i - 1]) else 0

            green = "Yes" if close > row["open"] else "No"

            new_high = "No"
            if close > post_exit_high:
                post_exit_high = close
                new_high = "YES"

            print(f"  {day_count:>4d} {row['date']:<11s} ${close:>7.2f} {vs_entry:>+8.1f}% "
                  f"{vs_exit:>+8.1f}% {vol_ratio:>7.1f}x {rsi_val:>7.1f} {green:>7s} {new_high:>9s}")

    # ======================================================================
    # SECTION 2: ENTRY PRICE RECROSS ANALYSIS
    # ======================================================================
    print("\n" + "=" * 120)
    print("  SECTION 2: ENTRY PRICE RECROSS ANALYSIS")
    print("=" * 120)

    for key, data in loser_post_data.items():
        post_df = data["post_df"]
        exit_in_post = data["exit_in_post"]
        entry_price = data["entry_price"]
        exit_price = data["exit_price"]
        ticker = data["ticker"]

        print(f"\n  {ticker}  ({data['signal_date']})  Entry=${entry_price:.2f}  Exit=${exit_price:.2f}")

        # Track crosses above/below entry
        above_entry = False
        first_above_day = None
        first_above_date = None
        cross_above_count = 0
        cross_below_count = 0
        consecutive_above_runs = []
        current_above_run = 0
        max_gain_from_entry = -999
        max_gain_before_drop = []
        total_days = 0

        day_count = 0
        for i in range(exit_in_post + 1, len(post_df)):
            day_count += 1
            if day_count > 60:
                break
            total_days += 1

            close = post_df.iloc[i]["close"]
            now_above = close > entry_price
            gain_from_entry = (close - entry_price) / entry_price * 100

            if gain_from_entry > max_gain_from_entry:
                max_gain_from_entry = gain_from_entry

            if now_above and not above_entry:
                # Crossed above
                cross_above_count += 1
                if first_above_day is None:
                    first_above_day = day_count
                    first_above_date = post_df.iloc[i]["date"]
                current_above_run = 1
            elif now_above and above_entry:
                current_above_run += 1
            elif not now_above and above_entry:
                # Dropped below
                cross_below_count += 1
                consecutive_above_runs.append(current_above_run)
                max_gain_before_drop.append(max_gain_from_entry)
                current_above_run = 0

            above_entry = now_above

        # Close out any open run
        if above_entry and current_above_run > 0:
            consecutive_above_runs.append(current_above_run)

        chop_count = cross_above_count + cross_below_count

        if first_above_day is not None:
            print(f"    First close above entry: Day {first_above_day} ({first_above_date})")
            stayed_above = cross_below_count == 0 and above_entry
            print(f"    Stayed above entry after first cross? {'YES' if stayed_above else 'NO'}")
            print(f"    Crosses above entry: {cross_above_count}")
            print(f"    Crosses below entry (after being above): {cross_below_count}")
            print(f"    Chop count (total crosses): {chop_count}")
            if consecutive_above_runs:
                print(f"    Consecutive days above entry per run: {consecutive_above_runs}")
                print(f"    Avg consecutive days above: {mean(consecutive_above_runs):.1f}")
                print(f"    Max consecutive days above: {max(consecutive_above_runs)}")
            print(f"    Max gain from entry in 60d: {max_gain_from_entry:+.2f}%")
            if max_gain_before_drop:
                print(f"    Max gain before each drop below entry: {[f'{g:+.1f}%' for g in max_gain_before_drop]}")
        else:
            print(f"    NEVER crossed back above entry in 60 trading days")
            print(f"    Max gain from entry in 60d: {max_gain_from_entry:+.2f}%")

    # ======================================================================
    # SECTION 3: RE-ENTRY SIGNAL TESTING
    # ======================================================================
    print("\n" + "=" * 120)
    print("  SECTION 3: RE-ENTRY SIGNAL TESTING")
    print("  (Each re-entry uses 0.5x ATR trailing stop, 15d max hold from re-entry)")
    print("=" * 120)

    reentry_signals = {
        "A. Close > entry": {},
        "B. Close > entry + vol > 1.0x": {},
        "C. Close > entry + 2 consec days": {},
        "D. Close > entry + green candle": {},
        "E. Close > exit price": {},
        "F. Close > exit + vol > 1.0x": {},
        "G. 5-day high breakout": {},
        "H. Close > entry + RSI > 50": {},
        "I. Close > 10d SMA recross": {},
    }

    for key, data in loser_post_data.items():
        post_df = data["post_df"]
        exit_in_post = data["exit_in_post"]
        entry_price = data["entry_price"]
        exit_price = data["exit_price"]
        ticker = data["ticker"]

        # Pre-compute indicators
        all_closes = post_df["close"].values
        rsi_series = compute_rsi(all_closes, 14)

        # 10-day SMA
        sma_10 = post_df["close"].rolling(10).mean()

        # 20d avg volume (from pre-exit)
        pre_exit_vols = post_df["volume"].iloc[max(0, exit_in_post - 20):exit_in_post].tolist()
        avg_vol_20 = mean(pre_exit_vols) if pre_exit_vols else 1

        # Track state for signals
        consec_above_entry = 0
        five_day_highs = []
        was_below_entry = True  # start as below (we just got stopped)
        was_below_sma10 = True

        for day_num_offset in range(1, 61):
            i = exit_in_post + day_num_offset
            if i >= len(post_df):
                break

            row = post_df.iloc[i]
            close = row["close"]
            open_price = row["open"]
            volume = row["volume"]
            vol_ratio = volume / avg_vol_20 if avg_vol_20 > 0 else 0
            rsi_val = rsi_series.iloc[i - 1] if i - 1 < len(rsi_series) and not pd.isna(rsi_series.iloc[i - 1]) else 50
            sma_10_val = sma_10.iloc[i] if i < len(sma_10) and not pd.isna(sma_10.iloc[i]) else close
            green = close > open_price

            five_day_highs.append(close)
            if len(five_day_highs) > 5:
                five_day_highs.pop(0)

            above_entry = close > entry_price
            above_exit = close > exit_price

            if above_entry:
                consec_above_entry += 1
            else:
                consec_above_entry = 0

            # Check if we were below 10d SMA at some point
            if close < sma_10_val:
                was_below_sma10 = True

            # Signal A: Close above entry
            if "A. Close > entry" not in reentry_signals.get("_done_" + key, set()):
                if above_entry:
                    reentry_signals["A. Close > entry"][key] = {
                        "day": day_num_offset,
                        "reentry_idx": i,
                        "reentry_price": close,
                    }
                    reentry_signals.setdefault("_done_" + key, set()).add("A. Close > entry")

            # Signal B: Close above entry + volume > 1.0x
            if "B. Close > entry + vol > 1.0x" not in reentry_signals.get("_done_" + key, set()):
                if above_entry and vol_ratio >= 1.0:
                    reentry_signals["B. Close > entry + vol > 1.0x"][key] = {
                        "day": day_num_offset,
                        "reentry_idx": i,
                        "reentry_price": close,
                    }
                    reentry_signals.setdefault("_done_" + key, set()).add("B. Close > entry + vol > 1.0x")

            # Signal C: Close above entry + 2 consecutive days
            if "C. Close > entry + 2 consec days" not in reentry_signals.get("_done_" + key, set()):
                if consec_above_entry >= 2:
                    reentry_signals["C. Close > entry + 2 consec days"][key] = {
                        "day": day_num_offset,
                        "reentry_idx": i,
                        "reentry_price": close,
                    }
                    reentry_signals.setdefault("_done_" + key, set()).add("C. Close > entry + 2 consec days")

            # Signal D: Close above entry + green candle
            if "D. Close > entry + green candle" not in reentry_signals.get("_done_" + key, set()):
                if above_entry and green:
                    reentry_signals["D. Close > entry + green candle"][key] = {
                        "day": day_num_offset,
                        "reentry_idx": i,
                        "reentry_price": close,
                    }
                    reentry_signals.setdefault("_done_" + key, set()).add("D. Close > entry + green candle")

            # Signal E: Close above exit price
            if "E. Close > exit price" not in reentry_signals.get("_done_" + key, set()):
                if above_exit:
                    reentry_signals["E. Close > exit price"][key] = {
                        "day": day_num_offset,
                        "reentry_idx": i,
                        "reentry_price": close,
                    }
                    reentry_signals.setdefault("_done_" + key, set()).add("E. Close > exit price")

            # Signal F: Close above exit + volume > 1.0x
            if "F. Close > exit + vol > 1.0x" not in reentry_signals.get("_done_" + key, set()):
                if above_exit and vol_ratio >= 1.0:
                    reentry_signals["F. Close > exit + vol > 1.0x"][key] = {
                        "day": day_num_offset,
                        "reentry_idx": i,
                        "reentry_price": close,
                    }
                    reentry_signals.setdefault("_done_" + key, set()).add("F. Close > exit + vol > 1.0x")

            # Signal G: 5-day high breakout (after having been below entry)
            if "G. 5-day high breakout" not in reentry_signals.get("_done_" + key, set()):
                if len(five_day_highs) >= 5 and was_below_entry:
                    prev_high = max(five_day_highs[:-1])
                    if close > prev_high:
                        reentry_signals["G. 5-day high breakout"][key] = {
                            "day": day_num_offset,
                            "reentry_idx": i,
                            "reentry_price": close,
                        }
                        reentry_signals.setdefault("_done_" + key, set()).add("G. 5-day high breakout")

            # Signal H: Close above entry + RSI > 50
            if "H. Close > entry + RSI > 50" not in reentry_signals.get("_done_" + key, set()):
                if above_entry and rsi_val > 50:
                    reentry_signals["H. Close > entry + RSI > 50"][key] = {
                        "day": day_num_offset,
                        "reentry_idx": i,
                        "reentry_price": close,
                    }
                    reentry_signals.setdefault("_done_" + key, set()).add("H. Close > entry + RSI > 50")

            # Signal I: Close above 10d SMA after being below it
            if "I. Close > 10d SMA recross" not in reentry_signals.get("_done_" + key, set()):
                if was_below_sma10 and close > sma_10_val:
                    reentry_signals["I. Close > 10d SMA recross"][key] = {
                        "day": day_num_offset,
                        "reentry_idx": i,
                        "reentry_price": close,
                    }
                    reentry_signals.setdefault("_done_" + key, set()).add("I. Close > 10d SMA recross")

            # Update below-entry tracking
            if not above_entry:
                was_below_entry = True

    # Now simulate re-entry trades for each signal
    signal_names = [
        "A. Close > entry",
        "B. Close > entry + vol > 1.0x",
        "C. Close > entry + 2 consec days",
        "D. Close > entry + green candle",
        "E. Close > exit price",
        "F. Close > exit + vol > 1.0x",
        "G. 5-day high breakout",
        "H. Close > entry + RSI > 50",
        "I. Close > 10d SMA recross",
    ]

    all_reentry_results = {}

    for signal_name in signal_names:
        triggers = reentry_signals.get(signal_name, {})
        if not triggers:
            print(f"\n  {signal_name}")
            print(f"    Triggered: 0 / {len(loser_post_data)} losers")
            all_reentry_results[signal_name] = []
            continue

        print(f"\n  {signal_name}")
        print(f"    Triggered: {len(triggers)} / {len(loser_post_data)} losers")
        print(f"    {'Ticker':<8s} {'Days':>5s} {'ReEntry$':>9s} {'Exit$':>8s} {'Return':>8s} "
              f"{'Hold':>5s} {'ExitRsn':<10s} {'Peak':>7s} {'W/L':>4s} {'OrigLoss':>9s} {'Combined':>9s}")
        print("    " + "-" * 90)

        results = []
        for loser_key, trigger_info in triggers.items():
            data = loser_post_data[loser_key]
            post_df = data["post_df"]
            reentry_idx = trigger_info["reentry_idx"]

            trade = simulate_reentry_trade(post_df, reentry_idx, data["entry_price"],
                                           atr_mult=0.5, max_days=15)
            if trade is None:
                continue

            orig_loss = data["original_return"]
            combined = orig_loss + trade["return"]
            wl = "W" if trade["return"] > 0 else "L"

            print(f"    {data['ticker']:<8s} {trigger_info['day']:>5d} "
                  f"${trade['reentry_price']:>8.2f} ${trade['exit_price']:>7.2f} "
                  f"{trade['return']:>+7.2f}% {trade['hold_days']:>4d}d "
                  f"{trade['exit_reason']:<10s} {trade['peak_return']:>+6.2f}% {wl:>4s} "
                  f"{orig_loss:>+8.2f}% {combined:>+8.2f}%")

            results.append({
                "ticker": data["ticker"],
                "signal_date": data["signal_date"],
                "days_to_trigger": trigger_info["day"],
                "reentry_price": trade["reentry_price"],
                "exit_price": trade["exit_price"],
                "return": trade["return"],
                "hold_days": trade["hold_days"],
                "exit_reason": trade["exit_reason"],
                "peak_return": trade["peak_return"],
                "original_loss": orig_loss,
                "combined": orig_loss + trade["return"],
                "win": trade["return"] > 0,
            })

        all_reentry_results[signal_name] = results

        if results:
            wins = [r for r in results if r["win"]]
            wr = len(wins) / len(results) * 100
            avg_ret = mean([r["return"] for r in results])
            avg_days = mean([r["days_to_trigger"] for r in results])
            avg_combined = mean([r["combined"] for r in results])
            avg_peak = mean([r["peak_return"] for r in results])
            print(f"    ---")
            print(f"    WR: {wr:.0f}% ({len(wins)}/{len(results)})  "
                  f"Avg Return: {avg_ret:+.2f}%  Avg Days to Trigger: {avg_days:.1f}  "
                  f"Avg Peak: {avg_peak:+.2f}%  Avg Combined (orig+re): {avg_combined:+.2f}%")

    # ======================================================================
    # SECTION 4: CHOP ZONE ANALYSIS
    # ======================================================================
    print("\n" + "=" * 120)
    print("  SECTION 4: CHOP ZONE ANALYSIS")
    print("  (Price behavior around the entry price level, +/- 5%)")
    print("=" * 120)

    for key, data in loser_post_data.items():
        post_df = data["post_df"]
        exit_in_post = data["exit_in_post"]
        entry_price = data["entry_price"]
        ticker = data["ticker"]

        print(f"\n  {ticker}  ({data['signal_date']})  Entry=${entry_price:.2f}")

        chop_zone_upper = entry_price * 1.05
        chop_zone_lower = entry_price * 0.95

        days_in_chop = 0
        days_above_entry_in_chop = 0
        days_below_entry_in_chop = 0
        total_days = 0
        first_clean_breakout = None
        clean_breakout_day = None
        consec_above_chop = 0
        max_consec_above_chop = 0

        day_count = 0
        for i in range(exit_in_post + 1, len(post_df)):
            day_count += 1
            if day_count > 60:
                break
            total_days += 1

            close = post_df.iloc[i]["close"]
            in_chop = chop_zone_lower <= close <= chop_zone_upper
            above_chop = close > chop_zone_upper

            if in_chop:
                days_in_chop += 1
                if close >= entry_price:
                    days_above_entry_in_chop += 1
                else:
                    days_below_entry_in_chop += 1
                consec_above_chop = 0
            elif above_chop:
                consec_above_chop += 1
                if consec_above_chop > max_consec_above_chop:
                    max_consec_above_chop = consec_above_chop
                if consec_above_chop >= 3 and first_clean_breakout is None:
                    first_clean_breakout = close
                    clean_breakout_day = day_count
            else:
                consec_above_chop = 0

        if total_days == 0:
            print(f"    No post-exit data available")
            continue

        chop_pct = days_in_chop / total_days * 100
        above_pct = (days_above_entry_in_chop / days_in_chop * 100) if days_in_chop > 0 else 0
        below_pct = (days_below_entry_in_chop / days_in_chop * 100) if days_in_chop > 0 else 0

        print(f"    Days in chop zone (+/-5% of entry): {days_in_chop} / {total_days} ({chop_pct:.0f}%)")
        print(f"    Of chop zone days: {above_pct:.0f}% above entry, {below_pct:.0f}% below entry")

        if first_clean_breakout is not None:
            print(f"    Clean breakout (3+ days above +5%): Day {clean_breakout_day} at ${first_clean_breakout:.2f}")
            print(f"    Pattern: CLEAN BREAKOUT")
        elif max_consec_above_chop > 0:
            print(f"    Max consecutive days above +5%: {max_consec_above_chop}")
            if chop_pct > 50:
                print(f"    Pattern: MESSY / CHOPPY")
            else:
                print(f"    Pattern: MOSTLY BELOW / WEAK RECOVERY")
        else:
            if chop_pct > 30:
                print(f"    Pattern: STUCK IN CHOP ZONE")
            else:
                print(f"    Pattern: CONTINUED DECLINE / NO RECOVERY")

        # Stabilization analysis: look for decreasing volatility near entry
        closes_post = []
        for i in range(exit_in_post + 1, min(exit_in_post + 61, len(post_df))):
            closes_post.append(post_df.iloc[i]["close"])

        if len(closes_post) >= 20:
            # Compare first-half vs second-half volatility
            half = len(closes_post) // 2
            first_half_std = np.std(closes_post[:half])
            second_half_std = np.std(closes_post[half:])
            first_half_range = (max(closes_post[:half]) - min(closes_post[:half])) / entry_price * 100
            second_half_range = (max(closes_post[half:]) - min(closes_post[half:])) / entry_price * 100

            print(f"    Volatility (std): First 30d=${first_half_std:.2f}, Last 30d=${second_half_std:.2f} "
                  f"({'stabilizing' if second_half_std < first_half_std else 'still volatile'})")
            print(f"    Price range: First 30d={first_half_range:.1f}%, Last 30d={second_half_range:.1f}%")

    # ======================================================================
    # SECTION 5: SUMMARY
    # ======================================================================
    print("\n" + "=" * 120)
    print("  SECTION 5: SUMMARY AND RECOMMENDATIONS")
    print("=" * 120)

    print(f"\n  Total losers analyzed: {len(loser_post_data)}")

    # Rank re-entry signals
    print(f"\n  RE-ENTRY SIGNAL COMPARISON:")
    print(f"  {'Signal':<40s} {'Triggered':>10s} {'WR':>6s} {'AvgRet':>8s} {'AvgDays':>8s} "
          f"{'AvgPeak':>8s} {'AvgCombined':>12s}")
    print("  " + "-" * 100)

    best_signal = None
    best_combined = -999

    for signal_name in signal_names:
        results = all_reentry_results.get(signal_name, [])
        if not results:
            print(f"  {signal_name:<40s} {'0/' + str(len(loser_post_data)):>10s} "
                  f"{'N/A':>6s} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s} {'N/A':>12s}")
            continue

        wins = [r for r in results if r["win"]]
        wr = len(wins) / len(results) * 100
        avg_ret = mean([r["return"] for r in results])
        avg_days = mean([r["days_to_trigger"] for r in results])
        avg_peak = mean([r["peak_return"] for r in results])
        avg_combined = mean([r["combined"] for r in results])

        triggered = f"{len(results)}/{len(loser_post_data)}"
        print(f"  {signal_name:<40s} {triggered:>10s} {wr:>5.0f}% {avg_ret:>+7.2f}% "
              f"{avg_days:>7.1f}d {avg_peak:>+7.2f}% {avg_combined:>+11.2f}%")

        if avg_combined > best_combined:
            best_combined = avg_combined
            best_signal = signal_name

    # Summary statistics
    print(f"\n  BEST RE-ENTRY SIGNAL: {best_signal}")
    if best_signal:
        best_results = all_reentry_results.get(best_signal, [])
        if best_results:
            wins = [r for r in best_results if r["win"]]
            avg_ret = mean([r["return"] for r in best_results])
            avg_orig_loss = mean([r["original_loss"] for r in best_results])
            avg_combined = mean([r["combined"] for r in best_results])

            print(f"    Average original loss:     {avg_orig_loss:+.2f}%")
            print(f"    Average re-entry return:   {avg_ret:+.2f}%")
            print(f"    Average combined return:   {avg_combined:+.2f}%")
            print(f"    Re-entry win rate:         {len(wins)}/{len(best_results)} "
                  f"({len(wins)/len(best_results)*100:.0f}%)")

    # Overall assessment
    print(f"\n  PRACTICAL RECOMMENDATION:")

    all_losers_orig = [data["original_return"] for data in loser_post_data.values()]
    avg_orig_loss = mean(all_losers_orig) if all_losers_orig else 0

    # Count how many losers eventually recovered above entry
    recovered = 0
    for key, data in loser_post_data.items():
        post_df = data["post_df"]
        exit_in_post = data["exit_in_post"]
        entry_price = data["entry_price"]
        for i in range(exit_in_post + 1, min(exit_in_post + 61, len(post_df))):
            if post_df.iloc[i]["close"] > entry_price:
                recovered += 1
                break

    print(f"    Losers that recovered above entry within 60d: {recovered}/{len(loser_post_data)}")
    print(f"    Average original loss: {avg_orig_loss:+.2f}%")

    if best_signal and all_reentry_results.get(best_signal):
        best_results = all_reentry_results[best_signal]
        wins = [r for r in best_results if r["win"]]
        wr = len(wins) / len(best_results) * 100
        avg_combined = mean([r["combined"] for r in best_results])

        if wr >= 50 and avg_combined > avg_orig_loss * 0.5:
            print(f"    Verdict: RE-ENTRY MAY BE WORTH ATTEMPTING")
            print(f"    The {best_signal} signal recovers an average of "
                  f"{mean([r['return'] for r in best_results]):+.2f}% per re-entry trade.")
            print(f"    Combined (loss + re-entry), the average outcome improves from "
                  f"{avg_orig_loss:+.2f}% to {avg_combined:+.2f}%.")
        elif wr >= 40:
            print(f"    Verdict: MARGINAL - Re-entry partially offsets losses but is not reliable.")
            print(f"    Win rate is only {wr:.0f}% and average combined return is {avg_combined:+.2f}%.")
            print(f"    Consider using re-entry only with strongest confirmation signals.")
        else:
            print(f"    Verdict: NOT RECOMMENDED - Re-entry is too choppy and unreliable.")
            print(f"    Win rate is only {wr:.0f}% and average combined return is {avg_combined:+.2f}%.")
            print(f"    Better to accept the stop loss and move on to the next signal.")
    else:
        print(f"    Verdict: INSUFFICIENT DATA - Not enough re-entry triggers to draw conclusions.")
        print(f"    With only {len(loser_post_data)} losers, sample size is too small for reliable statistics.")

    print(f"\n  NOTE: Sample size is small ({len(loser_post_data)} losers). These results are suggestive, not definitive.")


if __name__ == "__main__":
    main()
