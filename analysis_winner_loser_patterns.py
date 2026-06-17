"""
Deep dive on the CURRENT SYSTEM's 12 trades.
Current system: D0 close entry, vol >= 1.2x, configs EPS>=25%/30d and EPS>=30%/60d
combined and deduped, 0.5x ATR trailing stop, 15d max hold.

Sections:
1. Winner hold time spread & day-by-day trajectory
2. Patterns before exit on winners
3. Post-exit continuation on winners
4. Loser reversal patterns
5. Summary table
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
# The two configs that make up the "current system"
# ---------------------------------------------------------------------------
CONFIGS = [(25.0, 30), (30.0, 60)]

# ---------------------------------------------------------------------------
# Helpers
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
    """Compute RSI from a list/array of close prices."""
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = pd.Series(gains).rolling(period).mean().iloc[-1]
    avg_loss = pd.Series(losses).rolling(period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_rsi_series(closes_series, period=14):
    """Compute RSI as a pandas Series from a close-price Series."""
    delta = closes_series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def gather_signals(db):
    """Run both configs, apply C17-like filters + vol>=1.2x, combine & dedup."""
    all_signals = []

    for eps_thresh, trend_window in CONFIGS:
        sc = ScannerConfig(
            min_price=1.0, max_price=50.0,
            min_market_cap=50_000_000, max_market_cap=10_000_000_000,
            ma_crossover_pairs=[(20, 50)],
            eps_change_threshold=eps_thresh,
            trend_window_days=trend_window,
            direction="both",
        )
        bc = BacktestConfig(
            start_date="2022-01-01", end_date="2026-03-19",
            forward_return_days=[5, 10, 15, 30, 60],
            ma_crossover_pairs=[(20, 50)],
            eps_thresholds=[eps_thresh],
            trend_windows=[trend_window],
        )
        result = Backtester(db, sc, bc).run()
        enriched = enrich_deep(db, result["signals"])
        # C17-like filter: bullish, positive EPS, dollar vol >= 500K, |EPS| < 100%, days > 10
        filtered = [s for s in enriched
                    if s["signal_type"] == "bullish"
                    and s["eps_change_pct"] > 0
                    and s.get("avg_dollar_vol", 0) >= 500_000
                    and abs(s["eps_change_pct"]) < 100
                    and s["days_between"] > 10]
        for s in filtered:
            s["_config"] = f"EPS>={eps_thresh}/{trend_window}d"
        all_signals.extend(filtered)

    # Dedup by (ticker, trend_date) -- keep the first occurrence
    seen = set()
    deduped = []
    for s in all_signals:
        key = (s["ticker"], s["trend_date"])
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    return deduped


def simulate_current_system(db, signal, max_days=15, atr_mult=0.5):
    """Simulate a single trade with CURRENT SYSTEM rules:
    - Entry: D0 close (signal day close)
    - Vol filter: signal day vol >= 1.2x 20d avg
    - Stop: 0.5x ATR trailing from highest close
    - Max hold: 15 trading days
    Returns dict with day-by-day details or None if filtered out.
    """
    ticker = signal["ticker"]
    trend_date = signal["trend_date"]
    trend_dt = date.fromisoformat(trend_date)
    fetch_start = (trend_dt - timedelta(days=60)).isoformat()
    fetch_end = (trend_dt + timedelta(days=int(max_days * 2.5) + 10)).isoformat()

    rows = db.get_daily_prices(ticker, fetch_start, fetch_end)
    if not rows:
        return None

    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df.reset_index(drop=True)

    # Find signal date index
    signal_idx_list = df.index[df["date"] == trend_date].tolist()
    if not signal_idx_list:
        return None
    signal_idx = signal_idx_list[0]

    # Volume filter: signal day vol >= 1.2x 20d avg
    if signal_idx < 20:
        return None
    vol_20d_avg = df["volume"].iloc[signal_idx - 20:signal_idx].mean()
    signal_vol = df["volume"].iloc[signal_idx]
    if vol_20d_avg <= 0 or signal_vol < 1.2 * vol_20d_avg:
        return None
    vol_ratio = signal_vol / vol_20d_avg

    # Entry = D0 close
    entry_price = df["close"].iloc[signal_idx]
    if entry_price <= 0:
        return None

    # Compute ATR series
    atr_series = compute_atr(df)

    # Walk forward from day after signal
    highest_close = entry_price
    daily_data = []
    exit_day = None
    exit_price = None
    exit_reason = None

    for day_num in range(1, max_days + 1):
        cur_idx = signal_idx + day_num
        if cur_idx >= len(df):
            break

        row = df.iloc[cur_idx]
        close = row["close"]
        day_open = row["open"]
        high = row["high"]
        low = row["low"]
        volume = row["volume"]
        trade_date = row["date"]

        # Volume relative to 20d avg (use the 20 days ending before this day)
        if cur_idx >= 20:
            day_vol_20avg = df["volume"].iloc[cur_idx - 20:cur_idx].mean()
        else:
            day_vol_20avg = vol_20d_avg
        day_rel_vol = volume / day_vol_20avg if day_vol_20avg > 0 else 1.0

        # Day return from entry
        day_return = (close - entry_price) / entry_price * 100

        # Green or red candle
        is_green = close >= day_open

        # ATR trailing stop check
        atr_val = atr_series.iloc[cur_idx - 1] if cur_idx - 1 < len(atr_series) and pd.notna(atr_series.iloc[cur_idx - 1]) else None
        if atr_val is None or pd.isna(atr_val):
            # Fallback
            lookback = df.iloc[max(0, cur_idx - 14):cur_idx]
            atr_val = (lookback["high"] - lookback["low"]).mean()

        stop_level = highest_close - atr_mult * atr_val
        stopped = close < stop_level

        # Gap from prior close
        prev_close = df["close"].iloc[cur_idx - 1]
        gap_pct = (day_open - prev_close) / prev_close * 100 if prev_close > 0 else 0

        daily_data.append({
            "day_num": day_num,
            "date": trade_date,
            "open": day_open,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "rel_vol": round(day_rel_vol, 2),
            "return_from_entry": round(day_return, 2),
            "is_green": is_green,
            "highest_close_so_far": round(highest_close, 4),
            "atr_val": round(atr_val, 4),
            "stop_level": round(stop_level, 4),
            "gap_pct": round(gap_pct, 2),
        })

        if stopped and exit_day is None:
            exit_day = day_num
            exit_price = close
            exit_reason = "atr_stop"
            # Record exit but still store this day's data
            break

        # Update trailing high
        if close > highest_close:
            highest_close = close

    # If no stop triggered, exit at horizon
    if exit_day is None and daily_data:
        last = daily_data[-1]
        exit_day = last["day_num"]
        exit_price = last["close"]
        exit_reason = "horizon"

    if exit_price is None:
        return None

    trade_return = (exit_price - entry_price) / entry_price * 100

    # Find peak day
    peak_return = max(d["return_from_entry"] for d in daily_data)
    peak_day = [d for d in daily_data if d["return_from_entry"] == peak_return][0]["day_num"]

    return {
        "ticker": ticker,
        "trend_date": trend_date,
        "entry_price": round(entry_price, 4),
        "exit_price": round(exit_price, 4),
        "exit_day": exit_day,
        "exit_reason": exit_reason,
        "trade_return": round(trade_return, 2),
        "is_winner": trade_return > 0,
        "peak_return": round(peak_return, 2),
        "peak_day": peak_day,
        "give_back": round(peak_return - trade_return, 2),
        "daily_data": daily_data,
        "vol_ratio_signal": round(vol_ratio, 2),
        "eps_change_pct": signal.get("eps_change_pct", 0),
        "signal": signal,
    }


def get_post_exit_data(db, ticker, exit_date, exit_price, n_days=30):
    """Fetch price data for n_days TRADING days after exit."""
    exit_dt = date.fromisoformat(exit_date)
    fetch_end = (exit_dt + timedelta(days=int(n_days * 2) + 10)).isoformat()
    rows = db.get_daily_prices(ticker, exit_date, fetch_end)
    if not rows:
        return None

    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df.reset_index(drop=True)

    # Find exit date index
    exit_idx_list = df.index[df["date"] == exit_date].tolist()
    if not exit_idx_list:
        return None
    exit_idx = exit_idx_list[0]

    post_data = []
    for i in range(1, n_days + 1):
        idx = exit_idx + i
        if idx >= len(df):
            break
        row = df.iloc[idx]
        ret = (row["close"] - exit_price) / exit_price * 100
        post_data.append({
            "day_after": i,
            "date": row["date"],
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row["volume"],
            "return_from_exit": round(ret, 2),
        })

    return post_data


def get_exit_day_context(db, ticker, trade_result):
    """Get RSI, volume, gap info around exit day."""
    trend_date = trade_result["trend_date"]
    trend_dt = date.fromisoformat(trend_date)
    fetch_start = (trend_dt - timedelta(days=100)).isoformat()
    exit_date = trade_result["daily_data"][-1]["date"]
    fetch_end_dt = date.fromisoformat(exit_date) + timedelta(days=5)

    rows = db.get_daily_prices(ticker, fetch_start, fetch_end_dt.isoformat())
    if not rows or len(rows) < 20:
        return {}

    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df.reset_index(drop=True)

    # RSI series
    rsi_series = compute_rsi_series(df["close"])

    # Find exit day and day before
    exit_idx_list = df.index[df["date"] == exit_date].tolist()
    if not exit_idx_list:
        return {}
    exit_idx = exit_idx_list[0]

    exit_rsi = rsi_series.iloc[exit_idx] if exit_idx < len(rsi_series) else None
    prev_rsi = rsi_series.iloc[exit_idx - 1] if exit_idx > 0 else None

    # Volume context
    vol_20avg = df["volume"].iloc[max(0, exit_idx - 20):exit_idx].mean() if exit_idx >= 20 else df["volume"].iloc[:exit_idx].mean()
    exit_vol = df["volume"].iloc[exit_idx]
    exit_vol_ratio = exit_vol / vol_20avg if vol_20avg > 0 else 1.0

    # Prior day volume context (look at last 3 days before exit)
    vol_trend = "flat"
    if exit_idx >= 3:
        vols = [df["volume"].iloc[exit_idx - 2], df["volume"].iloc[exit_idx - 1], df["volume"].iloc[exit_idx]]
        if vols[2] > vols[1] > vols[0]:
            vol_trend = "increasing"
        elif vols[2] < vols[1] < vols[0]:
            vol_trend = "decreasing"
        elif vols[2] > vols[0]:
            vol_trend = "rising"
        else:
            vol_trend = "declining"

    # Gap down on exit day
    prev_close = df["close"].iloc[exit_idx - 1] if exit_idx > 0 else None
    exit_open = df["open"].iloc[exit_idx]
    gap_down = False
    gap_pct = 0
    if prev_close and prev_close > 0:
        gap_pct = (exit_open - prev_close) / prev_close * 100
        gap_down = gap_pct < -0.5

    # Was stock making new highs the day before?
    making_new_highs = False
    if exit_idx >= 2:
        close_before_exit = df["close"].iloc[exit_idx - 1]
        close_2_before = df["close"].iloc[exit_idx - 2]
        making_new_highs = close_before_exit > close_2_before

    # Exit day return
    exit_close = df["close"].iloc[exit_idx]
    exit_open_val = df["open"].iloc[exit_idx]
    exit_day_return = (exit_close - prev_close) / prev_close * 100 if prev_close and prev_close > 0 else 0

    # ATR as % of price
    atr_df = df.iloc[max(0, exit_idx - 14):exit_idx + 1].copy()
    if len(atr_df) > 1:
        atr_df["prev_close"] = atr_df["close"].shift(1)
        atr_df["tr"] = atr_df.apply(lambda r: max(
            r["high"] - r["low"],
            abs(r["high"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
            abs(r["low"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
        ), axis=1)
        atr_val = atr_df["tr"].mean()
        atr_pct_of_price = atr_val / exit_close * 100 if exit_close > 0 else 0
    else:
        atr_pct_of_price = 0

    # How far from peak
    peak_close = max(d["close"] for d in trade_result["daily_data"])
    pct_below_peak = (peak_close - exit_close) / peak_close * 100 if peak_close > 0 else 0

    return {
        "exit_rsi": round(exit_rsi, 1) if exit_rsi and not pd.isna(exit_rsi) else None,
        "prev_rsi": round(prev_rsi, 1) if prev_rsi and not pd.isna(prev_rsi) else None,
        "exit_vol_ratio": round(exit_vol_ratio, 2),
        "vol_trend_3d": vol_trend,
        "gap_down": gap_down,
        "gap_pct": round(gap_pct, 2),
        "making_new_highs_before_exit": making_new_highs,
        "exit_day_return": round(exit_day_return, 2),
        "atr_pct_of_price": round(atr_pct_of_price, 2),
        "pct_below_peak": round(pct_below_peak, 2),
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    db = get_db()

    print("=" * 100)
    print("  CURRENT SYSTEM DEEP DIVE: Winner/Loser Pattern Analysis")
    print("  Rules: D0 close entry, vol >= 1.2x, EPS>=25%/30d + EPS>=30%/60d")
    print("         combined & deduped, 0.5x ATR trailing stop, 15d max hold")
    print("=" * 100)

    # Gather signals from both configs
    print("\nGathering signals from both configs...")
    deduped = gather_signals(db)
    print(f"  Deduped C17-filtered signals: {len(deduped)}")

    # Simulate each trade with current system rules
    print("Simulating trades with current system rules...")
    trades = []
    for s in deduped:
        result = simulate_current_system(db, s)
        if result is not None:
            trades.append(result)

    print(f"  Trades after vol>=1.2x filter: {len(trades)}")
    winners = [t for t in trades if t["is_winner"]]
    losers = [t for t in trades if not t["is_winner"]]
    print(f"  Winners: {len(winners)}, Losers: {len(losers)}")

    # Quick overview
    print(f"\n  {'Ticker':<8s} {'Date':<12s} {'Entry$':>8s} {'Exit$':>8s} {'Ret%':>7s} {'W/L':>4s} "
          f"{'Hold':>5s} {'Peak%':>7s} {'PkDay':>5s} {'GiveBack':>8s} {'Exit':>8s} {'VolRat':>7s} {'EPS%':>7s}")
    print("  " + "-" * 110)
    for t in sorted(trades, key=lambda x: x["trend_date"]):
        wl = "W" if t["is_winner"] else "L"
        print(f"  {t['ticker']:<8s} {t['trend_date']:<12s} ${t['entry_price']:>7.2f} ${t['exit_price']:>7.2f} "
              f"{t['trade_return']:>+6.1f}% {wl:>4s} {t['exit_day']:>4d}d {t['peak_return']:>+6.1f}% "
              f"{t['peak_day']:>4d}d {t['give_back']:>+7.1f}% {t['exit_reason']:>8s} {t['vol_ratio_signal']:>6.1f}x "
              f"{t['eps_change_pct']:>+6.1f}%")

    # ======================================================================
    # SECTION 1: Winner Hold Time Spread & Day-by-Day Trajectory
    # ======================================================================
    print("\n" + "=" * 100)
    print("  SECTION 1: WINNER HOLD TIME SPREAD & DAY-BY-DAY TRAJECTORY")
    print("=" * 100)

    if winners:
        sorted_winners = sorted(winners, key=lambda t: t["exit_day"])
        quickest = sorted_winners[0]
        longest = sorted_winners[-1]
        print(f"\n  QUICKEST winner: {quickest['ticker']} ({quickest['exit_day']}d hold, {quickest['trade_return']:+.1f}%)")
        print(f"  LONGEST winner:  {longest['ticker']} ({longest['exit_day']}d hold, {longest['trade_return']:+.1f}%)")
        print(f"  Average hold:    {mean([t['exit_day'] for t in winners]):.1f}d")
        print(f"  Median hold:     {median([t['exit_day'] for t in winners]):.1f}d")

    for t in sorted(winners, key=lambda x: x["trend_date"]):
        print(f"\n  --- {t['ticker']} (entered {t['trend_date']}) ---")
        print(f"      Entry: ${t['entry_price']:.2f}  |  Exit: ${t['exit_price']:.2f}  |  "
              f"Return: {t['trade_return']:+.2f}%  |  Hold: {t['exit_day']}d  |  "
              f"Peak: Day {t['peak_day']} ({t['peak_return']:+.2f}%)  |  "
              f"Stop fired: {'Day ' + str(t['exit_day']) if t['exit_reason'] == 'atr_stop' else 'N/A (horizon)'}")
        print(f"      {'Day':>5s} {'Date':>12s} {'Close':>8s} {'Ret%':>7s} {'RelVol':>7s} {'G/R':>4s} "
              f"{'Peak$':>8s} {'StopLvl':>8s} {'Gap%':>7s}")
        for d in t["daily_data"]:
            gr = "G" if d["is_green"] else "R"
            is_peak = " <-- PEAK" if d["day_num"] == t["peak_day"] else ""
            is_exit = " <-- EXIT" if d["day_num"] == t["exit_day"] else ""
            marker = is_peak + is_exit
            print(f"      {d['day_num']:>5d} {d['date']:>12s} ${d['close']:>7.2f} {d['return_from_entry']:>+6.1f}% "
                  f"{d['rel_vol']:>6.1f}x {gr:>4s} ${d['highest_close_so_far']:>7.2f} "
                  f"${d['stop_level']:>7.2f} {d['gap_pct']:>+6.1f}%{marker}")

    # ======================================================================
    # SECTION 2: Patterns Before Exit on Winners
    # ======================================================================
    print("\n" + "=" * 100)
    print("  SECTION 2: PATTERNS BEFORE EXIT ON WINNERS")
    print("=" * 100)

    exit_contexts = []
    for t in sorted(winners, key=lambda x: x["trend_date"]):
        ctx = get_exit_day_context(db, t["ticker"], t)
        ctx["ticker"] = t["ticker"]
        ctx["trade_return"] = t["trade_return"]
        ctx["exit_reason"] = t["exit_reason"]
        exit_contexts.append(ctx)

        print(f"\n  --- {t['ticker']} (return: {t['trade_return']:+.1f}%, exit: {t['exit_reason']}) ---")
        print(f"      Making new highs day before exit?  {ctx.get('making_new_highs_before_exit', 'N/A')}")
        print(f"      Volume trend (3d into exit):       {ctx.get('vol_trend_3d', 'N/A')}")
        print(f"      RSI on exit day:                   {ctx.get('exit_rsi', 'N/A')}")
        print(f"      RSI day before exit:               {ctx.get('prev_rsi', 'N/A')}")
        print(f"      Gap down on exit day?              {ctx.get('gap_down', 'N/A')} ({ctx.get('gap_pct', 0):+.2f}%)")
        print(f"      % below peak on exit:              {ctx.get('pct_below_peak', 0):.2f}%")
        print(f"      Exit day return:                   {ctx.get('exit_day_return', 0):+.2f}%")
        print(f"      Exit day vol vs 20d avg:           {ctx.get('exit_vol_ratio', 0):.2f}x")
        print(f"      ATR as % of price:                 {ctx.get('atr_pct_of_price', 0):.2f}%")

    # Small vs big winners
    small_winners = [c for c in exit_contexts if c["trade_return"] <= 10]
    big_winners = [c for c in exit_contexts if c["trade_return"] > 10]

    print(f"\n  --- SMALL WINNERS (<= 10% return): {len(small_winners)} trades ---")
    if small_winners:
        gap_down_pct = sum(1 for c in small_winners if c.get("gap_down")) / len(small_winners) * 100
        new_high_pct = sum(1 for c in small_winners if c.get("making_new_highs_before_exit")) / len(small_winners) * 100
        vol_trends = [c.get("vol_trend_3d", "?") for c in small_winners]
        avg_pct_below = mean([c.get("pct_below_peak", 0) for c in small_winners])
        avg_exit_ret = mean([c.get("exit_day_return", 0) for c in small_winners])
        avg_rsi = mean([c.get("exit_rsi", 50) for c in small_winners if c.get("exit_rsi")])
        print(f"      Gap down on exit:      {gap_down_pct:.0f}%")
        print(f"      Making new highs:      {new_high_pct:.0f}%")
        print(f"      Volume trend pattern:  {vol_trends}")
        print(f"      Avg % below peak:      {avg_pct_below:.2f}%")
        print(f"      Avg exit day return:   {avg_exit_ret:+.2f}%")
        print(f"      Avg exit RSI:          {avg_rsi:.1f}")
        for c in small_winners:
            print(f"        {c['ticker']:7s}  ret={c['trade_return']:+.1f}%  gap_down={c.get('gap_down')}  "
                  f"vol_trend={c.get('vol_trend_3d')}  rsi={c.get('exit_rsi')}  "
                  f"below_peak={c.get('pct_below_peak', 0):.1f}%")

    print(f"\n  --- BIG WINNERS (> 10% return): {len(big_winners)} trades ---")
    if big_winners:
        gap_down_pct = sum(1 for c in big_winners if c.get("gap_down")) / len(big_winners) * 100
        new_high_pct = sum(1 for c in big_winners if c.get("making_new_highs_before_exit")) / len(big_winners) * 100
        vol_trends = [c.get("vol_trend_3d", "?") for c in big_winners]
        avg_pct_below = mean([c.get("pct_below_peak", 0) for c in big_winners])
        avg_exit_ret = mean([c.get("exit_day_return", 0) for c in big_winners])
        avg_rsi = mean([c.get("exit_rsi", 50) for c in big_winners if c.get("exit_rsi")])
        print(f"      Gap down on exit:      {gap_down_pct:.0f}%")
        print(f"      Making new highs:      {new_high_pct:.0f}%")
        print(f"      Volume trend pattern:  {vol_trends}")
        print(f"      Avg % below peak:      {avg_pct_below:.2f}%")
        print(f"      Avg exit day return:   {avg_exit_ret:+.2f}%")
        print(f"      Avg exit RSI:          {avg_rsi:.1f}")
        for c in big_winners:
            print(f"        {c['ticker']:7s}  ret={c['trade_return']:+.1f}%  gap_down={c.get('gap_down')}  "
                  f"vol_trend={c.get('vol_trend_3d')}  rsi={c.get('exit_rsi')}  "
                  f"below_peak={c.get('pct_below_peak', 0):.1f}%")

    # ======================================================================
    # SECTION 3: Post-Exit Continuation on Winners
    # ======================================================================
    print("\n" + "=" * 100)
    print("  SECTION 3: POST-EXIT CONTINUATION ON WINNERS (30 trading days)")
    print("=" * 100)

    for t in sorted(winners, key=lambda x: x["trend_date"]):
        exit_date = t["daily_data"][-1]["date"]
        post = get_post_exit_data(db, t["ticker"], exit_date, t["exit_price"], n_days=30)
        if not post:
            print(f"\n  --- {t['ticker']}: No post-exit data available ---")
            continue

        # Compute post-exit metrics
        returns = [d["return_from_exit"] for d in post]
        max_gain = max(returns) if returns else 0
        max_dd = min(returns) if returns else 0

        # Returns at specific points
        ret_5d = post[4]["return_from_exit"] if len(post) >= 5 else None
        ret_10d = post[9]["return_from_exit"] if len(post) >= 10 else None
        ret_20d = post[19]["return_from_exit"] if len(post) >= 20 else None
        ret_30d = post[29]["return_from_exit"] if len(post) >= 30 else None

        # Did it make new high above trade peak?
        trade_peak = t["entry_price"] * (1 + t["peak_return"] / 100)
        new_high_above_peak = False
        new_high_day = None
        for d in post:
            if d["close"] > trade_peak:
                new_high_above_peak = True
                new_high_day = d["day_after"]
                break

        # Categorize
        if new_high_above_peak and new_high_day and new_high_day <= 10:
            category = "KEPT RUNNING"
        elif new_high_above_peak and new_high_day and new_high_day <= 30:
            category = "MODEST CONTINUATION"
        elif max_dd < -20:
            category = "COLLAPSED"
        elif max_dd < -10:
            category = "REVERSED"
        else:
            category = "TOPPED OUT"

        t["_post_exit"] = {
            "max_gain": max_gain,
            "max_dd": max_dd,
            "ret_5d": ret_5d,
            "ret_10d": ret_10d,
            "ret_20d": ret_20d,
            "ret_30d": ret_30d,
            "new_high_above_peak": new_high_above_peak,
            "new_high_day": new_high_day,
            "category": category,
        }

        print(f"\n  --- {t['ticker']} (trade return: {t['trade_return']:+.1f}%, exited {exit_date}) ---")
        print(f"      Category:                  {category}")
        print(f"      Max additional gain:       {max_gain:+.2f}%")
        print(f"      Max drawdown from exit:    {max_dd:+.2f}%")
        print(f"      Return  5d after exit:     {ret_5d:+.2f}%" if ret_5d is not None else "      Return  5d after exit:     N/A")
        print(f"      Return 10d after exit:     {ret_10d:+.2f}%" if ret_10d is not None else "      Return 10d after exit:     N/A")
        print(f"      Return 20d after exit:     {ret_20d:+.2f}%" if ret_20d is not None else "      Return 20d after exit:     N/A")
        print(f"      Return 30d after exit:     {ret_30d:+.2f}%" if ret_30d is not None else "      Return 30d after exit:     N/A")
        print(f"      New high above trade peak? {'Yes (day ' + str(new_high_day) + ')' if new_high_above_peak else 'No'}")

        # Day-by-day first 10d
        print(f"      First 10 days after exit:")
        print(f"      {'Day':>5s} {'Date':>12s} {'Close':>8s} {'Ret%':>7s}")
        for d in post[:10]:
            marker = ""
            if d["return_from_exit"] == max_gain:
                marker = " <-- MAX GAIN"
            if d["return_from_exit"] == max_dd and max_dd < 0:
                marker = " <-- MAX DD"
            print(f"      {d['day_after']:>5d} {d['date']:>12s} ${d['close']:>7.2f} {d['return_from_exit']:>+6.1f}%{marker}")

    # Commonalities analysis
    print(f"\n  --- COMMONALITIES: 'Kept running' / 'Modest continuation' vs 'Topped out' ---")
    kept_running = [t for t in winners if t.get("_post_exit", {}).get("category") in ("KEPT RUNNING", "MODEST CONTINUATION")]
    topped_out = [t for t in winners if t.get("_post_exit", {}).get("category") in ("TOPPED OUT", "REVERSED", "COLLAPSED")]

    for label, group in [("Kept Running / Modest Continuation", kept_running), ("Topped Out / Reversed / Collapsed", topped_out)]:
        print(f"\n      {label}: {len(group)} trades")
        if group:
            print(f"        Avg EPS change %:       {mean([t['eps_change_pct'] for t in group]):.1f}%")
            print(f"        Avg signal vol ratio:    {mean([t['vol_ratio_signal'] for t in group]):.2f}x")
            print(f"        Avg days to peak:        {mean([t['peak_day'] for t in group]):.1f}")
            print(f"        Avg entry price:         ${mean([t['entry_price'] for t in group]):.2f}")
            print(f"        Avg trade return:        {mean([t['trade_return'] for t in group]):+.1f}%")
            for t in group:
                pe = t.get("_post_exit", {})
                print(f"          {t['ticker']:7s}  ret={t['trade_return']:+.1f}%  eps={t['eps_change_pct']:+.1f}%  "
                      f"vol={t['vol_ratio_signal']:.1f}x  peak_day={t['peak_day']}  "
                      f"price=${t['entry_price']:.2f}  post_cat={pe.get('category', '?')}")

    # ======================================================================
    # SECTION 4: Loser Reversal Patterns
    # ======================================================================
    print("\n" + "=" * 100)
    print("  SECTION 4: LOSER REVERSAL PATTERNS (30 trading days post-exit)")
    print("=" * 100)

    loser_bottoms = []
    for t in sorted(losers, key=lambda x: x["trend_date"]):
        exit_date = t["daily_data"][-1]["date"]
        post = get_post_exit_data(db, t["ticker"], exit_date, t["exit_price"], n_days=30)
        if not post:
            print(f"\n  --- {t['ticker']}: No post-exit data available ---")
            continue

        returns = [d["return_from_exit"] for d in post]
        max_dd = min(returns) if returns else 0
        max_dd_day = [d for d in post if d["return_from_exit"] == max_dd][0]["day_after"] if returns else 0
        max_gain = max(returns) if returns else 0

        # When does it bottom?
        bottom_day = max_dd_day

        # When does it recover above exit price?
        recovery_day = None
        for d in post:
            if d["return_from_exit"] > 0:
                recovery_day = d["day_after"]
                break

        # Does it recover above entry price?
        entry_price = t["entry_price"]
        recovery_above_entry = None
        for d in post:
            if d["close"] > entry_price:
                recovery_above_entry = d["day_after"]
                break

        # Volume at bottom
        bottom_idx = bottom_day - 1
        if bottom_idx < len(post) and bottom_idx >= 0:
            bottom_vol = post[bottom_idx]["volume"]
            # Avg volume around bottom
            nearby_vols = [post[i]["volume"] for i in range(max(0, bottom_idx - 2), min(len(post), bottom_idx + 3))]
            avg_nearby_vol = mean(nearby_vols) if nearby_vols else 0
        else:
            bottom_vol = 0
            avg_nearby_vol = 0

        # RSI at bottom (need fresh fetch)
        trend_dt = date.fromisoformat(t["trend_date"])
        fetch_start = (trend_dt - timedelta(days=60)).isoformat()
        bottom_date = post[bottom_idx]["date"] if bottom_idx < len(post) else exit_date
        fetch_end = (date.fromisoformat(bottom_date) + timedelta(days=5)).isoformat()
        price_rows = db.get_daily_prices(t["ticker"], fetch_start, fetch_end)
        bottom_rsi = None
        if price_rows and len(price_rows) > 20:
            pdf = pd.DataFrame(price_rows)
            pdf["close"] = pdf["close"].astype(float)
            rsi_s = compute_rsi_series(pdf["close"])
            bidx = pdf.index[pdf["date"] == bottom_date].tolist()
            if bidx:
                bottom_rsi = rsi_s.iloc[bidx[0]]
                if pd.notna(bottom_rsi):
                    bottom_rsi = round(float(bottom_rsi), 1)
                else:
                    bottom_rsi = None

        # Is there volume spike at bottom?
        vol_spike_at_bottom = False
        if bottom_idx < len(post) and bottom_idx >= 0:
            # Get 20d avg vol before bottom
            exit_dt = date.fromisoformat(exit_date)
            pre_fetch = (exit_dt - timedelta(days=40)).isoformat()
            pre_rows = db.get_daily_prices(t["ticker"], pre_fetch, exit_date)
            if pre_rows and len(pre_rows) >= 20:
                pre_df = pd.DataFrame(pre_rows)
                pre_df["volume"] = pre_df["volume"].astype(float)
                avg_vol_20 = pre_df["volume"].tail(20).mean()
                if avg_vol_20 > 0 and bottom_vol > 1.5 * avg_vol_20:
                    vol_spike_at_bottom = True

        loser_info = {
            "ticker": t["ticker"],
            "trade_return": t["trade_return"],
            "exit_day": t["exit_day"],
            "max_dd": max_dd,
            "max_dd_day": max_dd_day,
            "max_gain_post": max_gain,
            "bottom_day": bottom_day,
            "recovery_day": recovery_day,
            "recovery_above_entry": recovery_above_entry,
            "bottom_rsi": bottom_rsi,
            "vol_spike_at_bottom": vol_spike_at_bottom,
        }
        loser_bottoms.append(loser_info)

        # Returns at specific points
        ret_5d = post[4]["return_from_exit"] if len(post) >= 5 else None
        ret_10d = post[9]["return_from_exit"] if len(post) >= 10 else None
        ret_20d = post[19]["return_from_exit"] if len(post) >= 20 else None
        ret_30d = post[29]["return_from_exit"] if len(post) >= 30 else None

        t["_post_exit"] = {
            "max_gain": max_gain,
            "max_dd": max_dd,
            "ret_5d": ret_5d,
            "ret_10d": ret_10d,
            "ret_20d": ret_20d,
            "ret_30d": ret_30d,
        }

        print(f"\n  --- {t['ticker']} (trade: {t['trade_return']:+.1f}%, exited day {t['exit_day']}, {exit_date}) ---")
        print(f"      Max drawdown from exit:    {max_dd:+.2f}% (day {max_dd_day})")
        print(f"      Max gain from exit:        {max_gain:+.2f}%")
        print(f"      Recovery above exit:       {'Day ' + str(recovery_day) if recovery_day else 'Never (in 30d)'}")
        print(f"      Recovery above entry:      {'Day ' + str(recovery_above_entry) if recovery_above_entry else 'Never (in 30d)'}")
        print(f"      RSI at bottom:             {bottom_rsi}")
        print(f"      Volume spike at bottom?    {vol_spike_at_bottom}")
        print(f"      Return  5d after exit:     {ret_5d:+.2f}%" if ret_5d is not None else "      Return  5d after exit:     N/A")
        print(f"      Return 10d after exit:     {ret_10d:+.2f}%" if ret_10d is not None else "      Return 10d after exit:     N/A")
        print(f"      Return 20d after exit:     {ret_20d:+.2f}%" if ret_20d is not None else "      Return 20d after exit:     N/A")
        print(f"      Return 30d after exit:     {ret_30d:+.2f}%" if ret_30d is not None else "      Return 30d after exit:     N/A")

        # Day-by-day for first 10 days
        print(f"      First 10 days after exit:")
        print(f"      {'Day':>5s} {'Date':>12s} {'Close':>8s} {'Ret%':>7s}")
        for d in post[:10]:
            marker = ""
            if d["return_from_exit"] == max_dd and max_dd < 0:
                marker = " <-- BOTTOM"
            if d["day_after"] == recovery_day:
                marker += " <-- RECOVERY"
            print(f"      {d['day_after']:>5d} {d['date']:>12s} ${d['close']:>7.2f} {d['return_from_exit']:>+6.1f}%{marker}")

    # Common reversal patterns
    print(f"\n  --- COMMON REVERSAL PATTERNS ---")
    if loser_bottoms:
        bottom_days = [lb["bottom_day"] for lb in loser_bottoms]
        recovery_days = [lb["recovery_day"] for lb in loser_bottoms if lb["recovery_day"] is not None]
        recovery_above_entry_days = [lb["recovery_above_entry"] for lb in loser_bottoms if lb["recovery_above_entry"] is not None]
        rsis_at_bottom = [lb["bottom_rsi"] for lb in loser_bottoms if lb["bottom_rsi"] is not None]
        vol_spikes = [lb["vol_spike_at_bottom"] for lb in loser_bottoms]

        print(f"      Days to bottom:                {bottom_days}  avg={mean(bottom_days):.1f}  med={median(bottom_days):.1f}")
        if recovery_days:
            print(f"      Days to recover above exit:    {recovery_days}  avg={mean(recovery_days):.1f}  med={median(recovery_days):.1f}")
        else:
            print(f"      Days to recover above exit:    NONE recovered")
        if recovery_above_entry_days:
            print(f"      Days to recover above entry:   {recovery_above_entry_days}  avg={mean(recovery_above_entry_days):.1f}")
        else:
            print(f"      Days to recover above entry:   NONE recovered above entry")
        if rsis_at_bottom:
            print(f"      RSI at bottom:                 {rsis_at_bottom}  avg={mean(rsis_at_bottom):.1f}")
        print(f"      Volume spike at bottom:        {sum(vol_spikes)}/{len(vol_spikes)} ({sum(vol_spikes)/len(vol_spikes)*100:.0f}%)")

        # Second entry opportunities
        print(f"\n      SECOND ENTRY OPPORTUNITIES:")
        for lb in loser_bottoms:
            if lb["recovery_day"] is not None:
                print(f"        {lb['ticker']}: Recovered above exit on day {lb['recovery_day']} "
                      f"(trade was {lb['trade_return']:+.1f}%, bottomed day {lb['bottom_day']} "
                      f"at {lb['max_dd']:+.1f}% from exit)")
                if lb["recovery_above_entry"] is not None:
                    print(f"                  Recovered above ENTRY on day {lb['recovery_above_entry']}")
            else:
                print(f"        {lb['ticker']}: Never recovered above exit in 30d "
                      f"(bottomed day {lb['bottom_day']} at {lb['max_dd']:+.1f}%)")

    # ======================================================================
    # SECTION 5: Summary Table
    # ======================================================================
    print("\n" + "=" * 140)
    print("  SECTION 5: COMPREHENSIVE SUMMARY TABLE")
    print("=" * 140)

    # Header
    print(f"\n  {'Ticker':<7s} {'Date':<11s} {'EPS%':>6s} {'SigVol':>6s} {'Entry$':>7s} {'Exit$':>7s} "
          f"{'Ret%':>6s} {'Hold':>4s} {'W/L':>3s} {'PeakR%':>6s} {'PkDay':>5s} {'GvBk%':>5s} "
          f"{'Post10d':>7s} {'Post30d':>7s} {'ExGap?':>6s} {'ExVol':>5s} {'ExRSI':>5s} "
          f"{'LsrBot':>6s} {'LsrDD':>6s} {'LsrRec':>6s}")
    print("  " + "-" * 138)

    for t in sorted(trades, key=lambda x: x["trend_date"]):
        wl = "W" if t["is_winner"] else "L"

        # Get exit context
        ctx = get_exit_day_context(db, t["ticker"], t)
        exit_gap = "Y" if ctx.get("gap_down") else "N"
        exit_vol = f"{ctx.get('exit_vol_ratio', 0):.1f}x"
        exit_rsi = f"{ctx.get('exit_rsi', 0):.0f}" if ctx.get("exit_rsi") else "?"

        # Post-exit data
        pe = t.get("_post_exit", {})
        post_10d = f"{pe.get('ret_10d', 0):+.1f}%" if pe.get("ret_10d") is not None else "N/A"
        post_30d = f"{pe.get('ret_30d', 0):+.1f}%" if pe.get("ret_30d") is not None else "N/A"

        # Loser-specific columns
        if not t["is_winner"]:
            lb = next((lb for lb in loser_bottoms if lb["ticker"] == t["ticker"]), None)
            if lb:
                bot_day = f"{lb['bottom_day']}d"
                max_dd = f"{lb['max_dd']:+.1f}%"
                rec = f"{lb['recovery_above_entry']}d" if lb["recovery_above_entry"] else "never"
            else:
                bot_day = "?"
                max_dd = "?"
                rec = "?"
        else:
            bot_day = "-"
            max_dd = "-"
            rec = "-"

        print(f"  {t['ticker']:<7s} {t['trend_date']:<11s} {t['eps_change_pct']:>+5.0f}% "
              f"{t['vol_ratio_signal']:>5.1f}x ${t['entry_price']:>6.2f} ${t['exit_price']:>6.2f} "
              f"{t['trade_return']:>+5.1f}% {t['exit_day']:>3d}d {wl:>3s} "
              f"{t['peak_return']:>+5.1f}% {t['peak_day']:>4d}d {t['give_back']:>+4.1f}% "
              f"{post_10d:>7s} {post_30d:>7s} {exit_gap:>6s} {exit_vol:>5s} {exit_rsi:>5s} "
              f"{bot_day:>6s} {max_dd:>6s} {rec:>6s}")

    # Final stats
    print(f"\n  --- AGGREGATE STATS ---")
    if trades:
        all_returns = [t["trade_return"] for t in trades]
        w_returns = [t["trade_return"] for t in winners]
        l_returns = [t["trade_return"] for t in losers]
        wr = len(winners) / len(trades) * 100
        gross_gains = sum(r for r in all_returns if r > 0)
        gross_losses = sum(r for r in all_returns if r < 0)
        pf = gross_gains / abs(gross_losses) if gross_losses else float("inf")
        pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"

        print(f"      Total trades:        {len(trades)}")
        print(f"      Win rate:            {wr:.1f}%")
        print(f"      Avg return:          {mean(all_returns):+.2f}%")
        print(f"      Avg winner:          {mean(w_returns):+.2f}%")
        print(f"      Avg loser:           {mean(l_returns):+.2f}%")
        print(f"      Profit factor:       {pf_str}")
        print(f"      Total gross gains:   {gross_gains:+.2f}%")
        print(f"      Total gross losses:  {gross_losses:+.2f}%")
        print(f"      Avg hold (all):      {mean([t['exit_day'] for t in trades]):.1f}d")
        print(f"      Avg hold (winners):  {mean([t['exit_day'] for t in winners]):.1f}d")
        print(f"      Avg hold (losers):   {mean([t['exit_day'] for t in losers]):.1f}d")
        print(f"      Avg peak return:     {mean([t['peak_return'] for t in trades]):+.2f}%")
        print(f"      Avg give-back:       {mean([t['give_back'] for t in trades]):.2f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
