"""Two-day confirmation rule:
  Day 1: close > open (green candle)
  Day 2: open > Day 1 close (gap up)
  Entry: Day 2 open

Questions to answer:
  1. Does this cost % on non-marginal winners?
  2. Do dropped stocks (fail gap-up) go on to move higher?
  3. Does this also filter out losers?
"""

import sys
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import DB_PATH, BacktestConfig, ScannerConfig
from core.backtest import Backtester
from core.database import Database
from analysis_winners_losers import enrich_deep


def compute_atr(df, period=14):
    d = df.copy()
    d["prev_close"] = d["close"].shift(1)
    d["tr"] = d.apply(lambda r: max(
        r["high"] - r["low"],
        abs(r["high"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
        abs(r["low"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
    ), axis=1)
    return d["tr"].rolling(period).mean()


def get_trade_data(db, ticker, signal_date, max_days=15):
    """Get all the price data we need for a signal."""
    signal_dt = date.fromisoformat(signal_date)
    fetch_start = (signal_dt - timedelta(days=50)).isoformat()
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

    d1_idx = sig_idx + 1
    d2_idx = sig_idx + 2
    if d1_idx >= len(df) or d2_idx >= len(df):
        return None

    return {
        "df": df, "atr_series": atr_series, "sig_idx": sig_idx,
        "d0": df.iloc[sig_idx], "d1": df.iloc[d1_idx], "d2": df.iloc[d2_idx],
        "d1_idx": d1_idx, "d2_idx": d2_idx,
    }


def simulate_from_entry(df, atr_series, entry_idx, entry_price, atr_mult=0.5, max_days=15):
    """Run ATR trailing stop from a given entry point."""
    highest_close = entry_price

    for day_offset in range(1, max_days + 1):
        cur_idx = entry_idx + day_offset
        if cur_idx >= len(df):
            break

        row = df.iloc[cur_idx]
        close = row["close"]

        # ATR trailing stop
        atr_val = atr_series.iloc[cur_idx - 1] if cur_idx - 1 < len(atr_series) else None
        if atr_val is None or pd.isna(atr_val):
            atr_val = (df["high"].iloc[max(0, cur_idx-14):cur_idx].mean() -
                       df["low"].iloc[max(0, cur_idx-14):cur_idx].mean())
        stop_level = highest_close - atr_mult * atr_val
        if close < stop_level:
            ret = (close - entry_price) / entry_price * 100
            return {"return": ret, "hold_days": day_offset, "exit_reason": "atr_stop",
                    "exit_price": close, "exit_date": row["date"]}

        if close > highest_close:
            highest_close = close

    last_idx = min(entry_idx + max_days, len(df) - 1)
    exit_price = df.iloc[last_idx]["close"]
    ret = (exit_price - entry_price) / entry_price * 100
    return {"return": ret, "hold_days": max_days, "exit_reason": "horizon",
            "exit_price": exit_price, "exit_date": df.iloc[last_idx]["date"]}


def forward_max_return(df, start_idx, days=10):
    """What's the max close reached in N days after start_idx?"""
    best = 0
    start_price = df.iloc[start_idx]["close"]
    for i in range(1, days + 1):
        idx = start_idx + i
        if idx >= len(df):
            break
        close = df.iloc[idx]["close"]
        ret = (close - start_price) / start_price * 100
        if ret > best:
            best = ret
    return best


def main():
    db = Database(DB_PATH)
    db.initialize()

    configs = [(10.0, 60), (20.0, 45), (20.0, 60), (25.0, 30), (25.0, 60), (30.0, 30), (30.0, 60)]

    signal_cache = {}
    for eps_thresh, tw in configs:
        sc = ScannerConfig(
            min_price=1.0, max_price=50.0, min_market_cap=50_000_000, max_market_cap=10_000_000_000,
            ma_crossover_pairs=[(20, 50)], eps_change_threshold=eps_thresh,
            trend_window_days=tw, direction="both",
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

    # Process all unique signals
    seen = set()
    all_signals = []

    for (eps_thresh, tw), (c17, raw) in sorted(signal_cache.items()):
        for s in c17:
            key = (s["ticker"], s["trend_date"])
            if key in seen:
                continue
            seen.add(key)

            data = get_trade_data(db, s["ticker"], s["trend_date"])
            if not data:
                continue

            d0, d1, d2 = data["d0"], data["d1"], data["d2"]
            df, atr_series = data["df"], data["atr_series"]

            green_candle = d1["close"] > d1["open"]
            gap_up = d2["open"] > d1["close"]
            d1_mid = (d1["high"] + d1["low"]) / 2

            # Simulate three entry scenarios
            # A) Original unrealistic: d1 midpoint, green candle only
            trade_a = None
            if green_candle:
                trade_a = simulate_from_entry(df, atr_series, data["d1_idx"], d1_mid)
                trade_a["entry_price"] = d1_mid
                trade_a["entry_date"] = d1["date"]

            # B) Two-day confirm: green candle + gap up, enter d2 open
            trade_b = None
            if green_candle and gap_up:
                trade_b = simulate_from_entry(df, atr_series, data["d2_idx"], d2["open"])
                trade_b["entry_price"] = d2["open"]
                trade_b["entry_date"] = d2["date"]

            # C) What happens to trades that PASS green candle but FAIL gap up?
            #    (entered at d1 midpoint unrealistically)
            trade_dropped = None
            if green_candle and not gap_up:
                trade_dropped = simulate_from_entry(df, atr_series, data["d1_idx"], d1_mid)
                trade_dropped["entry_price"] = d1_mid
                trade_dropped["entry_date"] = d1["date"]

            # D) Forward max return from d1 close (for dropped trades)
            fwd_max_10d = forward_max_return(df, data["d1_idx"], days=10)

            all_signals.append({
                "ticker": s["ticker"],
                "signal_date": s["trend_date"],
                "eps_pct": s.get("eps_change_pct", 0),
                "d1_open": d1["open"], "d1_close": d1["close"],
                "d1_mid": d1_mid,
                "d2_open": d2["open"],
                "green_candle": green_candle,
                "gap_up": gap_up,
                "gap_pct": (d2["open"] - d1["close"]) / d1["close"] * 100 if green_candle else None,
                "trade_a": trade_a,  # unrealistic green candle
                "trade_b": trade_b,  # two-day confirm
                "trade_dropped": trade_dropped,  # green candle pass, gap up fail
                "fwd_max_10d": fwd_max_10d if green_candle else None,
                "configs": f"EPS>={eps_thresh:.0f}%/{tw}d",
            })

    all_signals.sort(key=lambda x: (x["signal_date"], x["ticker"]))

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 1: Side-by-side comparison of original winners
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 180)
    print("  SECTION 1: GREEN CANDLE WINNERS — What happens with two-day confirmation?")
    print("  Original: d1 midpoint entry if green candle. New: d2 open entry if green candle + gap up.")
    print("=" * 180)
    print()
    print(f"  {'Ticker':7s} {'Date':12s} {'EPS':>5s} | {'D1 Close':>8s} {'D2 Open':>8s} {'Gap%':>6s} "
          f"{'Pass?':>5s} | {'Orig Ret':>8s} {'Orig W/L':>8s} | {'New Ret':>8s} {'New W/L':>8s} | {'Cost':>6s}")
    print("  " + "-" * 150)

    orig_winners = [s for s in all_signals if s["trade_a"] and s["trade_a"]["return"] > 0]

    passed_winners = []
    dropped_winners = []

    for s in orig_winners:
        orig = s["trade_a"]
        orig_wl = "W" if orig["return"] > 0 else "L"

        if s["gap_up"]:
            new = s["trade_b"]
            new_ret = f"{new['return']:>+7.1f}%"
            new_wl = "W" if new["return"] > 0 else "L"
            cost = orig["return"] - new["return"]
            cost_s = f"{cost:>+5.1f}pp"
            pass_s = "YES"
            passed_winners.append(s)
        else:
            new_ret = "  DROPPED"
            new_wl = " ---"
            cost_s = "  ---"
            pass_s = "NO"
            dropped_winners.append(s)

        print(f"  {s['ticker']:7s} {s['signal_date']:12s} {s['eps_pct']:>+4.0f}% | "
              f"${s['d1_close']:>7.2f} ${s['d2_open']:>7.2f} {s['gap_pct']:>+5.1f}% "
              f"{pass_s:>5s} | {orig['return']:>+7.1f}% {orig_wl:>8s} | "
              f"{new_ret:>8s} {new_wl:>8s} | {cost_s:>6s}")

    print()
    print(f"  Original winners: {len(orig_winners)}")
    print(f"  Pass two-day filter: {len(passed_winners)}")
    print(f"  Dropped by gap-up requirement: {len(dropped_winners)}")

    if passed_winners:
        orig_rets = [s["trade_a"]["return"] for s in passed_winners]
        new_rets = [s["trade_b"]["return"] for s in passed_winners]
        print(f"\n  PASSED winners avg cost of waiting: {mean(orig_rets) - mean(new_rets):+.1f}pp "
              f"(orig avg {mean(orig_rets):+.1f}% -> new avg {mean(new_rets):+.1f}%)")

        # Break down by size
        big_winners = [s for s in passed_winners if s["trade_a"]["return"] > 5]
        small_winners = [s for s in passed_winners if s["trade_a"]["return"] <= 5]

        if big_winners:
            orig_big = [s["trade_a"]["return"] for s in big_winners]
            new_big = [s["trade_b"]["return"] for s in big_winners]
            print(f"  Non-marginal winners (>5%): {len(big_winners)} trades, "
                  f"avg cost: {mean(orig_big) - mean(new_big):+.1f}pp "
                  f"(orig {mean(orig_big):+.1f}% -> new {mean(new_big):+.1f}%)")

        if small_winners:
            orig_sm = [s["trade_a"]["return"] for s in small_winners]
            new_sm = [s["trade_b"]["return"] for s in small_winners]
            flipped = sum(1 for s in small_winners if s["trade_b"]["return"] <= 0)
            print(f"  Marginal winners (<=5%): {len(small_winners)} trades, "
                  f"avg cost: {mean(orig_sm) - mean(new_sm):+.1f}pp "
                  f"(orig {mean(orig_sm):+.1f}% -> new {mean(new_sm):+.1f}%), "
                  f"{flipped} flipped to losses")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 2: What happens to DROPPED winners?
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*180}")
    print("  SECTION 2: DROPPED WINNERS — Do they go on to move higher?")
    print("  These passed green candle but failed gap up (D2 open <= D1 close)")
    print("=" * 180)
    print()
    print(f"  {'Ticker':7s} {'Date':12s} {'EPS':>5s} | {'D1 Close':>8s} {'D2 Open':>8s} {'Gap%':>6s} | "
          f"{'Orig Ret':>8s} {'W/L':>4s} {'Hold':>5s} | {'Max 10d':>8s} (from D1 close)")
    print("  " + "-" * 120)

    for s in dropped_winners:
        orig = s["trade_a"]
        wl = "W" if orig["return"] > 0 else "L"
        print(f"  {s['ticker']:7s} {s['signal_date']:12s} {s['eps_pct']:>+4.0f}% | "
              f"${s['d1_close']:>7.2f} ${s['d2_open']:>7.2f} {s['gap_pct']:>+5.1f}% | "
              f"{orig['return']:>+7.1f}% {wl:>4s} {orig['hold_days']:>4d}d | "
              f"{s['fwd_max_10d']:>+7.1f}%")

    if dropped_winners:
        orig_dropped_rets = [s["trade_a"]["return"] for s in dropped_winners]
        fwd_maxes = [s["fwd_max_10d"] for s in dropped_winners]
        print(f"\n  Dropped winners: {len(dropped_winners)} trades")
        print(f"  Their original (unrealistic) avg return: {mean(orig_dropped_rets):+.1f}%")
        print(f"  Their max forward 10d return from D1 close: avg {mean(fwd_maxes):+.1f}%, "
              f"max {max(fwd_maxes):+.1f}%, min {min(fwd_maxes):+.1f}%")
        big_movers = sum(1 for f in fwd_maxes if f > 10)
        print(f"  Moved >10% higher within 10d: {big_movers} of {len(dropped_winners)}")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 3: What about LOSERS? Does the filter avoid them?
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*180}")
    print("  SECTION 3: ORIGINAL LOSERS — Does the two-day filter avoid them?")
    print("  Original losers under green candle rule: how many pass/fail gap-up?")
    print("=" * 180)
    print()

    orig_losers = [s for s in all_signals if s["trade_a"] and s["trade_a"]["return"] <= 0]

    print(f"  {'Ticker':7s} {'Date':12s} {'EPS':>5s} | {'D1 Close':>8s} {'D2 Open':>8s} {'Gap%':>6s} "
          f"{'Pass?':>5s} | {'Orig Ret':>8s} | {'New Ret':>8s} {'New W/L':>8s}")
    print("  " + "-" * 120)

    loser_passed = []
    loser_dropped = []

    for s in orig_losers:
        orig = s["trade_a"]
        if s["gap_up"]:
            new = s["trade_b"]
            new_ret = f"{new['return']:>+7.1f}%"
            new_wl = "W" if new["return"] > 0 else "L"
            pass_s = "YES"
            loser_passed.append(s)
        else:
            new_ret = "  DROPPED"
            new_wl = " ---"
            pass_s = "NO"
            loser_dropped.append(s)

        print(f"  {s['ticker']:7s} {s['signal_date']:12s} {s['eps_pct']:>+4.0f}% | "
              f"${s['d1_close']:>7.2f} ${s['d2_open']:>7.2f} {s['gap_pct']:>+5.1f}% "
              f"{pass_s:>5s} | {orig['return']:>+7.1f}% | {new_ret:>8s} {new_wl:>8s}")

    print()
    print(f"  Original losers: {len(orig_losers)}")
    print(f"  Pass two-day filter (STILL ENTERED): {len(loser_passed)}")
    print(f"  Dropped by gap-up (AVOIDED): {len(loser_dropped)}")

    if loser_passed:
        orig_passed_rets = [s["trade_a"]["return"] for s in loser_passed]
        new_passed_rets = [s["trade_b"]["return"] for s in loser_passed]
        flipped_to_win = sum(1 for s in loser_passed if s["trade_b"]["return"] > 0)
        print(f"  Losers that still enter: avg orig ret {mean(orig_passed_rets):+.1f}%, "
              f"avg new ret {mean(new_passed_rets):+.1f}%, {flipped_to_win} flipped to wins")

    if loser_dropped:
        dropped_rets = [s["trade_a"]["return"] for s in loser_dropped]
        print(f"  Losers avoided: avg loss was {mean(dropped_rets):+.1f}%")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 4: Net effect — full system comparison
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*180}")
    print("  SECTION 4: FULL SYSTEM COMPARISON")
    print("=" * 180)
    print()

    # All green candle trades (original)
    gc_trades = [s for s in all_signals if s["trade_a"]]
    gc_rets = [s["trade_a"]["return"] for s in gc_trades]
    gc_wins = sum(1 for r in gc_rets if r > 0)

    # Two-day confirm trades
    td_trades = [s for s in all_signals if s["trade_b"]]
    td_rets = [s["trade_b"]["return"] for s in td_trades]
    td_wins = sum(1 for r in td_rets if r > 0)

    # No filter baseline (all C17 signals, d1 midpoint)
    all_c17 = [s for s in all_signals]
    baseline_trades = []
    for s in all_c17:
        data = get_trade_data(db, s["ticker"], s["signal_date"])
        if not data:
            continue
        d1_mid = (data["d1"]["high"] + data["d1"]["low"]) / 2
        trade = simulate_from_entry(data["df"], data["atr_series"], data["d1_idx"], d1_mid)
        if trade:
            baseline_trades.append(trade["return"])

    bl_wins = sum(1 for r in baseline_trades if r > 0)

    def calc_pf(rets):
        gg = sum(r for r in rets if r > 0)
        gl = sum(r for r in rets if r <= 0)
        return gg / abs(gl) if gl else float("inf")

    print(f"  {'System':50s} | {'Trades':>6s} | {'WR':>6s} | {'Avg Ret':>8s} | {'PF':>6s}")
    print("  " + "-" * 90)
    if baseline_trades:
        wr = bl_wins / len(baseline_trades) * 100
        pf = calc_pf(baseline_trades)
        print(f"  {'No filter (all C17, d1 midpoint)':50s} | {len(baseline_trades):>6d} | "
              f"{wr:>5.1f}% | {mean(baseline_trades):>+7.1f}% | {pf:>6.2f}")
    if gc_rets:
        wr = gc_wins / len(gc_rets) * 100
        pf = calc_pf(gc_rets)
        print(f"  {'Green candle only (d1 midpoint, UNREALISTIC)':50s} | {len(gc_rets):>6d} | "
              f"{wr:>5.1f}% | {mean(gc_rets):>+7.1f}% | {pf:>6.2f}")
    if td_rets:
        wr = td_wins / len(td_rets) * 100
        pf = calc_pf(td_rets)
        print(f"  {'Green candle + gap up (d2 open, REALISTIC)':50s} | {len(td_rets):>6d} | "
              f"{wr:>5.1f}% | {mean(td_rets):>+7.1f}% | {pf:>6.2f}")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 5: Signals that FAIL green candle — what happens to them?
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*180}")
    print("  SECTION 5: NON-GREEN-CANDLE SIGNALS — What are we missing?")
    print("  Signals where Day 1 closed red (close < open). Do any become big winners?")
    print("=" * 180)
    print()

    red_candle_signals = [s for s in all_signals if not s["green_candle"]]
    print(f"  {'Ticker':7s} {'Date':12s} {'EPS':>5s} | {'D1 Open':>8s} {'D1 Close':>8s} {'Chg%':>6s} | "
          f"{'10d Max':>8s}")
    print("  " + "-" * 80)

    red_fwd = []
    for s in red_candle_signals:
        data = get_trade_data(db, s["ticker"], s["signal_date"])
        if not data:
            continue
        fwd = forward_max_return(data["df"], data["d1_idx"], days=10)
        d1_chg = (data["d1"]["close"] - data["d1"]["open"]) / data["d1"]["open"] * 100
        red_fwd.append(fwd)
        flag = " <-- BIG MISS" if fwd > 15 else ""
        print(f"  {s['ticker']:7s} {s['signal_date']:12s} {s['eps_pct']:>+4.0f}% | "
              f"${data['d1']['open']:>7.2f} ${data['d1']['close']:>7.2f} {d1_chg:>+5.1f}% | "
              f"{fwd:>+7.1f}%{flag}")

    if red_fwd:
        print(f"\n  Red candle signals: {len(red_fwd)}")
        print(f"  Avg max 10d forward return: {mean(red_fwd):+.1f}%")
        big_misses = sum(1 for f in red_fwd if f > 15)
        print(f"  Big misses (>15% in 10d): {big_misses}")


if __name__ == "__main__":
    main()
