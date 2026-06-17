"""
Bearish trigger analysis:
1. Per-config short performance — which config is the best bearish signal?
2. Discretionary-rejected signals from current system — are those bearish?
3. Combined bearish universe
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

# Discretionary rejects from the current bullish system
DISCRETIONARY_REJECTS = {
    ("NRDS", "2025-05-12"),   # unfilled gap overhead
    ("U", "2025-05-13"),       # no trend break, sideways range
    ("ARRY", "2025-05-14"),    # unfilled gaps
    ("BILL", "2025-05-15"),    # downtrend, no reversal
    ("SPCE", "2025-07-25"),    # no trend change, still going down
    ("PLUG", "2026-01-22"),    # consolidation, not a trend change
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


def simulate_short(db, ticker, signal_date, atr_mult=0.5, max_days=15):
    """
    SHORT: sell at D0 close, cover using inverse stop logic.
    Day 1: cover if close > previous day's HIGH
    Day 2+: trailing cover: cover if close > lowest_close + ATR * mult
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

    entry_price = d0["close"]
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
            prev_high = prev_row["high"]
            if close > prev_high:
                ret = (entry_price - close) / entry_price * 100
                return {"return": ret, "hold_days": day_offset, "exit_reason": "stop_d1",
                        "entry_price": entry_price, "exit_price": close, "vol_ratio": sig_vol_ratio}
        else:
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


def simulate_long(db, ticker, signal_date, atr_mult=0.5, max_days=15):
    """Standard long simulation for reference."""
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
            "gg": gg, "gl": gl}


def main():
    db = Database(DB_PATH)
    db.initialize()

    # ===================================================================
    # Load all signals
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
    for (eps_thresh, tw) in CURRENT_CONFIGS:
        for s in all_signals.get((eps_thresh, tw), []):
            key = (s["ticker"], s["trend_date"])
            trade = simulate_long(db, s["ticker"], s["trend_date"])
            if trade and not trade.get("filtered"):
                current_keys.add(key)

    print(f"  Current bullish system: {len(current_keys)} trades")
    print(f"  Discretionary rejects: {len(DISCRETIONARY_REJECTS)}")

    # ===================================================================
    # PART 1: Per-config SHORT performance (NEW trades only, not in current system)
    # ===================================================================
    print(f"\n\n{'=' * 120}")
    print("  PART 1: SHORT PERFORMANCE BY CONFIG (new trades only, not in current bullish system)")
    print(f"{'=' * 120}")

    config_results = {}
    all_other_short_trades = []

    for eps_thresh, tw in OTHER_CONFIGS:
        seen = set()
        trades = []
        for s in all_signals.get((eps_thresh, tw), []):
            key = (s["ticker"], s["trend_date"])
            if key in seen or key in current_keys:
                continue
            seen.add(key)
            trade = simulate_short(db, s["ticker"], s["trend_date"])
            if not trade or trade.get("filtered"):
                continue
            trade["ticker"] = s["ticker"]
            trade["signal_date"] = s["trend_date"]
            trade["eps_pct"] = s.get("eps_change_pct", 0)
            trade["config"] = f"EPS>={eps_thresh:.0f}%/{tw}d"
            trades.append(trade)

        label = f"EPS>={eps_thresh:.0f}% / {tw}d"
        print(f"\n  --- {label} ---")
        print(f"  {'Ticker':7s} {'Signal':12s} {'EPS%':>6s} {'Vol':>5s} {'Short@':>8s} "
              f"{'Cover@':>8s} {'Return':>8s} {'Hold':>5s} {'Exit':10s} {'W/L':>4s}")
        print("  " + "-" * 90)
        for t in sorted(trades, key=lambda x: x["signal_date"]):
            wl = "W" if t["return"] > 0 else "L"
            print(f"  {t['ticker']:7s} {t['signal_date']:12s} {t['eps_pct']:>+5.0f}% "
                  f"{t['vol_ratio']:>4.1f}x ${t['entry_price']:>7.2f} ${t['exit_price']:>7.2f} "
                  f"{t['return']:>+7.1f}% {t['hold_days']:>4d}d {t['exit_reason']:10s} {wl:>3s}")

        stats = show_stats(f"SHORT: {label}", trades)
        config_results[(eps_thresh, tw)] = stats
        all_other_short_trades.extend(trades)

    # Dedup the combined other trades (some signals appear in multiple configs)
    deduped_other = {}
    for t in all_other_short_trades:
        key = (t["ticker"], t["signal_date"])
        if key not in deduped_other:
            deduped_other[key] = t
    all_other_deduped = list(deduped_other.values())

    # ===================================================================
    # PART 2: Config comparison table
    # ===================================================================
    print(f"\n\n{'=' * 120}")
    print("  PART 2: BEARISH CONFIG COMPARISON (which config is the best short signal?)")
    print(f"{'=' * 120}")

    print(f"\n  {'Config':20s} {'Trades':>7s} {'WR':>8s} {'AvgRet':>8s} {'PF':>8s} {'GG':>8s} {'GL':>8s}")
    print("  " + "-" * 70)
    for eps_thresh, tw in OTHER_CONFIGS:
        s = config_results.get((eps_thresh, tw), {})
        if not s:
            print(f"  EPS>={eps_thresh:.0f}%/{tw}d{'':12s} {'0':>7s} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s}")
            continue
        pf_s = f"{s['pf']:.2f}" if s['pf'] != float('inf') else "inf"
        print(f"  EPS>={eps_thresh:.0f}%/{tw}d{'':12s} {s['n']:>7d} {s['wr']:>7.1f}% {s['avg']:>+7.1f}% "
              f"{pf_s:>8s} {s['gg']:>+7.1f}% {s['gl']:>+7.1f}%")

    # ===================================================================
    # PART 3: Discretionary rejects as SHORT signals
    # ===================================================================
    print(f"\n\n{'=' * 120}")
    print("  PART 3: DISCRETIONARY REJECTS AS SHORTS")
    print("  (Bullish signals from current system that you'd skip based on chart review)")
    print(f"{'=' * 120}")

    reject_trades = []
    for ticker, sig_date in sorted(DISCRETIONARY_REJECTS):
        short_trade = simulate_short(db, ticker, sig_date)
        long_trade = simulate_long(db, ticker, sig_date)

        if short_trade and not short_trade.get("filtered"):
            short_trade["ticker"] = ticker
            short_trade["signal_date"] = sig_date
            long_ret = long_trade["return"] if long_trade and "return" in long_trade else None
            short_trade["long_return"] = long_ret
            reject_trades.append(short_trade)

    print(f"\n  {'Ticker':7s} {'Signal':12s} {'Reason':30s} {'Long Ret':>9s} {'Short Ret':>10s} "
          f"{'Hold':>5s} {'Exit':10s} {'W/L':>4s}")
    print("  " + "-" * 100)

    reasons = {
        "NRDS": "unfilled gap overhead",
        "U": "no trend break, sideways",
        "ARRY": "unfilled gaps",
        "BILL": "downtrend, no reversal",
        "SPCE": "no trend change, still down",
        "PLUG": "consolidation, not trend change",
    }

    for t in sorted(reject_trades, key=lambda x: x["signal_date"]):
        wl = "W" if t["return"] > 0 else "L"
        reason = reasons.get(t["ticker"], "")
        long_s = f"{t['long_return']:>+8.1f}%" if t["long_return"] is not None else "N/A"
        print(f"  {t['ticker']:7s} {t['signal_date']:12s} {reason:30s} {long_s:>9s} "
              f"{t['return']:>+9.1f}% {t['hold_days']:>4d}d {t['exit_reason']:10s} {wl:>3s}")

    show_stats("SHORT: Discretionary rejects from current system", reject_trades)

    # ===================================================================
    # PART 4: What if we also check — do "discretionary reject" patterns
    # appear in other config signals? (overlap analysis)
    # ===================================================================
    print(f"\n\n{'=' * 120}")
    print("  PART 4: OVERLAP — Are the discretionary reject tickers also in other config signals?")
    print(f"{'=' * 120}")

    reject_tickers = {t for t, d in DISCRETIONARY_REJECTS}
    for eps_thresh, tw in OTHER_CONFIGS:
        overlap = []
        for s in all_signals.get((eps_thresh, tw), []):
            if s["ticker"] in reject_tickers:
                overlap.append(s)
        if overlap:
            print(f"\n  EPS>={eps_thresh:.0f}%/{tw}d has {len(overlap)} signals for reject tickers:")
            for s in overlap:
                key = (s["ticker"], s["trend_date"])
                in_reject = "YES" if key in DISCRETIONARY_REJECTS else "no (different date)"
                print(f"    {s['ticker']:7s} {s['trend_date']:12s} EPS {s['eps_change_pct']:>+.0f}%  — exact match: {in_reject}")

    # ===================================================================
    # PART 5: COMBINED BEARISH UNIVERSE
    # All shorts: other config NEW trades + discretionary rejects (deduped)
    # ===================================================================
    print(f"\n\n{'=' * 120}")
    print("  PART 5: COMBINED BEARISH UNIVERSE")
    print("  Source A: NEW signals from other configs (not in current bullish system)")
    print("  Source B: Discretionary rejects from current bullish system")
    print(f"{'=' * 120}")

    combined_seen = set()
    combined_shorts = []

    # Source A: other config trades
    for t in all_other_deduped:
        key = (t["ticker"], t["signal_date"])
        if key not in combined_seen:
            combined_seen.add(key)
            t["source"] = "other_config"
            combined_shorts.append(t)

    # Source B: discretionary rejects
    for t in reject_trades:
        key = (t["ticker"], t["signal_date"])
        if key not in combined_seen:
            combined_seen.add(key)
            t["source"] = "disc_reject"
            combined_shorts.append(t)

    print(f"\n  {'Source':12s} {'Ticker':7s} {'Signal':12s} {'Vol':>5s} {'Short@':>8s} "
          f"{'Cover@':>8s} {'Return':>8s} {'Hold':>5s} {'Exit':10s} {'W/L':>4s}")
    print("  " + "-" * 100)
    for t in sorted(combined_shorts, key=lambda x: x["signal_date"]):
        wl = "W" if t["return"] > 0 else "L"
        src = "OtherCfg" if t["source"] == "other_config" else "DiscReject"
        print(f"  {src:12s} {t['ticker']:7s} {t['signal_date']:12s} "
              f"{t['vol_ratio']:>4.1f}x ${t['entry_price']:>7.2f} ${t['exit_price']:>7.2f} "
              f"{t['return']:>+7.1f}% {t['hold_days']:>4d}d {t['exit_reason']:10s} {wl:>3s}")

    show_stats("COMBINED BEARISH: Other configs + Discretionary rejects", combined_shorts)

    # Source breakdown
    src_a = [t for t in combined_shorts if t["source"] == "other_config"]
    src_b = [t for t in combined_shorts if t["source"] == "disc_reject"]
    show_stats("  Source A: Other config signals", src_a)
    show_stats("  Source B: Discretionary rejects", src_b)

    # ===================================================================
    # PART 6: FULL SYSTEM — Long bullish + Short bearish
    # ===================================================================
    print(f"\n\n{'=' * 120}")
    print("  PART 6: FULL SYSTEM — Long bullish (after disc filter) + Short bearish")
    print(f"{'=' * 120}")

    # Get filtered long trades
    long_trades_all = []
    long_seen2 = set()
    for eps_thresh, tw in CURRENT_CONFIGS:
        for s in all_signals.get((eps_thresh, tw), []):
            key = (s["ticker"], s["trend_date"])
            if key in long_seen2:
                continue
            long_seen2.add(key)
            if key in DISCRETIONARY_REJECTS:
                continue  # Skip rejects
            trade = simulate_long(db, s["ticker"], s["trend_date"])
            if not trade or trade.get("filtered"):
                continue
            trade["ticker"] = s["ticker"]
            trade["signal_date"] = s["trend_date"]
            trade["direction"] = "LONG"
            long_trades_all.append(trade)

    full_system = []
    for t in long_trades_all:
        full_system.append(t)
    for t in combined_shorts:
        t2 = dict(t)
        t2["direction"] = "SHORT"
        full_system.append(t2)

    print(f"\n  {'Dir':5s} {'Src':10s} {'Ticker':7s} {'Signal':12s} {'Vol':>5s} {'Entry$':>8s} "
          f"{'Exit$':>8s} {'Return':>8s} {'Hold':>5s} {'W/L':>4s}")
    print("  " + "-" * 85)
    for t in sorted(full_system, key=lambda x: x["signal_date"]):
        wl = "W" if t["return"] > 0 else "L"
        src = ""
        if t["direction"] == "LONG":
            src = "Bullish"
        elif t.get("source") == "disc_reject":
            src = "DiscReject"
        else:
            src = "OtherCfg"
        print(f"  {t['direction']:5s} {src:10s} {t['ticker']:7s} {t['signal_date']:12s} "
              f"{t['vol_ratio']:>4.1f}x ${t['entry_price']:>7.2f} ${t['exit_price']:>7.2f} "
              f"{t['return']:>+7.1f}% {t['hold_days']:>4d}d {wl:>3s}")

    show_stats("FULL SYSTEM: Long bullish + Short bearish", full_system)

    longs = [t for t in full_system if t["direction"] == "LONG"]
    shorts = [t for t in full_system if t["direction"] == "SHORT"]
    show_stats("  Long leg (after disc filter)", longs)
    show_stats("  Short leg (all bearish)", shorts)

    print(f"\n  Monthly frequency: {len(full_system)} trades / 10 months = {len(full_system)/10:.1f}/month")
    print(f"  Long: {len(longs)} trades ({len(longs)/10:.1f}/month)")
    print(f"  Short: {len(shorts)} trades ({len(shorts)/10:.1f}/month)")


if __name__ == "__main__":
    main()
