"""Test softer gap filters and oversold-bounce combinations.

Filters tested:
  Gap variants:
    A) D2 open > D1 close (strict, already tested)
    B) D2 open > D1 midpoint (softer)
    C) D2 open > D1 open (softest)
    D) D2 open > D1 VWAP proxy ((H+L+C)/3)

  Oversold bounce indicators (on Day 1):
    E) RSI(14) crossed above 30 in last 3 days (classic oversold exit)
    F) RSI(14) < 50 on signal day, > 50 on Day 1 (momentum shift)
    G) RSI(14) rising (Day 1 RSI > Day 0 RSI)
    H) Price above lower Bollinger Band (20,2) after being below within 5 days
    I) Stochastic %K > %D and %K was < 20 within 3 days

  Combinations:
    Green candle + soft gap + oversold bounce
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


def compute_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_bollinger(df, period=20, std_mult=2):
    sma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return sma, upper, lower


def compute_stochastic(df, k_period=14, d_period=3):
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def enrich_technicals(df):
    """Add RSI, Bollinger, Stochastic to dataframe."""
    df = df.copy()
    df["rsi"] = compute_rsi(df)
    df["bb_mid"], df["bb_upper"], df["bb_lower"] = compute_bollinger(df)
    df["stoch_k"], df["stoch_d"] = compute_stochastic(df)
    return df


def simulate_from_entry(df, atr_series, entry_idx, entry_price, atr_mult=0.75, max_days=15):
    """Run ATR trailing stop from entry."""
    highest_close = entry_price
    for day_offset in range(1, max_days + 1):
        cur_idx = entry_idx + day_offset
        if cur_idx >= len(df):
            break
        row = df.iloc[cur_idx]
        close = row["close"]

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


def get_signal_data(db, ticker, signal_date):
    """Get enriched price data around a signal."""
    signal_dt = date.fromisoformat(signal_date)
    fetch_start = (signal_dt - timedelta(days=60)).isoformat()
    fetch_end = (signal_dt + timedelta(days=40)).isoformat()

    rows = db.get_daily_prices(ticker, fetch_start, fetch_end)
    if not rows:
        return None

    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df = enrich_technicals(df)
    atr_series = compute_atr(df)

    sig_idx_list = df.index[df["date"] == signal_date].tolist()
    if not sig_idx_list:
        return None
    sig_idx = sig_idx_list[0]

    d1_idx = sig_idx + 1
    d2_idx = sig_idx + 2
    if d2_idx >= len(df):
        return None

    return {
        "df": df, "atr_series": atr_series, "sig_idx": sig_idx,
        "d0": df.iloc[sig_idx], "d1": df.iloc[d1_idx], "d2": df.iloc[d2_idx],
        "d1_idx": d1_idx, "d2_idx": d2_idx,
    }


def check_filters(data):
    """Check all filter conditions, return dict of booleans."""
    d0, d1, d2 = data["d0"], data["d1"], data["d2"]
    df = data["df"]
    sig_idx = data["sig_idx"]
    d1_idx = data["d1_idx"]

    d1_mid = (d1["high"] + d1["low"]) / 2
    d1_vwap = (d1["high"] + d1["low"] + d1["close"]) / 3

    filters = {}

    # Green candle
    filters["green_candle"] = d1["close"] > d1["open"]

    # Gap variants
    filters["gap_strict"] = d2["open"] > d1["close"]           # D2 open > D1 close
    filters["gap_mid"] = d2["open"] > d1_mid                   # D2 open > D1 midpoint
    filters["gap_open"] = d2["open"] > d1["open"]              # D2 open > D1 open
    filters["gap_vwap"] = d2["open"] > d1_vwap                 # D2 open > D1 VWAP

    # RSI conditions
    d0_rsi = d0.get("rsi", np.nan)
    d1_rsi = d1.get("rsi", np.nan)

    # RSI crossed above 30 recently (was below 30 in last 5 bars, now above)
    rsi_crossed_30 = False
    if pd.notna(d1_rsi) and d1_rsi > 30:
        for lookback in range(1, 6):
            idx = d1_idx - lookback
            if idx >= 0 and pd.notna(df.iloc[idx].get("rsi", np.nan)):
                if df.iloc[idx]["rsi"] < 30:
                    rsi_crossed_30 = True
                    break
    filters["rsi_cross_30"] = rsi_crossed_30

    # RSI shifted from <50 to >50
    filters["rsi_cross_50"] = (pd.notna(d0_rsi) and pd.notna(d1_rsi)
                                and d0_rsi < 50 and d1_rsi >= 50)

    # RSI rising
    filters["rsi_rising"] = (pd.notna(d0_rsi) and pd.notna(d1_rsi)
                              and d1_rsi > d0_rsi)

    # RSI was oversold (below 40) in last 5 days
    rsi_was_oversold = False
    for lookback in range(0, 6):
        idx = d1_idx - lookback
        if idx >= 0:
            r = df.iloc[idx].get("rsi", np.nan)
            if pd.notna(r) and r < 40:
                rsi_was_oversold = True
                break
    filters["rsi_was_oversold_5d"] = rsi_was_oversold

    # RSI < 50 on Day 1 (not overbought)
    filters["rsi_below_50"] = pd.notna(d1_rsi) and d1_rsi < 50

    # RSI between 30-60 on Day 1 (sweet spot: not overbought, recovering)
    filters["rsi_30_60"] = pd.notna(d1_rsi) and 30 <= d1_rsi <= 60

    # Bollinger band bounce: price was below lower band in last 5 days, now above
    bb_bounce = False
    d1_lower = d1.get("bb_lower", np.nan)
    if pd.notna(d1_lower) and d1["close"] > d1_lower:
        for lookback in range(1, 6):
            idx = d1_idx - lookback
            if idx >= 0:
                row = df.iloc[idx]
                lower = row.get("bb_lower", np.nan)
                if pd.notna(lower) and row["close"] < lower:
                    bb_bounce = True
                    break
    filters["bb_bounce"] = bb_bounce

    # Stochastic oversold bounce: %K was < 20 in last 3 days and now %K > %D
    stoch_bounce = False
    d1_k = d1.get("stoch_k", np.nan)
    d1_d = d1.get("stoch_d", np.nan)
    if pd.notna(d1_k) and pd.notna(d1_d) and d1_k > d1_d:
        for lookback in range(0, 4):
            idx = d1_idx - lookback
            if idx >= 0:
                k = df.iloc[idx].get("stoch_k", np.nan)
                if pd.notna(k) and k < 20:
                    stoch_bounce = True
                    break
    filters["stoch_bounce"] = stoch_bounce

    # Store raw values for display
    filters["_d1_rsi"] = d1_rsi
    filters["_d0_rsi"] = d0_rsi
    filters["_d1_stoch_k"] = d1.get("stoch_k", np.nan)
    filters["_d1_bb_lower"] = d1_lower
    filters["_d1_mid"] = d1_mid
    filters["_d1_vwap"] = d1_vwap

    return filters


def show(label, results):
    if not results:
        print(f"  {label:80s} | n=  0")
        return {}
    rets = [r["return"] for r in results]
    wins = [r for r in results if r["return"] > 0]
    losses = [r for r in results if r["return"] <= 0]
    wr = len(wins) / len(results) * 100
    avg = mean(rets)
    avg_w = mean([r["return"] for r in wins]) if wins else 0
    avg_l = mean([r["return"] for r in losses]) if losses else 0
    gg = sum(r["return"] for r in wins)
    gl = sum(r["return"] for r in losses)
    pf = gg / abs(gl) if gl else float("inf")
    pf_s = f"{pf:.2f}" if pf != float("inf") else "inf"
    hold = mean([r["hold_days"] for r in results])

    print(f"  {label:80s} | n={len(results):>3d} | WR={wr:>5.1f}% | Avg={avg:>+7.1f}% "
          f"| AvgW={avg_w:>+6.1f}% | AvgL={avg_l:>+6.1f}% | PF={pf_s:>6s} | Hold={hold:.1f}d")
    return {"n": len(results), "wr": wr, "avg": avg, "pf": pf}


def show_trades(results):
    for r in sorted(results, key=lambda x: x["signal_date"]):
        wl = "W" if r["return"] > 0 else "L"
        rsi_s = f"RSI={r.get('d1_rsi', 0):>4.0f}" if pd.notna(r.get("d1_rsi")) else "RSI=  ??"
        print(f"      {r['ticker']:7s} {r['signal_date']}  ${r['entry_price']:>7.2f}->${r['exit_price']:>7.2f}  "
              f"{r['return']:>+7.1f}%  {wl}  hold={r['hold_days']:>2d}d  {r['exit_reason']:12s}  "
              f"{rsi_s}  EPS={r.get('eps_pct',0):>+.0f}%")


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

    # Build enriched signal data
    seen = set()
    all_data = []

    for (eps_thresh, tw), (c17, raw) in sorted(signal_cache.items()):
        for s in c17:
            key = (s["ticker"], s["trend_date"])
            if key in seen:
                continue
            seen.add(key)

            data = get_signal_data(db, s["ticker"], s["trend_date"])
            if not data:
                continue

            filters = check_filters(data)
            all_data.append({
                "ticker": s["ticker"],
                "signal_date": s["trend_date"],
                "eps_pct": s.get("eps_change_pct", 0),
                "data": data,
                "filters": filters,
            })

    all_data.sort(key=lambda x: (x["signal_date"], x["ticker"]))

    def run_filter_combo(filter_fn, entry_mode, atr_mult=0.75):
        """Run a filter combo and return trade results."""
        results = []
        for item in all_data:
            if not filter_fn(item["filters"]):
                continue

            data = item["data"]
            d1, d2 = data["d1"], data["d2"]

            if entry_mode == "d1_mid":
                entry_price = (d1["high"] + d1["low"]) / 2
                entry_idx = data["d1_idx"]
            elif entry_mode == "d1_close":
                entry_price = d1["close"]
                entry_idx = data["d1_idx"]
            elif entry_mode == "d2_open":
                entry_price = d2["open"]
                entry_idx = data["d2_idx"]
            else:
                continue

            if entry_price <= 0:
                continue

            trade = simulate_from_entry(data["df"], data["atr_series"],
                                        entry_idx, entry_price, atr_mult=atr_mult)
            if trade:
                trade["entry_price"] = entry_price
                trade["ticker"] = item["ticker"]
                trade["signal_date"] = item["signal_date"]
                trade["eps_pct"] = item["eps_pct"]
                trade["d1_rsi"] = item["filters"]["_d1_rsi"]
                results.append(trade)
        return results

    # ═══════════════════════════════════════════════════════════════════
    print("=" * 180)
    print("  SOFT GAP FILTERS + OVERSOLD CONDITIONS (0.75x ATR stop)")
    print("=" * 180)

    # ── Baselines ──
    print("\n  --- BASELINES ---")
    r = run_filter_combo(lambda f: True, "d1_mid")
    show("No filter, d1 midpoint (unrealistic baseline)", r)

    r = run_filter_combo(lambda f: f["green_candle"], "d1_mid")
    show("Green candle, d1 midpoint (unrealistic)", r)

    r = run_filter_combo(lambda f: f["green_candle"], "d1_close")
    show("Green candle, d1 close (last hour)", r)

    # ── Gap variants (all with green candle, d2 open entry) ──
    print("\n  --- GAP FILTER VARIANTS (green candle + d2 open entry) ---")

    r = run_filter_combo(lambda f: f["green_candle"] and f["gap_strict"], "d2_open")
    show("Green + D2 open > D1 close (strict)", r)

    r = run_filter_combo(lambda f: f["green_candle"] and f["gap_mid"], "d2_open")
    show("Green + D2 open > D1 midpoint (softer)", r)

    r = run_filter_combo(lambda f: f["green_candle"] and f["gap_open"], "d2_open")
    show("Green + D2 open > D1 open (softest)", r)

    r = run_filter_combo(lambda f: f["green_candle"] and f["gap_vwap"], "d2_open")
    show("Green + D2 open > D1 VWAP (medium)", r)

    # ── Oversold bounce filters (green candle, d1 close entry) ──
    print("\n  --- OVERSOLD BOUNCE FILTERS (green candle + d1 close entry) ---")

    r = run_filter_combo(lambda f: f["green_candle"] and f["rsi_cross_30"], "d1_close")
    show("Green + RSI crossed above 30 in 5d", r)

    r = run_filter_combo(lambda f: f["green_candle"] and f["rsi_cross_50"], "d1_close")
    show("Green + RSI crossed from <50 to >=50", r)

    r = run_filter_combo(lambda f: f["green_candle"] and f["rsi_rising"], "d1_close")
    show("Green + RSI rising (D1 > D0)", r)

    r = run_filter_combo(lambda f: f["green_candle"] and f["rsi_was_oversold_5d"], "d1_close")
    show("Green + RSI was <40 in last 5 days", r)

    r = run_filter_combo(lambda f: f["green_candle"] and f["rsi_below_50"], "d1_close")
    show("Green + RSI < 50 on Day 1 (not overbought)", r)

    r = run_filter_combo(lambda f: f["green_candle"] and f["rsi_30_60"], "d1_close")
    show("Green + RSI 30-60 on Day 1 (sweet spot)", r)

    r = run_filter_combo(lambda f: f["green_candle"] and f["bb_bounce"], "d1_close")
    show("Green + Bollinger Band bounce (5d)", r)

    r = run_filter_combo(lambda f: f["green_candle"] and f["stoch_bounce"], "d1_close")
    show("Green + Stochastic bounce (%K>%D from <20)", r)

    # ── Oversold bounce WITHOUT green candle (catch red candle + oversold) ──
    print("\n  --- OVERSOLD BOUNCE WITHOUT GREEN CANDLE (d1 close entry) ---")

    r = run_filter_combo(lambda f: f["rsi_cross_30"], "d1_close")
    show("RSI crossed above 30 (no candle filter)", r)

    r = run_filter_combo(lambda f: f["rsi_was_oversold_5d"], "d1_close")
    show("RSI was <40 in 5d (no candle filter)", r)

    r = run_filter_combo(lambda f: f["bb_bounce"], "d1_close")
    show("Bollinger bounce (no candle filter)", r)

    r = run_filter_combo(lambda f: f["stoch_bounce"], "d1_close")
    show("Stochastic bounce (no candle filter)", r)

    r = run_filter_combo(lambda f: f["rsi_rising"] and f["rsi_was_oversold_5d"], "d1_close")
    show("RSI rising + was <40 in 5d (no candle filter)", r)

    # ── Best combos: soft gap + oversold ──
    print("\n  --- COMBINED: SOFT GAP + OVERSOLD (green candle + d2 open) ---")

    r = run_filter_combo(lambda f: f["green_candle"] and f["gap_open"] and f["rsi_rising"], "d2_open")
    show("Green + D2>D1open + RSI rising", r)

    r = run_filter_combo(lambda f: f["green_candle"] and f["gap_open"] and f["rsi_was_oversold_5d"], "d2_open")
    show("Green + D2>D1open + RSI was <40 in 5d", r)

    r = run_filter_combo(lambda f: f["green_candle"] and f["gap_mid"] and f["rsi_rising"], "d2_open")
    show("Green + D2>D1mid + RSI rising", r)

    r = run_filter_combo(lambda f: f["green_candle"] and f["gap_mid"] and f["rsi_was_oversold_5d"], "d2_open")
    show("Green + D2>D1mid + RSI was <40 in 5d", r)

    r = run_filter_combo(lambda f: f["green_candle"] and f["gap_vwap"] and f["rsi_rising"], "d2_open")
    show("Green + D2>D1vwap + RSI rising", r)

    r = run_filter_combo(lambda f: f["green_candle"] and f["gap_open"] and f["bb_bounce"], "d2_open")
    show("Green + D2>D1open + BB bounce", r)

    r = run_filter_combo(lambda f: f["green_candle"] and f["gap_open"] and f["stoch_bounce"], "d2_open")
    show("Green + D2>D1open + Stoch bounce", r)

    # ── Combined: soft gap + oversold with d1_close entry ──
    print("\n  --- COMBINED: OVERSOLD + D1 CLOSE ENTRY (no gap needed) ---")

    r = run_filter_combo(lambda f: f["green_candle"] and f["rsi_rising"] and f["rsi_was_oversold_5d"], "d1_close")
    show("Green + RSI rising + was <40 in 5d, d1 close", r)

    r = run_filter_combo(lambda f: f["green_candle"] and f["rsi_rising"] and f["rsi_below_50"], "d1_close")
    show("Green + RSI rising + RSI<50, d1 close", r)

    r = run_filter_combo(lambda f: f["green_candle"] and f["rsi_rising"] and f["rsi_30_60"], "d1_close")
    show("Green + RSI rising + RSI 30-60, d1 close", r)

    r = run_filter_combo(lambda f: f["green_candle"] and f["bb_bounce"] and f["rsi_rising"], "d1_close")
    show("Green + BB bounce + RSI rising, d1 close", r)

    r = run_filter_combo(lambda f: f["green_candle"] and f["stoch_bounce"] and f["rsi_rising"], "d1_close")
    show("Green + Stoch bounce + RSI rising, d1 close", r)

    r = run_filter_combo(lambda f: f["rsi_rising"] and f["rsi_was_oversold_5d"], "d1_close")
    show("RSI rising + was <40 in 5d (no green candle), d1 close", r)

    r = run_filter_combo(lambda f: f["rsi_cross_50"] and f["rsi_was_oversold_5d"], "d1_close")
    show("RSI cross 50 + was <40 in 5d (no green candle), d1 close", r)

    # ═══════════════════════════════════════════════════════════════════
    # Show individual trades for the most promising combos
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*180}")
    print("  INDIVIDUAL TRADES FOR BEST COMBOS")
    print(f"{'='*180}")

    best_combos = [
        ("Green + D2>D1open + RSI rising (d2 open)",
         lambda f: f["green_candle"] and f["gap_open"] and f["rsi_rising"], "d2_open"),
        ("Green + RSI rising + RSI<50 (d1 close)",
         lambda f: f["green_candle"] and f["rsi_rising"] and f["rsi_below_50"], "d1_close"),
        ("Green + RSI rising + was <40 in 5d (d1 close)",
         lambda f: f["green_candle"] and f["rsi_rising"] and f["rsi_was_oversold_5d"], "d1_close"),
        ("Green + D2>D1open (d2 open, softgap only)",
         lambda f: f["green_candle"] and f["gap_open"], "d2_open"),
        ("RSI rising + was <40 in 5d (no green, d1 close)",
         lambda f: f["rsi_rising"] and f["rsi_was_oversold_5d"], "d1_close"),
        ("Green + D2>D1mid + RSI rising (d2 open)",
         lambda f: f["green_candle"] and f["gap_mid"] and f["rsi_rising"], "d2_open"),
    ]

    for label, fn, entry in best_combos:
        r = run_filter_combo(fn, entry)
        print(f"\n  >> {label}")
        show(label, r)
        show_trades(r)

    # ═══════════════════════════════════════════════════════════════════
    # RSI distribution of winners vs losers
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*180}")
    print("  RSI ON DAY 1: WINNERS VS LOSERS (all C17, no entry filter)")
    print(f"{'='*180}")

    all_trades = run_filter_combo(lambda f: True, "d1_close")
    winners = [t for t in all_trades if t["return"] > 0 and pd.notna(t.get("d1_rsi"))]
    losers = [t for t in all_trades if t["return"] <= 0 and pd.notna(t.get("d1_rsi"))]

    if winners:
        w_rsi = [t["d1_rsi"] for t in winners]
        print(f"\n  Winners ({len(winners)}): RSI avg={mean(w_rsi):.1f}, "
              f"min={min(w_rsi):.1f}, max={max(w_rsi):.1f}")
        # Distribution
        for lo, hi in [(0,30), (30,40), (40,50), (50,60), (60,70), (70,100)]:
            cnt = sum(1 for r in w_rsi if lo <= r < hi)
            print(f"    RSI {lo:>3d}-{hi:<3d}: {cnt:>3d} ({cnt/len(w_rsi)*100:.0f}%)")

    if losers:
        l_rsi = [t["d1_rsi"] for t in losers]
        print(f"\n  Losers ({len(losers)}): RSI avg={mean(l_rsi):.1f}, "
              f"min={min(l_rsi):.1f}, max={max(l_rsi):.1f}")
        for lo, hi in [(0,30), (30,40), (40,50), (50,60), (60,70), (70,100)]:
            cnt = sum(1 for r in l_rsi if lo <= r < hi)
            print(f"    RSI {lo:>3d}-{hi:<3d}: {cnt:>3d} ({cnt/len(l_rsi)*100:.0f}%)")


if __name__ == "__main__":
    main()
