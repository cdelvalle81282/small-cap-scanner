"""
Chart structure feature analysis for EPS trend-change signals.

Computes technical features at the time of each MA crossover, then:
1. Shows correlation of each feature with 30-day outcome
2. Compares feature distributions between top-quartile winners and bottom-quartile losers
3. Tests single-feature filters to find which ones restore edge
4. Tests a composite filter combining the best features

Features computed at the crossover date:
  ma_spread_pct      - spread between 20 and 50 SMA as % of price (compression)
  ma_200_spread_pct  - spread between 50 and 200 SMA as % of price
  all_ma_tight       - max spread between 20/50/200 as % of price (overall compression)
  pct_above_52w_low  - % above 52-week low (near lows = lower for longs)
  pct_below_52w_high - % below 52-week high (off highs = lower for shorts)
  vol_ratio          - crossover day volume / 20-day avg volume
  atr_ratio          - ATR(5) / ATR(20) (< 1 = contracting range = consolidating)
  sma20_slope        - % change in 20 SMA over last 5 days
  price_vs_200       - % price is above/below 200 SMA
  range_10d_pct      - 10-day high-low range as % of price
  above_200          - bool: price above 200 SMA at crossover
"""

import sys
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import DB_PATH, ScannerConfig, BacktestConfig
from core.database import Database
from core.backtest import Backtester
from analysis_winners_losers import enrich_deep, filt_eps_trend
from analysis_exit_strategies import trade_result, get_price_df


# ── Feature computation ───────────────────────────────────────────────────────

def compute_features(db, sig):
    signal_date = sig["trend_date"]
    ticker      = sig["ticker"]
    direction   = sig["signal_type"]

    signal_dt = date.fromisoformat(signal_date)
    # Fetch 400 days before signal to have enough history for 200 SMA + 52w calculations
    start = (signal_dt - timedelta(days=400)).isoformat()
    end   = (signal_dt + timedelta(days=5)).isoformat()
    rows  = db.get_daily_prices(ticker, start, end)
    if not rows or len(rows) < 60:
        return None

    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Find crossover row
    cross_idx = df[df["date"] == pd.Timestamp(signal_date)].index
    if cross_idx.empty:
        # Use last row at or before signal date
        mask = df["date"] <= pd.Timestamp(signal_date)
        if not mask.any():
            return None
        cross_idx = [df[mask].index[-1]]
    ci = cross_idx[0]

    if ci < 50:
        return None  # not enough history

    window = df.iloc[:ci + 1]

    # SMAs
    sma20  = window["close"].rolling(20).mean().iloc[-1]
    sma50  = window["close"].rolling(50).mean().iloc[-1]
    sma200 = window["close"].rolling(200).mean().iloc[-1] if len(window) >= 200 else None
    price  = window["close"].iloc[-1]

    if price <= 0 or pd.isna(sma20) or pd.isna(sma50):
        return None

    # MA spread (compression metrics)
    ma_spread_pct = abs(sma50 - sma20) / price * 100

    if sma200 is not None and not pd.isna(sma200):
        ma_200_spread_pct = abs(sma200 - sma50) / price * 100
        all_ma_tight = (max(sma20, sma50, sma200) - min(sma20, sma50, sma200)) / price * 100
        price_vs_200 = (price - sma200) / sma200 * 100
        above_200    = price > sma200
    else:
        ma_200_spread_pct = None
        all_ma_tight      = ma_spread_pct
        price_vs_200      = None
        above_200         = None

    # 52-week levels (252 trading days back)
    lookback_52w = window.tail(252)
    high_52w = lookback_52w["high"].max()
    low_52w  = lookback_52w["low"].min()
    pct_above_52w_low  = (price - low_52w)  / low_52w  * 100 if low_52w  > 0 else None
    pct_below_52w_high = (high_52w - price) / high_52w * 100 if high_52w > 0 else None

    # Volume ratio: crossover day vs 20-day avg
    vol_today = window["volume"].iloc[-1]
    avg_vol_20 = window["volume"].tail(21).iloc[:-1].mean()
    vol_ratio = vol_today / avg_vol_20 if avg_vol_20 > 0 else None

    # ATR ratio: ATR(5) / ATR(20) — < 1 means range contracting
    def atr_series(w, n):
        d = w.copy()
        d["prev"] = d["close"].shift(1)
        d["tr"] = d.apply(lambda r: max(
            r["high"] - r["low"],
            abs(r["high"] - r["prev"]) if pd.notna(r["prev"]) else 0,
            abs(r["low"]  - r["prev"]) if pd.notna(r["prev"]) else 0,
        ), axis=1)
        return d["tr"].tail(n).mean()

    atr5  = atr_series(window, 5)
    atr20 = atr_series(window, 20)
    atr_ratio = atr5 / atr20 if atr20 > 0 else None

    # 20 SMA slope over last 5 days (% change)
    sma20_series = window["close"].rolling(20).mean()
    sma20_5_ago  = sma20_series.iloc[-6] if len(sma20_series) >= 6 else None
    sma20_now    = sma20_series.iloc[-1]
    sma20_slope  = (sma20_now - sma20_5_ago) / sma20_5_ago * 100 if (
        sma20_5_ago and not pd.isna(sma20_5_ago) and sma20_5_ago > 0
    ) else None

    # 10-day price range as % of price
    last10 = window.tail(10)
    range_10d_pct = (last10["high"].max() - last10["low"].min()) / price * 100

    return {
        "ma_spread_pct":      round(ma_spread_pct, 3),
        "ma_200_spread_pct":  round(ma_200_spread_pct, 3) if ma_200_spread_pct is not None else None,
        "all_ma_tight":       round(all_ma_tight, 3),
        "pct_above_52w_low":  round(pct_above_52w_low, 1)  if pct_above_52w_low  is not None else None,
        "pct_below_52w_high": round(pct_below_52w_high, 1) if pct_below_52w_high is not None else None,
        "vol_ratio":          round(vol_ratio, 2)  if vol_ratio  is not None else None,
        "atr_ratio":          round(atr_ratio, 3)  if atr_ratio  is not None else None,
        "sma20_slope":        round(sma20_slope, 3) if sma20_slope is not None else None,
        "price_vs_200":       round(price_vs_200, 1) if price_vs_200 is not None else None,
        "range_10d_pct":      round(range_10d_pct, 2),
        "above_200":          above_200,
    }


def get_30d_return(db, sig):
    df = get_price_df(db, sig["ticker"], sig["trend_date"])
    if df is None or len(df) < 10:
        return None
    signal_dt = pd.Timestamp(sig["trend_date"])
    after = df[df["date"] > signal_dt]
    if after.empty:
        return None
    entry_idx = after.index[0]
    ep = (df.iloc[entry_idx]["high"] + df.iloc[entry_idx]["low"]) / 2
    if ep <= 0:
        return None
    i30 = min(entry_idx + 30, len(df) - 1)
    return trade_result(ep, df.iloc[i30]["close"], sig["signal_type"])


# ── Stats helpers ─────────────────────────────────────────────────────────────

def pearson(x, y):
    xs = np.array(x, dtype=float)
    ys = np.array(y, dtype=float)
    mask = ~(np.isnan(xs) | np.isnan(ys))
    xs, ys = xs[mask], ys[mask]
    if len(xs) < 10:
        return float("nan"), len(xs)
    r = np.corrcoef(xs, ys)[0, 1]
    return round(r, 3), len(xs)


def group_stats(vals):
    if not vals:
        return {"n": 0}
    return {
        "n":      len(vals),
        "mean":   round(mean(vals), 2),
        "median": round(median(vals), 2),
        "p25":    round(float(np.percentile(vals, 25)), 2),
        "p75":    round(float(np.percentile(vals, 75)), 2),
    }


def filter_stats(signals_with_features, feature, threshold, op, ret_key="ret30"):
    subset = [s for s in signals_with_features
              if s.get(feature) is not None and s.get(ret_key) is not None
              and (s[feature] < threshold if op == "<" else s[feature] > threshold)]
    if not subset:
        return None
    returns = [s[ret_key] for s in subset]
    winners = [r for r in returns if r > 0]
    losers  = [r for r in returns if r <= 0]
    n = len(returns)
    wr = len(winners) / n * 100
    avg_w = mean(winners) if winners else 0
    avg_l = mean(losers)  if losers  else 0
    gross_l = abs(sum(r for r in losers if r < 0))
    pf = sum(winners) / gross_l if gross_l > 0 else float("inf")
    exp = (len(winners) / n) * avg_w + (len(losers) / n) * avg_l
    return {"n": n, "wr": round(wr, 1), "pf": round(pf, 2),
            "exp": round(exp, 2), "avg_w": round(avg_w, 1), "avg_l": round(avg_l, 1)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    db = Database(DB_PATH)
    sc = ScannerConfig()
    bc = BacktestConfig(start_date="2018-01-01", end_date="2026-06-09")

    print("Loading signals...")
    bt = Backtester(db, sc, bc)
    results = bt.run()
    enriched = enrich_deep(db, results["signals"])
    signals  = filt_eps_trend(enriched)
    print(f"Signals after liquidity filter: {len(signals)}")

    print("Computing chart features and 30d returns (this takes a few minutes)...")
    rows = []
    for i, sig in enumerate(signals):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(signals)}...")
        feats = compute_features(db, sig)
        ret30 = get_30d_return(db, sig)
        if feats is None or ret30 is None:
            continue
        rows.append({**sig, **feats, "ret30": ret30})

    print(f"Complete: {len(rows)} signals with features\n")

    # ── 1. Correlation table ──────────────────────────────────────────────────
    features = [
        "ma_spread_pct", "all_ma_tight", "ma_200_spread_pct",
        "pct_above_52w_low", "pct_below_52w_high",
        "vol_ratio", "atr_ratio", "sma20_slope",
        "price_vs_200", "range_10d_pct",
    ]

    print("=== Feature correlation with 30-day return ===")
    print(f"{'Feature':<22} {'Corr':>6}  {'n':>5}")
    print("-" * 38)
    correlations = {}
    for feat in features:
        x = [r[feat] for r in rows if r.get(feat) is not None]
        y = [r["ret30"] for r in rows if r.get(feat) is not None]
        r_val, n = pearson(x, y)
        correlations[feat] = r_val
        print(f"{feat:<22} {r_val:>6.3f}  {n:>5}")
    print()

    # ── 2. Winners vs losers quartiles ────────────────────────────────────────
    all_rets = sorted([r["ret30"] for r in rows])
    q1 = np.percentile(all_rets, 25)
    q3 = np.percentile(all_rets, 75)
    top_q  = [r for r in rows if r["ret30"] >= q3]
    bot_q  = [r for r in rows if r["ret30"] <= q1]
    print(f"=== Top quartile (>={q3:.1f}%) vs Bottom quartile (<={q1:.1f}%) ===")
    print(f"{'Feature':<22} {'Top-Q med':>10} {'Bot-Q med':>10} {'Diff':>8}")
    print("-" * 54)
    for feat in features:
        top_vals = [r[feat] for r in top_q if r.get(feat) is not None]
        bot_vals = [r[feat] for r in bot_q if r.get(feat) is not None]
        if not top_vals or not bot_vals:
            continue
        tm = median(top_vals)
        bm = median(bot_vals)
        print(f"{feat:<22} {tm:>10.2f} {bm:>10.2f} {tm-bm:>+8.2f}")
    print()

    # ── 3. Single-feature filter sweep ───────────────────────────────────────
    print("=== Single-feature filter performance (30d return, n>50) ===")
    print(f"{'Filter':<35} {'n':>5} {'WR%':>6} {'PF':>6} {'Exp%':>6} {'AvgW':>6} {'AvgL':>6}")
    print("-" * 70)

    # Baseline (no filter)
    all_rets_list = [r["ret30"] for r in rows]
    winners_all = [r for r in all_rets_list if r > 0]
    losers_all  = [r for r in all_rets_list if r <= 0]
    gross_l = abs(sum(r for r in losers_all if r < 0))
    pf_all  = sum(winners_all) / gross_l if gross_l > 0 else float("inf")
    exp_all = (len(winners_all)/len(all_rets_list))*mean(winners_all) + \
              (len(losers_all)/len(all_rets_list))*mean(losers_all)
    print(f"{'[baseline - no filter]':<35} {len(rows):>5} "
          f"{len(winners_all)/len(rows)*100:>6.1f} {pf_all:>6.2f} {exp_all:>6.2f} "
          f"{mean(winners_all):>6.1f} {mean(losers_all):>6.1f}")
    print()

    # Directional filters
    for direction in ["bullish", "bearish"]:
        subset = [r for r in rows if r["signal_type"] == direction]
        if len(subset) < 50:
            continue
        ret_s = [r["ret30"] for r in subset]
        w_s = [r for r in ret_s if r > 0]
        l_s = [r for r in ret_s if r <= 0]
        gl  = abs(sum(r for r in l_s if r < 0))
        pf_s = sum(w_s) / gl if gl > 0 else float("inf")
        exp_s = (len(w_s)/len(ret_s))*mean(w_s) + (len(l_s)/len(ret_s))*mean(l_s) if w_s and l_s else mean(ret_s)
        print(f"{'direction='+direction:<35} {len(subset):>5} "
              f"{len(w_s)/len(ret_s)*100:>6.1f} {pf_s:>6.2f} {exp_s:>6.2f} "
              f"{mean(w_s):>6.1f} {mean(l_s):>6.1f}")
    print()

    # Feature threshold filters
    sweep = [
        ("all_ma_tight",       "<",  5.0),
        ("all_ma_tight",       "<",  10.0),
        ("all_ma_tight",       "<",  15.0),
        ("ma_spread_pct",      "<",  3.0),
        ("ma_spread_pct",      "<",  5.0),
        ("pct_above_52w_low",  "<",  30.0),
        ("pct_above_52w_low",  "<",  50.0),
        ("pct_above_52w_low",  "<",  80.0),
        ("pct_below_52w_high", "<",  20.0),
        ("pct_below_52w_high", "<",  40.0),
        ("vol_ratio",          ">",  1.5),
        ("vol_ratio",          ">",  2.0),
        ("atr_ratio",          "<",  0.8),
        ("atr_ratio",          "<",  1.0),
        ("range_10d_pct",      "<",  15.0),
        ("range_10d_pct",      "<",  20.0),
        ("price_vs_200",       "<",  0.0),   # below 200 SMA
        ("price_vs_200",       "<",  20.0),  # within 20% of 200 SMA
        ("sma20_slope",        ">",  0.0),   # 20 SMA turning up
    ]

    for feat, op, thresh in sweep:
        s = filter_stats(rows, feat, thresh, op)
        if s is None or s["n"] < 50:
            continue
        label = f"{feat} {op} {thresh}"
        print(f"{label:<35} {s['n']:>5} {s['wr']:>6.1f} {s['pf']:>6.2f} "
              f"{s['exp']:>6.2f} {s['avg_w']:>6.1f} {s['avg_l']:>6.1f}")
    print()

    # ── 4. Composite filter ───────────────────────────────────────────────────
    # Based on what we see above, test combinations
    print("=== Composite filter tests ===")
    print(f"{'Filter combo':<45} {'n':>5} {'WR%':>6} {'PF':>6} {'Exp%':>6}")
    print("-" * 68)

    combos = [
        ("long + tight base",
         lambda r: r["signal_type"] == "bullish"
                   and (r.get("all_ma_tight") or 99) < 10
                   and (r.get("pct_above_52w_low") or 99) < 80),
        ("long + near lows + compressed",
         lambda r: r["signal_type"] == "bullish"
                   and (r.get("all_ma_tight") or 99) < 10
                   and (r.get("pct_above_52w_low") or 99) < 50),
        ("long + near lows + vol spike",
         lambda r: r["signal_type"] == "bullish"
                   and (r.get("pct_above_52w_low") or 99) < 80
                   and (r.get("vol_ratio") or 0) > 1.5),
        ("long + compressed + vol spike",
         lambda r: r["signal_type"] == "bullish"
                   and (r.get("all_ma_tight") or 99) < 10
                   and (r.get("vol_ratio") or 0) > 1.5),
        ("long + all 3: tight + near low + vol",
         lambda r: r["signal_type"] == "bullish"
                   and (r.get("all_ma_tight") or 99) < 15
                   and (r.get("pct_above_52w_low") or 99) < 80
                   and (r.get("vol_ratio") or 0) > 1.2),
        ("short + near high + compressed",
         lambda r: r["signal_type"] == "bearish"
                   and (r.get("pct_below_52w_high") or 99) < 30
                   and (r.get("all_ma_tight") or 99) < 15),
        ("short + near high + vol spike",
         lambda r: r["signal_type"] == "bearish"
                   and (r.get("pct_below_52w_high") or 99) < 30
                   and (r.get("vol_ratio") or 0) > 1.5),
    ]

    for label, fn in combos:
        subset = [r for r in rows if fn(r)]
        if len(subset) < 20:
            print(f"{label:<45} {len(subset):>5}  (too few)")
            continue
        ret_s = [r["ret30"] for r in subset]
        w_s = [r for r in ret_s if r > 0]
        l_s = [r for r in ret_s if r <= 0]
        gl  = abs(sum(r for r in l_s if r < 0))
        pf_s = sum(w_s) / gl if gl > 0 else float("inf")
        exp_s = (len(w_s)/len(ret_s))*mean(w_s) + (len(l_s)/len(ret_s))*mean(l_s) if w_s and l_s else mean(ret_s)
        print(f"{label:<45} {len(subset):>5} {len(w_s)/len(ret_s)*100:>6.1f} {pf_s:>6.2f} {exp_s:>6.2f}")


if __name__ == "__main__":
    main()
