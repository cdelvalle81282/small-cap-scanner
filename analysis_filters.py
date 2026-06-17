"""
Backtest filter analysis: test each research-based criterion individually,
then combine the best ones to find optimal rule sets.

Criteria tested:
 1. Baseline (no filters)
 2. Bullish-only
 3. Positive EPS only (eps_change > 0)
 4. EPS cap (|eps_change| < 100%)
 5. Liquidity filter (avg daily dollar volume >= $500K)
 6. Volume filter (avg daily volume >= 100K shares)
 7. Slippage modeling (0.20% >$5, 0.50% $1-$5)
 8. MA slope filter (slow MA slope > 0 for bullish)
 9. ATR confirmation (price breaks MA by >= 0.5x ATR)
10. Days-between filter (days_between > 10)
11. 52-week high proximity (within 30% of 52w high)
12. 20-day momentum filter (> 0%)
13. Longer horizons (test 30d and 60d vs 10d and 15d)
"""

import sys
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import DB_PATH, BacktestConfig, ScannerConfig
from core.backtest import Backtester
from core.database import Database


def get_db():
    db = Database(DB_PATH)
    db.initialize()
    return db


def run_baseline_backtest(db):
    """Run backtest with default params: 20/50 MA, 10% EPS, 30d window, both directions."""
    sc = ScannerConfig(
        min_price=1.0, max_price=50.0,
        min_market_cap=50_000_000, max_market_cap=10_000_000_000,
        ma_crossover_pairs=[(20, 50)],
        eps_change_threshold=10.0, trend_window_days=30, direction="both",
    )
    bc = BacktestConfig(
        start_date="2022-01-01", end_date="2026-03-19",
        forward_return_days=[5, 10, 15, 30, 60],
        ma_crossover_pairs=[(20, 50)],
        eps_thresholds=[10.0], trend_windows=[30],
    )
    bt = Backtester(db, sc, bc)
    return bt.run()


def compute_metrics(signals, horizon=15):
    """Compute standard metrics from a list of signal dicts."""
    returns = []
    for s in signals:
        fr = s.get("forward_returns", {})
        r = fr.get(horizon)
        if r is not None:
            returns.append(r)

    if not returns:
        return {
            "count": 0, "win_rate": 0, "avg_return": 0, "median_return": 0,
            "profit_factor": 0, "expectancy": 0, "max_gain": 0, "max_loss": 0,
            "avg_winner": 0, "avg_loser": 0, "win_count": 0, "loss_count": 0,
            "max_consec_losses": 0, "avg_trade_days": 0,
        }

    winners = [r for r in returns if r > 0]
    losers = [r for r in returns if r <= 0]
    gross_gains = sum(winners)
    gross_losses = sum(r for r in losers if r < 0)
    win_rate = len(winners) / len(returns) * 100
    avg_winner = mean(winners) if winners else 0
    avg_loser = mean(losers) if losers else 0
    pf = gross_gains / abs(gross_losses) if gross_losses != 0 else (float("inf") if gross_gains > 0 else 0)
    expectancy = (len(winners)/len(returns)) * avg_winner + (len(losers)/len(returns)) * avg_loser

    # Max consecutive losses
    max_consec = 0
    current_consec = 0
    for r in returns:
        if r <= 0:
            current_consec += 1
            max_consec = max(max_consec, current_consec)
        else:
            current_consec = 0

    # Average trade duration (days from entry to exit at this horizon)
    trade_days = []
    for s in signals:
        td = s.get("trade_details", {}).get(horizon, {})
        entry = s.get("entry_date")
        exit_d = td.get("exit_date")
        if entry and exit_d:
            try:
                d1 = date.fromisoformat(entry)
                d2 = date.fromisoformat(exit_d)
                trade_days.append((d2 - d1).days)
            except (ValueError, TypeError):
                pass

    return {
        "count": len(returns),
        "win_rate": round(win_rate, 1),
        "avg_return": round(mean(returns), 2),
        "median_return": round(sorted(returns)[len(returns)//2], 2),
        "profit_factor": round(pf, 2) if pf != float("inf") else "inf",
        "expectancy": round(expectancy, 2),
        "max_gain": round(max(returns), 2),
        "max_loss": round(min(returns), 2),
        "avg_winner": round(avg_winner, 2),
        "avg_loser": round(avg_loser, 2),
        "win_count": len(winners),
        "loss_count": len(losers),
        "max_consec_losses": max_consec,
        "avg_trade_days": round(mean(trade_days), 1) if trade_days else 0,
    }


def enrich_signals(db, signals):
    """Add computed features to each signal for filtering."""
    enriched = []
    for s in signals:
        ticker = s["ticker"]
        trend_date = s.get("trend_change_date")
        entry_date = s.get("entry_date")
        entry_price = s.get("entry_price")
        if not trend_date or not entry_date or not entry_price:
            continue

        trend_dt = date.fromisoformat(trend_date)
        # Fetch 260 trading days (~1 year) of history before signal
        fetch_start = (trend_dt - timedelta(days=400)).isoformat()
        fetch_end = trend_date
        prices = db.get_daily_prices(ticker, fetch_start, fetch_end)

        if len(prices) < 50:
            enriched.append({**s, "_skip": True})
            continue

        df = pd.DataFrame(prices)
        df["close"] = df["close"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["volume"] = df["volume"].astype(float)

        # --- Liquidity: avg daily dollar volume (20d) ---
        last_20 = df.tail(20)
        avg_vol_20 = last_20["volume"].mean()
        avg_price_20 = last_20["close"].mean()
        avg_dollar_vol = avg_vol_20 * avg_price_20

        # --- ATR (14-day) ---
        df["prev_close"] = df["close"].shift(1)
        df["tr"] = df.apply(lambda r: max(
            r["high"] - r["low"],
            abs(r["high"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
            abs(r["low"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
        ), axis=1)
        atr_14 = df["tr"].tail(14).mean()
        atr_pct = (atr_14 / entry_price * 100) if entry_price > 0 else 0

        # --- MA slope (slow MA = 50-day SMA) ---
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        last_row = df.iloc[-1]
        sma_50_now = last_row.get("sma_50")
        sma_50_5ago = df["sma_50"].iloc[-6] if len(df) >= 6 else None
        if sma_50_now and sma_50_5ago and sma_50_5ago != 0:
            sma50_slope_pct = (sma_50_now - sma_50_5ago) / sma_50_5ago * 100
        else:
            sma50_slope_pct = None

        sma_20_now = last_row.get("sma_20")
        sma_20_5ago = df["sma_20"].iloc[-6] if len(df) >= 6 else None
        if sma_20_now and sma_20_5ago and sma_20_5ago != 0:
            sma20_slope_pct = (sma_20_now - sma_20_5ago) / sma_20_5ago * 100
        else:
            sma20_slope_pct = None

        # --- ATR confirmation: did price break MA by >= 0.5x ATR? ---
        sma_fast = last_row.get("sma_20")
        sma_slow = last_row.get("sma_50")
        if sma_fast and sma_slow and atr_14 > 0:
            ma_gap = abs(sma_fast - sma_slow)
            atr_confirmed = ma_gap >= 0.5 * atr_14
        else:
            atr_confirmed = False

        # --- 52-week high proximity ---
        year_prices = df.tail(252) if len(df) >= 252 else df
        high_52w = year_prices["high"].max()
        pct_off_52w_high = ((high_52w - entry_price) / high_52w * 100) if high_52w > 0 else 0

        # --- 20-day momentum ---
        if len(df) >= 21:
            close_20ago = df["close"].iloc[-21]
            momentum_20d = ((last_row["close"] - close_20ago) / close_20ago * 100) if close_20ago > 0 else 0
        else:
            momentum_20d = 0

        # --- Volume ratio (current vs 20d avg) ---
        signal_day_vol = last_row["volume"] if last_row["volume"] else 0
        vol_ratio = signal_day_vol / avg_vol_20 if avg_vol_20 > 0 else 0

        enriched.append({
            **s,
            "_skip": False,
            "avg_dollar_vol": avg_dollar_vol,
            "avg_vol_20": avg_vol_20,
            "atr_14": atr_14,
            "atr_pct": atr_pct,
            "sma50_slope_pct": sma50_slope_pct,
            "sma20_slope_pct": sma20_slope_pct,
            "atr_confirmed": atr_confirmed,
            "pct_off_52w_high": pct_off_52w_high,
            "momentum_20d": momentum_20d,
            "vol_ratio": vol_ratio,
            "entry_price": entry_price,
        })

    return [s for s in enriched if not s.get("_skip", False)]


def apply_slippage(signals, slippage_map=None):
    """Adjust returns for slippage. Default: 0.20% >$5, 0.50% $1-$5."""
    adjusted = []
    for s in signals:
        ep = s.get("entry_price", 0)
        slip = 0.50 if ep < 5 else 0.20
        # Round-trip slippage cost = 2 * slip
        total_slip = 2 * slip
        new_s = dict(s)
        new_fr = {}
        for h, r in s.get("forward_returns", {}).items():
            if r is not None:
                new_fr[h] = round(r - total_slip, 4)
            else:
                new_fr[h] = r
        new_s["forward_returns"] = new_fr
        adjusted.append(new_s)
    return adjusted


def print_metrics(label, metrics):
    pf = metrics["profit_factor"]
    pf_str = f"{pf}" if isinstance(pf, str) else f"{pf:.2f}"
    print(f"  {label:42s} | n={metrics['count']:>4d} | WR={metrics['win_rate']:>5.1f}% "
          f"| Avg={metrics['avg_return']:>+6.2f}% | PF={pf_str:>6s} "
          f"| Exp={metrics['expectancy']:>+6.2f}% | W/L={metrics['win_count']}/{metrics['loss_count']} "
          f"| MaxConsecL={metrics['max_consec_losses']} | AvgDays={metrics['avg_trade_days']}")


def main():
    db = get_db()

    print("=" * 120)
    print("RUNNING BASELINE BACKTEST (20/50 MA, 10% EPS, 30d window, both directions, $1-$50)")
    print("=" * 120)
    result = run_baseline_backtest(db)
    signals = result["signals"]
    print(f"\nTotal raw signals: {len(signals)}")

    print("\nEnriching signals with computed features...")
    enriched = enrich_signals(db, signals)
    print(f"Enriched signals: {len(enriched)}")

    # Show distribution of key features
    print("\n--- Feature Distributions ---")
    dollar_vols = [s["avg_dollar_vol"] for s in enriched if s["avg_dollar_vol"]]
    avg_vols = [s["avg_vol_20"] for s in enriched if s["avg_vol_20"]]
    atr_pcts = [s["atr_pct"] for s in enriched if s["atr_pct"]]
    momentums = [s["momentum_20d"] for s in enriched]
    off_52w = [s["pct_off_52w_high"] for s in enriched]
    slopes = [s["sma50_slope_pct"] for s in enriched if s["sma50_slope_pct"] is not None]

    for name, vals in [
        ("Avg Dollar Volume ($K)", [v/1000 for v in dollar_vols]),
        ("Avg Daily Volume (K shares)", [v/1000 for v in avg_vols]),
        ("ATR %", atr_pcts),
        ("20d Momentum %", momentums),
        ("% Off 52w High", off_52w),
        ("SMA50 Slope %", slopes),
    ]:
        if vals:
            sv = sorted(vals)
            p10 = sv[int(len(sv)*0.1)]
            p25 = sv[int(len(sv)*0.25)]
            p50 = sv[int(len(sv)*0.5)]
            p75 = sv[int(len(sv)*0.75)]
            p90 = sv[int(len(sv)*0.9)]
            print(f"  {name:30s}: p10={p10:>8.1f}  p25={p25:>8.1f}  p50={p50:>8.1f}  p75={p75:>8.1f}  p90={p90:>8.1f}")

    # ======================================================================
    # INDIVIDUAL FILTER TESTS
    # ======================================================================
    print("\n" + "=" * 120)
    print("INDIVIDUAL FILTER TESTS (15-day horizon)")
    print("=" * 120)

    horizon = 15

    # 1. Baseline
    baseline = compute_metrics(enriched, horizon)
    print_metrics("1. BASELINE (no filters)", baseline)

    # 2. Bullish only
    bull_only = [s for s in enriched if s["signal_type"] == "bullish"]
    print_metrics("2. Bullish only", compute_metrics(bull_only, horizon))

    # 3. Bearish only
    bear_only = [s for s in enriched if s["signal_type"] == "bearish"]
    print_metrics("3. Bearish only", compute_metrics(bear_only, horizon))

    # 4. Positive EPS only
    pos_eps = [s for s in enriched if s.get("eps_change_pct", 0) > 0]
    print_metrics("4. Positive EPS only", compute_metrics(pos_eps, horizon))

    # 5. EPS cap < 100%
    eps_capped = [s for s in enriched if abs(s.get("eps_change_pct", 0)) < 100]
    print_metrics("5. |EPS change| < 100%", compute_metrics(eps_capped, horizon))

    # 6. Liquidity: avg dollar vol >= $500K
    liq_500k = [s for s in enriched if s.get("avg_dollar_vol", 0) >= 500_000]
    print_metrics("6. Avg dollar vol >= $500K", compute_metrics(liq_500k, horizon))

    # 7. Liquidity: avg dollar vol >= $1M
    liq_1m = [s for s in enriched if s.get("avg_dollar_vol", 0) >= 1_000_000]
    print_metrics("7. Avg dollar vol >= $1M", compute_metrics(liq_1m, horizon))

    # 8. Volume: avg vol >= 100K shares/day
    vol_100k = [s for s in enriched if s.get("avg_vol_20", 0) >= 100_000]
    print_metrics("8. Avg volume >= 100K shares", compute_metrics(vol_100k, horizon))

    # 9. Volume: avg vol >= 500K shares/day
    vol_500k = [s for s in enriched if s.get("avg_vol_20", 0) >= 500_000]
    print_metrics("9. Avg volume >= 500K shares", compute_metrics(vol_500k, horizon))

    # 10. Slippage applied
    slipped = apply_slippage(enriched)
    print_metrics("10. With slippage (0.2-0.5%)", compute_metrics(slipped, horizon))

    # 11. MA slope filter: SMA50 slope > 0 (bullish aligned)
    slope_pos = [s for s in enriched if s.get("sma50_slope_pct") is not None and s["sma50_slope_pct"] > 0]
    print_metrics("11. SMA50 slope > 0%", compute_metrics(slope_pos, horizon))

    # 12. MA slope filter: SMA20 slope > 0
    slope20_pos = [s for s in enriched if s.get("sma20_slope_pct") is not None and s["sma20_slope_pct"] > 0]
    print_metrics("12. SMA20 slope > 0%", compute_metrics(slope20_pos, horizon))

    # 13. MA slope filter: SMA20 slope > -2% (not falling hard)
    slope20_mild = [s for s in enriched if s.get("sma20_slope_pct") is not None and s["sma20_slope_pct"] > -2]
    print_metrics("13. SMA20 slope > -2%", compute_metrics(slope20_mild, horizon))

    # 14. ATR confirmation
    atr_conf = [s for s in enriched if s.get("atr_confirmed", False)]
    print_metrics("14. ATR confirmation (0.5x ATR)", compute_metrics(atr_conf, horizon))

    # 15. Days between > 10
    days_gt10 = [s for s in enriched if s.get("days_between", 0) > 10]
    print_metrics("15. Days between > 10", compute_metrics(days_gt10, horizon))

    # 16. Days between > 5
    days_gt5 = [s for s in enriched if s.get("days_between", 0) > 5]
    print_metrics("16. Days between > 5", compute_metrics(days_gt5, horizon))

    # 17. Within 30% of 52w high
    near_high = [s for s in enriched if s.get("pct_off_52w_high", 100) <= 30]
    print_metrics("17. Within 30% of 52w high", compute_metrics(near_high, horizon))

    # 18. Within 50% of 52w high
    near_high50 = [s for s in enriched if s.get("pct_off_52w_high", 100) <= 50]
    print_metrics("18. Within 50% of 52w high", compute_metrics(near_high50, horizon))

    # 19. 20d momentum > 0%
    mom_pos = [s for s in enriched if s.get("momentum_20d", 0) > 0]
    print_metrics("19. 20d momentum > 0%", compute_metrics(mom_pos, horizon))

    # 20. 20d momentum > -5% (not crashing)
    mom_mild = [s for s in enriched if s.get("momentum_20d", 0) > -5]
    print_metrics("20. 20d momentum > -5%", compute_metrics(mom_mild, horizon))

    # 21. ATR% < 8% (not too volatile)
    low_atr = [s for s in enriched if s.get("atr_pct", 100) < 8]
    print_metrics("21. ATR < 8%", compute_metrics(low_atr, horizon))

    # 22. ATR% < 12%
    low_atr12 = [s for s in enriched if s.get("atr_pct", 100) < 12]
    print_metrics("22. ATR < 12%", compute_metrics(low_atr12, horizon))

    # 23. Volume ratio > 0.8 (signal day not abnormally low)
    vol_ratio_ok = [s for s in enriched if s.get("vol_ratio", 0) >= 0.8]
    print_metrics("23. Vol ratio >= 0.8x avg", compute_metrics(vol_ratio_ok, horizon))

    # ======================================================================
    # MULTI-HORIZON COMPARISON
    # ======================================================================
    print("\n" + "=" * 120)
    print("HORIZON COMPARISON (baseline, all signals)")
    print("=" * 120)
    for h in [5, 10, 15, 30, 60]:
        m = compute_metrics(enriched, h)
        print_metrics(f"  Horizon {h}d", m)

    # ======================================================================
    # COMBINATION TESTS
    # ======================================================================
    print("\n" + "=" * 120)
    print("COMBINATION TESTS (15-day horizon)")
    print("=" * 120)

    def filt(sigs, **kwargs):
        out = list(sigs)
        for key, val in kwargs.items():
            if key == "bullish":
                out = [s for s in out if s["signal_type"] == "bullish"]
            elif key == "pos_eps":
                out = [s for s in out if s.get("eps_change_pct", 0) > 0]
            elif key == "eps_cap":
                out = [s for s in out if abs(s.get("eps_change_pct", 0)) < val]
            elif key == "min_dollar_vol":
                out = [s for s in out if s.get("avg_dollar_vol", 0) >= val]
            elif key == "min_avg_vol":
                out = [s for s in out if s.get("avg_vol_20", 0) >= val]
            elif key == "sma50_slope_gt":
                out = [s for s in out if s.get("sma50_slope_pct") is not None and s["sma50_slope_pct"] > val]
            elif key == "sma20_slope_gt":
                out = [s for s in out if s.get("sma20_slope_pct") is not None and s["sma20_slope_pct"] > val]
            elif key == "atr_confirmed":
                out = [s for s in out if s.get("atr_confirmed", False)]
            elif key == "days_between_gt":
                out = [s for s in out if s.get("days_between", 0) > val]
            elif key == "max_off_52w":
                out = [s for s in out if s.get("pct_off_52w_high", 100) <= val]
            elif key == "min_momentum":
                out = [s for s in out if s.get("momentum_20d", -999) > val]
            elif key == "max_atr_pct":
                out = [s for s in out if s.get("atr_pct", 100) < val]
            elif key == "slippage":
                out = apply_slippage(out)
            elif key == "min_vol_ratio":
                out = [s for s in out if s.get("vol_ratio", 0) >= val]
        return out

    combos = [
        ("C1: Bull + PosEPS",
         dict(bullish=True, pos_eps=True)),
        ("C2: Bull + PosEPS + EPS<100%",
         dict(bullish=True, pos_eps=True, eps_cap=100)),
        ("C3: Bull + PosEPS + Liq$500K",
         dict(bullish=True, pos_eps=True, min_dollar_vol=500_000)),
        ("C4: Bull + PosEPS + Liq$1M",
         dict(bullish=True, pos_eps=True, min_dollar_vol=1_000_000)),
        ("C5: Bull + PosEPS + SMA50slope>0",
         dict(bullish=True, pos_eps=True, sma50_slope_gt=0)),
        ("C6: Bull + PosEPS + Mom20d>0",
         dict(bullish=True, pos_eps=True, min_momentum=0)),
        ("C7: Bull + PosEPS + Within30%52wH",
         dict(bullish=True, pos_eps=True, max_off_52w=30)),
        ("C8: Bull + PosEPS + Days>10",
         dict(bullish=True, pos_eps=True, days_between_gt=10)),
        ("C9: Bull + PosEPS + ATRconf",
         dict(bullish=True, pos_eps=True, atr_confirmed=True)),
        ("C10: Bull + PosEPS + ATR<8%",
         dict(bullish=True, pos_eps=True, max_atr_pct=8)),

        # Layered combinations
        ("C11: Bull+PosEPS+Liq$500K+EPS<100",
         dict(bullish=True, pos_eps=True, min_dollar_vol=500_000, eps_cap=100)),
        ("C12: Bull+PosEPS+Liq$500K+SMA50>0",
         dict(bullish=True, pos_eps=True, min_dollar_vol=500_000, sma50_slope_gt=0)),
        ("C13: Bull+PosEPS+Liq$500K+Mom>0",
         dict(bullish=True, pos_eps=True, min_dollar_vol=500_000, min_momentum=0)),
        ("C14: Bull+PosEPS+Liq$500K+52w<30%",
         dict(bullish=True, pos_eps=True, min_dollar_vol=500_000, max_off_52w=30)),
        ("C15: Bull+PosEPS+Liq$500K+Days>10",
         dict(bullish=True, pos_eps=True, min_dollar_vol=500_000, days_between_gt=10)),
        ("C16: Bull+PosEPS+Liq$500K+ATR<12%",
         dict(bullish=True, pos_eps=True, min_dollar_vol=500_000, max_atr_pct=12)),

        # Triple+ layered
        ("C17: Bull+PosEPS+Liq$500K+EPS<100+Days>10",
         dict(bullish=True, pos_eps=True, min_dollar_vol=500_000, eps_cap=100, days_between_gt=10)),
        ("C18: Bull+PosEPS+Liq$500K+SMA50>0+Mom>0",
         dict(bullish=True, pos_eps=True, min_dollar_vol=500_000, sma50_slope_gt=0, min_momentum=0)),
        ("C19: Bull+PosEPS+Liq$500K+52w<30%+ATR<12",
         dict(bullish=True, pos_eps=True, min_dollar_vol=500_000, max_off_52w=30, max_atr_pct=12)),
        ("C20: Bull+PosEPS+Liq$500K+Days>10+SMA50>0",
         dict(bullish=True, pos_eps=True, min_dollar_vol=500_000, days_between_gt=10, sma50_slope_gt=0)),

        # Kitchen sink combos
        ("C21: Bull+PosEPS+Liq$500K+EPS<100+Days>10+SMA50>0",
         dict(bullish=True, pos_eps=True, min_dollar_vol=500_000, eps_cap=100, days_between_gt=10, sma50_slope_gt=0)),
        ("C22: Bull+PosEPS+Liq$500K+EPS<100+52w<30%+Mom>0",
         dict(bullish=True, pos_eps=True, min_dollar_vol=500_000, eps_cap=100, max_off_52w=30, min_momentum=0)),
        ("C23: Bull+PosEPS+Liq$500K+EPS<100+ATR<12+Mom>-5",
         dict(bullish=True, pos_eps=True, min_dollar_vol=500_000, eps_cap=100, max_atr_pct=12, min_momentum=-5)),
        ("C24: Bull+PosEPS+Liq$500K+EPS<100+Days>5+SMA50>0",
         dict(bullish=True, pos_eps=True, min_dollar_vol=500_000, eps_cap=100, days_between_gt=5, sma50_slope_gt=0)),

        # With slippage on best combos
        ("C25: C17 + slippage",
         dict(bullish=True, pos_eps=True, min_dollar_vol=500_000, eps_cap=100, days_between_gt=10, slippage=True)),
        ("C26: C21 + slippage",
         dict(bullish=True, pos_eps=True, min_dollar_vol=500_000, eps_cap=100, days_between_gt=10, sma50_slope_gt=0, slippage=True)),

        # Relaxed combos for more signals
        ("C27: Bull+PosEPS+Vol100K+EPS<200",
         dict(bullish=True, pos_eps=True, min_avg_vol=100_000, eps_cap=200)),
        ("C28: Bull+PosEPS+Vol100K+Mom>-5",
         dict(bullish=True, pos_eps=True, min_avg_vol=100_000, min_momentum=-5)),
        ("C29: Bull+Liq$500K+EPS<100",
         dict(bullish=True, min_dollar_vol=500_000, eps_cap=100)),
        ("C30: PosEPS+Liq$500K+SMA50>0",
         dict(pos_eps=True, min_dollar_vol=500_000, sma50_slope_gt=0)),
    ]

    for label, params in combos:
        filtered = filt(enriched, **params)
        m = compute_metrics(filtered, horizon)
        print_metrics(label, m)

    # ======================================================================
    # BEST COMBOS ACROSS ALL HORIZONS
    # ======================================================================
    print("\n" + "=" * 120)
    print("BEST COMBOS ACROSS ALL HORIZONS")
    print("=" * 120)

    best_combos = [
        ("BASELINE", {}),
        ("C1: Bull + PosEPS", dict(bullish=True, pos_eps=True)),
        ("C11: Bull+PosEPS+Liq$500K+EPS<100", dict(bullish=True, pos_eps=True, min_dollar_vol=500_000, eps_cap=100)),
        ("C17: +Days>10", dict(bullish=True, pos_eps=True, min_dollar_vol=500_000, eps_cap=100, days_between_gt=10)),
        ("C21: +SMA50>0", dict(bullish=True, pos_eps=True, min_dollar_vol=500_000, eps_cap=100, days_between_gt=10, sma50_slope_gt=0)),
        ("C22: +52w<30%+Mom>0", dict(bullish=True, pos_eps=True, min_dollar_vol=500_000, eps_cap=100, max_off_52w=30, min_momentum=0)),
    ]

    for label, params in best_combos:
        filtered = filt(enriched, **params) if params else enriched
        print(f"\n  {label}:")
        for h in [5, 10, 15, 30, 60]:
            m = compute_metrics(filtered, h)
            pf = m["profit_factor"]
            pf_str = f"{pf}" if isinstance(pf, str) else f"{pf:.2f}"
            print(f"    {h:>2d}d: n={m['count']:>4d} | WR={m['win_rate']:>5.1f}% | Avg={m['avg_return']:>+6.2f}% "
                  f"| PF={pf_str:>6s} | Exp={m['expectancy']:>+6.2f}%")

    # ======================================================================
    # SIGNAL FREQUENCY ANALYSIS
    # ======================================================================
    print("\n" + "=" * 120)
    print("SIGNAL FREQUENCY (signals per month)")
    print("=" * 120)

    for label, params in best_combos:
        filtered = filt(enriched, **params) if params else enriched
        if not filtered:
            print(f"  {label}: 0 signals")
            continue
        dates = sorted([s["trend_change_date"] for s in filtered])
        first = date.fromisoformat(dates[0])
        last = date.fromisoformat(dates[-1])
        months = max(1, (last - first).days / 30.44)
        per_month = len(filtered) / months
        print(f"  {label:50s}: {len(filtered):>4d} signals over {months:.0f} months = {per_month:.1f}/month")


if __name__ == "__main__":
    main()
