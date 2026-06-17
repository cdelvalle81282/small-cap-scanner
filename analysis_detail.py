"""Detailed win rate and hold time breakdown for each filter and recommended combos."""

import sys
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, median

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


def enrich_signals(db, signals):
    enriched = []
    for s in signals:
        ticker = s["ticker"]
        trend_date = s.get("trend_change_date")
        entry_date = s.get("entry_date")
        entry_price = s.get("entry_price")
        if not trend_date or not entry_date or not entry_price:
            continue

        trend_dt = date.fromisoformat(trend_date)
        fetch_start = (trend_dt - timedelta(days=400)).isoformat()
        prices = db.get_daily_prices(ticker, fetch_start, trend_date)

        if len(prices) < 50:
            continue

        df = pd.DataFrame(prices)
        df["close"] = df["close"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["volume"] = df["volume"].astype(float)

        last_20 = df.tail(20)
        avg_vol_20 = last_20["volume"].mean()
        avg_price_20 = last_20["close"].mean()
        avg_dollar_vol = avg_vol_20 * avg_price_20

        df["sma_50"] = df["close"].rolling(50).mean()
        last_row = df.iloc[-1]
        sma_50_now = last_row.get("sma_50")
        sma_50_5ago = df["sma_50"].iloc[-6] if len(df) >= 6 else None
        sma50_slope = ((sma_50_now - sma_50_5ago) / sma_50_5ago * 100) if sma_50_now and sma_50_5ago and sma_50_5ago != 0 else None

        year_prices = df.tail(252) if len(df) >= 252 else df
        high_52w = year_prices["high"].max()
        pct_off_52w = ((high_52w - entry_price) / high_52w * 100) if high_52w > 0 else 0

        if len(df) >= 21:
            close_20ago = df["close"].iloc[-21]
            momentum_20d = ((last_row["close"] - close_20ago) / close_20ago * 100) if close_20ago > 0 else 0
        else:
            momentum_20d = 0

        enriched.append({
            **s,
            "avg_dollar_vol": avg_dollar_vol,
            "avg_vol_20": avg_vol_20,
            "sma50_slope_pct": sma50_slope,
            "pct_off_52w_high": pct_off_52w,
            "momentum_20d": momentum_20d,
        })
    return enriched


def detailed_metrics(signals, horizon):
    """Return detailed metrics including winner/loser hold time breakdown."""
    results = []
    for s in signals:
        fr = s.get("forward_returns", {})
        td = s.get("trade_details", {}).get(horizon, {})
        r = fr.get(horizon)
        if r is None:
            continue

        entry = s.get("entry_date")
        exit_d = td.get("exit_date")
        exit_reason = td.get("exit_reason", "unknown")
        hold_days = 0
        if entry and exit_d:
            try:
                hold_days = (date.fromisoformat(exit_d) - date.fromisoformat(entry)).days
            except (ValueError, TypeError):
                pass

        results.append({
            "ticker": s["ticker"],
            "return": r,
            "hold_days": hold_days,
            "exit_reason": exit_reason,
            "is_win": r > 0,
        })

    if not results:
        return None

    all_returns = [r["return"] for r in results]
    winners = [r for r in results if r["is_win"]]
    losers = [r for r in results if not r["is_win"]]
    stopped = [r for r in results if r["exit_reason"] == "stop_loss"]
    horizon_exits = [r for r in results if r["exit_reason"] == "horizon"]

    win_hold = [w["hold_days"] for w in winners]
    lose_hold = [l["hold_days"] for l in losers]
    stop_hold = [s["hold_days"] for s in stopped]

    return {
        "total": len(results),
        "wins": len(winners),
        "losses": len(losers),
        "win_pct": round(len(winners) / len(results) * 100, 1),
        "avg_return": round(mean(all_returns), 2),
        "avg_winner": round(mean([w["return"] for w in winners]), 2) if winners else 0,
        "avg_loser": round(mean([l["return"] for l in losers]), 2) if losers else 0,
        "avg_hold_all": round(mean([r["hold_days"] for r in results]), 1),
        "avg_hold_winners": round(mean(win_hold), 1) if win_hold else 0,
        "avg_hold_losers": round(mean(lose_hold), 1) if lose_hold else 0,
        "median_hold_all": round(median([r["hold_days"] for r in results]), 1),
        "median_hold_winners": round(median(win_hold), 1) if win_hold else 0,
        "median_hold_losers": round(median(lose_hold), 1) if lose_hold else 0,
        "stopped_out": len(stopped),
        "stopped_pct": round(len(stopped) / len(results) * 100, 1),
        "horizon_exits": len(horizon_exits),
        "avg_hold_stopped": round(mean(stop_hold), 1) if stop_hold else 0,
        "max_consec_losses": max_consec_losses([r["is_win"] for r in results]),
    }


def max_consec_losses(wins):
    max_c = 0
    curr = 0
    for w in wins:
        if not w:
            curr += 1
            max_c = max(max_c, curr)
        else:
            curr = 0
    return max_c


def print_detail(label, m):
    if m is None:
        print(f"\n  {label}: NO SIGNALS")
        return
    print(f"\n  {label}")
    print(f"    Signals:          {m['total']:>4d}  ({m['wins']}W / {m['losses']}L)")
    print(f"    Win Rate:         {m['win_pct']:>5.1f}%")
    print(f"    Avg Return:       {m['avg_return']:>+6.2f}%")
    print(f"    Avg Winner:       {m['avg_winner']:>+6.2f}%")
    print(f"    Avg Loser:        {m['avg_loser']:>+6.2f}%")
    print(f"    Stopped Out:      {m['stopped_out']:>4d}  ({m['stopped_pct']:.0f}%)")
    print(f"    Horizon Exits:    {m['horizon_exits']:>4d}")
    print(f"    Max Consec Losses:{m['max_consec_losses']:>4d}")
    print(f"    -- Hold Times --")
    print(f"    Avg (all):        {m['avg_hold_all']:>5.1f} days")
    print(f"    Avg (winners):    {m['avg_hold_winners']:>5.1f} days")
    print(f"    Avg (losers):     {m['avg_hold_losers']:>5.1f} days")
    print(f"    Median (all):     {m['median_hold_all']:>5.1f} days")
    print(f"    Median (winners): {m['median_hold_winners']:>5.1f} days")
    print(f"    Median (losers):  {m['median_hold_losers']:>5.1f} days")
    print(f"    Avg (stopped):    {m['avg_hold_stopped']:>5.1f} days")


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
        elif key == "sma50_slope_gt":
            out = [s for s in out if s.get("sma50_slope_pct") is not None and s["sma50_slope_pct"] > val]
        elif key == "days_between_gt":
            out = [s for s in out if s.get("days_between", 0) > val]
        elif key == "max_off_52w":
            out = [s for s in out if s.get("pct_off_52w_high", 100) <= val]
        elif key == "min_momentum":
            out = [s for s in out if s.get("momentum_20d", -999) > val]
    return out


def main():
    db = get_db()
    result = run_baseline_backtest(db)
    signals = result["signals"]
    enriched = enrich_signals(db, signals)

    filters = [
        ("1. BASELINE (no filters)", {}),
        ("2. Bullish only", dict(bullish=True)),
        ("3. Bearish only", {}),  # special
        ("4. Positive EPS only", dict(pos_eps=True)),
        ("5. |EPS change| < 100%", dict(eps_cap=100)),
        ("6. Avg dollar vol >= $500K", dict(min_dollar_vol=500_000)),
        ("7. SMA50 slope > 0%", dict(sma50_slope_gt=0)),
        ("8. Days between > 10", dict(days_between_gt=10)),
        ("9. Within 30% of 52w high", dict(max_off_52w=30)),
        ("10. 20d momentum > 0%", dict(min_momentum=0)),
    ]

    for horizon in [5, 10, 15]:
        print("=" * 90)
        print(f"  INDIVIDUAL FILTERS — {horizon}-DAY HORIZON")
        print("=" * 90)

        for label, params in filters:
            if label == "3. Bearish only":
                filtered = [s for s in enriched if s["signal_type"] == "bearish"]
            else:
                filtered = filt(enriched, **params) if params else enriched
            m = detailed_metrics(filtered, horizon)
            print_detail(label, m)

    # Recommended combinations
    combos = [
        ("C17: Bull+PosEPS+Liq$500K+EPS<100+Days>10 (RECOMMENDED)",
         dict(bullish=True, pos_eps=True, min_dollar_vol=500_000, eps_cap=100, days_between_gt=10)),
        ("C22: Bull+PosEPS+Liq$500K+EPS<100+52w<30%+Mom>0 (HIGH QUALITY)",
         dict(bullish=True, pos_eps=True, min_dollar_vol=500_000, eps_cap=100, max_off_52w=30, min_momentum=0)),
        ("C11: Bull+PosEPS+Liq$500K+EPS<100 (BROADER)",
         dict(bullish=True, pos_eps=True, min_dollar_vol=500_000, eps_cap=100)),
        ("C1: Bull+PosEPS (SIMPLEST PROFITABLE)",
         dict(bullish=True, pos_eps=True)),
    ]

    for horizon in [5, 10, 15]:
        print("\n" + "=" * 90)
        print(f"  RECOMMENDED COMBOS — {horizon}-DAY HORIZON")
        print("=" * 90)

        for label, params in combos:
            filtered = filt(enriched, **params)
            m = detailed_metrics(filtered, horizon)
            print_detail(label, m)

    # Per-trade detail for C17 and C22
    print("\n" + "=" * 90)
    print("  C17 INDIVIDUAL TRADES (15-day horizon)")
    print("=" * 90)

    c17 = filt(enriched, bullish=True, pos_eps=True, min_dollar_vol=500_000, eps_cap=100, days_between_gt=10)
    print(f"  {'Ticker':<8s} {'Signal Date':<12s} {'Entry Date':<12s} {'Entry$':>8s} "
          f"{'Exit Date':<12s} {'Exit$':>8s} {'Return':>8s} {'Hold':>5s} {'Exit Reason':<12s} {'EPS%':>7s}")
    print("  " + "-" * 105)
    for s in sorted(c17, key=lambda x: x.get("trend_change_date", "")):
        td = s.get("trade_details", {}).get(15, {})
        ep = s.get("entry_price", 0)
        ex_p = td.get("exit_price", 0)
        ret = s.get("forward_returns", {}).get(15)
        entry = s.get("entry_date", "")
        exit_d = td.get("exit_date", "")
        hold = 0
        if entry and exit_d:
            try:
                hold = (date.fromisoformat(exit_d) - date.fromisoformat(entry)).days
            except:
                pass
        wl = "W" if ret and ret > 0 else "L"
        print(f"  {s['ticker']:<8s} {s.get('trend_change_date',''):<12s} {entry:<12s} ${ep:>7.2f} "
              f"{exit_d:<12s} ${ex_p if ex_p else 0:>7.2f} {ret if ret else 0:>+7.2f}% {hold:>4d}d {td.get('exit_reason',''):<12s} {s.get('eps_change_pct',0):>+6.1f}% {wl}")

    print("\n" + "=" * 90)
    print("  C22 INDIVIDUAL TRADES (15-day horizon)")
    print("=" * 90)

    c22 = filt(enriched, bullish=True, pos_eps=True, min_dollar_vol=500_000, eps_cap=100, max_off_52w=30, min_momentum=0)
    print(f"  {'Ticker':<8s} {'Signal Date':<12s} {'Entry Date':<12s} {'Entry$':>8s} "
          f"{'Exit Date':<12s} {'Exit$':>8s} {'Return':>8s} {'Hold':>5s} {'Exit Reason':<12s} {'EPS%':>7s}")
    print("  " + "-" * 105)
    for s in sorted(c22, key=lambda x: x.get("trend_change_date", "")):
        td = s.get("trade_details", {}).get(15, {})
        ep = s.get("entry_price", 0)
        ex_p = td.get("exit_price", 0)
        ret = s.get("forward_returns", {}).get(15)
        entry = s.get("entry_date", "")
        exit_d = td.get("exit_date", "")
        hold = 0
        if entry and exit_d:
            try:
                hold = (date.fromisoformat(exit_d) - date.fromisoformat(entry)).days
            except:
                pass
        wl = "W" if ret and ret > 0 else "L"
        print(f"  {s['ticker']:<8s} {s.get('trend_change_date',''):<12s} {entry:<12s} ${ep:>7.2f} "
              f"{exit_d:<12s} ${ex_p if ex_p else 0:>7.2f} {ret if ret else 0:>+7.2f}% {hold:>4d}d {td.get('exit_reason',''):<12s} {s.get('eps_change_pct',0):>+6.1f}% {wl}")


if __name__ == "__main__":
    main()
