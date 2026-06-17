"""For each winner under the green candle rule (d1 midpoint, close>open),
show how much above the Day 1 close the Day 2 open was."""

import sys
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import DB_PATH, BacktestConfig, ScannerConfig
from core.backtest import Backtester
from core.database import Database
from analysis_winners_losers import enrich_deep
from analysis_optimization import simulate_trade_v2


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

    # Collect all unique winner trades across configs (avoid duplicates)
    seen = set()
    all_trades = []

    for (eps_thresh, tw), (c17, raw) in sorted(signal_cache.items()):
        for s in c17:
            match = None
            for rs in raw:
                if rs["ticker"] == s["ticker"] and rs.get("trend_change_date") == s["trend_date"]:
                    match = rs
                    break
            if not match:
                continue

            # Run the unrealistic sim: d1 midpoint entry, green candle required
            trade = simulate_trade_v2(
                db, s["ticker"], s["trend_date"], "bullish",
                entry_type="midpoint", atr_mult=0.5, max_days=15,
                require_entry_confirmation=True,
            )
            if not trade or trade["return"] <= 0:
                continue

            key = (s["ticker"], s["trend_date"])
            if key in seen:
                continue
            seen.add(key)

            # Now get the actual Day 1 and Day 2 prices
            signal_dt = date.fromisoformat(s["trend_date"])
            fetch_start = (signal_dt - timedelta(days=5)).isoformat()
            fetch_end = (signal_dt + timedelta(days=20)).isoformat()
            rows = db.get_daily_prices(s["ticker"], fetch_start, fetch_end)
            if not rows:
                continue

            df = pd.DataFrame(rows)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            sig_idx_list = df.index[df["date"] == s["trend_date"]].tolist()
            if not sig_idx_list:
                continue
            sig_idx = sig_idx_list[0]

            d1_idx = sig_idx + 1
            d2_idx = sig_idx + 2
            if d1_idx >= len(df) or d2_idx >= len(df):
                continue

            d0 = df.iloc[sig_idx]
            d1 = df.iloc[d1_idx]
            d2 = df.iloc[d2_idx]

            d1_mid = (d1["high"] + d1["low"]) / 2
            d1_close = d1["close"]
            d2_open = d2["open"]

            gap_pct = (d2_open - d1_close) / d1_close * 100
            gap_vs_mid = (d2_open - d1_mid) / d1_mid * 100

            all_trades.append({
                "ticker": s["ticker"],
                "signal_date": s["trend_date"],
                "eps_pct": s.get("eps_change_pct", 0),
                "d1_open": d1["open"],
                "d1_high": d1["high"],
                "d1_low": d1["low"],
                "d1_mid": d1_mid,
                "d1_close": d1_close,
                "d2_open": d2_open,
                "gap_pct": gap_pct,
                "gap_vs_mid": gap_vs_mid,
                "trade_return": trade["return"],
                "hold_days": trade["hold_days"],
                "exit_reason": trade["exit_reason"],
                "configs": f"EPS>={eps_thresh:.0f}%/{tw}d",
            })

    # Sort by signal date
    all_trades.sort(key=lambda x: (x["signal_date"], x["ticker"]))

    print("=" * 180)
    print("  GREEN CANDLE WINNERS: Day 1 close vs Day 2 open")
    print("  (These are all winners from the unrealistic d1_midpoint + green candle filter)")
    print("=" * 180)
    print()
    print(f"  {'Ticker':7s} {'Signal Date':12s} {'EPS%':>6s} | {'D1 Open':>8s} {'D1 High':>8s} {'D1 Low':>8s} "
          f"{'D1 Mid':>8s} {'D1 Close':>8s} | {'D2 Open':>8s} {'Gap%':>7s} {'vs Mid':>7s} | "
          f"{'Return':>8s} {'Hold':>5s}")
    print("  " + "-" * 170)

    gaps = []
    gaps_vs_mid = []

    for t in all_trades:
        print(f"  {t['ticker']:7s} {t['signal_date']:12s} {t['eps_pct']:>+5.0f}% | "
              f"${t['d1_open']:>7.2f} ${t['d1_high']:>7.2f} ${t['d1_low']:>7.2f} "
              f"${t['d1_mid']:>7.2f} ${t['d1_close']:>7.2f} | "
              f"${t['d2_open']:>7.2f} {t['gap_pct']:>+6.1f}% {t['gap_vs_mid']:>+6.1f}% | "
              f"{t['trade_return']:>+7.1f}% {t['hold_days']:>4d}d")
        gaps.append(t["gap_pct"])
        gaps_vs_mid.append(t["gap_vs_mid"])

    print()
    print(f"  {len(all_trades)} unique winners")
    print(f"  D2 open vs D1 close:   avg gap = {mean(gaps):+.2f}%,  "
          f"min = {min(gaps):+.2f}%,  max = {max(gaps):+.2f}%")
    print(f"  D2 open vs D1 midpoint: avg gap = {mean(gaps_vs_mid):+.2f}%,  "
          f"min = {min(gaps_vs_mid):+.2f}%,  max = {max(gaps_vs_mid):+.2f}%")

    # Count how many gapped up vs down
    gap_up = sum(1 for g in gaps if g > 0)
    gap_down = sum(1 for g in gaps if g < 0)
    gap_flat = sum(1 for g in gaps if g == 0)
    print(f"  Gapped up: {gap_up},  Gapped down: {gap_down},  Flat: {gap_flat}")

    # What if you entered at D2 open instead of D1 midpoint?
    print()
    print("  Return comparison: D1 midpoint entry vs D2 open entry")
    print("  " + "-" * 80)
    for t in all_trades:
        d2_entry_return = (t["trade_return"] * t["d1_mid"] / t["d2_open"]) if t["d2_open"] > 0 else 0
        # Actually let's compute it properly: if exit price is the same,
        # return from d2_open = (exit_price - d2_open) / d2_open
        # We know: exit_price = d1_mid * (1 + trade_return/100)
        exit_price = t["d1_mid"] * (1 + t["trade_return"] / 100)
        d2_return = (exit_price - t["d2_open"]) / t["d2_open"] * 100
        cost = t["trade_return"] - d2_return
        print(f"  {t['ticker']:7s} {t['signal_date']:12s}  "
              f"d1_mid entry: {t['trade_return']:>+7.1f}%  |  "
              f"d2_open entry: {d2_return:>+7.1f}%  |  cost of waiting: {cost:>+5.1f}pp")


if __name__ == "__main__":
    main()
