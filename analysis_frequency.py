"""Check signal frequency for the best filter combos."""

import sys
from datetime import date
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent))

from config import DB_PATH, BacktestConfig, ScannerConfig
from core.backtest import Backtester
from core.database import Database
from analysis_winners_losers import enrich_deep, filt_c17


def show(label, sigs):
    if not sigs:
        print(f"  {label:55s}: 0 signals")
        return
    dates = sorted([s["trend_date"] for s in sigs])
    first = date.fromisoformat(dates[0])
    last = date.fromisoformat(dates[-1])
    months = max(1, (last - first).days / 30.44)
    per_mo = len(sigs) / months

    rets = [s["return_15d"] for s in sigs if s["return_15d"] is not None]
    w = [r for r in rets if r > 0]
    l = [r for r in rets if r <= 0]
    wr = len(w) / len(rets) * 100 if rets else 0
    avg = mean(rets) if rets else 0
    gg = sum(w)
    gl = sum(r for r in l if r < 0)
    pf = gg / abs(gl) if gl else float("inf")
    pf_s = f"{pf:.2f}" if pf != float("inf") else "inf"

    print(f"  {label:55s}: {len(sigs):>3d} sigs | {per_mo:.1f}/mo | WR={wr:.0f}% | Avg={avg:>+.1f}% | PF={pf_s}")
    print(f"    Dates: {dates[0]} to {dates[-1]}")
    for s in sorted(sigs, key=lambda x: x["trend_date"]):
        r = s["return_15d"] if s["return_15d"] else 0
        wl = "W" if r > 0 else "L"
        print(f"      {s['ticker']:7s} {s['trend_date']}  {r:>+7.1f}%  {wl}  "
              f"updays={s['consec_up_days']}  slope5d={s['slope_5d']:+.2f}  gap={s['gap_pct']:+.2f}%")


def main():
    db = Database(DB_PATH)
    db.initialize()
    sc = ScannerConfig(
        min_price=1.0, max_price=50.0, min_market_cap=50_000_000, max_market_cap=10_000_000_000,
        ma_crossover_pairs=[(20, 50)], eps_change_threshold=10.0, trend_window_days=30, direction="both",
    )
    bc = BacktestConfig(
        start_date="2022-01-01", end_date="2026-03-19",
        forward_return_days=[5, 10, 15, 30, 60], ma_crossover_pairs=[(20, 50)],
        eps_thresholds=[10.0], trend_windows=[30],
    )
    result = Backtester(db, sc, bc).run()
    enriched = enrich_deep(db, result["signals"])
    c17 = filt_c17(enriched)

    # Filter variants
    up2 = [s for s in c17 if s.get("consec_up_days", 0) >= 2]
    up1 = [s for s in c17 if s.get("consec_up_days", 0) >= 1]
    slope5 = [s for s in c17 if s.get("slope_5d", 0) > 0]
    gap_slope = [s for s in c17 if s.get("gap_pct", 0) > -2 and s.get("slope_5d", 0) > 0]
    full_best = [s for s in c17 if s.get("slope_5d", 0) > 0 and s.get("gap_pct", 0) > -2 and s.get("consec_up_days", 0) >= 2]

    # Also test with both MA pairs to increase signal count
    sc2 = ScannerConfig(
        min_price=1.0, max_price=50.0, min_market_cap=50_000_000, max_market_cap=10_000_000_000,
        ma_crossover_pairs=[(20, 50), (50, 200)], eps_change_threshold=10.0, trend_window_days=30, direction="both",
    )
    bc2 = BacktestConfig(
        start_date="2022-01-01", end_date="2026-03-19",
        forward_return_days=[5, 10, 15, 30, 60], ma_crossover_pairs=[(20, 50), (50, 200)],
        eps_thresholds=[10.0], trend_windows=[30],
    )
    result2 = Backtester(db, sc2, bc2).run()
    enriched2 = enrich_deep(db, result2["signals"])

    # C17 with both MA pairs
    c17_both = [s for s in enriched2
                if s["signal_type"] == "bullish"
                and s["eps_change_pct"] > 0
                and s.get("avg_dollar_vol", 0) >= 500_000
                and abs(s["eps_change_pct"]) < 100
                and s["days_between"] > 10]
    c17_both_up2 = [s for s in c17_both if s.get("consec_up_days", 0) >= 2]
    c17_both_up1 = [s for s in c17_both if s.get("consec_up_days", 0) >= 1]
    c17_both_slope = [s for s in c17_both if s.get("slope_5d", 0) > 0]

    # Also try wider trend window (45 days)
    sc3 = ScannerConfig(
        min_price=1.0, max_price=50.0, min_market_cap=50_000_000, max_market_cap=10_000_000_000,
        ma_crossover_pairs=[(20, 50)], eps_change_threshold=10.0, trend_window_days=45, direction="both",
    )
    bc3 = BacktestConfig(
        start_date="2022-01-01", end_date="2026-03-19",
        forward_return_days=[5, 10, 15, 30, 60], ma_crossover_pairs=[(20, 50)],
        eps_thresholds=[10.0], trend_windows=[45],
    )
    result3 = Backtester(db, sc3, bc3).run()
    enriched3 = enrich_deep(db, result3["signals"])
    c17_45d = [s for s in enriched3
               if s["signal_type"] == "bullish"
               and s["eps_change_pct"] > 0
               and s.get("avg_dollar_vol", 0) >= 500_000
               and abs(s["eps_change_pct"]) < 100
               and s["days_between"] > 10]
    c17_45d_up2 = [s for s in c17_45d if s.get("consec_up_days", 0) >= 2]
    c17_45d_up1 = [s for s in c17_45d if s.get("consec_up_days", 0) >= 1]

    print("=" * 100)
    print("  FREQUENCY CHECK: Can we hit 2-4 signals/month?")
    print("=" * 100)

    print("\n--- 20/50 MA only, 30d trend window ---")
    show("C17 BASE (20/50, 30d)", c17)
    print()
    show("C17 + 2+ consecutive up days", up2)
    print()
    show("C17 + 1+ consecutive up days", up1)
    print()
    show("C17 + slope_5d > 0", slope5)
    print()
    show("C17 + gap>-2% + slope_5d>0", gap_slope)
    print()
    show("C17 + slope_5d>0 + gap>-2% + 2+ up days", full_best)

    print("\n--- Both MA pairs (20/50 + 50/200), 30d trend window ---")
    show("C17 both MAs BASE", c17_both)
    print()
    show("C17 both MAs + 2+ up days", c17_both_up2)
    print()
    show("C17 both MAs + 1+ up days", c17_both_up1)
    print()
    show("C17 both MAs + slope_5d>0", c17_both_slope)

    print("\n--- 20/50 MA only, 45d trend window ---")
    show("C17 45d window BASE", c17_45d)
    print()
    show("C17 45d window + 2+ up days", c17_45d_up2)
    print()
    show("C17 45d window + 1+ up days", c17_45d_up1)


if __name__ == "__main__":
    main()
