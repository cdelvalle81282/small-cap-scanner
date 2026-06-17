"""Find filter combos that achieve >50% win rate while maximizing signal frequency."""

import sys
from datetime import date
from pathlib import Path
from statistics import mean
from itertools import product

sys.path.insert(0, str(Path(__file__).parent))

from config import DB_PATH, BacktestConfig, ScannerConfig
from core.backtest import Backtester
from core.database import Database
from analysis_winners_losers import enrich_deep, filt_c17


def show(label, sigs, min_n=3):
    if not sigs:
        return
    rets = [s["return_15d"] for s in sigs if s["return_15d"] is not None]
    if len(rets) < min_n:
        return
    w = [r for r in rets if r > 0]
    l = [r for r in rets if r <= 0]
    wr = len(w) / len(rets) * 100
    avg = mean(rets)
    gg = sum(w)
    gl = sum(r for r in l if r < 0)
    pf = gg / abs(gl) if gl else float("inf")
    pf_s = f"{pf:.2f}" if pf != float("inf") else "inf"

    dates = sorted([s["trend_date"] for s in sigs])
    first = date.fromisoformat(dates[0])
    last = date.fromisoformat(dates[-1])
    months = max(1, (last - first).days / 30.44)
    per_mo = len(sigs) / months

    flag = " ***" if wr >= 50 else ""
    print(f"  {label:65s}: {len(sigs):>3d} sigs | {per_mo:.1f}/mo | WR={wr:.0f}% | Avg={avg:>+.1f}% | PF={pf_s}{flag}")


def run_all():
    db = Database(DB_PATH)
    db.initialize()

    results_by_window = {}

    for trend_window in [30, 45, 60]:
        sc = ScannerConfig(
            min_price=1.0, max_price=50.0,
            min_market_cap=50_000_000, max_market_cap=10_000_000_000,
            ma_crossover_pairs=[(20, 50)],
            eps_change_threshold=10.0, trend_window_days=trend_window, direction="both",
        )
        bc = BacktestConfig(
            start_date="2022-01-01", end_date="2026-03-19",
            forward_return_days=[5, 10, 15, 30, 60],
            ma_crossover_pairs=[(20, 50)],
            eps_thresholds=[10.0], trend_windows=[trend_window],
        )
        result = Backtester(db, sc, bc).run()
        enriched = enrich_deep(db, result["signals"])
        c17 = filt_c17(enriched)
        results_by_window[trend_window] = c17

    print("=" * 110)
    print("  TARGET: >50% WIN RATE — SEARCHING ALL FILTER COMBINATIONS")
    print("=" * 110)

    for tw, c17 in results_by_window.items():
        print(f"\n{'='*110}")
        print(f"  TREND WINDOW = {tw} DAYS  ({len(c17)} C17 signals)")
        print(f"{'='*110}")

        # ── Single momentum filters at different thresholds ──
        print(f"\n  --- Single filters ---")
        show(f"C17 base", c17)
        for n in [1, 2, 3]:
            show(f"consec_up_days >= {n}", [s for s in c17 if s.get("consec_up_days", 0) >= n])
        for t in [0, 0.5, 1.0, 2.0]:
            show(f"slope_5d > {t}", [s for s in c17 if s.get("slope_5d", 0) > t])
        for t in [0, 0.5, 1.0]:
            show(f"slope_10d > {t}", [s for s in c17 if s.get("slope_10d", 0) > t])
        show(f"vol_acceleration > 50%", [s for s in c17 if s.get("vol_acceleration", 0) > 50])
        show(f"vol_acceleration > 20%", [s for s in c17 if s.get("vol_acceleration", 0) > 20])
        show(f"vol_spike = True", [s for s in c17 if s.get("vol_spike")])
        show(f"bb_above_upper = True", [s for s in c17 if s.get("bb_above_upper")])
        show(f"bb_pct_b > 0.8", [s for s in c17 if s.get("bb_pct_b", 0) > 0.8])
        show(f"bb_pct_b > 0.5", [s for s in c17 if s.get("bb_pct_b", 0) > 0.5])
        show(f"gap_pct > 0", [s for s in c17 if s.get("gap_pct", 0) > 0])
        show(f"gap_pct > -2%", [s for s in c17 if s.get("gap_pct", 0) > -2])
        show(f"pct_above_sma20 > 0", [s for s in c17 if s.get("pct_above_sma20", 0) > 0])
        show(f"pct_above_sma20 > 5", [s for s in c17 if s.get("pct_above_sma20", 0) > 5])
        show(f"momentum_20d > 0", [s for s in c17 if s.get("momentum_20d", 0) > 0])
        show(f"momentum_20d > 10", [s for s in c17 if s.get("momentum_20d", 0) > 10])
        show(f"atr_pct < 5", [s for s in c17 if s.get("atr_pct", 99) < 5])
        show(f"atr_pct < 8", [s for s in c17 if s.get("atr_pct", 99) < 8])
        show(f"pct_off_52w < 20", [s for s in c17 if s.get("pct_off_52w", 99) < 20])
        show(f"pct_off_52w < 30", [s for s in c17 if s.get("pct_off_52w", 99) < 30])
        show(f"consolidation_pct < 15", [s for s in c17 if s.get("consolidation_pct", 99) < 15])
        show(f"consolidation_pct < 20", [s for s in c17 if s.get("consolidation_pct", 99) < 20])
        show(f"bull_candles_5d >= 3", [s for s in c17 if s.get("bull_candles_5d", 0) >= 3])
        show(f"bull_candles_5d >= 4", [s for s in c17 if s.get("bull_candles_5d", 0) >= 4])

        # ── Promising 2-filter combos ──
        print(f"\n  --- 2-filter combos ---")
        show("up>=2 + slope_5d>0", [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("slope_5d", 0) > 0])
        show("up>=2 + slope_5d>1", [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("slope_5d", 0) > 1])
        show("up>=2 + vol_accel>20", [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("vol_acceleration", 0) > 20])
        show("up>=2 + bb_pct_b>0.8", [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("bb_pct_b", 0) > 0.8])
        show("up>=2 + bb_above_upper", [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("bb_above_upper")])
        show("up>=2 + gap>0", [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("gap_pct", 0) > 0])
        show("up>=2 + above_sma20>0", [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("pct_above_sma20", 0) > 0])
        show("up>=2 + mom20d>0", [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("momentum_20d", 0) > 0])
        show("up>=2 + atr<8", [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("atr_pct", 99) < 8])
        show("up>=2 + pct_off_52w<30", [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("pct_off_52w", 99) < 30])
        show("up>=2 + consol<20", [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("consolidation_pct", 99) < 20])
        show("up>=2 + bull_candles>=3", [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("bull_candles_5d", 0) >= 3])
        show("slope_5d>1 + bb_pct_b>0.8", [s for s in c17 if s.get("slope_5d", 0) > 1 and s.get("bb_pct_b", 0) > 0.8])
        show("slope_5d>1 + vol_accel>20", [s for s in c17 if s.get("slope_5d", 0) > 1 and s.get("vol_acceleration", 0) > 20])
        show("slope_5d>1 + gap>0", [s for s in c17 if s.get("slope_5d", 0) > 1 and s.get("gap_pct", 0) > 0])
        show("slope_5d>0 + bb_above_upper", [s for s in c17 if s.get("slope_5d", 0) > 0 and s.get("bb_above_upper")])
        show("slope_5d>0 + vol_accel>50", [s for s in c17 if s.get("slope_5d", 0) > 0 and s.get("vol_acceleration", 0) > 50])
        show("bb_above_upper + vol_spike", [s for s in c17 if s.get("bb_above_upper") and s.get("vol_spike")])
        show("gap>0 + slope_5d>0", [s for s in c17 if s.get("gap_pct", 0) > 0 and s.get("slope_5d", 0) > 0])
        show("gap>0 + up>=1", [s for s in c17 if s.get("gap_pct", 0) > 0 and s.get("consec_up_days", 0) >= 1])
        show("gap>0 + up>=2", [s for s in c17 if s.get("gap_pct", 0) > 0 and s.get("consec_up_days", 0) >= 2])
        show("mom20d>10 + up>=1", [s for s in c17 if s.get("momentum_20d", 0) > 10 and s.get("consec_up_days", 0) >= 1])
        show("slope_10d>0 + up>=2", [s for s in c17 if s.get("slope_10d", 0) > 0 and s.get("consec_up_days", 0) >= 2])
        show("above_sma20>5 + up>=1", [s for s in c17 if s.get("pct_above_sma20", 0) > 5 and s.get("consec_up_days", 0) >= 1])

        # ── 3-filter combos ──
        print(f"\n  --- 3-filter combos ---")
        show("up>=2 + slope_5d>0 + gap>-2", [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("slope_5d", 0) > 0 and s.get("gap_pct", 0) > -2])
        show("up>=2 + slope_5d>0 + gap>0", [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("slope_5d", 0) > 0 and s.get("gap_pct", 0) > 0])
        show("up>=2 + slope_5d>1 + gap>-2", [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("slope_5d", 0) > 1 and s.get("gap_pct", 0) > -2])
        show("up>=2 + slope_5d>0 + bb_pct_b>0.5", [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("slope_5d", 0) > 0 and s.get("bb_pct_b", 0) > 0.5])
        show("up>=2 + slope_5d>0 + vol_accel>20", [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("slope_5d", 0) > 0 and s.get("vol_acceleration", 0) > 20])
        show("up>=2 + slope_5d>0 + above_sma20>0", [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("slope_5d", 0) > 0 and s.get("pct_above_sma20", 0) > 0])
        show("up>=2 + slope_5d>0 + mom20d>0", [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("slope_5d", 0) > 0 and s.get("momentum_20d", 0) > 0])
        show("up>=2 + slope_5d>0 + atr<8", [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("slope_5d", 0) > 0 and s.get("atr_pct", 99) < 8])
        show("up>=2 + slope_5d>0 + pct_off_52w<30", [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("slope_5d", 0) > 0 and s.get("pct_off_52w", 99) < 30])
        show("up>=2 + bb_above_upper + vol_accel>20", [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("bb_above_upper") and s.get("vol_acceleration", 0) > 20])
        show("up>=2 + gap>0 + vol_accel>20", [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("gap_pct", 0) > 0 and s.get("vol_acceleration", 0) > 20])
        show("up>=2 + gap>0 + slope_5d>0", [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("gap_pct", 0) > 0 and s.get("slope_5d", 0) > 0])
        show("up>=1 + slope_5d>1 + gap>0", [s for s in c17 if s.get("consec_up_days", 0) >= 1 and s.get("slope_5d", 0) > 1 and s.get("gap_pct", 0) > 0])
        show("up>=1 + slope_5d>0 + bb_pct_b>0.8", [s for s in c17 if s.get("consec_up_days", 0) >= 1 and s.get("slope_5d", 0) > 0 and s.get("bb_pct_b", 0) > 0.8])
        show("up>=1 + slope_5d>0 + vol_accel>20", [s for s in c17 if s.get("consec_up_days", 0) >= 1 and s.get("slope_5d", 0) > 0 and s.get("vol_acceleration", 0) > 20])
        show("up>=1 + slope_5d>1 + vol_accel>20", [s for s in c17 if s.get("consec_up_days", 0) >= 1 and s.get("slope_5d", 0) > 1 and s.get("vol_acceleration", 0) > 20])
        show("slope_5d>0 + bb_above_upper + vol_accel>20", [s for s in c17 if s.get("slope_5d", 0) > 0 and s.get("bb_above_upper") and s.get("vol_acceleration", 0) > 20])
        show("slope_5d>0 + gap>0 + bull_candles>=3", [s for s in c17 if s.get("slope_5d", 0) > 0 and s.get("gap_pct", 0) > 0 and s.get("bull_candles_5d", 0) >= 3])
        show("up>=1 + bb_pct_b>0.5 + vol_accel>20", [s for s in c17 if s.get("consec_up_days", 0) >= 1 and s.get("bb_pct_b", 0) > 0.5 and s.get("vol_acceleration", 0) > 20])

        # ── 4-filter combos ──
        print(f"\n  --- 4-filter combos ---")
        show("up>=2 + slope>0 + gap>-2 + bb>0.5", [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("slope_5d", 0) > 0 and s.get("gap_pct", 0) > -2 and s.get("bb_pct_b", 0) > 0.5])
        show("up>=2 + slope>0 + gap>-2 + vol_accel>20", [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("slope_5d", 0) > 0 and s.get("gap_pct", 0) > -2 and s.get("vol_acceleration", 0) > 20])
        show("up>=2 + slope>0 + gap>0 + bb>0.5", [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("slope_5d", 0) > 0 and s.get("gap_pct", 0) > 0 and s.get("bb_pct_b", 0) > 0.5])
        show("up>=2 + slope>0 + gap>0 + vol_accel>20", [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("slope_5d", 0) > 0 and s.get("gap_pct", 0) > 0 and s.get("vol_acceleration", 0) > 20])
        show("up>=1 + slope>1 + gap>0 + bb>0.5", [s for s in c17 if s.get("consec_up_days", 0) >= 1 and s.get("slope_5d", 0) > 1 and s.get("gap_pct", 0) > 0 and s.get("bb_pct_b", 0) > 0.5])
        show("up>=1 + slope>1 + gap>0 + vol_accel>20", [s for s in c17 if s.get("consec_up_days", 0) >= 1 and s.get("slope_5d", 0) > 1 and s.get("gap_pct", 0) > 0 and s.get("vol_acceleration", 0) > 20])
        show("up>=2 + slope>0 + above_sma20>0 + gap>-2", [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("slope_5d", 0) > 0 and s.get("pct_above_sma20", 0) > 0 and s.get("gap_pct", 0) > -2])
        show("up>=2 + slope>0 + pct_off_52w<30 + gap>-2", [s for s in c17 if s.get("consec_up_days", 0) >= 2 and s.get("slope_5d", 0) > 0 and s.get("pct_off_52w", 99) < 30 and s.get("gap_pct", 0) > -2])

        # ── Show individual trades for any combo with >=50% WR and n>=3 ──
        print(f"\n  --- Trades for best combos (WR >= 50%, n >= 3) ---")

        combos_to_detail = [
            ("up>=2 + slope>0 + gap>-2", lambda s: s.get("consec_up_days", 0) >= 2 and s.get("slope_5d", 0) > 0 and s.get("gap_pct", 0) > -2),
            ("up>=2 + slope>0 + gap>0", lambda s: s.get("consec_up_days", 0) >= 2 and s.get("slope_5d", 0) > 0 and s.get("gap_pct", 0) > 0),
            ("up>=2 + slope>0", lambda s: s.get("consec_up_days", 0) >= 2 and s.get("slope_5d", 0) > 0),
            ("up>=2 + gap>0", lambda s: s.get("consec_up_days", 0) >= 2 and s.get("gap_pct", 0) > 0),
            ("up>=2 + bb_above_upper", lambda s: s.get("consec_up_days", 0) >= 2 and s.get("bb_above_upper")),
            ("up>=2 + vol_accel>20", lambda s: s.get("consec_up_days", 0) >= 2 and s.get("vol_acceleration", 0) > 20),
            ("up>=2 + slope>0 + bb>0.5", lambda s: s.get("consec_up_days", 0) >= 2 and s.get("slope_5d", 0) > 0 and s.get("bb_pct_b", 0) > 0.5),
            ("up>=1 + slope>1 + gap>0", lambda s: s.get("consec_up_days", 0) >= 1 and s.get("slope_5d", 0) > 1 and s.get("gap_pct", 0) > 0),
            ("up>=3", lambda s: s.get("consec_up_days", 0) >= 3),
            ("slope>2 + up>=1", lambda s: s.get("slope_5d", 0) > 2 and s.get("consec_up_days", 0) >= 1),
        ]

        for label, filt_fn in combos_to_detail:
            filtered = [s for s in c17 if filt_fn(s)]
            rets = [s["return_15d"] for s in filtered if s["return_15d"] is not None]
            if len(rets) < 3:
                continue
            wr = sum(1 for r in rets if r > 0) / len(rets) * 100
            if wr < 50:
                continue
            print(f"\n  >> {label}  (WR={wr:.0f}%, n={len(filtered)})")
            for s in sorted(filtered, key=lambda x: x["trend_date"]):
                r = s["return_15d"] if s["return_15d"] else 0
                wl = "W" if r > 0 else "L"
                print(f"      {s['ticker']:7s} {s['trend_date']}  {r:>+7.1f}%  {wl}  "
                      f"updays={s['consec_up_days']}  slope5d={s['slope_5d']:+.2f}  "
                      f"gap={s['gap_pct']:+.2f}%  bb={s['bb_pct_b']:.2f}  "
                      f"vol_accel={s['vol_acceleration']:+.0f}%")


if __name__ == "__main__":
    run_all()
