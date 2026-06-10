"""
Minimum 5-day hold + trailing exit strategies for C17 signals.

Hypothesis: the tight stop kicks trades out before the move develops (~day 13 peak).
A min-5-day hold gives the trade room to breathe, then a trailing exit captures
the peak without holding all the way to 30d where losers expand.

Strategies (all: no exit before day 5, then trailing kicks in):
  baselines  - time_5, time_10, time_15, time_30  (no stop, for reference)
  min5+atr   - ATR trailing at 0.5x / 1x / 2x  after day 5
  min5+sma   - exit when close < 5/10/20-day SMA  after day 5
  min5+ma    - exit when fast MA crosses slow MA   after day 5 (5/10, 5/20, 10/20)
  min5+peak  - exit when return drops > X% from peak  (trailing % drawdown on return)
"""

import sys
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, median

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import DB_PATH, ScannerConfig, BacktestConfig
from core.database import Database
from core.backtest import Backtester
from analysis_winners_losers import enrich_deep, filt_c17_both as filt_c17
from analysis_exit_strategies import get_price_df, compute_atr, trade_result

MAX_HOLD = 30
MIN_HOLD = 5


# ── Exit implementations (all enforce MIN_HOLD before checking) ───────────────

def sim_time(df, entry_idx, ep, direction, hold_days):
    i = min(entry_idx + hold_days, len(df) - 1)
    return {"return": trade_result(ep, df.iloc[i]["close"], direction),
            "hold": i - entry_idx, "exit": "horizon"}


def sim_min5_atr(df, atr_series, entry_idx, ep, direction, atr_mult):
    trail_anchor = ep
    for offset in range(1, MAX_HOLD + 1):
        i = entry_idx + offset
        if i >= len(df):
            break
        close = df.iloc[i]["close"]
        if direction == "bullish":
            trail_anchor = max(trail_anchor, close)
        else:
            trail_anchor = min(trail_anchor, close)

        if offset >= MIN_HOLD:
            atr_val = atr_series.iloc[i - 1] if i - 1 < len(atr_series) else None
            if atr_val is None or pd.isna(atr_val):
                atr_val = (df["high"].iloc[max(0, i - 14):i].mean() -
                           df["low"].iloc[max(0, i - 14):i].mean())
            if direction == "bullish" and close < trail_anchor - atr_mult * atr_val:
                return {"return": trade_result(ep, close, direction),
                        "hold": offset, "exit": "stop"}
            if direction == "bearish" and close > trail_anchor + atr_mult * atr_val:
                return {"return": trade_result(ep, close, direction),
                        "hold": offset, "exit": "stop"}

    last = df.iloc[min(entry_idx + MAX_HOLD, len(df) - 1)]
    return {"return": trade_result(ep, last["close"], direction),
            "hold": MAX_HOLD, "exit": "horizon"}


def sim_min5_sma(df, entry_idx, ep, direction, period):
    df = df.copy()
    df[f"sma"] = df["close"].rolling(period).mean()
    for offset in range(1, MAX_HOLD + 1):
        i = entry_idx + offset
        if i >= len(df):
            break
        sma = df["sma"].iloc[i]
        close = df["close"].iloc[i]
        if pd.isna(sma) or offset < MIN_HOLD:
            continue
        if direction == "bullish" and close < sma:
            return {"return": trade_result(ep, close, direction),
                    "hold": offset, "exit": "stop"}
        if direction == "bearish" and close > sma:
            return {"return": trade_result(ep, close, direction),
                    "hold": offset, "exit": "stop"}
    last = df.iloc[min(entry_idx + MAX_HOLD, len(df) - 1)]
    return {"return": trade_result(ep, last["close"], direction),
            "hold": MAX_HOLD, "exit": "horizon"}


def sim_min5_ma_cross(df, entry_idx, ep, direction, fast, slow):
    df = df.copy()
    df["fast_ma"] = df["close"].rolling(fast).mean()
    df["slow_ma"] = df["close"].rolling(slow).mean()
    for offset in range(1, MAX_HOLD + 1):
        i = entry_idx + offset
        if i >= len(df) or i < 1 or offset < MIN_HOLD:
            continue
        fp, sp = df["fast_ma"].iloc[i - 1], df["slow_ma"].iloc[i - 1]
        fc, sc = df["fast_ma"].iloc[i],     df["slow_ma"].iloc[i]
        if any(pd.isna(v) for v in [fp, sp, fc, sc]):
            continue
        if direction == "bullish" and fp >= sp and fc < sc:
            close = df["close"].iloc[i]
            return {"return": trade_result(ep, close, direction),
                    "hold": offset, "exit": "stop"}
        if direction == "bearish" and fp <= sp and fc > sc:
            close = df["close"].iloc[i]
            return {"return": trade_result(ep, close, direction),
                    "hold": offset, "exit": "stop"}
    last = df.iloc[min(entry_idx + MAX_HOLD, len(df) - 1)]
    return {"return": trade_result(ep, last["close"], direction),
            "hold": MAX_HOLD, "exit": "horizon"}


def sim_min5_peak_drawdown(df, entry_idx, ep, direction, drawdown_pct):
    """Exit when return drops more than drawdown_pct from its peak (after day 5)."""
    peak_ret = None
    for offset in range(1, MAX_HOLD + 1):
        i = entry_idx + offset
        if i >= len(df):
            break
        close = df.iloc[i]["close"]
        ret = trade_result(ep, close, direction)
        if peak_ret is None or ret > peak_ret:
            peak_ret = ret
        if offset >= MIN_HOLD and peak_ret is not None:
            if ret < peak_ret - drawdown_pct:
                return {"return": ret, "hold": offset, "exit": "stop"}
    last = df.iloc[min(entry_idx + MAX_HOLD, len(df) - 1)]
    return {"return": trade_result(ep, last["close"], direction),
            "hold": MAX_HOLD, "exit": "horizon"}


# ── Stats & display ───────────────────────────────────────────────────────────

def stats(results):
    returns = [r["return"] for r in results if r]
    if not returns:
        return None
    winners = [r for r in returns if r > 0]
    losers  = [r for r in returns if r <= 0]
    n = len(returns)
    wr = len(winners) / n * 100
    avg_w = mean(winners) if winners else 0
    avg_l = mean(losers) if losers else 0
    gross_l = abs(sum(r for r in losers if r < 0))
    pf = sum(winners) / gross_l if gross_l > 0 else float("inf")
    exp = (len(winners) / n) * avg_w + (len(losers) / n) * avg_l
    avg_hold = mean(r["hold"] for r in results if r)
    stop_pct = sum(1 for r in results if r and r["exit"] == "stop") / n * 100
    return {"n": n, "wr": wr, "pf": pf, "exp": exp,
            "avg_w": avg_w, "avg_l": avg_l, "med": median(returns),
            "avg_hold": avg_hold, "stop_pct": stop_pct}


def print_table(all_stats):
    hdr = f"{'Strategy':<18} {'n':>4} {'WR%':>6} {'PF':>6} {'Exp%':>6} {'AvgW%':>7} {'AvgL%':>7} {'Med%':>7} {'Hold':>5} {'Stop%':>6}"
    print(hdr)
    print("-" * len(hdr))
    for name, s in all_stats.items():
        if s is None:
            print(f"{name:<18}  no data")
            continue
        pf_str = f"{s['pf']:.2f}" if s['pf'] != float("inf") else "  inf"
        print(f"{name:<18} {s['n']:>4} {s['wr']:>6.1f} {pf_str:>6} {s['exp']:>6.2f}"
              f" {s['avg_w']:>7.1f} {s['avg_l']:>7.1f} {s['med']:>7.2f}"
              f" {s['avg_hold']:>5.1f} {s['stop_pct']:>6.1f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    db = Database(DB_PATH)
    sc = ScannerConfig()
    bc = BacktestConfig(start_date="2022-01-01", end_date="2026-06-09")

    print("Running backtest and applying C17 filter...")
    bt = Backtester(db, sc, bc)
    results = bt.run()
    enriched = enrich_deep(db, results["signals"])
    c17 = filt_c17(enriched)
    print(f"C17 signals: {len(c17)}\n")

    strategies = {
        # Reference baselines
        "time_5":          [],
        "time_10":         [],
        "time_15":         [],
        "time_30":         [],
        # Min-5 + ATR trailing
        "min5+atr_0.5x":   [],
        "min5+atr_1x":     [],
        "min5+atr_2x":     [],
        # Min-5 + SMA support
        "min5+sma5":       [],
        "min5+sma10":      [],
        "min5+sma20":      [],
        # Min-5 + MA cross
        "min5+ma_5_10":    [],
        "min5+ma_5_20":    [],
        "min5+ma_10_20":   [],
        # Min-5 + peak drawdown trailing
        "min5+peak_3%":    [],
        "min5+peak_5%":    [],
        "min5+peak_8%":    [],
        "min5+peak_10%":   [],
    }

    skipped = 0
    for sig in c17:
        ticker    = sig["ticker"]
        signal_date = sig["trend_date"]
        direction = sig["signal_type"]

        df = get_price_df(db, ticker, signal_date)
        if df is None or len(df) < 30:
            skipped += 1
            continue

        signal_dt = pd.Timestamp(signal_date)
        after = df[df["date"] > signal_dt]
        if after.empty:
            skipped += 1
            continue
        entry_idx = after.index[0]
        entry_row = df.iloc[entry_idx]
        ep = (entry_row["high"] + entry_row["low"]) / 2
        if ep <= 0:
            skipped += 1
            continue

        atr_series = compute_atr(df)

        strategies["time_5"].append(sim_time(df, entry_idx, ep, direction, 5))
        strategies["time_10"].append(sim_time(df, entry_idx, ep, direction, 10))
        strategies["time_15"].append(sim_time(df, entry_idx, ep, direction, 15))
        strategies["time_30"].append(sim_time(df, entry_idx, ep, direction, 30))

        strategies["min5+atr_0.5x"].append(sim_min5_atr(df, atr_series, entry_idx, ep, direction, 0.5))
        strategies["min5+atr_1x"].append(sim_min5_atr(df, atr_series, entry_idx, ep, direction, 1.0))
        strategies["min5+atr_2x"].append(sim_min5_atr(df, atr_series, entry_idx, ep, direction, 2.0))

        strategies["min5+sma5"].append(sim_min5_sma(df, entry_idx, ep, direction, 5))
        strategies["min5+sma10"].append(sim_min5_sma(df, entry_idx, ep, direction, 10))
        strategies["min5+sma20"].append(sim_min5_sma(df, entry_idx, ep, direction, 20))

        strategies["min5+ma_5_10"].append(sim_min5_ma_cross(df, entry_idx, ep, direction, 5, 10))
        strategies["min5+ma_5_20"].append(sim_min5_ma_cross(df, entry_idx, ep, direction, 5, 20))
        strategies["min5+ma_10_20"].append(sim_min5_ma_cross(df, entry_idx, ep, direction, 10, 20))

        strategies["min5+peak_3%"].append(sim_min5_peak_drawdown(df, entry_idx, ep, direction, 3))
        strategies["min5+peak_5%"].append(sim_min5_peak_drawdown(df, entry_idx, ep, direction, 5))
        strategies["min5+peak_8%"].append(sim_min5_peak_drawdown(df, entry_idx, ep, direction, 8))
        strategies["min5+peak_10%"].append(sim_min5_peak_drawdown(df, entry_idx, ep, direction, 10))

    if skipped:
        print(f"(Skipped {skipped} signals — insufficient price data)\n")

    all_stats = {name: stats(res) for name, res in strategies.items()}

    print("-- Baselines (no stop) --")
    print_table({k: all_stats[k] for k in ["time_5", "time_10", "time_15", "time_30"]})
    print()
    print("-- Min-5 hold + ATR trailing --")
    print_table({k: all_stats[k] for k in ["min5+atr_0.5x", "min5+atr_1x", "min5+atr_2x"]})
    print()
    print("-- Min-5 hold + SMA support --")
    print_table({k: all_stats[k] for k in ["min5+sma5", "min5+sma10", "min5+sma20"]})
    print()
    print("-- Min-5 hold + MA cross --")
    print_table({k: all_stats[k] for k in ["min5+ma_5_10", "min5+ma_5_20", "min5+ma_10_20"]})
    print()
    print("-- Min-5 hold + peak drawdown trailing --")
    print_table({k: all_stats[k] for k in ["min5+peak_3%", "min5+peak_5%", "min5+peak_8%", "min5+peak_10%"]})
    print()

    best_pf  = max(((k, s) for k, s in all_stats.items() if s and s["pf"] != float("inf")), key=lambda x: x[1]["pf"])
    best_exp = max(((k, s) for k, s in all_stats.items() if s), key=lambda x: x[1]["exp"])
    print(f"Best profit factor : {best_pf[0]}  PF={best_pf[1]['pf']:.2f}  Exp={best_pf[1]['exp']:.2f}%  Hold={best_pf[1]['avg_hold']:.1f}d")
    print(f"Best expectancy    : {best_exp[0]}  PF={best_exp[1]['pf']:.2f}  Exp={best_exp[1]['exp']:.2f}%  Hold={best_exp[1]['avg_hold']:.1f}d")


if __name__ == "__main__":
    main()
