"""Test realistic entry: enter at Day 1 close (last hour) if green candle confirmed.

Rules:
  - Signal: MA crossover on Day 0 (detected after close)
  - Confirmation: Day 1 must close green (close > open)
  - Entry: Day 1 close price (entered in last hour of trading)
  - Emergency stop: If Day 1 close < prior day open, exit Day 2 at open
    (UNLESS Day 2 opens above prior day open — then hold)
  - Normal stop Day 2+: trailing at highest_close - 0.5x ATR

Also tests "close > prior close" as the confirmation filter.
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
from analysis_winners_losers import enrich_deep, filt_c17


def compute_atr(df, period=14):
    d = df.copy()
    d["prev_close"] = d["close"].shift(1)
    d["tr"] = d.apply(lambda r: max(
        r["high"] - r["low"],
        abs(r["high"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
        abs(r["low"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
    ), axis=1)
    return d["tr"].rolling(period).mean()


def simulate_realistic(
    db, ticker, signal_date, direction,
    confirm_mode="green_candle",  # "green_candle" or "close_above_prior"
    atr_mult=0.5,
    max_days=15,
):
    """
    Realistic trade simulation:
      Day 0: signal (MA crossover after close)
      Day 1: confirmation day — enter at close if confirmation passes
      Day 2+: trailing ATR stop

    Emergency rule: if Day 1 close < Day 0 open, exit Day 2 at open
    UNLESS Day 2 opens >= Day 0 open (then hold, normal stop applies).
    """
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

    signal_idx_list = df.index[df["date"] == signal_date].tolist()
    if not signal_idx_list:
        return None
    signal_idx = signal_idx_list[0]

    d0_row = df.iloc[signal_idx]  # signal day
    d1_idx = signal_idx + 1
    if d1_idx >= len(df):
        return None

    d1_row = df.iloc[d1_idx]  # confirmation day

    # ── Confirmation check ──
    if confirm_mode == "green_candle":
        if d1_row["close"] <= d1_row["open"]:
            return None  # not a green candle
    elif confirm_mode == "close_above_prior":
        if d1_row["close"] <= d0_row["close"]:
            return None  # didn't close above prior
    elif confirm_mode == "none":
        pass  # no filter

    # ── Entry at Day 1 close ──
    entry_price = d1_row["close"]
    if entry_price <= 0:
        return None

    # ── Check emergency stop: Day 1 close < Day 0 open ──
    d0_open = d0_row["open"]

    d2_idx = d1_idx + 1
    if d2_idx >= len(df):
        # Can't check day 2, just return the entry as a trade held to end
        return None

    d2_row = df.iloc[d2_idx]

    if direction == "bullish" and entry_price < d0_open:
        # Emergency condition: entered but close is below prior day open
        # Exit at Day 2 open UNLESS Day 2 opens above Day 0 open
        if d2_row["open"] < d0_open:
            # Exit at Day 2 open
            exit_price = d2_row["open"]
            ret = (exit_price - entry_price) / entry_price * 100
            return {
                "return": ret, "hold_days": 1, "exit_reason": "emergency_stop",
                "entry_price": entry_price, "exit_price": exit_price,
                "ticker": ticker, "signal_date": signal_date,
                "entry_date": d1_row["date"], "exit_date": d2_row["date"],
            }
        # else: Day 2 opened above d0_open, hold — fall through to normal stop logic

    # ── Walk forward from Day 2 with ATR trailing stop ──
    highest_close = max(entry_price, d1_row["close"])

    for day_offset in range(2, max_days + 2):
        cur_idx = d1_idx + day_offset
        if cur_idx >= len(df):
            break

        row = df.iloc[cur_idx]
        close = row["close"]
        hold_days = day_offset  # days after entry

        # ATR trailing stop
        if atr_mult is not None:
            atr_val = atr_series.iloc[cur_idx - 1] if cur_idx - 1 < len(atr_series) else None
            if atr_val is None or pd.isna(atr_val):
                atr_val = (df["high"].iloc[max(0, cur_idx-14):cur_idx].mean() -
                           df["low"].iloc[max(0, cur_idx-14):cur_idx].mean())
            stop_level = highest_close - atr_mult * atr_val
            if direction == "bullish" and close < stop_level:
                ret = (close - entry_price) / entry_price * 100
                return {
                    "return": ret, "hold_days": hold_days, "exit_reason": "atr_stop",
                    "entry_price": entry_price, "exit_price": close,
                    "ticker": ticker, "signal_date": signal_date,
                    "entry_date": d1_row["date"], "exit_date": row["date"],
                }

        if close > highest_close:
            highest_close = close

    # Held to horizon
    last_idx = min(d1_idx + max_days + 1, len(df) - 1)
    exit_price = df.iloc[last_idx]["close"]
    ret = (exit_price - entry_price) / entry_price * 100
    return {
        "return": ret, "hold_days": max_days, "exit_reason": "horizon",
        "entry_price": entry_price, "exit_price": exit_price,
        "ticker": ticker, "signal_date": signal_date,
        "entry_date": d1_row["date"], "exit_date": df.iloc[last_idx]["date"],
    }


def run_sim(db, signals, raw_signals, **kwargs):
    results = []
    for s in signals:
        match = None
        for rs in raw_signals:
            if rs["ticker"] == s["ticker"] and rs.get("trend_change_date") == s["trend_date"]:
                match = rs
                break
        if not match:
            continue

        trade = simulate_realistic(db, s["ticker"], s["trend_date"], "bullish", **kwargs)
        if trade:
            trade["ticker"] = s["ticker"]
            trade["eps_change_pct"] = s.get("eps_change_pct", 0)
            results.append(trade)
    return results


def show(label, results, show_trades=False):
    if not results:
        print(f"  {label:75s} | n=  0")
        return

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

    # Count exit reasons
    reasons = {}
    for r in results:
        reasons[r["exit_reason"]] = reasons.get(r["exit_reason"], 0) + 1
    reason_str = ", ".join(f"{k}={v}" for k, v in sorted(reasons.items()))

    print(f"  {label:75s} | n={len(results):>3d} | WR={wr:>5.1f}% | Avg={avg:>+7.1f}% "
          f"| AvgW={avg_w:>+6.1f}% | AvgL={avg_l:>+6.1f}% | PF={pf_s:>6s} | Hold={hold:.1f}d")
    if reason_str:
        print(f"  {'':75s}   exits: {reason_str}")

    if show_trades:
        for r in sorted(results, key=lambda x: x.get("entry_date", x["signal_date"])):
            wl = "W" if r["return"] > 0 else "L"
            print(f"      {r['ticker']:7s} sig={r['signal_date']}  entry={r.get('entry_date','?')}  "
                  f"exit={r.get('exit_date','?')}  ${r['entry_price']:>7.2f} -> ${r['exit_price']:>7.2f}  "
                  f"{r['return']:>+7.1f}%  {wl}  hold={r['hold_days']:>2d}d  {r['exit_reason']}")


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

    # ═══════════════════════════════════════════════════════════════════
    print("=" * 160)
    print("  REALISTIC ENTRY: Last-hour entry at Day 1 close + emergency stop if close < prior open")
    print("  Emergency rule: if entry close < Day 0 open -> exit Day 2 open (unless Day 2 opens above Day 0 open)")
    print("=" * 160)

    for eps_thresh, tw in configs:
        c17, raw = signal_cache[(eps_thresh, tw)]
        if len(c17) < 3:
            continue

        print(f"\n{'='*160}")
        print(f"  EPS>={eps_thresh:.0f}% / {tw}d window — {len(c17)} C17 signals")
        print(f"{'='*160}")

        scenarios = [
            ("BASELINE: no confirmation filter, enter at d1 close",
             {"confirm_mode": "none"}),
            ("Green candle (close>open) + enter at d1 close + emergency stop",
             {"confirm_mode": "green_candle"}),
            ("Close > prior close + enter at d1 close + emergency stop",
             {"confirm_mode": "close_above_prior"}),
        ]

        for label, kwargs in scenarios:
            results = run_sim(db, c17, raw, atr_mult=0.5, max_days=15, **kwargs)
            show(label, results)

    # ═══════════════════════════════════════════════════════════════════
    # Test ATR multiplier variants with the realistic setup
    print(f"\n\n{'='*160}")
    print("  ATR MULTIPLIER SWEEP with realistic entry (green candle + d1 close + emergency stop)")
    print(f"{'='*160}")

    for eps_thresh, tw in [(25.0, 30), (25.0, 60), (20.0, 60)]:
        c17, raw = signal_cache[(eps_thresh, tw)]
        if len(c17) < 3:
            continue

        print(f"\n  --- EPS>={eps_thresh:.0f}% / {tw}d ---")
        for mult in [0.25, 0.3, 0.4, 0.5, 0.75, 1.0]:
            results = run_sim(db, c17, raw, confirm_mode="green_candle", atr_mult=mult, max_days=15)
            show(f"Green candle + {mult:.2f}x ATR", results)

        print()
        for mult in [0.25, 0.3, 0.4, 0.5, 0.75, 1.0]:
            results = run_sim(db, c17, raw, confirm_mode="close_above_prior", atr_mult=mult, max_days=15)
            show(f"Close>prior + {mult:.2f}x ATR", results)

    # ═══════════════════════════════════════════════════════════════════
    # Individual trade details
    print(f"\n\n{'='*160}")
    print("  INDIVIDUAL TRADES")
    print(f"{'='*160}")

    for eps_thresh, tw in [(25.0, 30), (20.0, 60), (30.0, 60)]:
        c17, raw = signal_cache[(eps_thresh, tw)]
        if len(c17) < 3:
            continue

        for confirm in ["green_candle", "close_above_prior"]:
            label = f"EPS>={eps_thresh:.0f}%/{tw}d — {confirm} — d1_close + emergency stop"
            print(f"\n  >> {label}")
            results = run_sim(db, c17, raw, confirm_mode=confirm, atr_mult=0.5, max_days=15)
            show(label, results, show_trades=True)


if __name__ == "__main__":
    main()
