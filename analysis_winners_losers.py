"""
Deep analysis of what differentiates winners from losers in C17 and C22.
Examines: RSI, MACD, candlestick patterns, trendlines, volume patterns,
Bollinger Bands, support/resistance, and pre-signal price action.
"""

import sys
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, median, stdev

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import DB_PATH, BacktestConfig, ScannerConfig
from core.backtest import Backtester
from core.database import Database


def get_db():
    db = Database(DB_PATH)
    db.initialize()
    return db


def run_backtest(db):
    sc = ScannerConfig(
        min_price=1.0, max_price=50.0,
        min_market_cap=50_000_000, max_market_cap=10_000_000_000,
        ma_crossover_pairs=[(20, 50)],
        eps_change_threshold=10.0, trend_window_days=30, direction="both",
    )
    bc = BacktestConfig(
        start_date="2022-01-01", end_date="2026-03-19",
        forward_return_days=[5, 10, 15, 30, 60],
        ma_crossover_pairs=[(20, 50)], eps_thresholds=[10.0], trend_windows=[30],
    )
    return Backtester(db, sc, bc).run()


def compute_rsi(closes, period=14):
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = pd.Series(gains).rolling(period).mean().iloc[-1]
    avg_loss = pd.Series(losses).rolling(period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(closes):
    s = pd.Series(closes)
    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return {
        "macd": macd_line.iloc[-1],
        "signal": signal_line.iloc[-1],
        "histogram": histogram.iloc[-1],
        "macd_above_signal": macd_line.iloc[-1] > signal_line.iloc[-1],
        "histogram_positive": histogram.iloc[-1] > 0,
        "histogram_rising": len(histogram) >= 2 and histogram.iloc[-1] > histogram.iloc[-2],
    }


def compute_bollinger(closes, period=20):
    s = pd.Series(closes)
    sma = s.rolling(period).mean().iloc[-1]
    std = s.rolling(period).std().iloc[-1]
    upper = sma + 2 * std
    lower = sma - 2 * std
    last = closes[-1]
    pct_b = (last - lower) / (upper - lower) if (upper - lower) > 0 else 0.5
    bandwidth = (upper - lower) / sma * 100 if sma > 0 else 0
    return {
        "bb_pct_b": pct_b,
        "bb_bandwidth": bandwidth,
        "above_upper": last > upper,
        "below_lower": last < lower,
        "near_middle": abs(last - sma) < 0.5 * std,
    }


def analyze_candlesticks(df_last5):
    """Analyze last 5 candles for patterns."""
    patterns = []
    for i in range(len(df_last5)):
        row = df_last5.iloc[i]
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        body = abs(c - o)
        total_range = h - l if h > l else 0.001
        body_pct = body / total_range * 100

        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        if body_pct < 10:
            patterns.append("doji")
        elif c > o and lower_wick > 2 * body and upper_wick < body * 0.5:
            patterns.append("hammer")
        elif c < o and upper_wick > 2 * body and lower_wick < body * 0.5:
            patterns.append("shooting_star")
        elif c > o and body_pct > 60:
            patterns.append("strong_bull")
        elif c < o and body_pct > 60:
            patterns.append("strong_bear")
        elif c > o:
            patterns.append("bull")
        else:
            patterns.append("bear")

    # Summary patterns
    result = {
        "last_candle": patterns[-1] if patterns else "unknown",
        "bull_candles": sum(1 for p in patterns if "bull" in p or p == "hammer"),
        "bear_candles": sum(1 for p in patterns if "bear" in p or p == "shooting_star"),
        "doji_count": sum(1 for p in patterns if p == "doji"),
        "pattern_sequence": "_".join(patterns[-3:]) if len(patterns) >= 3 else "_".join(patterns),
    }

    # Engulfing pattern (last 2 candles)
    if len(df_last5) >= 2:
        prev = df_last5.iloc[-2]
        curr = df_last5.iloc[-1]
        if prev["close"] < prev["open"] and curr["close"] > curr["open"]:
            if curr["close"] > prev["open"] and curr["open"] < prev["close"]:
                result["engulfing"] = "bullish_engulfing"
            else:
                result["engulfing"] = "none"
        elif prev["close"] > prev["open"] and curr["close"] < curr["open"]:
            if curr["open"] > prev["close"] and curr["close"] < prev["open"]:
                result["engulfing"] = "bearish_engulfing"
            else:
                result["engulfing"] = "none"
        else:
            result["engulfing"] = "none"
    else:
        result["engulfing"] = "unknown"

    return result


def compute_trendline_slope(closes, period=10):
    """Linear regression slope of last N closes."""
    if len(closes) < period:
        return 0
    y = np.array(closes[-period:])
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    # Normalize as % per day
    avg_price = np.mean(y)
    return (slope / avg_price * 100) if avg_price > 0 else 0


def analyze_volume_pattern(volumes, period=20):
    """Volume trend and acceleration."""
    if len(volumes) < period:
        return {"vol_trend": 0, "vol_acceleration": 0, "vol_spike": False}
    recent = volumes[-5:]
    baseline = volumes[-period:-5] if len(volumes) > 5 else volumes
    avg_recent = mean(recent)
    avg_baseline = mean(baseline) if baseline else 1
    vol_trend = (avg_recent / avg_baseline - 1) * 100 if avg_baseline > 0 else 0

    # Is latest volume a spike (>2x avg)?
    vol_spike = recent[-1] > 2 * avg_baseline if avg_baseline > 0 else False

    # Volume acceleration (is volume increasing day over day?)
    if len(recent) >= 3:
        vol_accel = (recent[-1] - recent[-3]) / recent[-3] * 100 if recent[-3] > 0 else 0
    else:
        vol_accel = 0

    return {"vol_trend": vol_trend, "vol_acceleration": vol_accel, "vol_spike": vol_spike}


def enrich_deep(db, signals):
    """Add deep technical features to each signal."""
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
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        closes = df["close"].tolist()
        volumes = df["volume"].tolist()

        # RSI
        rsi_14 = compute_rsi(closes, 14) if len(closes) > 15 else 50

        # MACD
        macd_data = compute_macd(closes) if len(closes) > 26 else {
            "macd": 0, "signal": 0, "histogram": 0,
            "macd_above_signal": False, "histogram_positive": False, "histogram_rising": False,
        }

        # Bollinger Bands
        bb_data = compute_bollinger(closes) if len(closes) >= 20 else {
            "bb_pct_b": 0.5, "bb_bandwidth": 0, "above_upper": False,
            "below_lower": False, "near_middle": True,
        }

        # Candlestick patterns (last 5 candles)
        candle_data = analyze_candlesticks(df.tail(5))

        # Trendline slopes
        slope_5d = compute_trendline_slope(closes, 5)
        slope_10d = compute_trendline_slope(closes, 10)
        slope_20d = compute_trendline_slope(closes, 20)

        # Volume patterns
        vol_data = analyze_volume_pattern(volumes)

        # Price relative to MAs
        sma_20 = mean(closes[-20:]) if len(closes) >= 20 else closes[-1]
        sma_50 = mean(closes[-50:]) if len(closes) >= 50 else closes[-1]
        pct_above_sma20 = (closes[-1] - sma_20) / sma_20 * 100 if sma_20 > 0 else 0
        pct_above_sma50 = (closes[-1] - sma_50) / sma_50 * 100 if sma_50 > 0 else 0

        # ATR
        df["prev_close"] = df["close"].shift(1)
        df["tr"] = df.apply(lambda r: max(
            r["high"] - r["low"],
            abs(r["high"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
            abs(r["low"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
        ), axis=1)
        atr_14 = df["tr"].tail(14).mean()
        atr_pct = (atr_14 / entry_price * 100) if entry_price > 0 else 0

        # Recent high/low range (consolidation measure)
        last_10_high = df["high"].tail(10).max()
        last_10_low = df["low"].tail(10).min()
        consolidation_pct = ((last_10_high - last_10_low) / last_10_low * 100) if last_10_low > 0 else 0

        # Gap from previous close to signal day open
        if len(df) >= 2:
            prev_close = df["close"].iloc[-2]
            signal_open = df["open"].iloc[-1]
            gap_pct = (signal_open - prev_close) / prev_close * 100 if prev_close > 0 else 0
        else:
            gap_pct = 0

        # Days since last earnings
        days_between = s.get("days_between", 0)

        # 52w high proximity
        year_prices = df.tail(252) if len(df) >= 252 else df
        high_52w = year_prices["high"].max()
        pct_off_52w = ((high_52w - entry_price) / high_52w * 100) if high_52w > 0 else 0

        # 20d momentum
        if len(closes) >= 21:
            momentum_20d = (closes[-1] - closes[-21]) / closes[-21] * 100 if closes[-21] > 0 else 0
        else:
            momentum_20d = 0

        # Dollar volume
        last_20 = df.tail(20)
        avg_dollar_vol = (last_20["volume"] * last_20["close"]).mean()

        # Consecutive up/down days before signal
        up_days = 0
        for i in range(len(closes) - 1, 0, -1):
            if closes[i] > closes[i-1]:
                up_days += 1
            else:
                break
        down_days = 0
        for i in range(len(closes) - 1, 0, -1):
            if closes[i] < closes[i-1]:
                down_days += 1
            else:
                break

        # Return for this trade (15d horizon)
        ret_15 = s.get("forward_returns", {}).get(15)
        ret_5 = s.get("forward_returns", {}).get(5)
        td_15 = s.get("trade_details", {}).get(15, {})

        # Hold time
        hold_days = 0
        if entry_date and td_15.get("exit_date"):
            try:
                hold_days = (date.fromisoformat(td_15["exit_date"]) - date.fromisoformat(entry_date)).days
            except:
                pass

        enriched.append({
            "ticker": ticker,
            "trend_date": trend_date,
            "entry_date": entry_date,
            "entry_price": entry_price,
            "signal_type": s.get("signal_type"),
            "eps_change_pct": s.get("eps_change_pct", 0),
            "days_between": days_between,
            "return_15d": ret_15,
            "return_5d": ret_5,
            "hold_days": hold_days,
            "exit_reason": td_15.get("exit_reason"),
            "is_winner": ret_15 is not None and ret_15 > 0,
            # Technical indicators
            "rsi_14": round(rsi_14, 1),
            "macd_histogram": round(macd_data["histogram"], 4),
            "macd_above_signal": macd_data["macd_above_signal"],
            "macd_hist_positive": macd_data["histogram_positive"],
            "macd_hist_rising": macd_data["histogram_rising"],
            "bb_pct_b": round(bb_data["bb_pct_b"], 3),
            "bb_bandwidth": round(bb_data["bb_bandwidth"], 1),
            "bb_above_upper": bb_data["above_upper"],
            "bb_below_lower": bb_data["below_lower"],
            # Candlesticks
            "last_candle": candle_data["last_candle"],
            "bull_candles_5d": candle_data["bull_candles"],
            "bear_candles_5d": candle_data["bear_candles"],
            "doji_count": candle_data["doji_count"],
            "engulfing": candle_data["engulfing"],
            "candle_pattern": candle_data["pattern_sequence"],
            # Trendlines
            "slope_5d": round(slope_5d, 3),
            "slope_10d": round(slope_10d, 3),
            "slope_20d": round(slope_20d, 3),
            # Volume
            "vol_trend": round(vol_data["vol_trend"], 1),
            "vol_acceleration": round(vol_data["vol_acceleration"], 1),
            "vol_spike": vol_data["vol_spike"],
            # Position
            "pct_above_sma20": round(pct_above_sma20, 2),
            "pct_above_sma50": round(pct_above_sma50, 2),
            "atr_pct": round(atr_pct, 2),
            "consolidation_pct": round(consolidation_pct, 1),
            "gap_pct": round(gap_pct, 2),
            "pct_off_52w": round(pct_off_52w, 1),
            "momentum_20d": round(momentum_20d, 2),
            "avg_dollar_vol": avg_dollar_vol,
            "consec_up_days": up_days,
            "consec_down_days": down_days,
        })

    return enriched


def filt_c17(signals):
    """Bullish crossover + improving YoY EPS (long entries only)."""
    return [s for s in signals
            if s["signal_type"] == "bullish"
            and s["eps_change_pct"] > 0
            and s.get("avg_dollar_vol", 0) >= 500_000
            and abs(s["eps_change_pct"]) < 100
            and s["days_between"] > 10]


def filt_c17_both(signals):
    """Both directions: long on bullish crossover + improving EPS,
    short on bearish crossover + deteriorating EPS."""
    return [s for s in signals
            if (
                (s["signal_type"] == "bullish" and s.get("eps_change_pct", 0) > 0) or
                (s["signal_type"] == "bearish" and s.get("eps_change_pct", 0) < 0)
            )
            and s.get("avg_dollar_vol", 0) >= 500_000
            and abs(s.get("eps_change_pct") or 0) < 100
            and s["days_between"] > 10]


def filt_eps_trend(signals):
    """Trade the MA crossover direction regardless of EPS direction.
    EPS is just the catalyst — we trade whichever way the trend breaks.
    Requires: liquidity >= $500K avg dollar vol, days_between > 10."""
    return [s for s in signals
            if s.get("avg_dollar_vol", 0) >= 500_000
            and s["days_between"] > 10
            and s.get("eps_change_pct") is not None]


def compare_groups(winners, losers, field, label=None):
    label = label or field
    w_vals = [s[field] for s in winners if s[field] is not None and not (isinstance(s[field], float) and np.isnan(s[field]))]
    l_vals = [s[field] for s in losers if s[field] is not None and not (isinstance(s[field], float) and np.isnan(s[field]))]
    if not w_vals or not l_vals:
        return
    w_avg = mean(w_vals)
    l_avg = mean(l_vals)
    w_med = median(w_vals)
    l_med = median(l_vals)
    diff = w_avg - l_avg
    print(f"  {label:30s} | Winners: avg={w_avg:>8.2f}  med={w_med:>8.2f} | Losers: avg={l_avg:>8.2f}  med={l_med:>8.2f} | Diff={diff:>+8.2f}")


def compare_bool(winners, losers, field, label=None):
    label = label or field
    w_pct = sum(1 for s in winners if s.get(field)) / len(winners) * 100 if winners else 0
    l_pct = sum(1 for s in losers if s.get(field)) / len(losers) * 100 if losers else 0
    print(f"  {label:30s} | Winners: {w_pct:>5.1f}% True | Losers: {l_pct:>5.1f}% True | Diff={w_pct-l_pct:>+5.1f}pp")


def compare_categorical(winners, losers, field, label=None):
    label = label or field
    from collections import Counter
    w_counts = Counter(s[field] for s in winners)
    l_counts = Counter(s[field] for s in losers)
    all_keys = sorted(set(list(w_counts.keys()) + list(l_counts.keys())))
    print(f"  {label}:")
    for k in all_keys:
        wc = w_counts.get(k, 0)
        lc = l_counts.get(k, 0)
        wp = wc / len(winners) * 100 if winners else 0
        lp = lc / len(losers) * 100 if losers else 0
        print(f"    {str(k):25s} | Winners: {wc:>2d} ({wp:>5.1f}%) | Losers: {lc:>2d} ({lp:>5.1f}%)")


def test_filter(all_signals, field, threshold, direction="gt", label=None):
    """Test a filter and show impact."""
    label = label or f"{field} {'>' if direction == 'gt' else '<'} {threshold}"
    if direction == "gt":
        filtered = [s for s in all_signals if s.get(field, 0) > threshold]
    elif direction == "lt":
        filtered = [s for s in all_signals if s.get(field, 999) < threshold]
    elif direction == "gte":
        filtered = [s for s in all_signals if s.get(field, 0) >= threshold]
    elif direction == "lte":
        filtered = [s for s in all_signals if s.get(field, 999) <= threshold]
    elif direction == "eq":
        filtered = [s for s in all_signals if s.get(field) == threshold]
    elif direction == "bool_true":
        filtered = [s for s in all_signals if s.get(field)]
    elif direction == "bool_false":
        filtered = [s for s in all_signals if not s.get(field)]
    else:
        filtered = all_signals

    if not filtered:
        print(f"  {label:50s} | n=   0 | (no signals)")
        return filtered

    returns = [s["return_15d"] for s in filtered if s["return_15d"] is not None]
    if not returns:
        print(f"  {label:50s} | n=   0 | (no returns)")
        return filtered

    winners = [r for r in returns if r > 0]
    losers = [r for r in returns if r <= 0]
    wr = len(winners) / len(returns) * 100
    avg = mean(returns)
    gross_gains = sum(winners)
    gross_losses = sum(r for r in losers if r < 0)
    pf = gross_gains / abs(gross_losses) if gross_losses != 0 else float("inf")
    pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"

    print(f"  {label:50s} | n={len(returns):>3d} | WR={wr:>5.1f}% | Avg={avg:>+7.2f}% | PF={pf_str:>6s} | W/L={len(winners)}/{len(losers)}")
    return filtered


def main():
    db = get_db()
    result = run_backtest(db)
    signals = result["signals"]

    print("Enriching all signals with deep technical features...")
    enriched = enrich_deep(db, signals)
    print(f"Total enriched: {len(enriched)}")

    # Apply C17 filter
    c17 = filt_c17(enriched)
    print(f"C17 signals: {len(c17)}")

    winners = [s for s in c17 if s["is_winner"]]
    losers = [s for s in c17 if not s["is_winner"]]
    print(f"Winners: {len(winners)}, Losers: {len(losers)}")

    # ======================================================================
    print("\n" + "=" * 110)
    print("  WINNERS vs LOSERS: TECHNICAL INDICATOR COMPARISON (C17, 15d horizon)")
    print("=" * 110)

    print("\n--- RSI ---")
    compare_groups(winners, losers, "rsi_14", "RSI (14)")

    print("\n--- MACD ---")
    compare_groups(winners, losers, "macd_histogram", "MACD Histogram")
    compare_bool(winners, losers, "macd_above_signal", "MACD > Signal Line")
    compare_bool(winners, losers, "macd_hist_positive", "MACD Histogram > 0")
    compare_bool(winners, losers, "macd_hist_rising", "MACD Histogram Rising")

    print("\n--- Bollinger Bands ---")
    compare_groups(winners, losers, "bb_pct_b", "BB %B (0=lower, 1=upper)")
    compare_groups(winners, losers, "bb_bandwidth", "BB Bandwidth %")
    compare_bool(winners, losers, "bb_above_upper", "Above Upper BB")
    compare_bool(winners, losers, "bb_below_lower", "Below Lower BB")

    print("\n--- Trendline Slopes (%/day) ---")
    compare_groups(winners, losers, "slope_5d", "5-day slope")
    compare_groups(winners, losers, "slope_10d", "10-day slope")
    compare_groups(winners, losers, "slope_20d", "20-day slope")

    print("\n--- Candlestick Patterns (last 5 candles) ---")
    compare_groups(winners, losers, "bull_candles_5d", "Bull candles (of 5)")
    compare_groups(winners, losers, "bear_candles_5d", "Bear candles (of 5)")
    compare_groups(winners, losers, "doji_count", "Doji count")
    compare_categorical(winners, losers, "last_candle", "Last candle type")
    compare_categorical(winners, losers, "engulfing", "Engulfing pattern")

    print("\n--- Volume ---")
    compare_groups(winners, losers, "vol_trend", "Vol trend (vs 20d avg) %")
    compare_groups(winners, losers, "vol_acceleration", "Vol acceleration %")
    compare_bool(winners, losers, "vol_spike", "Volume spike (>2x avg)")

    print("\n--- Price Position ---")
    compare_groups(winners, losers, "pct_above_sma20", "% above SMA20")
    compare_groups(winners, losers, "pct_above_sma50", "% above SMA50")
    compare_groups(winners, losers, "pct_off_52w", "% off 52w high")
    compare_groups(winners, losers, "momentum_20d", "20d momentum %")

    print("\n--- Volatility & Structure ---")
    compare_groups(winners, losers, "atr_pct", "ATR %")
    compare_groups(winners, losers, "consolidation_pct", "10d consolidation %")
    compare_groups(winners, losers, "gap_pct", "Gap % (prev close to open)")

    print("\n--- Consecutive Days ---")
    compare_groups(winners, losers, "consec_up_days", "Consecutive up days")
    compare_groups(winners, losers, "consec_down_days", "Consecutive down days")

    print("\n--- Other ---")
    compare_groups(winners, losers, "eps_change_pct", "EPS change %")
    compare_groups(winners, losers, "days_between", "Days between EPS & cross")
    compare_groups(winners, losers, "hold_days", "Hold days (to exit)")

    # ======================================================================
    print("\n" + "=" * 110)
    print("  INDIVIDUAL TRADE DETAIL (C17)")
    print("=" * 110)
    print(f"  {'Ticker':<7s} {'Date':<11s} {'Ret%':>7s} {'W/L':>4s} {'RSI':>5s} {'MACD_H':>7s} "
          f"{'MACD>S':>6s} {'BB%B':>5s} {'Slp5d':>6s} {'Slp10d':>6s} {'Candle':<12s} "
          f"{'VolTr%':>6s} {'VSpike':>6s} {'Gap%':>6s} {'52wOff':>6s} {'Mom20':>6s} {'ATR%':>5s}")
    print("  " + "-" * 130)

    for s in sorted(c17, key=lambda x: x.get("trend_date", "")):
        wl = "W" if s["is_winner"] else "L"
        macd_s = "Y" if s["macd_above_signal"] else "N"
        vspike = "Y" if s["vol_spike"] else "N"
        ret = s["return_15d"] if s["return_15d"] else 0
        print(f"  {s['ticker']:<7s} {s['trend_date']:<11s} {ret:>+6.1f}% {wl:>4s} "
              f"{s['rsi_14']:>5.1f} {s['macd_histogram']:>7.3f} {macd_s:>6s} {s['bb_pct_b']:>5.3f} "
              f"{s['slope_5d']:>6.2f} {s['slope_10d']:>6.2f} {s['last_candle']:<12s} "
              f"{s['vol_trend']:>6.1f} {vspike:>6s} {s['gap_pct']:>6.2f} {s['pct_off_52w']:>6.1f} "
              f"{s['momentum_20d']:>6.1f} {s['atr_pct']:>5.1f}")

    # ======================================================================
    print("\n" + "=" * 110)
    print("  FILTER IMPACT TESTS (starting from C17 base, 15d horizon)")
    print("=" * 110)

    print("\n--- RSI Filters ---")
    test_filter(c17, "rsi_14", 40, "gt", "RSI > 40")
    test_filter(c17, "rsi_14", 50, "gt", "RSI > 50")
    test_filter(c17, "rsi_14", 50, "lt", "RSI < 50")
    test_filter(c17, "rsi_14", 60, "lt", "RSI < 60")
    test_filter(c17, "rsi_14", 70, "lt", "RSI < 70 (not overbought)")
    test_filter(c17, "rsi_14", 30, "gt", "RSI > 30 (not oversold)")

    print("\n--- MACD Filters ---")
    test_filter(c17, "macd_above_signal", True, "bool_true", "MACD above signal line")
    test_filter(c17, "macd_above_signal", False, "bool_false", "MACD below signal line")
    test_filter(c17, "macd_hist_positive", True, "bool_true", "MACD histogram positive")
    test_filter(c17, "macd_hist_rising", True, "bool_true", "MACD histogram rising")

    print("\n--- Bollinger Band Filters ---")
    test_filter(c17, "bb_pct_b", 0.5, "gt", "BB %B > 0.5 (upper half)")
    test_filter(c17, "bb_pct_b", 0.5, "lt", "BB %B < 0.5 (lower half)")
    test_filter(c17, "bb_pct_b", 0.3, "gt", "BB %B > 0.3")
    test_filter(c17, "bb_pct_b", 0.8, "lt", "BB %B < 0.8 (not extended)")
    test_filter(c17, "bb_bandwidth", 15, "lt", "BB Bandwidth < 15% (tight)")
    test_filter(c17, "bb_bandwidth", 15, "gt", "BB Bandwidth > 15% (wide)")

    print("\n--- Trendline Slope Filters ---")
    test_filter(c17, "slope_5d", 0, "gt", "5d slope > 0 (short uptrend)")
    test_filter(c17, "slope_5d", 0, "lt", "5d slope < 0 (short downtrend)")
    test_filter(c17, "slope_10d", 0, "gt", "10d slope > 0")
    test_filter(c17, "slope_10d", 0, "lt", "10d slope < 0")
    test_filter(c17, "slope_20d", 0, "gt", "20d slope > 0")

    print("\n--- Candlestick Filters ---")
    test_filter(c17, "last_candle", "strong_bull", "eq", "Last candle = strong bull")
    test_filter(c17, "last_candle", "bull", "eq", "Last candle = bull")
    test_filter(c17, "last_candle", "bear", "eq", "Last candle = bear")
    test_filter(c17, "last_candle", "strong_bear", "eq", "Last candle = strong bear")
    test_filter(c17, "bull_candles_5d", 3, "gte", "3+ bull candles in last 5")
    test_filter(c17, "bear_candles_5d", 3, "gte", "3+ bear candles in last 5")

    print("\n--- Volume Filters ---")
    test_filter(c17, "vol_spike", True, "bool_true", "Volume spike on signal day")
    test_filter(c17, "vol_spike", False, "bool_false", "No volume spike")
    test_filter(c17, "vol_trend", 0, "gt", "Volume trending up (5d vs 20d)")
    test_filter(c17, "vol_trend", 50, "gt", "Volume up >50%")

    print("\n--- Gap Filters ---")
    test_filter(c17, "gap_pct", 0, "gt", "Gap up on signal day")
    test_filter(c17, "gap_pct", 0, "lt", "Gap down on signal day")
    test_filter(c17, "gap_pct", 2, "lt", "Gap < 2% (no big gap)")
    test_filter(c17, "gap_pct", -2, "gt", "Gap > -2% (no big gap down)")

    print("\n--- ATR / Volatility ---")
    test_filter(c17, "atr_pct", 5, "lt", "ATR < 5%")
    test_filter(c17, "atr_pct", 8, "lt", "ATR < 8%")
    test_filter(c17, "consolidation_pct", 20, "lt", "10d consolidation < 20%")

    print("\n--- Consecutive Day Patterns ---")
    test_filter(c17, "consec_up_days", 2, "gte", "2+ consecutive up days")
    test_filter(c17, "consec_up_days", 0, "eq", "0 consecutive up days")
    test_filter(c17, "consec_down_days", 2, "gte", "2+ consecutive down days")

    # ======================================================================
    print("\n" + "=" * 110)
    print("  BEST COMBINED FILTERS (layered on C17)")
    print("=" * 110)

    # Build combinations of the most promising filters
    def multi_filter(sigs, filters_dict):
        out = list(sigs)
        for field, (direction, threshold) in filters_dict.items():
            if direction == "gt":
                out = [s for s in out if s.get(field, 0) > threshold]
            elif direction == "lt":
                out = [s for s in out if s.get(field, 999) < threshold]
            elif direction == "gte":
                out = [s for s in out if s.get(field, 0) >= threshold]
            elif direction == "bool_true":
                out = [s for s in out if s.get(field)]
            elif direction == "bool_false":
                out = [s for s in out if not s.get(field)]
        return out

    combos = [
        ("RSI<70 + MACD hist rising",
         {"rsi_14": ("lt", 70), "macd_hist_rising": ("bool_true", None)}),
        ("RSI>30 + slope_5d>0",
         {"rsi_14": ("gt", 30), "slope_5d": ("gt", 0)}),
        ("RSI>40 + MACD>signal",
         {"rsi_14": ("gt", 40), "macd_above_signal": ("bool_true", None)}),
        ("MACD>signal + slope_5d>0",
         {"macd_above_signal": ("bool_true", None), "slope_5d": ("gt", 0)}),
        ("MACD hist+ + no vol spike",
         {"macd_hist_positive": ("bool_true", None), "vol_spike": ("bool_false", None)}),
        ("BB%B>0.3 + slope_10d>0",
         {"bb_pct_b": ("gt", 0.3), "slope_10d": ("gt", 0)}),
        ("BB%B<0.8 + MACD rising",
         {"bb_pct_b": ("lt", 0.8), "macd_hist_rising": ("bool_true", None)}),
        ("Gap>0 + MACD>signal",
         {"gap_pct": ("gt", 0), "macd_above_signal": ("bool_true", None)}),
        ("3+ bull candles + RSI>40",
         {"bull_candles_5d": ("gte", 3), "rsi_14": ("gt", 40)}),
        ("No gap down + slope_5d>0 + RSI>40",
         {"gap_pct": ("gt", -2), "slope_5d": ("gt", 0), "rsi_14": ("gt", 40)}),
        ("MACD>signal + BB%B>0.3 + ATR<8",
         {"macd_above_signal": ("bool_true", None), "bb_pct_b": ("gt", 0.3), "atr_pct": ("lt", 8)}),
        ("MACD rising + slope_10d>0 + no spike",
         {"macd_hist_rising": ("bool_true", None), "slope_10d": ("gt", 0), "vol_spike": ("bool_false", None)}),
        ("RSI 40-60 + MACD>signal",
         {"rsi_14": ("gt", 40), "macd_above_signal": ("bool_true", None)}),
        ("slope_5d>0 + gap>-2% + consec_up>=2",
         {"slope_5d": ("gt", 0), "gap_pct": ("gt", -2), "consec_up_days": ("gte", 2)}),
    ]

    for label, filters_dict in combos:
        filtered = multi_filter(c17, filters_dict)
        if not filtered:
            print(f"  {label:55s} | n=   0")
            continue
        returns = [s["return_15d"] for s in filtered if s["return_15d"] is not None]
        if not returns:
            print(f"  {label:55s} | n=   0")
            continue
        winners_r = [r for r in returns if r > 0]
        losers_r = [r for r in returns if r <= 0]
        wr = len(winners_r) / len(returns) * 100
        avg = mean(returns)
        gl = sum(r for r in losers_r if r < 0)
        gg = sum(winners_r)
        pf = gg / abs(gl) if gl != 0 else float("inf")
        pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
        # Avg hold
        holds = [s["hold_days"] for s in filtered if s["hold_days"]]
        avg_h = mean(holds) if holds else 0
        print(f"  {label:55s} | n={len(returns):>3d} | WR={wr:>5.1f}% | Avg={avg:>+7.2f}% "
              f"| PF={pf_str:>6s} | W/L={len(winners_r)}/{len(losers_r)} | AvgHold={avg_h:.1f}d")


if __name__ == "__main__":
    main()
