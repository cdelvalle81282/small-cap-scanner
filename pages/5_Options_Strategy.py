import sys
import math
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DB_PATH, BacktestConfig, ScannerConfig
from core.backtest import Backtester
from core.database import Database
from core.options_model import estimate_call_return
from core.scanner import Scanner
from analysis_winners_losers import enrich_deep, filt_c17

st.set_page_config(page_title="Options Strategy", page_icon="🎯", layout="wide")


@st.cache_resource
def get_db() -> Database:
    db = Database(DB_PATH)
    db.initialize()
    return db


db = get_db()

# ── Header ────────────────────────────────────────────────────────────────

st.title("Options Strategy")
st.markdown("""
Backtesting found an **extreme winner/loser asymmetry**: winners average
**+36-44% stock gains in 4-7 days**, while losers average just **-3.4% in
1-3 days**. ATM call options amplify this — 2-week calls returned an
estimated **+511% avg** vs **+34.8% stock avg** on winning trades.
""")

# ── Helpers ───────────────────────────────────────────────────────────────

MOMENTUM_FILTERS = {
    "slope_20d_pos": ("slope_20d", "gt", 0),
    "momentum_20d_pos": ("momentum_20d", "gt", 0),
    "vol_trend_pos": ("vol_trend", "gt", 0),
}


def apply_momentum_filters(signals, strict=True):
    """Apply optional momentum filters on top of C17."""
    out = list(signals)
    for name, (field, direction, threshold) in MOMENTUM_FILTERS.items():
        if strict:
            if direction == "gt":
                out = [s for s in out if s.get(field, 0) > threshold]
            elif direction == "lt":
                out = [s for s in out if s.get(field, 999) < threshold]
    return out


def compute_options_returns(signals, iv=0.60, dte_2wk=14, dte_monthly=30):
    """Add estimated call option returns to each signal."""
    for s in signals:
        stock_ret = s.get("return_15d")
        entry_price = s.get("entry_price", 0)
        hold = s.get("hold_days", 5) or 5

        if stock_ret is not None and entry_price > 0:
            s["call_return_2wk"] = round(
                estimate_call_return(stock_ret, entry_price, iv, dte_2wk, hold), 1
            )
            s["call_return_monthly"] = round(
                estimate_call_return(stock_ret, entry_price, iv, dte_monthly, hold), 1
            )
        else:
            s["call_return_2wk"] = None
            s["call_return_monthly"] = None
    return signals


def format_pct(value, signed=True):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    sign = "+" if signed else ""
    return f"{value:{sign}.1f}%"


def safe_mean(values):
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return mean(clean) if clean else None


def count_optional_filters(s):
    """Count how many optional momentum filters a signal passes (0-3)."""
    score = 0
    if s.get("slope_20d", 0) > 0:
        score += 1
    if s.get("momentum_20d", 0) > 0:
        score += 1
    if s.get("vol_trend", 0) > 0:
        score += 1
    return score


# ── Tabs ──────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Strategy Guide", "Backtest Results", "Live Signals"])

# ══════════════════════════════════════════════════════════════════════════
# Tab 1: Strategy Guide
# ══════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("Signal Criteria")
    st.markdown("""
The optimized signal uses **C17 base filters** plus momentum confirmation.
Only **bullish** signals are traded — bearish signals historically
underperform (27% WR vs 39% bullish).

| # | Filter | Rule | Rationale |
|---|--------|------|-----------|
| 1 | Direction | Bullish only | Bearish signals have poor win rate |
| 2 | EPS Change | Positive (surprise to upside) | Negative EPS = fundamental headwind |
| 3 | EPS Magnitude | < 100% change | Extreme outliers are noisy |
| 4 | Timing | > 10 days between EPS & crossover | Filters out same-day gap noise |
| 5 | Liquidity | Avg daily dollar volume >= $500K | Ensures tradeable options market |
| 6 | 20d Momentum | Positive (optional, strict mode) | Confirms trend direction |
| 7 | SMA20 Slope | Positive (optional, strict mode) | Rising moving average = uptrend |
""")

    st.header("Why These Filters Work")
    st.markdown("""
Research across 183 tickers (2022-2026) found:

- **Bullish + positive EPS + liquidity + timing** (C17) produces a base
  win rate of ~50% with strong profit factor
- Adding **momentum confirmation** (filters 6-7) increases win rate at
  the cost of fewer signals
- Winners are **fast and large** (avg +36-44% in 4-7 days), losers are
  **small and quick** (avg -3.4% in 1-3 days via stop loss)
- This asymmetry makes options highly attractive: you risk the premium
  on losers but capture leveraged upside on winners
""")

    st.header("Winner / Loser Asymmetry")
    asym_df = pd.DataFrame({
        "Metric": ["Avg Stock Return", "Avg Hold Days", "Avg ATR %", "Exit Reason"],
        "Winners": ["+36 to +44%", "4 - 7 days", "~5%", "Horizon (held to target)"],
        "Losers": ["-3.4%", "1 - 3 days", "~6%", "Stop loss (prev day low)"],
    })
    st.table(asym_df)

    st.header("Options Buy Rules")
    st.markdown("""
- **Instrument**: ATM (at-the-money) call options
- **Expiry**: **2-week expiry preferred** over monthly
  - 2-week calls have higher gamma → more responsive to quick moves
  - Winners resolve in 4-7 days, so shorter expiry captures the move
    with less time premium wasted
  - Monthly calls still work but have lower estimated return due to
    higher entry cost
- **Entry**: Buy when the stock signal triggers (next trading day after
  MA crossover)
- **Position Size**: Fixed dollar amount per trade (e.g. $500-$1000 per
  option position)
""")

    st.header("Options Sell Rules")
    st.markdown("""
Exit the call position when **any** of these occur:

1. **Stock trailing stop triggers**
   - Day 1: close < previous day's low
   - Day 2+: close < 0.5x ATR trailing stop
2. **Expiration**: sell before expiry if still holding
3. **Profit target** (optional): sell at 200-500% gain on the option

The trailing stop is the primary exit — it limits losses to roughly one
day's adverse move on the stock, which translates to a partial loss on
the option premium rather than total loss.
""")

    st.info("""
**Disclaimer**: Option returns shown are *estimated* using the
Black-Scholes model with assumed 60% IV. Actual returns depend on
real implied volatility, bid/ask spreads, and Greeks at time of trade.
These estimates illustrate the leverage effect, not guaranteed outcomes.
""")

# ══════════════════════════════════════════════════════════════════════════
# Tab 2: Backtest Results
# ══════════════════════════════════════════════════════════════════════════

with tab2:
    with st.sidebar:
        st.header("Options Backtest Settings")
        filter_mode = st.selectbox(
            "Filter Mode",
            options=["Strict", "Standard", "Relaxed"],
            index=1,
            help="Strict = C17 + all momentum filters. Standard = C17 only. Relaxed = bullish + positive EPS only.",
        )
        trend_window = st.selectbox(
            "Trend Window (days)", options=[30, 45, 60], index=0
        )
        assumed_iv = st.slider(
            "Assumed IV (%)", min_value=30, max_value=120, value=60, step=5
        ) / 100.0

    if st.button("Run Options Backtest", type="primary", key="run_options_bt"):
        sc = ScannerConfig(
            min_price=1.0,
            max_price=50.0,
            min_market_cap=50_000_000,
            max_market_cap=10_000_000_000,
            ma_crossover_pairs=[(20, 50)],
            eps_change_threshold=10.0,
            trend_window_days=int(trend_window),
            direction="both",
        )
        bc = BacktestConfig(
            start_date="2022-01-01",
            end_date="2026-03-19",
            forward_return_days=[5, 10, 15, 30, 60],
            ma_crossover_pairs=[(20, 50)],
            eps_thresholds=[10.0],
            trend_windows=[int(trend_window)],
        )

        with st.spinner("Running backtest & enriching signals..."):
            result = Backtester(db, sc, bc).run()
            enriched = enrich_deep(db, result["signals"])

            # Apply filters based on mode
            if filter_mode == "Relaxed":
                filtered = [
                    s for s in enriched
                    if s["signal_type"] == "bullish" and s["eps_change_pct"] > 0
                ]
            elif filter_mode == "Standard":
                filtered = filt_c17(enriched)
            else:  # Strict
                filtered = apply_momentum_filters(filt_c17(enriched), strict=True)

            filtered = compute_options_returns(filtered, iv=assumed_iv)

        st.session_state["options_bt_results"] = filtered

    results = st.session_state.get("options_bt_results")
    if results:
        st.subheader(f"Results: {len(results)} qualifying trades")

        # Summary metrics
        stock_rets = [s["return_15d"] for s in results if s["return_15d"] is not None]
        call_2wk = [s["call_return_2wk"] for s in results if s.get("call_return_2wk") is not None]
        call_mo = [s["call_return_monthly"] for s in results if s.get("call_return_monthly") is not None]

        winners_stock = [r for r in stock_rets if r > 0]
        losers_stock = [r for r in stock_rets if r <= 0]
        wr = len(winners_stock) / len(stock_rets) * 100 if stock_rets else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Win Rate", f"{wr:.0f}%")
        col2.metric("Avg Stock Return", format_pct(safe_mean(stock_rets)))
        col3.metric("Avg 2wk Call Return", format_pct(safe_mean(call_2wk)))
        col4.metric("Avg Monthly Call Return", format_pct(safe_mean(call_mo)))

        # Comparison table
        st.subheader("Stock vs Options Comparison")
        comp_data = {
            "Metric": [
                "Avg Return (all trades)",
                "Avg Winner",
                "Avg Loser",
                "Max Gain",
                "Max Loss",
            ],
        }

        def _stats(rets):
            if not rets:
                return ["N/A"] * 5
            w = [r for r in rets if r > 0]
            l = [r for r in rets if r <= 0]
            return [
                format_pct(safe_mean(rets)),
                format_pct(safe_mean(w)) if w else "N/A",
                format_pct(safe_mean(l)) if l else "N/A",
                format_pct(max(rets)),
                format_pct(min(rets)),
            ]

        comp_data["Stock (15d)"] = _stats(stock_rets)
        comp_data["2-Week Call (est.)"] = _stats(call_2wk)
        comp_data["Monthly Call (est.)"] = _stats(call_mo)
        st.table(pd.DataFrame(comp_data))

        # Individual trades
        st.subheader("Individual Trades")
        rows = []
        for s in sorted(results, key=lambda x: x.get("trend_date", "")):
            rows.append({
                "Ticker": s["ticker"],
                "Date": s["trend_date"],
                "Entry $": f"${s['entry_price']:.2f}" if s.get("entry_price") else "N/A",
                "Stock Ret": format_pct(s.get("return_15d")),
                "Hold Days": s.get("hold_days") or "N/A",
                "Exit": s.get("exit_reason") or "N/A",
                "2wk Call": format_pct(s.get("call_return_2wk")),
                "Mo. Call": format_pct(s.get("call_return_monthly")),
                "Strength": "+" * count_optional_filters(s),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# Tab 3: Live Signals
# ══════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("Scan for **today's qualifying signals** using the optimized filter set.")

    if st.button("Scan Now", type="primary", key="scan_live"):
        sc = ScannerConfig(
            min_price=1.0,
            max_price=50.0,
            min_market_cap=50_000_000,
            max_market_cap=10_000_000_000,
            ma_crossover_pairs=[(20, 50)],
            eps_change_threshold=10.0,
            trend_window_days=60,
            direction="bullish",
        )

        today = date.today().isoformat()

        with st.spinner("Scanning universe..."):
            scanner = Scanner(db, sc)
            raw_signals = scanner.scan(today)

            # Filter to recent signals (last 60 days)
            cutoff = (date.today() - timedelta(days=60)).isoformat()
            recent = [s for s in raw_signals if s.get("trend_change_date", "") >= cutoff]

            if not recent:
                st.session_state["live_signals"] = []
            else:
                # Build minimal backtest data for enrichment
                bc = BacktestConfig(
                    start_date=cutoff,
                    end_date=today,
                    forward_return_days=[5, 10, 15],
                    ma_crossover_pairs=[(20, 50)],
                    eps_thresholds=[10.0],
                    trend_windows=[60],
                )
                bt_result = Backtester(db, sc, bc).run()
                enriched = enrich_deep(db, bt_result["signals"])
                filtered = filt_c17(enriched)
                st.session_state["live_signals"] = filtered

    live = st.session_state.get("live_signals")
    if live is not None:
        if not live:
            st.info("No qualifying signals found. Check back after new earnings reports or MA crossovers.")
        else:
            st.success(f"Found **{len(live)}** qualifying signal(s)")
            for s in live:
                strength = count_optional_filters(s)
                strength_bar = "🟢" * strength + "⚪" * (3 - strength)

                with st.expander(
                    f"**{s['ticker']}** — {s['trend_date']} — "
                    f"EPS: {s['eps_change_pct']:+.1f}% — Strength: {strength_bar}"
                ):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Entry Price", f"${s['entry_price']:.2f}" if s.get("entry_price") else "N/A")
                    c2.metric("EPS Change", f"{s['eps_change_pct']:+.1f}%")
                    c3.metric("Days EPS→Cross", s.get("days_between", "N/A"))
                    c4.metric("Signal Strength", f"{strength}/3")

                    st.markdown(f"""
| Indicator | Value |
|-----------|-------|
| 20d Momentum | {s.get('momentum_20d', 0):+.1f}% |
| SMA20 Slope | {s.get('slope_20d', 0):+.3f}%/day |
| Volume Trend | {s.get('vol_trend', 0):+.1f}% vs 20d avg |
| ATR % | {s.get('atr_pct', 0):.1f}% |
| RSI (14) | {s.get('rsi_14', 0):.1f} |
| Avg Dollar Volume | ${s.get('avg_dollar_vol', 0):,.0f} |
""")

                    if st.button(
                        f"View {s['ticker']} Chart",
                        key=f"live_{s['ticker']}_{s['trend_date']}",
                    ):
                        st.session_state["selected_ticker"] = s["ticker"]
                        st.switch_page("pages/2_Stock_Detail.py")
