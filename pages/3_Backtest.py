import sys
import math
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DB_PATH, BacktestConfig, ScannerConfig
from core.backtest import Backtester
from core.database import Database

st.set_page_config(page_title="Backtest", page_icon="📊", layout="wide")


@st.cache_resource
def get_db() -> Database:
    db = Database(DB_PATH)
    db.initialize()
    return db


db = get_db()

MA_PAIR_OPTIONS = {
    "20/50": (20, 50),
    "50/200": (50, 200),
}


def format_pct(value: float | None, signed: bool = True) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    sign = "+" if signed else ""
    return f"{value:{sign}.2f}%"


def format_price(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"${value:.2f}"


def format_ratio(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if math.isinf(value):
        return "∞"
    return f"{value:.2f}"

# --- Sidebar ---
with st.sidebar:
    st.header("Backtest Settings")

    start_date = st.date_input("Start Date", value=date(2022, 1, 1))
    end_date = st.date_input("End Date", value=date(2026, 3, 17))
    min_price = st.number_input("Min Price ($)", value=1.0, min_value=0.01, step=0.5)
    max_price = st.number_input("Max Price ($)", value=20.0, min_value=0.01, step=1.0)
    direction = st.selectbox("Direction", options=["both", "bullish", "bearish"], index=0)

st.title("Backtest")

tab1, tab2 = st.tabs(["Single Backtest", "Parameter Sweep"])

# ---------------------------------------------------------------------------
# Tab 1: Single Backtest
# ---------------------------------------------------------------------------
with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        ma_pair_label = st.selectbox(
            "MA Crossover", options=list(MA_PAIR_OPTIONS.keys()), index=0, key="single_ma"
        )
    with col2:
        eps_threshold = st.number_input(
            "EPS Change Threshold (%)", value=10.0, min_value=0.0, step=1.0, key="single_eps"
        )
    with col3:
        trend_window = st.number_input(
            "Trend Window (days)", value=30, min_value=1, step=5, key="single_trend"
        )

    if st.button("Run Backtest", type="primary", key="run_single"):
        ma_pair = MA_PAIR_OPTIONS[ma_pair_label]
        scanner_cfg = ScannerConfig(
            min_price=float(min_price),
            max_price=float(max_price),
            ma_crossover_pairs=[ma_pair],
            eps_change_threshold=float(eps_threshold),
            trend_window_days=int(trend_window),
            direction=direction,
        )
        backtest_cfg = BacktestConfig(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )

        with st.spinner("Running backtest..."):
            backtester = Backtester(db, scanner_cfg, backtest_cfg)
            result = backtester.run()

        st.session_state["backtest_results"] = result

    # --- Display persisted results ---
    bt_result = st.session_state.get("backtest_results")
    if bt_result is not None:
        signals = bt_result["signals"]
        summary = bt_result["summary"]
        total = summary.get("total_signals", 0)

        st.metric("Total Signals", total)

        if total == 0:
            st.info("No signals found for the given parameters and date range.")
        else:
            by_horizon = summary.get("by_horizon", {})

            # Summary by horizon — expandable to see individual signals
            st.subheader("Performance by Horizon")
            for horizon, stats in sorted(by_horizon.items()):
                win_rate = format_pct(stats["win_rate"], signed=False)
                avg_ret = format_pct(stats["avg_return"])
                median_ret = format_pct(stats["median_return"])
                profit_factor = format_ratio(stats.get("profit_factor"))
                sample = stats["sample_size"]
                win_count = stats.get("win_count", 0)
                loss_count = stats.get("loss_count", 0)

                label = (
                    f"{horizon}-Day  |  Win Rate: {win_rate}  |  "
                    f"W/L: {win_count}/{loss_count}  |  PF: {profit_factor}  |  "
                    f"Avg: {avg_ret}  |  Median: {median_ret}  |  "
                    f"Signals: {sample}"
                )
                with st.expander(label):
                    # Collect signals that have a return for this horizon
                    horizon_signals = []
                    for sig in signals:
                        fwd = sig.get("forward_returns", {})
                        ret = fwd.get(horizon)
                        if ret is not None:
                            horizon_signals.append({**sig, "_return": ret})

                    if not horizon_signals:
                        st.info("No signals with data for this horizon.")
                        continue

                    horizon_signals.sort(key=lambda s: s["_return"], reverse=True)

                    # Build a dataframe for clean display
                    rows = []
                    for sig in horizon_signals:
                        trade_detail = sig.get("trade_details", {}).get(horizon, {})
                        rows.append({
                            "Ticker": sig.get("ticker", ""),
                            "Signal": sig.get("signal_type", ""),
                            "MA Cross": f"{sig.get('fast_ma', '')}/{sig.get('slow_ma', '')}",
                            "EPS Change": f"{sig['eps_change_pct']:+.1f}%" if sig.get("eps_change_pct") is not None else "N/A",
                            "EPS Date": sig.get("eps_change_date", ""),
                            "Cross Date": sig.get("trend_change_date", ""),
                            "Entry Date": sig.get("entry_date", ""),
                            "Entry Price": format_price(sig.get("entry_price")),
                            "Exit Date": trade_detail.get("exit_date") or "N/A",
                            "Exit Price": format_price(trade_detail.get("exit_price")),
                            "Exit Reason": trade_detail.get("exit_reason") or "N/A",
                            f"{horizon}d Return": format_pct(sig["_return"]),
                        })
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

                    # Clickable ticker buttons to navigate to Stock Detail
                    st.caption("Click a ticker to view its chart:")
                    btn_cols = st.columns(min(len(horizon_signals), 8))
                    for i, sig in enumerate(horizon_signals):
                        col_idx = i % min(len(horizon_signals), 8)
                        ticker_val = sig.get("ticker", "")
                        if btn_cols[col_idx].button(
                            ticker_val,
                            key=f"hz_{horizon}_{ticker_val}_{sig.get('trend_change_date')}",
                        ):
                            st.session_state["selected_ticker"] = ticker_val
                            st.session_state["signal_data"] = sig
                            st.switch_page("pages/2_Stock_Detail.py")

            # Box plot: forward return distribution by horizon
            st.subheader("Return Distribution by Horizon")
            box_rows = []
            for sig in signals:
                fwd = sig.get("forward_returns", {})
                for horizon, ret in fwd.items():
                    if ret is not None:
                        box_rows.append({"Horizon (days)": str(horizon), "Return %": ret})

            if box_rows:
                box_df = pd.DataFrame(box_rows)
                # Sort horizons numerically
                horizon_order = sorted(box_df["Horizon (days)"].unique(), key=lambda x: int(x))
                fig_box = px.box(
                    box_df,
                    x="Horizon (days)",
                    y="Return %",
                    category_orders={"Horizon (days)": horizon_order},
                    template="plotly_dark",
                )
                fig_box.add_hline(y=0, line=dict(color="white", width=1, dash="dash"))
                fig_box.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(
                    fig_box,
                    use_container_width=True,
                    config={"scrollZoom": True},
                )

            # Individual signals table
            st.subheader("Individual Signals")
            header_cols = st.columns([1, 2, 2, 2, 2, 2, 2])
            header_cols[0].markdown("**Ticker**")
            header_cols[1].markdown("**Signal**")
            header_cols[2].markdown("**MA Cross**")
            header_cols[3].markdown("**EPS Change**")
            header_cols[4].markdown("**Cross Date**")
            header_cols[5].markdown("**EPS Date**")
            header_cols[6].markdown("**Days Between**")
            for sig in signals:
                cols = st.columns([1, 2, 2, 2, 2, 2, 2])
                ticker_val = sig.get("ticker", "")
                if cols[0].button(ticker_val, key=f"sig_{ticker_val}_{sig.get('trend_change_date')}"):
                    st.session_state["selected_ticker"] = ticker_val
                    st.session_state["signal_data"] = sig
                    st.switch_page("pages/2_Stock_Detail.py")
                cols[1].write(sig.get("signal_type", ""))
                cols[2].write(f"{sig.get('fast_ma', '')}/{sig.get('slow_ma', '')}")
                eps_chg = sig.get("eps_change_pct")
                cols[3].write(f"{eps_chg:+.1f}%" if eps_chg is not None else "N/A")
                cols[4].write(sig.get("trend_change_date", ""))
                cols[5].write(sig.get("eps_change_date", ""))
                days = sig.get("days_between")
                cols[6].write(str(days) if days is not None else "N/A")

# ---------------------------------------------------------------------------
# Tab 2: Parameter Sweep
# ---------------------------------------------------------------------------
with tab2:
    sw_col1, sw_col2, sw_col3, sw_col4 = st.columns(4)
    with sw_col1:
        sweep_ma_pairs = st.multiselect(
            "MA Crossovers",
            options=list(MA_PAIR_OPTIONS.keys()),
            default=list(MA_PAIR_OPTIONS.keys()),
            key="sweep_ma",
        )
    with sw_col2:
        sweep_eps_thresholds = st.multiselect(
            "EPS Thresholds (%)", options=[5.0, 10.0, 25.0, 50.0], default=[5.0, 10.0, 25.0], key="sweep_eps"
        )
    with sw_col3:
        sweep_trend_windows = st.multiselect(
            "Trend Windows (days)", options=[15, 30, 45], default=[15, 30, 45], key="sweep_trend"
        )
    with sw_col4:
        sweep_horizon = st.selectbox(
            "Comparison Horizon (days)",
            options=[5, 10, 15, 30, 60],
            index=2,
            key="sweep_horizon",
        )

    if not sweep_ma_pairs or not sweep_eps_thresholds or not sweep_trend_windows:
        st.warning("Select at least one value for each parameter.")
    elif st.button("Run Parameter Sweep", type="primary", key="run_sweep"):
        selected_pairs = [MA_PAIR_OPTIONS[label] for label in sweep_ma_pairs]
        scanner_cfg = ScannerConfig(
            min_price=float(min_price),
            max_price=float(max_price),
            direction=direction,
        )
        backtest_cfg = BacktestConfig(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            forward_return_days=[int(sweep_horizon)],
            ma_crossover_pairs=selected_pairs,
            eps_thresholds=[float(t) for t in sweep_eps_thresholds],
            trend_windows=[int(w) for w in sweep_trend_windows],
        )

        with st.spinner("Running parameter sweep (this may take a while)..."):
            backtester = Backtester(db, scanner_cfg, backtest_cfg)
            sweep_results = backtester.parameter_sweep()

        st.session_state["sweep_results"] = sweep_results

    # --- Display persisted sweep results ---
    sweep_results = st.session_state.get("sweep_results")
    if sweep_results is not None:
        if not sweep_results:
            st.info("No results returned from parameter sweep.")
        else:
            sweep_df = pd.DataFrame(sweep_results)
            sweep_df = sweep_df.sort_values("win_rate", ascending=False).reset_index(drop=True)

            st.subheader("Parameter Sweep Results")
            display_df = sweep_df.copy()
            display_df["win_rate"] = display_df["win_rate"].apply(
                lambda v: format_pct(v, signed=False)
            )
            display_df["avg_return"] = display_df["avg_return"].apply(
                format_pct
            )
            display_df["profit_factor"] = display_df["profit_factor"].apply(
                format_ratio
            )
            display_df["expectancy"] = display_df["expectancy"].apply(
                format_pct
            )
            display_df.columns = [
                "MA Crossover", "EPS Threshold %", "Trend Window (days)",
                "Total Signals", "Win Rate", "Avg Return", "Profit Factor",
                "Expectancy", "Sample Size",
            ]
            st.dataframe(display_df, hide_index=True, use_container_width=True)

            # Heatmap: win_rate by MA Crossover vs EPS Threshold
            if len(sweep_df["ma_crossover"].unique()) > 1 and len(sweep_df["eps_threshold"].unique()) > 1:
                st.subheader("Win Rate Heatmap (MA Crossover vs EPS Threshold)")
                pivot_df = (
                    sweep_df[sweep_df["win_rate"].notna()]
                    .groupby(["ma_crossover", "eps_threshold"])["win_rate"]
                    .mean()
                    .reset_index()
                    .pivot(index="ma_crossover", columns="eps_threshold", values="win_rate")
                )

                if not pivot_df.empty:
                    fig_heat = px.imshow(
                        pivot_df,
                        labels=dict(x="EPS Threshold %", y="MA Crossover", color="Win Rate %"),
                        color_continuous_scale="RdYlGn",
                        text_auto=".1f",
                        template="plotly_dark",
                    )
                    fig_heat.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(
                        fig_heat,
                        use_container_width=True,
                        config={"scrollZoom": True},
                    )
