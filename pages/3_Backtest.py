import sys
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
        ma_period = st.selectbox("MA Period", options=[20, 50, 200], index=0, key="single_ma")
    with col2:
        eps_threshold = st.number_input(
            "EPS Change Threshold (%)", value=10.0, min_value=0.0, step=1.0, key="single_eps"
        )
    with col3:
        trend_window = st.number_input(
            "Trend Window (days)", value=30, min_value=1, step=5, key="single_trend"
        )

    if st.button("Run Backtest", type="primary", key="run_single"):
        scanner_cfg = ScannerConfig(
            min_price=float(min_price),
            max_price=float(max_price),
            ma_periods=[int(ma_period)],
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

        signals = result["signals"]
        summary = result["summary"]
        total = summary.get("total_signals", 0)

        st.metric("Total Signals", total)

        if total == 0:
            st.info("No signals found for the given parameters and date range.")
        else:
            by_horizon = summary.get("by_horizon", {})

            # Summary table
            st.subheader("Performance by Horizon")
            horizon_rows = []
            for horizon, stats in sorted(by_horizon.items()):
                horizon_rows.append({
                    "Horizon (days)": horizon,
                    "Win Rate %": f"{stats['win_rate']:.1f}%" if stats["win_rate"] is not None else "N/A",
                    "Avg Return %": f"{stats['avg_return']:+.2f}%" if stats["avg_return"] is not None else "N/A",
                    "Median Return %": f"{stats['median_return']:+.2f}%" if stats["median_return"] is not None else "N/A",
                    "Max Gain %": f"{stats['max_gain']:+.2f}%" if stats["max_gain"] is not None else "N/A",
                    "Max Loss %": f"{stats['max_loss']:+.2f}%" if stats["max_loss"] is not None else "N/A",
                    "Sample Size": stats["sample_size"],
                })
            st.dataframe(pd.DataFrame(horizon_rows), hide_index=True, use_container_width=True)

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
                st.plotly_chart(fig_box, use_container_width=True)

            # Individual signals table
            st.subheader("Individual Signals")
            for sig in signals:
                cols = st.columns([1, 2, 2, 2, 2, 2, 2, 2])
                ticker_val = sig.get("ticker", "")
                if cols[0].button(ticker_val, key=f"sig_{ticker_val}_{sig.get('trend_change_date')}"):
                    st.session_state["selected_ticker"] = ticker_val
                    st.session_state["signal_data"] = sig
                    st.switch_page("pages/2_Stock_Detail.py")
                cols[1].write(sig.get("signal_type", ""))
                cols[2].write(str(sig.get("ma_period", "")))
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
        sweep_ma_periods = st.multiselect(
            "MA Periods", options=[20, 50, 200], default=[20, 50, 200], key="sweep_ma"
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
            options=[5, 10, 20, 30, 60],
            index=2,
            key="sweep_horizon",
        )

    if not sweep_ma_periods or not sweep_eps_thresholds or not sweep_trend_windows:
        st.warning("Select at least one value for each parameter.")
    elif st.button("Run Parameter Sweep", type="primary", key="run_sweep"):
        scanner_cfg = ScannerConfig(
            min_price=float(min_price),
            max_price=float(max_price),
            direction=direction,
        )
        backtest_cfg = BacktestConfig(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            forward_return_days=[int(sweep_horizon)],
            ma_periods=[int(p) for p in sweep_ma_periods],
            eps_thresholds=[float(t) for t in sweep_eps_thresholds],
            trend_windows=[int(w) for w in sweep_trend_windows],
        )

        with st.spinner("Running parameter sweep (this may take a while)..."):
            backtester = Backtester(db, scanner_cfg, backtest_cfg)
            sweep_results = backtester.parameter_sweep()

        if not sweep_results:
            st.info("No results returned from parameter sweep.")
        else:
            sweep_df = pd.DataFrame(sweep_results)
            sweep_df = sweep_df.sort_values("win_rate", ascending=False).reset_index(drop=True)

            st.subheader("Parameter Sweep Results")
            display_df = sweep_df.copy()
            display_df["win_rate"] = display_df["win_rate"].apply(
                lambda v: f"{v:.1f}%" if pd.notna(v) else "N/A"
            )
            display_df["avg_return"] = display_df["avg_return"].apply(
                lambda v: f"{v:+.2f}%" if pd.notna(v) else "N/A"
            )
            display_df.columns = [
                "MA Period", "EPS Threshold %", "Trend Window (days)",
                "Total Signals", "Win Rate", "Avg Return", "Sample Size",
            ]
            st.dataframe(display_df, hide_index=True, use_container_width=True)

            # Heatmap: win_rate by MA Period vs EPS Threshold
            if len(sweep_ma_periods) > 1 and len(sweep_eps_thresholds) > 1:
                st.subheader("Win Rate Heatmap (MA Period vs EPS Threshold)")
                # Aggregate by ma_period and eps_threshold (mean over trend windows)
                pivot_df = (
                    sweep_df[sweep_df["win_rate"].notna()]
                    .groupby(["ma_period", "eps_threshold"])["win_rate"]
                    .mean()
                    .reset_index()
                    .pivot(index="ma_period", columns="eps_threshold", values="win_rate")
                )

                if not pivot_df.empty:
                    fig_heat = px.imshow(
                        pivot_df,
                        labels=dict(x="EPS Threshold %", y="MA Period", color="Win Rate %"),
                        color_continuous_scale="RdYlGn",
                        text_auto=".1f",
                        template="plotly_dark",
                    )
                    fig_heat.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig_heat, use_container_width=True)
