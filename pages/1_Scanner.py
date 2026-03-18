import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DB_PATH, ScannerConfig
from core.database import Database
from core.scanner import Scanner

st.set_page_config(page_title="Scanner", page_icon="🔍", layout="wide")


@st.cache_resource
def get_db() -> Database:
    db = Database(DB_PATH)
    db.initialize()
    return db


db = get_db()

st.title("Small Cap Scanner")

# Sidebar controls
with st.sidebar:
    st.header("Scan Parameters")
    min_price = st.number_input("Min Price ($)", value=1.0, min_value=0.01, step=0.5)
    max_price = st.number_input("Max Price ($)", value=20.0, min_value=0.01, step=1.0)
    ma_period = st.selectbox("MA Period", options=[20, 50, 200], index=0)
    eps_threshold = st.slider("Min EPS Change %", min_value=1, max_value=100, value=10)
    trend_window = st.slider("Trend Window (days)", min_value=5, max_value=90, value=30)
    direction = st.selectbox("Signal Direction", options=["both", "bullish", "bearish"], index=0)

run_scan = st.button("Run Scanner", type="primary")

if run_scan:
    config = ScannerConfig(
        min_price=min_price,
        max_price=max_price,
        ma_periods=[ma_period],
        eps_change_threshold=float(eps_threshold),
        trend_window_days=trend_window,
        direction=direction,
    )
    scanner = Scanner(db=db, config=config)
    as_of_date = datetime.now().strftime("%Y-%m-%d")

    with st.spinner("Scanning..."):
        results = scanner.scan(as_of_date)

    if results:
        st.success(f"Found {len(results)} signal(s).")

        # Header row
        cols = st.columns(7)
        headers = ["Ticker", "Signal", "MA Period", "EPS Chg %", "EPS Date", "MA Cross Date", "Days Between"]
        for col, header in zip(cols, headers):
            col.markdown(f"**{header}**")

        st.divider()

        for result in results:
            cols = st.columns(7)
            ticker = result["ticker"]

            with cols[0]:
                if st.button(ticker, key=f"ticker_{ticker}_{result['trend_change_date']}"):
                    st.session_state["selected_ticker"] = ticker
                    st.session_state["signal_data"] = result
                    st.switch_page("pages/2_Stock_Detail.py")

            signal_type = result.get("signal_type", "")
            cols[1].markdown(
                f":green[{signal_type}]" if signal_type == "bullish" else f":red[{signal_type}]"
            )
            cols[2].write(result.get("ma_period", ""))

            eps_chg = result.get("eps_change_pct")
            if eps_chg is not None:
                cols[3].markdown(
                    f":green[+{eps_chg:.1f}%]" if eps_chg >= 0 else f":red[{eps_chg:.1f}%]"
                )
            else:
                cols[3].write("N/A")

            cols[4].write(result.get("eps_change_date", ""))
            cols[5].write(result.get("trend_change_date", ""))
            cols[6].write(result.get("days_between", ""))
    else:
        st.info(
            "No signals found for the selected parameters. "
            "Try lowering the EPS threshold, widening the trend window, or running the pipeline to load more data."
        )
