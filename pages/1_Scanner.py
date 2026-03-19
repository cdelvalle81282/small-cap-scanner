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

MA_PAIR_OPTIONS = {
    "20/50": (20, 50),
    "50/200": (50, 200),
}

# Sidebar controls
with st.sidebar:
    st.header("Scan Parameters")
    min_price = st.number_input("Min Price ($)", value=1.0, min_value=0.01, step=0.5)
    max_price = st.number_input("Max Price ($)", value=50.0, min_value=0.01, step=1.0)
    ma_pair_label = st.selectbox("MA Crossover", options=list(MA_PAIR_OPTIONS.keys()), index=0)
    eps_threshold = st.slider("Min EPS Change %", min_value=1, max_value=100, value=10)
    trend_window = st.slider("Trend Window (days)", min_value=5, max_value=90, value=30)
    direction = st.selectbox("Signal Direction", options=["both", "bullish", "bearish"], index=0)

run_scan = st.button("Run Scanner", type="primary")

if run_scan:
    ma_pair = MA_PAIR_OPTIONS[ma_pair_label]
    config = ScannerConfig(
        min_price=min_price,
        max_price=max_price,
        ma_crossover_pairs=[ma_pair],
        eps_change_threshold=float(eps_threshold),
        trend_window_days=trend_window,
        direction=direction,
    )
    scanner = Scanner(db=db, config=config)
    as_of_date = datetime.now().strftime("%Y-%m-%d")

    with st.spinner("Scanning..."):
        all_signals = scanner.scan(as_of_date)

    # Keep only the most recent signal per ticker (most recent by trend_change_date)
    latest_by_ticker: dict[str, dict] = {}
    for sig in all_signals:
        ticker = sig["ticker"]
        if ticker not in latest_by_ticker or sig["trend_change_date"] > latest_by_ticker[ticker]["trend_change_date"]:
            latest_by_ticker[ticker] = sig
    results = sorted(latest_by_ticker.values(), key=lambda s: s["trend_change_date"], reverse=True)

    st.session_state["scan_results"] = results

# Display results (persisted across page navigations)
results = st.session_state.get("scan_results")

if results is not None:
    if results:
        import pandas as pd

        # --- Total results count at the top ---
        st.metric("Total Signals", len(results))

        # --- Clickable ticker buttons ---
        st.subheader("Tickers")
        n_cols = min(len(results), 10)
        rows = [results[i : i + n_cols] for i in range(0, len(results), n_cols)]
        for row_chunk in rows:
            btn_cols = st.columns(n_cols)
            for i, result in enumerate(row_chunk):
                ticker = result["ticker"]
                signal = result.get("signal_type", "")
                label = f"{'🟢' if signal == 'bullish' else '🔴'} {ticker}"
                with btn_cols[i]:
                    if st.button(label, key=f"ticker_{ticker}_{result['trend_change_date']}"):
                        st.session_state["selected_ticker"] = ticker
                        st.session_state["signal_data"] = result
                        st.switch_page("pages/2_Stock_Detail.py")

        # --- Compact dataframe (no side scroll) ---
        st.subheader("Details")
        table_rows = []
        for r in results:
            eps_chg = r.get("eps_change_pct")
            table_rows.append(
                {
                    "Ticker": r["ticker"],
                    "Signal": r.get("signal_type", ""),
                    "MA Cross": f"{r.get('fast_ma', '')}/{r.get('slow_ma', '')}",
                    "EPS Chg %": f"{eps_chg:+.1f}%" if eps_chg is not None else "N/A",
                    "EPS Date": r.get("eps_change_date", ""),
                    "Cross Date": r.get("trend_change_date", ""),
                    "Days Between": r.get("days_between", ""),
                }
            )
        df = pd.DataFrame(table_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info(
            "No signals found for the selected parameters. "
            "Try lowering the EPS threshold, widening the trend window, or running the pipeline to load more data."
        )
