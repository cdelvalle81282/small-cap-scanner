import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
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

SORT_OPTIONS = {
    "Cross Date (newest)": ("trend_change_date", True),
    "Avg $ Volume (highest)": ("avg_dollar_vol", True),
    "Market Cap (highest)": ("market_cap", True),
    "Market Cap (lowest)": ("market_cap", False),
    "Price (highest)": ("latest_close", True),
    "Price (lowest)": ("latest_close", False),
    "EPS Change % (highest)": ("eps_change_pct", True),
}


def parse_optional_float(val: str) -> float | None:
    val = val.strip()
    if not val:
        return None
    try:
        return float(val.replace(",", ""))
    except ValueError:
        return None


def parse_optional_market_cap(val: str) -> float | None:
    """Parse market cap shorthand: 500M → 500_000_000, 2B → 2_000_000_000."""
    val = val.strip().upper().replace(",", "")
    if not val:
        return None
    try:
        if val.endswith("B"):
            return float(val[:-1]) * 1_000_000_000
        if val.endswith("M"):
            return float(val[:-1]) * 1_000_000
        return float(val)
    except ValueError:
        return None


# Sidebar controls
with st.sidebar:
    st.header("Scan Parameters")

    st.caption("Price Filter (leave blank for no limit)")
    col_min, col_max = st.columns(2)
    with col_min:
        min_price_str = st.text_input("Min $", value="", placeholder="e.g. 1")
    with col_max:
        max_price_str = st.text_input("Max $", value="", placeholder="e.g. 50")

    st.caption("Market Cap Filter (leave blank for no limit)")
    col_mmin, col_mmax = st.columns(2)
    with col_mmin:
        min_cap_str = st.text_input("Min Cap", value="", placeholder="e.g. 50M")
    with col_mmax:
        max_cap_str = st.text_input("Max Cap", value="", placeholder="e.g. 10B")

    ma_pair_label = st.selectbox("MA Crossover", options=list(MA_PAIR_OPTIONS.keys()), index=0)
    eps_threshold = st.slider("Min EPS Change %", min_value=1, max_value=100, value=10)
    trend_window = st.slider("Trend Window (days)", min_value=5, max_value=90, value=30)
    direction = st.selectbox("Signal Direction", options=["both", "bullish", "bearish"], index=0)
    st.divider()
    sort_label = st.selectbox("Sort By", options=list(SORT_OPTIONS.keys()), index=0)
    st.divider()
    run_scan = st.button("Run Scanner", type="primary", use_container_width=True)

# Parse optional filters
min_price = parse_optional_float(min_price_str)
max_price = parse_optional_float(max_price_str)
min_cap   = parse_optional_market_cap(min_cap_str)
max_cap   = parse_optional_market_cap(max_cap_str)

# Validation feedback
errors = []
if min_price_str.strip() and min_price is None:
    errors.append("Min Price is not a valid number.")
if max_price_str.strip() and max_price is None:
    errors.append("Max Price is not a valid number.")
if min_cap_str.strip() and min_cap is None:
    errors.append("Min Cap is not a valid number (use e.g. 50M or 2B).")
if max_cap_str.strip() and max_cap is None:
    errors.append("Max Cap is not a valid number (use e.g. 50M or 2B).")
for e in errors:
    st.warning(e)

if run_scan and not errors:
    ma_pair = MA_PAIR_OPTIONS[ma_pair_label]
    config = ScannerConfig(
        min_price=min_price if min_price is not None else 0.0,
        max_price=max_price if max_price is not None else 1_000_000.0,
        min_market_cap=min_cap if min_cap is not None else 0.0,
        max_market_cap=max_cap if max_cap is not None else 100_000_000_000_000.0,
        ma_crossover_pairs=[ma_pair],
        eps_change_threshold=float(eps_threshold),
        trend_window_days=trend_window,
        direction=direction,
    )
    scanner = Scanner(db=db, config=config)
    as_of_date = datetime.now().strftime("%Y-%m-%d")

    with st.spinner("Scanning..."):
        all_signals = scanner.scan(as_of_date)

    # Keep only the most recent signal per ticker
    latest_by_ticker: dict[str, dict] = {}
    for sig in all_signals:
        t = sig["ticker"]
        if t not in latest_by_ticker or sig["trend_change_date"] > latest_by_ticker[t]["trend_change_date"]:
            latest_by_ticker[t] = sig

    raw_results = list(latest_by_ticker.values())

    # Enrich with market cap, price, volume
    tickers = [r["ticker"] for r in raw_results]
    enrichment = db.get_signal_enrichment(tickers)
    for r in raw_results:
        snap = enrichment.get(r["ticker"], {})
        r["market_cap"] = snap.get("market_cap")
        r["latest_close"] = snap.get("latest_close")
        r["avg_dollar_vol"] = snap.get("avg_dollar_vol")

    st.session_state["scan_results"] = raw_results
    st.session_state["scan_sort"] = sort_label

# Display results (persisted across page navigations)
results = st.session_state.get("scan_results")

if results is not None:
    if results:
        # Apply sort
        sort_key, sort_desc = SORT_OPTIONS[sort_label]
        results_sorted = sorted(
            results,
            key=lambda r: (r.get(sort_key) is None, r.get(sort_key) or 0),
            reverse=sort_desc,
        )

        st.metric("Total Signals", len(results_sorted))

        # --- Clickable ticker buttons ---
        st.subheader("Tickers")
        n_cols = min(len(results_sorted), 10)
        rows_chunked = [results_sorted[i : i + n_cols] for i in range(0, len(results_sorted), n_cols)]
        for row_chunk in rows_chunked:
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

        # --- Details table ---
        st.subheader("Details")

        def fmt_cap(v):
            if v is None:
                return "N/A"
            if v >= 1_000_000_000:
                return f"${v / 1_000_000_000:.1f}B"
            return f"${v / 1_000_000:.0f}M"

        def fmt_dvol(v):
            if v is None:
                return "N/A"
            if v >= 1_000_000:
                return f"${v / 1_000_000:.1f}M"
            return f"${v / 1_000:.0f}K"

        table_rows = []
        for r in results_sorted:
            eps_chg = r.get("eps_change_pct")
            table_rows.append({
                "Ticker": r["ticker"],
                "Signal": r.get("signal_type", ""),
                "MA Cross": f"{r.get('fast_ma', '')}/{r.get('slow_ma', '')}",
                "EPS Chg %": f"{eps_chg:+.1f}%" if eps_chg is not None else "N/A",
                "EPS Date": r.get("eps_change_date", ""),
                "Cross Date": r.get("trend_change_date", ""),
                "Days Between": r.get("days_between", ""),
                "Price": f"${r['latest_close']:.2f}" if r.get("latest_close") else "N/A",
                "Mkt Cap": fmt_cap(r.get("market_cap")),
                "Avg $ Vol": fmt_dvol(r.get("avg_dollar_vol")),
            })

        df = pd.DataFrame(table_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info(
            "No signals found for the selected parameters. "
            "Try lowering the EPS threshold, widening the trend window, or running the pipeline to load more data."
        )
