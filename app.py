import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from config import DB_PATH
from core.database import Database

st.set_page_config(page_title="Small Cap Scanner", page_icon="📊", layout="wide")
st.title("Small Cap EPS + Trend Scanner")
st.markdown(
    "Identifies small cap stocks ($1-$20) where a moving average trend change occurs "
    "within a configurable window of an earnings-per-share change."
)


@st.cache_resource
def get_db() -> Database:
    db = Database(DB_PATH)
    db.initialize()
    return db


db = get_db()

# DB status
stock_count = len(db.get_stock_universe(0, 100, 0, 100_000_000_000))
st.metric("Stocks in Database", stock_count)

if stock_count == 0:
    st.warning("No data in database. Run the pipeline first: `python pipeline.py`")
else:
    st.success("Database loaded. Use the sidebar to navigate.")

# Pipeline runner
st.divider()
st.subheader("Data Pipeline")
st.markdown("Fetch stock data from YFinance and store in the local database.")

col1, col2 = st.columns(2)
with col1:
    pipeline_start = st.date_input(
        "Pipeline Start", value=pd.to_datetime("2022-01-01"), key="pipe_start"
    )
with col2:
    pipeline_end = st.date_input(
        "Pipeline End", value=pd.to_datetime("2026-03-17"), key="pipe_end"
    )

if st.button("Run Pipeline", type="primary"):
    from core.providers.yfinance_provider import YFinanceProvider
    from pipeline import Pipeline

    provider = YFinanceProvider()
    pipeline = Pipeline(db=db, provider=provider)
    tickers = provider.get_small_cap_universe(1.0, 50.0)
    total = len(tickers)

    progress = st.progress(0, text="Starting pipeline...")
    status = st.empty()
    status.text(f"Found {total} candidates")

    for i, ticker in enumerate(tickers):
        try:
            progress.progress(
                (i + 1) / total, text=f"Processing {ticker} ({i + 1}/{total})"
            )
            pipeline._process_ticker(
                ticker,
                pipeline_start.strftime("%Y-%m-%d"),
                pipeline_end.strftime("%Y-%m-%d"),
            )
        except Exception as e:
            status.warning(f"Skipped {ticker}: {e}")
        if (i + 1) % 5 == 0:
            time.sleep(1)

    progress.progress(1.0, text="Complete!")
    st.success(f"Pipeline finished. Processed {total} tickers.")
    st.rerun()
