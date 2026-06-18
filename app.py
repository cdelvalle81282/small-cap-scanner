import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from config import DB_PATH
from core.database import Database

st.set_page_config(page_title="Small Cap Scanner", page_icon="📊", layout="wide")
from core.styles import inject_styles
inject_styles()
st.title("Small Cap EPS + Trend Scanner")
st.markdown(
    "Identifies small cap stocks ($1-$50) where a moving average trend change occurs "
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
    st.warning("No data in database. Run the pipeline first.")
else:
    st.success("Database loaded. Use the sidebar to navigate.")

# Alert banner
unread_alerts = db.get_unread_alerts()
if unread_alerts:
    st.error(
        f"**{len(unread_alerts)} price alert{'s' if len(unread_alerts) > 1 else ''}** — "
        f"[View in Tracking](/Tracking)"
    )

# ── Pipeline runner ───────────────────────────────────────────────────────────
st.divider()
st.subheader("Data Pipeline")
st.markdown("Fetch stock data and store in the local database.")


def parse_cap(val: str) -> float | None:
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


# Date range
col1, col2 = st.columns(2)
with col1:
    pipeline_start = st.date_input(
        "Pipeline Start", value=pd.to_datetime("2022-01-01"), key="pipe_start"
    )
with col2:
    pipeline_end = st.date_input(
        "Pipeline End", value=pd.Timestamp.today(), key="pipe_end"
    )

# Universe settings
st.caption("Universe Filters")
col1, col2, col3, col4 = st.columns(4)
with col1:
    price_min_str = st.text_input("Min Price ($)", value="1", placeholder="e.g. 1")
with col2:
    price_max_str = st.text_input("Max Price ($)", value="50", placeholder="e.g. 50")
with col3:
    cap_floor_str = st.text_input("Market Cap Floor", value="50M", placeholder="e.g. 50M")
with col4:
    cap_ceil_str = st.text_input("Market Cap Ceiling", value="10B", placeholder="e.g. 10B")

price_min = float(price_min_str) if price_min_str.strip() else 1.0
price_max = float(price_max_str) if price_max_str.strip() else 50.0
cap_floor = parse_cap(cap_floor_str)
cap_ceil  = parse_cap(cap_ceil_str)

# Provider
provider_choice = st.selectbox(
    "Data Provider",
    options=["yfinance (hardcoded universe)", "Polygon (dynamic NYSE/NASDAQ universe)"],
    help="Polygon dynamically discovers all active NYSE/NASDAQ/AMEX stocks meeting your price filter. Requires POLYGON_API_KEY in .env.",
)
use_polygon = provider_choice.startswith("Polygon")

if use_polygon:
    st.info(
        f"Polygon will discover all active NYSE/NASDAQ/AMEX common stocks priced ${price_min:.0f}–${price_max:.0f}. "
        f"Market cap filter ${cap_floor_str}–{cap_ceil_str} applied per-ticker during pipeline. "
        "Expect 1,500–3,000 tickers and a longer first run (~2–3 hrs)."
    )

if st.button("Run Pipeline", type="primary"):
    import os
    from pipeline import Pipeline

    if use_polygon:
        from core.providers.polygon_provider import PolygonProvider
        api_key = os.environ.get("POLYGON_API_KEY")
        if not api_key:
            st.error("POLYGON_API_KEY not set in .env file.")
            st.stop()
        provider = PolygonProvider(api_key)
    else:
        from core.providers.yfinance_provider import YFinanceProvider
        provider = YFinanceProvider()

    pipeline = Pipeline(
        db=db,
        provider=provider,
        min_market_cap=cap_floor,
        max_market_cap=cap_ceil,
    )
    tickers = provider.get_small_cap_universe(price_min, price_max)
    total = len(tickers)

    progress = st.progress(0, text="Starting pipeline...")
    status = st.empty()
    status.text(f"Found {total} candidates (market cap filter applied per-ticker)")

    skipped = 0
    for i, ticker in enumerate(tickers):
        try:
            progress.progress(
                (i + 1) / total,
                text=f"Processing {ticker} ({i + 1}/{total}, skipped {skipped})",
            )
            result = pipeline._process_ticker(
                ticker,
                pipeline_start.strftime("%Y-%m-%d"),
                pipeline_end.strftime("%Y-%m-%d"),
            )
            if result == "skipped":
                skipped += 1
        except Exception as e:
            status.warning(f"Skipped {ticker}: {e}")
            skipped += 1
        if (i + 1) % 5 == 0:
            time.sleep(1)

    progress.progress(1.0, text="Complete!")
    st.success(
        f"Pipeline finished. Processed {total - skipped} tickers "
        f"({skipped} skipped — outside market cap range)."
    )
    st.rerun()
