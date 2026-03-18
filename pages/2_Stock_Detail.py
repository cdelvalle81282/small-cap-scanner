import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DB_PATH
from core.database import Database

st.set_page_config(page_title="Stock Detail", page_icon="📈", layout="wide")


@st.cache_resource
def get_db() -> Database:
    db = Database(DB_PATH)
    db.initialize()
    return db


db = get_db()

# --- Ticker selection ---
session_ticker = st.session_state.get("selected_ticker", "")
signal_data = st.session_state.get("signal_data")

with st.sidebar:
    st.header("Stock Detail")
    manual_ticker = st.text_input(
        "Enter ticker",
        value=session_ticker,
        placeholder="e.g. AAPL",
    ).upper().strip()

ticker = manual_ticker or session_ticker

if not ticker:
    st.warning("No ticker selected. Enter a ticker in the sidebar or navigate from the scanner.")
    st.stop()

# --- Load data ---
stock = db.get_stock(ticker)
fundamentals = db.get_fundamentals(ticker)
earnings = db.get_earnings(ticker)

end_date = datetime.today().strftime("%Y-%m-%d")
start_date = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
prices_raw = db.get_daily_prices(ticker, start_date, end_date)

# --- Sidebar stats ---
with st.sidebar:
    st.divider()
    if stock:
        st.subheader(stock.get("name") or ticker)

        def fmt_cap(v):
            if v is None:
                return "N/A"
            if v >= 1_000_000_000:
                return f"${v / 1_000_000_000:.2f}B"
            return f"${v / 1_000_000:.0f}M"

        st.metric("Market Cap", fmt_cap(stock.get("market_cap")))
        st.metric("Sector", stock.get("sector") or "N/A")

        float_shares = stock.get("shares_float")
        if float_shares is not None:
            if float_shares >= 1_000_000:
                float_str = f"{float_shares / 1_000_000:.1f}M"
            else:
                float_str = f"{float_shares:,.0f}"
            st.metric("Float", float_str)
        else:
            st.metric("Float", "N/A")

        si_pct = stock.get("short_interest_pct")
        st.metric("Short Interest %", f"{si_pct:.1f}%" if si_pct is not None else "N/A")

        sr = stock.get("short_ratio")
        st.metric("Short Ratio", f"{sr:.1f}" if sr is not None else "N/A")

        if fundamentals:
            latest = fundamentals[-1]
            st.divider()
            st.subheader("Fundamentals")
            rev = latest.get("revenue")
            if rev is not None:
                if rev >= 1_000_000_000:
                    rev_str = f"${rev / 1_000_000_000:.2f}B"
                else:
                    rev_str = f"${rev / 1_000_000:.0f}M"
                st.metric("Revenue", rev_str)

            gm = latest.get("gross_margin")
            st.metric("Gross Margin", f"{gm:.1f}%" if gm is not None else "N/A")

            om = latest.get("operating_margin")
            st.metric("Operating Margin", f"{om:.1f}%" if om is not None else "N/A")
    else:
        st.info(f"No stock info found for **{ticker}**.")

# --- Main content ---
st.title(f"{ticker}" + (f" — {stock['name']}" if stock and stock.get("name") else ""))

if not prices_raw:
    st.warning(f"No price data available for {ticker} in the last year.")
    st.stop()

df = pd.DataFrame(prices_raw)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# --- Moving averages ---
ma_configs = [
    (20, "SMA 20", "orange", "dash"),
    (50, "SMA 50", "green", "dash"),
    (200, "SMA 200", "red", "dash"),
]

# --- Build chart ---
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.75, 0.25],
    vertical_spacing=0.03,
)

# Price line
fig.add_trace(
    go.Scatter(
        x=df["date"],
        y=df["close"],
        name="Price",
        line=dict(color="royalblue", width=2),
        showlegend=True,
    ),
    row=1,
    col=1,
)

# MA overlays
for period, label, color, dash in ma_configs:
    if len(df) >= period:
        df[f"sma_{period}"] = df["close"].rolling(window=period).mean()
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df[f"sma_{period}"],
                name=label,
                line=dict(color=color, width=1.5, dash=dash),
                showlegend=True,
            ),
            row=1,
            col=1,
        )

# EPS markers
chart_start = df["date"].min()
chart_end = df["date"].max()

for earn in earnings:
    try:
        rd = pd.to_datetime(earn["report_date"])
    except Exception:
        continue
    if rd < chart_start or rd > chart_end:
        continue

    chg = earn.get("eps_change_pct")
    line_color = "green" if (chg is not None and chg >= 0) else "red"
    chg_label = f"+{chg:.1f}%" if (chg is not None and chg >= 0) else (f"{chg:.1f}%" if chg is not None else "N/A")

    fig.add_vline(
        x=rd,
        line=dict(color=line_color, width=1, dash="dot"),
        annotation_text=f"EPS: {chg_label}",
        annotation_position="top",
        annotation=dict(font_size=10, font_color=line_color),
        row=1,
        col=1,
    )

# Signal highlight
if signal_data:
    try:
        trend_date = pd.to_datetime(signal_data.get("trend_change_date"))
        ma_period = signal_data.get("ma_period", "?")
        direction = signal_data.get("signal_type", "Cross")
        fig.add_vline(
            x=trend_date,
            line=dict(color="purple", width=2, dash="solid"),
            annotation_text=f"MA{ma_period} Cross ({direction})",
            annotation_position="top",
            annotation=dict(font_size=11, font_color="purple"),
            row=1,
            col=1,
        )
    except Exception:
        pass

# Volume bars
fig.add_trace(
    go.Bar(
        x=df["date"],
        y=df["volume"],
        name="Volume",
        marker=dict(color="gray", opacity=0.3),
        showlegend=True,
    ),
    row=2,
    col=1,
)

fig.update_layout(
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    xaxis2=dict(rangeslider=dict(visible=False)),
    margin=dict(l=0, r=0, t=40, b=0),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
)
fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)")
fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")

st.plotly_chart(fig, use_container_width=True)

# --- Earnings history table ---
if earnings:
    st.subheader("Earnings History")
    earnings_df = pd.DataFrame(earnings)[
        ["report_date", "period", "eps_actual", "eps_prior", "eps_change_pct"]
    ].copy()
    earnings_df.columns = ["Report Date", "Period", "EPS Actual", "EPS Prior", "EPS Change %"]
    earnings_df = earnings_df.sort_values("Report Date", ascending=False).reset_index(drop=True)

    # Format numeric columns
    for col in ["EPS Actual", "EPS Prior"]:
        earnings_df[col] = earnings_df[col].apply(
            lambda v: f"${v:.2f}" if pd.notna(v) else "N/A"
        )
    earnings_df["EPS Change %"] = earnings_df["EPS Change %"].apply(
        lambda v: f"{v:+.1f}%" if pd.notna(v) else "N/A"
    )

    st.dataframe(earnings_df, hide_index=True, use_container_width=True)
else:
    st.info("No earnings data available for this ticker.")
