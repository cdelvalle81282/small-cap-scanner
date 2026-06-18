import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
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


# --- Trendline helpers ---

def find_swing_points(df: pd.DataFrame, lookback: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find swing highs and lows using a rolling window comparison."""
    highs = []
    lows = []
    for i in range(lookback, len(df) - lookback):
        window_high = df["high"].iloc[i - lookback : i + lookback + 1]
        if df["high"].iloc[i] == window_high.max():
            highs.append(i)
        window_low = df["low"].iloc[i - lookback : i + lookback + 1]
        if df["low"].iloc[i] == window_low.min():
            lows.append(i)
    return df.iloc[highs] if highs else pd.DataFrame(), df.iloc[lows] if lows else pd.DataFrame()


def build_trendline(points: pd.DataFrame, price_col: str, df: pd.DataFrame):
    """Fit a line through the last 2+ swing points and extend to chart end."""
    if len(points) < 2:
        return None
    # Use last 2 swing points
    pts = points.tail(2)
    x_idx = np.array([df.index.get_loc(i) if i in df.index else 0 for i in pts.index])
    if x_idx[1] == x_idx[0]:
        return None
    y_vals = pts[price_col].values.astype(float)
    slope = (y_vals[1] - y_vals[0]) / (x_idx[1] - x_idx[0])
    intercept = y_vals[0] - slope * x_idx[0]
    # Extend from first swing point to end of chart
    x_range = np.arange(x_idx[0], len(df))
    y_range = slope * x_range + intercept
    dates = df["date"].iloc[x_range]
    return dates, y_range


def build_regression_channel(df: pd.DataFrame, window: int = 90, num_std: float = 1.5):
    """Fit a linear regression on the last `window` closes with ±std bands."""
    subset = df.tail(window).copy()
    if len(subset) < 20:
        return None
    x = np.arange(len(subset))
    y = subset["close"].values.astype(float)
    coeffs = np.polyfit(x, y, 1)
    fit = np.polyval(coeffs, x)
    residuals = y - fit
    std = residuals.std()
    return subset["date"], fit, fit + num_std * std, fit - num_std * std


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
start_date = "2020-01-01"
prices_raw = db.get_daily_prices(ticker, start_date, end_date)

# Chart overlay defaults (controls only rendered after data is confirmed)
show_ma = True
show_trendlines = True
show_regression = False
swing_lookback = 5
reg_window, reg_std = 90, 1.5

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
    st.warning(f"No price data available for {ticker}. Run the pipeline to load data.")
    st.stop()

df = pd.DataFrame(prices_raw)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# --- Sidebar: chart overlays (only shown when data exists) ---
with st.sidebar:
    st.divider()
    st.subheader("Chart Overlays")
    show_ma = st.checkbox("Moving Averages", value=True)
    show_trendlines = st.checkbox("Support / Resistance", value=True)
    show_regression = st.checkbox("Regression Channel", value=False)
    if show_trendlines:
        swing_lookback = st.slider("Swing Lookback (bars)", min_value=3, max_value=15, value=5)
    else:
        swing_lookback = 5
    if show_regression:
        reg_window = st.slider("Regression Window (days)", min_value=30, max_value=250, value=90)
        reg_std = st.slider("Channel Width (std dev)", min_value=1.0, max_value=3.0, value=1.5, step=0.25)
    else:
        reg_window, reg_std = 90, 1.5

# --- Build chart ---
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.75, 0.25],
    vertical_spacing=0.03,
)

# Candlestick chart
fig.add_trace(
    go.Candlestick(
        x=df["date"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Price",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        showlegend=True,
    ),
    row=1,
    col=1,
)

# MA overlays — solid lines
if show_ma:
    ma_configs = [
        (20,  "SMA 20",  "#ff9800", 1.5),
        (50,  "SMA 50",  "#4caf50", 1.5),
        (200, "SMA 200", "#f44336", 2.0),
    ]
    for period, label, color, width in ma_configs:
        if len(df) >= period:
            df[f"sma_{period}"] = df["close"].rolling(window=period).mean()
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df[f"sma_{period}"],
                    name=label,
                    line=dict(color=color, width=width),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

# Swing-based support/resistance trendlines
if show_trendlines:
    swing_highs, swing_lows = find_swing_points(df, lookback=swing_lookback)

    # Resistance line (swing highs) — solid red
    if len(swing_highs) >= 2:
        result = build_trendline(swing_highs, "high", df)
        if result is not None:
            dates, y_vals = result
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=y_vals,
                    name="Resistance",
                    line=dict(color="#ef5350", width=2.0),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

    # Support line (swing lows) — solid green
    if len(swing_lows) >= 2:
        result = build_trendline(swing_lows, "low", df)
        if result is not None:
            dates, y_vals = result
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=y_vals,
                    name="Support",
                    line=dict(color="#26a69a", width=2.0),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

# Linear regression channel
if show_regression:
    reg_result = build_regression_channel(df, window=reg_window, num_std=reg_std)
    if reg_result is not None:
        reg_dates, reg_fit, reg_upper, reg_lower = reg_result
        fig.add_trace(
            go.Scatter(
                x=reg_dates,
                y=reg_fit,
                name="Regression",
                line=dict(color="yellow", width=1.5),
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=reg_dates,
                y=reg_upper,
                name="Upper Band",
                line=dict(color="yellow", width=1, dash="dot"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=reg_dates,
                y=reg_lower,
                name="Lower Band",
                line=dict(color="yellow", width=1, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(255, 255, 0, 0.05)",
                showlegend=False,
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

    fig.add_shape(
        type="line",
        x0=rd, x1=rd, y0=0, y1=1,
        yref="y domain",
        line=dict(color=line_color, width=1, dash="dot"),
        row=1,
        col=1,
    )
    fig.add_annotation(
        x=rd,
        y=1,
        yref="y domain",
        text=f"EPS: {chg_label}",
        showarrow=False,
        font=dict(size=10, color=line_color),
        yanchor="bottom",
        row=1,
        col=1,
    )

# Signal highlight
if signal_data:
    try:
        trend_date = pd.to_datetime(signal_data.get("trend_change_date"))
        fast_ma = signal_data.get("fast_ma", "?")
        slow_ma = signal_data.get("slow_ma", "?")
        direction = signal_data.get("signal_type", "Cross")
        fig.add_shape(
            type="line",
            x0=trend_date, x1=trend_date, y0=0, y1=1,
            yref="y domain",
            line=dict(color="purple", width=2, dash="solid"),
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=trend_date,
            y=1,
            yref="y domain",
            text=f"SMA{fast_ma}/{slow_ma} Cross ({direction})",
            showarrow=False,
            font=dict(size=11, color="purple"),
            yanchor="bottom",
            row=1,
            col=1,
        )
    except Exception:
        pass

# Volume bars — color by up/down day
vol_colors = [
    "#26a69a" if c >= o else "#ef5350"
    for c, o in zip(df["close"], df["open"])
]
fig.add_trace(
    go.Bar(
        x=df["date"],
        y=df["volume"],
        name="Volume",
        marker=dict(color=vol_colors, opacity=0.4),
        showlegend=False,
    ),
    row=2,
    col=1,
)

# Default view: 1 year ending at the last data point (no future space)
chart_end = df["date"].max().strftime("%Y-%m-%d")
chart_1y  = (df["date"].max() - pd.DateOffset(years=1)).strftime("%Y-%m-%d")

fig.update_layout(
    height=650,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    xaxis_rangeslider_visible=False,
    xaxis2=dict(rangeslider=dict(visible=False)),
    margin=dict(l=0, r=0, t=40, b=0),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
)
fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)")
fig.update_xaxes(
    gridcolor="rgba(255,255,255,0.05)",
    rangebreaks=[dict(bounds=["sat", "mon"])],
)
# Range selector on the price panel only; default to 1Y with no future gap
fig.update_xaxes(
    range=[chart_1y, chart_end],
    rangeselector=dict(
        buttons=[
            dict(count=3,  label="3M", step="month", stepmode="backward"),
            dict(count=6,  label="6M", step="month", stepmode="backward"),
            dict(count=9,  label="9M", step="month", stepmode="backward"),
            dict(count=1,  label="1Y", step="year",  stepmode="backward"),
            dict(label="All", step="all"),
        ],
        activecolor="#4caf50",
        bgcolor="#1e1e2e",
        bordercolor="#555",
        borderwidth=1,
        font=dict(color="white", size=11),
        x=0, y=1.0,
        xanchor="left",
        yanchor="bottom",
    ),
    row=1, col=1,
)

# --- Overlay AI-identified support/resistance levels if analysis has been run ---
for key, val in st.session_state.items():
    if key.startswith(f"ai_analysis_{ticker}_") and isinstance(val, dict):
        levels = val.get("levels", [])
        trend_break = val.get("trend_break_price")
        trend_break_prices = {l.get("price") for l in levels}

        for level in levels:
            price = level.get("price")
            ltype = level.get("type", "")
            label = level.get("label", "")
            if not price:
                continue
            color = "#ef5350" if ltype == "resistance" else "#26a69a"
            fig.add_hline(
                y=price,
                line_color=color,
                line_dash="solid",
                line_width=2.0,
                annotation_text=f"  {ltype.title()} ${price:.2f} — {label[:40]}",
                annotation_position="top left",
                annotation_font=dict(color=color, size=10),
                row=1, col=1,
            )

        # Trend break line in purple if it's a distinct price
        if trend_break and trend_break not in trend_break_prices:
            fig.add_hline(
                y=trend_break,
                line_color="#b388ff",
                line_dash="solid",
                line_width=2.0,
                annotation_text=f"  Trend Break ${trend_break:.2f}",
                annotation_position="top left",
                annotation_font=dict(color="#b388ff", size=10),
                row=1, col=1,
            )
        break

st.plotly_chart(fig, use_container_width=True)

# --- AI Chart Analysis ---
st.divider()
st.subheader("AI Chart Analysis")

import json as _json
from datetime import timedelta as _timedelta

cache_key = f"ai_analysis_{ticker}"

# Load from DB into session_state if not already there
if cache_key not in st.session_state:
    saved = db.get_ai_analysis(ticker)
    if saved:
        st.session_state[cache_key] = {
            "text": saved.get("text", ""),
            "levels": saved.get("levels", []),
            "trend_break_price": saved.get("trend_break_price"),
            "trend_break_condition": saved.get("trend_break_condition", ""),
            "_analysis_date": saved.get("analysis_date", ""),
        }


def render_analysis(result: dict) -> None:
    levels = result.get("levels", [])
    if levels:
        st.subheader("Key Price Levels")
        levels_df = pd.DataFrame(levels)[["type", "price", "label"]]
        levels_df.columns = ["Type", "Price", "Significance"]
        levels_df["Price"] = levels_df["Price"].apply(lambda p: f"${p:.2f}")
        st.dataframe(levels_df, hide_index=True, use_container_width=True)

    trend_break = result.get("trend_break_condition")
    if trend_break:
        st.info(f"**Trend break condition:** {trend_break}")

    st.markdown(result.get("text", ""))


def run_analysis() -> None:
    from core.chart_analyzer import analyze_chart, build_signal_chart
    with st.spinner("Building signal chart and analyzing with Claude..."):
        try:
            sig = signal_data or {
                "ticker": ticker,
                "signal_type": "bullish",
                "eps_change_pct": None,
                "eps_change_date": "",
                "trend_change_date": df["date"].max().strftime("%Y-%m-%d"),
                "days_between": 0,
                "fast_ma": 20,
                "slow_ma": 50,
            }
            signal_fig = build_signal_chart(df, sig, earnings)
            result = analyze_chart(signal_fig, sig)
            result["_analysis_date"] = datetime.now().strftime("%Y-%m-%d")
            st.session_state[cache_key] = result
            db.save_ai_analysis(
                ticker, result,
                signal_date=sig.get("trend_change_date"),
            )
            st.rerun()
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Analysis failed: {e}")


if cache_key in st.session_state:
    result = st.session_state[cache_key]

    analysis_date = result.get("_analysis_date", "")
    col_hdr, col_rerun = st.columns([5, 1])
    with col_hdr:
        if analysis_date:
            st.caption(f"Analysis from {analysis_date}")
    with col_rerun:
        if st.button("Re-run", help="Re-run AI analysis to get fresh chart read"):
            run_analysis()

    render_analysis(result)

    # Watchlist button (only when we have a real signal)
    if signal_data:
        existing = db.get_watchlist_entry(ticker)
        if existing:
            st.success(f"On watchlist — expires {existing['expiry_date']}")
            if st.button("Remove from Watchlist"):
                db.remove_from_watchlist(existing["id"])
                st.rerun()
        else:
            if st.button("Add to Watchlist (30-day monitoring)", type="primary"):
                eps_date = signal_data.get("eps_change_date", "")
                expiry = (
                    (datetime.fromisoformat(eps_date) + _timedelta(days=30)).strftime("%Y-%m-%d")
                    if eps_date else ""
                )
                db.add_to_watchlist({
                    "ticker": ticker,
                    "signal_type": signal_data.get("signal_type", ""),
                    "eps_change_pct": signal_data.get("eps_change_pct"),
                    "eps_date": eps_date,
                    "signal_date": signal_data.get("trend_change_date", ""),
                    "fast_ma": signal_data.get("fast_ma"),
                    "slow_ma": signal_data.get("slow_ma"),
                    "levels_json": _json.dumps(result.get("levels", [])),
                    "trend_break_price": result.get("trend_break_price"),
                    "trend_break_condition": result.get("trend_break_condition", ""),
                    "ai_analysis": result.get("text", ""),
                    "expiry_date": expiry,
                    "added_date": datetime.now().strftime("%Y-%m-%d"),
                })
                st.success(f"Added {ticker} to watchlist until {expiry}")
                st.rerun()
else:
    if st.button("Analyze Chart with AI", type="primary"):
        run_analysis()

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
