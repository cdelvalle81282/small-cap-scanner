import base64
import json
import os
import re
from datetime import timedelta
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import anthropic

_MODEL = "claude-sonnet-4-6"


def build_signal_chart(
    df: pd.DataFrame,
    signal_data: dict,
    earnings: list,
) -> go.Figure:
    """Build a focused candlestick chart around the signal date for AI analysis.

    Window: 90 calendar days before EPS date → 45 calendar days after crossover date.
    """
    eps_date = pd.to_datetime(signal_data.get("eps_change_date"))
    cross_date = pd.to_datetime(signal_data.get("trend_change_date"))

    window_start = eps_date - timedelta(days=90)
    window_end = cross_date + timedelta(days=45)

    subset = df[(df["date"] >= window_start) & (df["date"] <= window_end)].copy()
    if len(subset) < 5:
        subset = df.tail(120).copy()

    # Compute MAs on the full df first so rolling windows are correct
    for period in (20, 50, 200):
        col = f"sma_{period}"
        if col not in df.columns:
            df[col] = df["close"].rolling(window=period).mean()

    subset = subset.merge(
        df[["date", "sma_20", "sma_50", "sma_200"]],
        on="date",
        how="left",
        suffixes=("", "_calc"),
    )
    for col in ("sma_20", "sma_50", "sma_200"):
        calc_col = f"{col}_calc"
        if calc_col in subset.columns:
            subset[col] = subset[calc_col]
            subset.drop(columns=[calc_col], inplace=True)

    # kaleido 1.x uses orjson which can't serialize pandas Timestamps — convert to str
    subset["date"] = subset["date"].dt.strftime("%Y-%m-%d")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
    )

    # Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=subset["date"],
            open=subset["open"],
            high=subset["high"],
            low=subset["low"],
            close=subset["close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )

    # MAs
    ma_styles = [
        ("sma_20", "SMA 20", "orange"),
        ("sma_50", "SMA 50", "#00e676"),
        ("sma_200", "SMA 200", "#ef5350"),
    ]
    for col, label, color in ma_styles:
        if col in subset.columns:
            fig.add_trace(
                go.Scatter(
                    x=subset["date"],
                    y=subset[col],
                    name=label,
                    line=dict(color=color, width=1.5, dash="dash"),
                ),
                row=1, col=1,
            )

    # EPS marker
    date_min, date_max = subset["date"].min(), subset["date"].max()
    for earn in earnings:
        try:
            rd = pd.to_datetime(earn["report_date"]).strftime("%Y-%m-%d")
        except Exception:
            continue
        if rd < date_min or rd > date_max:
            continue
        chg = earn.get("eps_change_pct")
        color = "lime" if (chg is not None and chg >= 0) else "#ef5350"
        label = f"EPS {chg:+.1f}%" if chg is not None else "EPS"
        fig.add_vline(x=rd, line_color=color, line_dash="dot", line_width=1, row=1, col=1)
        fig.add_annotation(
            x=rd, y=1, yref="y domain",
            text=label, showarrow=False,
            font=dict(size=10, color=color), yanchor="bottom",
            row=1, col=1,
        )

    # Signal/crossover marker
    cross_date_str = cross_date.strftime("%Y-%m-%d")
    fig.add_vline(x=cross_date_str, line_color="purple", line_width=2, row=1, col=1)
    direction = signal_data.get("signal_type", "cross")
    fast = signal_data.get("fast_ma", "")
    slow = signal_data.get("slow_ma", "")
    fig.add_annotation(
        x=cross_date_str, y=0.95, yref="y domain",
        text=f"SMA{fast}/{slow} {direction}",
        showarrow=False,
        font=dict(size=10, color="purple"), yanchor="top",
        row=1, col=1,
    )

    # Volume bars
    vol_colors = [
        "#26a69a" if c >= o else "#ef5350"
        for c, o in zip(subset["close"], subset["open"])
    ]
    fig.add_trace(
        go.Bar(
            x=subset["date"],
            y=subset["volume"],
            name="Volume",
            marker=dict(color=vol_colors, opacity=0.5),
            showlegend=False,
        ),
        row=2, col=1,
    )

    fig.update_layout(
        height=600,
        width=1200,
        title=f"{signal_data.get('ticker', '')} — Signal Chart",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_rangeslider_visible=False,
        plot_bgcolor="#1e1e2e",
        paper_bgcolor="#1e1e2e",
        font=dict(color="white"),
    )
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)")

    return fig


def analyze_chart(fig: go.Figure, signal_data: dict) -> dict:
    """Export chart to PNG and send to Claude vision for technical analysis.

    Returns a dict with keys:
        text: full markdown analysis (JSON block stripped out)
        levels: list of {type, price, label} dicts
        trend_break_price: float or None
        trend_break_condition: str
    """
    img_bytes = fig.to_image(format="png", scale=2)
    img_b64 = base64.standard_b64encode(img_bytes).decode("utf-8")

    ticker = signal_data.get("ticker", "")
    direction = signal_data.get("signal_type", "")
    eps_chg = signal_data.get("eps_change_pct") or 0
    fast_ma = signal_data.get("fast_ma", "")
    slow_ma = signal_data.get("slow_ma", "")
    cross_date = signal_data.get("trend_change_date", "")
    eps_date = signal_data.get("eps_change_date", "")
    days_between = signal_data.get("days_between", "")
    trade_side = "long" if direction == "bullish" else "short"

    prompt = f"""You are an expert technical analyst reviewing a daily candlestick chart for {ticker}, a small/mid-cap stock.

Signal context:
- Direction: {direction} ({trade_side} trade idea)
- EPS change: {eps_chg:+.1f}% reported on {eps_date} (dotted vertical line)
- SMA{fast_ma}/SMA{slow_ma} crossover on {cross_date} (purple vertical line), {days_between} days after the EPS report
- Overlays: SMA 20 (orange), SMA 50 (green), SMA 200 (red), volume bars below

IMPORTANT: Begin your response with a ```json code block in exactly this format (replace example values with real ones from the chart):
```json
{{
  "levels": [
    {{"type": "resistance", "price": 11.25, "label": "former support turned ceiling, SMAs converging"}},
    {{"type": "support", "price": 9.00, "label": "EPS spike low"}}
  ],
  "trend_break_price": 11.25,
  "trend_break_condition": "Close above $11.25 on volume would invalidate the bearish thesis"
}}
```
Include 2-4 levels. trend_break_price is the single most critical level whose breach invalidates the primary trade direction.

Then provide the full technical analysis in markdown with these sections:

**Trend Direction**
What is the primary trend at the time of the crossover? Is this signal aligned with or fighting the larger trend?

**Chart Pattern**
Identify any recognizable pattern (base, breakout, consolidation, cup/handle, flag, wedge, etc.). If none, say so.

**Support & Resistance**
The 2-3 most significant levels visible, with approximate price values.

**Candlestick Patterns**
Notable candlestick patterns in the 5-10 bars around the signal date.

**Volume Analysis**
Does volume confirm the move? Any surges, dry-ups, or divergences worth noting?

**Setup Quality**
Rate: Strong / Moderate / Weak — with a 1-2 sentence reason.

**Key Risk**
The single biggest technical risk to this {trade_side} trade.

Be specific and concise. Reference price levels where visible."""

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("ANTHROPIC_API_KEY")
        except Exception:
            pass
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. Add it to .env, environment variables, "
            "or Streamlit secrets (Settings → Secrets in Streamlit Cloud)."
        )

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=_MODEL,
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )

    response_text = message.content[0].text

    # Parse the structured JSON block
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        try:
            structured = json.loads(json_match.group(1))
        except json.JSONDecodeError:
            structured = {"levels": [], "trend_break_price": None, "trend_break_condition": ""}
    else:
        structured = {"levels": [], "trend_break_price": None, "trend_break_condition": ""}

    # Strip the JSON block from the text portion
    text = re.sub(r'```json.*?```', '', response_text, flags=re.DOTALL).strip()

    return {
        "text": text,
        "levels": structured.get("levels", []),
        "trend_break_price": structured.get("trend_break_price"),
        "trend_break_condition": structured.get("trend_break_condition", ""),
    }
