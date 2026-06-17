"""
Interactive chart viewer for C17 signals.

Generates a self-contained HTML file (no CDN needed) with:
- 1-year daily candlestick chart
- 20 / 50 / 200-day SMAs
- All EPS dates marked (orange)
- MA crossover date marked (green=bullish, red=bearish)
- Entry price line
- 30-day return in the title

Usage:
  python chart_signals.py                      # all 2025+ signals
  python chart_signals.py --from 2025-06-01    # custom start
  python chart_signals.py --ticker RDW         # one ticker
  python chart_signals.py --winners            # only +10%+ at 30d
  python chart_signals.py --losers             # only sub -10% at 30d
  python chart_signals.py --limit 30           # cap number of charts
"""

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))

from config import DB_PATH, ScannerConfig, BacktestConfig
from core.database import Database
from core.backtest import Backtester
from analysis_winners_losers import enrich_deep, filt_c17_both
from analysis_exit_strategies import trade_result

LOOKBACK_DAYS = 365
FORWARD_DAYS  = 50


def get_chart_data(db, ticker, signal_date):
    signal_dt = date.fromisoformat(signal_date)
    start = (signal_dt - timedelta(days=LOOKBACK_DAYS)).isoformat()
    end   = (signal_dt + timedelta(days=FORWARD_DAYS)).isoformat()
    rows  = db.get_daily_prices(ticker, start, end)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["sma20"]  = df["close"].rolling(20).mean()
    df["sma50"]  = df["close"].rolling(50).mean()
    df["sma200"] = df["close"].rolling(200).mean()
    return df


def get_all_eps_dates(db, ticker):
    rows = db.get_earnings(ticker)
    if not rows:
        return []
    return [r["report_date"] for r in rows if r.get("report_date")]


def make_chart(sig, df, all_eps_dates, ret30):
    signal_dt = pd.Timestamp(sig["trend_date"])
    direction = sig["signal_type"]
    cross_color = "#26a69a" if direction == "bullish" else "#ef5350"
    ret_str   = f"+{ret30:.1f}%" if ret30 > 0 else f"{ret30:.1f}%"
    ret_color = "lime" if ret30 > 0 else "tomato"

    title = (
        f"<b>{sig['ticker']}</b>  {direction.upper()}  |  "
        f"EPS YoY: {sig['eps_change_pct']:+.1f}%  |  "
        f"Days EPS→cross: {sig['days_between']}  |  "
        f"30d return: <span style='color:{ret_color}'>{ret_str}</span>"
    )

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.02,
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df["date"],
        open=df["open"], high=df["high"],
        low=df["low"],   close=df["close"],
        name="Price",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a",
        decreasing_fillcolor="#ef5350",
    ), row=1, col=1)

    # SMAs
    for col, clr, dash, w in [
        ("sma20",  "#FFD700", "solid",  1.5),
        ("sma50",  "#FF8C00", "dash",   1.5),
        ("sma200", "#9370DB", "dot",    2.0),
    ]:
        valid = df.dropna(subset=[col])
        fig.add_trace(go.Scatter(
            x=valid["date"], y=valid[col],
            name=col.upper(),
            line=dict(color=clr, width=w, dash=dash),
        ), row=1, col=1)

    # Volume bars
    vol_colors = [
        "#26a69a" if c >= o else "#ef5350"
        for c, o in zip(df["close"], df["open"])
    ]
    fig.add_trace(go.Bar(
        x=df["date"], y=df["volume"],
        name="Volume",
        marker_color=vol_colors,
        opacity=0.55,
        showlegend=False,
    ), row=2, col=1)

    # All EPS dates (orange vertical lines)
    chart_start = df["date"].min()
    chart_end   = df["date"].max()
    for eps_date in all_eps_dates:
        eps_dt = pd.Timestamp(eps_date)
        if not (chart_start <= eps_dt <= chart_end):
            continue
        fig.add_vline(
            x=eps_dt.timestamp() * 1000,
            line_width=1.5, line_dash="dot", line_color="#FFA500",
            row="all", col=1,
        )
        # Label on price chart
        eps_row = df[df["date"] == eps_dt]
        y_pos = eps_row["high"].values[0] if not eps_row.empty else df["high"].max()
        fig.add_annotation(
            x=eps_dt, y=y_pos,
            text="E", showarrow=False,
            font=dict(color="#FFA500", size=10, family="monospace"),
            yshift=8, row=1, col=1,
        )

    # MA crossover date (signal)
    fig.add_vline(
        x=signal_dt.timestamp() * 1000,
        line_width=2.5, line_dash="dash", line_color=cross_color,
        row="all", col=1,
    )
    cross_row = df[df["date"] == signal_dt]
    y_cross = cross_row["high"].values[0] if not cross_row.empty else df["high"].max()
    fig.add_annotation(
        x=signal_dt, y=y_cross,
        text="X", showarrow=True, arrowhead=2,
        arrowcolor=cross_color,
        font=dict(color=cross_color, size=12, family="monospace"),
        yshift=12, row=1, col=1,
    )

    # Entry price line
    after = df[df["date"] > signal_dt]
    if not after.empty:
        ep = (df.iloc[after.index[0]]["high"] + df.iloc[after.index[0]]["low"]) / 2
        fig.add_hline(
            y=ep, line_width=1, line_dash="longdash", line_color="#888888",
            annotation_text=f"  entry {ep:.2f}",
            annotation_font=dict(color="#888888", size=10),
            row=1, col=1,
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=13), x=0),
        xaxis_rangeslider_visible=False,
        height=520,
        margin=dict(l=60, r=60, t=55, b=10),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#d0d0d0", size=11),
        legend=dict(orientation="h", y=1.04, x=0, font=dict(size=10)),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#2a2a3e", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#2a2a3e", zeroline=False)

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from",    dest="from_date", default="2025-01-01")
    parser.add_argument("--ticker",  default=None)
    parser.add_argument("--winners", action="store_true")
    parser.add_argument("--losers",  action="store_true")
    parser.add_argument("--limit",   type=int, default=None)
    parser.add_argument("--out",     default="signal_charts.html")
    args = parser.parse_args()

    db = Database(DB_PATH)
    sc = ScannerConfig()
    bc = BacktestConfig(start_date="2018-01-01", end_date="2026-06-09")

    print("Loading signals...")
    bt = Backtester(db, sc, bc)
    results = bt.run()
    enriched = enrich_deep(db, results["signals"])
    c17 = filt_c17_both(enriched)

    signals = [s for s in c17 if s["trend_date"] >= args.from_date]
    if args.ticker:
        signals = [s for s in signals if s["ticker"] == args.ticker.upper()]
    signals.sort(key=lambda x: x["trend_date"], reverse=True)

    print(f"Computing returns for {len(signals)} signals...")
    rows = []
    for sig in signals:
        df = get_chart_data(db, sig["ticker"], sig["trend_date"])
        if df is None or len(df) < 20:
            continue
        signal_dt = pd.Timestamp(sig["trend_date"])
        after = df[df["date"] > signal_dt]
        if after.empty:
            continue
        entry_idx = after.index[0]
        ep = (df.iloc[entry_idx]["high"] + df.iloc[entry_idx]["low"]) / 2
        if ep <= 0:
            continue
        i30 = min(entry_idx + 30, len(df) - 1)
        ret30 = trade_result(ep, df.iloc[i30]["close"], sig["signal_type"])
        if args.winners and ret30 < 10:
            continue
        if args.losers and ret30 > -10:
            continue
        rows.append((sig, df, ret30))

    if args.limit:
        rows = rows[:args.limit]

    print(f"Generating {len(rows)} charts...")
    html_parts = []
    for i, (sig, df, ret30) in enumerate(rows):
        all_eps = get_all_eps_dates(db, sig["ticker"])
        fig = make_chart(sig, df, all_eps, ret30)
        include_js = (i == 0)  # embed Plotly once in the first chart
        html_parts.append(fig.to_html(
            full_html=False,
            include_plotlyjs=include_js,
        ))

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>C17 Signal Charts</title>
<style>
  body {{
    background: #0f0f1a;
    color: #d0d0d0;
    font-family: 'Segoe UI', sans-serif;
    margin: 16px;
  }}
  h2 {{ color: #9370DB; margin-bottom: 4px; }}
  p  {{ color: #888; margin-top: 0; font-size: 13px; }}
  .chart-wrap {{ margin-bottom: 8px; }}
  hr {{ border: none; border-top: 1px solid #2a2a3e; margin: 12px 0; }}
  .legend {{
    font-size: 12px; color: #aaa;
    margin-bottom: 16px; line-height: 1.8;
  }}
  .leg-eps   {{ color: #FFA500; font-weight: bold; }}
  .leg-cross {{ color: #26a69a; font-weight: bold; }}
  .leg-entry {{ color: #888;    font-weight: bold; }}
</style>
</head>
<body>
<h2>C17 Signal Charts</h2>
<p>{args.from_date} onward &mdash; {len(rows)} signals</p>
<div class="legend">
  <span class="leg-eps">&#9646; E = EPS report date (orange dotted)</span> &nbsp;|&nbsp;
  <span class="leg-cross">&#9646; X = MA crossover / signal date (colored dashed)</span> &nbsp;|&nbsp;
  <span class="leg-entry">&#9646; Gray dashed = entry price (next-day midpoint)</span>
</div>
{"<hr>".join(f'<div class="chart-wrap">{p}</div>' for p in html_parts)}
</body>
</html>"""

    out = Path(args.out)
    out.write_text(html, encoding="utf-8")
    print(f"Saved: {out.resolve()}")
    print("Open in browser: file:///" + str(out.resolve()).replace("\\", "/"))


if __name__ == "__main__":
    main()
