"""
Daily signal scanner for automated alerts.
Detects new EPS + MA crossover signals from the last 3 calendar days.

Run via GitHub Actions (see .github/workflows/daily_scan.yml) or manually:
    python scan_and_notify.py

Environment variables (set in GitHub Actions secrets or .env):
    SMTP_USERNAME   — Gmail address to send from
    SMTP_PASSWORD   — Gmail App Password (not your regular password)
    SMTP_TO         — recipient address(es), comma-separated
    SLACK_WEBHOOK_URL — Slack incoming webhook URL (optional)
"""

import os
import smtplib
import sys
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))

from core.providers.yfinance_provider import SMALL_CAP_UNIVERSE

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass


# ── Config ────────────────────────────────────────────────────────────────────

EPS_CHANGE_THRESHOLD = 10.0   # % minimum EPS change to qualify
TREND_WINDOW_DAYS    = 30     # days after EPS to look for MA crossover
MA_PAIRS             = [(20, 50), (50, 200)]
LOOKBACK_DAYS        = 3      # report signals from the last N calendar days
PRICE_HISTORY_MONTHS = "6mo"  # yfinance period for price data


# ── Price + MA logic ──────────────────────────────────────────────────────────

def find_new_crossovers(close: pd.Series, eps_events: list[dict], today: datetime) -> list[dict]:
    """Find MA crossovers within TREND_WINDOW_DAYS of any EPS event,
    where the crossover happened in the last LOOKBACK_DAYS."""
    cutoff = today - timedelta(days=LOOKBACK_DAYS)
    signals = []

    for fast, slow in MA_PAIRS:
        if len(close) < slow:
            continue
        sma_fast = close.rolling(fast).mean()
        sma_slow = close.rolling(slow).mean()

        for eps in eps_events:
            eps_dt = pd.to_datetime(eps["report_date"])
            window_end = eps_dt + timedelta(days=TREND_WINDOW_DAYS)

            mask = (close.index >= eps_dt) & (close.index <= window_end)
            w = pd.DataFrame({"fast": sma_fast[mask], "slow": sma_slow[mask]}).dropna()
            if len(w) < 2:
                continue

            for i in range(1, len(w)):
                prev, curr = w.iloc[i - 1], w.iloc[i]
                cross_date = curr.name

                # Only report if crossover is recent
                if cross_date < cutoff:
                    continue

                prev_above = prev["fast"] > prev["slow"]
                curr_above = curr["fast"] > curr["slow"]

                if not prev_above and curr_above:
                    direction = "bullish"
                elif prev_above and not curr_above:
                    direction = "bearish"
                else:
                    continue

                signals.append({
                    "direction": direction,
                    "fast_ma": fast,
                    "slow_ma": slow,
                    "cross_date": cross_date.strftime("%Y-%m-%d"),
                    "eps_date": eps["report_date"],
                    "eps_change_pct": eps["eps_change_pct"],
                    "days_between": (cross_date - eps_dt).days,
                    "close": round(float(close.loc[cross_date]), 2),
                })

    return signals


def get_earnings_events(ticker: str, lookback_days: int = 120) -> list[dict]:
    """Fetch recent earnings with significant EPS changes."""
    try:
        t = yf.Ticker(ticker)
        hist = t.quarterly_earnings
        if hist is None or hist.empty:
            return []

        cutoff = datetime.now() - timedelta(days=lookback_days)
        events = []
        for date, row in hist.iterrows():
            dt = pd.to_datetime(date)
            if dt < cutoff:
                continue
            actual = row.get("Earnings") if "Earnings" in row else None
            estimate = row.get("Estimate") if "Estimate" in row else None
            if actual is None or estimate is None or estimate == 0:
                continue
            chg = (actual - estimate) / abs(estimate) * 100
            if abs(chg) >= EPS_CHANGE_THRESHOLD:
                events.append({
                    "report_date": dt.strftime("%Y-%m-%d"),
                    "eps_change_pct": round(chg, 2),
                })
        return events
    except Exception:
        return []


# ── Notification ──────────────────────────────────────────────────────────────

def send_email(subject: str, body_html: str) -> bool:
    smtp_user = os.environ.get("SMTP_USERNAME")
    smtp_pass = os.environ.get("SMTP_PASSWORD")
    smtp_to   = os.environ.get("SMTP_TO", smtp_user)

    if not smtp_user or not smtp_pass:
        print("Email not configured (SMTP_USERNAME / SMTP_PASSWORD not set)")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = smtp_user
    msg["To"]      = smtp_to
    msg.attach(MIMEText(body_html, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, smtp_to.split(","), msg.as_string())
        print(f"Email sent to {smtp_to}")
        return True
    except Exception as e:
        print(f"Email failed: {e}")
        return False


def send_slack(text: str) -> bool:
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook:
        return False
    try:
        import urllib.request, json as _json
        payload = _json.dumps({"text": text}).encode()
        req = urllib.request.Request(webhook, data=payload, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
        print("Slack notification sent")
        return True
    except Exception as e:
        print(f"Slack failed: {e}")
        return False


def build_email_html(signals_by_ticker: dict[str, list[dict]]) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    rows = ""
    for ticker, sigs in signals_by_ticker.items():
        for s in sigs:
            arrow = "🟢" if s["direction"] == "bullish" else "🔴"
            rows += f"""
            <tr>
                <td><b>{ticker}</b></td>
                <td>{arrow} {s['direction']}</td>
                <td>SMA{s['fast_ma']}/{s['slow_ma']}</td>
                <td>{s['eps_change_pct']:+.1f}%</td>
                <td>{s['eps_date']}</td>
                <td>{s['cross_date']}</td>
                <td>{s['days_between']}d</td>
                <td>${s['close']}</td>
            </tr>"""

    return f"""
    <html><body>
    <h2>Small Cap Scanner — New Signals ({today})</h2>
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;font-family:monospace">
        <tr style="background:#333;color:white">
            <th>Ticker</th><th>Direction</th><th>MA Cross</th>
            <th>EPS Chg</th><th>EPS Date</th><th>Cross Date</th>
            <th>Days</th><th>Price</th>
        </tr>
        {rows}
    </table>
    <p style="color:#888;font-size:12px">
        Run <code>python scan_and_notify.py</code> to check again,
        or open the Streamlit app for chart analysis.
    </p>
    </body></html>"""


def build_slack_text(signals_by_ticker: dict[str, list[dict]]) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    lines = [f"*Small Cap Scanner — New Signals ({today})*"]
    for ticker, sigs in signals_by_ticker.items():
        for s in sigs:
            arrow = "🟢" if s["direction"] == "bullish" else "🔴"
            lines.append(
                f"{arrow} *{ticker}* SMA{s['fast_ma']}/{s['slow_ma']} {s['direction']} | "
                f"EPS {s['eps_change_pct']:+.1f}% on {s['eps_date']} | "
                f"Cross {s['cross_date']} ({s['days_between']}d later) | ${s['close']}"
            )
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    today = datetime.now()
    tickers = list(SMALL_CAP_UNIVERSE)
    print(f"Scanning {len(tickers)} tickers for signals in last {LOOKBACK_DAYS} days...")

    # Batch download prices
    print("Downloading price data...")
    raw = yf.download(
        tickers, period=PRICE_HISTORY_MONTHS,
        auto_adjust=True, group_by="ticker",
        threads=True, progress=False,
    )

    signals_by_ticker: dict[str, list[dict]] = {}

    for i, ticker in enumerate(tickers):
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(tickers)}")

        # Extract close series
        try:
            if len(tickers) == 1:
                close = raw["Close"].dropna()
            else:
                close = raw["Close"][ticker].dropna()
            if len(close) < 50:
                continue
        except (KeyError, TypeError):
            continue

        earnings = get_earnings_events(ticker)
        if not earnings:
            continue

        new_sigs = find_new_crossovers(close, earnings, today)
        if new_sigs:
            signals_by_ticker[ticker] = new_sigs

    total = sum(len(v) for v in signals_by_ticker.values())
    print(f"\nFound {total} new signal(s) across {len(signals_by_ticker)} ticker(s)")

    if not signals_by_ticker:
        print("No new signals. No notifications sent.")
        return

    for ticker, sigs in signals_by_ticker.items():
        for s in sigs:
            arrow = "🟢" if s["direction"] == "bullish" else "🔴"
            print(
                f"  {arrow} {ticker} | SMA{s['fast_ma']}/{s['slow_ma']} {s['direction']} | "
                f"EPS {s['eps_change_pct']:+.1f}% | Cross {s['cross_date']}"
            )

    send_email(
        subject=f"Small Cap Scanner: {total} new signal(s) — {today.strftime('%Y-%m-%d')}",
        body_html=build_email_html(signals_by_ticker),
    )
    send_slack(build_slack_text(signals_by_ticker))


if __name__ == "__main__":
    run()
