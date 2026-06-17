"""Backfill historical earnings using yfinance get_earnings_dates(limit=40).

The normal pipeline uses earnings_history which only returns ~4 quarters.
get_earnings_dates returns ~20 quarters (back to 2021), unlocking 3+ more
years of backtesting. No API key needed.

Usage:
    python backfill_fmp_earnings.py              # backfill all tickers
    python backfill_fmp_earnings.py --ticker SOFI # single ticker test
    python backfill_fmp_earnings.py --dry-run    # show remaining work
    python backfill_fmp_earnings.py --reset      # clear progress, restart
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import DB_PATH
from core.database import Database
from core.providers.yfinance_provider import SMALL_CAP_UNIVERSE

PROGRESS_FILE = Path(__file__).parent / "data" / "backfill_progress.json"


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed": [], "failed": []}


def save_progress(progress: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def fetch_earnings(ticker: str) -> list[dict]:
    """Fetch up to 20 quarters of earnings via yfinance."""
    t = yf.Ticker(ticker)
    try:
        df = t.get_earnings_dates(limit=40)
    except Exception:
        return []

    if df is None or df.empty:
        return []

    rows = []
    for idx, row in df.iterrows():
        eps_actual = row.get("Reported EPS")
        eps_estimate = row.get("EPS Estimate")

        if pd.isna(eps_actual):
            continue

        report_date = idx
        if hasattr(report_date, "strftime"):
            report_date = report_date.strftime("%Y-%m-%d")
        else:
            report_date = str(report_date)[:10]

        eps_change_pct = None
        surprise = row.get("Surprise(%)")
        if pd.notna(surprise):
            eps_change_pct = round(float(surprise), 2)
        elif pd.notna(eps_estimate) and eps_estimate != 0:
            eps_change_pct = round(
                (float(eps_actual) - float(eps_estimate)) / abs(float(eps_estimate)) * 100, 2
            )

        rows.append({
            "ticker": ticker,
            "report_date": report_date,
            "period": report_date,
            "eps_actual": float(eps_actual),
            "eps_prior": float(eps_estimate) if pd.notna(eps_estimate) else None,
            "eps_change_pct": eps_change_pct,
        })

    return rows


def backfill(
    dry_run: bool = False,
    single_ticker: str | None = None,
    reset: bool = False,
) -> None:
    db = Database(DB_PATH)
    progress = load_progress() if not reset else {"completed": [], "failed": []}

    if single_ticker:
        tickers = [single_ticker.upper()]
    else:
        done = set(progress["completed"])
        tickers = [t for t in SMALL_CAP_UNIVERSE if t not in done]

    if dry_run:
        print(f"Remaining tickers: {len(tickers)}")
        print(f"Already completed: {len(progress['completed'])}")
        print(f"Previously failed: {len(progress['failed'])}")
        return

    total = len(tickers)
    print(f"Backfilling earnings — {total} tickers to process")
    inserted_total = 0

    for i, ticker in enumerate(tickers):
        print(f"[{i + 1}/{total}] {ticker}...", end=" ", flush=True)

        try:
            rows = fetch_earnings(ticker)
        except Exception as e:
            print(f"error: {e}")
            progress["failed"].append(ticker)
            save_progress(progress)
            time.sleep(1)
            continue

        if rows:
            db.insert_earnings(rows)
            inserted_total += len(rows)
            print(f"{len(rows)} earnings")
        else:
            print("no data")

        if not single_ticker:
            progress["completed"].append(ticker)
            save_progress(progress)

        # Rate limit: yfinance can get throttled by Yahoo
        time.sleep(0.5)

    print(f"\nDone. {inserted_total} total earnings inserted across {total} tickers.")
    if progress["failed"]:
        print(f"Failed: {len(progress['failed'])} — {progress['failed'][:10]}")


def main():
    parser = argparse.ArgumentParser(description="Backfill earnings via yfinance")
    parser.add_argument("--reset", action="store_true", help="Clear progress, start over")
    parser.add_argument("--dry-run", action="store_true", help="Show remaining without fetching")
    parser.add_argument("--ticker", type=str, help="Backfill single ticker (for testing)")
    args = parser.parse_args()

    if args.reset:
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
        print("Progress cleared.")

    backfill(
        dry_run=args.dry_run,
        single_ticker=args.ticker,
        reset=args.reset,
    )


if __name__ == "__main__":
    main()
