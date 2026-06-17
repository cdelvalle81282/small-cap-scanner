"""
Daily price monitor for the Small Cap Scanner watchlist.

Run manually or via Windows Task Scheduler:
    python monitor.py

Fetches the latest close for each active watchlist entry and triggers alerts
when price crosses a watched support/resistance level.
"""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))

from config import DB_PATH
from core.database import Database
from core.notifier import send_alert

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_latest_close(ticker: str) -> float | None:
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception:
        logger.exception("Failed to fetch price for %s", ticker)
        return None


def check_level_breach(close: float, level: dict, signal_type: str) -> bool:
    """Return True if close has crossed a level in a way that matters for the trade."""
    price = level["price"]
    ltype = level["type"]

    if signal_type == "bullish":
        # For longs: breaching support (closing below) is the danger
        if ltype == "support" and close < price:
            return True
        # Clearing resistance is a positive confirmation
        if ltype == "resistance" and close > price:
            return True
    else:  # bearish
        # For shorts: recovering above resistance invalidates
        if ltype == "resistance" and close > price:
            return True
        # Breaking below support confirms
        if ltype == "support" and close < price:
            return True
    return False


def run_monitor() -> None:
    db = Database(DB_PATH)
    db.initialize()

    today = datetime.now().strftime("%Y-%m-%d")

    # Expire stale entries first
    expired = db.expire_watchlist(today)
    if expired:
        logger.info("Expired %d watchlist entries", expired)

    entries = db.get_watchlist(active_only=True)
    logger.info("Checking %d active watchlist entries", len(entries))

    triggered = 0
    for entry in entries:
        ticker = entry["ticker"]
        close = get_latest_close(ticker)
        if close is None:
            logger.warning("No price for %s, skipping", ticker)
            continue

        try:
            levels = json.loads(entry["levels_json"])
        except (json.JSONDecodeError, TypeError):
            levels = []

        for level in levels:
            if not isinstance(level.get("price"), (int, float)):
                continue
            if db.alert_exists_for_level(entry["id"], level["price"]):
                continue  # already alerted for this level
            if check_level_breach(close, level, entry["signal_type"]):
                alert = {
                    "watchlist_id": entry["id"],
                    "ticker": ticker,
                    "alert_date": today,
                    "level_type": level["type"],
                    "level_price": level["price"],
                    "level_label": level.get("label", ""),
                    "triggered_close": close,
                }
                db.add_price_alert(alert)
                send_alert(alert)
                logger.info(
                    "ALERT: %s closed at $%.2f, crossed %s $%.2f (%s)",
                    ticker, close, level["type"], level["price"], level.get("label", ""),
                )
                triggered += 1

    logger.info("Monitor complete. %d alerts triggered.", triggered)


if __name__ == "__main__":
    run_monitor()
