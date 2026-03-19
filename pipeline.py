import argparse
import logging
import time

from config import DB_PATH
from core.database import Database
from core.providers.base import DataProvider
from core.providers.yfinance_provider import YFinanceProvider

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, db: Database, provider: DataProvider, batch_delay: float = 1.0) -> None:
        self.db = db
        self.provider = provider
        self.batch_delay = batch_delay

    def run(self, start_date: str, end_date: str) -> None:
        tickers = self.provider.get_small_cap_universe(1.0, 50.0)
        logger.info("Universe contains %d tickers", len(tickers))

        for i, ticker in enumerate(tickers):
            try:
                logger.info("Processing %s (%d/%d)", ticker, i + 1, len(tickers))
                self._process_ticker(ticker, start_date, end_date)
            except Exception:
                logger.warning("Skipping %s due to unexpected error", ticker, exc_info=True)

            if (i + 1) % 5 == 0 and self.batch_delay > 0:
                time.sleep(self.batch_delay)

        logger.info("Pipeline complete")

    def _process_ticker(self, ticker: str, start_date: str, end_date: str) -> None:
        # Stock info
        info = self.provider.get_stock_info(ticker)
        info["ticker"] = ticker
        self.db.upsert_stock(info)

        # Price history
        prices_df = self.provider.get_price_history(ticker, start_date, end_date)
        if not prices_df.empty:
            prices_df["ticker"] = ticker
            self.db.insert_daily_prices(prices_df.to_dict("records"))

        # Earnings
        earnings_df = self.provider.get_earnings(ticker)
        if not earnings_df.empty:
            earnings_df["ticker"] = ticker
            self.db.insert_earnings(earnings_df.to_dict("records"))

        # Fundamentals
        fundamentals_df = self.provider.get_fundamentals(ticker)
        if not fundamentals_df.empty:
            fundamentals_df["ticker"] = ticker
            self.db.insert_fundamentals(fundamentals_df.to_dict("records"))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the small-cap data pipeline")
    parser.add_argument("--start", default="2021-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-03-17", help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    db = Database(DB_PATH)
    db.initialize()
    provider = YFinanceProvider()
    pipeline = Pipeline(db, provider)
    pipeline.run(args.start, args.end)


if __name__ == "__main__":
    main()
