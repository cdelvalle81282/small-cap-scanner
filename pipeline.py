import argparse
import logging
import os
import time

from config import DB_PATH
from core.database import Database
from core.providers.base import DataProvider
from core.providers.yfinance_provider import YFinanceProvider

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        db: Database,
        provider: DataProvider,
        batch_delay: float = 1.0,
        min_market_cap: float | None = None,
        max_market_cap: float | None = None,
    ) -> None:
        self.db = db
        self.provider = provider
        self.batch_delay = batch_delay
        self.min_market_cap = min_market_cap
        self.max_market_cap = max_market_cap

    def run(self, start_date: str, end_date: str, min_price: float = 1.0, max_price: float = 50.0) -> None:
        tickers = self.provider.get_small_cap_universe(min_price, max_price)
        logger.info("Universe contains %d tickers", len(tickers))

        skipped = 0
        for i, ticker in enumerate(tickers):
            try:
                logger.info("Processing %s (%d/%d)", ticker, i + 1, len(tickers))
                result = self._process_ticker(ticker, start_date, end_date)
                if result == "skipped":
                    skipped += 1
                    logger.info("Skipped %s — outside market cap range", ticker)
            except Exception:
                logger.warning("Skipping %s due to unexpected error", ticker, exc_info=True)
                skipped += 1

            if (i + 1) % 5 == 0 and self.batch_delay > 0:
                time.sleep(self.batch_delay)

        logger.info("Pipeline complete — processed %d, skipped %d", len(tickers) - skipped, skipped)

    def _process_ticker(self, ticker: str, start_date: str, end_date: str) -> str | None:
        # Stock info (always fetch — needed for market cap check)
        info = self.provider.get_stock_info(ticker)
        info["ticker"] = ticker

        # Market cap gate — skip price history and earnings if outside range
        market_cap = info.get("market_cap")
        if market_cap is not None:
            if self.min_market_cap is not None and market_cap < self.min_market_cap:
                return "skipped"
            if self.max_market_cap is not None and market_cap > self.max_market_cap:
                return "skipped"

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

        return None


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the small-cap data pipeline")
    parser.add_argument("--start", default="2022-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD, default today)")
    parser.add_argument("--price-min", type=float, default=1.0, help="Min price filter")
    parser.add_argument("--price-max", type=float, default=50.0, help="Max price filter")
    parser.add_argument("--min-cap", type=float, default=None, help="Market cap floor (e.g. 50000000)")
    parser.add_argument("--max-cap", type=float, default=None, help="Market cap ceiling (e.g. 10000000000)")
    parser.add_argument(
        "--provider", default="yfinance", choices=["yfinance", "polygon"],
        help="Data provider (default: yfinance)",
    )
    args = parser.parse_args()

    from datetime import date
    end_date = args.end or date.today().isoformat()

    db = Database(DB_PATH)
    db.initialize()

    if args.provider == "polygon":
        from core.providers.polygon_provider import PolygonProvider
        api_key = os.environ.get("POLYGON_API_KEY")
        if not api_key:
            raise SystemExit("POLYGON_API_KEY environment variable not set")
        provider: DataProvider = PolygonProvider(api_key)
    else:
        provider = YFinanceProvider()

    pipeline = Pipeline(
        db, provider,
        min_market_cap=args.min_cap,
        max_market_cap=args.max_cap,
    )
    pipeline.run(args.start, end_date, args.price_min, args.price_max)


if __name__ == "__main__":
    main()
