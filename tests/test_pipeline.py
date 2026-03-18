from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from core.database import Database
from pipeline import Pipeline


@pytest.fixture()
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    d.initialize()
    return d


def _make_provider(tickers=None):
    if tickers is None:
        tickers = ["ACMR", "KNDI"]

    provider = MagicMock()
    provider.get_small_cap_universe.return_value = tickers

    provider.get_stock_info.return_value = {
        "name": "Test Corp",
        "market_cap": 500_000_000,
        "sector": "Technology",
        "shares_float": 10_000_000,
        "short_interest_pct": 3.5,
        "short_ratio": 2.1,
    }

    provider.get_price_history.return_value = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "open": 5.0,
                "high": 5.5,
                "low": 4.9,
                "close": 5.2,
                "volume": 100_000,
            },
            {
                "date": "2024-01-03",
                "open": 5.2,
                "high": 5.8,
                "low": 5.1,
                "close": 5.6,
                "volume": 120_000,
            },
        ]
    )

    provider.get_earnings.return_value = pd.DataFrame(
        [
            {
                "report_date": "2024-01-15",
                "period": "Q4 2023",
                "eps_actual": 0.25,
                "eps_prior": 0.20,
                "eps_change_pct": 25.0,
            }
        ]
    )

    provider.get_fundamentals.return_value = pd.DataFrame(
        [
            {
                "period": "2023-12-31",
                "revenue": 50_000_000.0,
                "gross_margin": 45.0,
                "operating_margin": 12.0,
            }
        ]
    )

    return provider


def test_pipeline_fetches_and_stores(db):
    tickers = ["ACMR", "KNDI"]
    provider = _make_provider(tickers)

    pipeline = Pipeline(db, provider, batch_delay=0)
    pipeline.run("2024-01-01", "2024-01-31")

    # Both tickers should be in stocks table
    for ticker in tickers:
        stock = db.get_stock(ticker)
        assert stock is not None, f"Expected {ticker} in stocks table"
        assert stock["name"] == "Test Corp"
        assert stock["market_cap"] == 500_000_000

    # Price rows should be present
    for ticker in tickers:
        prices = db.get_daily_prices(ticker, "2024-01-01", "2024-01-31")
        assert len(prices) == 2
        assert prices[0]["close"] == pytest.approx(5.2)
        assert prices[1]["close"] == pytest.approx(5.6)

    # Earnings rows
    for ticker in tickers:
        earnings = db.get_earnings(ticker)
        assert len(earnings) == 1
        assert earnings[0]["eps_actual"] == pytest.approx(0.25)
        assert earnings[0]["eps_change_pct"] == pytest.approx(25.0)

    # Fundamentals rows
    for ticker in tickers:
        fundamentals = db.get_fundamentals(ticker)
        assert len(fundamentals) == 1
        assert fundamentals[0]["revenue"] == pytest.approx(50_000_000.0)


def test_pipeline_handles_provider_errors(db):
    tickers = ["ACMR", "KNDI", "MVST"]
    provider = _make_provider(tickers)

    # Make get_stock_info raise for the second ticker only
    def stock_info_side_effect(ticker):
        if ticker == "KNDI":
            raise RuntimeError("API timeout")
        return {
            "name": "Test Corp",
            "market_cap": 500_000_000,
            "sector": "Technology",
            "shares_float": 10_000_000,
            "short_interest_pct": 3.5,
            "short_ratio": 2.1,
        }

    provider.get_stock_info.side_effect = stock_info_side_effect

    pipeline = Pipeline(db, provider, batch_delay=0)
    # Should complete without raising
    pipeline.run("2024-01-01", "2024-01-31")

    # ACMR and MVST should be stored; KNDI should be skipped
    assert db.get_stock("ACMR") is not None
    assert db.get_stock("MVST") is not None
    assert db.get_stock("KNDI") is None
