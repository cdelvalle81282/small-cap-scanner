import pytest

from core.database import Database


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    return database


def test_initialize_creates_tables(db):
    tables = db.get_tables()
    assert "stocks" in tables
    assert "daily_prices" in tables
    assert "earnings" in tables
    assert "fundamentals" in tables
    assert "scan_results" in tables


def test_upsert_stock(db):
    stock = {
        "ticker": "AAPL",
        "name": "Apple Inc.",
        "market_cap": 3_000_000_000,
        "sector": "Technology",
        "shares_float": 15_000_000_000,
        "short_interest_pct": 0.5,
        "short_ratio": 1.2,
    }
    db.upsert_stock(stock)
    result = db.get_stock("AAPL")
    assert result is not None
    assert result["ticker"] == "AAPL"
    assert result["name"] == "Apple Inc."
    assert result["market_cap"] == 3_000_000_000
    assert result["sector"] == "Technology"
    assert result["last_updated"] is not None


def test_upsert_stock_updates_existing(db):
    stock = {
        "ticker": "AAPL",
        "name": "Apple Inc.",
        "market_cap": 1_000_000_000,
        "sector": "Technology",
        "shares_float": None,
        "short_interest_pct": None,
        "short_ratio": None,
    }
    db.upsert_stock(stock)

    updated = dict(stock)
    updated["market_cap"] = 2_000_000_000
    updated["name"] = "Apple Inc. Updated"
    db.upsert_stock(updated)

    result = db.get_stock("AAPL")
    assert result["market_cap"] == 2_000_000_000
    assert result["name"] == "Apple Inc. Updated"


def test_insert_daily_prices_and_get(db):
    db.upsert_stock({
        "ticker": "MSFT",
        "name": "Microsoft",
        "market_cap": 500_000_000,
        "sector": "Technology",
        "shares_float": None,
        "short_interest_pct": None,
        "short_ratio": None,
    })
    rows = [
        {
            "ticker": "MSFT",
            "date": "2024-01-01",
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "close": 103.0,
            "volume": 1_000_000,
        },
        {
            "ticker": "MSFT",
            "date": "2024-01-02",
            "open": 103.0,
            "high": 108.0,
            "low": 102.0,
            "close": 107.0,
            "volume": 1_200_000,
            "avg_volume_20d": 1_100_000.0,
            "relative_volume": 1.09,
        },
    ]
    db.insert_daily_prices(rows)
    result = db.get_daily_prices("MSFT", "2024-01-01", "2024-01-02")
    assert len(result) == 2
    assert result[0]["close"] == 103.0
    assert result[1]["close"] == 107.0
    assert result[0]["avg_volume_20d"] is None
    assert result[1]["avg_volume_20d"] == 1_100_000.0


def test_insert_earnings_and_get(db):
    rows = [
        {
            "ticker": "AAPL",
            "report_date": "2024-01-25",
            "period": "Q1 2024",
            "eps_actual": 2.18,
            "eps_prior": 1.88,
            "eps_change_pct": 15.96,
        },
        {
            "ticker": "AAPL",
            "report_date": "2024-04-25",
            "period": "Q2 2024",
            "eps_actual": 1.53,
            "eps_prior": 1.52,
            "eps_change_pct": 0.66,
        },
    ]
    db.insert_earnings(rows)
    result = db.get_earnings("AAPL")
    assert len(result) == 2
    assert result[0]["report_date"] == "2024-01-25"
    assert result[0]["eps_change_pct"] == pytest.approx(15.96)
    assert result[1]["period"] == "Q2 2024"


def test_insert_fundamentals_and_get(db):
    rows = [
        {
            "ticker": "GOOG",
            "period": "2023-Q4",
            "revenue": 86_310_000_000,
            "gross_margin": 0.56,
            "operating_margin": 0.27,
        },
        {
            "ticker": "GOOG",
            "period": "2024-Q1",
            "revenue": 80_539_000_000,
            "gross_margin": 0.58,
            "operating_margin": 0.28,
        },
    ]
    db.insert_fundamentals(rows)
    result = db.get_fundamentals("GOOG")
    assert len(result) == 2
    assert result[0]["period"] == "2023-Q4"
    assert result[1]["gross_margin"] == pytest.approx(0.58)


def test_get_stock_universe(db):
    # Insert stocks with different market caps
    stocks = [
        {"ticker": "SMAL", "name": "Small Co", "market_cap": 100_000_000, "sector": "Tech",
         "shares_float": None, "short_interest_pct": None, "short_ratio": None},
        {"ticker": "MEDI", "name": "Medium Co", "market_cap": 500_000_000, "sector": "Health",
         "shares_float": None, "short_interest_pct": None, "short_ratio": None},
        {"ticker": "LARG", "name": "Large Co", "market_cap": 5_000_000_000, "sector": "Finance",
         "shares_float": None, "short_interest_pct": None, "short_ratio": None},
    ]
    for s in stocks:
        db.upsert_stock(s)

    # Insert latest prices — SMAL and MEDI in range, LARG price too high
    prices = [
        {"ticker": "SMAL", "date": "2024-01-02", "open": 5.0, "high": 6.0, "low": 4.5,
         "close": 5.5, "volume": 500_000},
        {"ticker": "MEDI", "date": "2024-01-02", "open": 10.0, "high": 11.0, "low": 9.5,
         "close": 10.5, "volume": 300_000},
        {"ticker": "LARG", "date": "2024-01-02", "open": 50.0, "high": 55.0, "low": 49.0,
         "close": 52.0, "volume": 2_000_000},
    ]
    db.insert_daily_prices(prices)

    # Filter: price 1-20, market cap 50M-2B
    result = db.get_stock_universe(
        min_price=1.0,
        max_price=20.0,
        min_market_cap=50_000_000,
        max_market_cap=2_000_000_000,
    )
    tickers = [r["ticker"] for r in result]
    assert "SMAL" in tickers
    assert "MEDI" in tickers
    assert "LARG" not in tickers  # market cap too large and price too high


def test_get_stock_universe_price_filter(db):
    db.upsert_stock({"ticker": "CHIP", "name": "Cheap Co", "market_cap": 200_000_000,
                     "sector": "Retail", "shares_float": None, "short_interest_pct": None,
                     "short_ratio": None})
    db.insert_daily_prices([
        {"ticker": "CHIP", "date": "2024-01-02", "open": 0.5, "high": 0.6, "low": 0.4,
         "close": 0.55, "volume": 100_000},
    ])
    result = db.get_stock_universe(1.0, 20.0, 50_000_000, 2_000_000_000)
    tickers = [r["ticker"] for r in result]
    assert "CHIP" not in tickers  # price below min
