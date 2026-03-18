from datetime import date, timedelta

import pytest

from config import ScannerConfig
from core.database import Database
from core.scanner import Scanner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stock(ticker: str, market_cap: float = 200_000_000) -> dict:
    return {
        "ticker": ticker,
        "name": ticker,
        "market_cap": market_cap,
        "sector": "Tech",
        "shares_float": None,
        "short_interest_pct": None,
        "short_ratio": None,
    }


def _make_prices(ticker: str, base_date: date, closes: list[float]) -> list[dict]:
    """Build daily_prices rows from a list of close values starting at base_date."""
    rows = []
    for i, close in enumerate(closes):
        d = base_date + timedelta(days=i)
        rows.append({
            "ticker": ticker,
            "date": d.isoformat(),
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": 500_000,
        })
    return rows


def _bullish_closes(n_low: int = 40, n_high: int = 20, low: float = 4.0, high: float = 6.0) -> list[float]:
    """
    Returns closes that are below `low` for n_low days then jump to `high`
    for n_high days, producing a clear bullish MA-20 crossover.
    """
    return [low] * n_low + [high] * n_high


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "scanner_test.db")
    database.initialize()
    return database


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_scanner_finds_bullish_signal(db):
    """
    BULL has 60 days of prices: 40 days at 4.0, then 20 days at 6.0.
    The MA-20 will be dragged up over time; the cross from below to above
    occurs within the window around the EPS report date.
    EPS report on day 35 (5 days into the high-price run) with 50% change.
    Scanner (threshold=10%, direction="both") should detect a bullish signal.
    """
    base = date(2024, 1, 1)
    db.upsert_stock(_stock("BULL"))

    closes = _bullish_closes(n_low=40, n_high=20)
    db.insert_daily_prices(_make_prices("BULL", base, closes))

    # EPS report on day 35 — price is already in the "high" zone
    eps_date = (base + timedelta(days=35)).isoformat()
    db.insert_earnings([{
        "ticker": "BULL",
        "report_date": eps_date,
        "period": "Q1 2024",
        "eps_actual": 1.50,
        "eps_prior": 1.00,
        "eps_change_pct": 50.0,
    }])

    config = ScannerConfig(
        min_price=1.0,
        max_price=20.0,
        min_market_cap=50_000_000,
        max_market_cap=2_000_000_000,
        ma_periods=[20],
        eps_change_threshold=10.0,
        trend_window_days=30,
        direction="both",
    )
    scanner = Scanner(db, config)
    as_of_date = (base + timedelta(days=59)).isoformat()
    results = scanner.scan(as_of_date)

    bull_signals = [r for r in results if r["ticker"] == "BULL"]
    assert len(bull_signals) >= 1, "Expected at least one bullish signal for BULL"
    assert bull_signals[0]["signal_type"] == "bullish"
    assert bull_signals[0]["ma_period"] == 20
    assert bull_signals[0]["eps_change_pct"] == pytest.approx(50.0)


def test_scanner_respects_eps_threshold(db):
    """
    Same price/earnings data as above but threshold=60% > 50% change.
    Scanner should find no signals for BULL.
    """
    base = date(2024, 1, 1)
    db.upsert_stock(_stock("BULL"))

    closes = _bullish_closes(n_low=40, n_high=20)
    db.insert_daily_prices(_make_prices("BULL", base, closes))

    eps_date = (base + timedelta(days=35)).isoformat()
    db.insert_earnings([{
        "ticker": "BULL",
        "report_date": eps_date,
        "period": "Q1 2024",
        "eps_actual": 1.50,
        "eps_prior": 1.00,
        "eps_change_pct": 50.0,
    }])

    config = ScannerConfig(
        min_price=1.0,
        max_price=20.0,
        min_market_cap=50_000_000,
        max_market_cap=2_000_000_000,
        ma_periods=[20],
        eps_change_threshold=60.0,   # higher than 50%
        trend_window_days=30,
        direction="both",
    )
    scanner = Scanner(db, config)
    as_of_date = (base + timedelta(days=59)).isoformat()
    results = scanner.scan(as_of_date)

    assert results == [], f"Expected no signals, got: {results}"


def test_scanner_respects_direction_filter(db):
    """
    BULL has a bullish crossover. Setting direction='bearish' should exclude it.
    """
    base = date(2024, 1, 1)
    db.upsert_stock(_stock("BULL"))

    closes = _bullish_closes(n_low=40, n_high=20)
    db.insert_daily_prices(_make_prices("BULL", base, closes))

    eps_date = (base + timedelta(days=35)).isoformat()
    db.insert_earnings([{
        "ticker": "BULL",
        "report_date": eps_date,
        "period": "Q1 2024",
        "eps_actual": 1.50,
        "eps_prior": 1.00,
        "eps_change_pct": 50.0,
    }])

    config = ScannerConfig(
        min_price=1.0,
        max_price=20.0,
        min_market_cap=50_000_000,
        max_market_cap=2_000_000_000,
        ma_periods=[20],
        eps_change_threshold=10.0,
        trend_window_days=30,
        direction="bearish",          # only want bearish
    )
    scanner = Scanner(db, config)
    as_of_date = (base + timedelta(days=59)).isoformat()
    results = scanner.scan(as_of_date)

    bull_signals = [r for r in results if r["ticker"] == "BULL"]
    assert bull_signals == [], "Bullish signal should be excluded by direction='bearish'"


def test_flat_stock_produces_no_signal(db):
    """
    FLAT has an EPS change of only 2%, below any reasonable threshold.
    Scanner should return nothing for it.
    """
    base = date(2024, 1, 1)
    db.upsert_stock(_stock("FLAT"))

    # Flat prices — no crossover possible
    closes = [5.0] * 60
    db.insert_daily_prices(_make_prices("FLAT", base, closes))

    eps_date = (base + timedelta(days=30)).isoformat()
    db.insert_earnings([{
        "ticker": "FLAT",
        "report_date": eps_date,
        "period": "Q1 2024",
        "eps_actual": 1.02,
        "eps_prior": 1.00,
        "eps_change_pct": 2.0,
    }])

    config = ScannerConfig(
        min_price=1.0,
        max_price=20.0,
        min_market_cap=50_000_000,
        max_market_cap=2_000_000_000,
        ma_periods=[20],
        eps_change_threshold=10.0,
        trend_window_days=30,
        direction="both",
    )
    scanner = Scanner(db, config)
    as_of_date = (base + timedelta(days=59)).isoformat()
    results = scanner.scan(as_of_date)

    assert results == [], f"Expected no signals for FLAT stock, got: {results}"
