from datetime import date, timedelta

import pytest

from config import ScannerConfig
from core.database import Database
from core.performance import (
    follow_through,
    relative_volume,
    summarize_horizons,
    ticker_follow_through,
)


def _make_prices(ticker, base_date, closes, volume=500_000):
    rows = []
    for i, close in enumerate(closes):
        d = base_date + timedelta(days=i)
        rows.append({
            "ticker": ticker, "date": d.isoformat(),
            "open": close, "high": close + 0.1, "low": close - 0.1,
            "close": close, "volume": volume,
        })
    return rows


def _bullish_closes():
    # 100 flat days, then a jump that keeps rising so forward returns are real
    return [4.0] * 100 + [round(6.0 + i * 0.04, 2) for i in range(60)]


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "perf_test.db")
    database.initialize()
    base = date(2024, 1, 1)
    database.upsert_stock({
        "ticker": "BULL", "name": "BULL", "market_cap": 200_000_000,
        "sector": "Tech", "shares_float": None,
        "short_interest_pct": None, "short_ratio": None,
    })
    database.insert_daily_prices(_make_prices("BULL", base, _bullish_closes()))
    database.insert_earnings([{
        "ticker": "BULL", "report_date": (base + timedelta(days=98)).isoformat(),
        "period": "Q1 2024", "eps_actual": 1.50, "eps_prior": 1.00, "eps_change_pct": 50.0,
    }])
    return database, base


def _config():
    return ScannerConfig(
        min_price=1.0, max_price=20.0, min_market_cap=50_000_000, max_market_cap=2_000_000_000,
        ma_crossover_pairs=[(20, 50)], eps_change_threshold=10.0, trend_window_days=30, direction="both",
    )


def test_follow_through_produces_forward_returns(db):
    database, base = db
    end = (base + timedelta(days=159)).isoformat()
    result = follow_through(database, _config(), horizons=(5, 10, 15), start_date="2024-01-01", end_date=end)

    assert result["signals"], "expected at least one signal with follow-through"
    sig = next(s for s in result["signals"] if s["ticker"] == "BULL")
    assert set(sig["forward_returns"]) == {5, 10, 15}
    # BULL keeps rising after the cross -> positive return at some horizon
    vals = [v for v in sig["forward_returns"].values() if v is not None]
    assert vals and max(vals) > 0


def test_ticker_follow_through_single(db):
    database, _ = db
    # cross happens in the 6.0 run; pick a date we know has forward data
    out = ticker_follow_through(database, "BULL", (date(2024, 1, 1) + timedelta(days=110)).isoformat(), "bullish", horizons=(5, 10))
    assert out is not None
    assert set(out["forward_returns"]) == {5, 10}
    assert out["forward_returns"][10] > 0  # rising series held 10d is positive


def test_relative_volume(tmp_path):
    database = Database(tmp_path / "rvol.db")
    database.initialize()
    base = date(2024, 1, 1)
    # 24 normal days then a big volume spike on the last day
    rows = _make_prices("VOL", base, [5.0] * 25)
    rows[-1]["volume"] = 2_000_000  # 4x the 500k baseline
    database.insert_daily_prices(rows)
    rvol = relative_volume(database, "VOL", as_of=rows[-1]["date"])
    assert rvol is not None and rvol > 3.5


def test_summarize_horizons():
    signals = [
        {"forward_returns": {15: 10.0, 30: -5.0}},
        {"forward_returns": {15: 20.0, 30: 15.0}},
        {"forward_returns": {15: -4.0, 30: None}},
    ]
    out = summarize_horizons(signals, horizons=(15, 30))
    assert out[15]["sample"] == 3
    assert out[15]["win_rate"] == pytest.approx(66.7, abs=0.1)
    assert out[30]["sample"] == 2  # None dropped
