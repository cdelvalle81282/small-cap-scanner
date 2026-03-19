from datetime import date, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

from config import BacktestConfig, ScannerConfig
from core.backtest import Backtester
from core.database import Database
from core.scanner import Scanner
from pipeline import Pipeline

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

START = date(2022, 1, 1)
END = date(2023, 6, 1)
N_DAYS = (END - START).days + 1  # 517 days


def _price_rows(closes: list[float]) -> list[dict]:
    rows = []
    for i, close in enumerate(closes):
        d = START + timedelta(days=i)
        rows.append({
            "date": d.isoformat(),
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": 500_000,
        })
    return rows


def _bull_closes() -> list[float]:
    # 200 days at 4.0, then slow uptrend from 6.0
    low = [4.0] * 200
    high = [6.0 + i * 0.01 for i in range(N_DAYS - 200)]
    return low + high


def _bear_closes() -> list[float]:
    # 200 days at 10.0, then slow downtrend to ~7.0
    high = [10.0] * 200
    low = [10.0 - i * 0.01 for i in range(N_DAYS - 200)]
    return high + low


def _flat_closes() -> list[float]:
    return [8.0] * N_DAYS


def _eps_date() -> str:
    # EPS report near the price transition — day 200
    return (START + timedelta(days=200)).isoformat()


def _make_provider() -> MagicMock:
    provider = MagicMock()
    provider.get_small_cap_universe.return_value = ["BULL", "BEAR", "FLAT"]

    stock_info = {
        "BULL": {"name": "Bull Corp", "market_cap": 500_000_000, "sector": "Tech",
                 "shares_float": 10_000_000, "short_interest_pct": 2.0, "short_ratio": 1.5},
        "BEAR": {"name": "Bear Corp", "market_cap": 300_000_000, "sector": "Finance",
                 "shares_float": 8_000_000, "short_interest_pct": 5.0, "short_ratio": 3.0},
        "FLAT": {"name": "Flat Corp", "market_cap": 200_000_000, "sector": "Health",
                 "shares_float": 5_000_000, "short_interest_pct": 1.0, "short_ratio": 0.5},
    }

    def get_stock_info(ticker):
        return stock_info[ticker]

    provider.get_stock_info.side_effect = get_stock_info

    price_data = {
        "BULL": pd.DataFrame(_price_rows(_bull_closes())),
        "BEAR": pd.DataFrame(_price_rows(_bear_closes())),
        "FLAT": pd.DataFrame(_price_rows(_flat_closes())),
    }

    def get_price_history(ticker, start, end):
        return price_data[ticker]

    provider.get_price_history.side_effect = get_price_history

    eps_date = _eps_date()
    earnings_data = {
        "BULL": pd.DataFrame([{
            "report_date": eps_date,
            "period": "Q2 2022",
            "eps_actual": 2.00,
            "eps_prior": 1.00,
            "eps_change_pct": 100.0,
        }]),
        "BEAR": pd.DataFrame([{
            "report_date": eps_date,
            "period": "Q2 2022",
            "eps_actual": 0.10,
            "eps_prior": 0.30,
            "eps_change_pct": -66.67,
        }]),
        "FLAT": pd.DataFrame([{
            "report_date": eps_date,
            "period": "Q2 2022",
            "eps_actual": 1.00,
            "eps_prior": 1.00,
            "eps_change_pct": 0.0,
        }]),
    }

    def get_earnings(ticker):
        return earnings_data[ticker]

    provider.get_earnings.side_effect = get_earnings

    def get_fundamentals(ticker):
        return pd.DataFrame([{
            "period": "2022-Q2",
            "revenue": 100_000_000.0,
            "gross_margin": 40.0,
            "operating_margin": 15.0,
        }])

    provider.get_fundamentals.side_effect = get_fundamentals

    return provider


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def full_system(tmp_path):
    db = Database(tmp_path / "integration.db")
    db.initialize()

    provider = _make_provider()
    Pipeline(db, provider, batch_delay=0).run(
        START.isoformat(), END.isoformat()
    )
    return db


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_database_populated_correctly(full_system):
    db = full_system

    bull = db.get_stock("BULL")
    assert bull is not None
    assert bull["market_cap"] == 500_000_000

    earnings = db.get_earnings("BULL")
    assert len(earnings) >= 1
    assert earnings[0]["eps_change_pct"] == pytest.approx(100.0)

    prices = db.get_daily_prices("BULL", START.isoformat(), END.isoformat())
    assert len(prices) > 0


def test_full_pipeline_to_scan(full_system):
    config = ScannerConfig(
        min_price=1.0,
        max_price=20.0,
        min_market_cap=50_000_000,
        max_market_cap=2_000_000_000,
        ma_crossover_pairs=[(20, 50)],
        eps_change_threshold=10.0,
        trend_window_days=30,
        direction="both",
    )
    scanner = Scanner(full_system, config)
    results = scanner.scan(END.isoformat())

    flat_signals = [r for r in results if r["ticker"] == "FLAT"]
    assert flat_signals == [], (
        "FLAT has 0% EPS change so must not appear in scan results"
    )


def test_full_pipeline_to_backtest(full_system):
    scanner_config = ScannerConfig(
        min_price=1.0,
        max_price=20.0,
        min_market_cap=50_000_000,
        max_market_cap=2_000_000_000,
        ma_crossover_pairs=[(20, 50)],
        eps_change_threshold=10.0,
        trend_window_days=30,
        direction="both",
    )
    backtest_config = BacktestConfig(
        start_date=START.isoformat(),
        end_date=END.isoformat(),
        forward_return_days=[10, 15],
        ma_crossover_pairs=[(20, 50)],
        eps_thresholds=[10.0],
        trend_windows=[30],
    )
    bt = Backtester(full_system, scanner_config, backtest_config)
    result = bt.run()

    assert "signals" in result
    assert "summary" in result
    summary = result["summary"]
    assert "total_signals" in summary

    if summary["total_signals"] > 0:
        assert "by_horizon" in summary
        for horizon in [10, 15]:
            assert horizon in summary["by_horizon"]
            stats = summary["by_horizon"][horizon]
            for key in ["profit_factor", "expectancy", "win_count", "loss_count"]:
                assert key in stats
            assert stats["win_count"] + stats["loss_count"] == stats["sample_size"]
