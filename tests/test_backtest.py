from datetime import date, timedelta

import pytest

from config import BacktestConfig, ScannerConfig
from core.backtest import Backtester
from core.database import Database


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(ticker: str, base_date: date, closes: list[float]) -> list[dict]:
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASE = date(2023, 1, 1)


@pytest.fixture
def db_with_history(tmp_path):
    """
    BULL stock:
    - 500 days of prices starting 2023-01-01
    - Days 0-199: close = 4.0  (flat low)
    - Days 200-499: close starts at 6.0 and increases by 0.01/day
    - EPS report on day 200 with 100% change
    The MA-20 crossover (strict close > sma) will occur a few days after day 200
    when enough 6.0+ days pull the 20-day average above 4.0.
    With 300 days remaining after crossover there's ample room for all forward horizons.
    """
    db = Database(tmp_path / "backtest_test.db")
    db.initialize()

    db.upsert_stock({
        "ticker": "BULL",
        "name": "Bull Corp",
        "market_cap": 500_000_000,
        "sector": "Tech",
        "shares_float": None,
        "short_interest_pct": None,
        "short_ratio": None,
    })

    closes = [4.0] * 200 + [6.0 + i * 0.01 for i in range(300)]
    db.insert_daily_prices(_make_prices("BULL", BASE, closes))

    eps_date = (BASE + timedelta(days=200)).isoformat()
    db.insert_earnings([{
        "ticker": "BULL",
        "report_date": eps_date,
        "period": "Q1 2023",
        "eps_actual": 2.00,
        "eps_prior": 1.00,
        "eps_change_pct": 100.0,
    }])

    return db


def _scanner_config() -> ScannerConfig:
    return ScannerConfig(
        min_price=1.0,
        max_price=20.0,
        min_market_cap=50_000_000,
        max_market_cap=2_000_000_000,
        ma_periods=[20],
        eps_change_threshold=10.0,
        trend_window_days=30,
        direction="both",
    )


def _backtest_config(base: date = BASE) -> BacktestConfig:
    return BacktestConfig(
        start_date=base.isoformat(),
        end_date=(base + timedelta(days=499)).isoformat(),
        forward_return_days=[5, 10, 20],
        ma_periods=[20],
        eps_thresholds=[10.0],
        trend_windows=[30],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_backtester_produces_results(db_with_history):
    bt = Backtester(db_with_history, _scanner_config(), _backtest_config())
    result = bt.run()

    assert "signals" in result
    assert "summary" in result
    assert isinstance(result["signals"], list)
    assert isinstance(result["summary"], dict)
    assert len(result["signals"]) >= 1, "Expected at least one signal for BULL"
    assert result["summary"]["total_signals"] >= 1


def test_backtester_computes_forward_returns(db_with_history):
    bt = Backtester(db_with_history, _scanner_config(), _backtest_config())
    result = bt.run()

    assert len(result["signals"]) >= 1
    signal = result["signals"][0]

    assert "forward_returns" in signal
    fwd = signal["forward_returns"]

    # All configured horizons should be present
    for horizon in [5, 10, 20]:
        assert horizon in fwd, f"Missing horizon {horizon} in forward_returns"
        # The price trend is upward so returns should be positive
        ret = fwd[horizon]
        assert ret is not None, f"Return for horizon {horizon} should not be None"
        assert isinstance(ret, float)


def test_parameter_sweep(db_with_history):
    sc = ScannerConfig(
        min_price=1.0,
        max_price=20.0,
        min_market_cap=50_000_000,
        max_market_cap=2_000_000_000,
        ma_periods=[20, 50],
        eps_change_threshold=10.0,
        trend_window_days=30,
        direction="both",
    )
    bc = BacktestConfig(
        start_date=BASE.isoformat(),
        end_date=(BASE + timedelta(days=499)).isoformat(),
        forward_return_days=[5, 10],
        ma_periods=[20, 50],
        eps_thresholds=[10.0, 50.0],
        trend_windows=[30],
    )
    bt = Backtester(db_with_history, sc, bc)
    sweep = bt.parameter_sweep()

    assert isinstance(sweep, list)
    assert len(sweep) > 0

    expected_keys = {"ma_period", "eps_threshold", "trend_window",
                     "total_signals", "win_rate", "avg_return", "sample_size"}
    for row in sweep:
        assert expected_keys.issubset(row.keys()), f"Missing keys in sweep row: {row}"

    # With ma_periods=[20,50] x eps_thresholds=[10,50] x trend_windows=[30] = 4 combos
    assert len(sweep) == 4
