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


def _make_custom_prices(ticker: str, base_date: date, candles: list[dict]) -> list[dict]:
    rows = []
    for i, candle in enumerate(candles):
        close = candle["close"]
        rows.append({
            "ticker": ticker,
            "date": (base_date + timedelta(days=i)).isoformat(),
            "open": candle.get("open", close),
            "high": candle.get("high", close),
            "low": candle.get("low", close),
            "close": close,
            "volume": candle.get("volume", 500_000),
        })
    return rows


def _make_trade_backtester(
    tmp_path,
    ticker: str,
    candles: list[dict],
    horizons: list[int],
) -> Backtester:
    db = Database(tmp_path / f"{ticker.lower()}_trade.db")
    db.initialize()
    db.insert_daily_prices(_make_custom_prices(ticker, BASE, candles))
    return Backtester(
        db,
        _scanner_config(),
        BacktestConfig(
            start_date=BASE.isoformat(),
            end_date=(BASE + timedelta(days=len(candles) - 1)).isoformat(),
            forward_return_days=horizons,
            ma_crossover_pairs=[(20, 50)],
            eps_thresholds=[10.0],
            trend_windows=[30],
        ),
    )


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
    The SMA20 will cross above SMA50 shortly after day 200 when the fast
    average rises from 4.0 faster than the slow average.
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

    # EPS report 5 days before the price jump — crossover happens after
    eps_date = (BASE + timedelta(days=195)).isoformat()
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
        ma_crossover_pairs=[(20, 50)],
        eps_change_threshold=10.0,
        trend_window_days=30,
        direction="both",
    )


def _backtest_config(base: date = BASE) -> BacktestConfig:
    return BacktestConfig(
        start_date=base.isoformat(),
        end_date=(base + timedelta(days=499)).isoformat(),
        forward_return_days=[5, 10, 15],
        ma_crossover_pairs=[(20, 50)],
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

    assert "entry_date" in signal
    assert "entry_price" in signal
    assert "trade_details" in signal
    assert "forward_returns" in signal
    assert signal["entry_date"] > signal["trend_change_date"]
    assert isinstance(signal["entry_price"], float)

    fwd = signal["forward_returns"]
    trade_details = signal["trade_details"]

    # All configured horizons should be present
    for horizon in [5, 10, 15]:
        assert horizon in fwd, f"Missing horizon {horizon} in forward_returns"
        assert horizon in trade_details, f"Missing horizon {horizon} in trade_details"

        # The price trend is upward so returns should be positive
        ret = fwd[horizon]
        assert ret is not None, f"Return for horizon {horizon} should not be None"
        assert isinstance(ret, float)
        assert trade_details[horizon]["return_pct"] == ret
        assert trade_details[horizon]["exit_reason"] == "horizon"
        assert trade_details[horizon]["exit_price"] is not None
        assert trade_details[horizon]["exit_date"] is not None


def test_parameter_sweep(db_with_history):
    sc = ScannerConfig(
        min_price=1.0,
        max_price=20.0,
        min_market_cap=50_000_000,
        max_market_cap=2_000_000_000,
        ma_crossover_pairs=[(20, 50), (50, 200)],
        eps_change_threshold=10.0,
        trend_window_days=30,
        direction="both",
    )
    bc = BacktestConfig(
        start_date=BASE.isoformat(),
        end_date=(BASE + timedelta(days=499)).isoformat(),
        forward_return_days=[5, 10],
        ma_crossover_pairs=[(20, 50), (50, 200)],
        eps_thresholds=[10.0, 50.0],
        trend_windows=[30],
    )
    bt = Backtester(db_with_history, sc, bc)
    sweep = bt.parameter_sweep()

    assert isinstance(sweep, list)
    assert len(sweep) > 0

    expected_keys = {
        "ma_crossover",
        "eps_threshold",
        "trend_window",
        "total_signals",
        "win_rate",
        "avg_return",
        "profit_factor",
        "expectancy",
        "sample_size",
    }
    for row in sweep:
        assert expected_keys.issubset(row.keys()), f"Missing keys in sweep row: {row}"

    # With ma_crossover_pairs=[(20,50),(50,200)] x eps_thresholds=[10,50] x trend_windows=[30] = 4 combos
    assert len(sweep) == 4


def test_stop_loss_triggers_for_long(tmp_path):
    bt = _make_trade_backtester(
        tmp_path,
        "LONG",
        [
            {"high": 11.0, "low": 9.0, "close": 10.0},
            {"high": 12.0, "low": 10.0, "close": 11.0},
            {"high": 11.2, "low": 10.2, "close": 10.8},
            {"high": 10.0, "low": 9.0, "close": 9.5},
        ],
        [5, 10],
    )

    trade = bt._compute_forward_returns("LONG", BASE.isoformat(), "bullish")

    assert trade is not None
    assert trade["entry_date"] == (BASE + timedelta(days=1)).isoformat()
    assert trade["entry_price"] == pytest.approx(11.0)
    expected_return = round((9.5 - 11.0) / 11.0 * 100, 4)

    for horizon in [5, 10]:
        detail = trade["horizons"][horizon]
        assert detail["exit_reason"] == "stop_loss"
        assert detail["exit_date"] == (BASE + timedelta(days=3)).isoformat()
        assert detail["exit_price"] == pytest.approx(9.5)
        assert detail["return_pct"] == pytest.approx(expected_return)


def test_bearish_signal_short_returns(tmp_path):
    bt = _make_trade_backtester(
        tmp_path,
        "SHORT",
        [
            {"high": 10.5, "low": 9.5, "close": 10.0},
            {"high": 10.0, "low": 8.0, "close": 8.5},
            {"high": 9.0, "low": 7.8, "close": 8.0},
            {"high": 8.5, "low": 7.0, "close": 7.5},
        ],
        [3],
    )

    trade = bt._compute_forward_returns("SHORT", BASE.isoformat(), "bearish")

    assert trade is not None
    assert trade["entry_price"] == pytest.approx(9.0)
    detail = trade["horizons"][3]
    assert detail["exit_reason"] == "horizon"
    assert detail["exit_date"] == (BASE + timedelta(days=3)).isoformat()
    assert detail["return_pct"] == pytest.approx(round((9.0 - 7.5) / 9.0 * 100, 4))
    assert detail["return_pct"] > 0


def test_entry_price_is_next_day_midpoint(tmp_path):
    bt = _make_trade_backtester(
        tmp_path,
        "MID",
        [
            {"high": 10.0, "low": 9.0, "close": 9.5},
            {"high": 12.0, "low": 8.0, "close": 11.0},
            {"high": 11.5, "low": 10.5, "close": 11.0},
        ],
        [1],
    )

    trade = bt._compute_forward_returns("MID", BASE.isoformat(), "bullish")

    assert trade is not None
    assert trade["entry_date"] == (BASE + timedelta(days=1)).isoformat()
    assert trade["entry_price"] == pytest.approx(10.0)
