import pytest

from core.providers.yfinance_provider import YFinanceProvider


@pytest.fixture(scope="module")
def provider():
    return YFinanceProvider()


@pytest.mark.integration
def test_get_price_history(provider):
    df = provider.get_price_history("AAPL", "2024-01-01", "2024-01-31")
    assert not df.empty
    for col in ("date", "open", "high", "low", "close", "volume"):
        assert col in df.columns, f"Missing column: {col}"
    assert len(df) > 0


@pytest.mark.integration
def test_get_stock_info(provider):
    info = provider.get_stock_info("AAPL")
    assert isinstance(info, dict)
    assert info.get("name") is not None
    assert info.get("market_cap") is not None
    assert info["market_cap"] > 0


@pytest.mark.integration
def test_get_earnings(provider):
    df = provider.get_earnings("AAPL")
    assert not df.empty
    assert "eps_actual" in df.columns


@pytest.mark.integration
def test_get_small_cap_universe(provider):
    tickers = provider.get_small_cap_universe(1.0, 20.0)
    assert isinstance(tickers, list)
    assert len(tickers) > 0
