import pytest

from core.providers.base import DataProvider


def test_data_provider_is_abstract():
    with pytest.raises(TypeError):
        DataProvider()  # type: ignore[abstract]


def test_data_provider_defines_required_methods():
    required = {
        "get_price_history",
        "get_earnings",
        "get_small_cap_universe",
        "get_fundamentals",
        "get_stock_info",
    }
    assert required.issubset(set(DataProvider.__abstractmethods__))
