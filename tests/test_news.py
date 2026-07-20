"""Unit tests for YFinanceProvider.get_news, all mocked (no network).

The key behavior under test is the failure contract: get_news must raise
NewsFetchError (never return a silent empty list) when the fetch errors or the
payload shape can't be parsed, while a genuinely empty feed returns [] quietly.
"""
import pytest

import core.providers.yfinance_provider as yp
from core.providers.base import NewsFetchError
from core.providers.yfinance_provider import YFinanceProvider


class _FakeTicker:
    def __init__(self, news):
        self._news = news

    @property
    def news(self):
        if isinstance(self._news, Exception):
            raise self._news
        return self._news


@pytest.fixture
def fake_news(monkeypatch):
    """Patch yf.Ticker(...).news to return whatever the test provides."""
    def _install(payload):
        monkeypatch.setattr(yp.yf, "Ticker", lambda _sym: _FakeTicker(payload))
    return _install


def _nested(title="Headline", url="https://finance.yahoo.com/a",
            publisher="Reuters", when="2026-07-20T02:03:00Z"):
    return {"id": "x", "content": {
        "title": title,
        "provider": {"displayName": publisher},
        "pubDate": when,
        "clickThroughUrl": {"url": url},
        "canonicalUrl": {"url": "https://origin.example/a"},
    }}


def test_parses_nested_schema_and_caps_to_limit(fake_news):
    fake_news([_nested(title=f"H{i}") for i in range(10)])
    items = YFinanceProvider().get_news("AAPL", limit=4)
    assert len(items) == 4
    first = items[0]
    assert first["title"] == "H0"
    assert first["url"] == "https://finance.yahoo.com/a"
    assert first["publisher"] == "Reuters"
    assert first["published"] == "2026-07-20T02:03:00Z"


def test_prefers_click_through_url(fake_news):
    fake_news([_nested()])
    assert YFinanceProvider().get_news("AAPL")[0]["url"] == "https://finance.yahoo.com/a"


def test_falls_back_to_canonical_url(fake_news):
    item = _nested()
    del item["content"]["clickThroughUrl"]
    fake_news([item])
    assert YFinanceProvider().get_news("AAPL")[0]["url"] == "https://origin.example/a"


def test_old_flat_schema(fake_news):
    fake_news([{"title": "Old", "link": "https://y/z", "publisher": "AP",
               "providerPublishTime": 1_700_000_000}])
    items = YFinanceProvider().get_news("AAPL")
    assert items[0]["title"] == "Old"
    assert items[0]["url"] == "https://y/z"
    assert items[0]["published"].startswith("20")  # epoch -> ISO


def test_empty_list_returns_empty(fake_news):
    fake_news([])
    assert YFinanceProvider().get_news("AAPL") == []


def test_none_returns_empty(fake_news):
    fake_news(None)
    assert YFinanceProvider().get_news("AAPL") == []


def test_non_list_raises(fake_news):
    fake_news({"unexpected": "dict"})
    with pytest.raises(NewsFetchError):
        YFinanceProvider().get_news("AAPL")


def test_unparseable_items_raise(fake_news):
    # non-empty payload, but nothing has a title+url -> schema change signal
    fake_news([{"content": {"description": "no title or url"}}, {"foo": "bar"}])
    with pytest.raises(NewsFetchError):
        YFinanceProvider().get_news("AAPL")


def test_upstream_error_raises(fake_news):
    fake_news(RuntimeError("yfinance boom"))
    with pytest.raises(NewsFetchError):
        YFinanceProvider().get_news("AAPL")
