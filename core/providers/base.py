from abc import ABC, abstractmethod

import pandas as pd


class DataProvider(ABC):
    @abstractmethod
    def get_price_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Returns DataFrame with columns: date, open, high, low, close, volume"""
        ...

    @abstractmethod
    def get_earnings(self, ticker: str) -> pd.DataFrame:
        """Returns DataFrame with columns: report_date, period, eps_actual, eps_prior, eps_change_pct"""
        ...

    @abstractmethod
    def get_small_cap_universe(self, min_price: float, max_price: float) -> list[str]:
        """Returns list of ticker symbols matching the price filter"""
        ...

    @abstractmethod
    def get_fundamentals(self, ticker: str) -> pd.DataFrame:
        """Returns DataFrame with columns: period, revenue, gross_margin, operating_margin"""
        ...

    @abstractmethod
    def get_stock_info(self, ticker: str) -> dict:
        """Returns dict with keys: name, market_cap, sector, shares_float, short_interest_pct, short_ratio"""
        ...
