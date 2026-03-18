from datetime import date, timedelta

import pandas as pd

from config import ScannerConfig
from core.database import Database


class Scanner:
    def __init__(self, db: Database, config: ScannerConfig) -> None:
        self.db = db
        self.config = config

    def scan(self, as_of_date: str) -> list[dict]:
        universe = self.db.get_stock_universe(
            min_price=self.config.min_price,
            max_price=self.config.max_price,
            min_market_cap=self.config.min_market_cap,
            max_market_cap=self.config.max_market_cap,
        )
        signals: list[dict] = []
        for row in universe:
            signals.extend(self._check_ticker(row["ticker"], as_of_date))
        return signals

    def _check_ticker(self, ticker: str, as_of_date: str) -> list[dict]:
        earnings = self.db.get_earnings(ticker)
        signals: list[dict] = []

        for earning in earnings:
            eps_change = earning.get("eps_change_pct")
            if eps_change is None:
                continue
            if abs(eps_change) < self.config.eps_change_threshold:
                continue

            eps_date = earning["report_date"]
            # Only look at earnings on or before as_of_date
            if eps_date > as_of_date:
                continue

            for ma_period in self.config.ma_periods:
                crossover = self._find_ma_crossover(
                    ticker,
                    eps_date,
                    ma_period,
                    self.config.trend_window_days,
                )
                if crossover is None:
                    continue

                direction = crossover["direction"]
                if self.config.direction != "both" and direction != self.config.direction:
                    continue

                signals.append({
                    "ticker": ticker,
                    "scan_date": as_of_date,
                    "signal_type": direction,
                    "ma_period": ma_period,
                    "eps_change_pct": eps_change,
                    "trend_change_date": crossover["date"],
                    "eps_change_date": eps_date,
                    "days_between": crossover["days_between"],
                })

        return signals

    def _find_ma_crossover(
        self,
        ticker: str,
        eps_date: str,
        ma_period: int,
        window_days: int,
    ) -> dict | None:
        eps_dt = date.fromisoformat(eps_date)

        # Fetch enough history to compute the MA before the window starts
        fetch_start = eps_dt - timedelta(days=ma_period + window_days + 30)
        fetch_end = eps_dt + timedelta(days=window_days)

        rows = self.db.get_daily_prices(
            ticker,
            fetch_start.isoformat(),
            fetch_end.isoformat(),
        )
        if len(rows) < ma_period:
            return None

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df["sma"] = df["close"].rolling(window=ma_period, min_periods=ma_period).mean()

        # Restrict crossover search to the window around the EPS date
        window_start = eps_dt - timedelta(days=window_days)
        window_end = eps_dt + timedelta(days=window_days)
        mask = (df["date"].dt.date >= window_start) & (df["date"].dt.date <= window_end)
        window_df = df[mask].dropna(subset=["sma"]).reset_index(drop=True)

        if len(window_df) < 2:
            return None

        # Walk through consecutive pairs to find the first crossover
        for i in range(1, len(window_df)):
            prev = window_df.iloc[i - 1]
            curr = window_df.iloc[i]

            prev_above = prev["close"] > prev["sma"]
            curr_above = curr["close"] > curr["sma"]

            if not prev_above and curr_above:
                direction = "bullish"
            elif prev_above and not curr_above:
                direction = "bearish"
            else:
                continue

            cross_date = curr["date"].date()
            days_between = abs((cross_date - eps_dt).days)
            return {
                "date": cross_date.isoformat(),
                "direction": direction,
                "days_between": days_between,
            }

        return None
