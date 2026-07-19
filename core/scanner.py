import sqlite3
from datetime import date, timedelta

import numpy as np
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
        tickers = [row["ticker"] for row in universe]
        earnings_by_ticker = self.db.get_earnings_bulk(tickers)

        signals: list[dict] = []
        with self.db.read_connection() as conn:
            for ticker in tickers:
                signals.extend(
                    self._check_ticker(ticker, earnings_by_ticker.get(ticker, []), as_of_date, conn)
                )
        return signals

    def _check_ticker(
        self,
        ticker: str,
        earnings: list[dict],
        as_of_date: str,
        conn: sqlite3.Connection,
    ) -> list[dict]:
        qualifying = []
        for earning in earnings:
            eps_change = earning.get("eps_change_pct")
            if eps_change is None:
                continue
            if abs(eps_change) < self.config.eps_change_threshold:
                continue
            if earning["report_date"] > as_of_date:
                continue
            qualifying.append((earning["report_date"], eps_change))

        if not qualifying:
            return []

        rows = self.db.get_price_history(ticker, conn=conn)
        if not rows:
            return []

        # Fetch the ticker's price history ONCE and reuse it (via fast binary search on
        # `dates`) for every earning x MA-pair combo below, instead of re-querying per combo.
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        dates = df["date"].to_numpy()

        signals: list[dict] = []
        for eps_date, eps_change in qualifying:
            for fast_period, slow_period in self.config.ma_crossover_pairs:
                crossover = self._find_ma_crossover(
                    df, dates, eps_date, fast_period, slow_period, self.config.trend_window_days,
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
                    "fast_ma": fast_period,
                    "slow_ma": slow_period,
                    "eps_change_pct": eps_change,
                    "trend_change_date": crossover["date"],
                    "eps_change_date": eps_date,
                    "days_between": crossover["days_between"],
                })

        return signals

    def _find_ma_crossover(
        self,
        df: pd.DataFrame,
        dates: np.ndarray,
        eps_date: str,
        fast_period: int,
        slow_period: int,
        window_days: int,
    ) -> dict | None:
        eps_dt = date.fromisoformat(eps_date)

        # Match the original per-earning fetch bounds exactly: MAs are only warmed up
        # from rows inside [fetch_start, fetch_end], not the ticker's full history. A
        # ticker with a data gap in that span can have fewer than `slow_period` valid
        # rows here even though earlier history exists, which changes where the MA
        # becomes non-NaN — so this window must be reproduced rather than computed
        # over the full series.
        fetch_start = eps_dt - timedelta(days=slow_period + window_days + 30)
        fetch_end = eps_dt + timedelta(days=window_days)
        lo = np.searchsorted(dates, pd.Timestamp(fetch_start).to_datetime64(), side="left")
        hi = np.searchsorted(dates, pd.Timestamp(fetch_end).to_datetime64(), side="right")
        if hi - lo < slow_period:
            return None

        sub = df.iloc[lo:hi]
        sma_fast = sub["close"].rolling(window=fast_period, min_periods=fast_period).mean().to_numpy()
        sma_slow = sub["close"].rolling(window=slow_period, min_periods=slow_period).mean().to_numpy()

        # Only count crossovers that happen AFTER the EPS report
        window_start = pd.Timestamp(eps_dt).to_datetime64()
        window_end = pd.Timestamp(eps_dt + timedelta(days=window_days)).to_datetime64()
        sub_dates = dates[lo:hi]
        w_lo = np.searchsorted(sub_dates, window_start, side="left")
        w_hi = np.searchsorted(sub_dates, window_end, side="right")

        valid = ~(np.isnan(sma_fast[w_lo:w_hi]) | np.isnan(sma_slow[w_lo:w_hi]))
        idxs = np.arange(w_lo, w_hi)[valid]
        if len(idxs) < 2:
            return None

        # First consecutive pair where fast crosses slow (prev_above -> curr_above changes)
        above = sma_fast[idxs] > sma_slow[idxs]
        transitions = np.diff(above.astype(np.int8))
        crossings = np.flatnonzero(transitions)
        if crossings.size == 0:
            return None

        first = crossings[0]
        direction = "bullish" if transitions[first] > 0 else "bearish"
        cross_date = sub["date"].iloc[idxs[first + 1]].date()
        days_between = abs((cross_date - eps_dt).days)
        return {
            "date": cross_date.isoformat(),
            "direction": direction,
            "days_between": days_between,
        }
