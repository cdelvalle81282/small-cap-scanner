from datetime import date, timedelta
from itertools import product
from statistics import mean, median

from config import BacktestConfig, ScannerConfig
from core.database import Database
from core.scanner import Scanner


class Backtester:
    def __init__(
        self,
        db: Database,
        scanner_config: ScannerConfig,
        backtest_config: BacktestConfig,
    ) -> None:
        self.db = db
        self.scanner_config = scanner_config
        self.backtest_config = backtest_config

    def run(self) -> dict:
        scanner = Scanner(self.db, self.scanner_config)
        raw_signals = scanner.scan(self.backtest_config.end_date)

        # Filter to signals whose trend_change_date falls within the backtest range
        signals = []
        for sig in raw_signals:
            trend_date = sig.get("trend_change_date")
            if trend_date is None:
                continue
            if trend_date < self.backtest_config.start_date:
                continue
            if trend_date > self.backtest_config.end_date:
                continue

            fwd = self._compute_forward_returns(sig["ticker"], trend_date)
            signals.append({**sig, "forward_returns": fwd})

        summary = self._compute_summary(signals)
        return {"signals": signals, "summary": summary}

    def _compute_forward_returns(
        self, ticker: str, signal_date: str
    ) -> dict[int, float | None]:
        horizons = self.backtest_config.forward_return_days
        max_horizon = max(horizons)

        signal_dt = date.fromisoformat(signal_date)
        fetch_end = signal_dt + timedelta(days=max_horizon + 10)

        rows = self.db.get_daily_prices(
            ticker,
            signal_date,
            fetch_end.isoformat(),
        )
        if not rows:
            return {h: None for h in horizons}

        entry_price = rows[0]["close"]
        if not entry_price:
            return {h: None for h in horizons}

        # Build a list of (date_obj, close) for easy lookup
        dated = [(date.fromisoformat(r["date"]), r["close"]) for r in rows]

        result: dict[int, float | None] = {}
        for horizon in horizons:
            target_dt = signal_dt + timedelta(days=horizon)
            # Find the trading day with the date closest to target (on or after)
            candidates = [(d, c) for d, c in dated if d >= target_dt]
            if not candidates:
                result[horizon] = None
                continue
            # Pick the closest date
            exit_date, exit_price = min(candidates, key=lambda x: (x[0] - target_dt).days)
            if exit_price is None or entry_price == 0:
                result[horizon] = None
            else:
                result[horizon] = round((exit_price - entry_price) / entry_price * 100, 4)

        return result

    def _compute_summary(self, signals: list[dict]) -> dict:
        if not signals:
            return {"total_signals": 0}

        horizons = self.backtest_config.forward_return_days
        by_horizon: dict[int, dict] = {}

        for horizon in horizons:
            returns = [
                s["forward_returns"][horizon]
                for s in signals
                if s.get("forward_returns") and s["forward_returns"].get(horizon) is not None
            ]
            if not returns:
                by_horizon[horizon] = {
                    "win_rate": None,
                    "avg_return": None,
                    "median_return": None,
                    "max_gain": None,
                    "max_loss": None,
                    "sample_size": 0,
                }
                continue

            wins = sum(1 for r in returns if r > 0)
            by_horizon[horizon] = {
                "win_rate": round(wins / len(returns) * 100, 2),
                "avg_return": round(mean(returns), 4),
                "median_return": round(median(returns), 4),
                "max_gain": round(max(returns), 4),
                "max_loss": round(min(returns), 4),
                "sample_size": len(returns),
            }

        return {"total_signals": len(signals), "by_horizon": by_horizon}

    def parameter_sweep(self) -> list[dict]:
        results = []
        first_horizon = self.backtest_config.forward_return_days[0]

        for ma_period, eps_threshold, trend_window in product(
            self.backtest_config.ma_periods,
            self.backtest_config.eps_thresholds,
            self.backtest_config.trend_windows,
        ):
            sc = ScannerConfig(
                min_price=self.scanner_config.min_price,
                max_price=self.scanner_config.max_price,
                min_market_cap=self.scanner_config.min_market_cap,
                max_market_cap=self.scanner_config.max_market_cap,
                ma_periods=[ma_period],
                eps_change_threshold=eps_threshold,
                trend_window_days=trend_window,
                direction=self.scanner_config.direction,
            )
            bt = Backtester(self.db, sc, self.backtest_config)
            outcome = bt.run()

            summary = outcome["summary"]
            horizon_stats = (
                summary.get("by_horizon", {}).get(first_horizon, {})
                if summary.get("total_signals", 0) > 0
                else {}
            )

            results.append({
                "ma_period": ma_period,
                "eps_threshold": eps_threshold,
                "trend_window": trend_window,
                "total_signals": summary.get("total_signals", 0),
                "win_rate": horizon_stats.get("win_rate"),
                "avg_return": horizon_stats.get("avg_return"),
                "sample_size": horizon_stats.get("sample_size", 0),
            })

        return results
