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

            trade_data = self._compute_forward_returns(
                sig["ticker"],
                trend_date,
                sig["signal_type"],
            )
            if trade_data is None:
                continue

            trade_details = trade_data["horizons"]
            forward_returns = {
                horizon: detail["return_pct"]
                for horizon, detail in trade_details.items()
            }
            signals.append({
                **sig,
                "entry_date": trade_data["entry_date"],
                "entry_price": trade_data["entry_price"],
                "trade_details": trade_details,
                "forward_returns": forward_returns,
            })

        summary = self._compute_summary(signals)
        return {"signals": signals, "summary": summary}

    def _compute_forward_returns(
        self, ticker: str, signal_date: str, direction: str
    ) -> dict | None:
        if direction not in {"bullish", "bearish"}:
            return None

        horizons = sorted(self.backtest_config.forward_return_days)
        max_horizon = max(horizons)

        signal_dt = date.fromisoformat(signal_date)
        fetch_end = signal_dt + timedelta(days=int(max_horizon * 1.6) + 10)

        rows = self.db.get_daily_prices(
            ticker,
            signal_date,
            fetch_end.isoformat(),
        )
        if len(rows) < 2:
            return None

        signal_idx = next(
            (idx for idx, row in enumerate(rows) if row["date"] == signal_date),
            None,
        )
        if signal_idx is None or signal_idx + 1 >= len(rows):
            return None

        entry_row = rows[signal_idx + 1]
        entry_price = self._get_entry_price(entry_row, direction)
        if entry_price is None or entry_price <= 0:
            return None

        trade_details = {
            horizon: {
                "exit_date": None,
                "exit_price": None,
                "return_pct": None,
                "exit_reason": None,
            }
            for horizon in horizons
        }

        prev_row = rows[signal_idx]
        unresolved = set(horizons)

        for trading_day, row in enumerate(rows[signal_idx + 1 :], start=1):
            stop_detail = self._get_stop_detail(prev_row, row, entry_price, direction)
            if stop_detail is not None:
                for horizon in sorted(unresolved):
                    trade_details[horizon] = stop_detail.copy()
                unresolved.clear()
                break

            if trading_day in unresolved:
                trade_details[trading_day] = self._build_trade_detail(
                    row,
                    entry_price,
                    direction,
                    "horizon",
                )
                unresolved.remove(trading_day)

            if not unresolved:
                break

            prev_row = row

        return {
            "entry_date": entry_row["date"],
            "entry_price": round(entry_price, 4),
            "horizons": trade_details,
        }

    def _get_entry_price(self, row: dict, direction: str) -> float | None:
        high = row.get("high")
        low = row.get("low")
        if high is None or low is None:
            return None

        midpoint = (high + low) / 2
        slippage = self.backtest_config.slippage_pct / 100
        if direction == "bullish":
            return midpoint * (1 + slippage)
        return midpoint * (1 - slippage)

    def _get_stop_detail(
        self,
        prev_row: dict,
        row: dict,
        entry_price: float,
        direction: str,
    ) -> dict | None:
        close = row.get("close")
        prev_low = prev_row.get("low")
        prev_high = prev_row.get("high")

        if close is None:
            return None

        stop_triggered = (
            direction == "bullish"
            and prev_low is not None
            and close < prev_low
        ) or (
            direction == "bearish"
            and prev_high is not None
            and close > prev_high
        )
        if not stop_triggered:
            return None

        return self._build_trade_detail(row, entry_price, direction, "stop_loss")

    def _build_trade_detail(
        self,
        row: dict,
        entry_price: float,
        direction: str,
        exit_reason: str,
    ) -> dict:
        exit_price = row.get("close")
        if exit_price is None or entry_price <= 0:
            return {
                "exit_date": row.get("date"),
                "exit_price": None,
                "return_pct": None,
                "exit_reason": exit_reason,
            }

        if direction == "bullish":
            return_pct = (exit_price - entry_price) / entry_price * 100
        else:
            return_pct = (entry_price - exit_price) / entry_price * 100

        return {
            "exit_date": row.get("date"),
            "exit_price": round(exit_price, 4),
            "return_pct": round(return_pct, 4),
            "exit_reason": exit_reason,
        }

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
                    "win_count": 0,
                    "loss_count": 0,
                    "avg_winner": None,
                    "avg_loser": None,
                    "profit_factor": None,
                    "expectancy": None,
                    "sample_size": 0,
                }
                continue

            winners = [r for r in returns if r > 0]
            losers = [r for r in returns if r <= 0]
            win_count = len(winners)
            loss_count = len(losers)
            gross_gains = sum(winners)
            gross_losses = sum(r for r in losers if r < 0)

            if gross_losses == 0:
                profit_factor = float("inf") if gross_gains > 0 else None
            else:
                profit_factor = gross_gains / abs(gross_losses)

            avg_winner = mean(winners) if winners else None
            avg_loser = mean(losers) if losers else None
            win_rate = win_count / len(returns)
            loss_rate = loss_count / len(returns)
            expectancy = (win_rate * (avg_winner or 0.0)) + (loss_rate * (avg_loser or 0.0))

            by_horizon[horizon] = {
                "win_rate": round(win_rate * 100, 2),
                "avg_return": round(mean(returns), 4),
                "median_return": round(median(returns), 4),
                "max_gain": round(max(returns), 4),
                "max_loss": round(min(returns), 4),
                "win_count": win_count,
                "loss_count": loss_count,
                "avg_winner": round(avg_winner, 4) if avg_winner is not None else None,
                "avg_loser": round(avg_loser, 4) if avg_loser is not None else None,
                "profit_factor": (
                    round(profit_factor, 4)
                    if profit_factor not in {None, float("inf")}
                    else profit_factor
                ),
                "expectancy": round(expectancy, 4),
                "sample_size": len(returns),
            }

        return {"total_signals": len(signals), "by_horizon": by_horizon}

    def parameter_sweep(self) -> list[dict]:
        results = []
        first_horizon = self.backtest_config.forward_return_days[0]

        for ma_pair, eps_threshold, trend_window in product(
            self.backtest_config.ma_crossover_pairs,
            self.backtest_config.eps_thresholds,
            self.backtest_config.trend_windows,
        ):
            sc = ScannerConfig(
                min_price=self.scanner_config.min_price,
                max_price=self.scanner_config.max_price,
                min_market_cap=self.scanner_config.min_market_cap,
                max_market_cap=self.scanner_config.max_market_cap,
                ma_crossover_pairs=[ma_pair],
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
                "ma_crossover": f"{ma_pair[0]}/{ma_pair[1]}",
                "eps_threshold": eps_threshold,
                "trend_window": trend_window,
                "total_signals": summary.get("total_signals", 0),
                "win_rate": horizon_stats.get("win_rate"),
                "avg_return": horizon_stats.get("avg_return"),
                "profit_factor": horizon_stats.get("profit_factor"),
                "expectancy": horizon_stats.get("expectancy"),
                "sample_size": horizon_stats.get("sample_size", 0),
            })

        return results
