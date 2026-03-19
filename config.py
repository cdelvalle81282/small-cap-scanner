from dataclasses import dataclass, field
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "scanner.db"

@dataclass
class ScannerConfig:
    min_price: float = 1.0
    max_price: float = 20.0
    min_market_cap: float = 50_000_000       # $50M
    max_market_cap: float = 10_000_000_000   # $10B
    ma_crossover_pairs: list[tuple[int, int]] = field(
        default_factory=lambda: [(20, 50), (50, 200)]
    )  # (fast_period, slow_period)
    eps_change_threshold: float = 10.0       # percent
    trend_window_days: int = 30
    direction: str = "both"                  # "bullish", "bearish", "both"

@dataclass
class BacktestConfig:
    start_date: str = "2021-01-01"
    end_date: str = "2026-03-17"
    forward_return_days: list[int] = field(default_factory=lambda: [5, 10, 15, 30, 60])
    slippage_pct: float = 0.0
    ma_crossover_pairs: list[tuple[int, int]] = field(
        default_factory=lambda: [(20, 50), (50, 200)]
    )
    eps_thresholds: list[float] = field(default_factory=lambda: [5.0, 10.0, 25.0, 50.0])
    trend_windows: list[int] = field(default_factory=lambda: [15, 30, 45])
