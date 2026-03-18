import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


class Database:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS stocks (
                    ticker TEXT PRIMARY KEY,
                    name TEXT,
                    market_cap REAL,
                    sector TEXT,
                    shares_float REAL,
                    short_interest_pct REAL,
                    short_ratio REAL,
                    last_updated TEXT
                );

                CREATE TABLE IF NOT EXISTS daily_prices (
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    avg_volume_20d REAL,
                    relative_volume REAL,
                    PRIMARY KEY (ticker, date)
                );

                CREATE TABLE IF NOT EXISTS earnings (
                    ticker TEXT NOT NULL,
                    report_date TEXT NOT NULL,
                    period TEXT,
                    eps_actual REAL,
                    eps_prior REAL,
                    eps_change_pct REAL,
                    PRIMARY KEY (ticker, report_date)
                );

                CREATE TABLE IF NOT EXISTS fundamentals (
                    ticker TEXT NOT NULL,
                    period TEXT NOT NULL,
                    revenue REAL,
                    gross_margin REAL,
                    operating_margin REAL,
                    PRIMARY KEY (ticker, period)
                );

                CREATE TABLE IF NOT EXISTS scan_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    scan_date TEXT,
                    signal_type TEXT,
                    ma_period INTEGER,
                    eps_change_pct REAL,
                    trend_change_date TEXT,
                    eps_change_date TEXT,
                    days_between INTEGER
                );

                CREATE INDEX IF NOT EXISTS idx_daily_prices_ticker_date
                    ON daily_prices (ticker, date);

                CREATE INDEX IF NOT EXISTS idx_earnings_ticker
                    ON earnings (ticker);

                CREATE INDEX IF NOT EXISTS idx_scan_results_scan_date
                    ON scan_results (scan_date);
            """)

    def get_tables(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        return [row["name"] for row in rows]

    def upsert_stock(self, stock: dict) -> None:
        stock = dict(stock)
        stock["last_updated"] = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO stocks
                    (ticker, name, market_cap, sector, shares_float,
                     short_interest_pct, short_ratio, last_updated)
                VALUES
                    (:ticker, :name, :market_cap, :sector, :shares_float,
                     :short_interest_pct, :short_ratio, :last_updated)
                """,
                stock,
            )

    def get_stock(self, ticker: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM stocks WHERE ticker = ?", (ticker,)
            ).fetchone()
        return dict(row) if row else None

    def insert_daily_prices(self, rows: list[dict]) -> None:
        with self._connect() as conn:
            for row in rows:
                row = dict(row)
                row.setdefault("avg_volume_20d", None)
                row.setdefault("relative_volume", None)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO daily_prices
                        (ticker, date, open, high, low, close, volume,
                         avg_volume_20d, relative_volume)
                    VALUES
                        (:ticker, :date, :open, :high, :low, :close, :volume,
                         :avg_volume_20d, :relative_volume)
                    """,
                    row,
                )

    def get_daily_prices(
        self, ticker: str, start_date: str, end_date: str
    ) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM daily_prices
                WHERE ticker = ? AND date >= ? AND date <= ?
                ORDER BY date
                """,
                (ticker, start_date, end_date),
            ).fetchall()
        return [dict(row) for row in rows]

    def insert_earnings(self, rows: list[dict]) -> None:
        with self._connect() as conn:
            for row in rows:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO earnings
                        (ticker, report_date, period, eps_actual, eps_prior, eps_change_pct)
                    VALUES
                        (:ticker, :report_date, :period, :eps_actual, :eps_prior, :eps_change_pct)
                    """,
                    row,
                )

    def get_earnings(self, ticker: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM earnings WHERE ticker = ? ORDER BY report_date",
                (ticker,),
            ).fetchall()
        return [dict(row) for row in rows]

    def insert_fundamentals(self, rows: list[dict]) -> None:
        with self._connect() as conn:
            for row in rows:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO fundamentals
                        (ticker, period, revenue, gross_margin, operating_margin)
                    VALUES
                        (:ticker, :period, :revenue, :gross_margin, :operating_margin)
                    """,
                    row,
                )

    def get_fundamentals(self, ticker: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM fundamentals WHERE ticker = ? ORDER BY period",
                (ticker,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_stock_universe(
        self,
        min_price: float,
        max_price: float,
        min_market_cap: float,
        max_market_cap: float,
    ) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.*, dp.close AS last_close
                FROM stocks s
                JOIN daily_prices dp ON dp.ticker = s.ticker
                    AND dp.date = (
                        SELECT MAX(date) FROM daily_prices WHERE ticker = s.ticker
                    )
                WHERE dp.close >= ?
                  AND dp.close <= ?
                  AND s.market_cap >= ?
                  AND s.market_cap <= ?
                ORDER BY s.ticker
                """,
                (min_price, max_price, min_market_cap, max_market_cap),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_scan_results(self, results: list[dict]) -> None:
        with self._connect() as conn:
            for row in results:
                conn.execute(
                    """
                    INSERT INTO scan_results
                        (ticker, scan_date, signal_type, ma_period, eps_change_pct,
                         trend_change_date, eps_change_date, days_between)
                    VALUES
                        (:ticker, :scan_date, :signal_type, :ma_period, :eps_change_pct,
                         :trend_change_date, :eps_change_date, :days_between)
                    """,
                    row,
                )

    def get_scan_results(self, scan_date: str | None = None) -> list[dict]:
        with self._connect() as conn:
            if scan_date is not None:
                rows = conn.execute(
                    "SELECT * FROM scan_results WHERE scan_date = ? ORDER BY id",
                    (scan_date,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM scan_results ORDER BY id"
                ).fetchall()
        return [dict(row) for row in rows]
