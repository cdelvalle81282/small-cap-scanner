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

                CREATE TABLE IF NOT EXISTS signal_watchlist (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    eps_change_pct REAL,
                    eps_date TEXT NOT NULL,
                    signal_date TEXT NOT NULL,
                    fast_ma INTEGER,
                    slow_ma INTEGER,
                    levels_json TEXT NOT NULL,
                    trend_break_price REAL,
                    trend_break_condition TEXT,
                    ai_analysis TEXT,
                    expiry_date TEXT NOT NULL,
                    added_date TEXT NOT NULL,
                    active INTEGER DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS price_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    watchlist_id INTEGER NOT NULL,
                    ticker TEXT NOT NULL,
                    alert_date TEXT NOT NULL,
                    level_type TEXT,
                    level_price REAL,
                    level_label TEXT,
                    triggered_close REAL,
                    read_flag INTEGER DEFAULT 0,
                    FOREIGN KEY (watchlist_id) REFERENCES signal_watchlist(id)
                );

                CREATE INDEX IF NOT EXISTS idx_watchlist_active
                    ON signal_watchlist (active, expiry_date);

                CREATE TABLE IF NOT EXISTS signal_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    alert_date TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    fast_ma INTEGER,
                    slow_ma INTEGER,
                    eps_change_pct REAL,
                    eps_date TEXT,
                    cross_date TEXT,
                    days_between INTEGER,
                    close_price REAL,
                    UNIQUE(ticker, alert_date, signal_type, fast_ma, slow_ma)
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    status TEXT DEFAULT 'open',
                    entry_date TEXT,
                    entry_price REAL,
                    shares REAL,
                    stop_price REAL,
                    target_price REAL,
                    exit_date TEXT,
                    exit_price REAL,
                    notes TEXT,
                    signal_type TEXT,
                    eps_change_pct REAL,
                    eps_date TEXT,
                    added_date TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_signal_alerts_date
                    ON signal_alerts (alert_date DESC);

                CREATE INDEX IF NOT EXISTS idx_trades_status
                    ON trades (status);
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

    def get_signal_enrichment(self, tickers: list[str]) -> dict[str, dict]:
        """Returns {ticker: {market_cap, latest_close, avg_dollar_vol}} for a list of tickers."""
        if not tickers:
            return {}
        placeholders = ",".join("?" * len(tickers))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                WITH recent AS (
                    SELECT ticker, close, volume,
                           ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
                    FROM daily_prices
                    WHERE ticker IN ({placeholders})
                ),
                avg_dvol AS (
                    SELECT ticker, AVG(close * volume) AS avg_dollar_vol
                    FROM recent WHERE rn <= 20
                    GROUP BY ticker
                ),
                latest AS (
                    SELECT ticker, close AS latest_close
                    FROM recent WHERE rn = 1
                )
                SELECT s.ticker, s.market_cap, l.latest_close, a.avg_dollar_vol
                FROM stocks s
                LEFT JOIN latest l ON l.ticker = s.ticker
                LEFT JOIN avg_dvol a ON a.ticker = s.ticker
                WHERE s.ticker IN ({placeholders})
                """,
                tickers + tickers,
            ).fetchall()
        return {row["ticker"]: dict(row) for row in rows}

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

    # --- Watchlist methods ---

    def add_to_watchlist(self, entry: dict) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO signal_watchlist
                    (ticker, signal_type, eps_change_pct, eps_date, signal_date,
                     fast_ma, slow_ma, levels_json, trend_break_price,
                     trend_break_condition, ai_analysis, expiry_date, added_date, active)
                VALUES
                    (:ticker, :signal_type, :eps_change_pct, :eps_date, :signal_date,
                     :fast_ma, :slow_ma, :levels_json, :trend_break_price,
                     :trend_break_condition, :ai_analysis, :expiry_date, :added_date, 1)
                """,
                entry,
            )
            return cur.lastrowid

    def get_watchlist(self, active_only: bool = True) -> list[dict]:
        with self._connect() as conn:
            if active_only:
                rows = conn.execute(
                    "SELECT * FROM signal_watchlist WHERE active = 1 ORDER BY added_date DESC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM signal_watchlist ORDER BY added_date DESC"
                ).fetchall()
        return [dict(r) for r in rows]

    def get_watchlist_entry(self, ticker: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM signal_watchlist WHERE ticker = ? AND active = 1 ORDER BY id DESC LIMIT 1",
                (ticker,),
            ).fetchone()
        return dict(row) if row else None

    def remove_from_watchlist(self, watchlist_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE signal_watchlist SET active = 0 WHERE id = ?",
                (watchlist_id,),
            )

    def expire_watchlist(self, today: str) -> int:
        """Mark entries past their expiry date as inactive. Returns count expired."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE signal_watchlist SET active = 0 WHERE active = 1 AND expiry_date < ?",
                (today,),
            )
            return cur.rowcount

    # --- Price alert methods ---

    def add_price_alert(self, alert: dict) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO price_alerts
                    (watchlist_id, ticker, alert_date, level_type, level_price,
                     level_label, triggered_close, read_flag)
                VALUES
                    (:watchlist_id, :ticker, :alert_date, :level_type, :level_price,
                     :level_label, :triggered_close, 0)
                """,
                alert,
            )
            return cur.lastrowid

    def get_unread_alerts(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM price_alerts WHERE read_flag = 0 ORDER BY alert_date DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_alerts(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM price_alerts ORDER BY alert_date DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_alert_read(self, alert_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE price_alerts SET read_flag = 1 WHERE id = ?",
                (alert_id,),
            )

    def mark_all_alerts_read(self) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE price_alerts SET read_flag = 1")

    def alert_exists_for_level(self, watchlist_id: int, level_price: float) -> bool:
        """Prevent duplicate alerts for the same level breach."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM price_alerts WHERE watchlist_id = ? AND level_price = ?",
                (watchlist_id, level_price),
            ).fetchone()
        return row is not None

    # --- signal_alerts ---

    def save_signal_alert(self, alert: dict) -> None:
        """Insert a signal alert, ignore if duplicate."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO signal_alerts
                    (ticker, alert_date, signal_type, fast_ma, slow_ma,
                     eps_change_pct, eps_date, cross_date, days_between, close_price)
                VALUES
                    (:ticker, :alert_date, :signal_type, :fast_ma, :slow_ma,
                     :eps_change_pct, :eps_date, :cross_date, :days_between, :close_price)
                """,
                alert,
            )

    def get_signal_alerts(self, start_date: str, end_date: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM signal_alerts
                WHERE alert_date >= ? AND alert_date <= ?
                ORDER BY alert_date DESC, ticker
                """,
                (start_date, end_date),
            ).fetchall()
        return [dict(r) for r in rows]

    # --- trades ---

    def add_trade(self, trade: dict) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO trades
                    (ticker, direction, status, entry_date, entry_price, shares,
                     stop_price, target_price, exit_date, exit_price, notes,
                     signal_type, eps_change_pct, eps_date, added_date)
                VALUES
                    (:ticker, :direction, :status, :entry_date, :entry_price, :shares,
                     :stop_price, :target_price, :exit_date, :exit_price, :notes,
                     :signal_type, :eps_change_pct, :eps_date, :added_date)
                """,
                trade,
            )
            return cur.lastrowid

    def get_trades(self, status: str | None = None) -> list[dict]:
        with self._connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM trades WHERE status = ? ORDER BY entry_date DESC",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM trades ORDER BY added_date DESC"
                ).fetchall()
        return [dict(r) for r in rows]

    def close_trade(self, trade_id: int, exit_date: str, exit_price: float) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE trades
                SET status = 'closed', exit_date = ?, exit_price = ?
                WHERE id = ?
                """,
                (exit_date, exit_price, trade_id),
            )

    def delete_trade(self, trade_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM trades WHERE id = ?", (trade_id,))
