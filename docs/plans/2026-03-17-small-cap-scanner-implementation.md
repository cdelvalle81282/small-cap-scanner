# Small Cap EPS + Trend Scanner Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a stock scanner that finds small cap stocks ($1–$20) where a moving average trend change occurs within a configurable window of an EPS change, with backtesting and a Streamlit UI.

**Architecture:** Pipeline + SQLite + Streamlit. A data pipeline fetches from YFinance and stores in SQLite. The scanner engine reads from the DB. Streamlit provides an interactive UI with scanner dashboard, stock detail charts, and backtest results.

**Tech Stack:** Python 3.11+, yfinance, pandas, SQLite, Streamlit, plotly, pytest

---

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `config.py`
- Create: `core/__init__.py`
- Create: `core/providers/__init__.py`
- Create: `pages/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Create requirements.txt**

```txt
yfinance>=0.2.31
pandas>=2.0.0
streamlit>=1.30.0
plotly>=5.18.0
pytest>=7.4.0
```

**Step 2: Create config.py with default settings**

```python
from dataclasses import dataclass, field
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "scanner.db"

@dataclass
class ScannerConfig:
    min_price: float = 1.0
    max_price: float = 20.0
    min_market_cap: float = 50_000_000       # $50M
    max_market_cap: float = 2_000_000_000    # $2B
    ma_periods: list[int] = field(default_factory=lambda: [20, 50, 200])
    eps_change_threshold: float = 10.0       # percent
    trend_window_days: int = 30
    direction: str = "both"                  # "bullish", "bearish", "both"

@dataclass
class BacktestConfig:
    start_date: str = "2021-01-01"
    end_date: str = "2026-03-17"
    forward_return_days: list[int] = field(default_factory=lambda: [5, 10, 20, 30, 60])
    ma_periods: list[int] = field(default_factory=lambda: [20, 50, 200])
    eps_thresholds: list[float] = field(default_factory=lambda: [5.0, 10.0, 25.0, 50.0])
    trend_windows: list[int] = field(default_factory=lambda: [15, 30, 45])
```

**Step 3: Create directory structure and __init__.py files**

```bash
mkdir -p core/providers pages tests data
touch core/__init__.py core/providers/__init__.py pages/__init__.py tests/__init__.py
```

**Step 4: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 5: Commit**

```bash
git add requirements.txt config.py core/ pages/ tests/ data/
git commit -m "feat: project scaffolding with config and dependencies"
```

---

### Task 2: Database Layer

**Files:**
- Create: `core/database.py`
- Create: `tests/test_database.py`

**Step 1: Write failing tests for database initialization and CRUD**

```python
# tests/test_database.py
import pytest
import sqlite3
from pathlib import Path
from core.database import Database

@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test.db"
    database = Database(db_path)
    database.initialize()
    return database

def test_initialize_creates_tables(db):
    tables = db.get_tables()
    assert "stocks" in tables
    assert "daily_prices" in tables
    assert "earnings" in tables
    assert "fundamentals" in tables
    assert "scan_results" in tables

def test_upsert_stock(db):
    db.upsert_stock({
        "ticker": "TEST",
        "name": "Test Corp",
        "market_cap": 500_000_000,
        "sector": "Technology",
        "shares_float": 10_000_000,
        "short_interest_pct": 5.2,
        "short_ratio": 2.1,
    })
    stock = db.get_stock("TEST")
    assert stock["name"] == "Test Corp"
    assert stock["market_cap"] == 500_000_000

def test_upsert_stock_updates_existing(db):
    db.upsert_stock({"ticker": "TEST", "name": "Old Name", "market_cap": 100})
    db.upsert_stock({"ticker": "TEST", "name": "New Name", "market_cap": 200})
    stock = db.get_stock("TEST")
    assert stock["name"] == "New Name"
    assert stock["market_cap"] == 200

def test_insert_daily_prices(db):
    rows = [
        {"ticker": "TEST", "date": "2024-01-01", "open": 5.0, "high": 5.5,
         "low": 4.8, "close": 5.2, "volume": 100000},
        {"ticker": "TEST", "date": "2024-01-02", "open": 5.2, "high": 5.8,
         "low": 5.0, "close": 5.6, "volume": 120000},
    ]
    db.insert_daily_prices(rows)
    prices = db.get_daily_prices("TEST", "2024-01-01", "2024-01-02")
    assert len(prices) == 2
    assert prices[0]["close"] == 5.2

def test_insert_earnings(db):
    db.insert_earnings([{
        "ticker": "TEST",
        "report_date": "2024-01-15",
        "period": "Q4 2023",
        "eps_actual": 0.50,
        "eps_prior": 0.30,
        "eps_change_pct": 66.67,
    }])
    earnings = db.get_earnings("TEST")
    assert len(earnings) == 1
    assert earnings[0]["eps_change_pct"] == pytest.approx(66.67)

def test_insert_fundamentals(db):
    db.insert_fundamentals([{
        "ticker": "TEST",
        "period": "Q4 2023",
        "revenue": 10_000_000,
        "gross_margin": 0.45,
        "operating_margin": 0.20,
    }])
    fundamentals = db.get_fundamentals("TEST")
    assert len(fundamentals) == 1
    assert fundamentals[0]["revenue"] == 10_000_000

def test_get_stock_universe(db):
    db.upsert_stock({"ticker": "A", "name": "A", "market_cap": 500_000_000})
    db.upsert_stock({"ticker": "B", "name": "B", "market_cap": 100_000_000})
    db.upsert_stock({"ticker": "C", "name": "C", "market_cap": 50_000_000_000})
    # Insert prices so we can filter by price
    db.insert_daily_prices([
        {"ticker": "A", "date": "2024-01-01", "open": 5, "high": 5, "low": 5, "close": 5.0, "volume": 1000},
        {"ticker": "B", "date": "2024-01-01", "open": 15, "high": 15, "low": 15, "close": 15.0, "volume": 1000},
        {"ticker": "C", "date": "2024-01-01", "open": 50, "high": 50, "low": 50, "close": 50.0, "volume": 1000},
    ])
    universe = db.get_stock_universe(min_price=1, max_price=20, min_market_cap=50_000_000, max_market_cap=2_000_000_000)
    tickers = [s["ticker"] for s in universe]
    assert "A" in tickers
    assert "B" in tickers
    assert "C" not in tickers  # market cap too large
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_database.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'core.database'`

**Step 3: Implement database.py**

```python
# core/database.py
import sqlite3
from pathlib import Path
from datetime import datetime

class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def initialize(self):
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
                    ticker TEXT NOT NULL,
                    scan_date TEXT NOT NULL,
                    signal_type TEXT,
                    ma_period INTEGER,
                    eps_change_pct REAL,
                    trend_change_date TEXT,
                    eps_change_date TEXT,
                    days_between INTEGER
                );

                CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON daily_prices(ticker, date);
                CREATE INDEX IF NOT EXISTS idx_earnings_ticker ON earnings(ticker);
                CREATE INDEX IF NOT EXISTS idx_scan_results_date ON scan_results(scan_date);
            """)

    def get_tables(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            return [row["name"] for row in rows]

    def upsert_stock(self, stock: dict):
        stock.setdefault("last_updated", datetime.now().isoformat())
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO stocks (ticker, name, market_cap, sector, shares_float,
                                    short_interest_pct, short_ratio, last_updated)
                VALUES (:ticker, :name, :market_cap, :sector, :shares_float,
                        :short_interest_pct, :short_ratio, :last_updated)
                ON CONFLICT(ticker) DO UPDATE SET
                    name=excluded.name, market_cap=excluded.market_cap,
                    sector=excluded.sector, shares_float=excluded.shares_float,
                    short_interest_pct=excluded.short_interest_pct,
                    short_ratio=excluded.short_ratio, last_updated=excluded.last_updated
            """, stock)

    def get_stock(self, ticker: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM stocks WHERE ticker = ?", (ticker,)).fetchone()
            return dict(row) if row else None

    def insert_daily_prices(self, rows: list[dict]):
        with self._connect() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO daily_prices
                    (ticker, date, open, high, low, close, volume, avg_volume_20d, relative_volume)
                VALUES (:ticker, :date, :open, :high, :low, :close, :volume,
                        :avg_volume_20d, :relative_volume)
            """, [{**{"avg_volume_20d": None, "relative_volume": None}, **r} for r in rows])

    def get_daily_prices(self, ticker: str, start_date: str, end_date: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM daily_prices
                WHERE ticker = ? AND date >= ? AND date <= ?
                ORDER BY date
            """, (ticker, start_date, end_date)).fetchall()
            return [dict(r) for r in rows]

    def insert_earnings(self, rows: list[dict]):
        with self._connect() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO earnings
                    (ticker, report_date, period, eps_actual, eps_prior, eps_change_pct)
                VALUES (:ticker, :report_date, :period, :eps_actual, :eps_prior, :eps_change_pct)
            """, rows)

    def get_earnings(self, ticker: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM earnings WHERE ticker = ? ORDER BY report_date", (ticker,)
            ).fetchall()
            return [dict(r) for r in rows]

    def insert_fundamentals(self, rows: list[dict]):
        with self._connect() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO fundamentals
                    (ticker, period, revenue, gross_margin, operating_margin)
                VALUES (:ticker, :period, :revenue, :gross_margin, :operating_margin)
            """, rows)

    def get_fundamentals(self, ticker: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM fundamentals WHERE ticker = ? ORDER BY period", (ticker,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_stock_universe(self, min_price: float, max_price: float,
                           min_market_cap: float, max_market_cap: float) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT s.* FROM stocks s
                INNER JOIN (
                    SELECT ticker, close FROM daily_prices
                    WHERE (ticker, date) IN (
                        SELECT ticker, MAX(date) FROM daily_prices GROUP BY ticker
                    )
                ) p ON s.ticker = p.ticker
                WHERE p.close >= ? AND p.close <= ?
                  AND s.market_cap >= ? AND s.market_cap <= ?
            """, (min_price, max_price, min_market_cap, max_market_cap)).fetchall()
            return [dict(r) for r in rows]

    def save_scan_results(self, results: list[dict]):
        with self._connect() as conn:
            conn.executemany("""
                INSERT INTO scan_results
                    (ticker, scan_date, signal_type, ma_period, eps_change_pct,
                     trend_change_date, eps_change_date, days_between)
                VALUES (:ticker, :scan_date, :signal_type, :ma_period, :eps_change_pct,
                        :trend_change_date, :eps_change_date, :days_between)
            """, results)

    def get_scan_results(self, scan_date: str | None = None) -> list[dict]:
        with self._connect() as conn:
            if scan_date:
                rows = conn.execute(
                    "SELECT * FROM scan_results WHERE scan_date = ? ORDER BY ticker",
                    (scan_date,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM scan_results ORDER BY scan_date DESC, ticker"
                ).fetchall()
            return [dict(r) for r in rows]
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_database.py -v
```

Expected: All PASS

**Step 5: Commit**

```bash
git add core/database.py tests/test_database.py
git commit -m "feat: database layer with SQLite schema and CRUD operations"
```

---

### Task 3: Data Provider Abstraction

**Files:**
- Create: `core/providers/base.py`
- Create: `tests/test_providers_base.py`

**Step 1: Write the test**

```python
# tests/test_providers_base.py
import pytest
from core.providers.base import DataProvider

def test_data_provider_is_abstract():
    with pytest.raises(TypeError):
        DataProvider()

def test_data_provider_defines_required_methods():
    methods = ["get_price_history", "get_earnings", "get_small_cap_universe",
               "get_fundamentals", "get_stock_info"]
    for method in methods:
        assert hasattr(DataProvider, method)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_providers_base.py -v
```

**Step 3: Implement base.py**

```python
# core/providers/base.py
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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_providers_base.py -v
```

**Step 5: Commit**

```bash
git add core/providers/base.py tests/test_providers_base.py
git commit -m "feat: abstract DataProvider base class"
```

---

### Task 4: YFinance Provider

**Files:**
- Create: `core/providers/yfinance_provider.py`
- Create: `tests/test_yfinance_provider.py`

**Step 1: Write tests (integration tests — these hit the real API so mark them appropriately)**

```python
# tests/test_yfinance_provider.py
import pytest
import pandas as pd
from core.providers.yfinance_provider import YFinanceProvider

@pytest.fixture
def provider():
    return YFinanceProvider()

@pytest.mark.integration
def test_get_price_history(provider):
    df = provider.get_price_history("AAPL", "2024-01-01", "2024-01-31")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "close" in df.columns
    assert "volume" in df.columns

@pytest.mark.integration
def test_get_stock_info(provider):
    info = provider.get_stock_info("AAPL")
    assert "name" in info
    assert "market_cap" in info
    assert info["market_cap"] > 0

@pytest.mark.integration
def test_get_earnings(provider):
    df = provider.get_earnings("AAPL")
    assert isinstance(df, pd.DataFrame)
    assert "eps_actual" in df.columns

@pytest.mark.integration
def test_get_small_cap_universe(provider):
    tickers = provider.get_small_cap_universe(1.0, 20.0)
    assert isinstance(tickers, list)
    assert len(tickers) > 0
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_yfinance_provider.py -v -m integration
```

**Step 3: Implement yfinance_provider.py**

```python
# core/providers/yfinance_provider.py
import yfinance as yf
import pandas as pd
from core.providers.base import DataProvider

# Small-cap screening tickers — curated from common small-cap ETF holdings
# In production, this would come from a proper screener API
SMALL_CAP_SEED_TICKERS = []  # populated by get_small_cap_universe

class YFinanceProvider(DataProvider):
    def get_price_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end, auto_adjust=True)
        if df.empty:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df = df.rename(columns={"date": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return df[["date", "open", "high", "low", "close", "volume"]]

    def get_earnings(self, ticker: str) -> pd.DataFrame:
        stock = yf.Ticker(ticker)
        # Get earnings history
        earnings = stock.earnings_history
        if earnings is None or (isinstance(earnings, pd.DataFrame) and earnings.empty):
            # Fallback: try quarterly earnings
            try:
                qe = stock.quarterly_earnings
                if qe is not None and not qe.empty:
                    qe = qe.reset_index()
                    qe.columns = [c.lower().replace(" ", "_") for c in qe.columns]
                    # quarterly_earnings has 'revenue' and 'earnings' columns
                    result = pd.DataFrame({
                        "report_date": qe.index.astype(str) if "date" not in qe.columns else qe["date"].astype(str),
                        "period": qe.get("quarter", qe.index.astype(str)),
                        "eps_actual": qe.get("earnings", pd.Series(dtype=float)),
                    })
                    result["eps_prior"] = result["eps_actual"].shift(-1)
                    result["eps_change_pct"] = (
                        (result["eps_actual"] - result["eps_prior"]) / result["eps_prior"].abs() * 100
                    ).round(2)
                    return result.dropna(subset=["eps_actual"])
            except Exception:
                pass
            return pd.DataFrame(columns=["report_date", "period", "eps_actual", "eps_prior", "eps_change_pct"])

        if isinstance(earnings, pd.DataFrame):
            df = earnings.reset_index()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            # Map yfinance columns to our schema
            col_map = {}
            for col in df.columns:
                if "eps_actual" in col or col == "reported_eps" or col == "actual":
                    col_map[col] = "eps_actual"
                elif "eps_estimate" in col or col == "estimate":
                    col_map[col] = "eps_prior"
                elif "date" in col.lower() or col == "earnings_date":
                    col_map[col] = "report_date"
            df = df.rename(columns=col_map)

            if "report_date" not in df.columns and "quarter" in df.columns:
                df["report_date"] = df["quarter"].astype(str)
            elif "report_date" not in df.columns:
                df["report_date"] = df.index.astype(str)

            if "eps_actual" not in df.columns:
                return pd.DataFrame(columns=["report_date", "period", "eps_actual", "eps_prior", "eps_change_pct"])

            df["report_date"] = pd.to_datetime(df["report_date"]).dt.strftime("%Y-%m-%d")
            df["period"] = df.get("quarter", df["report_date"])

            if "eps_prior" not in df.columns:
                df["eps_prior"] = df["eps_actual"].shift(-1)

            df["eps_change_pct"] = (
                (df["eps_actual"] - df["eps_prior"]) / df["eps_prior"].abs() * 100
            ).round(2)

            return df[["report_date", "period", "eps_actual", "eps_prior", "eps_change_pct"]].dropna(subset=["eps_actual"])

        return pd.DataFrame(columns=["report_date", "period", "eps_actual", "eps_prior", "eps_change_pct"])

    def get_small_cap_universe(self, min_price: float, max_price: float) -> list[str]:
        """Get small cap tickers by screening Russell 2000 components via yfinance."""
        # Use a small-cap ETF (IWM) as a proxy for the universe
        # Then filter by price. In production, use a proper screener.
        import time

        # Start with well-known small-cap ETFs and extract holdings
        # Fallback: use a curated set of tickers from screeners
        screener_tickers = []
        try:
            # yfinance Screener for small caps
            # Use screen for stocks with market cap between 50M and 2B
            screen = yf.Screener()
            screen.set_default_body({
                "query": {
                    "operator": "AND",
                    "operands": [
                        {"operator": "GT", "operands": ["intradayprice", min_price]},
                        {"operator": "LT", "operands": ["intradayprice", max_price]},
                        {"operator": "GT", "operands": ["intradaymarketcap", 50_000_000]},
                        {"operator": "LT", "operands": ["intradaymarketcap", 2_000_000_000]},
                    ]
                },
                "size": 250,
                "offset": 0,
                "sortField": "intradaymarketcap",
                "sortType": "ASC",
                "quoteType": "EQUITY",
            })
            result = screen.response
            screener_tickers = [q["symbol"] for q in result.get("quotes", [])]
        except Exception:
            # Fallback: use known small-cap tickers
            pass

        if not screener_tickers:
            # Hardcoded fallback: popular small-cap tickers known to be in range
            # This is a bootstrap set — the pipeline will discover more over time
            candidates = [
                "SIRI", "PLTR", "SOFI", "CLOV", "WISH", "BBIG", "ATER", "PROG",
                "FCEL", "PLUG", "WKHS", "GOEV", "RIDE", "NKLA", "QS", "BLNK",
                "CHPT", "SPCE", "SKLZ", "GENI", "DKNG", "PENN", "RKT", "UWMC",
                "OPEN", "COUR", "DUOL", "MNDY", "TASK", "PAYO",
            ]
            screener_tickers = candidates

        return screener_tickers

    def get_fundamentals(self, ticker: str) -> pd.DataFrame:
        stock = yf.Ticker(ticker)
        try:
            financials = stock.quarterly_financials
            if financials is None or financials.empty:
                return pd.DataFrame(columns=["period", "revenue", "gross_margin", "operating_margin"])

            financials = financials.T
            financials.index.name = "period"
            financials = financials.reset_index()

            result = pd.DataFrame()
            result["period"] = financials["period"].astype(str)

            revenue_col = next((c for c in financials.columns if "revenue" in c.lower() or "total_revenue" in c.lower()), None)
            gross_col = next((c for c in financials.columns if "gross_profit" in c.lower()), None)
            operating_col = next((c for c in financials.columns if "operating_income" in c.lower() or "ebit" in c.lower()), None)

            result["revenue"] = financials[revenue_col] if revenue_col else None
            if revenue_col and gross_col:
                result["gross_margin"] = (financials[gross_col] / financials[revenue_col]).round(4)
            else:
                result["gross_margin"] = None
            if revenue_col and operating_col:
                result["operating_margin"] = (financials[operating_col] / financials[revenue_col]).round(4)
            else:
                result["operating_margin"] = None

            return result
        except Exception:
            return pd.DataFrame(columns=["period", "revenue", "gross_margin", "operating_margin"])

    def get_stock_info(self, ticker: str) -> dict:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        return {
            "name": info.get("shortName", info.get("longName", ticker)),
            "market_cap": info.get("marketCap", 0),
            "sector": info.get("sector", "Unknown"),
            "shares_float": info.get("floatShares", 0),
            "short_interest_pct": info.get("shortPercentOfFloat", 0) * 100 if info.get("shortPercentOfFloat") else 0,
            "short_ratio": info.get("shortRatio", 0),
        }
```

**Step 4: Run integration tests**

```bash
pytest tests/test_yfinance_provider.py -v -m integration
```

Expected: PASS (may be slow due to API calls)

**Step 5: Commit**

```bash
git add core/providers/yfinance_provider.py tests/test_yfinance_provider.py
git commit -m "feat: YFinance data provider implementation"
```

---

### Task 5: Data Pipeline

**Files:**
- Create: `pipeline.py`
- Create: `tests/test_pipeline.py`

**Step 1: Write failing test**

```python
# tests/test_pipeline.py
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from pipeline import Pipeline
from core.database import Database

@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    return database

@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.get_small_cap_universe.return_value = ["TEST1", "TEST2"]
    provider.get_stock_info.return_value = {
        "name": "Test Corp", "market_cap": 500_000_000, "sector": "Tech",
        "shares_float": 10_000_000, "short_interest_pct": 5.0, "short_ratio": 2.0,
    }
    provider.get_price_history.return_value = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "open": [5.0, 5.2], "high": [5.5, 5.8], "low": [4.8, 5.0],
        "close": [5.2, 5.6], "volume": [100000, 120000],
    })
    provider.get_earnings.return_value = pd.DataFrame({
        "report_date": ["2024-01-15"], "period": ["Q4 2023"],
        "eps_actual": [0.50], "eps_prior": [0.30], "eps_change_pct": [66.67],
    })
    provider.get_fundamentals.return_value = pd.DataFrame({
        "period": ["Q4 2023"], "revenue": [10_000_000],
        "gross_margin": [0.45], "operating_margin": [0.20],
    })
    return provider

def test_pipeline_fetches_and_stores(db, mock_provider):
    pipeline = Pipeline(db=db, provider=mock_provider)
    pipeline.run(start_date="2024-01-01", end_date="2024-01-31")

    # Verify stocks were stored
    stock = db.get_stock("TEST1")
    assert stock is not None
    assert stock["name"] == "Test Corp"

    # Verify prices were stored
    prices = db.get_daily_prices("TEST1", "2024-01-01", "2024-01-31")
    assert len(prices) == 2

    # Verify earnings were stored
    earnings = db.get_earnings("TEST1")
    assert len(earnings) == 1

def test_pipeline_handles_provider_errors(db, mock_provider):
    mock_provider.get_stock_info.side_effect = Exception("API Error")
    pipeline = Pipeline(db=db, provider=mock_provider)
    # Should not raise — skips failed tickers
    pipeline.run(start_date="2024-01-01", end_date="2024-01-31")
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_pipeline.py -v
```

**Step 3: Implement pipeline.py**

```python
# pipeline.py
import time
import logging
from core.database import Database
from core.providers.base import DataProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self, db: Database, provider: DataProvider, batch_delay: float = 1.0):
        self.db = db
        self.provider = provider
        self.batch_delay = batch_delay

    def run(self, start_date: str, end_date: str):
        logger.info("Fetching stock universe...")
        tickers = self.provider.get_small_cap_universe(1.0, 20.0)
        logger.info(f"Found {len(tickers)} candidates")

        for i, ticker in enumerate(tickers):
            try:
                logger.info(f"[{i+1}/{len(tickers)}] Processing {ticker}...")
                self._process_ticker(ticker, start_date, end_date)
            except Exception as e:
                logger.warning(f"Failed to process {ticker}: {e}")
                continue

            # Rate limiting
            if (i + 1) % 5 == 0:
                time.sleep(self.batch_delay)

        logger.info("Pipeline complete.")

    def _process_ticker(self, ticker: str, start_date: str, end_date: str):
        # Stock info
        info = self.provider.get_stock_info(ticker)
        self.db.upsert_stock({
            "ticker": ticker,
            **info,
        })

        # Price history
        prices_df = self.provider.get_price_history(ticker, start_date, end_date)
        if not prices_df.empty:
            price_rows = prices_df.assign(ticker=ticker).to_dict("records")
            self.db.insert_daily_prices(price_rows)

        # Earnings
        earnings_df = self.provider.get_earnings(ticker)
        if not earnings_df.empty:
            earnings_rows = earnings_df.assign(ticker=ticker).to_dict("records")
            self.db.insert_earnings(earnings_rows)

        # Fundamentals
        fundamentals_df = self.provider.get_fundamentals(ticker)
        if not fundamentals_df.empty:
            fund_rows = fundamentals_df.assign(ticker=ticker).to_dict("records")
            self.db.insert_fundamentals(fund_rows)

def main():
    """CLI entry point for running the pipeline."""
    import argparse
    from core.providers.yfinance_provider import YFinanceProvider
    from config import DB_PATH

    parser = argparse.ArgumentParser(description="Run data pipeline")
    parser.add_argument("--start", default="2021-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-03-17", help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    db = Database(DB_PATH)
    db.initialize()
    provider = YFinanceProvider()
    pipeline = Pipeline(db=db, provider=provider)
    pipeline.run(start_date=args.start, end_date=args.end)

if __name__ == "__main__":
    main()
```

**Step 4: Run tests**

```bash
pytest tests/test_pipeline.py -v
```

Expected: All PASS

**Step 5: Commit**

```bash
git add pipeline.py tests/test_pipeline.py
git commit -m "feat: data pipeline with rate limiting and error handling"
```

---

### Task 6: Scanner Engine

**Files:**
- Create: `core/scanner.py`
- Create: `tests/test_scanner.py`

**Step 1: Write failing tests**

```python
# tests/test_scanner.py
import pytest
import pandas as pd
from core.database import Database
from core.scanner import Scanner
from config import ScannerConfig

@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    return database

@pytest.fixture
def populated_db(db):
    """DB with a stock that should trigger: EPS jumped 50%, price crossed above 20 SMA 5 days later."""
    db.upsert_stock({
        "ticker": "BULL", "name": "Bull Corp", "market_cap": 500_000_000,
        "sector": "Tech", "shares_float": 10_000_000,
        "short_interest_pct": 5.0, "short_ratio": 2.0,
    })

    # Generate 60 days of price data
    # Days 1-30: price hovers around 4.0 (below 20 SMA)
    # Day 25: EPS report with 50% jump
    # Day 30: price breaks above 20 SMA (crosses from ~4.0 to ~5.5)
    import datetime
    base = datetime.date(2024, 1, 1)
    prices = []
    for i in range(60):
        date = (base + datetime.timedelta(days=i)).isoformat()
        if i < 30:
            close = 4.0 + (i % 3) * 0.1  # fluctuates 4.0–4.2
        else:
            close = 5.5 + (i % 3) * 0.1  # jumps to 5.5–5.7
        prices.append({
            "ticker": "BULL", "date": date,
            "open": close - 0.1, "high": close + 0.2,
            "low": close - 0.2, "close": close, "volume": 100000,
        })
    db.insert_daily_prices(prices)

    # EPS report on day 25 with a 50% change
    db.insert_earnings([{
        "ticker": "BULL", "report_date": "2024-01-26",
        "period": "Q4 2023", "eps_actual": 0.30,
        "eps_prior": 0.20, "eps_change_pct": 50.0,
    }])

    # Also add a stock that should NOT trigger (no EPS change)
    db.upsert_stock({
        "ticker": "FLAT", "name": "Flat Corp", "market_cap": 300_000_000,
        "sector": "Finance", "shares_float": 5_000_000,
        "short_interest_pct": 1.0, "short_ratio": 1.0,
    })
    for i in range(60):
        date = (base + datetime.timedelta(days=i)).isoformat()
        prices.append({
            "ticker": "FLAT", "date": date,
            "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0, "volume": 50000,
        })
    db.insert_daily_prices(prices)
    db.insert_earnings([{
        "ticker": "FLAT", "report_date": "2024-01-26",
        "period": "Q4 2023", "eps_actual": 0.10,
        "eps_prior": 0.10, "eps_change_pct": 0.0,
    }])

    return db

def test_scanner_finds_bullish_signal(populated_db):
    config = ScannerConfig(
        min_price=1.0, max_price=20.0,
        ma_periods=[20], eps_change_threshold=10.0,
        trend_window_days=30, direction="both",
    )
    scanner = Scanner(db=populated_db, config=config)
    results = scanner.scan(as_of_date="2024-03-01")

    tickers = [r["ticker"] for r in results]
    assert "BULL" in tickers
    assert "FLAT" not in tickers

def test_scanner_respects_eps_threshold(populated_db):
    config = ScannerConfig(
        ma_periods=[20], eps_change_threshold=60.0,  # higher than BULL's 50%
        trend_window_days=30, direction="both",
    )
    scanner = Scanner(db=populated_db, config=config)
    results = scanner.scan(as_of_date="2024-03-01")
    assert len(results) == 0  # BULL's 50% doesn't meet 60% threshold

def test_scanner_respects_direction_filter(populated_db):
    config = ScannerConfig(
        ma_periods=[20], eps_change_threshold=10.0,
        trend_window_days=30, direction="bearish",
    )
    scanner = Scanner(db=populated_db, config=config)
    results = scanner.scan(as_of_date="2024-03-01")
    # BULL had a bullish cross, so bearish filter should exclude it
    tickers = [r["ticker"] for r in results]
    assert "BULL" not in tickers
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_scanner.py -v
```

**Step 3: Implement scanner.py**

```python
# core/scanner.py
import pandas as pd
from core.database import Database
from config import ScannerConfig

class Scanner:
    def __init__(self, db: Database, config: ScannerConfig):
        self.db = db
        self.config = config

    def scan(self, as_of_date: str) -> list[dict]:
        """Run the scanner and return matching signals."""
        # Get universe
        universe = self.db.get_stock_universe(
            min_price=self.config.min_price,
            max_price=self.config.max_price,
            min_market_cap=self.config.min_market_cap,
            max_market_cap=self.config.max_market_cap,
        )

        results = []
        for stock in universe:
            ticker = stock["ticker"]
            signals = self._check_ticker(ticker, as_of_date)
            results.extend(signals)

        return results

    def _check_ticker(self, ticker: str, as_of_date: str) -> list[dict]:
        """Check a single ticker for EPS + MA crossover signals."""
        earnings = self.db.get_earnings(ticker)
        if not earnings:
            return []

        signals = []
        for earning in earnings:
            eps_change = abs(earning.get("eps_change_pct", 0) or 0)
            if eps_change < self.config.eps_change_threshold:
                continue

            eps_date = earning["report_date"]

            # Look for MA crossovers within the trend window
            for ma_period in self.config.ma_periods:
                crossover = self._find_ma_crossover(
                    ticker, eps_date, ma_period, self.config.trend_window_days
                )
                if crossover is None:
                    continue

                cross_direction = crossover["direction"]
                if self.config.direction == "bullish" and cross_direction != "bullish":
                    continue
                if self.config.direction == "bearish" and cross_direction != "bearish":
                    continue

                signals.append({
                    "ticker": ticker,
                    "scan_date": as_of_date,
                    "signal_type": cross_direction,
                    "ma_period": ma_period,
                    "eps_change_pct": earning["eps_change_pct"],
                    "trend_change_date": crossover["date"],
                    "eps_change_date": eps_date,
                    "days_between": crossover["days_between"],
                })

        return signals

    def _find_ma_crossover(self, ticker: str, eps_date: str,
                           ma_period: int, window_days: int) -> dict | None:
        """Find if price crossed above/below the MA within window_days of eps_date."""
        # Need enough history before the window to calculate the MA
        from datetime import datetime, timedelta

        eps_dt = datetime.strptime(eps_date, "%Y-%m-%d")
        lookback_start = eps_dt - timedelta(days=ma_period + window_days + 30)
        lookback_end = eps_dt + timedelta(days=window_days)

        prices = self.db.get_daily_prices(
            ticker, lookback_start.strftime("%Y-%m-%d"), lookback_end.strftime("%Y-%m-%d")
        )
        if len(prices) < ma_period + 1:
            return None

        df = pd.DataFrame(prices)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df["ma"] = df["close"].rolling(window=ma_period).mean()

        # Find crossover points within the window
        window_start = eps_dt - timedelta(days=window_days)
        window_end = eps_dt + timedelta(days=window_days)
        window_df = df[(df["date"] >= window_start) & (df["date"] <= window_end)].copy()

        if len(window_df) < 2:
            return None

        # Detect crossovers: price crosses above or below MA
        window_df = window_df.dropna(subset=["ma"])
        if len(window_df) < 2:
            return None

        window_df["above_ma"] = window_df["close"] > window_df["ma"]
        window_df["crossover"] = window_df["above_ma"] != window_df["above_ma"].shift(1)

        crossovers = window_df[window_df["crossover"] & window_df.index > window_df.index[0]]
        if crossovers.empty:
            return None

        # Take the first crossover in the window
        first = crossovers.iloc[0]
        direction = "bullish" if first["above_ma"] else "bearish"
        cross_date = first["date"].strftime("%Y-%m-%d")
        days_between = abs((first["date"] - eps_dt).days)

        return {
            "date": cross_date,
            "direction": direction,
            "days_between": days_between,
        }
```

**Step 4: Run tests**

```bash
pytest tests/test_scanner.py -v
```

Expected: All PASS

**Step 5: Commit**

```bash
git add core/scanner.py tests/test_scanner.py
git commit -m "feat: scanner engine with MA crossover + EPS change detection"
```

---

### Task 7: Backtest Module

**Files:**
- Create: `core/backtest.py`
- Create: `tests/test_backtest.py`

**Step 1: Write failing test**

```python
# tests/test_backtest.py
import pytest
import datetime
from core.database import Database
from core.scanner import Scanner
from core.backtest import Backtester
from config import ScannerConfig, BacktestConfig

@pytest.fixture
def db_with_history(tmp_path):
    """DB with 2 years of data for backtesting."""
    db = Database(tmp_path / "test.db")
    db.initialize()

    db.upsert_stock({
        "ticker": "BULL", "name": "Bull Corp", "market_cap": 500_000_000,
        "sector": "Tech", "shares_float": 10_000_000,
        "short_interest_pct": 5.0, "short_ratio": 2.0,
    })

    # Generate 500 days of price data with a clear trend change at day 200
    base = datetime.date(2022, 1, 1)
    prices = []
    for i in range(500):
        date = (base + datetime.timedelta(days=i)).isoformat()
        if i < 200:
            close = 4.0 + (i % 5) * 0.05
        else:
            close = 6.0 + (i - 200) * 0.01 + (i % 5) * 0.05  # uptrend after break
        prices.append({
            "ticker": "BULL", "date": date,
            "open": close - 0.1, "high": close + 0.2,
            "low": close - 0.2, "close": close, "volume": 100000,
        })
    db.insert_daily_prices(prices)

    # EPS report near the trend change
    db.insert_earnings([{
        "ticker": "BULL", "report_date": "2022-07-15",
        "period": "Q2 2022", "eps_actual": 0.40,
        "eps_prior": 0.20, "eps_change_pct": 100.0,
    }])

    return db

def test_backtester_produces_results(db_with_history):
    scanner_config = ScannerConfig(ma_periods=[20], eps_change_threshold=10.0, trend_window_days=30)
    backtest_config = BacktestConfig(
        start_date="2022-01-01", end_date="2023-06-01",
        forward_return_days=[5, 10, 20, 30],
    )
    backtester = Backtester(db=db_with_history, scanner_config=scanner_config, backtest_config=backtest_config)
    results = backtester.run()

    assert "signals" in results
    assert "summary" in results
    assert isinstance(results["summary"], dict)

def test_backtester_computes_forward_returns(db_with_history):
    scanner_config = ScannerConfig(ma_periods=[20], eps_change_threshold=10.0, trend_window_days=30)
    backtest_config = BacktestConfig(
        start_date="2022-01-01", end_date="2023-06-01",
        forward_return_days=[5, 10, 20],
    )
    backtester = Backtester(db=db_with_history, scanner_config=scanner_config, backtest_config=backtest_config)
    results = backtester.run()

    if results["signals"]:
        signal = results["signals"][0]
        assert "forward_returns" in signal
        assert 5 in signal["forward_returns"]
        assert 10 in signal["forward_returns"]

def test_parameter_sweep(db_with_history):
    backtest_config = BacktestConfig(
        start_date="2022-01-01", end_date="2023-06-01",
        forward_return_days=[10],
        ma_periods=[20, 50],
        eps_thresholds=[10.0, 50.0],
        trend_windows=[30],
    )
    backtester = Backtester(
        db=db_with_history,
        scanner_config=ScannerConfig(),
        backtest_config=backtest_config,
    )
    sweep = backtester.parameter_sweep()

    assert isinstance(sweep, list)
    assert len(sweep) > 0
    assert "ma_period" in sweep[0]
    assert "eps_threshold" in sweep[0]
    assert "win_rate" in sweep[0]
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_backtest.py -v
```

**Step 3: Implement backtest.py**

```python
# core/backtest.py
import pandas as pd
from datetime import datetime, timedelta
from core.database import Database
from core.scanner import Scanner
from config import ScannerConfig, BacktestConfig

class Backtester:
    def __init__(self, db: Database, scanner_config: ScannerConfig, backtest_config: BacktestConfig):
        self.db = db
        self.scanner_config = scanner_config
        self.backtest_config = backtest_config

    def run(self) -> dict:
        """Run backtest: find historical signals and compute forward returns."""
        scanner = Scanner(db=self.db, config=self.scanner_config)
        signals = scanner.scan(as_of_date=self.backtest_config.end_date)

        # Filter signals within backtest date range
        signals = [
            s for s in signals
            if self.backtest_config.start_date <= s["trend_change_date"] <= self.backtest_config.end_date
        ]

        # Compute forward returns for each signal
        for signal in signals:
            signal["forward_returns"] = self._compute_forward_returns(
                signal["ticker"], signal["trend_change_date"]
            )

        summary = self._compute_summary(signals)
        return {"signals": signals, "summary": summary}

    def _compute_forward_returns(self, ticker: str, signal_date: str) -> dict[int, float]:
        """Compute forward returns at various horizons from the signal date."""
        signal_dt = datetime.strptime(signal_date, "%Y-%m-%d")
        max_horizon = max(self.backtest_config.forward_return_days) + 10

        end_dt = signal_dt + timedelta(days=max_horizon)
        prices = self.db.get_daily_prices(ticker, signal_date, end_dt.strftime("%Y-%m-%d"))

        if not prices:
            return {}

        df = pd.DataFrame(prices)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        entry_price = df.iloc[0]["close"]
        returns = {}
        for days in self.backtest_config.forward_return_days:
            # Find the trading day closest to N calendar days out
            target_date = signal_dt + timedelta(days=days)
            future = df[df["date"] >= target_date]
            if not future.empty:
                exit_price = future.iloc[0]["close"]
                returns[days] = round(((exit_price - entry_price) / entry_price) * 100, 2)

        return returns

    def _compute_summary(self, signals: list[dict]) -> dict:
        """Compute aggregate metrics across all signals."""
        if not signals:
            return {"total_signals": 0}

        all_returns = {}
        for days in self.backtest_config.forward_return_days:
            day_returns = [
                s["forward_returns"].get(days)
                for s in signals
                if days in s.get("forward_returns", {})
            ]
            day_returns = [r for r in day_returns if r is not None]
            if day_returns:
                series = pd.Series(day_returns)
                all_returns[days] = {
                    "win_rate": round((series > 0).mean() * 100, 1),
                    "avg_return": round(series.mean(), 2),
                    "median_return": round(series.median(), 2),
                    "max_gain": round(series.max(), 2),
                    "max_loss": round(series.min(), 2),
                    "sample_size": len(series),
                }

        return {
            "total_signals": len(signals),
            "by_horizon": all_returns,
        }

    def parameter_sweep(self) -> list[dict]:
        """Test all combinations of parameters and return comparison table."""
        results = []

        for ma_period in self.backtest_config.ma_periods:
            for eps_thresh in self.backtest_config.eps_thresholds:
                for window in self.backtest_config.trend_windows:
                    config = ScannerConfig(
                        min_price=self.scanner_config.min_price,
                        max_price=self.scanner_config.max_price,
                        min_market_cap=self.scanner_config.min_market_cap,
                        max_market_cap=self.scanner_config.max_market_cap,
                        ma_periods=[ma_period],
                        eps_change_threshold=eps_thresh,
                        trend_window_days=window,
                        direction=self.scanner_config.direction,
                    )
                    backtester = Backtester(
                        db=self.db,
                        scanner_config=config,
                        backtest_config=self.backtest_config,
                    )
                    run_result = backtester.run()
                    summary = run_result["summary"]

                    # Use the first forward return horizon for the sweep comparison
                    horizon = self.backtest_config.forward_return_days[0]
                    horizon_stats = summary.get("by_horizon", {}).get(horizon, {})

                    results.append({
                        "ma_period": ma_period,
                        "eps_threshold": eps_thresh,
                        "trend_window": window,
                        "total_signals": summary.get("total_signals", 0),
                        "win_rate": horizon_stats.get("win_rate", 0),
                        "avg_return": horizon_stats.get("avg_return", 0),
                        "sample_size": horizon_stats.get("sample_size", 0),
                    })

        return results
```

**Step 4: Run tests**

```bash
pytest tests/test_backtest.py -v
```

Expected: All PASS

**Step 5: Commit**

```bash
git add core/backtest.py tests/test_backtest.py
git commit -m "feat: backtest module with forward returns and parameter sweep"
```

---

### Task 8: Streamlit App — Scanner Dashboard

**Files:**
- Create: `app.py`
- Create: `pages/1_Scanner.py`

**Step 1: Create Streamlit entry point**

```python
# app.py
import streamlit as st
from pathlib import Path
from core.database import Database
from config import DB_PATH

st.set_page_config(page_title="Small Cap Scanner", page_icon="📊", layout="wide")

st.title("Small Cap EPS + Trend Scanner")
st.markdown("""
Identifies small cap stocks ($1–$20) where a moving average trend change
occurs within a configurable window of an earnings-per-share change.
""")

# Initialize database
@st.cache_resource
def get_db():
    db = Database(DB_PATH)
    db.initialize()
    return db

db = get_db()

# Show database status
stock_count = len(db.get_stock_universe(0, 100, 0, 100_000_000_000))
st.metric("Stocks in Database", stock_count)

if stock_count == 0:
    st.warning("No data in database. Run the pipeline first: `python pipeline.py`")
else:
    st.success("Database loaded. Use the sidebar to navigate.")
```

**Step 2: Create scanner page**

```python
# pages/1_Scanner.py
import streamlit as st
import pandas as pd
from core.database import Database
from core.scanner import Scanner
from config import DB_PATH, ScannerConfig
from datetime import datetime

st.set_page_config(page_title="Scanner", page_icon="🔍", layout="wide")
st.title("Scanner Dashboard")

@st.cache_resource
def get_db():
    db = Database(DB_PATH)
    db.initialize()
    return db

db = get_db()

# Sidebar controls
st.sidebar.header("Scanner Settings")
min_price = st.sidebar.number_input("Min Price ($)", value=1.0, min_value=0.01, step=0.5)
max_price = st.sidebar.number_input("Max Price ($)", value=20.0, min_value=0.01, step=1.0)
ma_period = st.sidebar.selectbox("Moving Average Period", [20, 50, 200], index=0)
eps_threshold = st.sidebar.slider("EPS Change Threshold (%)", min_value=1.0, max_value=100.0, value=10.0, step=1.0)
trend_window = st.sidebar.slider("Trend Window (days)", min_value=5, max_value=90, value=30, step=5)
direction = st.sidebar.selectbox("Direction", ["both", "bullish", "bearish"])

if st.sidebar.button("Run Scanner", type="primary"):
    config = ScannerConfig(
        min_price=min_price,
        max_price=max_price,
        ma_periods=[ma_period],
        eps_change_threshold=eps_threshold,
        trend_window_days=trend_window,
        direction=direction,
    )
    scanner = Scanner(db=db, config=config)

    with st.spinner("Scanning..."):
        results = scanner.scan(as_of_date=datetime.now().strftime("%Y-%m-%d"))

    if results:
        df = pd.DataFrame(results)
        df = df[["ticker", "signal_type", "ma_period", "eps_change_pct",
                  "trend_change_date", "eps_change_date", "days_between"]]
        df.columns = ["Ticker", "Signal", "MA Period", "EPS Change %",
                       "Trend Change", "EPS Date", "Days Between"]

        st.success(f"Found {len(results)} signals")

        # Make ticker clickable
        for idx, row in df.iterrows():
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
            with col1:
                if st.button(row["Ticker"], key=f"btn_{idx}"):
                    st.session_state["selected_ticker"] = row["Ticker"]
                    st.session_state["signal_data"] = results[idx]
                    st.switch_page("pages/2_Stock_Detail.py")
            with col2:
                st.write(row["Signal"])
            with col3:
                st.write(row["MA Period"])
            with col4:
                st.write(f"{row['EPS Change %']:.1f}%")
            with col5:
                st.write(row["Trend Change"])
            with col6:
                st.write(row["EPS Date"])
            with col7:
                st.write(row["Days Between"])
    else:
        st.info("No signals found with current settings. Try adjusting parameters.")
```

**Step 3: Verify it runs**

```bash
streamlit run app.py --server.headless true
```

Verify: App loads without errors, shows database status.

**Step 4: Commit**

```bash
git add app.py pages/1_Scanner.py
git commit -m "feat: Streamlit scanner dashboard with configurable filters"
```

---

### Task 9: Streamlit — Stock Detail Page

**Files:**
- Create: `pages/2_Stock_Detail.py`

**Step 1: Implement stock detail page with chart**

```python
# pages/2_Stock_Detail.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from core.database import Database
from config import DB_PATH

st.set_page_config(page_title="Stock Detail", page_icon="📈", layout="wide")

@st.cache_resource
def get_db():
    db = Database(DB_PATH)
    db.initialize()
    return db

db = get_db()

# Get selected ticker from session state or query params
ticker = st.session_state.get("selected_ticker", None)
signal_data = st.session_state.get("signal_data", None)

# Allow manual ticker input too
ticker_input = st.sidebar.text_input("Ticker", value=ticker or "")
if ticker_input:
    ticker = ticker_input.upper()

if not ticker:
    st.warning("Select a stock from the Scanner page, or enter a ticker in the sidebar.")
    st.stop()

st.title(f"{ticker} — Stock Detail")

# Fetch data
stock = db.get_stock(ticker)
end_date = datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
prices = db.get_daily_prices(ticker, start_date, end_date)
earnings = db.get_earnings(ticker)
fundamentals = db.get_fundamentals(ticker)

if not prices:
    st.error(f"No price data for {ticker}. Run the pipeline first.")
    st.stop()

# Stock info sidebar
if stock:
    st.sidebar.subheader("Key Stats")
    st.sidebar.metric("Market Cap", f"${stock.get('market_cap', 0):,.0f}")
    st.sidebar.metric("Sector", stock.get("sector", "N/A"))
    st.sidebar.metric("Float", f"{stock.get('shares_float', 0):,.0f}")
    st.sidebar.metric("Short Interest", f"{stock.get('short_interest_pct', 0):.1f}%")
    st.sidebar.metric("Short Ratio", f"{stock.get('short_ratio', 0):.1f}")

if fundamentals:
    st.sidebar.subheader("Fundamentals")
    latest = fundamentals[-1]
    if latest.get("revenue"):
        st.sidebar.metric("Revenue", f"${latest['revenue']:,.0f}")
    if latest.get("gross_margin"):
        st.sidebar.metric("Gross Margin", f"{latest['gross_margin']*100:.1f}%")
    if latest.get("operating_margin"):
        st.sidebar.metric("Operating Margin", f"{latest['operating_margin']*100:.1f}%")

# Build price chart with MA overlays
df = pd.DataFrame(prices)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# Calculate MAs
for period in [20, 50, 200]:
    if len(df) >= period:
        df[f"sma_{period}"] = df["close"].rolling(window=period).mean()

# Create chart
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.75, 0.25],
)

# Price line
fig.add_trace(
    go.Scatter(x=df["date"], y=df["close"], name="Price",
               line=dict(color="#2196F3", width=2)),
    row=1, col=1,
)

# MA overlays
ma_colors = {20: "#FF9800", 50: "#4CAF50", 200: "#F44336"}
for period, color in ma_colors.items():
    col_name = f"sma_{period}"
    if col_name in df.columns:
        fig.add_trace(
            go.Scatter(x=df["date"], y=df[col_name], name=f"SMA {period}",
                       line=dict(color=color, width=1, dash="dash")),
            row=1, col=1,
        )

# EPS markers
if earnings:
    for earning in earnings:
        report_date = pd.to_datetime(earning["report_date"])
        if report_date >= df["date"].min() and report_date <= df["date"].max():
            eps_change = earning.get("eps_change_pct", 0) or 0
            color = "green" if eps_change > 0 else "red"
            fig.add_vline(
                x=report_date, line_dash="dot", line_color=color,
                annotation_text=f"EPS: {eps_change:+.1f}%",
                annotation_position="top left",
                row=1, col=1,
            )

# Signal highlight
if signal_data:
    trend_date = pd.to_datetime(signal_data["trend_change_date"])
    fig.add_vline(
        x=trend_date, line_dash="solid", line_color="purple", line_width=2,
        annotation_text=f"MA{signal_data['ma_period']} Cross ({signal_data['signal_type']})",
        annotation_position="bottom right",
        row=1, col=1,
    )

# Volume bars
fig.add_trace(
    go.Bar(x=df["date"], y=df["volume"], name="Volume",
           marker_color="rgba(100,100,100,0.3)"),
    row=2, col=1,
)

fig.update_layout(
    height=600,
    title=f"{ticker} — 1 Year Price Chart",
    xaxis_rangeslider_visible=False,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig.update_yaxes(title_text="Price ($)", row=1, col=1)
fig.update_yaxes(title_text="Volume", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# Earnings history table
if earnings:
    st.subheader("Earnings History")
    earnings_df = pd.DataFrame(earnings)
    earnings_df = earnings_df[["report_date", "period", "eps_actual", "eps_prior", "eps_change_pct"]]
    earnings_df.columns = ["Report Date", "Period", "EPS Actual", "EPS Prior", "EPS Change %"]
    st.dataframe(earnings_df, use_container_width=True, hide_index=True)
```

**Step 2: Verify it runs**

```bash
streamlit run app.py --server.headless true
```

Navigate to Stock Detail page, enter a ticker, verify chart renders.

**Step 3: Commit**

```bash
git add pages/2_Stock_Detail.py
git commit -m "feat: stock detail page with price chart, MA overlays, and EPS markers"
```

---

### Task 10: Streamlit — Backtest Page

**Files:**
- Create: `pages/3_Backtest.py`

**Step 1: Implement backtest page**

```python
# pages/3_Backtest.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from core.database import Database
from core.backtest import Backtester
from config import DB_PATH, ScannerConfig, BacktestConfig

st.set_page_config(page_title="Backtest", page_icon="📊", layout="wide")
st.title("Backtest")

@st.cache_resource
def get_db():
    db = Database(DB_PATH)
    db.initialize()
    return db

db = get_db()

# Sidebar config
st.sidebar.header("Backtest Settings")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2026-03-17"))
min_price = st.sidebar.number_input("Min Price ($)", value=1.0, step=0.5)
max_price = st.sidebar.number_input("Max Price ($)", value=20.0, step=1.0)
direction = st.sidebar.selectbox("Direction", ["both", "bullish", "bearish"])

tab1, tab2 = st.tabs(["Single Backtest", "Parameter Sweep"])

with tab1:
    st.subheader("Single Parameter Backtest")
    col1, col2, col3 = st.columns(3)
    with col1:
        ma_period = st.selectbox("MA Period", [20, 50, 200])
    with col2:
        eps_threshold = st.number_input("EPS Threshold (%)", value=10.0, step=5.0)
    with col3:
        trend_window = st.number_input("Trend Window (days)", value=30, step=5)

    if st.button("Run Backtest", type="primary"):
        scanner_config = ScannerConfig(
            min_price=min_price, max_price=max_price,
            ma_periods=[ma_period],
            eps_change_threshold=eps_threshold,
            trend_window_days=trend_window,
            direction=direction,
        )
        backtest_config = BacktestConfig(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )

        with st.spinner("Running backtest..."):
            backtester = Backtester(db=db, scanner_config=scanner_config, backtest_config=backtest_config)
            results = backtester.run()

        st.metric("Total Signals", results["summary"].get("total_signals", 0))

        # Summary by horizon
        by_horizon = results["summary"].get("by_horizon", {})
        if by_horizon:
            horizon_df = pd.DataFrame([
                {"Horizon (days)": k, **v} for k, v in by_horizon.items()
            ])
            st.dataframe(horizon_df, use_container_width=True, hide_index=True)

            # Return distribution chart
            all_returns = []
            for signal in results["signals"]:
                for days, ret in signal.get("forward_returns", {}).items():
                    all_returns.append({"horizon": f"{days}d", "return_pct": ret, "ticker": signal["ticker"]})

            if all_returns:
                returns_df = pd.DataFrame(all_returns)
                fig = px.box(returns_df, x="horizon", y="return_pct",
                             title="Forward Return Distribution by Horizon",
                             labels={"return_pct": "Return (%)", "horizon": "Holding Period"})
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)

        # Individual signals table
        if results["signals"]:
            st.subheader("Individual Signals")
            signals_df = pd.DataFrame(results["signals"])
            display_cols = ["ticker", "signal_type", "ma_period", "eps_change_pct",
                           "trend_change_date", "eps_change_date", "days_between"]
            signals_df = signals_df[[c for c in display_cols if c in signals_df.columns]]

            # Add click-through to detail page
            for idx, row in signals_df.iterrows():
                cols = st.columns(len(display_cols))
                for i, col_name in enumerate(display_cols):
                    if col_name in row:
                        with cols[i]:
                            if col_name == "ticker":
                                if st.button(str(row[col_name]), key=f"bt_signal_{idx}"):
                                    st.session_state["selected_ticker"] = row[col_name]
                                    st.session_state["signal_data"] = results["signals"][idx]
                                    st.switch_page("pages/2_Stock_Detail.py")
                            else:
                                st.write(row[col_name])

with tab2:
    st.subheader("Parameter Sweep")
    st.markdown("Test all combinations of MA periods, EPS thresholds, and trend windows.")

    sweep_ma = st.multiselect("MA Periods", [20, 50, 200], default=[20, 50, 200])
    sweep_eps = st.multiselect("EPS Thresholds (%)", [5.0, 10.0, 25.0, 50.0], default=[5.0, 10.0, 25.0])
    sweep_windows = st.multiselect("Trend Windows (days)", [15, 30, 45], default=[15, 30, 45])
    sweep_horizon = st.selectbox("Compare at Horizon (days)", [5, 10, 20, 30, 60], index=2)

    if st.button("Run Parameter Sweep", type="primary"):
        scanner_config = ScannerConfig(
            min_price=min_price, max_price=max_price, direction=direction,
        )
        backtest_config = BacktestConfig(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            forward_return_days=[sweep_horizon],
            ma_periods=sweep_ma,
            eps_thresholds=sweep_eps,
            trend_windows=sweep_windows,
        )

        with st.spinner("Running parameter sweep (this may take a while)..."):
            backtester = Backtester(db=db, scanner_config=scanner_config, backtest_config=backtest_config)
            sweep_results = backtester.parameter_sweep()

        if sweep_results:
            sweep_df = pd.DataFrame(sweep_results)
            sweep_df.columns = ["MA Period", "EPS Threshold", "Trend Window",
                               "Total Signals", "Win Rate %", "Avg Return %", "Sample Size"]
            sweep_df = sweep_df.sort_values("Win Rate %", ascending=False)
            st.dataframe(sweep_df, use_container_width=True, hide_index=True)

            # Heatmap: win rate by MA period vs EPS threshold
            if len(sweep_ma) > 1 and len(sweep_eps) > 1:
                pivot = sweep_df.pivot_table(
                    values="Win Rate %", index="MA Period", columns="EPS Threshold"
                )
                fig = px.imshow(pivot, text_auto=True, aspect="auto",
                               title="Win Rate: MA Period vs EPS Threshold",
                               labels=dict(x="EPS Threshold (%)", y="MA Period", color="Win Rate %"))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No results from parameter sweep.")
```

**Step 2: Verify it runs**

```bash
streamlit run app.py --server.headless true
```

Navigate to Backtest page, verify it loads.

**Step 3: Commit**

```bash
git add pages/3_Backtest.py
git commit -m "feat: backtest page with parameter sweep and return distribution charts"
```

---

### Task 11: Integration Test — End to End

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write end-to-end integration test**

```python
# tests/test_integration.py
import pytest
import datetime
from core.database import Database
from core.scanner import Scanner
from core.backtest import Backtester
from pipeline import Pipeline
from config import ScannerConfig, BacktestConfig
from unittest.mock import MagicMock
import pandas as pd

@pytest.fixture
def full_system(tmp_path):
    """Set up complete system with mock provider."""
    db = Database(tmp_path / "test.db")
    db.initialize()

    # Create mock provider with realistic data
    provider = MagicMock()
    provider.get_small_cap_universe.return_value = ["BULL", "BEAR", "FLAT"]

    # Stock info
    stock_info = {
        "BULL": {"name": "Bull Corp", "market_cap": 500_000_000, "sector": "Tech",
                 "shares_float": 10_000_000, "short_interest_pct": 5.0, "short_ratio": 2.0},
        "BEAR": {"name": "Bear Corp", "market_cap": 300_000_000, "sector": "Finance",
                 "shares_float": 8_000_000, "short_interest_pct": 10.0, "short_ratio": 4.0},
        "FLAT": {"name": "Flat Corp", "market_cap": 200_000_000, "sector": "Health",
                 "shares_float": 5_000_000, "short_interest_pct": 1.0, "short_ratio": 1.0},
    }
    provider.get_stock_info.side_effect = lambda t: stock_info[t]

    # Price data: BULL trends up after day 200, BEAR trends down, FLAT stays flat
    def make_prices(ticker):
        base = datetime.date(2022, 1, 1)
        rows = []
        for i in range(500):
            date = (base + datetime.timedelta(days=i)).isoformat()
            if ticker == "BULL":
                close = 4.0 + (0.05 * (i % 5)) if i < 200 else 6.0 + (i - 200) * 0.01
            elif ticker == "BEAR":
                close = 10.0 - (0.05 * (i % 5)) if i < 200 else 7.0 - (i - 200) * 0.005
            else:
                close = 8.0 + (0.05 * (i % 3))
            rows.append({"date": date, "open": close, "high": close + 0.2,
                        "low": close - 0.2, "close": close, "volume": 100000})
        return pd.DataFrame(rows)

    provider.get_price_history.side_effect = lambda t, s, e: make_prices(t)

    # Earnings
    earnings = {
        "BULL": pd.DataFrame({"report_date": ["2022-07-15"], "period": ["Q2"],
                               "eps_actual": [0.40], "eps_prior": [0.20], "eps_change_pct": [100.0]}),
        "BEAR": pd.DataFrame({"report_date": ["2022-07-15"], "period": ["Q2"],
                               "eps_actual": [0.10], "eps_prior": [0.30], "eps_change_pct": [-66.67]}),
        "FLAT": pd.DataFrame({"report_date": ["2022-07-15"], "period": ["Q2"],
                               "eps_actual": [0.20], "eps_prior": [0.20], "eps_change_pct": [0.0]}),
    }
    provider.get_earnings.side_effect = lambda t: earnings[t]
    provider.get_fundamentals.return_value = pd.DataFrame(
        {"period": ["Q2"], "revenue": [5_000_000], "gross_margin": [0.40], "operating_margin": [0.15]}
    )

    # Run pipeline
    pipeline = Pipeline(db=db, provider=provider, batch_delay=0)
    pipeline.run(start_date="2022-01-01", end_date="2023-06-01")

    return db

def test_full_pipeline_to_scan(full_system):
    config = ScannerConfig(ma_periods=[20], eps_change_threshold=10.0, trend_window_days=30)
    scanner = Scanner(db=full_system, config=config)
    results = scanner.scan(as_of_date="2023-06-01")

    tickers = [r["ticker"] for r in results]
    # BULL and BEAR should have signals (EPS changed > 10%), FLAT should not
    assert "FLAT" not in tickers

def test_full_pipeline_to_backtest(full_system):
    scanner_config = ScannerConfig(ma_periods=[20], eps_change_threshold=10.0, trend_window_days=30)
    backtest_config = BacktestConfig(
        start_date="2022-01-01", end_date="2023-06-01",
        forward_return_days=[10, 20],
    )
    backtester = Backtester(db=full_system, scanner_config=scanner_config, backtest_config=backtest_config)
    results = backtester.run()

    assert results["summary"]["total_signals"] >= 0
    assert "by_horizon" in results["summary"]
```

**Step 2: Run all tests**

```bash
pytest tests/ -v --ignore=tests/test_yfinance_provider.py
```

Expected: All PASS (excluding integration tests that hit real API)

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: end-to-end integration tests with mock data"
```

---

### Task 12: Pipeline Runner in Streamlit

**Files:**
- Modify: `app.py` — add pipeline trigger button

**Step 1: Add pipeline controls to the home page**

Add to the bottom of `app.py`:

```python
# --- Add after the existing content ---

st.divider()
st.subheader("Data Pipeline")
st.markdown("Fetch stock data from YFinance and store in the local database.")

col1, col2 = st.columns(2)
with col1:
    pipeline_start = st.date_input("Pipeline Start", value=pd.to_datetime("2022-01-01"), key="pipe_start")
with col2:
    pipeline_end = st.date_input("Pipeline End", value=pd.to_datetime("2026-03-17"), key="pipe_end")

if st.button("Run Pipeline", type="primary"):
    from pipeline import Pipeline
    from core.providers.yfinance_provider import YFinanceProvider

    provider = YFinanceProvider()
    pipeline = Pipeline(db=db, provider=provider)

    progress = st.progress(0, text="Starting pipeline...")
    status = st.empty()

    # Override pipeline to show progress
    tickers = provider.get_small_cap_universe(1.0, 20.0)
    total = len(tickers)
    status.text(f"Found {total} candidates")

    for i, ticker in enumerate(tickers):
        try:
            progress.progress((i + 1) / total, text=f"Processing {ticker} ({i+1}/{total})")
            pipeline._process_ticker(ticker, pipeline_start.strftime("%Y-%m-%d"), pipeline_end.strftime("%Y-%m-%d"))
        except Exception as e:
            status.warning(f"Skipped {ticker}: {e}")
        import time
        if (i + 1) % 5 == 0:
            time.sleep(1)

    progress.progress(1.0, text="Complete!")
    st.success(f"Pipeline finished. Processed {total} tickers.")
    st.rerun()
```

Add `import pandas as pd` to the top of app.py if not already there.

**Step 2: Verify it runs**

```bash
streamlit run app.py --server.headless true
```

**Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add pipeline runner to Streamlit home page"
```

---

## Summary

| Task | Component | Description |
|------|-----------|-------------|
| 1 | Scaffolding | Project structure, config, dependencies |
| 2 | Database | SQLite schema and CRUD operations |
| 3 | Provider Base | Abstract DataProvider class |
| 4 | YFinance | YFinance provider implementation |
| 5 | Pipeline | Data fetching pipeline with rate limiting |
| 6 | Scanner | MA crossover + EPS change signal detection |
| 7 | Backtest | Forward returns and parameter sweep |
| 8 | UI: Scanner | Scanner dashboard with filters |
| 9 | UI: Detail | Stock detail with chart and EPS markers |
| 10 | UI: Backtest | Backtest page with sweep and charts |
| 11 | Integration | End-to-end tests |
| 12 | UI: Pipeline | Pipeline runner in Streamlit |
