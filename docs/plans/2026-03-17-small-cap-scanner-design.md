# Small Cap EPS + Trend Scanner — Design Document

## Overview

A stock scanner that identifies small cap stocks ($1–$20) where a moving average trend change occurs within a configurable window of an earnings-per-share change. Includes backtesting for signal validation and a Streamlit UI for interactive exploration.

## Architecture

**Approach: Pipeline + SQLite + Streamlit**

```
[YFinance API] → [Data Pipeline] → [SQLite DB] → [Scanner Engine] → [Streamlit UI]
                                         ↑                              ↓
                                    [Backtest Module] ←────────────────┘
```

Future: `BloombergProvider` replaces `YFinanceProvider` with no changes to scanner/backtest/UI.

## Data Layer

### Data Provider Abstraction

Abstract `DataProvider` base class with methods:
- `get_price_history(ticker, start, end)`
- `get_earnings(ticker)`
- `get_small_cap_universe(min_price, max_price)`
- `get_fundamentals(ticker)`
- `get_stock_info(ticker)` (float, short interest)

`YFinanceProvider` is the initial implementation. Future `BloombergProvider` swaps in via config.

### SQLite Schema

**stocks**: ticker, name, market_cap, sector, shares_float, short_interest_pct, short_ratio, last_updated

**daily_prices**: ticker, date, open, high, low, close, volume, avg_volume_20d, relative_volume

**earnings**: ticker, report_date, period, eps_actual, eps_prior, eps_change_pct

**fundamentals**: ticker, period, revenue, gross_margin, operating_margin

**scan_results**: ticker, scan_date, signal_type, ma_period, eps_change_pct, trend_change_date, eps_change_date, days_between

### Data Pipeline

- Fetches stock universe, price history, earnings, and fundamentals from YFinance
- Stores in SQLite with incremental updates (only fetch new data since last run)
- Handles rate limiting with batching and delays
- Can be run manually via CLI or from the Streamlit UI

## Scanner Engine

### Signal Logic

1. **Universe filter**: Price $1–$20, small cap market cap range (configurable)
2. **EPS change detection**: EPS changed by more than X% (configurable threshold, default 10%) in a recent earnings report
3. **Trend change detection**: Price crossed above or below a configurable MA (20/50/200 SMA) within N days (configurable, default 30) of the EPS report date
4. **Direction**: Configurable — bullish only, bearish only, or both

### Output per match

ticker, EPS change %, EPS report date, MA period crossed, cross direction, cross date, days between EPS and cross

## Backtesting

### Signal Validation

- Run scanner historically over configurable date range (e.g., 2–5 years)
- For each historical signal, measure forward returns at 5, 10, 20, 30, 60 days
- Metrics: win rate, average return, median return, max drawdown
- Group by: MA period, EPS threshold, direction, sector

### Parameter Sweep

- Test combinations: MA periods (20/50/200) x EPS thresholds (5%/10%/25%/50%) x time windows (15/30/45 days)
- Output comparison table: parameter combo → win rate, avg return, sample size

Phase 2 (future): Full trade simulation with entry/exit rules, position sizing, stop losses.

## Streamlit UI

### Page 1: Scanner Dashboard

- **Sidebar**: Price range, MA period, EPS threshold, time window, direction filter
- **Results table**: Sortable — ticker, EPS change %, MA crossed, signal date, days between
- **Row click**: Navigates to stock detail page

### Page 2: Stock Detail

- **Price chart (1 year)**: Line/candlestick with 20/50/200 MA overlays
- **EPS markers**: Vertical lines on chart at earnings dates, annotated with EPS change %
- **Signal highlight**: The MA cross that triggered the scanner, highlighted
- **Stats sidebar**: Float, short interest, revenue, margins, volume metrics

### Page 3: Backtest

- **Config**: Date range, parameter selections
- **Results table**: Parameter combo → win rate, avg return, sample size
- **Charts**: Return distribution, equity curve for selected parameters
- **Drill-down**: Click into individual historical signals

## Tech Stack

- Python 3.11+
- yfinance (data provider)
- pandas (data manipulation)
- SQLite (storage)
- Streamlit (UI)
- plotly (charts)

## Project Structure

```
small_cap_scanner/
├── app.py                  # Streamlit entry point
├── pages/
│   ├── scanner.py          # Scanner dashboard page
│   ├── stock_detail.py     # Stock detail page
│   └── backtest.py         # Backtest page
├── core/
│   ├── providers/
│   │   ├── base.py         # Abstract DataProvider
│   │   └── yfinance_provider.py
│   ├── scanner.py          # Scanner engine
│   ├── backtest.py         # Backtest module
│   └── database.py         # SQLite operations
├── pipeline.py             # Data pipeline CLI
├── config.py               # Default settings
├── requirements.txt
└── docs/plans/
```
