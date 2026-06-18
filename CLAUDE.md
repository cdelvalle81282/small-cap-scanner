# Small Cap Scanner — Claude Instructions

## What this project is
EPS + MA crossover scanner for small/mid-cap stocks ($1–$50). Streamlit UI with AI chart analysis, watchlist monitoring, and trade tracking. Deployed on DigitalOcean.

## Deployment
- **Live:** https://smallcapepsscan.duckdns.org
- **Server:** deploy@167.99.167.244 — /home/deploy/small-cap-scanner
- **Deploy sequence:** `git push origin master` → `ssh deploy@167.99.167.244 "cd ~/small-cap-scanner && git pull && sudo systemctl restart small-cap-scanner"`
- **Never** reference Streamlit Community Cloud — we use DigitalOcean

## Architecture
```
YFinance/Polygon → pipeline.py → SQLite (data/scanner.db) → Scanner/Backtest → Streamlit UI
```
- `pipeline.py` — fetches price + earnings + fundamentals, supports --provider yfinance|polygon, --min-cap, --max-cap
- `core/database.py` — single Database class, all tables in one SQLite file
- `core/providers/` — DataProvider ABC, YFinanceProvider, PolygonProvider
- `core/chart_analyzer.py` — Claude vision AI analysis via kaleido PNG export
- `core/notifier.py` — alert dispatch stubs (email/Slack not yet wired)
- `monitor.py` — daily watchlist price level checker
- `scan_and_notify.py` — daily new signal scanner, writes to signal_alerts table

## Pages
1. `app.py` — home, pipeline runner (provider/price/market cap config)
2. `pages/1_Scanner.py` — scan with optional filters, sort, recency filter, URL-param persistence
3. `pages/2_Stock_Detail.py` — chart with AI overlay, AI analysis, watchlist button
4. `pages/3_Backtest.py` — single + sweep backtest
5. `pages/4_Methodology.py` — docs
6. `pages/5_Tracking.py` — 3 tabs: Alerts | Watchlist | Trades

## Database tables
stocks, daily_prices, earnings, fundamentals, scan_results,
signal_watchlist, price_alerts, signal_alerts, trades

## API Keys (on droplet in /home/deploy/small-cap-scanner/.env)
- ANTHROPIC_API_KEY — AI chart analysis
- POLYGON_API_KEY — dynamic universe discovery (also in ~/orb-finder/.env)

## Key conventions
- Tests: `python -m pytest tests/ -q --ignore=tests/test_yfinance_provider.py` — must stay green
- Always run tests before committing
- Commit message style: `feat:`, `fix:`, `refactor:` prefix
- No comments unless the WHY is non-obvious
- Kaleido requires Chrome: installed at /home/deploy/.local/share/choreographer/deps/chrome-linux64/chrome

## Cron jobs on droplet
- `35 13 * * 1-5` — scan_and_notify.py (9:35 AM ET)
- `0 1 * * 2-6` — run_pipeline.sh (9 PM ET incremental)
- `*/5 * * * *` — DuckDNS IP updater
