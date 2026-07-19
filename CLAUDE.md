# Small Cap Scanner — Claude Instructions

## What this project is
EPS + MA crossover scanner for small/mid-cap stocks ($1–$50). Web app — FastAPI backend + vanilla-JS single-page frontend — with signal quality scoring, follow-through analytics, AI chart analysis, watchlist monitoring, and trade tracking. Deployed on DigitalOcean. The original Streamlit UI (`app.py` / `pages/`) is retired; kept in-repo for reference only.

## Deployment
- **Live:** https://smallcapepsscan.duckdns.org
- **Server:** deploy@167.99.167.244 — /home/deploy/small-cap-scanner
- **Deploy sequence:** `git push origin master` → `ssh deploy@167.99.167.244 "cd ~/small-cap-scanner && git pull && sudo systemctl restart small-cap-web"`
- **Services (systemd):** `small-cap-web` = FastAPI/uvicorn on :8600 (current, serves `web/` + `/api`). nginx (basic-auth via `/etc/nginx/.htpasswd`, user `editorial`) proxies :8600. `small-cap-scanner` = old Streamlit on :8503, **disabled**. Roll back: flip nginx `proxy_pass` to 8503 (backup at `/etc/nginx/sites-available/small-cap-scanner.bak`) + re-enable the service.
- **Never** reference Streamlit Community Cloud — we use DigitalOcean

## Architecture
```
YFinance/Polygon → pipeline.py → SQLite (data/scanner.db) → core/ (Scanner·Backtest·scoring·performance·AI) → FastAPI (api/) → web/ SPA
```
- `pipeline.py` — fetches price + earnings + fundamentals, supports --provider yfinance|polygon, --min-cap, --max-cap
- `core/database.py` — single Database class, all tables in one SQLite file
- `core/providers/` — DataProvider ABC, YFinanceProvider, PolygonProvider
- `core/scoring.py` — signal quality score (0-100 + factor breakdown)
- `core/performance.py` — follow-through: forward returns at 15/30/60/90d
- `core/chart_analyzer.py` — Claude vision AI analysis via kaleido PNG export (model: claude-sonnet-5)
- `api/main.py` — FastAPI JSON API (meta, signals, performance, ticker, watchlist, trades, analyze); also serves `web/`
- `core/notifier.py` — alert dispatch stubs (email/Slack not yet wired)
- `monitor.py` — daily watchlist price level checker
- `scan_and_notify.py` — daily new signal scanner, writes to signal_alerts table

## Frontend (web/ SPA — current)
Vanilla HTML/CSS/JS single-page app served by FastAPI, hash-routed (`web/js/views/`):
- **Overview** — KPI tiles, freshness, top signals by quality
- **Scanner** — filter rail, signal table (sparklines · RVOL · quality score), linked focus panel
- **Follow-Through** — realized returns at 15/30/60/90d per trigger + aggregates
- **Detail** — lightweight-charts candles + SMA, forward returns, AI analysis
- **Tracking** — Alerts | Watchlist | Trades

Legacy Streamlit (`app.py`, `pages/1_Scanner.py` … `pages/5_Tracking.py`) is retired but kept in-repo for reference.

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
