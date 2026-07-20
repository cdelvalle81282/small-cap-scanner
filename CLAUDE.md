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
- `core/performance.py` — follow-through: current (mark-to-market) + forward returns at 15/30/60/90d per trigger, direction-aware, no stop
- `core/chart_analyzer.py` — Claude vision AI analysis via kaleido PNG export (model: claude-sonnet-5). Saved analyses auto-expire after 7 days (`get_ai_analysis` TTL)
- `api/main.py` — FastAPI JSON API (meta, signals, performance, ticker, news, watchlist, trades, analyze); also serves `web/`. **Both `/api/signals` and `/api/performance` are cached** (keyed on config + latest price date) and prewarmed on startup — `/api/performance` is the heaviest call, keep it cached + universe-constrained or the 502 returns (see memory `perf-endpoint-cost`)
- `/api/news/{sym}` — lazy Yahoo Finance headlines (top 4) via yfinance `.news`, 30-min in-memory cache. A fetch failure or an unparseable payload (likely a yfinance schema change) is **never swallowed**: it fires a throttled ops alert via `notify_ops` and returns a visible `error` to the UI. `YFinanceProvider.get_news` raises `NewsFetchError` on failure but returns `[]` for a genuinely empty feed
- `core/notifier.py` — `send_alert` (price-level; email/Slack still stubs) + `notify_ops(subject, message)` (wired to `SLACK_WEBHOOK_URL` / SMTP env, used for the news-feed failure alerts)
- `monitor.py` — daily watchlist price level checker
- `scan_and_notify.py` — daily new signal scanner, writes to signal_alerts table

## Frontend (web/ SPA — current)
Vanilla HTML/CSS/JS single-page app served by FastAPI, hash-routed (`web/js/views/`):
- **Overview** — KPI tiles, freshness, top signals by quality
- **Scanner** — filter rail, signal table (sparklines · RVOL · quality score), linked focus panel. Clicking a ticker cell opens Detail
- **Triggered** (route `performance`, nav label "Triggered") — track record of every trigger: current + 15/30/60/90d returns, aggregate cards, sortable columns, table capped to top 300 sorted rows
- **Detail** — lightweight-charts candles + SMA 20/50/200, diagonal swing-pivot trendlines, AI support/resistance + trend-break levels, EPS/cross markers, forward returns, AI analysis, latest-news panel (4 Yahoo headlines, click out to source). Chart opens on the recent ~240 bars (~1 trading year) so earnings labels stay legible; scroll/zoom back for full history
- **Tracking** — Alerts | Watchlist | Trades

Frontend gotchas:
- `createPriceLine` does NOT render in the standalone lightweight-charts build — draw horizontal levels and trendlines as `addLineSeries` instead (trendlines use `autoscaleInfoProvider: () => null` so they don't distort the candle scale). See memory `lightweight-charts-lines`.
- Detail chart: do NOT `fitContent()` over the full multi-year history — every earnings report is a labelled marker, so at full zoom the labels collide into an unreadable cluster. `drawChart` defaults the visible range to the last ~240 bars via `setVisibleLogicalRange`; SMA200 stays populated across it and the user can still scroll back. See memory `detail-chart-window`.
- Sortable tables use `.ft-tbl th{position:sticky;top:0}`, NOT `top:78px`. Each `.ft-tbl` lives inside its own `overflow:auto` wrapper (Triggered has `max-height:60vh`; Tracking tables get `overflow-y:auto` implicitly), i.e. its own scroll container — a topbar-height offset floats the header ~64px down into the table and hides the top rows. See memory `sticky-table-header`.

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
- Repo hygiene (gitignored): logs (`*.log`), backups (`*.bak`), `venv/`, `plan.md`, and the analysis-script chart artifacts (`signal_charts.html`, `losers_charts.png`, `winners_charts.png`, `signal_charts_sample.png`, `signal_detail.png`). Those 5 artifacts were also untracked from git (~6.8 MB, ~99% of the pack) — a `git filter-repo` history rewrite to purge them from past commits is documented but not yet run. `run_pipeline.sh` is droplet-only (gitignored, edit over SSH — see memory `pipeline-cron-polygon`)

## Cron jobs on droplet
- `35 13 * * 1-5` — scan_and_notify.py (9:35 AM ET)
- `0 1 * * 2-6` — run_pipeline.sh (9 PM ET incremental)
- `*/5 * * * *` — DuckDNS IP updater
