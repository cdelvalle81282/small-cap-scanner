import logging
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from core.providers.base import DataProvider
from core.providers.yfinance_provider import SMALL_CAP_UNIVERSE

logger = logging.getLogger(__name__)

_BASE = "https://api.polygon.io"
_SNAPSHOT_BATCH = 200  # tickers per snapshot request


class PolygonProvider(DataProvider):
    """DataProvider backed by the Polygon.io REST API.

    EPS signal: year-over-year actual EPS change (Polygon has no analyst estimates).
    This differs from YFinanceProvider which uses analyst-surprise when available.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY env var not set and no key passed")
        self._session = requests.Session()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get(self, path: str, params: dict | None = None) -> dict:
        p = dict(params or {})
        p["apiKey"] = self.api_key
        resp = self._session.get(f"{_BASE}{path}", params=p, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _paginate(self, path: str, params: dict | None = None, max_pages: int = 5) -> list:
        """Follow next_url pagination, returning combined results list."""
        results = []
        p = dict(params or {})
        p["apiKey"] = self.api_key
        url = f"{_BASE}{path}"
        for _ in range(max_pages):
            resp = self._session.get(url, params=p, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            results.extend(data.get("results") or [])
            next_url = data.get("next_url")
            if not next_url:
                break
            # next_url already includes the apiKey param
            url = next_url
            p = {}
        return results

    # ── DataProvider interface ────────────────────────────────────────────────

    def get_price_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        empty = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        try:
            path = f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
            data = self._get(path, {"adjusted": "true", "sort": "asc", "limit": 50000})
            results = data.get("results") or []
            if not results:
                return empty
            rows = []
            for bar in results:
                ts_ms = bar.get("t")
                if ts_ms is None:
                    continue
                dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                rows.append({
                    "date":   dt.strftime("%Y-%m-%d"),
                    "open":   bar.get("o"),
                    "high":   bar.get("h"),
                    "low":    bar.get("l"),
                    "close":  bar.get("c"),
                    "volume": bar.get("v"),
                })
            return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
        except Exception:
            logger.exception("get_price_history failed for %s", ticker)
            return empty

    def get_earnings(self, ticker: str) -> pd.DataFrame:
        """Returns quarterly EPS with YoY change (actual vs same quarter prior year)."""
        empty = pd.DataFrame(
            columns=["report_date", "period", "eps_actual", "eps_prior", "eps_change_pct"]
        )
        try:
            results = self._paginate(
                "/vX/reference/financials",
                {"ticker": ticker, "timeframe": "quarterly", "order": "asc", "limit": 100},
            )
            if not results:
                return empty

            rows = []
            for r in results:
                inc = r.get("financials", {}).get("income_statement", {})
                eps_val = (
                    inc.get("diluted_earnings_per_share") or
                    inc.get("basic_earnings_per_share")
                )
                if eps_val is None:
                    continue
                eps_actual = eps_val.get("value")
                if eps_actual is None:
                    continue
                # Q4 is filed in annual 10-K; filing_date may be None — fall back to end_date
                report_date = r.get("filing_date") or r.get("end_date")
                rows.append({
                    "report_date": report_date,
                    "fiscal_year": r.get("fiscal_year"),
                    "fiscal_period": r.get("fiscal_period"),
                    "period": f"{r.get('fiscal_year')} {r.get('fiscal_period')}",
                    "eps_actual": float(eps_actual),
                })

            if not rows:
                return empty

            df = pd.DataFrame(rows).sort_values("report_date").reset_index(drop=True)

            # YoY: match each row to the same fiscal_period from the prior fiscal_year
            prior_map = {
                (r["fiscal_year"], r["fiscal_period"]): r["eps_actual"]
                for r in rows
            }

            def yoy_prior(row):
                try:
                    return prior_map.get((str(int(row["fiscal_year"]) - 1), row["fiscal_period"]))
                except (ValueError, TypeError):
                    return None

            df["eps_prior"] = df.apply(yoy_prior, axis=1)
            df["eps_change_pct"] = df.apply(
                lambda r: (
                    round((r["eps_actual"] - r["eps_prior"]) / abs(r["eps_prior"]) * 100, 2)
                    if pd.notna(r.get("eps_prior")) and r.get("eps_prior") != 0
                    else None
                ),
                axis=1,
            )

            keep = ["report_date", "period", "eps_actual", "eps_prior", "eps_change_pct"]
            return df[[c for c in keep if c in df.columns]]
        except Exception:
            logger.exception("get_earnings failed for %s", ticker)
            return empty

    def get_small_cap_universe(self, min_price: float, max_price: float) -> list[str]:
        """Filter the hardcoded universe to tickers currently within the price range."""
        candidates = list(SMALL_CAP_UNIVERSE)
        try:
            valid = []
            for i in range(0, len(candidates), _SNAPSHOT_BATCH):
                batch = candidates[i : i + _SNAPSHOT_BATCH]
                data = self._get(
                    "/v2/snapshot/locale/us/markets/stocks/tickers",
                    {"tickers": ",".join(batch)},
                )
                for snap in data.get("tickers") or []:
                    ticker = snap.get("ticker")
                    # Prefer today's close; fall back to previous close
                    day = snap.get("day") or {}
                    prev = snap.get("prevDay") or {}
                    price = day.get("c") or prev.get("c")
                    if price and min_price <= price <= max_price:
                        valid.append(ticker)
                time.sleep(0.1)  # be polite between batches

            if valid:
                logger.info(
                    "Price-filtered universe: %d/%d tickers in $%.0f-$%.0f range",
                    len(valid), len(candidates), min_price, max_price,
                )
                return valid

        except Exception:
            logger.exception("Snapshot price filter failed, returning full universe")

        return candidates

    def get_fundamentals(self, ticker: str) -> pd.DataFrame:
        empty = pd.DataFrame(columns=["period", "revenue", "gross_margin", "operating_margin"])
        try:
            results = self._paginate(
                "/vX/reference/financials",
                {"ticker": ticker, "timeframe": "quarterly", "order": "asc", "limit": 100},
            )
            rows = []
            for r in results:
                inc = r.get("financials", {}).get("income_statement", {})
                rev_obj = inc.get("revenues") or inc.get("net_revenues") or {}
                gp_obj  = inc.get("gross_profit", {})
                oi_obj  = inc.get("operating_income_loss") or inc.get("operating_income", {})

                revenue = rev_obj.get("value") if isinstance(rev_obj, dict) else None
                if not revenue:
                    continue

                gross_profit     = gp_obj.get("value") if isinstance(gp_obj, dict) else None
                operating_income = oi_obj.get("value") if isinstance(oi_obj, dict) else None

                rows.append({
                    "period": f"{r.get('fiscal_year')} {r.get('fiscal_period')}",
                    "revenue": float(revenue),
                    "gross_margin": (
                        round(float(gross_profit) / float(revenue) * 100, 2)
                        if gross_profit is not None else None
                    ),
                    "operating_margin": (
                        round(float(operating_income) / float(revenue) * 100, 2)
                        if operating_income is not None else None
                    ),
                })

            return pd.DataFrame(rows) if rows else empty
        except Exception:
            logger.exception("get_fundamentals failed for %s", ticker)
            return empty

    def get_stock_info(self, ticker: str) -> dict:
        empty: dict = {
            "name": None, "market_cap": None, "sector": None,
            "shares_float": None, "short_interest_pct": None, "short_ratio": None,
        }
        try:
            data = self._get(f"/v3/reference/tickers/{ticker}")
            result = data.get("results") or {}
            return {
                "name":               result.get("name"),
                "market_cap":         result.get("market_cap"),
                "sector":             result.get("sic_description"),
                # Polygon does not provide short interest data
                "shares_float":       None,
                "short_interest_pct": None,
                "short_ratio":        None,
            }
        except Exception:
            logger.exception("get_stock_info failed for %s", ticker)
            return empty
