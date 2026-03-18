import logging

import pandas as pd
import yfinance as yf

from core.providers.base import DataProvider

logger = logging.getLogger(__name__)

FALLBACK_SMALL_CAPS = [
    "ACMR", "AEHR", "ALEC", "APTO", "AREC",
    "AULT", "BSFC", "BTBT", "CLOV", "COMS",
    "DARE", "DPRO", "EKSO", "ELOX", "EOLS",
    "GFAI", "GOED", "HCDI", "HPNN", "IMVT",
    "JBDI", "KNDI", "LFLY", "LPSN", "MFIN",
    "MVST", "NKLA", "OCGN", "OPAD", "PRST",
]


class YFinanceProvider(DataProvider):
    def get_price_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        try:
            t = yf.Ticker(ticker)
            df = t.history(start=start, end=end)
            if df.empty:
                return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            # Normalize the date column name — yfinance uses "Date" or "Datetime"
            for candidate in ("datetime", "date"):
                if candidate in df.columns:
                    df = df.rename(columns={candidate: "date"})
                    break
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            keep = ["date", "open", "high", "low", "close", "volume"]
            return df[[c for c in keep if c in df.columns]]
        except Exception:
            logger.exception("get_price_history failed for %s", ticker)
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    def get_earnings(self, ticker: str) -> pd.DataFrame:
        empty = pd.DataFrame(
            columns=["report_date", "period", "eps_actual", "eps_prior", "eps_change_pct"]
        )
        try:
            t = yf.Ticker(ticker)
            df = None

            # Try earnings_history first (newer yfinance versions)
            hist = getattr(t, "earnings_history", None)
            if hist is not None and not hist.empty:
                df = hist.copy()

            if df is None or df.empty:
                return empty

            df.columns = [c.lower().replace(" ", "_") for c in df.columns]

            # Map column names from yfinance to our schema
            col_map = {
                "epsactual": "eps_actual",
                "epsestimate": "eps_estimate",
                "epsdifference": "eps_difference",
                "surprisepercent": "surprise_pct",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

            if "eps_actual" not in df.columns:
                return empty

            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]

            # Determine report_date column
            for candidate in ("quarter", "date", "earningsdate"):
                if candidate in df.columns:
                    df = df.rename(columns={candidate: "report_date"})
                    break

            if "report_date" not in df.columns:
                df["report_date"] = None

            df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce").dt.strftime(
                "%Y-%m-%d"
            )

            if "period" not in df.columns:
                df["period"] = df.get("report_date", None)

            # Compute eps_prior (shift eps_actual by 4 quarters = 1 year prior)
            df = df.sort_values("report_date")
            df["eps_prior"] = df["eps_actual"].shift(4)
            df["eps_change_pct"] = df.apply(
                lambda r: (
                    (r["eps_actual"] - r["eps_prior"]) / abs(r["eps_prior"]) * 100
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
        try:
            screener = yf.Screener()
            screener.set_predefined_body("small_cap_gainers")
            result = screener.response
            quotes = result.get("quotes", [])
            tickers = [
                q["symbol"]
                for q in quotes
                if min_price <= q.get("regularMarketPrice", 0) <= max_price
            ]
            if tickers:
                return tickers
        except Exception:
            logger.warning("YFinance screener failed, using fallback list")
        return list(FALLBACK_SMALL_CAPS)

    def get_fundamentals(self, ticker: str) -> pd.DataFrame:
        empty = pd.DataFrame(
            columns=["period", "revenue", "gross_margin", "operating_margin"]
        )
        try:
            t = yf.Ticker(ticker)
            fin = t.quarterly_financials
            if fin is None or fin.empty:
                return empty

            rows = []
            for col in fin.columns:
                period = str(col)[:10]
                revenue = fin[col].get("Total Revenue")
                gross_profit = fin[col].get("Gross Profit")
                operating_income = fin[col].get("Operating Income")

                if revenue is None or pd.isna(revenue) or revenue == 0:
                    continue

                gross_margin = (
                    float(gross_profit) / float(revenue) * 100
                    if gross_profit is not None and not pd.isna(gross_profit)
                    else None
                )
                operating_margin = (
                    float(operating_income) / float(revenue) * 100
                    if operating_income is not None and not pd.isna(operating_income)
                    else None
                )
                rows.append(
                    {
                        "period": period,
                        "revenue": float(revenue),
                        "gross_margin": gross_margin,
                        "operating_margin": operating_margin,
                    }
                )

            if not rows:
                return empty
            return pd.DataFrame(rows, columns=["period", "revenue", "gross_margin", "operating_margin"])
        except Exception:
            logger.exception("get_fundamentals failed for %s", ticker)
            return empty

    def get_stock_info(self, ticker: str) -> dict:
        empty: dict = {
            "name": None,
            "market_cap": None,
            "sector": None,
            "shares_float": None,
            "short_interest_pct": None,
            "short_ratio": None,
        }
        try:
            info = yf.Ticker(ticker).info
            short_pct_float = info.get("shortPercentOfFloat")
            return {
                "name": info.get("longName") or info.get("shortName"),
                "market_cap": info.get("marketCap"),
                "sector": info.get("sector"),
                "shares_float": info.get("floatShares"),
                "short_interest_pct": (
                    float(short_pct_float) * 100
                    if short_pct_float is not None
                    else None
                ),
                "short_ratio": info.get("shortRatio"),
            }
        except Exception:
            logger.exception("get_stock_info failed for %s", ticker)
            return empty
