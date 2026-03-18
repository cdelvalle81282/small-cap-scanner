import logging

import pandas as pd
import yfinance as yf

from core.providers.base import DataProvider

logger = logging.getLogger(__name__)

# Validated tickers ($1-$20, confirmed active March 2026)
# Covers biotech, tech, EV/energy, fintech, space, consumer, cannabis
# get_small_cap_universe() dynamically filters by current price range
SMALL_CAP_UNIVERSE = [
    "ACB", "AGEN", "AI", "ALEC", "ARBK", "AREC", "ARRY", "BBAI",
    "BCRX", "BIRD", "BMBL", "BRZE", "BTBT", "CGC", "CHPT", "CLF",
    "CLOV", "COUR", "CRMD", "DARE", "DNA", "DPRO", "EOLS", "EVGO",
    "FATE", "FCEL", "FIGS", "FLYW", "FRSH", "GENI", "GERN", "HYLN",
    "IBRX", "IQ", "KNDI", "LPSN", "LUNR", "MARA", "MAXN", "MFIN",
    "MNKD", "MNTS", "MVST", "NAVI", "NCMI", "OCGN", "OPEN", "PAYO",
    "PLBY", "PLUG", "PSFE", "PUBM", "QBTS", "QS", "RCKT", "RDW",
    "RENT", "RGNX", "RGTI", "RIOT", "RKT", "RUN", "RXRX", "SATL",
    "SAVA", "SKIN", "SKLZ", "SMPL", "SNDL", "SOFI", "SPCE", "SPT",
    "STEM", "TASK", "TLRY", "UWMC", "VUZI", "WEAV", "WKHS", "XNCR",
    "XPOF", "YEXT", "ZNTL",
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

            # Use eps_estimate (analyst consensus) as the prior for computing EPS surprise.
            # yfinance only returns ~4 quarters so shift(4) for YoY won't work.
            # Earnings surprise vs estimate is the more meaningful signal anyway.
            df = df.sort_values("report_date")
            if "eps_estimate" in df.columns:
                df["eps_prior"] = df["eps_estimate"]
            else:
                # Fallback: quarter-over-quarter comparison
                df["eps_prior"] = df["eps_actual"].shift(1)
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
        """Return tickers from our universe that currently trade within the price range.

        Batch-downloads recent prices to filter, so only one API call is needed.
        """
        candidates = list(SMALL_CAP_UNIVERSE)
        try:
            import pandas as pd

            data = yf.download(candidates, period="5d", progress=False, threads=True)
            if data.empty:
                logger.warning("Batch download returned no data, returning full universe")
                return candidates

            # Handle both single-ticker and multi-ticker DataFrame shapes
            if isinstance(data.columns, pd.MultiIndex):
                last_prices = data["Close"].iloc[-1]
            else:
                last_prices = data["Close"].iloc[-1:]

            valid = []
            for ticker in candidates:
                try:
                    price = (
                        last_prices[ticker]
                        if isinstance(last_prices, pd.Series)
                        else float(last_prices.iloc[0])
                    )
                    if pd.notna(price) and min_price <= price <= max_price:
                        valid.append(ticker)
                except (KeyError, IndexError):
                    continue

            if valid:
                logger.info("Price-filtered universe: %d/%d tickers in $%.0f-$%.0f range",
                            len(valid), len(candidates), min_price, max_price)
                return valid

        except Exception:
            logger.exception("Batch price filter failed, returning full universe")

        return candidates

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
