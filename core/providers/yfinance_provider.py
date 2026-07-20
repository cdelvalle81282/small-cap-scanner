import logging
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from core.providers.base import DataProvider, NewsFetchError

logger = logging.getLogger(__name__)


def _nested(d, *keys):
    """Safely walk nested dict keys, returning None if any hop is missing."""
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d

# Validated tickers ($1-$50, confirmed active March 2026)
# Covers biotech, tech, EV/energy, fintech, space, consumer, cannabis, ADRs
# get_small_cap_universe() dynamically filters by current price range
SMALL_CAP_UNIVERSE = [
    # ── Original universe (83 tickers, mostly $1-$20) ──
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
    # ── Expansion: ADRs (Chinese/Asian tech, LatAm, European) ──
    "ABEV", "BILI", "FINV", "GGB", "GOTU", "GRAB", "JD", "LI",
    "LSPD", "LX", "NIO", "NU", "PAGS", "PHI", "SID", "STNE",
    "TAL", "TEVA", "TIGR", "TME", "TUYA", "UGP", "VALE", "VNET",
    "VTRS", "WB", "XPEV", "YMM", "ZTO",
    # ── Expansion: US small/mid caps ($5-$50) ──
    "AFRM", "APPS", "ASAN", "BILL", "CARG", "CHWY", "CLSK", "COMP",
    "CRNC", "CRSP", "DKNG", "DOCS", "ENPH", "ENVX", "EVTL", "FLNC",
    "GDRX", "GRPN", "HIMS", "IONQ", "JOBY", "LCID", "MGNI", "MQ",
    "NNDM", "NRDS", "OUST", "PATH", "PERI", "PRCH", "PSNY", "RNW",
    "RPAY", "SDGR", "SEDG", "SHLS", "SPWR", "TALK", "TOST", "TPVG",
    "U", "UPST", "UPWK", "VLY", "WULF", "XERS", "XOS", "ZETA",
    # ── Expansion: Biotech/pharma ──
    "ACAD", "AGIO", "ALKS", "AMRX", "APLS", "ARVN", "BEAM", "CRBU",
    "DNLI", "EDIT", "FOLD", "LEGN", "NRIX", "NUVB", "PRTA", "RARE",
    "REPL", "RLAY", "ROIV", "TGTX", "VCNX",
    # ── Expansion: Clean energy / EV ──
    "AES", "BLDP",
    # ── Expansion: Screened US equities $1.2B-$2.0B market cap (March 2026) ──
    "AAMI", "ADNT", "ADUS", "AERO", "AESI", "AFYA", "AGBK", "AGM",
    "AGYS", "AHCO", "AIN", "ALEX", "ALG", "ALGT", "ALKT", "ALX",
    "AMLX", "AMRC", "AMSC", "ANAB", "ANIP", "AORT", "APC", "APPN",
    "ARCO", "ARDT", "ARDX", "ARI", "ARLO", "ARR", "ASGN", "ASTH",
    "ATAI", "ATEC", "ATEN", "ATKR", "ATRC", "AUPH", "AVAH", "AXGN",
    "BAK", "BANR", "BB", "BFC", "BHC", "BHE", "BHVN", "BITF",
    "BLBD", "BLX", "BOBS", "BORR", "BRAI", "BUR", "BVC", "BY",
    "CAPR", "CBZ", "CCS", "CDNL", "CDRE", "CHA", "CHCO", "CLBK",
    "CLDX", "CLVT", "CMPR", "CNOB", "CNXC", "CNXN", "COTY", "CRI",
    "CSWC", "CTOS", "CTS", "CXM", "CYD", "DCH", "DCO", "DCOM",
    "DEI", "DFH", "DFIN", "DFTX", "DGII", "DMLP", "DOLE", "DQ",
    "DRH", "DRVN", "DSGR", "DV", "ECO", "ECPG", "ECVT", "EDN",
    "EFC", "EFSC", "ELVN", "ENOV", "EOSE", "EPAC", "ESTA", "EVCM",
    "EVTC", "FA", "FCF", "FDP", "FIHL", "FLNG", "FLO", "FMC",
    "FOR", "FUN", "GABC", "GAM", "GBX", "GCMG", "GCT", "GENB",
    "GILT", "GLIBA", "GLIBK", "GLOB", "GLP", "GRAL", "GRC", "GSHD",
    "GSL", "GT", "GTM", "GTY", "HCSG", "HCXY", "HIMX", "HLF",
    "HLMN", "HLX", "HMN", "HOPE", "HRMY", "ICHR", "IIPR", "IMAX",
    "IMCR", "IMKTA", "IMOS", "IMTX", "INOD", "INSP", "INVA", "INVX",
    "IOSP", "IOVA", "IRMD", "IVA", "JJSF", "KARO", "KDK", "KMPR",
    "KOD", "KOS", "KRP", "KSS", "KW", "LC", "LEG", "LILA",
    "LILAK", "LKFN", "LOB", "LOMA", "LPG", "LTC", "LZB", "MANE",
    "MBIN", "MBX", "MCRI", "MD", "MESO", "MLYS", "MXL", "NBTX",
    "NCNO", "NEOG", "NEXA", "NEXT", "NGL", "NMM", "NOMD", "NRP",
    "NSSC", "NTLA", "NVAX", "NVCR", "NVGS", "NVRI", "NWBI", "NWL",
    "OBK", "OCS", "OCUL", "OFG", "OGN", "OI", "OLMA", "OMCL",
    "OPRA", "ORC", "ORKA", "PAX", "PBI", "PDFS", "PDS", "PENN",
    "PGNY", "PHVS", "PICS", "PLPC", "PLSE", "PLUS", "PRA", "PRCT",
    "PRGO", "PRGS", "PRKS", "PRLB", "PSIX", "PTON", "PUMP", "PVLA",
    "PWP", "QCRH", "QFIN", "QUBT", "RAMP", "RAPP", "RBCAA", "RCAT",
    "RES", "REX", "RHLD", "ROG", "RUM", "RVLV", "SBET", "SBH",
    "SEMR", "SFL", "SGRY", "SHO", "SION", "SKWD", "SKYT", "SLNO",
    "SLVM", "SNDA", "SONO", "SPB", "SPH", "SRCE", "SRPT", "STBA",
    "STC", "STEL", "STOK", "STRA", "SVV", "SYBT", "TCBK", "TE",
    "TFIN", "TGLS", "THR", "TIC", "TILE", "TNDM", "TNET", "TPB",
    "TRS", "TRVI", "TSHA", "TSLX", "TV", "TY", "UAMY", "UAN",
    "UFPT", "UNIT", "UVV", "VCEL", "VECO", "VIR", "VOYG", "VRE",
    "VTOL", "WD", "WEN", "WERN", "WINA", "WKC", "WLFC", "WLY",
    "WLYB", "WMK", "WS", "WT", "WTTR", "WWW", "XHR", "XPRO",
    "XZO", "YELP", "ZBIO", "ZD", "ZYME",
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

    def get_news(self, ticker: str, limit: int = 4) -> list[dict]:
        """Return up to `limit` most-recent Yahoo Finance news items for a ticker.

        Each item: {title, url, publisher, published}. An empty list means the
        ticker legitimately has no news, a normal, quiet outcome.

        Unlike the other fetchers here, this does NOT swallow errors into an empty
        result: a fetch failure, a non-list payload, or a non-empty payload we
        can't parse (the likely symptom of a yfinance schema change) all raise
        NewsFetchError so the caller can alert. Silent blanks would hide breakage.
        """
        try:
            raw = yf.Ticker(ticker).news
        except Exception as e:
            raise NewsFetchError(f"yfinance .news raised for {ticker}: {e!r}") from e

        if raw is None:
            return []
        if not isinstance(raw, list):
            raise NewsFetchError(
                f"yfinance .news returned {type(raw).__name__} (expected list) for {ticker}"
            )
        if not raw:
            return []

        items: list[dict] = []
        for entry in raw:
            # yfinance 1.x nests fields under "content"; older versions were flat.
            c = entry.get("content", entry) if isinstance(entry, dict) else {}
            title = c.get("title") or (entry.get("title") if isinstance(entry, dict) else None)
            url = (
                _nested(c, "clickThroughUrl", "url")
                or _nested(c, "canonicalUrl", "url")
                or (entry.get("link") if isinstance(entry, dict) else None)
            )
            if not title or not url:
                continue

            published = c.get("pubDate") or (
                entry.get("providerPublishTime") if isinstance(entry, dict) else None
            )
            if isinstance(published, (int, float)):  # old schema: epoch seconds
                published = datetime.fromtimestamp(published, tz=timezone.utc).isoformat()

            items.append({
                "title": str(title),
                "url": str(url),
                "publisher": _nested(c, "provider", "displayName")
                or (entry.get("publisher") if isinstance(entry, dict) else "")
                or "",
                "published": published or "",
            })
            if len(items) >= limit:
                break

        # Non-empty payload but nothing parseable -> the shape almost certainly
        # changed upstream. Treat as a failure so it gets reported, not blanked.
        if not items:
            raise NewsFetchError(
                f"yfinance returned {len(raw)} news item(s) for {ticker} but none had a "
                f"title+url in the expected shape (schema change?)"
            )
        return items
