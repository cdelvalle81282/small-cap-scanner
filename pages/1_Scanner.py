import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DB_PATH, ScannerConfig
from core.database import Database
from core.scanner import Scanner

st.set_page_config(page_title="Scanner", page_icon="🔍", layout="wide")
from core.styles import inject_styles
inject_styles()


@st.cache_resource
def get_db() -> Database:
    db = Database(DB_PATH)
    db.initialize()
    return db


db = get_db()

st.title("Small Cap Scanner")

MA_PAIR_OPTIONS = {
    "20/50": (20, 50),
    "50/200": (50, 200),
}

SORT_OPTIONS = {
    "Cross Date (newest)": ("trend_change_date", True),
    "Avg Volume (highest)": ("avg_volume", True),
    "Avg $ Volume (highest)": ("avg_dollar_vol", True),
    "Market Cap (highest)": ("market_cap", True),
    "Market Cap (lowest)": ("market_cap", False),
    "Price (highest)": ("latest_close", True),
    "Price (lowest)": ("latest_close", False),
    "EPS Change % (highest)": ("eps_change_pct", True),
}


def parse_optional_float(val: str) -> float | None:
    val = val.strip()
    if not val:
        return None
    try:
        return float(val.replace(",", ""))
    except ValueError:
        return None


def parse_optional_market_cap(val: str) -> float | None:
    val = val.strip().upper().replace(",", "")
    if not val:
        return None
    try:
        if val.endswith("B"):
            return float(val[:-1]) * 1_000_000_000
        if val.endswith("M"):
            return float(val[:-1]) * 1_000_000
        return float(val)
    except ValueError:
        return None


# --- Restore params from URL if navigating back ---
qp = st.query_params

# Sidebar controls
with st.sidebar:
    st.header("Scan Parameters")

    st.caption("Price Filter (leave blank for no limit)")
    col_min, col_max = st.columns(2)
    with col_min:
        min_price_str = st.text_input("Min $", value=qp.get("min_p", ""), placeholder="e.g. 1")
    with col_max:
        max_price_str = st.text_input("Max $", value=qp.get("max_p", ""), placeholder="e.g. 50")

    st.caption("Market Cap Filter (leave blank for no limit)")
    col_mmin, col_mmax = st.columns(2)
    with col_mmin:
        min_cap_str = st.text_input("Min Cap", value=qp.get("min_cap", ""), placeholder="e.g. 50M")
    with col_mmax:
        max_cap_str = st.text_input("Max Cap", value=qp.get("max_cap", ""), placeholder="e.g. 10B")

    ma_pair_label = st.selectbox(
        "MA Crossover",
        options=list(MA_PAIR_OPTIONS.keys()),
        index=list(MA_PAIR_OPTIONS.keys()).index(qp.get("ma", "20/50")) if qp.get("ma") in MA_PAIR_OPTIONS else 0,
    )
    eps_threshold = st.slider("Min EPS Change %", min_value=1, max_value=100, value=int(qp.get("eps", 10)))
    trend_window = st.slider("Trend Window (days)", min_value=5, max_value=90, value=int(qp.get("window", 30)))
    direction = st.selectbox(
        "Signal Direction",
        options=["both", "bullish", "bearish"],
        index=["both", "bullish", "bearish"].index(qp.get("dir", "both")) if qp.get("dir") in ["both", "bullish", "bearish"] else 0,
    )
    max_days_old = st.slider(
        "Max days since crossover",
        min_value=1, max_value=365,
        value=int(qp.get("recency", 30)),
        help="Only show signals where the MA crossover happened within this many days",
    )
    st.divider()
    sort_label = st.selectbox("Sort By", options=list(SORT_OPTIONS.keys()), index=0)
    st.divider()
    run_scan = st.button("Run Scanner", type="primary", use_container_width=True)

# Parse optional filters
min_price = parse_optional_float(min_price_str)
max_price = parse_optional_float(max_price_str)
min_cap   = parse_optional_market_cap(min_cap_str)
max_cap   = parse_optional_market_cap(max_cap_str)

# Validation
errors = []
if min_price_str.strip() and min_price is None:
    errors.append("Min Price is not a valid number.")
if max_price_str.strip() and max_price is None:
    errors.append("Max Price is not a valid number.")
if min_cap_str.strip() and min_cap is None:
    errors.append("Min Cap is not a valid number (use e.g. 50M or 2B).")
if max_cap_str.strip() and max_cap is None:
    errors.append("Max Cap is not a valid number (use e.g. 50M or 2B).")
for e in errors:
    st.warning(e)

def _db_alert_to_signal(a: dict) -> dict:
    """Convert a signal_alerts DB row back to scanner signal format."""
    return {
        "ticker": a["ticker"],
        "scan_date": a.get("alert_date", ""),
        "signal_type": a["signal_type"],
        "fast_ma": a.get("fast_ma", 20),
        "slow_ma": a.get("slow_ma", 50),
        "eps_change_pct": a.get("eps_change_pct"),
        "trend_change_date": a.get("cross_date", ""),
        "eps_change_date": a.get("eps_date", ""),
        "days_between": a.get("days_between"),
    }


def _enrich(signals: list[dict]) -> list[dict]:
    tickers = [r["ticker"] for r in signals]
    enrichment = db.get_signal_enrichment(tickers)
    for r in signals:
        snap = enrichment.get(r["ticker"], {})
        r["market_cap"] = snap.get("market_cap")
        r["latest_close"] = snap.get("latest_close")
        r["avg_dollar_vol"] = snap.get("avg_dollar_vol")
        r["avg_volume"] = snap.get("avg_volume")
    return signals


def _merge_into(base: dict[str, dict], incoming: list[dict]) -> dict[str, dict]:
    """Merge incoming signals into base dict keyed by ticker, keeping most recent crossover."""
    for sig in incoming:
        t = sig["ticker"]
        if not sig.get("trend_change_date"):
            continue
        if t not in base or sig["trend_change_date"] > base[t]["trend_change_date"]:
            base[t] = sig
    return base


# ── Load persisted signals from DB on every page open ────────────────────────
today = date.today()
cutoff = (today - timedelta(days=max_days_old)).isoformat()
db_alerts = db.get_signal_alerts(cutoff, today.isoformat())
db_signals: dict[str, dict] = {}
if db_alerts:
    for a in db_alerts:
        sig = _db_alert_to_signal(a)
        t = sig["ticker"]
        if not sig["trend_change_date"]:
            continue
        if t not in db_signals or sig["trend_change_date"] > db_signals[t]["trend_change_date"]:
            db_signals[t] = sig

# ── Manual scan ───────────────────────────────────────────────────────────────
if run_scan and not errors:
    ma_pair = MA_PAIR_OPTIONS[ma_pair_label]
    config = ScannerConfig(
        min_price=min_price if min_price is not None else 0.0,
        max_price=max_price if max_price is not None else 1_000_000.0,
        min_market_cap=min_cap if min_cap is not None else 0.0,
        max_market_cap=max_cap if max_cap is not None else 100_000_000_000_000.0,
        ma_crossover_pairs=[ma_pair],
        eps_change_threshold=float(eps_threshold),
        trend_window_days=trend_window,
        direction=direction,
    )
    scanner = Scanner(db=db, config=config)
    as_of_date = datetime.now().strftime("%Y-%m-%d")

    with st.spinner("Scanning..."):
        all_signals = scanner.scan(as_of_date)

    # Keep most recent per ticker from this scan
    scan_latest: dict[str, dict] = {}
    for sig in all_signals:
        t = sig["ticker"]
        if t not in scan_latest or sig["trend_change_date"] > scan_latest[t]["trend_change_date"]:
            scan_latest[t] = sig

    # Save new signals to DB so they persist across sessions
    for sig in scan_latest.values():
        db.save_signal_alert({
            "ticker": sig["ticker"],
            "alert_date": as_of_date,
            "signal_type": sig["signal_type"],
            "fast_ma": sig.get("fast_ma"),
            "slow_ma": sig.get("slow_ma"),
            "eps_change_pct": sig.get("eps_change_pct"),
            "eps_date": sig.get("eps_change_date", ""),
            "cross_date": sig.get("trend_change_date", ""),
            "days_between": sig.get("days_between"),
            "close_price": None,
        })

    # Merge scan results into the DB-loaded set
    db_signals = _merge_into(db_signals, list(scan_latest.values()))
    st.session_state["scan_run_time"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    st.query_params.update({
        "min_p": min_price_str, "max_p": max_price_str,
        "min_cap": min_cap_str, "max_cap": max_cap_str,
        "ma": ma_pair_label, "eps": str(eps_threshold),
        "window": str(trend_window), "dir": direction,
        "recency": str(max_days_old),
    })

# ── Build final result set ────────────────────────────────────────────────────
run_time = st.session_state.get("scan_run_time", "")
results: list[dict] | None = None
if db_signals:
    results = _enrich(list(db_signals.values()))

if results is not None:
    if results:
        today = date.today()

        # Apply recency filter
        results_recent = [
            r for r in results
            if (today - date.fromisoformat(r["trend_change_date"])).days <= max_days_old
        ]
        filtered_out = len(results) - len(results_recent)

        # Apply sort
        sort_key, sort_desc = SORT_OPTIONS[sort_label]
        results_sorted = sorted(
            results_recent,
            key=lambda r: (r.get(sort_key) is None, r.get(sort_key) or 0),
            reverse=sort_desc,
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Signals", len(results_sorted))
        col2.metric("Filtered (too old)", filtered_out)
        source = f"Last manual scan: {run_time}" if run_time else f"Loaded from saved alerts (last {max_days_old} days)"
        col3.caption(source)

        if not results_sorted:
            st.info(f"No signals within {max_days_old} days. Widen the 'Max days since crossover' slider or re-run with a broader trend window.")
            st.stop()

        # --- Clickable ticker buttons with date ---
        st.subheader("Tickers")
        n_cols = min(len(results_sorted), 8)
        rows_chunked = [results_sorted[i : i + n_cols] for i in range(0, len(results_sorted), n_cols)]
        for row_chunk in rows_chunked:
            btn_cols = st.columns(n_cols)
            for i, result in enumerate(row_chunk):
                ticker = result["ticker"]
                signal = result.get("signal_type", "")
                arrow = "🟢" if signal == "bullish" else "🔴"
                try:
                    cross_dt = datetime.fromisoformat(result["trend_change_date"])
                    date_str = cross_dt.strftime("%b %d")
                except Exception:
                    date_str = ""
                label = f"{arrow} {ticker}\n{date_str}"
                with btn_cols[i]:
                    if st.button(label, key=f"ticker_{ticker}_{result['trend_change_date']}"):
                        st.session_state["selected_ticker"] = ticker
                        st.session_state["signal_data"] = result
                        st.switch_page("pages/2_Stock_Detail.py")

        # --- Details table ---
        st.subheader("Details")

        def fmt_cap(v):
            if v is None:
                return "N/A"
            if v >= 1_000_000_000:
                return f"${v / 1_000_000_000:.1f}B"
            return f"${v / 1_000_000:.0f}M"

        def fmt_dvol(v):
            if v is None:
                return "N/A"
            if v >= 1_000_000:
                return f"${v / 1_000_000:.1f}M"
            return f"${v / 1_000:.0f}K"

        table_rows = []
        for r in results_sorted:
            eps_chg = r.get("eps_change_pct")
            try:
                days_ago = (today - date.fromisoformat(r["trend_change_date"])).days
            except Exception:
                days_ago = None
            table_rows.append({
                "Ticker": r["ticker"],
                "Signal": r.get("signal_type", ""),
                "MA Cross": f"{r.get('fast_ma', '')}/{r.get('slow_ma', '')}",
                "EPS Chg %": f"{eps_chg:+.1f}%" if eps_chg is not None else "N/A",
                "EPS Date": r.get("eps_change_date", ""),
                "Cross Date": r.get("trend_change_date", ""),
                "Days Ago": days_ago,
                "Price": f"${r['latest_close']:.2f}" if r.get("latest_close") else "N/A",
                "Mkt Cap": fmt_cap(r.get("market_cap")),
                "Avg Vol": f"{int(r['avg_volume']):,}" if r.get("avg_volume") else "N/A",
                "Avg $ Vol": fmt_dvol(r.get("avg_dollar_vol")),
            })

        df = pd.DataFrame(table_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info(
            f"No saved signals in the last {max_days_old} days. "
            "Run the scanner to search for new signals, or widen the 'Max days since crossover' slider."
        )
