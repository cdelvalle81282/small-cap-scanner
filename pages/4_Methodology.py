import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DB_PATH
from core.database import Database

st.set_page_config(page_title="Methodology", page_icon="📖", layout="wide")


@st.cache_resource
def get_db() -> Database:
    db = Database(DB_PATH)
    db.initialize()
    return db


db = get_db()

st.title("Methodology")

# ── Section 1: How Signals Are Calculated ─────────────────────────────────

st.header("How Signals Are Calculated")

st.markdown("""
The scanner identifies stocks where a **moving average trend change** occurs
shortly after a significant **earnings-per-share (EPS) change**. A signal
requires two ingredients:

**1. EPS Filter**

Each earnings report is checked for a large enough move. The percentage change
between the reported EPS and the analyst estimate (consensus) must exceed the
configured threshold (default **10 %**). When analyst estimates are unavailable,
the scanner falls back to quarter-over-quarter EPS comparison. Both positive
and negative EPS surprises qualify.

**2. MA Crossover Detection**

After an EPS report passes the filter, the scanner looks for a moving average
crossover within a configurable **trend window** (default **30 days** after the
report date).

| Crossover | Condition | Signal |
|-----------|-----------|--------|
| **Bullish** | Fast SMA crosses **above** Slow SMA | Buy / Long |
| **Bearish** | Fast SMA crosses **below** Slow SMA | Sell / Short |

The default MA pairs are **20/50** and **50/200** (simple moving averages).
Both pairs are checked independently, so a single EPS event can produce
multiple signals.

**3. Stock Universe**

The scanner monitors **183 tickers** across multiple categories:

| Category | Count | Examples |
|----------|-------|---------|
| US small caps ($1--$20) | 83 | SOFI, PLUG, MARA, RIOT, LUNR |
| US-traded ADRs | 29 | NIO, JD, VALE, NU, GRAB, XPEV |
| US small/mid caps ($5--$50) | 48 | DKNG, HIMS, IONQ, UPST, CHWY |
| Biotech / pharma | 21 | BEAM, CRSP, TGTX, LEGN, EDIT |
| Clean energy / EV | 2 | AES, BLDP |

The universe is dynamically filtered at scan time by current price,
so only tickers trading within the configured price range are included.

**4. Configurable Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| Price range | $1 -- $50 | Stock price filter |
| Market cap | $50M -- $10B | Market cap filter |
| MA pairs | 20/50, 50/200 | Fast / slow SMA periods |
| EPS threshold | 10% | Minimum absolute EPS change |
| Trend window | 30 days | Window after EPS report to look for crossover |
| Direction | both | Filter for bullish, bearish, or both |
""")

# ── Section 2: How Buy / Hold / Sell Is Determined ────────────────────────

st.header("How Buy / Hold / Sell Is Determined")

st.markdown("""
Once a signal is identified, the backtester simulates a trade to measure
forward performance.

**Entry**

- Entry occurs on the **next trading day** after the crossover date
  (avoids look-ahead bias -- the crossover isn't confirmed until after
  the close).
- Entry price = midpoint of that day's range: **(High + Low) / 2**.
- An optional slippage percentage can be applied (default 0%).

**Direction**

| Signal | Position | Positive return means... |
|--------|----------|------------------------|
| Bullish | Long | Stock went **up** |
| Bearish | Short | Stock went **down** |

**Stop Loss**

Positions are protected by a trailing stop based on the previous day's range:

- **Long**: exit at close if today's close **< previous day's low**
- **Short**: exit at close if today's close **> previous day's high**

When a stop triggers, the exit price is that day's close and all remaining
horizons lock in the stop-loss return.

**Horizons**

Forward returns are measured at **5, 10, 15, 30, and 60 trading days**
(not calendar days). If a stop loss fires before a horizon is reached,
that horizon records the stop-loss return instead of waiting.

**Metrics**

| Metric | Formula |
|--------|---------|
| Win Rate | wins / total trades |
| Profit Factor | gross gains / abs(gross losses) |
| Expectancy | (win rate x avg winner) + (loss rate x avg loser) |
| Avg Winner / Loser | Mean return of winning / losing trades |
""")

# ── Section 3: Last Data Sync ─────────────────────────────────────────────

st.header("Last Data Sync")

with db._connect() as conn:
    last_updated_row = conn.execute(
        "SELECT MAX(last_updated) AS last_updated FROM stocks"
    ).fetchone()
    last_updated = last_updated_row["last_updated"] if last_updated_row else None

    ticker_count_row = conn.execute(
        "SELECT COUNT(DISTINCT ticker) AS cnt FROM stocks"
    ).fetchone()
    ticker_count = ticker_count_row["cnt"] if ticker_count_row else 0

    price_range_row = conn.execute(
        "SELECT MIN(date) AS earliest, MAX(date) AS latest, "
        "COUNT(*) AS total_rows FROM daily_prices"
    ).fetchone()

    earnings_count_row = conn.execute(
        "SELECT COUNT(*) AS cnt FROM earnings"
    ).fetchone()
    earnings_count = earnings_count_row["cnt"] if earnings_count_row else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Tickers", ticker_count)
col2.metric("Price Records", f"{price_range_row['total_rows']:,}" if price_range_row else "0")
col3.metric("Earnings Records", f"{earnings_count:,}")

if last_updated:
    # Show just date + time, trim timezone/microseconds
    display_ts = last_updated[:19].replace("T", " ")
    col4.metric("Last Sync", display_ts)
else:
    col4.metric("Last Sync", "Never")

if price_range_row and price_range_row["earliest"]:
    st.info(
        f"Price data covers **{price_range_row['earliest']}** to "
        f"**{price_range_row['latest']}**"
    )
else:
    st.warning("No price data in database. Run the pipeline from the home page.")
