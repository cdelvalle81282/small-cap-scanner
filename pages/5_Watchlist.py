import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DB_PATH
from core.database import Database

st.set_page_config(page_title="Watchlist", page_icon="👁", layout="wide")


@st.cache_resource
def get_db() -> Database:
    db = Database(DB_PATH)
    db.initialize()
    return db


db = get_db()

st.title("Watchlist & Alerts")

# --- Unread alerts banner ---
unread = db.get_unread_alerts()
if unread:
    st.error(f"**{len(unread)} unread alert{'s' if len(unread) > 1 else ''}**")
    for alert in unread:
        st.warning(
            f"**{alert['ticker']}** closed at **${alert['triggered_close']:.2f}** — "
            f"crossed {alert['level_type']} ${alert['level_price']:.2f} "
            f"({alert['level_label']}) on {alert['alert_date']}"
        )
    if st.button("Mark all as read"):
        db.mark_all_alerts_read()
        st.rerun()
    st.divider()

# --- Active watchlist ---
st.subheader("Active Watchlist")
entries = db.get_watchlist(active_only=True)

if not entries:
    st.info("No active watchlist entries. Run the scanner, analyze a signal, and click 'Add to Watchlist'.")
else:
    for entry in entries:
        eps_chg = entry.get("eps_change_pct")
        eps_chg_str = f"{eps_chg:+.1f}%" if eps_chg is not None else "N/A"
        with st.expander(
            f"{'🟢' if entry['signal_type'] == 'bullish' else '🔴'} "
            f"{entry['ticker']} — {entry['signal_type']} | "
            f"EPS {eps_chg_str} | expires {entry['expiry_date']}"
        ):
            col1, col2 = st.columns([3, 1])

            with col1:
                try:
                    levels = json.loads(entry["levels_json"])
                    if levels:
                        levels_df = pd.DataFrame(levels)[["type", "price", "label"]]
                        levels_df.columns = ["Type", "Price", "Significance"]
                        levels_df["Price"] = levels_df["Price"].apply(lambda p: f"${p:.2f}")
                        st.dataframe(levels_df, hide_index=True, use_container_width=True)
                except Exception:
                    st.write("No levels data.")

                if entry.get("trend_break_condition"):
                    st.info(f"**Trend break:** {entry['trend_break_condition']}")

            with col2:
                st.metric("Signal Date", entry["signal_date"])
                st.metric("EPS Date", entry["eps_date"])
                if st.button("Remove", key=f"remove_{entry['id']}"):
                    db.remove_from_watchlist(entry["id"])
                    st.rerun()

# --- Alert history ---
st.divider()
st.subheader("Alert History")
all_alerts = db.get_all_alerts()
if all_alerts:
    alerts_df = pd.DataFrame(all_alerts)[[
        "alert_date", "ticker", "level_type", "level_price", "triggered_close", "level_label", "read_flag"
    ]].copy()
    alerts_df.columns = ["Date", "Ticker", "Level Type", "Level $", "Close $", "Significance", "Read"]
    alerts_df["Level $"] = alerts_df["Level $"].apply(lambda p: f"${p:.2f}" if pd.notna(p) else "")
    alerts_df["Close $"] = alerts_df["Close $"].apply(lambda p: f"${p:.2f}" if pd.notna(p) else "")
    alerts_df["Read"] = alerts_df["Read"].map({0: "Unread", 1: "Read"})
    st.dataframe(alerts_df, hide_index=True, use_container_width=True)
else:
    st.info("No alerts yet. Run `python monitor.py` daily to check prices.")
