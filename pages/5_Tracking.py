import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DB_PATH
from core.database import Database

st.set_page_config(page_title="Tracking", page_icon="📋", layout="wide")


@st.cache_resource
def get_db() -> Database:
    db = Database(DB_PATH)
    db.initialize()
    return db


@st.cache_data(ttl=30)
def get_open_trades_cached() -> list[dict]:
    return get_db().get_trades(status="open")


@st.cache_data(ttl=30)
def get_closed_trades_cached() -> list[dict]:
    return get_db().get_trades(status="closed")


@st.cache_data(ttl=300)
def fetch_current_prices(tickers: tuple[str, ...]) -> dict[str, float]:
    if not tickers:
        return {}
    try:
        raw = yf.download(list(tickers), period="2d", auto_adjust=True, progress=False)
        prices = {}
        for t in tickers:
            try:
                col = raw["Close"][t] if len(tickers) > 1 else raw["Close"]
                prices[t] = float(col.dropna().iloc[-1])
            except Exception:
                prices[t] = None
        return prices
    except Exception:
        return {}


def calc_pnl(trade: dict, current_price: float | None) -> dict:
    ep = trade.get("entry_price")
    shares = trade.get("shares") or 1
    direction = trade.get("direction", "long")

    if trade["status"] == "closed":
        xp = trade.get("exit_price")
        if ep and xp:
            raw = (xp - ep) * shares if direction == "long" else (ep - xp) * shares
            pct = (xp - ep) / ep * 100 if direction == "long" else (ep - xp) / ep * 100
            return {"pnl": raw, "pnl_pct": pct}
    elif current_price and ep:
        raw = (current_price - ep) * shares if direction == "long" else (ep - current_price) * shares
        pct = (current_price - ep) / ep * 100 if direction == "long" else (ep - current_price) / ep * 100
        return {"pnl": raw, "pnl_pct": pct}
    return {"pnl": None, "pnl_pct": None}


db = get_db()

st.title("Tracking")

tab_alerts, tab_watchlist, tab_trades = st.tabs(["📢 Alerts", "👁 Watchlist", "📈 Trades"])

# ── TAB 1: ALERTS ─────────────────────────────────────────────────────────────
with tab_alerts:
    st.subheader("Signal Alert History")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", value=date.today() - timedelta(days=30), key="alert_start")
    with col2:
        end_date = st.date_input("To", value=date.today(), key="alert_end")

    alerts = db.get_signal_alerts(start_date.isoformat(), end_date.isoformat())

    if not alerts:
        st.info(
            "No alerts in this date range. Alerts are recorded when `scan_and_notify.py` "
            "runs and finds new signals. Run the daily scan manually to populate: "
            "`python scan_and_notify.py`"
        )
    else:
        st.metric("Alerts found", len(alerts))

        for alert in alerts:
            arrow = "🟢" if alert["signal_type"] == "bullish" else "🔴"
            with st.expander(
                f"{arrow} {alert['ticker']} — {alert['signal_type']} | "
                f"SMA{alert['fast_ma']}/{alert['slow_ma']} | "
                f"EPS {alert.get('eps_change_pct', 0):+.1f}% | {alert['alert_date']}"
            ):
                col1, col2, col3 = st.columns(3)
                col1.metric("Cross Date", alert.get("cross_date", "N/A"))
                col2.metric("EPS Date", alert.get("eps_date", "N/A"))
                col3.metric("Days Between", alert.get("days_between", "N/A"))
                if alert.get("close_price"):
                    st.metric("Price at Signal", f"${alert['close_price']:.2f}")

                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    if st.button("Add to Watchlist", key=f"wl_{alert['id']}"):
                        if not db.get_watchlist_entry(alert["ticker"]):
                            eps_dt = alert.get("eps_date", "")
                            expiry = (
                                (datetime.fromisoformat(eps_dt) + timedelta(days=30)).strftime("%Y-%m-%d")
                                if eps_dt else (date.today() + timedelta(days=30)).isoformat()
                            )
                            db.add_to_watchlist({
                                "ticker": alert["ticker"],
                                "signal_type": alert["signal_type"],
                                "eps_change_pct": alert.get("eps_change_pct"),
                                "eps_date": eps_dt,
                                "signal_date": alert.get("cross_date", ""),
                                "fast_ma": alert.get("fast_ma"),
                                "slow_ma": alert.get("slow_ma"),
                                "levels_json": "[]",
                                "trend_break_price": None,
                                "trend_break_condition": "",
                                "ai_analysis": "",
                                "expiry_date": expiry,
                                "added_date": date.today().isoformat(),
                            })
                            st.success(f"Added {alert['ticker']} to watchlist")
                            st.rerun()
                        else:
                            st.info("Already on watchlist")
                with bcol2:
                    if st.button("Enter Trade", key=f"trade_{alert['id']}"):
                        st.session_state["prefill_trade"] = {
                            "ticker": alert["ticker"],
                            "direction": "long" if alert["signal_type"] == "bullish" else "short",
                            "signal_type": alert["signal_type"],
                            "eps_change_pct": alert.get("eps_change_pct"),
                            "eps_date": alert.get("eps_date", ""),
                            "entry_price": alert.get("close_price"),
                        }
                        st.info("Trade pre-filled — click the Trades tab to complete entry.")

# ── TAB 2: WATCHLIST ──────────────────────────────────────────────────────────
with tab_watchlist:
    st.subheader("Active Watchlist")

    unread = db.get_unread_alerts()
    if unread:
        st.error(f"**{len(unread)} price level alert{'s' if len(unread) > 1 else ''} triggered**")
        for a in unread:
            st.warning(
                f"**{a['ticker']}** closed at **${a['triggered_close']:.2f}** — "
                f"crossed {a['level_type']} ${a['level_price']:.2f} "
                f"({a['level_label']}) on {a['alert_date']}"
            )
        if st.button("Mark all as read"):
            db.mark_all_alerts_read()
            st.rerun()
        st.divider()

    entries = db.get_watchlist(active_only=True)
    if not entries:
        st.info(
            "No active watchlist entries. Run the scanner, analyze a signal with AI, "
            "then click 'Add to Watchlist' — or use the Alerts tab above."
        )
    else:
        for entry in entries:
            arrow = "🟢" if entry["signal_type"] == "bullish" else "🔴"
            with st.expander(
                f"{arrow} {entry['ticker']} — {entry['signal_type']} | "
                f"EPS {entry.get('eps_change_pct', 0):+.1f}% | expires {entry['expiry_date']}"
            ):
                col1, col2 = st.columns([3, 1])
                with col1:
                    try:
                        levels = json.loads(entry["levels_json"])
                        if levels:
                            ldf = pd.DataFrame(levels)[["type", "price", "label"]]
                            ldf.columns = ["Type", "Price", "Significance"]
                            ldf["Price"] = ldf["Price"].apply(lambda p: f"${p:.2f}")
                            st.dataframe(ldf, hide_index=True, use_container_width=True)
                        else:
                            st.caption("No AI levels — navigate to Stock Detail and run AI analysis to extract levels.")
                    except Exception:
                        pass
                    if entry.get("trend_break_condition"):
                        st.info(f"**Trend break:** {entry['trend_break_condition']}")
                with col2:
                    st.metric("Signal Date", entry["signal_date"])
                    st.metric("EPS Date", entry["eps_date"])
                    if st.button("Enter Trade", key=f"wl_trade_{entry['id']}"):
                        st.session_state["prefill_trade"] = {
                            "ticker": entry["ticker"],
                            "direction": "long" if entry["signal_type"] == "bullish" else "short",
                            "signal_type": entry["signal_type"],
                            "eps_change_pct": entry.get("eps_change_pct"),
                            "eps_date": entry.get("eps_date", ""),
                            "entry_price": None,
                        }
                        st.info("Trade pre-filled — click the Trades tab to complete entry.")
                    if st.button("Remove", key=f"wl_rm_{entry['id']}"):
                        db.remove_from_watchlist(entry["id"])
                        st.rerun()

    st.divider()
    st.subheader("Price Level Alert History")
    all_price_alerts = db.get_all_alerts()
    if all_price_alerts:
        adf = pd.DataFrame(all_price_alerts)[
            ["alert_date", "ticker", "level_type", "level_price", "triggered_close", "level_label", "read_flag"]
        ].copy()
        adf.columns = ["Date", "Ticker", "Level Type", "Level $", "Close $", "Significance", "Read"]
        adf["Level $"] = adf["Level $"].apply(lambda p: f"${p:.2f}" if pd.notna(p) else "")
        adf["Close $"] = adf["Close $"].apply(lambda p: f"${p:.2f}" if pd.notna(p) else "")
        adf["Read"] = adf["Read"].map({0: "Unread", 1: "✓"})
        st.dataframe(adf, hide_index=True, use_container_width=True)
    else:
        st.info("No price level alerts yet. Run `python monitor.py` to check watchlist prices.")

# ── TAB 3: TRADES ─────────────────────────────────────────────────────────────
with tab_trades:
    prefill = st.session_state.pop("prefill_trade", None)

    with st.expander("➕ Add New Trade", expanded=bool(prefill)):
        with st.form("add_trade_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                ticker_in = st.text_input(
                    "Ticker", value=(prefill or {}).get("ticker", "")
                ).upper().strip()
                direction_in = st.selectbox(
                    "Direction",
                    ["long", "short"],
                    index=0 if (prefill or {}).get("direction", "long") == "long" else 1,
                )
                entry_date_in = st.date_input("Entry Date", value=date.today())
                entry_price_in = st.number_input(
                    "Entry Price ($)", min_value=0.0, step=0.01,
                    value=float((prefill or {}).get("entry_price") or 0.0),
                )
            with col2:
                shares_in = st.number_input("Shares", min_value=0.0, step=1.0, value=0.0)
                stop_in = st.number_input("Stop Price ($)", min_value=0.0, step=0.01, value=0.0)
                target_in = st.number_input("Target Price ($)", min_value=0.0, step=0.01, value=0.0)
                notes_in = st.text_area("Notes", value="")

            submitted = st.form_submit_button("Add Trade", type="primary")
            if submitted and ticker_in and entry_price_in > 0:
                db.add_trade({
                    "ticker": ticker_in,
                    "direction": direction_in,
                    "status": "open",
                    "entry_date": entry_date_in.isoformat(),
                    "entry_price": entry_price_in,
                    "shares": shares_in or None,
                    "stop_price": stop_in or None,
                    "target_price": target_in or None,
                    "exit_date": None,
                    "exit_price": None,
                    "notes": notes_in or None,
                    "signal_type": (prefill or {}).get("signal_type"),
                    "eps_change_pct": (prefill or {}).get("eps_change_pct"),
                    "eps_date": (prefill or {}).get("eps_date"),
                    "added_date": date.today().isoformat(),
                })
                st.cache_data.clear()
                st.success(f"Added {direction_in} trade on {ticker_in}")
            elif submitted:
                st.error("Ticker and Entry Price are required.")

    st.subheader("Open Positions")
    open_trades = get_open_trades_cached()

    if open_trades:
        open_tickers = tuple(t["ticker"] for t in open_trades)
        live_prices = fetch_current_prices(open_tickers)

        open_rows = []
        for t in open_trades:
            cp = live_prices.get(t["ticker"])
            pnl = calc_pnl(t, cp)
            days_held = (
                (date.today() - date.fromisoformat(t["entry_date"])).days
                if t.get("entry_date") else None
            )
            open_rows.append({
                "ID": t["id"],
                "Ticker": t["ticker"],
                "Dir": t["direction"],
                "Entry Date": t.get("entry_date", ""),
                "Entry $": f"${t['entry_price']:.2f}" if t.get("entry_price") else "N/A",
                "Shares": t.get("shares") or "—",
                "Stop $": f"${t['stop_price']:.2f}" if t.get("stop_price") else "—",
                "Target $": f"${t['target_price']:.2f}" if t.get("target_price") else "—",
                "Current $": f"${cp:.2f}" if cp else "N/A",
                "P&L $": f"${pnl['pnl']:+.2f}" if pnl["pnl"] is not None else "N/A",
                "P&L %": f"{pnl['pnl_pct']:+.1f}%" if pnl["pnl_pct"] is not None else "N/A",
                "Days": days_held,
            })

        st.dataframe(pd.DataFrame(open_rows).drop(columns=["ID"]), hide_index=True, use_container_width=True)

        st.subheader("Close a Position")
        trade_opts = {f"{t['ticker']} ({t['direction']}) — entered {t.get('entry_date', '?')}": t["id"] for t in open_trades}
        selected_label = st.selectbox("Select trade to close", list(trade_opts.keys()))
        selected_id = trade_opts[selected_label]

        col1, col2, col3 = st.columns(3)
        with col1:
            exit_date_in = st.date_input("Exit Date", value=date.today(), key="exit_date")
        with col2:
            exit_price_in = st.number_input("Exit Price ($)", min_value=0.0, step=0.01, key="exit_price")
        with col3:
            st.write("")
            st.write("")
            if st.button("Close Trade", type="primary"):
                if exit_price_in > 0:
                    db.close_trade(selected_id, exit_date_in.isoformat(), exit_price_in)
                    st.cache_data.clear()
                    st.success("Trade closed.")
                    st.rerun()
                else:
                    st.error("Enter an exit price.")
    else:
        st.info("No open positions. Add a trade above or click 'Enter Trade' from the Alerts or Watchlist tabs.")

    st.divider()
    st.subheader("Closed Positions")
    closed_trades = get_closed_trades_cached()

    if closed_trades:
        closed_rows = []
        for t in closed_trades:
            pnl = calc_pnl(t, None)
            closed_rows.append({
                "Ticker": t["ticker"],
                "Dir": t["direction"],
                "Entry Date": t.get("entry_date", ""),
                "Exit Date": t.get("exit_date", ""),
                "Entry $": f"${t['entry_price']:.2f}" if t.get("entry_price") else "N/A",
                "Exit $": f"${t['exit_price']:.2f}" if t.get("exit_price") else "N/A",
                "Shares": t.get("shares") or "—",
                "P&L $": f"${pnl['pnl']:+.2f}" if pnl["pnl"] is not None else "N/A",
                "P&L %": f"{pnl['pnl_pct']:+.1f}%" if pnl["pnl_pct"] is not None else "N/A",
                "Result": "✅ Win" if (pnl["pnl"] or 0) > 0 else ("❌ Loss" if (pnl["pnl"] or 0) < 0 else "—"),
                "Notes": t.get("notes") or "",
            })

        closed_df = pd.DataFrame(closed_rows)
        st.dataframe(closed_df, hide_index=True, use_container_width=True)

        st.divider()
        st.subheader("Performance Summary")
        pnls = [
            calc_pnl(t, None)["pnl"]
            for t in closed_trades
            if calc_pnl(t, None)["pnl"] is not None
        ]
        if pnls:
            winners = [p for p in pnls if p > 0]
            losers = [p for p in pnls if p < 0]
            total_pnl = sum(pnls)
            win_rate = len(winners) / len(pnls) * 100 if pnls else 0
            avg_win = sum(winners) / len(winners) if winners else 0
            avg_loss = sum(losers) / len(losers) if losers else 0

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total P&L", f"${total_pnl:+,.2f}")
            m2.metric("Win Rate", f"{win_rate:.0f}%")
            m3.metric("Avg Winner", f"${avg_win:+,.2f}")
            m4.metric("Avg Loser", f"${avg_loss:+,.2f}")
            m5.metric("Trades", f"{len(pnls)} ({len(winners)}W / {len(losers)}L)")
    else:
        st.info("No closed positions yet.")
