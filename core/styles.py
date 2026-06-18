import streamlit as st


def inject_styles() -> None:
    """Inject global CSS for Terminal Pro design system."""
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Design tokens ─────────────────────────────────────────── */
:root {
  --bg:           #080d17;
  --bg-2:         #0f1729;
  --bg-3:         #141e30;
  --border:       #1e2d45;
  --border-hi:    #2a3f5f;
  --cyan:         #06b6d4;
  --cyan-dim:     rgba(6,182,212,.12);
  --green:        #10b981;
  --green-dim:    rgba(16,185,129,.12);
  --red:          #ef4444;
  --red-dim:      rgba(239,68,68,.12);
  --amber:        #f59e0b;
  --text-1:       #e2e8f0;
  --text-2:       #94a3b8;
  --text-3:       #4b6080;
  --mono:         'IBM Plex Mono', ui-monospace, monospace;
  --sans:         'IBM Plex Sans', system-ui, sans-serif;
  --radius:       6px;
}

/* ── Base ──────────────────────────────────────────────────── */
html, body, [data-testid="stApp"],
.stApp, .main .block-container          { font-family: var(--sans) !important; }

[data-testid="stApp"]                   { background: var(--bg) !important; }

/* Main content padding */
.main .block-container {
  padding: 2rem 2.5rem 4rem !important;
  max-width: 1200px;
}

/* ── Sidebar ───────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--bg-2) !important;
  border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] > div:first-child {
  padding-top: 1.5rem;
}

/* Sidebar nav links */
[data-testid="stSidebarNavLink"] {
  font-family: var(--mono) !important;
  font-size: 0.7rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  color: var(--text-2) !important;
  padding: 0.45rem 1rem !important;
  border-radius: 0 !important;
  border-left: 2px solid transparent !important;
  transition: all 0.15s ease !important;
}

[data-testid="stSidebarNavLink"]:hover {
  color: var(--cyan) !important;
  border-left-color: var(--cyan) !important;
  background: var(--cyan-dim) !important;
}

[data-testid="stSidebarNavLink"][aria-current="page"] {
  color: var(--cyan) !important;
  border-left-color: var(--cyan) !important;
  background: var(--cyan-dim) !important;
  font-weight: 600 !important;
}

/* ── Typography ────────────────────────────────────────────── */
h1 {
  font-family: var(--sans) !important;
  font-size: 1.6rem !important;
  font-weight: 600 !important;
  letter-spacing: -0.02em !important;
  color: var(--text-1) !important;
  margin-bottom: 0.25rem !important;
}

h2 {
  font-family: var(--sans) !important;
  font-size: 0.9rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  color: var(--text-3) !important;
  margin-top: 1.5rem !important;
  margin-bottom: 0.75rem !important;
}

h3 {
  font-family: var(--sans) !important;
  font-size: 0.85rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  color: var(--text-2) !important;
}

p, .stMarkdown p {
  color: var(--text-2);
  font-size: 0.875rem;
  line-height: 1.6;
}

/* ── Metric cards ──────────────────────────────────────────── */
[data-testid="metric-container"] {
  background: var(--bg-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 0.875rem 1rem !important;
}

[data-testid="stMetricLabel"] {
  font-family: var(--mono) !important;
  font-size: 0.65rem !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  color: var(--text-3) !important;
}

[data-testid="stMetricValue"] {
  font-family: var(--mono) !important;
  font-size: 1.6rem !important;
  font-weight: 600 !important;
  color: var(--cyan) !important;
  line-height: 1.2 !important;
}

[data-testid="stMetricDelta"] {
  font-family: var(--mono) !important;
  font-size: 0.75rem !important;
}

/* ── Buttons ───────────────────────────────────────────────── */
.stButton > button {
  font-family: var(--mono) !important;
  font-size: 0.72rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  background: var(--bg-3) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-2) !important;
  border-radius: var(--radius) !important;
  padding: 0.45rem 1rem !important;
  transition: all 0.15s ease !important;
  height: auto !important;
}

.stButton > button:hover {
  background: var(--bg-2) !important;
  border-color: var(--cyan) !important;
  color: var(--cyan) !important;
}

/* Primary buttons */
.stButton > button[kind="primary"],
[data-testid="baseButton-primary"] {
  background: var(--cyan) !important;
  border-color: var(--cyan) !important;
  color: #000 !important;
  font-weight: 600 !important;
}

.stButton > button[kind="primary"]:hover,
[data-testid="baseButton-primary"]:hover {
  background: #0891b2 !important;
  border-color: #0891b2 !important;
  color: #000 !important;
}

/* ── Form inputs ───────────────────────────────────────────── */
.stTextInput input,
.stNumberInput input,
.stDateInput input {
  font-family: var(--mono) !important;
  font-size: 0.8rem !important;
  background: var(--bg-3) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  color: var(--text-1) !important;
  padding: 0.45rem 0.75rem !important;
}

.stTextInput input:focus,
.stNumberInput input:focus {
  border-color: var(--cyan) !important;
  box-shadow: 0 0 0 2px var(--cyan-dim) !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
  background: var(--bg-3) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  font-family: var(--mono) !important;
  font-size: 0.8rem !important;
  color: var(--text-1) !important;
}

/* Slider */
[data-testid="stSlider"] label {
  font-family: var(--mono) !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  color: var(--text-3) !important;
}

[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
  background: var(--cyan) !important;
  border-color: var(--cyan) !important;
}

/* ── Tabs ──────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
  padding: 0 !important;
}

.stTabs [data-baseweb="tab"] {
  font-family: var(--mono) !important;
  font-size: 0.7rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  color: var(--text-3) !important;
  background: transparent !important;
  padding: 0.6rem 1.25rem !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  transition: all 0.15s ease !important;
}

.stTabs [aria-selected="true"] {
  color: var(--cyan) !important;
  border-bottom-color: var(--cyan) !important;
  background: transparent !important;
}

.stTabs [data-baseweb="tab"]:hover {
  color: var(--text-1) !important;
  background: transparent !important;
}

/* ── Expanders ─────────────────────────────────────────────── */
[data-testid="stExpander"] {
  background: var(--bg-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  margin-bottom: 0.375rem !important;
}

[data-testid="stExpander"] summary {
  font-family: var(--mono) !important;
  font-size: 0.78rem !important;
  padding: 0.6rem 0.875rem !important;
  color: var(--text-1) !important;
}

[data-testid="stExpander"] summary:hover {
  color: var(--cyan) !important;
}

[data-testid="stExpander"] > div:last-child {
  padding: 0.5rem 0.875rem 0.875rem !important;
}

/* ── Dataframes ────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  overflow: hidden !important;
}

[data-testid="stDataFrame"] table {
  font-family: var(--mono) !important;
  font-size: 0.775rem !important;
}

[data-testid="stDataFrame"] th {
  background: var(--bg-3) !important;
  color: var(--text-3) !important;
  font-size: 0.65rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  padding: 0.5rem 0.75rem !important;
  border-bottom: 1px solid var(--border) !important;
  white-space: nowrap !important;
}

[data-testid="stDataFrame"] td {
  color: var(--text-1) !important;
  padding: 0.4rem 0.75rem !important;
  border-bottom: 1px solid var(--border) !important;
}

/* ── Alert boxes ───────────────────────────────────────────── */
[data-testid="stAlert"] {
  border-radius: var(--radius) !important;
  border-left-width: 3px !important;
  font-family: var(--sans) !important;
  font-size: 0.825rem !important;
  padding: 0.6rem 0.875rem !important;
}

/* ── Captions ──────────────────────────────────────────────── */
.stCaption, [data-testid="stCaptionContainer"] {
  font-family: var(--mono) !important;
  font-size: 0.68rem !important;
  color: var(--text-3) !important;
  letter-spacing: 0.04em !important;
}

/* ── Dividers ──────────────────────────────────────────────── */
hr {
  border: none !important;
  border-top: 1px solid var(--border) !important;
  margin: 1.25rem 0 !important;
}

/* ── Info/Success banners ──────────────────────────────────── */
.stSuccess { background: var(--green-dim) !important; border-left-color: var(--green) !important; }
.stError   { background: var(--red-dim)   !important; border-left-color: var(--red)   !important; }
.stWarning { background: rgba(245,158,11,.1) !important; border-left-color: var(--amber) !important; }
.stInfo    { background: var(--cyan-dim)  !important; border-left-color: var(--cyan)  !important; }

/* ── Sidebar labels ────────────────────────────────────────── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption {
  font-family: var(--mono) !important;
  font-size: 0.68rem !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  color: var(--text-3) !important;
}

/* ── Scrollbar ─────────────────────────────────────────────── */
::-webkit-scrollbar               { width: 5px; height: 5px; }
::-webkit-scrollbar-track         { background: var(--bg); }
::-webkit-scrollbar-thumb         { background: var(--border-hi); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover   { background: var(--text-3); }

/* ── Hide Streamlit branding ───────────────────────────────── */
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }
</style>
""", unsafe_allow_html=True)
