"""SEC Filings Dashboard — pull EDGAR XBRL data and visualize fundamentals.

Run:  uv run streamlit run app.py --server.port 8601
Set a real contact string first:  export SEC_USER_AGENT="Your Name you@email.com"
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # run from any cwd

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:  # On Streamlit Cloud the SEC contact string comes from secrets
    if "SEC_USER_AGENT" in st.secrets:
        os.environ["SEC_USER_AGENT"] = st.secrets["SEC_USER_AGENT"]
except FileNotFoundError:  # no secrets.toml locally — env var is used instead
    pass

import edgar
import metrics
import pdf_extract
from universe import PRESETS, load_options

# --- dataviz palette (validated categorical slots + chart chrome) -------------
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
POS, NEG = "#2a78d6", "#e34948"
GRID, AXIS, MUTED = "#2c2c2a", "#383835", "#898781"

MONEY_KEYS = {"revenue", "cost_of_revenue", "gross_profit", "operating_income", "net_income",
              "rnd", "op_cash_flow", "assets", "liabilities", "equity", "cash", "long_term_debt"}
PCT_KEYS = {"gross_margin", "operating_margin", "net_margin", "roe", "roa",
            "rnd_pct_revenue", "ocf_margin", "revenue_growth", "net_income_growth"}

COMPARE_METRICS = ["revenue", "revenue_growth", "gross_margin", "operating_margin", "net_margin",
                   "roe", "roa", "debt_to_equity", "rnd_pct_revenue", "ocf_margin",
                   "eps_diluted", "net_income", "op_cash_flow", "assets"]

MAX_COMPANIES = 8

st.set_page_config(page_title="SEC Filings Dashboard", page_icon="📊", layout="wide")
st.markdown(
    """<style>
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] { color: #a9a8a1; }
    </style>""",
    unsafe_allow_html=True,
)


# --- data loading (cached) ----------------------------------------------------
@st.cache_data(ttl=24 * 3600, show_spinner="Loading company list…")
def ticker_options() -> list[str]:
    """Domestic 10-K filers — bundled universe.json, or built live (see universe.py)."""
    return load_options()


@st.cache_resource
def runtime_blocklist() -> set[str]:
    """Tickers discovered at load time to lack US-GAAP data (session-shared)."""
    return set()


@st.cache_data(ttl=3600, show_spinner=False)
def load_facts(ticker: str):
    info = edgar.resolve_cik(ticker)
    facts = edgar.get_company_facts(info["cik"])
    filings = edgar.get_recent_filings(info["cik"], limit=8)
    return info, facts, filings


@st.cache_data(show_spinner=False)
def parse_pdf_cached(data: bytes) -> pdf_extract.PdfExtraction:
    return pdf_extract.parse_pdf(data)


# --- formatting helpers -------------------------------------------------------
def human(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    a = abs(v)
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"${v / div:.2f}{suf}"
    return f"${v:,.0f}"


def fmt_cell(v) -> str:
    """Table cell: commas for dollar magnitudes, 2 decimals for ratios/margins."""
    if v is None or pd.isna(v):
        return "—"
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.2f}"


def style_fig(fig: go.Figure, ytitle: str, pct: bool = False) -> go.Figure:
    fig.update_layout(
        colorway=PALETTE,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        bargap=0.28,
        bargroupgap=0.08,
        height=340,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=AXIS, tickcolor=AXIS,
                     color=MUTED, dtick=1, tickformat="d")
    fig.update_yaxes(title_text=ytitle, showgrid=True, gridcolor=GRID, zeroline=True,
                     zerolinecolor=AXIS, linecolor="rgba(0,0,0,0)", color=MUTED,
                     ticksuffix="%" if pct else "")
    return fig


def bn(series: pd.Series) -> pd.Series:
    return series / 1e9


# --- chart builders -----------------------------------------------------------
def chart_grouped_bars(df: pd.DataFrame, cols: list[tuple[str, str, str]],
                       ytitle: str, unit: str = "$", suffix: str = "B") -> go.Figure:
    fig = go.Figure()
    for key, name, color in cols:
        if key in df:
            fig.add_bar(x=df.index, y=bn(df[key]), name=name, marker_color=color,
                        hovertemplate=f"{name}: {unit}%{{y:.2f}}{suffix}<extra></extra>")
    fig.update_layout(barmode="group")
    return style_fig(fig, ytitle)


def chart_margins(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for key, name, color in (("gross_margin", "Gross", PALETTE[0]),
                             ("operating_margin", "Operating", PALETTE[2]),
                             ("net_margin", "Net", PALETTE[1])):
        if key in df:
            fig.add_scatter(x=df.index, y=df[key], name=name, mode="lines+markers",
                            line=dict(color=color, width=2), marker=dict(size=7),
                            hovertemplate=f"{name}: %{{y:.1f}}%<extra></extra>")
    return style_fig(fig, "Margin", pct=True)


def chart_growth(df: pd.DataFrame, col: str = "revenue_growth",
                 label: str = "Revenue growth") -> go.Figure:
    g = df[col].dropna()
    colors = [POS if v >= 0 else NEG for v in g]
    fig = go.Figure(go.Bar(x=g.index, y=g, marker_color=colors,
                           hovertemplate=f"{label}: %{{y:.1f}}%<extra></extra>"))
    return style_fig(fig, "YoY growth", pct=True)


def chart_eps(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=df.index, y=df["eps_diluted"], marker_color=PALETTE[4],
                           hovertemplate="Diluted EPS: $%{y:.2f}<extra></extra>"))
    return style_fig(fig, "USD per share")


def metric_kind(key: str) -> str:
    if key in PCT_KEYS:
        return "pct"
    if key in MONEY_KEYS:
        return "money"
    return "ratio"


def chart_compare(frames: dict[str, pd.DataFrame], col: str, rebase: bool = False) -> go.Figure:
    kind = metric_kind(col)
    fig = go.Figure()
    for i, (ticker, df) in enumerate(frames.items()):
        if col not in df:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        if rebase:
            s = s / s.iloc[0] * 100
            hover = f"{ticker}: %{{y:.0f}}<extra></extra>"
        elif kind == "money":
            s = bn(s)
            hover = f"{ticker}: $%{{y:.1f}}B<extra></extra>"
        elif kind == "pct":
            hover = f"{ticker}: %{{y:.1f}}%<extra></extra>"
        else:
            hover = f"{ticker}: %{{y:.2f}}<extra></extra>"
        fig.add_scatter(x=s.index, y=s, name=ticker, mode="lines+markers",
                        line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                        marker=dict(size=7), hovertemplate=hover)
    ytitle = ("Index" if rebase else
              "USD (billions)" if kind == "money" else
              metrics.label_for(col) if kind == "pct" else metrics.label_for(col))
    return style_fig(fig, ytitle, pct=(kind == "pct" and not rebase))


# --- views --------------------------------------------------------------------
def render_overview(ticker: str, info: dict, df: pd.DataFrame, filings: list[dict]) -> None:
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else None

    st.subheader(f"{info['title']}  ·  {ticker}")
    st.caption(f"CIK {info['cik']} · fiscal years {df.index.min()}–{df.index.max()}")

    def delta(key):
        if prev is None or key not in df:
            return None
        cur, base = latest.get(key), prev.get(key)
        if pd.isna(cur) or pd.isna(base) or not base:
            return None
        return f"{(cur - base) / abs(base) * 100:+.1f}% YoY"

    cols = st.columns(5)
    cols[0].metric("Revenue", human(latest.get("revenue")), delta("revenue"))
    cols[1].metric("Net income", human(latest.get("net_income")), delta("net_income"))
    cols[2].metric(
        "Net margin",
        f"{latest['net_margin']:.1f}%" if "net_margin" in df else "—",
        f"{latest['net_margin'] - prev['net_margin']:+.1f} pts"
        if prev is not None and "net_margin" in df and pd.notna(prev.get("net_margin")) else None,
    )
    cols[3].metric(
        "Diluted EPS",
        f"${latest['eps_diluted']:.2f}" if "eps_diluted" in df else "—",
        delta("eps_diluted"),
    )
    cols[4].metric("Total assets", human(latest.get("assets")) if "assets" in df else "—")

    st.markdown("#### Revenue & net income")
    st.plotly_chart(
        chart_grouped_bars(df, [("revenue", "Revenue", PALETTE[0]),
                                ("net_income", "Net income", PALETTE[1])], "USD (billions)"),
        width="stretch", key=f"rev_{ticker}")

    c1, c2 = st.columns(2)
    with c1:
        if any(k in df for k in ("gross_margin", "operating_margin", "net_margin")):
            st.markdown("#### Profitability margins")
            st.plotly_chart(chart_margins(df), width="stretch", key=f"marg_{ticker}")
    with c2:
        if "revenue_growth" in df and df["revenue_growth"].notna().any():
            st.markdown("#### Revenue growth (YoY)")
            st.plotly_chart(chart_growth(df), width="stretch", key=f"grow_{ticker}")

    c3, c4 = st.columns(2)
    with c3:
        if any(k in df for k in ("assets", "liabilities", "equity")):
            st.markdown("#### Balance sheet")
            st.plotly_chart(
                chart_grouped_bars(df, [("assets", "Assets", PALETTE[0]),
                                        ("liabilities", "Liabilities", PALETTE[5]),
                                        ("equity", "Equity", PALETTE[1])], "USD (billions)"),
                width="stretch", key=f"bs_{ticker}")
    with c4:
        if "op_cash_flow" in df and "net_income" in df:
            st.markdown("#### Earnings quality: cash flow vs net income")
            st.plotly_chart(
                chart_grouped_bars(df, [("net_income", "Net income", PALETTE[1]),
                                        ("op_cash_flow", "Operating cash flow", PALETTE[4])],
                                   "USD (billions)"),
                width="stretch", key=f"ocf_{ticker}")
        elif "eps_diluted" in df:
            st.markdown("#### Diluted EPS")
            st.plotly_chart(chart_eps(df), width="stretch", key=f"eps_{ticker}")

    st.markdown("#### Financials")
    as_yoy = st.toggle("Show YoY % change", key=f"yoy_{ticker}")
    if as_yoy:
        raw_cols = [c for c in df.columns if c in MONEY_KEYS or c == "eps_diluted"]
        show = df[raw_cols].pct_change() * 100
        show.columns = [metrics.label_for(c) for c in show.columns]
        st.dataframe(show.style.format("{:+.1f}%", na_rep="—"), width="stretch")
    else:
        show = df.copy()
        show.columns = [metrics.label_for(c) for c in show.columns]
        st.dataframe(show.style.format(fmt_cell, na_rep="—"), width="stretch")
    st.download_button("Download CSV", df.to_csv().encode(),
                       f"{ticker}_financials.csv", "text/csv", key=f"dl_{ticker}")

    st.markdown("#### Recent filings")
    for f in filings:
        st.markdown(f"- **{f['form']}** · {f['filed']} · [{f['description'] or 'document'}]({f['url']})")


def render_compare(loaded: dict) -> None:
    frames = {t: v[1] for t, v in loaded.items()}
    st.subheader("Compare companies")
    st.caption(", ".join(f"{t} ({v[0]['title']})" for t, v in loaded.items()))

    kcols = st.columns(len(loaded))
    for i, (t, (info, df, _)) in enumerate(loaded.items()):
        latest = df.iloc[-1]
        rev = latest.get("revenue")
        margin = latest.get("net_margin")
        kcols[i].metric(
            f"{t} revenue",
            f"${rev / 1e9:,.0f}B" if pd.notna(rev) else "—",
            f"{margin:.1f}% net margin" if pd.notna(margin) else None,
        )

    available = [m for m in COMPARE_METRICS if any(m in df for df in frames.values())]
    picked = st.multiselect(
        "Metrics to compare", available,
        default=[m for m in ("revenue", "net_margin", "roe", "revenue_growth") if m in available],
        format_func=metrics.label_for,
    )
    show_rebased = st.toggle("Add indexed revenue (first year = 100)", value=True)

    charts: list[tuple[str, go.Figure]] = []
    if show_rebased and "revenue" in available:
        charts.append(("Indexed revenue (first year = 100)",
                       chart_compare(frames, "revenue", rebase=True)))
    charts.extend((metrics.label_for(m), chart_compare(frames, m)) for m in picked)

    for row_start in range(0, len(charts), 2):
        cols = st.columns(2)
        for col_st, (title, fig) in zip(cols, charts[row_start:row_start + 2]):
            with col_st:
                st.markdown(f"#### {title}")
                st.plotly_chart(fig, width="stretch", key=f"cmp_{title}")

    st.markdown("#### Latest fiscal year — side by side")
    comp = pd.DataFrame({t: df.iloc[-1] for t, df in frames.items()})
    comp.index = [metrics.label_for(i) for i in comp.index]
    st.dataframe(comp.style.format(fmt_cell, na_rep="—"), width="stretch")
    st.download_button("Download comparison CSV", comp.to_csv().encode(),
                       "comparison.csv", "text/csv")


def render_pdf_tab() -> None:
    st.subheader("Import an annual report PDF")
    st.caption(
        "For reports that never hit EDGAR — foreign filers, older reports, private-ish "
        "disclosures. Drop in a text-based PDF (10-K, annual report); key line items are "
        "extracted heuristically and fully editable below before charting."
    )
    up = st.file_uploader("Annual report PDF", type="pdf")
    if not up:
        st.info("Upload a PDF to extract revenue, profit, balance-sheet items, and cash flow by year.")
        return

    with st.spinner("Parsing PDF…"):
        ext = parse_pdf_cached(up.getvalue())

    if not ext.ok:
        st.error(
            "No financial line items found. This parser needs a text-based PDF "
            "(scanned/image reports won't work) with statement-style tables."
        )
        return

    scale_name = {1e9: "billions", 1e6: "millions", 1e3: "thousands", 1.0: "units"}[ext.scale]
    st.success(
        f"Extracted **{len(ext.df.columns)} metrics × {len(ext.df)} years** "
        f"({ext.rows_matched} data points from pages {ext.pages_matched[:8]}…). "
        f"Detected currency **{ext.currency}**, figures in **{scale_name}**."
    )

    st.markdown("#### Extracted figures — edit anything the parser got wrong")
    st.caption(f"Money values shown in {ext.currency} millions; per-share values as reported.")
    editor_df = ext.df.copy()
    money_cols = [c for c in editor_df.columns if c not in pdf_extract.PER_SHARE_KEYS]
    editor_df[money_cols] = editor_df[money_cols] / 1e6
    edited = st.data_editor(
        editor_df,
        column_config={c: st.column_config.NumberColumn(metrics.label_for(c), format="%.1f")
                       for c in editor_df.columns},
        width="stretch",
    )
    df = edited.copy()
    df[money_cols] = df[money_cols] * 1e6
    df = pdf_extract.add_pdf_derived(df)

    cur = ext.currency
    c1, c2 = st.columns(2)
    with c1:
        if "revenue" in df:
            st.markdown("#### Revenue & profit")
            st.plotly_chart(
                chart_grouped_bars(df, [("revenue", "Revenue", PALETTE[0]),
                                        ("net_income", "Net income", PALETTE[1]),
                                        ("operating_income", "Operating income", PALETTE[2])],
                                   f"{cur} (billions)", unit="", suffix=f"B {cur}"),
                width="stretch", key="pdf_rev")
    with c2:
        if any(k in df for k in ("gross_margin", "operating_margin", "net_margin")):
            st.markdown("#### Margins")
            st.plotly_chart(chart_margins(df), width="stretch", key="pdf_marg")

    c3, c4 = st.columns(2)
    with c3:
        if "revenue_growth" in df and df["revenue_growth"].notna().any():
            st.markdown("#### Revenue growth (YoY)")
            st.plotly_chart(chart_growth(df), width="stretch", key="pdf_grow")
    with c4:
        if any(k in df for k in ("assets", "liabilities", "equity")):
            st.markdown("#### Balance sheet")
            st.plotly_chart(
                chart_grouped_bars(df, [("assets", "Assets", PALETTE[0]),
                                        ("liabilities", "Liabilities", PALETTE[5]),
                                        ("equity", "Equity", PALETTE[1])],
                                   f"{cur} (billions)", unit="", suffix=f"B {cur}"),
                width="stretch", key="pdf_bs")

    st.download_button("Download extracted CSV", df.to_csv().encode(),
                       "pdf_financials.csv", "text/csv")


# --- header & controls (main area, always visible) ----------------------------
st.markdown("## 📊 SEC Filings Dashboard")
st.caption(
    "Fundamentals straight from [SEC EDGAR](https://www.sec.gov/edgar) XBRL filings. "
    "Fiscal years labeled by period-end year."
)

_blocked = runtime_blocklist()
options = [o for o in ticker_options() if o.split(" — ")[0] not in _blocked]
if "companies" not in st.session_state:
    st.session_state.companies = [o for o in options if o.startswith(("AAPL —", "MSFT —"))]
else:
    # Self-heal: drop selections that failed a previous load (foreign filers).
    st.session_state.companies = [
        o for o in st.session_state.companies
        if o.split(" — ")[0].strip().upper() not in _blocked
    ]


def _apply_preset() -> None:
    group = PRESETS.get(st.session_state.preset_pick)
    if group:
        by_ticker = {o.split(" — ")[0]: o for o in options}
        st.session_state.companies = [by_ticker.get(t, t) for t in group]


cc1, cc2, cc3, cc4 = st.columns([4, 1.6, 1.1, 1.8], vertical_alignment="bottom")
selected = cc1.multiselect(
    "Companies", options, key="companies",
    accept_new_options=True,
    help="Pick from the list or type any US-listed ticker and press Enter.",
    max_selections=MAX_COMPANIES,
)
cc2.selectbox("Peer group", list(PRESETS), key="preset_pick")
cc3.button("Load preset", on_click=_apply_preset, width="stretch")
years = cc4.slider("Years of history", min_value=4, max_value=15, value=8)

tickers = []
for opt in selected:
    t = opt.split(" — ")[0].strip().upper()
    if t and t not in tickers:
        tickers.append(t)

# --- load ---------------------------------------------------------------------
tab_overview, tab_compare, tab_pdf = st.tabs(["📈 Overview", "⚖️ Compare", "📄 PDF import"])

loaded: dict[str, tuple] = {}
skipped: list[str] = []
errors: list[str] = []
for t in tickers[:MAX_COMPANIES]:
    try:
        info, facts, filings = load_facts(t)
        df = metrics.build_financials(facts, years=years)
        if df.empty or "revenue" not in df:
            runtime_blocklist().add(t)
            skipped.append(t)
            continue
        loaded[t] = (info, df, filings)
    except edgar.EdgarError as exc:
        if "404" in str(exc):
            runtime_blocklist().add(t)
            skipped.append(t)
        else:
            errors.append(f"{t}: {exc}")
    except Exception as exc:  # noqa: BLE001 — surface any parse issue to the UI
        errors.append(f"{t}: unexpected error — {exc}")

if skipped:
    st.warning(
        f"Skipped **{', '.join(skipped)}** — no US-GAAP filings on EDGAR (foreign "
        "filer or OTC listing). Use the **📄 PDF import** tab for these companies."
    )
for e in errors:
    st.error(e)

with tab_overview:
    if not loaded:
        st.info("Pick at least one company in the picker above.")
    elif len(loaded) == 1:
        t, (info, df, filings) = next(iter(loaded.items()))
        render_overview(t, info, df, filings)
    else:
        t = st.selectbox("Company", list(loaded), format_func=lambda x: f"{x} — {loaded[x][0]['title']}")
        info, df, filings = loaded[t]
        render_overview(t, info, df, filings)

with tab_compare:
    if len(loaded) < 2:
        st.info("Add two or more companies in the picker above (or load a peer-group preset) to compare.")
    else:
        render_compare(loaded)

with tab_pdf:
    render_pdf_tab()
