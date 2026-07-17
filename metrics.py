"""Map raw XBRL company-facts into tidy annual financial time series.

Companies tag the same economic concept with different us-gaap elements
(e.g. revenue may be ``Revenues`` or
``RevenueFromContractWithCustomerExcludingAssessedTax``), so each metric
lists candidate tags tried in order. Values are reduced to one figure per
fiscal-year-end: full-year durations for flow items, year-end snapshots
for balance-sheet (instant) items, de-duplicating restatements by keeping
the latest-filed value.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    tags: tuple[str, ...]
    kind: str = "duration"  # "duration" (flow) or "instant" (stock)
    unit: str = "USD"
    group: str = "Income statement"


METRICS: list[Metric] = [
    Metric("revenue", "Revenue", (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues", "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    )),
    Metric("cost_of_revenue", "Cost of revenue", (
        "CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold",
    )),
    Metric("gross_profit", "Gross profit", ("GrossProfit",)),
    Metric("operating_income", "Operating income", ("OperatingIncomeLoss",)),
    Metric("net_income", "Net income", ("NetIncomeLoss", "ProfitLoss")),
    Metric("rnd", "R&D expense", ("ResearchAndDevelopmentExpense",)),
    Metric("op_cash_flow", "Operating cash flow", (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ), group="Cash flow"),
    Metric("eps_diluted", "Diluted EPS", ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"),
           unit="USD/shares", group="Per share"),
    Metric("assets", "Total assets", ("Assets",), kind="instant", group="Balance sheet"),
    Metric("liabilities", "Total liabilities", ("Liabilities",), kind="instant", group="Balance sheet"),
    Metric("equity", "Stockholders' equity", (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ), kind="instant", group="Balance sheet"),
    Metric("cash", "Cash & equivalents", ("CashAndCashEquivalentsAtCarryingValue",),
           kind="instant", group="Balance sheet"),
    Metric("long_term_debt", "Long-term debt", ("LongTermDebtNoncurrent", "LongTermDebt"),
           kind="instant", group="Balance sheet"),
]

METRICS_BY_KEY = {m.key: m for m in METRICS}


def _series_candidates(facts: dict, metric: Metric) -> list[list[dict]]:
    """Return each matching tag's units list, in the metric's priority order."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    out: list[list[dict]] = []
    for tag in metric.tags:
        node = us_gaap.get(tag)
        if not node:
            continue
        units = node.get("units", {})
        series = units.get(metric.unit) or next(iter(units.values()), None)
        if series:
            out.append(series)
    return out


def _extract(facts: dict, metric: Metric) -> dict[int, float]:
    """Merge annual points across candidate tags; higher-priority tag wins a year.

    Companies rename XBRL elements over time (e.g. MSFT moved from
    ``CostOfRevenue`` to ``CostOfGoodsAndServicesSold``), so a single tag can
    cover only part of the history. Lower-priority tags fill the gaps.
    """
    merged: dict[int, float] = {}
    for series in _series_candidates(facts, metric):
        for yr, val in _annual_points(series, metric.kind).items():
            merged.setdefault(yr, val)
    return merged


def _annual_points(series: list[dict], kind: str) -> dict[int, float]:
    """Reduce raw XBRL points to {fiscal_year_end_year: value}."""
    # (end_date, value, filed) tuples that qualify as annual observations
    candidates: list[tuple[date, float, str]] = []
    for e in series:
        val = e.get("val")
        end = e.get("end")
        if val is None or not end:
            continue
        try:
            end_d = date.fromisoformat(end)
        except ValueError:
            continue
        form = e.get("form") or ""
        if not form.startswith("10-K"):
            # 10-Qs also carry ~365-day windows (LTM comparatives) that would
            # fabricate phantom fiscal years; annual figures come from 10-Ks.
            continue
        if kind == "duration":
            start = e.get("start")
            if not start:
                continue
            try:
                span = (end_d - date.fromisoformat(start)).days
            except ValueError:
                continue
            if not (340 <= span <= 380):  # full fiscal year only
                continue
        else:  # instant: year-end snapshot
            if e.get("start"):
                continue
        candidates.append((end_d, float(val), e.get("filed", "")))

    # Keep one value per period-end: the latest-filed (handles restatements).
    by_end: dict[date, tuple[float, str]] = {}
    for end_d, val, filed in candidates:
        prev = by_end.get(end_d)
        if prev is None or filed >= prev[1]:
            by_end[end_d] = (val, filed)

    # Collapse to one value per fiscal-year (year of the period end).
    by_year: dict[int, tuple[date, float]] = {}
    for end_d, (val, _) in by_end.items():
        prev = by_year.get(end_d.year)
        if prev is None or end_d > prev[0]:
            by_year[end_d.year] = (end_d, val)
    return {yr: val for yr, (_, val) in by_year.items()}


def build_financials(facts: dict, years: int = 10) -> pd.DataFrame:
    """Build a fiscal-year-indexed DataFrame of raw + derived metrics."""
    data: dict[str, dict[int, float]] = {}
    for metric in METRICS:
        pts = _extract(facts, metric)
        if pts:
            data[metric.key] = pts

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data).sort_index()
    df.index.name = "fiscal_year"

    # Trim to the most recent N fiscal years.
    if years and len(df) > years:
        df = df.iloc[-years:]

    _add_derived(df)
    return df


def _add_derived(df: pd.DataFrame) -> None:
    """Add margins, growth, and returns in place where inputs exist."""
    def has(*cols: str) -> bool:
        return all(c in df.columns for c in cols)

    if has("gross_profit", "revenue"):
        df["gross_margin"] = df["gross_profit"] / df["revenue"] * 100
    elif has("revenue", "cost_of_revenue"):
        df["gross_margin"] = (df["revenue"] - df["cost_of_revenue"]) / df["revenue"] * 100
    if has("operating_income", "revenue"):
        df["operating_margin"] = df["operating_income"] / df["revenue"] * 100
    if has("net_income", "revenue"):
        df["net_margin"] = df["net_income"] / df["revenue"] * 100
    if has("net_income", "equity"):
        df["roe"] = df["net_income"] / df["equity"] * 100
    if has("net_income", "assets"):
        df["roa"] = df["net_income"] / df["assets"] * 100
    if has("long_term_debt", "equity"):
        df["debt_to_equity"] = df["long_term_debt"] / df["equity"]
    if has("rnd", "revenue"):
        df["rnd_pct_revenue"] = df["rnd"] / df["revenue"] * 100
    if has("op_cash_flow", "revenue"):
        df["ocf_margin"] = df["op_cash_flow"] / df["revenue"] * 100
    if "revenue" in df.columns:
        df["revenue_growth"] = df["revenue"].pct_change() * 100
    if "net_income" in df.columns:
        df["net_income_growth"] = df["net_income"].pct_change() * 100


# Human-readable labels for derived columns (raw labels come from METRICS_BY_KEY).
DERIVED_LABELS = {
    "gross_margin": "Gross margin %",
    "operating_margin": "Operating margin %",
    "net_margin": "Net margin %",
    "roe": "Return on equity %",
    "roa": "Return on assets %",
    "debt_to_equity": "Debt / equity",
    "rnd_pct_revenue": "R&D % of revenue",
    "ocf_margin": "OCF margin %",
    "revenue_growth": "Revenue growth % YoY",
    "net_income_growth": "Net income growth % YoY",
}


def label_for(key: str) -> str:
    if key in METRICS_BY_KEY:
        return METRICS_BY_KEY[key].label
    return DERIVED_LABELS.get(key, key.replace("_", " ").title())
