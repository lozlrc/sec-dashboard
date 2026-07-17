"""Extract annual financial metrics from a report PDF (10-K, annual report).

Heuristic text-layer parser: find statement-style lines ("<label> ... <n> <n>"),
match labels against metric synonyms, and assign the numeric columns to the
fiscal years announced in the nearest preceding year-header line. Works on
text-based PDFs (most annual reports); scanned/image PDFs are out of scope.

Extraction is best-effort by design — the UI shows the result in an editable
table so a human can fix anything the heuristics missed.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import pandas as pd
import pdfplumber

# Metric key -> label synonyms (lowercase, matched against the whole label).
# Order within a tuple = priority; first metric to claim a (key, year) wins.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "revenue": ("total revenues", "total revenue", "revenues", "revenue",
                "net sales", "total net sales", "turnover"),
    "cost_of_revenue": ("cost of revenues", "cost of revenue", "cost of sales",
                        "cost of goods sold"),
    "gross_profit": ("gross profit",),
    "operating_income": ("operating profit", "operating income", "income from operations",
                         "profit from operations"),
    "net_income": ("profit for the year", "profit attributable to equity holders",
                   "net income attributable", "net income", "net profit",
                   "profit for the period"),
    "rnd": ("research and development expenses", "research and development"),
    "op_cash_flow": ("net cash generated from operating activities",
                     "net cash provided by operating activities",
                     "net cash flows from operating activities",
                     "cash generated from operations"),
    "assets": ("total assets",),
    "liabilities": ("total liabilities",),
    "equity": ("total equity", "total shareholders' equity", "total stockholders' equity",
               "equity attributable to equity holders"),
    "cash": ("cash and cash equivalents",),
    "eps_diluted": ("diluted earnings per share", "diluted eps",
                    "earnings per share - diluted", "diluted (in"),
}

# EPS-style metrics are per-share: never scaled by the document multiplier.
PER_SHARE_KEYS = {"eps_diluted"}

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_NUM_RE = re.compile(r"\(?-?(?:US\$|HK\$|RMB|\$|€|£|¥)?\s?\d[\d,]*(?:\.\d+)?\)?%?")

_SCALE_PATTERNS = [
    (re.compile(r"in\s+billions|(?:US\$|RMB|HK\$)\s*billions?", re.I), 1e9),
    (re.compile(r"in\s+millions|(?:US\$|RMB|HK\$)\s*millions?", re.I), 1e6),
    (re.compile(r"in\s+thousands|'000|’000", re.I), 1e3),
]
_CURRENCY_PATTERNS = [
    (re.compile(r"\bRMB\b|renminbi", re.I), "RMB"),
    (re.compile(r"\bHK\$|hong kong dollar", re.I), "HKD"),
    (re.compile(r"\bUS\$|U\.S\. dollar|\bUSD\b", re.I), "USD"),
    (re.compile(r"€|\beuro", re.I), "EUR"),
    (re.compile(r"£"), "GBP"),
]


@dataclass
class PdfExtraction:
    df: pd.DataFrame  # index = fiscal year, columns = metric keys (absolute units)
    currency: str = "USD"
    scale: float = 1.0
    pages_matched: list[int] = field(default_factory=list)
    rows_matched: int = 0

    @property
    def ok(self) -> bool:
        return not self.df.empty


def _parse_number(tok: str) -> float | None:
    tok = tok.strip().rstrip("%")
    neg = tok.startswith("(") and tok.endswith(")")
    tok = tok.strip("()")
    tok = re.sub(r"US\$|HK\$|RMB|\$|€|£|¥|\s", "", tok)
    if not tok or tok in {"-", "—"}:
        return None
    try:
        val = float(tok.replace(",", ""))
    except ValueError:
        return None
    return -val if neg else val


def _year_header(line: str) -> list[int] | None:
    """A line announcing year columns, e.g. '2023 2022' or 'FY2024 FY2023'."""
    years = [int(m.group(0)) for m in _YEAR_RE.finditer(line)]
    if len(years) < 1:
        return None
    # Reject prose: the line should be mostly years/labels, not a sentence.
    words = line.split()
    if len(words) > 12:
        return None
    plausible = [y for y in years if 1995 <= y <= 2035]
    # Statement headers list recent, distinct, descending-ish years.
    if len(plausible) >= 1 and len(set(plausible)) == len(plausible):
        return plausible
    return None


def _match_metric(label: str) -> str | None:
    """Longest matching synonym wins, so 'cost of revenues' is never
    claimed by the shorter 'revenues' synonym of the revenue metric."""
    label = label.lower().strip()
    label = re.sub(r"\(note\s*\d+\)|note\s*\d+", "", label)  # strip note refs
    label = re.sub(r"[^a-z'\- ()]", " ", label).strip()
    best: tuple[int, str] | None = None
    for key, names in SYNONYMS.items():
        for name in names:
            if name in label and (best is None or len(name) > best[0]):
                best = (len(name), key)
    return best[1] if best else None


def _extract_rows(line: str) -> tuple[str, list[float]] | None:
    """Split a statement line into (label, numeric columns)."""
    tokens = _NUM_RE.findall(line)
    if not tokens:
        return None
    first = _NUM_RE.search(line)
    label = line[: first.start()].strip(" .·…")
    if len(label) < 4:
        return None
    values = [v for v in (_parse_number(t) for t in tokens) if v is not None]
    if not values:
        return None
    return label, values


def parse_pdf(data: bytes, max_pages: int = 250) -> PdfExtraction:
    """Parse a financial-report PDF into an annual metrics DataFrame."""
    points: dict[tuple[str, int], float] = {}
    pages_matched: set[int] = set()
    rows_matched = 0
    currency = None
    scale = None

    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page_no, page in enumerate(pdf.pages[:max_pages], start=1):
            text = page.extract_text() or ""
            if not text:
                continue
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            years: list[int] = []
            page_has_statement_words = bool(
                re.search(r"revenue|net sales|total assets|operating activities|turnover",
                          text, re.I))
            if not page_has_statement_words:
                continue
            if scale is None:
                for pat, mult in _SCALE_PATTERNS:
                    if pat.search(text):
                        scale = mult
                        break
            if currency is None:
                for pat, cur in _CURRENCY_PATTERNS:
                    if pat.search(text):
                        currency = cur
                        break

            for line in lines:
                hdr = _year_header(line)
                if hdr:
                    years = hdr
                    continue
                if not years:
                    continue
                row = _extract_rows(line)
                if not row:
                    continue
                label, values = row
                key = _match_metric(label)
                if not key:
                    continue
                # Drop a leading note-reference column: one extra small int
                # in front of plausibly-large figures (e.g. "Revenues 5 86,392 82,186").
                if len(values) == len(years) + 1 and abs(values[0]) < 100 and values[0] == int(values[0]):
                    values = values[1:]
                if len(values) < len(years):
                    continue
                values = values[: len(years)]
                for yr, val in zip(years, values):
                    points.setdefault((key, yr), val)
                    rows_matched += 1
                    pages_matched.add(page_no)

    if not points:
        return PdfExtraction(df=pd.DataFrame(), currency=currency or "USD",
                             scale=scale or 1.0)

    scale = scale or 1.0
    by_metric: dict[str, dict[int, float]] = {}
    for (key, yr), val in points.items():
        mult = 1.0 if key in PER_SHARE_KEYS else scale
        by_metric.setdefault(key, {})[yr] = val * mult
    df = pd.DataFrame(by_metric).sort_index()
    df.index.name = "fiscal_year"
    return PdfExtraction(df=df, currency=currency or "USD", scale=scale,
                         pages_matched=sorted(pages_matched), rows_matched=rows_matched)


def add_pdf_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Derived margins for a PDF-extracted frame (subset of metrics.py logic)."""
    out = df.copy()
    if "gross_profit" in out and "revenue" in out:
        out["gross_margin"] = out["gross_profit"] / out["revenue"] * 100
    elif "cost_of_revenue" in out and "revenue" in out:
        # IFRS-style statements report costs in parentheses (negative)
        out["gross_margin"] = (out["revenue"] - out["cost_of_revenue"].abs()) / out["revenue"] * 100
    if "operating_income" in out and "revenue" in out:
        out["operating_margin"] = out["operating_income"] / out["revenue"] * 100
    if "net_income" in out and "revenue" in out:
        out["net_margin"] = out["net_income"] / out["revenue"] * 100
    if "revenue" in out:
        out["revenue_growth"] = out["revenue"].pct_change() * 100
    return out
