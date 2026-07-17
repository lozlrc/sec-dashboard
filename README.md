# SEC Filings Dashboard

Pulls companies' financial filings straight from the **SEC EDGAR** XBRL API and
turns them into an interactive dashboard — fundamentals, margins, growth, and
balance-sheet trends for any US-listed ticker, plus side-by-side peer
comparison and a **PDF import mode** for annual reports that never hit EDGAR.
No paid data vendor; everything comes from primary-source filings.

## What it does

**📈 Overview** — pick a company: KPI tiles with YoY deltas (revenue, net
income, margin, EPS, assets), revenue & net income, profitability margins, YoY
revenue growth, balance sheet, an earnings-quality panel (operating cash flow
vs net income), a full financials table with a YoY-%-change toggle, CSV
export, and links to the underlying filings on SEC.gov.

**⚖️ Compare** — select up to 8 companies from a searchable picker (or load a
peer-group preset: Big Tech, Semiconductors, Banks, Retail, …) and choose
which metrics to chart: revenue, growth, margins, ROE/ROA, debt/equity, R&D
intensity, OCF margin, EPS, and more, rendered as a small-multiples grid with
an indexed-revenue (first year = 100) view and a latest-fiscal-year table.

**📄 PDF import** — drop in a text-based annual report (10-K or a foreign
filer's report, e.g. a Tencent annual report in RMB). A heuristic parser finds
statement line items, detects currency and scale ("RMB in millions"), strips
note-reference columns, treats parenthesized figures as negatives, and keeps
per-share values unscaled. Results land in an **editable table** — fix
anything the parser missed — and chart the same way as EDGAR data.

## Architecture

| File | Responsibility |
|------|----------------|
| `edgar.py` | SEC EDGAR REST client — ticker→CIK resolution, company-facts, recent filings. On-disk TTL cache + backoff to respect fair-access limits. Pure module, no UI. |
| `metrics.py` | Maps raw XBRL tags to clean financial metrics, one figure per fiscal year, plus derived margins/growth/returns. |
| `pdf_extract.py` | Heuristic annual-report PDF parser (pdfplumber): synonym matching, year-header detection, currency/scale detection. |
| `app.py` | Streamlit + Plotly dashboard (three tabs). |
| `tests/` | Parser tests against a synthetic annual-report PDF; `smoke_test.py` checks live EDGAR extraction. |

The non-trivial parts are in extraction:

- **Tag drift** — companies tag the same concept with different us-gaap
  elements *and rename them over time* (MSFT moved cost of revenue from
  `CostOfRevenue` to `CostOfGoodsAndServicesSold` in 2018). Each metric lists
  candidate tags; annual points are merged across tags with the
  higher-priority tag winning a year.
- **Phantom fiscal years** — 10-Qs carry ~365-day LTM comparison windows that
  look like fiscal years; only 10-K values are accepted.
- **Restatements** — duplicate (metric, period) values are de-duplicated
  keeping the latest-filed figure.
- **PDF label collisions** — "Cost of revenues" must not be claimed by the
  shorter "revenues" synonym; longest-synonym-wins matching.

## Run it

```bash
# uv handles the environment (https://docs.astral.sh/uv/)
export SEC_USER_AGENT="Your Name your@email.com"   # SEC requires a contact string
uv run streamlit run app.py --server.port 8601
```

Then open http://localhost:8601. First load per company hits the SEC API;
results are cached locally for 24h.

```bash
uv run pytest            # PDF parser tests
uv run python smoke_test.py   # live extraction check against EDGAR
```

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub (public).
2. At [share.streamlit.io](https://share.streamlit.io) → **New app**, pick the
   repo, branch `main`, main file `app.py`, Python 3.12.
3. In **Advanced settings → Secrets**, add your SEC contact string:

   ```toml
   SEC_USER_AGENT = "Your Name your@email.com"
   ```

4. Deploy. Dependencies come from `requirements.txt`
   (regenerate with `uv export --no-dev --no-hashes -o requirements.txt`).

Cold starts stay fast because the company-picker list ships precomputed in
`universe.json`. Refresh it occasionally (new IPOs appear, delistings drop):

```bash
uv run python scripts/build_universe.py   # then commit universe.json
```

## Design notes

- **Fiscal years** are labeled by period-end year for cross-company alignment.
- Charts follow a single-axis, colorblind-safe convention — no dual y-axes;
  categorical hues assigned in fixed order.
- **The company picker only offers domestic 10-K filers.** Eligibility is
  computed from EDGAR's quarterly form indexes (filed a 10-K in the last ~5
  quarters) plus a real exchange listing — which cleanly excludes foreign
  private issuers (20-F/IFRS) and OTC ADRs rather than letting them error at
  load time. Foreign companies are what the PDF import mode is for.

## Data source

[SEC EDGAR](https://www.sec.gov/edgar) `company-facts` and `submissions` APIs.
Public-domain filing data; usage is subject to the SEC's
[fair-access policy](https://www.sec.gov/os/webmaster-faq#developers).
