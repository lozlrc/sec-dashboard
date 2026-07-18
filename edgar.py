"""Client for the SEC EDGAR REST APIs.

Pure module (no Streamlit) so it stays testable and reusable. Handles
ticker->CIK resolution, XBRL company-facts, and recent-filing metadata,
with a polite on-disk cache and backoff to respect SEC fair-access limits.

SEC requires a descriptive User-Agent that identifies the caller. Set the
SEC_USER_AGENT env var to "Your Name your@email.com"; see
https://www.sec.gov/os/webmaster-faq#developers
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_TICKERS_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

CACHE_DIR = Path(os.environ.get("SEC_CACHE_DIR", Path.home() / ".cache" / "sec-dashboard"))
CACHE_TTL_SECONDS = int(os.environ.get("SEC_CACHE_TTL", 24 * 3600))

# SEC asks for <=10 req/s; we stay far below and cache aggressively.
_MIN_INTERVAL = 0.15
_last_request = 0.0
_session: requests.Session | None = None


class EdgarError(RuntimeError):
    """Raised when EDGAR data cannot be retrieved or parsed."""


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        # Env is read lazily so the app can inject SEC_USER_AGENT (e.g. from
        # Streamlit secrets) after import but before the first request.
        ua = os.environ.get("SEC_USER_AGENT", "sec-dashboard admin@example.com")
        s = requests.Session()
        s.headers.update({"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
        _session = s
    return _session


def _cache_path(url: str) -> Path:
    slug = url.replace("https://", "").replace("/", "_").replace(":", "_")
    return CACHE_DIR / f"{slug}.json"


def _get_json(url: str, use_cache: bool = True) -> dict | list:
    """GET JSON with an on-disk TTL cache, polite pacing, and backoff."""
    global _last_request
    cache_file = _cache_path(url)
    if use_cache and cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            try:
                return json.loads(cache_file.read_text())
            except (json.JSONDecodeError, OSError):
                pass  # corrupt cache -> refetch

    last_err: Exception | None = None
    for attempt in range(4):
        wait = _MIN_INTERVAL - (time.time() - _last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = _get_session().get(url, timeout=30)
            _last_request = time.time()
            if resp.status_code == 200:
                data = resp.json()
                try:
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(json.dumps(data))
                except OSError:
                    pass  # cache is best-effort
                return data
            if resp.status_code == 404:
                raise EdgarError(f"Not found (404): {url}")
            if resp.status_code in (403, 429, 500, 502, 503):
                last_err = EdgarError(f"HTTP {resp.status_code} from {url}")
                time.sleep(1.5 * (attempt + 1))
                continue
            resp.raise_for_status()
        except requests.RequestException as exc:
            last_err = exc
            time.sleep(1.0 * (attempt + 1))
    raise EdgarError(f"Failed to fetch {url}: {last_err}")


def load_ticker_map() -> dict[str, dict]:
    """Return {TICKER: {"cik": "0000320193", "title": "Apple Inc."}}."""
    raw = _get_json(SEC_TICKERS_URL)
    out: dict[str, dict] = {}
    # company_tickers.json is a dict keyed by row index.
    rows = raw.values() if isinstance(raw, dict) else raw
    for row in rows:
        ticker = str(row["ticker"]).upper()
        out[ticker] = {"cik": f"{int(row['cik_str']):010d}", "title": row["title"]}
    return out


def load_exchange_map() -> dict[str, str]:
    """Return {TICKER: exchange} ('Nasdaq', 'NYSE', 'OTC', 'CBOE', or '')."""
    raw = _get_json(SEC_TICKERS_EXCHANGE_URL)
    fields, data = raw["fields"], raw["data"]
    i_t, i_e = fields.index("ticker"), fields.index("exchange")
    return {str(r[i_t]).upper(): (r[i_e] or "") for r in data}


def _get_text(url: str) -> str:
    """GET plain text with the same on-disk TTL cache and pacing as _get_json."""
    global _last_request
    cache_file = _cache_path(url).with_suffix(".txt")
    if cache_file.exists() and time.time() - cache_file.stat().st_mtime < CACHE_TTL_SECONDS:
        try:
            return cache_file.read_text()
        except OSError:
            pass
    last_err: Exception | None = None
    for attempt in range(4):
        wait = _MIN_INTERVAL - (time.time() - _last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = _get_session().get(url, timeout=60)
            _last_request = time.time()
            if resp.status_code == 200:
                try:
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(resp.text)
                except OSError:
                    pass
                return resp.text
            if resp.status_code == 404:
                raise EdgarError(f"Not found (404): {url}")
            last_err = EdgarError(f"HTTP {resp.status_code} from {url}")
            time.sleep(1.5 * (attempt + 1))
        except requests.RequestException as exc:
            last_err = exc
            time.sleep(1.0 * (attempt + 1))
    raise EdgarError(f"Failed to fetch {url}: {last_err}")


def get_document(url: str) -> str:
    """Fetch a filing's primary document (HTML/text), cached like other calls."""
    return _get_text(url)


def load_recent_10k_ciks(quarters: int = 5) -> set[int]:
    """CIKs that filed a 10-K in the last `quarters` quarterly form indexes.

    This is the authoritative "domestic filer" test: foreign private issuers
    file 20-F/40-F instead of 10-K, and our extractor needs 10-K data anyway.
    """
    from datetime import date

    today = date.today()
    year, qtr = today.year, (today.month - 1) // 3 + 1
    ciks: set[int] = set()
    for _ in range(quarters):
        url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{qtr}/form.idx"
        try:
            text = _get_text(url)
        except EdgarError:
            text = ""  # index for a brand-new quarter may not exist yet
        for line in text.splitlines():
            if not line.startswith("10-K"):
                continue
            # Fixed-width row ending in ".../edgar/data/<CIK>/<accession>.txt"
            path = line.rsplit(None, 1)[-1]
            parts = path.split("/")
            if len(parts) >= 3 and parts[0] == "edgar" and parts[2].isdigit():
                ciks.add(int(parts[2]))
        qtr -= 1
        if qtr == 0:
            year, qtr = year - 1, 4
    return ciks


def resolve_cik(ticker: str) -> dict:
    """Resolve a ticker symbol to {"cik", "title", "ticker"}."""
    ticker = ticker.strip().upper()
    if not ticker:
        raise EdgarError("Empty ticker")
    mapping = load_ticker_map()
    if ticker not in mapping:
        raise EdgarError(f"Ticker '{ticker}' not found in SEC company list")
    entry = mapping[ticker]
    return {"cik": entry["cik"], "title": entry["title"], "ticker": ticker}


def get_company_facts(cik: str) -> dict:
    """Return the raw XBRL company-facts document for a 10-digit CIK."""
    cik = f"{int(cik):010d}"
    return _get_json(SEC_FACTS_URL.format(cik=cik))


def get_recent_filings(cik: str, forms=("10-K", "10-Q"), limit: int = 15) -> list[dict]:
    """Return recent filings (metadata + document URL) for the given forms."""
    cik = f"{int(cik):010d}"
    data = _get_json(SEC_SUBMISSIONS_URL.format(cik=cik))
    recent = data.get("filings", {}).get("recent", {})
    cik_int = int(cik)
    out: list[dict] = []
    for form, acc, doc, date, primary_desc in zip(
        recent.get("form", []),
        recent.get("accessionNumber", []),
        recent.get("primaryDocument", []),
        recent.get("filingDate", []),
        recent.get("primaryDocDescription", []),
    ):
        if forms and form not in forms:
            continue
        acc_nodash = acc.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"
        out.append(
            {"form": form, "filed": date, "accession": acc, "description": primary_desc, "url": url}
        )
        if len(out) >= limit:
            break
    return out
