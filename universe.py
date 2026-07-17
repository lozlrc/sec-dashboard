"""The company-picker universe: domestic 10-K filers on real exchanges.

Building the list live needs EDGAR's quarterly form indexes (~15-20s cold),
so deployments bundle a precomputed ``universe.json`` (see
``scripts/build_universe.py``); ``load_options()`` uses it when present and
falls back to the live build otherwise.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import edgar

UNIVERSE_FILE = Path(__file__).resolve().parent / "universe.json"
EXCHANGES_OK = {"Nasdaq", "NYSE", "CBOE"}
MAX_OPTIONS = 400

PRESETS = {
    "Big Tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"],
    "Semiconductors": ["NVDA", "AMD", "INTC", "AVGO", "QCOM", "MU"],
    "Banks": ["JPM", "BAC", "WFC", "GS", "MS"],
    "Retail": ["WMT", "COST", "TGT", "HD"],
    "Beverages": ["KO", "PEP", "MNST"],
    "Autos": ["TSLA", "F", "GM"],
}


def build_options() -> list[str]:
    """Compute 'TICKER — Company Name' options live from EDGAR.

    Two filters: a real exchange listing (drops OTC, mostly unsponsored ADRs
    of foreign companies) and a 10-K filed within the last ~5 quarters (drops
    foreign private issuers, who file 20-F/40-F under IFRS instead — the
    EDGAR path reads us-gaap 10-K data only).
    """
    mapping = edgar.load_ticker_map()
    exchanges = edgar.load_exchange_map()
    tenk_ciks = edgar.load_recent_10k_ciks()

    def eligible(ticker: str, info: dict) -> bool:
        return (exchanges.get(ticker, "") in EXCHANGES_OK
                and int(info["cik"]) in tenk_ciks)

    opts = [f"{t} — {v['title']}" for t, v in mapping.items() if eligible(t, v)][:MAX_OPTIONS]
    listed = {o.split(" — ")[0] for o in opts}
    for group in PRESETS.values():
        for t in group:
            if t not in listed and t in mapping:
                opts.append(f"{t} — {mapping[t]['title']}")
                listed.add(t)
    return opts


def save_universe(path: Path = UNIVERSE_FILE) -> list[str]:
    options = build_options()
    path.write_text(json.dumps(
        {"generated": date.today().isoformat(), "options": options}, indent=0))
    return options


def load_options() -> list[str]:
    """Bundled universe if available, else live build."""
    if UNIVERSE_FILE.exists():
        try:
            data = json.loads(UNIVERSE_FILE.read_text())
            options = data.get("options", [])
            if options:
                return options
        except (json.JSONDecodeError, OSError):
            pass
    return build_options()
