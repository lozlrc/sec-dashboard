"""Precompute universe.json (the company-picker list) for fast cold starts.

Run occasionally (e.g. before a deploy) and commit the result:

    export SEC_USER_AGENT="Your Name your@email.com"
    uv run python scripts/build_universe.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import universe

if __name__ == "__main__":
    options = universe.save_universe()
    print(f"Wrote {universe.UNIVERSE_FILE.name}: {len(options)} companies")
