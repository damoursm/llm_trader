"""Dump the per-ticker scores to JSON. Two invocations of this script
should produce identical files when caches are warm. Comparing them
reveals every source of cross-process non-determinism that affects scores.
"""
from __future__ import annotations

import json
import sys
from loguru import logger
logger.remove()

from src.signals.aggregator import build_signals
from src.data.cache import load_news


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "scripts/_signals_dump.json"
    tickers = ["NVDA", "AAPL", "MSFT", "JPM", "XLK", "XLF", "GLD", "SLV", "GOOGL", "INSM"]
    sigs = build_signals(tickers, load_news() or [])
    data = {s.ticker: s.model_dump(mode="json") for s in sigs}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)
    print(f"Wrote {len(data)} ticker signals to {out_path}")


if __name__ == "__main__":
    main()
