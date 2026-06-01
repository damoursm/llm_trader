"""Run build_signals twice on the same cached inputs and report any
score that drifts between calls. This isolates the AGGREGATOR's determinism
from the LLM-synthesis stage — anything that flips here is a determinism
bug in scoring, fetching, or ordering. Throwaway test script.
"""
from __future__ import annotations

from loguru import logger
logger.remove()

from src.signals.aggregator import build_signals
from src.data.cache import load_news


def run_once(tickers):
    articles = load_news() or []
    sigs = build_signals(tickers, articles)
    return {s.ticker: s.model_dump(mode="json") for s in sigs}


def diff(a: dict, b: dict, tol: float = 1e-9):
    diffs = {}
    keys = set(a) | set(b)
    for t in sorted(keys):
        if t not in a or t not in b:
            diffs[t] = {"_missing": [t in a, t in b]}
            continue
        per_field = {}
        for k in set(a[t]) | set(b[t]):
            va, vb = a[t].get(k), b[t].get(k)
            if isinstance(va, float) and isinstance(vb, float):
                if abs(va - vb) > tol:
                    per_field[k] = (va, vb)
            elif va != vb:
                per_field[k] = (va, vb)
        if per_field:
            diffs[t] = per_field
    return diffs


def main():
    # A representative slice — mix of pinned watchlist + sector ETFs + commodities
    # + one open-trade ticker so we exercise the OHLCV-refresh path too.
    tickers = ["NVDA", "AAPL", "MSFT", "JPM", "XLK", "XLF", "GLD", "SLV", "GOOGL", "INSM"]
    print(f"Running build_signals twice on {len(tickers)} tickers …")
    a = run_once(tickers)
    b = run_once(tickers)
    d = diff(a, b)
    if not d:
        print("PASS — both runs produced identical per-ticker scores.")
        return
    print(f"FAIL — {len(d)} ticker(s) drifted between runs:")
    for t, fields in d.items():
        print(f"\n  [{t}]")
        for k, (va, vb) in fields.items():
            print(f"    {k}: {va!r}  ->  {vb!r}")


if __name__ == "__main__":
    main()
