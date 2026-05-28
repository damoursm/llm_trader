"""Ticker → sector ETF benchmark resolver.

Used by sector-relative momentum (and any future "strip out beta" signal) to
isolate idiosyncratic alpha from sector-driven beta moves.

Resolution order:
  1. Hand-curated ``_SECTOR_MAP`` from ``aggregator`` (the popular names).
  2. Cached yfinance ``info["sector"]`` lookup, mapped to the SPDR sector ETF.
  3. SPY fallback for stocks the broader lookup can't classify.
  4. ``None`` for ETFs / commodities — they have no meaningful equity sector.

The cache file ``cache/sector_benchmark_map.json`` persists forever
(sectors essentially don't change). To force a re-lookup, delete the file
or remove the specific ticker entry. Per-run lookups are bounded so a cold
cache doesn't add minutes to the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

from config import settings

CACHE_FILE = Path("cache/sector_benchmark_map.json")

# Lookup limit per pipeline run — sector lookups are slow (~0.5–1 s each)
# and not essential. After the cache warms up over a few runs, every ticker
# the system sees will be resolved without any new fetches.
_MAX_LOOKUPS_PER_RUN = 25
_lookups_done = {"n": 0}

# yfinance sector strings → SPDR sector ETF
_SECTOR_TO_SPDR: Dict[str, str] = {
    "Technology":             "XLK",
    "Financial Services":     "XLF",
    "Financial":              "XLF",
    "Healthcare":             "XLV",
    "Health Care":            "XLV",
    "Consumer Cyclical":      "XLY",
    "Consumer Discretionary": "XLY",
    "Consumer Defensive":     "XLP",
    "Consumer Staples":       "XLP",
    "Industrials":            "XLI",
    "Energy":                 "XLE",
    "Basic Materials":        "XLB",
    "Materials":              "XLB",
    "Utilities":              "XLU",
    "Real Estate":            "XLRE",
    "Communication Services": "XLC",
    "Communications":         "XLC",
}


def _load_cache() -> Dict[str, str]:
    if not CACHE_FILE.exists():
        return {}
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"[sector_bench] Could not load cache: {e}")
        return {}


def _save_cache(cache: Dict[str, str]) -> None:
    CACHE_FILE.parent.mkdir(exist_ok=True)
    try:
        CACHE_FILE.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as e:
        logger.warning(f"[sector_bench] Could not save cache: {e}")


_CACHE: Dict[str, str] = _load_cache()


def _seed_from_aggregator() -> Dict[str, str]:
    """Pull the hand-curated mapping in aggregator._SECTOR_MAP."""
    try:
        from src.signals.aggregator import _SECTOR_MAP
        return {t.upper(): etf.upper() for t, etf in _SECTOR_MAP.items()}
    except Exception:
        return {}


_SEED = _seed_from_aggregator()


def _yfinance_sector(ticker: str) -> Optional[str]:
    """Look up the ticker's sector via yfinance. ``None`` on any failure.

    Bounded by ``_MAX_LOOKUPS_PER_RUN`` so a cold cache warm-up doesn't
    snowball into a multi-minute fetch.
    """
    if _lookups_done["n"] >= _MAX_LOOKUPS_PER_RUN:
        return None
    if not settings.enable_fetch_data:
        return None
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        _lookups_done["n"] += 1
        sector = info.get("sector") or info.get("category")
        if isinstance(sector, str) and sector.strip():
            return sector.strip()
    except Exception as e:
        logger.debug(f"[sector_bench] yfinance sector lookup failed for {ticker}: {e}")
    return None


def _infer_asset_type(ticker: str) -> str:
    """Best-effort classification for callers that don't track asset type.

    Matches the conventions used by tracker._sector_key / hypothetical_tracker
    so the benchmark resolution is consistent across the codebase.
    """
    tk = ticker.upper()
    try:
        if tk in {s.upper() for s in settings.commodities_list}:
            return "COMMODITY"
        if tk in {s.upper() for s in settings.sectors_list}:
            return "ETF"
        if tk in {s.upper() for s in settings.factor_list}:
            return "ETF"
    except Exception:
        pass
    return "STOCK"


def get_sector_benchmark(ticker: str, asset_type: Optional[str] = None) -> Optional[str]:
    """Return the benchmark ETF for *ticker*, or ``None`` when none applies.

    * STOCK: sector ETF via map / yfinance lookup, falling back to SPY.
    * ETF / factor / sector ETF: SPY (broad-market benchmark).
    * COMMODITY: ``None`` — no equity sector benchmark is meaningful.

    When ``asset_type`` is ``None`` it's inferred from settings (commodities /
    sector / factor lists), so callers that don't track type don't have to.

    Lookups are cached in ``cache/sector_benchmark_map.json`` so subsequent
    runs are instant.
    """
    if not ticker:
        return None
    tk = ticker.upper()
    atype = (asset_type or _infer_asset_type(tk)).upper()

    # SPY is the universal benchmark; benchmarking it against itself adds no signal.
    if tk == "SPY":
        return None
    if atype == "COMMODITY":
        return None

    # ETF: benchmark against SPY (broad market). Avoids the awkward XLK-vs-XLK
    # self-comparison and lets factor ETFs (MTUM, QUAL) get a real read.
    if atype == "ETF":
        return None if tk == "SPY" else "SPY"

    # STOCK — try cached → seed → yfinance → SPY.
    if tk in _CACHE:
        b = _CACHE[tk]
        return b if b else None       # explicit empty string = "no benchmark"
    if tk in _SEED:
        bench = _SEED[tk]
        _CACHE[tk] = bench
        _save_cache(_CACHE)
        return bench

    sector = _yfinance_sector(tk)
    if sector:
        bench = _SECTOR_TO_SPDR.get(sector, "SPY")
        _CACHE[tk] = bench
        _save_cache(_CACHE)
        return bench

    # Unknown sector and lookup unavailable — fall back to SPY without caching
    # (so the next run can try the lookup again).
    return "SPY"


def reset_lookup_counter() -> None:
    """Reset the per-run lookup budget. Called from the pipeline at start of run."""
    _lookups_done["n"] = 0
