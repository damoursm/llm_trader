"""
Related-company peer discovery — Massive/Polygon's related-companies graph.

For each seed ticker, pulls up to ~10 peers (one call/seed) and returns a deduped,
validated list of NEW symbols to widen the universe. The caller (pipeline Step 0)
routes the result through the liquidity gate, so untradeable microcap peers are
dropped — never injected raw (cf. the smart-money universe-leak incident).

Cached daily. Returns [] when disabled / no key.
"""

import json
from datetime import date
from pathlib import Path
from typing import List

from loguru import logger

from config import settings
from src.data import polygon_client
from src.data.market_data import is_valid_ticker

CACHE_DIR = Path("cache")


def _cache_path() -> Path:
    return CACHE_DIR / f"related_discovery_{date.today().isoformat()}.json"


def discover_related_tickers(seed_tickers: List[str], max_results: int = 25) -> List[str]:
    """Peers of the seed names (Massive related-companies), deduped/validated/capped.

    Excludes the seeds themselves and anything already seen. Cached daily. Returns []
    when disabled or Polygon is unavailable. The caller gates the result for liquidity
    before adding it to the universe."""
    if not settings.enable_related_discovery or not polygon_client.is_available():
        return []

    path = _cache_path()
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            # The cache is date-keyed (seeds are stable within a day), but defend
            # against a seed leaking back in if today's seed set differs from the
            # one that built the cache.
            _seeds = {s.upper() for s in seed_tickers}
            cached = [t for t in cached if t.upper() not in _seeds]
            logger.info(f"[related] loaded {len(cached)} cached peer(s)")
            return cached
        except Exception as e:
            logger.warning(f"[related] cache load failed: {e}")

    if not settings.enable_fetch_data:
        return []

    seeds = [s.upper() for s in dict.fromkeys(seed_tickers) if is_valid_ticker(s)]
    seen = set(seeds)
    peers: List[str] = []
    for s in seeds:
        for p in polygon_client.get_related_companies(s):
            pu = p.upper()
            if pu not in seen and is_valid_ticker(pu):
                seen.add(pu)
                peers.append(pu)
    peers = peers[:max_results]

    CACHE_DIR.mkdir(exist_ok=True)
    try:
        path.write_text(json.dumps(peers, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"[related] cache save failed: {e}")

    logger.info(f"[related] {len(peers)} peer(s) from {len(seeds)} seed(s)")
    return peers
