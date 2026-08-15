"""Ticker → sector-ETF benchmark resolver (`src/signals/sector_benchmark.py`).

This is the denominator of `sector_relative_momentum` (a weighted method), so a
wrong answer here does not error — it silently measures the stock against the
wrong thing. Two failure modes matter and neither is visible downstream:

* **benchmarking a name against itself** (XLK vs XLK, SPY vs SPY) yields a
  structural ~0 score that reads as "no relative strength" rather than "no
  measurement";
* **caching a fallback** would freeze a ticker on SPY forever, because the cache
  file is documented to persist indefinitely. The module deliberately returns
  the SPY fallback WITHOUT caching so the next run can retry the lookup, and
  that distinction is invisible from the return value alone.

Every test isolates the three pieces of module-level state — the cache dict
loaded at import, the cache FILE (a relative `cache/…` path that would otherwise
be the developer's real one) and the per-run lookup budget.
"""

from __future__ import annotations

import json

import pytest

from src.signals import sector_benchmark as sb


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Yields the REAL `_yfinance_sector` so the few tests that exercise the
    lookup itself can restore it; everything else gets a fail-fast stub, because
    a stray network call in this suite would be slow, flaky and untraceable."""
    real_lookup = sb._yfinance_sector
    monkeypatch.setattr(sb, "CACHE_FILE", tmp_path / "sector_benchmark_map.json")
    monkeypatch.setattr(sb, "_CACHE", {})
    monkeypatch.setattr(sb, "_lookups_done", {"n": 0})
    monkeypatch.setattr(sb, "_SEED_CACHE", {})
    monkeypatch.setattr(sb, "_yfinance_sector",
                        lambda t: pytest.fail(f"unexpected yfinance lookup for {t}"))
    yield real_lookup


# ── the self-benchmark cases ────────────────────────────────────────────────

def test_spy_has_no_benchmark():
    """SPY is the universal benchmark; measuring it against itself is a constant
    zero dressed up as a signal."""
    assert sb.get_sector_benchmark("SPY") is None
    assert sb.get_sector_benchmark("SPY", asset_type="ETF") is None
    assert sb.get_sector_benchmark("SPY", asset_type="STOCK") is None


def test_etfs_benchmark_against_the_broad_market_not_themselves():
    """A sector ETF's own sector ETF is itself — so ETFs get SPY, which also
    lets factor ETFs (MTUM, QUAL) get a real read."""
    assert sb.get_sector_benchmark("XLK", asset_type="ETF") == "SPY"
    assert sb.get_sector_benchmark("MTUM", asset_type="ETF") == "SPY"


def test_commodities_have_no_equity_sector():
    assert sb.get_sector_benchmark("GLD", asset_type="COMMODITY") is None


def test_empty_ticker_is_none():
    assert sb.get_sector_benchmark("") is None
    assert sb.get_sector_benchmark(None) is None


# ── asset-type inference ────────────────────────────────────────────────────

def test_asset_type_is_inferred_from_settings_when_not_supplied():
    """Callers that don't track asset type must still get the right answer —
    the inference reads the same commodity / sector / factor lists the rest of
    the system uses."""
    from config.settings import settings
    assert sb._infer_asset_type(settings.commodities_list[0]) == "COMMODITY"
    assert sb._infer_asset_type(settings.sectors_list[0]) == "ETF"
    assert sb._infer_asset_type(settings.factor_list[0]) == "ETF"
    assert sb._infer_asset_type("MSFT") == "STOCK"
    # ...and the inferred type reaches the resolution, uncached.
    assert sb.get_sector_benchmark(settings.commodities_list[0]) is None
    assert sb.get_sector_benchmark(settings.sectors_list[0]) == "SPY"


def test_explicit_asset_type_overrides_inference():
    from config.settings import settings
    etf = settings.sectors_list[0]
    assert sb.get_sector_benchmark(etf, asset_type="COMMODITY") is None


# ── resolution order: cache → seed → lookup → fallback ──────────────────────

def test_cache_hit_short_circuits_everything():
    sb._CACHE["ZZZZ"] = "XLV"
    assert sb.get_sector_benchmark("ZZZZ") == "XLV"        # no lookup (fixture fails)
    assert sb.get_sector_benchmark("zzzz") == "XLV"        # case-folded


def test_cached_empty_string_means_deliberately_no_benchmark():
    """An empty cache value is a recorded 'this ticker has none', distinct from
    a missing key (which means 'not looked up yet'). Returning "" would make
    the caller benchmark against a ticker named ''."""
    sb._CACHE["ZZZZ"] = ""
    assert sb.get_sector_benchmark("ZZZZ") is None


def test_hand_curated_seed_is_used_and_then_cached():
    """The aggregator's `_SECTOR_MAP` is authoritative for the popular names —
    it must beat a yfinance lookup, not just precede it."""
    from src.signals.aggregator import _SECTOR_MAP
    ticker, expected = next(iter(_SECTOR_MAP.items()))
    assert sb.get_sector_benchmark(ticker, asset_type="STOCK") == expected.upper()
    assert sb._CACHE[ticker.upper()] == expected.upper()
    assert json.loads(sb.CACHE_FILE.read_text(encoding="utf-8"))[ticker.upper()] \
        == expected.upper()


def test_seed_is_populated_despite_the_circular_import():
    """The 2026-08-14 bug, pinned. `aggregator` imports this module (via
    sector_relative_momentum) BEFORE it defines `_SECTOR_MAP`, so resolving the
    seed at import time raised, hit the bare `except`, and left it empty in
    every process — silently, forever. Nothing downstream could tell: the
    curated names still resolved, just through yfinance, so UBER sat on XLK
    where the map says XLY. Resolving lazily is what fixes it, and importing
    the aggregator FIRST here reproduces the exact order that broke."""
    import src.signals.aggregator  # noqa: F401  (establish the import order)
    from src.signals.aggregator import _SECTOR_MAP

    assert sb._seed(), "seed is empty — the aggregator map import broke silently"
    assert set(sb._seed()) == {t.upper() for t in _SECTOR_MAP}
    assert len(sb._seed()) == len(_SECTOR_MAP)


def test_curated_map_overrides_a_stale_cached_benchmark():
    """The cache never expires, so a wrong yfinance answer is permanent unless
    the curated map outranks it. This is the one mechanism for correcting a
    misclassification (UBER: yfinance says Technology/XLK, the map says XLY),
    and it only works if the seed is consulted BEFORE the cache."""
    from src.signals.aggregator import _SECTOR_MAP
    ticker, expected = next(iter(_SECTOR_MAP.items()))
    sb._CACHE[ticker.upper()] = "XLU"                  # a stale, wrong answer
    assert sb.get_sector_benchmark(ticker, asset_type="STOCK") == expected.upper()
    assert sb._CACHE[ticker.upper()] == expected.upper(), "correction not persisted"


@pytest.mark.parametrize("sector,expected", [
    ("Technology", "XLK"),
    ("Financial Services", "XLF"),
    ("Healthcare", "XLV"),
    ("Consumer Cyclical", "XLY"),
    ("Communication Services", "XLC"),
    ("Real Estate", "XLRE"),
])
def test_known_yfinance_sectors_map_to_their_spdr(monkeypatch, sector, expected):
    monkeypatch.setattr(sb, "_yfinance_sector", lambda t: sector)
    assert sb.get_sector_benchmark("ZZZZ", asset_type="STOCK") == expected
    assert sb._CACHE["ZZZZ"] == expected              # resolved once, kept


def test_unrecognised_sector_falls_back_to_spy_and_is_cached(monkeypatch):
    """The lookup SUCCEEDED — the sector string is simply not one we map. That
    is a settled answer, so caching it (unlike the no-lookup fallback below) is
    correct: retrying would return the same unknown string forever."""
    monkeypatch.setattr(sb, "_yfinance_sector", lambda t: "Shell Companies")
    assert sb.get_sector_benchmark("ZZZZ", asset_type="STOCK") == "SPY"
    assert sb._CACHE["ZZZZ"] == "SPY"


def test_failed_lookup_falls_back_to_spy_WITHOUT_caching(monkeypatch):
    """The load-bearing distinction. The cache file is documented to persist
    forever, so writing a fallback born of an unavailable lookup would pin the
    ticker to SPY permanently — the next run must be free to try again."""
    monkeypatch.setattr(sb, "_yfinance_sector", lambda t: None)
    assert sb.get_sector_benchmark("ZZZZ", asset_type="STOCK") == "SPY"
    assert "ZZZZ" not in sb._CACHE
    assert not sb.CACHE_FILE.exists()


# ── the per-run lookup budget ───────────────────────────────────────────────

def _fake_yfinance(monkeypatch, ticker_factory):
    import sys
    monkeypatch.setitem(sys.modules, "yfinance",
                        type("m", (), {"Ticker": ticker_factory}))


def test_lookup_budget_is_bounded_and_resettable(_isolated, monkeypatch):
    """A cold cache must not add minutes to a tick: lookups are capped per run.
    Past the cap `_yfinance_sector` returns None WITHOUT calling yfinance, so
    the caller quietly gets the SPY fallback."""
    calls = {"n": 0}

    class _FakeTicker:
        def __init__(self, t):
            pass

        @property
        def info(self):
            calls["n"] += 1
            return {"sector": "Technology"}

    monkeypatch.setattr(sb, "_yfinance_sector", _isolated)   # the real one
    _fake_yfinance(monkeypatch, _FakeTicker)

    for i in range(sb._MAX_LOOKUPS_PER_RUN + 5):
        sb._yfinance_sector(f"T{i}")
    assert calls["n"] == sb._MAX_LOOKUPS_PER_RUN, "budget did not bound the fetches"

    sb.reset_lookup_counter()
    assert sb._lookups_done["n"] == 0
    assert sb._yfinance_sector("AGAIN") == "Technology"


def test_no_lookup_when_data_fetching_is_disabled(_isolated, monkeypatch):
    """`enable_fetch_data=False` is the offline switch — it must stop the
    network call, not just the caching of its result."""
    monkeypatch.setattr(sb, "_yfinance_sector", _isolated)
    monkeypatch.setattr(sb.settings, "enable_fetch_data", False)
    _fake_yfinance(monkeypatch,
                   lambda t: pytest.fail("network hit while fetching is disabled"))
    assert sb._yfinance_sector("ZZZZ") is None


def test_lookup_failure_is_swallowed(_isolated, monkeypatch):
    """yfinance raising must degrade to 'unknown sector', never propagate into
    the scoring loop."""
    monkeypatch.setattr(sb, "_yfinance_sector", _isolated)

    def _boom(t):
        raise RuntimeError("yfinance down")

    _fake_yfinance(monkeypatch, _boom)
    assert sb._yfinance_sector("ZZZZ") is None


def test_category_is_accepted_when_sector_is_absent(_isolated, monkeypatch):
    """Funds report `category` rather than `sector`; the lookup reads either."""
    monkeypatch.setattr(sb, "_yfinance_sector", _isolated)
    _fake_yfinance(monkeypatch, lambda t: type(
        "T", (), {"info": {"category": "Healthcare"}})())
    assert sb._yfinance_sector("ZZZZ") == "Healthcare"


# ── cache file I/O ──────────────────────────────────────────────────────────

def test_corrupt_cache_file_is_ignored_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(sb, "CACHE_FILE", tmp_path / "broken.json")
    sb.CACHE_FILE.write_text("{not json", encoding="utf-8")
    assert sb._load_cache() == {}


def test_missing_cache_file_loads_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(sb, "CACHE_FILE", tmp_path / "absent.json")
    assert sb._load_cache() == {}


def test_save_then_load_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(sb, "CACHE_FILE", tmp_path / "sub" / "map.json")
    sb._save_cache({"AAA": "XLK", "BBB": ""})
    assert sb._load_cache() == {"AAA": "XLK", "BBB": ""}


def test_every_mapped_sector_points_at_a_real_spdr():
    """A typo in the sector→ETF table resolves to a ticker with no price data,
    which makes the relative-momentum score abstain rather than error."""
    assert set(sb._SECTOR_TO_SPDR.values()) <= {
        "XLK", "XLF", "XLV", "XLY", "XLP", "XLI", "XLE", "XLB", "XLU", "XLRE", "XLC"}
