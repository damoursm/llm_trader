"""The shadow discovery's Alpha Vantage cache (2026-10-26): the free key's 25 requests a day were spent by midday by
two uncached calls every tick — each answer is now kept for its hours, and an empty answer (the quota message) never."""
from __future__ import annotations

import json
import time

from config import settings


def test_av_answers_are_kept_for_their_hours_and_empty_ones_never():
    from src.data import trending
    calls = []

    def fetch():
        calls.append(1)
        return ["NVDA", "AMD"]
    assert trending._av_cached("movers", 24.0, fetch) == ["NVDA", "AMD"]
    assert trending._av_cached("movers", 24.0, fetch) == ["NVDA", "AMD"] and len(calls) == 1   # kept
    p = trending.Path(settings.trending_av_cache_dir) / "trending_av_movers.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["t"] = time.time() - 25 * 3600                                     # a day old: asked again
    p.write_text(json.dumps(d), encoding="utf-8")
    assert trending._av_cached("movers", 24.0, fetch) == ["NVDA", "AMD"] and len(calls) == 2
    empty = []

    def quota():
        empty.append(1)
        return []                                                        # "standard API rate limit is 25 ..."
    assert trending._av_cached("news", 12.0, quota) == [] and trending._av_cached("news", 12.0, quota) == []
    assert len(empty) == 2 and not (p.parent / "trending_av_news.json").exists()


def test_the_discovery_reads_both_alpha_vantage_calls_through_the_cache(monkeypatch):
    from src.data import trending
    n = {"movers": 0, "news": 0}
    monkeypatch.setattr(trending, "_yahoo_trending", lambda: [])
    monkeypatch.setattr(trending, "_newsapi_discovered_tickers", lambda: [])
    monkeypatch.setattr(trending, "_alpha_vantage_movers", lambda: n.__setitem__("movers", n["movers"] + 1) or ["MOVR"])
    monkeypatch.setattr(trending, "_alpha_vantage_news_tickers",
                        lambda: n.__setitem__("news", n["news"] + 1) or ["NEWZ"])
    monkeypatch.setattr(settings, "enable_stocktwits_discovery", False)
    monkeypatch.setattr(settings, "enable_wsb_discovery", False)
    for _ in range(3):                                                   # three ticks
        out = trending.get_trending_tickers(["AAPL"], ["XLK"])
    assert "MOVR" in out and "NEWZ" in out and n == {"movers": 1, "news": 1}
