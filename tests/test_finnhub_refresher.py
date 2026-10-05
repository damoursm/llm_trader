"""The Finnhub refresher (2026-09-25, `src/data/finnhub_refresher.py`).

The free tier allows 60 calls/min; ~400 names would hold Step 1 for ~7 minutes,
so a background thread keeps a per-ticker cache of the leg's exact request and
the tick reads it. What must hold: the tick serves only TODAY's, fresh-enough
entries; it fetches the rest inline through the SAME limiter up to a budget
(the pre-2026-09-25 behaviour when no refresher runs); a 429 blocks every
caller; the loop always works on the stalest name.
"""
from __future__ import annotations

import time
from datetime import date, timedelta

import pytest

from config.settings import settings
from src.data import finnhub_refresher as fr
from src.data import provider_news as pn


@pytest.fixture
def finnhub_on(monkeypatch):
    monkeypatch.setattr(settings, "enable_finnhub_news", True)
    monkeypatch.setattr(settings, "finnhub_api_key", "x")
    monkeypatch.setattr(settings, "finnhub_calls_per_minute", 60000)      # no waiting in tests
    monkeypatch.setattr(settings, "finnhub_inline_budget", 60)
    monkeypatch.setattr(settings, "finnhub_cache_max_age_seconds", 1800)
    fr.reset()
    yield
    fr.reset()


class _Resp:
    def __init__(self, payload, status=200):
        self.status_code = status
        self._p = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._p


def _payload(tk, n=3):
    ts = int(time.time())
    return [{"datetime": ts - i * 60, "headline": f"{tk} catalyst {i}", "url": f"http://f/{tk}/{i}",
             "source": "Reuters", "summary": "s"} for i in range(n)]


def _fake_get(calls, status=200):
    def get(url, params=None, timeout=None):
        calls.append(params["symbol"])
        return _Resp(_payload(params["symbol"]), status=status)
    return get


def test_the_leg_serves_the_cache_first_and_fetches_the_rest_inline(finnhub_on, monkeypatch):
    calls = []
    monkeypatch.setattr(pn.httpx, "get", _fake_get(calls))
    monkeypatch.setattr(settings, "finnhub_inline_budget", 2)
    out1 = pn.fetch_finnhub_news(["AAA", "BBB", "CCC"])
    assert calls == ["AAA", "BBB"]                       # budget 2; CCC missing this call
    assert {a.tickers[0] for a in out1} == {"AAA", "BBB"}
    out2 = pn.fetch_finnhub_news(["AAA", "BBB", "CCC"])
    assert calls == ["AAA", "BBB", "CCC"]                # AAA/BBB from the cache
    assert {a.tickers[0] for a in out2} == {"AAA", "BBB", "CCC"}
    assert all(a.provider_sentiment_source == "finnhub" for a in out2)


def test_no_cap_by_default_every_name_is_covered(finnhub_on, monkeypatch):
    calls = []
    monkeypatch.setattr(pn.httpx, "get", _fake_get(calls))
    monkeypatch.setattr(settings, "finnhub_inline_budget", 500)
    names = [f"N{i:03d}" for i in range(120)]
    out = pn.fetch_finnhub_news(names)
    assert len(calls) == 120 and len({a.tickers[0] for a in out}) == 120


def test_only_todays_fresh_entries_are_served(finnhub_on, monkeypatch):
    fr._ENTRIES["OLD"] = {"t": time.time(), "day": (date.today() - timedelta(days=1)).isoformat(),
                          "items": []}
    fr._ENTRIES["STALE"] = {"t": time.time() - 3600, "day": date.today().isoformat(), "items": []}
    fr._ENTRIES["FRESH"] = {"t": time.time() - 60, "day": date.today().isoformat(),
                            "items": [{"datetime": int(time.time()), "headline": "h", "url": "u",
                                       "source": "s", "summary": ""}]}
    assert fr.cached_items("OLD", 1800) is None           # the request's window moved at midnight
    assert fr.cached_items("STALE", 1800) is None
    items, age = fr.cached_items("fresh", 1800)
    assert len(items) == 1 and 0 <= age < 120


def test_a_429_blocks_every_caller(finnhub_on, monkeypatch):
    calls = []
    monkeypatch.setattr(pn.httpx, "get", _fake_get(calls, status=429))
    assert fr.fetch_now("AAA") is None
    assert fr.status()["rate_limited"] == 1
    # blocked: a caller that will not wait gets nothing and makes no request
    assert fr.fetch_now("BBB", max_wait=0.0) is None
    assert calls == ["AAA"]


def test_the_request_is_the_legs_exact_request_and_processing(finnhub_on, monkeypatch):
    seen = {}
    ts = int(time.time())
    payload = [{"datetime": ts, "headline": "Which S&P500 stocks are moving?", "url": "http://n",
                "source": "ChartMill"}] + \
              [{"datetime": ts - i, "headline": f"Real story {i}", "url": f"http://r/{i}",
                "source": "Reuters", "summary": "x" * 2000} for i in range(20)]

    def get(url, params=None, timeout=None):
        seen.update(params)
        return _Resp(payload)
    monkeypatch.setattr(pn.httpx, "get", get)
    items, dropped = pn.finnhub_company_news("aaa")
    assert seen["symbol"] == "AAA"
    assert seen["from"] == (date.today() - timedelta(days=3)).isoformat()
    assert seen["to"] == date.today().isoformat()
    assert dropped == 1 and len(items) == 15              # noise dropped, newest 15 kept
    assert items[0]["headline"] == "Real story 0" and len(items[0]["summary"]) == 1000


def test_the_loop_works_on_the_stalest_target(finnhub_on):
    now = time.time()
    fr.set_targets(["AAA", "BBB", "CCC"])
    today = date.today().isoformat()
    fr._ENTRIES["AAA"] = {"t": now - 100, "day": today, "items": []}
    fr._ENTRIES["BBB"] = {"t": now - 700, "day": today, "items": []}
    tk, wait = fr._next_due(now, 600)
    assert tk == "CCC" and wait == 0                      # never fetched: stalest of all
    fr._ENTRIES["CCC"] = {"t": now - 50, "day": today, "items": []}
    tk, wait = fr._next_due(now, 600)
    assert tk == "BBB" and wait == 0                      # 700 s > the 600 s cycle
    fr._ENTRIES["BBB"] = {"t": now - 10, "day": today, "items": []}
    tk, wait = fr._next_due(now, 600)
    assert tk == "AAA" and wait == pytest.approx(500, abs=1)


def test_the_cache_persists_only_todays_entries(finnhub_on, tmp_path, monkeypatch):
    monkeypatch.setattr(fr, "CACHE_PATH", tmp_path / "fh.json")
    today = date.today().isoformat()
    fr._ENTRIES["AAA"] = {"t": time.time(), "day": today, "items": [{"headline": "h"}]}
    fr._ENTRIES["OLD"] = {"t": time.time(), "day": "2000-01-01", "items": []}
    fr._save()
    fr.reset()
    assert fr._load() == 1
    assert set(fr._ENTRIES) == {"AAA"}


def test_keepalive_revives_idle_targets_and_nothing_else(finnhub_on):
    fr.keepalive()
    assert fr._TARGETS == []                              # no targets: nothing to keep alive
    fr.set_targets(["aaa", "AAA", "bbb"])
    assert fr._TARGETS == ["AAA", "BBB"]
    fr._TARGETS_AT = 0.0
    fr.keepalive()
    assert time.time() - fr._TARGETS_AT < 5


def test_the_refresher_does_not_start_without_a_key(monkeypatch):
    monkeypatch.setattr(settings, "enable_finnhub_news", True)
    monkeypatch.setattr(settings, "finnhub_api_key", "")
    assert fr.start() is False and not fr.is_running()


def test_the_limiter_spaces_calls_and_refuses_a_long_wait():
    lim = fr._Limiter()
    assert lim.acquire(600) is True                       # reserves the next slot 0.1 s out
    t0 = time.monotonic()
    assert lim.acquire(600) is True
    assert time.monotonic() - t0 >= 0.08
    lim.block(30)
    assert lim.acquire(600, max_wait=1.0) is False


def test_the_state_file_round_trips_and_stop_publishes_idle(finnhub_on):
    """The backfill (another PROCESS on the same key) pulls only while this
    reports idle."""
    fr.write_state(True)
    st = fr.read_state()
    assert st["active"] is True and time.time() - st["at"] < 5
    fr.stop(timeout=0.1)
    assert fr.read_state()["active"] is False


def test_the_idle_threshold_outlasts_every_weekday_gap_between_slots():
    """Idling mid-week would starve the next tick's Finnhub leg. The longest gap
    between two slots on a regular weekday, plus the 20-minute keepalive lead,
    must stay under the threshold — derived from the live slot grid, so a grid
    change that opens a longer gap fails here instead of silently idling."""
    from datetime import date as _d, datetime as _dt, timedelta as _td
    from src.scheduler import runner
    slots, _end = runner._session_slots()
    day = _d(2026, 9, 22)                                 # a Tuesday, no holiday
    stamps = sorted(_dt.combine(day + _td(days=k), t) for k in (0, 1) for t, kind in slots
                    if runner._slot_is_valid(day + _td(days=k), t, kind))
    gaps = [(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])]
    assert max(gaps) + 20 * 60 < fr._IDLE_AFTER_SECONDS
