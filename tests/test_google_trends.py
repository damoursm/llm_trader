"""google_trends: a rate-limited fetch must RAISE (not cache []) — so it doesn't
poison the daily cache and masquerade as a legit 'no trending spikes' result."""

import types

import pandas as pd
import pytest

from src.data import google_trends as gt


class _FakePT:
    def __init__(self, df):
        self._df = df

    def build_payload(self, *a, **k):
        pass

    def interest_over_time(self):
        return self._df


def _install(monkeypatch, df):
    import pytrends.request as pr
    monkeypatch.setattr(pr, "TrendReq", lambda *a, **k: _FakePT(df))
    monkeypatch.setattr(gt, "_load_cache", lambda: None)            # force a fresh fetch
    monkeypatch.setattr(gt, "time", types.SimpleNamespace(sleep=lambda *a, **k: None))


def test_all_batches_empty_raises_and_does_not_cache(monkeypatch):
    saved = {"n": 0}
    monkeypatch.setattr(gt, "_save_cache", lambda arts: saved.__setitem__("n", saved["n"] + 1))
    _install(monkeypatch, pd.DataFrame())          # empty frame for every batch = rate-limited
    with pytest.raises(RuntimeError):
        gt.fetch_google_trends(["AAPL", "NVDA"])
    assert saved["n"] == 0                          # never cached → no poison, retries next tick


def test_spike_produces_article_and_caches(monkeypatch):
    # 93 daily points: flat ~12 baseline, recent 7 surge to 30 → +150%, cur>10 → emits.
    df = pd.DataFrame({"NVDA": [12.0] * 86 + [30.0] * 7})
    saved = {"arts": None}
    monkeypatch.setattr(gt, "_save_cache", lambda arts: saved.__setitem__("arts", arts))
    _install(monkeypatch, df)
    arts = gt.fetch_google_trends(["NVDA"])
    assert len(arts) == 1 and "NVDA" in arts[0].title
    assert saved["arts"] is not None               # a successful fetch IS cached


def test_fetched_but_no_spike_caches_empty(monkeypatch):
    # Flat interest = a genuine "no spikes" day → returns [] AND caches it (legit
    # empty, distinct from the rate-limited failure above).
    df = pd.DataFrame({"NVDA": [20.0] * 93})
    saved = {"called": False}
    monkeypatch.setattr(gt, "_save_cache", lambda arts: saved.__setitem__("called", True))
    _install(monkeypatch, df)
    arts = gt.fetch_google_trends(["NVDA"])
    assert arts == [] and saved["called"] is True
