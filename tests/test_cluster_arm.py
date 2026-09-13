"""Cluster arm (2026-09-08): one call per news CLUSTER, blended by recency mass.

It decides nothing — it accrues paired against the live single-call verdict so
the comparison can be made on LIVE digests rather than the reconstructed ones
the pilot used. The properties pinned here are the ones whose failure would be
invisible: that it stays off by default, off the critical path, drained, and
that it never lets a fallen-through engine contaminate a blend.
"""

from datetime import datetime, timedelta, timezone

import pytest

from config.settings import settings
from src.analysis import sentiment as sent
from src.models import NewsArticle

NOW = datetime(2026, 8, 20, 15, 0, tzinfo=timezone.utc)


def _art(hours, url=None):
    return NewsArticle(title="headline", summary="s" * 40, url=url or f"u{hours}",
                       source="Reuters", published_at=NOW - timedelta(hours=hours))


@pytest.fixture(autouse=True)
def _clean():
    sent.pop_cluster_arm_rows()
    yield
    sent.pop_cluster_arm_rows()


def test_off_by_default():
    """It costs ~2 extra local calls per sampled ticker on a box that now serves
    100% of live sentiment on the critical path — so it must be opt-in.

    Asserts the CODE default, not the live value: a deployment may enable it
    (this one does, via `.env`, since 2026-09-08), and that must not make the
    test fail — what matters is that a fresh environment does not inherit the
    cost silently."""
    from config.settings import Settings
    assert Settings.model_fields["enable_cluster_arm"].default is False


def test_cluster_split_matches_the_priced_in_rule():
    """One definition of "a cluster": `news_priced_in.cluster_bounds` and the arm
    must partition a digest identically, or two experiments disagree about what
    they measured."""
    from src.signals.news_priced_in import cluster_bounds
    arts = [_art(1), _art(3), _art(96), _art(100), _art(101)]
    groups = sent.cluster_split(arts, NOW)
    bounds = cluster_bounds(arts, as_of=NOW)
    assert [len(g) for g in groups] == [n for _s, _m, n in bounds]
    assert sum(len(g) for g in groups) == len(arts)


def test_single_cluster_digest_is_skipped(monkeypatch):
    """With one cluster the arm IS the single call — running it would spend a
    call to reproduce a verdict we already have."""
    monkeypatch.setattr(settings, "enable_cluster_arm", True, raising=False)
    calls = []
    monkeypatch.setattr(sent, "analyse_sentiment",
                        lambda *a, **k: calls.append(1) or (0.1, "r", {"raw_score": 0.1}))
    sent._run_cluster_arm("AAA", [_art(1), _art(2)], "local", 0.2, 0.1, "d1", "run1", NOW)
    assert calls == []
    assert sent.pop_cluster_arm_rows() == []


def test_blend_is_recency_mass_weighted(monkeypatch):
    """The blend reuses the house convention (`_provider_sentiment_score`'s
    weighting), so a win belongs to the decomposition and not to a new
    weighting scheme."""
    monkeypatch.setattr(settings, "enable_cluster_arm", True, raising=False)
    fresh, stale = [_art(1), _art(2)], [_art(120), _art(124)]
    scores = {id(fresh[0]): 1.0, id(stale[0]): -1.0}

    def fake(ticker, arts, force_engine=None, as_of=None):
        return 0.0, "r", {"raw_score": scores[id(arts[0])], "engine": "local"}

    monkeypatch.setattr(sent, "analyse_sentiment", fake)
    sent._run_cluster_arm("AAA", fresh + stale, "local", 0.5, 0.4, "d1", "run1", NOW)
    row = sent.pop_cluster_arm_rows()[0]
    # the fresh cluster dominates: its recency weights are ~1.0 against ~0.01
    assert row["n_clusters"] == 2
    assert row["arm_raw"] > 0.9
    assert row["primary_raw"] == 0.5


def test_a_fallen_through_engine_never_enters_the_blend(monkeypatch):
    """`force_engine` only LEADS the fallback order — a local failure produced a
    DeepSeek verdict in the pilot (2/14 rows). A cluster answered by another
    engine is DROPPED, not averaged in."""
    monkeypatch.setattr(settings, "enable_cluster_arm", True, raising=False)
    fresh, stale = [_art(1), _art(2)], [_art(120), _art(124)]

    def fake(ticker, arts, force_engine=None, as_of=None):
        if arts[0] in stale:
            return 0.0, "r", {"raw_score": -1.0, "engine": "deepseek"}   # fell through
        return 0.0, "r", {"raw_score": 1.0, "engine": "local"}

    monkeypatch.setattr(sent, "analyse_sentiment", fake)
    sent._run_cluster_arm("AAA", fresh + stale, "local", 0.5, 0.4, "d1", "run1", NOW)
    row = sent.pop_cluster_arm_rows()[0]
    import json
    assert json.loads(row["cluster_scores"]) == [1.0, None]      # the foreign one dropped
    assert row["arm_raw"] == pytest.approx(1.0)


def test_sampling_is_deterministic_per_run_and_ticker(monkeypatch):
    """A retried ticker must get the same decision, or the sample
    over-represents exactly the calls that failed once."""
    monkeypatch.setattr(settings, "cluster_arm_share", 0.5, raising=False)
    monkeypatch.setattr(sent, "_CURRENT_RUN_ID", "run-x", raising=False)
    first = {t: sent._arm_sampled(t) for t in ("AAA", "BBB", "CCC", "DDD", "EEE")}
    assert all(sent._arm_sampled(t) is v for t, v in first.items())
    monkeypatch.setattr(settings, "cluster_arm_share", 0.0, raising=False)
    assert sent._arm_sampled("AAA") is False
    monkeypatch.setattr(settings, "cluster_arm_share", 1.0, raising=False)
    assert sent._arm_sampled("AAA") is True


def test_rows_are_drained_by_persist_run():
    """An undrained buffer is indistinguishable from a working pass — the same
    defect the catalyst-repair drain was added for."""
    import inspect
    import src.pipeline as pipe
    src = inspect.getsource(pipe._persist_run)
    assert "pop_cluster_arm_rows()" in src
    assert "insert_sentiment_cluster_arm" in src


def test_arm_never_tallies_into_the_run_provider():
    """Forced calls must not count toward `runs.llm_sentiment_provider`, or the
    arm would rewrite the engine attribution of the run it is measuring."""
    import inspect
    src = inspect.getsource(sent._run_cluster_arm)
    assert "force_engine=engine" in src
    assert "_record_sentiment_provider" not in src
