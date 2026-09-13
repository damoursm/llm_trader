"""The FULL news-pool archive (`news_articles`, 2026-09-11).

WHY IT EXISTS. Measured on the live pipeline: the merged per-tick pool is
**~2,433 articles**, and only the **~600**-article yfinance/NewsAPI leg was ever
persisted (`cache/news_*.json`). `sentiment_digests` keeps the top-20 cut that
reached a scorer — and nothing at all for a ticker the passing-mention rule
abstained on, since that returns before the digest is recorded. So ~75% of every
tick was discarded the moment the tick ended.

That is precisely the gap behind `memory/news-backfill-fidelity-2026-09`: a
replay could recover only ~3.1 of a live digest's ~6.4 articles and ran at 0.40x
magnitude, which is why "historical news cannot be faithfully regenerated" is
recorded as a finding rather than a bug. This table is the fix going FORWARD; it
cannot repair the past.
"""
from datetime import datetime, timedelta, timezone

import pytest

from src.db import repo
from src.models import NewsArticle

NOW = datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)


def _art(i, url=None, title=None, tickers=("ACME",)):
    return NewsArticle(title=title if title is not None else f"Story {i}",
                       url=url if url is not None else f"http://x/{i}",
                       source="Reuters", summary=f"body {i}",
                       published_at=NOW - timedelta(hours=i), tickers=list(tickers))


def _iso(dt):
    return dt.isoformat()


# ── the writer ──────────────────────────────────────────────────────────────

def test_the_whole_pool_is_archived_url_deduped():
    pool = [_art(i) for i in range(5)] + [_art(0)]      # one duplicate URL
    out = repo.insert_news_articles("r1", _iso(NOW), pool)
    assert out == {"new": 5, "seen": 0}
    assert int(repo.fetch_df("SELECT count(*) AS n FROM news_articles").n[0]) == 5


def test_a_repeat_sighting_never_moves_first_seen_at():
    """The property a point-in-time replay depends on. An article must not be
    visible to a tick that ran before anything had fetched it, so a later
    sighting bumps the TAIL (`last_seen_at`, `n_sightings`) and leaves the head
    alone. Get this backwards and every replayed tick sees tomorrow's news."""
    pool = [_art(i) for i in range(3)]
    repo.insert_news_articles("r1", _iso(NOW), pool)
    later = NOW + timedelta(hours=2)
    out = repo.insert_news_articles("r2", _iso(later), pool)
    assert out == {"new": 0, "seen": 3}
    d = repo.fetch_df("SELECT first_seen_at, last_seen_at, n_sightings, first_run_id "
                      "FROM news_articles")
    assert set(d.first_seen_at) == {_iso(NOW)}
    assert set(d.last_seen_at) == {_iso(later)}
    assert set(d.n_sightings) == {2}
    assert set(d.first_run_id) == {"r1"}


def test_a_growing_pool_adds_only_what_is_new():
    base = [_art(i) for i in range(4)]
    repo.insert_news_articles("r1", _iso(NOW), base)
    out = repo.insert_news_articles("r2", _iso(NOW + timedelta(hours=1)),
                                    base + [_art(7), _art(8)])
    assert out == {"new": 2, "seen": 4}


def test_an_empty_or_unusable_pool_is_a_no_op():
    assert repo.insert_news_articles("r1", _iso(NOW), []) == {"new": 0, "seen": 0}
    assert repo.insert_news_articles("r1", _iso(NOW), None) == {"new": 0, "seen": 0}
    # no url AND no title carries nothing to key on
    assert repo.insert_news_articles("r1", _iso(NOW), [_art(1, url="", title="")]) \
        == {"new": 0, "seen": 0}


def test_an_article_without_a_url_still_archives_on_its_title():
    """Some feeds (8-K, a few wires) carry no canonical link. Dropping them
    would silently bias the archive toward the sources that happen to have one."""
    out = repo.insert_news_articles("r1", _iso(NOW), [_art(1, url="")])
    assert out["new"] == 1
    assert repo.fetch_df("SELECT title FROM news_articles").title[0] == "Story 1"


def test_the_fields_a_backfill_needs_survive_the_round_trip():
    repo.insert_news_articles("r1", _iso(NOW), [_art(3, tickers=("ACME", "BETA"))])
    d = repo.fetch_df("SELECT url, title, source, published_at, summary, tickers_json "
                      "FROM news_articles")
    r = d.iloc[0]
    assert r.url == "http://x/3" and r.title == "Story 3" and r.source == "Reuters"
    assert r.summary == "body 3"
    assert "ACME" in r.tickers_json and "BETA" in r.tickers_json
    assert r.published_at.startswith("2026-09-11T09:00")


# ── wiring ──────────────────────────────────────────────────────────────────

def test_the_archive_runs_BEFORE_anything_cuts_the_pool():
    """It has to sit on the merged, URL-deduped pool — after per-ticker
    relevance filtering it would archive the same subset `sentiment_digests`
    already holds, which is the thing that was not enough."""
    import inspect

    from src import pipeline
    src = inspect.getsource(pipeline.run_pipeline)
    assert src.index("_dedupe_by_url(articles)") < src.index('_safe("news_archive"')


def test_retention_prunes_on_FIRST_sighting():
    """Pruning on `last_seen_at` would keep a years-old article alive forever
    just because a feed re-served it, and would delete a recently-discovered old
    article — both wrong for a point-in-time archive."""
    import inspect

    from src.db import retention
    src = inspect.getsource(retention.run_retention)
    assert '"news_articles", "first_seen_at"' in src


def test_the_defaults_are_a_long_archive_not_a_cache():
    """The table exists to be read years later by a backfill; rows are one per
    unique ARTICLE for all time, not one per tick, so growth tracks the rate of
    new articles rather than pool size."""
    from config.settings import Settings
    assert Settings.model_fields["enable_news_archive"].default is True
    assert Settings.model_fields["news_archive_retention_days"].default >= 365


def test_no_safe_call_site_references_an_unbound_name():
    """The bug the first version of this shipped with, generalised.

    `_safe(name, fn, *args)` guards the CALL, not the evaluation of its
    arguments: Python builds the argument list first, so a NameError in an
    argument fires BEFORE the guard is entered and takes the whole tick down.
    The first archive call passed `generated_at`, which does not exist in
    `run_pipeline` — two production ticks crashed and an alert fired, while the
    suite stayed green because the only test of the wiring was a source-INDEX
    check, which proves ordering and not executability.

    This walks every `_safe(...)` in `run_pipeline` and asserts each name in its
    arguments is bound somewhere in the function, a module global, or a
    builtin."""
    import ast
    import builtins
    import inspect

    from src import pipeline

    fn = ast.parse(inspect.getsource(pipeline.run_pipeline)).body[0]
    bound = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for al in node.names:
                bound.add((al.asname or al.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.withitem) and isinstance(node.optional_vars, ast.Name):
            bound.add(node.optional_vars.id)

    unbound = []
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_safe"):
            continue
        label = getattr(node.args[0], "value", "?") if node.args else "?"
        for arg in node.args:
            for nm in ast.walk(arg):
                if isinstance(nm, ast.Name) and isinstance(nm.ctx, ast.Load):
                    if (nm.id not in bound and not hasattr(pipeline, nm.id)
                            and not hasattr(builtins, nm.id)):
                        unbound.append(f"_safe({label!r}) -> {nm.id}")
    assert not unbound, unbound


def test_the_archive_renders_its_timestamp_inside_the_guarded_call():
    """Corollary: keep attribute access OUT of the argument list. The call site
    passes the `start` datetime and `_archive_articles` renders it, so nothing
    in the arguments can raise before `_safe` is entered."""
    import inspect

    from src import pipeline
    assert "_safe(\"news_archive\", _archive_articles, run_id, start, articles)"         in inspect.getsource(pipeline.run_pipeline)
    body = inspect.getsource(pipeline._archive_articles)
    assert 'hasattr(start, "isoformat")' in body
