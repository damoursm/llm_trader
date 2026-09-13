"""Per-tick news replay (2026-09-07): the point-in-time guarantees.

Every test here is a NO-TIME-TRAVEL probe. The module regenerates historical
news features from the archived hourly bundles, and the only thing that makes
such a regeneration legitimate is that it cannot see past the tick it replays —
so each channel through which the future could leak is pinned separately.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.analysis import news_replay as nr
from src.models import NewsArticle

NOW = datetime(2026, 8, 20, 15, 0, tzinfo=timezone.utc)


def _art(hours_before, url=None):
    return NewsArticle(title="headline", summary="s" * 40, url=url or f"u{hours_before}",
                       source="Reuters", published_at=NOW - timedelta(hours=hours_before))


def test_bundle_for_never_picks_a_later_file(tmp_path):
    """The tick used the bundle for ITS hour; a later file is the future."""
    index = [(NOW - timedelta(hours=2), tmp_path / "a.json"),
             (NOW - timedelta(hours=1), tmp_path / "b.json"),
             (NOW + timedelta(hours=1), tmp_path / "c.json")]      # written later
    stamp, path = nr._bundle_for(index, NOW)
    assert path.name == "b.json" and stamp == NOW - timedelta(hours=1)
    # a tick before every bundle gets nothing rather than the oldest one
    assert nr._bundle_for(index, NOW - timedelta(days=5)) == (None, None)


def test_a_few_future_articles_are_dropped_not_fatal(monkeypatch, tmp_path):
    """A bundle's hour-STAMP is not its write time: the hour-H file is written BY
    the tick running in hour H, so an article published during that tick's own
    fetch carries a timestamp after the run's `generated_at`. It was available to
    the run; it postdates only the run's start. Dropping a few is correct —
    aborting the tick would have blocked the backfill on any run whose fetch
    overlapped a publication."""
    # published 10 minutes into the run's own fetch
    monkeypatch.setattr(nr, "_load_bundle",
                        lambda p: [_art(2), _art(3), _art(-1 / 6, "future")])
    monkeypatch.setattr(nr, "_polygon_as_of", lambda when, universe: [])
    index = [(NOW - timedelta(hours=1), tmp_path / "b.json")]
    pool, prov = nr.build_tick_pool(NOW, {"AAA"}, index=index)
    assert "future" not in {a.url for a in pool}
    assert prov["n_bundle"] == 2


def test_a_bundle_mostly_in_the_future_still_raises(monkeypatch, tmp_path):
    """Hours into the future is not a fetch overlap — it means the file is
    misnamed or a clock is wrong, and that must be loud rather than quietly
    cleaned up."""
    monkeypatch.setattr(nr, "_load_bundle",
                        lambda p: [_art(2), _art(-3, "f1"), _art(-4, "f2")])
    monkeypatch.setattr(nr, "_polygon_as_of", lambda when, universe: [])
    index = [(NOW - timedelta(hours=1), tmp_path / "b.json")]
    with pytest.raises(RuntimeError, match="not point-in-time"):
        nr.build_tick_pool(NOW, {"AAA"}, index=index)


def test_pool_drops_articles_older_than_the_scorers_own_window(monkeypatch, tmp_path):
    monkeypatch.setattr(nr, "_load_bundle",
                        lambda p: [_art(2), _art(24), _art(24 * 9, "ancient")])
    monkeypatch.setattr(nr, "_polygon_as_of", lambda when, universe: [])
    index = [(NOW - timedelta(hours=1), tmp_path / "b.json")]
    pool, prov = nr.build_tick_pool(NOW, {"AAA"}, index=index)
    assert {a.url for a in pool} == {"u2", "u24"}
    assert prov["n_bundle"] == 2


def test_attention_baseline_is_bounded_by_the_replayed_date(tmp_db):
    """`news_shock.load_attention_baselines` hardcodes CURRENT_DATE, which is
    correct live and is time travel in a replay. `analysis_asof` does not cover
    it either — it bounds the panel LOADERS and this issues its own SQL — so the
    replay computes a date-bounded baseline of its own."""
    from src.db import repo
    from src.db.schema import SIGNAL_METHOD_COLUMNS
    base = {"type": "STOCK", "direction": "BULLISH", "combined_score": 0.1,
            "confidence": 0.5, "n_methods_agreeing": 1, "dominant_method": "news",
            "price": 10.0, "scores": {m: 0.0 for m in SIGNAL_METHOD_COLUMNS}}
    # six covered days BEFORE the replayed date, all mass 1.0 ...
    for i in range(6):
        d = f"2026-08-{10 + i:02d}"
        repo.insert_signals(f"r{i}", f"{d}T20:00:00+00:00", d,
                            [{**base, "ticker": "AAA", "news_recency_mass": 1.0}])
    # ... and six much heavier days AFTER it, which must not reach the baseline
    for i in range(6):
        d = f"2026-08-{21 + i:02d}"
        repo.insert_signals(f"r9{i}", f"{d}T20:00:00+00:00", d,
                            [{**base, "ticker": "AAA", "news_recency_mass": 99.0}])
    assert nr.baselines_as_of("2026-08-20")["AAA"] == pytest.approx(1.0)
    # the same query run later DOES see them — proving the bound is what excluded
    # them, not an empty table
    assert nr.baselines_as_of("2026-08-30")["AAA"] == pytest.approx(50.0)


def test_replay_uses_the_visible_ohlcv_window_not_todays_frame():
    """The OHLCV-consuming news methods (`news_bear_fresh`, `news_unpriced`)
    must be handed the bars the tick could see. Pinned at the source: the
    replay calls `replay.visible_history`, which reproduces the forming-bar
    rule as well."""
    import inspect
    src = inspect.getsource(nr._replay_one)
    assert "visible_history" in src
    assert "load_ohlcv(ticker)" in src and "visible_history(load_ohlcv" in src


def test_derived_layers_run_under_the_asof_cutoff():
    import inspect
    src = inspect.getsource(nr.replay_tick)
    assert "analysis_asof(info[\"signal_date\"])" in src


def test_replayed_rows_land_in_their_own_table_not_signals_replay():
    """The reconstruction is measurably NOT the same quantity as a live score
    (2026-09-07: 3.1 of 6.4 articles, rho 0.24, 67.5% sign agreement over 418
    pairs), so it must not be able to reach the panel by accident."""
    import inspect
    src = inspect.getsource(nr)
    assert "insert_news_replay" in src
    assert "insert_signals_replay" not in src
    assert "restore_replayed" not in src


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "t.db"))


# ── replay fidelity to live (2026-09-12) ────────────────────────────────────

def test_a_forced_engine_can_PIN_without_suppressing_the_provider_shortcut():
    """Two different reasons force an engine, and they want opposite things.

    A hold review forces one to RE-JUDGE, so it must not take the provider
    shortcut. The news replay forces one to PIN which model answers, and must
    take the shortcut wherever live's would have — measured 2026-09-12, the
    provider path serves ~42 of ~130 scored tickers per tick (~32%), so a replay
    that always calls the LLM diverges from live on a third of its rows AND pays
    for calls live never made."""
    import inspect

    import src.analysis.sentiment as sent
    src = inspect.getsource(sent.analyse_sentiment)
    assert "_allow_provider = (force_engine is None) if allow_provider is None" in src
    assert "if _allow_provider:" in src
    # default is unchanged: a plain forced call still bypasses the shortcut
    sig = inspect.signature(sent.analyse_sentiment)
    assert sig.parameters["allow_provider"].default is None


def test_the_replay_does_not_pollute_the_live_digest_store():
    """`sentiment_digests` records what the LIVE scorer saw, and `news_replay`
    keeps no `digest_id` — so replayed digests would land orphaned: unjoinable to
    the rows they came from, stamped with whatever run id happened to be current,
    and counted against the 180-day retention. ~34k rows on a full backfill."""
    import inspect

    import src.analysis.sentiment as sent
    from src.analysis import news_replay as nr
    assert "if store_digest:" in inspect.getsource(sent.analyse_sentiment)
    assert inspect.signature(sent.analyse_sentiment).parameters["store_digest"].default is True
    body = inspect.getsource(nr)
    assert "store_digest=False" in body and "allow_provider=True" in body


def test_the_replay_reconstructs_provider_insights():
    """The data was always in the response — `get_ticker_news_history` returns
    `insights` — the replay simply never mapped them onto `provider_insights`,
    which is what made the provider shortcut unreachable there. The RAW label is
    kept, exactly as `provider_news.fetch_polygon_news` does."""
    import inspect

    from src.analysis import news_replay as nr
    src = inspect.getsource(nr)
    assert "provider_insights=_ins" in src
    assert 'item.get("insights")' in src


def test_the_replay_installs_the_clustering_corpus_once_per_tick():
    """`news_cluster_mode` is hybrid, and the content-aware modes merge on IDF
    from the TICK'S WHOLE POOL. With no corpus `cluster_articles` degrades to the
    bare TIME partition — safe live, a silent infidelity here. Installed from
    `pool` and never per ticker from its own digest, since within-digest IDF
    zeroes exactly the shared-story terms the merge depends on."""
    import inspect

    from src.analysis import news_replay as nr
    tick = inspect.getsource(nr.replay_tick)
    assert "set_corpus(pool)" in tick
    assert tick.index("pool, prov = build_tick_pool") < tick.index("set_corpus(pool)")


def test_the_replay_regenerates_every_weighted_news_method():
    """`news_quiet` (0.10) and `news_bull_fresh` (0.08) shipped after this module
    was written. A backfill without them regenerates a news family missing its
    two newest voters, and the gap is invisible — the columns would not exist
    rather than read wrong."""
    from src.analysis.news_replay import NEWS_REPLAY_COLUMNS
    from src.signals.method_epochs import NEWS_FAMILY
    missing = [m for m in NEWS_FAMILY if m not in NEWS_REPLAY_COLUMNS]
    assert not missing, f"news family methods not regenerated: {missing}"


def test_the_pair_queries_are_ROLE_AGNOSTIC():
    """`paired_tickers` asked for `shadow_engine='local'`, which was right while
    DeepSeek was primary and went silently FALSE on 2026-09-09 when local took
    100% of the primary route. Every recent run then returned ZERO paired
    tickers — reported as "no paired tickers", indistinguishable from "nothing
    accrued", which is how it survived unnoticed.

    Exactly the defect `catalyst_repair`'s pair query carried. What the fidelity
    comparison needs is a LOCAL verdict, on whichever side of the pair it sits;
    a query keyed on today's engine ROLES breaks the day routing changes."""
    import inspect

    from src.analysis import news_replay as nr
    src = inspect.getsource(nr)
    # both the ticker selector and the fidelity join must accept either side
    assert src.count("primary_engine = 'local'") >= 2, (
        "a pair query still assumes local is the SHADOW")
    for q in ("paired_tickers", "fidelity"):
        pass
    body = inspect.getsource(nr.paired_tickers)
    assert "primary_engine = 'local'" in body and "shadow_engine = 'local'" in body


def test_the_writer_column_list_covers_every_replayed_feature():
    """`repo._NEWS_REPLAY_COLS` is what the INSERT names, and it is a SECOND
    copy of the list `news_replay` computes. They drifted on 2026-09-12: the two
    methods added that day reached `NEWS_REPLAY_COLUMNS` and the schema but not
    the writer, so every row computed them and the writer dropped them — 0.0%
    coverage on methods that read ~1% and ~18% live, with nothing erroring.

    Same relationship `tests/test_db_signals.py` pins for the signals panel."""
    from src.analysis.news_replay import NEWS_REPLAY_COLUMNS
    from src.db.repo import _NEWS_REPLAY_COLS
    missing = [c for c in NEWS_REPLAY_COLUMNS if c not in _NEWS_REPLAY_COLS]
    assert not missing, f"computed but never written: {missing}"


def test_every_replayed_feature_has_a_schema_column():
    """The third place the same list lives. A column present in the writer but
    absent from the table would fail loudly on INSERT — this pins it anyway so
    the failure is a test, not a dead backfill run."""
    from src.analysis.news_replay import NEWS_REPLAY_COLUMNS
    from src.db import schema
    ddl = "".join(schema._CREATE_STATEMENTS) if hasattr(schema, "_CREATE_STATEMENTS") else ""
    src = ddl or open("src/db/schema.py", encoding="utf-8").read()
    add = {c for _t, c, _ty in schema._ADD_COLUMNS if _t == "news_replay"}
    for c in NEWS_REPLAY_COLUMNS:
        assert c in src or c in add, f"{c} has no news_replay schema column"


def test_the_shock_baseline_comes_from_the_SAME_generator_as_its_numerator(tmp_db):
    """`news_shock` is a RATIO of today's evidence mass to the ticker's own
    normal, so both sides must be produced the same way.

    The first full backfill divided a REPLAYED numerator by the LIVE `signals`
    series, and a replayed pool carries ~0.19x the live mass (older articles,
    and only the yfinance/NewsAPI leg survives) — so the ratio sat structurally
    below 1, `clip(log2(ratio)/3, 0, 1)` returned exactly 0, and the method
    scored on 1.2% of rows against 13.7% on the consistent basis. The failure
    was invisible: an abstention is what "attention is normal" looks like too.
    """
    from src.db import repo
    from src.db.schema import SIGNAL_METHOD_COLUMNS
    base = {"type": "STOCK", "direction": "BULLISH", "combined_score": 0.1,
            "confidence": 0.5, "n_methods_agreeing": 1, "dominant_method": "news",
            "price": 10.0, "scores": {m: 0.0 for m in SIGNAL_METHOD_COLUMNS}}
    # LIVE panel: a heavy baseline. REPLAYED rows: a much lighter one.
    for i in range(6):
        d = f"2026-08-{10 + i:02d}"
        repo.insert_signals(f"r{i}", f"{d}T20:00:00+00:00", d,
                            [{**base, "ticker": "AAA", "news_recency_mass": 100.0}])
        repo.insert_news_replay([{
            "run_id": f"nr{i}", "ticker": "AAA", "signal_date": d,
            "generated_at": f"{d}T20:00:00+00:00", "replayed_at": "2026-09-12T00:00:00+00:00",
            "replay_version": "t", "engine": "local", "pool_spec": "union168h",
            "news": 0.5, "news_recency_mass": 1.0, "news_article_count": 3}])
    live_basis = nr.baselines_as_of("2026-08-20")
    pool_basis = nr.baselines_as_of("2026-08-20", "union168h")
    assert live_basis["AAA"] == pytest.approx(100.0)
    assert pool_basis["AAA"] == pytest.approx(1.0)
    # and NO silent fallback between the two: an unknown pool shape abstains
    # rather than quietly borrowing the live series.
    assert nr.baselines_as_of("2026-08-20", "union999h") == {}


def test_repair_shock_is_point_in_time_and_dry_by_default(tmp_db):
    """The repair pass sees the whole window at once, which is exactly why it
    has to restate the strictly-earlier rule: a later day's attention must never
    reach an earlier day's baseline."""
    from src.db import repo
    rows = []
    for i in range(6):                      # six quiet days, mass 1.0
        d = f"2026-08-{10 + i:02d}"
        rows.append({"run_id": f"q{i}", "ticker": "AAA", "signal_date": d,
                     "generated_at": f"{d}T20:00:00+00:00",
                     "replayed_at": "2026-09-12T00:00:00+00:00", "replay_version": "t",
                     "engine": "local", "pool_spec": "union168h",
                     "news": 0.5, "news_shock": 0.0,
                     "news_recency_mass": 1.0, "news_article_count": 3})
    # the day under test: 8x its own normal -> full shock, sign of the verdict
    rows.append({"run_id": "loud", "ticker": "AAA", "signal_date": "2026-08-20",
                 "generated_at": "2026-08-20T20:00:00+00:00",
                 "replayed_at": "2026-09-12T00:00:00+00:00", "replay_version": "t",
                 "engine": "local", "pool_spec": "union168h",
                 "news": -0.5, "news_shock": 0.0,
                 "news_recency_mass": 8.0, "news_article_count": 9})
    repo.insert_news_replay(rows)

    dry = nr.repair_shock(pool_spec="union168h")
    assert dry["changed"] >= 1 and dry["applied"] is False
    unchanged = repo.fetch_df(
        "SELECT news_shock FROM news_replay WHERE run_id = 'loud'")
    assert float(unchanged.news_shock[0]) == 0.0      # dry run wrote nothing

    out = nr.repair_shock(pool_spec="union168h", apply=True)
    assert out["applied"] is True
    got = repo.fetch_df("SELECT news_shock FROM news_replay WHERE run_id = 'loud'")
    assert float(got.news_shock[0]) == pytest.approx(-1.0)   # 8x normal, bearish read
    # The quiet days themselves have too few EARLIER days to qualify, so they
    # stay at 0 — proving the window is trailing, not the whole table.
    early = repo.fetch_df("SELECT news_shock FROM news_replay WHERE run_id = 'q0'")
    assert float(early.news_shock[0]) == 0.0
