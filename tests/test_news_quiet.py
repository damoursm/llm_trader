"""`news_quiet` — the news read once the story has gone quiet (2026-09-09).

The measured claim: splitting on the age of the digest's FRESHEST news cluster,
reads on stories >= 48h old pay +1.90 pp/decision (t +2.65, same-sign halves)
while reads on fresh ones pay -0.23 pp. SHORT carries it (+2.90, t +3.10);
LONG is below the bar (+0.90, t +0.86, halves opposite) and ships on the user's
explicit call.

These tests pin the choices that make the method mean what the measurement
measured — the RAW verdict rather than the scaled one, the STORY's start rather
than the newest article, abstention rather than inversion on a loud story — and
the registrations that keep a new method from being half-wired.
"""
from datetime import datetime, timedelta, timezone

import pytest

from src.models import NewsArticle
from src.signals import news_quiet as nq

NOW = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _pin_floor(monkeypatch):
    """These tests are about the RULE, not the tuned threshold. Pinning the
    floor keeps them meaningful when the setting moves (it went 48h -> 72h on
    2026-09-10); `test_the_threshold_is_a_setting_and_moves_the_verdict` is the
    one that owns the value."""
    from config.settings import settings
    monkeypatch.setattr(settings, "news_quiet_min_age_hours", 48.0, raising=False)


def _art(hours_ago, title="t"):
    return NewsArticle(title=title, summary="s", source="Reuters", url=f"u{hours_ago}",
                       published_at=NOW - timedelta(hours=hours_ago))


# ── the gate ────────────────────────────────────────────────────────────────

def test_a_loud_story_abstains_and_a_quiet_one_carries_the_verdict():
    """The whole method: same verdict, opposite treatment by story age."""
    loud = [_art(2), _art(3), _art(5)]
    quiet = [_art(60), _art(64), _art(70)]
    assert nq.compute_news_quiet("AAA", 0.42, loud, as_of=NOW)[0] == 0.0
    assert nq.compute_news_quiet("AAA", 0.42, quiet, as_of=NOW)[0] == pytest.approx(0.42)
    # bearish reads are treated identically — the method is symmetric, and the
    # measured long/short asymmetry is left to the per-side machinery
    assert nq.compute_news_quiet("AAA", -0.42, quiet, as_of=NOW)[0] == pytest.approx(-0.42)
    assert nq.compute_news_quiet("AAA", -0.42, loud, as_of=NOW)[0] == 0.0


def test_a_loud_story_abstains_rather_than_inverting():
    """A fresh read measured worth ~0 (LONG +0.50 pp, SHORT -1.07 pp), NOT
    reliably wrong. Scoring its negative would be a far stronger claim than the
    evidence supports — and in the rank basis a 0.0 leaves the cross-section,
    which is the honest expression of "no view here"."""
    loud = [_art(1), _art(2)]
    for verdict in (0.9, -0.9, 0.05, -0.05):
        assert nq.compute_news_quiet("AAA", verdict, loud, as_of=NOW)[0] == 0.0


def test_the_threshold_is_a_setting_and_moves_the_verdict(monkeypatch):
    from config.settings import settings
    arts = [_art(50), _art(55)]
    assert nq.compute_news_quiet("AAA", 0.3, arts, as_of=NOW)[0] == pytest.approx(0.3)
    monkeypatch.setattr(settings, "news_quiet_min_age_hours", 72.0, raising=False)
    assert nq.compute_news_quiet("AAA", 0.3, arts, as_of=NOW)[0] == 0.0


def test_age_is_measured_from_the_STORYS_start_not_the_newest_article():
    """The unit is the story, not the article. A cluster that BEGAN three days
    ago and is still being syndicated is old, and a genuinely new story inside
    an old digest is fresh — that distinction is the entire feature, and it is
    why this reuses `news_priced_in.cluster_bounds` rather than taking a max
    over publication times."""
    # one long-running story: many articles, all part of the same cluster
    running = [_art(h) for h in (60, 62, 64, 66, 70)]
    age = nq.freshest_cluster_age_hours(running, as_of=NOW)
    assert age == pytest.approx(70.0, abs=1.5)          # the story's START
    # a fresh story on top of an old one: the FRESHEST cluster is what counts
    mixed = [_art(1), _art(2)] + [_art(h) for h in (90, 95, 100)]
    assert nq.freshest_cluster_age_hours(mixed, as_of=NOW) < 6
    assert nq.compute_news_quiet("AAA", 0.5, mixed, as_of=NOW)[0] == 0.0


def test_the_clock_is_hour_quantised():
    """Two runs inside one hour must not land on opposite sides of the
    threshold for the same article set — the same rule `sentiment._clock`
    enforces so a borderline article cannot flip the digest between ticks."""
    arts = [_art(48.4)]
    a = nq.freshest_cluster_age_hours(arts, as_of=NOW)
    b = nq.freshest_cluster_age_hours(arts, as_of=NOW + timedelta(minutes=45))
    assert a == b


# ── the input choice ────────────────────────────────────────────────────────

def test_it_reads_the_RAW_verdict_not_the_scaled_news_score():
    """The scaled `news` score multiplies by evidence mass x source diversity,
    and the quiet cohort is thin BY CONSTRUCTION (median 6 articles vs 12) — so
    the scaled score would shrink exactly the names this method exists to
    express. Raw is also the quantity the split was measured on.

    Pinned structurally: the aggregator must hand it `raw_score`, not
    `sentiment_score`."""
    import inspect

    import src.signals.aggregator as agg
    src = inspect.getsource(agg.build_signals)
    i = src.index("compute_news_quiet(")
    call = src[i:i + 220]
    assert 'raw_score' in call, call
    assert "sentiment_score" not in call, call


def test_a_zero_or_missing_verdict_abstains():
    """No news view, nothing to carry into the quiet period."""
    quiet = [_art(60), _art(64)]
    assert nq.compute_news_quiet("AAA", 0.0, quiet, as_of=NOW)[0] == 0.0
    assert nq.compute_news_quiet("AAA", None, quiet, as_of=NOW)[0] == 0.0
    assert nq.compute_news_quiet("AAA", float("nan"), quiet, as_of=NOW)[0] == 0.0
    assert nq.compute_news_quiet("AAA", 0.5, [], as_of=NOW)[0] == 0.0


def test_it_fails_soft_and_clamps():
    """A scorer that raises takes the whole tick's ticker down; this one
    abstains instead. And the contract is [-1, +1] whatever the caller passes."""
    assert nq.compute_news_quiet("AAA", 0.5, object(), as_of=NOW) == (0.0, None)
    quiet = [_art(60), _art(64)]
    assert nq.compute_news_quiet("AAA", 4.2, quiet, as_of=NOW)[0] == 1.0
    assert nq.compute_news_quiet("AAA", -4.2, quiet, as_of=NOW)[0] == -1.0


def test_the_flag_switches_it_off(monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "enable_news_quiet", False, raising=False)
    assert nq.compute_news_quiet("AAA", 0.42, [_art(99)], as_of=NOW) == (0.0, None)


# ── registration: a half-wired method is worse than none ────────────────────

def test_it_is_registered_everywhere_a_method_must_be():
    from src.analysis.code_version import METHOD_SOURCES
    from src.db.schema import SIGNAL_BASE_METHOD_COLUMNS
    from src.performance.tracker import _ALL_METHODS, METHOD_CATEGORIES, METHOD_LABELS
    assert "news_quiet" in _ALL_METHODS
    assert "news_quiet" in SIGNAL_BASE_METHOD_COLUMNS
    assert "news_quiet" in METHOD_LABELS
    assert "news_quiet" in METHOD_SOURCES
    assert sum("news_quiet" in v for v in METHOD_CATEGORIES.values()) == 1


def test_it_votes_inside_the_SENTIMENT_family_not_as_a_second_voter():
    """It is the SAME information as `news` — the same verdict, gated on the
    story having gone quiet. The family layer exists so correlated methods are
    one voter; a `news_quiet` in a family of its own would let one sentiment
    read carry two of the seven family votes."""
    from src.signals.agreement import METHOD_FAMILIES
    fams = [f for f, ms in METHOD_FAMILIES.items() if "news_quiet" in ms]
    assert fams == ["Sentiment"]
    assert "news" in METHOD_FAMILIES["Sentiment"]


def test_it_is_weighted_and_the_weight_is_modest():
    """Shipped as a WEIGHTED method, not panel-first — the user's call. The
    weight stays below `sent_velocity` (0.12) because the finding is
    exploratory, and far below `news` (0.40) whose verdict it re-uses."""
    from src.signals.aggregator import _BASE_WEIGHTS
    assert 0 < _BASE_WEIGHTS["news_quiet"] <= 0.12
    assert _BASE_WEIGHTS["news_quiet"] < _BASE_WEIGHTS["news"]


def test_it_shares_the_news_family_epoch():
    """It shipped 2026-09-09 with NO epoch, correctly: an epoch masks history
    produced by a superseded implementation and a brand-new method has none.

    That stopped being true on 2026-09-10. `news_quiet` carries the RAW VERDICT
    on quiet names, and the verdict has since changed categorically four times —
    the passing-mention abstention, the logprob expectation replacing the argmax,
    prompt v7dir (11% of signs), and the catalyst-class cap. Unlike the rest of
    that family it is NOT panel-first: at weight 0.10 its win-rate filter and
    per-side adaptive tilt read its history, and with no epoch registered
    `score_is_comparable` failed OPEN and pooled all four eras."""
    from src.signals.method_epochs import METHOD_SCORER_EPOCH, NEWS_FAMILY
    assert "news_quiet" in NEWS_FAMILY
    assert len({str(METHOD_SCORER_EPOCH[m]) for m in NEWS_FAMILY}) == 1


def test_every_verdict_derived_method_is_in_the_family():
    """The rule is "the whole news family shares ONE boundary", and the way it
    breaks is by omission — a new verdict-derived method gets a weight and a
    column and nobody adds it here, so it silently pools scorer eras. Anything
    scoring off the sentiment verdict belongs in NEWS_FAMILY."""
    from src.signals.method_epochs import NEWS_FAMILY
    for m in ("news", "sent_velocity", "news_shock", "news_bear_fresh",
              "news_bull_fresh", "catalyst_tilt", "news_quiet",
              "news_unpriced", "news_unpriced_all"):
        assert m in NEWS_FAMILY, m


def test_the_score_and_its_age_both_reach_the_signals_table(tmp_path, monkeypatch):
    """A scorer wired into the aggregator but missing from the insert reads as
    a permanently-abstaining method, which looks exactly like a method with no
    views — so the write is verified mechanically, not by review.

    `news_quiet_age_h` is the trap here: it is NOT a method score, so it does
    not ride `SIGNAL_METHOD_COLUMNS` and needs its own column group all the way
    through schema -> migration -> insert -> the pipeline row.
    """
    from config.settings import settings
    from src.db import repo
    from src.db.schema import SIGNAL_METHOD_COLUMNS
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "t.db"))
    scores = {m: 0.0 for m in SIGNAL_METHOD_COLUMNS}
    scores["news_quiet"] = -0.375
    repo.insert_signals("run-1", "2026-09-09T14:00:00+00:00", "2026-09-09",
                        [{"ticker": "AAA", "direction": "BEARISH", "scores": scores,
                          "news_quiet_age_h": 61.5}])
    df = repo.fetch_df("SELECT news_quiet, news_quiet_age_h FROM signals",
                       read_only=False)
    assert df.iloc[0]["news_quiet"] == pytest.approx(-0.375)
    assert df.iloc[0]["news_quiet_age_h"] == pytest.approx(61.5)


def test_the_pipeline_row_carries_the_age():
    """The insert can only write what the row dict holds."""
    import inspect

    import src.pipeline as pipeline
    assert '"news_quiet_age_h": getattr(s, "news_quiet_age_h"' in inspect.getsource(pipeline)


def test_news_replay_regenerates_it_end_to_end():
    """This test previously asserted the OPPOSITE — that `news_quiet` must NOT
    appear in the replay's columns — and it was right when written: the replay
    did not compute it, and a column nothing populates is a permanently-NULL
    field wearing the name of a real method.

    On 2026-09-12 the replay was extended to regenerate it (it is WEIGHTED 0.10;
    a backfill without it produces a news family missing one of its voters). So
    the guard flips rather than disappears: the column must now be present in
    ALL THREE lists — computed, written, and in the schema — because being in
    only some of them is exactly the orphan-column failure the original test
    existed to prevent. It was: the writer's list was missed, so the value was
    computed on every row and silently dropped."""
    from src.analysis.news_replay import NEWS_REPLAY_COLUMNS
    from src.db.repo import _NEWS_REPLAY_COLS
    from src.db import schema
    assert "news_quiet" in NEWS_REPLAY_COLUMNS          # computed
    assert "news_quiet" in _NEWS_REPLAY_COLS            # written
    assert ("news_replay", "news_quiet", "DOUBLE") in schema._ADD_COLUMNS   # stored


def test_every_weighted_method_has_an_active_flag():
    """THE bug this method actually shipped with, caught on the first live run
    (`nq=0%` in the weight log).

    `_normalised_weights` iterates `active_flags`, NOT `_BASE_WEIGHTS`:

        raw = {m: (profile.get(m, ...) if on else 0.0) for m, on in active_flags.items()}

    so a method with a real base weight that is missing from that dict never
    enters `weights` at all. It still scores, still persists to the panel, still
    ranks, still shows up in the dashboard — and contributes exactly nothing to
    the combine. Every observable says "working"; only the weight says otherwise.
    That is the project's silent-failure class, so it is checked mechanically.
    """
    import ast
    import inspect

    import src.signals.aggregator as agg

    src = inspect.getsource(agg.build_signals)
    tree = ast.parse(inspect.cleandoc(src))
    flagged = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                and any(getattr(t, "id", "") == "_raw_active" for t in node.targets)):
            flagged = {k.value for k in node.value.keys
                       if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    assert flagged, "could not locate the _raw_active dict — update this guard"
    missing = [m for m, w in agg._BASE_WEIGHTS.items() if w and m not in flagged]
    assert not missing, (
        f"weighted method(s) {missing} have no entry in `_raw_active`, so "
        f"`_normalised_weights` gives them weight 0 while every other surface "
        f"reports them as working")


def test_news_quiet_is_a_stacker_feature():
    """It was the ONE news-family method missing from the list, and the one that
    could least afford to be: with `ml_combine_arm_share` at 1.0 the stackers ARE
    the combine (`combine_source` reads "ml" on 100% of live rows), so a base
    weight only reaches a per-side fail-soft that never fires. Its 0.10 decided
    nothing about direction — it reached only coherence, `sources_agreeing` and
    the Sentiment family vote, i.e. confidence and position SIZE."""
    from src.analysis.ml_stacker import STACKER_SIGNED_FEATURES
    from src.analysis.ml_exit_dataset import EXIT_METHODS
    assert "news_quiet" in STACKER_SIGNED_FEATURES
    assert "news_quiet" in EXIT_METHODS          # by derivation, not by a second list


def test_every_weighted_news_method_is_a_stacker_feature():
    """The general form of the same miss. A method carrying a base weight while
    the stackers are the combine reaches nothing unless it is in this list, and
    the gap is invisible: every other surface still reports it as working."""
    from src.analysis.ml_stacker import STACKER_SIGNED_FEATURES, STACKER_CONTEXT_FEATURES
    from src.signals.aggregator import _BASE_WEIGHTS
    from src.signals.method_epochs import NEWS_FAMILY
    feats = set(STACKER_SIGNED_FEATURES) | set(STACKER_CONTEXT_FEATURES)
    for m in NEWS_FAMILY:
        if _BASE_WEIGHTS.get(m, 0.0) > 0:
            assert m in feats, f"{m} carries weight {_BASE_WEIGHTS[m]} but is not a stacker feature"


def test_the_weight_log_cannot_go_out_of_date_with_the_book():
    """It used to be a hand-written f-string naming ~29 methods — a second copy
    of the method book — and it drifted: the three methods promoted off weight 0
    on 2026-09-11 carried real weight and appeared nowhere in it.

    That matters because this line is the monitoring surface that caught
    `news_quiet` contributing ZERO two days earlier. A weighted method invisible
    here is the same silent-failure exposure the promotion was meant to close, so
    the line is now GENERATED from the weights dict and this test asserts every
    weighted method reaches it."""
    from src.signals.aggregator import _BASE_WEIGHTS, _format_weight_log, _WEIGHT_LOG_LABELS
    line = _format_weight_log({m: 0.05 for m in _BASE_WEIGHTS})
    for m in _BASE_WEIGHTS:
        label = _WEIGHT_LOG_LABELS.get(m, m)
        assert f"{label}=" in line, f"{m} carries weight but is missing from the weight log"


def test_the_weight_log_omits_zero_and_survives_an_empty_book():
    """~30 methods sit at weight 0; printing them would make the line
    unreadable, which is how the hand-written version justified its fixed list
    in the first place."""
    from src.signals.aggregator import _format_weight_log
    line = _format_weight_log({"news": 0.4, "tech": 0.0, "news_shock": 0.04})
    assert "news=40%" in line and "nshock=4%" in line and "tech" not in line
    assert _format_weight_log({}) == "none"
