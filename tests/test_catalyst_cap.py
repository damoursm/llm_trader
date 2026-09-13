"""Catalyst-class conviction cap (2026-09-11).

Measured over 67 days / 5,959 labelled+typed rows (per-day pivot IC, paired
against the uncapped baseline, day-clustered): capping `analyst` verdicts to
+/-0.10 scores **+0.0121, t +2.46, same-sign halves** — the only one of eight
class-level interventions to clear the house bar. Analyst reads are weak on the
tape: oriented -0.882 pp, hit 47.1% over 745 rows.

These tests pin what the cap must and must NOT do — above all that it never
touches `macro_sector`, the class the search started from, which turned out to
be the BEST-reading class and measures NEGATIVE when capped in all three windows.
"""
import pytest

import src.analysis.sentiment as sent


# ── the cap ─────────────────────────────────────────────────────────────────

def test_it_caps_the_named_class_only():
    assert sent.apply_catalyst_cap(0.55, "analyst") == pytest.approx(0.03)
    assert sent.apply_catalyst_cap(-0.55, "analyst") == pytest.approx(-0.03)
    for other in ("earnings", "guidance", "ma_deal", "legal_regulatory", None, "", "none"):
        assert sent.apply_catalyst_cap(0.55, other) == pytest.approx(0.55)


def test_it_NEVER_caps_macro_sector():
    """The class the whole search started from. v6's LEAN rule is aimed at it and
    the audit flagged 42% of it as over-scored — but it is the BEST-reading
    class (+0.657 pp, hit 53.5% over 1,067 rows) and capping it measured
    NEGATIVE in the combined window, the live half AND the backfill half. A
    structural defect is not a defect until its consumer is measured."""
    assert sent.apply_catalyst_cap(0.9, "macro_sector") == pytest.approx(0.9)
    assert "macro_sector" not in sent._cap_classes()


def test_it_never_touches_the_SIGN():
    """A conviction limit, not a direction claim."""
    for s in (0.9, -0.9, 0.11, -0.11, 0.05, -0.05):
        out = sent.apply_catalyst_cap(s, "analyst")
        assert (out > 0) == (s > 0)
        assert abs(out) <= abs(s)


def test_a_score_already_inside_the_band_is_untouched():
    for s in (0.01, -0.01, 0.03, -0.03, 0.0):
        assert sent.apply_catalyst_cap(s, "analyst") == pytest.approx(s)


def test_the_class_list_is_configurable(monkeypatch):
    """`legal_regulatory` (+0.0060, t +1.64) and `ma_deal` (which did NOT
    replicate across labellers) are excluded on evidence, not on code — adding
    one later must be config."""
    from config.settings import settings
    assert sent.apply_catalyst_cap(0.5, "legal_regulatory") == pytest.approx(0.5)
    monkeypatch.setattr(settings, "catalyst_cap_classes", "analyst,legal_regulatory",
                        raising=False)
    assert sent.apply_catalyst_cap(0.5, "legal_regulatory") == pytest.approx(0.03)


def test_the_flag_switches_it_off(monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "enable_catalyst_class_cap", False, raising=False)
    assert sent.apply_catalyst_cap(0.55, "analyst") == pytest.approx(0.55)


def test_the_shipped_default_is_analyst_only():
    from config.settings import Settings
    assert Settings.model_fields["catalyst_cap_classes"].default == "analyst"
    # swept 2026-09-11: the curve is monotone with an interior optimum, and
    # 0.0 (abstain) is clearly worse than a small lean. 0.03 sits mid-plateau.
    assert Settings.model_fields["catalyst_cap_limit"].default == 0.03


# ── end to end ──────────────────────────────────────────────────────────────

def test_parse_applies_the_cap_on_every_engine_path():
    """It lives in `_parse_response`, so hosted and local verdicts and the
    salvage path all get it — the measurement was on the scorer's output, not
    on one engine's."""
    s, _r, c = sent._parse_response(
        '{"rationale": "r", "direction": "UP", "catalyst": "analyst", "score": 0.45}')
    assert s == pytest.approx(0.03) and c == "analyst"
    s2, _r2, _c2 = sent._parse_response(
        '{"rationale": "r", "direction": "UP", "catalyst": "macro_sector", "score": 0.45}')
    assert s2 == pytest.approx(0.45)
    # truncated / salvaged responses too
    s3, _r3, c3 = sent._parse_response(
        '{"rationale": "cut off, "catalyst": "analyst", "score": -0.62')
    assert s3 == pytest.approx(-0.03) and c3 == "analyst"


def test_the_cap_runs_AFTER_the_direction_enforcement():
    """Order matters: direction decides the SIGN, the cap limits the MAGNITUDE.
    Capping first then flipping would be the same number, but enforcing the
    direction on an already-capped value hides a disagreement the invariant is
    meant to surface."""
    s, _r, c = sent._parse_response(
        '{"rationale": "r", "direction": "DOWN", "catalyst": "analyst", "score": 0.62}')
    assert s == pytest.approx(-0.03) and c == "analyst"
    assert "apply_direction(" in __import__("inspect").getsource(sent._parse_response_uncapped)


def test_the_uncapped_parser_is_what_the_engine_loop_reads():
    """`_parse_response` caps; `_parse_response_uncapped` does not. The engine
    loop MUST use the uncapped one — see the test below for why."""
    assert sent._parse_response_uncapped(
        '{"rationale": "r", "catalyst": "analyst", "score": 0.45}') == (0.45, "r", "analyst")
    import inspect
    body = inspect.getsource(sent.analyse_sentiment)
    assert "_parse_response_uncapped(" in body
    assert "= _parse_response(" not in body


def test_the_logprob_EXPECTATION_is_measured_against_the_UNCAPPED_argmax():
    """With the cap inside the parser, the local branch handed `expected_score`
    an already-clamped argmax while the response text still read the model's own
    number — and `expected_score` REFUSES when those disagree (it reads that as
    logprobs describing a different response). So the expectation was silently
    switched OFF for every capped-class row: the one engine-level improvement
    shipped the day before, disabled on exactly the rows the cap touches, with
    no error and no log line.

    Verified directly: with tokens whose text says 0.45, `expected_score(t, 0.45)`
    returns 0.446 and `expected_score(t, 0.10)` returns None.

    (This is NOT what produced the two uncapped 0.4263 rows on the first live
    run — see the cache test below. Two separate defects, one ordering fix.)"""
    import inspect
    body = inspect.getsource(sent.analyse_sentiment)
    assert body.index("raw_score = _expected") < body.index(
        "raw_score = apply_catalyst_cap(raw_score, catalyst)")
    # ...and the expectation is still measured against the model's OWN argmax
    assert body.index("_argmax = raw_score") < body.index("_expected = expected_score(")


def test_the_cache_stores_the_UNCAPPED_verdict():
    """So that editing `catalyst_cap_classes` takes effect on the next tick
    rather than waiting out the 3 h TTL. The cap is a clamp on a stored number,
    not a different question to ask the model, so re-scoring to change it would
    be waste — and salting the cache key with the class list would force exactly
    that."""
    import inspect
    body = inspect.getsource(sent.analyse_sentiment)
    assert body.index("_sentiment_cache_put(") < body.index(
        "raw_score = apply_catalyst_cap(raw_score, catalyst)")


def test_a_cached_verdict_is_capped_on_the_way_out():
    """THIS is what produced the two uncapped `analyst` rows (0.4263) on the
    first live run after the cap shipped. A clamp placed in the parser cannot
    cover a cache hit, because a cache hit never reaches a parser — and with a
    180-minute TTL, every verdict scored in the three hours before the deploy
    kept being served uncapped. The cap now runs once, last, on the value EVERY
    path produced."""
    import inspect
    body = inspect.getsource(sent.analyse_sentiment)
    cap = body.index("raw_score = apply_catalyst_cap(raw_score, catalyst)")
    assert body.index('raw_score = float(cached["raw_score"])') < cap
    assert cap < body.index("adjusted_score = round(raw_score * precision_scale")


def test_the_news_family_epoch_covers_it():
    """It changes what the scorer OUTPUTS for a subset of rows, so history
    either side is not poolable by a calibration."""
    from src.signals.method_epochs import METHOD_SCORER_EPOCH, NEWS_FAMILY
    fam = NEWS_FAMILY
    assert len({str(METHOD_SCORER_EPOCH[m]) for m in fam}) == 1
    assert str(METHOD_SCORER_EPOCH["news"])[:10] >= "2026-09-11"


# ── ordering and provenance, found by reviewing the shipped change ──────────

def test_the_cap_judges_the_label_the_panel_will_STORE():
    """`fund_catalyst_override` retypes a FUND's company-event class to
    `macro_sector`, and that rewritten label is what `signals.news_catalyst`
    stores and what the +0.0121 was measured on.

    Applied before the override, a fund the model labelled `analyst` was clipped
    to the LEAN band and then persisted as `macro_sector` — a row whose stored
    class and applied cap disagree, and `macro_sector` is the one class the cap
    must never touch. So the override has to be resolved FIRST."""
    import inspect
    body = inspect.getsource(sent.analyse_sentiment)
    assert body.index("fund_catalyst_override(") < body.index(
        "raw_score = apply_catalyst_cap(raw_score, catalyst)")


def test_expected_score_still_IS_the_verdict_after_the_cap():
    """The column pair's contract is that `expected_score`, when present, IS
    `raw_score` — `argmax_score` is the greedy value kept for provenance. The
    cap broke that for capped classes: the verdict was clipped and the
    expectation persisted uncapped, so `signals.news_expected_score` carried the
    pre-cap number for exactly the rows the cap exists to limit."""
    import inspect
    body = inspect.getsource(sent.analyse_sentiment)
    cap = body.index("raw_score = apply_catalyst_cap(raw_score, catalyst)")
    assert body.index("_expected = raw_score", cap) > cap
    assert body.index("_argmax = apply_catalyst_cap(_argmax, catalyst)") > cap


def test_logprob_provenance_cannot_survive_an_engine_fallthrough():
    """The local branch sets `_argmax`/`_expected` and only THEN logs, tallies
    and writes the cache. An exception in any of those is caught by the engine
    loop, which moves on to DeepSeek — and without a per-attempt reset the
    DeepSeek verdict is persisted wearing the local model's `expected_score`."""
    import inspect
    body = inspect.getsource(sent.analyse_sentiment)
    loop = body.index("for engine in (order if raw_score is None else [])")
    reset = body.index("_expected = _argmax = None", loop)
    assert reset < body.index("_argmax = raw_score", loop)


def test_a_zero_limit_means_ABSTAIN_not_the_default(monkeypatch):
    """`catalyst_cap_limit = 0` clips the class to zero — an abstention, which
    was one of the eight arms measured. Read through `or 0.10` it silently
    became the default instead: a setting that looks configured and does
    something else, which is the exact class `tests/test_inert_settings.py`
    exists for."""
    from config.settings import settings
    monkeypatch.setattr(settings, "catalyst_cap_limit", 0.0, raising=False)
    assert sent.apply_catalyst_cap(0.55, "analyst") == 0.0
    assert sent.apply_catalyst_cap(-0.55, "analyst") == 0.0
    monkeypatch.setattr(settings, "catalyst_cap_limit", None, raising=False)
    assert sent.apply_catalyst_cap(0.55, "analyst") == pytest.approx(0.03)
