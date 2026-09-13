"""v7dir — name the direction before the number (2026-09-10).

A blind judge shown only the rationale — never the score, so it cannot anchor —
disagreed with the emitted SIGN on 7.8% of live verdicts: the model argues one
way and scores the other.

Detect-and-flip was rejected as unsafe: the best detector available (a bull/bear
word count) is 21% precise, because concessive netting is exactly what it gets
wrong — "beat estimates, HOWEVER the stock fell on weak guidance" is a correct
bearish net and the word count scores the losing clause. So the inconsistency is
made IMPOSSIBLE instead of caught: field order is generation order with
reasoning off, so a `direction` emitted between the rationale and the score is
chosen from the argument and the number has to follow it.

Measured paired on 90 digests, both arms judged blind: contradiction 7.8% ->
3.3%, direction agreeing with its own score 100% of the time on 99% of rows —
but McNemar one-sided exact **p = 0.109 (NOT significant)**, 10/90 SIGN FLIPS
and mean |score| 0.367 -> 0.422. Shipped on the user's directive with those
caveats recorded.
"""
import pytest

import src.analysis.sentiment as sent


# ── the enforcement ─────────────────────────────────────────────────────────

def test_the_number_follows_the_direction():
    """The invariant the whole change exists for. Today this is a NO-OP —
    direction and score agree 100% of the time — and it is kept as the
    guarantee: if a future checkpoint drifts, the contradiction cannot reach
    the panel."""
    assert sent.apply_direction(0.40, "UP") == pytest.approx(0.40)
    assert sent.apply_direction(-0.40, "DOWN") == pytest.approx(-0.40)
    # disagreement: keep the MAGNITUDE, take the direction's sign
    assert sent.apply_direction(-0.40, "UP") == pytest.approx(0.40)
    assert sent.apply_direction(0.40, "DOWN") == pytest.approx(-0.40)


def test_NONE_abstains():
    """It maps onto v6's own two abstention cases, and a 0.0 is an ABSTENTION
    in the rank-consumed combine — the ticker leaves the cross-section."""
    assert sent.apply_direction(0.40, "NONE") == 0.0
    assert sent.apply_direction(-0.9, "none") == 0.0


def test_a_missing_or_junk_direction_leaves_the_score_alone():
    """Fail-soft: a cached pre-v7dir verdict, or a model that skipped the
    field, must not lose its read."""
    for d in (None, "", "sideways", "0.4", "UPWARD-ISH"):
        assert sent.apply_direction(0.35, d) == pytest.approx(0.35)
    assert sent.apply_direction(0.0, "UP") == 0.0


def test_it_is_case_and_whitespace_tolerant():
    assert sent.apply_direction(-0.2, " up ") == pytest.approx(0.2)
    assert sent.apply_direction(0.2, "Down") == pytest.approx(-0.2)


# ── the prompt ──────────────────────────────────────────────────────────────

def test_v7dir_is_derived_from_v6_so_the_shared_rubric_cannot_drift():
    """Two independent literals is how the confidence rubric drifted; here the
    DIFF is the experiment, so a v6 edit that invalidates a replacement fails at
    IMPORT rather than shipping a half-converted prompt."""
    with pytest.raises(RuntimeError, match="no longer contains"):
        sent._build_direction_prefix("a prompt sharing nothing with v6")
    for shared in ("CROSS-SECTIONALLY", "SOURCE TIERS", "PRICED-IN CHECK", "±0.01–0.10 LEAN"):
        assert shared in sent._SENTIMENT_PREFIX_DIRECTION


def test_the_direction_is_asked_for_BEFORE_the_score():
    """Field order IS generation order with reasoning off. A direction emitted
    after the number would be a rationalisation of it, which is the defect, not
    the fix."""
    p = sent._SENTIMENT_PREFIX_DIRECTION
    schema = p.split("Respond with ONLY")[1]
    assert schema.index('"direction"') < schema.index('"score"')
    skeleton = [ln for ln in p.splitlines() if ln.startswith('{"rationale"')][-1]
    assert skeleton.index('"direction"') < skeleton.index('"score"')
    assert skeleton.index('"rationale"') < skeleton.index('"direction"')


def test_it_carries_its_own_cache_salt():
    """A prompt change keeping the old salt serves OLD-prompt verdicts for the
    whole TTL — the 2026-08-14 defect `_SENT_PROMPT_VERSION` exists to stop."""
    assert sent._SENT_PROMPT_VERSION_DIRECTION != sent._SENT_PROMPT_VERSION
    assert sent._prompt_pair()[1] == sent._SENT_PROMPT_VERSION_DIRECTION


def test_v7d_still_wins_when_both_flags_are_on(monkeypatch):
    """The decomposition arm is measured-and-rejected but revivable; if someone
    turns it on for a re-test it must not be silently overridden."""
    from config.settings import settings
    monkeypatch.setattr(settings, "enable_priced_in_decomposition", True, raising=False)
    monkeypatch.setattr(settings, "enable_direction_field", True, raising=False)
    assert sent._prompt_pair()[1] == sent._SENT_PROMPT_VERSION_DECOMPOSED


def test_turning_the_flag_off_restores_v6_exactly(monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "enable_priced_in_decomposition", False, raising=False)
    monkeypatch.setattr(settings, "enable_direction_field", False, raising=False)
    assert sent._prompt_pair() == (sent._SENTIMENT_PREFIX, sent._SENT_PROMPT_VERSION)


# ── end to end ──────────────────────────────────────────────────────────────

def test_parse_applies_the_direction():
    s, _r, c = sent._parse_response(
        '{"rationale": "r", "direction": "DOWN", "catalyst": "guidance", "score": 0.35}')
    assert s == pytest.approx(-0.35) and c == "guidance"
    s2, _r2, _c2 = sent._parse_response(
        '{"rationale": "r", "direction": "NONE", "catalyst": "none", "score": 0.35}')
    assert s2 == 0.0


def test_the_salvage_path_applies_it_too():
    """A truncated response that still carries both fields is worth saving —
    through the same enforcement, never around it."""
    s, _r, _c = sent._parse_response(
        '{"rationale": "cut off mid sentence, "direction": "UP", '
        '"catalyst": "earnings", "score": -0.40}')
    assert s == pytest.approx(0.40)


def test_a_v6_era_response_is_unaffected():
    """Cached verdicts and the shadow/repair paths still emit no direction.

    Uses `guidance`, not `analyst`: since 2026-09-11 the analyst class is held
    to the LEAN band, so this fixture would have been testing the cap instead of
    the thing it is named for."""
    assert sent._parse_response(
        '{"rationale": "x", "catalyst": "guidance", "score": 0.41}') == (0.41, "x", "guidance")


def test_the_news_family_epoch_moved_for_it():
    """It changes the SIGN of ~11% of verdicts — categorically a different
    scorer, not a refinement — so the whole family shares a boundary placed at
    the deploy."""
    from src.signals.method_epochs import METHOD_SCORER_EPOCH, NEWS_FAMILY
    fam = NEWS_FAMILY
    stamps = {str(METHOD_SCORER_EPOCH[m]) for m in fam}
    assert len(stamps) == 1, stamps
    assert str(METHOD_SCORER_EPOCH["news"])[:10] >= "2026-09-10"
