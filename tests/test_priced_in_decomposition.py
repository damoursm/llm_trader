"""The priced-in discount, made arithmetic (sentiment prompt v7d, 2026-09-09).

WHY THIS EXISTS. Measured over 146 re-scored clusters (38 days, rationales
kept): v6's rationale field REQUIRES the model to name "what is already priced
in", so it does — in 77% of rationales — and then scores as though it had not.
Clusters whose rationale says the move is already made are scored HIGHER, not
lower (mean |s| 0.327 vs 0.309); 48% of them still land in the CLEAR band or
above, 16% in the HIGH band, and the worst case scored -0.85 on a rationale
reading "the stock has already reacted, with a steep decline".

The fix splits the judgement into two required fields and multiplies them
OURSELVES, so the discount cannot be skipped. These tests pin the properties
that make that safe — above all that a discounted read stays a signed LEAN
instead of collapsing into an ABSTENTION, which in the rank-consumed combine
removes the ticker from the cross-section entirely.

⚠ THE FAMILY IS CLOSED AND THE FLAG IS OFF (2026-09-11). Four independent ways
of acting on "already priced in" were measured and none helped: this
decomposition (the model states the fraction), the price-measured version
(`news_unpriced`), the volume-measured version (surge decay back to the
pre-catalyst level), and a discount on rows whose rationale makes the claim in
words. The reason is in the last section of this file: the CLAIM IS NOT A
MEASUREMENT. The machinery stays tested and one flag from returning, per the
revivable-dead-branch convention — but read that section before attempting a
fifth.
"""
import re

import pytest

import src.analysis.sentiment as sent


# ── the arithmetic ──────────────────────────────────────────────────────────

def test_score_is_the_product_and_the_discount_cannot_be_skipped():
    """The one property the whole change exists for: a catalyst the model calls
    largely priced CANNOT come back at full conviction."""
    assert sent.apply_priced_in(0.80, 0.00) == pytest.approx(0.80)
    assert sent.apply_priced_in(0.80, 0.50) == pytest.approx(0.40)
    assert sent.apply_priced_in(0.80, 0.90) == pytest.approx(0.08)
    # the sample's worst case: a HIGH-band catalyst the rationale says is spent
    assert abs(sent.apply_priced_in(-0.85, 0.85)) < 0.15


def test_a_discounted_read_stays_a_lean_never_an_abstention():
    """A 0.0 is an ABSTENTION — the ticker leaves the ranked cross-section
    altogether. A well-priced catalyst is a WEAK read, not an absent one, so it
    floors at the LEAN band instead of rounding away. This is the v5/v6 lesson
    restated in arithmetic."""
    for pi in (0.995, 0.999, 0.9999):
        v = sent.apply_priced_in(0.50, pi)
        assert v != 0.0 and abs(v) == pytest.approx(sent._MIN_LEAN)
    assert sent.apply_priced_in(-0.50, 0.999) == pytest.approx(-sent._MIN_LEAN)


def test_the_two_abstention_cases_still_reach_exactly_zero():
    """(a) nothing connects to the target -> no catalyst at all; (b) a genuinely
    nil remaining move (an all-cash target pinned at the offer) -> the fraction
    at 1.0. Those are the ONLY routes to 0.0."""
    assert sent.apply_priced_in(0.0, 0.0) == 0.0
    assert sent.apply_priced_in(0.0, 0.9) == 0.0       # no catalyst wins
    assert sent.apply_priced_in(0.75, 1.0) == 0.0      # case (b)


def test_an_overshoot_flips_the_sign():
    """What the v6 prose asked for and could not express: 'after an outsized
    one-day spike that residual is small and often NEGATIVE'."""
    assert sent.apply_priced_in(0.60, 1.25) < 0
    assert sent.apply_priced_in(-0.60, 1.25) > 0
    # bounded, so a mis-stated fraction cannot invert a read arbitrarily far
    assert sent.apply_priced_in(1.0, 99.0) == pytest.approx(-(sent.PRICED_IN_MAX - 1.0))


def test_a_negative_fraction_cannot_amplify_past_the_band():
    """A model inventing 'not yet priced, so -0.2' must not push a catalyst
    beyond the magnitude band it was placed in."""
    assert sent.apply_priced_in(0.50, -0.20) == pytest.approx(0.50)
    assert abs(sent.apply_priced_in(1.0, -5.0)) <= 1.0


def test_precision_survives_the_multiplication():
    """The product of two two-decimal numbers is FINER than either, which is a
    side benefit on an engine that quantises its own output onto a 0.05 grid.
    A coarse round here would re-merge what the two fields separated."""
    vals = {sent.apply_priced_in(cs, pi)
            for cs in (0.25, 0.30, 0.35) for pi in (0.10, 0.15, 0.20)}
    assert len(vals) == 9


# ── the prompt ──────────────────────────────────────────────────────────────

def test_v7d_is_derived_from_v6_so_the_shared_rubric_cannot_drift():
    """Two independent literals is how the confidence rubric drifted. v7d is
    built by asserted replacement of v6, so a v6 edit that invalidates one of
    them fails at IMPORT rather than shipping a half-converted prompt."""
    with pytest.raises(RuntimeError, match="no longer contains"):
        sent._build_decomposed_prefix("a prompt that shares nothing with v6")
    # the parts v7d does not touch are still there verbatim
    for shared in ("CROSS-SECTIONALLY", "SOURCE TIERS", "FUNDS:", "±0.01–0.10 LEAN"):
        assert shared in sent._SENTIMENT_PREFIX
        assert shared in sent._SENTIMENT_PREFIX_DECOMPOSED


def test_v7d_asks_for_the_two_fields_and_forbids_the_score():
    """The model must not emit a `score`: the arithmetic is ours. If it kept
    emitting one it would anchor on it and the decomposition would be theatre."""
    p = sent._SENTIMENT_PREFIX_DECOMPOSED
    schema = p.split("Respond with ONLY")[1]
    assert '"catalyst_score"' in schema and '"priced_in"' in schema
    assert 'Do NOT output a "score" field' in p
    assert '"score": <signed two-decimal number>' not in p     # skeleton replaced


def test_v7d_moves_the_discount_out_of_the_band_placement():
    """The discount must be taken ONCE. v6 asked for it inside the placement
    rule AND in its own section; if the band placement still mentioned it, the
    model would shrink the catalyst and we would shrink it again."""
    p = sent._SENTIMENT_PREFIX_DECOMPOSED
    place = [ln for ln in p.splitlines() if ln.startswith('- PLACE "catalyst_score"')]
    assert len(place) == 1
    assert "does NOT belong here" in place[0]
    assert "never shrink it twice" in p
    assert "PRICED-IN FRACTION" in p and "PRICED-IN CHECK" not in p


def test_v7d_plants_no_example_numbers():
    """Twice now a number written into a prompt became a modal output (sentiment
    v2, confidence placement v1). The only numerals v7d adds are band edges and
    the endpoint definitions of the fraction."""
    added = set(re.findall(r"-?\d+\.\d+", sent._SENTIMENT_PREFIX_DECOMPOSED)) - \
        set(re.findall(r"-?\d+\.\d+", sent._SENTIMENT_PREFIX))
    assert added <= {"0.0", "1.0", "-1.0", "+1.0"}, added


# ── wiring ──────────────────────────────────────────────────────────────────

def test_the_flag_routes_the_prompt_and_its_own_cache_salt(monkeypatch):
    """A prompt change that kept the v6 salt would serve v6 verdicts from cache
    for the whole TTL — the 2026-08-14 defect `_SENT_PROMPT_VERSION` exists to
    prevent. v7d carries its own version, so flipping the flag RE-SCORES."""
    from config.settings import settings
    monkeypatch.setattr(settings, "enable_priced_in_decomposition", False, raising=False)
    # v7dir also sits between v6 and the caller since 2026-09-11
    monkeypatch.setattr(settings, "enable_direction_field", False, raising=False)
    assert sent._prompt_pair() == (sent._SENTIMENT_PREFIX, sent._SENT_PROMPT_VERSION)
    monkeypatch.setattr(settings, "enable_priced_in_decomposition", True, raising=False)
    prefix, version = sent._prompt_pair()
    assert prefix is sent._SENTIMENT_PREFIX_DECOMPOSED
    assert version != sent._SENT_PROMPT_VERSION


def test_the_flag_is_off_by_default():
    """Default OFF until the paired offline read clears the house bar. Asserted
    on the FIELD DEFAULT, not on the live settings object — pinning deployment
    config in a test makes the test fail the day the config is legitimately
    changed."""
    from config.settings import Settings
    assert Settings.model_fields["enable_priced_in_decomposition"].default is False


def test_the_two_prompts_key_different_cache_entries(monkeypatch):
    """The verdict cache outlives the flag flip, so the two prompts must never
    share a key: a v6 verdict served under v7d would carry an undiscounted score
    that no longer means what the column says it means."""
    from config.settings import settings
    from src.models import NewsArticle
    from datetime import datetime, timezone
    monkeypatch.setattr(settings, "enable_priced_in_decomposition", False, raising=False)
    arts = [NewsArticle(title="t", summary="s", source="s", url="u",
                        published_at=datetime(2026, 9, 1, tzinfo=timezone.utc))]
    key_off = sent._sentiment_cache_key("AAA", "local", arts)
    monkeypatch.setattr(settings, "enable_priced_in_decomposition", True, raising=False)
    assert sent._sentiment_cache_key("AAA", "local", arts) != key_off


def test_a_monkeypatched_prompt_still_wins_while_the_flag_is_off(monkeypatch):
    """The prompt A/B scripts swap `_SENTIMENT_PREFIX` / `_SENT_PROMPT_VERSION`
    on the module. `_prompt_pair` must read those attributes, not a snapshot, or
    those harnesses would silently keep scoring the live prompt."""
    from config.settings import settings
    monkeypatch.setattr(settings, "enable_priced_in_decomposition", False, raising=False)
    monkeypatch.setattr(settings, "enable_direction_field", False, raising=False)
    monkeypatch.setattr(sent, "_SENTIMENT_PREFIX", "ARM PROMPT", raising=False)
    monkeypatch.setattr(sent, "_SENT_PROMPT_VERSION", "arm-v0", raising=False)
    assert sent._prompt_pair() == ("ARM PROMPT", "arm-v0")


# ── the parser ──────────────────────────────────────────────────────────────

def test_parse_computes_the_score_and_never_trusts_a_model_supplied_one():
    """Even if a model emits a `score` alongside the two fields, ours wins."""
    s, r, c = sent._parse_response(
        '{"rationale": "r", "catalyst": "guidance", "catalyst_score": 0.60, '
        '"priced_in": 0.75, "score": 0.60}')
    assert s == pytest.approx(0.15) and c == "guidance" and r == "r"


def test_legacy_v6_responses_still_parse():
    """Every cached pre-v7d verdict, and every v6 run of the shadow/repair
    paths, must keep working unchanged.

    Uses `guidance`, not `analyst`: the analyst class is held to the LEAN
    band since 2026-09-11, so that fixture would have been testing the cap
    rather than legacy parsing."""
    assert sent._parse_response(
        '{"rationale": "x", "catalyst": "guidance", "score": 0.41}') == (0.41, "x", "guidance")


def test_the_number_repair_covers_the_new_fields():
    """JSON forbids a leading `+` and a bare `.75`, and qwen3:8b emits both. A
    repair anchored on `"score"` alone would go INERT on the very prompt that
    removed that field — and a perfect verdict would again be lost as a neutral
    0.0 inside the 0.40-weight `news` method."""
    s, _r, _c = sent._parse_response(
        '{"rationale": "r", "catalyst": "guidance", "catalyst_score": +0.60, '
        '"priced_in": .75}')
    assert s == pytest.approx(0.15)


def test_the_repair_never_touches_a_plus_in_the_rationale():
    s, r, _c = sent._parse_response(
        '{"rationale": "revenue +12% yoy and margins +3pp", "catalyst": "earnings", '
        '"catalyst_score": 0.40, "priced_in": 0.0}')
    assert r == "revenue +12% yoy and margins +3pp" and s == pytest.approx(0.40)


def test_a_truncated_v7d_response_is_refused_not_read_undiscounted():
    """A response that lost its `priced_in` to a token cap must NOT be salvaged
    as an undiscounted catalyst_score — that is exactly the failure the change
    is built to remove, arriving through the back door."""
    with pytest.raises(Exception):
        sent._parse_response('{"rationale": "trunc", "catalyst": "analyst", '
                             '"catalyst_score": 0.85')


def test_the_salvage_path_applies_the_discount_when_both_fields_survive():
    """A response that is invalid JSON but carries both numbers is still worth
    saving — through the same arithmetic, never around it."""
    s, _r, c = sent._parse_response(
        '{"rationale": "cut off mid sentence, "catalyst": "fda_clinical", '
        '"catalyst_score": -0.80, "priced_in": 0.50}')
    assert s == pytest.approx(-0.40) and c == "fda_clinical"


# ── why the family is closed ────────────────────────────────────────────────
#
# Measured 2026-09-11 on 4,418 stored rationales (`catalyst_repairs.rationale`,
# 4 days, both engines) joined to the move the tape had actually made — which
# the pipeline already computes exactly, as `news_unpriced = news x clip(1 - z/2)`
# where z is the realized move in the news direction since the story's own
# anchor, in the volatility that ruled when it landed. So the model's claim can
# be scored against the thing it is a claim ABOUT, with no forward label needed
# (which matters: scorer rationales are stored only from 2026-09-08 and settled
# pivot labels stop at 2026-09-08, so exactly ONE day overlaps — a labelled test
# of this cohort is not possible before roughly 2026-09-25).
#
#   * the claim fires on 44.4% of rationales under a detector that REQUIRES an
#     intensifier bound to the phrase ("already/largely/fully ... priced in");
#     a bare mention fires on 53.4%, because v6's rubric MANDATES the check.
#   * rows making the claim score HIGHER, not lower (mean |raw| 0.351 vs 0.294,
#     MINOR+ 65.4% vs 54.9%) — they are the bigger, better-covered stories.
#   * the claim barely tracks the move: z(claim) - z(no claim) = +0.096 sigma,
#     +0.082 stratified by digest size.
#   * PER ENGINE IT POINTS OPPOSITE WAYS: local qwen +0.164, DeepSeek -0.108.
#     Any uniform rule therefore averages a roughly-right response on half the
#     rows with a backwards one on the other half — which is what four measured
#     failures look like from the inside.
#   * PAIRED on the 1,771 digests BOTH engines rationalised (same digest, same
#     instant, so day, ticker, story size and regime all cancel): the two agree
#     on the claim 53.5% of the time against 49.3% expected by chance — Cohen's
#     kappa ~0.08, i.e. no agreement at all. Local-only claims sit at z +0.656,
#     DeepSeek-only at +0.489 (Welch t +2.74 between them), and the CONSENSUS
#     claim — both engines saying priced-in — sits at z -0.056 BELOW the rows
#     neither flagged. The most confident form of the claim is anti-grounded.
#
# Conclusion: the claim is a noisy, engine-inconsistent, partly anti-correlated
# estimator of a quantity the pipeline measures directly. Acting on the text can
# only add noise on top of the measurement. The measured version is already in
# production per SIDE and fitted rather than hand-signed — `news_bear_fresh`
# (un-fallen tape guards a bear read) and `news_bull_fresh` shipped INVERTED
# (un-moved tape is anti-signal for a bull read) — and `news_unpriced` /
# `news_unpriced_all` ride the stacker feature set for the model to sign.
#
# NOT DONE, deliberately: the prompt's PRICED-IN CHECK block was left in place.
# The stated claim being noise is not evidence that asking for it harms the
# SCORE, v6 as a whole measured well, and removing it would cost a prompt
# version, a news-family epoch and a cache flush to test a hypothesis nothing
# supports.


def test_the_decomposition_arm_stays_OFF_by_default():
    """Pinned rather than deleted (revivable-dead-branch convention). If a fifth
    attempt is coming, it has to turn this on deliberately and read the section
    above first."""
    from config.settings import Settings
    assert Settings.model_fields["enable_priced_in_decomposition"].default is False


def test_the_measured_replacement_is_what_ships():
    """The claim is out; the measured move is in, per side and fitted. These are
    the methods that carry the idea in production, so a refactor that drops one
    should fail here rather than quietly reopening the gap."""
    from src.signals import news_bull_fresh, news_bear_fresh  # noqa: F401
    from config.settings import Settings
    assert Settings.model_fields["news_bull_fresh_invert"].default is True
    assert Settings.model_fields["enable_news_priced_in"].default is True
