"""IV Expression (`src/signals/iv_expr.py`) — a weighted method.

Reads TRUE market-implied vol out of the options chain already fetched for GEX
and combines it with the options market's own positioning (`oi_skew`). Its
defining property is that the score's SIGN FLIPS across the IV regime: the same
bullish skew is bullish when options are cheap and BEARISH when they are rich
(vol mean-reverts, so a richly-priced directional bet is the thing to fade).

That makes this the one scorer where a "simplification" is most likely to be a
silent inversion — which is why the regime boundaries and the sign in each
branch are pinned individually rather than through a couple of happy paths.

The IV rank itself has two modes: a real percentile against the ticker's own
trailing `cache/gex_*.json` history, and a universe-relative fallback for
cold-cache days. Every test points `CACHE_DIR` at tmp_path so the developer's
real gex caches never decide an assertion.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from src.models import GEXContext, GEXSignal
from src.signals import iv_expr as ie


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ie, "CACHE_DIR", tmp_path)


def _sig(ticker="AAA", em=5.0, skew=0.0, gex="NEUTRAL") -> GEXSignal:
    return GEXSignal(
        ticker=ticker, spot_price=100.0, net_gex_bn=0.0, gex_normalized=0.0,
        gex_signal=gex, gamma_flip=100.0, max_pain=100.0, expected_move_pct=em,
        max_pain_bias="NEUTRAL", oi_skew=skew, dominant_expiry="2026-09-18",
        report_date=date.today(), summary="s")


def _ctx(*sigs) -> GEXContext:
    return GEXContext(signals=list(sigs), report_date=date.today(), summary="s")


def _write_history(tmp_path, ticker, values, start_days_ago=1):
    """One prior gex cache file per value, each on its own past report_date.

    MERGES into an existing file rather than overwriting it — the real caches
    hold every ticker for a given day, and a test writing two tickers would
    otherwise silently erase the first one's history (both map to the same
    `gex_<date>.json` names)."""
    for i, v in enumerate(values):
        d = date.today() - timedelta(days=start_days_ago + i)
        path = tmp_path / f"gex_{d.isoformat()}.json"
        payload = {"report_date": d.isoformat(), "signals": []}
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
        payload["signals"].append({"ticker": ticker, "expected_move_pct": v})
        path.write_text(json.dumps(payload), encoding="utf-8")


# ── no data ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ctx", [None, GEXContext(signals=[], report_date=date.today(),
                                                  summary="s")])
def test_a_ticker_with_no_options_data_abstains(ctx):
    """Neutral IV rank of 50 and a zero score — the method contributes nothing
    rather than a confident-looking reading built from an absent chain."""
    assert ie.compute_iv_expr_score("AAA", ctx) == (0.0, 50.0, 0.0, "NO_OPTIONS_DATA")


@pytest.mark.parametrize("em", [0.0, -1.0])
def test_an_impossible_expected_move_abstains(em):
    """A zero or negative implied move is a broken chain, not a quiet one — the
    percentile and the universe ratio are both undefined there. (The `None`
    branch in the guard is unreachable through the model, which requires a
    float; it is defence against a hand-built context.)"""
    assert ie.compute_iv_expr_score("AAA", _ctx(_sig(em=em)))[3] == "NO_OPTIONS_DATA"


def test_ticker_lookup_is_case_insensitive():
    ctx = _ctx(_sig(ticker="aaa", em=5.0, skew=0.9))
    assert ie.compute_iv_expr_score("AAA", ctx)[3] != "NO_OPTIONS_DATA"


# ── the regime sign flip ────────────────────────────────────────────────────

def test_rich_options_FADE_a_strong_skew(tmp_path):
    """HIGH IV + decisive positioning → score AGAINST the options direction.
    The single most invertible line in the module."""
    _write_history(tmp_path, "AAA", [1.0] * 8)            # today's 9.0 ranks 100
    score, rank, skew, label = ie.compute_iv_expr_score(
        "AAA", _ctx(_sig(em=9.0, skew=+0.9)))
    assert rank >= ie._HIGH_IV
    assert label == "FADE_PREMIUM"
    assert score < 0, "a bullish skew at rich IV must score BEARISH"
    assert skew == 0.9

    score_dn, _r, _s, label_dn = ie.compute_iv_expr_score(
        "AAA", _ctx(_sig(em=9.0, skew=-0.9)))
    assert label_dn == "FADE_PREMIUM" and score_dn > 0


def test_cheap_options_CONFIRM_a_strong_skew(tmp_path):
    """LOW IV + decisive positioning → score WITH the options direction. Same
    skew as above, opposite sign, purely because premium is cheap."""
    _write_history(tmp_path, "AAA", [9.0] * 8)            # today's 1.0 ranks 0
    up = ie.compute_iv_expr_score("AAA", _ctx(_sig(em=1.0, skew=+0.9)))
    dn = ie.compute_iv_expr_score("AAA", _ctx(_sig(em=1.0, skew=-0.9)))
    assert up[1] <= ie._LOW_IV
    assert up[3] == "CHEAP_DIRECTIONAL_LONG" and up[0] > 0
    assert dn[3] == "CHEAP_DIRECTIONAL_SHORT" and dn[0] < 0


def test_the_same_skew_flips_sign_across_the_iv_regime(tmp_path):
    """Stated as the property rather than per-branch: this IS the method."""
    _write_history(tmp_path, "RICH", [1.0] * 8)
    _write_history(tmp_path, "CHEAP", [9.0] * 8)
    rich = ie.compute_iv_expr_score("RICH", _ctx(_sig(ticker="RICH", em=9.0, skew=0.9)))
    cheap = ie.compute_iv_expr_score("CHEAP", _ctx(_sig(ticker="CHEAP", em=1.0, skew=0.9)))
    assert rich[0] < 0 < cheap[0]


def test_weak_skew_gets_the_muted_labels(tmp_path):
    _write_history(tmp_path, "AAA", [1.0] * 8)
    weak_rich = ie.compute_iv_expr_score("AAA", _ctx(_sig(em=9.0, skew=0.2)))
    assert weak_rich[3] == "EXPENSIVE_NEUTRAL"
    assert abs(weak_rich[0]) < 0.55

    _write_history(tmp_path, "BBB", [9.0] * 8)
    weak_cheap = ie.compute_iv_expr_score("BBB", _ctx(_sig(ticker="BBB", em=1.0, skew=0.2)))
    assert weak_cheap[3] == "CHEAP_COMPLACENT"
    assert 0 < weak_cheap[0] < 0.55


def test_a_negligible_skew_scores_zero_in_both_extremes(tmp_path):
    """No directional information from the options market — the IV regime alone
    is not a direction."""
    _write_history(tmp_path, "AAA", [1.0] * 8)
    assert ie.compute_iv_expr_score("AAA", _ctx(_sig(em=9.0, skew=0.01)))[0] == 0.0
    _write_history(tmp_path, "BBB", [9.0] * 8)
    assert ie.compute_iv_expr_score("BBB", _ctx(_sig(ticker="BBB", em=1.0,
                                                     skew=0.01)))[0] == 0.0


def test_mid_iv_uses_the_raw_skew_for_a_smooth_gradient(tmp_path):
    """Between the bands the score is proportional to the skew rather than
    bucketed, so a name drifting across the mid regime doesn't jump."""
    _write_history(tmp_path, "AAA", [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5])
    small = ie.compute_iv_expr_score("AAA", _ctx(_sig(em=5.6, skew=0.20)))
    big = ie.compute_iv_expr_score("AAA", _ctx(_sig(em=5.6, skew=0.45)))
    assert ie._LOW_IV < small[1] < ie._HIGH_IV
    assert 0 < small[0] < big[0]
    assert big[3] == "MID_IV_DIRECTIONAL"


# ── the gamma adjustment ────────────────────────────────────────────────────

def test_amplified_gamma_pushes_further_in_the_score_direction(tmp_path):
    """Short-gamma dealers accelerate moves — the boost is applied along the
    score's OWN sign, so it never flips a verdict, only strengthens it."""
    _write_history(tmp_path, "AAA", [9.0] * 8)
    plain = ie.compute_iv_expr_score("AAA", _ctx(_sig(em=1.0, skew=0.9)))
    amped = ie.compute_iv_expr_score("AAA", _ctx(_sig(em=1.0, skew=0.9, gex="AMPLIFIED")))
    assert amped[0] > plain[0] > 0
    assert amped[0] == pytest.approx(plain[0] + 0.10, abs=1e-6)

    neg = ie.compute_iv_expr_score("AAA", _ctx(_sig(em=1.0, skew=-0.9, gex="AMPLIFIED")))
    neg_plain = ie.compute_iv_expr_score("AAA", _ctx(_sig(em=1.0, skew=-0.9)))
    assert neg[0] < neg_plain[0] < 0


def test_pinned_gamma_gets_no_boost(tmp_path):
    _write_history(tmp_path, "AAA", [9.0] * 8)
    plain = ie.compute_iv_expr_score("AAA", _ctx(_sig(em=1.0, skew=0.9)))
    pinned = ie.compute_iv_expr_score("AAA", _ctx(_sig(em=1.0, skew=0.9, gex="PINNED")))
    assert pinned[0] == plain[0]


def test_amplified_does_not_resurrect_a_zero_score(tmp_path):
    """A no-view must stay a no-view — the boost only amplifies an existing
    direction."""
    _write_history(tmp_path, "AAA", [9.0] * 8)
    assert ie.compute_iv_expr_score(
        "AAA", _ctx(_sig(em=1.0, skew=0.0, gex="AMPLIFIED")))[0] == 0.0


def test_score_stays_bounded(tmp_path):
    _write_history(tmp_path, "AAA", [9.0] * 8)
    for skew in (-1.0, 1.0):
        s = ie.compute_iv_expr_score("AAA", _ctx(_sig(em=1.0, skew=skew,
                                                      gex="AMPLIFIED")))[0]
        assert -1.0 <= s <= 1.0


# ── IV ranking ──────────────────────────────────────────────────────────────

def test_history_mode_is_a_percentile_of_prior_readings(tmp_path):
    _write_history(tmp_path, "AAA", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    # 3 of 6 prior readings are below 3.5 -> rank 50.
    assert ie.compute_iv_expr_score("AAA", _ctx(_sig(em=3.5)))[1] == pytest.approx(50.0)
    assert ie.compute_iv_expr_score("AAA", _ctx(_sig(em=0.5)))[1] == pytest.approx(0.0)
    assert ie.compute_iv_expr_score("AAA", _ctx(_sig(em=9.9)))[1] == pytest.approx(100.0)


def test_thin_history_falls_back_to_the_universe(tmp_path):
    """Fewer than `_MIN_HIST_N` readings is not a percentile. The fallback ranks
    the ticker against today's universe median so cold-cache days still score."""
    _write_history(tmp_path, "AAA", [1.0] * (ie._MIN_HIST_N - 1))
    ctx = _ctx(_sig(ticker="AAA", em=10.0, skew=0.9), _sig(ticker="B", em=5.0),
               _sig(ticker="C", em=5.0), _sig(ticker="D", em=5.0))
    rank = ie.compute_iv_expr_score("AAA", ctx)[1]
    assert rank > 50.0, "twice the universe median must rank as expensive"


def test_universe_fallback_needs_a_real_cross_section():
    """Under three usable readings there is no median worth the name; the rank
    collapses to neutral rather than inventing a regime."""
    ctx = _ctx(_sig(ticker="AAA", em=10.0), _sig(ticker="B", em=5.0))
    assert ie.compute_iv_expr_score("AAA", ctx)[1] == pytest.approx(50.0)


def test_universe_median_ignores_missing_and_non_positive_readings():
    ctx = _ctx(_sig(em=5.0), _sig(ticker="B", em=5.0), _sig(ticker="C", em=5.0),
               _sig(ticker="D", em=0.0))
    assert ie._universe_median_iv(ctx) == pytest.approx(5.0)
    assert ie._universe_median_iv(None) is None
    assert ie._universe_median_iv(_ctx()) is None


def test_history_excludes_today_and_anything_beyond_the_window(tmp_path):
    """Today's own cache would make the ranking self-referential; a 6-month-old
    reading is a different vol regime."""
    today = date.today()
    (tmp_path / f"gex_{today.isoformat()}.json").write_text(json.dumps({
        "report_date": today.isoformat(),
        "signals": [{"ticker": "AAA", "expected_move_pct": 99.0}]}), encoding="utf-8")
    stale = today - timedelta(days=ie._HISTORY_DAYS + 5)
    (tmp_path / f"gex_{stale.isoformat()}.json").write_text(json.dumps({
        "report_date": stale.isoformat(),
        "signals": [{"ticker": "AAA", "expected_move_pct": 88.0}]}), encoding="utf-8")
    _write_history(tmp_path, "AAA", [1.0, 2.0])
    hist = ie._load_iv_history("AAA", today)
    assert hist == [2.0, 1.0] or hist == [1.0, 2.0]
    assert 99.0 not in hist and 88.0 not in hist


def test_history_is_returned_oldest_first(tmp_path):
    _write_history(tmp_path, "AAA", [10.0, 20.0, 30.0])    # 1, 2, 3 days ago
    assert ie._load_iv_history("AAA", date.today()) == [30.0, 20.0, 10.0]


def test_history_reading_is_fail_soft(tmp_path):
    """A truncated or half-written cache file must not take the scorer down."""
    (tmp_path / "gex_bad.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "gex_nodate.json").write_text(json.dumps({"signals": []}),
                                              encoding="utf-8")
    _write_history(tmp_path, "AAA", [1.0, 2.0])
    assert len(ie._load_iv_history("AAA", date.today())) == 2


def test_missing_cache_dir_yields_no_history(tmp_path, monkeypatch):
    monkeypatch.setattr(ie, "CACHE_DIR", tmp_path / "absent")
    assert ie._load_iv_history("AAA", date.today()) == []


def test_history_is_scoped_to_the_ticker(tmp_path):
    _write_history(tmp_path, "OTHER", [1.0, 2.0, 3.0])
    assert ie._load_iv_history("AAA", date.today()) == []


# ── the band constants ──────────────────────────────────────────────────────

def test_the_regime_bands_leave_a_real_middle():
    assert 0 < ie._LOW_IV < ie._HIGH_IV < 100
    assert 0 < ie._STRONG_OI_SKEW <= 1.0
