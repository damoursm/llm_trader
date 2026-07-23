"""Cross-family agreement + tape confirmation (2026-07-19).

METHOD_FAMILIES rolls the 21 weighted methods into 7 independent information
families; family-level votes (magnitude-weighted, real vote threshold) replace
pseudo-replication (five correlated technicals ≠ five confirmations) in the
confidence factor and are exposed to the synthesis prompt. Tape confirmation is
a score-independent raw price/volume state check (range position + signed
volume share + RVOL×direction) that qualifies agreement, never joins the combine.
"""

import json

import pandas as pd
import pytest

from config.settings import settings
import src.signals.aggregator as agg
from src.signals.agreement import (METHOD_FAMILIES, FAMILY_OF, FamilyAgreement,
                                   compute_family_agreement,
                                   compute_tape_confirmation, tape_factor)


# ── family taxonomy drift ────────────────────────────────────────────────────

def test_families_cover_base_weights_exactly():
    # Every weighted method in exactly one family; no stray names. A method
    # added to _BASE_WEIGHTS must be assigned a family (or this fails).
    assert set(FAMILY_OF) == set(agg._BASE_WEIGHTS)
    flat = [m for members in METHOD_FAMILIES.values() for m in members]
    assert len(flat) == len(set(flat))          # no method in two families


# ── family vote math ─────────────────────────────────────────────────────────

def test_family_vote_is_magnitude_weighted():
    # Within Sentiment: news +0.8, sent_velocity −0.2 → strong member dominates:
    # (0.8·0.8 − 0.2·0.2) / (0.8+0.2) = 0.60 → one bullish family vote.
    fa = compute_family_agreement({"news": 0.8, "sent_velocity": -0.2}, combined=0.3)
    assert fa.family_scores["Sentiment"] == pytest.approx(0.6)
    assert fa.agreeing == 1 and fa.opposing == 0


def test_same_family_pileon_is_one_vote():
    # Five technicals agreeing = ONE family vote; news+insider+put_call = THREE.
    tech_pile = {"tech": 0.5, "momentum": 0.5, "trend_strength": 0.5,
                 "pattern": 0.5, "vwap": 0.5}
    fa_pile = compute_family_agreement(tech_pile, combined=0.3)
    assert fa_pile.agreeing == 1

    cross = {"news": 0.5, "insider": 0.5, "put_call": 0.5}
    fa_cross = compute_family_agreement(cross, combined=0.3)
    assert fa_cross.agreeing == 3
    # And the factor rewards the cross-family case strictly more.
    assert fa_cross.factor(0.12) > fa_pile.factor(0.12)


def test_vote_threshold_filters_hairline_leans():
    # A 0.01 family lean is NOT a vote (the legacy sources_agreeing bug class).
    fa = compute_family_agreement({"news": 0.01}, combined=0.3, vote_threshold=0.05)
    assert fa.agreeing == 0 and fa.opposing == 0
    assert fa.factor(0.12) == 1.0               # no votes → neutral


def test_opposing_family_counts_and_drags_factor():
    scores = {"news": 0.6, "insider": 0.5, "put_call": -0.4}
    fa = compute_family_agreement(scores, combined=0.3)
    assert fa.agreeing == 2 and fa.opposing == 1
    # (2 − 1.5 − 1)/3 = −1/6 → factor below 1.0
    assert fa.factor(0.12) < 1.0


def test_factor_anchors():
    assert FamilyAgreement(agreeing=1, opposing=0).factor(0.12) == 1.0    # single family neutral
    assert FamilyAgreement(agreeing=2, opposing=0).factor(0.12) == pytest.approx(1.04)
    assert FamilyAgreement(agreeing=4, opposing=0).factor(0.12) == pytest.approx(1.12)
    assert FamilyAgreement(agreeing=0, opposing=0).factor(0.12) == 1.0    # no votes neutral


def test_neutral_combined_reports_scores_without_alignment():
    fa = compute_family_agreement({"news": 0.5, "insider": -0.5}, combined=0.0)
    assert fa.agreeing == 0 and fa.opposing == 0        # nothing to align against
    assert "Sentiment" in fa.detail and "Smart-Money" in fa.detail


# ── tape confirmation ────────────────────────────────────────────────────────

def _tape_df(trend: float, up_vol_heavy: bool, n: int = 30) -> pd.DataFrame:
    """Synthetic OHLCV: monotone drift `trend`/bar; volume concentrated on
    up (or down) days. Non-degenerate range by construction."""
    rows = []
    px = 100.0
    for i in range(n):
        prev = px
        px = px + trend
        up_day = px > prev
        vol = 2_000_000 if (up_day == up_vol_heavy) else 500_000
        hi, lo = max(prev, px) + 0.5, min(prev, px) - 0.5
        rows.append({"open": prev, "high": hi, "low": lo, "close": px, "volume": vol})
    idx = pd.date_range("2026-01-01", periods=n, freq="B")
    return pd.DataFrame(rows, index=idx)


def test_tape_bullish_structure():
    tc = compute_tape_confirmation("FAKE", df=_tape_df(trend=+1.0, up_vol_heavy=True))
    assert tc.label == "BULLISH_TAPE" and tc.score > 0.5
    assert "range" in tc.detail and "volume" in tc.detail


def test_tape_bearish_structure():
    tc = compute_tape_confirmation("FAKE", df=_tape_df(trend=-1.0, up_vol_heavy=False))
    assert tc.label == "BEARISH_TAPE" and tc.score < -0.5


def test_tape_no_data_paths():
    assert compute_tape_confirmation("ZZZQQQFAKE", df=None).label == "NO_DATA"  # cold cache
    short = _tape_df(+1.0, True, n=10)
    assert compute_tape_confirmation("FAKE", df=short).label == "NO_DATA"       # too few bars
    bad = _tape_df(+1.0, True)
    bad.iloc[-1, bad.columns.get_loc("close")] = float("nan")                   # NaN close guard
    assert compute_tape_confirmation("FAKE", df=bad).label == "NO_DATA"
    zero = _tape_df(+1.0, True)
    zero.iloc[-3, zero.columns.get_loc("close")] = 0.0                          # zero-close guard
    assert compute_tape_confirmation("FAKE", df=zero).label == "NO_DATA"


def test_tape_factor_confirm_and_diverge():
    from src.signals.agreement import TapeCheck
    bull = TapeCheck(score=0.8, label="BULLISH_TAPE")
    assert tape_factor(bull, combined=+0.4, span=0.08) == pytest.approx(1.064)  # confirms
    assert tape_factor(bull, combined=-0.4, span=0.08) == pytest.approx(0.936)  # diverges
    assert tape_factor(bull, combined=0.0, span=0.08) == 1.0                    # no direction
    assert tape_factor(TapeCheck(), combined=0.4, span=0.08) == 1.0             # NO_DATA
    assert tape_factor(None, combined=0.4, span=0.08) == 1.0


# ── build_signals wiring ─────────────────────────────────────────────────────

_OFF = [
    "enable_sentiment_velocity", "enable_technical_analysis", "enable_options_flow",
    "enable_sec_filings", "enable_insider_trades", "enable_put_call", "enable_gex",
    "enable_vwap", "enable_pattern_recognition", "enable_price_momentum",
    "enable_sector_relative_momentum", "enable_market_relative_momentum",
    "enable_money_flow", "enable_trend_strength", "enable_pead", "enable_iv_rank",
    "enable_iv_expr", "enable_cointegration", "enable_cross_sectional",
    "enable_adaptive_weights", "enable_ic_weights", "enable_winrate_method_filter",
    "enable_market_mode_switching", "enable_catalyst_timing",
    "enable_multi_timeframe_signals", "enable_extended_gap",
    "enable_trend_predictability_methods",
    "enable_high_52w", "enable_momentum_12_1", "enable_st_reversal",
    "enable_ttm_squeeze", "enable_iv_term_structure", "enable_anchored_vwap",
    "enable_residual_momentum", "enable_volume_profile",
]


def _setup_build(monkeypatch, massive=True, news_score=0.6, massive_score=0.8):
    for flag in _OFF:
        monkeypatch.setattr(settings, flag, False)
    monkeypatch.setattr(settings, "enable_news_sentiment", True)
    monkeypatch.setattr(settings, "enable_massive_tech", massive)
    monkeypatch.setattr(settings, "massive_tech_max_tickers", 0)
    monkeypatch.setattr(settings, "signal_scoring_max_workers", 2)
    monkeypatch.setattr(settings, "inverted_methods", "")
    monkeypatch.setattr(agg, "analyse_sentiment",
                        lambda t, a, force_engine=None: (news_score, "n"))
    if massive:
        import src.signals.massive_tech as mt
        monkeypatch.setattr(mt, "compute_massive_tech_score", lambda t: massive_score)


def test_build_signals_populates_family_fields(monkeypatch):
    # news (Sentiment) + massive (Price/Trend) both bullish → 2 families aligned.
    _setup_build(monkeypatch)
    monkeypatch.setattr(settings, "enable_tape_confirmation", False)  # deterministic
    sigs = agg.build_signals(["FAKEAAA"], [])
    s = sigs[0]
    assert s.families_agreeing == 2 and s.families_opposing == 0
    assert "Sentiment" in s.family_detail and "Price/Trend" in s.family_detail
    assert s.family_net_score > 0


def test_two_families_boost_confidence_vs_disabled(monkeypatch):
    _setup_build(monkeypatch)
    monkeypatch.setattr(settings, "enable_tape_confirmation", False)
    monkeypatch.setattr(settings, "enable_family_agreement", True)
    with_fam = agg.build_signals(["FAKEAAA"], [])[0].confidence
    monkeypatch.setattr(settings, "enable_family_agreement", False)
    without = agg.build_signals(["FAKEAAA"], [])[0].confidence
    # 2 aligned families → factor 1.04 → strictly ≥ (rounding can equalise at caps)
    assert with_fam >= without
    assert agg.build_signals(["FAKEAAA"], [])[0].families_agreeing == 0  # disabled → unset


def test_single_family_is_confidence_neutral(monkeypatch):
    # news alone (one family) — factor anchored at exactly 1.0, so enabling the
    # feature must not change confidence for single-source names.
    _setup_build(monkeypatch, massive=False)
    monkeypatch.setattr(settings, "enable_tape_confirmation", False)
    monkeypatch.setattr(settings, "enable_family_agreement", True)
    a = agg.build_signals(["FAKEAAA"], [])[0].confidence
    monkeypatch.setattr(settings, "enable_family_agreement", False)
    b = agg.build_signals(["FAKEAAA"], [])[0].confidence
    assert a == b


def test_winrate_filtered_method_excluded_from_family_votes(monkeypatch):
    # massive filtered by the win-rate filter → its Price/Trend family vote
    # disappears; only Sentiment (news) remains.
    import src.performance.tracker as tracker
    _setup_build(monkeypatch)
    monkeypatch.setattr(settings, "enable_tape_confirmation", False)
    monkeypatch.setattr(settings, "enable_winrate_method_filter", True)
    monkeypatch.setattr(settings, "winrate_filter_threshold", 0.50)
    monkeypatch.setattr(settings, "winrate_filter_min_trades", 10)
    monkeypatch.setattr(settings, "enable_oos_validation", False)
    monkeypatch.setattr(tracker, "compute_solo_method_gross_winrate",
                        lambda split=None, **kw: {"massive": {"trades": 20, "win_rate": 30.0}})
    agg.reset_winrate_filter_cache()
    s = agg.build_signals(["FAKEAAA"], [])[0]
    assert s.families_agreeing == 1
    assert "Price/Trend" not in s.family_detail


def test_inverted_method_votes_with_corrected_sign(monkeypatch):
    # massive raw +0.8 but INVERTED → its family votes BEARISH (the corrected
    # sign the combine consumes); news bullish → 1 aligned vs 1 opposed... the
    # combined direction depends on weights; just assert the Price/Trend family
    # score flipped negative in the detail.
    _setup_build(monkeypatch)
    monkeypatch.setattr(settings, "enable_tape_confirmation", False)
    monkeypatch.setattr(settings, "inverted_methods", "massive")
    s = agg.build_signals(["FAKEAAA"], [])[0]
    assert "Price/Trend:-" in s.family_detail


def test_tape_factor_moves_confidence(monkeypatch):
    # Same signal, tape bullish vs bearish → confidence differs in the tape's
    # favour (bullish combined here). Small scores keep confidence below the
    # 1.0 cap so the factor is visible.
    import src.signals.aggregator as A
    from src.signals.agreement import TapeCheck
    _setup_build(monkeypatch, news_score=0.2, massive_score=0.2)
    monkeypatch.setattr(settings, "enable_tape_confirmation", True)
    monkeypatch.setattr(A, "compute_tape_confirmation",
                        lambda t, df=None: TapeCheck(score=0.9, label="BULLISH_TAPE", detail="d"))
    conf_bull = agg.build_signals(["FAKEAAA"], [])[0]
    monkeypatch.setattr(A, "compute_tape_confirmation",
                        lambda t, df=None: TapeCheck(score=-0.9, label="BEARISH_TAPE", detail="d"))
    conf_bear = agg.build_signals(["FAKEAAA"], [])[0]
    assert conf_bull.confidence > conf_bear.confidence
    assert conf_bull.tape_confirmation_label == "BULLISH_TAPE"
    assert conf_bear.tape_confirmation_score == -0.9


# ── synthesis prompt exposure ────────────────────────────────────────────────

def _capture_prompt(monkeypatch, sig, blind=False):
    import src.analysis.claude_analyst as ca
    monkeypatch.setattr(settings, "llm_ab_synthesis_models", "deepseek-v4-flash")
    captured = {}

    def fake(prompt, **kwargs):
        captured["prompt"] = prompt
        return json.dumps([{"ticker": sig.ticker, "type": "STOCK", "direction": "BULLISH",
                            "action": "WATCH", "confidence": 0.5, "rationale": "x"}])

    monkeypatch.setattr(ca, "_call_claude_analyst", fake)
    monkeypatch.setattr(ca, "_call_deepseek_analyst", fake)
    monkeypatch.setattr(ca, "_call_qwen_analyst", fake)
    # blind_synthesis is a PARAMETER (the pipeline flips the coin), not a setting.
    ca.generate_recommendations([sig], blind_synthesis=blind)
    return captured["prompt"]


def _prompt_signal():
    from src.models import TickerSignal
    return TickerSignal(
        ticker="AAA", direction="BULLISH", confidence=0.5,
        sentiment_score=0.4, technical_score=0.0, insider_score=0.0,
        sources_agreeing=2, rationale="r", price=50.0,
        families_agreeing=3, families_opposing=1,
        family_detail="Sentiment:+0.42|Options:+0.18|Smart-Money:-0.12",
        family_net_score=0.3,
        tape_confirmation_score=0.55, tape_confirmation_label="BULLISH_TAPE",
        tape_confirmation_detail="close at 82% of 20d range",
    )


def test_prompt_shows_family_agreement_and_tape(monkeypatch):
    p = _capture_prompt(monkeypatch, _prompt_signal(), blind=False)
    assert "FAMILY AGREEMENT: 3 independent families aligned" in p
    assert "Sentiment:+0.42|Options:+0.18|Smart-Money:-0.12" in p
    assert "TAPE STRUCTURE=+0.55 [BULLISH_TAPE]" in p
    assert "AGREEMENT QUALITY" in p                       # instruction block present
    assert "ONE independent confirmation" in p


def test_blind_prompt_gets_neutral_rollup_only(monkeypatch):
    p = _capture_prompt(monkeypatch, _prompt_signal(), blind=True)
    assert "FAMILY ROLLUP" in p                           # neutral rollup shown
    assert "aligned with the read" not in p               # no alignment framing (verdict leak)
    assert "TAPE STRUCTURE=+0.55" in p                    # raw-data line stays


def test_prompt_omits_lines_when_absent(monkeypatch):
    from src.models import TickerSignal
    bare = TickerSignal(
        ticker="AAA", direction="BULLISH", confidence=0.5,
        sentiment_score=0.4, technical_score=0.0, insider_score=0.0,
        sources_agreeing=1, rationale="r", price=50.0,
        tape_confirmation_label="NO_DATA",
    )
    p = _capture_prompt(monkeypatch, bare, blind=False)
    # Anchor on the per-ticker LINE formats ("FAMILY AGREEMENT: <n> …" /
    # "TAPE STRUCTURE=…") — the method names alone also appear in the standing
    # instruction text, which is present regardless of per-ticker data.
    assert "FAMILY AGREEMENT:" not in p
    assert "TAPE STRUCTURE=" not in p
