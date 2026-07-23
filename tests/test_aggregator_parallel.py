"""Parallelised per-ticker scoring (build_signals thread pool) must be IDENTICAL
to the sequential path — same order, same scores — and the per-run caps that used
to be sequential counters (massive_tech_max_tickers / intraday_30m_max_tickers)
must still pick the first-N tickers deterministically regardless of concurrency."""

import pytest

from config.settings import settings
import src.signals.aggregator as agg

# Everything data-heavy / network-touching OFF; news stays on (mocked) so each
# ticker gets a distinct, deterministic combined score and order is observable.
_OFF = [
    "enable_sentiment_velocity", "enable_technical_analysis", "enable_options_flow",
    "enable_sec_filings", "enable_insider_trades", "enable_put_call", "enable_gex",
    "enable_vwap", "enable_pattern_recognition", "enable_price_momentum",
    "enable_sector_relative_momentum", "enable_market_relative_momentum",
    "enable_money_flow", "enable_trend_strength", "enable_pead", "enable_iv_rank",
    "enable_iv_expr", "enable_cointegration", "enable_cross_sectional",
    "enable_adaptive_weights", "enable_market_mode_switching", "enable_catalyst_timing",
    "enable_multi_timeframe_signals", "enable_extended_gap",
    "enable_trend_predictability_methods",
    "enable_high_52w", "enable_momentum_12_1", "enable_st_reversal",
    "enable_ttm_squeeze", "enable_iv_term_structure", "enable_anchored_vwap",
    "enable_residual_momentum", "enable_volume_profile",
]


def _fake_sentiment(ticker, articles, force_engine=None):
    # Deterministic per-ticker score in [-1, 1] (stable across processes — no hash()).
    return round(((sum(ord(c) for c in ticker) % 200) - 100) / 100.0, 3), f"news {ticker}"


def _setup(monkeypatch, massive=False):
    for flag in _OFF:
        monkeypatch.setattr(settings, flag, False)
    monkeypatch.setattr(settings, "enable_news_sentiment", True)
    monkeypatch.setattr(settings, "enable_massive_tech", massive)
    monkeypatch.setattr(agg, "analyse_sentiment", _fake_sentiment)


def test_parallel_scoring_identical_to_sequential(monkeypatch):
    _setup(monkeypatch)
    tickers = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "XLE",
               "XLF", "GLD", "SPY", "QQQ", "INTC", "MU"]

    monkeypatch.setattr(settings, "signal_scoring_max_workers", 1)
    seq = agg.build_signals(list(tickers), [])

    monkeypatch.setattr(settings, "signal_scoring_max_workers", 8)
    par = agg.build_signals(list(tickers), [])

    # Order is preserved (ex.map keeps input order) ...
    assert [s.ticker for s in seq] == tickers
    assert [s.ticker for s in par] == tickers
    # ... and every field of every signal is byte-identical.
    assert [s.model_dump() for s in seq] == [s.model_dump() for s in par]
    # Sanity: the scores are actually non-trivial / distinct (test would be vacuous otherwise).
    assert len({s.combined_score for s in par}) > 1


def test_parallel_massive_cap_picks_first_n_deterministically(monkeypatch):
    _setup(monkeypatch, massive=True)
    monkeypatch.setattr(settings, "massive_tech_max_tickers", 2)
    import src.signals.massive_tech as mt
    monkeypatch.setattr(mt, "compute_massive_tech_score", lambda t: 0.42)

    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    monkeypatch.setattr(settings, "signal_scoring_max_workers", 8)
    sigs = agg.build_signals(list(tickers), [])

    # Only the first 2 tickers (in input order) get a Massive score, even though the
    # work ran concurrently — the cap is pre-selected, not a racy counter.
    scored = [s.ticker for s in sigs if s.massive_score != 0.0]
    assert scored == ["AAA", "BBB"]


def test_massive_is_weighted_into_combined_score(monkeypatch):
    # massive_tech is now a WEIGHTED member of combined_score (promoted), not a
    # diagnostic. With news neutral (0), the SIGN of combined_score must track massive.
    _setup(monkeypatch, massive=True)
    monkeypatch.setattr(agg, "analyse_sentiment", lambda t, a, force_engine=None: (0.0, "neutral"))
    monkeypatch.setattr(settings, "massive_tech_max_tickers", 0)   # uncapped (the new default)
    import src.signals.massive_tech as mt
    tickers = ["AAA", "BBB"]
    monkeypatch.setattr(settings, "signal_scoring_max_workers", 4)

    monkeypatch.setattr(mt, "compute_massive_tech_score", lambda t: 0.8)
    pos = agg.build_signals(list(tickers), [])
    monkeypatch.setattr(mt, "compute_massive_tech_score", lambda t: -0.8)
    neg = agg.build_signals(list(tickers), [])

    assert all(s.massive_score == 0.8 for s in pos)         # every ticker scored (uncapped)
    assert all(s.combined_score > 0 for s in pos)           # +massive pushes combined positive
    assert all(s.combined_score < 0 for s in neg)           # −massive pushes it negative


def test_inverted_methods_parsing(monkeypatch):
    from src.signals.aggregator import _inverted_methods
    monkeypatch.setattr(settings, "inverted_methods", "")
    assert _inverted_methods() == frozenset()
    monkeypatch.setattr(settings, "inverted_methods", " Insider ,MASSIVE, tech ")
    assert _inverted_methods() == frozenset({"insider", "massive", "tech"})


def test_inversion_flips_combined_score_but_keeps_panel_raw(monkeypatch):
    """A method in inverted_methods contributes with a FLIPPED sign in
    combined_score, while the signals panel keeps its RAW score (so the inversion
    stays re-validatable). Exercised via `massive` (in _BASE_WEIGHTS, easy to inject)."""
    _setup(monkeypatch, massive=True)
    monkeypatch.setattr(agg, "analyse_sentiment", lambda t, a, force_engine=None: (0.0, "neutral"))
    monkeypatch.setattr(settings, "enable_ic_weights", False)
    monkeypatch.setattr(settings, "massive_tech_max_tickers", 0)
    monkeypatch.setattr(settings, "signal_scoring_max_workers", 4)
    import src.signals.massive_tech as mt
    monkeypatch.setattr(mt, "compute_massive_tech_score", lambda t: 0.8)
    tickers = ["AAA", "BBB"]

    monkeypatch.setattr(settings, "inverted_methods", "")
    base = agg.build_signals(list(tickers), [])
    assert base and all(s.combined_score > 0 for s in base)       # +massive → positive

    monkeypatch.setattr(settings, "inverted_methods", "massive")
    inv = agg.build_signals(list(tickers), [])
    assert inv and all(s.combined_score < 0 for s in inv)         # inverted → NEGATIVE
    assert all(s.massive_score == 0.8 for s in inv)               # panel keeps the RAW score


def test_coherence_and_sources_agreeing_use_effective_sign(monkeypatch):
    """coherence_factor / sources_agreeing must judge agreement on each method's
    EFFECTIVE (inversion-corrected) sign, not its RAW sign — unified 2026-07-20
    with family_factor's pre-existing convention. An inverted method's raw score
    is typically the OPPOSITE of what it actually contributes to combined_score,
    so reading raw misclassifies it as "disagreeing" even when it's the very
    method driving combined_score's direction (exercised via `massive`, the only
    active weighted method here so the agreement math is unambiguous)."""
    _setup(monkeypatch, massive=True)
    monkeypatch.setattr(agg, "analyse_sentiment", lambda t, a, force_engine=None: (0.0, "neutral"))
    monkeypatch.setattr(settings, "enable_ic_weights", False)
    monkeypatch.setattr(settings, "massive_tech_max_tickers", 0)
    monkeypatch.setattr(settings, "signal_scoring_max_workers", 4)
    monkeypatch.setattr(settings, "inverted_methods", "massive")
    import src.signals.massive_tech as mt
    monkeypatch.setattr(mt, "compute_massive_tech_score", lambda t: 0.8)   # raw bullish

    sigs = agg.build_signals(["AAA"], [])
    s = sigs[0]
    assert s.combined_score < 0            # inverted massive is the sole driver → BEARISH
    assert s.massive_score == 0.8          # panel still keeps the RAW (bullish) score
    # massive's EFFECTIVE score (-0.8) agrees with the bearish combined → counted as
    # agreeing. Pre-unification this read massive's RAW (+0.8) against a bearish
    # combined → opposite signs → wrongly "disagreeing" (coherence_ratio 0.0, the
    # 0.45x confidence-penalty floor) despite massive being the only contributor.
    assert s.sources_agreeing == 1
    from src.signals.aggregator import _coherence_factor
    # Rebuild coherence directly to assert the exact ratio (not just >0): full
    # agreement from the sole active method → ratio 1.0, not the 0.0 a raw-sign
    # read would have produced.
    ratio, factor = _coherence_factor(s.combined_score, [(True, -0.8)])
    assert ratio == 1.0 and factor == 1.35


def test_confidence_components_reproduce_stored_confidence(monkeypatch):
    """The 5 persisted confidence-factor fields (+ raw_confidence) on
    TickerSignal must multiply out to EXACTLY the stored `confidence` — the
    invariant src/analysis/confidence_components.py depends on. If the formula
    in aggregator._score_ticker ever changes without updating the corresponding
    TickerSignal(...) field assignments, this catches the drift immediately
    (the component-isolation dashboard tables would otherwise silently analyse
    a formula that no longer matches production)."""
    _setup(monkeypatch, massive=True)
    monkeypatch.setattr(settings, "enable_ic_weights", False)
    monkeypatch.setattr(settings, "massive_tech_max_tickers", 0)
    monkeypatch.setattr(settings, "signal_scoring_max_workers", 4)
    import src.signals.massive_tech as mt
    monkeypatch.setattr(mt, "compute_massive_tech_score", lambda t: 0.6)

    sigs = agg.build_signals(["AAA", "BBB", "CCC"], [])
    assert sigs
    for s in sigs:
        product = (s.raw_confidence * s.coherence_factor * s.movement_factor
                  * s.volume_factor * s.family_conf_factor * s.tape_conf_factor)
        assert s.confidence == pytest.approx(round(min(1.0, product), 2), abs=1e-9)
    # Sanity: the factors aren't all trivially 1.0 / the test isn't vacuous.
    assert any(s.raw_confidence > 0 for s in sigs)


def test_market_momentum_weighted_into_combined_score(monkeypatch):
    # market_momentum promoted from diagnostic → weighted (light). News neutral, so
    # the sign of combined_score must track market_momentum. Pin inverted_methods
    # empty so this tests the RAW weighting mechanism independent of the ops
    # inversion setting (which now inverts market_momentum by default — its
    # sign-flip behaviour is covered by test_inversion_flips_combined_score_...).
    _setup(monkeypatch)
    monkeypatch.setattr(agg, "analyse_sentiment", lambda t, a, force_engine=None: (0.0, "neutral"))
    monkeypatch.setattr(settings, "inverted_methods", "")
    monkeypatch.setattr(settings, "enable_market_relative_momentum", True)
    tickers = ["AAA", "BBB"]
    monkeypatch.setattr(settings, "signal_scoring_max_workers", 4)

    monkeypatch.setattr(agg, "compute_market_relative_momentum_score",
                        lambda t: (0.8, 0.0, 0.0, ""))
    pos = agg.build_signals(list(tickers), [])
    monkeypatch.setattr(agg, "compute_market_relative_momentum_score",
                        lambda t: (-0.8, 0.0, 0.0, ""))
    neg = agg.build_signals(list(tickers), [])

    assert all(s.market_momentum_score == 0.8 for s in pos)
    assert all(s.combined_score > 0 for s in pos)
    assert all(s.combined_score < 0 for s in neg)


def test_fundamentals_overlay_nudges_combined_score(monkeypatch):
    # The 4 Massive fundamental factors fold into combined_score as an additive overlay.
    _setup(monkeypatch)
    monkeypatch.setattr(agg, "analyse_sentiment", lambda t, a, force_engine=None: (0.0, "neutral"))
    monkeypatch.setattr(settings, "fundamental_factor_weight", 0.5)
    ff = {"AAA": {"f_value": 0.4, "f_quality": 0.4, "f_growth": 0.0, "f_short_squeeze": 0.0}}
    monkeypatch.setattr(settings, "signal_scoring_max_workers", 4)

    sigs = agg.build_signals(["AAA", "BBB"], [], fundamental_factors=ff)
    by = {s.ticker: s for s in sigs}
    assert by["AAA"].combined_score > by["BBB"].combined_score   # only AAA has factors
    assert by["AAA"].combined_score > 0
    assert by["BBB"].combined_score == 0.0                       # no factors, news neutral


def test_single_worker_is_the_sequential_path(monkeypatch):
    # workers <= 1 must still produce correct output (the legacy escape hatch).
    _setup(monkeypatch)
    monkeypatch.setattr(settings, "signal_scoring_max_workers", 1)
    sigs = agg.build_signals(["AAPL", "MSFT", "NVDA"], [])
    assert [s.ticker for s in sigs] == ["AAPL", "MSFT", "NVDA"]
