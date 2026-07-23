"""Win-rate method filter (2026-07-19): a method whose GROSS solo win rate (pure
directional hit BEFORE fees/spread) is confidently sub-50% (≥ winrate_filter_min_trades
attributed trades AND win_rate < winrate_filter_threshold) is DROPPED from
combined_score + coherence + sources_agreeing (aggregator) and from the synthesis
prompt — yet is still SCORED and persisted to the signals panel, so simulation
monitoring is unaffected. Inverted methods are exempt.

Covers the selection logic (aggregator.winrate_filtered_methods, fed by
tracker.compute_solo_method_gross_winrate) and its wiring into build_signals (weight
→ 0, renormalised survivors, raw score still on the signal).
"""

import pytest

from config.settings import settings
import src.signals.aggregator as agg
import src.performance.tracker as tracker


# ── selection logic ──────────────────────────────────────────────────────────

def _perf(**methods):
    """methods: name -> (n_trades, win_rate_pct) → the gross-winrate dict shape."""
    return {m: {"trades": n, "win_rate": wr} for m, (n, wr) in methods.items()}


def _patch_perf(monkeypatch, perf):
    monkeypatch.setattr(tracker, "compute_solo_method_gross_winrate",
                        lambda split=None, **kw: perf)
    agg.reset_winrate_filter_cache()


def _base(monkeypatch):
    monkeypatch.setattr(settings, "enable_winrate_method_filter", True)
    monkeypatch.setattr(settings, "winrate_filter_threshold", 0.50)
    monkeypatch.setattr(settings, "winrate_filter_min_trades", 10)
    monkeypatch.setattr(settings, "inverted_methods", "")
    monkeypatch.setattr(settings, "enable_oos_validation", False)
    agg.reset_winrate_filter_cache()


def test_drops_confident_sub50_method(monkeypatch):
    _base(monkeypatch)
    _patch_perf(monkeypatch, _perf(news=(20, 40.0), momentum=(20, 60.0)))
    assert agg.winrate_filtered_methods() == frozenset({"news"})


def test_keeps_method_below_min_trades(monkeypatch):
    _base(monkeypatch)
    # 30% WR but only 5 trades → not enough evidence to judge → kept.
    _patch_perf(monkeypatch, _perf(news=(5, 30.0)))
    assert agg.winrate_filtered_methods() == frozenset()


def test_exactly_50_is_kept(monkeypatch):
    _base(monkeypatch)
    _patch_perf(monkeypatch, _perf(news=(20, 50.0)))     # not strictly < 50 → kept
    assert agg.winrate_filtered_methods() == frozenset()


def test_inverted_method_is_exempt(monkeypatch):
    _base(monkeypatch)
    monkeypatch.setattr(settings, "inverted_methods", "insider")
    _patch_perf(monkeypatch, _perf(insider=(20, 25.0), news=(20, 40.0)))
    # insider is anti-predictive RAW but INVERTED (sign already corrected) → exempt;
    # news is genuinely sub-50% and not inverted → dropped.
    assert agg.winrate_filtered_methods() == frozenset({"news"})


def test_flag_off_is_noop(monkeypatch):
    _base(monkeypatch)
    monkeypatch.setattr(settings, "enable_winrate_method_filter", False)
    _patch_perf(monkeypatch, _perf(news=(20, 10.0)))
    assert agg.winrate_filtered_methods() == frozenset()


def test_non_base_methods_never_filtered(monkeypatch):
    # A panel-first / non-_BASE_WEIGHTS name in the perf dict is ignored by the filter.
    _base(monkeypatch)
    _patch_perf(monkeypatch, _perf(hi52=(50, 5.0), cross_sectional=(50, 5.0)))
    assert agg.winrate_filtered_methods() == frozenset()


def test_gross_winrate_is_directional_and_cost_free(monkeypatch):
    # The gross win rate counts a right-DIRECTION move as a win even when it's tiny
    # (smaller than any round-trip cost), and skips no-view (score 0) trades.
    trades = [
        # long view, stock rose only +0.1% → gross WIN (a net/cost-adjusted metric
        # would score this a LOSS once spread+commission are charged)
        {"status": "CLOSED", "action": "BUY", "type": "STOCK",
         "entry_price": 100.0, "exit_price": 100.1, "method_scores": {"tech": 0.5}},
        # long view, stock fell → gross LOSS
        {"status": "CLOSED", "action": "BUY", "type": "STOCK",
         "entry_price": 100.0, "exit_price": 99.0, "method_scores": {"tech": 0.5}},
        # short view (score < 0), stock fell → gross WIN
        {"status": "CLOSED", "action": "SELL", "type": "STOCK",
         "entry_price": 100.0, "exit_price": 98.0, "method_scores": {"tech": -0.5}},
        # no view (score 0) → skipped entirely
        {"status": "CLOSED", "action": "BUY", "type": "STOCK",
         "entry_price": 100.0, "exit_price": 90.0, "method_scores": {"tech": 0.0}},
    ]
    monkeypatch.setattr(tracker, "_load_trades", lambda: trades)
    out = tracker.compute_solo_method_gross_winrate(split=None)
    assert out["tech"]["trades"] == 3                    # the score-0 trade skipped
    assert out["tech"]["win_rate"] == round(2 / 3 * 100, 1)   # 2 of 3 directions right


def test_perf_unavailable_fails_soft(monkeypatch):
    _base(monkeypatch)

    def _boom(split=None, **kw):
        raise RuntimeError("no db")

    monkeypatch.setattr(tracker, "compute_solo_method_gross_winrate", _boom)
    agg.reset_winrate_filter_cache()
    assert agg.winrate_filtered_methods() == frozenset()


# ── wiring into build_signals ────────────────────────────────────────────────

# Everything data-heavy / network-touching OFF; news + massive stay on so the
# combine has exactly two weighted members we control.
_OFF = [
    "enable_sentiment_velocity", "enable_technical_analysis", "enable_options_flow",
    "enable_sec_filings", "enable_insider_trades", "enable_put_call", "enable_gex",
    "enable_vwap", "enable_pattern_recognition", "enable_price_momentum",
    "enable_sector_relative_momentum", "enable_market_relative_momentum",
    "enable_money_flow", "enable_trend_strength", "enable_pead", "enable_iv_rank",
    "enable_iv_expr", "enable_cointegration", "enable_cross_sectional",
    "enable_adaptive_weights", "enable_ic_weights", "enable_market_mode_switching",
    "enable_catalyst_timing", "enable_multi_timeframe_signals", "enable_extended_gap",
    "enable_trend_predictability_methods",
    "enable_high_52w", "enable_momentum_12_1", "enable_st_reversal",
    "enable_ttm_squeeze", "enable_iv_term_structure", "enable_anchored_vwap",
    "enable_residual_momentum", "enable_volume_profile",
]


def _setup_build(monkeypatch, massive_score):
    for flag in _OFF:
        monkeypatch.setattr(settings, flag, False)
    monkeypatch.setattr(settings, "enable_news_sentiment", True)
    monkeypatch.setattr(settings, "enable_massive_tech", True)
    monkeypatch.setattr(settings, "massive_tech_max_tickers", 0)   # uncapped → all scored
    monkeypatch.setattr(settings, "signal_scoring_max_workers", 4)
    monkeypatch.setattr(settings, "inverted_methods", "")
    monkeypatch.setattr(agg, "analyse_sentiment",
                        lambda t, a, force_engine=None: (0.0, "neutral"))
    import src.signals.massive_tech as mt
    monkeypatch.setattr(mt, "compute_massive_tech_score", lambda t: massive_score)


def test_filter_removes_method_from_combine_but_keeps_panel_score(monkeypatch):
    _base(monkeypatch)
    _setup_build(monkeypatch, massive_score=0.8)

    # Baseline — massive is a healthy (>50%) method → +massive drives combined > 0
    # and it counts toward sources_agreeing.
    _patch_perf(monkeypatch, _perf(massive=(20, 60.0)))
    base = agg.build_signals(["AAA", "BBB"], [])
    assert base and all(s.combined_score > 0 for s in base)
    assert all(s.massive_score == 0.8 for s in base)
    assert all(s.sources_agreeing >= 1 for s in base)

    # Now massive is confidently sub-50% → DROPPED from the weighted combine +
    # coherence + agreement. News is neutral (0) and the only surviving weighted
    # method, so combined collapses to 0 and nothing agrees — but the RAW massive
    # score is STILL computed and on the signal (panel monitoring intact).
    _patch_perf(monkeypatch, _perf(massive=(20, 30.0)))
    filt = agg.build_signals(["AAA", "BBB"], [])
    assert filt and all(s.combined_score == 0.0 for s in filt)
    assert all(s.sources_agreeing == 0 for s in filt)
    assert all(s.massive_score == 0.8 for s in filt)   # STILL scored + persisted


def test_filtered_method_line_hidden_from_synthesis_prompt(monkeypatch):
    # The synthesis prompt must HIDE a filtered method's per-ticker score line while
    # keeping an unfiltered one — so the LLM never leans on a sub-coin-flip method.
    import json
    import src.analysis.claude_analyst as ca
    from src.models import TickerSignal

    _base(monkeypatch)
    # vwap is confidently sub-50% → filtered; momentum is healthy → kept.
    _patch_perf(monkeypatch, _perf(vwap=(20, 30.0), momentum=(20, 60.0)))
    # Pin the synthesis pool to a single mocked engine so the run is deterministic
    # and never touches a real API (the .env pool includes a Qwen arm).
    monkeypatch.setattr(settings, "llm_ab_synthesis_models", "deepseek-v4-flash")

    captured = {}

    def fake(prompt, **kwargs):   # tolerate model= / thinking= kwargs from the bake-off
        captured["prompt"] = prompt
        return json.dumps([{"ticker": "AAA", "type": "STOCK", "direction": "BULLISH",
                            "action": "WATCH", "confidence": 0.5, "rationale": "x"}])

    monkeypatch.setattr(ca, "_call_claude_analyst", fake)
    monkeypatch.setattr(ca, "_call_deepseek_analyst", fake)
    monkeypatch.setattr(ca, "_call_qwen_analyst", fake)

    sig = TickerSignal(
        ticker="AAA", direction="BULLISH", confidence=0.5,
        sentiment_score=0.0, technical_score=0.0, insider_score=0.0,
        sources_agreeing=1, rationale="r", price=50.0,
        vwap_score=0.5, vwap_distance_pct=-2.0,        # would print a VWAP_score line…
        momentum_score=0.5, momentum_1m_pct=3.0,        # …and a Momentum_score line
    )
    ca.generate_recommendations([sig])
    p = captured["prompt"]
    # Anchor on the per-ticker VALUE line ("<Method>_score=+0.50"), not the method
    # name alone — the method name also appears in the prompt's methodology preamble.
    assert "Momentum_score=+0.50" in p     # unfiltered method's per-ticker line present
    assert "VWAP_score=+0.50" not in p     # filtered method's per-ticker line hidden


def test_filter_suppressed_when_it_would_empty_the_book(monkeypatch):
    # If EVERY active weighted method is sub-50%, the filter must be suppressed for
    # that run (never zero the whole book → all-NEUTRAL). Both active members
    # (news + massive) are sub-50% here; news is neutral (0) and massive = +0.8, so a
    # SUPPRESSED filter leaves massive driving combined > 0. Had the filter NOT been
    # suppressed, both weights would be 0 and combined would collapse to 0.
    _base(monkeypatch)
    _setup_build(monkeypatch, massive_score=0.8)
    _patch_perf(monkeypatch, _perf(news=(20, 30.0), massive=(20, 30.0)))
    sigs = agg.build_signals(["AAA", "BBB"], [])
    assert sigs and all(s.combined_score > 0 for s in sigs)
