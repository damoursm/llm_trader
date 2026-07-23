"""Guards for the two evaluation-path data bugs found 2026-07-22.

1. SPLIT CONTAMINATION — `_build_pseudo_trades` anchored each call at the LIVE
   snapshot price recorded with the recommendation, then marked it against the
   OHLCV cache, which is adjusted RETROACTIVELY for splits. Mixing bases invents
   a return of exactly the split ratio (observed: SNDQ −1,239% in one day).

2. ATTRIBUTION CONTAMINATION — `recommendations.llm_provider` is one per-run
   value written to every row, but the synthesis model is only asked about the
   ~40 highest-confidence tickers; the rest are rule-based back-fills that
   inherited the model's name (~22% of all rows).
"""

import pandas as pd
import pytest

from src.performance import tracker
from src.performance.tracker import (
    RULE_FILL_MODEL, _is_rule_based_fill, _build_pseudo_trades,
)


# ── 1. split guard ───────────────────────────────────────────────────────────

def _bars(prices):
    idx = pd.to_datetime([d for d, _ in prices])
    return pd.DataFrame({"Close": [p for _, p in prices]}, index=idx)


@pytest.fixture
def _cache(monkeypatch):
    """Patch the OHLCV loader used inside _build_pseudo_trades."""
    store = {}

    def fake_load(ticker, *a, **k):
        return store.get(ticker)

    monkeypatch.setattr("src.data.cache.load_ohlcv", fake_load)
    return store


def _call(snap, day="2026-07-06", ticker="SOXS", action="SELL"):
    return {
        "ticker": ticker, "type": "STOCK", "action": action,
        "entry_date": day, "entry_datetime": f"{day}T12:00:00-04:00",
        "snap_price": snap,
    }


def test_split_contaminated_snapshot_is_discarded(_cache):
    """A snapshot on the PRE-split scale (1:10 reverse split → cache is 10x)
    must not be used as the entry anchor: it would fabricate a ~-900% return."""
    _cache["SOXS"] = _bars([("2026-07-03", 40.0), ("2026-07-06", 41.7),
                            ("2026-07-07", 48.0)])
    out = _build_pseudo_trades([_call(snap=4.17)])          # 4.17 vs cached 41.70
    assert len(out) == 1
    # Anchored on the cached PRIOR close (same basis as the mark), not 4.17.
    assert out[0]["entry_price"] == pytest.approx(40.0)
    # Sanity: the return is now a plausible single-digit magnitude, not -900%.
    assert abs(out[0]["return_pct"]) < 50


def test_forward_split_snapshot_also_discarded(_cache):
    """The mirror case: a forward split makes the cache a FRACTION of the
    recorded snapshot (ratio ~0.05). Also a mixed basis."""
    _cache["MUU"] = _bars([("2026-07-03", 0.95), ("2026-07-06", 1.0),
                           ("2026-07-07", 1.1)])
    out = _build_pseudo_trades([_call(snap=21.0, ticker="MUU", action="BUY")])
    assert len(out) == 1
    assert out[0]["entry_price"] == pytest.approx(0.95)


def test_normal_snapshot_is_still_preferred(_cache):
    """The guard must NOT disturb the common case — an intraday snapshot that
    agrees with its own day's close stays the anchor (it is the decision price,
    and using the prior close instead would credit pre-decision drift)."""
    _cache["AAPL"] = _bars([("2026-07-03", 100.0), ("2026-07-06", 104.0),
                            ("2026-07-07", 106.0)])
    out = _build_pseudo_trades([_call(snap=103.0, ticker="AAPL", action="BUY")])
    assert len(out) == 1
    assert out[0]["entry_price"] == pytest.approx(103.0)


def test_large_but_plausible_intraday_move_is_kept(_cache):
    """A real 40% intraday move is inside the band and must survive — the guard
    is for split-scale (>=2x) disagreement, not for volatile microcaps."""
    _cache["ZYBT"] = _bars([("2026-07-03", 1.00), ("2026-07-06", 1.40),
                            ("2026-07-07", 1.50)])
    out = _build_pseudo_trades([_call(snap=1.00, ticker="ZYBT", action="BUY")])
    assert len(out) == 1
    assert out[0]["entry_price"] == pytest.approx(1.00)


def test_guard_is_noop_when_no_same_day_bar(_cache):
    """With no cached bar for the call's own day there is nothing to validate
    against; the snapshot is trusted (unchanged legacy behaviour)."""
    _cache["XYZ"] = _bars([("2026-07-03", 10.0), ("2026-07-07", 11.0)])
    out = _build_pseudo_trades([_call(snap=10.5, ticker="XYZ", action="BUY")])
    assert len(out) == 1
    assert out[0]["entry_price"] == pytest.approx(10.5)


# ── 2. rule-based fill attribution ───────────────────────────────────────────

def test_new_rows_flagged_by_explicit_provider():
    assert _is_rule_based_fill(RULE_FILL_MODEL, "anything at all") is True


def test_legacy_rows_flagged_by_aggregator_rationale():
    # The aggregator assembles a signal rationale from these fragments; a fill
    # reuses it verbatim.
    assert _is_rule_based_fill("deepseek-v4-flash",
                               "No recent news articles found. | Technical score: +0.59") is True
    assert _is_rule_based_fill("deepseek-v4-pro-thinking",
                               "Some news. | Put/call signal: +0.70") is True
    assert _is_rule_based_fill("deepseek-v4-flash", "No rationale available.") is True


def test_genuine_llm_rationale_not_flagged():
    """Authored prose — including a model echoing the prompt's 'Technical
    score=' spelling (equals sign, not colon) — must NOT be treated as a fill."""
    assert _is_rule_based_fill(
        "deepseek-v4-flash",
        "Truist initiated coverage with a Buy rating and $98 target, a hard "
        "catalyst 4h ago that typically drives short-term repricing.") is False
    assert _is_rule_based_fill("deepseek-v4-flash",
                               "Momentum is fading; Technical score=+0.21 in the prompt.") is False


def test_fill_rationale_with_sentiment_prose_is_still_a_fill():
    """The NEWS half of a signal rationale is written by the SENTIMENT model, so
    a fill often reads authored. It is still not the SYNTHESIS model's call —
    this is the case that made the bug hard to see."""
    rationale = ("Fed rate decision imminent with Kevin Warsh taking over, creating "
                 "near-term uncertainty for tech. | Technical score: +0.12")
    assert _is_rule_based_fill("deepseek-v4-flash-thinking", rationale) is True


def test_empty_and_none_are_not_fills():
    assert _is_rule_based_fill("deepseek-v4-flash", None) is False
    assert _is_rule_based_fill(None, None) is False


def test_recommendation_model_defaults_to_not_filled():
    from datetime import datetime, timezone
    from src.models import Recommendation
    r = Recommendation(ticker="AAPL", direction="BULLISH", confidence=0.9,
                       action="BUY", rationale="x", generated_at=datetime.now(timezone.utc))
    assert r.rule_filled is False
    r.rule_filled = True
    assert r.rule_filled is True


def test_generate_recommendations_marks_its_own_backfills(monkeypatch):
    """END-TO-END: the REAL synthesis path must flag the tickers it back-filled.

    Verified separately from the helper above because that only proves the field
    exists — this proves `generate_recommendations` actually sets it, which is
    what makes `pipeline._persist_run` stamp RULE_FILL_MODEL instead of the run's
    model id. Easy to miss in production: a fill only reaches the persisted
    top-10 when its confidence beats the LLM's own calls, so a run can legitimately
    show zero fills and hide a broken flag.
    """
    import json
    import src.analysis.claude_analyst as ca
    from src.models import TickerSignal

    sigs = [TickerSignal(ticker=f"T{i}", direction="BULLISH", confidence=0.9 - 0.01 * i,
                         combined_score=0.5, sentiment_score=0.3, technical_score=0.2,
                         rationale=f"agg text {i} | Technical score: +0.20")
            for i in range(6)]
    payload = {"recommendations": [
        {"ticker": "T0", "type": "STOCK", "direction": "BULLISH", "action": "BUY",
         "time_horizon": "SWING", "confidence": 0.88, "rationale": "Authored prose T0."},
        {"ticker": "T1", "type": "STOCK", "direction": "BEARISH", "action": "SELL",
         "time_horizon": "SWING", "confidence": 0.83, "rationale": "Authored prose T1."},
    ]}
    for fn in ("_call_deepseek_analyst", "_call_qwen_analyst", "_call_claude_analyst"):
        monkeypatch.setattr(ca, fn, lambda *a, **k: json.dumps(payload))

    recs = {r.ticker: r for r in ca.generate_recommendations(sigs)}
    assert len(recs) == 6
    # the two the model answered are its own…
    assert recs["T0"].rule_filled is False and recs["T1"].rule_filled is False
    # …every ticker it was never asked about is flagged
    assert all(recs[f"T{i}"].rule_filled is True for i in range(2, 6))
    # and the persist-time expression turns exactly those into the rule-based label
    assert [t for t in sorted(recs) if recs[t].rule_filled] == ["T2", "T3", "T4", "T5"]
