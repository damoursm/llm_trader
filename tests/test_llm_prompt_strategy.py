"""LLM prompt/strategy improvements (2026-07-12):

#1 Blind-synthesis A/B — the aggregator's verdict (direction / combined_confidence /
   sources_agreeing + HORIZON/EXPECTED-MOVE opinion lines + the "trust the
   pre-computed confidence" instruction) is hidden from the MAIN synthesis on the
   blind arm, so the LLM must judge from raw per-method evidence. Sighted arm stays
   byte-compatible with the pre-A/B prompt. Entries stamp entry_blind_synthesis;
   compute_blind_synthesis_eval aggregates closed outcomes ON vs OFF.

#2 Two-tick confirmation on the noisy LLM exits (llm_signal_flipped /
   llm_confidence_loss): first breaching review ARMS llm_exit_pending (position
   holds), second consecutive breach closes, a holding review DISARMS, an absent
   review leaves the marker untouched. Non-LLM reasons pass through.

#3 Guaranteed-JSON synthesis: the output spec asks for {"recommendations": [...]};
   the parser accepts both the wrapped object and the legacy bare array (plus the
   truncated-wrapped repair); _call_qwen_analyst requests json_object mode and
   retries once without it if rejected.
"""

import json
import types

import pytest

import src.analysis.claude_analyst as ca
import src.performance.tracker as tr
from config.settings import settings
from src.models import TickerSignal


def _signal(ticker="XLE", **over):
    kw = dict(
        ticker=ticker, direction="BULLISH", confidence=0.4,
        action_suggestion="WATCH", news_sentiment_score=0.0,
        sentiment_score=0.0, insider_score=0.0, technical_score=0.1,
        sources_agreeing=1, key_reasons=["r"], rationale="r", price=57.0,
    )
    kw.update(over)
    return TickerSignal(**kw)


def _capture_prompt(monkeypatch, response='{"recommendations": []}'):
    captured = {}

    def fake(prompt, **kwargs):
        captured["prompt"] = prompt
        return response

    monkeypatch.setattr(ca, "_call_claude_analyst", fake)
    monkeypatch.setattr(ca, "_call_deepseek_analyst", fake)
    monkeypatch.setattr(ca, "_call_qwen_analyst", fake)
    return captured


# ── #1 blind-synthesis prompt rendering ─────────────────────────────────────

def test_sighted_prompt_carries_the_aggregate_verdict(monkeypatch):
    cap = _capture_prompt(monkeypatch)
    ca.generate_recommendations([_signal()], blind_synthesis=False)
    p = cap["prompt"]
    # 2026-07-22: the buy/sell split sides ride the sighted verdict line between
    # confidence and sources_agreeing (a contested 0.55-vs-0.40 read is visible).
    assert "direction=BULLISH, combined_confidence=40%, buy_score=" in p
    assert "sell_score=" in p and "sources_agreeing=1" in p
    assert "sources_agreeing ≥ 2 → eligible" in p
    assert "The pre-computed confidence already reflects" in p
    assert "Trust it" in p


def test_blind_prompt_hides_verdict_and_swaps_instructions(monkeypatch):
    cap = _capture_prompt(monkeypatch)
    ca.generate_recommendations([_signal()], blind_synthesis=True)
    p = cap["prompt"]
    # verdict trio gone
    assert "combined_confidence=" not in p
    assert "sources_agreeing=1" not in p
    assert "direction=BULLISH" not in p
    # anchor instructions gone, own-judgment calibration present
    assert "The pre-computed confidence already reflects" not in p
    assert "Trust it" not in p
    assert "NO pre-computed verdict is provided" in p
    assert "0.5 = coin flip" in p
    # raw evidence still present (the ingredients survive)
    assert "Technical score=+0.10" in p
    # the ticker line itself survives, bare
    assert "- XLE:" in p


def test_blind_prompt_suppresses_horizon_and_expected_move(monkeypatch):
    monkeypatch.setattr(settings, "enable_horizon_synthesis", True)
    monkeypatch.setattr(settings, "enable_expected_move_ranking", True)
    sig = _signal(target_horizon="1w", horizon_label="SWING",
                  horizon_net_edge_pct=1.2, horizon_conviction=0.5,
                  expected_move_pct=2.3, upside_score=0.4, market_aligned="aligned")
    cap = _capture_prompt(monkeypatch)
    ca.generate_recommendations([sig], blind_synthesis=False)
    assert "HORIZON MODEL" in cap["prompt"]
    assert "EXPECTED MOVE" in cap["prompt"]
    cap2 = _capture_prompt(monkeypatch)
    ca.generate_recommendations([sig], blind_synthesis=True)
    assert "HORIZON MODEL" not in cap2["prompt"]
    assert "EXPECTED MOVE" not in cap2["prompt"]


# ── #1 stamping + eval ──────────────────────────────────────────────────────

def test_blind_synthesis_eval_groups_by_entry_stamp():
    trades = [
        {"status": "CLOSED", "entry_blind_synthesis": True, "return_pct": 2.0},
        {"status": "CLOSED", "entry_blind_synthesis": True, "return_pct": -1.0},
        {"status": "CLOSED", "entry_blind_synthesis": False, "return_pct": -3.0},
        {"status": "CLOSED", "return_pct": 9.9},              # pre-experiment: excluded
        {"status": "OPEN", "entry_blind_synthesis": True, "return_pct": 5.0},  # open: excluded
    ]
    ev = tr.compute_blind_synthesis_eval(trades)
    assert ev["on"] == {"trades": 2, "win_rate": 50.0, "avg_return": 0.5}
    assert ev["off"] == {"trades": 1, "win_rate": 0.0, "avg_return": -3.0}


# ── #2 two-tick confirmation ────────────────────────────────────────────────

def _review(action="SELL", conf=0.9):
    return types.SimpleNamespace(action=action, confidence=conf)


@pytest.fixture
def _confirm_on(monkeypatch):
    monkeypatch.setattr(settings, "enable_llm_exit_confirmation", True)


def test_first_breach_arms_and_holds(_confirm_on):
    t = {"ticker": "XLE"}
    reason, dirty = tr._confirm_llm_exit(t, "llm_signal_flipped", _review())
    assert reason is None and dirty is True
    assert t["llm_exit_pending"]["reason"] == "llm_signal_flipped"


def test_second_consecutive_breach_closes(_confirm_on):
    t = {"ticker": "XLE"}
    tr._confirm_llm_exit(t, "llm_confidence_loss", _review("BUY", 0.3))
    reason, dirty = tr._confirm_llm_exit(t, "llm_confidence_loss", _review("BUY", 0.2))
    assert reason == "llm_confidence_loss" and dirty is True
    assert "llm_exit_pending" not in t
    assert t["llm_exit_confirmed_from"]["reason"] == "llm_confidence_loss"


def test_either_llm_reason_confirms(_confirm_on):
    """Flip then confidence-loss = two consecutive 'get out' judgments."""
    t = {"ticker": "XLE"}
    tr._confirm_llm_exit(t, "llm_signal_flipped", _review())
    reason, _ = tr._confirm_llm_exit(t, "llm_confidence_loss", _review("BUY", 0.2))
    assert reason == "llm_confidence_loss"


def test_holding_review_disarms(_confirm_on):
    t = {"ticker": "XLE"}
    tr._confirm_llm_exit(t, "llm_signal_flipped", _review())
    reason, dirty = tr._confirm_llm_exit(t, None, _review("BUY", 0.9))  # review holds
    assert reason is None and dirty is True
    assert "llm_exit_pending" not in t
    # a fresh breach after the disarm must arm again, not close
    reason, _ = tr._confirm_llm_exit(t, "llm_signal_flipped", _review())
    assert reason is None


def test_absent_review_leaves_marker_armed(_confirm_on):
    t = {"ticker": "XLE"}
    tr._confirm_llm_exit(t, "llm_signal_flipped", _review())
    reason, dirty = tr._confirm_llm_exit(t, None, None)   # no review this tick
    assert reason is None and dirty is False
    assert "llm_exit_pending" in t                         # still armed
    reason, _ = tr._confirm_llm_exit(t, "llm_signal_flipped", _review())
    assert reason == "llm_signal_flipped"                  # next breach confirms


def test_non_llm_reasons_pass_through(_confirm_on):
    t = {"ticker": "XLE", "llm_exit_pending": {"reason": "llm_signal_flipped"}}
    reason, _ = tr._confirm_llm_exit(t, "macro_regime_exit", _review())
    assert reason == "macro_regime_exit"                   # closes regardless


def test_flag_off_is_single_review_passthrough(monkeypatch):
    monkeypatch.setattr(settings, "enable_llm_exit_confirmation", False)
    t = {"ticker": "XLE"}
    reason, dirty = tr._confirm_llm_exit(t, "llm_signal_flipped", _review())
    assert reason == "llm_signal_flipped" and dirty is False
    assert "llm_exit_pending" not in t


# ── #3 parser: wrapped object / bare array / truncated-wrapped repair ───────

def _rec(ticker="XLE"):
    return {"ticker": ticker, "type": "ETF", "direction": "BULLISH", "action": "BUY",
            "time_horizon": "SWING", "confidence": 0.8, "rationale": "r"}


def test_extract_rec_list_accepts_both_shapes():
    assert ca._extract_rec_list([_rec()]) == [_rec()]
    assert ca._extract_rec_list({"recommendations": [_rec()]}) == [_rec()]
    assert ca._extract_rec_list({"other_key": [_rec()]}) == [_rec()]   # defensive
    assert ca._extract_rec_list({"n": 1}) == []
    assert ca._extract_rec_list("junk") == []


def test_wrapped_object_response_parses(monkeypatch):
    _capture_prompt(monkeypatch, response=json.dumps({"recommendations": [_rec()]}))
    recs = ca.generate_recommendations([_signal()])
    assert any(r.ticker == "XLE" and r.action == "BUY" for r in recs)


def test_truncated_wrapped_response_repairs(monkeypatch):
    # Cut mid-second-object: the wrapper prefix must be stripped so the array
    # repair recovers the first complete object.
    full = json.dumps({"recommendations": [_rec("XLE"), _rec("SPY")]})
    truncated = full[: full.find('"SPY"') + 5]
    _capture_prompt(monkeypatch, response=truncated)
    recs = ca.generate_recommendations([_signal()])
    assert any(r.ticker == "XLE" for r in recs)


def test_qwen_json_mode_requests_response_format(monkeypatch):
    monkeypatch.setattr(settings, "qwen_json_mode", True)
    cap = {}

    class _Stream:
        def __enter__(self): return []
        def __exit__(self, *a): return False

    class _Completions:
        def create(self, **kw):
            cap.update(kw)
            return _Stream()

    class _Client:
        chat = type("C", (), {"completions": _Completions()})()

    monkeypatch.setattr(ca, "_get_qwen_analyst_client", lambda: _Client())
    ca._call_qwen_analyst("hi")
    assert cap["response_format"] == {"type": "json_object"}


def test_qwen_json_mode_retries_without_on_rejection(monkeypatch):
    monkeypatch.setattr(settings, "qwen_json_mode", True)
    attempts = []

    class _Stream:
        def __enter__(self): return []
        def __exit__(self, *a): return False

    class _Completions:
        def create(self, **kw):
            attempts.append("json" if "response_format" in kw else "plain")
            if "response_format" in kw:
                raise RuntimeError("response_format not supported with enable_thinking")
            return _Stream()

    class _Client:
        chat = type("C", (), {"completions": _Completions()})()

    monkeypatch.setattr(ca, "_get_qwen_analyst_client", lambda: _Client())
    ca._call_qwen_analyst("hi")                        # must not raise
    assert attempts == ["json", "plain"]               # one retry, sans response_format


def test_qwen_json_mode_off_never_sends_response_format(monkeypatch):
    monkeypatch.setattr(settings, "qwen_json_mode", False)
    cap = {}

    class _Stream:
        def __enter__(self): return []
        def __exit__(self, *a): return False

    class _Completions:
        def create(self, **kw):
            cap.update(kw)
            return _Stream()

    class _Client:
        chat = type("C", (), {"completions": _Completions()})()

    monkeypatch.setattr(ca, "_get_qwen_analyst_client", lambda: _Client())
    ca._call_qwen_analyst("hi")
    assert "response_format" not in cap
