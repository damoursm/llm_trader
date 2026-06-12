"""Duplicate-recommendation defenses (the 2026-06-11 XLE incident).

DeepSeek's synthesis response repeated several tickers; the duplicates flowed
through record_new_trades into two identical ledger trades, two broker entries
under one orderRef, and two full-position exits that flipped the paper account
short. Two independent layers must each stop this:

  1. claude_analyst._dedupe_recommendations — kill duplicates at the source.
  2. tracker.record_new_trades — within-batch already_open guard, so a
     duplicated ticker opens exactly one trade even if a duplicate sneaks in.

(The third layer — reconcile's same-tick client_ref guard — is covered in
tests/test_broker_drift.py P6.)
"""

from datetime import datetime, timezone

from config.settings import settings
from src.models import Recommendation


def _rec(ticker, action="BUY", direction="BULLISH"):
    return Recommendation(
        ticker=ticker, type="ETF", direction=direction, action=action,
        confidence=0.85, time_horizon="1-3 months", rationale="test",
        generated_at=datetime.now(timezone.utc),
    )


# ── layer 1: dedupe at the LLM-response parse ──────────────────────────────

def test_dedupe_keeps_first_occurrence_per_ticker():
    from src.analysis.claude_analyst import _dedupe_recommendations
    recs = [_rec("XLE"), _rec("SPY"), _rec("XLE", action="SELL", direction="BEARISH"),
            _rec("ITA"), _rec("SPY")]
    out = _dedupe_recommendations(recs, "test-model")
    assert [r.ticker for r in out] == ["XLE", "SPY", "ITA"]
    assert out[0].action == "BUY"   # first occurrence wins, even over a conflict


def test_dedupe_is_noop_for_unique_tickers():
    from src.analysis.claude_analyst import _dedupe_recommendations
    recs = [_rec("XLE"), _rec("SPY"), _rec("ITA")]
    out = _dedupe_recommendations(recs, "test-model")
    assert [r.ticker for r in out] == ["XLE", "SPY", "ITA"]


# ── layer 2: record_new_trades opens a duplicated ticker exactly once ──────

def test_record_new_trades_opens_duplicate_ticker_once(tmp_path, monkeypatch):
    from src.performance import tracker
    monkeypatch.setattr(tracker, "TRADES_FILE", tmp_path / "no-legacy.json")
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: 57.12)
    monkeypatch.setattr(settings, "enable_intraday_timing", False)

    rec = _rec("XLE")
    diag = tracker.record_new_trades([rec, rec.model_copy()], run_id="r1")

    assert diag["opened"] == 1
    assert diag["skipped_already_open"] == 1
    trades = tracker._load_trades()
    assert len([t for t in trades if t["ticker"] == "XLE"]) == 1
