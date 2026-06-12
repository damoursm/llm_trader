"""Held-positions prompt A/B experiment (2026-06-12).

Half the runs (open_positions_prompt_share, default 0.5) tell the LLM what
the system currently holds via <open_positions_context> + instruction §27;
half leave it blind (the pre-experiment behavior). The per-run coin flip is
stamped on every trade CLOSED that run (exit_hold_prompt) by BOTH close
paths, and compute_hold_prompt_eval aggregates exit outcomes ON vs OFF for
the dashboard's method-evaluation table.
"""

from datetime import datetime, timezone

from src.models import Recommendation, TickerSignal


def _signal(ticker="XLE"):
    return TickerSignal(
        ticker=ticker, direction="BULLISH", confidence=0.4,
        action_suggestion="WATCH", news_sentiment_score=0.0,
        sentiment_score=0.0, insider_score=0.0, technical_score=0.1,
        sources_agreeing=1, key_reasons=["r"], rationale="r", price=57.0,
    )


def _position(ticker="XLE", action="BUY"):
    return {"ticker": ticker, "action": action, "entry_date": "2026-06-11",
            "entry_price": 57.12, "current_price": 57.50,
            "return_pct": 0.45, "days_held": 1}


# ── prompt rendering ───────────────────────────────────────────────────────

def test_open_positions_reach_the_synthesis_prompt(monkeypatch):
    import src.analysis.claude_analyst as ca

    captured = {}

    def fake(prompt):
        captured["prompt"] = prompt
        return "[]"

    monkeypatch.setattr(ca, "_call_claude_analyst", fake)
    monkeypatch.setattr(ca, "_call_deepseek_analyst", fake)

    ca.generate_recommendations(
        [_signal()], open_positions=[_position(), _position("TRUP", "SELL")])
    p = captured["prompt"]
    assert "<open_positions_context>" in p
    assert "XLE: LONG (opened BUY) since 2026-06-11" in p
    assert "TRUP: SHORT (opened SELL)" in p
    assert "27. Held-position review" in p
    assert "ZERO endowment bias" in p


def test_prompt_off_run_has_no_block(monkeypatch):
    import src.analysis.claude_analyst as ca

    captured = {}

    def fake(prompt):
        captured["prompt"] = prompt
        return "[]"

    monkeypatch.setattr(ca, "_call_claude_analyst", fake)
    monkeypatch.setattr(ca, "_call_deepseek_analyst", fake)

    ca.generate_recommendations([_signal()], open_positions=None)
    assert "<open_positions_context>" not in captured["prompt"]
    assert "27. Held-position review" not in captured["prompt"]


# ── exit stamping (both close paths) ───────────────────────────────────────

def _open_trade(ticker="XLE", action="BUY"):
    return {
        "ticker": ticker, "type": "ETF", "action": action, "status": "OPEN",
        "entry_date": "2026-06-11", "entry_price": 57.12,
        "current_price": 57.50, "current_price_datetime": "2026-06-12T15:00:00+00:00",
        "position_size_multiplier": 1.0, "return_pct": 0.45,
        "signal_at_entry": {"combined_score": 0.5, "confidence": 0.85,
                            "direction": "BULLISH", "methods_agreeing": ["news"]},
    }


def test_reversal_close_stamps_the_coin_flip(tmp_path, monkeypatch):
    from src.performance import tracker
    monkeypatch.setattr(tracker, "TRADES_FILE", tmp_path / "no-legacy.json")
    tracker._save_trades([_open_trade()])

    rec = Recommendation(ticker="XLE", type="ETF", direction="BEARISH",
                         action="SELL", confidence=0.9, time_horizon="1w",
                         rationale="reversal",
                         generated_at=datetime.now(timezone.utc))
    closed = tracker.close_trades_on_signal_reversal([rec], hold_prompt_active=True)
    assert closed == 1
    t = tracker._load_trades()[0]
    assert t["status"] == "CLOSED" and t["exit_reason"] == "signal_reversal"
    assert t["exit_hold_prompt"] is True


def test_monitor_close_stamps_the_coin_flip(tmp_path, monkeypatch):
    from config.settings import settings
    from src.performance import tracker
    monkeypatch.setattr(tracker, "TRADES_FILE", tmp_path / "no-legacy.json")
    monkeypatch.setattr(settings, "enable_signal_decay_exits", True)
    monkeypatch.setattr(settings, "enable_intraday_exit", False)
    tracker._save_trades([_open_trade()])

    # Today's signal flipped hard against the long → signal_flipped exit.
    flipped = _signal("XLE").model_copy(update={"combined_score": -0.6, "confidence": 0.2})
    closed = tracker.monitor_open_positions(
        signals_by_ticker={"XLE": flipped}, hold_prompt_active=False)
    assert closed == 1
    t = tracker._load_trades()[0]
    assert t["status"] == "CLOSED"
    assert t["exit_hold_prompt"] is False


# ── A/B aggregation ────────────────────────────────────────────────────────

def test_hold_prompt_eval_groups_by_flag():
    from src.performance.tracker import compute_hold_prompt_eval
    trades = [
        {"status": "CLOSED", "exit_hold_prompt": True,  "return_pct": 2.0},
        {"status": "CLOSED", "exit_hold_prompt": True,  "return_pct": -1.0},
        {"status": "CLOSED", "exit_hold_prompt": False, "return_pct": 0.5},
        {"status": "CLOSED", "exit_hold_prompt": None,  "return_pct": 9.9},  # pre-experiment
        {"status": "OPEN",   "exit_hold_prompt": True,  "return_pct": 1.0},  # not closed
    ]
    ev = compute_hold_prompt_eval(trades)
    assert ev["on"] == {"trades": 2, "win_rate": 50.0, "avg_return": 0.5}
    assert ev["off"] == {"trades": 1, "win_rate": 100.0, "avg_return": 0.5}


def test_hold_prompt_eval_empty_segments_are_none():
    from src.performance.tracker import compute_hold_prompt_eval
    assert compute_hold_prompt_eval([]) == {"on": None, "off": None}
