"""Score-sign convention guardrail.

The whole stack — ``combined_score``, ``direction``, the IC tables, and the
inversion logic (``signal_panel`` / ``simulated_trades`` IC<0 ⇒ use −score, plus
the ``edge_curve`` auto-flip) — relies on ONE invariant: every method's score SIGN
is the predicted direction of the STOCK's forward return (+ = up/bullish, − =
down/bearish), position-independent. Mean-reversion / contrarian methods bake the
reversal INTO the sign rather than reporting raw state.

These tests lock the *surprising* (contrarian) signs and the direction→trade→flip
mapping, so the inversion logic stays well-defined. If a refactor flips one of these,
the combine and every IC/inversion readout silently invert with it — hence the guard.
"""

import numpy as np
import pandas as pd
import pytest


# ── Mean-reversion / contrarian methods encode PREDICTED DIRECTION, not state ──

def test_put_call_is_contrarian_predicted_direction():
    # Heavy PUTS = fear ⇒ predicted UP (+); heavy CALLS = greed ⇒ predicted DOWN (−).
    # The sign is the STOCK's expected move, NOT "the book is put-heavy".
    from src.signals.aggregator import _PC_SIGNAL_SCORE
    assert _PC_SIGNAL_SCORE["EXTREME_PUTS"] > 0
    assert _PC_SIGNAL_SCORE["PUTS_HEAVY"] > 0
    assert _PC_SIGNAL_SCORE["BALANCED"] == 0.0
    assert _PC_SIGNAL_SCORE["CALLS_HEAVY"] < 0
    assert _PC_SIGNAL_SCORE["EXTREME_CALLS"] < 0


def test_insider_tx_signs_are_predicted_direction():
    # Accumulation ⇒ +, distribution ⇒ −.
    from src.signals.aggregator import _TX_SCORE
    for tx in ("unusual_call", "13d_activist_stake", "13f_new_position", "13f_increase"):
        assert _TX_SCORE[tx][0] > 0
    for tx in ("unusual_put", "planned_sale_144", "13f_exit", "13f_decrease"):
        assert _TX_SCORE[tx][0] < 0


def _iv_rank_df(last5_change_pct: float) -> pd.DataFrame:
    """220 calm bars then a sharp |move| over the final 5 → high realized-vol
    percentile + a strongly normalised 5-day return (|z_ret| >> 1)."""
    rng = np.random.default_rng(0)
    calm = 100.0 + np.cumsum(rng.normal(0.0, 0.03, 215))
    move = np.linspace(calm[-1], calm[-1] * (1.0 + last5_change_pct), 6)[1:]
    close = np.concatenate([calm, move])
    return pd.DataFrame({
        "Open": close, "High": close * 1.002, "Low": close * 0.998,
        "Close": close, "Volume": 1_000_000,
    })


def test_iv_rank_contrarian_sign_is_predicted_direction():
    # A mean-reversion scorer: in a high realized-vol regime a sharp SELLOFF is a
    # CAPITULATION_BUY → POSITIVE (predicted up); a sharp RALLY is FADE → NEGATIVE.
    from src.signals.iv_rank import compute_iv_rank_score
    sell_score, ir_down, _, label_down = compute_iv_rank_score("TEST", df=_iv_rank_df(-0.20))
    rally_score, ir_up, _, label_up = compute_iv_rank_score("TEST", df=_iv_rank_df(+0.20))
    assert ir_down >= 70.0 and ir_up >= 70.0          # both are high-vol regimes
    assert sell_score > 0 and label_down == "CAPITULATION_BUY"   # selloff → bullish
    assert rally_score < 0 and label_up == "FADE_EXTREME"        # rally  → bearish


# ── direction → trade → flip mapping is sign-consistent (positive = long) ──────

def test_flip_trade_inverts_action_and_direction_together():
    from src.performance.tracker import _flip_trade
    long_t = {"ticker": "AAA", "action": "BUY", "direction": "BULLISH",
              "status": "CLOSED", "entry_price": 100.0, "exit_price": 110.0,
              "type": "STOCK"}
    flipped = _flip_trade(long_t)
    assert flipped["action"] == "SELL" and flipped["direction"] == "BEARISH"
    # A long that gained becomes a short that lost (sign of return flips).
    assert flipped["return_pct"] < 0
    # Round-trip restores the original orientation.
    back = _flip_trade(flipped)
    assert back["action"] == "BUY" and back["direction"] == "BULLISH"


def test_solo_method_direction_follows_score_sign():
    # The inversion-relevant mapping used by both the ledger solo perf and the
    # simulated panel: a method's standalone call is LONG iff its score is positive.
    from src.analysis.simulated_trades import backfill_from_signals
    from src.db import repo
    from src.db.schema import SIGNAL_METHOD_COLUMNS
    scores = {m: 0.0 for m in SIGNAL_METHOD_COLUMNS}
    scores["news"] = 0.5      # bullish view → BUY
    scores["tech"] = -0.2     # bearish view → SELL
    repo.insert_signals(
        "run-conv", "2026-06-09T14:00:00+00:00", "2026-06-09",
        [{"ticker": "AAPL", "type": "STOCK", "direction": "BULLISH",
          "combined_score": 0.3, "confidence": 0.8, "n_methods_agreeing": 1,
          "dominant_method": "news", "price": 100.0, "scores": scores}])
    backfill_from_signals()
    df = repo.fetch_df("SELECT * FROM simulated_trades", read_only=False)
    assert df[df.method == "news"].iloc[0]["direction"] == "BUY"
    assert df[df.method == "tech"].iloc[0]["direction"] == "SELL"
