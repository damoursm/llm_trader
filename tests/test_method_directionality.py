"""Directional integrity of the per-ticker scorers (2026-07-24 audit).

The aggregator's core invariant: every method returns a score in [-1, +1] whose
SIGN is the predicted direction of the stock. A method that cannot change sign
between a rising and a falling tape is not merely weak — it is structurally
unable to express direction, and its weight in the combine is wasted.

These tests feed each scorer mirror-image synthetic series (same noise, opposite
drift) and assert the sign actually flips. All pure, no network, no cache.
"""

import numpy as np
import pandas as pd
import pytest


def _series(drift: float, n: int = 300, seed: int = 7) -> pd.DataFrame:
    """Synthetic OHLCV with a controlled drift and reproducible noise."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.006, n)))
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    hi = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    lo = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    op = np.r_[close[0], close[:-1]]
    vol = rng.integers(5_000_000, 15_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": op,
         "High": np.maximum(hi, np.maximum(op, close)),
         "Low": np.minimum(lo, np.minimum(op, close)),
         "Close": close, "Volume": vol},
        index=idx,
    )


UP, DOWN = _series(+0.004), _series(-0.004)


# ── money_flow: two directional defects found and fixed 2026-07-24 ──────────

def test_money_flow_sign_follows_the_tape():
    """The regression that motivated the audit: money_flow scored POSITIVE on
    both a rising and a falling tape (+0.31 / +0.28), because the contrarian
    MFI term and the trend-following CMF term were equally weighted and
    cancelled. It must now flip sign with the trend."""
    from src.signals.money_flow import compute_money_flow_score
    up, _, _ = compute_money_flow_score("TEST", UP)
    down, _, _ = compute_money_flow_score("TEST", DOWN)
    assert up > 0, f"rising tape should read bullish, got {up:+.3f}"
    assert down < 0, f"falling tape should read bearish, got {down:+.3f}"


def test_money_flow_v3_score_is_cmf_only():
    """v3 (2026-08-16): the score is exactly tanh(tanh(cmf/0.15)/0.6) — the
    3-year gated battery measured CMF alone at IC +0.048/t +11.8 vs the
    composite's +0.035, with OBV informationless (t +0.97) and the contrarian
    MFI term anti-predictive (t −2.72). MFI survives only as the aux display
    value; it must NOT move the score."""
    import numpy as np
    from src.signals.money_flow import compute_money_flow_score
    for df in (UP, DOWN, _series(0.0)):
        score, mfi, cmf = compute_money_flow_score("TEST", df)
        expected = round(float(np.tanh(np.tanh(cmf / 0.15) / 0.6)), 3)
        assert score == pytest.approx(expected, abs=1e-9)
        assert 0.0 <= mfi <= 100.0                     # aux value still returned


def test_money_flow_tail_slice_matches_full_history():
    """The scorer computes on the trailing slice for latency — outputs must be
    IDENTICAL to a full-history compute (rolling windows at the last bar only
    depend on the trailing rows)."""
    from src.signals.money_flow import _TAIL_BARS, compute_money_flow_score
    long_up = _series(+0.004, n=1200)
    assert len(long_up) > _TAIL_BARS
    full = compute_money_flow_score("TEST", long_up)
    tail = compute_money_flow_score("TEST", long_up.tail(_TAIL_BARS))
    assert full == tail


def test_money_flow_stays_in_range():
    from src.signals.money_flow import compute_money_flow_score
    for df in (UP, DOWN, _series(0.0)):
        score, mfi, cmf = compute_money_flow_score("TEST", df)
        assert -1.0 <= score <= 1.0
        assert 0.0 <= mfi <= 100.0


# ── the invariant, across every OHLCV scorer that claims to be two-sided ────

@pytest.mark.parametrize("name", [
    "vwap", "momentum", "trend_strength", "money_flow", "hi52", "mom_12_1", "avwap",
])
def test_scorer_flips_sign_between_up_and_down_tape(name):
    """Each of these claims a two-sided directional read, so a mirror-image
    tape must produce opposite signs. (Deliberately excluded: contrarian
    scorers whose sign is inverted by design, context-gated methods that
    abstain on one side, and detectors that may find no setup at all.)"""
    from src.signals.vwap import compute_vwap_score
    from src.signals.price_momentum import compute_price_momentum_score
    from src.signals.trend_strength import compute_trend_strength_score
    from src.signals.money_flow import compute_money_flow_score
    from src.signals.classic_anomalies import compute_high_52w_score, compute_momentum_12_1_score
    from src.signals.anchored_vwap import compute_anchored_vwap_score

    fns = {
        "vwap": compute_vwap_score,            # contrarian: sign is inverted vs trend
        "momentum": compute_price_momentum_score,
        "trend_strength": compute_trend_strength_score,
        "money_flow": compute_money_flow_score,
        "hi52": compute_high_52w_score,
        "mom_12_1": compute_momentum_12_1_score,
        "avwap": compute_anchored_vwap_score,
    }
    fn = fns[name]
    up = fn("TEST", UP)
    down = fn("TEST", DOWN)
    up = up[0] if isinstance(up, tuple) else up
    down = down[0] if isinstance(down, tuple) else down
    assert -1.0 <= up <= 1.0 and -1.0 <= down <= 1.0, "scores must stay in [-1, +1]"
    assert up * down < 0, (
        f"{name} returned the same sign on a rising and falling tape "
        f"(up={up:+.3f}, down={down:+.3f}) — it cannot express direction"
    )
