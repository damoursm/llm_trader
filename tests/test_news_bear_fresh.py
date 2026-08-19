"""news_bear_fresh (2026-08-15, panel-first weight 0): formula + wiring.

The measured basis (39 daily cross-sections, gated, pivot target): within
bear-news events the modulated score's daily IC is +0.095 (t +2.96) vs +0.029
for news alone — the guard is a genuine product effect. These tests pin the
guard's shape (abstain in the hole, passthrough flat, capped boost when the
tape ignored the news), the bear-only contract, and the add-method checklist.
"""

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.signals.news_bear_fresh import compute_news_bear_fresh


def _frame(move3=0.0, n=30):
    """Synthetic daily closes: alternating ±0.5% noise, then a final 3-bar move
    of `move3` cumulative (equal per-bar steps)."""
    closes = [100.0]
    for i in range(n - 4):
        closes.append(closes[-1] * (1 + (0.005 if i % 2 == 0 else -0.005)))
    step = (1.0 + move3) ** (1.0 / 3.0)
    for _ in range(3):
        closes.append(closes[-1] * step)
    idx = pd.date_range("2026-06-01", periods=len(closes), freq="B")
    return pd.DataFrame({"Open": closes, "High": closes, "Low": closes,
                         "Close": closes, "Volume": [1e6] * len(closes)}, index=idx)


# ── the guard's shape ───────────────────────────────────────────────────────

def test_flat_tape_passes_the_news_through():
    score, z3 = compute_news_bear_fresh("TST", -0.6, df=_frame(0.0))
    assert score == pytest.approx(-0.6, abs=0.02)      # guard ≈ 1.0
    assert abs(z3) < 0.1


def test_never_short_into_the_hole():
    """A −5% 3-day aligned decline (≈4σ here) → the guard hits 0: abstain."""
    score, z3 = compute_news_bear_fresh("TST", -0.9, df=_frame(-0.05))
    assert score == 0.0
    assert z3 < -2.0


def test_unfallen_short_is_boosted_and_capped():
    """Price ROSE 6% against the bad news → guard caps at 1.5x."""
    score, z3 = compute_news_bear_fresh("TST", -0.4, df=_frame(+0.06))
    assert score == pytest.approx(-0.6, abs=1e-6)      # −0.4 × 1.5 exactly
    assert z3 > 2.0
    # and the floor: a huge bear read boosted 1.5x still clips at −1
    score2, _ = compute_news_bear_fresh("TST", -0.9, df=_frame(+0.06))
    assert score2 == -1.0


def test_monotone_in_the_aligned_move():
    scores = [compute_news_bear_fresh("TST", -0.5, df=_frame(m))[0]
              for m in (+0.04, +0.01, 0.0, -0.01, -0.04)]
    assert scores == sorted(scores)                    # more fallen → closer to 0
    assert scores[0] < scores[2] < scores[-1] == 0.0


# ── the bear-only contract + fail-soft ──────────────────────────────────────

def test_bull_and_zero_news_abstain():
    assert compute_news_bear_fresh("TST", 0.6, df=_frame(0.0)) == (0.0, 0.0)
    assert compute_news_bear_fresh("TST", 0.0, df=_frame(0.0)) == (0.0, 0.0)
    assert compute_news_bear_fresh("TST", None, df=_frame(0.0)) == (0.0, 0.0)


def test_missing_or_short_history_abstains():
    assert compute_news_bear_fresh("ZZNOPE_NOT_CACHED", -0.6, df=None) == (0.0, 0.0)
    assert compute_news_bear_fresh("TST", -0.6, df=_frame(0.0, n=10)) == (0.0, 0.0)
    bad = _frame(0.0).drop(columns=["Close"])
    assert compute_news_bear_fresh("TST", -0.6, df=bad) == (0.0, 0.0)


def test_degenerate_vol_abstains():
    df = _frame(0.0)
    df["Close"] = 100.0                                # zero vol
    assert compute_news_bear_fresh("TST", -0.6, df=df) == (0.0, 0.0)


# ── add-method wiring (mirrors the news_shock checklist) ────────────────────

def test_news_bear_fresh_wiring_complete():
    from src.analysis.code_version import METHOD_SOURCES, unmapped_methods
    from src.db.schema import SIGNAL_BASE_METHOD_COLUMNS, _ADD_COLUMNS
    from src.models import TickerSignal
    from src.performance.tracker import _ALL_METHODS, METHOD_CATEGORIES, METHOD_LABELS
    from src.signals.agreement import FAMILY_OF
    from src.signals.aggregator import _BASE_WEIGHTS

    assert "news_bear_fresh" in _ALL_METHODS
    assert "news_bear_fresh" in SIGNAL_BASE_METHOD_COLUMNS
    assert "news_bear_fresh" in METHOD_CATEGORIES["Sentiment"]
    assert "news_bear_fresh" in METHOD_LABELS
    assert "news_bear_fresh" in METHOD_SOURCES and not unmapped_methods()
    assert ("signals", "news_bear_fresh") in {(t, c) for t, c, _ in _ADD_COLUMNS}
    assert "news_bear_fresh_score" in TickerSignal.model_fields
    # PANEL-FIRST: not weighted, not a family voter — mirrors news_shock.
    assert "news_bear_fresh" not in _BASE_WEIGHTS
    assert "news_bear_fresh" not in FAMILY_OF


def test_score_flows_to_the_signal(monkeypatch):
    """End to end through build_signals with everything else off."""
    import src.signals.aggregator as agg
    import src.signals.news_bear_fresh as nbf
    from tests.test_news_events import _OFF
    for flag in _OFF:
        monkeypatch.setattr(settings, flag, False)
    monkeypatch.setattr(settings, "enable_news_sentiment", True)
    monkeypatch.setattr(settings, "enable_massive_tech", False)
    monkeypatch.setattr(settings, "enable_news_bear_fresh", True)
    monkeypatch.setattr(settings, "signal_scoring_max_workers", 2)
    monkeypatch.setattr(settings, "inverted_methods", "")
    monkeypatch.setattr(agg, "analyse_sentiment",
                        lambda t, a, force_engine=None: (-0.6, "bad news"))
    monkeypatch.setattr(nbf, "compute_news_bear_fresh",
                        lambda ticker, news, df=None: (-0.42, 0.5))
    s = agg.build_signals(["TST"], articles=[], snapshots=[])[0]
    assert s.news_bear_fresh_score == pytest.approx(-0.42)
