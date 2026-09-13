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
    # Promoted 2026-09-11 (user request) off weight 0 — a weight-0 method is
    # excluded from coherence, `sources_agreeing` and the family vote entirely,
    # and that participation is the stated reason. It does NOT reach direction:
    # the stackers are the combine at `ml_combine_arm_share` 1.0.
    assert 0 < _BASE_WEIGHTS["news_bear_fresh"] <= 0.08
    # Joined the Sentiment family on 2026-09-11 with its weight: a weight-0
    # method is excluded from the family vote entirely, and participating in it
    # is the stated reason for the promotion. Sentiment rather than a family of
    # its own — it is a function of the same verdict as `news`, and the family
    # layer exists so correlated methods are ONE voter (the family COUNT is
    # unchanged at 7).
    assert FAMILY_OF["news_bear_fresh"] == "Sentiment"


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


# ── news_bull_fresh: the bull counterpart (2026-09-10) ──────────────────────

def test_bull_fresh_is_a_bull_specialist():
    """Bearish and zero news abstain — a bear passthrough would only duplicate
    `news_bear_fresh` in the panel, exactly as that method abstains on bulls."""
    import numpy as np
    import pandas as pd

    from src.signals.news_bull_fresh import compute_news_bull_fresh
    idx = pd.bdate_range(end="2026-09-09", periods=40)
    flat = pd.DataFrame({"Close": 100 + np.random.RandomState(1).normal(0, 0.5, 40)}, index=idx)
    assert compute_news_bull_fresh("X", -0.5, flat) == (0.0, 0.0)
    assert compute_news_bull_fresh("X", 0.0, flat) == (0.0, 0.0)
    assert compute_news_bull_fresh("X", None, flat) == (0.0, 0.0)
    assert compute_news_bull_fresh("X", 0.5, flat)[0] > 0


def test_bull_fresh_ships_INVERTED_and_that_is_deliberate(monkeypatch):
    """It was asked for as a mirror of the bear guard and shipped inverted,
    because the mirror was measured and is the wrong direction.

    357 bull-news rows / 50 days, per-day pivot IC vs `news` alone (+0.1533):
    mirror +0.0832 (paired -0.0701, t -1.17); invert +0.2090 (paired +0.0556,
    t +1.63); and the guard BY ITSELF is significantly anti-predictive in the
    mirror direction (-0.2577, t -2.86), so the bear logic does not transfer.
    """
    import numpy as np
    import pandas as pd

    from config.settings import Settings, settings
    from src.signals.news_bull_fresh import compute_news_bull_fresh
    assert Settings.model_fields["news_bull_fresh_invert"].default is True

    idx = pd.bdate_range(end="2026-09-09", periods=40)
    df = pd.DataFrame({"Close": 100 + np.random.RandomState(2).normal(0, 0.4, 40)}, index=idx)
    df.iloc[-3:, 0] = [100.0, 103.0, 106.0]          # a strong aligned run-up

    monkeypatch.setattr(settings, "news_bull_fresh_invert", True, raising=False)
    boosted, z3 = compute_news_bull_fresh("X", 0.4, df)
    monkeypatch.setattr(settings, "news_bull_fresh_invert", False, raising=False)
    discounted, _ = compute_news_bull_fresh("X", 0.4, df)
    assert z3 > 0, "fixture no longer exercises an aligned move"
    assert boosted > discounted, (boosted, discounted)
    # and the mirror really does discount below the raw read
    assert discounted < 0.4


def test_bull_fresh_is_weighted_modestly_and_votes_once():
    """Promoted out of panel-first on 2026-09-10 (user request: "add it to live
    production with some weight, the stackers will decide on their weight").

    0.08 is the smallest weight in the book and deliberately below `news_quiet`'s
    0.10: news_quiet CLEARED the house bar (t +2.65) and this did not (+0.0556
    paired IC within bull events, t +1.63). And because it is the SAME
    information as `news` on bull rows, it joins the Sentiment family rather
    than voting a second time — the whole point of the family layer."""
    from src.signals.aggregator import _BASE_WEIGHTS
    from src.signals.agreement import METHOD_FAMILIES
    assert 0 < _BASE_WEIGHTS["news_bull_fresh"] <= _BASE_WEIGHTS["news_quiet"]
    # It is no longer the single smallest: the 2026-09-11 promotions sit BELOW
    # it precisely because their measurements cannot currently be confirmed
    # (zero labelled non-masked rows post-epoch), where news_bull_fresh's could.
    assert _BASE_WEIGHTS["news_bull_fresh"] <= 0.08
    assert _BASE_WEIGHTS["news_shock"] < _BASE_WEIGHTS["news_bull_fresh"], (
        "news_shock has NO measured IC at all — it must stay the smallest")
    fams = [f for f, ms in METHOD_FAMILIES.items() if "news_bull_fresh" in ms]
    assert fams == ["Sentiment"]


def test_bull_fresh_has_an_active_flag_so_its_weight_is_real():
    """`_normalised_weights` iterates `_raw_active`, not `_BASE_WEIGHTS`, so a
    weighted method missing from that dict scores, persists, ranks and shows in
    the dashboard while contributing weight ZERO. `news_quiet` shipped that way
    on 2026-09-09 and only the live weight log caught it."""
    from src.signals.aggregator import _BASE_WEIGHTS, _normalised_weights
    w = _normalised_weights({m: True for m in _BASE_WEIGHTS})
    assert w["news_bull_fresh"] > 0


def test_bull_fresh_is_a_stacker_feature():
    """The base weight above only reaches the per-side FAIL-SOFT: with
    `ml_combine_arm_share` at 1.0 the stackers ARE the combine. This list is
    what a retrain learns a coefficient over — and `EXIT_METHODS` is derived
    from it, so the exit timer picks it up at the same retrain."""
    from src.analysis.ml_stacker import (STACKER_RANKED_FEATURES,
                                         STACKER_SIGNED_FEATURES)
    assert "news_bull_fresh" in STACKER_SIGNED_FEATURES
    # it is a news-family score, so it rides the rank basis like the others
    assert "news_bull_fresh" in STACKER_RANKED_FEATURES


def test_bull_fresh_fails_soft():
    from src.signals.news_bull_fresh import compute_news_bull_fresh
    assert compute_news_bull_fresh("X", 0.5, object()) == (0.0, 0.0)
    assert compute_news_bull_fresh("X", 0.5, None)[0] >= 0.0     # cache path, no raise


def test_bull_fresh_is_registered_everywhere():
    from src.analysis.code_version import METHOD_SOURCES
    from src.db.schema import SIGNAL_BASE_METHOD_COLUMNS
    from src.performance.tracker import _ALL_METHODS, METHOD_CATEGORIES, METHOD_LABELS
    from src.signals.method_epochs import METHOD_SCORER_EPOCH
    assert "news_bull_fresh" in _ALL_METHODS
    assert "news_bull_fresh" in SIGNAL_BASE_METHOD_COLUMNS
    assert "news_bull_fresh" in METHOD_LABELS
    assert "news_bull_fresh" in METHOD_SOURCES
    assert sum("news_bull_fresh" in v for v in METHOD_CATEGORIES.values()) == 1
    # it consumes the sentiment verdict, so it shares the family's ONE boundary
    from src.signals.method_epochs import NEWS_FAMILY
    assert len({str(METHOD_SCORER_EPOCH[m]) for m in NEWS_FAMILY}) == 1
