"""Buy/sell split combine (2026-07-22 user directive).

Every weighted method's inversion-corrected view is decomposed into a BUY
component max(0, eff) and a SELL component max(0, −eff); each side is
weight-averaged over the methods HOLDING that view only (its own camp);
combined_score = combined_buy − combined_sell (+ the additive overlays), and
direction fires on the difference clearing ``buy_sell_diff_threshold``.

Tested through build_signals with every optional layer off and exactly the
news + massive methods controllable (the same harness as
tests/test_family_agreement.py), so the camp arithmetic is legible.
"""

import pytest

from config.settings import settings
from src.signals import aggregator as agg

_OFF = [
    "enable_sentiment_velocity", "enable_technical_analysis", "enable_options_flow",
    "enable_sec_filings", "enable_insider_trades", "enable_put_call", "enable_gex",
    "enable_vwap", "enable_pattern_recognition", "enable_price_momentum",
    "enable_sector_relative_momentum", "enable_market_relative_momentum",
    "enable_money_flow", "enable_trend_strength", "enable_pead", "enable_iv_rank",
    "enable_iv_expr", "enable_cointegration", "enable_cross_sectional",
    "enable_adaptive_weights", "enable_ic_weights", "enable_winrate_method_filter",
    "enable_market_mode_switching", "enable_catalyst_timing",
    "enable_multi_timeframe_signals", "enable_extended_gap",
    "enable_trend_predictability_methods", "enable_family_agreement",
    "enable_tape_confirmation",
    "enable_high_52w", "enable_momentum_12_1", "enable_st_reversal",
    "enable_ttm_squeeze", "enable_iv_term_structure", "enable_anchored_vwap",
    "enable_residual_momentum", "enable_volume_profile",
]


@pytest.fixture
def world(monkeypatch):
    """News + massive active, nothing else; both scores injectable."""
    for flag in _OFF:
        monkeypatch.setattr(settings, flag, False)
    monkeypatch.setattr(settings, "enable_news_sentiment", True)
    monkeypatch.setattr(settings, "enable_massive_tech", True)
    monkeypatch.setattr(settings, "massive_tech_max_tickers", 0)
    monkeypatch.setattr(settings, "signal_scoring_max_workers", 2)
    monkeypatch.setattr(settings, "inverted_methods", "")
    monkeypatch.setattr(settings, "buy_sell_diff_threshold", 0.15)
    scores = {"news": 0.0, "massive": 0.0}
    monkeypatch.setattr(agg, "analyse_sentiment",
                        lambda t, a, force_engine=None: (scores["news"], "n"))
    import src.signals.massive_tech as mt
    monkeypatch.setattr(mt, "compute_massive_tech_score",
                        lambda t: scores["massive"])
    return scores


def _one(ticker="TST"):
    sigs = agg.build_signals([ticker], articles=[], snapshots=[])
    assert len(sigs) == 1
    return sigs[0]


def test_opposing_camps_each_side_is_its_sole_member(world):
    """news bullish 0.6, massive bearish 0.4 → one-member camps report their
    member's conviction verbatim, whatever the two methods' pool weights."""
    world.update(news=0.6, massive=-0.4)
    s = _one()
    assert s.combined_buy_score == pytest.approx(0.6, abs=1e-6)
    assert s.combined_sell_score == pytest.approx(0.4, abs=1e-6)
    # combined = diff (+ the small interaction adjustment on top)
    assert s.combined_score == pytest.approx(0.6 - 0.4, abs=0.1)


def test_same_camp_weight_averages(world):
    """Both bullish → sell side 0, buy side = weight-average over the camp."""
    world.update(news=0.8, massive=0.2)
    s = _one()
    w_news, w_mas = agg._BASE_WEIGHTS["news"], agg._BASE_WEIGHTS["massive"]
    expected = (w_news * 0.8 + w_mas * 0.2) / (w_news + w_mas)
    assert s.combined_sell_score == 0.0
    assert s.combined_buy_score == pytest.approx(expected, abs=1e-4)   # signal rounds to 4dp


def test_abstainer_does_not_dilute(world):
    """massive = 0 (no view) → the buy camp is news ALONE at its own conviction
    — the defining difference from the old single normalised pool, where an
    abstainer's weight stayed in the denominator and dampened the score."""
    world.update(news=0.6, massive=0.0)
    s = _one()
    assert s.combined_buy_score == pytest.approx(0.6, abs=1e-6)
    assert s.combined_sell_score == 0.0


def test_direction_band_from_setting(world, monkeypatch):
    world.update(news=0.6, massive=-0.4)                  # diff ≈ +0.2
    assert _one().direction == "BULLISH"
    monkeypatch.setattr(settings, "buy_sell_diff_threshold", 0.30)
    assert _one().direction == "NEUTRAL"                  # band widened → neutral
    monkeypatch.setattr(settings, "buy_sell_diff_threshold", 0.15)
    world.update(news=-0.6, massive=0.4)                  # mirrored
    assert _one().direction == "BEARISH"


def test_inverted_method_swaps_camp(world, monkeypatch):
    """An inverted method's RAW bullish score must land in the SELL camp (its
    effective, sign-corrected view), mirroring the combine's negated weight."""
    # _inverted_methods reads settings on every call (no cache) — the
    # monkeypatched setting is picked up directly.
    monkeypatch.setattr(settings, "inverted_methods", "news")
    world.update(news=0.6, massive=0.0)
    s = _one()
    assert s.combined_sell_score == pytest.approx(0.6, abs=1e-6)
    assert s.combined_buy_score == 0.0
    assert s.combined_score < 0


def test_all_neutral_is_all_zero(world):
    world.update(news=0.0, massive=0.0)
    s = _one()
    assert s.combined_buy_score == 0.0
    assert s.combined_sell_score == 0.0
    assert s.combined_score == pytest.approx(0.0, abs=1e-9)
    assert s.direction == "NEUTRAL"


def test_sides_bounded_zero_one(world):
    world.update(news=1.0, massive=-1.0)
    s = _one()
    assert 0.0 <= s.combined_buy_score <= 1.0
    assert 0.0 <= s.combined_sell_score <= 1.0


def test_panel_side_eval_columns_derived():
    """build_panel must derive the SIGNED eval columns: cmb_buy as-is, cmb_sell
    NEGATED (a bearish view in the standard sign convention)."""
    import pandas as pd
    from src.analysis.signal_panel import build_panel
    df = pd.DataFrame([
        {"signal_date": "2026-07-20", "generated_at": "2026-07-20T14:00:00+00:00",
         "ticker": "AAA", "combined_score": 0.2,
         "combined_buy_score": 0.5, "combined_sell_score": 0.3},
    ])
    panel = build_panel(horizons=(1,), signals_df=df)
    assert panel.loc[0, "cmb_buy"] == pytest.approx(0.5)
    assert panel.loc[0, "cmb_sell"] == pytest.approx(-0.3)


def test_compute_ic_side_filter():
    """side='buy' evaluates only positive scores; side='sell' only negatives —
    and the sell side's hit is the share that FELL."""
    import pandas as pd
    from src.analysis.signal_panel import compute_ic
    rows = []
    for i in range(30):                       # 30 bullish calls, 60% rose
        rows.append({"signal_date": f"2026-07-{(i % 10) + 1:02d}", "ticker": f"B{i}",
                     "news": 0.5, "fwd_ret_1d": 1.0 if i < 18 else -1.0})
    for i in range(30):                       # 30 bearish calls, 70% fell
        rows.append({"signal_date": f"2026-07-{(i % 10) + 1:02d}", "ticker": f"S{i}",
                     "news": -0.5, "fwd_ret_1d": -1.0 if i < 21 else 1.0})
    panel = pd.DataFrame(rows)
    buy = compute_ic(panel, horizons=(1,), min_n=5, side="buy")
    sell = compute_ic(panel, horizons=(1,), min_n=5, side="sell")
    b = buy[buy.method == "news"].iloc[0]
    s = sell[sell.method == "news"].iloc[0]
    assert b["n_1d"] == 30 and s["n_1d"] == 30
    assert b["hit_1d"] == pytest.approx(60.0)
    assert s["hit_1d"] == pytest.approx(70.0)
    assert s["simret_1d"] == pytest.approx((21 * 1.0 - 9 * 1.0) / 30)   # short P&L


def test_signal_roundtrip_persists_sides(tmp_path, monkeypatch):
    """insert_signals must write the two side columns and read them back."""
    import src.db.connection as conn_mod
    from src.db import repo
    monkeypatch.setattr(conn_mod, "DB_PATH", tmp_path / "t.db", raising=False)
    repo.set_read_only(False)
    repo.insert_signals("r1", generated_at="2026-07-22T14:00:00+00:00",
                        signal_date="2026-07-22",
                        rows=[{"ticker": "AAA", "type": "STOCK", "direction": "BULLISH",
                               "combined_score": 0.25, "confidence": 0.9,
                               "n_methods_agreeing": 3, "dominant_method": "news",
                               "price": 100.0, "universe_source": "watchlist",
                               "combined_buy_score": 0.55, "combined_sell_score": 0.30,
                               "scores": {"news": 0.55}}])
    df = repo.fetch_df("SELECT combined_buy_score, combined_sell_score FROM signals")
    assert df.combined_buy_score.iloc[0] == pytest.approx(0.55)
    assert df.combined_sell_score.iloc[0] == pytest.approx(0.30)
