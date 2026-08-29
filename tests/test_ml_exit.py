"""Tests for the ML exit-model. Phase 0 (dataset) centre of gravity is LEAKAGE —
the simulated position state must be causal (only prices up to the held day), and
the label / baseline must never be features. Phase 1 (live wiring) adds: train/
serve feature PARITY, fail-soft, the exit_signals registration, the consensus
EXCLUSION (ml_exit must not contaminate the baseline it competes with), and the
ARM COUPLING — ml_exit closes only trades stamped ``ml_arm``, and only past
the min-hold window."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.analysis import ml_exit_dataset as me


def _panel(prices, direction="BULLISH", start_day=1, fwd=None):
    """A synthetic one-ticker signals panel: rising/falling prices, constant
    method scores, a forward-return column."""
    n = len(prices)
    dates = [f"2026-06-{start_day + i:02d}" for i in range(n)]
    rows = []
    for i, (d, p) in enumerate(zip(dates, prices)):
        r = {"signal_date": d, "ticker": "AAA", "price": float(p), "direction": direction,
             "combined_score": 0.30, "fwd_ret_1d": (fwd[i] if fwd else 1.0)}
        for m in me.EXIT_METHODS:
            r[m] = 0.2
        rows.append(r)
    return pd.DataFrame(rows)


def _mock(monkeypatch, panel):
    """Point build_panel at the synthetic panel AND the benchmark grid at a dense
    daily calendar covering it (so end-date resolution doesn't hit the real SPY
    cache)."""
    from datetime import date
    from src.analysis import signal_panel
    monkeypatch.setattr(signal_panel, "build_panel", lambda **k: panel)
    grid = [date(2026, 6, d) for d in range(1, 29)]
    monkeypatch.setattr(me, "_benchmark_series", lambda *a, **k: (grid, [100.0] * len(grid)))


def test_exit_features_exclude_label_and_baseline():
    # Circularity guard: the model must never see its own label or the baseline it
    # is being compared against, nor a weight-derived aggregate as a raw feature.
    for banned in ("fwd_ret_pos_1d", "fwd_ret_pos_3d", "ex_consensus",
                   "combined_score", "confidence"):
        assert banned not in me.EXIT_FEATURE_COLUMNS


def test_state_invariants_mfe_ge_ret_ge_mae(monkeypatch):
    p = _panel([100, 102, 105, 103, 99, 104])
    _mock(monkeypatch, p)
    df = me.build_exit_dataset(horizon=1, max_hold=5, entry_stride=1)
    assert not df.empty
    # Path invariants of the excursion features.
    assert (df["ex_mfe"] >= df["ex_ret"] - 1e-9).all()
    assert (df["ex_ret"] >= df["ex_mae"] - 1e-9).all()
    assert (df["ex_giveback"] >= -1e-9).all()          # mfe - ret >= 0
    assert (df["ex_from_mae"] >= -1e-9).all()           # ret - mae >= 0
    assert (df["days_held"] >= 1).all()


def test_label_is_oriented_by_direction(monkeypatch):
    from src.analysis import signal_panel
    # A SHORT: a +2% forward stock move is a −2% oriented (held) return.
    p = _panel([100, 100, 100, 100], direction="BEARISH", fwd=[2.0, 2.0, 2.0, 2.0])
    monkeypatch.setattr(signal_panel, "build_panel", lambda **k: p)
    df = me.build_exit_dataset(horizon=1, max_hold=3)
    assert not df.empty
    # oriented held return = dir_sign(−1) × +2 = −2 (holding the short hurt).
    assert (df["fwd_ret_pos_1d"].dropna() == -2.0).all()


def test_exit_state_is_causal_future_days_do_not_move_past_rows(monkeypatch):
    base = [100, 101, 103, 102, 101, 104, 106]
    _mock(monkeypatch, _panel(base))
    d1 = me.build_exit_dataset(horizon=1, max_hold=6)

    # Append EXTREME future days; already-emitted (entry, held-day) rows must not move.
    _mock(monkeypatch, _panel(base + [500, 900]))
    d2 = me.build_exit_dataset(horizon=1, max_hold=6)

    key = ["ticker", "signal_date", "days_held"]
    m = d1.merge(d2, on=key, suffixes=("_a", "_b"))
    assert len(m) > 0
    for c in me.EXIT_STATE_FEATURES:
        if c in key:
            continue                       # join key — equal by construction
        assert np.allclose(m[f"{c}_a"], m[f"{c}_b"], equal_nan=True), \
            f"a future bar moved a past held-position feature ({c}) — leak"


def test_entries_all_widens_to_combine_sign_hypotheticals(monkeypatch):
    """entries="all" (2026-08-23): a NEUTRAL-direction name with a nonzero
    combine gets a hypothetical position oriented by the combine's SIGN and
    stamped entry_directional=False; a zero-combine neutral name still gets
    nothing; the directional population is unchanged and remains the default."""
    rows = []
    for i, d in enumerate(f"2026-06-{dd:02d}" for dd in range(1, 7)):
        for tk, direction, comb in (("DIR", "BULLISH", 0.30),
                                    ("NEU", "NEUTRAL", -0.20),
                                    ("ZER", "NEUTRAL", 0.0)):
            r = {"signal_date": d, "ticker": tk, "price": 100.0 + i,
                 "direction": direction, "combined_score": comb, "fwd_ret_1d": 1.0}
            for m in me.EXIT_METHODS:
                r[m] = 0.2
            rows.append(r)
    _mock(monkeypatch, pd.DataFrame(rows))

    d_dir = me.build_exit_dataset(horizon=1, max_hold=4)          # default population
    assert set(d_dir["ticker"]) == {"DIR"}
    assert d_dir["entry_directional"].all()

    d_all = me.build_exit_dataset(horizon=1, max_hold=4, entries="all")
    assert set(d_all["ticker"]) == {"DIR", "NEU"}                 # ZER: no orientation
    neu = d_all[d_all["ticker"] == "NEU"]
    assert (~neu["entry_directional"]).all()
    # oriented by the combine's sign (−0.20 => short): a rising tape is a
    # NEGATIVE oriented held return for the hypothetical short.
    assert (neu["ex_ret"] < 0).all()
    # the directional subpopulation is byte-identical to the default build
    sub = d_all[d_all["entry_directional"]].reset_index(drop=True)
    assert len(sub) == len(d_dir)
    assert np.allclose(sub["ex_ret"], d_dir["ex_ret"], equal_nan=True)


# ── Phase 1: live wiring ─────────────────────────────────────────────────────

def test_ml_exit_registered_as_exit_decision_method():
    # Persisted + IC-tracked like the other exit-decision overlays, and EXCLUDED
    # from the hand-built consensus it competes against (must not contaminate it).
    from src.analysis.exit_methods import EXIT_DECISION_METHODS, EXIT_METHOD_LABELS, exit_category_for
    from src.analysis.exit_conviction import _CONSENSUS_SKIP
    assert "ml_exit" in EXIT_DECISION_METHODS
    assert "ml_exit" in EXIT_METHOD_LABELS
    assert "ml_exit" in _CONSENSUS_SKIP
    assert exit_category_for("ml_exit") == "Exit decision (synthesized review + overlays)"


def test_consensus_never_reads_ml_exit():
    # Two signal methods say EXIT; a strongly-positive ml_exit must not flip the
    # consensus (it is skipped), so the anti-predictive baseline stays uncontaminated.
    from src.analysis.exit_conviction import exit_method_consensus
    c = exit_method_consensus({"tech": -0.4, "momentum": -0.5, "ml_exit": +0.9})
    assert c is not None and c < 0


def test_compute_exit_model_score_fail_soft_without_artifact(tmp_path, monkeypatch):
    # No artifact / lightgbm => None, so build_exit_scores omits ml_exit and the
    # caller keeps the hand-built exit. An invisible degradation is impossible.
    monkeypatch.setattr(me, "_EXIT_MODEL_PATH", tmp_path / "nope.pkl")
    me.reset_exit_caches()
    trade = {"ticker": "AAA", "action": "BUY", "direction": "BULLISH",
             "entry_price": 100.0, "current_price": 105.0, "entry_date": "2026-06-01",
             "signal_at_entry": {"combined_score": 0.3}, "method_scores": {"tech": 0.2}}
    assert me.compute_exit_model_score(trade, {}, None) is None


def test_build_exit_scores_adds_ml_exit_only_when_enabled(monkeypatch):
    # Gate + fail-soft wiring at the persist point: on + a score => present; the
    # persisted value is the model's signed hold-conviction verbatim.
    from config.settings import settings
    import src.analysis.exit_methods as em
    # build_exit_scores lazy-imports compute_exit_model_score from ml_exit_dataset,
    # so patch it on the SOURCE module (that is what the local import binds).
    monkeypatch.setattr(me, "compute_exit_model_score", lambda *a, **k: -0.8)
    trade = {"ticker": "AAA", "action": "BUY", "direction": "BULLISH",
             "entry_price": 100.0, "current_price": 90.0, "method_scores": {}}
    monkeypatch.setattr(settings, "enable_ml_exit_model", True)
    assert em.build_exit_scores(trade, None, {}, None).get("ml_exit") == -0.8
    monkeypatch.setattr(settings, "enable_ml_exit_model", False)
    assert "ml_exit" not in em.build_exit_scores(trade, None, {}, None)


def test_live_features_match_the_dataset_state(monkeypatch):
    # Train/serve PARITY: live_exit_features must reproduce the dataset's state
    # features for the same held position. Locks the two paths (dataset walk vs
    # live OHLCV path) to one definition of MFE/MAE/ret/combine/elapsed.
    from src.analysis import signal_panel
    from src.performance import tracker
    from src.data import cache

    prices = [100, 102, 105, 103, 99, 104]
    dates = [f"2026-06-{1 + i:02d}" for i in range(len(prices))]
    rows = []
    for d, p in zip(dates, prices):
        r = {"signal_date": d, "ticker": "AAA", "price": float(p), "direction": "BULLISH",
             "combined_score": 0.30, "fwd_ret_5d": 1.0}
        for mth in me.EXIT_METHODS:
            r[mth] = 0.2
        rows.append(r)
    panel = pd.DataFrame(rows)
    monkeypatch.setattr(signal_panel, "build_panel", lambda **k: panel)
    grid = [date(2026, 6, d) for d in range(1, 29)]
    monkeypatch.setattr(me, "_benchmark_series", lambda *a, **k: (grid, [100.0] * len(grid)))
    monkeypatch.setattr(tracker, "_method_scores_from_signal",
                        lambda tk, dr, s: {m: 0.2 for m in me.EXIT_METHODS})
    monkeypatch.setattr(tracker, "_trading_days_held", lambda ed: 2)   # match k=2 panel offset
    monkeypatch.setattr(cache, "load_ohlcv", lambda t, interval="1d": pd.DataFrame(
        {"Close": [100.0, 102.0, 105.0]},
        index=pd.to_datetime(["2026-06-01", "2026-06-02", "2026-06-03"])))

    ds = me.build_exit_dataset(horizon=5, max_hold=5, entry_stride=1)
    row = ds[ds.signal_date == "2026-06-03"].iloc[0]      # entry 06-01, held day k=2 (price 105)

    class Sig:
        combined_score = 0.30                              # consistent with the panel's held-day combine
        tape_confirmation_score = 0.2                      # the live twin of the panel's tape_score
    trade = {"ticker": "AAA", "action": "BUY", "direction": "BULLISH", "entry_price": 100.0,
             "current_price": 105.0, "entry_date": "2026-06-01",
             "signal_at_entry": {"combined_score": 0.30},
             "method_scores": {m: 0.2 for m in me.EXIT_METHODS}}
    lf = me.live_exit_features(trade, {"AAA": Sig()}, Sig())
    # ex_tape_score rides the parity check (2026-08-22): the dataset reads the
    # panel's replay-derived tape_score column; the live side must resolve the
    # SAME quantity from the TickerSignal's tape field, oriented identically.
    for c in me.EXIT_STATE_FEATURES + ["ex_tech", "ex_momentum", "ex_tape_score"]:
        a, b = float(row[c]), float(lf[c])
        assert (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-6, f"parity mismatch on {c}"


# ── the arm coupling — ml_exit closes only ml_arm trades, past min-hold ──

def _arm_open_trade(ml_arm=True, days_ago=30):
    return {"ticker": "XLE", "type": "ETF", "action": "BUY", "status": "OPEN",
            "confidence": 0.85, "ml_arm": ml_arm,
            "llm_synthesis_model": "deepseek-v4-flash",
            "entry_date": (date.today() - timedelta(days=days_ago)).isoformat(),
            "entry_datetime": (date.today() - timedelta(days=days_ago)).isoformat() + "T15:00:00+00:00",
            "entry_price": 57.0, "current_price": 57.5, "position_size_multiplier": 1.0,
            "current_price_datetime": date.today().isoformat() + "T15:00:00+00:00",
            "signal_at_entry": {"combined_score": 0.05, "confidence": 0.06}}


def _seed_and_monitor(tmp_path, monkeypatch, trade, ml_exit_score):
    """Save one trade, mock build_exit_scores to a fixed ml_exit conviction, run
    the monitor with no review (LLM logic holds), and return the reloaded trade."""
    from config.settings import settings
    from src.performance import tracker
    import src.analysis.exit_methods as em
    monkeypatch.setattr(settings, "enable_ml_exit_model", True)
    monkeypatch.setattr(settings, "enable_mechanical_exit", True)   # so _escores is built
    monkeypatch.setattr(settings, "horizon_default_window", "")     # no zombie time-stop
    monkeypatch.setattr(em, "build_exit_scores", lambda *a, **k: {"ml_exit": ml_exit_score})
    tracker._save_trades([trade])
    tracker.monitor_open_positions(signals_by_ticker={}, hold_reviews={})
    return tracker._load_trades()[0]


def test_ml_exit_closes_arm_trade(tmp_path, monkeypatch):
    # Default config (min-hold 0, measured harmful): a confident exit conviction
    # closes an arm trade immediately — no forced holding period.
    t = _seed_and_monitor(tmp_path, monkeypatch, _arm_open_trade(days_ago=1), ml_exit_score=-0.9)
    assert t["status"] == "CLOSED"
    assert t["exit_reason"] == "ml_exit"


def test_ml_exit_does_not_touch_non_arm_trade(tmp_path, monkeypatch):
    # Same strong exit conviction, but the trade was NOT opened under the arm →
    # ml_exit must not close it (the consensus is skipped for ml_exit, so nothing fires).
    t = _seed_and_monitor(tmp_path, monkeypatch, _arm_open_trade(ml_arm=False, days_ago=30),
                          ml_exit_score=-0.9)
    assert t["status"] == "OPEN"


def test_min_hold_is_off_by_default():
    # The forced holding period was MEASURED HARMFUL (exit_policy_sim: it destroyed
    # ~37% of the exit model's timing edge), so it must ship inert. A regression
    # that silently reintroduces it would re-impose a constraint with no upside.
    from config.settings import settings
    assert int(settings.ml_arm_min_hold_days) == 0


def test_min_hold_still_works_when_deliberately_set(tmp_path, monkeypatch):
    # The knob is kept (a future regime could differ) — when explicitly set it
    # still suppresses a conviction exit inside the window.
    from config.settings import settings
    monkeypatch.setattr(settings, "ml_arm_min_hold_days", 5)
    t = _seed_and_monitor(tmp_path, monkeypatch, _arm_open_trade(days_ago=0), ml_exit_score=-0.9)
    assert t["status"] == "OPEN"


def test_mechanical_exit_is_deliberately_off():
    # The mechanical-consensus exit is anti-predictive (IC -0.066 / hit 46%) AND
    # its trigger cannot reach the configured threshold (consensus range
    # [-0.26,+0.32] vs a 0.35 bar). It was inert by ACCIDENT, which protected the
    # book; this pins it OFF deliberately so a future threshold tweak cannot
    # silently switch a backwards exit rule back on.
    from config.settings import settings
    assert settings.enable_mechanical_exit is False


def test_ml_exit_survives_mechanical_exit_being_off(tmp_path, monkeypatch):
    # Coupling regression: _escores (which carries ml_exit) used to be built only
    # when exit-conviction / edge-decay / mechanical-exit were on. With
    # mechanical_exit now off by default, ml_exit must still fire on its own —
    # otherwise disabling a dead rule would silently kill the live one.
    from config.settings import settings
    from src.performance import tracker
    import src.analysis.exit_methods as em
    monkeypatch.setattr(settings, "enable_ml_exit_model", True)
    monkeypatch.setattr(settings, "enable_mechanical_exit", False)
    monkeypatch.setattr(settings, "enable_exit_conviction", False)
    monkeypatch.setattr(settings, "enable_edge_decay_exit", False)
    monkeypatch.setattr(settings, "horizon_default_window", "")
    monkeypatch.setattr(em, "build_exit_scores", lambda *a, **k: {"ml_exit": -0.9})
    tracker._save_trades([_arm_open_trade(days_ago=1)])
    tracker.monitor_open_positions(signals_by_ticker={}, hold_reviews={})
    t = tracker._load_trades()[0]
    assert t["status"] == "CLOSED" and t["exit_reason"] == "ml_exit"


def test_ml_model_owns_the_confidence_degradation_exit_for_arm_trades():
    # The confidence-DEGRADATION exit is permanently replaced by the learned exit
    # timer for arm trades (not merely delayed) — it must be suppressed even with
    # the min-hold off and the position held a long time.
    from src.performance import tracker
    old = {"ticker": "XLE", "ml_arm": True,
           "entry_date": (date.today() - timedelta(days=60)).isoformat()}
    assert tracker._arm_suppresses_exit(old, "llm_confidence_loss") is True
    # ...but a genuine thesis break / safety exit is never suppressed,
    # and ml_exit itself is free to fire.
    for live in ("llm_signal_flipped", "macro_regime_exit", "adverse_stop", "ml_exit"):
        assert tracker._arm_suppresses_exit(old, live) is False
    # A NON-arm trade keeps the classic degradation exit.
    assert tracker._arm_suppresses_exit({"ticker": "X", "ml_arm": False,
                                         "entry_date": date.today().isoformat()},
                                        "llm_confidence_loss") is False
