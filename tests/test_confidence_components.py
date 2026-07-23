"""Confidence-formula component isolation (src/analysis/confidence_components.py).

Covers: the variant-value math (raw / raw×factor capped at 1.0 / live), IC +
conviction-band computation with a contrived known signal, min_n gating, the
entry-side (own combined_score direction) vs exit-side (held-interval join
against the trades ledger, oriented by the TRADE's direction) reports, and the
signals-panel schema/repo round-trip for the six persisted factor columns.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import src.analysis.confidence_components as cc
from config.settings import settings


# ── _variant_values ──────────────────────────────────────────────────────────

def test_variant_values_raw_and_live_pass_through():
    df = pd.DataFrame({
        "raw_confidence": [0.5, 0.8],
        "confidence": [0.42, 0.61],
        "coherence_factor": [1.0, 1.0],
        "movement_factor": [1.0, 1.0],
        "volume_factor": [1.0, 1.0],
        "family_conf_factor": [1.0, 1.0],
        "tape_conf_factor": [1.0, 1.0],
    })
    vals = cc._variant_values(df)
    assert list(vals["raw"]) == [0.5, 0.8]
    assert list(vals["live"]) == [0.42, 0.61]      # the STORED confidence, not re-derived


def test_variant_values_multiplies_single_factor_and_caps_at_one():
    df = pd.DataFrame({
        "raw_confidence": [1.0, 0.5],
        "confidence": [0.0, 0.0],
        "coherence_factor": [1.35, 1.35],   # top of the real span
        "movement_factor": [1.0, 1.0],
        "volume_factor": [1.0, 1.0],
        "family_conf_factor": [1.0, 1.0],
        "tape_conf_factor": [1.0, 1.0],
    })
    vals = cc._variant_values(df)
    # raw=1.0 × coherence=1.35 = 1.35 → capped at 1.0 (same ceiling the live
    # formula applies to the FULL product; a variant must not exceed the
    # [0,1] scale the conviction bands are calibrated to).
    assert vals["raw_coherence"].iloc[0] == pytest.approx(1.0)
    # raw=0.5 × 1.35 = 0.675 — under the cap, passes through untouched.
    assert vals["raw_coherence"].iloc[1] == pytest.approx(0.675)


def test_has_component_data():
    assert cc.has_component_data(None) is False
    assert cc.has_component_data(pd.DataFrame()) is False
    assert cc.has_component_data(pd.DataFrame({"ticker": ["A"]})) is False   # no column
    assert cc.has_component_data(pd.DataFrame({"raw_confidence": [None, None]})) is False
    assert cc.has_component_data(pd.DataFrame({"raw_confidence": [None, 0.5]})) is True


# ── compute_component_ic ─────────────────────────────────────────────────────

def _factor_panel(n=40):
    """raw_confidence CONSTANT across rows so a variant's variation is driven
    purely by its own factor column (isolates what the IC is actually testing).
    combined_score alternates sign so dir_sign varies too."""
    rng = np.random.RandomState(0)
    combined = np.where(np.arange(n) % 2 == 0, 0.3, -0.3)
    # coherence_factor deliberately tracks the ORIENTED outcome: high factor on
    # rows that go on to win, low factor on rows that go on to lose — a clean,
    # known positive signal to detect.
    oriented_outcome = np.where(np.arange(n) < n // 2, 1.0, -1.0)   # first half "wins"
    coherence = np.where(oriented_outcome > 0, 1.30, 0.70)
    # fwd_ret oriented by combined_score's sign reproduces oriented_outcome:
    # for combined>0 rows fwd_ret = +oriented_outcome; for combined<0 rows
    # fwd_ret = -oriented_outcome (so sign(combined) * fwd_ret == oriented_outcome).
    fwd = np.where(combined > 0, oriented_outcome, -oriented_outcome) * 2.0  # scale, arbitrary
    # movement_factor is CONSTANT — a genuinely inert factor, no relationship at all.
    movement = np.full(n, 1.05)
    return pd.DataFrame({
        "signal_date": ["2026-07-01"] * n,
        "ticker": [f"T{i}" for i in range(n)],
        "combined_score": combined,
        "raw_confidence": np.full(n, 0.6),
        "confidence": np.full(n, 0.5),
        "coherence_factor": coherence,
        "movement_factor": movement,
        "volume_factor": np.full(n, 1.0),
        "family_conf_factor": np.full(n, 1.0),
        "tape_conf_factor": np.full(n, 1.0),
        "fwd_ret_5d": fwd,
    })


def test_compute_component_ic_detects_known_signal_and_is_inert_on_constant_factor():
    df = _factor_panel()
    cs = df["combined_score"]
    dir_sign = pd.Series(np.where(cs > 0, 1.0, -1.0), index=df.index)
    ic = cc.compute_component_ic(df, dir_sign, horizons=(5,), min_n=10)
    coh = ic[ic.variant == "raw_coherence"].iloc[0]
    # coherence_factor is high exactly on rows that go on to win → raw_coherence
    # (raw is constant, so ALL of the variation is the factor) ranks with the
    # oriented outcome → strong positive IC. (win%/ret% are intentionally NOT
    # on this ungated table — see compute_component_ic's docstring; they'd be
    # identical across every variant here since they don't depend on v.)
    assert coh["ic_5d"] > 0.9
    # movement_factor is perfectly CONSTANT → raw_movement has zero variance →
    # Spearman is degenerate → None, not a crash and not a fabricated number.
    # (DataFrame column dtype coerces the row-dict's None to NaN once mixed
    # with float rows — pd.isna is the right check, matching signal_panel's
    # own compute_ic tests.)
    mov = ic[ic.variant == "raw_movement"].iloc[0]
    assert pd.isna(mov["ic_5d"])
    # "raw" alone (no factor) is also constant here (raw_confidence is constant
    # by construction) → likewise degenerate.
    raw = ic[ic.variant == "raw"].iloc[0]
    assert pd.isna(raw["ic_5d"])


def test_compute_component_ic_min_n_gate():
    df = _factor_panel(n=8)   # below default min_n
    cs = df["combined_score"]
    dir_sign = pd.Series(np.where(cs > 0, 1.0, -1.0), index=df.index)
    ic = cc.compute_component_ic(df, dir_sign, horizons=(5,), min_n=20)
    row = ic[ic.variant == "raw_coherence"].iloc[0]
    assert row["n_5d"] == 8
    assert pd.isna(row["ic_5d"]) and pd.isna(row["icir_5d"])


def test_compute_component_ic_all_seven_variants_present():
    df = _factor_panel()
    dir_sign = pd.Series(np.where(df["combined_score"] > 0, 1.0, -1.0), index=df.index)
    ic = cc.compute_component_ic(df, dir_sign, horizons=(5,), min_n=10)
    assert list(ic["variant"]) == [k for k, _l, _c in cc.VARIANTS]


# ── compute_component_bands ──────────────────────────────────────────────────

def test_compute_component_bands_routes_by_boundary():
    # raw_confidence values straddling every band edge (factor columns neutral
    # so raw == every non-"live" variant's value == raw_confidence itself).
    # Each edge value repeated exactly 5x (explicit, not a truncated cycle, so
    # every value gets identical representation).
    vals = [v for v in (0.05, 0.10, 0.34, 0.35, 0.64, 0.65, 0.99) for _ in range(5)]
    n = len(vals)   # 35
    df = pd.DataFrame({
        "combined_score": [0.3] * n,
        "raw_confidence": vals,
        "confidence": vals,
        "coherence_factor": [1.0] * n,
        "movement_factor": [1.0] * n,
        "volume_factor": [1.0] * n,
        "family_conf_factor": [1.0] * n,
        "tape_conf_factor": [1.0] * n,
        "fwd_ret_5d": [1.0] * n,   # constant win for every row — isolates BAND ROUTING
    })
    dir_sign = pd.Series(1.0, index=df.index)
    bands = cc.compute_component_bands(df, dir_sign, horizons=(5,), min_n=1)
    raw_bands = bands[bands.variant == "raw"].set_index("band")
    # 0.05 -> below Low's floor (0.10) => excluded from every band (not counted anywhere)
    # 0.10, 0.34 -> low; 0.35, 0.64 -> med; 0.65, 0.99 -> high
    assert raw_bands.loc["low", "n_5d"] == 2 * 5    # {0.10, 0.34} × 5 repeats
    assert raw_bands.loc["med", "n_5d"] == 2 * 5    # {0.35, 0.64} × 5 repeats
    assert raw_bands.loc["high", "n_5d"] == 2 * 5   # {0.65, 0.99} × 5 repeats
    # every included row wins (fwd_ret_5d=1.0, dir_sign=+1) → 100% in every band
    assert raw_bands.loc["low", "win_5d"] == pytest.approx(100.0)
    assert raw_bands.loc["high", "win_5d"] == pytest.approx(100.0)


def test_compute_component_bands_shows_rising_accuracy():
    """A well-behaved component: win rate genuinely rises with its own band
    because the factor tracks the outcome (same construction as the IC test)."""
    df = _factor_panel()
    dir_sign = pd.Series(np.where(df["combined_score"] > 0, 1.0, -1.0), index=df.index)
    bands = cc.compute_component_bands(df, dir_sign, horizons=(5,), min_n=5)
    coh = bands[bands.variant == "raw_coherence"].set_index("band")
    # coherence 0.70 on losers -> raw_coherence = 0.6*0.70 = 0.42 -> Medium band;
    # coherence 1.30 on winners -> raw_coherence = 0.6*1.30 = 0.78 -> High band.
    # (Low is empty here — this factor only ever produces those two values —
    # which is itself fine: n<min_n there just means "no evidence", not a claim.)
    assert coh.loc["med", "win_5d"] < 50
    assert coh.loc["high", "win_5d"] > 50
    assert coh.loc["high", "win_5d"] > coh.loc["med", "win_5d"]


# ── compute_entry_component_report ───────────────────────────────────────────

def test_entry_report_no_factor_columns_reports_has_factors_false(monkeypatch):
    panel = pd.DataFrame({
        "signal_date": ["2026-07-01"], "ticker": ["AAA"],
        "combined_score": [0.3], "confidence": [0.5],
    })
    import src.analysis.signal_panel as sp
    monkeypatch.setattr(sp, "_close_series", lambda tk: {})
    rep = cc.compute_entry_component_report(signals_df=panel)
    assert rep["has_factors"] is False
    assert rep["ic"].empty and rep["bands"].empty


def test_entry_report_end_to_end(monkeypatch):
    import src.analysis.signal_panel as sp
    closes = {date(2026, 7, 1): 100.0, date(2026, 7, 2): 100.0, date(2026, 7, 3): 100.0,
              date(2026, 7, 4): 100.0, date(2026, 7, 5): 100.0, date(2026, 7, 6): 110.0}
    monkeypatch.setattr(sp, "_close_series", lambda tk: closes)
    panel = pd.DataFrame([{
        "generated_at": "2026-07-01T14:00:00+00:00", "signal_date": "2026-07-01",
        "ticker": "AAA", "combined_score": 0.4, "confidence": 0.5,
        "raw_confidence": 0.8, "coherence_factor": 1.1, "movement_factor": 1.0,
        "volume_factor": 1.0, "family_conf_factor": 1.0, "tape_conf_factor": 1.0,
    }])
    rep = cc.compute_entry_component_report(horizons=(5,), min_n=1, signals_df=panel)
    assert rep["has_factors"] is True
    assert rep["panel_rows"] == 1
    ic = rep["ic"]
    assert int(ic[ic.variant == "live"].iloc[0]["n_5d"]) == 1
    # win%/return live on the banded table (variant value determines band
    # membership, unlike the ungated IC table). "live" confidence=0.5 -> Medium.
    bands = rep["bands"]
    live_med = bands[(bands.variant == "live") & (bands.band == "med")].iloc[0]
    # combined_score > 0 → dir_sign=+1; the stock rose 100 -> 110 over 5 sessions
    # -> fwd_ret_5d=+10%; oriented = +10.
    assert live_med["ret_5d"] == pytest.approx(10.0)
    assert live_med["win_5d"] == pytest.approx(100.0)


# ── restrict_to_held_intervals (the exit-side join) ──────────────────────────

def _panel_days(ticker, start, n):
    return pd.DataFrame([
        {"generated_at": f"{(start + timedelta(days=i)).isoformat()}T14:00:00+00:00",
         "signal_date": (start + timedelta(days=i)).isoformat(),
         "ticker": ticker, "combined_score": -0.3, "confidence": 0.4,   # deliberately
         "raw_confidence": 0.6, "coherence_factor": 1.0, "movement_factor": 1.0,
         "volume_factor": 1.0, "family_conf_factor": 1.0, "tape_conf_factor": 1.0,
         "fwd_ret_1d": 1.0}
        for i in range(n)
    ])


def test_restrict_to_held_intervals_excludes_entry_day_and_post_exit():
    start = date(2026, 7, 1)
    panel = _panel_days("XYZ", start, 6)   # day0..day5 = Jul1..Jul6
    trades = pd.DataFrame([{
        "ticker": "XYZ", "direction": "BUY",
        "entry_date": "2026-07-02", "exit_date": "2026-07-04", "status": "CLOSED",
    }])
    held = cc.restrict_to_held_intervals(panel, trades)
    kept_days = sorted(held["signal_date"].tolist())
    # entry_date=Jul2 itself excluded (that's an ENTRY event, not a hold re-read);
    # Jul3, Jul4 included (strictly after entry, through & including exit_date);
    # Jul1 (before entry) and Jul5/Jul6 (after exit) excluded.
    assert kept_days == ["2026-07-03", "2026-07-04"]


def test_restrict_to_held_intervals_still_open_runs_through_today():
    start = date.today() - timedelta(days=4)
    panel = _panel_days("OPEN1", start, 5)   # today-4 .. today
    trades = pd.DataFrame([{
        "ticker": "OPEN1", "direction": "SELL",
        "entry_date": start.isoformat(), "exit_date": None, "status": "OPEN",
    }])
    held = cc.restrict_to_held_intervals(panel, trades)
    # entry day excluded; every day after through today included (4 remaining rows).
    assert len(held) == 4
    assert (date.today()).isoformat() in held["signal_date"].tolist()


def test_restrict_to_held_intervals_uses_trade_direction_not_panel_direction():
    """The panel row carries combined_score < 0 (a since-reversed bearish call)
    but the trade actually held is a BUY — the exit question is about the
    POSITION, so _dir_sign must come from the trade, not the ticker's current
    (possibly since-flipped) view."""
    start = date(2026, 7, 1)
    panel = _panel_days("ZZZ", start, 3)          # combined_score is -0.3 throughout
    trades = pd.DataFrame([{
        "ticker": "ZZZ", "direction": "BUY",
        "entry_date": "2026-07-01", "exit_date": "2026-07-03", "status": "CLOSED",
    }])
    held = cc.restrict_to_held_intervals(panel, trades)
    assert not held.empty
    assert (held["_dir_sign"] == 1.0).all()       # long, from the TRADE not combined_score<0


def test_restrict_to_held_intervals_empty_when_no_trades():
    panel = _panel_days("NOPOS", date(2026, 7, 1), 3)
    assert cc.restrict_to_held_intervals(panel, pd.DataFrame()).empty
    assert cc.restrict_to_held_intervals(pd.DataFrame(), pd.DataFrame()).empty


# ── compute_exit_component_report ────────────────────────────────────────────

def test_exit_report_empty_when_no_trades(monkeypatch):
    import src.analysis.signal_panel as sp
    monkeypatch.setattr(sp, "_close_series", lambda tk: {})
    panel = pd.DataFrame([{
        "generated_at": "2026-07-01T14:00:00+00:00", "signal_date": "2026-07-01",
        "ticker": "AAA", "combined_score": 0.4, "confidence": 0.5,
        "raw_confidence": 0.8, "coherence_factor": 1.1, "movement_factor": 1.0,
        "volume_factor": 1.0, "family_conf_factor": 1.0, "tape_conf_factor": 1.0,
    }])
    rep = cc.compute_exit_component_report(signals_df=panel, trades_df=pd.DataFrame())
    assert rep["panel_rows"] == 0
    assert rep["ic"].empty and rep["bands"].empty


def test_exit_report_direction_filter(monkeypatch):
    import src.analysis.signal_panel as sp
    closes = {date(2026, 7, 1): 100.0, date(2026, 7, 2): 100.0, date(2026, 7, 3): 100.0,
              date(2026, 7, 4): 105.0, date(2026, 7, 5): 95.0, date(2026, 7, 6): 90.0}
    monkeypatch.setattr(sp, "_close_series", lambda tk: closes)
    panel = pd.concat([_panel_days("LONGX", date(2026, 7, 1), 5),
                       _panel_days("SHORTY", date(2026, 7, 1), 5)], ignore_index=True)
    for col in ("raw_confidence", "coherence_factor", "movement_factor",
               "volume_factor", "family_conf_factor", "tape_conf_factor"):
        panel[col] = panel.get(col, 0.6 if col == "raw_confidence" else 1.0)
    trades = pd.DataFrame([
        {"ticker": "LONGX", "direction": "BUY", "entry_date": "2026-07-01",
         "exit_date": "2026-07-05", "status": "CLOSED"},
        {"ticker": "SHORTY", "direction": "SELL", "entry_date": "2026-07-01",
         "exit_date": "2026-07-05", "status": "CLOSED"},
    ])
    rep_long = cc.compute_exit_component_report(horizons=(1,), min_n=1, signals_df=panel,
                                                trades_df=trades, direction="long")
    assert rep_long["panel_rows"] > 0
    rep_short = cc.compute_exit_component_report(horizons=(1,), min_n=1, signals_df=panel,
                                                 trades_df=trades, direction="short")
    assert rep_short["panel_rows"] > 0
    assert rep_long["panel_rows"] + rep_short["panel_rows"] == \
        cc.compute_exit_component_report(horizons=(1,), min_n=1, signals_df=panel,
                                        trades_df=trades)["panel_rows"]
