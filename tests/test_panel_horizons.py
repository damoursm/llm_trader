"""Reporting horizons vs DECISION horizons (2026-08-15).

The dashboard stopped reporting the three intraday horizons (30m/3h/6h): they
are the most expensive columns on the page — computing them reads the whole
30-min OHLCV cache (~2,700 files / ~190 MB), measured ~28 s of the warm sweep
across the three simulated panels — and nothing on the page depended on them.

The trap this file exists to pin: the same curve is ALSO a live decision
surface. `edge_curve` picks each position's target holding window from it, and
`6h` is the single most-used `target_horizon` in the ledger (59 of the closed
book's `horizon_expired` exits, plus open positions). Narrowing the SHARED
default — rather than only what the dashboard asks for — would silently change
which exits fire, and `cap_horizon`'s "the LLM may shorten" rule would invert:
`HORIZON_ORDER.get("6h", <longest>)` defaults a missing label to the LONGEST, so
a SHORT-TERM call would stop shortening anything instead of erroring.
"""

import pandas as pd
import pytest

from dashboard import data as dash_data
from src.db import repo

# Importing dashboard.data flips the repo into read-only process-wide; undo it
# so the rest of the suite is unaffected (the convention in test_dashboard_*).
repo.set_read_only(False)


INTRADAY = ("30m", "3h", "6h")


# ── resolve_horizons ─────────────────────────────────────────────────────────

def test_default_is_every_horizon():
    from src.analysis.simulated_trades import HORIZONS, resolve_horizons
    assert resolve_horizons() == HORIZONS
    assert resolve_horizons(None) == HORIZONS


def test_narrowing_keeps_canonical_order_and_ignores_unknown_labels():
    from src.analysis.simulated_trades import resolve_horizons
    got = tuple(h[0] for h in resolve_horizons(["1w", "1d", "nonsense"]))
    assert got == ("1d", "1w")          # HORIZONS order, unknown label dropped
    assert resolve_horizons([]) == ()


# ── the live decision surface must keep the intraday horizons ────────────────

def test_intraday_horizons_are_still_in_the_shared_default():
    from src.analysis.simulated_trades import HORIZON_LABELS
    for lbl in INTRADAY:
        assert lbl in HORIZON_LABELS, (
            f"{lbl} left the default horizon set — edge_curve selects holding "
            "windows from it and 6h is its most-used pick; narrow the DASHBOARD's "
            "request (dashboard.data.PANEL_HORIZONS), never the default.")


def test_llm_can_still_shorten_a_horizon_to_the_short_term_bucket():
    # The silent-failure case: with "6h" gone from HORIZON_ORDER this returns
    # the unshortened "1w" instead, because .get() defaults to the LONGEST.
    from src.signals import edge_curve as ec
    assert ec.cap_horizon("1w", "SHORT-TERM") == "6h"
    assert ec.horizon_hours("6h") == 6.0


def test_edge_curve_does_not_narrow_the_horizons_it_asks_for():
    # Source-level: get_ic_matrix must not pass horizons= (a narrowed matrix
    # would remove candidate holding windows from live horizon synthesis).
    import ast
    import inspect
    from src.signals import edge_curve as ec
    tree = ast.parse(inspect.getsource(ec.get_ic_matrix))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") == "compute_method_perf"]
    assert calls, "get_ic_matrix no longer calls compute_method_perf"
    for c in calls:
        assert not any(k.arg == "horizons" for k in c.keywords)


# ── the dashboard asks for a narrower set, and only that ─────────────────────

def test_panel_horizons_are_real_labels_and_exclude_the_intraday_ones():
    from src.analysis.simulated_trades import HORIZON_LABELS
    assert dash_data.PANEL_HORIZONS, "the dashboard must report some horizon"
    for lbl in dash_data.PANEL_HORIZONS:
        # A typo here renders silently-empty columns rather than raising.
        assert lbl in HORIZON_LABELS, f"{lbl!r} is not a horizon label"
    assert not set(dash_data.PANEL_HORIZONS) & set(INTRADAY)


def test_rendered_columns_are_derived_from_the_computed_ones():
    from dashboard import app as dash_app
    assert dash_app._SIM_HORIZONS == ("pv",) + tuple(dash_data.PANEL_HORIZONS)


# ── narrowing changes which columns exist, never a number ────────────────────

def _sim_frame():
    return pd.DataFrame([
        {"generated_at": "2026-06-01T14:00:00+00:00", "signal_date": "2026-06-01",
         "ticker": "A", "method": "news", "score": +0.5, "direction": "BUY"},
        {"generated_at": "2026-06-01T14:00:00+00:00", "signal_date": "2026-06-01",
         "ticker": "B", "method": "news", "score": -0.5, "direction": "SELL"},
    ])


@pytest.fixture
def _series(monkeypatch):
    from datetime import date
    import src.analysis.simulated_trades as st
    d0, d1 = date(2026, 6, 1), date(2026, 6, 2)
    monkeypatch.setattr(st, "_daily_series", lambda tk: ([d0, d1], {d0: 100.0, d1: 110.0}))
    monkeypatch.setattr(st, "_intraday_series", lambda tk: [(1_700_000_000_000_000_000, 100.0),
                                                            (1_700_000_001_000_000_000, 110.0)])
    return st


def test_sim_perf_narrowing_drops_columns_without_moving_values(_series):
    st = _series
    sim = _sim_frame()
    full = st.compute_method_perf(sim_df=sim, min_n=1)
    slim = st.compute_method_perf(sim_df=sim, min_n=1,
                                  horizons=dash_data.PANEL_HORIZONS)
    for lbl in INTRADAY:
        assert f"n_{lbl}" in full.columns
        assert f"n_{lbl}" not in slim.columns
    assert "n_pv" in slim.columns          # the pivot basis always rides along
    keep = [c for c in slim.columns if c in full.columns]
    pd.testing.assert_frame_equal(slim[keep].reset_index(drop=True),
                                  full[keep].reset_index(drop=True))


def test_exit_panels_narrow_the_same_way(_series):
    import src.analysis.exit_panel as ep
    ex = pd.DataFrame([
        {"run_id": "r1", "reviewed_at": "2026-06-01T14:00:00+00:00",
         "signal_date": "2026-06-01", "ticker": "A", "position_id": "p1",
         "entry_direction": "BULLISH", "method": "tech", "score": -0.4, "price": 100.0},
    ])
    kw = dict(min_n=1, min_per_day=1, min_days=1, review_df=pd.DataFrame())
    full = ep.compute_exit_method_perf(exit_df=ex, **kw)
    slim = ep.compute_exit_method_perf(exit_df=ex, horizons=dash_data.PANEL_HORIZONS, **kw)
    assert "n_6h" in full.columns and "n_6h" not in slim.columns
    assert slim.iloc[0]["n_1d"] == full.iloc[0]["n_1d"]
