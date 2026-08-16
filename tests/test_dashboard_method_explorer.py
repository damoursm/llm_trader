"""The Entry Performance method explorer (2026-08-16).

Replaces the three always-visible buy/sell/all IC blocks — ~15 tables of ~30
columns rendered on every tab open — with one dropdown over methods and
families. Two properties matter and neither is visible by inspection:

  1. **Opening the tab reads nothing.** The old section called
     ``data.signal_ic()`` while the tab was being built, so the panel join (the
     most expensive query on the page) gated the first paint. The explorer must
     render its dropdown from static metadata only, and hit the panel solely on
     a selection. A regression here looks identical to working code — just slow.

  2. **Every panel column is reachable.** The grouping is assembled from three
     sources (the 7 information families, the timeframe/fundamental schema
     lists, a catch-all). A column missing from all of them is simply
     unreachable in the UI, with nothing anywhere to say so.
"""

import pandas as pd
import pytest

from dashboard import app as dash_app
from dashboard import data as dash_data
from src.db import repo

# dashboard.data flips the repo read-only process-wide at import, which is right
# for the dashboard and wrong for the rest of the suite.
repo.set_read_only(False)


# ── grouping covers the panel exactly once ───────────────────────────────────

def test_every_panel_score_column_is_reachable():
    from src.analysis.signal_panel import PANEL_SCORE_COLUMNS
    grouped = [m for members in dash_app._ic_family_groups().values() for m in members]
    assert sorted(grouped) == sorted(PANEL_SCORE_COLUMNS), (
        "a panel column is missing from (or duplicated across) the dropdown groups — "
        "it would be unreachable in the UI")
    assert len(grouped) == len(set(grouped)), "a method appears in two groups"


def test_information_families_are_the_grouping():
    """The 7 families the combine actually votes by must be the primary grouping —
    that is what makes 'how did Options do' a question about the system."""
    from src.signals.agreement import METHOD_FAMILIES
    groups = dash_app._ic_family_groups()
    for family in METHOD_FAMILIES:
        assert f"Family · {family}" in groups


def test_dropdown_offers_families_and_every_method():
    opts = dash_app._ic_dropdown_options()
    values = [o["value"] for o in opts]
    groups = dash_app._ic_family_groups()
    assert len(values) == len(set(values)), "duplicate dropdown values"
    for family in groups:
        assert f"fam:{family}" in values
    for members in groups.values():
        for m in members:
            assert f"m:{m}" in values


@pytest.mark.parametrize("value,expected", [
    (None, []),
    ("", []),
    ("m:news", ["news"]),
    ("fam:Family · Volume-Flow", ["money_flow"]),
    ("nonsense", []),
    ("fam:no such family", []),
])
def test_resolve(value, expected):
    assert dash_app._ic_resolve(value) == expected


# ── the tab must not touch the panel until something is picked ───────────────

def test_section_renders_without_reading_the_panel(monkeypatch):
    """THE load-time regression. Building the section must not call signal_ic."""
    def boom(*a, **k):
        raise AssertionError("_ic_section read the signals panel at render time")
    monkeypatch.setattr(dash_data, "signal_ic", boom)
    assert dash_app._ic_section() is not None


def test_empty_selection_renders_without_reading_the_panel(monkeypatch):
    """The guard has to sit AHEAD of the data call, not after it."""
    def boom(*a, **k):
        raise AssertionError("the placeholder path read the signals panel")
    monkeypatch.setattr(dash_data, "signal_ic", boom)
    for value in (None, "", "nonsense"):
        assert dash_app._ic_body(value, ["pv"]) is not None


def test_a_selection_does_read_the_panel(monkeypatch):
    calls = []
    monkeypatch.setattr(dash_data, "signal_ic",
                        lambda *a, **k: calls.append(1) or {"ic": pd.DataFrame(),
                                                            "panel_rows": 0, "tickers": 0})
    dash_app._ic_body("m:news", ["pv"])
    assert calls, "picking a method did not read the panel"


# ── row shape ────────────────────────────────────────────────────────────────

def _res(**frames):
    base = {"panel_rows": 10, "tickers": 3,
            "ic": pd.DataFrame(), "ic_buy": pd.DataFrame(), "ic_sell": pd.DataFrame()}
    base.update(frames)
    return base


def _df(method, views, ic):
    return pd.DataFrame([{"method": method, "views": views, "n_pv": 100,
                          "ic_pv": ic, "icstd_pv": 0.1, "icir_pv": 0.5,
                          "hit_pv": 55.0, "simret_pv": 1.25}])


def test_sides_become_rows_of_the_selected_method():
    res = _res(ic=_df("vwap", 900, 0.03),
               ic_buy=_df("vwap", 400, 0.05),
               ic_sell=_df("vwap", 500, 0.01))
    rows = dash_app._ic_rows(res, ["vwap"], [("pv", "pivot")])
    assert [r["side"] for r in rows] == ["▲ Buy (bullish calls)",
                                         "▼ Sell (bearish calls)",
                                         "● All calls"]
    assert [r["ic_pv"] for r in rows] == [0.05, 0.01, 0.03]
    assert all(r["method"] == "VWAP Distance" for r in rows)


def test_zero_view_sides_are_dropped_not_shown_blank():
    """An epoch-masked method (or a one-sided one) must not render as dashes —
    that reads as 'measured zero' when the truth is 'nothing to measure'."""
    res = _res(ic=_df("vwap", 0, None),
               ic_buy=_df("vwap", 400, 0.05),
               ic_sell=_df("vwap", 0, None))
    rows = dash_app._ic_rows(res, ["vwap"], [("pv", "pivot")])
    assert len(rows) == 1 and rows[0]["side"].startswith("▲")


def test_a_fully_masked_method_gets_the_explanatory_empty_state(monkeypatch):
    monkeypatch.setattr(dash_data, "signal_ic",
                        lambda *a, **k: _res(ic=_df("news", 0, None)))
    out = dash_app._ic_body("m:news", ["pv"])
    assert out is not None                     # renders the note, does not crash


def test_family_selection_lists_every_member():
    res = _res(ic=pd.concat([_df("put_call", 10, 0.1), _df("max_pain", 20, 0.2)]))
    rows = dash_app._ic_rows(res, ["put_call", "max_pain"], [("pv", "pivot")])
    assert len(rows) == 2
    assert {r["method"] for r in rows} == {"Put/Call Ratio", "Max Pain (GEX)"}


def test_horizon_picker_trims_columns():
    res = _res(ic=_df("vwap", 900, 0.03))
    one = dash_app._ic_table(dash_app._ic_rows(res, ["vwap"], [("pv", "pivot")]),
                             [("pv", "pivot")])
    allh = dash_app._ic_table(dash_app._ic_rows(res, ["vwap"], list(dash_app._IC_BLOCKS)),
                              list(dash_app._IC_BLOCKS))
    assert len(one.columns) < len(allh.columns)
    # Method / Side / Views + 6 metrics per horizon.
    assert len(one.columns) == 3 + 6
    assert len(allh.columns) == 3 + 6 * len(dash_app._IC_BLOCKS)


def test_pivot_leads_the_horizon_grid():
    """The pivot pseudo-horizon is the DECISION basis (2026-08-12 directive), so
    it must be the first horizon block, not buried after the fixed monitors."""
    assert dash_app._IC_BLOCKS[0] == ("pv", "pivot")
