"""Return / score by price and dollar-volume band (`src/analysis/price_volume_perf.py`).

The evidence behind the two-tier liquidity floors: do penny and thin names
behave differently from liquid ones? It is a read-only dashboard surface, so the
risk is a WRONG ANSWER rather than a crash — and the answer feeds a decision
about where to set the discovery gate.

The bands are half-open `[lo, hi)` and deliberately straddle the live floors
($1 / $5 price, $5M / $20M dollar volume). An off-by-one at a boundary puts the
$5.00 stock in the "$1–5" bucket and quietly moves the evidence for the very
threshold being argued about, so the boundary cases are pinned individually.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis import price_volume_perf as pv


# ── banding ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("price,expected", [
    (0.0, "<$1"), (0.99, "<$1"),
    (1.0, "$1–5"), (4.99, "$1–5"),
    (5.0, "$5–20"),                     # the trade floor lands in the HIGHER band
    (19.99, "$5–20"), (20.0, "$20–50"), (49.99, "$20–50"),
    (50.0, "$50–200"), (199.99, "$50–200"),
    (200.0, "$200+"), (1e9, "$200+"),
])
def test_price_bands_are_half_open_at_the_documented_floors(price, expected):
    assert pv._band(price, pv.PRICE_BANDS) == expected


@pytest.mark.parametrize("dvol,expected", [
    (0.0, "<$5M"), (4.99, "<$5M"),
    (5.0, "$5–20M"), (19.99, "$5–20M"),
    (20.0, "$20–100M"), (99.99, "$20–100M"),
    (100.0, "$100M–1B"), (999.99, "$100M–1B"),
    (1000.0, "$1B+"),
])
def test_dvol_bands_are_in_millions(dvol, expected):
    """The panel carries dollar volume in $M — banding a raw dollar figure here
    would drop every real name into `$1B+`."""
    assert pv._band(dvol, pv.DVOL_BANDS) == expected


@pytest.mark.parametrize("bad", [None, "", "abc", float("nan"), -1.0])
def test_unusable_values_get_no_band(bad):
    """None (not a band) so the row is DROPPED rather than silently pooled into
    the cheapest bucket — a negative or missing price is absence of data, not
    evidence about penny stocks."""
    assert pv._band(bad, pv.PRICE_BANDS) is None


def test_bands_are_contiguous_and_cover_the_line():
    for bands in (pv.PRICE_BANDS, pv.DVOL_BANDS):
        for (_, hi, _), (lo2, _, _) in zip(bands, bands[1:]):
            assert hi == lo2, f"gap or overlap at {hi} in {bands}"
        assert bands[0][0] == 0 and bands[-1][1] == float("inf")


def test_iso_date_accepts_dates_and_datetimes_and_rejects_junk():
    assert pv._iso_date("2026-07-01") == "2026-07-01"
    assert pv._iso_date("2026-07-01T14:30:00+00:00") == "2026-07-01"
    for bad in (None, "", "not-a-date", "2026-13-45"):
        assert pv._iso_date(bad) is None


# ── bucket statistics ───────────────────────────────────────────────────────

def test_every_band_is_reported_even_when_empty():
    """The dashboard renders one row per band; dropping empty ones would make
    'no penny trades yet' visually indistinguishable from 'penny trades did
    fine'."""
    rows = pv._bucket_stats(pd.DataFrame(), "x", "price_band", pv.PRICE_BANDS)
    assert [r["band"] for r in rows] == [b[2] for b in pv.PRICE_BANDS]
    assert all(r["n"] == 0 and r["mean"] is None for r in rows)


def test_bucket_mean_ignores_unparseable_values():
    df = pd.DataFrame({"price_band": ["<$1", "<$1", "<$1"],
                       "return_pct": [10.0, "junk", 20.0]})
    row = next(r for r in pv._bucket_stats(df, "return_pct", "price_band", pv.PRICE_BANDS)
               if r["band"] == "<$1")
    assert row["n"] == 2 and row["mean"] == pytest.approx(15.0)


# ── the trade-ledger view ───────────────────────────────────────────────────

def _trade(ticker="AAA", price=10.0, ret=5.0, entry="2026-07-01", **kw):
    t = {"ticker": ticker, "entry_price": price, "return_pct": ret, "entry_date": entry}
    t.update(kw)
    return t


@pytest.fixture
def _no_feature_panel(monkeypatch):
    """`build_feature_panel` reads the OHLCV cache to attach as-of dollar volume.
    Off by default here so the price-band half is tested in isolation."""
    import src.analysis.predictability as pred
    monkeypatch.setattr(pred, "build_feature_panel",
                        lambda df: df.assign(dollar_vol=None))


def test_empty_ledger_returns_the_full_empty_grid():
    out = pv.trade_return_by_price_volume([])
    assert out["n_trades"] == 0 and out["n_with_dvol"] == 0
    assert len(out["by_price"]) == len(pv.PRICE_BANDS)
    assert len(out["by_dvol"]) == len(pv.DVOL_BANDS)


def test_trades_are_bucketed_by_entry_price(_no_feature_panel):
    out = pv.trade_return_by_price_volume([
        _trade(price=0.50, ret=-20.0),
        _trade(price=7.00, ret=10.0),
        _trade(price=9.00, ret=20.0),
    ])
    assert out["n_trades"] == 3
    by = {r["band"]: r for r in out["by_price"]}
    assert by["<$1"]["n"] == 1 and by["<$1"]["mean"] == pytest.approx(-20.0)
    assert by["$5–20"]["n"] == 2 and by["$5–20"]["mean"] == pytest.approx(15.0)


@pytest.mark.parametrize("bad", [
    {"price": None}, {"price": 0.0}, {"price": "n/a"},
    {"ret": None}, {"entry": "garbage"}, {"ticker": None},
])
def test_unusable_trades_are_dropped_not_defaulted(_no_feature_panel, bad):
    assert pv.trade_return_by_price_volume([_trade(**bad)])["n_trades"] == 0


def test_dvol_view_counts_only_rows_that_have_a_dollar_volume(monkeypatch):
    """`n_with_dvol` is reported separately from `n_trades` precisely because
    the as-of volume join is partial — reporting one number would let a thin
    join read as a thin ledger."""
    import src.analysis.predictability as pred
    monkeypatch.setattr(pred, "build_feature_panel",
                        lambda df: df.assign(dollar_vol=[3.0, None]))
    out = pv.trade_return_by_price_volume([
        _trade(price=10.0, ret=5.0), _trade(price=11.0, ret=7.0, entry="2026-07-02")])
    assert out["n_trades"] == 2 and out["n_with_dvol"] == 1
    by = {r["band"]: r for r in out["by_dvol"]}
    assert by["<$5M"]["n"] == 1 and by["<$5M"]["mean"] == pytest.approx(5.0)


def test_a_failing_feature_panel_degrades_to_price_only(monkeypatch):
    """The volume join is an enrichment; losing it must not lose the price
    analysis too."""
    import src.analysis.predictability as pred
    monkeypatch.setattr(pred, "build_feature_panel",
                        lambda df: (_ for _ in ()).throw(RuntimeError("no cache")))
    out = pv.trade_return_by_price_volume([_trade(price=10.0, ret=5.0)])
    assert out["n_trades"] == 1 and out["n_with_dvol"] == 0
    assert {r["band"]: r["n"] for r in out["by_price"]}["$5–20"] == 1


# ── the signals-panel view ──────────────────────────────────────────────────

def _panel() -> pd.DataFrame:
    return pd.DataFrame({
        "signal_date": ["2026-07-01"] * 3,
        "ticker": ["AAA", "BBB", "CCC"],
        "price": [0.50, 10.0, 300.0],
        "combined_score": [0.10, 0.20, 0.30],
        "fwd_ret_pivot": [1.0, 2.0, 3.0],
        "fwd_ret_5d": [9.0, 9.0, 9.0],
    })


@pytest.fixture
def _panel_stubs(monkeypatch):
    import src.analysis.predictability as pred
    import src.analysis.signal_panel as spanel
    monkeypatch.setattr(pred, "build_feature_panel",
                        lambda df: df.assign(dollar_vol=[2.0, 50.0, 5000.0]))
    monkeypatch.setattr(spanel, "build_panel", lambda **kw: _panel())


def test_score_by_price_volume_buckets_both_axes(_panel_stubs):
    out = pv.score_by_price_volume()
    assert out["n_rows"] == 3
    by_p = {r["band"]: r for r in out["by_price"]}
    assert by_p["<$1"]["mean"] == pytest.approx(0.10)
    assert by_p["$200+"]["mean"] == pytest.approx(0.30)
    by_v = {r["band"]: r for r in out["by_dvol"]}
    assert by_v["<$5M"]["n"] == 1 and by_v["$1B+"]["n"] == 1


def test_forward_return_prefers_the_pivot_column(_panel_stubs):
    """2026-08-13 standardization: the H/L pivot target is the headline outcome
    everywhere a forward return is judged. 5d is only the fallback."""
    out = pv.score_by_price_volume()
    assert out["fwd_col"] == "fwd_ret_pivot"
    by = {r["band"]: r for r in out["fwd_by_price"]}
    assert by["<$1"]["mean"] == pytest.approx(1.0)      # pivot values, not the 9.0s


def test_falls_back_to_5d_when_the_pivot_column_is_all_null(monkeypatch):
    """A panel whose pivot labels have not settled yet must not report an empty
    forward column — it falls back rather than going blank."""
    import src.analysis.predictability as pred
    import src.analysis.signal_panel as spanel
    p = _panel()
    p["fwd_ret_pivot"] = [None, None, None]
    monkeypatch.setattr(spanel, "build_panel", lambda **kw: p)
    monkeypatch.setattr(pred, "build_feature_panel",
                        lambda df: df.assign(dollar_vol=[2.0, 50.0, 5000.0]))
    out = pv.score_by_price_volume()
    assert out["fwd_col"] == "fwd_ret_5d"


@pytest.mark.parametrize("panel", [None, pd.DataFrame()])
def test_missing_panel_returns_the_empty_grid(monkeypatch, panel):
    import src.analysis.signal_panel as spanel
    monkeypatch.setattr(spanel, "build_panel", lambda **kw: panel)
    out = pv.score_by_price_volume()
    assert out["n_rows"] == 0 and out["fwd_col"] is None
    assert len(out["by_price"]) == len(pv.PRICE_BANDS)


def test_a_raising_panel_is_fail_soft(monkeypatch):
    import src.analysis.signal_panel as spanel
    monkeypatch.setattr(spanel, "build_panel",
                        lambda **kw: (_ for _ in ()).throw(RuntimeError("db down")))
    assert pv.score_by_price_volume()["n_rows"] == 0
