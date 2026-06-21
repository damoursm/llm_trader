"""Signal-panel IC engine: forward-return join + the OHLCV-refresh that unblocks it.

Root cause this guards: the IC join needs a cached close at signal_date + h. The
cache is only warmed incidentally by a running pipeline, so forward bars were
missing and every IC reported n=0. `refresh_panel_ohlcv` force-warms the cache so
the instrument actually measures.
"""

from datetime import date

import pandas as pd
import pytest

import src.analysis.signal_panel as sp


def _row(ticker, score, day="2026-01-02"):
    return {"signal_date": day, "ticker": ticker,
            "generated_at": f"{day}T20:00:00", "tech": score}


def test_build_panel_computes_forward_returns_and_ic(monkeypatch):
    series = {
        "AAA": {date(2026, 1, 2): 100.0, date(2026, 1, 3): 110.0},   # +10%
        "BBB": {date(2026, 1, 2): 100.0, date(2026, 1, 3): 102.0},   # +2%
        "CCC": {date(2026, 1, 2): 100.0, date(2026, 1, 3): 95.0},    # -5%
    }
    monkeypatch.setattr(sp, "_close_series", lambda tk: series.get(tk, {}))
    df = pd.DataFrame([_row("AAA", 0.8), _row("BBB", 0.2), _row("CCC", -0.5)])

    panel = sp.build_panel(horizons=(1,), signals_df=df)
    assert panel["fwd_ret_1d"].tolist() == pytest.approx([10.0, 2.0, -5.0])

    ic = sp.compute_ic(panel, horizons=(1,), min_n=3)
    tech = ic[ic["method"] == "tech"].iloc[0]
    assert tech["n_1d"] == 3
    assert tech["ic_1d"] == 1.0       # score ranks perfectly with forward return
    assert tech["hit_1d"] == 100.0


def test_forward_return_nan_when_cache_stops_at_signal_date(monkeypatch):
    # The exact bug: cache frozen at the signal day → no forward bar → NaN → n=0.
    monkeypatch.setattr(sp, "_close_series", lambda tk: {date(2026, 1, 2): 100.0})
    panel = sp.build_panel(horizons=(1,), signals_df=pd.DataFrame([_row("AAA", 0.5)]))
    assert pd.isna(panel["fwd_ret_1d"].iloc[0])
    ic = sp.compute_ic(panel, horizons=(1,), min_n=1)
    assert int(ic[ic["method"] == "tech"].iloc[0]["n_1d"]) == 0


def test_zero_scores_excluded_from_ic(monkeypatch):
    series = {t: {date(2026, 1, 2): 100.0, date(2026, 1, 3): 105.0} for t in ("AAA", "BBB")}
    monkeypatch.setattr(sp, "_close_series", lambda tk: series.get(tk, {}))
    df = pd.DataFrame([_row("AAA", 0.0), _row("BBB", 0.4)])   # AAA = "no view"
    ic = sp.compute_ic(sp.build_panel(horizons=(1,), signals_df=df), horizons=(1,), min_n=1)
    assert int(ic[ic["method"] == "tech"].iloc[0]["views"]) == 1   # only BBB counts


def test_compute_ic_simulated_return_winning(monkeypatch):
    # A method aligned with forward moves: a long that rose AND a short that fell
    # both "win" → simulated solo return is positive, win rate 100%.
    series = {
        "AAA": {date(2026, 1, 2): 100.0, date(2026, 1, 3): 110.0},   # +10%
        "BBB": {date(2026, 1, 2): 100.0, date(2026, 1, 3): 90.0},    # -10%
    }
    monkeypatch.setattr(sp, "_close_series", lambda tk: series.get(tk, {}))
    df = pd.DataFrame([_row("AAA", 0.8), _row("BBB", -0.5)])   # long AAA, short BBB
    ic = sp.compute_ic(sp.build_panel(horizons=(1,), signals_df=df), horizons=(1,), min_n=2)
    tech = ic[ic["method"] == "tech"].iloc[0]
    assert tech["simret_1d"] == pytest.approx(10.0)   # sign(score)×fwd: +10 and +10
    assert tech["hit_1d"] == pytest.approx(100.0)
    assert tech["category"] == sp.IC_CATEGORY_1D


def test_compute_ic_simulated_return_losing(monkeypatch):
    # Same scores, opposite forward moves: both lose → negative simulated return.
    series = {
        "AAA": {date(2026, 1, 2): 100.0, date(2026, 1, 3): 90.0},    # -10%
        "BBB": {date(2026, 1, 2): 100.0, date(2026, 1, 3): 110.0},   # +10%
    }
    monkeypatch.setattr(sp, "_close_series", lambda tk: series.get(tk, {}))
    df = pd.DataFrame([_row("AAA", 0.8), _row("BBB", -0.5)])
    ic = sp.compute_ic(sp.build_panel(horizons=(1,), signals_df=df), horizons=(1,), min_n=2)
    tech = ic[ic["method"] == "tech"].iloc[0]
    assert tech["simret_1d"] == pytest.approx(-10.0)
    assert tech["hit_1d"] == pytest.approx(0.0)


def test_category_for_mapping():
    assert sp.category_for("tech") == sp.IC_CATEGORY_1D
    assert sp.category_for("vwap") == sp.IC_CATEGORY_1D
    assert sp.category_for("tech_30m") == sp.IC_CATEGORY_30M
    assert sp.category_for("sector_momentum_30m") == sp.IC_CATEGORY_30M
    assert sp.category_for("sector_momentum_1w") == sp.IC_CATEGORY_1W
    assert sp.category_for("news") == sp.IC_CATEGORY_OTHER
    assert sp.category_for("combined_score") == sp.IC_CATEGORY_OTHER


def test_refresh_warms_each_unique_ticker_force(monkeypatch):
    import src.data.market_data as md
    calls = []
    monkeypatch.setattr(md, "get_history",
                        lambda tk, force_refresh=False: (calls.append((tk, force_refresh))
                                                         or pd.DataFrame({"Close": [1.0]})))
    n = sp.refresh_panel_ohlcv(["AAA", "BBB", "AAA", "", None])   # dedupe + drop blanks
    assert n == 2
    assert [c[0] for c in calls] == ["AAA", "BBB"]
    assert all(c[1] is True for c in calls)        # force_refresh bypasses the TTL


def test_refresh_respects_max_and_is_fail_soft(monkeypatch):
    import src.data.market_data as md
    calls = []

    def gh(tk, force_refresh=False):
        calls.append(tk)
        if tk == "B":
            raise RuntimeError("network")          # one bad ticker must not abort
        return pd.DataFrame({"Close": [1.0]})

    monkeypatch.setattr(md, "get_history", gh)
    n = sp.refresh_panel_ohlcv(["A", "B", "C", "D"], max_tickers=3)
    assert calls == ["A", "B", "C"]                # capped at 3
    assert n == 2                                  # B failed, A+C warmed
