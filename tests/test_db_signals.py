"""Tests for the signals panel: schema↔tracker drift guard, insert_signals
round-trip, forward-return join, and the IC computation."""

from datetime import date

import pandas as pd
import pytest

from config.settings import settings
from src.db.schema import (
    SIGNAL_BASE_METHOD_COLUMNS,
    SIGNAL_METHOD_COLUMNS,
    SIGNAL_TIMEFRAME_COLUMNS,
)


# ── drift guard: BASE schema columns must mirror tracker._ALL_METHODS ──────

def test_signal_base_columns_match_tracker():
    from src.performance.tracker import _ALL_METHODS
    assert tuple(SIGNAL_BASE_METHOD_COLUMNS) == tuple(_ALL_METHODS), (
        "schema.SIGNAL_BASE_METHOD_COLUMNS must mirror tracker._ALL_METHODS — "
        "the trade-attribution set. When adding a base method, add its column "
        "here (and ALTER TABLE signals ADD COLUMN <m> DOUBLE on existing DBs)."
    )


def test_signal_timeframe_columns_convention():
    """The panel-only multi-timeframe columns follow ``{method}_{tf}`` for a
    known technical method × non-daily timeframe, and compose with the base
    set without overlap."""
    from src.signals.multi_timeframe import TECHNICAL_METHODS, NON_DAILY_TIMEFRAMES
    valid = {f"{m}_{tf}" for m in TECHNICAL_METHODS for tf in NON_DAILY_TIMEFRAMES}
    from src.db.schema import SIGNAL_FUNDAMENTAL_COLUMNS
    assert set(SIGNAL_TIMEFRAME_COLUMNS) == valid
    # Panel columns = base (trade-attributed) + timeframe + fundamentals factors
    # (the latter two are panel-only diagnostics, NOT in _ALL_METHODS).
    assert tuple(SIGNAL_METHOD_COLUMNS) == (tuple(SIGNAL_BASE_METHOD_COLUMNS)
                                            + tuple(SIGNAL_TIMEFRAME_COLUMNS)
                                            + tuple(SIGNAL_FUNDAMENTAL_COLUMNS))
    assert len(set(SIGNAL_METHOD_COLUMNS)) == len(SIGNAL_METHOD_COLUMNS)   # no dupes
    assert not (set(SIGNAL_BASE_METHOD_COLUMNS) & set(SIGNAL_TIMEFRAME_COLUMNS))
    assert not (set(SIGNAL_BASE_METHOD_COLUMNS) & set(SIGNAL_FUNDAMENTAL_COLUMNS))


# ── insert_signals round-trip (temporary DuckDB file) ─────────────────────

@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "test.db"))


def _row(ticker="AAPL", news=0.5):
    scores = {m: 0.0 for m in SIGNAL_METHOD_COLUMNS}
    scores["news"] = news
    return {"ticker": ticker, "type": "STOCK", "direction": "BULLISH",
            "combined_score": 0.4, "confidence": 0.8, "n_methods_agreeing": 3,
            "dominant_method": "news", "price": 100.0, "scores": scores}


def test_insert_signals_roundtrip(tmp_db):
    from src.db import repo
    repo.insert_signals("run-1", "2026-06-09T14:00:00+00:00", "2026-06-09",
                        [_row("AAPL", 0.5), _row("MSFT", -0.3)])
    df = repo.fetch_df("SELECT * FROM signals ORDER BY ticker", read_only=False)
    assert len(df) == 2
    assert df.iloc[0]["ticker"] == "AAPL"
    assert df.iloc[0]["news"] == pytest.approx(0.5)        # projected method column
    assert df.iloc[1]["news"] == pytest.approx(-0.3)
    assert df.iloc[0]["signal_date"] == "2026-06-09"
    assert df.iloc[0]["dominant_method"] == "news"
    assert '"news": 0.5' in df.iloc[0]["scores"]           # full dict kept as JSON


def test_insert_signals_timeframe_columns(tmp_db):
    from src.db import repo
    row = _row("AAPL")
    row["scores"]["tech_30m"] = 0.42
    row["scores"]["sector_momentum_1w"] = -0.31
    repo.insert_signals("run-1", "2026-06-09T14:00:00+00:00", "2026-06-09", [row])
    df = repo.fetch_df("SELECT * FROM signals", read_only=False)
    assert df.iloc[0]["tech_30m"] == pytest.approx(0.42)         # projected tf column
    assert df.iloc[0]["sector_momentum_1w"] == pytest.approx(-0.31)


def test_insert_signals_idempotent_per_run(tmp_db):
    from src.db import repo
    at = "2026-06-09T14:00:00+00:00"
    repo.insert_signals("run-1", at, "2026-06-09", [_row("AAPL")])
    repo.insert_signals("run-1", at, "2026-06-09", [_row("AAPL")])   # replaced
    repo.insert_signals("run-2", at, "2026-06-09", [_row("AAPL")])   # appended
    df = repo.fetch_df("SELECT count(*) AS n FROM signals", read_only=False)
    assert int(df.iloc[0]["n"]) == 2


# ── build_panel: forward-return join + per-day dedupe ─────────────────────

def test_build_panel_forward_returns(monkeypatch):
    import src.analysis.signal_panel as sp
    closes = {date(2026, 6, 1): 100.0, date(2026, 6, 2): 110.0,
              date(2026, 6, 3): 99.0, date(2026, 6, 4): 132.0}
    monkeypatch.setattr(sp, "_close_series", lambda tk: closes)
    sig = pd.DataFrame([
        {"generated_at": "t1", "signal_date": "2026-06-01", "ticker": "STK", "news": 0.5},
        {"generated_at": "t2", "signal_date": "2026-06-02", "ticker": "STK", "news": -0.2},
    ])
    panel = sp.build_panel(horizons=(1, 2), signals_df=sig)
    r1 = panel[panel.signal_date == "2026-06-01"].iloc[0]
    assert r1["fwd_ret_1d"] == pytest.approx(10.0)      # 100 → 110
    assert r1["fwd_ret_2d"] == pytest.approx(-1.0)      # 100 → 99
    r2 = panel[panel.signal_date == "2026-06-02"].iloc[0]
    assert r2["fwd_ret_1d"] == pytest.approx(-10.0)     # 110 → 99
    assert r2["fwd_ret_2d"] == pytest.approx(20.0)      # 110 → 132


def test_build_panel_fwd_nan_when_history_too_short(monkeypatch):
    import src.analysis.signal_panel as sp
    monkeypatch.setattr(sp, "_close_series",
                        lambda tk: {date(2026, 6, 1): 100.0})
    sig = pd.DataFrame([{"generated_at": "t1", "signal_date": "2026-06-01",
                         "ticker": "STK", "news": 0.5}])
    panel = sp.build_panel(horizons=(1,), signals_df=sig)
    assert pd.isna(panel.iloc[0]["fwd_ret_1d"])


def test_build_panel_dedupes_to_last_run_per_day(monkeypatch):
    import src.analysis.signal_panel as sp
    monkeypatch.setattr(sp, "_close_series", lambda tk: {})
    sig = pd.DataFrame([
        {"generated_at": "2026-06-01T14:00:00", "signal_date": "2026-06-01",
         "ticker": "STK", "news": 0.1},
        {"generated_at": "2026-06-01T20:00:00", "signal_date": "2026-06-01",
         "ticker": "STK", "news": 0.9},     # later run wins
    ])
    panel = sp.build_panel(horizons=(1,), signals_df=sig, dedupe="last")
    assert len(panel) == 1
    assert panel.iloc[0]["news"] == pytest.approx(0.9)
    # dedupe="all" keeps both
    assert len(sp.build_panel(horizons=(1,), signals_df=sig, dedupe="all")) == 2


# ── compute_ic ─────────────────────────────────────────────────────────────

def test_compute_ic_perfect_and_inverse_ranking():
    import src.analysis.signal_panel as sp
    n = 30
    panel = pd.DataFrame({
        "signal_date": ["2026-06-01"] * n,
        "ticker": [f"T{i}" for i in range(n)],
        "tech": [(i + 1) / n for i in range(n)],            # same ranking as fwd
        "vwap": [-(i + 1) / n for i in range(n)],           # inverse ranking
        "news": [0.0] * n,                                  # no view → excluded
        "fwd_ret_1d": [float(i + 1) for i in range(n)],
    })
    ic = sp.compute_ic(panel, horizons=(1,), min_n=10)
    tech = ic[ic.method == "tech"].iloc[0]
    assert tech["ic_1d"] == pytest.approx(1.0)
    assert tech["hit_1d"] == pytest.approx(100.0)
    vwap = ic[ic.method == "vwap"].iloc[0]
    assert vwap["ic_1d"] == pytest.approx(-1.0)
    assert vwap["hit_1d"] == pytest.approx(0.0)
    news = ic[ic.method == "news"].iloc[0]
    assert news["views"] == 0
    assert pd.isna(news["ic_1d"])                           # zero scores excluded


def test_compute_ic_min_n_gate():
    import src.analysis.signal_panel as sp
    panel = pd.DataFrame({
        "signal_date": ["2026-06-01"] * 5,
        "ticker": [f"T{i}" for i in range(5)],
        "tech": [0.1, 0.2, 0.3, 0.4, 0.5],
        "fwd_ret_1d": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    ic = sp.compute_ic(panel, horizons=(1,), min_n=20)
    tech = ic[ic.method == "tech"].iloc[0]
    assert tech["n_1d"] == 5
    assert pd.isna(tech["ic_1d"])                           # below min_n → unreported
