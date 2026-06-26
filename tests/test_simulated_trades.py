"""Tests for the simulated single-method trades feature:
insert_simulated_trades round-trip + idempotency, the signals→simulated_trades
backfill reshape, and compute_method_perf's forward-return join / win-return
math / dedupe / min-n gate (DB-free via a fed sim_df + monkeypatched series)."""

from datetime import date

import pandas as pd
import pytest

from src.db.schema import SIGNAL_METHOD_COLUMNS


# ── insert_simulated_trades round-trip (isolated DuckDB via conftest) ──────

def _sim(ticker="AAPL", method="news", score=0.5, price=100.0):
    return {"ticker": ticker, "method": method, "score": score,
            "direction": "BUY" if score > 0 else "SELL", "entry_price": price}


def test_insert_simulated_trades_roundtrip():
    from src.db import repo
    repo.insert_simulated_trades(
        "run-1", "2026-06-09T14:00:00+00:00", "2026-06-09",
        [_sim("AAPL", "news", 0.5), _sim("MSFT", "tech", -0.3)])
    df = repo.fetch_df("SELECT * FROM simulated_trades ORDER BY ticker", read_only=False)
    assert len(df) == 2
    aapl = df[df.ticker == "AAPL"].iloc[0]
    assert aapl["method"] == "news"
    assert aapl["direction"] == "BUY"
    assert aapl["score"] == pytest.approx(0.5)
    assert aapl["signal_date"] == "2026-06-09"
    assert df[df.ticker == "MSFT"].iloc[0]["direction"] == "SELL"


def test_insert_simulated_trades_idempotent_per_run():
    from src.db import repo
    at = "2026-06-09T14:00:00+00:00"
    repo.insert_simulated_trades("run-1", at, "2026-06-09", [_sim("AAPL")])
    repo.insert_simulated_trades("run-1", at, "2026-06-09", [_sim("AAPL")])  # replace
    repo.insert_simulated_trades("run-2", at, "2026-06-09", [_sim("AAPL")])  # append
    df = repo.fetch_df("SELECT count(*) AS n FROM simulated_trades", read_only=False)
    assert int(df.iloc[0]["n"]) == 2


# ── backfill: signals (wide) → simulated_trades (long) ─────────────────────

def test_backfill_from_signals_reshapes_nonzero_methods():
    from src.db import repo
    from src.analysis.simulated_trades import backfill_from_signals

    scores = {m: 0.0 for m in SIGNAL_METHOD_COLUMNS}
    scores["news"] = 0.5
    scores["tech"] = -0.2
    repo.insert_signals(
        "run-1", "2026-06-09T14:00:00+00:00", "2026-06-09",
        [{"ticker": "AAPL", "type": "STOCK", "direction": "BULLISH",
          "combined_score": 0.4, "confidence": 0.8, "n_methods_agreeing": 2,
          "dominant_method": "news", "price": 100.0, "scores": scores}])

    backfill_from_signals()
    df = repo.fetch_df("SELECT * FROM simulated_trades", read_only=False)
    methods = set(df["method"])
    assert {"news", "tech", "combined_score"} <= methods   # non-zero scores kept
    assert "vwap" not in methods                            # zero score → no view
    assert df[df.method == "news"].iloc[0]["direction"] == "BUY"
    assert df[df.method == "tech"].iloc[0]["direction"] == "SELL"
    assert df[df.method == "news"].iloc[0]["entry_price"] == pytest.approx(100.0)


# ── compute_method_perf: forward-return join + win/return math ─────────────

def _two_day_series():
    closes = {date(2026, 6, 1): 100.0, date(2026, 6, 2): 110.0}   # +10% next session
    return [date(2026, 6, 1), date(2026, 6, 2)], closes


def _patch_series(monkeypatch):
    import src.analysis.simulated_trades as st
    dates, closes = _two_day_series()
    monkeypatch.setattr(st, "_daily_series", lambda tk: (dates, closes))
    monkeypatch.setattr(st, "_intraday_series", lambda tk: [])    # no 30m data
    return st


def test_compute_method_perf_long_is_a_win_short_is_a_loss(monkeypatch):
    st = _patch_series(monkeypatch)
    sim = pd.DataFrame([
        {"generated_at": "t1", "signal_date": "2026-06-01", "ticker": "STK",
         "method": "tech", "score": 0.5, "direction": "BUY"},
        {"generated_at": "t1", "signal_date": "2026-06-01", "ticker": "STK2",
         "method": "vwap", "score": -0.5, "direction": "SELL"},
    ])
    perf = st.compute_method_perf(sim_df=sim, min_n=1)
    tech = perf[perf.method == "tech"].iloc[0]
    assert tech["n_1d"] == 1
    assert tech["win_1d"] == pytest.approx(100.0)       # BUY into +10% → right
    assert tech["ret_1d"] == pytest.approx(10.0)
    vwap = perf[perf.method == "vwap"].iloc[0]
    assert vwap["win_1d"] == pytest.approx(0.0)         # SELL into +10% → wrong
    assert vwap["ret_1d"] == pytest.approx(-10.0)       # signed return is negated


def test_compute_method_perf_aggregates_win_rate(monkeypatch):
    st = _patch_series(monkeypatch)
    # Same +10% move: one BUY (right) + one SELL (wrong) for the SAME method →
    # win rate 50%, mean signed return 0.
    sim = pd.DataFrame([
        {"generated_at": "t1", "signal_date": "2026-06-01", "ticker": "A",
         "method": "news", "score": 0.5, "direction": "BUY"},
        {"generated_at": "t1", "signal_date": "2026-06-01", "ticker": "B",
         "method": "news", "score": -0.5, "direction": "SELL"},
    ])
    perf = st.compute_method_perf(sim_df=sim, min_n=1)
    news = perf[perf.method == "news"].iloc[0]
    assert news["n_1d"] == 2
    assert news["win_1d"] == pytest.approx(50.0)
    assert news["ret_1d"] == pytest.approx(0.0)
    assert news["views"] == 2


def test_compute_method_perf_min_n_gate(monkeypatch):
    st = _patch_series(monkeypatch)
    sim = pd.DataFrame([
        {"generated_at": "t1", "signal_date": "2026-06-01", "ticker": "STK",
         "method": "tech", "score": 0.5, "direction": "BUY"},
    ])
    perf = st.compute_method_perf(sim_df=sim, min_n=20)
    tech = perf[perf.method == "tech"].iloc[0]
    assert tech["n_1d"] == 1
    assert pd.isna(tech["win_1d"])                       # below min_n → unreported
    assert pd.isna(tech["ret_1d"])


def test_compute_method_perf_dedupes_to_last_run(monkeypatch):
    st = _patch_series(monkeypatch)
    sim = pd.DataFrame([
        {"generated_at": "2026-06-01T14:00", "signal_date": "2026-06-01",
         "ticker": "STK", "method": "tech", "score": 0.5, "direction": "BUY"},
        {"generated_at": "2026-06-01T20:00", "signal_date": "2026-06-01",
         "ticker": "STK", "method": "tech", "score": -0.5, "direction": "SELL"},  # wins
    ])
    perf = st.compute_method_perf(sim_df=sim, min_n=1)        # default dedupe="last"
    tech = perf[perf.method == "tech"].iloc[0]
    assert tech["views"] == 1                            # only the later run kept
    assert tech["win_1d"] == pytest.approx(0.0)          # SELL into +10% → wrong
    # dedupe="all" keeps both legs → win 50%
    both = st.compute_method_perf(sim_df=sim, min_n=1, dedupe="all")
    assert both[both.method == "tech"].iloc[0]["views"] == 2


def test_compute_method_perf_ic_spearman(monkeypatch):
    """ic_<h> is Spearman(score, forward return): rank-aligned → +1, inverse → −1."""
    import src.analysis.simulated_trades as st
    d1, d2 = date(2026, 6, 1), date(2026, 6, 2)
    # 5 tickers, distinct +1%..+5% next-session returns.
    series = {f"T{i}": ([d1, d2], {d1: 100.0, d2: 100.0 + (i + 1)}) for i in range(5)}
    monkeypatch.setattr(st, "_daily_series", lambda tk: series[tk])
    monkeypatch.setattr(st, "_intraday_series", lambda tk: [])

    # scores rank-aligned with returns → IC ≈ +1; all BUY into gains → win 100%
    aligned = pd.DataFrame([
        {"generated_at": "t1", "signal_date": "2026-06-01", "ticker": f"T{i}",
         "method": "tech", "score": (i + 1) / 5.0, "direction": "BUY"} for i in range(5)])
    tech = st.compute_method_perf(sim_df=aligned, min_n=5)
    tech = tech[tech.method == "tech"].iloc[0]
    assert tech["ic_1d"] == pytest.approx(1.0)
    assert tech["win_1d"] == pytest.approx(100.0)

    # scores inverse to returns → IC ≈ −1
    inverse = pd.DataFrame([
        {"generated_at": "t1", "signal_date": "2026-06-01", "ticker": f"T{i}",
         "method": "vwap", "score": (5 - i) / 5.0, "direction": "BUY"} for i in range(5)])
    vwap = st.compute_method_perf(sim_df=inverse, min_n=5)
    assert vwap[vwap.method == "vwap"].iloc[0]["ic_1d"] == pytest.approx(-1.0)


def test_compute_method_perf_horizon_steps(monkeypatch):
    """3d/1w/2w/1m return None when the cache doesn't reach that many sessions
    forward — only the 1d horizon has data in a 2-row series."""
    st = _patch_series(monkeypatch)
    sim = pd.DataFrame([
        {"generated_at": "t1", "signal_date": "2026-06-01", "ticker": "STK",
         "method": "tech", "score": 0.5, "direction": "BUY"},
    ])
    perf = st.compute_method_perf(sim_df=sim, min_n=1)
    tech = perf[perf.method == "tech"].iloc[0]
    assert tech["n_1d"] == 1
    # Intraday horizons (3h/6h) have no 30m data here; daily horizons beyond 1d
    # run off the 2-row series — all None.
    for lbl in ("3h", "6h", "3d", "1w", "2w", "1m"):
        assert tech[f"n_{lbl}"] == 0
        assert pd.isna(tech[f"win_{lbl}"])
