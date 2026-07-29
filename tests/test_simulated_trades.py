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


def test_compute_method_perf_dedupe_modes(monkeypatch):
    st = _patch_series(monkeypatch)
    sim = pd.DataFrame([
        {"generated_at": "2026-06-01T14:00", "signal_date": "2026-06-01",
         "ticker": "STK", "method": "tech", "score": 0.5, "direction": "BUY"},
        {"generated_at": "2026-06-01T20:00", "signal_date": "2026-06-01",
         "ticker": "STK", "method": "tech", "score": -0.5, "direction": "SELL"},
    ])
    # "last" — the legacy one-row-per-(day, ticker, method) panel convention,
    # PINNED by the live horizon/edge-curve IC matrix: only the later run kept.
    perf = st.compute_method_perf(sim_df=sim, min_n=1, dedupe="last")
    tech = perf[perf.method == "tech"].iloc[0]
    assert tech["views"] == 1
    assert tech["win_1d"] == pytest.approx(0.0)          # SELL into +10% → wrong
    # "events" (default) — BUY→SELL is a sign flip: BOTH are entry decisions.
    ev = st.compute_method_perf(sim_df=sim, min_n=1)
    assert ev[ev.method == "tech"].iloc[0]["views"] == 2
    # dedupe="all" keeps raw rows → also both legs here.
    both = st.compute_method_perf(sim_df=sim, min_n=1, dedupe="all")
    assert both[both.method == "tech"].iloc[0]["views"] == 2


def test_compute_method_perf_events_collapse_standing_calls(monkeypatch):
    """A call re-affirmed run after run is ONE trade (the method can't re-enter a
    position it already holds); a sign flip or a >3-day gap opens a new one.
    Events are unique moments, so session buckets partition them (All = Σ)."""
    st = _patch_series(monkeypatch)
    sim = pd.DataFrame([
        # Three consecutive re-affirmations of the same BUY call → 1 trade.
        {"generated_at": "2026-06-01T14:00:00+00:00", "signal_date": "2026-06-01",
         "ticker": "STK", "method": "tech", "score": 0.5, "direction": "BUY"},
        {"generated_at": "2026-06-01T14:30:00+00:00", "signal_date": "2026-06-01",
         "ticker": "STK", "method": "tech", "score": 0.6, "direction": "BUY"},
        {"generated_at": "2026-06-01T15:00:00+00:00", "signal_date": "2026-06-01",
         "ticker": "STK", "method": "tech", "score": 0.4, "direction": "BUY"},
        # Sign flip → a second trade (the SELL entry).
        {"generated_at": "2026-06-01T20:00:00+00:00", "signal_date": "2026-06-01",
         "ticker": "STK", "method": "tech", "score": -0.5, "direction": "SELL"},
        # Same sign again after a >3-day silence → a third trade (re-entry).
        {"generated_at": "2026-06-08T14:00:00+00:00", "signal_date": "2026-06-08",
         "ticker": "STK", "method": "tech", "score": -0.5, "direction": "SELL"},
    ])
    ev = st.extract_entry_events(sim)
    assert len(ev) == 3
    assert list(ev["generated_at"]) == ["2026-06-01T14:00:00+00:00",
                                        "2026-06-01T20:00:00+00:00",
                                        "2026-06-08T14:00:00+00:00"]


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


def test_compute_method_perf_emits_icstd_and_icir(monkeypatch):
    """icstd_<h>/icir_<h> = stdev & info-ratio of the PER-DAY IC. Three days with a
    daily IC of +1 / -1 / +1 ⇒ std=1.1547, ICIR=0.289 (same fixture as the panel test)."""
    from datetime import timedelta
    import src.analysis.simulated_trades as st

    def mkseries(base, pct):
        nxt = base + timedelta(days=1)
        return ([base, nxt], {base: 100.0, nxt: 100.0 * (1 + pct / 100.0)})

    series = {
        "A1": mkseries(date(2026, 6, 1), 1), "A2": mkseries(date(2026, 6, 1), 2),
        "A3": mkseries(date(2026, 6, 1), 3),
        "B1": mkseries(date(2026, 6, 2), 3), "B2": mkseries(date(2026, 6, 2), 2),
        "B3": mkseries(date(2026, 6, 2), 1),
        "C1": mkseries(date(2026, 6, 3), 1), "C2": mkseries(date(2026, 6, 3), 2),
        "C3": mkseries(date(2026, 6, 3), 3),
    }
    monkeypatch.setattr(st, "_daily_series", lambda tk: series[tk])
    monkeypatch.setattr(st, "_intraday_series", lambda tk: [])

    rows = []
    for day, tks in (("2026-06-01", ["A1", "A2", "A3"]),
                     ("2026-06-02", ["B1", "B2", "B3"]),
                     ("2026-06-03", ["C1", "C2", "C3"])):
        for sc, tk in zip([0.1, 0.2, 0.3], tks):
            rows.append({"generated_at": f"{day}T20:00", "signal_date": day, "ticker": tk,
                         "method": "tech", "score": sc, "direction": "BUY"})
    perf = st.compute_method_perf(sim_df=pd.DataFrame(rows), min_n=9,
                                  min_per_day=3, min_days=3)
    tech = perf[perf.method == "tech"].iloc[0]
    assert tech["n_1d"] == 9
    assert tech["icstd_1d"] == pytest.approx(1.1547, abs=1e-3)
    assert tech["icir_1d"] == pytest.approx(0.289, abs=1e-3)
    # Intraday horizons have no 30m data → IC and its confidence are unreported.
    assert pd.isna(tech["icstd_3h"]) and pd.isna(tech["icir_3h"])


def test_compute_directional_perf_emits_icstd_and_icir(monkeypatch):
    """The market-neutral path also reports the per-day IC's stdev/ICIR — the
    inversion readout. With a FLAT benchmark, market-relative return == raw return,
    so the per-day ICs are again +1 / -1 / +1 ⇒ std=1.1547, ICIR=0.289."""
    from datetime import timedelta
    import src.analysis.simulated_trades as st

    def mkseries(base, pct):
        nxt = base + timedelta(days=1)
        return ([base, nxt], {base: 100.0, nxt: 100.0 * (1 + pct / 100.0)})

    series = {
        "A1": mkseries(date(2026, 6, 1), 1), "A2": mkseries(date(2026, 6, 1), 2),
        "A3": mkseries(date(2026, 6, 1), 3),
        "B1": mkseries(date(2026, 6, 2), 3), "B2": mkseries(date(2026, 6, 2), 2),
        "B3": mkseries(date(2026, 6, 2), 1),
        "C1": mkseries(date(2026, 6, 3), 1), "C2": mkseries(date(2026, 6, 3), 2),
        "C3": mkseries(date(2026, 6, 3), 3),
        # Flat benchmark across all sessions → market-relative return == raw return.
        "SPY": ([date(2026, 6, d) for d in (1, 2, 3, 4)],
                {date(2026, 6, d): 100.0 for d in (1, 2, 3, 4)}),
    }
    monkeypatch.setattr(st, "_daily_series", lambda tk: series[tk])
    monkeypatch.setattr(st, "_intraday_series", lambda tk: [])

    rows = []
    for day, tks in (("2026-06-01", ["A1", "A2", "A3"]),
                     ("2026-06-02", ["B1", "B2", "B3"]),
                     ("2026-06-03", ["C1", "C2", "C3"])):
        for sc, tk in zip([0.1, 0.2, 0.3], tks):
            rows.append({"generated_at": f"{day}T20:00", "signal_date": day, "ticker": tk,
                         "method": "tech", "score": sc, "direction": "BUY"})
    dperf = st.compute_directional_perf(sim_df=pd.DataFrame(rows), benchmark="SPY",
                                        min_n=9, min_per_day=3, min_days=3)
    bull = dperf[(dperf.method == "tech") & (dperf.side == "bull")].iloc[0]
    assert bull["n_1d"] == 9
    assert bull["icstd_1d"] == pytest.approx(1.1547, abs=1e-3)
    assert bull["icir_1d"] == pytest.approx(0.289, abs=1e-3)
    assert int(bull["icdays_1d"]) == 3        # day count the shadow IC weighting gates on


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


# ── scorer-epoch protection (2026-07-24) ────────────────────────────────────
#
# simulated_trades is a SEPARATE materialized table from the signals panel
# (written live by the pipeline each tick, one row per method's directional
# call), feeding its own dashboard toggle explicitly labeled "unbiased ...
# thousands of observations". It has its own read choke point
# (load_sim_trades) and therefore needed its OWN scorer-epoch protection —
# signal_panel.build_panel's masking does not reach this table at all.

def test_load_sim_trades_drops_rows_from_before_a_scorer_changed(monkeypatch):
    from src.analysis import simulated_trades as st
    monkeypatch.setattr(st, "METHOD_SCORER_EPOCH", {}, raising=False)
    import src.signals.method_epochs as me
    monkeypatch.setattr(me, "METHOD_SCORER_EPOCH",
                        {"money_flow": pd.Timestamp("2026-07-24T20:01:00Z").to_pydatetime()})
    df = pd.DataFrame({
        "run_id": ["r1", "r2", "r3"],
        "generated_at": ["2026-07-24T10:00:00+00:00",   # pre-epoch → dropped
                         "2026-07-24T20:30:00+00:00",   # post-epoch → kept
                         "2026-07-24T10:00:00+00:00"],  # different method → kept
        "signal_date": ["2026-07-24"] * 3,
        "ticker": ["AAA", "AAA", "BBB"],
        "method": ["money_flow", "money_flow", "tech"],
        "score": [0.5, 0.6, 0.7],
        "direction": ["BUY", "BUY", "BUY"],
        "entry_price": [10.0, 10.0, 20.0],
    })
    out = st._drop_superseded_scorer_rows(df)
    assert len(out) == 2
    assert set(out["ticker"] + out["method"]) == {"AAAmoney_flow", "BBBtech"}
    kept_mf = out[out["method"] == "money_flow"]
    assert kept_mf.iloc[0]["generated_at"] == "2026-07-24T20:30:00+00:00"


def test_load_sim_trades_is_a_noop_for_unchanged_methods(monkeypatch):
    from src.analysis import simulated_trades as st
    import src.signals.method_epochs as me
    monkeypatch.setattr(me, "METHOD_SCORER_EPOCH", {})
    df = pd.DataFrame({
        "generated_at": ["2020-01-01T00:00:00+00:00"], "signal_date": ["2020-01-01"],
        "ticker": ["AAA"], "method": ["tech"], "score": [0.5],
        "direction": ["BUY"], "entry_price": [10.0],
    })
    out = st._drop_superseded_scorer_rows(df)
    assert len(out) == 1


def test_load_sim_trades_handles_empty_and_missing_method_column():
    from src.analysis import simulated_trades as st
    assert st._drop_superseded_scorer_rows(pd.DataFrame()).empty
    no_method = pd.DataFrame({"ticker": ["AAA"], "score": [0.5]})
    out = st._drop_superseded_scorer_rows(no_method)
    assert len(out) == 1   # fails open — nothing to filter on


# ── DuckDB event-extraction push-down (2026-07-24) ──────────────────────────
#
# extract_entry_events is a lag() window over (ticker, method) ordered by
# generated_at, so it runs in SQL and 97% of rows never reach pandas. The SQL
# and pandas definitions MUST agree — verified on live data (156,047 events
# both ways) and pinned here on a hand-built case.

def test_sql_and_pandas_event_definitions_agree(monkeypatch):
    """Same event semantics: first call, sign flip, and re-entry after a gap."""
    from src.analysis import simulated_trades as st
    rows = [
        # first call for (AAA, tech) -> EVENT
        ("2026-07-01T10:00:00+00:00", "AAA", "tech", 0.5),
        # same sign, no gap -> re-affirmation, NOT an event
        ("2026-07-01T11:00:00+00:00", "AAA", "tech", 0.6),
        # sign flip -> EVENT
        ("2026-07-01T12:00:00+00:00", "AAA", "tech", -0.4),
        # >3 day gap, same sign -> EVENT
        ("2026-07-06T12:00:00+00:00", "AAA", "tech", -0.3),
        # separate (ticker, method) partition, first call -> EVENT
        ("2026-07-01T10:00:00+00:00", "BBB", "vwap", 0.2),
    ]
    df = pd.DataFrame({
        "run_id": [f"r{i}" for i in range(len(rows))],
        "generated_at": [r[0] for r in rows],
        "signal_date": [r[0][:10] for r in rows],
        "ticker": [r[1] for r in rows],
        "method": [r[2] for r in rows],
        "score": [r[3] for r in rows],
        "direction": ["BUY" if r[3] > 0 else "SELL" for r in rows],
        "entry_price": [10.0] * len(rows),
    })
    ev = st.extract_entry_events(df)
    assert len(ev) == 4, f"expected 4 entry events, got {len(ev)}"
    got = set(zip(ev["ticker"], ev["generated_at"]))
    assert ("AAA", "2026-07-01T11:00:00+00:00") not in got, "re-affirmation is not an event"
    assert ("AAA", "2026-07-01T12:00:00+00:00") in got, "sign flip is an event"
    assert ("AAA", "2026-07-06T12:00:00+00:00") in got, "post-gap re-entry is an event"
    assert ("BBB", "2026-07-01T10:00:00+00:00") in got, "each partition's first call is an event"


def test_epoch_predicate_is_emitted_for_registered_methods(monkeypatch):
    """The SQL twin of the row filter must name the epoch'd method and its
    cutoff, so the filter still runs BEFORE the window function."""
    from datetime import datetime, timezone
    from src.analysis import simulated_trades as st
    import src.signals.method_epochs as me
    monkeypatch.setattr(me, "METHOD_SCORER_EPOCH",
                        {"money_flow": datetime(2026, 7, 24, 20, 1, tzinfo=timezone.utc)})
    sql = st._epoch_sql_predicate()
    assert "money_flow" in sql and "2026-07-24 20:01:00" in sql
    monkeypatch.setattr(me, "METHOD_SCORER_EPOCH", {})
    assert st._epoch_sql_predicate() == ""


def test_fwd_intraday_precomputed_args_match_the_plain_call():
    """The optional times/entry_ns fast-path args must not change the result."""
    from src.analysis.simulated_trades import _fwd_intraday
    series = [(1_000, 10.0), (2_000, 11.0), (3_000, 12.0), (4_000, 13.0)]
    plain = _fwd_intraday(series, "1970-01-01T00:00:00.000002Z", 1)
    fast = _fwd_intraday(series, "1970-01-01T00:00:00.000002Z", 1,
                         times=[t for t, _ in series], entry_ns=2_000)
    assert plain == fast == pytest.approx((12.0 / 11.0 - 1.0) * 100.0)


def test_intraday_series_is_vectorised_but_equivalent(tmp_path, monkeypatch):
    """The vectorised index->epoch-ns conversion must match the per-row form
    exactly (verified on 150 real tickers with 0 mismatches; pinned here)."""
    import src.data.cache as C
    from src.analysis.simulated_trades import _intraday_series
    idx = pd.to_datetime(["2026-07-01T13:30:00Z", "2026-07-01T14:00:00Z",
                          "2026-07-01T14:30:00Z"])
    df = pd.DataFrame({"Open": [1.0, 2.0, 3.0], "High": [1.0, 2.0, 3.0],
                       "Low": [1.0, 2.0, 3.0], "Close": [10.0, 0.0, 12.0],
                       "Volume": [1, 1, 1]}, index=idx)
    monkeypatch.setattr(C, "load_ohlcv", lambda tk, interval="1d": df)
    out = _intraday_series("AAA")
    # The zero close is dropped; the rest carry exact epoch-ns timestamps.
    assert out == [(int(pd.Timestamp(idx[0]).value), 10.0),
                   (int(pd.Timestamp(idx[2]).value), 12.0)]


def test_ohlcv_parse_cache_is_byte_bounded(monkeypatch, tmp_path):
    """The memo is bounded in MB, not entries — 30-min frames are ~6x larger
    than daily ones, so an entry cap would thrash on one and over-commit the
    other. A zero budget disables it."""
    import src.data.cache as C
    from config.settings import settings
    C.clear_ohlcv_parse_cache()
    monkeypatch.setattr(settings, "ohlcv_parse_cache_mb", 0)
    assert C.ohlcv_parse_cache_stats()["budget_mb"] == 0
    monkeypatch.setattr(settings, "ohlcv_parse_cache_mb", 160)
    assert C.ohlcv_parse_cache_stats()["budget_mb"] == 160.0
    C.clear_ohlcv_parse_cache()
    assert C.ohlcv_parse_cache_stats() == {"entries": 0, "mb": 0.0, "budget_mb": 160.0}
