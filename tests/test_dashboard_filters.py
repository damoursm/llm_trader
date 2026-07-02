"""Session/Direction filters across the dashboard's analysis tables (2026-07-02).

The Method Performance tab's "All scored tickers" source and the whole Exit
Performance tab used to ignore the Session and Direction toggles — the panels
carry a generation/review timestamp and a call/position side, so both filters
are computable everywhere. Covers: the vectorized session classifier, the
simulated single-method panel, the held + shadow exit-method panels, the
llm_review row, and the realized exit-reason breakdown.
"""

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest


# ── signal_panel.session_of_ts / session_filter_mask ──────────────────────────

def test_session_of_ts_boundaries():
    from src.analysis.signal_panel import session_of_ts
    s = pd.Series([
        "2026-07-01T14:30:00+00:00",   # 10:30 ET → rth
        "2026-07-01T08:00:00+00:00",   # 04:00 ET → premarket
        "2026-07-01T20:30:00+00:00",   # 16:30 ET → afterhours
        "2026-07-02T01:00:00+00:00",   # 21:00 ET (Jul 1) → overnight
        "garbage",
    ])
    assert list(session_of_ts(s)) == ["rth", "premarket", "afterhours", "overnight", ""]


def test_session_filter_mask_extended_is_both_halves():
    from src.analysis.signal_panel import session_filter_mask
    s = pd.Series(["2026-07-01T08:00:00+00:00",    # premarket
                   "2026-07-01T20:30:00+00:00",    # afterhours
                   "2026-07-01T14:30:00+00:00"])   # rth
    assert list(session_filter_mask(s, "extended")) == [True, True, False]
    assert list(session_filter_mask(s, "rth")) == [False, False, True]
    assert list(session_filter_mask(s, None)) == [True, True, True]


# ── simulated_trades.compute_method_perf(session=, direction=) ────────────────

def _patch_sim_series(monkeypatch):
    import src.analysis.simulated_trades as st
    d0, d1 = date(2026, 6, 1), date(2026, 6, 2)
    monkeypatch.setattr(st, "_daily_series",
                        lambda tk: ([d0, d1], {d0: 100.0, d1: 110.0}))
    monkeypatch.setattr(st, "_intraday_series", lambda tk: [])
    return st


def _sim_row(ticker, score, generated_at):
    return {"generated_at": generated_at, "signal_date": "2026-06-01",
            "ticker": ticker, "method": "news", "score": score,
            "direction": "BUY" if score > 0 else "SELL"}


def test_sim_perf_direction_filter(monkeypatch):
    st = _patch_sim_series(monkeypatch)
    sim = pd.DataFrame([
        _sim_row("A", +0.5, "2026-06-01T14:00:00+00:00"),   # long call — right (+10%)
        _sim_row("B", -0.5, "2026-06-01T14:00:00+00:00"),   # short call — wrong
    ])
    longs = st.compute_method_perf(sim_df=sim, min_n=1, direction="long")
    assert longs[longs.method == "news"].iloc[0]["win_1d"] == pytest.approx(100.0)
    shorts = st.compute_method_perf(sim_df=sim, min_n=1, direction="short")
    assert shorts[shorts.method == "news"].iloc[0]["win_1d"] == pytest.approx(0.0)
    both = st.compute_method_perf(sim_df=sim, min_n=1)
    assert both[both.method == "news"].iloc[0]["n_1d"] == 2


def test_sim_perf_session_filter(monkeypatch):
    st = _patch_sim_series(monkeypatch)
    sim = pd.DataFrame([
        _sim_row("A", +0.5, "2026-06-01T14:00:00+00:00"),   # 10:00 ET — rth
        _sim_row("B", +0.5, "2026-06-01T08:30:00+00:00"),   # 04:30 ET — premarket
    ])
    rth = st.compute_method_perf(sim_df=sim, min_n=1, session="rth")
    assert rth[rth.method == "news"].iloc[0]["views"] == 1
    pre = st.compute_method_perf(sim_df=sim, min_n=1, session="premarket")
    assert pre[pre.method == "news"].iloc[0]["views"] == 1
    ext = st.compute_method_perf(sim_df=sim, min_n=1, session="extended")
    assert ext[ext.method == "news"].iloc[0]["views"] == 1   # the premarket row
    on = st.compute_method_perf(sim_df=sim, min_n=1, session="overnight")
    assert on.empty


def test_sim_perf_sessions_partition_the_events(monkeypatch):
    # A BUY call decided in RTH, flipped to SELL after-hours: two entry EVENTS
    # in two different sessions. Each session bucket holds exactly its own
    # event and All sessions equals their sum — sessions partition the trades.
    st = _patch_sim_series(monkeypatch)
    sim = pd.DataFrame([
        _sim_row("A", +0.5, "2026-06-01T14:00:00+00:00"),   # rth entry (BUY)
        _sim_row("A", -0.5, "2026-06-01T21:00:00+00:00"),   # afterhours flip (SELL entry)
    ])
    rth = st.compute_method_perf(sim_df=sim, min_n=1, session="rth")
    row = rth[rth.method == "news"].iloc[0]
    assert row["views"] == 1 and row["win_1d"] == pytest.approx(100.0)  # the +0.5 rth entry
    ah = st.compute_method_perf(sim_df=sim, min_n=1, session="afterhours")
    assert ah[ah.method == "news"].iloc[0]["views"] == 1
    both = st.compute_method_perf(sim_df=sim, min_n=1)
    assert both[both.method == "news"].iloc[0]["views"] == 2            # All = Σ sessions


# ── exit_panel: held + shadow filters ─────────────────────────────────────────

def _make_daily(fwd_by_ticker):
    def _f(tk):
        p = fwd_by_ticker.get(tk)
        if p is None:
            return [], {}
        d0, d1 = date(2026, 6, 1), date(2026, 6, 2)
        return [d0, d1], {d0: 100.0, d1: 100.0 * (1 + p / 100.0)}
    return _f


def _exit_row(ticker, direction, pos, reviewed_at="2026-06-01T14:00:00+00:00"):
    # score −0.5 = an ACTIVATION (the method turned against the position) —
    # only activations enter the exit tables.
    return {"run_id": "r", "reviewed_at": reviewed_at, "signal_date": "2026-06-01",
            "ticker": ticker, "position_id": pos, "entry_direction": direction,
            "method": "aggregator", "score": -0.5, "price": 100.0}


def test_exit_perf_direction_and_session_filters(monkeypatch):
    import src.analysis.exit_panel as ep
    monkeypatch.setattr(ep, "_daily_series", _make_daily({"L": 10.0, "S": 10.0}))
    monkeypatch.setattr(ep, "_intraday_series", lambda tk: [])
    rows = [
        _exit_row("L", "BULLISH", "p1", "2026-06-01T14:00:00+00:00"),   # long, rth review
        _exit_row("S", "BEARISH", "p2", "2026-06-01T21:00:00+00:00"),   # short, afterhours review
    ]
    kw = dict(min_n=1, min_per_day=1, min_days=1, review_df=pd.DataFrame())
    longs = ep.compute_exit_method_perf(exit_df=pd.DataFrame(rows), direction="long", **kw)
    assert longs[longs.method == "aggregator"].iloc[0]["views"] == 1
    ah = ep.compute_exit_method_perf(exit_df=pd.DataFrame(rows), session="afterhours", **kw)
    assert ah[ah.method == "aggregator"].iloc[0]["views"] == 1
    none = ep.compute_exit_method_perf(exit_df=pd.DataFrame(rows), session="overnight", **kw)
    assert none.empty or "aggregator" not in set(none.get("method", []))


def test_shadow_exit_perf_filters(monkeypatch):
    import src.analysis.exit_panel as ep
    monkeypatch.setattr(ep, "_daily_series", _make_daily({"L": 10.0, "S": 10.0}))
    monkeypatch.setattr(ep, "_intraday_series", lambda tk: [])
    sig = pd.DataFrame([
        # Long L: aggregator −0.4 turned against it (activation) in RTH.
        {"signal_date": "2026-06-01", "ticker": "L", "direction": "BULLISH",
         "generated_at": "2026-06-01T14:00:00+00:00", "combined_score": -0.4},
        # Short S: aggregator +0.4 raw → conviction −0.4 vs the short (activation) pre-market.
        {"signal_date": "2026-06-01", "ticker": "S", "direction": "BEARISH",
         "generated_at": "2026-06-01T08:30:00+00:00", "combined_score": 0.4},
    ])
    kw = dict(min_n=1, min_per_day=1, min_days=1)
    longs = ep.compute_shadow_exit_method_perf(signals_df=sig, direction="long", **kw)
    assert longs[longs.method == "aggregator"].iloc[0]["views"] == 1
    pre = ep.compute_shadow_exit_method_perf(signals_df=sig, session="premarket", **kw)
    assert pre[pre.method == "aggregator"].iloc[0]["views"] == 1
    on = ep.compute_shadow_exit_method_perf(signals_df=sig, session="overnight", **kw)
    assert on.empty
    both = ep.compute_shadow_exit_method_perf(signals_df=sig, **kw)
    assert both[both.method == "aggregator"].iloc[0]["views"] == 2   # All = Σ sessions


def test_llm_review_row_honors_filters(monkeypatch):
    import src.analysis.exit_panel as ep
    monkeypatch.setattr(ep, "_daily_series", _make_daily({"L": 10.0, "S": 10.0}))
    monkeypatch.setattr(ep, "_intraday_series", lambda tk: [])
    # Both reviews FLIP against their position (activations): the long is judged
    # SELL during RTH, the short judged BUY after-hours.
    reviews = pd.DataFrame([
        {"ticker": "L", "position_id": "p1", "entry_action": "BUY", "action": "SELL",
         "confidence": 0.8, "reviewed_at": "2026-06-01T14:00:00+00:00"},
        {"ticker": "S", "position_id": "p2", "entry_action": "SELL", "action": "BUY",
         "confidence": 0.8, "reviewed_at": "2026-06-01T21:00:00+00:00"},
    ])
    kw = dict(min_n=1, min_per_day=1, min_days=1)
    row = ep.compute_llm_review_perf_from_reviews(review_df=reviews, direction="short", **kw)
    assert row is not None and row["views"] == 1
    row = ep.compute_llm_review_perf_from_reviews(review_df=reviews, session="rth", **kw)
    assert row is not None and row["views"] == 1
    row = ep.compute_llm_review_perf_from_reviews(review_df=reviews, session="overnight", **kw)
    assert row is None
    row = ep.compute_llm_review_perf_from_reviews(review_df=reviews, **kw)
    assert row is not None and row["views"] == 2                     # All = Σ sessions


# ── tracker.compute_exit_reason_perf(session=, direction=) ────────────────────

def _closed(ticker, action, exit_iso, reason, ret=1.0):
    return {"ticker": ticker, "action": action, "status": "CLOSED",
            "entry_date": "2026-06-01", "entry_datetime": "2026-06-01T14:00:00+00:00",
            "entry_price": 100.0, "exit_date": exit_iso[:10], "exit_datetime": exit_iso,
            "exit_price": 101.0, "return_pct": ret, "exit_reason": reason,
            "position_size_multiplier": 1.0}


def test_exit_reason_perf_filters(monkeypatch):
    from src.performance import tracker
    tracker._save_trades([
        _closed("A", "BUY", "2026-06-02T15:00:00+00:00", "horizon_expired"),        # rth exit, long
        _closed("B", "SELL", "2026-06-02T21:00:00+00:00", "llm_confidence_loss"),   # afterhours, short
    ])
    all_rows = {r["exit_reason"]: r["trades"] for r in tracker.compute_exit_reason_perf()}
    assert all_rows == {"horizon_expired": 1, "llm_confidence_loss": 1}
    rth = {r["exit_reason"] for r in tracker.compute_exit_reason_perf(session="rth")}
    assert rth == {"horizon_expired"}
    ah = {r["exit_reason"] for r in tracker.compute_exit_reason_perf(session="afterhours")}
    assert ah == {"llm_confidence_loss"}
    shorts = {r["exit_reason"] for r in tracker.compute_exit_reason_perf(direction="short")}
    assert shorts == {"llm_confidence_loss"}
    assert tracker.compute_exit_reason_perf(session="overnight") == []
