"""exit_panel analysis — direction-oriented forward-return join, IC, win/ret,
the trade_reviews-backed llm_review row, and the min_n gate."""

from datetime import date

import pandas as pd
import pytest


def _make_daily(fwd_by_ticker):
    """A fake _daily_series: each ticker closes 100 today and 100·(1+pct/100)
    the next session, so its 1-day forward return is ``pct`` %."""
    def _f(tk):
        p = fwd_by_ticker.get(tk)
        if p is None:
            return [], {}
        d0, d1 = date(2026, 6, 1), date(2026, 6, 2)
        return [d0, d1], {d0: 100.0, d1: 100.0 * (1 + p / 100.0)}
    return _f


def _exit_row(ticker, method, score, direction, pos):
    return {"run_id": "r", "reviewed_at": "2026-06-01T14:00:00+00:00",
            "signal_date": "2026-06-01", "ticker": ticker, "position_id": pos,
            "entry_direction": direction, "method": method, "score": score, "price": 100.0}


def test_long_positive_ic_and_orientation(monkeypatch):
    import src.analysis.exit_panel as ep
    n = 12
    fwd = {f"T{i}": float(i + 1) for i in range(n)}          # T0..T11 rise +1..+12 %
    monkeypatch.setattr(ep, "_daily_series", _make_daily(fwd))
    monkeypatch.setattr(ep, "_intraday_series", lambda tk: [])
    rows = [_exit_row(f"T{i}", "aggregator", (i + 1) / n, "BULLISH", f"p{i}") for i in range(n)]
    df = ep.compute_exit_method_perf(min_n=5, min_per_day=1, min_days=1,
                                     exit_df=pd.DataFrame(rows), review_df=pd.DataFrame())
    agg = df[df.method == "aggregator"].iloc[0]
    assert agg["n_1d"] == n
    assert agg["ic_1d"] == pytest.approx(1.0)                # hold-conviction rank-aligned with fwd
    assert agg["win_1d"] == pytest.approx(100.0)            # all longs rose, all held → win
    assert agg["ret_1d"] == pytest.approx(6.5)             # mean(+1..+12)


def test_short_orientation_penalises_rally(monkeypatch):
    import src.analysis.exit_panel as ep
    n = 10
    monkeypatch.setattr(ep, "_daily_series", _make_daily({f"S{i}": 10.0 for i in range(n)}))
    monkeypatch.setattr(ep, "_intraday_series", lambda tk: [])
    rows = [_exit_row(f"S{i}", "aggregator", 0.5, "BEARISH", f"q{i}") for i in range(n)]
    df = ep.compute_exit_method_perf(min_n=5, min_per_day=1, min_days=1,
                                     exit_df=pd.DataFrame(rows), review_df=pd.DataFrame())
    agg = df[df.method == "aggregator"].iloc[0]
    # score +0.5 says "hold the short", but the stock ROSE +10 % → oriented fwd = −10 → loss.
    assert agg["win_1d"] == pytest.approx(0.0)
    assert agg["ret_1d"] == pytest.approx(-10.0)


def test_llm_review_from_trade_reviews(monkeypatch):
    import src.analysis.exit_panel as ep
    monkeypatch.setattr(ep, "_daily_series", _make_daily({"AAA": 10.0}))
    monkeypatch.setattr(ep, "_intraday_series", lambda tk: [])
    rev = pd.DataFrame([{"reviewed_at": "2026-06-01T14:00:00+00:00", "ticker": "AAA",
                         "position_id": "p", "action": "BUY", "confidence": 0.8,
                         "entry_action": "BUY"}])
    row = ep.compute_llm_review_perf_from_reviews(min_n=1, min_per_day=1, min_days=1, review_df=rev)
    assert row is not None and row["method"] == "llm_review"
    assert row["n_1d"] == 1
    assert row["win_1d"] == pytest.approx(100.0)            # reaffirm BUY, stock rose → correct hold
    assert row["ret_1d"] == pytest.approx(10.0)


def test_llm_review_flip_is_oriented_negative(monkeypatch):
    import src.analysis.exit_panel as ep
    monkeypatch.setattr(ep, "_daily_series", _make_daily({"AAA": 10.0}))
    monkeypatch.setattr(ep, "_intraday_series", lambda tk: [])
    # Review FLIPS to SELL on a long that then rose: the exit call was wrong.
    rev = pd.DataFrame([{"reviewed_at": "2026-06-01T14:00:00+00:00", "ticker": "AAA",
                         "position_id": "p", "action": "SELL", "confidence": 0.7,
                         "entry_action": "BUY"}])
    row = ep.compute_llm_review_perf_from_reviews(min_n=1, min_per_day=1, min_days=1, review_df=rev)
    # score −0.7 (exit), fwd +10 → signed = −fwd = −10 → wrong exit.
    assert row["win_1d"] == pytest.approx(0.0)
    assert row["ret_1d"] == pytest.approx(-10.0)


def test_review_row_replaces_panel_llm_review(monkeypatch):
    import src.analysis.exit_panel as ep
    monkeypatch.setattr(ep, "_daily_series", _make_daily({"AAA": 10.0}))
    monkeypatch.setattr(ep, "_intraday_series", lambda tk: [])
    # exit_signals carries a (wrong) llm_review flip; trade_reviews carries the reaffirm.
    ex = pd.DataFrame([_exit_row("AAA", "llm_review", -0.5, "BULLISH", "p")])
    rev = pd.DataFrame([{"reviewed_at": "2026-06-01T14:00:00+00:00", "ticker": "AAA",
                         "position_id": "p", "action": "BUY", "confidence": 0.8,
                         "entry_action": "BUY"}])
    df = ep.compute_exit_method_perf(min_n=1, min_per_day=1, min_days=1, exit_df=ex, review_df=rev)
    llm = df[df.method == "llm_review"]
    assert len(llm) == 1                                    # exactly one llm_review row
    assert llm.iloc[0]["win_1d"] == pytest.approx(100.0)   # from trade_reviews, not the panel flip


def test_min_n_gate(monkeypatch):
    import src.analysis.exit_panel as ep
    monkeypatch.setattr(ep, "_daily_series", _make_daily({"T0": 5.0}))
    monkeypatch.setattr(ep, "_intraday_series", lambda tk: [])
    ex = pd.DataFrame([_exit_row("T0", "aggregator", 0.5, "BULLISH", "p")])
    df = ep.compute_exit_method_perf(min_n=20, exit_df=ex, review_df=pd.DataFrame())
    agg = df[df.method == "aggregator"].iloc[0]
    assert agg["n_1d"] == 1
    assert pd.isna(agg["ic_1d"])                            # below min_n → unreported


def test_empty_inputs_return_empty(monkeypatch):
    import src.analysis.exit_panel as ep
    df = ep.compute_exit_method_perf(exit_df=pd.DataFrame(), review_df=pd.DataFrame())
    assert df is not None and df.empty


# ── shadow book: exit methods simulated over ALL scored tickers (signals panel) ─

def _sig_row(ticker, direction, combined, **methods):
    r = {"run_id": "r", "generated_at": "2026-06-01T14:00:00+00:00", "signal_date": "2026-06-01",
         "ticker": ticker, "direction": direction, "combined_score": combined}
    r.update(methods)
    return r


def test_shadow_orientation_long_and_short(monkeypatch):
    import src.analysis.exit_panel as ep
    monkeypatch.setattr(ep, "_daily_series", _make_daily({"AAA": 10.0, "BBB": 10.0}))
    monkeypatch.setattr(ep, "_intraday_series", lambda tk: [])
    sig = pd.DataFrame([
        _sig_row("AAA", "BULLISH", 0.5, news=0.6),    # long rises → hold-win; bullish news → hold-win
        _sig_row("BBB", "BEARISH", -0.5, news=0.6),   # short rises → hold-loss; bullish news → exit-short = win
    ])
    df = ep.compute_shadow_exit_method_perf(min_n=1, min_per_day=1, min_days=1, signals_df=sig)
    news = df[df.method == "news"].iloc[0]
    assert news["n_1d"] == 2
    assert news["win_1d"] == pytest.approx(100.0)     # both directionally correct as exit signals
    assert news["ret_1d"] == pytest.approx(10.0)
    agg = df[df.method == "aggregator"].iloc[0]        # = combined_score, oriented
    assert agg["win_1d"] == pytest.approx(50.0)        # AAA hold-long win, BBB hold-short loss
    assert agg["ret_1d"] == pytest.approx(0.0)


def test_shadow_excludes_held_only_methods(monkeypatch):
    import src.analysis.exit_panel as ep
    monkeypatch.setattr(ep, "_daily_series", _make_daily({"AAA": 5.0}))
    monkeypatch.setattr(ep, "_intraday_series", lambda tk: [])
    sig = pd.DataFrame([_sig_row("AAA", "BULLISH", 0.5, news=0.6)])
    df = ep.compute_shadow_exit_method_perf(min_n=1, min_per_day=1, min_days=1, signals_df=sig)
    methods = set(df["method"])
    assert {"aggregator", "news"} <= methods
    # horizon / llm_review / macro_regime need a real entry / opener → held-only, never here.
    assert not ({"horizon", "llm_review", "macro_regime"} & methods)


def test_shadow_category_grouping(monkeypatch):
    import src.analysis.exit_panel as ep
    from src.analysis.exit_methods import EXIT_CATEGORY_DECISION, EXIT_CATEGORY_SIGNAL
    monkeypatch.setattr(ep, "_daily_series", _make_daily({"AAA": 5.0}))
    monkeypatch.setattr(ep, "_intraday_series", lambda tk: [])
    sig = pd.DataFrame([_sig_row("AAA", "BULLISH", 0.5, news=0.6, tech=0.3)])
    df = ep.compute_shadow_exit_method_perf(min_n=1, min_per_day=1, min_days=1, signals_df=sig)
    assert df[df.method == "aggregator"].iloc[0]["category"] == EXIT_CATEGORY_DECISION
    assert df[df.method == "news"].iloc[0]["category"] == EXIT_CATEGORY_SIGNAL


def test_shadow_empty_returns_empty():
    import src.analysis.exit_panel as ep
    df = ep.compute_shadow_exit_method_perf(signals_df=pd.DataFrame())
    assert df is not None and df.empty
