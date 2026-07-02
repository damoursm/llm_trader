"""exit_panel analysis — ACTIVATION-event extraction (the tick an exit method
first turns against a position), direction-oriented forward-return join, IC,
win/ret, the trade_reviews-backed llm_review row, and the min_n gate.

Semantics (2026-07-02): a per-tick hold-conviction row enters the tables only
at an ACTIVATION — the tick its score first crosses into negative territory
(no previous tick / previous ≥ 0 / a new position epoch). Metrics are scored
in the direction the method called (sign(score)×oriented forward return), so
Win% > 50 / positive Ret / positive IC still mean "the method's exit calls
were vindicated by the subsequent move".
"""

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


def _exit_row(ticker, method, score, direction, pos,
              reviewed_at="2026-06-01T14:00:00+00:00"):
    return {"run_id": "r", "reviewed_at": reviewed_at,
            "signal_date": "2026-06-01", "ticker": ticker, "position_id": pos,
            "entry_direction": direction, "method": method, "score": score, "price": 100.0}


def test_activation_ic_and_orientation_on_longs(monkeypatch):
    # A GOOD exit method: the deeper its exit-conviction, the harder the long
    # fell after it fired — positive IC, 100% win, positive mean oriented return.
    import src.analysis.exit_panel as ep
    n = 12
    fwd = {f"T{i}": -float(i + 1) for i in range(n)}         # T0..T11 fall −1..−12 %
    monkeypatch.setattr(ep, "_daily_series", _make_daily(fwd))
    monkeypatch.setattr(ep, "_intraday_series", lambda tk: [])
    rows = [_exit_row(f"T{i}", "aggregator", -(i + 1) / n, "BULLISH", f"p{i}")
            for i in range(n)]
    df = ep.compute_exit_method_perf(min_n=5, min_per_day=1, min_days=1,
                                     exit_df=pd.DataFrame(rows), review_df=pd.DataFrame())
    agg = df[df.method == "aggregator"].iloc[0]
    assert agg["n_1d"] == n
    assert agg["ic_1d"] == pytest.approx(1.0)      # deeper conviction ↔ bigger drop
    assert agg["win_1d"] == pytest.approx(100.0)   # every exit call vindicated
    assert agg["ret_1d"] == pytest.approx(6.5)     # mean |drop| in the called direction


def test_short_activation_rewarded_by_rally(monkeypatch):
    import src.analysis.exit_panel as ep
    n = 10
    monkeypatch.setattr(ep, "_daily_series", _make_daily({f"S{i}": 10.0 for i in range(n)}))
    monkeypatch.setattr(ep, "_intraday_series", lambda tk: [])
    rows = [_exit_row(f"S{i}", "aggregator", -0.5, "BEARISH", f"q{i}") for i in range(n)]
    df = ep.compute_exit_method_perf(min_n=5, min_per_day=1, min_days=1,
                                     exit_df=pd.DataFrame(rows), review_df=pd.DataFrame())
    agg = df[df.method == "aggregator"].iloc[0]
    # −0.5 says "exit the short"; the stock ROSE +10 % (oriented −10 for the
    # short) → the exit call was right: signed = +10.
    assert agg["win_1d"] == pytest.approx(100.0)
    assert agg["ret_1d"] == pytest.approx(10.0)


def test_repeated_exit_calls_count_once(monkeypatch):
    # A method repeating "exit" tick after tick fired ONCE; a recovery back to
    # positive re-arms it, so the next negative crossing is a second event.
    import src.analysis.exit_panel as ep
    monkeypatch.setattr(ep, "_daily_series", _make_daily({"AAA": -5.0}))
    monkeypatch.setattr(ep, "_intraday_series", lambda tk: [])
    rows = [
        _exit_row("AAA", "aggregator", -0.5, "BULLISH", "p", "2026-06-01T14:00:00+00:00"),
        _exit_row("AAA", "aggregator", -0.6, "BULLISH", "p", "2026-06-01T14:30:00+00:00"),
        _exit_row("AAA", "aggregator", +0.2, "BULLISH", "p", "2026-06-01T15:00:00+00:00"),
        _exit_row("AAA", "aggregator", -0.4, "BULLISH", "p", "2026-06-01T15:30:00+00:00"),
    ]
    df = ep.compute_exit_method_perf(min_n=1, min_per_day=1, min_days=1,
                                     exit_df=pd.DataFrame(rows), review_df=pd.DataFrame())
    agg = df[df.method == "aggregator"].iloc[0]
    assert agg["views"] == 2                       # 14:00 activation + 15:30 re-fire


def test_llm_review_from_trade_reviews(monkeypatch):
    import src.analysis.exit_panel as ep
    monkeypatch.setattr(ep, "_daily_series", _make_daily({"AAA": -10.0}))
    monkeypatch.setattr(ep, "_intraday_series", lambda tk: [])
    # The engine FLIPS against the long (activation) and the stock then fell:
    # the exit call was right.
    rev = pd.DataFrame([{"reviewed_at": "2026-06-01T14:00:00+00:00", "ticker": "AAA",
                         "position_id": "p", "action": "SELL", "confidence": 0.8,
                         "entry_action": "BUY"}])
    row = ep.compute_llm_review_perf_from_reviews(min_n=1, min_per_day=1, min_days=1, review_df=rev)
    assert row is not None and row["method"] == "llm_review"
    assert row["n_1d"] == 1
    assert row["win_1d"] == pytest.approx(100.0)
    assert row["ret_1d"] == pytest.approx(10.0)
    # Reaffirming reviews never activate → nothing enters the data.
    reaffirm = pd.DataFrame([{"reviewed_at": "2026-06-01T14:00:00+00:00", "ticker": "AAA",
                              "position_id": "p", "action": "BUY", "confidence": 0.8,
                              "entry_action": "BUY"}])
    assert ep.compute_llm_review_perf_from_reviews(
        min_n=1, min_per_day=1, min_days=1, review_df=reaffirm) is None


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
    monkeypatch.setattr(ep, "_daily_series", _make_daily({"AAA": 10.0, "BBB": 10.0}))
    monkeypatch.setattr(ep, "_intraday_series", lambda tk: [])
    # exit_signals carries ONE llm_review activation; trade_reviews carries TWO
    # (different positions) — the history-backed review row must win.
    ex = pd.DataFrame([_exit_row("AAA", "llm_review", -0.5, "BULLISH", "p")])
    rev = pd.DataFrame([
        {"reviewed_at": "2026-06-01T14:00:00+00:00", "ticker": "AAA",
         "position_id": "p", "action": "SELL", "confidence": 0.8, "entry_action": "BUY"},
        {"reviewed_at": "2026-06-01T14:00:00+00:00", "ticker": "BBB",
         "position_id": "q", "action": "SELL", "confidence": 0.7, "entry_action": "BUY"},
    ])
    df = ep.compute_exit_method_perf(min_n=1, min_per_day=1, min_days=1, exit_df=ex, review_df=rev)
    llm = df[df.method == "llm_review"]
    assert len(llm) == 1                                    # exactly one llm_review row
    assert llm.iloc[0]["views"] == 2                        # from trade_reviews, not the panel


def test_min_n_gate(monkeypatch):
    import src.analysis.exit_panel as ep
    monkeypatch.setattr(ep, "_daily_series", _make_daily({"T0": 5.0}))
    monkeypatch.setattr(ep, "_intraday_series", lambda tk: [])
    ex = pd.DataFrame([_exit_row("T0", "aggregator", -0.5, "BULLISH", "p")])
    df = ep.compute_exit_method_perf(min_n=20, exit_df=ex, review_df=pd.DataFrame())
    agg = df[df.method == "aggregator"].iloc[0]
    assert agg["n_1d"] == 1
    assert pd.isna(agg["ic_1d"])                            # below min_n → unreported


def test_empty_inputs_return_empty(monkeypatch):
    import src.analysis.exit_panel as ep
    df = ep.compute_exit_method_perf(exit_df=pd.DataFrame(), review_df=pd.DataFrame())
    assert df is not None and df.empty


# ── shadow book: exit methods simulated over ALL scored tickers (signals panel) ─

def _sig_row(ticker, direction, combined, generated_at="2026-06-01T14:00:00+00:00",
             **methods):
    r = {"run_id": "r", "generated_at": generated_at, "signal_date": "2026-06-01",
         "ticker": ticker, "direction": direction, "combined_score": combined}
    r.update(methods)
    return r


def test_shadow_orientation_long_and_short(monkeypatch):
    import src.analysis.exit_panel as ep
    monkeypatch.setattr(ep, "_daily_series", _make_daily({"AAA": 10.0, "BBB": 10.0}))
    monkeypatch.setattr(ep, "_intraday_series", lambda tk: [])
    sig = pd.DataFrame([
        # Long AAA: aggregator −0.5 and news −0.6 both turned against it → two
        # activations; the stock then ROSE +10 → both exit calls were wrong.
        _sig_row("AAA", "BULLISH", -0.5, news=-0.6),
        # Short BBB: bullish news (+0.6 raw → conviction −0.6) turned against the
        # short → activation; the rally proved it right. The aggregator (−0.5 raw
        # → conviction +0.5) kept endorsing the short → no event.
        _sig_row("BBB", "BEARISH", -0.5, news=0.6),
    ])
    df = ep.compute_shadow_exit_method_perf(min_n=1, min_per_day=1, min_days=1, signals_df=sig)
    news = df[df.method == "news"].iloc[0]
    assert news["n_1d"] == 2
    assert news["win_1d"] == pytest.approx(50.0)      # wrong on AAA, right on BBB
    assert news["ret_1d"] == pytest.approx(0.0)
    agg = df[df.method == "aggregator"].iloc[0]
    assert agg["n_1d"] == 1                           # only the AAA activation
    assert agg["win_1d"] == pytest.approx(0.0)        # AAA kept rising → wrong exit
    assert agg["ret_1d"] == pytest.approx(-10.0)


def test_shadow_direction_flip_starts_new_epoch(monkeypatch):
    # The hypothetical position IS the ticker's aggregate direction: when that
    # flips, the old position is gone — a negative conviction against the NEW
    # position is a fresh activation even though the previous tick was negative.
    import src.analysis.exit_panel as ep
    monkeypatch.setattr(ep, "_daily_series", _make_daily({"AAA": 10.0}))
    monkeypatch.setattr(ep, "_intraday_series", lambda tk: [])
    sig = pd.DataFrame([
        _sig_row("AAA", "BULLISH", -0.5, generated_at="2026-06-01T14:00:00+00:00",
                 news=-0.6),                                  # activation vs the long
        _sig_row("AAA", "BEARISH", -0.5, generated_at="2026-06-01T15:00:00+00:00",
                 news=0.6),                                   # flip: news conviction −0.6 vs the NEW short
    ])
    df = ep.compute_shadow_exit_method_perf(min_n=1, min_per_day=1, min_days=1, signals_df=sig)
    assert df[df.method == "news"].iloc[0]["views"] == 2


def test_shadow_excludes_held_only_methods(monkeypatch):
    import src.analysis.exit_panel as ep
    monkeypatch.setattr(ep, "_daily_series", _make_daily({"AAA": 5.0}))
    monkeypatch.setattr(ep, "_intraday_series", lambda tk: [])
    sig = pd.DataFrame([_sig_row("AAA", "BULLISH", -0.5, news=-0.6)])
    df = ep.compute_shadow_exit_method_perf(min_n=1, min_per_day=1, min_days=1, signals_df=sig)
    methods = set(df["method"])
    assert {"aggregator", "news"} <= methods
    # horizon / llm_review / macro_regime / mfe / mae need a real held position
    # (entry, opener, or excursion history) → held-only, never in the shadow book.
    assert not ({"horizon", "llm_review", "macro_regime", "mfe", "mae"} & methods)


def test_shadow_category_grouping(monkeypatch):
    import src.analysis.exit_panel as ep
    from src.analysis.exit_methods import EXIT_CATEGORY_DECISION, EXIT_CATEGORY_SIGNAL
    monkeypatch.setattr(ep, "_daily_series", _make_daily({"AAA": 5.0}))
    monkeypatch.setattr(ep, "_intraday_series", lambda tk: [])
    sig = pd.DataFrame([_sig_row("AAA", "BULLISH", -0.5, news=-0.6, tech=-0.3)])
    df = ep.compute_shadow_exit_method_perf(min_n=1, min_per_day=1, min_days=1, signals_df=sig)
    assert df[df.method == "aggregator"].iloc[0]["category"] == EXIT_CATEGORY_DECISION
    assert df[df.method == "news"].iloc[0]["category"] == EXIT_CATEGORY_SIGNAL


def test_shadow_empty_returns_empty():
    import src.analysis.exit_panel as ep
    df = ep.compute_shadow_exit_method_perf(signals_df=pd.DataFrame())
    assert df is not None and df.empty
