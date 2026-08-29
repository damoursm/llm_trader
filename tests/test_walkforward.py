"""Point-in-time cutoff and walk-forward calibration.

The value of a walk-forward is entirely in what it CANNOT see. So these tests
are mostly negative: that the cutoff reaches every data source, that it does not
leak an outcome resolved after it, that a stale cache cannot answer for the
wrong date, and that a missing calibration is skipped rather than silently
backfilled with today's weights.

The cache property is the one that would fail silently and look like success:
every layer of the weighting stack memoises on a TTL, not on the cutoff, so
without a flush every step of a walk reuses step one's answer and the whole run
collapses into a single calibration wearing a walk-forward label.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis import asof as asof_mod
from src.analysis.asof import analysis_asof, before_cutoff, current_asof


# ── the cutoff itself ─────────────────────────────────────────────────────────

def test_cutoff_is_unset_by_default_and_restored_after():
    assert current_asof() is None
    with analysis_asof("2026-07-10"):
        assert current_asof() == "2026-07-10"
    assert current_asof() is None, "cutoff leaked out of its context"


def test_nested_cutoffs_restore_the_outer_value():
    with analysis_asof("2026-07-10"):
        with analysis_asof("2026-07-05"):
            assert current_asof() == "2026-07-05"
        assert current_asof() == "2026-07-10"


def test_cutoff_survives_an_exception():
    with pytest.raises(ValueError):
        with analysis_asof("2026-07-10"):
            raise ValueError("boom")
    assert current_asof() is None


def test_malformed_cutoff_is_refused_not_degraded():
    """A bad date spliced into SQL would silently match nothing or everything —
    both look like a working walk-forward with a surprising answer."""
    with pytest.raises(ValueError):
        with analysis_asof("not-a-date"):
            pass
    assert current_asof() is None


def test_before_cutoff_is_strict_and_rejects_unknown_dates():
    with analysis_asof("2026-07-10"):
        assert before_cutoff("2026-07-09")
        assert not before_cutoff("2026-07-10"), "cutoff must be STRICT"
        assert not before_cutoff("2026-07-11")
        assert not before_cutoff(None), "an undated row cannot be proven visible"
    assert before_cutoff("2099-01-01"), "no cutoff => everything visible"


def test_sql_clause_is_empty_without_a_cutoff():
    assert asof_mod.asof_sql_clause("signal_date") == ""
    with analysis_asof("2026-07-10"):
        assert "signal_date < '2026-07-10'" in asof_mod.asof_sql_clause("signal_date")


# ── the cache trap ────────────────────────────────────────────────────────────

def test_entering_a_cutoff_flushes_the_weighting_caches():
    """Without this the second step of a walk reuses the first step's weights
    and the run silently becomes one calibration, not a walk-forward."""
    import src.signals.aggregator as agg

    agg._IC_WEIGHT_CACHE["sentinel"] = {"ts": 1e18, "data": {"x": 1.0}}
    agg._WINRATE_FILTER_CACHE["sentinel"] = {"ts": 1e18, "data": {"x"}}
    with analysis_asof("2026-07-10"):
        assert "sentinel" not in agg._IC_WEIGHT_CACHE
        assert "sentinel" not in agg._WINRATE_FILTER_CACHE


def test_leaving_a_cutoff_flushes_too():
    """Point-in-time calibrations computed inside the context must not leak into
    live decisions afterwards."""
    import src.signals.aggregator as agg
    with analysis_asof("2026-07-10"):
        agg._IC_WEIGHT_CACHE["stale"] = {"ts": 1e18, "data": {"x": 1.0}}
    assert "stale" not in agg._IC_WEIGHT_CACHE


def test_reset_is_fail_soft_when_a_hook_is_missing(monkeypatch):
    import src.signals.aggregator as agg

    def boom():
        raise RuntimeError("no hook")
    monkeypatch.setattr(agg, "reset_ic_weight_cache", boom)
    asof_mod.reset_all_calibration_caches()      # must not raise


def test_every_registered_module_actually_exposes_a_reset_hook():
    """The fail-soft above has a hole the walk-forward cannot survive.

    `reset_all_calibration_caches` walks a list of module paths and calls
    whichever of `reset_cache` / `reset_panel_cache` it finds — guarded by
    `hasattr`. So a module whose hook is RENAMED (or a path listed with a typo)
    raises nothing, logs nothing and resets nothing: the TTL cache simply
    survives the cutoff change and every subsequent walk-forward step answers
    with the previous step's calibration. That is the failure this whole module
    exists to prevent, and it is invisible from the outside — exactly the
    "verify mechanically, never by review" class.

    Behavioural, not a source re-read: a sentinel is planted in each registered
    module's cache and must be gone afterwards."""
    import ast
    import inspect

    src = inspect.getsource(asof_mod.reset_all_calibration_caches)
    paths = {c.value for node in ast.walk(ast.parse(src.strip()))
             if isinstance(node, ast.Tuple)
             for c in node.elts
             if isinstance(c, ast.Constant) and isinstance(c.value, str)
             and c.value.startswith("src.")}
    assert len(paths) >= 4, f"registration list not found (got {paths})"

    planted = []
    for path in sorted(paths):
        mod = __import__(path, fromlist=["x"])
        hooks = [h for h in ("reset_cache", "reset_panel_cache") if hasattr(mod, h)]
        assert hooks, (
            f"{path} is registered for as-of flushing but exposes neither "
            f"reset_cache nor reset_panel_cache — its cache silently survives "
            f"every cutoff change (rename the hook back, or drop the entry)")
        caches = [n for n in dir(mod)
                  if n.endswith("CACHE") and isinstance(getattr(mod, n), dict)]
        assert caches, f"{path} has a reset hook but no dict cache to reset"
        for name in caches:
            cache = getattr(mod, name)
            if "ts" in cache:
                # Slot-style ({"ts": …, "<payload>": …}): the hook UPDATES the
                # slots rather than clearing, so a fresh key would survive
                # legitimately. Poison the slots instead.
                cache["ts"] = 1e18
                for k in cache:
                    if k != "ts":
                        cache[k] = "__asof_sentinel__"
            else:
                cache["__asof_sentinel__"] = object()
            planted.append((path, name))

    asof_mod.reset_all_calibration_caches()

    survived = []
    for path, name in planted:
        cache = getattr(__import__(path, fromlist=["x"]), name)
        if "ts" in cache:
            stale = (cache["ts"] != 0.0
                     or any(v == "__asof_sentinel__"
                            for k, v in cache.items() if k != "ts"))
        else:
            stale = "__asof_sentinel__" in cache
        if stale:
            survived.append(f"{path}.{name}")
    assert not survived, f"cache(s) survived the as-of flush: {survived}"


def test_the_newest_ttl_caches_are_registered():
    """rank_shaping (the shaped rank->payoff curves, 6h TTL) and news_shock (the
    per-ticker attention baselines, 20min TTL) both feed live SCORES, so a stale
    curve or baseline inside a walk-forward step is look-ahead, not just noise.
    Pinned by name because "I added a module-level TTL cache" is the moment the
    registration is easiest to forget."""
    import ast
    import inspect

    src = inspect.getsource(asof_mod.reset_all_calibration_caches)
    for path in ("src.signals.rank_shaping", "src.signals.news_shock"):
        assert path in src, f"{path} is not registered for as-of cache flushing"
    assert ast.parse(src.strip())          # the source really is this function


# ── the ledger leaks outcomes, not entries ────────────────────────────────────

def test_ledger_visibility_keys_on_the_EXIT_not_the_entry():
    """A trade entered before the cutoff but closed after it carries a realised
    return that had not happened yet — admitting it would leak precisely the
    outcome a win-rate calibration is trying to predict."""
    from src.performance.tracker import _closed_before

    cut = "2026-07-10"
    assert _closed_before({"status": "CLOSED", "entry_date": "2026-07-01",
                           "exit_date": "2026-07-05"}, cut)
    assert not _closed_before({"status": "CLOSED", "entry_date": "2026-07-01",
                               "exit_date": "2026-07-20"}, cut), \
        "a trade resolving AFTER the cutoff is future information"
    assert not _closed_before({"status": "OPEN", "entry_date": "2026-07-01"}, cut), \
        "an open position has no outcome to contribute"
    assert not _closed_before({"status": "CLOSED", "entry_date": "2026-07-01"}, cut), \
        "closed but undated cannot be proven visible"


# ── weight history resolution ─────────────────────────────────────────────────

def _history():
    """Weight states keyed on REAL method names.

    Using placeholder names here silently filters every score out of the
    combine (`m in weights`), so the backtest returns nothing and the test fails
    for a reason that has nothing to do with what it is checking.
    """
    import json
    mk = lambda d, n: {
        "as_of": d, "computed_at": "x", "n_active": n,
        "weights": json.dumps({"tech": 0.5, "vwap": 0.3}),
        "inverted": "[]", "filtered": "[]",
        "buy_filtered": "[]", "sell_filtered": "[]",
        "buy_mults": "{}", "sell_mults": "{}"}
    return pd.DataFrame([mk("2026-07-01", 12), mk("2026-07-05", 10),
                         mk("2026-07-09", 11)])


def test_weights_for_date_uses_the_latest_STRICTLY_earlier_calibration():
    from src.analysis.walkforward import weights_for_date

    h = _history()
    # Same-day calibration must NOT be used: it saw that day's own data.
    assert weights_for_date("2026-07-05", h)["as_of"] == "2026-07-01"
    assert weights_for_date("2026-07-06", h)["as_of"] == "2026-07-05"
    assert weights_for_date("2026-07-20", h)["as_of"] == "2026-07-09"


def test_weights_for_date_returns_none_before_any_calibration():
    from src.analysis.walkforward import weights_for_date
    assert weights_for_date("2026-06-01", _history()) is None


def test_backtest_skips_rows_with_no_prior_calibration(monkeypatch):
    """Falling back to TODAY's weights for an uncalibrated early date would
    reintroduce exactly the look-ahead walk-forward removes, so the row is
    dropped instead."""
    from src.analysis import backtest as bt
    from src.analysis import walkforward as wf

    monkeypatch.setattr(wf, "load_weight_history", lambda: _history())
    import src.db.repo as repo
    monkeypatch.setattr(repo, "fetch_df", lambda *a, **k: pd.DataFrame({
        "signal_date": ["2026-06-01", "2026-07-10"],
        "ticker": ["AAA", "BBB"],
        "generated_at": ["2026-06-01T14:00", "2026-07-10T14:00"],
        "tech": [0.8, 0.8], "vwap": [0.4, 0.4],
        "rp_tech": [None, None], "rp_vwap": [None, None],
        "rp_movement_factor": [None, None], "rp_tape_score": [None, None],
        "rp_vol_ratio": [None, None],
    }))
    out = bt.run_backtest(walk_forward=True)
    dates = set(out["signal_date"]) if not out.empty else set()
    assert "2026-06-01" not in dates, "row predating every calibration must be skipped"


def test_walkforward_stamps_the_calibration_date_not_a_global_hash():
    """A walk-forward row must say WHICH point-in-time calibration produced it;
    a single global hash would make the whole point unauditable."""
    from src.analysis import backtest as bt
    from src.analysis import walkforward as wf
    import src.db.repo as repo

    hist = _history()
    orig_load = wf.load_weight_history
    wf.load_weight_history = lambda: hist
    orig_fetch = repo.fetch_df
    repo.fetch_df = lambda *a, **k: pd.DataFrame({
        "signal_date": ["2026-07-10"], "ticker": ["BBB"],
        "generated_at": ["2026-07-10T14:00"],
        "tech": [0.8], "vwap": [0.4],
        "rp_tech": [None], "rp_vwap": [None], "rp_movement_factor": [None],
        "rp_tape_score": [None], "rp_vol_ratio": [None]})
    try:
        out = bt.run_backtest(walk_forward=True)
        assert not out.empty
        # Since 2026-08-20 the stamp is "<arch>|wf:<as_of>" — the entry
        # architecture is part of what a backtest ran under, and rows from
        # different architectures must never be pooled.
        tag = out.iloc[0]["weight_set"]
        assert "|wf:" in tag, tag
        assert tag.split("|", 1)[0] in ("rank-v1", "abs-v1"), tag
    finally:
        wf.load_weight_history = orig_load
        repo.fetch_df = orig_fetch


# ── the horizon guard: the cutoff must bound OUTCOMES, not just signals ───────

def test_forward_returns_may_not_reach_past_the_cutoff():
    """The subtle half of a point-in-time cutoff.

    Filtering `signal_date < D` is not enough: a row dated D-1 with a 5-day
    horizon reads a close at D+4, an outcome that had not happened. The row
    LOOKS correctly dated, so the leak is invisible — and it feeds the
    market-relative filter and the method-horizon states, i.e. which methods the
    walk-forward believes were active.

    Availability must therefore recede with the horizon: at cutoff D a 1-day
    return exists up to D-1, a 5-day one only to about D-5.
    """
    from datetime import date
    import unittest.mock as mock
    from src.analysis import signal_panel as sp

    idx = pd.bdate_range("2026-05-01", periods=40)
    closes = {d.date(): 100.0 + i for i, d in enumerate(idx)}
    all_dates = sorted(closes)
    cutoff = all_dates[30]

    # Driven through the REAL build_panel (signals_df bypasses the panel cache);
    # asserting against a private helper would pass whether or not the live path
    # honours the guard.
    signals = pd.DataFrame({
        "signal_date": [d.isoformat() for d in all_dates[:30]],
        "ticker": ["AAA"] * 30,
        "generated_at": [f"{d.isoformat()}T14:00:00" for d in all_dates[:30]],
        "combined_score": [0.5] * 30,
    })

    with mock.patch.object(sp, "_close_series", lambda tk: closes):
        with analysis_asof(cutoff.isoformat()):
            out = sp.build_panel(horizons=(1, 5), signals_df=signals)

    assert not out.empty, "expected panel rows under the cutoff"
    checked = 0
    for h in (1, 5):
        col = f"fwd_ret_{h}d"
        got = out[out[col].notna()]
        if got.empty:
            continue
        checked += 1
        latest = date.fromisoformat(str(got["signal_date"].max()))
        i = all_dates.index(latest)
        assert all_dates[i + h] < cutoff, (
            f"{col} used an end bar at/after the cutoff — future outcome leaked")
    assert checked, "no horizon produced any forward return — test proved nothing"


def test_sim_forward_return_respects_the_cutoff():
    """Same guard on the simulated-trades path, which feeds the market-relative
    filter and the method-horizon states."""
    from src.analysis.simulated_trades import _fwd_intraday

    base = pd.Timestamp("2026-07-01T14:00:00")
    series = [(int((base + pd.Timedelta(days=i)).value), 100.0 + i) for i in range(12)]
    gen = base.isoformat()

    assert _fwd_intraday(series, gen, 2) is not None, "no cutoff => available"
    with analysis_asof("2026-07-02"):
        assert _fwd_intraday(series, gen, 2) is None, \
            "end bar lands after the cutoff — must be withheld"
    with analysis_asof("2026-07-10"):
        assert _fwd_intraday(series, gen, 2) is not None, \
            "end bar precedes the cutoff — must be available"


# ── degraded steps: a fail-soft that fabricates a permissive calibration ──────

def test_a_step_is_flagged_degraded_when_nothing_filters_despite_history(monkeypatch):
    """The layers inside `weights_as_of` are fail-soft and several return EMPTY
    rather than raising, so a transient DB read failure yields "nothing
    filtered" — a materially more permissive calibration that looks completely
    legitimate. Observed once for real: 2026-07-26 stored 21 active methods
    between neighbours at 10, and recomputing the same cutoff gave 10.

    An empty filter IS legitimate early on, so the discriminator is how much
    history was VISIBLE, not the emptiness itself.
    """
    import src.analysis.walkforward as wf
    import src.signals.aggregator as agg
    from config.settings import settings

    monkeypatch.setattr(agg, "winrate_filtered_methods", lambda: set())
    monkeypatch.setattr(agg, "_inverted_methods", lambda: set())
    monkeypatch.setattr(agg, "side_filtered_methods", lambda side: set())
    monkeypatch.setattr(agg, "side_weight_multipliers", lambda side: {})
    monkeypatch.setattr(settings, "walkforward_min_rows_to_filter", 100)

    import src.analysis.signal_panel as sp
    monkeypatch.setattr(sp, "_load_signals", lambda *a, **k: pd.DataFrame(
        {"signal_date": ["2026-07-01"] * 500}))
    assert wf.weights_as_of("2026-07-26")["degraded"] is True

    # Sparse history => an empty filter is expected, NOT degraded.
    monkeypatch.setattr(sp, "_load_signals", lambda *a, **k: pd.DataFrame(
        {"signal_date": ["2026-07-01"] * 10}))
    assert wf.weights_as_of("2026-06-20")["degraded"] is False


def test_weights_for_date_skips_a_degraded_step():
    """A degraded step is a fabricated permissive calibration; scoring a day
    under it would be worse than using the previous good one."""
    import json
    from src.analysis.walkforward import weights_for_date

    mk = lambda d, n, bad: {
        "as_of": d, "computed_at": "x", "n_active": n, "degraded": bad,
        "weights": json.dumps({"tech": 0.5}), "inverted": "[]", "filtered": "[]",
        "buy_filtered": "[]", "sell_filtered": "[]",
        "buy_mults": "{}", "sell_mults": "{}"}
    h = pd.DataFrame([mk("2026-07-01", 10, False), mk("2026-07-05", 21, True)])

    got = weights_for_date("2026-07-08", h)
    assert got["as_of"] == "2026-07-01", "resolved to the DEGRADED step"
