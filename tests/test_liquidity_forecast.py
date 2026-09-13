"""Tests for the expected-liquidity / drift-risk forecast
(src/performance/liquidity_forecast.py): estimator ordering on synthetic
tapes, fail-soft behaviour, the realized-fill blend, memo discipline, the
signals-table round-trip, and mechanical wiring guards (pipeline + tracker) —
a forecast that silently stops being computed or persisted must be a test
failure, not a NULL column nobody notices."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.performance import liquidity_forecast as lf


@pytest.fixture(autouse=True)
def _fresh():
    lf.reset_cache()
    yield
    lf.reset_cache()


# ── synthetic tapes ────────────────────────────────────────────────────────

def _bounce_tape(spread_frac: float, n: int = 400, seed: int = 7):
    """Mid follows a GBM; each day's H/L are the ask/bid extremes around the
    mid path, close alternates bid/ask (the classic bounce). Returns c, h, l."""
    rng = np.random.default_rng(seed)
    mid = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    half = spread_frac / 2.0
    intraday = np.abs(rng.normal(0, 0.004, n))
    h = mid * (1 + intraday + half)
    l = mid * (1 - intraday - half)
    side = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    c = mid * (1 + side * half)
    return c, h, l


def test_cs_orders_wide_above_tight():
    _, h1, l1 = _bounce_tape(0.002)
    _, h2, l2 = _bounce_tape(0.03)
    tight = lf.corwin_schultz_spread(h1, l1)
    wide = lf.corwin_schultz_spread(h2, l2)
    assert tight is not None and wide is not None
    assert wide > tight, f"CS must rank the wide-spread tape higher ({wide} vs {tight})"


def test_ar_orders_wide_above_tight():
    c1, h1, l1 = _bounce_tape(0.002)
    c2, h2, l2 = _bounce_tape(0.03)
    tight = lf.abdi_ranaldo_spread(c1, h1, l1)
    wide = lf.abdi_ranaldo_spread(c2, h2, l2)
    assert tight is not None and wide is not None
    assert wide > tight


def test_estimators_fail_soft_on_degenerate_input():
    assert lf.corwin_schultz_spread([100.0] * 5, [99.0] * 5) is None      # too short
    assert lf.abdi_ranaldo_spread([1.0] * 5, [1.0] * 5, [1.0] * 5) is None
    # NaN-riddled long series → None, never a raise
    bad = [float("nan")] * 60
    assert lf.corwin_schultz_spread(bad, bad) is None


# ── forecast assembly ──────────────────────────────────────────────────────

def _fake_ohlcv(monkeypatch, spread_frac=0.01, price_mult=1.0, n=200):
    c, h, l = _bounce_tape(spread_frac, n=n)
    df = pd.DataFrame({"Open": c, "High": h * price_mult, "Low": l * price_mult,
                       "Close": c * price_mult, "Volume": np.full(n, 1e6)})
    monkeypatch.setattr("src.data.cache.load_ohlcv", lambda tk, **kw: df.copy())


def test_forecast_returns_bps_and_risk(monkeypatch):
    _fake_ohlcv(monkeypatch, spread_frac=0.01)
    monkeypatch.setattr(lf, "_refresh_realized", lambda: None)
    f = lf.forecast_for("FAKE")
    assert f is not None
    assert f["exp_halfspread_bps"] > 0
    assert f["risk"] in ("low", "medium", "high")
    assert f["source"] in ("struct", "blend", "class", "realized")


def test_realized_fills_pull_the_forecast_on_the_CS_path(monkeypatch):
    """The per-ticker blend survives only where the structural estimate is
    CRUDE (CS/AR/class) — there its own history is the better information."""
    _fake_ohlcv(monkeypatch, spread_frac=0.01)
    monkeypatch.setattr(lf, "_refresh_realized", lambda: None)
    base = lf.forecast_for("FAKE")["exp_halfspread_bps"]
    assert lf.forecast_for("FAKE")["estimator"] == "cs"
    lf.reset_cache()
    _fake_ohlcv(monkeypatch, spread_frac=0.01)
    lf._REALIZED_CACHE.update(ts=float("inf"),
                              by_ticker={"FAKE": (base * 4.0, 12)})
    blended = lf.forecast_for("FAKE")
    assert blended["source"] == "blend"
    assert blended["exp_halfspread_bps"] > base, \
        "per-ticker realized evidence must pull a CS estimate toward itself"


def test_measured_spread_is_NOT_blended_with_ticker_history(monkeypatch):
    """Validated 2026-09-01 on all four held-out folds: blending a MEASURED
    spread toward the ticker's unconditional realized median makes the
    forecast WORSE (mean median-abs-error 5.71 vs 5.03), because the fitted
    mapping already conditions on today's book. Regression guard — the blend
    must not creep back onto the nbbo/ibkr path."""
    lf.reset_cache()
    monkeypatch.setattr(lf, "_refresh_realized", lambda: None)
    monkeypatch.setattr(lf, "_spread_mapping", lambda: (2.295, 0.647))
    # a ticker with a big realized history AND a live measured book
    lf._REALIZED_CACHE.update(ts=float("inf"), by_ticker={"MEGA": (40.0, 20)},
                              level=None, level_n=0, ibkr_map=None)
    lf.set_live_spreads({"MEGA": 1.0})
    f = lf.forecast_for("MEGA")
    assert f["estimator"] == "nbbo"
    assert f["source"] == "struct", "a measured spread must not be blended"
    assert f["exp_halfspread_bps"] == pytest.approx(2.295 * 1.0 ** 0.647, abs=0.05)
    lf.reset_cache()


def test_no_data_yields_none_not_a_guess(monkeypatch):
    monkeypatch.setattr("src.data.cache.load_ohlcv", lambda tk, **kw: None)
    monkeypatch.setattr(lf, "_refresh_realized", lambda: None)
    assert lf.forecast_for("GHOST") is None
    assert lf.drift_risk("GHOST") is None
    assert lf.expected_halfspread_bps("GHOST") is None


def test_disabled_flag_turns_everything_off(monkeypatch):
    _fake_ohlcv(monkeypatch)
    monkeypatch.setattr(settings, "enable_liquidity_forecast", False)
    assert lf.forecast_for("FAKE") is None
    assert lf.prime_liquidity_forecast(["FAKE"]) == {}
    assert lf.cached_bps("FAKE") is None


def test_memo_one_ohlcv_read_per_ticker_per_day(monkeypatch):
    calls = {"n": 0}
    c, h, l = _bounce_tape(0.01)
    df = pd.DataFrame({"High": h, "Low": l, "Close": c,
                       "Volume": np.full(len(c), 1e6)})

    def counting(tk, **kw):
        calls["n"] += 1
        return df.copy()

    monkeypatch.setattr("src.data.cache.load_ohlcv", counting)
    monkeypatch.setattr(lf, "_refresh_realized", lambda: None)
    lf.forecast_for("FAKE")
    lf.forecast_for("FAKE")
    lf.drift_risk("FAKE")
    assert calls["n"] == 1, "structural read must be memoised per (ticker, day)"


def test_cached_bps_never_computes(monkeypatch):
    """The persistence path records what the run computed — a cold memo must
    yield None, not trigger a fresh computation."""
    calls = {"n": 0}

    def counting(tk, **kw):
        calls["n"] += 1
        return None

    monkeypatch.setattr("src.data.cache.load_ohlcv", counting)
    assert lf.cached_bps("NEVERPRIMED") is None
    assert calls["n"] == 0


def test_session_multiplier_widens_offhours(monkeypatch):
    _fake_ohlcv(monkeypatch, spread_frac=0.01)
    monkeypatch.setattr(lf, "_refresh_realized", lambda: None)
    rth = lf.expected_halfspread_bps("FAKE")
    ext = lf.expected_halfspread_bps("FAKE", session="extended")
    assert ext > rth


def test_prime_summary_counts(monkeypatch):
    _fake_ohlcv(monkeypatch, spread_frac=0.01)
    monkeypatch.setattr(lf, "_refresh_realized", lambda: None)
    monkeypatch.setattr(lf, "_level_factor", lambda: 0.27)
    s = lf.prime_liquidity_forecast(["AAA", "BBB", "AAA", None, ""])
    assert s["n"] == 2                      # deduped, empties dropped
    assert s["forecast"] == 2
    assert s["median_bps"] and s["median_bps"] > 0
    # and the persistence read now hits the memo
    assert lf.cached_bps("AAA") == pytest.approx(s["median_bps"], abs=1e6)


# ── persistence round-trip ─────────────────────────────────────────────────

def test_insert_signals_roundtrips_exp_halfspread(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "t.db"))
    from src.db import repo
    row = {"ticker": "AAPL", "type": "STOCK", "direction": "bullish",
           "combined_score": 0.2, "confidence": 0.7, "price": 100.0,
           "exp_halfspread_bps": 12.34, "scores": {"news": 0.2}}
    repo.insert_signals("run-liq", "2026-08-30T14:00:00+00:00", "2026-08-30", [row])
    df = repo.fetch_df("SELECT ticker, exp_halfspread_bps FROM signals")
    assert df.iloc[0]["exp_halfspread_bps"] == pytest.approx(12.34)


# ── mechanical wiring guards ───────────────────────────────────────────────

def test_pipeline_primes_and_persists_the_forecast():
    """A forecast the pipeline stops priming or persisting fails HERE instead
    of degrading into a silently-NULL column (the inert-mechanism rule)."""
    import inspect
    import src.pipeline as p
    src = inspect.getsource(p)
    assert "prime_liquidity_forecast" in src
    assert '"exp_halfspread_bps"' in src


def test_tracker_stamps_the_forecast_on_new_trades():
    import inspect
    from src.performance import tracker
    src = inspect.getsource(tracker.record_new_trades)
    assert "exp_halfspread_bps_at_entry" in src
    assert "liq_risk_at_entry" in src


# ── IBKR measured-spread layer (2026-08-31) ────────────────────────────────

def _fake_ibkr_map(monkeypatch, mapping):
    monkeypatch.setattr(lf, "_ibkr_spread_map", lambda: mapping)
    monkeypatch.setattr(lf, "_memo_epoch", lambda: ("test-epoch", 0.0))


def _fixed_mapping(monkeypatch, a=2.295, b=0.647):
    """Pin the power law to its validated priors so a test asserts the SHAPE,
    not whatever the live fit currently reads."""
    monkeypatch.setattr(lf, "_spread_mapping", lambda: (a, b))


def test_ibkr_measurement_supersedes_cs(monkeypatch):
    _fake_ohlcv(monkeypatch, spread_frac=0.01)
    monkeypatch.setattr(lf, "_refresh_realized", lambda: None)
    _fake_ibkr_map(monkeypatch, {"FAKE": 0.6})
    _fixed_mapping(monkeypatch)
    monkeypatch.setattr(lf, "_sweep_to_pit_scale", lambda: 1.0)
    f = lf.forecast_for("FAKE")
    assert f["estimator"] == "ibkr"
    assert f["exp_halfspread_bps"] == pytest.approx(2.295 * 0.6 ** 0.647, abs=0.02)


def test_mapping_is_concave_not_additive(monkeypatch):
    """The refit (2026-09-01) replaced `floor + slope x spread` with a POWER
    law whose exponent is < 1. Two properties must hold, and the retired
    additive form got both backwards:

    * a sub-bp book forecasts a SMALL deviation (the old 10 bp floor
      over-charged those names ~7x: 10.2 predicted vs 1.5 measured), and
    * the response is CONCAVE — doubling the quoted spread must add LESS than
      double, because the LMT cap refuses the full touch on wide books.
    """
    monkeypatch.setattr(lf, "_refresh_realized", lambda: None)
    _fake_ibkr_map(monkeypatch, {"MEGA": 0.5, "MID": 10.0, "WIDE": 20.0})
    _fixed_mapping(monkeypatch)
    monkeypatch.setattr(lf, "_sweep_to_pit_scale", lambda: 1.0)
    mega = lf.forecast_for("MEGA")["exp_halfspread_bps"]
    mid = lf.forecast_for("MID")["exp_halfspread_bps"]
    wide = lf.forecast_for("WIDE")["exp_halfspread_bps"]
    assert mega < 3.0, "a sub-bp book must not inherit a 10 bp floor"
    assert mega < mid < wide, "monotone in the quoted spread"
    assert wide < 2 * mid, "concave: doubling the spread adds less than double"


def test_mapping_priors_hold_below_the_evidence_floor(monkeypatch):
    """Too few fitted legs ⇒ the validated priors hold exactly, so a thin or
    broken join can never silently install a fitted curve."""
    lf.reset_cache()
    monkeypatch.setattr(lf, "_refresh_realized", lambda: None)
    monkeypatch.setattr(lf, "_fit_pairs", lambda: [(5.0, 5.0)] * 3)
    a, b = lf._spread_mapping()
    assert (a, b) == pytest.approx((lf._MAP_A_PRIOR, lf._MAP_B_PRIOR))


def test_mapping_fit_recovers_a_known_power_law(monkeypatch):
    """Given clean synthetic pairs the fitter must recover the curve that
    generated them (shrinkage pulls toward the prior, so check it MOVED
    toward the truth rather than landing exactly)."""
    lf.reset_cache()
    monkeypatch.setattr(lf, "_refresh_realized", lambda: None)
    xs = np.linspace(0.5, 60, 400)
    monkeypatch.setattr(lf, "_fit_pairs", lambda: [(float(x), 5.0 * x ** 0.9) for x in xs])
    a, b = lf._spread_mapping()
    assert b > lf._MAP_B_PRIOR, "a steeper truth must pull the exponent up"
    assert a > lf._MAP_A_PRIOR, "a larger scale must pull the scale up"
    assert lf._MAP_B_CLAMP[0] <= b <= lf._MAP_B_CLAMP[1]


def test_level_factor_never_fits_on_ibkr_rows(monkeypatch):
    """The CS level factor must calibrate only on CS/AR rows: an IBKR-measured
    megacap (true spread 0.5 bp, realized 12 bp -> ratio 24x) would wreck the
    multiplicative rescale built for range-based estimators."""
    lf._REALIZED_CACHE.update(ts=float("inf"),
                              by_ticker={"MEGA": (12.0, 10)}, level=None,
                              level_n=0, ibkr_map=None)
    monkeypatch.setattr(lf, "_structural",
                        lambda tk: {"struct_half_bps": 0.5, "estimator": "ibkr",
                                    "price": None, "adv": None})
    val = lf._level_factor()
    assert val == pytest.approx(lf._LEVEL_PRIOR)   # no CS evidence -> prior holds
    assert lf._REALIZED_CACHE["level_n"] == 0


def test_stale_ibkr_entries_are_ignored(tmp_path, monkeypatch):
    import json
    from src.performance import spread_sweep as sw
    p = tmp_path / "ibkr_spread.json"
    fresh = {"half_bps": 5.0, "date": "2026-08-28", "bars": 5,
             "fetched_at": "2026-08-30T00:00:00+00:00"}
    stale = dict(fresh, fetched_at="2026-07-01T00:00:00+00:00")
    p.write_text(json.dumps({"FRESH": fresh, "STALE": stale}), encoding="utf-8")
    monkeypatch.setattr(sw, "SPREAD_STORE_PATH", p)

    class _FakeDate(lf.date):
        @classmethod
        def today(cls):
            return cls(2026, 8, 31)

    monkeypatch.setattr(lf, "date", _FakeDate)
    m = lf._ibkr_spread_map()
    assert m == {"FRESH": 5.0}


def test_sweep_store_roundtrip_and_bar_math(tmp_path):
    from src.performance import spread_sweep as sw

    class Bar:
        def __init__(self, b, a):
            self.open, self.close = b, a

    # avg-bid 99.95 / avg-ask 100.05 -> half-spread 5 bp; one crossed bar skipped
    half = sw.half_bps_from_bars([Bar(99.95, 100.05), Bar(100.1, 100.0),
                                  Bar(99.90, 100.10)])
    assert half == pytest.approx(7.5, abs=0.2)     # median of {5.0, 10.0}
    assert sw.half_bps_from_bars([]) is None
    assert sw.half_bps_from_bars([Bar(-1, -1)]) is None

    p = tmp_path / "s.json"
    sw.save_spread_store({"AAPL": {"half_bps": 0.6}}, path=p)
    assert sw.load_spread_store(path=p) == {"AAPL": {"half_bps": 0.6}}
    assert sw.load_spread_store(path=tmp_path / "missing.json") == {}


def test_sweep_rotation_prefers_unswept_then_stalest():
    from src.performance import spread_sweep as sw
    store = {"B": {"fetched_at": "2026-08-30T00:00:00"},
             "C": {"fetched_at": "2026-08-01T00:00:00"}}
    assert sw._rotation_order(["B", "C", "A"], store) == ["A", "C", "B"]


def test_sweep_population_excludes_non_equity_symbols(monkeypatch):
    """Futures/index context symbols (ES=F, ^VIX) have no SMART stock contract
    — the first live sweep burned request slots on them (IBKR Error 200)."""
    import pandas as pd

    from src.performance import spread_sweep as sw

    df = pd.DataFrame({"ticker": ["AAPL", "ES=F", "^VIX", "NQ=F"],
                       "price": [100.0, 5000.0, 15.0, 20000.0]})
    monkeypatch.setattr("src.db.repo.fetch_df", lambda *a, **k: df)
    monkeypatch.setattr("src.db.repo.load_trades", lambda: [])
    monkeypatch.setattr("src.data.cache.load_ohlcv",
                        lambda tk, **kw: pd.DataFrame({
                            "Close": [100.0] * 30, "Volume": [1e6] * 30}))
    pop = sw._sweep_population()
    assert pop == ["AAPL"]


def test_sweep_skips_without_ibkr_broker(monkeypatch):
    from src.performance import spread_sweep as sw
    monkeypatch.setattr(settings, "broker_mode", "off")
    assert "skipped" in sw.run_sweep()
    monkeypatch.setattr(settings, "enable_liquidity_forecast", False)
    assert "skipped" in sw.run_sweep()


def test_eod_chain_launches_the_sweep():
    import inspect
    import src.scheduler.runner as runner
    src = inspect.getsource(runner._eod_work)
    assert "spread_sweep" in src
    assert "enable_eod_spread_sweep" in src


# ── book-at-submit persistence (2026-08-31) ────────────────────────────────

def test_record_order_attaches_fresh_quote_and_roundtrips(tmp_path, monkeypatch):
    """_quote_for's stash → _record_order row → broker_orders columns. A stale
    stash (older than the freshness window) must NOT be attributed."""
    import time as _t

    from src.broker import reconcile as rc
    from src.db import repo

    monkeypatch.setattr(settings, "db_path", str(tmp_path / "t.db"))
    monkeypatch.setattr(settings, "broker_spread_aware_limits", True)

    class _Q:
        bid, ask = 99.98, 100.02

    class _B:
        def get_quote(self, tk):
            return _Q()

    rc._LAST_QUOTE.clear()
    assert rc._quote_for(_B(), "aapl") is not None      # stashes upper-cased
    report = {"orders": []}
    rc._record_order(report, event="SUBMIT", intent="ENTRY", ticker="AAPL",
                     side="BUY", order_type="LMT", requested_qty=5,
                     filled_qty=0, model_price=100.0, limit_price=100.02,
                     fill_price=None, commission=None, status="Submitted",
                     ok=True, error=None, order_id="1", client_ref="r1",
                     submitted_at=None)
    row = report["orders"][0]
    assert row["bid_at_submit"] == pytest.approx(99.98)
    assert row["ask_at_submit"] == pytest.approx(100.02)

    # stale stash → no attribution
    rc._LAST_QUOTE["MSFT"] = (10.0, 10.1, _t.time() - rc._QUOTE_FRESH_SECONDS - 1)
    rc._record_order(report, event="SUBMIT", intent="ENTRY", ticker="MSFT",
                     side="BUY", order_type="LMT", requested_qty=5,
                     filled_qty=0, model_price=10.0, limit_price=None,
                     fill_price=None, commission=None, status="Submitted",
                     ok=True, error=None, order_id="2", client_ref="r2",
                     submitted_at=None)
    assert report["orders"][1]["bid_at_submit"] is None

    repo.insert_broker_report("run-ba", {"orders": report["orders"], "mode": "test",
                                         "connected": True, "ok": True})
    df = repo.fetch_df("SELECT ticker, bid_at_submit, ask_at_submit "
                       "FROM broker_orders ORDER BY ticker")
    assert df.iloc[0]["bid_at_submit"] == pytest.approx(99.98)
    assert df.iloc[0]["ask_at_submit"] == pytest.approx(100.02)
    assert pd.isna(df.iloc[1]["bid_at_submit"])
    rc._LAST_QUOTE.clear()
