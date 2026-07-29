"""Per-regime validation (2026-07-26).

The Macro Regime Filter sets the actionable threshold (RISK_ON 0.79 → PANIC
0.95) and blocks BUYs outright in RISK_OFF/PANIC — among the most consequential
parameters in the stack, and nothing had ever measured whether the label carries
information.

Two properties are pinned, and both are about NOT overclaiming:

  * UNOBSERVED regimes are reported as such, never omitted. Over 802 runs only
    NEUTRAL and CAUTION have ever fired; PANIC, RISK_OFF and RISK_ON have not,
    so their thresholds have never been exercised and the BUY block has never
    executed in production. A report that silently dropped those rows would
    read as though the filter had been validated.
  * The DAY is the unit of evidence. A regime is a market-wide state, so every
    ticker scored on a day shares one shock; treating ~1,000 ticker-rows as
    independent would report n=722 for CAUTION when the real evidence is FIVE
    DAYS.

All synthetic, no network, no DB.
"""

import pandas as pd
import pytest

from src.analysis import regime_performance as rp


# ── coverage: absence must be visible ──────────────────────────────────────

def test_unobserved_regimes_are_reported_not_omitted(monkeypatch):
    monkeypatch.setattr(rp, "_q", lambda sql, params=None, tries=12: pd.DataFrame([
        {"regime": "NEUTRAL", "runs": 772, "n_days": 26, "first_day": "2026-06-17",
         "last_day": "2026-07-27", "avg_eff_threshold": 0.822,
         "buy_blocked_runs": 0, "actionable": 3839}]))
    cov = rp.regime_coverage()
    assert list(cov["regime"]) == list(rp.ALL_REGIMES), "every regime must appear"
    unobs = set(cov[~cov["observed"]]["regime"])
    assert unobs == {"PANIC", "RISK_OFF", "CAUTION", "RISK_ON"}
    assert (cov[cov["regime"] == "PANIC"]["runs"] == 0).all()


def test_coverage_reports_the_documented_threshold_even_when_unused(monkeypatch):
    """A never-fired regime still has a configured gate — showing it is how you
    see that 0.95 has never actually been exercised."""
    monkeypatch.setattr(rp, "_q", lambda sql, params=None, tries=12: pd.DataFrame())
    cov = rp.regime_coverage().set_index("regime")
    assert cov.at["PANIC", "doc_threshold"] == 0.95
    assert cov.at["RISK_ON", "doc_threshold"] == 0.79
    assert bool(cov.at["PANIC", "blocks_buys"]) is True
    assert bool(cov.at["CAUTION", "blocks_buys"]) is False


# ── the day-level unit ─────────────────────────────────────────────────────

def _rows(per_regime):
    """per_regime: {regime: [(day, [row returns])]} -> the _regime_rows frame."""
    recs = []
    for reg, dayrows in per_regime.items():
        for day, rets in dayrows:
            for r in rets:
                recs.append({"regime": reg, "signal_date": day, "_ret_1d": r})
    return pd.DataFrame(recs)


def test_row_count_does_not_change_the_verdict(monkeypatch):
    """The pseudo-replication guard, stated precisely.

    A t-test on DAILY MEANS must give the identical answer whether each day
    carried 2 tickers or 500, because a market-wide regime gives one shock per
    day however many names were scored under it. Row-wise pooling would instead
    make the 500-ticker version look ~16x more certain for no new information —
    which is exactly how a 5-day CAUTION sample could be misreported as n=722.
    """
    days_c = ["2026-07-01", "2026-07-02", "2026-07-03"]
    days_n = ["2026-07-06", "2026-07-07", "2026-07-08"]
    means_c, means_n = [1.5, 0.5, 1.0], [-0.4, 0.3, -0.9]

    def frame(rows_per_day):
        return _rows({
            "CAUTION": [(d, [m] * rows_per_day) for d, m in zip(days_c, means_c)],
            "NEUTRAL": [(d, [m] * rows_per_day) for d, m in zip(days_n, means_n)],
        })

    monkeypatch.setattr(rp, "_regime_rows", lambda h=(1,), d=None: frame(2))
    thin = rp.regime_day_test(horizons=(1,)).set_index("regime")
    monkeypatch.setattr(rp, "_regime_rows", lambda h=(1,), d=None: frame(500))
    fat = rp.regime_day_test(horizons=(1,)).set_index("regime")

    assert thin.at["CAUTION", "n_days"] == fat.at["CAUTION", "n_days"] == 3
    assert thin.at["CAUTION", "p"] == pytest.approx(fat.at["CAUTION", "p"]), (
        "250x more tickers per day changed the p-value — rows are being pooled "
        "as if independent")
    assert thin.at["CAUTION", "mean_daily"] == pytest.approx(fat.at["CAUTION", "mean_daily"])


def test_a_genuinely_different_regime_IS_detected(monkeypatch):
    """The test must still have power when the day-level evidence is real."""
    import numpy as np
    rng = np.random.default_rng(0)
    monkeypatch.setattr(rp, "_regime_rows", lambda h=(1,), d=None: _rows({
        "CAUTION": [(f"2026-06-{d:02d}", [float(x)]) for d, x in
                    zip(range(1, 21), rng.normal(-4.0, 0.5, 20))],
        "NEUTRAL": [(f"2026-07-{d:02d}", [float(x)]) for d, x in
                    zip(range(1, 21), rng.normal(+1.0, 0.5, 20))],
    }))
    out = rp.regime_day_test(horizons=(1,)).set_index("regime")
    assert out.at["CAUTION", "p"] < 0.05
    assert "DIFFERENT" in out.at["CAUTION", "verdict"]


def test_a_single_day_is_untestable_not_significant(monkeypatch):
    monkeypatch.setattr(rp, "_regime_rows", lambda h=(1,), d=None: _rows({
        "CAUTION": [("2026-07-01", [9.0] * 400)],
        "NEUTRAL": [(f"2026-07-{d:02d}", [-1.0]) for d in range(2, 20)],
    }))
    out = rp.regime_day_test(horizons=(1,)).set_index("regime")
    assert "untestable" in out.at["CAUTION", "verdict"]
    # No p-value is produced at all — an untestable arm must not be handed a
    # number a reader could mistake for evidence.
    assert "p" not in out.columns or pd.isna(out.at["CAUTION", "p"])


def test_never_observed_regimes_get_a_verdict_not_a_number(monkeypatch):
    monkeypatch.setattr(rp, "_regime_rows", lambda h=(1,), d=None: _rows({
        "NEUTRAL": [(f"2026-07-{d:02d}", [-1.0]) for d in range(1, 20)],
    }))
    out = rp.regime_day_test(horizons=(1,)).set_index("regime")
    assert out.at["PANIC", "verdict"] == "never observed"
    assert out.at["PANIC", "n_days"] == 0


def test_empty_input_degrades_quietly(monkeypatch):
    monkeypatch.setattr(rp, "_regime_rows", lambda h=(1,), d=None: pd.DataFrame())
    assert rp.regime_day_test().empty
