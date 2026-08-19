"""The ml_ohlcv promotion gate must not silently validate a NON-production model.

Discovered 2026-08-18: `ml_validate --target pivot` defaults `--from-parquet` to
the small offline set (`dataset_multi`: 1,500 tickers, stride 2 -> ~388k merged
training rows) while `ml_model.train_and_persist_pivot` trains the live artifact
on `dataset_full` (~9M rows). The gate printed "VERDICT: PROMOTE" with nothing
indicating the model it judged was not the model production runs — and its OOS
predictions correlated with the LIVE persisted `ml_ohlcv` scores at within-day
rank corr +0.044 / +0.604 on the two judgeable days.

That is the repo's silent-failure class: a mechanism whose failure mode is
indistinguishable from normal operation must be verified mechanically, never by
review. These tests are that verification.
"""

from __future__ import annotations

import inspect

import src.analysis.ml_validate as mv


def test_production_parquet_passes_without_a_banner():
    from src.signals.ml_model import _PIVOT_PARQUET
    assert mv._warn_if_not_production_training_set(_PIVOT_PARQUET) is None


def test_offline_parquet_raises_the_mismatch_banner():
    banner = mv._warn_if_not_production_training_set("cache/ml/dataset_multi.parquet")
    assert banner, "the small offline set must be flagged"
    assert "MISMATCH" in banner
    assert "dataset_full" in banner       # names the fix
    assert "NOT the model in production" in banner


def test_banner_tolerates_separator_and_path_style():
    from src.signals.ml_model import _PIVOT_PARQUET
    assert mv._warn_if_not_production_training_set(
        _PIVOT_PARQUET.replace("/", "\\")) is None
    assert mv._warn_if_not_production_training_set(
        "C:/somewhere/" + _PIVOT_PARQUET) is None


def test_banner_is_fail_soft_on_junk():
    for bad in (None, "", "no/such/file.parquet"):
        out = mv._warn_if_not_production_training_set(bad)
        assert out is None or isinstance(out, str)


def test_the_verdict_printer_is_wired_to_the_check():
    """The banner must reach the SAME output as the verdict — a warning logged
    somewhere else is exactly the silence this guards against."""
    src = inspect.getsource(mv._print_pivot)
    assert "_warn_if_not_production_training_set" in src
    assert "VERDICT" in src
    # and the CLI must pass the parquet it actually used, or the check is inert
    main_src = inspect.getsource(mv.main)
    assert "deep_parquet=a.from_parquet" in main_src


def test_the_stale_promotion_bar_is_flagged():
    """The pre-registered t>=2 bar predates the 2026-08-12 H/L pivot
    redefinition, which moved the same statistic from +0.064/t+2.31 to
    +0.244/t+9.32 with no comparable model change. The printer must say so, or
    a trivially-cleared bar keeps reading as a passed gate."""
    src = inspect.getsource(mv._print_pivot)
    assert "2026-08-12" in src
    assert "not comparable" in src
