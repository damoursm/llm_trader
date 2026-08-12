"""The degenerate-input Spearman must be SILENT, not just NaN.

Root cause of the 2026-08-08/09 production freezes: ``signal_panel._spearman``
on a constant vector let numpy's corrcoef emit a RuntimeWarning to stderr; the
nightly rescore runs thousands of per-day Spearmans, the flood filled the
scheduler child's undrained stderr pipe, and every subsequent stderr write in
the process blocked forever (numpy warnings, then loguru's console sink, then
the main loop) — a whole-process wedge invisible to both the supervisor and
the broker watchdog, because the process neither exits nor hangs a broker
call. The NaN-means-no-verdict contract already covers the degenerate case;
announcing it per call is what killed the process.

``simplefilter("error")`` promotes ANY warning to an exception, so this test
fails on the unfixed code path and pins the ``np.errstate`` guard.
"""

import warnings

import pandas as pd
import pytest

from src.analysis.signal_panel import _spearman


def test_degenerate_spearman_is_nan_and_emits_no_warning():
    const = pd.Series([2.5] * 60)
    moving = pd.Series(range(60), dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _spearman(const, moving) is None
        assert _spearman(moving, const) is None
        assert _spearman(const, const) is None


def test_normal_spearman_still_works():
    a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _spearman(a, a) == pytest.approx(1.0)
        assert _spearman(a, -a) == pytest.approx(-1.0)
