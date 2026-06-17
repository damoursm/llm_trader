"""Tests for src.analysis.confidence_calibration."""

from src.analysis.confidence_calibration import compute_calibration


def _t(conf, ret, mul=1.0):
    return {"confidence": conf, "return_pct": ret, "position_size_multiplier": mul}


def test_empty():
    rep = compute_calibration([])
    assert rep["n"] == 0
    assert rep["buckets"] == []
    assert rep["slope"] is None


def test_positive_slope_when_return_rises_with_confidence():
    trades = [_t(0.80, -1.0), _t(0.86, 1.0), _t(0.90, 2.0), _t(0.94, 4.0)]
    rep = compute_calibration(trades)
    assert rep["n"] == 4
    assert rep["slope"] > 0
    assert rep["spearman"] == 1.0           # strictly monotone
    assert "rises with" in rep["verdict"]


def test_flat_slope_flags_noise():
    trades = [_t(0.80, 1.0), _t(0.86, 1.0), _t(0.92, 1.0), _t(0.95, 1.0)]
    rep = compute_calibration(trades)
    assert abs(rep["slope"]) < 1e-6
    assert "⚠" in rep["verdict"]            # sizing-on-noise warning


def test_buckets_partition_by_confidence_tier():
    trades = [_t(0.70, 0.0), _t(0.80, 1.0), _t(0.88, 2.0), _t(0.93, 3.0)]
    rep = compute_calibration(trades)
    labels = [b["label"] for b in rep["buckets"]]
    # one trade in each of the four anchor ranges
    assert len(labels) == 4
    assert all(b["trades"] == 1 for b in rep["buckets"])


def test_open_trades_included_via_return_pct():
    # No status field needed — calibration keys off confidence+return_pct only.
    rep = compute_calibration([_t(0.90, 5.0), _t(0.90, -5.0)])
    bucket = [b for b in rep["buckets"] if b["trades"] == 2][0]
    assert bucket["win_rate"] == 50.0
    assert bucket["avg_return"] == 0.0
