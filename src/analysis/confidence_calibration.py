"""Confidence calibration — does realized return actually rise with confidence?

The entry confidence drives the position-size tier (``tracker._position_multiplier``:
1.0× at ≤0.78 ramping to 2.0× at ≥0.95), and the actionable gate is a confidence
threshold. So confidence is doing real work — sizing capital AND deciding what
trades at all. This module is the formal check that the work is justified: bucket
trades by their entry confidence and see whether higher-confidence buckets earn
more, plus a single fitted slope of return-vs-confidence over the whole book.

A flat or downward slope means the confidence number is not carrying directional
information worth sizing on — the tiers (and the gate) are sizing on noise. The
dashboard's "Return vs entry confidence" scatter shows the raw points; this is
its bucketed + slope summary.

Conventions
-----------
* Closed trades use realized ``return_pct``; open trades use the live M2M
  ``return_pct`` (maintained by ``update_open_trades``) — the same "treat open as
  a hypothetical exit" convention as the rest of the performance surface.
* Buckets are confidence ranges aligned to the sizing anchors, so each bucket is
  also a size tier. Empty buckets are omitted.
* Pure function of the trade list — no OHLCV/NAV walk, no network — so it is fast
  and deterministic for the dashboard and tests.

Usage:  python -m src.analysis.confidence_calibration
"""

from __future__ import annotations

from statistics import median
from typing import List, Optional, Sequence, Tuple

# Confidence-range buckets aligned to the _position_multiplier anchors (0.78 /
# 0.85 / 0.92 / 0.95). Each (lo, hi, label) is [lo, hi); the last is closed.
_BUCKETS: Tuple[Tuple[float, float, str], ...] = (
    (0.00, 0.78, "≤0.78 · 1.0×"),
    (0.78, 0.85, "0.78–0.85 · 1.0–1.5×"),
    (0.85, 0.92, "0.85–0.92 · 1.5–1.85×"),
    (0.92, 1.01, "≥0.92 · 1.85–2.0×"),
)


def _pairs(trades: List[dict]) -> List[Tuple[float, float, float]]:
    """``(confidence, return_pct, weight)`` for every trade with both values."""
    out: List[Tuple[float, float, float]] = []
    for t in trades or []:
        c, r = t.get("confidence"), t.get("return_pct")
        if c is None or r is None:
            continue
        try:
            out.append((float(c), float(r), float(t.get("position_size_multiplier", 1.0))))
        except (TypeError, ValueError):
            continue
    return out


def _bucket_stats(rows: List[Tuple[float, float, float]]) -> dict:
    """Summary metrics for one confidence bucket (returns are already %)."""
    rets = [r for _, r, _ in rows]
    wts = [w for _, _, w in rows]
    total_w = sum(wts)
    wins = [r for r in rets if r > 0]
    return {
        "trades":         len(rows),
        "win_rate":       round(100.0 * len(wins) / len(rets), 1) if rets else None,
        "avg_return":     round(sum(rets) / len(rets), 2) if rets else None,
        "median_return":  round(median(rets), 2) if rets else None,
        "wtd_avg_return": round(sum(r * w for _, r, w in rows) / total_w, 2) if total_w else None,
        "best":           round(max(rets), 2) if rets else None,
        "worst":          round(min(rets), 2) if rets else None,
    }


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation without scipy (Pearson on average-tie ranks)."""
    if len(xs) < 2:
        return None
    import pandas as pd
    sx, sy = pd.Series(xs), pd.Series(ys)
    if sx.nunique() < 2 or sy.nunique() < 2:   # constant series → undefined
        return None
    ic = sx.rank().corr(sy.rank())
    return None if pd.isna(ic) else round(float(ic), 3)


def _slope_and_corr(rows: List[Tuple[float, float, float]]) -> dict:
    """Least-squares slope (% return per 1 pp of confidence) + Pearson/Spearman.

    Slope x-units are confidence *percentage points* (confidence × 100) so the
    number reads as "each extra point of confidence is worth N% of return".
    Needs ≥2 trades at ≥2 distinct confidences."""
    xs = [c * 100.0 for c, _, _ in rows]
    ys = [r for _, r, _ in rows]
    if len(xs) < 2 or len(set(xs)) < 2:
        return {"slope": None, "pearson": None, "spearman": None}
    import numpy as np
    slope, _intercept = np.polyfit(xs, ys, 1)
    # Pearson is undefined when returns are constant (zero variance) — guard so it
    # returns None instead of a NaN-with-warning.
    if np.std(ys) == 0:
        pearson = None
    else:
        p = float(np.corrcoef(xs, ys)[0, 1])
        pearson = None if np.isnan(p) else round(p, 3)
    return {
        "slope":    round(float(slope), 4),
        "pearson":  pearson,
        "spearman": _spearman(xs, ys),
    }


def _verdict(fit: dict, n: int) -> str:
    slope, rho = fit.get("slope"), fit.get("spearman")
    if slope is None:
        return f"Not enough spread to judge calibration ({n} trade(s))."
    direction = "rises with" if slope > 0.005 else "falls with" if slope < -0.005 else "is flat vs"
    rho_txt = f", ρ={rho:+.2f}" if rho is not None else ""
    note = ("" if slope > 0.01 else
            "  ⚠ confidence is not carrying return-predictive information — the size tiers may be sizing on noise."
            if slope <= 0.005 else "")
    return f"Return {direction} confidence: {slope:+.3f}%/pt{rho_txt} over {n} trade(s).{note}"


def compute_calibration(trades: List[dict]) -> dict:
    """Confidence-calibration report over closed + open trades.

    Returns ``{n, slope, pearson, spearman, buckets:[{label, conf_lo, conf_hi,
    ...stats}], verdict}``. ``buckets`` omits empty ranges. Open trades are
    included at their live M2M return (treated as hypothetical exits)."""
    rows = _pairs(trades)
    fit = _slope_and_corr(rows)
    buckets = []
    for lo, hi, label in _BUCKETS:
        in_b = [row for row in rows if lo <= row[0] < hi]
        if not in_b:
            continue
        buckets.append({"label": label, "conf_lo": lo, "conf_hi": hi, **_bucket_stats(in_b)})
    return {
        "n":        len(rows),
        "slope":    fit["slope"],
        "pearson":  fit["pearson"],
        "spearman": fit["spearman"],
        "buckets":  buckets,
        "verdict":  _verdict(fit, len(rows)),
    }


def _print_report(rep: dict) -> None:
    if not rep["n"]:
        print("No trades with a stored confidence yet.")
        return
    print(f"\nConfidence calibration — {rep['n']} trade(s)")
    print(rep["verdict"])
    print()
    head = f"{'bucket':<24}{'trades':>7}{'win%':>7}{'avg%':>8}{'med%':>8}{'wtd%':>8}{'best%':>8}{'worst%':>8}"
    print(head)
    print("-" * len(head))
    for b in rep["buckets"]:
        def f(v, w=8):
            return f"{v:>{w}.2f}" if isinstance(v, (int, float)) else f"{'—':>{w}}"
        print(f"{b['label']:<24}{b['trades']:>7}{(b['win_rate'] or 0):>7.1f}"
              f"{f(b['avg_return'])}{f(b['median_return'])}{f(b['wtd_avg_return'])}"
              f"{f(b['best'])}{f(b['worst'])}")
    print("\nA well-calibrated book shows avg return rising across buckets and a "
          "positive slope; a flat/negative slope means confidence isn't worth sizing on.")


def main() -> None:
    import sys
    from src.db import repo
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # Windows console: render ρ/→/− glyphs
    except Exception:
        pass
    repo.set_read_only(True)   # never contend with a running scheduler's write lock
    try:
        trades = repo.load_trades()
    except Exception as e:
        print(f"Could not read trades ({e}).")
        return
    _print_report(compute_calibration(trades))


if __name__ == "__main__":
    main()
