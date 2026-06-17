"""Exit quality — are we exiting near the worst point, or giving back the peak?

Every open trade ratchets two excursions in ``tracker.update_open_trades``:
``max_favorable_excursion`` (MFE, the best M2M return seen) and
``max_adverse_excursion`` (MAE, the worst). Once closed they freeze. Bracketing
the realized ``return_pct`` between them tells us *where in the trade's own range*
the exit landed:

    exit_placement = (return − MAE) / (MFE − MAE)      ∈ [0, 1]
        0.0  → exited at the worst point (cut at MAE — bad timing)
        1.0  → exited at the peak (perfect profit-take)

plus two complementary lenses:
    capture   = return / MFE        (fraction of the favorable peak kept)
    give_back = MFE − return        (how much of the peak was surrendered)
    edge      = MFE / |MAE|         (was there more upside than downside on offer)

Aggregated, these flag systematic exit problems: a low average placement means
we routinely cut near the bottom (exits too late / stops too tight); low capture
with healthy MFE means we ride winners to the peak and give it all back (no
profit-taking). Complements the sim-vs-broker tracking-error work.

Caveat: MFE/MAE are ratchets *seeded at the first observed tick*, so a trade
opened and closed within one tick, or any legacy trade that predates the feature,
has a degenerate (MFE≈MAE≈return) band and is **excluded** — it would otherwise
read as a fake "perfect" exit.

Usage:  python -m src.analysis.exit_quality
"""

from __future__ import annotations

from statistics import mean
from typing import List, Optional

# A trade's excursion band must span at least this many return-% points for the
# placement/capture ratios to be meaningful (filters degenerate/legacy rows).
_MIN_BAND_PCT = 0.05
# Capture (return / MFE) is only meaningful when there was a real favorable peak
# to capture — below this MFE the ratio explodes on a near-zero denominator
# (e.g. a trade that went −13% with MFE 0.08% is not a "capture" story, it just
# never went favorable), so capture is left None there.
_CAPTURE_MIN_MFE = 1.0
# "exited near the worst point" / "gave back most of the peak" thresholds.
_NEAR_MAE_PLACEMENT = 0.20
_LOW_CAPTURE = 0.50


def _per_trade(t: dict) -> Optional[dict]:
    """Exit-quality metrics for one CLOSED trade, or None if not analysable."""
    if t.get("status") != "CLOSED":
        return None
    mfe = t.get("max_favorable_excursion")
    mae = t.get("max_adverse_excursion")
    ret = t.get("return_pct")
    if mfe is None or mae is None or ret is None:
        return None
    try:
        mfe, mae, ret = float(mfe), float(mae), float(ret)
    except (TypeError, ValueError):
        return None
    # The realized return must lie within the observed band (allow tiny float
    # slop); clamp so a seeding artifact can't push placement out of [0, 1].
    hi, lo = max(mfe, ret), min(mae, ret)
    band = hi - lo
    if band < _MIN_BAND_PCT:
        return None
    placement = (ret - lo) / band                      # 0 = at worst, 1 = at best
    capture = (ret / mfe) if mfe >= _CAPTURE_MIN_MFE else None
    edge = (mfe / abs(mae)) if mae < -_MIN_BAND_PCT else None
    return {
        "ticker":        t.get("ticker"),
        "return_pct":    round(ret, 3),
        "mfe":           round(hi, 3),
        "mae":           round(lo, 3),
        "exit_placement": round(placement, 3),
        "capture":       round(capture, 3) if capture is not None else None,
        "give_back":     round(hi - ret, 3),
        "edge_ratio":    round(edge, 2) if edge is not None else None,
        "exit_reason":   t.get("exit_reason"),
        "session":       t.get("exit_session") or t.get("entry_session") or "rth",
    }


def compute_exit_quality(trades: List[dict]) -> dict:
    """Exit-quality report. Returns ``{n, per_trade:[...], avg_placement,
    avg_capture, avg_give_back, pct_exited_near_mae, pct_gave_back_most_mfe,
    verdict}``. ``n`` counts analysable CLOSED trades (degenerate bands excluded)."""
    rows = [r for r in (_per_trade(t) for t in (trades or [])) if r is not None]
    if not rows:
        return {"n": 0, "per_trade": [], "verdict": "No analysable closed trades yet "
                "(needs closed trades with a non-degenerate MFE/MAE band)."}

    placements = [r["exit_placement"] for r in rows]
    captures = [r["capture"] for r in rows if r["capture"] is not None]
    give_backs = [r["give_back"] for r in rows]
    near_mae = sum(1 for p in placements if p < _NEAR_MAE_PLACEMENT)
    gave_back = sum(1 for r in rows if r["capture"] is not None and r["capture"] < _LOW_CAPTURE)
    avg_place = mean(placements)

    parts = [f"Avg exit placement {avg_place:.0%} of the MFE→MAE range"]
    if avg_place < 0.4:
        parts.append("⚠ exits skew toward the worst point (late exits / tight stops)")
    if captures and mean(captures) < _LOW_CAPTURE:
        parts.append(f"⚠ avg capture {mean(captures):.0%} — most of the favorable peak is given back (no profit-taking)")

    return {
        "n":                       len(rows),
        "per_trade":               rows,
        "avg_placement":           round(avg_place, 3),
        "avg_capture":             round(mean(captures), 3) if captures else None,
        "avg_give_back":           round(mean(give_backs), 3),
        "pct_exited_near_mae":     round(100.0 * near_mae / len(rows), 1),
        "pct_gave_back_most_mfe":  round(100.0 * gave_back / len(captures), 1) if captures else None,
        "verdict":                 "; ".join(parts) + ".",
    }


def _print_report(rep: dict) -> None:
    if not rep["n"]:
        print(rep["verdict"])
        return
    print(f"\nExit quality — {rep['n']} analysable closed trade(s)")
    print(rep["verdict"])
    print(f"  exited near MAE (placement <{int(_NEAR_MAE_PLACEMENT*100)}%): {rep['pct_exited_near_mae']}%   ·   "
          f"gave back >half the peak: {rep['pct_gave_back_most_mfe']}%")
    print()
    head = f"{'ticker':<8}{'ret%':>8}{'MFE%':>8}{'MAE%':>8}{'place':>7}{'capt':>7}{'giveb%':>8}  reason"
    print(head)
    print("-" * len(head))
    for r in sorted(rep["per_trade"], key=lambda x: x["exit_placement"]):
        cap = f"{r['capture']:>7.2f}" if r["capture"] is not None else f"{'—':>7}"
        print(f"{(r['ticker'] or ''):<8}{r['return_pct']:>8.2f}{r['mfe']:>8.2f}{r['mae']:>8.2f}"
              f"{r['exit_placement']:>7.2f}{cap}{r['give_back']:>8.2f}  {r['exit_reason'] or ''}")


def main() -> None:
    import sys
    from src.db import repo
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # Windows console: render →/⚠ glyphs
    except Exception:
        pass
    repo.set_read_only(True)   # never contend with a running scheduler's write lock
    try:
        trades = repo.load_trades()
    except Exception as e:
        print(f"Could not read trades ({e}).")
        return
    _print_report(compute_exit_quality(trades))


if __name__ == "__main__":
    main()
