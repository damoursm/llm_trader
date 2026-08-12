"""SEQUENTIAL exit-policy simulator — what a close rule actually EARNS.

``exit_policy_eval`` is MYOPIC by construction: it scores each position-day in
isolation ("was closing here better than holding here?"), which measures an exit
SIGNAL's predictive quality but cannot answer a question about a rule that spans
days. A minimum-hold window is exactly that kind of rule — it changes WHICH day
the position is allowed to close on, so its value only shows up in the realized
return of the whole position.

This module closes that gap. Over the simulated held-position dataset
(``ml_exit_dataset.build_exit_dataset`` — each directional ticker-day opens a
hypothetical position walked forward day by day), it runs a **close-once**
policy per position:

    walk the held-days in order → the first day the policy says EXIT, the
    position is realized at that day's oriented return (``ex_ret``) and the walk
    stops; if it never fires, the position is realized at its last observed day.

That yields, per policy, the realized-return distribution over the SAME set of
positions — so competing rules are compared on outcome, not on signal IC.

**Why this is the right test for the min-hold.** A min-hold cannot help a rule
that is already right; it can only help a rule that is WRONG EARLY. Measuring it
tells you which of those you have. The project standard is that a mechanism must
be shown to be better, not assumed (see the combined-score gate, which measuring
said keep INERT) — a forced holding period is a strong constraint on the book and
has to earn its place the same way.

**Costs matter here in a way they do not in the myopic view.** Exiting pays a
round trip (spread + commission, both legs). A rule that churns can beat
buy-and-hold gross and lose net, so every policy is reported gross AND net of a
per-exit cost. The cost is also what makes the principled alternative to a
min-hold work: an EV rule only exits when the expected gain from exiting exceeds
the round-trip cost, which produces hysteresis from the ECONOMICS rather than
from a clock.

CAVEAT (the honest one): these are SIMULATED positions over the forward panel,
not the live ledger's real trades — the entry rule is "every directional
ticker-day", not the gated book. It is the right substrate for RANKING exit
rules (thousands of positions, no selection bias) and the wrong one for
predicting the live book's P&L.

    python -m src.analysis.exit_policy_sim [--horizon 5] [--max-hold 15]
"""

from __future__ import annotations

import argparse
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

# A policy sees one held-day's state and returns True to CLOSE.
#   state keys: days_held, ex_ret, ex_mfe, ex_mae, ex_giveback, ex_combine,
#               ex_combine_delta, ex_elapsed_ratio, ex_consensus, ml_exit
Policy = Callable[[dict], bool]


# ── the policies under test ──────────────────────────────────────────────────

def p_hold_to_end(_s: dict) -> bool:
    """Never close early — the baseline every other rule must beat."""
    return False


def p_ml_exit(threshold: float = 0.35, min_hold: int = 0) -> Policy:
    """Close when the ML exit model's hold-conviction is confidently negative.
    ``min_hold`` > 0 forbids closing before that many held days (the current live
    arm behaviour); 0 is the unconstrained rule."""
    def _p(s: dict) -> bool:
        if s["days_held"] < min_hold:
            return False
        v = s.get("ml_exit")
        return v is not None and v == v and v <= -abs(threshold)
    return _p


def p_consensus(threshold: float = 0.35, min_hold: int = 0) -> Policy:
    """The CURRENT hand-built mechanical exit (``exit_method_consensus``)."""
    def _p(s: dict) -> bool:
        if s["days_held"] < min_hold:
            return False
        v = s.get("ex_consensus")
        return v is not None and v == v and v <= -abs(threshold)
    return _p


def p_trailing(arm_pct: float = 3.0, give_back: float = 0.5) -> Policy:
    """The live trailing stop: once MFE clears ``arm_pct``, close on giving back
    ``give_back`` of the peak."""
    def _p(s: dict) -> bool:
        mfe = s.get("ex_mfe") or 0.0
        return mfe > arm_pct and (s.get("ex_ret") or 0.0) <= mfe * (1.0 - give_back)
    return _p


def p_ev(cost_pct: float, edge_scale: float = 1.0, min_hold: int = 0) -> Policy:
    """**The principled alternative to a min-hold.**

    Exit only when the expected gain from exiting beats the ROUND-TRIP COST of
    doing so. The ML exit model's hold-conviction ``c ∈ [−1,+1]`` is a calibrated
    P(held return > 0) rescaled (``c = 2p − 1``), so ``−c`` is the model's
    expected-loss direction; scaling it by the position's own realized volatility
    proxy (its excursion range) turns it into a % expectation comparable to a cost.

        expected_gain_from_exiting ≈ (−c) × edge_scale × range
        exit  ⟺  expected_gain_from_exiting > cost_pct

    The cost term creates HYSTERESIS naturally: a marginal negative view cannot
    justify paying the spread, so weak signals hold — which is what a min-hold was
    crudely approximating, except this responds to the position's STATE instead of
    to a clock, so a genuinely broken position can still exit on day 1."""
    def _p(s: dict) -> bool:
        if s["days_held"] < min_hold:
            return False
        c = s.get("ml_exit")
        if c is None or c != c or c >= 0:
            return False
        rng = max(1.0, float(s.get("ex_mfe") or 0.0) - float(s.get("ex_mae") or 0.0))
        return (-c) * edge_scale * rng > cost_pct
    return _p


def p_fixed_day(day: int) -> Policy:
    """**Control.** Always close on held-day ``day``, ignoring every signal.

    The load-bearing control for this whole harness. On a book whose average
    position DECAYS, any rule that shortens the average hold looks good for a
    reason that has nothing to do with timing skill. Comparing a signal rule
    against the fixed-day rule with the SAME average hold separates the two: only
    the excess over the matched fixed-day exit is timing."""
    def _p(s: dict) -> bool:
        return s["days_held"] >= day
    return _p


def p_random(exit_rate: float, seed: int = 0) -> Policy:
    """**Control.** Close on a uniformly-random held-day at the given rate — the
    "same amount of exiting, no information" benchmark (mirrors the live book's
    exit-timing Monte Carlo)."""
    rng = np.random.default_rng(seed)
    def _p(s: dict) -> bool:
        # Per-day hazard chosen so a ~7-day position exits at about `exit_rate`.
        return bool(rng.random() < (1.0 - (1.0 - exit_rate) ** (1.0 / 7.0)))
    return _p


# ── the sequential simulator ─────────────────────────────────────────────────

def simulate(df: pd.DataFrame, policies: Dict[str, Policy],
             cost_pct: float = 0.0) -> pd.DataFrame:
    """Run each close-once policy over every simulated position in ``df``.

    ``df`` needs the exit-dataset columns plus an ``ml_exit`` column of
    OUT-OF-SAMPLE model scores (see ``walk_forward_scores``). Returns one row per
    policy: realized return (gross + net of a round-trip ``cost_pct`` charged only
    when the policy actually closes early), win rate, average hold, exit rate."""
    if df.empty:
        return pd.DataFrame()
    need = ["ticker", "entry_date", "days_held", "ex_ret"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"exit frame missing {c}")
    df = df.sort_values(["ticker", "entry_date", "days_held"])
    state_cols = [c for c in ("days_held", "ex_ret", "ex_mfe", "ex_mae", "ex_giveback",
                              "ex_from_mae", "ex_combine", "ex_combine_delta",
                              "ex_elapsed_ratio", "ex_consensus", "ml_exit")
                  if c in df.columns]

    # Pre-extract each position's day sequence once; every policy replays it.
    positions: List[List[dict]] = []
    for _key, g in df.groupby(["ticker", "entry_date"], sort=False):
        positions.append(g[state_cols].to_dict("records"))

    rows = []
    for name, pol in policies.items():
        rets, holds, n_exit = [], [], 0
        for days in positions:
            realized, held_for, exited = None, len(days), False
            for st in days:
                if pol(st):
                    realized, held_for, exited = st["ex_ret"], st["days_held"], True
                    break
            if realized is None:                      # never fired → carried to the end
                realized, held_for = days[-1]["ex_ret"], days[-1]["days_held"]
            rets.append(float(realized) - (cost_pct if exited else 0.0))
            holds.append(float(held_for))
            n_exit += int(exited)
        a = np.asarray(rets, dtype=float)
        rows.append({
            "policy": name, "positions": len(a),
            "mean_ret": round(float(a.mean()), 4),
            "median_ret": round(float(np.median(a)), 4),
            "win_pct": round(100.0 * float((a > 0).mean()), 2),
            "avg_hold_d": round(float(np.mean(holds)), 2),
            "exit_rate_pct": round(100.0 * n_exit / max(1, len(positions)), 1),
        })
    return pd.DataFrame(rows)


def walk_forward_scores(horizon: int = 5, days: Optional[int] = None, max_hold: int = 15,
                        deadband: float = 0.0, min_train_days: int = 8, step_days: int = 2,
                        min_train_rows: int = 500) -> pd.DataFrame:
    """The exit dataset with an added ``ml_exit`` column of OUT-OF-SAMPLE model
    scores (``2·P(keep)−1``), produced by the same point-in-time walk-forward the
    go/no-go uses — so a policy simulated on them never sees its own future."""
    from datetime import date as _d
    from src.analysis.ml_exit_dataset import EXIT_FEATURE_COLUMNS, build_exit_dataset
    from src.analysis.ml_train import label_from_return, make_model_factory

    df = build_exit_dataset(horizon=horizon, days=days, max_hold=max_hold)
    if df.empty:
        return df
    ycol, ecol = f"fwd_ret_pos_{horizon}d", f"end_date_{horizon}d"
    feats = [f for f in EXIT_FEATURE_COLUMNS if f in df.columns]
    work = df[df[ycol].notna()].copy().reset_index(drop=True)
    work["_y"] = pd.to_numeric(work[ycol], errors="coerce").map(
        lambda r: label_from_return(r, deadband))
    work = work[work["_y"].notna()].reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    work["_sig"] = work["signal_date"].map(lambda s: _d.fromisoformat(str(s)[:10]))
    work["_end"] = work[ecol].map(lambda s: _d.fromisoformat(str(s)[:10]))
    X = work[feats].to_numpy(dtype=float)
    y = work["_y"].to_numpy(dtype=int)
    uniq = sorted(work["_sig"].unique())
    work["ml_exit"] = np.nan
    factory = make_model_factory("gbm")
    for a in range(min_train_days, len(uniq), step_days):
        cutoff = uniq[a]
        nxt = uniq[a + step_days] if a + step_days < len(uniq) else None
        tr = work["_end"].to_numpy() < cutoff             # label printed before the cutoff
        sig = work["_sig"].to_numpy()
        te = (sig >= cutoff) if nxt is None else ((sig >= cutoff) & (sig < nxt))
        if tr.sum() < min_train_rows or te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        model = factory().fit(X[tr], y[tr])
        keep, _ = model.bull_bear(X[te])
        work.loc[te, "ml_exit"] = np.clip(2.0 * keep - 1.0, -1.0, 1.0)
    scored = work[work["ml_exit"].notna()].reset_index(drop=True)
    logger.info(f"[exit_policy_sim] {len(scored):,} OOS held-days over "
                f"{scored.groupby(['ticker', 'entry_date']).ngroups:,} positions")
    return scored


def add_excess_vs_matched_control(t: pd.DataFrame) -> pd.DataFrame:
    """Add ``excess`` = the policy's return MINUS the fixed-day control with the
    same average hold (linearly interpolated along the control curve).

    This is the headline number, not ``mean_ret``. On a decaying book the raw
    return of any rule is dominated by how long it holds, so comparing rules by
    ``mean_ret`` mostly ranks them by average hold. The fixed-day controls trace
    return-as-a-function-of-hold with ZERO information; the excess over that curve
    at the SAME hold is what the signal actually contributed."""
    ctrl = t[t["policy"].str.startswith("[control] always exit")]
    if ctrl.empty:
        t["excess"] = np.nan
        return t
    xs = ctrl["avg_hold_d"].to_numpy(dtype=float)
    ys = ctrl["mean_ret"].to_numpy(dtype=float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    t = t.copy()
    t["excess"] = [round(float(r["mean_ret"] - np.interp(r["avg_hold_d"], xs, ys)), 4)
                   for _, r in t.iterrows()]
    return t


def _print(t: pd.DataFrame, cost_pct: float) -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if t is None or t.empty:
        print("No policy results — the panel is too thin.")
        return
    print(f"\nSEQUENTIAL EXIT-POLICY SIMULATION (close-once, per simulated position)")
    print(f"Round-trip cost charged on an early close: {cost_pct:.2f}%   "
          f"positions: {int(t['positions'].iloc[0]):,}\n")
    head = (f"{'policy':<34}{'mean_ret%':>11}{'win%':>7}"
            f"{'hold_d':>8}{'exit%':>7}{'EXCESS':>9}")
    print(head); print("-" * len(head))
    for _, r in t.iterrows():
        ex = r.get("excess")
        exs = f"{ex:>+9.3f}" if ex is not None and pd.notna(ex) else f"{'—':>9}"
        print(f"{r['policy']:<34}{r['mean_ret']:>+11.4f}{r['win_pct']:>7.2f}"
              f"{r['avg_hold_d']:>8.2f}{r['exit_rate_pct']:>7.1f}{exs}")
    print("-" * len(head))
    print("\nEXCESS = return MINUS the fixed-day control at the SAME average hold — the")
    print("only column that is timing SKILL. On a decaying book `mean_ret` mostly ranks")
    print("rules by how long they hold, so it flatters any rule that exits sooner; the")
    print("controls trace that decay with zero information and are divided out here.")
    print("The min-hold pair (identical rule, min_hold 0 vs N) is the direct test of")
    print("whether forcing a holding period is STRICTLY BETTER.")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Sequential exit-policy simulation")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--max-hold", type=int, default=15)
    p.add_argument("--min-hold", type=int, default=5, help="the min-hold under test")
    p.add_argument("--cost-pct", type=float, default=None,
                   help="round-trip cost %% charged on an early close (default: measured)")
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--min-train-days", type=int, default=8)
    p.add_argument("--step-days", type=int, default=2)
    a = p.parse_args(argv)

    from src.db import repo
    repo.set_read_only(True)
    if a.cost_pct is None:
        try:
            from src.performance.spread import _one_side_cost
            a.cost_pct = 2.0 * 100.0 * _one_side_cost(50.0, "STOCK")   # round trip, mid-price name
        except Exception:
            a.cost_pct = 0.10
    df = walk_forward_scores(horizon=a.horizon, max_hold=a.max_hold,
                             min_train_days=a.min_train_days, step_days=a.step_days)
    if df.empty:
        print("No OOS exit scores — panel too thin.")
        return
    mh, thr = a.min_hold, a.threshold
    policies: Dict[str, Policy] = {
        "hold to end": p_hold_to_end,
        f"ml_exit <=-{thr:g}  (no min-hold)": p_ml_exit(thr, 0),
        f"ml_exit <=-{thr:g}  (min-hold {mh}d)": p_ml_exit(thr, mh),
        f"consensus <=-{thr:g} (no min-hold)": p_consensus(thr, 0),
        f"consensus <=-{thr:g} (min-hold {mh}d)": p_consensus(thr, mh),
        "trailing stop (3%/50%)": p_trailing(),
        f"EV rule (cost {a.cost_pct:.2f}%)": p_ev(a.cost_pct, 1.0, 0),
        f"EV rule + min-hold {mh}d": p_ev(a.cost_pct, 1.0, mh),
    }
    # Threshold sweep on the unconstrained ML rule — is 0.35 even the right bar?
    for t in (0.15, 0.25, 0.50, 0.70):
        policies[f"ml_exit <=-{t:g}  (no min-hold)"] = p_ml_exit(t, 0)
    # CONTROLS — the decay/holding-period effect, with no information at all.
    for d in (1, 2, 3, 4, 5, 7):
        policies[f"[control] always exit day {d}"] = p_fixed_day(d)
    policies["[control] random exits (~70%)"] = p_random(0.70, seed=0)
    table = add_excess_vs_matched_control(simulate(df, policies, cost_pct=a.cost_pct))
    _print(table, a.cost_pct)


if __name__ == "__main__":
    main()
