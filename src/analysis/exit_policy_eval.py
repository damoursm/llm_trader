"""Offline EXIT policy evaluation — counterfactual value of alternative close rules.

The entry harness (``policy_eval``) answers "does this SIZING earn its keep?".
This is its exit-side twin: "does this CLOSE rule earn its keep?" — the missing
validation before wiring any exit-conviction signal into the live decision
(``tracker._evaluate_decay``, today a single LLM-scalar comparison that ignores
the 27-method exit-conviction panel it already collects).

Structure differs from entry because an exit is close-vs-hold on a HELD
position, so the counterfactual reward is what the position DID NEXT:

    reward(decision) = oriented_forward_return   if the policy HOLDS
                     = 0                          if the policy CLOSES

A good exit therefore CLOSES the position-days whose oriented forward return is
about to go negative (cutting losers) and HOLDS the ones about to go positive.
The headline metrics per policy:

    avg_fwd_on_close  — mean oriented fwd of the days it closed (want NEGATIVE:
                        you avoided a drop)
    held_mean         — mean oriented fwd of the days it held
    exit_alpha        — held_mean − allhold_mean: how much better the book you
                        CARRY does than holding everything. > 0 ⇒ the close rule
                        adds value; the "always hold" baseline is 0 by definition.

Data = the ``exit_signals`` panel (per-tick signed HOLD-conviction, already
position-oriented: + = hold, − = exit) joined to forward returns from the OHLCV
cache — the same direct-method, selection-bias-free replay the entry harness
uses. Deduped to the LAST review per (position, day) so the ~30×-autocorrelated
per-tick reviews don't inflate the evidence (the exit-floor-calibration lesson).
MYOPIC per position-day (like the entry eval is per-decision): it scores the
exit SIGNAL's predictive quality, not the full sequential close-once policy.
Deterministic given the DB + OHLCV cache.

    python -m src.analysis.exit_policy_eval --days 90 --horizon 5
"""

from __future__ import annotations

import argparse
from datetime import date
from statistics import mean, pstdev
from typing import Callable, Dict, List, Optional

import pandas as pd

# An exit policy maps a per-decision method→score dict to close? (True=close).
ExitPolicy = Callable[[dict], bool]


def build_exit_decision_panel(days: Optional[int] = None, horizon: int = 5,
                              exit_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per (position_id, signal_date): the tick's exit-method scores
    (method columns) + ``fwd`` (oriented forward return % at ``horizon`` days
    from that day, position-direction applied). Deduped to the LAST review of
    each position-day. Rows without a forward return are dropped. Empty until
    the exit panel has forward history."""
    from src.analysis.exit_panel import _load_exit_signals, _dir_sign_of
    from src.analysis.simulated_trades import _daily_series, _fwd_daily

    df = exit_df if exit_df is not None else _load_exit_signals(days)
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame()
    df = df.copy()
    # Keep only the LAST review tick per (position, day), then pivot methods → cols.
    df["_last"] = df.groupby(["position_id", "signal_date"])["reviewed_at"].transform("max")
    last = df[df["reviewed_at"] == df["_last"]]
    meta = ["position_id", "signal_date", "ticker", "entry_direction"]
    wide = (last.pivot_table(index=meta, columns="method", values="score", aggfunc="last")
                .reset_index())
    if wide.empty:
        return pd.DataFrame()

    # Oriented forward return per row (one cache read per ticker).
    daily: Dict[str, tuple] = {}
    fwds: List[Optional[float]] = []
    for r in wide.itertuples(index=False):
        tk = getattr(r, "ticker")
        if tk not in daily:
            daily[tk] = _daily_series(tk)
        dates, closes = daily[tk]
        try:
            sigd = date.fromisoformat(getattr(r, "signal_date"))
        except Exception:
            fwds.append(None)
            continue
        raw = _fwd_daily(dates, closes, sigd, horizon)
        if raw is None:
            fwds.append(None)
            continue
        fwds.append(_dir_sign_of(getattr(r, "entry_direction")) * raw)
    wide["fwd"] = fwds
    return wide[wide["fwd"].notna()].reset_index(drop=True)


# Method columns are everything except the meta + fwd; a policy sees them as a dict.
_META_COLS = {"position_id", "signal_date", "ticker", "entry_direction", "fwd", "_last"}


def _row_scores(row) -> dict:
    out = {}
    for k in row._fields:
        if k in _META_COLS:
            continue
        v = getattr(row, k)
        if v is not None and pd.notna(v):
            out[k] = float(v)
    return out


def evaluate_exit_policy(panel: pd.DataFrame, policy: ExitPolicy) -> dict:
    """Replay ``policy`` over the exit decision panel → counterfactual close-rule
    value. See the module docstring for the metric definitions. Empty-safe."""
    if panel is None or panel.empty:
        return {"n_decisions": 0, "n_days": 0}
    n = n_close = 0
    close_fwd_sum = hold_fwd_sum = all_fwd_sum = 0.0
    by_day_held: Dict[str, List[float]] = {}
    by_day_all: Dict[str, List[float]] = {}
    for row in panel.itertuples(index=False):
        fwd = float(getattr(row, "fwd"))
        close = bool(policy(_row_scores(row)))
        n += 1
        all_fwd_sum += fwd
        by_day_all.setdefault(getattr(row, "signal_date"), []).append(fwd)
        if close:
            n_close += 1
            close_fwd_sum += fwd
        else:
            hold_fwd_sum += fwd
            by_day_held.setdefault(getattr(row, "signal_date"), []).append(fwd)
    if n == 0:
        return {"n_decisions": 0, "n_days": 0}
    n_hold = n - n_close
    allhold_mean = all_fwd_sum / n
    held_mean = (hold_fwd_sum / n_hold) if n_hold else 0.0
    # Per-day held-book mean → info ratio (each day once; days fully closed drop out).
    daily = [mean(v) for v in by_day_held.values() if v]
    dmean = mean(daily) if daily else None
    dstd = pstdev(daily) if len(daily) > 1 else None
    return {
        "n_decisions": n,
        "n_days": len(by_day_all),
        "close_rate": round(100.0 * n_close / n, 1),
        "avg_fwd_on_close": round(close_fwd_sum / n_close, 3) if n_close else None,
        "avg_fwd_on_hold": round(held_mean, 3),
        "allhold_mean": round(allhold_mean, 3),
        "exit_alpha": round(held_mean - allhold_mean, 3),
        "held_daily_mean": round(dmean, 3) if dmean is not None else None,
        "info_ratio": round(dmean / dstd, 3) if (dmean is not None and dstd and dstd > 1e-9) else None,
    }


# ── Concrete exit policies ────────────────────────────────────────────────────
# Exit scores are position-oriented: score < 0 = "this method says EXIT".

def exit_always_hold(_scores: dict) -> bool:
    return False                                    # the do-nothing baseline (alpha ≡ 0)


def exit_llm_review_current(scores: dict) -> bool:
    """Approximates the LIVE rule: close when the opener's hold-review flipped or
    lost conviction — here, the ``llm_review`` conviction below the exit floor.
    No review this tick → hold (matches ``_evaluate_decay``)."""
    from src.performance.tracker import _confidence_floor
    v = scores.get("llm_review")
    if v is None:
        return False
    return v < _confidence_floor(None)              # absolute floor (no entry-conf here)


def exit_aggregator(scores: dict) -> bool:
    """Close when the aggregator's combined exit-conviction turned negative."""
    v = scores.get("aggregator")
    return v is not None and v < 0.0


def _exit_breadth_frac(scores: dict) -> Optional[float]:
    """Fraction of the SIGNAL methods (excludes the decision-layer/excursion
    methods) voting EXIT (score < 0) — the exit-side analog of entry agreement
    breadth. None when nothing signal-method scored."""
    skip = {"aggregator", "llm_review", "horizon", "macro_regime", "mfe", "mae"}
    votes = [v for m, v in scores.items() if m not in skip]
    return (sum(1 for v in votes if v < 0) / len(votes)) if votes else None


def make_exit_breadth(threshold: float = 0.5) -> ExitPolicy:
    """Close when > ``threshold`` of the signal methods vote EXIT — the
    breadth-of-exit-agreement rule (the entry-breadth analog)."""
    def _p(scores: dict) -> bool:
        frac = _exit_breadth_frac(scores)
        return frac is not None and frac > threshold
    return _p


DEFAULT_EXIT_POLICIES: Dict[str, ExitPolicy] = {
    "always hold (baseline)": exit_always_hold,
    "llm_review floor (current)": exit_llm_review_current,
    "aggregator < 0": exit_aggregator,
    "exit-breadth > 50%": make_exit_breadth(0.5),
    "exit-breadth > 65%": make_exit_breadth(0.65),
}


def compare_exit_policies(days: Optional[int] = None, horizon: int = 5,
                          policies: Optional[Dict[str, ExitPolicy]] = None,
                          exit_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Every close rule over the SAME decision panel, ranked by ``exit_alpha``
    desc. A rule earns its keep only if its alpha beats the 'always hold'
    baseline (0) — i.e. the positions it keeps outperform keeping everything."""
    panel = build_exit_decision_panel(days=days, horizon=horizon, exit_df=exit_df)
    if panel is None or panel.empty:
        return pd.DataFrame()
    pols = policies or DEFAULT_EXIT_POLICIES
    rows = [{"policy": name, **evaluate_exit_policy(panel, pol)} for name, pol in pols.items()]
    df = pd.DataFrame(rows)
    if not df.empty and "exit_alpha" in df.columns:
        df = df.sort_values("exit_alpha", ascending=False, na_position="last").reset_index(drop=True)
    return df


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Offline EXIT policy evaluation (counterfactual)")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--horizon", type=int, default=5, help="holding horizon in trading days")
    args = ap.parse_args(argv)
    from src.db import repo
    repo.set_read_only(True)
    df = compare_exit_policies(days=args.days, horizon=args.horizon)
    if df.empty:
        print("No decidable exit decisions yet — the exit_signals panel needs "
              "forward-return history (accrues as positions are held; warm the "
              "cache with `python -m src.analysis.signal_panel --refresh`).")
        return
    print(f"Offline EXIT policy comparison - {args.horizon}-day horizon, last {args.days} days")
    print("exit_alpha = held_mean - allhold_mean (>0 => the close rule earns its keep); "
          "avg_fwd_on_close wants to be NEGATIVE (cutting losers).\n")
    cols = ["policy", "n_decisions", "n_days", "close_rate", "avg_fwd_on_close",
            "avg_fwd_on_hold", "allhold_mean", "exit_alpha", "info_ratio"]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
