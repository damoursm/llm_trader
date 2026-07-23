"""Offline policy evaluation — counterfactual P&L of alternative trading policies.

The ledger only records trades that passed the gates (selection bias), so it
can't answer "would sizing by breadth have beaten flat sizing?" — the trades
that a different policy *would* have taken aren't in it. But the SIGNALS PANEL
scores EVERY ticker every run, and forward returns are observable for all of
them from the OHLCV cache. That removes the need for importance sampling
(IPS/doubly-robust reweighting of logged actions): we have the actual outcome
of every candidate decision, so we can just REPLAY a policy over the full
unbiased cross-section and read off its counterfactual return — the direct
method of offline policy evaluation, and the clean choice for this system.

A POLICY is a function ``context -> (trade: bool, size: float)`` of a decision
row's features (confidence, combined_score, agreement breadth, session, method
scores …). Direction is the aggregate direction the panel already assigned;
the policy only decides WHETHER to trade and HOW BIG. Each traded decision
earns ``sign × forward_return − round_trip_cost`` (same calibrated cost the
real ledger charges), and decisions are aggregated into DAILY capital-weighted
portfolio returns, then summarized with the mean/std/info-ratio of the per-day
series — so same-day cross-sectional correlation can't inflate the confidence
(the periodic convention used across the analysis layer).

Holding the GATE fixed and varying only the SIZE function isolates the sizing
question ("does breadth sizing earn its keep?"); a policy may also change the
gate to evaluate threshold moves. Deterministic given the DB + OHLCV cache.

    python -m src.analysis.policy_eval --days 90 --horizon 5
"""

from __future__ import annotations

import argparse
from statistics import mean, pstdev
from typing import Callable, Dict, List, Optional

import pandas as pd

from config.settings import settings

# A policy maps a decision context (dict) to (trade?, size). Size is a positive
# capital weight; the gate is `trade`. Direction comes from the context.
Policy = Callable[[dict], "tuple[bool, float]"]


def _dir_sign(direction) -> int:
    d = str(direction or "").upper()
    return 1 if ("BULL" in d or d in ("BUY", "LONG")) else -1


def build_decision_panel(days: Optional[int] = None, horizon: int = 5,
                         signals_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per (signal_date, ticker) decision: the panel features + the
    oriented forward return at ``horizon`` days (``fwd`` = sign × close-to-close
    %, direction already applied) + the fine session. Rows without a forward
    return (cache doesn't reach signal_date + h yet) are dropped — they aren't
    decidable outcomes. Empty until the panel has forward history."""
    from src.analysis.signal_panel import build_panel, session_of_ts
    panel = build_panel(horizons=(horizon,), days=days, signals_df=signals_df)
    if panel is None or panel.empty:
        return pd.DataFrame()
    col = f"fwd_ret_{horizon}d"
    panel = panel[panel[col].notna()].copy()
    if panel.empty:
        return pd.DataFrame()
    panel["dir_sign"] = panel["direction"].map(_dir_sign)
    panel["fwd"] = panel["dir_sign"] * panel[col].astype(float)     # oriented return %
    panel["session"] = (session_of_ts(panel["generated_at"])
                        if "generated_at" in panel.columns else "rth")
    return panel


def _round_trip_cost_pct(price: Optional[float], session: str) -> float:
    """Round-trip cost % for a decision — the SAME calibrated cost the real
    ledger charges (entry + exit legs via ``spread._one_side_cost``), so
    counterfactual returns are cost-consistent with realized ones."""
    from src.performance.spread import _one_side_cost
    p = float(price) if price and price > 0 else 100.0
    sess = session if session in ("extended", "overnight") else "rth"
    return (_one_side_cost(p, "STOCK", sess) + _one_side_cost(p, "STOCK", sess)) * 100.0


def _row_context(row) -> dict:
    """Decision context handed to a policy — the features a live entry would
    have. Breadth fraction is recomputed from the row's method columns."""
    from src.db.schema import SIGNAL_BASE_METHOD_COLUMNS
    ms = {m: (float(getattr(row, m)) if getattr(row, m, None) is not None
              and pd.notna(getattr(row, m)) else 0.0)
          for m in SIGNAL_BASE_METHOD_COLUMNS if hasattr(row, m)}
    dsign = int(getattr(row, "dir_sign", 1))
    n_agree = sum(1 for v in ms.values() if v * dsign > 0)
    n_methods = len(ms) or 1
    return {
        "confidence": float(getattr(row, "confidence", 0.0) or 0.0),
        "combined_score": float(getattr(row, "combined_score", 0.0) or 0.0),
        "dir_sign": dsign,
        "n_agree": n_agree,
        "breadth_frac": n_agree / n_methods,
        "session": getattr(row, "session", "rth"),
        "price": float(getattr(row, "price", 0.0) or 0.0) if getattr(row, "price", None) else None,
        "method_scores": ms,
        "fwd": float(getattr(row, "fwd", 0.0)),
    }


def evaluate_policy(panel: pd.DataFrame, policy: Policy,
                    min_per_day: int = 1) -> dict:
    """Replay ``policy`` over the decision panel → counterfactual performance.

    Per traded decision: net return = ``fwd − round_trip_cost``. Decisions are
    grouped by signal day into a CAPITAL-WEIGHTED portfolio return
    ``Σ(size·net)/Σ(size)``; the daily series is summarized by its mean, std and
    info ratio (mean/std — a Sharpe-like confidence measure that treats each day
    once). Also returns the pooled decision count, win rate, mean net return and
    the capital-weighted mean. Empty-safe."""
    if panel is None or panel.empty:
        return {"n_decisions": 0, "n_days": 0}
    by_day: Dict[str, List[tuple]] = {}
    n_dec = wins = 0
    net_sum = tot_w = tot_wr = 0.0
    for row in panel.itertuples(index=False):
        ctx = _row_context(row)
        trade, size = policy(ctx)
        if not trade or size <= 0:
            continue
        net = ctx["fwd"] - _round_trip_cost_pct(ctx["price"], ctx["session"])
        n_dec += 1
        wins += 1 if net > 0 else 0
        net_sum += net
        tot_w += size
        tot_wr += size * net                       # capital-weighted aggregate
        by_day.setdefault(getattr(row, "signal_date"), []).append((size, net))
    if n_dec == 0:
        return {"n_decisions": 0, "n_days": 0}
    daily = []
    for _d, items in by_day.items():
        if len(items) < min_per_day:
            continue
        wsum = sum(w for w, _ in items)
        if wsum > 0:
            daily.append(sum(w * r for w, r in items) / wsum)
    dmean = mean(daily) if daily else None
    dstd = pstdev(daily) if len(daily) > 1 else None
    icir = (dmean / dstd) if (dmean is not None and dstd and dstd > 1e-9) else None
    return {
        "n_decisions": n_dec,
        "n_days": len(daily),
        "win_rate": round(100.0 * wins / n_dec, 1),
        "avg_net_ret": round(net_sum / n_dec, 3),
        "cap_wtd_ret": round(tot_wr / tot_w, 3) if tot_w > 0 else None,
        "mean_daily_ret": round(dmean, 3) if dmean is not None else None,
        "daily_std": round(dstd, 3) if dstd is not None else None,
        "info_ratio": round(icir, 3) if icir is not None else None,
    }


# ── Concrete policies ─────────────────────────────────────────────────────────
# A shared GATE (the actionable filter, approximated from panel fields) so the
# sizing policies below differ ONLY in their size function — isolating "does
# this sizing rule earn its keep?".

# Baseline actionable confidence threshold (the pre-regime default the pipeline
# starts from). Kept as a module constant so all policies share one gate.
# Tracks the live 0.85 NEUTRAL baseline (raised from 0.78 on 2026-07-21) so the
# counterfactual policy simulation gates on the same bar the pipeline uses.
GATE_CONFIDENCE = 0.85


def _passes_gate(ctx: dict) -> bool:
    return (ctx["confidence"] >= GATE_CONFIDENCE
            and abs(ctx["combined_score"]) >= float(settings.min_combined_score_for_entry)
            and ctx["n_agree"] >= 2)


def policy_flat(ctx: dict):
    return (_passes_gate(ctx), 1.0)


def policy_confidence(ctx: dict):
    if not _passes_gate(ctx):
        return (False, 0.0)
    from src.performance.tracker import _position_multiplier
    return (True, _position_multiplier(ctx["confidence"]))


def policy_breadth(ctx: dict):
    if not _passes_gate(ctx):
        return (False, 0.0)
    from src.performance.tracker import _breadth_multiplier, _breadth_calibration
    return (True, _breadth_multiplier(ctx["breadth_frac"], _breadth_calibration()))


def policy_conf_x_breadth(ctx: dict):
    if not _passes_gate(ctx):
        return (False, 0.0)
    from src.performance.tracker import (_position_multiplier, _breadth_multiplier,
                                         _breadth_calibration)
    size = (_position_multiplier(ctx["confidence"])
            * _breadth_multiplier(ctx["breadth_frac"], _breadth_calibration()))
    return (True, size)


DEFAULT_POLICIES: Dict[str, Policy] = {
    "flat (gate only, size 1)": policy_flat,
    "confidence-sized": policy_confidence,
    "breadth-sized": policy_breadth,
    "confidence × breadth (current)": policy_conf_x_breadth,
}


def compare_policies(days: Optional[int] = None, horizon: int = 5,
                     policies: Optional[Dict[str, Policy]] = None,
                     signals_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Every policy evaluated over the SAME decision panel — the head-to-head
    counterfactual. One row per policy, sorted by capital-weighted return desc.
    The 'flat' row is the sizing-agnostic baseline: a sizing policy earns its
    keep only if its cap-weighted return beats flat on the same decisions."""
    panel = build_decision_panel(days=days, horizon=horizon, signals_df=signals_df)
    pols = policies or DEFAULT_POLICIES
    rows = []
    for name, pol in pols.items():
        stats = evaluate_policy(panel, pol)
        rows.append({"policy": name, **stats})
    df = pd.DataFrame(rows)
    if not df.empty and "cap_wtd_ret" in df.columns:
        df = df.sort_values("cap_wtd_ret", ascending=False, na_position="last").reset_index(drop=True)
    return df


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Offline policy evaluation (counterfactual P&L)")
    ap.add_argument("--days", type=int, default=90, help="lookback window over the signals panel")
    ap.add_argument("--horizon", type=int, default=5, help="holding horizon in trading days")
    args = ap.parse_args(argv)
    from src.db import repo
    repo.set_read_only(True)
    # Match the ledger's cost basis (calibrated real-fill cost) for the replay.
    try:
        from src.performance.tracker import calibrate_sim_costs
        calibrate_sim_costs()
    except Exception:
        pass
    df = compare_policies(days=args.days, horizon=args.horizon)
    if df.empty:
        print("No decidable decisions yet — the signals panel needs forward-return "
              "history (warm it with `python -m src.analysis.signal_panel --refresh`).")
        return
    print(f"Offline policy comparison — {args.horizon}-day horizon, last {args.days} days")
    print("cap_wtd_ret = capital-weighted net return %; the sizing test is each "
          "row vs 'flat'.\n")
    cols = ["policy", "n_decisions", "n_days", "win_rate", "avg_net_ret",
            "cap_wtd_ret", "mean_daily_ret", "info_ratio"]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
