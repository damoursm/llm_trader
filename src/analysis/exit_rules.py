"""Per-EXIT-RULE funnel on the H/L PIVOT target -- the exit-side twin of
``gate_funnel``.

Each exit rule is judged by the positions it CLOSES versus the ones it lets run,
scored on the ORIENTED REMAINING MOVE to the next H/L pivot from that held day
(``fwd_ret_pos_pv``: + = the leg still runs our way after this tick).

    A rule EARNS its place when what it CLOSES was about to do WORSE than what
    it HELD.

Sign convention is deliberately the same as the gate table so the two read
alike::

    value = hold_excess - exit_excess       (+ = the rule closed the worse cohort)

and ``IC`` is computed on the HOLD indicator, so a positive IC again means "what
it keeps does better than what it drops".

Why this is not the existing exit-IC table: ``exit_panel`` measures each exit
METHOD's continuous conviction score against forward returns. This measures the
binary RULE -- the actual fire/don't-fire condition with its live thresholds --
which is what decides whether a position is closed. A method can have a useful
score and a badly-calibrated threshold, and only this view sees that.

The rules are evaluated in ``monitor_open_positions`` order and a position is
attributed to the FIRST rule that fires, so the cohorts partition.

SIMULATION CAVEATS, stated because a simulated rule that silently never fires is
indistinguishable from one that never triggers:

* ``llm_signal_flipped`` has no LLM in a simulation. Its mechanical analogue --
  the aggregator's own combine turning against the position (``ex_combine < 0``)
  -- is simulated under the name ``combine_flip`` and labelled as a PROXY, not
  the live rule.
* ``ml_exit`` is a model score, not a threshold on panel state, and it only ever
  drives ``ml_arm`` trades. Not simulated.
* ``horizon_expired`` needs each position's own target horizon;
  ``ex_elapsed_ratio`` carries it for only ~9% of rows, so the rule is simulated
  on that subset and its n is small by construction.
* ``llm_confidence_loss`` and ``mechanical_exit`` are OFF in production. They are
  simulated anyway -- that is how a retired rule is re-evaluated -- and flagged.

CLI:  python -m src.analysis.exit_rules [--days 45]
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config.settings import settings
from src.analysis.gate_funnel import _cohort_stats, _gate_ic

# Live evaluation order in tracker.monitor_open_positions. First rule to fire
# owns the close, so the cohorts partition top-to-bottom.
EXIT_RULES = (
    ("macro_regime_exit", "Regime exit (long in PANIC)"),
    ("combine_flip",      "Combine flipped against position  [PROXY for llm_signal_flipped]"),
    ("trailing_stop",     "Trailing stop (give back half the peak)"),
    ("adverse_stop",      "Adverse stop (long 8% / short 20%)"),
    ("mechanical_exit",   "Mechanical consensus  [RETIRED - OFF in production]"),
    ("horizon_expired",   "Horizon expired (held past target)"),
)

# Rules whose live form cannot be reconstructed from panel state.
UNSIMULATED_RULES = ("llm_signal_flipped", "llm_confidence_loss", "ml_exit",
                     "edge_decay", "method_horizon")


def simulate_exit_rules(days: Optional[int] = None, horizon: int = 3,
                        max_hold: int = 12) -> pd.DataFrame:
    """One row per simulated (position, held-day) with the FIRST rule that fires.

    Built on ``ml_exit_dataset.build_exit_dataset`` -- the same simulated
    held-position walk the ML exit-timer trains on, so the position state
    (days held, oriented return, MFE/MAE, give-back, combine) and the pivot
    label come from one already-validated source rather than a second
    reconstruction that could drift from it.
    """
    from src.analysis.ml_exit_dataset import build_exit_dataset
    from src.analysis.signal_panel import build_panel
    from src.db import repo

    ds = build_exit_dataset(horizon=horizon, days=days, max_hold=max_hold)
    if ds is None or ds.empty or "fwd_ret_pos_pv" not in ds.columns:
        return pd.DataFrame()
    df = ds.copy()
    df["ret_rem"] = pd.to_numeric(df["fwd_ret_pos_pv"], errors="coerce")
    df = df[df["ret_rem"].notna()]
    if df.empty:
        return pd.DataFrame()
    lo, hi = df["ret_rem"].quantile([0.01, 0.99])
    df["ret_rem"] = df["ret_rem"].clip(lo, hi)
    df["day"] = df["signal_date"].astype(str).str[:10]

    # DIRECTION is not carried by the exit dataset, and the adverse stop is
    # asymmetric (long 8% / short 20%) -- using one threshold for both would
    # mis-fire on every short. Recovered by joining the panel on the position's
    # ENTRY day, which is where its direction was decided.
    panel = build_panel(horizons=(horizon,), days=days, dedupe="last")
    dirs = pd.DataFrame()
    if panel is not None and not panel.empty and "direction" in panel.columns:
        dirs = panel[["signal_date", "ticker", "direction"]].copy()
        dirs["entry_date"] = dirs["signal_date"].astype(str).str[:10]
        dirs = dirs[["entry_date", "ticker", "direction"]].drop_duplicates(
            subset=["entry_date", "ticker"], keep="last")
    if not dirs.empty:
        df["entry_date"] = df["entry_date"].astype(str).str[:10]
        df = df.merge(dirs, on=["entry_date", "ticker"], how="left")
    else:
        df["direction"] = None
    df["is_long"] = df["direction"].astype(str).str.upper().str.contains("BULL")
    # The position P&L is already ORIENTED, so `sign` for the excess benchmark is
    # +1 throughout: a held position's remaining move is measured in its own
    # favour regardless of side.
    df["sign"] = 1.0
    df["oriented"] = df["ret_rem"]

    # regime per day for the PANIC exit
    runs = repo.fetch_df("SELECT run_id, macro_regime, started_at FROM runs")
    regime_by_day: dict = {}
    if runs is not None and not runs.empty:
        rmap = repo.fetch_df("SELECT DISTINCT run_id, signal_date FROM signals")
        day_of = {str(r.run_id): str(r.signal_date)[:10]
                  for r in rmap.itertuples(index=False)} if rmap is not None else {}
        for r in runs.itertuples(index=False):
            d = day_of.get(str(r.run_id))
            if d:                                   # a PANIC anywhere in the day
                cur = regime_by_day.get(d)
                reg = str(r.macro_regime or "NEUTRAL").upper()
                if cur != "PANIC":
                    regime_by_day[d] = reg
    df["regime"] = df["day"].map(regime_by_day).fillna("NEUTRAL")

    mfe = pd.to_numeric(df.get("ex_mfe"), errors="coerce")
    giveback = pd.to_numeric(df.get("ex_giveback"), errors="coerce")
    ret = pd.to_numeric(df.get("ex_ret"), errors="coerce")
    combine = pd.to_numeric(df.get("ex_combine"), errors="coerce")
    consensus = pd.to_numeric(df.get("ex_consensus"), errors="coerce")
    elapsed = pd.to_numeric(df.get("ex_elapsed_ratio"), errors="coerce")

    rule = pd.Series("held", index=df.index, dtype=object)
    open_ = pd.Series(True, index=df.index)

    def _fire(mask, label):
        nonlocal open_
        hit = open_ & mask.fillna(False)
        rule[hit] = label
        open_ = open_ & ~hit

    _fire((df["regime"] == "PANIC") & df["is_long"], "macro_regime_exit")
    _fire(combine < 0, "combine_flip")
    if settings.enable_trailing_exit:
        # ex_giveback is in PERCENTAGE POINTS (peak minus current), while the live
        # rule compares against a FRACTION of the peak -- so the threshold is
        # frac x peak, not the bare fraction.
        arm = float(settings.trailing_arm_pct)
        frac = float(settings.trailing_give_back_frac)
        _fire((mfe >= arm) & (giveback >= frac * mfe), "trailing_stop")
    if settings.enable_adverse_stop:
        lim = np.where(df["is_long"],
                       -abs(float(settings.adverse_stop_pct_long or settings.adverse_stop_pct)),
                       -abs(float(settings.adverse_stop_pct_short or settings.adverse_stop_pct)))
        _fire(ret <= pd.Series(lim, index=df.index), "adverse_stop")
    # Retired in production; simulated so it can be re-evaluated, flagged in the UI.
    _fire(consensus <= -abs(float(settings.mechanical_exit_threshold)), "mechanical_exit")
    _fire(elapsed >= 1.0, "horizon_expired")

    df["gate"] = rule                      # named `gate` so gate_funnel stats apply
    df.attrs["drift"] = float(df["ret_rem"].mean())
    df.attrs["days"] = int(df["day"].nunique())
    df.attrs["d0"], df.attrs["d1"] = str(df["day"].min()), str(df["day"].max())
    df.attrs["dir_known_pct"] = round(100.0 * df["direction"].notna().mean(), 1)
    return df


def compute_exit_rule_performance(days: Optional[int] = None,
                                  horizon: int = 3) -> dict:
    """``{rows: DataFrame, meta: dict}`` -- one row per exit rule, in fire order."""
    rec = simulate_exit_rules(days=days, horizon=horizon)
    if rec is None or rec.empty:
        return {"rows": pd.DataFrame(), "meta": {}}
    drift = rec.attrs.get("drift", 0.0)

    rows, remaining = [], rec.copy()
    for key, label in EXIT_RULES:
        seen = remaining
        fired = seen[seen["gate"] == key]
        held = seen[seen["gate"] != key]
        f, h = _cohort_stats(fired, drift), _cohort_stats(held, drift)
        ic, icir, ic_days = _gate_ic(seen, key)
        rows.append({
            "stage": label, "seen": int(len(seen)), "dropped": f["n"],
            "drop_win": f["win"], "drop_exc": f["exc"], "drop_t": f["t"],
            "kept": h["n"], "keep_win": h["win"], "keep_exc": h["exc"],
            # + = the rule closed positions that were about to do worse.
            "value": (h["exc"] - f["exc"]) if (f["n"] and h["n"]) else np.nan,
            "ic": ic, "icir": icir, "ic_days": ic_days,
        })
        remaining = held

    still_held = rec[rec["gate"] == "held"]
    for label, frame in (("-> STILL HELD (no rule fired)", still_held),
                         ("(reference) every held day, no exit rules", rec)):
        st = _cohort_stats(frame, drift)
        rows.append({"stage": label, "seen": int(len(rec)), "dropped": np.nan,
                     "drop_win": np.nan, "drop_exc": np.nan, "drop_t": np.nan,
                     "kept": st["n"], "keep_win": st["win"], "keep_exc": st["exc"],
                     "value": np.nan, "ic": np.nan, "icir": np.nan, "ic_days": 0})

    meta = {"calls": int(len(rec)), "days": rec.attrs.get("days", 0),
            "d0": rec.attrs.get("d0", ""), "d1": rec.attrs.get("d1", ""),
            "drift": drift, "passed": int(len(still_held)), "source": "simulated",
            "unsimulated": list(UNSIMULATED_RULES),
            "dir_known_pct": rec.attrs.get("dir_known_pct"),
            "cascade_value": ((_cohort_stats(still_held, drift)["exc"]
                               - _cohort_stats(rec, drift)["exc"])
                              if len(still_held) else np.nan)}
    return {"rows": pd.DataFrame(rows), "meta": meta}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Per-exit-rule funnel on the pivot target")
    ap.add_argument("--days", type=int, default=45)
    ap.add_argument("--horizon", type=int, default=3)
    a = ap.parse_args()
    res = compute_exit_rule_performance(days=a.days, horizon=a.horizon)
    df, meta = res["rows"], res["meta"]
    if df.empty:
        print("No simulated held-days with a settled pivot label yet.")
        return
    print(f"{meta['calls']:,} simulated held-days over {meta['days']} days "
          f"({meta['d0']} -> {meta['d1']}); mean remaining move {meta['drift']:+.3f}%")
    print(df.to_string(index=False))
    print(f"\nrules NOT simulable: {', '.join(meta['unsimulated'])}")
    print(f"direction recovered for {meta['dir_known_pct']}% of rows "
          f"(the adverse stop is asymmetric, so this matters)")
    print("+ value = the rule closed positions that were about to do WORSE.")


if __name__ == "__main__":
    main()
