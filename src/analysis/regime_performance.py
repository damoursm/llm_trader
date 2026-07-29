"""Per-REGIME validation — does the macro regime classification predict anything?

The Macro Regime Filter is the system's top-down risk overlay: it raises the
actionable confidence threshold as conditions worsen (RISK_ON 0.79 → NEUTRAL
0.85 → CAUTION 0.87 → RISK_OFF 0.89 → PANIC 0.95) and blocks BUY entries
outright in RISK_OFF and PANIC. Those are among the most consequential
parameters in the stack, and until now nothing measured whether the regime label
carries information at all.

Two views, deliberately both:

* SIMULATED (the honest one) — every scored ticker in a run stamped with that
  run's regime, oriented by `combined_score`'s own direction and joined to
  forward returns. Tens of thousands of observations per regime, no gate
  selection. This answers "does the system's edge actually differ by regime?"
* REALIZED — trades actually opened in each regime. Small and gate-selected,
  but it is what the money did.

**The coverage caveat dominates everything here.** Over 40 days and 802 runs
only TWO regimes have ever occurred — NEUTRAL (772 runs / 26 days) and CAUTION
(30 runs / 6 days). PANIC, RISK_OFF and RISK_ON have NEVER fired, which means:
  * their thresholds (0.95 / 0.89 / 0.79) have never been exercised;
  * the BUY block has never executed in production (`allow_buys` was True in
    every run ever recorded);
  * no amount of analysis can validate them — this module reports them as
    UNOBSERVED rather than inventing a verdict.

So `regime_coverage()` is not decoration: it is the precondition for reading
anything else in here.

CLI:  python -m src.analysis.regime_performance [--horizons 1,5,10] [--days N]
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import pandas as pd

from loguru import logger  # project configures loguru sinks only

# Declared order, worst → best. Includes regimes that have never fired so their
# absence is VISIBLE in every report rather than silently omitted.
ALL_REGIMES = ("PANIC", "RISK_OFF", "CAUTION", "NEUTRAL", "RISK_ON")

# The documented base threshold per regime (macro_regime._REGIME_THRESHOLD), for
# reporting what a regime's gate WOULD be against what it has actually done.
DOC_THRESHOLD = {"PANIC": 0.95, "RISK_OFF": 0.89, "CAUTION": 0.87,
                 "NEUTRAL": 0.85, "RISK_ON": 0.79}
BUY_BLOCKED = {"PANIC", "RISK_OFF"}


def _q(sql: str, params: Optional[list] = None, tries: int = 12):
    """Read with backoff — the pipeline holds the single write lock at persist."""
    from src.db import repo
    for i in range(tries):
        try:
            return repo.fetch_df(sql, params)
        except Exception as e:
            msg = str(e)
            if i == tries - 1 or ("another process" not in msg and "used by" not in msg):
                raise
            time.sleep(5)
    return pd.DataFrame()


def regime_coverage() -> pd.DataFrame:
    """Per regime: how much has actually been OBSERVED.

    Read this first. A regime with 0 runs cannot be validated at any sample
    size, and three of the five are in that state.
    """
    df = _q("""SELECT macro_regime AS regime, count(*) AS runs,
                      count(DISTINCT substr(started_at,1,10)) AS n_days,
                      min(substr(started_at,1,10)) AS first_day,
                      max(substr(started_at,1,10)) AS last_day,
                      round(avg(confidence_threshold),3) AS avg_eff_threshold,
                      sum(CASE WHEN allow_buys THEN 0 ELSE 1 END) AS buy_blocked_runs,
                      sum(n_actionable) AS actionable
               FROM runs GROUP BY 1""")
    seen = {} if df is None or df.empty else {r["regime"]: r for _, r in df.iterrows()}
    rows = []
    for reg in ALL_REGIMES:
        r = seen.get(reg)
        rows.append({
            "regime": reg,
            "runs": int(r["runs"]) if r is not None else 0,
            "n_days": int(r["n_days"]) if r is not None else 0,
            "first_day": r["first_day"] if r is not None else None,
            "last_day": r["last_day"] if r is not None else None,
            "doc_threshold": DOC_THRESHOLD[reg],
            "avg_eff_threshold": (round(float(r["avg_eff_threshold"]), 3)
                                  if r is not None else None),
            "blocks_buys": reg in BUY_BLOCKED,
            "buy_blocked_runs": int(r["buy_blocked_runs"] or 0) if r is not None else 0,
            "actionable": int(r["actionable"] or 0) if r is not None else 0,
            "observed": r is not None,
        })
    return pd.DataFrame(rows)


def _regime_rows(horizons: Sequence[int] = (1, 5, 10),
                 days: Optional[int] = None) -> pd.DataFrame:
    """The per-row frame behind both the summary and the day-level test."""
    return regime_simulated(horizons, days, _return_rows=True)


def regime_simulated(horizons: Sequence[int] = (1, 5, 10),
                     days: Optional[int] = None,
                     min_score: Optional[float] = None,
                     _return_rows: bool = False) -> pd.DataFrame:
    """Simulated per-regime performance over the signals panel.

    Every scored ticker in a run inherits that run's regime; the trade is taken
    in `combined_score`'s own direction and marked at the forward close. Gate-
    independent, so it measures the SIGNAL's behaviour in each regime rather
    than the (regime-dependent) gate's selection — which is what makes it a fair
    comparison across regimes at all: a stricter gate mechanically changes WHICH
    trades happen, so realized returns alone can never separate "the regime is
    informative" from "the gate was tighter".

    ``min_score`` defaults to ``buy_sell_diff_threshold`` — only rows that would
    actually fire a direction.
    """
    from config.settings import settings
    from src.analysis.signal_panel import build_panel

    thr = float(settings.buy_sell_diff_threshold if min_score is None else min_score)
    runs = _q("SELECT run_id, macro_regime AS regime FROM runs")
    if runs is None or runs.empty:
        return pd.DataFrame()

    # Regime is a per-RUN property, so the panel's default dedupe (last run per
    # ticker-day) silently discards every run that was not last of its day —
    # which on this data threw away 98% of the CAUTION rows (116 of 9,517),
    # because CAUTION mostly occurred mid-day between NEUTRAL runs. Dedupe to
    # one row per (day, ticker, REGIME) instead: each ticker-day contributes
    # once to each regime it was actually scored under, preserving attribution
    # without the pseudo-replication of keeping all ~28 runs per day.
    from src.analysis.signal_panel import _load_signals
    raw = _load_signals(days, dedupe="all")
    if raw is None or raw.empty:
        return pd.DataFrame()
    raw = raw.merge(runs, on="run_id", how="inner")
    if raw.empty:
        return pd.DataFrame()
    raw = (raw.sort_values("generated_at")
              .groupby(["signal_date", "ticker", "regime"], as_index=False).tail(1))
    panel = build_panel(horizons=horizons, days=days, dedupe="all", signals_df=raw)
    if panel is None or panel.empty:
        return pd.DataFrame()

    df = panel
    df["combined_score"] = pd.to_numeric(df["combined_score"], errors="coerce")
    df = df[df["combined_score"].abs() >= thr]
    if df.empty:
        return pd.DataFrame()
    df = df.copy()          # filtered above, so this is a view — take ownership
    df["_sign"] = df["combined_score"].apply(lambda v: 1.0 if v > 0 else -1.0)
    for _h in horizons:
        _f = pd.to_numeric(df.get(f"fwd_ret_{_h}d"), errors="coerce")
        df[f"_ret_{_h}d"] = df["_sign"] * _f

    if _return_rows:
        return df

    out: List[dict] = []
    for reg in ALL_REGIMES:
        sub = df[df["regime"] == reg]
        row: dict = {"regime": reg, "observed": not sub.empty,
                     "signal_rows": int(len(sub)),
                     "n_days": int(sub["signal_date"].nunique()) if not sub.empty else 0,
                     "buy_share": (round(100.0 * float((sub["_sign"] > 0).mean()), 1)
                                   if not sub.empty else None)}
        for h in horizons:
            fwd = pd.to_numeric(sub.get(f"fwd_ret_{h}d"), errors="coerce") if not sub.empty else None
            if fwd is None:
                row[f"n_{h}d"] = 0
                row[f"ret_{h}d"] = row[f"win_{h}d"] = None
                continue
            oriented = (sub["_sign"] * fwd).dropna()
            row[f"n_{h}d"] = int(len(oriented))
            row[f"ret_{h}d"] = round(float(oriented.mean()), 3) if len(oriented) else None
            row[f"win_{h}d"] = (round(100.0 * float((oriented > 0).mean()), 1)
                                if len(oriented) else None)
            # Raw (un-oriented) market move — is the TAPE different in this
            # regime, separately from whether our direction was right?
            raw = fwd.dropna()
            row[f"mkt_{h}d"] = round(float(raw.mean()), 3) if len(raw) else None
        out.append(row)
    return pd.DataFrame(out)


def regime_day_test(horizons: Sequence[int] = (1, 5, 10),
                    days: Optional[int] = None) -> pd.DataFrame:
    """Regime comparison at the DAY level — the only honest unit here.

    A macro regime is a market-wide state, so every ticker scored on the same
    day shares the same shock. Treating the ~1,000 ticker-rows of a day as
    independent observations is pseudo-replication of the worst kind: it would
    report n=722 for CAUTION when the real evidence is FIVE DAYS. Rows make the
    per-regime means precise; only DAYS make them significant.

    So the day's mean oriented return is one observation, and regimes are
    compared with Welch's t-test on those daily means (unequal variance, unequal
    n). With 5 CAUTION days against 29 NEUTRAL ones this will almost certainly
    return "not significant" — that is the correct answer, and stating it is the
    point.
    """
    sim_rows = _regime_rows(horizons, days)
    if sim_rows is None or sim_rows.empty:
        return pd.DataFrame()
    out: List[dict] = []
    for h in horizons:
        col = f"_ret_{h}d"
        if col not in sim_rows.columns:
            continue
        daily = (sim_rows.dropna(subset=[col])
                 .groupby(["regime", "signal_date"])[col].mean().reset_index())
        base = daily[daily["regime"] == "NEUTRAL"][col]
        for reg in ALL_REGIMES:
            if reg == "NEUTRAL":
                continue
            arm = daily[daily["regime"] == reg][col]
            row = {"horizon": f"{h}d", "regime": reg,
                   "n_days": int(len(arm)), "neutral_days": int(len(base)),
                   "mean_daily": round(float(arm.mean()), 3) if len(arm) else None,
                   "neutral_mean": round(float(base.mean()), 3) if len(base) else None}
            if len(arm) >= 2 and len(base) >= 2:
                # No try/except: this used to call scipy inside one, and since
                # scipy was never a declared dependency EVERY comparison fell
                # through to "test unavailable" — a silent off switch on the
                # only test that makes this table mean anything.
                from src.analysis.stats import welch_t_test
                t, pv = welch_t_test(arm.tolist(), base.tolist())
                if pv == pv:                    # NaN => genuinely undefined
                    row["t"] = round(float(t), 3)
                    row["p"] = round(float(pv), 4)
                    row["verdict"] = ("DIFFERENT (p<0.05)" if pv < 0.05
                                      else "indistinguishable from NEUTRAL")
                else:
                    row["verdict"] = "undefined (no variance)"
            else:
                row["verdict"] = ("never observed" if not len(arm)
                                  else f"only {len(arm)} day(s) — untestable")
            out.append(row)
    return pd.DataFrame(out)


def regime_realized(days: Optional[int] = None) -> pd.DataFrame:
    """Realized outcomes of trades actually OPENED in each regime.

    Small and gate-selected — a stricter regime gate admits fewer, higher-
    conviction trades, so a difference here confounds the regime with its own
    gate. Read alongside `regime_simulated`, never instead of it.
    """
    try:
        from src.performance.tracker import _load_trades
        trades = _load_trades()
    except Exception:
        return pd.DataFrame()
    runs = _q("SELECT run_id, macro_regime AS regime FROM runs")
    reg_of = {} if runs is None or runs.empty else dict(zip(runs["run_id"], runs["regime"]))

    by: Dict[str, List[dict]] = {r: [] for r in ALL_REGIMES}
    for t in trades:
        reg = reg_of.get(t.get("run_id"))
        if reg in by:
            by[reg].append(t)

    out: List[dict] = []
    for reg in ALL_REGIMES:
        ts = by[reg]
        closed = [t for t in ts if t.get("status") == "CLOSED"]
        rets = [float(t.get("return_pct") or 0.0) for t in closed]
        out.append({
            "regime": reg,
            "trades": len(ts),
            "closed": len(closed),
            "open": sum(1 for t in ts if t.get("status") == "OPEN"),
            "buys": sum(1 for t in ts if (t.get("action") or "").upper() == "BUY"),
            "sells": sum(1 for t in ts if (t.get("action") or "").upper() == "SELL"),
            "win_rate": (round(100.0 * sum(1 for r in rets if r > 0) / len(rets), 1)
                         if rets else None),
            "avg_return": round(sum(rets) / len(rets), 2) if rets else None,
        })
    return pd.DataFrame(out)


def evaluate(horizons: Sequence[int] = (1, 5, 10),
             days: Optional[int] = None) -> dict:
    return {"coverage": regime_coverage(),
            "simulated": regime_simulated(horizons, days),
            "day_test": regime_day_test(horizons, days),
            "realized": regime_realized(days),
            "horizons": list(horizons)}


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Per-regime validation")
    ap.add_argument("--horizons", default="1,5,10")
    ap.add_argument("--days", type=int, default=None)
    a = ap.parse_args()
    hs = [int(x) for x in a.horizons.split(",") if x.strip()]
    res = evaluate(horizons=hs, days=a.days)

    with pd.option_context("display.width", 200, "display.max_columns", 40):
        cov = res["coverage"]
        print("\n── COVERAGE — read this first ─────────────────────────────────")
        print(cov.to_string(index=False))
        unobs = list(cov[~cov["observed"]]["regime"])
        if unobs:
            print(f"\n  ⚠ NEVER OBSERVED: {unobs}")
            print("    Their thresholds have never been exercised and the BUY block")
            print("    has never executed. Nothing below can validate them.")
        print("\n── SIMULATED (signals panel, gate-independent) ────────────────")
        sim = res["simulated"]
        print(sim.to_string(index=False) if not sim.empty else "  (no rows)")
        print("\n── DAY-LEVEL TEST — the honest unit ───────────────────────────")
        print("  A regime is a MARKET-WIDE state: every ticker scored on a day shares")
        print("  the same shock, so ~1,000 ticker-rows are ONE observation, not 1,000.")
        dt = res["day_test"]
        print(dt.to_string(index=False) if not dt.empty else "  (no rows)")
        print("\n── REALIZED (trades opened in each regime) ────────────────────")
        rea = res["realized"]
        print(rea.to_string(index=False) if not rea.empty else "  (no rows)")
