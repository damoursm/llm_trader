"""Per-GATE entry-funnel performance on the H/L PIVOT target.

Each actionable gate is judged by the cohort it DROPS versus the cohort it lets
through, scored on the signed pivot move oriented by the direction that would
have been traded::

    oriented = sign(action) x fwd_ret_pivot        (+ = the decision was right)

A gate EARNS its place when what it drops has a WORSE oriented outcome than what
it keeps.

Three methodology points, each load-bearing (they come from the 2026-08-17 funnel
study, and reversing any one of them changes the verdicts):

1. **EXCESS, not the raw oriented mean.** Orientation makes raw means
   incomparable across cohorts with different BUY/SELL mixes: a short's oriented
   return is ``-1 x drift`` by construction, so a short-heavy cohort looks worse
   than a long-heavy one for no reason but the population's upward drift. Every
   cohort is therefore benchmarked against a random draw with ITS OWN side mix
   (``mean(sign) x population drift``), and ``excess`` is the reported quantity.

2. **Day-clustered t.** Same-day returns are heavily correlated, so a t computed
   on the raw row count is inflated several-fold. Each signal-day contributes ONE
   observation, and the t is computed on the per-day EXCESS.

3. **The join must use ``signal_date``, never ``generated_at``.** ``signal_date``
   is ET; ``generated_at`` is UTC. Every overnight tick from 20:00-23:59 ET
   carries the NEXT UTC date, so joining on ``generated_at`` mis-assigns ~4 of
   the 7 nightly slots -- which measurably FLIPPED a gate's verdict during the
   original study.

The gate stamps come from ``runs.gate_diag -> gate_outcomes`` (one per ticker per
run, recording the FIRST gate that rejected it, so the cohorts partition), joined
to the settled pivot label from ``signal_panel.build_panel``.

Standing caveat: in the original 45-day window NOTHING in the cascade was
statistically distinguishable (every |t| <= 1.3). Read the table as point
estimates and direction, not established effects -- and treat a gate whose n is
in the low tens as no answer at all.

CLI:  python -m src.analysis.gate_funnel [--days 90]
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import settings

# Cascade order = the order pipeline._apply_actionable_gates applies them. A drop
# is attributed to the FIRST gate that rejects it, so each gate is evaluated only
# on the candidates still alive when it runs.
GATE_STAGES = (
    ("below_threshold",   "Gate 1 - confidence threshold"),
    ("low_agreement",     "Gate 1b - agreement floor"),
    ("buy_blocked",       "Gate 2 - regime BUY block"),
    ("earnings_blackout", "Gate 3 - earnings blackout"),
    ("untradeable",       "Gate 4 - liquidity floor"),
    ("overextended",      "Gate 5 - anti-chase"),
)


def _day_clustered_t(frame: pd.DataFrame, col: str = "_exc") -> tuple:
    """``(t, n_days)`` on the per-day mean of ``col``. NaN below 5 days."""
    if frame.empty or col not in frame.columns:
        return float("nan"), 0
    per_day = frame.groupby("day")[col].mean().dropna()
    n = len(per_day)
    if n < 5:
        return float("nan"), n
    sd = per_day.std(ddof=1)
    if not sd or sd <= 0:
        return float("nan"), n
    return float(per_day.mean() / sd * np.sqrt(n)), n


def _cohort_stats(frame: pd.DataFrame, drift: float) -> dict:
    """n / win% / mean / excess / day-clustered t for one cohort."""
    blank = {"n": 0, "win": np.nan, "mean": np.nan, "exc": np.nan,
             "t": np.nan, "days": 0}
    if frame is None or frame.empty:
        return blank
    s = frame["oriented"].dropna()
    if s.empty:
        return blank
    bench = float(frame["sign"].mean()) * drift
    f = frame.copy()
    f["_exc"] = f["oriented"] - f["sign"] * drift
    t, days = _day_clustered_t(f)
    return {"n": int(len(s)), "win": float((s > 0).mean() * 100.0),
            "mean": float(s.mean()), "exc": float(s.mean()) - bench,
            "t": t, "days": days}


def _gate_ic(seen: pd.DataFrame, gate_key: str) -> tuple:
    """``(ic, icir, ic_days)`` for one gate, treating PASSING it as the score.

    A gate is binary, so its "IC" is the rank correlation between a pass/drop
    indicator and the oriented outcome -- positive = the names it lets through do
    better than the ones it stops. ICIR is the per-day mean/std of that IC, the
    same reliability measure the method tables use, so a gate that is right on
    average but flips sign day to day is visibly unreliable.
    """
    from src.analysis.signal_panel import _spearman, periodic_ic_stats
    if seen is None or len(seen) < 10:
        return np.nan, np.nan, 0
    passed = (seen["gate"] != gate_key).astype(float)
    if passed.nunique() < 2:                       # never fired => no contrast
        return np.nan, np.nan, 0
    ic = _spearman(passed, seen["oriented"])
    _mean, _std, icir, days = periodic_ic_stats(
        seen["day"], passed, seen["oriented"], min_per_day=5, min_days=3)
    return (ic if ic is not None else np.nan,
            icir if icir is not None else np.nan, int(days))


def load_gate_calls(days: Optional[int] = None) -> pd.DataFrame:
    """BUY/SELL recommendations carrying BOTH a gate stamp and a settled pivot
    label, deduped to the last call per (signal_date, ticker)."""
    from src.analysis.signal_panel import build_panel
    from src.db import repo

    panel = build_panel(horizons=(5,), days=days, dedupe="last")
    if panel is None or panel.empty or "fwd_ret_pivot" not in panel.columns:
        return pd.DataFrame()
    lab = panel[["signal_date", "ticker", "fwd_ret_pivot"]].copy()
    lab["fwd_ret_pivot"] = pd.to_numeric(lab["fwd_ret_pivot"], errors="coerce")
    lab = lab[lab["fwd_ret_pivot"].notna()]
    if lab.empty:
        return pd.DataFrame()
    # Winsorize the LABEL, not the cohorts: one bad print must not set a gate's
    # verdict, and the clip has to be identical across every cohort compared.
    lo, hi = lab["fwd_ret_pivot"].quantile([0.01, 0.99])
    lab["ret"] = lab["fwd_ret_pivot"].clip(lo, hi)
    lab["day"] = lab["signal_date"].astype(str).str[:10]

    rdf = repo.fetch_df("SELECT run_id, generated_at, ticker, action FROM "
                        "recommendations WHERE action IN ('BUY','SELL')")
    runs = repo.fetch_df("SELECT run_id, gate_diag FROM runs WHERE gate_diag IS NOT NULL")
    if rdf is None or rdf.empty or runs is None or runs.empty:
        return pd.DataFrame()
    outcomes = {}
    for r in runs.itertuples(index=False):
        try:
            outcomes[str(r.run_id)] = (json.loads(r.gate_diag) or {}).get("gate_outcomes") or {}
        except Exception:
            outcomes[str(r.run_id)] = {}

    # run_id -> ET signal_date. MUST come from `signals`: generated_at is UTC and
    # every 20:00-23:59 ET overnight tick carries the NEXT UTC date (see the
    # module docstring -- this mis-join flipped a verdict once already).
    rmap = repo.fetch_df("SELECT DISTINCT run_id, signal_date FROM signals")
    run_day = {str(r.run_id): str(r.signal_date)[:10] for r in rmap.itertuples(index=False)}
    rdf["day"] = rdf["run_id"].astype(str).map(run_day)
    rdf = rdf[rdf["day"].notna()]
    rdf["gate"] = [outcomes.get(str(rid), {}).get(tk)
                   for rid, tk in zip(rdf["run_id"], rdf["ticker"])]
    rdf = rdf[rdf["gate"].notna()]
    rdf = rdf.sort_values("generated_at").groupby(["day", "ticker"], as_index=False).tail(1)

    rec = rdf.merge(lab[["day", "ticker", "ret"]], on=["day", "ticker"], how="inner")
    if rec.empty:
        return rec
    rec["sign"] = np.where(rec["action"] == "BUY", 1.0, -1.0)
    rec["oriented"] = rec["sign"] * rec["ret"]
    rec.attrs["drift"] = float(lab["ret"].mean())
    rec.attrs["days"] = int(lab["day"].nunique())
    rec.attrs["d0"], rec.attrs["d1"] = str(lab["day"].min()), str(lab["day"].max())
    return rec




# ---------------------------------------------------------------------------
# SIMULATED gates -- the same cascade replayed over the WHOLE signals panel
# ---------------------------------------------------------------------------
#
# The stamped cascade (`load_gate_calls`) only sees tickers the LLM turned into a
# BUY/SELL, which is both small (~2k rows) and selection-biased: the gates are
# judged only on names one upstream stage already liked. Replaying the gate
# LOGIC over every scored ticker-day removes that bias and multiplies the sample
# ~10x, which is the same reason the method tables have a "simulated" source.
#
# Faithfulness rules followed here:
#   * same cascade ORDER, first-rejecting-gate wins, so cohorts partition;
#   * the direction traded is the AGGREGATOR's own (there is no LLM in a
#     simulation) -- so this measures the gates against the mechanical stream;
#   * every input is POINT-IN-TIME. `pipeline._recent_runup_pct` and
#     `liquidity.is_liquid` both read the CURRENT tail of the OHLCV cache, which
#     is correct live and would be look-ahead here, so both are recomputed from
#     bars visible on the signal date only.
#
# Gate 3 (earnings blackout) is NOT simulated: it needs the historical earnings
# calendar as it stood on each signal date, which is not stored. It is reported
# as "not simulable" rather than silently passed, because a gate that always
# passes looks identical to a gate that never rejects anything.

_REGIME_THRESHOLD = {"PANIC": 0.95, "RISK_OFF": 0.89, "CAUTION": 0.87,
                     "NEUTRAL": 0.85, "RISK_ON": 0.79}
_SIM_UNSIMULATED = ("earnings_blackout",)


def _pit_ohlcv_features(tickers, days_needed: dict) -> dict:
    """``{(ticker, day): {"runup": pct, "adv": dollars, "close": px}}`` computed
    from bars visible ON OR BEFORE each day -- never the current tail.

    One parse per ticker (``cache.load_ohlcv`` memoises), then a positional slice
    per requested day, so this is O(rows) rather than O(rows x bars).
    """
    from src.data.cache import load_ohlcv
    lookback = max(1, int(settings.overextension_lookback_bars))
    out: dict = {}
    for tk in tickers:
        wanted = days_needed.get(tk)
        if not wanted:
            continue
        try:
            bars = load_ohlcv(tk)
        except Exception:
            continue
        if bars is None or getattr(bars, "empty", True) or "Close" not in bars.columns:
            continue
        idx = pd.to_datetime(bars.index, errors="coerce")
        dates = pd.Series(idx).dt.strftime("%Y-%m-%d").tolist()
        closes = pd.to_numeric(bars["Close"], errors="coerce").tolist()
        vols = (pd.to_numeric(bars["Volume"], errors="coerce").tolist()
                if "Volume" in bars.columns else [float("nan")] * len(closes))
        for day in wanted:
            # bars strictly visible as of `day` (inclusive) -- the point-in-time cut
            hi = np.searchsorted(np.array(dates), day, side="right")
            if hi <= 0:
                continue
            last = closes[hi - 1]
            if not last or last != last or last <= 0:
                continue
            runup = np.nan
            if hi > lookback:
                prev = closes[hi - 1 - lookback]
                if prev and prev == prev and prev > 0:
                    runup = (last - prev) / prev * 100.0
            lo = max(0, hi - 20)
            dv = [c * v for c, v in zip(closes[lo:hi], vols[lo:hi])
                  if c == c and v == v and c > 0 and v > 0]
            out[(tk, day)] = {"runup": runup, "close": last,
                              "adv": (float(np.mean(dv)) if dv else np.nan)}
    return out


def simulate_gate_calls(days: Optional[int] = None) -> pd.DataFrame:
    """Replay the gate cascade over EVERY scored ticker-day.

    Returns the same shape as ``load_gate_calls`` (``gate`` / ``sign`` /
    ``oriented`` / ``day``), so the identical stats machinery scores both.
    """
    from src.analysis.signal_panel import build_panel
    from src.db import repo

    panel = build_panel(horizons=(5,), days=days, dedupe="last")
    if panel is None or panel.empty or "fwd_ret_pivot" not in panel.columns:
        return pd.DataFrame()
    cols = [c for c in ("signal_date", "ticker", "price", "confidence",
                        "n_methods_agreeing", "combined_score", "fwd_ret_pivot",
                        "run_id") if c in panel.columns]
    df = panel[cols].copy()
    df["fwd_ret_pivot"] = pd.to_numeric(df["fwd_ret_pivot"], errors="coerce")
    df["combined_score"] = pd.to_numeric(df["combined_score"], errors="coerce")
    df = df[df["fwd_ret_pivot"].notna() & df["combined_score"].notna()]
    df = df[df["combined_score"] != 0]
    if df.empty:
        return pd.DataFrame()

    lo, hi = df["fwd_ret_pivot"].quantile([0.01, 0.99])
    df["ret"] = df["fwd_ret_pivot"].clip(lo, hi)
    df["day"] = df["signal_date"].astype(str).str[:10]
    df["sign"] = np.sign(df["combined_score"])
    df["action"] = np.where(df["sign"] > 0, "BUY", "SELL")
    df["oriented"] = df["sign"] * df["ret"]

    # regime per run -> Gate 1 threshold and the Gate 2 BUY block
    runs = repo.fetch_df("SELECT run_id, macro_regime FROM runs")
    regime = {str(r.run_id): str(r.macro_regime or "NEUTRAL").upper()
              for r in runs.itertuples(index=False)} if runs is not None else {}
    df["regime"] = df["run_id"].astype(str).map(regime).fillna("NEUTRAL")
    df["thr"] = df["regime"].map(_REGIME_THRESHOLD).fillna(0.85)

    need: dict = {}
    for tk, day in zip(df["ticker"], df["day"]):
        need.setdefault(tk, set()).add(day)
    feats = _pit_ohlcv_features(list(need), {k: sorted(v) for k, v in need.items()})
    keys = list(zip(df["ticker"], df["day"]))
    df["runup"] = [feats.get(k, {}).get("runup", np.nan) for k in keys]
    df["adv"] = [feats.get(k, {}).get("adv", np.nan) for k in keys]
    df["px"] = [feats.get(k, {}).get("close", np.nan) for k in keys]
    df["px"] = df["px"].fillna(pd.to_numeric(df.get("price"), errors="coerce"))

    # CONFIDENCE MUST COME FROM THE RAW `signals` TABLE, NOT THE PANEL.
    # build_panel masks confidence to NaN wherever CONFIDENCE_EPOCH says the
    # value was produced by superseded code -- 90.5% of rows in the current
    # window. `conf < threshold` is then False everywhere and Gate 1 rejects
    # NOTHING while looking like it ran: the biggest gate in the cascade
    # silently disappears. (Observed exactly that on the first run: Gate 1
    # dropped 0 of 20,381.)
    #
    # Reading it unmasked is sound HERE because this is descriptive -- it
    # measures what the gate did, and never feeds a weight or a calibration.
    # Same precedent as `news_events`, which reads news unmasked with an era
    # column for exactly this reason. The masked share is reported so a mixed
    # era is visible rather than assumed away: confidence was rescaled by the
    # rank basis, so pre-epoch values are not on today's scale.
    raw_conf = repo.fetch_df("""
        SELECT signal_date, ticker, confidence, n_methods_agreeing FROM (
          SELECT signal_date, ticker, confidence, n_methods_agreeing,
                 row_number() OVER (PARTITION BY signal_date, ticker
                                    ORDER BY generated_at DESC) AS _rn
          FROM signals WHERE confidence IS NOT NULL) WHERE _rn = 1
    """)
    masked_share = float(pd.to_numeric(df.get("confidence"), errors="coerce").isna().mean())
    if raw_conf is not None and not raw_conf.empty:
        raw_conf["day"] = raw_conf["signal_date"].astype(str).str[:10]
        df = df.merge(raw_conf[["day", "ticker", "confidence", "n_methods_agreeing"]]
                      .rename(columns={"confidence": "_conf_raw",
                                       "n_methods_agreeing": "_agree_raw"}),
                      on=["day", "ticker"], how="left")
        conf = pd.to_numeric(df["_conf_raw"], errors="coerce")
        agree = pd.to_numeric(df["_agree_raw"], errors="coerce")
    else:
        conf = pd.to_numeric(df.get("confidence"), errors="coerce")
        agree = pd.to_numeric(df.get("n_methods_agreeing"), errors="coerce")
    # A row with no confidence at all cannot be judged by Gate 1; excluding it
    # is honest, silently passing it is not.
    df = df[conf.notna()].copy()
    conf, agree = conf[conf.notna()], agree.reindex(df.index)
    if df.empty:
        return pd.DataFrame()

    # The cascade, in order. Each condition is evaluated only where nothing
    # earlier already rejected the row, so `gate` records the FIRST rejection.
    gate = pd.Series("pass", index=df.index, dtype=object)
    open_ = pd.Series(True, index=df.index)

    def _reject(mask, label):
        nonlocal open_
        hit = open_ & mask.fillna(False)
        gate[hit] = label
        open_ = open_ & ~hit

    _reject(conf < df["thr"], "below_threshold")
    if settings.enable_agreement_gate:
        _reject(agree < float(settings.min_sources_agreeing_gate), "low_agreement")
    else:
        # Retired 2026-08-17. Simulated anyway (that is how you re-evaluate a
        # retired gate) but flagged, so a row of zeros is not read as "never fires".
        _reject(pd.Series(False, index=df.index), "low_agreement")
    _reject((df["action"] == "BUY") & df["regime"].isin(["PANIC"]), "buy_blocked")
    # Gate 3 not simulable -- no historical earnings calendar. Nothing rejected.
    if settings.enable_trade_liquidity_gate:
        thin = (df["px"] < float(settings.trade_min_price)) | \
               (df["adv"] < float(settings.trade_min_dollar_volume)) | df["adv"].isna()
        _reject(thin, "untradeable")
    if settings.enable_overextension_gate:
        _reject((df["action"] == "BUY") &
                (df["runup"] > float(settings.overextension_runup_pct)), "overextended")

    df["gate"] = gate
    df.attrs["drift"] = float(df["ret"].mean())
    df.attrs["days"] = int(df["day"].nunique())
    df.attrs["d0"], df.attrs["d1"] = str(df["day"].min()), str(df["day"].max())
    df.attrs["simulated"] = True
    df.attrs["conf_masked_pct"] = round(masked_share * 100.0, 1)
    return df


def compute_gate_performance(days: Optional[int] = None,
                             source: str = "stamped") -> dict:
    """``{rows: DataFrame, meta: dict}`` -- one row per gate, in cascade order.

    Each gate is evaluated only on the candidates STILL ALIVE when it runs, so
    the cohorts partition and the funnel reads top to bottom.
    """
    rec = (simulate_gate_calls(days=days) if source == "simulated"
           else load_gate_calls(days=days))
    if rec is None or rec.empty:
        return {"rows": pd.DataFrame(), "meta": {}}
    drift = rec.attrs.get("drift", 0.0)

    rows, remaining = [], rec.copy()
    for key, label in GATE_STAGES:
        seen = remaining
        dropped = seen[seen["gate"] == key]
        survived = seen[seen["gate"] != key]
        d, s = _cohort_stats(dropped, drift), _cohort_stats(survived, drift)
        ic, icir, ic_days = _gate_ic(seen, key)
        rows.append({
            "stage": label, "seen": int(len(seen)), "dropped": d["n"],
            "drop_win": d["win"], "drop_exc": d["exc"], "drop_t": d["t"],
            "kept": s["n"], "keep_win": s["win"], "keep_exc": s["exc"],
            # A gate's VALUE is what it adds to the surviving book: the excess it
            # keeps minus the excess it threw away. Positive = the drop was right.
            "value": (s["exc"] - d["exc"]) if (d["n"] and s["n"]) else np.nan,
            "ic": ic, "icir": icir, "ic_days": ic_days,
        })
        remaining = survived

    passed = rec[rec["gate"] == "pass"]
    for label, frame in (("-> PASS (traded)", passed),
                         ("(reference) every BUY/SELL, gates off", rec)):
        st = _cohort_stats(frame, drift)
        rows.append({"stage": label, "seen": int(len(rec)), "dropped": np.nan,
                     "drop_win": np.nan, "drop_exc": np.nan, "drop_t": np.nan,
                     "kept": st["n"], "keep_win": st["win"], "keep_exc": st["exc"],
                     "value": np.nan, "ic": np.nan, "icir": np.nan, "ic_days": 0})

    meta = {"calls": int(len(rec)), "days": rec.attrs.get("days", 0),
            "source": source,
            "unsimulated": list(_SIM_UNSIMULATED) if source == "simulated" else [],
            "conf_masked_pct": rec.attrs.get("conf_masked_pct"),
            "d0": rec.attrs.get("d0", ""), "d1": rec.attrs.get("d1", ""),
            "drift": drift, "passed": int(len(passed)),
            "cascade_value": ((_cohort_stats(passed, drift)["exc"]
                               - _cohort_stats(rec, drift)["exc"])
                              if len(passed) else np.nan)}
    return {"rows": pd.DataFrame(rows), "meta": meta}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Per-gate funnel on the pivot target")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--source", choices=("stamped", "simulated"),
                    default="stamped")
    a = ap.parse_args()
    res = compute_gate_performance(days=a.days, source=a.source)
    df, meta = res["rows"], res["meta"]
    if df.empty:
        print("No gate-stamped calls with a settled pivot label yet.")
        return
    print(f"{meta['calls']} gate-stamped BUY/SELL calls over {meta['days']} days "
          f"({meta['d0']} -> {meta['d1']}); population drift {meta['drift']:+.3f}%")
    print(df.to_string(index=False))
    print(f"\ncascade contribution vs trading every LLM call: "
          f"{meta['cascade_value']:+.3f} pp")
    print("Standing caveat: in the original study every |t| <= 1.3 -- read direction, "
          "not significance, and ignore any gate whose n is in the low tens.")


if __name__ == "__main__":
    main()
