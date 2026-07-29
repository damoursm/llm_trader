"""Tier 2 — recompute `combined_score` and the full confidence over history.

**This is a BACKTEST, not a recovery, and the distinction is the whole design.**

Tier 1 (`replay.py`) regenerates values that do NOT depend on the weights:
method scores, and the one confidence component (`movement_factor`) built from
raw OHLCV. Those are recoveries — the current code's honest output on inputs the
run actually had — so calibrations may fit them.

Everything here is different. `combined_score` is a weighted average, and
`raw_confidence` / `coherence_factor` / `family_conf_factor` / `volume_factor` /
`tape_conf_factor` all take it (or the weights) as an input. Recomputing them
with TODAY's weights answers "what would today's configuration have decided?" —
a legitimate and useful question, and NOT the same question as "what was true at
the time".

**The circularity, stated precisely.** Today's weights are calibrated FROM the
signals panel. If a backtested `combined_score` fed back into weight
calibration, the weights would be fitted on values derived from themselves —
future information laundered into the dataset that drives live sizing and
gating. That is not a small bias: it is self-confirmation, and it would look
like the configuration improving.

So the firewall is structural, not a convention:

  * writes to its OWN table, `signals_backtest` — never `signals`, never
    `signals_replay`;
  * `signal_panel.build_panel` does NOT read it (tier 1's `restore_replayed`
    reads `signals_replay` only), so no calibration can reach it by accident;
  * every row is stamped with `weight_set` — a hash of the weights actually
    used — so a backtest can never be mistaken for a different configuration's;
  * `tests/test_backtest.py` asserts no calibration module imports this one.

Read it to EVALUATE a configuration. Never to FIT one.

Non-OHLCV method scores (news, insider, the options family, pead, massive) come
from the stored `signals` row, which is correct: their scorers are unversioned,
so the stored value IS what today's code produces. Only the OHLCV six are
replayed, and only `money_flow` ever changed.

**SCOPE — what this recomputes, and what it does NOT.** It reproduces the
weighted buy/sell camp combine and the confidence chain. It does NOT apply the
ADDITIVE overlays that the live path lands on top of `combined` — the
corp-action/fundamental `f_*` factors, the `kaufman_*`/`adx_*` trend pair, the
`cross_sectional` overlay — nor `_interaction_adjustment`, nor the
cross-sectional rescale pass. Those are sparse, event-driven, and several need
context (fundamentals, corporate actions) that is not stored per ticker-day, so
including them would mean inventing inputs. Consequence to keep in view: a
backtested `combined_score` is the CAMP DIFFERENCE only, and will differ from a
live value by the overlay term. Treat the direction and the relative ordering as
the signal here, not the exact magnitude.

**FIRST READ (2026-07-28, 71,784 comparable rows, weight_set f3229c1cfcd5).**
Headline "47.7% direction agreement" is misleading and should not be quoted
alone. The crosstab shows the two configurations rarely CONTRADICT — flat
opposites (bull vs bear either way) are just 7.0% — and mostly differ in
DECISIVENESS: today's configuration turns ~23,500 live NEUTRALs into directional
calls, taking the NEUTRAL share from 51.8% to 31.6%. The mechanism is the
market-relative hard filter, which drops 12 of 21 weighted methods, leaving 10
active: fewer members per camp means less dilution, so |combined| clears the
0.15 band far more often (mean confidence 0.57 backtested vs 0.38 as run). That
is a real property of the current configuration worth knowing — it is markedly
more willing to take a side than the book that produced this history.

CLI:  python -m src.analysis.backtest --days 30 --write
      python -m src.analysis.backtest --compare        # backtest vs what ran
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from config.settings import settings

# Columns this module produces. Deliberately NOT added to any panel constant —
# nothing in the calibration path should be able to pick them up by iterating a
# shared list.
BACKTEST_COLUMNS = (
    "combined_buy_score", "combined_sell_score", "combined_score",
    "raw_confidence", "coherence_factor", "movement_factor", "volume_factor",
    "family_conf_factor", "tape_conf_factor", "confidence", "direction",
)


def weight_set_hash() -> str:
    """Short, stable fingerprint of the weights a backtest ran under.

    Two backtests are comparable only if this matches. Without it a stored row
    is uninterpretable — "today's weights" means nothing once today has passed.
    """
    from src.signals.aggregator import _BASE_WEIGHTS
    try:
        from src.signals.aggregator import (_inverted_methods,
                                            winrate_filtered_methods)
        payload = {
            "base": {k: round(float(v), 6) for k, v in sorted(_BASE_WEIGHTS.items())},
            "inverted": sorted(_inverted_methods()),
            "filtered": sorted(winrate_filtered_methods()),
            "diff_threshold": round(float(settings.buy_sell_diff_threshold), 6),
        }
    except Exception as e:                      # weights unavailable => say so
        logger.warning(f"[backtest] weight fingerprint degraded: {e}")
        payload = {"base": {k: round(float(v), 6)
                            for k, v in sorted(_BASE_WEIGHTS.items())}}
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _effective_weights() -> Dict[str, float]:
    """Today's weight vector, inversion signs applied — the same construction
    the live combine uses, read through the aggregator's own helpers so this
    cannot drift into a private copy of the weighting stack."""
    from src.signals.aggregator import _BASE_WEIGHTS, _inverted_methods
    inv = set(_inverted_methods())
    return {m: (-w if m in inv else w) for m, w in _BASE_WEIGHTS.items()}


def backtest_row(scores: Dict[str, float], weights: Dict[str, float],
                 buy_filtered=(), sell_filtered=(),
                 buy_mults=None, sell_mults=None,
                 movement_factor: Optional[float] = None,
                 tape_score: Optional[float] = None,
                 vol_ratio: Optional[float] = None) -> Dict[str, float]:
    """Recompute the derived layer for one ticker-day under ``weights``.

    ``scores`` maps method -> score (replayed where possible, stored otherwise).
    The OHLCV-derived context (``movement_factor``/``tape_score``/``vol_ratio``)
    comes from tier 1 and is a genuine recovery; everything computed here is not.
    """
    from src.signals.aggregator import (_coherence_factor, _volume_factor,
                                        combine_buy_sell)

    # method_score_map: method -> (active, score), matching the live contract.
    msm = {m: (True, float(s)) for m, s in scores.items()
           if s is not None and s == s and m in weights}
    if not msm:
        return {}

    buy, sell = combine_buy_sell(msm, weights, buy_filtered, sell_filtered,
                                 buy_mults, sell_mults)
    combined = buy - sell

    # Coherence reads each method's RAW score against the combined direction —
    # the same documented (and deliberate) convention as the live path. It takes
    # a SEQUENCE of (enabled, score) pairs, not the method->pair map the combine
    # takes; passing the map iterates its KEYS and silently feeds it strings.
    coherence_ratio, coherence = _coherence_factor(combined, list(msm.values()))

    raw_conf = min(1.0, abs(combined) / 0.5)
    conf = raw_conf * coherence
    out = {
        "combined_buy_score": round(buy, 4),
        "combined_sell_score": round(sell, 4),
        "combined_score": round(combined, 4),
        "raw_confidence": round(raw_conf, 4),
        "coherence_factor": round(coherence, 4),
    }
    if movement_factor is not None and movement_factor == movement_factor:
        out["movement_factor"] = round(float(movement_factor), 4)
        conf *= float(movement_factor)
    if vol_ratio is not None and vol_ratio == vol_ratio:
        vf = _volume_factor(float(vol_ratio), abs(combined), coherence_ratio)
        out["volume_factor"] = round(float(vf), 4)
        conf *= float(vf)
    if tape_score is not None and tape_score == tape_score:
        from src.signals.agreement import TapeCheck, tape_factor
        tf = tape_factor(TapeCheck(score=float(tape_score), label="REPLAY"),
                         combined, float(settings.tape_confirmation_factor_span))
        out["tape_conf_factor"] = round(float(tf), 4)
        conf *= float(tf)

    thr = float(settings.buy_sell_diff_threshold)
    out["confidence"] = round(min(1.0, conf), 4)
    out["direction"] = ("BULLISH" if combined >= thr
                        else "BEARISH" if combined <= -thr else "NEUTRAL")
    return out


def run_backtest(days: Optional[int] = None,
                 limit: Optional[int] = None,
                 walk_forward: bool = True) -> pd.DataFrame:
    """Recompute the derived layer across history.

    ``walk_forward=True`` (default) resolves the weight set PER SIGNAL DATE from
    `weight_history` — the calibration the system would have had then, fitted on
    strictly-earlier data. That is the production-faithful mode and the only one
    whose output is out-of-sample: weights at D cannot encode D's outcome.

    ``walk_forward=False`` applies TODAY's weights to everything. Useful for the
    narrow question "what would the CURRENT configuration do", but it is not a
    simulation of anything that could have happened — measured over this
    history, today's config keeps 10 methods while the system genuinely ran with
    18 in late June, so a fixed-weight backtest misstates the early period
    wholesale. It falls back to this mode automatically when `weight_history` is
    empty, and says so.
    """
    from src.db import repo
    from src.db.schema import (REPLAYABLE_METHOD_COLUMNS, SIGNAL_BASE_METHOD_COLUMNS)
    from src.signals.aggregator import (side_filtered_methods,
                                        side_weight_multipliers,
                                        winrate_filtered_methods)

    weights = _effective_weights()
    dropped = set(winrate_filtered_methods())
    for m in dropped:
        weights.pop(m, None)
    try:
        buy_filtered = set(side_filtered_methods("buy"))
        sell_filtered = set(side_filtered_methods("sell"))
        buy_mults = side_weight_multipliers("buy")
        sell_mults = side_weight_multipliers("sell")
    except Exception as e:
        logger.warning(f"[backtest] per-side layers unavailable: {e}")
        buy_filtered = sell_filtered = set()
        buy_mults = sell_mults = {}

    stored_cols = [c for c in SIGNAL_BASE_METHOD_COLUMNS]
    where, params = "", []
    if days:
        where = ("WHERE s.signal_date >= "
                 "(CURRENT_DATE - INTERVAL (?) DAY)::VARCHAR")
        params = [int(days)]
    rep_cols = ", ".join(f"r.{m} AS rp_{m}" for m in REPLAYABLE_METHOD_COLUMNS)
    sql = f"""SELECT s.signal_date, s.ticker, s.generated_at,
                     {', '.join(f's.{c}' for c in stored_cols)},
                     {rep_cols},
                     r.movement_factor AS rp_movement_factor,
                     r.tape_score AS rp_tape_score,
                     r.vol_ratio AS rp_vol_ratio
              FROM signals s
              LEFT JOIN signals_replay r
                ON s.signal_date = r.signal_date AND s.ticker = r.ticker
               AND s.generated_at = r.generated_at
              {where} ORDER BY s.signal_date, s.ticker"""
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = repo.fetch_df(sql, params)
    if rows is None or rows.empty:
        return pd.DataFrame()

    # Walk-forward: the weight state in force for each signal_date, resolved
    # once per DATE (not per row — a calibration step covers a whole day).
    wf_hist = None
    if walk_forward:
        from src.analysis.walkforward import load_weight_history, weights_for_date
        wf_hist = load_weight_history()
        if wf_hist is None or wf_hist.empty:
            logger.warning("[backtest] weight_history is empty — falling back to "
                           "TODAY's weights for ALL rows. That is NOT "
                           "production-faithful; run `python -m "
                           "src.analysis.walkforward --write` first.")
            wf_hist = None
    _wf_cache: Dict[str, Optional[dict]] = {}

    def _wf_for(d: str):
        if d not in _wf_cache:
            _wf_cache[d] = weights_for_date(d, wf_hist)
        return _wf_cache[d]

    ws = weight_set_hash()
    stamped = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out: List[dict] = []
    for r in rows.to_dict("records"):
        scores = {}
        for m in stored_cols:
            # Replayed value wins where it exists — it is the current scorer's
            # output; the stored one is only used where replay cannot reach.
            v = r.get(f"rp_{m}")
            if v is None or v != v:
                v = r.get(m)
            if v is not None and v == v:
                scores[m] = float(v)
        w_row, bf, sf, bm, sm, tag = (weights, buy_filtered, sell_filtered,
                                      buy_mults, sell_mults, ws)
        if wf_hist is not None:
            wf = _wf_for(str(r["signal_date"])[:10])
            if wf is None:
                # No calibration existed yet for this date. Emitting today's
                # weights here would be exactly the look-ahead walk-forward
                # exists to remove, so the row is SKIPPED instead.
                continue
            w_row = wf["weights"]
            bf, sf = set(wf["buy_filtered"]), set(wf["sell_filtered"])
            bm, sm = wf["buy_mults"], wf["sell_mults"]
            tag = f"wf:{wf['as_of']}"

        got = backtest_row(scores, w_row, bf, sf, bm, sm,
                           movement_factor=r.get("rp_movement_factor"),
                           tape_score=r.get("rp_tape_score"),
                           vol_ratio=r.get("rp_vol_ratio"))
        if not got:
            continue
        rec = {"signal_date": r["signal_date"], "ticker": r["ticker"],
               "generated_at": r["generated_at"], "weight_set": tag,
               "computed_at": stamped}
        rec.update(got)
        out.append(rec)
    return pd.DataFrame(out)


def materialize(days: Optional[int] = None) -> int:
    """Write the backtest to `signals_backtest`, replacing the same span."""
    from src.db.connection import connect

    df = run_backtest(days=days)
    if df.empty:
        logger.info("[backtest] nothing to write")
        return 0
    cols = (["signal_date", "ticker", "generated_at", "weight_set", "computed_at"]
            + [c for c in BACKTEST_COLUMNS if c in df.columns])
    df = df[cols]
    with connect() as con:
        if days:
            con.execute("DELETE FROM signals_backtest WHERE signal_date >= "
                        "(CURRENT_DATE - INTERVAL (?) DAY)::VARCHAR", [int(days)])
        else:
            con.execute("DELETE FROM signals_backtest")
        con.register("_bt_df", df)
        con.execute(f"INSERT INTO signals_backtest ({', '.join(cols)}) "
                    f"SELECT {', '.join(cols)} FROM _bt_df")
        con.unregister("_bt_df")
    logger.info(f"[backtest] wrote {len(df):,} rows under weight_set={df['weight_set'].iloc[0]}")
    return len(df)


def compare(days: Optional[int] = None) -> pd.DataFrame:
    """Backtest vs what actually ran, on the rows where both are comparable.

    Only rows AT OR AFTER the confidence epoch can be compared at all — before
    it the stored confidence is on a different scale, which is the whole reason
    it is masked. Rows before the epoch are reported separately as "backtest
    only": there is a number, but nothing legitimate to compare it against.
    """
    from src.db import repo
    from src.signals.method_epochs import confidence_epoch

    bt = run_backtest(days=days)
    if bt.empty:
        return pd.DataFrame()
    st = repo.fetch_df("SELECT signal_date, ticker, generated_at, combined_score, "
                       "confidence, direction FROM signals")
    for d in (bt, st):
        for c in ("signal_date", "generated_at"):
            d[c] = d[c].astype(str)
    m = bt.merge(st, on=["signal_date", "ticker", "generated_at"],
                 how="inner", suffixes=("_bt", "_live"))
    cep = confidence_epoch()
    rows = []
    for label, sub in (("comparable (post confidence-epoch)",
                        m[m["signal_date"] >= cep.isoformat()] if cep is not None else m),
                       ("backtest only (pre-epoch, live value masked)",
                        m[m["signal_date"] < cep.isoformat()] if cep is not None else m.iloc[0:0])):
        if sub.empty:
            continue
        agree = (sub["direction_bt"] == sub["direction_live"]).mean() * 100
        # Raw agreement conflates "contradicts" with "is more decisive", and the
        # two mean opposite things. A flat opposite is a genuine disagreement; a
        # NEUTRAL that became directional is the configuration taking a side the
        # old one declined. Report them separately or the headline misleads.
        opp = (((sub["direction_bt"] == "BULLISH") & (sub["direction_live"] == "BEARISH"))
               | ((sub["direction_bt"] == "BEARISH") & (sub["direction_live"] == "BULLISH"))
               ).mean() * 100
        rows.append({
            "cohort": label, "n": len(sub),
            "neutral_bt_pct": round(float((sub["direction_bt"] == "NEUTRAL").mean() * 100), 1),
            "neutral_live_pct": round(float((sub["direction_live"] == "NEUTRAL").mean() * 100), 1),
            "opposite_pct": round(float(opp), 1),
            "mean_combined_bt": round(float(sub["combined_score_bt"].mean()), 4),
            "mean_combined_live": round(float(sub["combined_score_live"].mean()), 4),
            "mean_conf_bt": round(float(sub["confidence_bt"].mean()), 4),
            "mean_conf_live": round(float(sub["confidence_live"].mean()), 4),
            "direction_agree_pct": round(float(agree), 1),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(
        description="Recompute combined_score + confidence under TODAY's weights "
                    "(a BACKTEST — never feed this to a calibration)")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()

    if a.write:
        print(f"wrote {materialize(days=a.days):,} rows")
    else:
        rep = compare(days=a.days)
        print(f"\nweight_set = {weight_set_hash()}\n")
        print(rep.to_string(index=False) if not rep.empty else "no rows")
        print("\nThis is what TODAY's configuration would have decided. It is NOT")
        print("what happened, and it must never be fed back into weight fitting.")
