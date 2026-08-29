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

**ARCHITECTURE PARITY (2026-08-20).** From the 2026-08-13 rank-basis cutover to
2026-08-20 this module silently kept reproducing the RETIRED absolute combine —
raw scores, the 0.5 divisor, the symmetric 0.15 band, no shaping, no family
factor — so "what would today's configuration have decided" was answered about
a configuration that no longer existed. It now dispatches on
``settings.method_score_basis`` exactly like the live combine and resolves
every architecture constant through the aggregator's OWN single-source helpers
(`_rank_transform_run` incl. payoff shaping, `_direction_bands`,
`_raw_confidence_scale`, `_confidence_from`, `compute_family_agreement`), so a
future combine change propagates here by construction. The rank transform is a
RUN-level cross-section, so `run_backtest` groups rows per `run_id` and ranks
within the run over the cache-only Gate-4 tradeable pool (per-row stored
`price`; ADV from the OHLCV cache tail — the same mild anachronism the live
cache-only pool carries). Rows are stamped ``<arch>|<weights>`` (`combine_arch`:
``rank-v1`` / ``abs-v1``) and the weight fingerprint now hashes the bands,
divisor and shaped curves too — rows from different architectures must never
be pooled, and the tag makes that mechanical. Shaping curves and per-side
bands are TODAY's in both walk-forward modes (they live outside
`weight_history`); the wf mode is therefore "current CODE + the weight
calibrations available at the time", the closest out-of-sample approximation
of the current system that stored history permits.

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
        # The entry ARCHITECTURE is part of what a backtest ran under — two
        # runs differing only in basis/bands/shaping are different strategies.
        from src.signals.aggregator import (_direction_bands,
                                            _raw_confidence_scale)
        payload["arch"] = combine_arch()
        payload["bands"] = [round(b, 6) for b in _direction_bands("weighted")]
        payload["divisor"] = round(float(_raw_confidence_scale("weighted")), 6)
        if payload["arch"].startswith("rank"):
            try:
                from src.signals.rank_shaping import get_rank_shapes
                shapes = get_rank_shapes() or {}
                payload["shapes"] = {m: [round(float(x), 4) for x in c]
                                     for m, c in sorted(shapes.items())}
            except Exception:
                payload["shapes"] = None
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


def combine_arch() -> str:
    """The entry-architecture tag stamped into every backtest row's weight_set.

    ``rank-v1`` = the 2026-08-13+ production combine: centered within-run ranks
    through the payoff-shaped curves, per-side direction bands, the rank
    confidence divisor, and the family breadth factor. ``abs-v1`` = the
    pre-rank absolute combine (still reachable via
    ``method_score_basis="absolute"``, same as live). Rows from different
    architectures answer different questions and must never be pooled — the
    tag is what makes that mechanical."""
    basis = str(getattr(settings, "method_score_basis", "rank")).lower()
    return "rank-v1" if basis == "rank" else "abs-v1"


def backtest_row(scores: Dict[str, float], weights: Dict[str, float],
                 buy_filtered=(), sell_filtered=(),
                 buy_mults=None, sell_mults=None,
                 movement_factor: Optional[float] = None,
                 tape_score: Optional[float] = None,
                 vol_ratio: Optional[float] = None,
                 abstained=frozenset()) -> Dict[str, float]:
    """Recompute the derived layer for one ticker-day under ``weights``.

    ``scores`` maps method -> score. On the RANK architecture the caller has
    already run the cross-section through ``_rank_transform_run`` (the ranks
    are a RUN-level quantity, not a row-level one), so ``scores`` here are the
    transformed values and ``abstained`` carries the methods too thin to rank
    this run — weight 0, excluded from coherence and family votes, exactly the
    live idiom. On the ABSOLUTE architecture ``scores`` are raw and
    ``abstained`` is empty.

    Direction bands and the raw-confidence divisor resolve through the
    aggregator's OWN helpers (`_direction_bands` / `_raw_confidence_scale`) —
    the same single-source-of-truth rule the cross-sectional overlay follows,
    and the reason this module cannot silently reproduce a retired
    architecture again (it did exactly that from 2026-08-13 to 2026-08-20:
    every row used the absolute 0.5 divisor and the symmetric 0.15 band while
    production ranked and shaped).

    The OHLCV-derived context (``movement_factor``/``tape_score``/``vol_ratio``)
    comes from tier 1 and is a genuine recovery; everything computed here is
    not. ``sector_conf_factor`` is NOT reproduced (the sector-alignment pass
    needs sector-ETF context that is not stored per ticker-day) — the live
    default of 1.0 is used, so a backtested confidence can sit above a stored
    one that took the 0.75 sector penalty.
    """
    from src.signals.aggregator import (_coherence_factor, _confidence_from,
                                        _direction_bands, _raw_confidence_scale,
                                        _volume_factor, combine_buy_sell)

    # method_score_map: method -> (active, score), matching the live contract.
    # Abstained methods keep weight 0 (the win-rate-filter idiom): out of the
    # combine, out of coherence, out of family votes.
    msm = {m: (True, float(s)) for m, s in scores.items()
           if s is not None and s == s and m in weights and m not in abstained}
    if not msm:
        return {}

    buy, sell = combine_buy_sell(msm, weights, buy_filtered, sell_filtered,
                                 buy_mults, sell_mults)
    combined = buy - sell

    # Coherence reads each method's score against the combined direction, on
    # the SAME map the combine consumed (rank basis => the transformed values;
    # the 2026-08-13 stream-separation contract). It takes a SEQUENCE of
    # (enabled, score) pairs, not the method->pair map the combine takes;
    # passing the map iterates its KEYS and silently feeds it strings.
    coherence_ratio, coherence = _coherence_factor(combined, list(msm.values()))

    raw_conf = min(1.0, abs(combined) / _raw_confidence_scale("weighted"))
    out = {
        "combined_buy_score": round(buy, 4),
        "combined_sell_score": round(sell, 4),
        "combined_score": round(combined, 4),
        "raw_confidence": round(raw_conf, 4),
        "coherence_factor": round(coherence, 4),
    }

    mv = vf = 1.0
    if movement_factor is not None and movement_factor == movement_factor:
        mv = float(movement_factor)
        out["movement_factor"] = round(mv, 4)
    if vol_ratio is not None and vol_ratio == vol_ratio:
        vf = float(_volume_factor(float(vol_ratio), abs(combined), coherence_ratio))
        out["volume_factor"] = round(vf, 4)
    tf = 1.0
    if tape_score is not None and tape_score == tape_score:
        from src.signals.agreement import TapeCheck, tape_factor
        tf = float(tape_factor(TapeCheck(score=float(tape_score), label="REPLAY"),
                               combined, float(settings.tape_confirmation_factor_span)))
        out["tape_conf_factor"] = round(tf, 4)

    # Family breadth factor — part of the live confidence chain since 2026-07,
    # previously omitted here (a backtested confidence silently lacked one of
    # the six multipliers). Votes run over the same effective map as the
    # combine; abstained/filtered methods are already out of `msm`.
    ff = 1.0
    try:
        from src.signals.agreement import compute_family_agreement
        eff = {m: s for m, (on, s) in msm.items() if on and s == s and s != 0.0}
        fam = compute_family_agreement(eff, combined,
                                       float(settings.family_vote_threshold))
        ff = float(fam.factor(float(settings.family_agreement_factor_span)))
        out["family_conf_factor"] = round(ff, 4)
    except Exception:
        pass

    out["confidence"] = round(min(1.0, _confidence_from(raw_conf, coherence, mv,
                                                        vf, ff, tf)), 4)
    long_band, short_band = _direction_bands("weighted")
    out["direction"] = ("BULLISH" if combined >= long_band
                        else "BEARISH" if combined <= -short_band else "NEUTRAL")
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
    sql = f"""SELECT s.run_id, s.signal_date, s.ticker, s.generated_at, s.price,
                     {', '.join(f's.{c}' for c in stored_cols)},
                     {rep_cols},
                     r.movement_factor AS rp_movement_factor,
                     r.tape_score AS rp_tape_score,
                     r.vol_ratio AS rp_vol_ratio
              FROM signals s
              LEFT JOIN signals_replay r
                ON s.signal_date = r.signal_date AND s.ticker = r.ticker
               AND s.generated_at = r.generated_at
              {where} ORDER BY s.run_id, s.ticker"""
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = repo.fetch_df(sql, params)
    if rows is None or rows.empty:
        return pd.DataFrame()
    # Production rows always carry run_id/price; a caller-supplied frame (tests,
    # ad-hoc reruns) may not. The run is the rank transform's cross-section, so
    # fall back to generated_at — the run-exact key replay itself joins on.
    if "run_id" not in rows.columns:
        rows = rows.assign(run_id=rows["generated_at"])
    if "price" not in rows.columns:
        rows = rows.assign(price=None)

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
    arch = combine_arch()
    rank_basis = arch.startswith("rank")
    stamped = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Cache-only 20d ADV per ticker for the tradeable rank pool (Gate-4 floors,
    # same numbers `rank_tradeable_only` judges against; the stored per-row
    # `price` is as-of, the ADV is the cache's current tail — the same mild
    # anachronism the live cache-only pool carries, stated rather than hidden).
    _adv_memo: Dict[str, float] = {}

    def _adv(tk: str) -> float:
        if tk not in _adv_memo:
            val = 0.0
            try:
                from src.data.cache import load_ohlcv
                d = load_ohlcv(tk)
                if d is not None and not d.empty and "Volume" in d.columns:
                    c = pd.to_numeric(d["Close"], errors="coerce")
                    v = pd.to_numeric(d["Volume"], errors="coerce")
                    dv = (c * v).tail(20).mean()
                    val = float(dv) if dv == dv else 0.0
            except Exception:
                val = 0.0
            _adv_memo[tk] = val
        return _adv_memo[tk]

    use_pool = (rank_basis and bool(getattr(settings, "rank_tradeable_only", True))
                and bool(getattr(settings, "enable_trade_liquidity_gate", True)))
    min_px = float(getattr(settings, "trade_min_price", 5.0))
    min_dv = float(getattr(settings, "trade_min_dollar_volume", 5e6))

    # Walk-forward SHAPES (2026-08-21): in wf mode each run consumes the curves
    # calibrated STRICTLY BEFORE its date (shape_history), never today's — the
    # 2026-08-20 finding was that today's curves alone flip this backtest's
    # pivot IC from −0.041 to +0.049, pure in-sample. No earlier calibration =>
    # IDENTITY ({}), the honest state of a date before shaping existed. Non-wf
    # mode ("what would TODAY's config decide") keeps today's live curves.
    shape_hist = None
    if rank_basis and wf_hist is not None:
        try:
            from src.signals.rank_shaping import load_shape_history
            shape_hist = load_shape_history()
            if not shape_hist:
                logger.warning("[backtest] shape_history is empty — wf rows will "
                               "use IDENTITY curves; run `python -m "
                               "src.signals.rank_shaping --materialize` first")
        except Exception as e:
            logger.warning(f"[backtest] shape history unavailable ({e}) — identity")
            shape_hist = {}

    out: List[dict] = []
    for run_id, grp in rows.groupby("run_id", sort=True):
        recs = grp.to_dict("records")
        # Per-row scores, replayed value winning where it exists — it is the
        # current scorer's output; the stored one is used where replay cannot
        # reach (non-replayable methods are the stored = current-code values,
        # or epoch-superseded ones the caller must judge via the arch tag).
        per_tk: Dict[str, Dict[str, float]] = {}
        for r in recs:
            scores = {}
            for m in stored_cols:
                v = r.get(f"rp_{m}")
                if v is None or v != v:
                    v = r.get(m)
                if v is not None and v == v:
                    scores[m] = float(v)
            per_tk[str(r["ticker"])] = scores

        # Weights per DATE (a run maps to one signal date).
        d0 = str(recs[0]["signal_date"])[:10]
        w_row, bf, sf, bm, sm, tag = (weights, buy_filtered, sell_filtered,
                                      buy_mults, sell_mults, f"{arch}|{ws}")
        if wf_hist is not None:
            wf = _wf_for(d0)
            if wf is None:
                # No calibration existed yet for this date. Emitting today's
                # weights here would be exactly the look-ahead walk-forward
                # exists to remove, so the run is SKIPPED instead.
                continue
            w_row = wf["weights"]
            bf, sf = set(wf["buy_filtered"]), set(wf["sell_filtered"])
            bm, sm = wf["buy_mults"], wf["sell_mults"]
            tag = f"{arch}|wf:{wf['as_of']}"

        abstained = frozenset()
        if rank_basis:
            # The rank transform is a RUN-level cross-section: build the run's
            # raw maps over the weighted methods, resolve the tradeable pool,
            # and transform through the aggregator's OWN helper (shaping
            # included) so this module cannot fork the architecture.
            from src.signals.aggregator import _rank_transform_run
            run_shapes = None                        # non-wf: today's live curves
            if shape_hist is not None:
                from src.signals.rank_shaping import shapes_for_date
                run_shapes = shapes_for_date(d0, shape_hist)
                sh_tag = "none" if run_shapes is None else "asof"
                if run_shapes is None:
                    run_shapes = {}                  # pre-shaping date: identity
                tag = f"{tag}|sh:{sh_tag}"
            raw_maps = {tk: {m: (True, s) for m, s in sc.items() if m in w_row}
                        for tk, sc in per_tk.items()}
            tradeable = None
            if use_pool:
                pool = set()
                for r in recs:
                    px = r.get("price")
                    if px is not None and px == px and float(px) >= min_px \
                            and _adv(str(r["ticker"])) >= min_dv:
                        pool.add(str(r["ticker"]))
                # live fail-soft: a pool too thin to rank falls back to the
                # full universe rather than zeroing the book
                tradeable = pool if len(pool) >= 30 else None
            maps, abstained = _rank_transform_run(raw_maps, tradeable,
                                                  shapes=run_shapes)
            per_tk = {tk: {m: float(s) for m, (on, s) in mm.items() if on}
                      for tk, mm in maps.items()}

        for r in recs:
            got = backtest_row(per_tk.get(str(r["ticker"]), {}), w_row, bf, sf,
                               bm, sm,
                               movement_factor=r.get("rp_movement_factor"),
                               tape_score=r.get("rp_tape_score"),
                               vol_ratio=r.get("rp_vol_ratio"),
                               abstained=abstained)
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
