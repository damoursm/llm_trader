"""Monte Carlo overfitting evaluation — luck-vs-skill for methods, the win-rate
filter's selection bias, and exit-rule timing vs random exits.

Every learning surface in this system selects or tilts on HISTORICAL performance
(the win-rate method filter, adaptive weights, exit-floor calibrations). With
~50–120-trade samples, a 51% win rate is statistically indistinguishable from a
coin flip — selection on such samples is the textbook overfitting mechanism.
This module quantifies that risk with three Monte Carlo tests:

1.  ``method_overfit_mc`` — per-method LUCK-VS-SKILL over the closed ledger's
    solo directional calls (same gross basis as the win-rate filter:
    ``sign(score)·(exit−entry)``, pre-cost):
      • bootstrap (resample trades with replacement) → 5–95% CI on the gross win
        rate and mean oriented return — how wide is the evidence?
      • sign-permutation null (each trade's direction call replaced by a fair
        coin, i.e. its oriented return's sign flipped with p=½) → one-sided
        p-value: the probability a NO-SKILL method with this trade count and
        these |moves| would look at least this good by luck.
    A method whose p ≥ 0.05 has a track record consistent with noise — keeping
    or dropping it on that record is a coin-flip decision either way.

2.  ``filter_selection_mc`` — the ≥50% win-rate filter's SELECTION BIAS: run the
    whole filter on synthetic no-skill methods (wins ~ Binomial(n_i, ½) at each
    method's REAL trade count) many times. Reports how many methods pure chance
    would KEEP at these sample sizes vs how many the live filter kept — if the
    two are similar, the kept set is not yet evidence of skill (expect churn as
    samples grow; don't compound decisions on top of it).

3.  ``exit_timing_mc`` — per exit rule (``exit_reason``): does the rule TIME
    exits better than random? For each closed trade the feasible exit window is
    every session close from the first session after entry through the actual
    hold + ``EXIT_MC_EXTRA_SESSIONS`` more; the null policy draws one random
    exit per trade per simulation. Both arms anchor entry at the real entry
    fill and exit at SESSION CLOSES (the actual arm at the actual exit date's
    close) so the comparison is apples-to-apples; gross of costs (both arms
    would pay the same one-way exit cost). Percentile ≈ 50 ⇒ the rule adds no
    timing skill over random; ≪ 50 ⇒ random exits would have BEATEN the rule.

All tests are deterministic given the DB + OHLCV cache (fixed RNG seed) and
fail-soft (thin data → reported as such, never guessed). Small-sample p-values
move a lot per added trade — re-read as the ledger grows, like every other
learning surface here.

Usage:  python -m src.analysis.monte_carlo
"""

from __future__ import annotations

from bisect import bisect_right
from datetime import date
from typing import Dict, List, Optional, Sequence

import numpy as np

DEFAULT_SEED = 4242
DEFAULT_SIMS = 2000
EXIT_MC_EXTRA_SESSIONS = 10   # random exits may land up to this many sessions past the actual hold
_SIG_P = 0.05                 # significance bar for the luck-vs-skill verdicts


# ── shared extraction (mirrors tracker.compute_solo_method_gross_winrate) ────

def method_gross_returns(closed_trades: List[dict]) -> Dict[str, List[float]]:
    """Per-method list of GROSS oriented solo returns (%), one per closed trade
    where the method had a view: ``sign(score) × (exit−entry)/entry × 100``.
    Same population / no-view guard / gross basis as the win-rate filter, so the
    MC judges exactly the number the filter selects on."""
    from src.performance.tracker import _ALL_METHODS, _METHOD_AGREE_THRESHOLD
    out: Dict[str, List[float]] = {}
    for t in closed_trades:
        if t.get("status") != "CLOSED" or not t.get("method_scores"):
            continue
        try:
            entry = float(t.get("entry_price") or 0.0)
            exit_ = float(t.get("exit_price") or 0.0)
        except (TypeError, ValueError):
            continue
        if entry <= 0 or exit_ <= 0:
            continue
        move_pct = (exit_ - entry) / entry * 100.0
        for m in _ALL_METHODS:
            s = t["method_scores"].get(m, 0.0)
            if s == 0.0 or abs(s) < _METHOD_AGREE_THRESHOLD:
                continue
            out.setdefault(m, []).append((1.0 if s > 0 else -1.0) * move_pct)
    return out


# ── 1: per-method luck-vs-skill ──────────────────────────────────────────────

def method_overfit_mc(returns_by_method: Dict[str, List[float]],
                      n_sims: int = DEFAULT_SIMS, seed: int = DEFAULT_SEED,
                      min_trades: int = 5) -> List[dict]:
    """Bootstrap CI + permutation-null p-value per method (see module doc).

    Returns rows sorted by n desc:
      ``{method, n, win_rate, wr_lo, wr_hi, p_luck, mean_ret, ret_lo, ret_hi,
         p_ret, verdict}``
    ``p_luck``: P(coin-flip win rate ≥ observed) — the sign-permutation null on
    the DIRECTIONAL HIT; ``p_ret``: same null on the mean oriented return
    (respects the actual |move| distribution — a method that was right on the
    BIG moves scores better here than raw hit rate shows). One-sided by design:
    the question is "is it better than no skill", not "is it different".
    """
    rng = np.random.default_rng(seed)
    rows: List[dict] = []
    for m, rets in returns_by_method.items():
        r = np.asarray(rets, dtype=float)
        n = len(r)
        if n < max(1, min_trades):
            continue
        wins = r > 0
        wr = float(wins.mean() * 100.0)
        mean_ret = float(r.mean())

        # Bootstrap: resample the n outcomes with replacement, n_sims times.
        idx = rng.integers(0, n, size=(n_sims, n))
        boot = r[idx]
        boot_wr = (boot > 0).mean(axis=1) * 100.0
        boot_mean = boot.mean(axis=1)

        # Permutation null: a no-skill method makes a fair-coin direction call on
        # each of the SAME n moves — flip each oriented return's sign with p=½.
        flips = rng.choice([-1.0, 1.0], size=(n_sims, n))
        null = np.abs(r)[None, :] * flips
        null_wr = (null > 0).mean(axis=1) * 100.0
        null_mean = null.mean(axis=1)
        # +1 smoothing so an MC p-value is never exactly 0 (resolution ≈ 1/n_sims)
        p_luck = float((np.sum(null_wr >= wr) + 1) / (n_sims + 1))
        p_ret = float((np.sum(null_mean >= mean_ret) + 1) / (n_sims + 1))

        verdict = ("SKILL (p<%.2f)" % _SIG_P if p_luck < _SIG_P
                   else "worse than chance" if p_luck > (1 - _SIG_P)
                   else "indistinguishable from luck")
        rows.append({
            "method": m, "n": n,
            "win_rate": round(wr, 1),
            "wr_lo": round(float(np.percentile(boot_wr, 5)), 1),
            "wr_hi": round(float(np.percentile(boot_wr, 95)), 1),
            "p_luck": round(p_luck, 3),
            "mean_ret": round(mean_ret, 3),
            "ret_lo": round(float(np.percentile(boot_mean, 5)), 3),
            "ret_hi": round(float(np.percentile(boot_mean, 95)), 3),
            "p_ret": round(p_ret, 3),
            "verdict": verdict,
        })
    return sorted(rows, key=lambda x: -x["n"])


# ── 2: win-rate filter selection-bias null ───────────────────────────────────

def filter_selection_mc(trade_counts: Dict[str, int], kept_actual: int,
                        threshold: float = 0.50, min_trades: int = 10,
                        n_sims: int = DEFAULT_SIMS, seed: int = DEFAULT_SEED) -> dict:
    """How many methods would the ≥threshold filter KEEP under the all-coin-flips
    null, at the REAL per-method trade counts?

    ``trade_counts`` — {method: n solo trades} for the methods the filter can
    judge (n ≥ min_trades, not inverted); ``kept_actual`` — how many of those the
    live filter actually kept. A method below min_trades is kept unjudged either
    way, so it is excluded from both sides. Returns::

        {n_judgeable, kept_actual, kept_null_mean, kept_null_sd,
         kept_null_lo, kept_null_hi, p_ge_actual, verdict}

    ``p_ge_actual`` = P(chance keeps ≥ kept_actual). A LOW p says the kept set is
    larger than luck explains (evidence of real >50% methods in it); p ≈ 0.5 says
    the filter's selections at these sample sizes look exactly like chance —
    expect the kept/dropped assignment to churn as trades accrue."""
    ns = np.asarray([n for n in trade_counts.values() if n >= min_trades], dtype=int)
    if ns.size == 0:
        return {"n_judgeable": 0, "kept_actual": kept_actual, "verdict":
                "No methods have enough trades for the filter to judge yet."}
    rng = np.random.default_rng(seed)
    # wins_i ~ Binomial(n_i, ½) per sim; kept when WR ≥ threshold (the filter
    # drops on STRICT <, so exactly-threshold survives — mirror that).
    wins = rng.binomial(ns[None, :].repeat(n_sims, axis=0), 0.5)
    kept = ((wins / ns[None, :]) >= threshold).sum(axis=1)
    p_ge = float((np.sum(kept >= kept_actual) + 1) / (n_sims + 1))
    out = {
        "n_judgeable": int(ns.size),
        "kept_actual": int(kept_actual),
        "kept_null_mean": round(float(kept.mean()), 2),
        "kept_null_sd": round(float(kept.std()), 2),
        "kept_null_lo": int(np.percentile(kept, 5)),
        "kept_null_hi": int(np.percentile(kept, 95)),
        "p_ge_actual": round(p_ge, 3),
    }
    if p_ge < _SIG_P:
        out["verdict"] = ("kept set is LARGER than chance explains — evidence of real "
                          ">50% methods among the kept")
    elif (1 - p_ge) < _SIG_P:
        out["verdict"] = ("kept set is SMALLER than chance would produce — the dropped "
                          "set likely contains genuinely bad methods")
    else:
        out["verdict"] = ("kept/dropped split is consistent with pure chance at these "
                          "sample sizes — treat the filter's current selections as "
                          "provisional (expect churn as trades accrue)")
    return out


# ── 3: exit-rule timing vs random exits ──────────────────────────────────────

def _close_series(ticker: str) -> Dict[date, float]:
    """Seam mirroring ``exit_forward._close_series`` (tests inject synthetic)."""
    from src.performance.daily_nav import _load_close_series
    return _load_close_series(ticker) or {}


def _exit_mc_trade(t: dict, extra: int) -> Optional[dict]:
    """Feasible-exit-return vector for one CLOSED trade, or None if unanchorable.

    ``rets[k]`` = gross oriented return (%) exiting at the close of the k-th
    session AFTER entry (k=0 → first session after entry date); ``act_k`` = the
    actual exit date's index in that vector (close-anchored, see module doc)."""
    from src.analysis.exit_forward import _oriented_sign
    if t.get("status") != "CLOSED":
        return None
    sign = _oriented_sign(t)
    try:
        entry_price = float(t.get("entry_price") or 0.0)
        entry_d = date.fromisoformat(str(t.get("entry_date") or "")[:10])
        exit_d = date.fromisoformat(str(t.get("exit_date") or "")[:10])
    except (TypeError, ValueError):
        return None
    if sign is None or entry_price <= 0 or exit_d < entry_d:
        return None
    closes = _close_series(str(t.get("ticker") or ""))
    dates = sorted(closes.keys())
    j0 = bisect_right(dates, entry_d)          # first session strictly after entry
    j_act = bisect_right(dates, exit_d) - 1    # session close at-or-before the exit
    if j_act < j0 or j0 >= len(dates):
        return None                            # exited same session / no bars
    k_max = min(j_act - j0 + int(extra), len(dates) - 1 - j0)
    rets = []
    for k in range(k_max + 1):
        c = closes[dates[j0 + k]]
        if c <= 0:
            return None                        # corrupt close row → skip trade
        rets.append(sign * (c / entry_price - 1.0) * 100.0)
    if len(rets) < 2:
        return None                            # no alternative exits to draw
    return {"reason": t.get("exit_reason") or "(unspecified)",
            "ticker": t.get("ticker"), "rets": np.asarray(rets, dtype=float),
            "act_k": j_act - j0}


def exit_timing_mc(closed_trades: List[dict], n_sims: int = DEFAULT_SIMS,
                   seed: int = DEFAULT_SEED,
                   extra_sessions: int = EXIT_MC_EXTRA_SESSIONS) -> dict:
    """Actual exit timing vs the random-exit null, per exit rule + overall.

    Returns ``{n, n_skipped, rows}`` where each row is::

        {reason, trades, actual_mean, null_mean, null_lo, null_hi,
         percentile, p_random_beats, verdict}

    ``percentile`` — where the rule's actual mean return lands inside its own
    random-exit null distribution (100 = better than every random draw);
    ``p_random_beats`` — P(random-exit mean ≥ actual mean), one-sided.
    Gross of costs, close-anchored both arms (see module doc)."""
    prepared = [r for r in (_exit_mc_trade(t, extra_sessions) for t in (closed_trades or []))
                if r is not None]
    skipped = len([t for t in (closed_trades or []) if t.get("status") == "CLOSED"]) - len(prepared)
    if not prepared:
        return {"n": 0, "n_skipped": skipped, "rows": [], "verdict":
                "No closed trades with enough cached sessions for the exit MC yet."}

    rng = np.random.default_rng(seed)
    groups: Dict[str, List[dict]] = {}
    for r in prepared:
        groups.setdefault(r["reason"], []).append(r)
    groups["ALL exits"] = prepared

    rows: List[dict] = []
    for reason, seg in sorted(groups.items(), key=lambda kv: (kv[0] == "ALL exits", -len(kv[1]))):
        actual = float(np.mean([tr["rets"][tr["act_k"]] for tr in seg]))
        # Null: each sim draws one uniform-random feasible exit per trade.
        sims = np.empty(n_sims, dtype=float)
        draws = [rng.integers(0, len(tr["rets"]), size=n_sims) for tr in seg]
        mat = np.stack([tr["rets"][d] for tr, d in zip(seg, draws)], axis=0)  # trades × sims
        sims = mat.mean(axis=0)
        # Mid-rank percentile (ties count half) — the discrete null makes exact
        # ties common on small groups; a strict < would cap a 1-trade group's
        # best-possible exit at 100·(1−1/K) and its worst at 0, biasing both tails.
        pct = float(((sims < actual).mean() + 0.5 * (sims == actual).mean()) * 100.0)
        p_rand = float((np.sum(sims >= actual) + 1) / (n_sims + 1))
        verdict = ("times exits BETTER than random" if pct >= 95.0
                   else "random exits would have BEATEN this rule" if pct <= 5.0
                   else "timing ≈ random (no evidence of exit skill)")
        rows.append({
            "reason": reason, "trades": len(seg),
            "actual_mean": round(actual, 3),
            "null_mean": round(float(sims.mean()), 3),
            "null_lo": round(float(np.percentile(sims, 5)), 3),
            "null_hi": round(float(np.percentile(sims, 95)), 3),
            "percentile": round(pct, 1),
            "p_random_beats": round(p_rand, 3),
            "verdict": verdict,
        })
    return {"n": len(prepared), "n_skipped": skipped, "rows": rows}


# ── ledger-loading report wrappers (dashboard + CLI entry points) ────────────

def compute_method_overfit_report(n_sims: int = DEFAULT_SIMS,
                                  seed: int = DEFAULT_SEED) -> dict:
    """Method luck-vs-skill + filter selection-bias over the live ledger.

    Uses the SAME data basis as the live win-rate filter (train split when OOS
    validation is on) so the p-values judge exactly the numbers the filter
    selects on. Each method row is annotated with the filter's treatment of it
    (KEPT / FILTERED / EXEMPT-inverted / thin-data)."""
    from config.settings import settings
    from src.performance.tracker import _filter_by_split, _load_trades
    from src.signals.aggregator import _inverted_methods

    split = "train" if settings.enable_oos_validation else None
    closed = [t for t in _load_trades()
              if t.get("status") == "CLOSED" and t.get("method_scores")]
    closed = _filter_by_split(closed, split)
    by_method = method_gross_returns(closed)
    rows = method_overfit_mc(by_method, n_sims=n_sims, seed=seed)

    thr = float(settings.winrate_filter_threshold) * 100.0
    min_n = max(1, int(settings.winrate_filter_min_trades))
    inverted = _inverted_methods()
    from src.signals.aggregator import _BASE_WEIGHTS
    judgeable: Dict[str, int] = {}
    kept_actual = 0
    for r in rows:
        m = r["method"]
        if m not in _BASE_WEIGHTS:
            r["filter_state"] = "—"            # panel-only / non-pool method
            continue
        if m in inverted:
            r["filter_state"] = "EXEMPT (inverted)"
            continue
        if r["n"] < min_n:
            r["filter_state"] = f"thin (<{min_n})"
            continue
        judgeable[m] = r["n"]
        if r["win_rate"] >= thr:
            r["filter_state"] = "KEPT"
            kept_actual += 1
        else:
            r["filter_state"] = "FILTERED"

    selection = filter_selection_mc(judgeable, kept_actual, threshold=thr / 100.0,
                                    min_trades=min_n, n_sims=n_sims, seed=seed)
    return {"rows": rows, "selection": selection,
            "split": split or "all", "n_closed": len(closed)}


def compute_exit_timing_report(session: Optional[str] = None,
                               direction: Optional[str] = None,
                               n_sims: int = DEFAULT_SIMS,
                               seed: int = DEFAULT_SEED) -> dict:
    """Exit-timing MC over the live ledger with the Exit-Performance tab's
    filter semantics (session = session the trade EXITED in)."""
    from src.performance.tracker import _exit_session_matches, _load_trades, _match_direction
    closed = [t for t in _load_trades()
              if t.get("status") == "CLOSED"
              and _exit_session_matches(t, session)
              and _match_direction(t, direction)]
    return exit_timing_mc(closed, n_sims=n_sims, seed=seed)


def _print_reports() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    rep = compute_method_overfit_report()
    print(f"\nMethod luck-vs-skill Monte Carlo — {rep['n_closed']} closed trade(s), "
          f"split={rep['split']}, {DEFAULT_SIMS} sims")
    head = (f"{'method':<16}{'state':<18}{'n':>5}{'WR%':>7}{'CI5-95':>14}{'p(luck)':>9}"
            f"{'mean%':>8}{'CI5-95':>16}{'p(ret)':>8}  verdict")
    print(head)
    print("-" * len(head))
    for r in rep["rows"]:
        wr_ci = "[{:.0f},{:.0f}]".format(r["wr_lo"], r["wr_hi"])
        ret_ci = "[{:.2f},{:.2f}]".format(r["ret_lo"], r["ret_hi"])
        print(f"{r['method']:<16}{r.get('filter_state', '—'):<18}{r['n']:>5}"
              f"{r['win_rate']:>7.1f}{wr_ci:>14}"
              f"{r['p_luck']:>9.3f}{r['mean_ret']:>8.2f}"
              f"{ret_ci:>16}{r['p_ret']:>8.3f}  {r['verdict']}")
    sel = rep["selection"]
    if sel.get("n_judgeable"):
        print(f"\nWin-rate filter selection-bias null ({sel['n_judgeable']} judgeable methods): "
              f"chance would keep {sel['kept_null_mean']}±{sel['kept_null_sd']} "
              f"[{sel['kept_null_lo']},{sel['kept_null_hi']}]; live filter kept "
              f"{sel['kept_actual']} (p≥actual = {sel['p_ge_actual']}).")
    print(f"→ {sel.get('verdict')}")

    ex = compute_exit_timing_report()
    print(f"\nExit-timing vs random-exit Monte Carlo — {ex['n']} closed trade(s)"
          + (f" ({ex['n_skipped']} skipped, no cached window)" if ex.get("n_skipped") else ""))
    if not ex["rows"]:
        print(ex.get("verdict", ""))
        return
    head2 = (f"{'exit reason':<26}{'n':>4}{'actual%':>9}{'null%':>8}{'null CI':>18}"
             f"{'pctile':>8}{'p(rand≥)':>10}  verdict")
    print(head2)
    print("-" * len(head2))
    for r in ex["rows"]:
        null_ci = "[{:.2f},{:.2f}]".format(r["null_lo"], r["null_hi"])
        print(f"{r['reason']:<26}{r['trades']:>4}{r['actual_mean']:>9.2f}{r['null_mean']:>8.2f}"
              f"{null_ci:>18}{r['percentile']:>8.1f}"
              f"{r['p_random_beats']:>10.3f}  {r['verdict']}")


def main() -> None:
    from src.db import repo
    repo.set_read_only(True)
    _print_reports()


if __name__ == "__main__":
    main()
