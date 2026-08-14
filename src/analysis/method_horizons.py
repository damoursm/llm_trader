"""Per-method horizon skill — where each method's edge actually lives.

Measured over the SOLO-METHOD simulation (`simulated_trades`), gated the way a
solo method would really be gated: a single method driving `combined_score`
fires a direction only when `|score| >= buy_sell_diff_threshold`, so only those
events count. That gives ~1,500-3,300 observations per method instead of the
78-174 attributed trades the ledger holds — the difference between "we cannot
tell" and a real answer.

Why horizons at all (2026-07-26). Judged at 1 day only, exactly ONE method
cleared p<0.05 and the book would have collapsed to a single signal. Judged
across 1/3/5/10 days the picture is completely different, and coherent:

    sent_velocity  52.4 (p .003)  50.0   50.9   50.7    fast spike, decays
    iv_expr        50.0   56.6 (p .040)  54.3   47.6    medium peak
    pattern        48.6   51.4   52.8 (p .015)  52.8 (p .025)   monotone build
    iv_rank        51.0   50.5   52.1 (p .017)  51.3    positive throughout
    oi_skew        45.9   47.9   53.1   57.7 (p .048)   slow build

A method is not "good" or "bad" — it is good over a particular holding period.
`sent_velocity` measures the RATE OF CHANGE of tone, so a one-day edge that
decays is exactly its expected shape; `pattern` needs days for a formation to
play out. Dropping either on a 1-day test would have been a measurement error,
not a finding.

Multiple comparisons: four horizons per method, and keeping on any hit does
inflate false positives. As with the cross-method case (see
`settings.inversion_require_replication`), a count-based correction is the wrong
tool because the tests are CORRELATED — a 3-day return contains the 1-day one.
The safeguard here is profile COHERENCE, reported per method: an edge that
builds monotonically or clears at two adjacent horizons is credible; a lone
spike surrounded by noise is not, and is flagged as such.

Three states, consumed by two different callers:
  Since 2026-08-12 the STATE is judged on the SIGNED PIVOT TARGET (``win_pv``,
  one binomial test at each name's own natural horizon — the promotion basis),
  with the fixed-horizon rule as fallback; the fixed 1/3/5/10d curve still
  decides ``best_horizon``/``best_days`` (the HOLDING period the exits use).

  * PROVEN     — significantly >50% at some horizon. Full weight; its best
                 horizon feeds the `method_horizon` exit signal.
  * DISPROVEN  — significantly <50% at EVERY horizon it has data for. Dropped.
  * UNPROVEN   — neither. Reduced weight: absence of evidence is not evidence
                 of absence, and on a five-week sample most methods land here.

CLI:  python -m src.analysis.method_horizons
"""

from __future__ import annotations

import time
from typing import Dict, Optional

from config.settings import settings

from loguru import logger  # project configures loguru sinks only

# Horizon label in `compute_method_perf` → trading days it represents.
HORIZON_DAYS: Dict[str, float] = {"1d": 1.0, "3d": 3.0, "1w": 5.0, "2w": 10.0}

PROVEN, UNPROVEN, DISPROVEN = "proven", "unproven", "disproven"

_CACHE: dict = {}


# NOTE: these deliberately do NOT swallow exceptions. They used to reach for
# scipy inside `try/except Exception: return None`, and since scipy was never a
# declared dependency the except branch was the only one that ever ran — so
# EVERY method classified UNPROVEN, permanently, and the reduced-weight branch
# was silently the only live path. A neutral verdict is indistinguishable from a
# real one, which is exactly why it must not be reachable by accident.

def _p_above_half(win_pct: float, n: int) -> Optional[float]:
    """One-sided p that the true win rate exceeds 50%."""
    if not n or win_pct is None:
        return None
    from src.analysis.stats import binom_p_greater
    return binom_p_greater(int(round(win_pct / 100.0 * n)), int(n), 0.5)


def _p_below_half(win_pct: float, n: int) -> Optional[float]:
    """One-sided p that the true win rate is below 50%."""
    if not n or win_pct is None:
        return None
    from src.analysis.stats import binom_p_less
    return binom_p_less(int(round(win_pct / 100.0 * n)), int(n), 0.5)


def compute_method_horizons(days: Optional[int] = None,
                            sim_df=None) -> Dict[str, dict]:
    """``{method: {state, best_horizon, best_days, best_win, best_p, profile,
    coherent}}``.

    ``profile`` is the per-horizon ``(win%, n, p)`` so a caller can see the shape
    that produced the verdict. ``coherent`` is True when the winning horizon is
    supported by an adjacent one (p<0.10) or the win rate builds monotonically —
    the guard against a lone significant spike.

    Cached for ``method_horizon_cache_seconds``; the underlying join is heavy and
    the answer moves over days. Fail-soft to ``{}``.
    """
    if not getattr(settings, "enable_method_horizons", False):
        return {}
    now = time.time()
    hit = _CACHE.get("v")
    if hit and (now - hit["ts"]) < float(settings.method_horizon_cache_seconds):
        return hit["out"]

    out: Dict[str, dict] = {}
    try:
        import pandas as pd
        from src.analysis.simulated_trades import load_sim_entry_events, compute_method_perf
        from src.signals.aggregator import _BASE_WEIGHTS

        ev = sim_df if sim_df is not None else load_sim_entry_events(days)
        if ev is None or len(ev) == 0:
            raise ValueError("no simulated entry events")
        # The gate a SOLO method would face: with one method holding a view,
        # combined_score IS that method's score, so a direction fires only at
        # |score| >= buy_sell_diff_threshold. Measuring ungated would credit a
        # method for calls the system would never have acted on.
        thr = float(settings.buy_sell_diff_threshold)
        gated = ev[ev["score"].abs() >= thr]
        perf = compute_method_perf(sim_df=gated,
                                   min_n=int(settings.method_horizon_min_obs))

        alpha = float(settings.method_horizon_alpha)
        for _, row in perf.iterrows():
            m = row.get("method")
            if m not in _BASE_WEIGHTS:
                continue
            profile, best, best_p = {}, None, 1.1
            below = []
            for label in HORIZON_DAYS:
                n = int(row.get(f"n_{label}") or 0)
                w = row.get(f"win_{label}")
                if not n or w is None or pd.isna(w):
                    continue
                w = float(w)
                pa, pb = _p_above_half(w, n), _p_below_half(w, n)
                profile[label] = {"win": round(w, 2), "n": n,
                                  "p_above": pa, "p_below": pb}
                if pa is not None and w > 50.0 and pa < best_p:
                    best_p, best = pa, label
                below.append(pb is not None and pb < alpha)

            # STATE — PIVOT basis when judgeable (2026-08-12, user directive):
            # proven/disproven is decided by the method's win rate on the
            # SIGNED PIVOT TARGET (one target, one binomial test — the same
            # statistic the promotions were justified by, at each name's own
            # natural horizon). The fixed-horizon rule below remains the
            # fallback for a perf frame predating the pivot column.
            n_pv = int(row.get("n_pv") or 0)
            w_pv = row.get("win_pv")
            pv_ok = n_pv > 0 and w_pv is not None and not pd.isna(w_pv)
            if pv_ok:
                w_pv = float(w_pv)
                pa_pv, pb_pv = _p_above_half(w_pv, n_pv), _p_below_half(w_pv, n_pv)
                profile["pv"] = {"win": round(w_pv, 2), "n": n_pv,
                                 "p_above": pa_pv, "p_below": pb_pv}
                if pa_pv is not None and w_pv > 50.0 and pa_pv < alpha:
                    state = PROVEN
                elif pb_pv is not None and pb_pv < alpha:
                    state = DISPROVEN      # loses on the pivot target
                else:
                    state = UNPROVEN
            elif not profile:
                continue
            elif best is not None and best_p < alpha:
                state = PROVEN
            elif below and all(below):
                state = DISPROVEN          # loses at EVERY horizon it can be judged on
            else:
                state = UNPROVEN

            # HOLDING PERIOD — always from the FIXED-horizon curve (the user
            # directive keeps calendar horizons for "how long", pivot for
            # "how good"): the winner is the significant fixed horizon when one
            # exists, else the best GUESS (lowest p_above) for a pivot-proven
            # method whose fixed curve is merely suggestive.
            has_fixed = any(k in HORIZON_DAYS for k in profile)
            if state == PROVEN and best is None and has_fixed:
                cands = [(v.get("p_above"), k) for k, v in profile.items()
                         if k in HORIZON_DAYS and v.get("p_above") is not None
                         and v.get("win", 0) > 50.0]
                if cands:
                    best_p, best = min(cands)

            out[m] = {
                "state": state,
                "best_horizon": best if state == PROVEN else None,
                "best_days": HORIZON_DAYS.get(best) if state == PROVEN else None,
                "best_win": profile.get(best, {}).get("win") if best else None,
                "best_p": round(best_p, 5) if best is not None and best_p <= 1 else None,
                "profile": profile,
                "coherent": _is_coherent(profile, best) if state == PROVEN else None,
            }
    except Exception as e:
        logger.debug(f"[method_horizons] unavailable: {e}")
        out = {}

    _CACHE["v"] = {"ts": now, "out": out}
    if out:
        pr = [m for m, d in out.items() if d["state"] == PROVEN]
        di = [m for m, d in out.items() if d["state"] == DISPROVEN]
        logger.info(f"[method_horizons] proven={sorted(pr)} disproven={sorted(di)} "
                    f"unproven={len(out) - len(pr) - len(di)}")
    return out


def _is_coherent(profile: dict, best: Optional[str]) -> bool:
    """Is the winning horizon supported by its neighbours, or a lone spike?

    Correlated horizons mean a count-based correction is wrong (a 3-day return
    contains the 1-day one), so shape is the guard instead: an adjacent horizon
    also leaning the same way (p<0.10), or a monotone build up to the winner.
    """
    if not best or best not in profile:
        return False
    order = list(HORIZON_DAYS)
    i = order.index(best)
    for j in (i - 1, i + 1):
        if 0 <= j < len(order):
            nb = profile.get(order[j])
            if nb and nb.get("p_above") is not None and nb["p_above"] < 0.10:
                return True
    wins = [profile[h]["win"] for h in order[:i + 1] if h in profile]
    return len(wins) >= 3 and all(b >= a - 0.5 for a, b in zip(wins, wins[1:]))


def method_state(method: str) -> str:
    """PROVEN / UNPROVEN / DISPROVEN for one method (UNPROVEN when unknown)."""
    return (compute_method_horizons().get(method) or {}).get("state", UNPROVEN)


def method_best_days(method: str) -> Optional[float]:
    """Trading days at which this method's edge is strongest, or None."""
    return (compute_method_horizons().get(method) or {}).get("best_days")


def reset_cache() -> None:
    _CACHE.clear()


if __name__ == "__main__":  # pragma: no cover
    res = compute_method_horizons()
    if not res:
        print("No horizon skill computed (needs the simulated_trades panel).")
        raise SystemExit(0)
    print(f"\n{'method':<17}" + "".join(f"{h:>16}" for h in HORIZON_DAYS)
          + f"{'state':>11}{'best':>7}{'coh':>6}")
    for m, d in sorted(res.items(), key=lambda x: (x[1]["state"] != PROVEN,
                                                   x[1].get("best_p") or 9)):
        cells = ""
        for h in HORIZON_DAYS:
            pr = d["profile"].get(h)
            cells += ("—".rjust(16) if not pr
                      else f"{pr['win']:>6.1f} ({pr['p_above']:>5.3f})".rjust(16))
        print(f"{m:<17}{cells}{d['state']:>11}{str(d['best_horizon'] or '—'):>7}"
              f"{('yes' if d['coherent'] else 'no' if d['coherent'] is not None else '—'):>6}")
