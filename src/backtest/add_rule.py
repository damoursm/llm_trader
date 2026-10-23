"""THE ADD RULE, mechanically (CLAUDE.md Standing rules: DEPLOYING; user directives 2026-10-10, 2026-10-11,
2026-10-13: "Apply the best decisions you're recommending", and 2026-10-14: the pooled history).

FROM 2026-10-14 A DECISION READS THE POOLED HISTORY — `pooled_verdict` (user directive 2026-10-14: "Instead of
checking for the three different sets (A0, A, and B), check for the full dataset (A0+A+B) ... doesn't need to have
growth all the time"): sets A0 + A + B (`periods.POOLED`, 2018-01-02 .. 2026-09-24), paired on the same seeds:

* growth: the POOLED gain — the years-weighted mean of the sets' paired gains (`periods.pool_weights`: one account
  chained through them), its standard error sqrt(sum w^2 se^2) — at least `MIN_GAIN` (+0.5) log points a year. No set
  needs to gain on its own and no similarity between sets is asked;
* not one episode: with per-year records (each draw's log growth per calendar-year part, `year_lp` from its daily
  NAV), the pooled gain stays above 0 with ANY ONE calendar year left out — a gain one year carries alone does not
  transfer;
* margin events AS A RATE (from 2026-10-23; the user: "update the add rule to judge margin calls as a rate. We
  shouldn't judge the performance based on number of margin calls anymore, just the performance in rates (return per
  day and growth)"): on the (set, draw) pairs where exactly ONE side meets a margin event, the change's excess refuses
  it only when a fair coin would rarely give it — the one-sided exact binomial (McNemar's exact test, `margin_events`)
  p < `MARGIN_P` (0.05): five more pairs and none fewer refuse, four do not; a pair or two is the draws' noise
  (PREREG184: the one-pair count refused changes that later gained with no more calls). ``margin="count"`` restores the
  count rule before — no more (set, draw) pairs with a margin call than the live configuration's on the same seeds,
  summed over the pool — and reproduces every decision before 2026-10-23;
* drawdown: the pooled history's worst drawdown (per seed the worst of its sets', averaged over the seeds) at most
  `MAX_DD_DEEPER` (1) point deeper.
Draws as below (20 a set; 40 in every set when the pooled gain lies within 2 SE of +0.5; futility from 5 in every set).
A variant whose growth fails while its margin events are significantly FEWER (the same test the other way; any
fewer under ``margin="count"``) or its drawdown more than 1 point shallower, with no deeper drawdown, is flagged
``risk_protection`` — the user's call, as before. The sets' own gains and each calendar
year's are REPORTED (where a change earns and where it costs), never required.

`verdict` below is the two-set rule (sets A and B) every study before 2026-10-14 decided on — kept, unchanged, to
reproduce them.

A variant of the live combination is ADDED when, on the production-mimicking simulator (`mimic.Mimic()`, stratified
draws), PAIRED against the live configuration on the SAME seeds, on BOTH borrow-data sets:

* growth: the mean paired gain is at least `MIN_GAIN` (+0.5) log points a year (100 x the change in ln(1 + growth)),
  and the smaller set's gain at least half the larger's;
* margin calls: the variant meets a margin call in NO MORE draws than the live configuration on the same seeds (the
  live account itself meets one in some draws of set B under PREREG94's borrow model — "none in any draw" could not
  be met by the live configuration);
* drawdown: the mean worst drawdown at most `MAX_DD_DEEPER` (1) point deeper.

DRAWS: `DRAWS` (20) paired draws per set — four stratified blocks of five; a set whose gain (or the half-rule's
comparison) lies within `UNDECIDED_SE` (2) standard errors of its threshold is UNDECIDED and runs `MORE_DRAWS` (40),
then is judged on its point estimate. FUTILITY: from `FUTILITY_DRAWS` (5) paired draws, a set whose gain lies more
than `FUTILITY_SE` (3) standard errors below +0.5 rejects the variant at once — more draws cannot lift it (PREREG95: a set-B rule change carries ~4-6 log points of noise a draw; five draws
cannot tell +0.5 from 0). Significance against the market's history is NOT required (user 2026-10-10) — the standard
error here is the SIMULATION's, which more draws remove.

`verdict(live, variant)` takes, per set, the per-seed records (seed -> {"growth", "maxdd", "calls"}: an account run's
growth as a fraction, its worst drawdown in points, its margin calls) and returns each criterion, the decision ("add" /
"reject" / "more draws") and the gains with their standard errors.

THE OVERFITTING REPORT (PREREG106, registered 2026-10-14 13:09 ET — REPORT ONLY, it changes no decision): with
``n_trials`` or ``model`` (the trial ledger, `src/backtest/trials.py`) and / or ``kind``, `verdict` adds an
``"overfitting"`` entry: each set's gain as a DEFLATED probability (its z against the expected maximum of N trials'
z) and its RETENTION-SHRUNK expectation (PREREG81's measured retention: rule tweaks ~0.15 (0.09-0.23), sizing ~0.5).
The formulas are in `verdict`'s docstring. Without those arguments every output is byte for byte the add rule's
(`tests/test_add_rule.py` pins it against the pre-PREREG106 module).

THE MARKET-NOISE REPORT (PREREG183, 2026-10-22 — REPORT ONLY): the draws' standard error says nothing about the
market's noise (one history, eight years). `market_noise(live, variant)` reads the change as the paired MONTHLY log
difference of the two configurations' NAVs (the mean over the seeds), with López de Prado, Lipton & Zoonekynd's
Sharpe inference (`src.analysis.stats`): its Newey-West t, its p with the months' autocorrelation, skewness and
kurtosis, deflated for K tries, the observed false discovery rate at a few priors, and the chance that a change
without effect shows the +0.5 gain; `pooled_verdict(..., market=True)` adds it as ``"market_noise"`` (off by default:
every verdict before stays byte for byte).
"""
from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Tuple, Union

MIN_GAIN = 0.5            # log points a year: each set (`verdict`), the pooled history (`pooled_verdict`)
POOLED_SETS = ("A0", "A", "B")   # `periods.POOLED`: the pooled history (2018-01-02 .. 2026-09-24)
LOYO_MIN = 0.0            # the pooled gain with any one calendar year left out stays above this (log points a year)
HALF = 0.5                # the smaller gain at least this share of the larger
MAX_DD_DEEPER = 1.0       # points
MARGIN_P = 0.05           # the margin events' rate test: the one-sided exact binomial on the discordant pairs (2026-10-23)
MARGIN_RULES = ("rate", "count")   # `pooled_verdict(margin=...)`: "rate" from 2026-10-23; "count" the rule before it
DRAWS = 20                # paired draws per set (four stratified blocks of five)
MORE_DRAWS = 40           # when a set is undecided after DRAWS
UNDECIDED_SE = 2.0        # a set within this many standard errors of a threshold is undecided
FUTILITY_DRAWS = 5        # from this many paired draws ...
FUTILITY_SE = 3.0         # ... a set this many standard errors below MIN_GAIN rejects the variant early
# PREREG106 (report only): PREREG81's RETENTION — the share of the in-sample best's gain over live kept out of sample
# across CSCV halvings (studies/prereg81/part4_81.json): PREREG75's rule variants 0.23 (set A) / 0.09 (set B),
# PREREG80's sizings 0.50 (A) / 0.47 (B). kind -> (the factor, its band).
RETENTION: Dict[str, Tuple[float, Tuple[float, float]]] = {
    "rule": (0.15, (0.09, 0.23)),
    "sizing": (0.5, (0.47, 0.50)),
}
RETENTION_SOURCE = ("PREREG81 (studies/prereg81/part4_81.json): rule variants 0.23 (A) / 0.09 (B), "
                    "sizings 0.50 (A) / 0.47 (B)")
FORMULA = ("z = gain / se (the paired draws' standard error); z0(N) = (1 - g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e)), "
           "g = 0.5772 (Euler), z0(1) = 0; p = Phi(z - z0(N)); shrunk = gain x retention")


def _lp(g: float) -> float:
    return 100.0 * math.log1p(float(g))


def paired(live: Mapping[int, dict], variant: Mapping[int, dict]) -> dict:
    """The paired comparison of one set: the seeds both have, the mean gain in log points a year with its standard
    error, the mean worst drawdowns, and the draws with a margin call on each side."""
    seeds = sorted(set(live) & set(variant))
    n = len(seeds)
    if n == 0:
        return {"n": 0}
    d = [_lp(variant[s]["growth"]) - _lp(live[s]["growth"]) for s in seeds]
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else float("nan")
    return {"n": n, "seeds": seeds, "gain": mean, "sd": sd, "se": sd / math.sqrt(n) if n > 1 else float("nan"),
            "gains": d,
            "dd_live": sum(float(live[s]["maxdd"]) for s in seeds) / n,
            "dd_variant": sum(float(variant[s]["maxdd"]) for s in seeds) / n,
            "calls_live": sum(1 for s in seeds if live[s].get("calls")),
            "calls_variant": sum(1 for s in seeds if variant[s].get("calls"))}


def _undecided(est: float, thr: float, se: float) -> bool:
    return bool(math.isfinite(se) and se > 0 and abs(est - thr) < UNDECIDED_SE * se)


def binom_upper(k: int, n: int) -> float:
    """P[X >= k] for X ~ Binomial(n, 1/2), exact; 1.0 when k <= 0 or n <= 0."""
    k, n = int(k), int(n)
    if k <= 0 or n <= 0:
        return 1.0
    return sum(math.comb(n, j) for j in range(k, n + 1)) / 2.0 ** n


def margin_events(live: Mapping[str, Mapping[int, dict]], variant: Mapping[str, Mapping[int, dict]],
                  sets=POOLED_SETS) -> dict:
    """THE MARGIN EVENTS AS A RATE (from 2026-10-23): over the (set, draw) pairs both sides hold, the pairs where only
    the live configuration meets a margin event (``only_live``), only the variant (``only_variant``), both and neither.
    With no change in the rate each DISCORDANT pair falls either way with probability 1/2 (McNemar's exact test):
    ``p_more`` = P[at least ``only_variant`` of them on the variant's side], ``p_fewer`` the same for the live side;
    ``more`` / ``fewer``: that side's excess significant at `MARGIN_P`."""
    only_l = only_v = both = neither = 0
    for st in sets:
        L, V = live.get(st, {}), variant.get(st, {})
        for k in set(L) & set(V):
            a, b = bool(L[k].get("calls")), bool(V[k].get("calls"))
            if a and b:
                both += 1
            elif a:
                only_l += 1
            elif b:
                only_v += 1
            else:
                neither += 1
    n = only_l + only_v
    p_more, p_fewer = binom_upper(only_v, n), binom_upper(only_l, n)
    return {"only_live": only_l, "only_variant": only_v, "both": both, "neither": neither, "discordant": n,
            "p_more": p_more, "p_fewer": p_fewer, "level": MARGIN_P,
            "more": bool(only_v > only_l and p_more < MARGIN_P),
            "fewer": bool(only_l > only_v and p_fewer < MARGIN_P)}


def _risk_note(margin: str) -> str:
    calls = "significantly fewer margin events" if margin == "rate" else "fewer margin calls"
    return f"RISK PROTECTION: {calls} or a shallower drawdown for the growth it costs: the user's call"


def verdict(live: Mapping[str, Mapping[int, dict]], variant: Mapping[str, Mapping[int, dict]],
            sets=("A", "B"), *, n_trials: Union[int, Tuple[int, int], None] = None, model: Optional[str] = None,
            kind: Optional[str] = None, ledger=None) -> dict:
    """The add rule on sets A and B (see the module docstring). Returns {"sets": {set: paired}, "criteria": {...},
    "undecided": [sets needing more draws], "decision": "add" | "reject" | "more draws", "why": [...]}.

    PREREG106's REPORT (it changes nothing above). ``n_trials`` (an int, or a (low, high) pair), or ``model`` (the
    trials it has had in the trial ledger, `trials.count`: PREREG103's seed plus the recorded trials, low and high;
    ``ledger`` names another ledger file), and / or ``kind`` ("rule" or "sizing") add ``"overfitting"``. Per set with
    paired draws:

        z      = gain / se              the paired mean gain over its standard error across the draws
        z0(N)  = (1 - γ) Φ⁻¹(1 - 1/N) + γ Φ⁻¹(1 - 1/(N e)),   γ = 0.5772 (Euler's constant)
                                        the expected maximum of N independent standard normals
                                        (`stats.expected_max_sharpe(N, 1)`; z0(1) = 0)
        p      = Φ(z - z0(N))           the DEFLATED probability that the gain is above 0, given that it is the best
                                        of N trials; N = 1 gives Φ(z), the undeflated one (``p_raw``)
        shrunk = gain × retention       PREREG81's measured retention (`RETENTION`): "rule" 0.15 (band 0.09-0.23),
                                        "sizing" 0.5 (band 0.47-0.50); ``clears_min_gain``: shrunk >= `MIN_GAIN`

    p is reported at the low and the high N (a single N when ``n_trials`` is one int). Record the candidate in the
    ledger BEFORE the verdict, so that it counts itself; N is at least 1. The standard error is the SIMULATION's (the
    draws'), not the market's: p reads selection against the draws' noise only, and `deflated_gain` takes any other
    standard error (a block bootstrap over the history) from the caller. Without these arguments the result is the
    add rule's alone, key for key and byte for byte."""
    if n_trials is None and model is None and kind is None:
        return _verdict(live, variant, sets)
    spec = _report_spec(n_trials, model, kind, ledger)         # a wrong argument fails before any work
    v = _verdict(live, variant, sets)
    v["overfitting"] = _report(v["sets"], spec)
    return v


def deflated_gain(gain: float, se: float, n_trials: int) -> dict:
    """The deflated probability of one gain (the formulas in `verdict`): {"N", "z", "z0", "p", "p_raw"} — ``z``,
    ``p`` and ``p_raw`` are None when ``se`` is not a positive number (a single draw) or the gain is not finite."""
    from src.analysis.stats import expected_max_sharpe
    from statistics import NormalDist
    n = int(n_trials)
    if n < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials!r}")
    z0 = expected_max_sharpe(n, 1.0)
    g, s = float(gain), float(se)
    if not (math.isfinite(g) and math.isfinite(s) and s > 0):
        return {"N": n, "z": None, "z0": z0, "p": None, "p_raw": None}
    z = g / s
    nd = NormalDist()
    return {"N": n, "z": z, "z0": z0, "p": nd.cdf(z - z0), "p_raw": nd.cdf(z)}


def _report_spec(n_trials, model, kind, ledger) -> dict:
    """The report's validated inputs: {"n": {"low", "high", "source"} or None, "kind", "retention"}."""
    if n_trials is not None and model is not None:
        raise ValueError("pass n_trials or model (the trial ledger's count), not both")
    n = None
    if n_trials is not None:
        import numbers
        pair = (n_trials, n_trials) if isinstance(n_trials, numbers.Integral) else n_trials
        try:
            lo, hi = pair
        except (TypeError, ValueError):
            lo = hi = None
        if not all(isinstance(x, numbers.Integral) and not isinstance(x, bool) for x in (lo, hi)) \
                or not 1 <= int(lo) <= int(hi):
            raise ValueError(f"n_trials must be an int >= 1 or a (low, high) pair with 1 <= low <= high, "
                             f"got {n_trials!r}")
        n = {"low": int(lo), "high": int(hi), "source": "given"}
    elif model is not None:
        from src.backtest import trials as TR
        c = TR.count(model, ledger)
        seed = c.get("seed") or {}
        n = {"low": max(1, int(c["low"])), "high": max(1, int(c["high"])),
             "source": (f"trial ledger {TR.ledger_path(ledger)}: {model!r} = seed {seed.get('low', 0)} / "
                        f"{seed.get('high', 0)} + {c['recorded']} recorded"
                        + (f" (+ {c['recorded_all_models']} of every model in the high count)"
                           if "recorded_all_models" in c else ""))}
    ret = None
    if kind is not None:
        if kind not in RETENTION:
            raise ValueError(f"no measured retention for kind {kind!r} (known: {sorted(RETENTION)}); pass kind=None "
                             f"for the deflated probability alone")
        f, band = RETENTION[kind]
        ret = {"factor": f, "band": list(band), "source": RETENTION_SOURCE}
    return {"n": n, "kind": kind, "retention": ret}


def _report(P: Mapping[str, dict], spec: dict) -> dict:
    """PREREG106's per-set report on the paired comparisons ``P`` (`paired`'s records)."""
    n, ret = spec["n"], spec["retention"]
    out = {}
    for s, p in P.items():
        if not p.get("n"):
            out[s] = {"n": 0}
            continue
        g, se = float(p["gain"]), float(p["se"])
        rec: Dict[str, object] = {"n": p["n"], "gain": g, "se": se}
        if n is not None:
            lo, hi = deflated_gain(g, se, n["low"]), deflated_gain(g, se, n["high"])
            rec.update(z=lo["z"], p_raw=lo["p_raw"],
                       deflated={"low": {k: lo[k] for k in ("N", "z0", "p")},
                                 "high": {k: hi[k] for k in ("N", "z0", "p")}})
        if ret is not None:
            a, b = g * ret["band"][0], g * ret["band"][1]
            rec["shrunk"] = {"gain": g * ret["factor"], "band": [min(a, b), max(a, b)],
                             "clears_min_gain": bool(g * ret["factor"] >= MIN_GAIN)}
        out[s] = rec
    return {"n_trials": n, "kind": spec["kind"], "retention": ret, "sets": out, "formula": FORMULA,
            "note": "report only (PREREG106): the decision above is the add rule's, unchanged"}


def _verdict(live: Mapping[str, Mapping[int, dict]], variant: Mapping[str, Mapping[int, dict]],
             sets=("A", "B")) -> dict:
    """The add rule itself (`verdict` without PREREG106's report)."""
    P = {s: paired(live.get(s, {}), variant.get(s, {})) for s in sets}
    why, undecided = [], []
    futile = [s for s in sets if P[s].get("n", 0) >= FUTILITY_DRAWS and math.isfinite(P[s]["se"]) and P[s]["se"] > 0
              and P[s]["gain"] + FUTILITY_SE * P[s]["se"] < MIN_GAIN]
    if futile:
        return {"sets": P, "criteria": {}, "undecided": [], "decision": "reject",
                "why": [f"set {s}: gain {P[s]['gain']:+.2f} +- {P[s]['se']:.2f} log pts after {P[s]['n']} draws, more "
                        f"than {FUTILITY_SE:g} SE below {MIN_GAIN:+.1f} (futility)" for s in futile]}
    for s in sets:
        n = P[s].get("n", 0)
        if n < DRAWS:
            undecided.append(s)
            why.append(f"set {s}: {n} paired draws, the rule needs {DRAWS}")
    if undecided:
        return {"sets": P, "criteria": {}, "undecided": undecided, "decision": "more draws", "why": why}
    gains = {s: P[s]["gain"] for s in sets}
    small_s = min(sets, key=lambda s: gains[s])
    large_s = max(sets, key=lambda s: gains[s])
    crit = {
        "gain_each": {s: gains[s] >= MIN_GAIN for s in sets},
        "similar": gains[large_s] > 0 and gains[small_s] >= HALF * gains[large_s],
        "margin_calls": {s: P[s]["calls_variant"] <= P[s]["calls_live"] for s in sets},
        "drawdown": {s: P[s]["dd_variant"] <= P[s]["dd_live"] + MAX_DD_DEEPER for s in sets},
    }
    # the simulation's own uncertainty: a set within UNDECIDED_SE standard errors of a growth threshold, judged on
    # fewer than MORE_DRAWS draws, runs more (the half-rule's threshold is half the other set's gain)
    for s in sets:
        se = P[s]["se"]
        thr_half = HALF * gains[large_s] if s == small_s else None
        near = _undecided(gains[s], MIN_GAIN, se) or (thr_half is not None and _undecided(gains[s], thr_half, se))
        if near and P[s]["n"] < MORE_DRAWS:
            undecided.append(s)
            why.append(f"set {s}: gain {gains[s]:+.2f} +- {se:.2f} log pts within {UNDECIDED_SE:g} SE of a threshold "
                       f"after {P[s]['n']} draws — run {MORE_DRAWS}")
    if undecided:
        return {"sets": P, "criteria": crit, "undecided": undecided, "decision": "more draws", "why": why}
    ok = all(crit["gain_each"].values()) and crit["similar"] and all(crit["margin_calls"].values()) \
        and all(crit["drawdown"].values())
    for s in sets:
        if not crit["gain_each"][s]:
            why.append(f"set {s}: gain {gains[s]:+.2f} log pts < {MIN_GAIN:+.1f}")
        if not crit["margin_calls"][s]:
            why.append(f"set {s}: margin calls in {P[s]['calls_variant']} draws vs the live {P[s]['calls_live']}")
        if not crit["drawdown"][s]:
            why.append(f"set {s}: drawdown {P[s]['dd_variant']:.1f} vs the live {P[s]['dd_live']:.1f} (> +{MAX_DD_DEEPER:g})")
    if not crit["similar"]:
        why.append(f"the gains are not similar: {gains[small_s]:+.2f} (set {small_s}) < half of {gains[large_s]:+.2f} "
                   f"(set {large_s})")
    return {"sets": P, "criteria": crit, "undecided": [], "decision": "add" if ok else "reject", "why": why}


# ── THE POOLED RULE (from 2026-10-14) ─────────────────────────────────────────

def year_lp(days, values) -> Dict[str, float]:
    """A draw's log growth (log points, 100 x ln) over each calendar year it spans, from its daily NAV (``days`` ISO
    dates, ``values`` the NAV): {"2018": lp, ...} — the first year from the NAV's first value, each next one from the
    previous year's last. The records' ``"years"`` for `pooled_verdict`."""
    out: Dict[str, float] = {}
    first: Dict[str, float] = {}
    last: Dict[str, float] = {}
    for d, v in zip(days, values):
        y = str(d)[:4]
        first.setdefault(y, float(v))
        last[y] = float(v)
    prev = None
    for y in sorted(last):
        a = prev if prev is not None else first[y]
        if a > 0 and last[y] > 0:
            out[y] = 100.0 * math.log(last[y] / a)
        prev = last[y]
    return out


def _year_len(y: str, lo: str, hi: str) -> float:
    """The part of calendar year ``y`` inside [lo, hi], in years (365.25 days)."""
    import datetime as _dt
    a = max(_dt.date(int(y), 1, 1), _dt.date.fromisoformat(lo))
    b = min(_dt.date(int(y), 12, 31), _dt.date.fromisoformat(hi))
    return max((b - a).days + 1, 0) / 365.25


def _pooled_years(live: Mapping[str, Mapping[int, dict]], variant: Mapping[str, Mapping[int, dict]],
                  sets) -> Optional[Dict[str, float]]:
    """The paired gain of each calendar year (log points over the year's part, summed over the sets that hold it),
    when every paired draw of every set carries ``"years"``; else None."""
    per: Dict[str, float] = {}
    for st in sets:
        seeds = sorted(set(live.get(st, {})) & set(variant.get(st, {})))
        if not seeds or not all("years" in live[st][k] and "years" in variant[st][k] for k in seeds):
            return None
        ys = sorted(set().union(*[set(live[st][k]["years"]) & set(variant[st][k]["years"]) for k in seeds]))
        for y in ys:
            d = [float(variant[st][k]["years"][y]) - float(live[st][k]["years"][y]) for k in seeds
                 if y in live[st][k]["years"] and y in variant[st][k]["years"]]
            if d:
                per[y] = per.get(y, 0.0) + sum(d) / len(d)
    return per


def pooled_verdict(live: Mapping[str, Mapping[int, dict]], variant: Mapping[str, Mapping[int, dict]],
                   sets=POOLED_SETS, weights: Optional[Mapping[str, float]] = None, *,
                   n_trials: Union[int, Tuple[int, int], None] = None, model: Optional[str] = None,
                   kind: Optional[str] = None, ledger=None, market: bool = False, margin: str = "rate") -> dict:
    """THE ADD RULE ON THE POOLED HISTORY (the module docstring; user directive 2026-10-14). ``live`` / ``variant``:
    per set, the per-seed records {"growth", "maxdd", "calls"} (+ ``"years"``: `year_lp` of the draw's NAV).
    ``weights`` default to `periods.pool_weights(sets)`. Returns {"sets": {set: paired}, "weights", "pooled": {"gain",
    "se", "calls_live", "calls_variant", "dd_live", "dd_variant", "dd_basis"}, "years": {year: gain} or None, "loyo":
    {year: the pooled gain without it} or None, "criteria", "undecided", "decision": "add" | "reject" | "more draws" |
    "needs per-year growth", "risk_protection", "why"}; with ``n_trials`` / ``model`` / ``kind``, PREREG106's report on
    the pooled gain (its standard error the draws' — pass a market one to `deflated_gain`); with ``market``,
    PREREG183's ``"market_noise"`` (`market_noise`: the gain against the MARKET's noise; the records must carry their
    daily NAV). Neither changes the decision. ``margin``: "rate" (from 2026-10-23: refused only when the change's
    margin events are significantly more frequent, `margin_events`; their record in ``pooled["margin_events"]``) or
    "count" (the rule before: no more pairs with a call than the live configuration's — every decision before
    2026-10-23)."""
    from src.backtest import periods
    if margin not in MARGIN_RULES:
        raise ValueError(f"margin must be one of {MARGIN_RULES}, not {margin!r}")
    spec = _report_spec(n_trials, model, kind, ledger) if (n_trials is not None or model is not None
                                                          or kind is not None) else None
    sets = tuple(sets)
    w = dict(weights) if weights is not None else periods.pool_weights(sets)
    P = {st: paired(live.get(st, {}), variant.get(st, {})) for st in sets}
    out: Dict[str, object] = {"sets": P, "weights": w, "criteria": {}, "undecided": [], "risk_protection": False,
                              "years": None, "loyo": None, "margin_rule": margin}
    ns = {st: P[st].get("n", 0) for st in sets}
    if all(ns[st] >= 2 for st in sets):
        gain = sum(w[st] * P[st]["gain"] for st in sets)
        se = math.sqrt(sum((w[st] * P[st]["se"]) ** 2 for st in sets))
    else:
        gain = se = float("nan")
    calls_l = sum(P[st].get("calls_live", 0) for st in sets)
    calls_v = sum(P[st].get("calls_variant", 0) for st in sets)
    me = margin_events(live, variant, sets)
    calls_ok = (not me["more"]) if margin == "rate" else calls_v <= calls_l
    calls_fewer = me["fewer"] if margin == "rate" else calls_v < calls_l
    common = sorted(set.intersection(*[set(P[st].get("seeds", [])) for st in sets])) if sets else []
    if common:
        dd_l = sum(max(float(live[st][k]["maxdd"]) for st in sets) for k in common) / len(common)
        dd_v = sum(max(float(variant[st][k]["maxdd"]) for st in sets) for k in common) / len(common)
        basis = f"per seed the worst of the sets' drawdowns, averaged over the {len(common)} seeds every set has"
    else:
        dd_l = max((P[st].get("dd_live", float("nan")) for st in sets), default=float("nan"))
        dd_v = max((P[st].get("dd_variant", float("nan")) for st in sets), default=float("nan"))
        basis = "the worst of the sets' mean drawdowns (no seed common to every set)"
    out["pooled"] = {"gain": gain, "se": se, "calls_live": calls_l, "calls_variant": calls_v,
                     "dd_live": dd_l, "dd_variant": dd_v, "dd_basis": basis, "margin_events": me}
    if spec is not None and math.isfinite(gain):
        out["overfitting"] = _report({"pooled": {"n": min(ns.values()), "gain": gain, "se": se}}, spec)
    if market:
        try:
            out["market_noise"] = market_noise(live, variant, sets)
        except Exception as e:                                         # noqa: BLE001 — a report, never a decision
            out["market_noise"] = {"error": str(e)}
    why: List[str] = []
    if all(ns[st] >= FUTILITY_DRAWS for st in sets) and math.isfinite(se) and se > 0             and gain + FUTILITY_SE * se < MIN_GAIN:
        why = [f"pooled gain {gain:+.2f} +- {se:.2f} log pts, more than {FUTILITY_SE:g} SE below {MIN_GAIN:+.1f} "
               f"(futility)"]
        if all(ns[st] >= DRAWS for st in sets):           # the risk reading needs the rule's draws
            out["years"] = _pooled_years(live, variant, sets)
            out["risk_protection"] = bool(dd_v <= dd_l + MAX_DD_DEEPER
                                          and (calls_fewer or dd_v < dd_l - MAX_DD_DEEPER))
            if out["risk_protection"]:
                why.append(_risk_note(margin))
        out.update(decision="reject", why=why)
        return out
    short = [st for st in sets if ns[st] < DRAWS]
    if short:
        out.update(undecided=short, decision="more draws",
                   why=[f"set {st}: {ns[st]} paired draws, the rule needs {DRAWS}" for st in short])
        return out
    yrs = _pooled_years(live, variant, sets)
    loyo = None
    if yrs:
        lo, hi = periods.pool_span(sets)
        lens = {y: _year_len(y, lo, hi) for y in yrs}
        T = sum(lens.values())
        tot = sum(yrs.values())
        loyo = {y: (tot - yrs[y]) / (T - lens[y]) for y in yrs if T - lens[y] > 0}
    out["years"], out["loyo"] = yrs, loyo
    crit = {"pooled_gain": gain >= MIN_GAIN,
            "not_one_year": (min(loyo.values()) > LOYO_MIN) if loyo else None,
            "margin_calls": calls_ok,
            "drawdown": dd_v <= dd_l + MAX_DD_DEEPER}
    out["criteria"] = crit
    if _undecided(gain, MIN_GAIN, se) and any(ns[st] < MORE_DRAWS for st in sets):
        und = [st for st in sets if ns[st] < MORE_DRAWS]
        out.update(undecided=und, decision="more draws",
                   why=[f"pooled gain {gain:+.2f} +- {se:.2f} log pts within {UNDECIDED_SE:g} SE of {MIN_GAIN:+.1f}: "
                        f"run {MORE_DRAWS} draws in sets {', '.join(und)}"])
        return out
    if not crit["pooled_gain"]:
        why.append(f"pooled gain {gain:+.2f} +- {se:.2f} log pts < {MIN_GAIN:+.1f}")
    if crit["not_one_year"] is False:
        y0 = min(loyo, key=loyo.get)
        why.append(f"one year carries the gain: without {y0} the pooled gain is {loyo[y0]:+.2f} log pts")
    if not crit["margin_calls"]:
        if margin == "rate":
            why.append(f"margin events significantly more frequent: {me['only_variant']} (set, draw) pairs only with the "
                       f"change against {me['only_live']} only without (one-sided exact binomial p {me['p_more']:.3f} "
                       f"< {MARGIN_P:g})")
        else:
            why.append(f"margin calls in {calls_v} (set, draw) pairs vs the live {calls_l}")
    if not crit["drawdown"]:
        why.append(f"pooled worst drawdown {dd_v:.1f} vs the live {dd_l:.1f} (> +{MAX_DD_DEEPER:g})")
    growth_ok = crit["pooled_gain"] and crit["not_one_year"] is not False
    risk_ok = crit["margin_calls"] and crit["drawdown"]
    if growth_ok and risk_ok and crit["not_one_year"] is None:
        out.update(decision="needs per-year growth",
                   why=["the records carry no per-year growth (`year_lp` of each draw's daily NAV): the one-year check "
                        "cannot run"])
        return out
    out["decision"] = "add" if (growth_ok and risk_ok) else "reject"
    out["risk_protection"] = bool(not growth_ok and dd_v <= dd_l + MAX_DD_DEEPER
                                  and (calls_fewer or dd_v < dd_l - MAX_DD_DEEPER))
    if out["risk_protection"]:
        why.append(_risk_note(margin))
    out["why"] = why
    return out


MARKET_LAGS = 3                       # PREREG183: Newey-West lags of the monthly paired difference
MARKET_PRIORS = (0.1, 0.25, 0.5)      # ... P[H1], the prior that a change is truly better, for the oFDR
MARKET_TRIALS = (10, 50)              # ... K of the deflated p (the best of K tries)


def market_noise(live: Mapping[str, Mapping[int, dict]], variant: Mapping[str, Mapping[int, dict]],
                 sets=POOLED_SETS, lags: int = MARKET_LAGS, priors=MARKET_PRIORS, trials=MARKET_TRIALS) -> dict:
    """PREREG183 (REPORT ONLY — no decision reads it): a change's gain against the MARKET's noise. The add rule's
    standard error is the draws' (`paired`): more draws remove the simulation's noise, never the history's — eight
    years hold one market. Each seed both sides have in every set: its monthly returns chained through the sets
    (`bank._months` of its daily NAV, `bank._chain`), the monthly LOG difference variant - live; their mean over the
    seeds = the change's monthly series d. Returns {"seeds", "months", "gain" (1200 x mean(d): log points a year, the
    pooled gain's monthly form), "se" (Newey-West with ``lags``, the same unit), "t", "sr_month", "skew", "kurt",
    "rho", "p" (1 - `stats.sharpe_psr`(SR; 0) with the series' autocorrelation, skewness and kurtosis: López de Prado,
    Lipton & Zoonekynd), "p_deflated" ({K: 1 - the FWER test's DSR, the change as the best of K tries}), "ofdr"
    ({P[H1]: `stats.ofdr` with the alternative at the observed Sharpe}), "addrule_false_pass" (P[a change WITHOUT
    effect shows the +0.5 gain the rule asks] = 1 - Φ(MIN_GAIN / se): the year, margin and drawdown guards not
    modelled, so an upper bound)}. Records without a daily NAV (``"nav": {"days", "values"}``, as the runs' pickles
    carry) give {"error"}."""
    from statistics import NormalDist

    import numpy as np
    import pandas as pd

    from src.analysis import stats as STS
    from src.backtest import bank as BK
    from src.backtest import periods
    sets = tuple(sets)
    seeds = sorted(set.intersection(*[set(live.get(s, {})) & set(variant.get(s, {})) for s in sets])) if sets else []
    if not seeds:
        return {"error": "no seed both sides have in every set"}
    if not all(live[s][k].get("nav") and variant[s][k].get("nav") for s in sets for k in seeds):
        return {"error": "the records carry no daily NAV (\"nav\")"}
    cols = []
    for k in seeds:
        a = BK._chain([BK._months(BK.nav_of(live[s][k], periods.get(s).lo)[0]) for s in sets])
        b = BK._chain([BK._months(BK.nav_of(variant[s][k], periods.get(s).lo)[0]) for s in sets])
        j = a.index.intersection(b.index)
        cols.append(pd.Series(np.log1p(b.loc[j].to_numpy(float)) - np.log1p(a.loc[j].to_numpy(float)), index=j))
    D = pd.concat(cols, axis=1).dropna()
    d = D.mean(axis=1).to_numpy(float)
    n = int(len(d))
    m = STS.sharpe_moments(d)
    if n < 3 or m["sr"] != m["sr"]:
        return {"error": f"{n} months: too few, or no variation"}
    se_m = float(STS.ols_hac(d, np.empty((n, 0)), lags=lags)["se"][0])
    rho = float(np.clip(m["rho"], -0.95, 0.95)) if m["rho"] == m["rho"] else 0.0
    kw = {"skew": m["skew"], "kurt": m["kurt"], "rho": rho}
    v0 = STS.sharpe_variance(0.0, n, rho=rho)
    gain, se = 1200.0 * float(d.mean()), 1200.0 * se_m
    return {"seeds": len(seeds), "months": n, "first": str(D.index[0]), "last": str(D.index[-1]),
            "gain": gain, "se": se, "t": gain / se if se > 0 else None, "lags": int(lags),
            "sr_month": m["sr"], "skew": m["skew"], "kurt": m["kurt"], "rho": m["rho"],
            "p": 1.0 - STS.sharpe_psr(m["sr"], 0.0, n, **kw),
            "p_deflated": {str(K): 1.0 - STS.deflated_sharpe_k(m["sr"], int(K), n, sr_var=v0, **kw)["dsr"]
                           for K in trials},
            "ofdr": {str(pr): (STS.ofdr(m["sr"], 0.0, m["sr"], n, float(pr), **kw) if m["sr"] > 0 else None)
                     for pr in priors},
            "ofdr_alternative": "the observed Sharpe (an effect as large as seen)",
            "addrule_false_pass": (1.0 - NormalDist().cdf(MIN_GAIN / se)) if se > 0 else None}


def pooled_summary(v: dict) -> str:
    """One line for a study's log: the pooled decision, gain, calls and drawdown, then each set's gain (reported)."""
    p = v.get("pooled", {})
    head = (f"{v['decision'].upper()} (pooled {'+'.join(v['sets'])}): {p.get('gain', float('nan')):+.2f} +- "
            f"{p.get('se', float('nan')):.2f} lp/yr, calls {p.get('calls_live')}->{p.get('calls_variant')}"
            + (f" (only with {p['margin_events']['only_variant']}, only without {p['margin_events']['only_live']}: p "
               f"{p['margin_events']['p_more']:.3f} more, {p['margin_events']['p_fewer']:.3f} fewer; "
               f"{v.get('margin_rule', 'rate')} rule)" if p.get("margin_events") else "")
            + f", dd {p.get('dd_live', float('nan')):.1f}->{p.get('dd_variant', float('nan')):.1f}")
    sets = "; ".join(f"{st} {r['gain']:+.2f}" for st, r in v["sets"].items() if r.get("n", 0) >= 2)
    line = head + (f" | sets {sets}" if sets else "")
    if v.get("loyo"):
        y0 = min(v["loyo"], key=v["loyo"].get)
        line += f" | without {y0}: {v['loyo'][y0]:+.2f}"
    if v.get("why"):
        line += " | " + "; ".join(v["why"])
    if "overfitting" in v:
        line += _report_line(v["overfitting"])
    mn = v.get("market_noise") or {}
    if mn.get("t") is not None:
        line += (f" | market noise (PREREG183): {mn['gain']:+.2f} +- {mn['se']:.2f} lp/yr over {mn['months']} months, "
                 f"t {mn['t']:.2f}, p {mn['p']:.3f}")
    return line


def summary(v: dict) -> str:
    """One line for a study's log (PREREG106's report appended when the verdict carries it)."""
    parts = []
    for s, p in v["sets"].items():
        if p.get("n"):
            parts.append(f"{s} {p['gain']:+.2f} +- {p['se']:.2f} lp (n {p['n']}, calls {p['calls_live']}->{p['calls_variant']}, "
                         f"dd {p['dd_live']:.1f}->{p['dd_variant']:.1f})")
    line = f"{v['decision'].upper()}: " + "; ".join(parts) + (f" — {'; '.join(v['why'])}" if v["why"] else "")
    if "overfitting" in v:
        line += _report_line(v["overfitting"])
    return line


def _report_line(o: dict) -> str:
    """PREREG106's report in a few words: per set z, the deflated p at the low..high N, the shrunk gain [band]."""
    parts = []
    for s, r in o["sets"].items():
        if not r.get("n"):
            continue
        bits = []
        d = r.get("deflated")
        if d is not None:
            lo, hi = d["low"], d["high"]
            if r.get("z") is None:
                bits.append("p n/a (no standard error)")
            else:
                ptxt = f"p {lo['p']:.3f}" if lo["N"] == hi["N"] else f"p {lo['p']:.3f}..{hi['p']:.3f}"
                bits.append(f"z {r['z']:.2f} {ptxt}")
        sh = r.get("shrunk")
        if sh is not None:
            bits.append(f"shrunk {sh['gain']:+.2f} [{sh['band'][0]:+.2f}..{sh['band'][1]:+.2f}] lp")
        if bits:
            parts.append(f"{s} " + ", ".join(bits))
    n = o.get("n_trials")
    head = "overfitting report"
    if n is not None:
        head += f" (N {n['low']}" + ("" if n["low"] == n["high"] else f"..{n['high']}") + ")"
    if o.get("kind"):
        head += f" ({o['kind']} retention)"
    return f" | {head}: " + "; ".join(parts) if parts else f" | {head}: no paired draws"
