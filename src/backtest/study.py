"""WEEKLY WALK-FORWARD RE-TUNING OF SEVERAL STRATEGIES TOGETHER on one account (user directives 2026-10-07: "Optuna
studies should always run on the weekly window", "tune on Sunday before the first evening session"; 2026-10-08:
"use it to get the growth ... and optimize our optuna studies. These studies could include multiple models, i.e. a
combination of a retuned vol arm and the dip buying arm").

A COMBO is one rule per arm (an arm may be absent). Every Sunday at 19:00 ET (`cut_ns`) every combo is scored on the
preceding window (3 / 6 / 12 months or everything since the start) by the ONE thing that matters: the window
account's log growth per year — a fresh $10,000 account (`account.simulate`) holding every arm's trades decided in the
window, each truncated at the cut (an open trade is marked at its last bar before it: nothing after the cut is
seen). A method then picks the combo that trades the following week:

* ``M0`` the window's best combo (argmax — the most overfit choice; a contrast);
* ``M4`` White's Reality Check (PREREG35's robust method): the best combo only if its edge over the REFERENCE combo
  (the live rules) survives the correction for having tried every combo (p < 0.10, paired 5-day-block bootstrap of
  the window's daily log returns), else the reference;
* ``optuna`` (``optuna_trials`` > 0): TPE over the grid, one categorical choice per arm, scored by the same window
  account; the best trial — seeded per week, so reproducible, but random across seeds (PREREG35).

The CHAINED account then runs once over the whole test span: each week's trades from that week's combo, every trade
to its own exit. It is compared with the reference combo's chained account (the same weeks, the live rules every
week): growth per year, log growth, return per day per arm, worst drawdown, margin calls, and the time-weighted
difference with a 21-day-block bootstrap CI.

Pieces come from adapters (`strategies/`), computed ONCE per (arm, rule) over the whole span — `window_pieces` slices
them by decision day and truncates them at a cut, which equals running the rule with that cut (the rules are causal;
the research evaluator's own cut does the same: entries after it dropped, open trades marked at the last close).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.backtest import account as A

DAY = 86400 * 10**9
EPOCH = pd.Timestamp("1970-01-01")
RC_B, RC_L, RC_P = 1000, 5, 0.10                     # Reality Check: resamples, block length (days), threshold


def dnum(x) -> int:
    return int((pd.Timestamp(x).normalize() - EPOCH).days)


def cut_of(sunday) -> int:
    """The re-tune instant: Sunday 19:00 ET (before the week's first evening session)."""
    return pd.Timestamp(f"{pd.Timestamp(sunday).date()} 19:00", tz="America/New_York").value


# ── pieces per window ────────────────────────────────────────────────────────

def window_pieces(pieces: Sequence[dict], lo_dn: int, hi_dn: int, cut_ns: Optional[int] = None) -> List[dict]:
    """The pieces decided on sessions [lo_dn, hi_dn] (``pick_day``) and entered before ``cut_ns``; a piece still open
    at the cut ends there, marked at its last bar before it (no bar yet: at its entry price), its borrow recomputed
    to that day."""
    out = []
    for p in pieces:
        d = int(p["pick_day"])
        if d < lo_dn or d > hi_dn:
            continue
        if cut_ns is not None and int(p["ens"]) >= cut_ns:
            continue
        if cut_ns is None or int(p["xns"]) <= cut_ns:
            out.append(p)
            continue
        pt = np.asarray(p["pt"], np.int64)
        k = int(np.searchsorted(pt, cut_ns, side="right"))
        q = dict(p)
        q["pt"] = pt[:k]
        for key in ("pc", "ph", "pl"):
            if key in p:
                q[key] = np.asarray(p[key], float)[:k]
        if k > 0:
            q["xns"], q["x"] = int(pt[k - 1]), float(np.asarray(p["pc"], float)[k - 1])
        else:
            q["xns"], q["x"] = int(cut_ns), float(p["e"])
        q["d_out"] = int(A.day_numbers([q["xns"]])[0])
        q.pop("full_ps", None)
        q["marked_at_cut"] = True
        out.append(q)
    return out


# ── combos ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Combo:
    """One rule per arm: ``rules`` maps an arm name to its rule key (None = the arm is off this week)."""
    label: str
    rules: Tuple[Tuple[str, Optional[str]], ...]

    def rule(self, arm: str) -> Optional[str]:
        return dict(self.rules).get(arm)


@dataclass
class Arm:
    """An arm of the study: its account rules and its pieces per rule key (each over the whole span)."""
    name: str
    strategy: A.Strategy
    pieces: Dict[str, List[dict]] = field(default_factory=dict)


def combo_pieces(arms: Dict[str, Arm], combo: Combo, lo_dn: int, hi_dn: int, cut_ns: Optional[int]) -> List[dict]:
    out: List[dict] = []
    for name, arm in arms.items():
        r = combo.rule(name)
        if r is None:
            continue
        ps = window_pieces(arm.pieces[r], lo_dn, hi_dn, cut_ns)
        out.extend(dict(p, strategy=name) for p in ps)
    return out


def daily_logret(res: dict, bdn: np.ndarray, final: float, start: float) -> np.ndarray:
    """The account's daily log returns on business days ``bdn``: the equity at each day's last event carried
    forward, the last day = the final value."""
    t = np.asarray(res.get("curve_t", []), np.int64)
    nav = np.ones(len(bdn))
    if len(t):
        v = np.asarray(res["curve_v"], float) / start
        dn = A.day_numbers(t)
        last = np.r_[np.flatnonzero(np.diff(dn) != 0), len(dn) - 1]
        j = np.searchsorted(dn[last], bdn, side="right") - 1
        nav = np.where(j >= 0, v[last][np.maximum(j, 0)], 1.0)
    nav[-1] = final / start
    return np.diff(np.log(np.r_[1.0, np.maximum(nav, 1e-6)]))


def score(arms, combo, lo, t, settle, rules, start=10_000.0):
    """One combo on the window [lo, t): (log growth per year, daily log returns, trades, margin calls)."""
    cut = cut_of(t)
    P = combo_pieces(arms, combo, dnum(lo), dnum(t) - 1, cut)
    strategies = [a.strategy for n, a in arms.items() if combo.rule(n) is not None]
    bdn = np.array([dnum(x) for x in pd.bdate_range(lo, pd.Timestamp(t) - pd.Timedelta(days=1))], np.int64)
    yrs = max((pd.Timestamp(t) - pd.Timestamp(lo)).days, 1) / 365.25
    if not P:
        return 0.0, np.zeros(len(bdn)), 0, 0
    res = A.simulate(P, strategies, rules, settle=settle, keep_curve=True)
    f = max(float(res["final"]), 1e-6 * start)
    return math.log(f / start) / yrs, daily_logret(res, bdn, f, start), int(res["trades"]), len(res["calls"])


# ── selection (PREREG35's, generic over combos) ──────────────────────────────

def _classes(X, g):
    keys, reps = {}, []
    cls = np.empty(len(X), int)
    for i in range(len(X)):
        k = (X[i].tobytes(), round(float(g[i]), 12))
        if k not in keys:
            keys[k] = len(reps)
            reps.append(i)
        cls[i] = keys[k]
    return np.array(reps), cls


def _boot_sums(Xc, nb, rng, n):
    C = np.concatenate([np.zeros((len(Xc), 1)), np.cumsum(Xc, axis=1)], axis=1)
    D = Xc.shape[1]
    s = rng.integers(0, D - RC_L + 1, size=(n, nb))
    out = np.empty((n, len(Xc)))
    for a in range(0, n, 25):
        ss = s[a:a + 25]
        out[a:a + 25] = (C[:, ss + RC_L] - C[:, ss]).sum(axis=2).T
    return out


def select(X: np.ndarray, g: np.ndarray, ref: int, seed: int) -> Tuple[Dict[str, int], dict]:
    """M0 (argmax, ties to the reference) and M4 (the Reality Check vs the reference) over the combos' daily log
    returns ``X`` (combos x days) and window log growth ``g``."""
    reps, cls = _classes(X, g)
    Xc, gc, rc = X[reps], g[reps], cls[ref]
    best = int(np.argmax(gc))
    if gc[rc] >= gc[best] - 1e-12:
        best = rc
    out = {"M0": int(reps[best])}
    D = Xc.shape[1]
    p_rc = 1.0
    if D >= RC_L and len(Xc) > 1:
        nb = max(1, D // RC_L)
        S = _boot_sums(Xc, nb, np.random.default_rng(seed), RC_B)
        d_obs = (Xc.sum(axis=1) - Xc[rc].sum()) / D
        others = np.arange(len(Xc)) != rc
        if d_obs[others].max() > 0:
            V = np.sqrt(D) * d_obs[others].max()
            dstar = (S - S[:, [rc]]) / (nb * RC_L)
            Vb = np.sqrt(D) * (dstar[:, others] - d_obs[others]).max(axis=1)
            p_rc = float((Vb >= V).mean())
    out["M4"] = out["M0"] if (out["M0"] != ref and p_rc < RC_P) else ref
    return out, {"rc_p": p_rc, "classes": int(len(reps))}


# ── the study ────────────────────────────────────────────────────────────────

@dataclass
class Study:
    """``arms`` (with their pieces per rule), the ``combos`` grid, the REFERENCE combo (the live rules), the
    Sundays [``first``, ``last``], the window lengths in months (None = everything since ``data_lo``), the account
    rules and the settlement calendar."""
    arms: Dict[str, Arm]
    combos: List[Combo]
    reference: int
    first: str
    last: str
    test_cap: str
    data_lo: str
    windows: Tuple[Optional[int], ...] = (12, None)
    rules: A.Rules = field(default_factory=A.Rules)
    methods: Tuple[str, ...] = ("M0", "M4")

    def sundays(self) -> List[pd.Timestamp]:
        return list(pd.date_range(self.first, self.last, freq="W-SUN"))

    def window_lo(self, t, months) -> pd.Timestamp:
        lo = pd.Timestamp(self.data_lo)
        return lo if months is None else max(lo, pd.Timestamp(t) - pd.DateOffset(months=months))


_W: dict = {}


def _init_worker(study_path: str, optuna_dir: Optional[str]) -> None:
    """A worker loads the study (arms with their pieces, combos, rules) once."""
    _W["study"] = pickle.load(open(study_path, "rb"))
    _W["settle"] = A.default_settle()
    if optuna_dir and optuna_dir not in __import__("sys").path:
        __import__("sys").path.insert(0, optuna_dir)


def _week(args):
    wi, t, months, trials = args
    study, settle = _W["study"], _W["settle"]
    lo = study.window_lo(t, months)
    n = len(study.combos)
    g = np.full(n, np.nan)
    X = None
    trades, calls = np.zeros(n, int), np.zeros(n, int)
    for i, c in enumerate(study.combos):
        gi, xi, tr, cl = score(study.arms, c, lo, t, settle, study.rules)
        if X is None:
            X = np.zeros((n, len(xi)))
        g[i], X[i], trades[i], calls[i] = gi, xi, tr, cl
    choice, diag = select(X, g, study.reference, seed=1000 + wi)
    if trials:
        choice["optuna"], diag["optuna_best"] = _optuna_pick(study, g, trials, seed=wi)
    return {"t": str(pd.Timestamp(t).date()), "lo": str(lo.date()), "months": months, "g": g, "trades": trades,
            "calls": calls, "choice": choice, "diag": diag}


def _optuna_pick(study: "Study", g: np.ndarray, trials: int, seed: int) -> Tuple[int, float]:
    """TPE over the grid — one categorical choice per arm (its rule, or off) — scored by the same window account
    (looked up: every combo was already scored). Returns (the best trial's combo, its score)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    index = {c.rules: i for i, c in enumerate(study.combos)}
    choices = {name: sorted({str(c.rule(name)) for c in study.combos}) for name in study.arms}

    def objective(trial):
        rules = tuple((name, None if (v := trial.suggest_categorical(name, choices[name])) == "None" else v)
                      for name in study.arms)
        i = index.get(rules)
        return float(g[i]) if i is not None and np.isfinite(g[i]) else -1e9
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(objective, n_trials=int(trials))
    rules = tuple((name, None if st.best_params[name] == "None" else st.best_params[name]) for name in study.arms)
    return index[rules], float(st.best_value)


def run_weeks(study: Study, study_path: str, workers: int = 8, optuna_trials: int = 0,
              optuna_dir: Optional[str] = None) -> Dict[Optional[int], List[dict]]:
    """Every Sunday x window: the combos' window scores and each method's pick (``optuna_trials`` > 0 adds the
    Optuna pick; ``optuna_dir`` = where the optuna package lives when the venv lacks it)."""
    save(study, study_path)
    jobs = [(wi, t, m, optuna_trials) for m in study.windows for wi, t in enumerate(study.sundays())]
    out: Dict[Optional[int], List[dict]] = {m: [] for m in study.windows}
    if workers <= 1:
        _init_worker(study_path, optuna_dir)
        res = [_week(j) for j in jobs]
    else:
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(study_path, optuna_dir)) as ex:
            res = list(ex.map(_week, jobs, chunksize=1))
    for j, r in zip(jobs, res):
        out[j[2]].append(r)
    return out


def chained(study: Study, picks: Sequence[int], settle, keep_curve: bool = True) -> dict:
    """One account over the test span: week w's trades (decided Sunday w .. Saturday) from combo ``picks[w]``, each
    to its own exit."""
    P: List[dict] = []
    used = set()
    for w, t in enumerate(study.sundays()):
        c = study.combos[picks[w]]
        hi = min(pd.Timestamp(t) + pd.Timedelta(days=6), pd.Timestamp(study.test_cap))
        for name, arm in study.arms.items():
            r = c.rule(name)
            if r is None:
                continue
            used.add(name)
            P.extend(dict(p, strategy=name) for p in window_pieces(arm.pieces[r], dnum(t), dnum(hi), None))
    strategies = [study.arms[n].strategy for n in study.arms]
    res = A.simulate(P, strategies, study.rules, settle=settle, keep_curve=keep_curve)
    lo = pd.Timestamp(study.first)
    end = pd.Timestamp(max([p["xns"] for p in P], default=lo.value), unit="ns") if P else lo
    m = A.metrics(res, lo, max(end, pd.Timestamp(study.test_cap)))
    return {"res": res, "metrics": m}


def nav_daily(res: dict, days: pd.DatetimeIndex, start: float = 10_000.0) -> np.ndarray:
    bdn = np.array([dnum(d) for d in days], np.int64)
    return np.exp(np.cumsum(daily_logret(res, bdn, max(float(res["final"]), 1e-6), start)))


def diff_ci(nav_a: np.ndarray, nav_b: np.ndarray, years: float, n: int = 4000, block: int = 21, seed: int = 7):
    """The time-weighted log-growth difference a - b per year with its 95% block-bootstrap CI and two-sided p."""
    d = np.diff(np.r_[0.0, np.log(np.maximum(nav_a, 1e-12)) - np.log(np.maximum(nav_b, 1e-12))])
    rng = np.random.default_rng(seed)
    D = len(d)
    nb = max(1, D // block)
    C = np.r_[0.0, np.cumsum(d)]
    s = rng.integers(0, max(1, D - block + 1), size=(n, nb))
    dist = (C[np.minimum(s + block, D)] - C[s]).sum(axis=1) * (D / (nb * block)) / years
    obs = float(d.sum() / years)
    p = float(min(1.0, 2 * min((dist <= 0).mean(), (dist >= 0).mean())))
    return obs, float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5)), p


def cache_key(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def save(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp"
    pickle.dump(obj, open(tmp, "wb"))
    os.replace(tmp, path)
