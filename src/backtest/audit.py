"""THE RECURRING OVERFITTING AUDIT (PREREG106 item 4, registered 2026-10-14 13:09 ET). PREREG103's global evaluation —
every live model on every set, in sample against out of sample — rerun from SAVED OUTPUTS, so that a model whose
out-of-sample label degrades shows the day it happens.

    python -m src.backtest.audit [--sets A0,A,B,D] [--day YYYY-MM-DD] [--no-write]
    python -m src.backtest.audit --parity          # this port against PREREG103's own numbers (global103.json)

WHAT IT READS. Never the research harness: no `R63._init()`, `live_books` or account run.
* Each arm's trades: the reliability cards' `trades_base.pkl` (`reliability.build`: every arm alone through the account
  engine, `Mimic()` seed 0). Sets A and B come from the latest `cache/ml/reliability/<day>/`, A0 from the latest
  `<day>_A0/`, D from the latest build holding D (`python -m src.backtest.reliability --build --sets D`, once the
  research stores reach set D).
* The combination: the live configuration's account draws (`<research>/studies/prereg95/<set>_LIVE_S2_s<seed>.pkl`,
  PREREG101's 20 stratified draws a set, seeds from `prereg101/account101.json`). A set without draw files has no
  combination label.
* The LONG histories (the dip book 2007-2020, the crash book 1988-2020): CARRIED from PREREG103's `global103.json`.
  They are a fixed history, and rebuilding them needs the research repository's long97.py.
* The live record (set C): the latest `cache/ml/live_calibration/<day>.json` (``since.arms``).
* The trial ledger (`trials.py`): each model's trial count for its deflated Sharpe ratio, and every set's uses.

WHAT IT COMPUTES. PREREG103's registered construction, ported; ``--parity`` reproduces its table 3 to the last digit.
* Per model x set: n, the average and median net trade, the hold, RETURN PER DAY with its 95% bootstrap interval
  (10,000 resamples over trades, seed 0), the GROSS win rate, the worst trade, the top-5 share, the t of the mean
  trade.
* Per model x OUT-OF-SAMPLE set: PREREG103's design, plus set D, which is out of sample for every model. DECAY = the
  OOS return per day / the IS one (IS = the average of the model's in-sample sets), with its bootstrap interval, and
  the registered LABEL: ROBUST (the OOS interval above 0, decay >= 0.5), WEAKER OUT OF SAMPLE (above 0, decay <
  0.5), FAILS OUT OF SAMPLE (the OOS interval includes 0). Set D, still accruing, is labelled only once it holds
  `MIN_D_TRADES` trades; the other sets keep PREREG103's rule as registered.
* Each arm's deflated Sharpe ratio on its OOS sets, N from the trial ledger (low and high).
* THE CHANGES against the previous audit (the latest earlier file; before any audit, PREREG103 itself). A label that
  moved down — ROBUST -> WEAKER -> FAILS — becomes a note for the email's scorer line (`health`): never an alarm,
  never a trading change.

Not rerun: PREREG103's regime-matched cells (table 4), the parameter grid (table 5) and the account's years against
SPY / QQQ (table 2's years). NOT WIRED: no scheduler or EOD step calls this module yet (the hook is in
docs/OVERFITTING.md).
"""
from __future__ import annotations

import json
import math
import os
import pickle
import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.backtest import periods

REPO = Path(__file__).resolve().parents[2]
AUDIT_DIR = REPO / "cache" / "ml" / "audit"
REL_DIR = REPO / "cache" / "ml" / "reliability"
ET = "America/New_York"
REGISTRATION = "PREREG106.md (2026-10-14 13:09 ET)"
B_BOOT, SEED = 10_000, 0                 # PREREG103: 10,000 resamples over trades, seed 0
ARMS = ("vol", "etf", "fall", "dip", "crash", "cap")
SETS = ("A0", "A", "B", "D")             # the sets the audit recomputes (LONG is carried, C is the live record)
MIN_D_TRADES = 10                        # set D (accruing) is labelled from this many trades (positions per draw)
DRAW_FILE = "{set}_LIVE_S2_s{seed}.pkl"  # the live configuration's account draws (PREREG101 / 103)
# PREREG103's design (fixed at its registration) + set D, out of sample for every model (no study may tune on it)
DESIGN = {
    "vol": {"IS": ("A", "B"), "OOS": ("A0", "D")},
    "etf": {"IS": ("A", "B"), "OOS": ("A0", "D")},
    "fall": {"IS": ("A",), "OOS": ("B", "A0", "D")},           # B is out of sample for the BASE rule
    "dip": {"IS": ("A", "B"), "OOS": ("LONG", "A0", "D")},
    "crash": {"IS": (), "OOS": ("D",)},                         # no IS split: context only, never a label
    "cap": {"IS": (), "OOS": ("D",)},                           # chosen on A0 + A + B (PREREG165): as the crash book
    "combination": {"IS": ("A", "B"), "OOS": ("A0", "D")},
}
RANK = {"FAILS OUT OF SAMPLE": 0, "WEAKER OUT OF SAMPLE": 1, "ROBUST": 2}
MODEL_NAME = {"vol": "the VOL arm", "etf": "the ETF arm", "fall": "the FALL arm", "dip": "the DIP book",
              "crash": "the CRASH book", "cap": "the CAPITULATION book", "combination": "THE COMBINATION"}
ROW_FIELDS = ("n", "rpd", "rpd_IS", "decay", "decay_lo95", "decay_hi95", "diff", "diff_lo95", "diff_hi95",
              "share_resamples_IS_le_0", "oos_boot_lo95", "oos_boot_hi95", "oos_lo95", "oos_hi95", "pairwise")


# ── the research repository's saved outputs ──────────────────────────────────

def research_dir() -> Path:
    from config.settings import settings
    return Path(settings.backtest_research_dir)


def draws_dir() -> Path:
    return research_dir() / "studies" / "prereg95"


def seeds_file() -> Path:
    return research_dir() / "studies" / "prereg101" / "account101.json"


def baseline_json() -> Path:
    """PREREG103's output: audit zero, and the LONG histories' carried numbers."""
    return research_dir() / "studies" / "prereg103" / "global103.json"


def bench_history_json() -> Path:
    """PREREG120's carried benchmark histories (the bank's bar before the daily cache's first session): the indices
    stitched before their ETF, the 3x funds synthetic before their inception."""
    return research_dir() / "studies" / "prereg120" / "bench_history.json"


def live_calibration_dir() -> Path:
    from config.settings import settings
    p = Path(getattr(settings, "live_calibration_dir", "cache/ml/live_calibration"))
    return p if p.is_absolute() else REPO / p


# ── PREREG103's statistics (global103.py, ported line for line; `--parity` checks them) ─────────

def per_day_vec(m, d):
    """`account.per_day` on arrays: (1 + m / 100)^(1 / d) - 1 in %, -100 where the average trade is <= -100%."""
    m, d = np.asarray(m, float), np.asarray(d, float)
    b = 1.0 + m / 100.0
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(b > 0, (np.abs(b) ** (1.0 / d) - 1.0) * 100.0, -100.0)
    return np.where(d > 0, out, np.nan)


def boot_rpd(net, days, rng, B=B_BOOT, chunk=500):
    """Return per day of ``B`` resamples of the trades (with replacement, the set's own n)."""
    net, days = np.asarray(net, float), np.asarray(days, float)
    n = len(net)
    out = np.empty(B)
    for s in range(0, B, chunk):
        k = min(chunk, B - s)
        idx = rng.integers(0, n, size=(k, n))
        out[s:s + k] = per_day_vec(net[idx].mean(1), days[idx].mean(1))
    return out


def pct(x, q):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(v) for v in np.percentile(x, q)] if len(x) else [None] * len(q)


def cell(t: pd.DataFrame, years=None, boot=True, seed=SEED) -> dict:
    """A model's statistics on a set's trades (the cards' `reliability.stats` + the t of the mean, the per-trade
    Sharpe and the return per day's bootstrap interval)."""
    from src.backtest import reliability as REL
    s = REL.stats(t)
    if not s.get("n"):
        return s
    net = t.net.to_numpy(float)
    sd = float(net.std(ddof=1)) if len(net) > 1 else float("nan")
    s["sd_trade"] = sd
    s["t_mean"] = float(net.mean() / (sd / math.sqrt(len(net)))) if sd > 0 else None
    s["sharpe_trade"] = float(net.mean() / sd) if sd > 0 else None
    if years:
        s["years"], s["per_year"] = years, s["n"] / years
    if boot and len(net) >= 2:
        d = boot_rpd(net, t.days.to_numpy(float), np.random.default_rng(seed))
        s["rpd_lo95"], s["rpd_hi95"] = pct(d, [2.5, 97.5])
    return s


def cluster_boot(draws: list, rng, mask=None, B=B_BOOT, chunk=500):
    """The draws' mean return per day under a bootstrap that resamples ENTRY DAYS (with replacement, the same days for
    every draw): the draws replay one history, so the history is resampled once and every draw is read on it."""
    P = [d if mask is None else d[mask(d)] for d in draws]
    P = [d for d in P if len(d)]
    if not P:
        return np.full(B, np.nan)
    days = np.array(sorted(set().union(*[set(d.entry_day) for d in P])), dtype="datetime64[ns]")
    U, D = len(days), len(P)
    CNT, SR, SH = np.zeros((U, D)), np.zeros((U, D)), np.zeros((U, D))
    for j, d in enumerate(P):
        ii = np.searchsorted(days, d.entry_day.to_numpy(dtype="datetime64[ns]"))
        np.add.at(CNT[:, j], ii, 1.0)
        np.add.at(SR[:, j], ii, d.net.to_numpy(float))
        np.add.at(SH[:, j], ii, d.days.to_numpy(float))
    out = np.empty(B)
    for s in range(0, B, chunk):
        k = min(chunk, B - s)
        idx = rng.integers(0, U, size=(k, U))
        C = np.bincount((np.arange(k)[:, None] * U + idx).ravel(), minlength=k * U).reshape(k, U).astype(float)
        n = C @ CNT
        with np.errstate(invalid="ignore", divide="ignore"):
            r = per_day_vec((C @ SR) / n, (C @ SH) / n)
        r[n == 0] = np.nan
        out[s:s + k] = np.nanmean(r, axis=1)
    return out


def draw_rpd(d: pd.DataFrame) -> float:
    from src.backtest.account import per_day
    return per_day(float(d.net.mean()), float(d.days.mean())) if len(d) else float("nan")


def label(oos_lo, decay):
    """PREREG103's registered labels: ROBUST = the OOS interval above 0 and decay >= 0.5; WEAKER OUT OF SAMPLE = the
    OOS interval above 0, decay < 0.5; FAILS OUT OF SAMPLE = the OOS interval includes 0 (or lies below)."""
    if oos_lo is None or decay is None:
        return None
    if oos_lo <= 0:
        return "FAILS OUT OF SAMPLE"
    return "ROBUST" if decay >= 0.5 else "WEAKER OUT OF SAMPLE"


def arm_sets(T, arm, s, mask=None):
    t = T[(T.book == arm) & (T.set == s)]
    return t if mask is None else t[mask(t)]


def decay_row(T, model, oos, IS, draws=None, mask=None, need=1):
    """OOS return per day / IS return per day (IS = the average of its IS sets' return per day), its bootstrap
    interval (each set's trades resampled on its own child of seed 0's SeedSequence: OOS first, then the IS sets in
    order) — PREREG103's `decay_row`."""
    from src.backtest.account import per_day
    gens = [np.random.default_rng(c) for c in np.random.SeedSequence(SEED).spawn(1 + len(IS))]
    if model == "combination":
        def get(s):
            return [r["pos"] for r in draws[s].values()]
        pt = {s: float(np.mean([draw_rpd(d if mask is None else d[mask(d)]) for d in get(s)])) for s in (oos,) + IS}
        ns = {s: float(np.mean([len(d if mask is None else d[mask(d)]) for d in get(s)])) for s in (oos,) + IS}
        dist = {s: cluster_boot(get(s), g, mask=mask) for s, g in zip((oos,) + IS, gens)}
    else:
        tabs = {s: arm_sets(T, model, s, mask) for s in (oos,) + IS}
        ns = {s: len(t) for s, t in tabs.items()}
        pt = {s: (per_day(float(t.net.mean()), float(t.days.mean())) if len(t) else float("nan")) for s, t in tabs.items()}
        dist = {s: (boot_rpd(t.net.to_numpy(float), t.days.to_numpy(float), g) if len(t) >= 2 else np.full(B_BOOT, np.nan))
                for (s, t), g in zip(tabs.items(), gens)}
    use_is = tuple(s for s in IS if ns[s] >= need)
    if not use_is or ns[oos] < need:
        return {"oos": oos, "IS": list(IS), "n": ns, "rpd": pt, "comparable": False}
    is_pt = float(np.mean([pt[s] for s in use_is]))
    is_d = np.mean([dist[s] for s in use_is], axis=0)
    ratio = dist[oos] / is_d
    diff = dist[oos] - is_d
    return {"oos": oos, "IS": list(IS), "IS_used": list(use_is), "n": ns, "rpd": pt, "rpd_IS": is_pt,
            "decay": float(pt[oos] / is_pt) if is_pt else None,
            "decay_lo95": pct(ratio, [2.5])[0], "decay_hi95": pct(ratio, [97.5])[0],
            "diff": float(pt[oos] - is_pt), "diff_lo95": pct(diff, [2.5])[0], "diff_hi95": pct(diff, [97.5])[0],
            "share_resamples_IS_le_0": float(np.mean(is_d <= 0)),
            "oos_boot_lo95": pct(dist[oos], [2.5])[0], "oos_boot_hi95": pct(dist[oos], [97.5])[0],
            "pairwise": {s: (float(pt[oos] / pt[s]) if pt[s] else None) for s in use_is}, "comparable": True}


def dsr_trades(net_pct, n_trials: int) -> dict:
    """PREREG103's per-trade deflated Sharpe: the mean / sd of the net trades deflated for ``n_trials`` trials, the
    trials' Sharpe variance being a skill-less trial's sampling variance 1 / (T - 1) (independent trials: conservative
    for the correlated variants a development tries)."""
    from src.analysis import stats as STS
    x = np.asarray(net_pct, float) / 100.0
    T = len(x)
    sr = float(x.mean() / x.std(ddof=1))
    sk = float(pd.Series(x).skew())
    ku = float(pd.Series(x).kurt()) + 3.0
    return STS.deflated_sharpe(sr, int(n_trials), T, sk, ku, sr_var=1.0 / (T - 1))


# ── the inputs ───────────────────────────────────────────────────────────────

def card_dirs(root=None) -> Dict[str, Path]:
    """The latest reliability build holding each set: {set: directory}. `reliability.build` names a build
    ``<day>`` for sets A and B and ``<day>_X-Y`` for any other selection (X, Y)."""
    found: Dict[str, Tuple[Tuple[str, float], Path]] = {}
    for p in Path(root if root is not None else REL_DIR).glob("20??-??-??*"):
        if not (p / "trades_base.pkl").exists():
            continue
        day, _, suffix = p.name.partition("_")
        try:
            date.fromisoformat(day)
        except ValueError:
            continue
        rank = (day, (p / "trades_base.pkl").stat().st_mtime)
        for s in (("A", "B") if not suffix else tuple(x for x in suffix.split("-") if x)):
            if s not in found or rank > found[s][0]:
                found[s] = (rank, p)
    return {s: p for s, (_, p) in found.items()}


def uniform(t: pd.DataFrame, keep: Sequence[str], now=None) -> Tuple[pd.DataFrame, int]:
    """One table for every source (PREREG103's `uniform`): book, ticker, entry_day (naive ET date), net / gross (%),
    days (24-hour hold), decision, year, set — the set of each trade's decision (`periods.set_of`, A0 and, when kept, D
    in C's place); the trades outside ``keep`` are dropped and counted."""
    ent = pd.to_datetime(t["entry"])
    if getattr(ent.dt, "tz", None) is not None:
        ent = ent.dt.tz_convert(ET).dt.tz_localize(None)
    out = pd.DataFrame({"book": t["book"].astype(str).to_numpy(), "ticker": t["ticker"].astype(str).to_numpy(),
                        "entry_day": ent.dt.normalize().to_numpy(), "net": t["net"].astype(float).to_numpy(),
                        "gross": t["gross"].astype(float).to_numpy(), "days": t["days"].astype(float).to_numpy(),
                        "decision": list(t["decision"])})
    out["year"] = pd.DatetimeIndex(out.entry_day).year
    dec = [d if d is not None else e.date() for d, e in zip(out.decision, pd.DatetimeIndex(out.entry_day))]
    out["set"] = [periods.set_of(d, extra=True, forward="D" in keep, now=now) for d in dec]
    inside = out.set.isin(list(keep)).to_numpy()
    return out[inside].reset_index(drop=True), int((~inside).sum())


def load_draws(st: str, seeds, root=None) -> dict:
    """The live configuration's account draws of set ``st`` (PREREG103's `load_draws`): {seed: {"pos": positions
    (book, entry_day, exit_day, net %, pnl, days), "growth", "maxdd", "calls", "rpd_all"}}; a missing file is
    skipped."""
    from src.backtest.bank import canonical_book
    out = {}
    for sd in seeds:
        p = Path(root if root is not None else draws_dir()) / DRAW_FILE.format(set=st, seed=sd)
        if not p.exists():
            continue
        r = pickle.load(open(p, "rb"))
        P = pd.DataFrame(r["pos"])
        ent = pd.to_datetime(P.entry_ns, unit="ns", utc=True).dt.tz_convert(ET).dt.tz_localize(None)
        ext = pd.to_datetime(P.exit_ns, unit="ns", utc=True).dt.tz_convert(ET).dt.tz_localize(None)
        P = pd.DataFrame({"book": P.strategy.map(canonical_book), "entry_day": ent.dt.normalize(), "exit_day": ext.dt.normalize(),
                          "net": P.ret.astype(float) * 100.0, "pnl": P.pnl.astype(float),
                          "days": (P.exit_ns - P.entry_ns).astype(float) / 86400e9})
        out[sd] = {"pos": P, "growth": r["growth"], "maxdd": r["maxdd"], "calls": r["calls"], "rpd_all": r["rpd_all"]}
    return out


def _mtime(p: Path) -> Optional[str]:
    try:
        return datetime.fromtimestamp(Path(p).stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        return None


def _now_et() -> pd.Timestamp:
    return pd.Timestamp.now(tz=ET)


def _asof(day: date) -> Optional[pd.Timestamp]:
    """The clock set D is cut at: now for today's audit, the day's end for an earlier day's."""
    now = _now_et()
    return None if day >= now.date() else pd.Timestamp(f"{day.isoformat()} 23:59", tz=ET)


def set_years(s: str, now=None) -> float:
    S = periods.get(s, now=now)
    return (pd.Timestamp(S.hi) - pd.Timestamp(S.lo)).days / 365.25


# ── the audit ────────────────────────────────────────────────────────────────

def _needed(sets: Sequence[str]) -> List[str]:
    """The sets to load: the ones asked for and every in-sample set their out-of-sample labels compare against."""
    need = set(sets)
    for d in DESIGN.values():
        if any(o in sets for o in d["OOS"]):
            need |= set(d["IS"])
    return [s for s in SETS if s in need]


def _labels(rec: Mapping) -> Dict[Tuple[str, str], dict]:
    """{(model, OOS set): {"label", "decay", ...}} of an audit file or of PREREG103's ``table3``."""
    rows = rec.get("rows")
    if rows is None:
        rows = (rec.get("table3") or {}).get("rows") or []
    return {(r["model"], r["oos"]): r for r in rows if r.get("model") and r.get("oos")}


def previous(day: date, audit_dir=None, baseline=None) -> Tuple[Optional[dict], str]:
    """The record the changes are read against: the latest audit before ``day``, else PREREG103 (audit zero)."""
    d = Path(audit_dir if audit_dir is not None else AUDIT_DIR)
    earlier = sorted(p for p in d.glob("????-??-??.json") if p.stem < day.isoformat()) if d.exists() else []
    for p in reversed(earlier):
        try:
            return json.loads(p.read_text(encoding="utf-8")), f"audit {p.stem}"
        except (OSError, ValueError):
            continue
    b = Path(baseline) if baseline is not None else baseline_json()
    try:
        return json.loads(b.read_text(encoding="utf-8")), "PREREG103 (global103.json)"
    except (OSError, ValueError):
        return None, "none"


def changes(now_rows: Sequence[dict], prev: Optional[Mapping]) -> List[dict]:
    """Each labelled (model, OOS set) against the previous record: degraded / improved / same / new."""
    before = _labels(prev) if prev else {}
    out = []
    for r in now_rows:
        if r.get("label") is None:
            continue
        k = (r["model"], r["oos"])
        b = before.get(k)
        was = b.get("label") if b else None
        if was is None:
            move = "new"
        elif RANK[r["label"]] < RANK[was]:
            move = "degraded"
        elif RANK[r["label"]] > RANK[was]:
            move = "improved"
        else:
            move = "same"
        out.append({"model": r["model"], "oos": r["oos"], "from": was, "to": r["label"], "move": move,
                    "decay_from": b.get("decay") if b else None, "decay_to": r.get("decay"),
                    "oos_lo95": r.get("oos_lo95"), "oos_hi95": r.get("oos_hi95"), "n": (r.get("n") or {}).get(r["oos"])})
    return out


def _fmt(x, nd=2, sign=False) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def notes_of(ch: Sequence[dict]) -> List[str]:
    """The scorer line's notes: every label that moved down."""
    out = []
    for c in ch:
        if c["move"] != "degraded":
            continue
        n = c.get("n")
        out.append(f"{MODEL_NAME.get(c['model'], c['model'])} on set {c['oos']}: {c['from']} -> {c['to']} (decay "
                   f"{_fmt(c['decay_from'])} -> {_fmt(c['decay_to'])}; out of sample {_fmt(c['oos_lo95'], sign=True)} .. "
                   f"{_fmt(c['oos_hi95'], sign=True)} %/day" + (f", n {n:.0f}" if isinstance(n, (int, float)) else "")
                   + ")")
    return out


def build(day: Optional[date] = None, sets: Sequence[str] = SETS, cards: Optional[Mapping[str, Path]] = None,
          rel_dir=None, draws_root=None, seeds_path=None, baseline=None, ledger=None, live_cal=None,
          audit_dir=None, now=None, compare: bool = True, log=print) -> dict:
    """The audit of ``day`` (default today, ET) over ``sets`` (each loaded with the in-sample sets its labels need).
    ``cards`` pins a reliability build per set (default: the latest holding it); the other arguments point at other
    saved outputs (tests, the parity check); ``compare=False`` skips the changes against the previous audit. Returns
    the record `write` saves."""
    t0 = time.time()
    day = day or _now_et().date()
    now = now if now is not None else (_asof(day) or _now_et())       # one instant: set D cut once for the whole audit
    sets = [s for s in sets if s in SETS]
    if not sets:
        raise ValueError(f"no set to audit among {list(SETS)}")
    load = _needed(sets)
    found = dict(cards) if cards is not None else card_dirs(rel_dir)
    from src.backtest import trials as TR

    # the arms' trades, per set from its latest build (one read per build)
    T_parts, inputs, outside, missing = [], {"cards": {}}, {}, []
    by_dir: Dict[Path, List[str]] = {}
    for s in load:
        p = found.get(s)
        if p is None:
            missing.append(s)
            continue
        by_dir.setdefault(Path(p), []).append(s)
    for p, ss in by_dir.items():
        raw = pd.read_pickle(Path(p) / "trades_base.pkl")
        u, k = uniform(raw, ss, now)
        T_parts.append(u.assign(src=Path(p).name))
        outside[Path(p).name] = k
        for s in ss:
            inputs["cards"][s] = {"dir": str(p), "modified": _mtime(Path(p) / "trades_base.pkl")}
    T = pd.concat(T_parts, ignore_index=True) if T_parts else pd.DataFrame(
        columns=["book", "ticker", "entry_day", "net", "gross", "days", "decision", "year", "set", "src"])
    days = sorted({Path(v["dir"]).name.split("_")[0] for v in inputs["cards"].values()})
    log(f"[audit] {len(T)} trades from {sorted(p.name for p in by_dir)}; no build for {missing or 'none'}")

    # the combination's draws
    seeds_p = Path(seeds_path) if seeds_path is not None else seeds_file()
    try:
        seeds = [int(x) for x in json.loads(seeds_p.read_text(encoding="utf-8"))["seeds"]]
    except (OSError, ValueError, KeyError):
        seeds = []
    draws = {s: load_draws(s, seeds, draws_root) for s in load} if seeds else {}
    draws = {s: d for s, d in draws.items() if d}
    inputs["draws"] = {"dir": str(Path(draws_root) if draws_root is not None else draws_dir()), "seeds_file": str(seeds_p),
                       "sets": {s: len(d) for s, d in draws.items()}}

    # per model x set
    models: Dict[str, dict] = {}
    for arm in ARMS:
        models[arm] = {}
        for s in load:
            t = T[(T.book == arm) & (T.set == s)]
            if len(t):
                models[arm][s] = cell(t, set_years(s, now))
    comb = {}
    for s, D in draws.items():
        pos = [r["pos"] for r in D.values()]
        mine = np.array([draw_rpd(p) for p in pos])
        lo95, hi95 = pct(cluster_boot(pos, np.random.default_rng(SEED)), [2.5, 97.5])
        g = np.array([float(r["growth"]) for r in D.values()])
        comb[s] = {"draws": len(D), "rpd_all": float(mine.mean()), "rpd_lo95": lo95, "rpd_hi95": hi95,
                   "rpd_draw_range": [float(mine.min()), float(mine.max())],
                   "growth": float(g.mean()), "growth_min": float(g.min()), "growth_max": float(g.max()),
                   "maxdd_mean": float(np.mean([float(r["maxdd"]) for r in D.values()])),
                   "margin_call_draws": int(sum(1 for r in D.values() if r["calls"])),
                   "positions_per_draw": float(np.mean([len(p) for p in pos]))}

    # decay and the registered labels
    base, base_src = None, None
    bpath = Path(baseline) if baseline is not None else baseline_json()
    try:
        base = json.loads(bpath.read_text(encoding="utf-8"))
        base_src = str(bpath)
    except (OSError, ValueError):
        pass
    carried = _labels(base) if base else {}
    rows = []
    for model, d in DESIGN.items():
        for oos in d["OOS"]:
            if oos == "LONG":
                r = carried.get((model, oos))
                if r is not None:
                    rows.append(dict(r, status="carried", source=f"PREREG103 {base.get('built_at')}"))
                continue
            if oos not in sets:
                continue
            have_oos = (oos in draws) if model == "combination" else (oos in models.get(model, {}))
            have_is = all((s in draws) if model == "combination" else (s in models.get(model, {})) for s in d["IS"])
            if not d["IS"]:
                rec = {"model": model, "oos": oos, "IS": [], "status": "no in-sample split (context only)",
                       "n": {oos: (models.get(model, {}).get(oos) or {}).get("n", 0)}, "label": None}
                rows.append(rec)
                continue
            if not have_oos or not have_is:
                rows.append({"model": model, "oos": oos, "IS": list(d["IS"]), "label": None,
                             "status": "no saved outputs" if not have_oos else "an in-sample set is missing"})
                continue
            r = decay_row(T, model, oos, tuple(d["IS"]), draws)
            src = comb.get(oos, {}) if model == "combination" else models[model][oos]
            oos_lo, oos_hi = src.get("rpd_lo95"), src.get("rpd_hi95")
            r.update(model=model, oos_lo95=oos_lo, oos_hi95=oos_hi, label=label(oos_lo, r.get("decay")), status="computed")
            n_oos = (r.get("n") or {}).get(oos, 0)
            if oos == "D" and (n_oos or 0) < MIN_D_TRADES:
                r.update(label=None, status=f"accruing: {n_oos:.0f} trades, labelled from {MIN_D_TRADES}")
            rows.append(r)

    # each arm's deflated Sharpe on its out-of-sample sets, N from the trial ledger
    cnt = {m: TR.count(m, ledger) for m in list(ARMS) + ["combination"]}
    dsr = {}
    for model in ARMS:
        c = cnt[model]
        dsr[model] = {"N_low": max(1, c["low"]), "N_high": max(1, c["high"]),
                      "N_source": "trial ledger" + ("" if c["seed"] or c["recorded"]
                                                     else " (no seed: python -m src.backtest.trials --seed)")}
        for s in DESIGN[model]["OOS"]:
            t = T[(T.book == model) & (T.set == s)] if s != "LONG" else None
            if t is None or len(t) < 3:
                continue
            net = t.net.to_numpy(float)
            r_lo, r_hi = dsr_trades(net, dsr[model]["N_low"]), dsr_trades(net, dsr[model]["N_high"])
            dsr[model][s] = {"T": int(len(t)), "sr_trade": r_lo["sr"], "psr0": r_lo["psr0"],
                             "dsr_low": r_lo["dsr"], "dsr_high": r_hi["dsr"]}

    # the live record (set C)
    live = None
    lc_dir = Path(live_cal) if live_cal is not None else live_calibration_dir()
    files = sorted(lc_dir.glob("????-??-??.json")) if lc_dir.exists() else []
    if files:
        try:
            lc = json.loads(files[-1].read_text(encoding="utf-8"))
            since = lc.get("since") or {}
            live = {"file": str(files[-1]), "from": since.get("from"), "to": since.get("to"),
                    "sessions": since.get("sessions"),
                    "arms": {a: {"closed": (since.get("arms") or {}).get(a, {}).get("n"),
                                 "open": (since.get("arms") or {}).get(a, {}).get("open_n"),
                                 "return_per_day": (since.get("arms") or {}).get(a, {}).get("return_per_day"),
                                 "verdict": (since.get("arms") or {}).get(a, {}).get("verdict")} for a in ARMS}}
        except (OSError, ValueError):
            live = None

    prev, prev_src = previous(day, audit_dir, baseline) if compare else (None, "not compared")
    ch = changes(rows, prev) if compare else []
    notes = notes_of(ch)
    windows = {s: [periods.get(s, now=now).lo, periods.get(s, now=now).hi] for s in load}
    rec = {"day": day.isoformat(), "built_at": _now_et().isoformat(timespec="seconds"), "registration": REGISTRATION,
           "design": {m: {"IS": list(d["IS"]), "OOS": list(d["OOS"])} for m, d in DESIGN.items()},
           "sets": {"asked": list(sets), "loaded": load, "windows": windows, "missing": missing,
                    "trades": {s: int((T.set == s).sum()) for s in load}, "outside_their_build": outside,
                    "mixed_builds": len(days) > 1, "build_days": days},
           "inputs": dict(inputs, baseline=base_src, ledger=str(TR.ledger_path(ledger))),
           "bootstrap": {"resamples": B_BOOT, "seed": SEED}, "min_d_trades": MIN_D_TRADES,
           "models": models, "combination": comb, "rows": rows, "dsr": dsr, "live": live,
           "trials": {"counts": cnt, "set_uses": TR.set_uses(ledger)},
           "previous": prev_src, "changes": ch, "notes": notes, "seconds": round(time.time() - t0, 1)}
    log(f"[audit] {sum(1 for r in rows if r.get('label'))} labels; changes vs {prev_src}: "
        f"{sum(1 for c in ch if c['move'] == 'degraded')} degraded; {rec['seconds']}s")
    return rec


def _json_ready(o):
    if isinstance(o, dict):
        return {str(k): _json_ready(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_ready(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return o.isoformat()
    return o


def out_path(day: date, audit_dir=None) -> Path:
    return Path(audit_dir if audit_dir is not None else AUDIT_DIR) / f"{day.isoformat()}.json"


def write(day: Optional[date] = None, audit_dir=None, **kw) -> Path:
    """Build the audit and save it as ``cache/ml/audit/<day>.json`` (atomically: the email may be reading it)."""
    day = day or _now_et().date()
    rec = build(day, audit_dir=audit_dir, **kw)
    p = out_path(day, audit_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(_json_ready(rec), indent=1, default=str), encoding="utf-8")
    from src.scheduler import db_fence
    db_fence.retry_os(lambda: os.replace(tmp, p))
    return p


def latest(audit_dir=None) -> Optional[dict]:
    try:
        d = Path(audit_dir if audit_dir is not None else AUDIT_DIR)
        files = sorted(d.glob("????-??-??.json")) if d.exists() else []
        return json.loads(files[-1].read_text(encoding="utf-8")) if files else None
    except (OSError, ValueError):
        return None


def health(audit_dir=None) -> dict:
    """The email's scorer line: one note for every out-of-sample label the latest audit found degraded since the
    audit before it (never a problem: the audit trades nothing)."""
    rec = latest(audit_dir)
    if not rec:
        return {"problems": [], "notes": []}
    return {"problems": [], "notes": [f"overfitting audit {rec.get('day')}: {n}" for n in rec.get("notes") or []]}


# ── the port against PREREG103 ───────────────────────────────────────────────

def parity(baseline=None, log=print) -> dict:
    """This module's statistics on PREREG103's own inputs (its reliability builds, PREREG101's draws) against
    PREREG103's table 3 and table 1 (sets A0, A, B): the largest absolute difference per field, the labels equal."""
    b = Path(baseline) if baseline is not None else baseline_json()
    G = json.loads(b.read_text(encoding="utf-8"))
    IN = G["inputs"]
    cards = {"A": Path(IN["cards_ab"]), "B": Path(IN["cards_ab"]), "A0": Path(IN["cards_a0"])}
    rec = build(date(2026, 10, 14), sets=("A0", "A", "B"), cards=cards, draws_root=IN["draws"],
                seeds_path=IN["account101"], baseline=b, compare=False, log=log)
    mine = {(r["model"], r["oos"]): r for r in rec["rows"] if r.get("status") == "computed"}
    ref = {(r["model"], r["oos"]): r for r in G["table3"]["rows"] if r.get("oos") != "LONG"}
    worst, labels, missing = 0.0, [], sorted(set(ref) - set(mine))

    def cmp(a, b_):
        nonlocal worst
        if isinstance(a, dict) and isinstance(b_, dict):
            for k in set(a) | set(b_):
                cmp(a.get(k), b_.get(k))
        elif isinstance(a, (int, float)) and isinstance(b_, (int, float)):
            worst = max(worst, abs(float(a) - float(b_)))
        elif a != b_:
            worst = float("inf")
    for k, r in ref.items():
        m = mine.get(k)
        if m is None:
            continue
        for f in ROW_FIELDS:
            cmp(_json_ready(m.get(f)), r.get(f))
        labels.append((k, m.get("label"), r.get("label")))
    t1 = 0.0
    for arm in ARMS:
        for s in ("A0", "A", "B"):
            a, c = rec["models"].get(arm, {}).get(s), G["table1"].get(arm, {}).get(s)
            if a is None or c is None:
                continue
            for f in ("n", "avg_trade", "med_trade", "avg_days", "return_per_day", "gross_win_rate", "worst",
                      "t_mean", "rpd_lo95", "rpd_hi95"):
                x, y = a.get(f), c.get(f)
                if x is not None and y is not None:
                    t1 = max(t1, abs(float(x) - float(y)))
    ok = not missing and worst < 1e-9 and t1 < 1e-9 and all(a == b_ for _, a, b_ in labels)
    out = {"identical": ok, "rows_compared": len(labels), "rows_missing": [list(k) for k in missing],
           "max_abs_diff_table3": worst, "max_abs_diff_table1": t1,
           "labels": [{"model": k[0], "oos": k[1], "audit": a, "prereg103": b_} for k, a, b_ in labels]}
    log(f"[audit] parity with PREREG103: {'IDENTICAL' if ok else 'DIFFERENT'} ({len(labels)} rows, table 3 max "
        f"|diff| {worst:.3g}, table 1 {t1:.3g})")
    return out


# ── the command line ─────────────────────────────────────────────────────────

def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="PREREG106's recurring overfitting audit (report only).")
    ap.add_argument("--sets", default=",".join(SETS), help="the sets to recompute (default A0,A,B,D)")
    ap.add_argument("--day", default=None, help="the audit's day (default today, ET)")
    ap.add_argument("--no-write", action="store_true", help="print the notes and changes, write nothing")
    ap.add_argument("--parity", action="store_true", help="this port against PREREG103's own numbers")
    a = ap.parse_args(argv)
    if a.parity:
        res = parity()
        print(json.dumps(_json_ready(res), indent=1))
        return 0 if res["identical"] else 1
    day = date.fromisoformat(a.day) if a.day else _now_et().date()
    sets = tuple(x.strip() for x in a.sets.split(",") if x.strip())
    if a.no_write:
        rec = build(day, sets)
        print(json.dumps(_json_ready({"changes": rec["changes"], "notes": rec["notes"], "sets": rec["sets"]}), indent=1))
        return 0
    p = write(day, sets=sets)
    rec = json.loads(p.read_text(encoding="utf-8"))
    print(json.dumps({"path": str(p), "notes": rec.get("notes"), "previous": rec.get("previous"),
                      "labels": [{"model": r["model"], "oos": r["oos"], "label": r.get("label"),
                                  "status": r.get("status")} for r in rec["rows"]]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
