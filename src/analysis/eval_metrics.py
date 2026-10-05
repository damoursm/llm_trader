"""The house MODEL metrics and the two TEST SETS, computed one way everywhere
(user directives 2026-09-24; `.claude/skills/evaluate/SKILL.md` §1 test sets, §2 metrics).

A model is evaluated on:

* **IC** — the mean per-day Spearman correlation between the model's score and
  the signed next-H/L-pivot return, over each day's scored cross-section;
* **t** — its day-clustered t (mean of the daily ICs over their standard error;
  one observation per signal date);
* **top / bottom 5%**, **3%** and **1% returns** — each day, the names with the
  highest (lowest) scores, ``ceil(pct x n)`` of them, and the mean of their
  signed next-pivot return; averaged over days, with a day-clustered t. At 1% a
  tail is 1-5 names a day on this universe, so its t is the noisiest of the set.
  The top tail is the long book a rank rule would buy (positive is good), the
  bottom tail the short book (NEGATIVE is good — its oriented return is the
  negation). The day's universe mean over the same rows is returned beside them
  as the baseline, never as a metric.

On two disjoint test sets, reported separately and never pooled: set 1 = signal
dates 2026-06-17 → 2026-09-27 (OHLCV + news since news collection began, news
from the per-source history), set 2 = 2026-09-28 → (the live pipeline with the
updated all-source news ingestion).

Rank the cross-section the model actually scores: a method that abstains with
0.0 must be passed its rows WITH a view, or its zeros become a tied block in the
middle of the ranking. Ties at a tail boundary break on the ticker, so a result
is reproducible.

The SELECTION OBJECTIVE (user directive 2026-09-25, `selection_objective` /
`selection_by_test_set`, section below) is what the next models train and are
chosen on: per run the top (long) or bottom (short) name, kept only when it is
also a new extreme against its own last 30 trading days of scores, one entry per
name and day, each earning its next-pivot return divided by the sessions it took.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

TAIL_PCTS: Tuple[float, ...] = (0.05, 0.03, 0.01)
MIN_DAY_ROWS = 20                 # a day with fewer scored, labelled rows does not count
TEST_SETS: Dict[str, Tuple[str, Optional[str]]] = {
    "set1_history": ("2026-06-17", "2026-09-27"),
    "set2_live_all_source": ("2026-09-28", None),
}


def _series_stats(values) -> Dict[str, float]:
    """Mean over days, day-clustered t, split-half means (chronological)."""
    import pandas as pd
    s = pd.Series(values, dtype=float).dropna()
    n = int(len(s))
    out = {"mean": float("nan"), "t": float("nan"), "half1": float("nan"),
           "half2": float("nan"), "days": n}
    if n == 0:
        return out
    out["mean"] = float(s.mean())
    if n >= 3:
        sd = float(s.std(ddof=1))
        out["t"] = out["mean"] / (sd / math.sqrt(n)) if sd > 0 else float("nan")
    if n >= 4:
        h = n // 2
        out["half1"], out["half2"] = float(s.iloc[:h].mean()), float(s.iloc[h:].mean())
    return out


def _tail_key(p: float) -> str:
    return f"{p * 100:g}%"


def model_metrics(df, score: str, label: str = "fwd_ret_pivot", date: str = "signal_date",
                  ticker: str = "ticker", tails: Sequence[float] = TAIL_PCTS,
                  min_rows: int = MIN_DAY_ROWS) -> dict:
    """IC to the next H/L pivot, its t, and the top/bottom tail returns (see the
    module doc). ``df`` holds one row per (day, name) the model scored; rows
    without a score or a label are dropped first."""
    import pandas as pd
    need = [date, score, label] + ([ticker] if ticker in df.columns else [])
    f = df[need].copy()
    f[score] = pd.to_numeric(f[score], errors="coerce")
    f[label] = pd.to_numeric(f[label], errors="coerce")
    f = f.dropna(subset=[score, label])
    if ticker not in f.columns:
        f[ticker] = range(len(f))
    days, ics, base = [], [], []
    tail_vals: Dict[str, list] = {f"{side}_{_tail_key(p)}": [] for p in tails for side in ("top", "bottom")}
    tail_names: Dict[str, list] = {k: [] for k in tail_vals}
    for d, g in f.groupby(date, sort=True):
        n = len(g)
        if n < min_rows or g[score].nunique() < 3:
            continue
        days.append(d)
        ics.append(float(g[score].rank().corr(g[label].rank())))
        base.append(float(g[label].mean()))
        ordered = g.sort_values([score, ticker], ascending=[False, True])
        for p in tails:
            k = max(1, math.ceil(p * n))
            top, bot = ordered.head(k), ordered.tail(k)
            tail_vals[f"top_{_tail_key(p)}"].append(float(top[label].mean()))
            tail_vals[f"bottom_{_tail_key(p)}"].append(float(bot[label].mean()))
            tail_names[f"top_{_tail_key(p)}"].append(k)
            tail_names[f"bottom_{_tail_key(p)}"].append(k)
    out = {"rows": int(len(f)), "days": len(days),
           "first_day": str(days[0]) if days else None, "last_day": str(days[-1]) if days else None,
           "ic": _series_stats(ics), "universe_mean": _series_stats(base)}
    for k, vals in tail_vals.items():
        st = _series_stats(vals)
        st["names_per_day"] = (sum(tail_names[k]) / len(tail_names[k])) if tail_names[k] else 0.0
        out[k] = st
    return out


def split_test_sets(df, date: str = "signal_date") -> dict:
    """``{set name: the rows of df inside it}`` (dates compared as ISO strings)."""
    d = df[date].astype(str).str[:10]
    out = {}
    for name, (lo, hi) in TEST_SETS.items():
        mask = d >= lo
        if hi is not None:
            mask &= d <= hi
        out[name] = df[mask]
    return out


def by_test_set(df, score: str, **kw) -> dict:
    """`model_metrics` on EACH test set, separately — the house report shape."""
    date = kw.get("date", "signal_date")
    return {name: (model_metrics(part, score, **kw) if len(part) else {"rows": 0, "days": 0})
            for name, part in split_test_sets(df, date=date).items()}


# ── the SELECTION OBJECTIVE (user directive 2026-09-25) ─────────────────────
#
# The next models (long and short trained separately; intraday and daily) are
# trained and chosen on what the book does with their scores:
#
#   1. each RUN, rank the scored names and take the top (long) or bottom
#      (short) `top_n` — 1 by default;
#   2. keep a pick only if its score is ALSO extreme for that name: strictly
#      above (long) / below (short) every score it got over the previous
#      `own_window_days` trading days — the live freshness rule
#      (`signals.score_history`), a name with fewer than `own_min_history`
#      prior scores kept (no standing is the limiting case of fresh);
#   3. one entry per name, side and day (its first qualifying run);
#   4. each entry earns its RETURN PER DAY: the side-oriented move to the next
#      H/L pivot divided by the sessions it took (30-minute RTH bars from the
#      entry to the pivot's extreme / 13), floored at `floor_days` so a pivot
#      one bar away cannot turn a 1% move into 13% a day.
#
# The objective is the mean over days of the day's mean entry return per day,
# with a day-clustered t — "a few entries a day or less" by construction. The
# pick is made on the SCORE alone; its label is looked up afterwards, so a
# top-ranked name without a label counts as an unlabeled entry and is never
# replaced by the next name (that would be look-ahead). Same hindsight caveat as
# every pivot metric: the pivot's extreme is known only afterwards, so a model
# chosen on this must still pass an implementable exit before it trades.

BARS_PER_DAY = 13                 # regular-hours 30-minute bars in a session
MIN_RUN_ROWS = 20                 # a run with fewer scored names is not a cross-section
OWN_WINDOW_DAYS = 30              # = settings.rank_entry_fresh_window_days
OWN_MIN_HISTORY = 10              # = settings.rank_entry_fresh_min_history


def return_per_day(ret_pct, bars_ahead, floor_days: float = 1.0,
                   bars_per_day: int = BARS_PER_DAY):
    """Return to the next pivot divided by the sessions it took (``bars_ahead``
    30-minute RTH bars from the entry through the pivot's extreme — the label's
    own ``bars_ahead`` — over ``bars_per_day``), floored at ``floor_days``.
    NaN in, NaN out."""
    import numpy as np
    r = np.asarray(ret_pct, dtype=float)
    d = np.asarray(bars_ahead, dtype=float) / float(bars_per_day)
    return r / np.maximum(d, float(floor_days))


def own_history_standing(df, score: str, date: str = "signal_date", ticker: str = "ticker",
                         window_days: int = OWN_WINDOW_DAYS):
    """``DataFrame[n_prior, prior_max, prior_min]`` aligned to ``df``: each row's
    ticker's scores over the ``window_days`` DISTINCT signal dates of the frame
    strictly before the row's date — every run of those days, never the row's
    own day, so the answer is stable across a day. The window of
    `signals.score_history.load_standings`."""
    import numpy as np
    import pandas as pd
    s = pd.to_numeric(df[score], errors="coerce").to_numpy(dtype=float)
    d = df[date].astype(str).str[:10].to_numpy()
    tk = df[ticker].astype(str).to_numpy()
    dates = pd.Index(sorted(set(d)))
    ok = ~np.isnan(s)
    g = (pd.DataFrame({"d": d[ok], "tk": tk[ok], "s": s[ok]})
         .groupby(["d", "tk"])["s"].agg(["max", "min", "count"]))
    w = int(window_days)
    n = len(df)
    n_prior, p_max, p_min = np.zeros(n), np.full(n, np.nan), np.full(n, np.nan)
    if not g.empty:
        mx = g["max"].unstack("tk").reindex(dates)
        pmx = mx.rolling(w, min_periods=1).max().shift(1).to_numpy()
        pmn = g["min"].unstack("tk").reindex(dates).rolling(w, min_periods=1).min().shift(1).to_numpy()
        pct = (g["count"].unstack("tk").reindex(dates).fillna(0)
               .rolling(w, min_periods=1).sum().shift(1).fillna(0).to_numpy())
        di = dates.get_indexer(d)
        ti = mx.columns.get_indexer(tk)
        has = ti >= 0                                # a ticker never scored has no standing
        n_prior[has] = pct[di[has], ti[has]]
        p_max[has] = pmx[di[has], ti[has]]
        p_min[has] = pmn[di[has], ti[has]]
    # positional, so a caller's duplicate index labels cannot misalign it
    return pd.DataFrame({"n_prior": n_prior, "prior_max": p_max, "prior_min": p_min}, index=df.index)


def fresh_mask(score_values, standing, side: str, min_history: int = OWN_MIN_HISTORY):
    """The live freshness rule on arrays: no standing (< ``min_history`` prior
    scores) OR a new extreme for the name — strictly above its prior max (long)
    or below its prior min (short)."""
    import numpy as np
    s = np.asarray(score_values, dtype=float)
    novice = standing["n_prior"].to_numpy(dtype=float) < float(min_history)
    with np.errstate(invalid="ignore"):
        if _side(side) == "long":
            ext = s > standing["prior_max"].to_numpy(dtype=float)
        else:
            ext = s < standing["prior_min"].to_numpy(dtype=float)
    return (novice | ext) & ~np.isnan(s)


def _side(side: str) -> str:
    s = str(side).lower()
    if s in ("long", "l", "buy", "bullish"):
        return "long"
    if s in ("short", "s", "sell", "bearish"):
        return "short"
    raise ValueError(f"side must be long or short, got {side!r}")


def _selection(df, score: str, side: str, label: str = "fwd_ret_pivot", bars: str = "bars_ahead",
               run: str = "run_id", date: str = "signal_date", ticker: str = "ticker",
               top_n: int = 1, own_history: bool = True, own_window_days: int = OWN_WINDOW_DAYS,
               own_min_history: int = OWN_MIN_HISTORY, floor_days: float = 1.0,
               bars_per_day: int = BARS_PER_DAY, min_run_rows: int = MIN_RUN_ROWS):
    """(picks, days, baseline): every run's top-``top_n`` picks, each flagged
    ``_fresh`` (passed the own-history rule) and ``_entry`` (taken: fresh when
    the rule is on, and the name's first pick of that day and side); the days
    with at least one eligible run; and each day's all-names mean return per
    day on the same side (what an arbitrary pick earned — context, never the
    metric)."""
    import numpy as np
    import pandas as pd
    sd = _side(side)
    sign = 1.0 if sd == "long" else -1.0
    f = df.copy()                        # every column rides along, for `selection_entries`
    f[score] = pd.to_numeric(f[score], errors="coerce")
    f = f[f[score].notna()]
    f["_d"] = f[date].astype(str).str[:10]
    st = own_history_standing(f, score, date="_d", ticker=ticker, window_days=own_window_days)
    f["_fresh"] = fresh_mask(f[score].to_numpy(), st, sd, own_min_history)
    lab = pd.to_numeric(f[label], errors="coerce") if label in f.columns else pd.Series(np.nan, index=f.index)
    bar = pd.to_numeric(f[bars], errors="coerce") if bars in f.columns else pd.Series(np.nan, index=f.index)
    f["_ret"] = sign * lab
    f["_days"] = bar / float(bars_per_day)
    f["_rpd"] = sign * return_per_day(lab.to_numpy(), bar.to_numpy(), floor_days, bars_per_day)
    size = f.groupby(run)[score].transform("size")
    f = f[size >= int(min_run_rows)]
    days = sorted(f["_d"].unique())
    baseline = f.groupby("_d")["_rpd"].mean()
    ordered = f.sort_values([run, score, ticker], ascending=[True, sd == "short", True], kind="mergesort")
    # the frame's own row labels are KEPT: a caller joins entries back to the
    # frame (e.g. a same-run control), so renumbering them would silently pair
    # each entry with some other row
    picks = ordered.groupby(run, sort=False).head(int(top_n))
    cand = picks["_fresh"].to_numpy() if own_history else np.ones(len(picks), bool)
    first = ~picks[cand].duplicated(["_d", ticker], keep="first")     # earliest run first (sorted)
    entry = np.zeros(len(picks), bool)
    entry[np.flatnonzero(cand)[first.to_numpy()]] = True
    picks = picks.assign(_entry=entry)
    return picks, days, baseline


def _selection_stats(picks, days, baseline, side: str) -> dict:
    import pandas as pd
    ent = picks[picks["_entry"]]
    lab = ent[ent["_rpd"].notna()]
    daily = lab.groupby("_d")["_rpd"].mean()
    daily_ret = lab.groupby("_d")["_ret"].mean()
    n_days = len(days)
    return {
        "side": _side(side), "days": n_days,
        "first_day": days[0] if days else None, "last_day": days[-1] if days else None,
        "entries": int(len(ent)), "entries_labeled": int(len(lab)),
        "entries_per_day": (len(ent) / n_days) if n_days else float("nan"),
        "days_with_entry": int(daily.size),
        "fresh_share": float(picks["_fresh"].mean()) if len(picks) else float("nan"),
        "objective": _series_stats(daily.sort_index()),              # mean return per day, per entry-day
        "return": _series_stats(daily_ret.sort_index()),              # the same entries' raw pivot return
        "days_to_pivot": float(lab["_days"].mean()) if len(lab) else float("nan"),
        "baseline": _series_stats(pd.Series(baseline).reindex(days).dropna()),
    }


def selection_entries(df, score: str, side: str, **kw):
    """The ENTRIES the selection rule takes (one row each, the frame's own
    columns plus ``_d`` / ``_ret`` / ``_days`` / ``_rpd``), for scoring the same
    picks on another return — e.g. a realizable exit, the check against the
    pivot label's hindsight."""
    picks, _days, _base = _selection(df, score, side, **kw)
    return picks[picks["_entry"]].drop(columns=["_entry"])


def selection_objective(df, score: str, side: str, **kw) -> dict:
    """The selection objective on one frame (see the section doc above).

    ``df`` holds one row per (run, name) the model scored — every run, not one
    per day, since the rule picks per run and the own-history window reads every
    run of the prior days. ``run`` must sort chronologically (the panel's
    ``run_id`` does; for arrays build e.g. ``day * 100 + bar``). ``label`` is the
    signed next-pivot return in % and ``bars`` its ``bars_ahead``; rows without a
    score are not in the cross-section, rows without a label still are."""
    picks, days, base = _selection(df, score, side, **kw)
    return _selection_stats(picks, days, base, side)


def selection_by_windows(df, score: str, side: str, windows: Dict[str, Tuple[str, Optional[str]]],
                         **kw) -> dict:
    """`selection_objective` on each ``{name: (first_day, last_day or None)}``
    window. The rule runs on the WHOLE frame first, so a window's first days
    read the own-history of the rows before it exactly as the live rule would;
    only the entries and days are split."""
    picks, days, base = _selection(df, score, side, **kw)
    out = {}
    for name, (lo, hi) in windows.items():
        keep = [d for d in days if d >= lo and (hi is None or d <= hi)]
        p = picks[picks["_d"].isin(keep)]
        out[name] = _selection_stats(p, keep, base, side) if keep else {"days": 0, "entries": 0}
    return out


def selection_by_test_set(df, score: str, side: str, **kw) -> dict:
    """`selection_objective` on EACH test set (`selection_by_windows` over
    `TEST_SETS`): set 2's first days read set 1's scores as the live rule would."""
    return selection_by_windows(df, score, side, TEST_SETS, **kw)
