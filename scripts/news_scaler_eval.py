"""News scaler (recency MASS x source DIVERSITY) — does a different curve rank better?

User 2026-09-29: "the raw score is transformed using the diversity and recency
weights before getting ranked ... do an analysis on the diversity curve and
recency mass to improve performance. Use the last version of the prompt to score
with the local Qwen ... evaluate using the current live implementation."

LIVE implementation (src/analysis/sentiment.py, epoch "news" 2026-09-11 21:50 UTC):
  w(age)   = exp(-ln2 * age_h / 18)   (0 past 168 h; "now" floored to the UTC hour)
  mass     = sum w over the DIGEST's articles
  evidence = min(1, 0.45 + 0.20 * log2(1 + mass))          (saturates at mass 5.73)
  divers.  = 1 - 0.30 * 0.5 ** (unique sources - 1)        (1 src 0.70, 2 0.85, ...)
  news     = raw * evidence * divers.  -> ranked within the run (zeros abstain)

DATA (verdicts from the CURRENT prompt v7dir on the LOCAL Qwen only):
  SET 1  = the per-source rebuild `src:all` (news_replay; scored 09-24 through the
           live `analyse_sentiment`, local engine): 70 dates 07-03 -> 09-24, one
           snapshot per name per date. Mass / diversity variants only (the
           digests' articles were not stored; the source count is implied exactly
           by the live diversity factor).
  LIVE   = the live panel 09-15 -> now, LLM-scored rows joined to their stored
           digest (sentiment_digests): every variant recomputed from the exact
           articles with the live functions, incl. the recency HALF-LIFE.
  SET 2  = LIVE rows from 09-28 (all-source ingestion) — reported separately.

LABEL: the next H/L pivot on 30-minute RTH bars from the row's own tick
(pivot_rows.pivot_fwd_row); unsettled rows at the last visible close.
METRIC: per-run Spearman IC over the run's non-zero scores, averaged per day;
day-clustered t; paired per-day difference vs LIVE; same-sign halves. Tails:
top/bottom 5% per run. Side ICs: within score>0 / score<0.

PRE-REGISTERED (from memory/news-scaler-review-2026-09.md):
  H1 raise/remove the mass ceiling (cap_20, nocap) beats live
  H2 mass on the LONG side only (long_mass_only) beats live
  H3 diversity is inert (div_none ~ live)
  H4 (exploratory, LIVE only) recency half-life 6/12/36/72 h
  Bar: paired t >= 2 AND same-sign halves, on SET 1 AND LIVE, never negative on SET 2.
"""
import json
import math
import sys
import time

import numpy as np
import pandas as pd

from src.analysis import pivot_rows as PR
from src.db.connection import connect

OUT = sys.argv[1] if len(sys.argv) > 1 else "news_scaler_eval.json"
t0 = time.time()


# ── the live scaler and the variants ─────────────────────────────────────────
def e_curve(m, a=0.20, cap=True):
    m = np.asarray(m, dtype=float)
    v = 0.45 + a * np.log2(1.0 + np.maximum(m, 0.0))
    v = np.minimum(1.0, v) if cap else v
    return np.where(m <= 0, 0.0, v)


def a_for_cap(x):                      # the slope that reaches 1.0 at mass x
    return 0.55 / math.log2(1.0 + x)


def d_curve(n, depth=0.30):
    n = np.asarray(n, dtype=float)
    return np.where(n <= 0, 1.0 - depth, 1.0 - depth * 0.5 ** (n - 1.0))


def hump(m, peak=12.0, slope=0.10, floor=0.70):
    m = np.asarray(m, dtype=float)
    over = np.where(m > peak, 1.0 - slope * np.log2(np.maximum(m, peak) / peak), 1.0)
    return np.maximum(floor, over)


def variants(raw, m, n, masses=None):
    """{name: score array}. m = the live 18-h mass; masses = {hl: mass} (LIVE only)."""
    e, d = e_curve(m), d_curve(n)
    out = {
        "live": raw * e * d,
        "no_scaler": raw.copy(),
        "no_mass": raw * d,
        "div_none": raw * e,
        "cap_10": raw * e_curve(m, a_for_cap(10)) * d,
        "cap_20": raw * e_curve(m, a_for_cap(20)) * d,
        "cap_50": raw * e_curve(m, a_for_cap(50)) * d,
        "nocap": raw * e_curve(m, cap=False) * d,
        "hump12": raw * e * hump(m) * d,
        "long_mass_only": np.where(raw > 0, raw * e * d, raw * d),
        "short_mass_only": np.where(raw < 0, raw * e * d, raw * d),
        "div_deep_045": raw * e * d_curve(n, 0.45),
        "div_shallow_015": raw * e * d_curve(n, 0.15),
    }
    for hl, mh in (masses or {}).items():
        out[f"hl_{hl:g}h"] = raw * e_curve(mh) * d
    return out


# ── labels ───────────────────────────────────────────────────────────────────
_last = {}


def label(tk, when, price):
    """Next-pivot return (%) from the row's tick; unsettled -> last visible close."""
    try:
        r = PR.pivot_fwd_row(tk, when, price if price and price > 0 else None,
                             fallback_close=_close_at(tk, when))
    except Exception:
        return np.nan, False
    if r is not None:
        return float(r[0]), True
    px = price if price and price > 0 else _close_at(tk, when)
    lc = _last_close(tk)
    if px and lc:
        return (lc / px - 1.0) * 100.0, False
    return np.nan, False


def _scan(tk):
    if tk not in _last:
        try:
            _last[tk] = PR._scan(tk, None, None)
        except Exception:
            _last[tk] = None
    return _last[tk]


def _last_close(tk):
    s = _scan(tk)
    if s is None:
        return None
    n = PR._visible_bars(s, None)
    return float(s[1][n - 1]) if n else None


def _close_at(tk, when):
    s = _scan(tk)
    if s is None:
        return None
    t = PR.to_naive_utc(when)
    if t is None:
        return None
    i = int(s[0].searchsorted(t, side="left")) - 1
    return float(s[1][i]) if i >= 0 else None


# ── metrics ──────────────────────────────────────────────────────────────────
def _t(x):
    x = np.asarray([v for v in x if v == v], dtype=float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def per_day_stats(df, score, day="day", run="run"):
    """Per-day: mean over runs of the run's Spearman IC (non-zero scores), the
    long / short within-side ICs and the top/bottom 5% label means."""
    rows = []
    for (d, r), g in df.groupby([day, run], sort=True):
        g = g[(g[score] != 0) & g["y"].notna()]
        if len(g) < 20 or g[score].nunique() < 3:
            continue
        ic = g[score].rank().corr(g["y"].rank())
        gl, gs = g[g[score] > 0], g[g[score] < 0]
        icl = gl[score].rank().corr(gl["y"].rank()) if len(gl) >= 10 else np.nan
        ics = gs[score].rank().corr(gs["y"].rank()) if len(gs) >= 10 else np.nan
        k = max(1, int(math.ceil(0.05 * len(g))))
        o = g.sort_values(score, ascending=False)
        rows.append((d, r, ic, icl, ics, o["y"].head(k).mean(), o["y"].tail(k).mean()))
    p = pd.DataFrame(rows, columns=["day", "run", "ic", "ic_long", "ic_short", "top5", "bot5"])
    return p.groupby("day").mean(numeric_only=True)


def report(df, names, tag):
    base = per_day_stats(df, "live")
    out = {"rows": int(len(df)), "days": int(base.shape[0]), "settled_share": float(df["settled"].mean())}
    res = {}
    for nme in names:
        s = base if nme == "live" else per_day_stats(df, nme)
        j = s.join(base, rsuffix="_live", how="inner")
        diff = (j["ic"] - j["ic_live"]).dropna()
        h = len(diff) // 2
        res[nme] = {
            "ic": float(s["ic"].mean()), "ic_t": _t(s["ic"]),
            "ic_long": float(s["ic_long"].mean()), "ic_short": float(s["ic_short"].mean()),
            "spread5": float((s["top5"] - s["bot5"]).mean()),
            "d_ic": float(diff.mean()) if len(diff) else float("nan"), "d_t": _t(diff),
            "halves": [float(diff.iloc[:h].mean()) if h else float("nan"),
                       float(diff.iloc[h:].mean()) if len(diff) > h else float("nan")],
            "d_long": float((j["ic_long"] - j["ic_long_live"]).mean()),
            "d_spr": float(((j["top5"] - j["bot5"]) - (j["top5_live"] - j["bot5_live"])).mean()),
            "d_spr_t": _t(((j["top5"] - j["bot5"]) - (j["top5_live"] - j["bot5_live"])).dropna()),
            "d_short": float((j["ic_short"] - j["ic_short_live"]).mean()),
        }
    out["variants"] = res
    print(f"\n=== {tag}: {out['rows']:,} rows, {out['days']} days, settled {out['settled_share']:.0%} ===")
    print(f"{'variant':18s} {'IC':>7s} {'t':>6s} {'IC_L':>7s} {'IC_S':>7s} {'spr5%':>7s} | "
          f"{'dIC':>8s} {'dt':>6s} {'halves':>17s} {'dL':>7s} {'dS':>7s} {'dSpr':>7s} {'t':>6s}")
    for nme, v in res.items():
        print(f"{nme:18s} {v['ic']:+.4f} {v['ic_t']:+6.2f} {v['ic_long']:+.4f} {v['ic_short']:+.4f} "
              f"{v['spread5']:+7.3f} | {v['d_ic']:+.5f} {v['d_t']:+6.2f} "
              f"{v['halves'][0]:+.4f}/{v['halves'][1]:+.4f} {v['d_long']:+.4f} {v['d_short']:+.4f} "
              f"{v['d_spr']:+7.3f} {v['d_spr_t']:+6.2f}")
    return out


results = {}

# ══ SET 1 — the per-source rebuild (src:all) ═════════════════════════════════
with connect(read_only=True) as con:
    s1 = con.execute("""
        SELECT ticker, signal_date, generated_at, news, news_raw_score AS raw, news_recency_mass AS mass,
               news_article_count AS n_art
        FROM news_replay WHERE pool_spec = 'src:all' AND news_raw_score IS NOT NULL
          AND news IS NOT NULL AND news <> 0 AND news_raw_score <> 0""").df()
print(f"set 1 rows loaded: {len(s1):,} ({time.time() - t0:.0f}s)")
s1["mass"] = s1["mass"].fillna(0.0)
e = e_curve(s1["mass"].to_numpy())
dimp = s1["news"].to_numpy() / (s1["raw"].to_numpy() * np.where(e > 0, e, np.nan))
n_imp = np.where(dimp >= 0.9995, 12.0,
                 np.round(1.0 + np.log2(0.30 / np.clip(1.0 - dimp, 1e-9, None))))
s1["n_src"] = n_imp
recon = s1["raw"].to_numpy() * e * d_curve(n_imp)
s1["fid"] = np.abs(recon - s1["news"].to_numpy()) <= 6e-4
print(f"set 1 fidelity (live formula reproduces the stored news): {s1['fid'].mean():.2%}")
s1 = s1[s1["fid"]].copy()
lab = [label(r.ticker, r.generated_at, None) for r in s1.itertuples()]
s1["y"], s1["settled"] = [a for a, _ in lab], [b for _, b in lab]
s1["day"] = s1["signal_date"].astype(str).str[:10]
s1["run"] = s1["day"]
V1 = variants(s1["raw"].to_numpy(), s1["mass"].to_numpy(), s1["n_src"].to_numpy())
for k, v in V1.items():
    s1[k] = v
print(f"set 1 labelled ({time.time() - t0:.0f}s): {s1['y'].notna().mean():.1%}")
results["set1"] = report(s1, list(V1), "SET 1 — rebuild src:all (07-03 -> 09-24)")

# ══ LIVE — the live panel's local-Qwen rows with their exact digests ═════════
with connect(read_only=True) as con:
    lv = con.execute("""
        WITH dg AS (SELECT digest_id, any_value(articles_json) AS arts FROM sentiment_digests
                    WHERE generated_at >= '2026-09-14' GROUP BY 1)
        SELECT s.run_id AS run, s.ticker, s.generated_at, s.price, s.news, s.news_raw_score AS raw,
               s.news_recency_mass AS mass_stored, dg.arts, r.llm_sentiment_provider AS prov
        FROM signals s JOIN dg ON dg.digest_id = s.news_digest_id JOIN runs r ON r.run_id = s.run_id
        WHERE s.generated_at >= '2026-09-15' AND s.news_raw_score IS NOT NULL AND s.news_raw_score <> 0
          AND s.news IS NOT NULL AND s.news <> 0""").df()
lv = lv[~lv["prov"].fillna("").str.contains("deepseek")].reset_index(drop=True)
print(f"\nLIVE rows loaded: {len(lv):,} ({time.time() - t0:.0f}s)")
HLS = (6.0, 12.0, 36.0, 72.0)
m18, mhl, nsrc = [], {h: [] for h in HLS}, []
for r in lv.itertuples():
    now = pd.Timestamp(r.generated_at).tz_convert("UTC").floor("h")
    arts = json.loads(r.arts) if isinstance(r.arts, str) else (r.arts or [])
    ages = np.array([(now - pd.Timestamp(a["published_at"]).tz_convert("UTC")).total_seconds() / 3600.0
                     for a in arts if a.get("published_at")], dtype=float)
    ages = np.maximum(ages, 0.0)
    ok = ages <= 168.0
    m18.append(float(np.exp(-np.log(2) * ages[ok] / 18.0).sum()))
    for h in HLS:
        mhl[h].append(float(np.exp(-np.log(2) * ages[ok] / h).sum()))
    nsrc.append(len({a.get("source") for a in arts}))
lv["mass"], lv["n_src"] = m18, nsrc
recon = lv["raw"].to_numpy() * e_curve(lv["mass"].to_numpy()) * d_curve(lv["n_src"].to_numpy())
lv["fid"] = np.abs(recon - lv["news"].to_numpy()) <= 6e-4
print(f"LIVE fidelity (recomputed from the digest's articles = the stored news): {lv['fid'].mean():.2%}")
lv = lv[lv["fid"]].copy()
lab = [label(r.ticker, r.generated_at, r.price) for r in lv.itertuples()]
lv["y"], lv["settled"] = [a for a, _ in lab], [b for _, b in lab]
lv["day"] = pd.to_datetime(lv["generated_at"], utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
# the half-life masses, aligned to the kept rows
keep = lv.index.to_numpy()
full_m = {h: np.asarray(mhl[h]) for h in HLS}
V2 = variants(lv["raw"].to_numpy(), lv["mass"].to_numpy(), lv["n_src"].to_numpy(),
              {h: full_m[h][keep] for h in HLS})
for k, v in V2.items():
    lv[k] = v
print(f"LIVE labelled ({time.time() - t0:.0f}s): {lv['y'].notna().mean():.1%}")
results["live"] = report(lv, list(V2), "LIVE — live panel 09-15 -> now, local Qwen, exact digests")
s2 = lv[lv["day"] >= "2026-09-28"]
if len(s2):
    results["set2"] = report(s2, list(V2), "SET 2 — live all-source (09-28 ->)")

# ── robustness: the combine's Gate-4 pool ($5 / 20-day mean $-volume $5M, known the day
#    before) and, for LIVE, one run a day (the day's first regular-hours run) ─────────
from src.data.cache import load_ohlcv

_g4 = {}


def gate4(tk, day):
    if tk not in _g4:
        try:
            d = load_ohlcv(tk)
            if d is None or not len(d):
                _g4[tk] = None
            else:
                d = d.copy()
                d.index = pd.to_datetime(d.index).tz_localize(None) if getattr(pd.to_datetime(d.index), "tz", None) is None else pd.to_datetime(d.index).tz_convert(None)
                dv = (d["Close"] * d["Volume"]).rolling(20, min_periods=20).mean().shift(1)
                _g4[tk] = pd.DataFrame({"px": d["Close"].shift(1), "dv": dv})
        except Exception:
            _g4[tk] = None
    f = _g4[tk]
    if f is None:
        return False
    i = f.index.searchsorted(pd.Timestamp(day), side="right") - 1
    if i < 0:
        return False
    r = f.iloc[i]
    return bool(r["px"] >= 5.0 and r["dv"] >= 5e6)


for key, frame, names in (("set1", s1, list(V1)), ("live", lv, list(V2))):
    frame["g4"] = [gate4(t, d) for t, d in zip(frame["ticker"], frame["day"])]
    results[key + "_gate4"] = report(frame[frame["g4"]], names, f"{key.upper()} - Gate-4 pool only ({frame['g4'].mean():.0%} of rows)")
et = pd.to_datetime(lv["generated_at"], utc=True).dt.tz_convert("America/New_York")
rth = lv[(et.dt.hour * 60 + et.dt.minute >= 9 * 60 + 30) & (et.dt.hour < 16)].copy()
first = rth.sort_values("generated_at").groupby("day")["run"].first()
one = rth[rth["run"].isin(set(first))]
results["live_first_rth_run"] = report(one, list(V2), "LIVE - the first regular-hours run of each day only")

with open(OUT, "w") as fh:
    json.dump(results, fh, indent=1, default=str)
print(f"\nwritten {OUT} ({time.time() - t0:.0f}s)")
