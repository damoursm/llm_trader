"""THE EARNINGS-DIP DEMOTION — the dip book's daily top pick passes over a dip that is its company's own earnings
reaction (PREREG137's DEMOTE_E; user 2026-10-17: "Fully deploy the earnings dip demotion in live"; from the 2026-10-19
session; `memory/dip-earnings-dips-2026-10.md`, report https://claude.ai/artifact/9gVkGD4XAdMnhRhAgsQNPn).

THE RULE. A signal of session D (the close of D-1) is an EARNINGS DIP when its company filed an 8-K or 8-K/A naming
Item 2.02 (Results of Operations and Financial Condition: the earnings release) ACCEPTED after 16:00 ET of the session
`dip_long_earnings_demote_sessions` (3) before the signal session and at or before 09:30 ET of D — the release behind
the drop, known before the order is placed (`window`). In the daily top pick (`dip_long._daily_top`, `dip_long.rank_key`)
an earnings dip ranks after every other signal of its day, the earnings dips among themselves by the same key (the
meta-model's score, then the RSI2): the day's best other dip is bought instead; a day whose every signal is an earnings
dip still buys its best one. Nothing else changes (the slice, the limit, the exit). The simulator's construction is
`mimic.earnings_demoted` on this module's `window_ns` / `releases` (`Mimic.dip_earnings`, production's by default).

THE DATA — EDGAR's acceptance instants as the deep store holds them (`sec_filings`, parsed by `sec._filings_frame`: the
study's construction, its quirk included — the submissions JSON's acceptanceDateTime is Eastern time labelled "Z" for
some filings accepted before ~09:00 ET, stored as UTC; the window's edges, 16:00 and 09:30 ET, lie outside those
hours). First the name's part of the store (`cache/ml/deep/sec_filings/parts/<T>.parquet`: the nightly full pass, then
the 08:30 pre-open run's pass over the companies on EDGAR's live feed, ~08:33); then, for a signal the store does not
flag, the company's submissions JSON asked NOW (`dip_long_earnings_edgar`; its recent block, real time; one request a
name, `EDGAR_BUDGET_SECONDS` for the day's signals, best-ranked first) — a release accepted after the pre-open pass and
before 09:30 (4 of the study's 551). The JSON's instants DRIFT once a filing is days old (+4 h, +8 h: `retimed`), so a
release EDGAR answers within 12 hours of the window takes its filing index page's "Accepted" instant (one request; the
JSON's instant only when the page cannot be read, noted ``acceptance_source``). The store's first-seen instants are
right for today's companies (2026-10-17: 185 of 186 sampled releases 2007-2026 equal their index pages, the 60 behind
PREREG137's flagged dips included); the delisted company store, the simulator's alone, holds most 4-5 h early (Eastern
time read as UTC: 28 of 34) — none moves a flag of the simulator's orders (386 of 386 on the index pages' instants; 2
of the 258 releases near a window were mis-timed). A company that files no 8-K (a foreign issuer's ADR files 6-Ks) is never an earnings dip, as in
the study.

FAIL-OPEN. A signal whose part of the store cannot be read is NOT demoted (``unknown``: the meta-model's ranking, as
without the rule) — a scorer-banner problem when it is the day's pick; EDGAR unreachable or the day's budget spent
(``store_only``: the store's filings through its last pass) is a note when it is the pick; an error of the whole check
demotes nothing (the ``top`` block's ``earnings.error``, a problem). Every EDGAR request is ONE request through the
store's limiter, at most `EDGAR_TIMEOUT_SECONDS`, never past the day's budget (`_get`): the opening tick waits for no
retry.

EVIDENCE (PREREG137 Amendment 1, EXPLORATORY: chosen after the top-pick table of the same years): the live account on
the production-mimicking simulator, 20 paired draws a set against the live configuration (L136), THE POOLED ADD RULE:
ADD, +3.25 pp a year (52.8% -> 56.1%), every set gaining (A0 +1.8, A +1.5, B +3.4 log points a year), 8 of 9 calendar
years, margin-call pairs 7 -> 7, the worst drawdown 26.4 -> 26.4%, the dip book's trades as many; the window on a plateau
(1 / 3 / 5 sessions: +2.3 / +3.3 / +2.9 pp — five refused by one more margin-call pair); set B's lendability shifted
+-0.37: +3.27 / +3.41 log points; NOT confirmed out of universe (PREREG140, the dip rule at $200M-$1B: -0.21 %/trade,
n.s.); the overfitting report's shrunk gain +0.31 log points a year. The effect is the earnings REACTION's (PREREG138 /
PREREG141: neither the guidance the release states nor the consensus surprise explains it).
"""
from __future__ import annotations

import re
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config import settings
from src.utils import ET

RULE = "PREREG137 DEMOTE_E: an earnings dip ranks after every other signal of its day"
ITEM = re.compile(r"\b2\.02\b")                 # Item 2.02 among the 8-K's items ("2.02,9.01"; never "12.02")
FORMS = ("8-K", "8-K/A")
COLUMNS = ("ticker", "cik", "accession", "acceptance", "form", "items")
EDGAR_TIMEOUT_SECONDS = 10.0                    # one request (no retry): a submissions JSON or an index page
EDGAR_BUDGET_SECONDS = 45.0                     # every EDGAR request of one day's signals together
# the simulator's table (`release_table`): today's companies (the deep store) and those delisted since 2021
DELISTED_FILINGS = Path("cache/ml/deep_delisted/sec_filings.parquet")


def enabled() -> bool:
    return bool(getattr(settings, "enable_dip_long_earnings_demote", False))


def edgar_enabled() -> bool:
    """The opening tick's EDGAR check of the signals the store does not flag (`dip_long_earnings_edgar`)."""
    return bool(getattr(settings, "dip_long_earnings_edgar", False))


def sessions() -> int:
    """The window's sessions before the signal session (`dip_long_earnings_demote_sessions`)."""
    n = int(getattr(settings, "dip_long_earnings_demote_sessions", 3))
    if n < 1:
        raise ValueError(f"dip_long_earnings_demote_sessions must be at least 1, not {n}")
    return n


# ── the window ───────────────────────────────────────────────────────────────

def window(d: date, n: Optional[int] = None) -> Tuple[datetime, datetime]:
    """(lo, hi) — aware UTC instants: 16:00 ET of the session ``n`` (default `sessions`) before the SIGNAL session
    (the session before ``d``, whose close fired the signal) and 09:30 ET of ``d``, the order's first session. A
    release counts when lo < its acceptance <= hi (PREREG137's E: three sessions' reactions and the night before D)."""
    from src.signals.sel_short import sessions_before
    n = sessions() if n is None else int(n)
    if n < 1:
        raise ValueError(f"the window needs at least one session, not {n}")
    signal = sessions_before(d, 1)[0]
    first = sessions_before(signal, n)[0]
    lo = datetime(first.year, first.month, first.day, 16, 0, tzinfo=ET).astimezone(timezone.utc)
    hi = datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET).astimezone(timezone.utc)
    return lo, hi


def window_ns(d: date, n: Optional[int] = None) -> Tuple[int, int]:
    """`window` as int64 nanoseconds UTC."""
    lo, hi = window(d, n)
    return int(pd.Timestamp(lo).value), int(pd.Timestamp(hi).value)


# ── the releases ─────────────────────────────────────────────────────────────

def releases(frame: Optional[pd.DataFrame]) -> pd.DataFrame:
    """The earnings releases among a company's filings — an 8-K / 8-K/A whose items name 2.02 — with their acceptance
    as int64 ns UTC (``ns``: the store's naive instants are UTC; an aware column is converted), oldest first; a filing
    without a readable acceptance is left out. Empty when there is none."""
    cols = [c for c in COLUMNS] + ["ns"]
    if frame is None or len(frame) == 0 or "form" not in frame or "items" not in frame or "acceptance" not in frame:
        return pd.DataFrame(columns=cols)
    form = frame["form"].fillna("").astype(str)
    items = frame["items"].fillna("").astype(str)
    f = frame[form.isin(FORMS) & items.str.contains(ITEM)]
    if f.empty:
        return pd.DataFrame(columns=cols)
    acc = pd.to_datetime(f["acceptance"], errors="coerce")
    if getattr(acc.dt, "tz", None) is not None:
        acc = acc.dt.tz_convert("UTC").dt.tz_localize(None)
    ok = acc.notna().to_numpy()
    out = f.loc[ok, [c for c in COLUMNS if c in f.columns]].copy()
    out["ns"] = pd.DatetimeIndex(acc[ok]).as_unit("ns").asi8
    return out.sort_values("ns", kind="mergesort").reset_index(drop=True)


def released(ns: np.ndarray, lo_ns: int, hi_ns: int) -> bool:
    """Whether any acceptance of ``ns`` falls in (lo, hi]."""
    a = np.asarray(ns, np.int64)
    return bool(((a > int(lo_ns)) & (a <= int(hi_ns))).any())


def in_window(rel: pd.DataFrame, lo_ns: int, hi_ns: int) -> Optional[dict]:
    """The LATEST release (`releases`' frame) accepted in (lo, hi] as a journal record; None when there is none."""
    if rel is None or rel.empty:
        return None
    m = (rel["ns"].to_numpy(np.int64) > int(lo_ns)) & (rel["ns"].to_numpy(np.int64) <= int(hi_ns))
    if not m.any():
        return None
    r = rel[m].iloc[-1]
    out = {"accession": str(r.get("accession")), "cik": None if r.get("cik") is None else str(r.get("cik")),
           "form": str(r.get("form")), "items": str(r.get("items")),
           "acceptance": pd.Timestamp(int(r["ns"]), tz="UTC").isoformat(), "n_in_window": int(m.sum())}
    if r.get("acceptance_source") is not None:          # EDGAR's answer: re-timed from the index page, or not
        out["acceptance_source"] = str(r.get("acceptance_source"))
    return out


# ── the two sources ──────────────────────────────────────────────────────────

def parts_dir() -> Path:
    """The deep store's `sec_filings` parts (one per ticker, the project's symbol form: BRK-B)."""
    from src.data import deep
    return Path(deep.DEEP_DIR) / "sec_filings" / "parts"


def store_filings(ticker: str) -> Optional[pd.DataFrame]:
    """The name's filings in the deep store's part; None when the store holds no part for it (raises when the part
    cannot be read)."""
    p = parts_dir() / f"{str(ticker).upper()}.parquet"
    if not p.exists():
        return None
    from src.data.deep import read_parquet
    return read_parquet(p, columns=list(COLUMNS))


def store_through() -> Optional[str]:
    """When the store's last `sec_filings` pass ended (the nightly full pass or the pre-open's incremental one), ISO;
    None when the refresh state holds neither."""
    try:
        from src.data.deep.refresh import load_state
        st = load_state()
    except Exception:                                           # noqa: BLE001
        return None
    stamps = [str((st.get(k) or {}).get("at") or "") for k in ("sec_filings", "sec_filings_incremental")]
    stamps = [s for s in stamps if s]
    return max(stamps) if stamps else None


def company_cik(store: Optional[pd.DataFrame]) -> Optional[str]:
    """The company's 10-digit CIK from its part (every row carries it); None when the part holds none."""
    if store is not None and len(store) and "cik" in store:
        c = store["cik"].dropna().astype(str)
        if len(c) and c.iloc[0].strip():
            return c.iloc[0].strip().zfill(10)
    return None


class BudgetSpent(RuntimeError):
    """The day's EDGAR time is spent: no further request."""


SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"


def _get(url: str, budget: Optional["Budget"] = None):
    """ONE request to EDGAR through the store's shared limiter, no retry, at most `EDGAR_TIMEOUT_SECONDS` and never past
    the day's budget (`deep.http_get` retries and honours a Retry-After of up to 90 s: not on the opening tick)."""
    import requests

    from src.data.deep import SEC_HEADERS, sec
    t = EDGAR_TIMEOUT_SECONDS if budget is None else min(EDGAR_TIMEOUT_SECONDS, budget.left())
    if t < 1.0:
        raise BudgetSpent("the day's EDGAR budget is spent")
    sec._LIMITER.wait()
    return requests.get(url, headers=SEC_HEADERS, timeout=t)


def edgar_filings(ticker: str, cik: str, budget: Optional["Budget"] = None) -> pd.DataFrame:
    """The company's recent filings from EDGAR NOW — the submissions JSON's recent block (real time, up to 1,001
    filings) through the store's own parser (`sec._filings_frame`). Raises on a failed request."""
    from src.data.deep import sec
    r = _get(SUBMISSIONS_URL.format(cik10=cik), budget)
    if r.status_code != 200:
        raise RuntimeError(f"submissions HTTP {r.status_code}")
    df = sec._filings_frame(((r.json() or {}).get("filings") or {}).get("recent") or {}, cik)
    if len(df):
        df.insert(0, "ticker", str(ticker).upper())
    return df


# The submissions JSON's acceptanceDateTime of a filing DRIFTS after its first days (measured 2026-10-17: 130 of 130
# sampled October releases re-read 4 h later than the store first read them, two September ones 8 h; the store's
# first-seen instants equal the filing index pages' "Accepted" on 185 of 186 of today's companies' releases sampled,
# the 60 behind PREREG137's flagged dips included), so a release EDGAR answers near the window is re-timed from its
# index page — the authoritative Eastern-time acceptance — before the window is applied.
INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/{accession}-index.htm"
_ACCEPTED = re.compile(r"Accepted</div>\s*<div class=\"info\">([^<]+)</div>")
DRIFT_NS = 12 * 3600 * 10 ** 9                  # an EDGAR release this close to the window is re-timed


def accepted_at(cik: str, accession: str, budget: Optional["Budget"] = None) -> Optional[int]:
    """The filing's acceptance as its EDGAR index page states it ("Accepted", Eastern time), int64 ns UTC; None when the
    page cannot be read (or the budget is spent)."""
    url = INDEX_URL.format(cik=int(str(cik)), folder=str(accession).replace("-", ""), accession=accession)
    try:
        r = _get(url, budget)
    except Exception:                                           # noqa: BLE001 — the JSON's instant stands
        return None
    m = _ACCEPTED.search(r.text) if r is not None and r.status_code == 200 else None
    if not m:
        return None
    try:
        return int(pd.Timestamp(m.group(1).strip()).tz_localize("America/New_York").value)
    except (ValueError, TypeError):
        return None


def retimed(rel: pd.DataFrame, lo_ns: int, hi_ns: int, budget: Optional["Budget"] = None) -> pd.DataFrame:
    """``rel`` (EDGAR's releases, `releases`) with every release within `DRIFT_NS` of the window (lo, hi] re-timed from
    its index page (`accepted_at`; ``acceptance_source`` "index"); one whose page cannot be read keeps the JSON's
    instant ("json")."""
    if rel is None or rel.empty:
        return rel
    out = rel.copy()
    ns = out["ns"].to_numpy(np.int64)
    near = (ns > int(lo_ns) - DRIFT_NS) & (ns <= int(hi_ns) + DRIFT_NS)
    src = np.array([None] * len(out), dtype=object)
    for i in np.flatnonzero(near):
        t = accepted_at(str(out["cik"].iloc[i]), str(out["accession"].iloc[i]), budget)
        if t is not None:
            ns[i] = t
            src[i] = "index"
        else:
            src[i] = "json"
    out["ns"] = ns
    out["acceptance_source"] = src
    return out.sort_values("ns", kind="mergesort").reset_index(drop=True)


# ── the check ────────────────────────────────────────────────────────────────

class Budget:
    """The day's EDGAR time: requests stop once ``seconds`` (default `EDGAR_BUDGET_SECONDS`) are spent."""

    def __init__(self, seconds: Optional[float] = None):
        self.deadline = time.monotonic() + float(EDGAR_BUDGET_SECONDS if seconds is None else seconds)

    def left(self) -> float:
        return self.deadline - time.monotonic()


STATUSES = ("flagged", "clear", "store_only", "no_filings", "unknown")


def check(ticker: str, d: date, n: Optional[int] = None, edgar: Optional[bool] = None,
          budget: Optional[Budget] = None) -> dict:
    """Whether the signal of ``ticker`` for session ``d`` is an earnings dip: ``earnings_dip`` True / False / None,
    the latest qualifying ``release``, the ``source`` that decided (store / edgar), whether EDGAR was asked
    (``edgar_asked``), an ``error`` and the ``status``:

    * ``flagged`` — a release in the window (the store's, else EDGAR's current filings);
    * ``clear`` — EDGAR's current filings hold none (complete through now);
    * ``store_only`` — the store holds none and EDGAR was not asked or did not answer (the store is complete through its
      last pass);
    * ``no_filings`` — the store holds no part for the name (no SEC filer in the universe's filings — an ADR whose issuer
      files 6-Ks has a part without 8-Ks; a listing newer than the store's last universe pass has none): never an
      earnings dip, as the study's unmapped names;
    * ``unknown`` — the name's part could not be read: ``earnings_dip`` None, NOT demoted (fail-open).

    Each EDGAR request is single and bounded (`_get`): ``budget`` (the day's) caps them all."""
    lo, hi = window_ns(d, n)
    out: dict = {"earnings_dip": None, "release": None, "source": None, "status": "unknown", "edgar_asked": False}
    try:
        store = store_filings(ticker)
    except Exception as e:                                      # noqa: BLE001
        return dict(out, error=f"store: {type(e).__name__}: {e}")
    if store is None:
        return dict(out, earnings_dip=False, status="no_filings")
    hit = in_window(releases(store), lo, hi)
    if hit is not None:
        return dict(out, earnings_dip=True, release=hit, source="store", status="flagged")
    out.update(earnings_dip=False, source="store", status="store_only")
    if not (edgar_enabled() if edgar is None else bool(edgar)):
        return out
    cik = company_cik(store)
    if cik is None:
        return dict(out, error="edgar: the part holds no CIK")
    if budget is not None and budget.left() < 1.0:
        return dict(out, error="edgar: the day's EDGAR budget is spent")
    out["edgar_asked"] = True
    try:
        hit = in_window(retimed(releases(edgar_filings(ticker, cik, budget)), lo, hi, budget), lo, hi)
    except Exception as e:                                      # noqa: BLE001 — the store's answer stands
        return dict(out, error=f"edgar: {type(e).__name__}: {e}")
    return dict(out, earnings_dip=hit is not None, release=hit, source="edgar",
                status="flagged" if hit is not None else "clear")


def flag_signals(sig: Sequence[dict], d: date) -> dict:
    """Stamp each of the day's signals (best-ranked first: the EDGAR budget goes to the likeliest picks) with
    ``earnings_dip`` (True / False / None), ``earnings_status`` and, when flagged, ``earnings_release``; returns the
    journal block of the daily top pick (``top.earnings``)."""
    n = sessions()
    lo, hi = window(d, n)
    budget = Budget()
    t0 = time.monotonic()
    statuses: Dict[str, str] = {}
    asked, failed = 0, []
    for s in sig:
        tk = str(s.get("ticker") or "")
        r = check(tk, d, n, budget=budget)
        s["earnings_dip"] = r["earnings_dip"]
        s["earnings_status"] = r["status"]
        if r.get("release"):
            s["earnings_release"] = dict(r["release"], source=r["source"])
        else:
            s.pop("earnings_release", None)
        asked += int(bool(r.get("edgar_asked")))
        if r.get("error"):
            s["earnings_error"] = r["error"]
            if "edgar:" in r["error"]:
                failed.append(tk)
        else:
            s.pop("earnings_error", None)
        statuses[tk] = r["status"]
    flagged = [str(s.get("ticker")) for s in sig if s.get("earnings_dip") is True]
    return {"rule": RULE, "sessions": n, "window": [lo.isoformat(), hi.isoformat()], "checked": len(sig),
            "flagged": flagged, "statuses": statuses,
            "unknown": [tk for tk, st in statuses.items() if st == "unknown"],
            "edgar": {"on": edgar_enabled(), "asked": asked, "failed": failed},
            "store_through": store_through(), "seconds": round(time.monotonic() - t0, 2), "error": None}


# ── the simulator's table (`mimic.earnings_demoted`) ─────────────────────────

_TABLE: Dict[Tuple[str, ...], Dict[str, np.ndarray]] = {}


def release_table(sources: Optional[Sequence[Path]] = None) -> Dict[str, np.ndarray]:
    """{company key: its earnings releases' acceptances, sorted int64 ns UTC} over whole consolidated filings tables
    (default: the deep store's `sec_filings.parquet` — today's companies by ticker — and the delisted company store's,
    ``XYZ_DELISTED``), through `releases`: the backtest's view of what this module reads per name. Loaded once per
    process and source list."""
    from src.data import deep
    srcs = tuple(str(Path(p)) for p in (sources if sources is not None
                                        else (Path(deep.DEEP_DIR) / "sec_filings.parquet", DELISTED_FILINGS)))
    if srcs not in _TABLE:
        import duckdb
        frames = []
        for p in srcs:
            if not Path(p).exists():
                logger.warning(f"[dip_earnings] no filings table {p} — its companies hold no release")
                continue
            con = duckdb.connect()
            try:
                frames.append(con.execute(
                    f"SELECT ticker, cik, accession, acceptance, form, items FROM '{Path(p).as_posix()}' "
                    "WHERE form IN ('8-K', '8-K/A') AND items LIKE '%2.02%'").df())
            finally:
                con.close()
        rel = releases(pd.concat(frames, ignore_index=True)) if frames else releases(None)
        _TABLE[srcs] = {str(k): np.sort(g["ns"].to_numpy(np.int64)) for k, g in rel.groupby("ticker")}
    return _TABLE[srcs]


def demoted(keys: Mapping[str, Optional[str]], n: int, table: Mapping[str, np.ndarray]) -> List[str]:
    """The order ids among ``keys`` ({order id "NAME|D": the company's key in ``table``, None when unknown}) whose
    company released earnings in the order's window (`window_ns` of D, ``n`` sessions) — the earnings dips."""
    out = []
    for oid, key in keys.items():
        a = table.get(key) if key is not None else None
        if a is None or not len(a):
            continue
        lo, hi = window_ns(date.fromisoformat(str(oid).split("|")[-1]), n)
        if released(a, lo, hi):
            out.append(str(oid))
    return out
