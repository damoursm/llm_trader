"""A stock's SECTOR for a per-sector cap (PREREG100): the Fama-French 12-industry group of its 4-digit SIC code
(Kenneth French's ``Siccodes12`` definitions, `ff12`) — a standard, coarse, pre-specified grouping, never searched.

THE SIC, in this order (`label_records`):

* Polygon's ticker details as the deep store holds them — today's names (``cache/ml/deep/ticker_details.parquet``)
  and the names delisted since 2021 (``cache/ml/deep_delisted/ticker_details.parquet``, ``XYZ_DELISTED`` there, the
  backtest's ``XYZ@D``);
* else SEC EDGAR's submissions record for the company's CIK (field ``sic``) — foreign filers / ADRs (ARM, TSM, ASML)
  file 20-F with an SIC that Polygon leaves blank. The CIK is Polygon's (or the delisted store's) when it has one,
  else a ticker -> CIK map (``cik_of``: the deep store's SEC filings, SEC's current ticker map — never asked for a
  delisted ``@D`` name: its symbol may belong to another company today). The lookups are INJECTED (``sec_sic``:
  CIK -> SIC, ``cik_of``: ticker -> CIK) so the labels can be built offline;
* else UNKNOWN (None) — never capped.

A fund (Polygon type ETF / ETN / ETV / ETS / FUND) has no sector (None). A missing or zero SIC is UNKNOWN, not
"Other". Today's SIC is used for every year (a company's industry code rarely changes — an assumption).

Backtest names: ``XYZ`` (the deep store), ``XYZ@D`` (delisted since 2021), ``XYZ@X`` (a stock outside the deep store:
looked up under its symbol today). ``stamp_groups`` writes each piece's ``group`` for the account engine's group cap
(`account.Strategy.group_cap`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

FF12 = ("NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq", "Telcm", "Utils", "Shops", "Hlth", "Money", "Other")
# Kenneth French's Siccodes12, inclusive ranges; any other SIC is "Other" (12)
FF12_RANGES: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "NoDur": ((100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)),
    "Durbl": ((2500, 2519), (2590, 2599), (3630, 3659), (3710, 3711), (3714, 3714), (3716, 3716), (3750, 3751),
              (3792, 3792), (3900, 3939), (3990, 3999)),
    "Manuf": ((2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569), (3580, 3629), (3700, 3709),
              (3712, 3713), (3715, 3715), (3717, 3749), (3752, 3791), (3793, 3799), (3830, 3839), (3860, 3899)),
    "Enrgy": ((1200, 1399), (2900, 2999)),
    "Chems": ((2800, 2829), (2840, 2899)),
    "BusEq": ((3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)),
    "Telcm": ((4800, 4899),),
    "Utils": ((4900, 4949),),
    "Shops": ((5000, 5999), (7200, 7299), (7600, 7699)),
    "Hlth": ((2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)),
    "Money": ((6000, 6999),),
}
FUND_TYPES = frozenset({"ETF", "ETN", "ETV", "ETS", "FUND"})
DEEP_DETAILS = Path("cache/ml/deep/ticker_details.parquet")
DELISTED_DETAILS = Path("cache/ml/deep_delisted/ticker_details.parquet")
DELISTED_SUFFIX = "_DELISTED"


def sic_int(sic) -> Optional[int]:
    """The SIC as an integer, None when missing, unparsable or zero ("0000" is no code)."""
    if sic is None:
        return None
    if isinstance(sic, float):
        if sic != sic:
            return None
        sic = int(sic)
    s = str(sic).strip()
    if not s:
        return None
    try:
        v = int(float(s))
    except ValueError:
        return None
    return v if v > 0 else None


def ff12(sic) -> Optional[str]:
    """The Fama-French 12-industry group of a SIC code (int or string); None for a missing / zero SIC."""
    v = sic_int(sic)
    if v is None:
        return None
    for name, ranges in FF12_RANGES.items():
        for lo, hi in ranges:
            if lo <= v <= hi:
                return name
    return "Other"


def split_name(name: str) -> Tuple[str, str]:
    """(symbol, suffix) of a backtest name: ``XYZ@D`` -> (``XYZ``, ``D``), ``XYZ`` -> (``XYZ``, "")."""
    s = str(name)
    if "@" in s:
        a, b = s.split("@", 1)
        return a, b
    return s, ""


def cik10(cik) -> Optional[str]:
    """A CIK as SEC's 10-digit string, None when missing."""
    v = sic_int(cik)
    return None if v is None else str(v).zfill(10)


def polygon_details(deep: Path = DEEP_DETAILS, delisted: Path = DELISTED_DETAILS) -> Dict[str, dict]:
    """{backtest name: {"sic", "cik", "type", "store"}} from Polygon's stored ticker details: a deep-store row under its
    ticker, a delisted row (``XYZ_DELISTED``) under ``XYZ@D``. Read with duckdb (no pyarrow)."""
    import duckdb
    out: Dict[str, dict] = {}
    for path, store in ((Path(deep), "deep"), (Path(delisted), "delisted")):
        if not path.exists():
            continue
        df = duckdb.query(f"select ticker, cik, sic_code, type from '{path.as_posix()}'").df()
        for tk, cik, sic, typ in zip(df["ticker"], df["cik"], df["sic_code"], df["type"]):
            if tk is None:
                continue
            tk = str(tk)
            if store == "delisted":
                if not tk.endswith(DELISTED_SUFFIX):
                    continue
                tk = tk[: -len(DELISTED_SUFFIX)] + "@D"
            out[tk] = {"sic": None if sic is None or (isinstance(sic, float) and sic != sic) else str(sic),
                       "cik": cik10(cik), "type": None if typ is None or (isinstance(typ, float) and typ != typ)
                       else str(typ), "store": store}
    return out


def label_records(names: Iterable[str], details: Optional[Mapping[str, dict]] = None,
                  sec_sic: Optional[Callable[[str], Optional[str]]] = None,
                  cik_of: Optional[Callable[[str], Optional[str]]] = None) -> Dict[str, dict]:
    """Per backtest name: {"sector", "sic", "source", "cik", "type"}. ``source``: "polygon" (the deep store's
    details), "polygon_delisted", "sec" (EDGAR's record for the CIK), "fund" (a fund type: no sector) or None
    (unknown). ``details``: `polygon_details()` (read when None); ``sec_sic(cik10)``: the SIC SEC records for a CIK
    (None: unknown; the argument None = no SEC lookups); ``cik_of(symbol)``: a current symbol's CIK (SEC's map)."""
    det = polygon_details() if details is None else details
    out: Dict[str, dict] = {}
    for name in names:
        sym, suf = split_name(name)
        rec = det.get(f"{sym}@D") if suf == "D" else det.get(sym)
        typ = rec.get("type") if rec else None
        cik = rec.get("cik") if rec else None
        if typ in FUND_TYPES:
            out[name] = {"sector": None, "sic": None, "source": "fund", "cik": cik, "type": typ}
            continue
        sic, src = None, None
        if rec and sic_int(rec.get("sic")) is not None:
            sic, src = str(rec["sic"]), ("polygon_delisted" if suf == "D" else "polygon")
        else:
            if cik is None and suf != "D" and cik_of is not None:
                cik = cik10(cik_of(sym))
            if cik is not None and sec_sic is not None:
                s = sec_sic(cik)
                if sic_int(s) is not None:
                    sic, src = str(s), "sec"
        out[name] = {"sector": ff12(sic), "sic": sic, "source": src, "cik": cik, "type": typ}
    return out


def load_labels(names: Iterable[str], details: Optional[Mapping[str, dict]] = None,
                sec_sic: Optional[Callable[[str], Optional[str]]] = None,
                cik_of: Optional[Callable[[str], Optional[str]]] = None) -> Dict[str, Optional[str]]:
    """{backtest name: its FF12 sector, or None (unknown, a fund)} — `label_records`' sectors."""
    return {k: v["sector"] for k, v in label_records(names, details, sec_sic, cik_of).items()}


def coverage(labels: Mapping[str, Optional[str]], names: Optional[Sequence[str]] = None,
             weights: Optional[Mapping[str, float]] = None) -> dict:
    """The share of ``names`` (all labelled names when None) with a sector, the count per sector and — with
    ``weights`` (say trades per name) — the weighted share. A name missing from ``labels`` counts as unknown."""
    ns: List[str] = list(labels) if names is None else list(dict.fromkeys(names))
    secs = [labels.get(n) for n in ns]
    known = [s for s in secs if s is not None]
    by: Dict[str, int] = {}
    for s in known:
        by[s] = by.get(s, 0) + 1
    out = {"n": len(ns), "with_sector": len(known), "share": (len(known) / len(ns)) if ns else None,
           "by_sector": dict(sorted(by.items(), key=lambda kv: -kv[1])),
           "unknown": [n for n, s in zip(ns, secs) if s is None]}
    if weights is not None:
        w = [float(weights.get(n, 0.0)) for n in ns]
        tot = sum(w)
        out["weighted_share"] = (sum(x for x, s in zip(w, secs) if s is not None) / tot) if tot > 0 else None
    return out


def stamp_groups(pieces: Iterable[dict], labels: Mapping[str, Optional[str]], key: str = "group") -> dict:
    """Write each piece's sector (``labels[piece["tkn"]]``, None when unknown or unlabelled) under ``key`` — the
    account engine's group (`account.Strategy.group_cap`). Returns the counts: pieces stamped with a sector, with
    None, and the names missing from ``labels``."""
    n_sec = n_none = 0
    missing = set()
    for p in pieces:
        tk = str(p["tkn"])
        if tk not in labels:
            missing.add(tk)
        s = labels.get(tk)
        p[key] = s
        if s is None:
            n_none += 1
        else:
            n_sec += 1
    return {"with_sector": n_sec, "none": n_none, "missing_names": sorted(missing)}
