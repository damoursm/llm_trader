"""Detect that an implementation changed — the trigger for automatic refactor.

Every stored value in this database was produced by some version of the code.
When that code changes, the stored values stop describing the current system,
and until now the only defence was a HAND-EDITED registry
(`METHOD_SCORER_EPOCH`) with a docstring asking whoever changed a scorer to
remember to add a timestamp. That is precisely the class of safeguard this
project keeps finding broken: correct only while someone remembers.

This module makes the detection mechanical. Each method is mapped to the
module(s) that compute it, and those modules are fingerprinted by their
**normalised AST**, not their text:

    * comments, blank lines, formatting and docstrings do NOT change the
      fingerprint — a reformat must not trigger a database refactor;
    * renaming a local, reordering a branch, or changing a constant DOES —
      those can change output.

A fingerprint change is a *suspicion* of changed output, not proof. What
follows from it depends on whether the method can be regenerated:

    REPLAYABLE      -> re-run the replay. A false positive costs CPU, nothing
                       else, so the detector can be liberal.
    NOT REPLAYABLE  -> the only remedy is masking (an epoch), which withholds
                       real history. A false positive is expensive there, so
                       those are recorded and surfaced rather than auto-masked
                       (see `refactor.py`).

**Why a fingerprint here is not the "version hash" that was rejected.** The
objection was to using a hash to SEGREGATE data into incomparable eras instead
of refactoring it. This hash does the opposite: it exists to TRIGGER the
refactor. It is a change detector, never a partition key, and no analysis reads
it.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
from typing import Dict, List, Optional, Tuple

from loguru import logger

# method -> the module(s) whose code determines its score. Several methods share
# a module (the classic anomalies, the fundamental factors), and a change in a
# shared module correctly marks all of its methods — a helper edit there really
# can move any of them.
METHOD_SOURCES: Dict[str, Tuple[str, ...]] = {
    # OHLCV technicals (replayable)
    "tech": ("src.analysis.technical",),
    "vwap": ("src.signals.vwap",),
    "momentum": ("src.signals.price_momentum",),
    "money_flow": ("src.signals.money_flow",),
    "trend_strength": ("src.signals.trend_strength",),
    "iv_rank": ("src.signals.iv_rank",),
    "pattern": ("src.signals.pattern_recognition",),
    "sector_momentum": ("src.signals.sector_relative_momentum",),
    # market_momentum is computed inline in the aggregator (no module of its own).
    "market_momentum": ("src.signals.aggregator",),
    # Sentiment / news
    "news": ("src.analysis.sentiment",),
    "sent_velocity": ("src.signals.sentiment_velocity",),
    # news_shock reads the sentiment module's attention_mass too — both modules
    # move its output, so both are fingerprinted.
    "news_shock": ("src.signals.news_shock", "src.analysis.sentiment"),
    # news_bear_fresh multiplies the news verdict, so the sentiment module moves
    # its output exactly as it moves news_shock's.
    "news_bear_fresh": ("src.signals.news_bear_fresh", "src.analysis.sentiment"),
    # catalyst_tilt multiplies the news verdict by a map calibrated from the
    # news-event dataset — all three modules move its output.
    "catalyst_tilt": ("src.signals.catalyst_tilt", "src.analysis.news_events",
                      "src.analysis.sentiment"),
    # Options family. `max_pain` and `oi_skew` have no module of their own —
    # they are scored INLINE in the aggregator from fetched chain data, so the
    # aggregator IS their source. Coarse (any aggregator edit marks them) but
    # correct: a change there really can move them, and for a non-replayable
    # method over-marking is caught by the review gate rather than acted on.
    "put_call": ("src.data.put_call", "src.signals.aggregator"),
    "max_pain": ("src.data.opex", "src.signals.aggregator"),
    "oi_skew": ("src.data.gamma_exposure", "src.signals.aggregator"),
    "iv_expr": ("src.signals.iv_expr",),
    "iv_term": ("src.signals.iv_term_structure",),
    # Event / smart money
    "insider": ("src.data.insider_trades", "src.signals.aggregator"),
    "pead": ("src.data.pead",),
    "ext_gap": ("src.signals.extended_session",),
    "coint": ("src.signals.cointegration",),
    "cross_sectional": ("src.signals.cross_sectional",),
    "massive": ("src.signals.massive_tech",),
    "broker_advisor": ("src.signals.broker_advisor",),
    # Panel-first families (one module each, several methods)
    "hi52": ("src.signals.classic_anomalies",),
    "mom_12_1": ("src.signals.classic_anomalies",),
    "st_reversal": ("src.signals.classic_anomalies",),
    "rsi2_rev": ("src.signals.classic_anomalies",),
    "dloc_rev": ("src.signals.classic_anomalies",),
    "squeeze": ("src.signals.ttm_squeeze",),
    "avwap": ("src.signals.anchored_vwap",),
    "resid_mom": ("src.signals.residual_momentum",),
    "vol_profile": ("src.signals.volume_profile",),
    # ML OHLCV model — the scoring orchestration + the feature code. A CODE change
    # to either is caught by the AST fingerprint (and, since ml_ohlcv is not
    # replayable, REPORTED rather than auto-masked at weight 0). A RETRAIN changes
    # the output without touching the code, so that comparability is the model
    # registry's job (ml_models), deferred to promotion.
    "ml_ohlcv": ("src.signals.ml_model", "src.analysis.ml_dataset",
                 "src.analysis.pivot_target"),
    # Trend predictability (one module, four sides)
    "kaufman_long": ("src.signals.trend_predictability",),
    "kaufman_short": ("src.signals.trend_predictability",),
    "adx_long": ("src.signals.trend_predictability",),
    "adx_short": ("src.signals.trend_predictability",),
    # Fundamental / corp-action factors
    "f_value": ("src.data.fundamentals",),
    "f_quality": ("src.data.fundamentals",),
    "f_growth": ("src.data.fundamentals",),
    "f_short_squeeze": ("src.data.fundamentals",),
    "f_split": ("src.data.corporate_actions",),
    "f_dividend": ("src.data.corporate_actions",),
}

# Derived quantities that are not per-method but are still stored, and whose
# formula changing invalidates the stored value just as surely.
DERIVED_SOURCES: Dict[str, Tuple[str, ...]] = {
    "confidence": ("src.signals.aggregator", "src.signals.agreement"),
    "combined_score": ("src.signals.aggregator",),
}


def _normalised_ast_hash(module_name: str) -> Optional[str]:
    """SHA of a module's AST with docstrings stripped.

    AST rather than source text so that reformatting, comment edits and
    docstring rewrites — which cannot change behaviour — do not trigger a
    refactor. `ast.dump` omits formatting entirely; docstrings are removed
    explicitly because they ARE AST nodes.
    """
    try:
        mod = __import__(module_name, fromlist=["x"])
        src = inspect.getsource(mod)
    except Exception as e:
        logger.debug(f"[code_version] {module_name} unreadable: {e}")
        return None
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        logger.warning(f"[code_version] {module_name} failed to parse: {e}")
        return None

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = getattr(node, "body", None)
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]

    dumped = ast.dump(tree, annotate_fields=False, include_attributes=False)
    return hashlib.sha256(dumped.encode()).hexdigest()[:16]


def fingerprint(name: str) -> Optional[str]:
    """Combined fingerprint for a method (or derived quantity), or None when it
    has no registered source. None means "cannot judge" and every caller must
    treat it as such — never as "unchanged"."""
    sources = METHOD_SOURCES.get(name) or DERIVED_SOURCES.get(name)
    if not sources:
        return None
    parts: List[str] = []
    for mod in sources:
        h = _normalised_ast_hash(mod)
        if h is None:
            return None                 # incomplete evidence => no verdict
        parts.append(f"{mod}:{h}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def all_fingerprints() -> Dict[str, str]:
    """Every registered name -> its current fingerprint (unjudgeable omitted)."""
    out: Dict[str, str] = {}
    for name in list(METHOD_SOURCES) + list(DERIVED_SOURCES):
        fp = fingerprint(name)
        if fp:
            out[name] = fp
    return out


def unmapped_methods() -> List[str]:
    """Stored method columns with no source mapping — invisible to the detector.

    Surfaced deliberately: an unmapped method silently opts out of automatic
    refactor, which is the same "looks covered, isn't" failure this module
    exists to remove. `tests/test_code_version.py` asserts the list stays empty.
    """
    from src.db.schema import SIGNAL_BASE_METHOD_COLUMNS
    return sorted(set(SIGNAL_BASE_METHOD_COLUMNS) - set(METHOD_SOURCES))


def load_stored() -> Dict[str, dict]:
    """`{name: {fingerprint, first_seen_at}}` from `code_versions`."""
    from src.db import repo
    try:
        df = repo.fetch_df("SELECT name, fingerprint, first_seen_at FROM "
                           "code_versions ORDER BY first_seen_at")
    except Exception as e:
        logger.debug(f"[code_version] no stored versions yet: {e}")
        return {}
    if df is None or df.empty:
        return {}
    out: Dict[str, dict] = {}
    for r in df.to_dict("records"):        # ordered => last wins = current
        out[r["name"]] = {"fingerprint": r["fingerprint"],
                          "first_seen_at": str(r["first_seen_at"])}
    return out


def detect_changes() -> Dict[str, dict]:
    """`{name: {"old": fp|None, "new": fp, "is_new": bool}}` for what moved.

    A name absent from `code_versions` is reported with ``is_new=True``: on the
    very first run everything is new, and treating that as "40 scorers changed"
    would trigger a pointless full refactor. `refactor.py` handles the two cases
    differently.
    """
    stored = load_stored()
    current = all_fingerprints()
    changed: Dict[str, dict] = {}
    for name, fp in current.items():
        prev = stored.get(name)
        if prev is None:
            changed[name] = {"old": None, "new": fp, "is_new": True}
        elif prev["fingerprint"] != fp:
            changed[name] = {"old": prev["fingerprint"], "new": fp,
                             "is_new": False}
    return changed


def record(names: Optional[List[str]] = None) -> int:
    """Persist current fingerprints for ``names`` (default: everything changed).

    Append-only: each row is "this fingerprint was first seen at this instant",
    which is what gives a non-replayable method its epoch for free.
    """
    from datetime import datetime, timezone
    from src.db.connection import connect

    current = all_fingerprints()
    if names is None:
        names = list(detect_changes())
    rows = [{"name": n, "fingerprint": current[n],
             "first_seen_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            for n in names if n in current]
    if not rows:
        return 0
    import pandas as pd
    df = pd.DataFrame(rows)
    with connect() as con:
        con.register("_cv_df", df)
        con.execute("INSERT INTO code_versions (name, fingerprint, first_seen_at) "
                    "SELECT name, fingerprint, first_seen_at FROM _cv_df")
        con.unregister("_cv_df")
    return len(rows)


if __name__ == "__main__":  # pragma: no cover
    unmapped = unmapped_methods()
    if unmapped:
        print(f"UNMAPPED (invisible to auto-refactor): {unmapped}\n")
    ch = detect_changes()
    if not ch:
        print("No implementation changes detected.")
    else:
        fresh = [n for n, d in ch.items() if d["is_new"]]
        moved = [n for n, d in ch.items() if not d["is_new"]]
        if fresh:
            print(f"first seen ({len(fresh)}): {sorted(fresh)}")
        if moved:
            print(f"CHANGED ({len(moved)}): {sorted(moved)}")
