"""Catalyst-label REPAIR pass — a score-free specialist re-types the sentiment
scorer's ``catalyst`` class on the exact digest the scorer saw (2026-09-06,
user directive).

Why this exists
---------------
The v6 sentiment prompt asks the model for a sentiment SCORE and, as a
by-product, a ``catalyst`` class from the fixed ``NEWS_CATALYST_TYPES``
taxonomy. Measured on 363 judged pairs (memory
``catalyst-label-error-rates-2026-09``): DeepSeek contradicts its own
rationale on ~6% of labels, the local qwen3:8b on ~14–19% — and on the local
engine the classes ``product`` / ``capital_structure`` / ``other`` were
essentially unusable (most such labels were wrong), while BOTH engines type an
ETF with one of its holdings' classes. The label feeds a live calibration
(``catalyst_tilt``, per-(catalyst, side) orientation), so a wrong class is
noise entering a decision surface.

The repair is deliberately OUTSIDE the scoring prompt: the scorer's prompt is
version-salted into the sentiment cache and the whole news family shares one
scorer epoch, so editing it for the sake of the label would cost a prompt
version, an epoch and a cache flush — for a field the score never reads.

What runs, in order
-------------------
1. MECHANICAL, zero cost. A FUND target (``sentiment._target_identity`` —
   a fund word in the SEC registrant name OR a Polygon fund type, the SAME
   verdict the scoring header printed) can only carry ``none`` or
   ``macro_sector`` — the scorer's label is overridden at read time in
   ``sentiment.fund_catalyst_override`` (the raw label survives as
   ``meta["catalyst_raw"]`` and on the shadow rows).
2. TRIGGERS (``trigger_for``): any one fires a specialist call — the label is
   in the MODEL's unreliable set (``catalyst_repair_unreliable_classes``,
   keyed per MODEL ID — ``local/qwen3:8b=…`` — with the bare engine name as the
   fallback; a model swapped in under the same engine name starts EMPTY and
   says so once in the log, because a measured error rate belongs to a
   checkpoint, not to a route); a KEYWORD vote on the headlines + the first-pass
   rationale disagrees with the label (``keyword_classes`` /
   ``keyword_disagrees`` — a disagreement DETECTOR, never a label); the target
   is a fund; or the row falls in a small random MONITOR sample
   (``catalyst_repair_monitor_share``) so the unflagged population stays
   measured.
3. SPECIALIST (``build_specialist_prompt`` + ``_call_specialist``): classify
   only, no score, own version salt, the local engine (``catalyst_repair_model``
   or ``local_sentiment_model``), enum-constrained through Ollama's structured
   output (``response_format`` json_schema — verified on 0.33.2; falls back to
   json_object + local validation). Input: an ENRICHED target header (company
   name + the Polygon industry line + FUND flag), the digest with source and
   age, the first-pass RATIONALE as an analyst's note — never the first-pass
   CLASS (an anchor). Output field order is reasoning order (thinking is off):
   what the item is ABOUT → quoted evidence → coarse family → fine class.
   Mechanical consistency: ``holdings``/``related`` ⇒ ``macro_sector``,
   ``unrelated``/``no_event`` ⇒ ``none``; an ``other`` whose evidence is empty
   or not quoted from the digest is a DISCARDED vote (``vote_from``), so the
   dumping-ground class can only ever be voted in on a named, quoted event.
4. VOTING (``classify_digest``): one call; agrees with the first pass → done.
   Disagrees → two more calls with a SHUFFLED class order (deterministic on the
   digest id), majority of the three → ``resolved``; no majority → the first
   pass stands, stamped ``unresolved``.
   THINK ARM (offline, NOT default): ``run_think_arm`` / ``--think-arm`` replays
   the live rows with thinking ON (``reasoning_effort`` in the server's own
   dialect + ``_THINK_MAX_TOKENS`` for chain + answer) on the SAME stored
   digest and rationale, landing as ``arm='think'`` beside the live row so
   ``evaluate`` can pair them. ~13× the live latency, so CLI territory when
   the scheduler is idle — never the tick path, never the EOD chain.
5. PERSISTENCE: ``signals.news_catalyst`` is never overwritten. One row per
   ``(digest_id, engine, arm)`` in ``catalyst_repairs`` (drained non-blocking by
   ``pipeline._persist_run`` like the shadow rows). A pair repaired in the last
   7 days is not re-offered: ``_prime_seen`` seeds the in-memory dedupe from the
   table once per process, so a restart does not re-run the specialist over
   every verdict the sentiment cache still serves. That dedupe is deliberately
   BLIND to ``specialist_version`` / ``specialist_model``: the row is idempotent
   per ``(digest_id, engine)``, so a version-aware re-run would OVERWRITE the
   previous specialist's verdict — a new specialist accrues FORWARD on new
   digests, and the two columns split the eras in ``--eval``. (The in-process
   vote memo IS salted with both, like the sentiment cache key.) CONSUMER RESOLUTION is BUILT and
   OFF (``enable_catalyst_repair_resolution``, gated on the measurement bar
   below): with it on, ``news_events.load_news_events`` resolves
   repair (``resolved`` | ``override``) → live → ``news_event_backfill`` via
   ``repair_lookup`` and exposes ``catalyst_quality``, and the ``catalyst_tilt``
   fit DROPS the ``unresolved`` events — a calibration excludes noise, never
   converts it, and "unchecked" is not "known bad", so the ~85% of events no
   trigger ever fired on stay in. With it off every consumer reads
   ``signals.news_catalyst`` (post fund override) → backfill exactly as before,
   so the pass is ACCRUAL ONLY: it changes no calibration and no prompt.
   ``signals.news_catalyst`` is never overwritten either way, and the live
   per-tick ``catalyst_tilt`` SCORE still reads the first pass — the repair for
   a digest lands after the verdict it repairs, so it can only ever reach the
   fit, not that tick's score.

Cost: ~15–20% of calls were ESTIMATED to trigger; at 1–3 specialist calls each
that is ~25–60 local calls per tick on a 1-worker background pool, behind the
sentiment shadow pass on the same 2-slot server. The estimate is MEASURED from
the first live tick: ``maybe_submit`` tallies every offer (``pop_trigger_counts``)
and ``pipeline._persist_run`` logs one ``[catalyst-repair] tick: offered …
triggered … queued …`` line per tick (``format_tick_summary``), so a trigger
rate drifting off the estimate is visible in the log, not inferred from the
table. Off the critical path: nothing here touches a score, and a dead local
server costs accrual, never a signal.

Measurement (pre-registered, ``python -m src.analysis.catalyst_repair --eval``):
gold = DeepSeek's label on the DeepSeek-primary / local-shadow pairs, overridden
by a judged correction in ``catalyst_judgments`` when one exists. The pass is
therefore ALSO triggered on the local SHADOW label — ``sentiment._run_shadow``
submits its verdict under ``role="shadow"`` (own ``(digest_id, engine)`` row,
trigger stamped ``<trigger>@shadow``), attributed to the engine that ANSWERED
the forced call (a shadow that fell through to the primary's own engine is
dropped as a self-pair) — which is the only place a local first pass, a
specialist verdict and a DeepSeek label coexist. Bar before the repair may FEED anything beyond ``catalyst_tilt``'s
row filter: on local-flagged rows the specialist's error < 25% (first pass
~85%), on the unflagged monitor sample agreement with gold not below the first
pass, both date halves same sign.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import threading
import time
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from loguru import logger

from config.settings import settings

SPECIALIST_VERSION = "cr1-2026-09-06"

# ── Taxonomy scaffolding ────────────────────────────────────────────────────
ABOUT_TARGET = ("target", "holdings", "related", "unrelated", "no_event")
EVENT_FAMILIES = ("results_outlook", "third_party_verdict", "ownership_capital",
                  "operations", "market_context", "none")

# Discriminators, not examples: each definition names the neighbour it is most
# often confused with (measured in the judged sample) and how to tell them
# apart. No worked examples — numbers and phrases written into a prompt become
# modal outputs (sentiment v2, confidence placement v1).
CLASS_DEFINITIONS: Dict[str, str] = {
    "earnings": ("the company's OWN reported quarter or year — revenue, EPS, profit, "
                 "margins, a beat or miss — including commentary ABOUT those results. "
                 "A peer's results are macro_sector, not earnings."),
    "guidance": ("the company's OWN forward outlook issued, raised, cut, reaffirmed or "
                 "withdrawn. Results and a guidance change together: guidance wins."),
    "analyst": ("a brokerage or research firm's rating, price-target or coverage ACTION. "
                "A journalist quoting 'analysts' with no rating or target action is not "
                "analyst — it is no_event."),
    "ma_deal": ("an acquisition, merger, tender offer, take-private, asset sale, or a "
                "stake taken WITH a stated intent to acquire or control."),
    "fda_clinical": ("regulatory decisions on drugs or devices, trial starts, data "
                     "readouts, PDUFA dates, clinical holds, approvals or rejections. "
                     "Trial data is fda_clinical, never product."),
    "legal_regulatory": ("lawsuits, investigations, fines, court rulings, policy or "
                         "tariff actions naming the company, non-FDA agency actions."),
    "management": ("CEO/CFO/board/founder appointments, departures, deaths, or an "
                   "activist board contest."),
    "capital_structure": ("the company's OWN share count or balance sheet — an offering, "
                          "dilution, buyback authorisation or progress, dividend, split, "
                          "debt raise, refinancing, credit rating. An INVESTOR's stake "
                          "filing is insider_activity, never capital_structure."),
    "distress": ("going concern doubt, bankruptcy, restructuring, covenant breach, "
                 "delisting notice."),
    "contract_partnership": ("a contract win or loss, partnership, licensing or supply "
                             "agreement with a named counterparty."),
    "product": ("launches, non-FDA approvals, recalls, outages, accidents, production "
                "or delivery numbers, store openings. Not trial data (fda_clinical), "
                "not a contract (contract_partnership), not a filing."),
    "index_membership": ("added to or removed from an index; an index rebalance naming "
                         "the company."),
    "insider_activity": ("ANY ownership disclosure as the news — 13F/13G/13D stakes, "
                         "Form 4 or 10b5-1 insider buys and sells, institutional "
                         "position changes, congressional trades — whatever the size."),
    "short_squeeze_social": ("short interest, squeeze talk, social-media or retail "
                             "attention, options positioning AS the story."),
    "macro_sector": ("sector-wide, commodity, rate, index-level or peer read-through "
                     "news; a fund's holdings or underlying; anything answered "
                     "'holdings' or 'related' above."),
    "company_pr": ("a promotional company-issued release — award, conference "
                   "appearance, milestone, product spotlight — with no hard numbers "
                   "and no dated event."),
    "other": ("a REAL dated, company-specific event that fits no class above. Requires "
              "a named event with a date; opinion or commentary is never other."),
    "none": ("no dated event — opinion, ranking, listicle, recap, a price-move "
             "explainer, or anything answered 'unrelated' or 'no_event' above."),
}

FAMILY_CLASSES: Dict[str, Tuple[str, ...]] = {
    "results_outlook": ("earnings", "guidance"),
    "third_party_verdict": ("analyst", "fda_clinical", "legal_regulatory", "index_membership"),
    "ownership_capital": ("ma_deal", "insider_activity", "capital_structure", "distress"),
    "operations": ("contract_partnership", "product", "management", "company_pr"),
    "market_context": ("macro_sector", "short_squeeze_social"),
    "none": ("none", "other"),
}


def _taxonomy() -> Tuple[str, ...]:
    from src.analysis.sentiment import NEWS_CATALYST_TYPES
    return tuple(NEWS_CATALYST_TYPES)


# ── Keyword vote — a DISAGREEMENT DETECTOR, never a label ───────────────────
# Whole-word, case-insensitive, run on the headlines AND the first-pass
# rationale. Deliberately narrow: a rule that fires on "analysts expect" or
# "shares" alone would flag most digests and turn the detector into "call the
# specialist always". A hit means "the text names an event of this class"; the
# model may still be right that another item dominates — the specialist decides.
_KW = {
    "insider_activity": [
        r"\b13[fdg]\b", r"\bschedule 13[dg]\b", r"\bform 4\b", r"\b10b5-1\b",
        r"\binsider (?:buy|buys|bought|buying|sell|sells|sold|selling|purchase|sale)s?\b",
        r"\b(?:takes|took|acquires|acquired|holds|owns|boosts|boosted|trims|trimmed|raises|"
        r"raised|lowers|lowered|increases|increased|reduces|reduced|cuts|cut|sells|sold|"
        r"buys|bought|purchases|purchased|adds to|adds|added|discloses|disclosed|reports) "
        r"(?:a |an |its |their |new |the )?(?:\d[\d.,]*\s?%\s?)?(?:stake|position|holdings?|shares)\b",
        r"\b(?:stake|position|holding) in\b",
    ],
    "fda_clinical": [
        r"\bphase (?:1|2|3|i|ii|iii|1/2|2/3|iib?|iiia?)\b", r"\bpdufa\b", r"\bfda\b",
        r"\bclinical (?:trial|hold|data|study|results)\b", r"\btopline\b", r"\btop-line\b",
        r"\btrial (?:results|data|readout|met|failed|misses|hits)\b",
        r"\bcomplete response letter\b", r"\bbreakthrough therapy\b",
        r"\bbiologics license\b", r"\bnew drug application\b", r"\bema\b", r"\bchmp\b",
    ],
    "earnings": [
        r"\beps\b", r"\bq[1-4]\b", r"\b(?:first|second|third|fourth)[- ]quarter\b",
        r"\bquarterly (?:results|earnings|revenue|profit|loss|report)\b",
        r"\b(?:beats?|beat|misses?|missed|topped|tops|trails) (?:\w+ )?(?:estimates|expectations|"
        r"consensus|the street|forecasts|views)\b",
        r"\bearnings (?:report|results|call|beat|miss|season|preview|recap)\b",
        r"\breports? (?:q[1-4]|first|second|third|fourth|quarterly|record|fiscal|full[- ]year)\b",
        r"\bfiscal (?:q[1-4]|20\d\d|year)\b.{0,30}\b(?:results|revenue|eps|profit|loss)\b",
    ],
    "guidance": [
        r"\bguidance\b",
        r"\b(?:raises|raised|cuts|cut|lowers|lowered|lifts|lifted|boosts|boosted|trims|trimmed|"
        r"reaffirms|reaffirmed|maintains|maintained|reiterates|reiterated|withdraws|withdrew|"
        r"issues|issued|updates|updated|narrows|narrowed) (?:its |their |the )?(?:full[- ]year |"
        r"fy ?\d* |fiscal |annual |20\d\d |q[1-4] |quarterly )?(?:outlook|forecast|guidance|targets?|view)\b",
        r"\b(?:outlook|forecast) (?:raised|cut|lowered|lifted|boosted|trimmed|reaffirmed|withdrawn|"
        r"maintained|reiterated)\b",
    ],
    "analyst": [
        r"\bupgrade[sd]?\b", r"\bdowngrade[sd]?\b", r"\bprice target\b", r"\bprice objective\b",
        r"\btarget price\b", r"\binitiat(?:es|ed|ing|ion of) coverage\b",
        r"\b(?:initiated|initiates|resumes|resumed|assumes|assumed) (?:at|with|coverage)\b",
        r"\b(?:overweight|underweight|outperform|underperform|sector perform|market perform|"
        r"equal[- ]weight|strong buy|buy rating|sell rating|hold rating|neutral rating|"
        r"top pick|conviction (?:buy|list))\b",
        r"\breiterat(?:es|ed) (?:a |its )?(?:buy|sell|hold|outperform|overweight|underweight|neutral)\b",
    ],
    "ma_deal": [
        r"\bacqui(?:re|res|red|sition|sitions|ring)\b", r"\bmergers?\b", r"\bmerge with\b",
        r"\btender offer\b", r"\btake[- ]private\b", r"\bbuyout\b", r"\bdefinitive agreement\b",
        r"\btakeover\b", r"\bagrees? to (?:buy|acquire|sell|be acquired)\b",
        r"\bdeal to (?:buy|acquire|sell)\b", r"\bto be acquired\b", r"\bin talks to (?:buy|acquire|sell)\b",
    ],
    "index_membership": [
        r"\b(?:join|joins|joining|joined|added to|removed from|drops? from|dropped from|enter|enters|"
        r"entering|exits?|exiting|deleted from|inclusion in|added into|to be added to|will replace|"
        r"replaces?)\b.{0,50}\b(?:s&p|russell|nasdaq[- ]100|dow jones|msci|ftse|index|indices)\b",
        r"\bindex (?:rebalanc\w*|reconstitution|inclusion|addition|deletion|change)\b",
        r"\b(?:s&p ?500|s&p ?midcap|s&p ?smallcap|russell ?\d{4}|nasdaq[- ]100|msci) "
        r"(?:add|addition|adds|inclusion|entry|rebalance|deletion|removal)\b",
    ],
}
_KW_COMPILED: Dict[str, List[re.Pattern]] = {
    cls: [re.compile(p, re.IGNORECASE) for p in pats] for cls, pats in _KW.items()
}
# Classes a keyword hit does NOT contradict (the same item legitimately types
# either way — results commentary is analyst-or-earnings, results-with-outlook
# is earnings-or-guidance).
KEYWORD_COMPAT: Tuple[frozenset, ...] = (
    frozenset({"earnings", "guidance"}),
    frozenset({"analyst", "earnings"}),
)


def keyword_classes(text: str) -> set:
    """Classes whose keyword rules fire on ``text`` (empty when none do)."""
    if not text:
        return set()
    hits = set()
    for cls, pats in _KW_COMPILED.items():
        if any(p.search(text) for p in pats):
            hits.add(cls)
    return hits


def keyword_disagrees(model_class: Optional[str], kw_classes: Iterable[str]) -> bool:
    """True when the keyword vote names an event class the model's label is
    neither in nor compatible with. Empty keyword vote ⇒ never disagrees."""
    kw = set(kw_classes or ())
    if not kw or model_class is None:
        return False
    allowed = set(kw)
    for group in KEYWORD_COMPAT:
        if kw & group:
            allowed |= group
    return model_class not in allowed


def keyword_text(headlines: Sequence[str], rationale: Optional[str]) -> str:
    """Headlines AND the first-pass rationale, one per line — the detector runs
    over both, so a rationale naming a price target flags ``analyst`` even when
    every headline is plain."""
    parts = [h for h in (headlines or ()) if h]
    if rationale:
        parts.append(rationale)
    return "\n".join(parts)


# ── Triggers ────────────────────────────────────────────────────────────────
def _unreliable_table() -> Dict[str, frozenset]:
    """``catalyst_repair_unreliable_classes`` parsed to ``{key: classes}``
    (``key=cls,cls;key2=cls``; keys lower-cased, a model id or an engine name)."""
    raw = (getattr(settings, "catalyst_repair_unreliable_classes", "") or "").strip()
    table: Dict[str, frozenset] = {}
    for part in re.split(r"[;\n]+", raw):
        if "=" not in part:
            continue
        name, classes = part.split("=", 1)
        table[name.strip().lower()] = frozenset(
            c.strip().lower() for c in classes.split(",") if c.strip())
    return table


def unreliable_classes(engine: Optional[str], model: Optional[str] = None) -> frozenset:
    """The classes measured unreliable for THIS model.

    Keyed per MODEL ID first (``local/qwen3:8b=…``) — a measured error rate
    belongs to a checkpoint, not to a route, so a model swapped in under the
    same engine name must NOT inherit it — with the bare engine name
    (``local=…``) as the fallback for a table written before model ids were
    known (an engine-level key is the operator's statement that it covers the
    whole route). Unknown model AND engine ⇒ empty set.
    """
    table = _unreliable_table()
    for key in (model, engine):
        if key:
            hit = table.get(str(key).strip().lower())
            if hit is not None:
                return hit
    return frozenset()


def _model_id(engine: Optional[str], model: Optional[str] = None) -> Optional[str]:
    """The model id the unreliable table is keyed on: the caller's when it has
    one, else the engine's CURRENT model (``sentiment_model_for``)."""
    if model:
        return str(model)
    if not engine:
        return None
    try:
        from src.analysis.sentiment import sentiment_model_for
        return sentiment_model_for(engine)
    except Exception:                                   # noqa: BLE001
        return None


def _log_set_once(model_id: Optional[str], engine: Optional[str], classes: frozenset) -> None:
    """Say ONCE per process which unreliable set a model resolved to. An empty
    set is the case worth seeing: after a model swap only the fund / keyword /
    monitor triggers fire for it, and nothing else in the log would show that."""
    key = model_id or engine or "?"
    with _LOCK:
        if key in _LOGGED_SETS:
            return
        _LOGGED_SETS.add(key)
    if classes:
        logger.info(f"[catalyst-repair] unreliable set for {key}: {', '.join(sorted(classes))}")
    else:
        logger.info(f"[catalyst-repair] unreliable set for {key}: NONE — no class measured "
                    f"unreliable for it, or an unmeasured model; only the fund / keyword / "
                    f"monitor triggers fire for it")


def _is_fund(ticker: str) -> bool:
    from src.analysis.sentiment import _is_fund_target
    return _is_fund_target(ticker)


def _monitor_sampled(ticker: str, share: float, run_id: Optional[str] = None) -> bool:
    """Deterministic per-(run, ticker) sample, salted apart from the shadow
    sample so the monitor rows are not the shadow's subset. ``run_id`` defaults
    to the scorer's current run; a shadow thread passes its own."""
    if share <= 0:
        return False
    if share >= 1.0:
        return True
    if run_id is None:
        from src.analysis import sentiment as _s
        run_id = _s._CURRENT_RUN_ID
    key = f"cr|{run_id or ''}|{ticker}".encode("utf-8")
    h = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(h, "big") / 2 ** 64 < share


def trigger_for(*, ticker: str, engine: Optional[str], first_pass: Optional[str],
                headlines: Sequence[str] = (), rationale: Optional[str] = None,
                allow_monitor: bool = True, model: Optional[str] = None,
                run_id: Optional[str] = None) -> Optional[str]:
    """Which trigger (if any) sends this first-pass label to the specialist.

    Priority: ``fund`` > ``unreliable_class`` > ``keyword`` > ``monitor``, with
    ``fund`` OFF since 2026-09-11 (`catalyst_repair_fund_trigger`): it was 43% of
    the pass's GPU for a measured wash (specialist 23.0% vs first pass 22.8% over
    3,213 pairs), because a fund's label is already corrected mechanically by
    `sentiment.fund_catalyst_override`. Funds still reach the specialist through
    the `monitor` sample, so they stay measured; the override is untouched.
    None when the pass is off, the engine cannot be reached (no local LLM), or
    there is no label to repair (the scorer omitted the field). The unreliable
    set is resolved for the engine's CURRENT model id (``model`` overrides).
    """
    if not getattr(settings, "enable_catalyst_repair", False):
        return None
    if not getattr(settings, "enable_local_llm", False):
        return None
    if first_pass is None:
        return None
    model_id = _model_id(engine, model)
    unrel = unreliable_classes(engine, model_id)
    _log_set_once(model_id, engine, unrel)
    if (bool(getattr(settings, "catalyst_repair_fund_trigger", False))
            and _is_fund(ticker)):
        return "fund"
    if first_pass in unrel:
        return "unreliable_class"
    if keyword_disagrees(first_pass, keyword_classes(keyword_text(headlines, rationale))):
        return "keyword"
    if allow_monitor and _monitor_sampled(
            ticker, float(getattr(settings, "catalyst_repair_monitor_share", 0.0) or 0.0),
            run_id=run_id):
        return "monitor"
    return None


# ── Specialist prompt ───────────────────────────────────────────────────────
def _enriched_header(ticker: str) -> str:
    """``TARGET: AR — Antero Resources Corp (Crude Petroleum & Natural Gas)``
    plus the fund line. Resolved through ``sentiment._target_identity`` — the
    SAME identity the scoring header and the fund trigger read — so a target
    the trigger routed as a fund is always presented to the specialist as one
    (a Polygon-typed fund whose name lacks a fund word included)."""
    sym = (ticker or "").strip().upper()
    name = None
    fund = False
    industry = None
    try:
        from src.analysis.sentiment import _target_identity
        sym, name, industry, fund = _target_identity(sym)
    except Exception as exc:                        # noqa: BLE001 - header is best effort
        logger.debug(f"[catalyst-repair] header enrichment failed for {sym}: {exc}")
    line = f"TARGET: {sym}"
    if name:
        line += f" — {name}"
    if industry:
        line += f" ({industry})"
    if fund:
        line += ("\nThe target is a FUND / ETF: its only events are about what it holds, "
                 "its index or its underlying asset (answer 'holdings' → macro_sector), "
                 "or nothing (none). A holding's own event is never the fund's class.")
    return line


def class_order(digest_id: Optional[str], call_idx: int) -> List[str]:
    """Taxonomy order for a vote: the canonical order on the first call, a
    deterministic shuffle (seeded on the digest id + call index) afterwards, so
    the two extra votes are not the first vote re-read with a different
    seed."""
    order = list(_taxonomy())
    if call_idx <= 0:
        return order
    rng = random.Random(f"{digest_id or ''}|{SPECIALIST_VERSION}|{call_idx}")
    rng.shuffle(order)
    return order


def build_specialist_prompt(ticker: str, digest_text: str, rationale: Optional[str],
                            *, order: Optional[Sequence[str]] = None,
                            header: Optional[str] = None) -> str:
    order = list(order or _taxonomy())
    defs = "\n".join(f"- {cls}: {CLASS_DEFINITIONS.get(cls, '')}" for cls in order)
    header = header if header is not None else _enriched_header(ticker)
    note = (rationale or "").strip() or "(no note)"
    return (
        "You are a news CLASSIFIER for an equity research desk. You do NOT score "
        "sentiment and you do NOT judge whether the news is good or bad. Your only job "
        "is to name the ONE event class that DOMINATES this news digest for the target "
        "security, using the fixed taxonomy below.\n\n"
        f"{header}\n\n"
        "Decide in this order and answer each step in its own JSON field:\n"
        "1. about_target — what the dominant item is about: 'target' (an event of the "
        "target company itself); 'holdings' (the target is a fund and the item concerns "
        "what it holds, its index or its underlying asset); 'related' (a named peer, "
        "customer, supplier, competitor, or sector-wide/macro news with a stated link to "
        "the target); 'unrelated' (a different company or entity with no stated link to "
        "the target); 'no_event' (opinion, ranking, recap or commentary with no dated "
        "event).\n"
        "2. evidence — the shortest phrase QUOTED verbatim from the digest that "
        "establishes the event (empty string for no_event).\n"
        "3. event_family — the coarse family: 'results_outlook' (the company's own "
        "reported numbers or outlook), 'third_party_verdict' (a regulator, court, agency, "
        "analyst or index provider acting on the company), 'ownership_capital' (deals, "
        "stakes, insider transactions, share count, debt, solvency), 'operations' "
        "(contracts, products, people, promotional releases), 'market_context' "
        "(sector/macro/holdings read-through, positioning or social attention), 'none'.\n"
        "4. catalyst — exactly one class from the taxonomy, consistent with the family.\n\n"
        "TAXONOMY:\n"
        f"{defs}\n\n"
        "Rules: one class only — the DOMINANT event of the digest, the one the analyst's "
        "note leans on. Routine items keep their own class (a small stake is "
        "insider_activity, a regular dividend is capital_structure, a conference "
        "appearance is company_pr). Do not infer an event the digest does not state. "
        "'holdings' and 'related' always resolve to macro_sector; 'unrelated' and "
        "'no_event' always resolve to none.\n\n"
        "<analyst_note>\n"
        f"{note}\n"
        "</analyst_note>\n"
        "(The note explains a sentiment read of the same digest. It may be wrong about "
        "WHAT KIND of event this is — classify from the digest, use the note only to see "
        "which item it leaned on.)\n\n"
        "<news>\n"
        f"{digest_text}\n"
        "</news>\n\n"
        "Respond with ONLY a JSON object with exactly these fields, in this order:\n"
        '{"about_target": "<target|holdings|related|unrelated|no_event>", '
        '"evidence": "<quoted phrase>", "event_family": "<family>", "catalyst": "<class>"}'
    )


def _json_schema(order: Sequence[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "about_target": {"type": "string", "enum": list(ABOUT_TARGET)},
            "evidence": {"type": "string"},
            "event_family": {"type": "string", "enum": list(EVENT_FAMILIES)},
            "catalyst": {"type": "string", "enum": list(order)},
        },
        "required": ["about_target", "evidence", "event_family", "catalyst"],
        "additionalProperties": False,
    }


def specialist_model() -> str:
    """Model id the specialist calls — ``catalyst_repair_model`` when set, else
    the local sentiment model (one server serves both)."""
    m = (getattr(settings, "catalyst_repair_model", "") or "").strip()
    return m or str(settings.local_sentiment_model)


_SPECIALIST_MAX_TOKENS = 512

# ── Specialist ARMS ─────────────────────────────────────────────────────────
# ``live`` is the tick-time pass (thinking OFF, the sentiment route's own
# dialect); ``think`` is the OFFLINE measurement arm — thinking ON for the
# specialist only, replayed over the live rows' stored digest + rationale by
# ``run_think_arm`` (CLI ``--think-arm``), never on the tick path and never in
# the EOD chain. Probed on the running Ollama 0.33.2 / qwen3:8b (2026-09-07,
# OpenAI endpoint, json_schema, temperature 0): any ``reasoning_effort`` other
# than ``"none"`` turns thinking ON and the LEVEL is ignored (``medium`` and
# ``high`` byte-identical); the chain comes back in ``message.reasoning`` with
# the content still clean, schema-enforced JSON; the reasoning tokens are
# counted INSIDE ``prompt_tokens`` (not completion_tokens), so the post-call
# truncation ratio would misread a healthy think call as a truncated one and
# is SKIPPED on this arm; ``max_tokens`` caps chain + answer together, and an
# exhausted budget returns finish_reason ``length`` with EMPTY content — a
# discarded vote, never a class; latency ~13× (1.3 s → ~17 s per call).
ARM_LIVE = "live"
ARM_THINK = "think"
_THINK_MAX_TOKENS = 1500
_THINK_EFFORT = "high"
# Ollama honours json_schema on the OpenAI endpoint (verified on the running
# 0.33.2 build, 2026-09-06: a prompt pushing an out-of-enum label came back
# INSIDE the enum under json_schema, strict/additionalProperties accepted, and
# OUTSIDE it under json_object — so the enum IS enforced, json_object is NOT,
# and the fallback path leans entirely on ``vote_from``'s local validation).
# An older build 400s on it, which is remembered per process so every later
# call goes straight to the fallback.
_SCHEMA_MODE = {"json_schema": True}
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_answer(raw: str) -> str:
    raw = (raw or "").strip()
    if "<think>" in raw:
        raw = _THINK_RE.sub("", raw).strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return raw


def think_extra_body() -> dict:
    """The sentiment route's server dialect with thinking switched ON.

    Starts from ``local_sentiment_extra_body`` (live: ``{"reasoning_effort":
    "none"}``) and replaces the effort — the specialist must speak the SAME
    dialect the server is known to honour, so a dialect without a
    ``reasoning_effort`` key is refused (fail closed) rather than guessed at:
    a key the server ignores would silently run the arm with thinking OFF and
    label the result as the think arm."""
    from src.analysis import sentiment as _s
    body = dict(_s._local_extra_body())
    if "reasoning_effort" not in body:
        raise RuntimeError(
            "local_sentiment_extra_body carries no 'reasoning_effort' key; "
            "the think arm cannot switch thinking on in this server's dialect")
    body["reasoning_effort"] = _THINK_EFFORT
    return body


def _call_specialist(prompt: str, order: Sequence[str], *, label: str,
                     think: bool = False) -> dict:
    """One specialist call. ``think=True`` selects the offline thinking-on arm:
    the output budget is ``_THINK_MAX_TOKENS`` (chain + answer), the pre-flight
    reserves that budget out of the context, the post-call truncation ratio is
    skipped (the server counts the chain inside ``prompt_tokens``), and an
    exhausted budget (finish_reason ``length``) is refused. The chain length
    rides back on the answer as ``_reasoning_chars`` for the vote record."""
    from src.analysis import local_llm
    from src.analysis import sentiment as _s
    local = _s._get_local()
    if local is None:
        raise RuntimeError("local LLM client unavailable")
    model = specialist_model()
    ctx = int(getattr(settings, "local_sentiment_context_tokens", 0) or 0)
    max_tokens = _THINK_MAX_TOKENS if think else _SPECIALIST_MAX_TOKENS
    # The answer (and, on the think arm, the chain) must fit INSIDE the context
    # beside the prompt: reserve the output budget before judging the prompt.
    fit_ctx = max(ctx - max_tokens, 0) if ctx > 0 else 0
    est = local_llm.check_fits(prompt, context_tokens=fit_ctx, label=label)
    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        seed=_s._LLM_SEED,
        extra_body=think_extra_body() if think else _s._local_extra_body(),
    )
    response = None
    if _SCHEMA_MODE["json_schema"]:
        try:
            response = local.chat.completions.create(
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "catalyst", "strict": True,
                                                 "schema": _json_schema(order)}},
                **kwargs)
        except Exception as exc:                    # noqa: BLE001
            msg = str(exc).lower()
            if "json_schema" in msg or "response_format" in msg or "400" in msg:
                logger.warning("[catalyst-repair] json_schema refused by the local "
                               "server ({}); falling back to json_object", str(exc)[:120])
                _SCHEMA_MODE["json_schema"] = False
            else:
                raise
    if response is None:
        response = local.chat.completions.create(
            response_format={"type": "json_object"}, **kwargs)
    if not think:
        local_llm.check_reported(
            getattr(getattr(response, "usage", None), "prompt_tokens", None),
            estimate=est, context_tokens=ctx, label=label)
    choice = response.choices[0]
    message = getattr(choice, "message", None)
    finish = getattr(choice, "finish_reason", None)
    content = _strip_answer(getattr(message, "content", None) or "")
    if think and finish == "length":
        raise RuntimeError(
            f"think budget exhausted ({_THINK_MAX_TOKENS} tokens, finish_reason=length)")
    if not content:
        raise RuntimeError("empty specialist answer")
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("specialist answer is not an object")
    if think:
        data["_reasoning_chars"] = len(getattr(message, "reasoning", None) or "")
    return data


def _norm_quote(text: str) -> str:
    """Case-, whitespace- and quote-mark-insensitive form of a phrase, so a
    phrase the model re-cased, re-quoted or wrapped across a line break still
    matches the digest it was lifted from. Quote marks (straight and smart)
    are dropped on BOTH sides — they carry no matching value."""
    text = (text or "").lower()
    for q in ("“", "”", "‘", "’", "'", '"'):
        text = text.replace(q, "")
    return " ".join(text.split())


def evidence_grounded(evidence: Optional[str], digest_text: Optional[str]) -> Optional[bool]:
    """Does the quoted evidence phrase actually appear in the digest? None when
    there is no digest to check against (or no phrase to check)."""
    if not digest_text:
        return None
    ev = _norm_quote(evidence or "")
    if len(ev) < 3:
        return None
    return ev in _norm_quote(digest_text)


def vote_from(data: dict, digest_text: Optional[str] = None) -> Optional[dict]:
    """Validate one specialist answer and apply the mechanical consistency
    rules. None ⇒ discarded vote (unknown class / no class).

    ``other`` is the one class the prompt cannot define positively ("a real
    dated event that fits nothing above"), which is how the local first pass
    made it a dumping ground — so it is the one class that must EARN its vote:
    it needs a non-empty evidence phrase, and when the digest is available
    that phrase must be quoted FROM it (``evidence_grounded``). An ``other``
    without one is a discarded vote, never a class, so it can only leave the
    row ``unresolved``. Every vote records ``grounded`` (None without a
    digest) so the eval can split quoted from invented evidence.
    """
    if not isinstance(data, dict):
        return None
    tax = set(_taxonomy())
    about = str(data.get("about_target") or "").strip().lower()
    fam = str(data.get("event_family") or "").strip().lower()
    cls = str(data.get("catalyst") or "").strip().lower().replace(" ", "_").replace("-", "_")
    if about not in ABOUT_TARGET:
        about = ""
    if fam not in EVENT_FAMILIES:
        fam = ""
    if about in ("holdings", "related"):
        cls = "macro_sector"
    elif about in ("unrelated", "no_event"):
        cls = "none"
    if cls not in tax:
        return None
    ev = data.get("evidence")
    ev_str = str(ev) if ev is not None else ""
    grounded = evidence_grounded(ev_str, digest_text)
    if cls == "other" and (len(_norm_quote(ev_str)) < 3 or grounded is False):
        return None
    return {
        "about_target": about or None,
        "evidence": (ev_str[:200] if ev is not None else None),
        "event_family": fam or None,
        "catalyst": cls,
        "grounded": grounded,
    }


# Per-(digest, call index) vote memo: a DeepSeek-primary digest and its local
# shadow verdict are repaired as two rows, but the specialist's answer to the
# SAME prompt is the same — the second row reuses the calls the first made.
# The key is salted with the specialist's own prompt version AND model id, the
# discipline ``_sentiment_cache_key`` enforces: a prompt edit or a model swap
# must never serve the previous specialist's answers — and with the ARM, since
# the think arm's whole point is a DIFFERENT answer to the same prompt.
_VOTE_CACHE: "OrderedDict[Tuple[str, str, str, str, int], Optional[dict]]" = OrderedDict()
_VOTE_CACHE_MAX = 4000
_VOTE_LOCK = threading.Lock()


def _vote_key(digest_id: str, idx: int, arm: str = ARM_LIVE) -> Tuple[str, str, str, str, int]:
    return (digest_id, SPECIALIST_VERSION, specialist_model(), arm, idx)


def _cached_vote(digest_id: Optional[str], idx: int, arm: str = ARM_LIVE):
    if not digest_id:
        return False, None
    key = _vote_key(digest_id, idx, arm)
    with _VOTE_LOCK:
        if key in _VOTE_CACHE:
            _VOTE_CACHE.move_to_end(key)
            return True, _VOTE_CACHE[key]
    return False, None


def _store_vote(digest_id: Optional[str], idx: int, vote: Optional[dict],
                arm: str = ARM_LIVE) -> None:
    if not digest_id:
        return
    key = _vote_key(digest_id, idx, arm)
    with _VOTE_LOCK:
        _VOTE_CACHE[key] = vote
        while len(_VOTE_CACHE) > _VOTE_CACHE_MAX:
            _VOTE_CACHE.popitem(last=False)


def specialist_version_for(arm: str) -> str:
    """The version stamp a row carries: the prompt version, suffixed by the arm
    when it is not the live one, so the two arms split into their own eras."""
    return SPECIALIST_VERSION if arm == ARM_LIVE else f"{SPECIALIST_VERSION}+{arm}"


def _one_vote(digest_id: Optional[str], ticker: str, digest_text: str,
              rationale: Optional[str], idx: int, header: str,
              arm: str = ARM_LIVE) -> Tuple[Optional[dict], float, Optional[str]]:
    """→ (vote | None, latency_s, error | None); memoised per (digest, arm, idx)."""
    hit, cached = _cached_vote(digest_id, idx, arm)
    if hit:
        return (dict(cached) if cached else None), 0.0, None
    order = class_order(digest_id, idx)
    prompt = build_specialist_prompt(ticker, digest_text, rationale, order=order, header=header)
    t0 = time.perf_counter()
    err = None
    vote = None
    # ``think`` is passed ONLY on the think arm so the live call keeps its
    # signature byte-for-byte (the test stubs take the live form).
    think_kw = {"think": True} if arm == ARM_THINK else {}
    try:
        data = _call_specialist(prompt, order, label=f"catalyst specialist ({ticker} #{idx})",
                                **think_kw)
        reasoning_chars = data.pop("_reasoning_chars", None) if isinstance(data, dict) else None
        vote = vote_from(data, digest_text=digest_text)
        if vote is not None:
            vote["shuffled"] = idx > 0
            if reasoning_chars is not None:
                vote["reasoning_chars"] = int(reasoning_chars)
    except Exception as exc:                        # noqa: BLE001
        err = f"{type(exc).__name__}: {str(exc)[:160]}"
        logger.debug(f"[catalyst-repair] {ticker} vote #{idx} ({arm}) failed: {err}")
    lat = time.perf_counter() - t0
    if err is None:
        _store_vote(digest_id, idx, vote, arm)
    return vote, lat, err


def classify_digest(*, digest_id: Optional[str], ticker: str, digest_text: str,
                    rationale: Optional[str], first_pass: Optional[str],
                    final_label: Optional[str] = None, trigger: Optional[str] = None,
                    arm: str = ARM_LIVE) -> dict:
    """The voting procedure. Returns the row fields the ledger stores.

    ``final_label`` is the label the first pass ENDED with after mechanical
    overrides (the fund override); when it differs from ``first_pass`` the
    override decided and the row is stamped ``override`` whatever the vote —
    the specialist call is monitoring only.

    ``arm`` selects the specialist configuration (``live`` | ``think``); it
    changes the calls, the memo key and the version stamp, never the procedure.
    """
    # The mechanical override has already decided this row, so the MAJORITY
    # cannot change the outcome — only the first vote is worth buying, and it is
    # bought for monitoring (that is what the docstring above means by
    # "monitoring only"). The two tie-break votes exist solely to resolve a
    # dissent into a majority, so on an overridden row they are discarded by
    # construction: measured 2026-09-11, override rows averaged 2.99 calls — the
    # MAXIMUM — because the specialist reliably dissents from a fund's
    # company-event first pass, and a dissent buys two more. 664 of 8,548
    # specialist calls (7.8%, ~2,280 s of the GPU that also serves live
    # sentiment) were spent this way.
    overridden = (final_label is not None and first_pass is not None
                  and final_label != first_pass)
    header = _enriched_header(ticker)
    votes: List[dict] = []
    errors: List[str] = []
    total = 0.0
    n_calls = 0
    v1, lat, err = _one_vote(digest_id, ticker, digest_text, rationale, 0, header, arm)
    total += lat
    n_calls += 1
    if err:
        errors.append(err)
    if v1 is not None:
        votes.append(dict(v1, latency_s=round(lat, 3)))
    final = first_pass
    quality = "unresolved"
    if v1 is not None and v1["catalyst"] == first_pass:
        final, quality = v1["catalyst"], "resolved"
    elif (v1 is not None or err is None) and not overridden:
        # Disagreement (or a discarded first vote): two more, shuffled order.
        for idx in (1, 2):
            v, lat, err = _one_vote(digest_id, ticker, digest_text, rationale, idx, header, arm)
            total += lat
            n_calls += 1
            if err:
                errors.append(err)
            if v is not None:
                votes.append(dict(v, latency_s=round(lat, 3)))
        tally = Counter(v["catalyst"] for v in votes)
        if tally:
            cls, n = tally.most_common(1)[0]
            if n >= 2:
                final, quality = cls, "resolved"
    if overridden:
        final, quality = final_label, "override"
    return {
        "votes": votes,
        "n_calls": n_calls,
        "final": final,
        "quality": quality,
        "about_target": (votes[0].get("about_target") if votes else None),
        "latency_s": round(total, 3),
        "error": ("; ".join(errors)[:300] if errors else None),
        "specialist_model": f"local/{specialist_model()}",
        "specialist_version": specialist_version_for(arm),
        "arm": arm,
        "trigger": trigger,
    }


# ── Background pool + ledger buffer ─────────────────────────────────────────
_POOL: Optional[ThreadPoolExecutor] = None
_POOL_LOCK = threading.Lock()
_LOCK = threading.Lock()
_ROWS: List[dict] = []
_PENDING = 0
_WARNED = {"cap": False}
_SEEN: "OrderedDict[Tuple[str, str], bool]" = OrderedDict()
_SEEN_MAX = 20000
# Per-tick trigger tally (``pop_trigger_counts`` copies and clears it):
# ``offered:<role>``, ``no_digest``, ``untriggered``, ``triggered:<trigger>``,
# ``dedup``, ``cap``, ``queued:<stamp>`` — offered = no_digest + untriggered +
# triggered, and triggered = dedup + cap + queued.
_COUNTS: Counter = Counter()
_LOGGED_SETS: set = set()
_PRIMED = {"done": False}
_PRIME_DAYS = 7


def _pool() -> ThreadPoolExecutor:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cat-repair")
        return _POOL


def _seen_before(digest_id: str, engine: str) -> bool:
    key = (digest_id, engine)
    with _LOCK:
        if key in _SEEN:
            return True
        _SEEN[key] = True
        while len(_SEEN) > _SEEN_MAX:
            _SEEN.popitem(last=False)
    return False


def _stamp(trigger: str, role: str) -> str:
    """The persisted trigger: bare on the primary path, ``<trigger>@<role>`` for
    any other role (``keyword@shadow``)."""
    return trigger if role == "primary" else f"{trigger}@{role}"


def _count(key: str, n: int = 1) -> None:
    with _LOCK:
        _COUNTS[key] += n


def pop_trigger_counts() -> Dict[str, int]:
    """Drain the per-tick trigger tally (copy-and-clear)."""
    with _LOCK:
        out = dict(_COUNTS)
        _COUNTS.clear()
    return out


def format_tick_summary(counts: Dict[str, int], n_written: int, pending: int) -> str:
    """One log line per tick, e.g. ``offered 175 (primary 161 / shadow 14);
    triggered 41 (23.4%: fund 3 / unreliable_class 9 / keyword 22 / monitor 7);
    queued 31 (shadow 12); dedup 10; cap 0; no digest 0; pending 4; persisted 27
    row(s)``. The rate is triggered over the offers that carried a digest."""
    counts = counts or {}
    offered = sum(v for k, v in counts.items() if k.startswith("offered:"))
    primary = int(counts.get("offered:primary", 0))
    shadow = offered - primary
    no_digest = int(counts.get("no_digest", 0))
    trig = {k.split(":", 1)[1]: int(v) for k, v in counts.items() if k.startswith("triggered:")}
    triggered = sum(trig.values())
    order = ["fund", "unreliable_class", "keyword", "monitor"]
    kinds = [k for k in order if k in trig] + sorted(k for k in trig if k not in order)
    base = offered - no_digest
    rate = f"{triggered / base * 100:.1f}%" if base > 0 else "n/a"
    detail = " / ".join(f"{k} {trig[k]}" for k in kinds) if kinds else "none"
    queued = {k.split(":", 1)[1]: int(v) for k, v in counts.items() if k.startswith("queued:")}
    n_queued = sum(queued.values())
    q_shadow = sum(v for k, v in queued.items() if "@" in k)
    return (f"offered {offered} (primary {primary} / shadow {shadow}); "
            f"triggered {triggered} ({rate}: {detail}); "
            f"queued {n_queued} (shadow {q_shadow}); "
            f"dedup {int(counts.get('dedup', 0))}; cap {int(counts.get('cap', 0))}; "
            f"no digest {no_digest}; pending {int(pending)}; persisted {int(n_written)} row(s)")


def _prime_seen() -> None:
    """Seed the in-memory dedupe from the last ``_PRIME_DAYS`` of
    ``catalyst_repairs`` once per process, so a restart does not re-run the
    specialist over every verdict the sentiment cache still serves (the
    sentiment cache outlives the process; ``_SEEN`` did not). Fail-soft: a
    failed read costs only duplicate repairs, which the repo write collapses."""
    with _LOCK:
        if _PRIMED["done"]:
            return
        _PRIMED["done"] = True
    try:
        from src.db import repo
        # ``generated_at`` is an ISO string column, so the cutoff is compared
        # lexicographically against the same format it was written in.
        cutoff = (datetime.now(timezone.utc) - timedelta(days=_PRIME_DAYS)).isoformat()
        df = repo.fetch_df("SELECT DISTINCT digest_id, engine FROM catalyst_repairs "
                           "WHERE generated_at >= ?", [cutoff])
        n = 0
        if df is not None and not df.empty:
            with _LOCK:
                for d, e in zip(df["digest_id"], df["engine"]):
                    if d and e:
                        _SEEN[(str(d), str(e))] = True
                        n += 1
                while len(_SEEN) > _SEEN_MAX:
                    _SEEN.popitem(last=False)
        logger.info(f"[catalyst-repair] primed {n} repaired (digest, engine) pair(s) "
                    f"from the last {_PRIME_DAYS} days")
    except Exception as exc:                        # noqa: BLE001
        logger.debug(f"[catalyst-repair] dedupe prime skipped: {exc}")


def maybe_submit(*, ticker: str, engine: Optional[str], first_pass: Optional[str],
                 final_label: Optional[str], rationale: Optional[str],
                 digest_id: Optional[str], digest_text: Optional[str],
                 articles: Sequence = (), role: str = "primary",
                 run_id: Optional[str] = None) -> Optional[str]:
    """Called for every LLM verdict — the primary's from the sentiment scorer
    (``role="primary"``) and the shadow engine's from ``sentiment._run_shadow``
    (``role="shadow"``, attributed to the engine that ANSWERED). Decides the
    trigger, and queues the specialist in the background. Returns the trigger
    name when queued, else None. Never raises.

    Every offer is COUNTED (``pop_trigger_counts``) so the per-tick volume is
    measured rather than estimated. ``run_id`` defaults to the scorer's current
    run — right on the primary path, NOT on a shadow thread that outlives its
    tick, so ``_run_shadow`` passes its own.
    """
    global _PENDING
    try:
        if not getattr(settings, "enable_catalyst_repair", False):
            return None
        if not getattr(settings, "enable_local_llm", False):
            return None
        _prime_seen()
        _count(f"offered:{role}")
        if not digest_id or not digest_text or not engine:
            _count("no_digest")
            return None
        if run_id is None:
            from src.analysis import sentiment as _s
            run_id = _s._CURRENT_RUN_ID
        headlines = [getattr(a, "title", "") or "" for a in (articles or ())]
        trig = trigger_for(ticker=ticker, engine=engine, first_pass=first_pass,
                           headlines=headlines, rationale=rationale, run_id=run_id)
        if trig is None:
            _count("untriggered")
            return None
        _count(f"triggered:{trig}")
        if _seen_before(digest_id, engine):
            _count("dedup")
            return None
        cap = int(getattr(settings, "catalyst_repair_max_pending", 200) or 0)
        stamp = _stamp(trig, role)
        with _LOCK:
            if cap and _PENDING >= cap:
                if not _WARNED["cap"]:
                    logger.warning(f"[catalyst-repair] {_PENDING} repairs pending ≥ cap {cap} — "
                                   f"the specialist is not keeping up; skipping until it drains")
                    _WARNED["cap"] = True
                # let the same digest be re-offered later
                _SEEN.pop((digest_id, engine), None)
                _COUNTS["cap"] += 1
                return None
            _PENDING += 1
            _COUNTS[f"queued:{stamp}"] += 1
        _pool().submit(_run_repair, run_id, ticker, engine, first_pass, final_label,
                       rationale, digest_id, digest_text, trig, role)
        return trig
    except Exception as exc:                        # noqa: BLE001
        logger.debug(f"[catalyst-repair] submit failed for {ticker}: {exc}")
        return None


def _run_repair(run_id, ticker, engine, first_pass, final_label, rationale,
                digest_id, digest_text, trigger, role) -> None:
    global _PENDING
    try:
        from src.analysis.sentiment import sentiment_model_for
        out = classify_digest(digest_id=digest_id, ticker=ticker, digest_text=digest_text,
                              rationale=rationale, first_pass=first_pass,
                              final_label=final_label, trigger=trigger)
        row = {
            "digest_id": digest_id,
            "run_id": run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ticker": ticker,
            "engine": engine,
            "model": sentiment_model_for(engine),
            "first_pass": first_pass,
            # The first-pass rationale is persisted nowhere else for a shadow
            # verdict; storing it here is what lets the offline think arm
            # replay the EXACT specialist input.
            "rationale": rationale,
            **out,
            # After ``**out``: classify_digest echoes the bare trigger, and the
            # role stamp (``keyword@shadow``) must win over that echo.
            "trigger": _stamp(trigger, role),
        }
        with _LOCK:
            _ROWS.append(row)
    except Exception as exc:                        # noqa: BLE001 - measurement only
        logger.debug(f"[catalyst-repair] {ticker} repair failed: {exc}")
    finally:
        with _LOCK:
            _PENDING -= 1


def pop_catalyst_repair_rows() -> List[dict]:
    """Drain finished rows (non-blocking — a repair still in flight lands on a
    later drain carrying its own run_id)."""
    with _LOCK:
        rows, _ROWS[:] = list(_ROWS), []
    return rows


def catalyst_repair_pending() -> int:
    with _LOCK:
        return _PENDING


def _reset_for_tests() -> None:
    global _PENDING
    with _LOCK:
        _ROWS.clear()
        _SEEN.clear()
        _COUNTS.clear()
        _LOGGED_SETS.clear()
        _PENDING = 0
        _WARNED["cap"] = False
        _PRIMED["done"] = True        # no DB prime under test unless a test arms it
    with _VOTE_LOCK:
        _VOTE_CACHE.clear()
    _SCHEMA_MODE["json_schema"] = True


def _drain_for_tests(timeout: float = 30.0) -> None:
    """Wait for every queued repair to finish (tests only)."""
    t0 = time.time()
    while catalyst_repair_pending() > 0 and time.time() - t0 < timeout:
        time.sleep(0.01)


# ── Consumer resolution (repair → live → backfill) ──────────────────────────
def repair_lookup(days: Optional[int] = None) -> Dict[Tuple[str, str], dict]:
    """``{(digest_id, starting_label): {"catalyst", "quality", "engine"}}`` over
    the LIVE arm, for consumers resolving repair → live → backfill.

    The key carries the label the repair STARTED from, not just the digest,
    because one digest can hold two rows — the primary engine's verdict and the
    shadow engine's, which routinely disagree. Joining on the digest alone would
    hand a consumer the OTHER engine's repair; keyed by the starting label, a
    caller matches the row that repaired the label it actually holds. That
    starting label is ``final`` on an ``override`` row (the fund override had
    already rewritten the label the panel stores) and ``first_pass`` otherwise.

    ``unresolved`` rows are RETURNED, not dropped — the caller needs to know a
    label was checked and stayed doubtful (the calibration then excludes it;
    excluding noise is the standing rule, converting it is not). A row whose
    repair errored still carries ``quality='unresolved'``.

    The offline ``think`` arm is excluded by construction: it is a measurement
    arm replayed later on the same digests, so letting it resolve a label would
    put a future call into a walk-forward view. Under ``analysis_asof`` the rows
    are bounded by their own ``generated_at`` — a repair is knowable only once
    it has run, and it runs after the verdict it repairs.
    """
    from src.db import repo
    where = [f"coalesce(arm, '{ARM_LIVE}') = '{ARM_LIVE}'", "digest_id IS NOT NULL"]
    params: list = []
    if days:
        where.append("generated_at >= ?")
        params.append(_days_cutoff_iso(days))
    try:
        from src.analysis.asof import current_asof
        _asof = current_asof()
        if _asof:
            # ISO strings compare lexicographically, so a plain date is a valid
            # upper bound on a timestamp ('2026-09-08T12:00+00:00' < '2026-09-09').
            where.append("generated_at < ?")
            params.append(str(_asof))
    except Exception:                                   # noqa: BLE001
        pass
    sql = ("SELECT digest_id, engine, trigger, first_pass, final, quality, generated_at "
           f"FROM catalyst_repairs WHERE {' AND '.join(where)} "
           "ORDER BY generated_at")
    try:
        df = repo.fetch_df(sql, params)
    except Exception as exc:                            # noqa: BLE001
        logger.warning(f"[catalyst-repair] repair lookup failed (consumers keep the "
                       f"live label): {exc}")
        return {}
    out: Dict[Tuple[str, str], dict] = {}
    if df is None or df.empty:
        return out
    for r in df.itertuples(index=False):
        quality = str(r.quality or "unresolved")
        start = r.final if quality == "override" else r.first_pass
        key = (str(r.digest_id), "" if start is None else str(start))
        shadow = str(r.trigger or "").endswith("@shadow")
        prev = out.get(key)
        # Deterministic pick when both engines repaired the same starting label:
        # a PRIMARY row always beats a shadow row (the panel's label is the
        # primary's), and between two rows of the same role the later one wins —
        # the frame arrives ordered by `generated_at`, so that is a plain
        # last-wins.
        if prev is not None and shadow and not prev["shadow"]:
            continue
        out[key] = {"catalyst": (None if r.final is None else str(r.final)),
                    "quality": quality,
                    "engine": (None if r.engine is None else str(r.engine)),
                    "shadow": shadow}
    return out


# ── Offline THINK arm ───────────────────────────────────────────────────────
_THINK_BATCH = 20


def _days_cutoff_iso(days: int) -> str:
    """ISO cutoff for a ``generated_at`` bound. Every timestamp column here is a
    VARCHAR ISO string (UTC, ``+00:00``), so the bound is a STRING compared
    lexicographically — DuckDB refuses ``varchar >= now() - INTERVAL`` outright
    (a binder error, not a silent miss), and ``_prime_seen`` already uses this
    idiom."""
    return (datetime.now(timezone.utc) - timedelta(days=int(days))).isoformat()


def run_think_arm(days: int = 7, limit: Optional[int] = None,
                  budget_seconds: Optional[float] = None) -> dict:
    """Replay the specialist with thinking ON over the live pass's rows.

    OFFLINE and NOT default: one call runs ~13× the live call, so this is CLI
    territory (``python -m src.analysis.catalyst_repair --think-arm``), never
    the tick path and never the EOD chain — it shares the single local server
    with the sentiment shadow pass, so run it when the scheduler is idle
    (weekends, or well outside the 04:00–20:00 ET tick window).

    Population = every live row of the last ``days`` days whose digest is still
    in ``sentiment_digests`` and which has no think row yet for the same
    ``(digest_id, engine)`` — so a killed run resumes where it stopped and the
    two arms line up on the SAME digests, the paired comparison ``evaluate``
    reads. Each row re-runs ``classify_digest`` on the stored digest text and
    the stored first-pass rationale (the exact live input), with the live row's
    first pass, override and trigger, and lands as ``arm='think'`` under its
    own idempotency key, so it can never overwrite the live row.
    Sequential, single worker: the server ceiling is the bottleneck.
    """
    from src.db import repo
    sql = f"""
        SELECT r.digest_id, r.run_id, r.ticker, r.engine, r.model, r.first_pass,
               r.trigger, r.final, r.quality, r.rationale, r.generated_at,
               d.digest_text
        FROM catalyst_repairs r
        JOIN sentiment_digests d ON d.digest_id = r.digest_id
        WHERE coalesce(r.arm, '{ARM_LIVE}') = '{ARM_LIVE}'
          AND r.generated_at >= ?
          AND NOT EXISTS (
              SELECT 1 FROM catalyst_repairs t
              WHERE t.digest_id = r.digest_id
                AND coalesce(t.engine, '') = coalesce(r.engine, '')
                AND t.arm = '{ARM_THINK}')
        ORDER BY r.generated_at, r.digest_id, r.engine
    """
    df = repo.fetch_df(sql, [_days_cutoff_iso(days)])
    summary = {"days": days, "selected": 0, "done": 0, "inserted": 0,
               "errors": 0, "elapsed_s": 0.0, "stopped": None}
    if df is None or df.empty:
        logger.info("[catalyst-repair] think arm: nothing to replay")
        return summary
    if limit:
        df = df.head(int(limit))
    summary["selected"] = int(len(df))
    logger.info(f"[catalyst-repair] think arm: {len(df)} live row(s) to replay "
                f"(last {days} days, thinking ON, {_THINK_MAX_TOKENS} output tokens)")
    t_start = time.perf_counter()
    batch: List[dict] = []

    def _flush() -> None:
        # A write that loses the lock race against the scheduler (the connect
        # already retries with backoff) keeps its rows for the next flush
        # point instead of dropping ~20 rows of GPU time; whatever is still
        # unwritten at the end is reported, and the next run re-selects it.
        if not batch:
            return
        try:
            repo.insert_catalyst_repairs(list(batch))
        except Exception as exc:                    # noqa: BLE001
            logger.warning(f"[catalyst-repair] think arm: write of {len(batch)} row(s) "
                           f"failed ({exc}); retrying at the next flush")
            return
        summary["inserted"] += len(batch)
        batch.clear()

    for i, r in enumerate(df.itertuples(index=False), 1):
        if budget_seconds and (time.perf_counter() - t_start) > float(budget_seconds):
            summary["stopped"] = "budget"
            logger.warning(f"[catalyst-repair] think arm: budget of {budget_seconds:.0f}s hit "
                           f"after {i - 1} row(s); rerun to resume")
            break
        first_pass = r.first_pass if isinstance(r.first_pass, str) else None
        # The live row's ``final`` is the override result only when it was an
        # override; otherwise the first pass is what the vote is judged against.
        final_label = r.final if (r.quality == "override" and isinstance(r.final, str)) else first_pass
        rationale = r.rationale if isinstance(r.rationale, str) else None
        trigger = r.trigger if isinstance(r.trigger, str) else None
        try:
            out = classify_digest(digest_id=r.digest_id, ticker=r.ticker,
                                  digest_text=r.digest_text or "", rationale=rationale,
                                  first_pass=first_pass, final_label=final_label,
                                  trigger=trigger, arm=ARM_THINK)
        except Exception as exc:                    # noqa: BLE001 - measurement only
            summary["errors"] += 1
            logger.debug(f"[catalyst-repair] think arm {r.ticker}: {exc}")
            continue
        if out.get("error"):
            summary["errors"] += 1
        batch.append({
            "digest_id": r.digest_id,
            "run_id": r.run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ticker": r.ticker,
            "engine": r.engine,
            "model": r.model,
            "first_pass": first_pass,
            "rationale": rationale,
            **out,
            "trigger": trigger,        # the live stamp, role suffix included
        })
        summary["done"] += 1
        if len(batch) >= _THINK_BATCH:
            _flush()
        if i % 10 == 0 or i == len(df):
            el = time.perf_counter() - t_start
            logger.info(f"[catalyst-repair] think arm: {i}/{len(df)} "
                        f"({el / i:.1f}s per row, {el:.0f}s elapsed)")
    _flush()
    if batch:
        summary["unwritten"] = len(batch)
        summary["stopped"] = summary["stopped"] or "write_failed"
    summary["elapsed_s"] = round(time.perf_counter() - t_start, 1)
    logger.info(f"[catalyst-repair] think arm: done {summary['done']}/{summary['selected']}, "
                f"inserted {summary['inserted']}, errors {summary['errors']}, "
                f"{summary['elapsed_s']}s")
    return summary


# ── Gold + evaluation ───────────────────────────────────────────────────────
def gold_label(judgment_verdict: Optional[str], judgment_correct: Optional[str],
               judged_model_catalyst: Optional[str], deepseek_label: Optional[str]) -> Optional[str]:
    """Gold for a digest: a judged correction wins (``err`` → the corrected
    class, ``ok`` → the judged model's own label, ``amb`` → no gold), else
    DeepSeek's label.

    Every argument arrives from a LEFT join against the judgments table, so an
    unjudged digest supplies pandas NaN — a float — not None. Coercing here is
    what lets `evaluate` run before a single judgment has been seeded, which is
    the state it is first read in.
    """
    def _s(x):
        return None if x is None or x != x else str(x)

    judgment_verdict, judgment_correct = _s(judgment_verdict), _s(judgment_correct)
    judged_model_catalyst, deepseek_label = _s(judged_model_catalyst), _s(deepseek_label)
    v = (judgment_verdict or "").strip().lower()
    if v == "err":
        return judgment_correct or None
    if v == "ok":
        return judged_model_catalyst or deepseek_label
    if v == "amb":
        return None
    return deepseek_label


def _halves(df, col="generated_at"):
    import pandas as pd
    if df.empty:
        return df, df
    ts = pd.to_datetime(df[col], utc=True, errors="coerce")
    mid = ts.sort_values().iloc[len(ts) // 2]
    return df[ts < mid], df[ts >= mid]


def _err_rate(a, b) -> Tuple[float, int]:
    import pandas as pd
    a = pd.Series(list(a), dtype="object")
    b = pd.Series(list(b), dtype="object")
    m = a.notna() & b.notna()
    if int(m.sum()) == 0:
        return float("nan"), 0
    return float((a[m] != b[m]).mean()), int(m.sum())


def _think_arm_block(live, think, pairs, judgments) -> dict:
    """Paired think-vs-live comparison on the ``(digest_id, engine)`` rows BOTH
    arms judged. Gold is the DeepSeek label for a local first pass (the pairs
    frame, judgments overriding) and the judgments alone for a DeepSeek first
    pass — the same gold ``evaluate`` uses for the live arm. Agreement between
    the arms is reported but is NOT skill; the error-vs-gold pair is."""
    import pandas as pd
    empty = {"n_pairs": 0}
    if think is None or think.empty or live is None or live.empty:
        return empty
    keep = ["digest_id", "engine", "final", "quality", "n_calls", "latency_s", "error", "generated_at"]
    lv = live[keep].sort_values("generated_at").drop_duplicates(["digest_id", "engine"], keep="last")
    th = think[keep + ["first_pass"]].sort_values("generated_at").drop_duplicates(
        ["digest_id", "engine"], keep="last")
    both = th.merge(lv, on=["digest_id", "engine"], suffixes=("_think", "_live"), how="inner")
    if both.empty:
        return empty
    # Gold: a judgment is digest-level truth (valid for either engine's row);
    # without one, DeepSeek's label is gold for a LOCAL first pass only — a
    # DeepSeek row cannot be graded against itself.
    judged_gold: Dict[str, Optional[str]] = {}
    if judgments is not None and not judgments.empty:
        for d, v, c, m in zip(judgments.digest_id, judgments.verdict,
                              judgments.correct_catalyst, judgments.model_catalyst):
            if isinstance(d, str):
                judged_gold[d] = gold_label(v, c, m, None)
    pair_gold: Dict[str, Optional[str]] = {}
    if pairs is not None and not pairs.empty and "gold" in pairs.columns:
        pair_gold = {d: g for d, g in zip(pairs.digest_id, pairs.gold) if isinstance(d, str)}
    both["gold"] = [judged_gold[d] if d in judged_gold
                    else (pair_gold.get(d) if e == "local" else None)
                    for d, e in zip(both.digest_id, both.engine)]
    judged = both[both.gold.notna()]
    res = {
        "n_pairs": int(len(both)),
        "agreement": float((both.final_think == both.final_live).mean()),
        "resolved_share": {"live": float((both.quality_live == "resolved").mean()),
                           "think": float((both.quality_think == "resolved").mean())},
        "mean_calls": {"live": float(pd.to_numeric(both.n_calls_live, errors="coerce").mean()),
                       "think": float(pd.to_numeric(both.n_calls_think, errors="coerce").mean())},
        "mean_latency_s": {"live": float(pd.to_numeric(both.latency_s_live, errors="coerce").mean()),
                           "think": float(pd.to_numeric(both.latency_s_think, errors="coerce").mean())},
        "error_share": {"live": float(both.error_live.notna().mean()),
                        "think": float(both.error_think.notna().mean())},
        "n_judged": int(len(judged)),
    }
    if not judged.empty:
        h1, h2 = _halves(judged, col="generated_at_live")
        res.update({
            "first_pass_err": _err_rate(judged.first_pass, judged.gold)[0],
            "live_err": _err_rate(judged.final_live, judged.gold)[0],
            "think_err": _err_rate(judged.final_think, judged.gold)[0],
            "halves_live_err": [_err_rate(h.final_live, h.gold)[0] for h in (h1, h2)],
            "halves_think_err": [_err_rate(h.final_think, h.gold)[0] for h in (h1, h2)],
            "by_engine": {k: {"n": int(len(v)),
                              "live_err": _err_rate(v.final_live, v.gold)[0],
                              "think_err": _err_rate(v.final_think, v.gold)[0]}
                          for k, v in judged.groupby("engine")},
        })
    return res


def _weekly_series(p) -> list:
    """Error rate per ISO week — the TRACKED SERIES the bar is read against.

    A single pooled number cannot tell "the specialist is working" from "the
    weeks it was measured on were easy": the first pass and the specialist are
    scored on the SAME rows each week, so the two curves move together when the
    digests get harder and apart only when the specialist earns it. Weeks are
    stamped from ``generated_at`` (UTC ISO week), judged rows only.
    """
    import pandas as pd
    if p is None or p.empty:
        return []
    g = p[p.gold.notna()].copy()
    if g.empty:
        return []
    ts = pd.to_datetime(g["generated_at"], utc=True, errors="coerce")
    iso = ts.dt.isocalendar()
    g["_week"] = iso.year.astype("Int64").astype(str) + "-W" + \
        iso.week.astype("Int64").astype(str).str.zfill(2)
    kind = g.trigger_kind.astype("string").fillna("")
    rows = []
    for week, w in g.groupby("_week"):
        flagged = w[kind.loc[w.index].isin(["unreliable_class", "keyword", "fund"])]
        rows.append({
            "week": str(week),
            "n": int(len(w)),
            "first_pass_err": _err_rate(w.shadow_catalyst, w.gold)[0],
            "specialist_err": _err_rate(w.final, w.gold)[0],
            "n_flagged": int(len(flagged)),
            "flagged_specialist_err": (_err_rate(flagged.final, flagged.gold)[0]
                                       if len(flagged) else None),
        })
    return sorted(rows, key=lambda r: r["week"])


def evaluate(days: int = 30) -> dict:
    """Pre-registered read of the repair pass (see the module docstring).
    Returns a dict and prints a report."""
    import pandas as pd
    from src.db import repo

    out: dict = {"days": days}
    # Baseline: the judged sample (first-pass error per engine / flagged classes).
    j = repo.fetch_df("SELECT * FROM catalyst_judgments")
    if j is not None and not j.empty:
        j["verdict"] = j["verdict"].str.lower()
        base = {}
        jj = j[j.verdict.isin(["ok", "err"])].copy()
        # Keyed the way the unreliable table is: by MODEL ID where the judgment
        # recorded one, by engine otherwise (a rate belongs to a checkpoint).
        if "model" in jj.columns:
            mdl = jj["model"].fillna("").astype(str).str.strip()
            jj["_key"] = mdl.where(mdl.str.len() > 0, jj["engine"].astype(str))
        else:
            jj["_key"] = jj["engine"].astype(str)
        for key, g in jj.groupby("_key"):
            eng = str(g["engine"].iloc[0])
            unrel = unreliable_classes(eng, None if key == eng else key)
            flagged = g[g.model_catalyst.isin(unrel)] if unrel else g.iloc[0:0]
            base[key] = {
                "engine": eng,
                "n": int(len(g)), "err": float((g.verdict == "err").mean()),
                "entity_err": float(g.entity_error.fillna(False).astype(bool).mean()),
                "flagged_n": int(len(flagged)),
                "flagged_err": (float((flagged.verdict == "err").mean()) if len(flagged) else float("nan")),
            }
        out["baseline"] = base
    # Pairs: a DeepSeek verdict and a local verdict on the SAME digest — in
    # EITHER role. The gold here is DeepSeek's LABEL, and a label does not care
    # which engine happened to drive the combine that run.
    #
    # This used to require `primary_engine='deepseek' AND shadow_engine='local'`,
    # which was true when it was written and became false on 2026-09-09 when
    # local took 100% of the primary route and DeepSeek became the shadow. The
    # population silently went to ZERO while 1,639 paired digests sat unused,
    # and the eval reported it as "no paired rows with a digest id yet" — a
    # message indistinguishable from "not accrued yet". Role-agnostic now, so a
    # future routing flip cannot repeat it.
    pairs = repo.fetch_df(f"""
        SELECT s.run_id, s.generated_at, s.ticker, s.digest_id,
               CASE WHEN s.primary_engine = 'deepseek' THEN s.primary_engine
                    ELSE s.shadow_engine END                      AS primary_engine,
               CASE WHEN s.primary_engine = 'deepseek' THEN s.primary_catalyst
                    ELSE s.shadow_catalyst END                    AS primary_catalyst,
               CASE WHEN s.primary_engine = 'deepseek' THEN s.shadow_engine
                    ELSE s.primary_engine END                     AS shadow_engine,
               CASE WHEN s.primary_engine = 'deepseek' THEN s.shadow_catalyst
                    ELSE s.primary_catalyst END                   AS shadow_catalyst,
               CASE WHEN s.primary_engine = 'deepseek' THEN s.primary_score
                    ELSE s.shadow_score END                       AS primary_score,
               CASE WHEN s.primary_engine = 'deepseek' THEN s.shadow_score
                    ELSE s.primary_score END                      AS shadow_score
        FROM sentiment_shadow s
        WHERE s.digest_id IS NOT NULL
          AND s.generated_at >= ?
          AND ((s.primary_engine = 'deepseek' AND s.shadow_engine = 'local')
            OR (s.primary_engine = 'local' AND s.shadow_engine = 'deepseek'))
    """, [_days_cutoff_iso(days)])
    # The main read is the LIVE arm only — the offline think arm is measured
    # against it in its own paired block below, never pooled with it.
    all_reps = repo.fetch_df(f"""
        SELECT digest_id, engine, first_pass, trigger, final, quality, n_calls,
               about_target, latency_s, error, generated_at, run_id,
               coalesce(arm, '{ARM_LIVE}') AS arm
        FROM catalyst_repairs
        WHERE generated_at >= ?
    """, [_days_cutoff_iso(days)])
    if all_reps is None or all_reps.empty:
        reps = all_reps
        think = None
    else:
        reps = all_reps[all_reps.arm == ARM_LIVE].copy()
        think = all_reps[all_reps.arm == ARM_THINK].copy()
    out["n_pairs"] = int(0 if pairs is None else len(pairs))
    out["n_repairs"] = int(0 if reps is None else len(reps))
    jg = (j[j.digest_id.notna()][["digest_id", "verdict", "correct_catalyst", "model_catalyst"]]
          .drop_duplicates("digest_id") if j is not None and not j.empty
          else pd.DataFrame(columns=["digest_id", "verdict", "correct_catalyst", "model_catalyst"]))
    if pairs is not None and not pairs.empty:
        p = pairs.merge(jg, on="digest_id", how="left")
        p["gold"] = [gold_label(v, c, m, d) for v, c, m, d in
                     zip(p.verdict, p.correct_catalyst, p.model_catalyst, p.primary_catalyst)]
    else:
        p = None
    out["think_arm"] = _think_arm_block(reps, think, p, jg)
    if p is None:
        _any = repo.fetch_df(
            "SELECT count(*) n FROM sentiment_shadow WHERE digest_id IS NOT NULL "
            "AND generated_at >= ?", [_days_cutoff_iso(days)])
        _n = int(_any.iloc[0]["n"]) if _any is not None and not _any.empty else 0
        out["status"] = (
            "no deepseek/local pairs in this window (the digest store began "
            f"2026-09-06); {_n} shadow row(s) WITH a digest id exist, so if that "
            "count is large the ENGINE PAIR is what is missing, not the data")
        _print_eval(out)
        return out
    local_reps = (reps[reps.engine == "local"] if reps is not None and not reps.empty
                  else pd.DataFrame(columns=["digest_id", "trigger", "final", "quality", "n_calls",
                                             "generated_at"]))
    if not local_reps.empty:
        local_reps = local_reps.sort_values("generated_at").drop_duplicates("digest_id", keep="last")
        p = p.merge(local_reps[["digest_id", "trigger", "final", "quality", "n_calls"]],
                    on="digest_id", how="left")
    else:
        for c in ("trigger", "final", "quality", "n_calls"):
            p[c] = None
    p["trigger_kind"] = p.trigger.astype("string").str.replace(r"@.*$", "", regex=True)
    judged = p[p.gold.notna()]
    fp_err, n_fp = _err_rate(judged.shadow_catalyst, judged.gold)
    out["local_first_pass"] = {"n": n_fp, "err_vs_gold": fp_err}
    # Flagged = the specialist actually ran with a non-monitor trigger.
    flagged = judged[judged.trigger_kind.isin(["unreliable_class", "keyword", "fund"])]
    monitor = judged[judged.trigger_kind == "monitor"]
    res = {}
    for name, g in (("flagged", flagged), ("monitor", monitor)):
        if g.empty:
            res[name] = {"n": 0}
            continue
        fp_e, n = _err_rate(g.shadow_catalyst, g.gold)
        sp_e, _ = _err_rate(g.final, g.gold)
        h1, h2 = _halves(g)
        res[name] = {
            "n": n, "first_pass_err": fp_e, "specialist_err": sp_e,
            "resolved_share": float((g.quality == "resolved").mean()),
            "halves_first_pass_err": [_err_rate(h.shadow_catalyst, h.gold)[0] for h in (h1, h2)],
            "halves_specialist_err": [_err_rate(h.final, h.gold)[0] for h in (h1, h2)],
            "by_trigger": {k: {"n": int(len(v)), "fp_err": _err_rate(v.shadow_catalyst, v.gold)[0],
                               "sp_err": _err_rate(v.final, v.gold)[0]}
                           for k, v in g.groupby("trigger_kind")},
        }
    out["local_specialist"] = res
    # The bar.
    f, m = res.get("flagged", {}), res.get("monitor", {})

    def _same_sign(r: dict, strict: bool) -> bool:
        pairs_ = list(zip(r.get("halves_specialist_err", []), r.get("halves_first_pass_err", [])))
        if len(pairs_) != 2:
            return False
        for a, b in pairs_:
            if a is None or b is None or a != a or b != b:
                return False
            if strict and not a < b:
                return False
            if not strict and not a <= b:
                return False
        return True

    bar = {
        "flagged_err_below_25": bool(f.get("n", 0) >= 30 and f.get("specialist_err", 1.0) < 0.25),
        "flagged_halves_same_sign": bool(f.get("n", 0) >= 30 and _same_sign(f, strict=True)),
        "monitor_not_worse": bool(m.get("n", 0) >= 30 and
                                  m.get("specialist_err", 1.0) <= m.get("first_pass_err", 0.0) + 1e-12),
        "monitor_halves_same_sign": bool(m.get("n", 0) >= 30 and _same_sign(m, strict=False)),
    }
    bar["PASS"] = all(bar.values())
    out["bar"] = bar
    out["series"] = _weekly_series(p)
    # DeepSeek side vs judgments only (no other gold exists for it).
    if reps is not None and not reps.empty and not jg.empty:
        dr = reps[reps.engine == "deepseek"].merge(jg, on="digest_id", how="inner")
        dr = dr[dr.verdict.str.lower().isin(["ok", "err"])]
        if not dr.empty:
            dr["gold"] = [gold_label(v, c, m, None) for v, c, m in
                          zip(dr.verdict, dr.correct_catalyst, dr.model_catalyst)]
            out["deepseek_specialist"] = {
                "n": int(len(dr)),
                "first_pass_err": _err_rate(dr.first_pass, dr.gold)[0],
                "specialist_err": _err_rate(dr.final, dr.gold)[0],
            }
    if reps is not None and not reps.empty:
        n_runs = int(reps["run_id"].dropna().nunique()) if "run_id" in reps.columns else 0
        out["repair_volume"] = {
            "rows": int(len(reps)),
            "runs": n_runs,
            "rows_per_run": (float(len(reps)) / n_runs if n_runs else float("nan")),
            "calls_per_run": (float(pd.to_numeric(reps.n_calls, errors="coerce").sum()) / n_runs
                              if n_runs else float("nan")),
            "by_quality": {k: int(v) for k, v in reps.quality.value_counts().items()},
            "by_trigger": {k: int(v) for k, v in reps.trigger.value_counts().items()},
            "mean_calls": float(pd.to_numeric(reps.n_calls, errors="coerce").mean()),
            "mean_latency_s": float(pd.to_numeric(reps.latency_s, errors="coerce").mean()),
            "error_share": float(reps.error.notna().mean()),
        }
    _print_eval(out)
    return out


def _pct(x) -> str:
    try:
        return "n/a" if x is None or x != x else f"{100 * float(x):.1f}%"
    except Exception:                               # noqa: BLE001
        return "n/a"


def _print_eval(out: dict) -> None:
    print(f"== Catalyst-repair eval - last {out.get('days')} days ==")
    for eng, b in (out.get("baseline") or {}).items():
        print(f"baseline judged {eng:16s} n={b['n']:4d}  err={_pct(b['err'])}  "
              f"entity={_pct(b['entity_err'])}  flagged-class n={b['flagged_n']} err={_pct(b['flagged_err'])}")
    if out.get("status"):
        print(out["status"])
        _print_think_arm(out.get("think_arm"))
        _print_series(out)
        return
    print(f"pairs (deepseek primary / local shadow, digest id): {out['n_pairs']}   "
          f"repair rows: {out['n_repairs']}")
    lf = out.get("local_first_pass") or {}
    print(f"local first pass vs gold: n={lf.get('n', 0)} err={_pct(lf.get('err_vs_gold'))}")
    for name, r in (out.get("local_specialist") or {}).items():
        if not r.get("n"):
            print(f"{name:8s}: no rows")
            continue
        print(f"{name:8s}: n={r['n']:4d}  first-pass err={_pct(r['first_pass_err'])}  "
              f"specialist err={_pct(r['specialist_err'])}  resolved={_pct(r['resolved_share'])}  "
              f"halves fp={[_pct(x) for x in r['halves_first_pass_err']]} "
              f"sp={[_pct(x) for x in r['halves_specialist_err']]}")
        for k, v in r.get("by_trigger", {}).items():
            print(f"           {k:16s} n={v['n']:4d} fp={_pct(v['fp_err'])} sp={_pct(v['sp_err'])}")
    ds = out.get("deepseek_specialist")
    if ds:
        print(f"deepseek side (judged only): n={ds['n']} first-pass err={_pct(ds['first_pass_err'])} "
              f"specialist err={_pct(ds['specialist_err'])}")
    rv = out.get("repair_volume")
    if rv:
        print(f"volume: {rv['rows']} rows over {rv['runs']} run(s) "
              f"({rv['rows_per_run']:.1f} rows / {rv['calls_per_run']:.1f} specialist calls per run)  "
              f"quality={rv['by_quality']}  trigger={rv['by_trigger']}  "
              f"calls/row={rv['mean_calls']:.2f}  latency={rv['mean_latency_s']:.1f}s  "
              f"error share={_pct(rv['error_share'])}")
    bar = out.get("bar") or {}
    if bar:
        print("BAR (flagged specialist err < 25%, monitor not worse, both halves same sign; "
              "n >= 30 each): " + ("PASS" if bar.get("PASS") else "NOT MET") +
              "  " + json.dumps({k: v for k, v in bar.items() if k != "PASS"}))
    _print_think_arm(out.get("think_arm"))
    _print_series(out)


def _print_think_arm(t: Optional[dict]) -> None:
    if not t or not t.get("n_pairs"):
        print("think arm: no paired rows (python -m src.analysis.catalyst_repair --think-arm)")
        return
    print(f"think arm (thinking ON, offline) vs live, paired on (digest, engine): n={t['n_pairs']}  "
          f"agreement={_pct(t['agreement'])}  "
          f"resolved live={_pct(t['resolved_share']['live'])} think={_pct(t['resolved_share']['think'])}  "
          f"calls/row live={t['mean_calls']['live']:.2f} think={t['mean_calls']['think']:.2f}  "
          f"latency live={t['mean_latency_s']['live']:.1f}s think={t['mean_latency_s']['think']:.1f}s  "
          f"error share live={_pct(t['error_share']['live'])} think={_pct(t['error_share']['think'])}")
    if t.get("n_judged"):
        print(f"           vs gold: n={t['n_judged']}  first-pass err={_pct(t['first_pass_err'])}  "
              f"live err={_pct(t['live_err'])}  think err={_pct(t['think_err'])}  "
              f"halves live={[_pct(x) for x in t['halves_live_err']]} "
              f"think={[_pct(x) for x in t['halves_think_err']]}")
        for k, v in t.get("by_engine", {}).items():
            print(f"           {k:10s} n={v['n']:4d} live={_pct(v['live_err'])} think={_pct(v['think_err'])}")
    else:
        print("           vs gold: no judged rows yet")


# ── the judged GOLD sample (~100 rows/week) ─────────────────────────
JUDGE_STRATA = ("flagged", "monitor")
_SHEET_VERSION = "sheet-v1"


def judge_sheet(n: int = 100, days: int = 7, out_path: Optional[str] = None,
                seed: Optional[int] = None) -> dict:
    """Write a fill-in review sheet of repair rows needing a human verdict.

    The bar in the module docstring is measured on TWO populations, so the
    sample is STRATIFIED and each stratum is drawn at RANDOM within itself:
    ``flagged`` (a trigger fired — where the specialist is supposed to earn its
    place) and ``monitor`` (the hash sample of untriggered verdicts — where it
    must not make things worse). Drawing the INTERESTING rows instead — the
    disagreements — would inflate both error rates and measure nothing, which
    is the whole reason the monitor sample exists.

    Rows already judged for that model are excluded, the draw is seeded on the
    ISO week so re-running the same week reproduces it, and every row carries
    what a human needs to decide (the headlines the scorer saw, its rationale,
    its label, the specialist's label and votes) plus the blank fields
    ``verdict`` / ``correct_catalyst`` / ``entity_error`` / ``reason``. The
    schema is exactly what ``seed_judgments`` ingests, so the sheet is filled in
    place and loaded back with ``--seed-judgments <path>``; blank verdicts are
    skipped there, so a partly-filled sheet is safe to load and re-load.
    """
    import os

    import pandas as pd

    from src.analysis.sentiment import NEWS_CATALYST_TYPES
    from src.db import repo
    out_path = out_path or (f"config/seeds/catalyst_judgments_"
                            f"{datetime.now(timezone.utc):%Y-%m-%d}.json")
    sql = f"""
        SELECT r.digest_id, r.run_id, r.ticker, r.engine, r.model, r.first_pass,
               r.trigger, r.final, r.quality, r.votes, r.rationale, r.generated_at,
               d.digest_text
        FROM catalyst_repairs r
        LEFT JOIN sentiment_digests d ON d.digest_id = r.digest_id
        WHERE coalesce(r.arm, '{ARM_LIVE}') = '{ARM_LIVE}'
          AND r.digest_id IS NOT NULL
          AND r.generated_at >= ?
          AND NOT EXISTS (SELECT 1 FROM catalyst_judgments j
                          WHERE j.digest_id = r.digest_id
                            AND coalesce(j.model, '') = coalesce(r.model, ''))
        ORDER BY r.generated_at
    """
    df = repo.fetch_df(sql, [_days_cutoff_iso(days)])
    summary = {"days": days, "requested": int(n), "written": 0, "path": out_path,
               "by_stratum": {}}
    if df is None or df.empty:
        logger.info("[catalyst-repair] judge sheet: no unjudged repair rows in the window")
        return summary
    kind = df.trigger.astype("string").str.replace(r"@.*$", "", regex=True).fillna("")
    df = df.assign(stratum=pd.Series(["monitor" if k == "monitor" else "flagged"
                                      for k in kind], index=df.index))
    rng = random.Random(seed if seed is not None
                        else datetime.now(timezone.utc).strftime("%Y-%W"))
    pools = {k: df[df.stratum == k].to_dict("records") for k in JUDGE_STRATA}
    for k in JUDGE_STRATA:
        rng.shuffle(pools[k])
    # Half the sheet to each stratum, and whatever one stratum cannot fill goes
    # to the other — a thin week should still spend its whole budget.
    want = {"flagged": n // 2, "monitor": n - n // 2}
    for k in JUDGE_STRATA:
        other = [x for x in JUDGE_STRATA if x != k][0]
        want[k] += max(0, want[other] - len(pools[other]))
    picks: List[dict] = []
    for k in JUDGE_STRATA:
        take = pools[k][:want[k]]
        summary["by_stratum"][k] = len(take)
        picks.extend(take)
    rows = []
    for r in picks:
        try:
            votes = [v.get("catalyst") for v in json.loads(r.get("votes") or "[]")]
        except Exception:                               # noqa: BLE001
            votes = []
        rows.append({
            # what `seed_judgments` reads:
            "judgment_id": f"{r['digest_id']}|{r.get('model') or r.get('engine')}",
            "judged_at": None,                          # stamped on load
            "source": "human",
            "engine": r.get("engine"),
            "model": r.get("model"),
            "ticker": r.get("ticker"),
            "run_id": r.get("run_id"),
            "digest_id": r.get("digest_id"),
            "model_catalyst": r.get("first_pass"),
            "verdict": "",                              # ok | err | amb  <- FILL IN
            "correct_catalyst": "",                     # required when verdict = err
            "entity_error": None,                       # true = the label belongs to another entity
            "reason": "",
            "rationale": r.get("rationale"),
            # context for the human, ignored by the loader:
            "_stratum": r.get("stratum"),
            "_trigger": r.get("trigger"),
            "_specialist": r.get("final"),
            "_specialist_quality": r.get("quality"),
            "_specialist_votes": votes,
            "_generated_at": r.get("generated_at"),
            "_digest": (r.get("digest_text") or "")[:4000],
        })
    payload = {
        "_sheet": _SHEET_VERSION,
        "_written_at": datetime.now(timezone.utc).isoformat(),
        "_instructions": ("Fill `verdict` with ok | err | amb (blank rows are skipped on load). "
                          "On `err` set `correct_catalyst` to one of the classes below; set "
                          "`entity_error` true when the label describes a different company (an "
                          "ETF typed with a holding's event). Load with: python -m "
                          "src.analysis.catalyst_repair --seed-judgments <this file>"),
        "_classes": list(NEWS_CATALYST_TYPES),
        "rows": rows,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    summary["written"] = len(rows)
    logger.info(f"[catalyst-repair] judge sheet: {len(rows)} row(s) "
                f"({summary['by_stratum']}) -> {out_path}")
    return summary


def _print_series(out: dict) -> None:
    series = out.get("series") or []
    if not series:
        print("weekly    : no judged rows yet - run --judge-sheet, fill it in, "
              "--seed-judgments it back")
        return
    print("weekly    : week      n   first-pass  specialist   flagged n / err")
    for r in series:
        print(f"            {r['week']:8s} {r['n']:4d}  {_pct(r['first_pass_err']):>9s}  "
              f"{_pct(r['specialist_err']):>10s}   {r['n_flagged']:4d} / "
              f"{_pct(r['flagged_specialist_err'])}")


def seed_judgments(path: str) -> int:
    """Load a judged-label file (a filled ``judge_sheet``, or any file with the
    same row schema) into ``catalyst_judgments``.

    A row with a BLANK verdict is not a judgment — it is a sheet line nobody
    reached — so it is skipped rather than stored, which is what makes a
    partly-filled sheet safe to load and re-load as it is worked through. An
    ``err`` row whose ``correct_catalyst`` is missing or outside the taxonomy is
    skipped too and NAMED in the log: a typo there becomes a wrong GOLD label,
    and a wrong gold is worse than a missing one — it moves every error rate
    measured against it.
    """
    from src.analysis.sentiment import NEWS_CATALYST_TYPES
    from src.db import repo
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)
    if isinstance(rows, dict):
        rows = rows.get("rows") or []
    keep, skipped, bad = [], 0, []
    for r in rows:
        verdict = str(r.get("verdict") or "").strip().lower()
        if verdict not in ("ok", "err", "amb"):
            skipped += 1
            continue
        correct = str(r.get("correct_catalyst") or "").strip().lower()
        if verdict == "err" and correct not in NEWS_CATALYST_TYPES:
            bad.append(f"{r.get('digest_id')}:{correct or '(blank)'}")
            continue
        row = {k: v for k, v in r.items() if not str(k).startswith("_")}
        row["verdict"], row["correct_catalyst"] = verdict, (correct or None)
        row.setdefault("source", "human")
        if not row.get("judged_at"):
            row["judged_at"] = datetime.now(timezone.utc).isoformat()
        keep.append(row)
    repo.insert_catalyst_judgments(keep)
    msg = f"[catalyst-repair] seeded {len(keep)} judgments from {path}"
    if skipped:
        msg += f" ({skipped} unjudged row(s) skipped)"
    if bad:
        msg += (f"; DROPPED {len(bad)} err row(s) with an unusable correct_catalyst: "
                f"{bad[:5]}")
    logger.info(msg)
    return len(keep)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Catalyst-label repair pass: evaluation + seeding")
    ap.add_argument("--eval", action="store_true", help="pre-registered read vs gold")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--seed-judgments", nargs="?", const="config/seeds/catalyst_judgments_2026-09-05.json",
                    help="load a judged-label seed file into catalyst_judgments")
    ap.add_argument("--judge-sheet", action="store_true",
                    help="write a stratified fill-in review sheet of unjudged repair rows")
    ap.add_argument("--n", type=int, default=100, help="--judge-sheet: how many rows (default 100)")
    ap.add_argument("--out", default=None, help="--judge-sheet: output path")
    ap.add_argument("--probe", metavar="TICKER", help="run the specialist once on a synthetic digest")
    ap.add_argument("--think", action="store_true",
                    help="with --probe: run the probe on the thinking-on arm")
    ap.add_argument("--think-arm", action="store_true",
                    help="OFFLINE: replay the live rows of the last --days days with thinking ON "
                         "(arm='think'; resumable; run when the scheduler is idle)")
    ap.add_argument("--limit", type=int, default=None, help="--think-arm: at most N rows")
    ap.add_argument("--budget-seconds", type=float, default=None,
                    help="--think-arm: stop after this many seconds (rerun resumes)")
    args = ap.parse_args(argv)
    if args.judge_sheet:
        print(json.dumps(judge_sheet(n=args.n, days=args.days, out_path=args.out), indent=2))
    if args.seed_judgments:
        n = seed_judgments(args.seed_judgments)
        print(f"seeded {n} judgments")
    if args.probe:
        digest = ("[Reuters | 3h ago] Board approves new $500 million share repurchase program\n"
                  "The company said its board authorised a buyback of up to $500 million.")
        out = classify_digest(digest_id="probe", ticker=args.probe, digest_text=digest,
                              rationale="Buyback authorisation is a mild positive.",
                              first_pass="product", trigger="probe",
                              arm=(ARM_THINK if args.think else ARM_LIVE))
        print(json.dumps(out, indent=2))
    if args.think_arm:
        print(json.dumps(run_think_arm(days=args.days, limit=args.limit,
                                       budget_seconds=args.budget_seconds), indent=2))
    if args.eval or not (args.seed_judgments or args.probe or args.think_arm
                         or args.judge_sheet):
        evaluate(days=args.days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
