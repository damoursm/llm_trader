"""Ticker → company name, and "does this text mention the company?" — the
relevance test the per-ticker news digest is built on (2026-09-03).

Why this exists. The sentiment scorer's abstention rate was measured at 73–79%
of LLM calls, and a rationale audit of 693 zero verdicts found ~74% of them
saying, correctly, that the digest was about OTHER companies. Two input defects
produced those digests: the yfinance per-ticker feed is Yahoo's *related* news
(tagged with the symbol whether or not the article is about it), and the
fallback keyword test was a bare SUBSTRING match on the lowercased symbol —
``"ar" in text`` for Antero Resources matches nearly every article ever written,
so a wide-open pool got capped to the 20 most recent random headlines and the
model was asked to find Antero in them. Only 32% of Yahoo's items mention the
symbol at all; articles say "Disney", not "DIS".

What this module provides:

* ``company_name(ticker)`` — the SEC ``company_tickers.json`` registrant title
  (one bulk download, disk-cached, ~11k names) with a per-ticker Polygon
  reference fallback (ETFs and foreign lines the SEC list lacks). Negative
  results are cached too, so an unknown symbol costs one call a week.
* ``mention_evidence(ticker, text)`` — the STRONGEST evidence that *text* is
  about *ticker*, or ``None``. Evidence, in order:
    ``symbol_explicit``  ``(AR)``, ``$AR``, ``NYSE: AR`` — unambiguous however
                         short the symbol;
    ``name_phrase``      the normalised name as a whole phrase ("general
                         dynamics"), the same name with its trailing generic
                         words dropped ("regeneron" from "Regeneron
                         Pharmaceuticals", "super micro" from "Super Micro
                         Computer"), or a curated alias ("jpmorgan",
                         "s&p 500"). A ONE-word remainder must be ≥ 4 letters,
                         not an ordinary word (the stoplist: "Target", "Visa",
                         "Shell" → alias map or symbol only) and not shared by
                         ≥ 6 SEC registrants ("Strategy", "General");
    ``symbol_word``      the symbol as a case-sensitive whole word, only for
                         symbols ≥ 3 letters that are not common acronyms
                         (``ticker_extract._STOPWORDS``: AI, EV, ALL, …) and
                         only outside all-caps headlines;
    ``name_token``       one DISTINCTIVE token of a multi-word name, written
                         Capitalised ("Disney", "Fargo", "Palantir") — ≥ 6
                         letters and neither generic nor an ordinary word.
                         Weakest tier, so ``allow_token=False`` (the default)
                         withholds it; the search-feed fetchers enable it to
                         CONFIRM a "related" tag, the general pool never uses it.
  Matching is whole-word, so "AR" never fires inside "market" and "lilly"
  never inside "rally".

The relevance filter (``sentiment.filter_relevant_articles``) accepts an
article when a feed tagged it (structured feeds tag what they are about;
the search-derived feeds tag only what a mention confirms) or when an untagged
general-pool article carries a mention. That last path is what finally
associates company-issued press releases — the most ticker-specific news there
is — with the names they are about.

Fail-soft throughout: no name ⇒ symbol evidence only; every network failure
degrades to "unknown", never raises.
"""
from __future__ import annotations

import json
import re
import threading
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Set

import httpx
from loguru import logger

from src.data.cache import CACHE_DIR
from src.data.ticker_extract import _STOPWORDS as _ACRONYM_STOPWORDS

_CACHE_PATH = CACHE_DIR / "company_names.json"
_SEC_URL = "https://www.sec.gov/files/company_tickers.json"
_SEC_HEADERS = {"User-Agent": "llm-trader research@example.com"}
_SEC_REFRESH_DAYS = 30      # the registrant list changes slowly
_NEGATIVE_TTL_DAYS = 7      # an unknown symbol is re-asked weekly, not per tick
_INDUSTRY_TTL_DAYS = 180    # a known industry line is re-asked twice a year
_GENERIC_TOKEN_MIN_REGISTRANTS = 6   # a name token shared this widely is not identifying

_lock = threading.RLock()
_state: Optional[dict] = None          # {"sec_loaded_at", "names": {SYM: {...}}, "industry": {SYM: {...}}}
_token_freq: Optional[Counter] = None  # token → number of SEC registrants using it
_keyword_cache: Dict[str, dict] = {}
_pattern_cache: Dict[str, dict] = {}

# Corporate boilerplate stripped from the END of a name, repeatedly.
_SUFFIX_TOKENS = frozenset({
    "inc", "incorporated", "corp", "corporation", "co", "company", "cos", "ltd",
    "limited", "plc", "llc", "lp", "l.p", "nv", "n.v", "sa", "s.a", "ag", "se",
    "ab", "asa", "oyj", "spa", "s.p.a", "bv", "b.v", "kk", "pty", "pbc",
    "holdings", "holding", "hldgs", "group", "grp", "trust", "etf", "fund",
    "common", "stock", "shares", "ordinary", "adr", "ads", "depositary", "units",
    "class", "cl", "a", "b", "c", "series", "1", "2", "i", "ii", "iii",
    "the", "&", "and", "of", "de", "reit", "bancorp", "bancshares", "new",
})
# Words that are ORDINARY in financial-news prose (or nationalities / places /
# fund-issuer names). A name token in this set is never evidence on its own;
# a multi-word PHRASE containing it still is ("option care", "range resources").
_ENGLISH_STOPLIST = frozenset("""
about above access account action active advance advanced advantage agile alliance alpha
alternative amber american analog anchor apex applied arch arena argo array arrow ascent asset
assets atlas atomic aurora avenue axis balance beacon beam best beta beyond black block blue
bold bolt boost bridge bright brilliant broad brown builder builders cable capital capitol
carbon care career carrier catalyst central century champion charter chase check choice circle
citizens city civic classic clean clear cloud coast coastal cobalt columbia comfort commerce
appalachian arctic atlantic baltic bakken gulf hudson marcellus nano nordic permian rio sierra
common community compass concept concrete connect consumer continental control core corner
country crown crystal cure custom cycle dawn deal delta delivery diamond direct discover
discovery dollar domain dominion door dream drive dynamic dynamics eagle earth east eastern edge
electric element elite ember emerald empire energy engine enterprise entertainment envoy epic
equity essential ever evolve exact excel express extra fair faith falcon family federal fidelity
first five flag flex focus forge fortune forward foundation fountain freedom fresh front frontier
fuel fusion future galaxy gap garden gate gateway general genesis giant global globe gold golden
good grand granite great green guardian harbor harmony harvest health heritage hero highland
home horizon hub icon ideal image impact independence infinity insight integrity iron island
jet jewel journey joy key keystone kind kinetic knight lake landmark leader legacy legend liberty
life light lighthouse lincoln lines link lion live logic lumen luna magic main major marathon
marine mark market master match matrix meridian merit metro micro mid midland mind modern
momentum monarch mosaic motion mountain national native natural nature navigator new next noble
north northern nova ocean omega open optimum orbit origin pacific paramount park partner
partners patriot peak pearl phoenix pilot pinnacle pioneer planet platinum plus polar portal
power precision premier prime prism pro progress progressive prospect pulse pure quantum quest
radiant rally range rapid ray real reliance renaissance republic resource rise river rocket
root royal safe sage sail secure select sentinel service shell shield shift signal silver simple
sky smart snap solar solid sound source south southern spark spectrum sphere spirit spring
square standard star state sterling stone strategic strong summit sun super superior swift
target terra titan total tower trade tradition trail trend trinity triumph true trust ultra
union unique united unity universal urban valley value vanguard vector velocity venture vertex
victory view visa vision vista vital voyager wave west western white wild wind wise world zenith
zero zoom
option options performance restaurant restaurants principal communication communications
staples discretionary semiconductor semiconductors genomics genetics famous cruise towers fisher
rhythm tortoise constellation protagonist trilogy factor sector sectors strategy strategies
platforms platform computer computers twist slide oscar relay riot shake materials financial
financials industrial industrials industries technology technologies therapeutics
pharmaceutical pharmaceuticals biosciences bioscience sciences science medical healthcare
systems solutions networks network software digital resources petroleum minerals mining metals
holdings insurance realty properties brands foods motors airlines airline interactive
international worldwide services products devices instruments automotive aerospace defense
security logistics shipping offshore midstream infrastructure environmental scientific research
development innovation innovations ventures growth income quality select exploration production
acquisition acquisitions corporation limited water nuclear renewable organic retail stores
markets trading exchange payments lending credit mortgage banking wealth management advisors
consulting media broadcasting publishing gaming sports fitness leisure travel hospitality
beverage tobacco cannabis apparel fashion beauty luxury jewelry furniture housing homes building
construction engineering machinery equipment chemical chemicals plastics paper packaging steel
aluminum copper lithium uranium precious homebuilders bancorporation savings federal
ishares spdr vaneck invesco proshares direxion wisdomtree schwab
norwegian canadian british german french chinese japanese korean indian australian european
mexican brazilian swiss dutch italian spanish irish israeli taiwanese texas california florida
boston chicago atlanta houston dallas denver seattle virginia carolina georgia arizona nevada
ohio michigan jersey oregon colorado atlantic
""".split())


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


def _parse_iso(s) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(s))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Disk cache + sources
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    global _state
    if _state is not None:
        return _state
    with _lock:
        if _state is not None:
            return _state
        st: dict = {"sec_loaded_at": None, "names": {}, "industry": {}}
        try:
            if _CACHE_PATH.exists():
                raw = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and isinstance(raw.get("names"), dict):
                    ind = raw.get("industry")
                    st = {"sec_loaded_at": raw.get("sec_loaded_at"),
                          "names": raw["names"],
                          "industry": ind if isinstance(ind, dict) else {}}
        except Exception as exc:                       # corrupt cache → rebuild
            logger.warning(f"[company_names] cache unreadable, rebuilding: {exc}")
        _state = st
        return _state


def _save_state() -> None:
    st = _state
    if st is None:
        return
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        tmp = _CACHE_PATH.with_suffix(f".tmp{threading.get_ident()}")
        tmp.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
        tmp.replace(_CACHE_PATH)
    except Exception as exc:
        logger.debug(f"[company_names] cache write failed: {exc}")


def _sec_stale(st: dict) -> bool:
    loaded = _parse_iso(st.get("sec_loaded_at"))
    return loaded is None or (_now() - loaded) > timedelta(days=_SEC_REFRESH_DAYS)


def _refresh_sec(st: dict) -> None:
    """One bulk download of the SEC registrant list → every symbol's title.
    Class shares arrive as ``BRK-B``; stored as-is AND under the dotted form so
    either spelling resolves."""
    try:
        resp = httpx.get(_SEC_URL, headers=_SEC_HEADERS, timeout=30)
        resp.raise_for_status()
        n = 0
        ts = _iso(_now())
        for entry in resp.json().values():
            sym = str(entry.get("ticker") or "").strip().upper()
            title = str(entry.get("title") or "").strip()
            if not sym or not title:
                continue
            for key in {sym, sym.replace("-", "."), sym.replace(".", "-")}:
                st["names"][key] = {"name": title, "source": "sec", "ts": ts}
            n += 1
        st["sec_loaded_at"] = ts
        logger.info(f"[company_names] SEC registrant list: {n:,} names")
    except Exception as exc:
        logger.warning(f"[company_names] SEC list unavailable: {exc}")
        # Don't hammer SEC every tick on an outage: back off a day.
        st["sec_loaded_at"] = _iso(_now() - timedelta(days=_SEC_REFRESH_DAYS - 1))


def _polygon_name(ticker: str) -> Optional[str]:
    try:
        from src.data import polygon_client as pc
        j = pc._get(f"/v3/reference/tickers/{pc.to_polygon_symbol(ticker)}", {})
        name = ((j or {}).get("results") or {}).get("name")
        return (str(name).strip() or None) if name else None
    except Exception as exc:
        logger.debug(f"[company_names] polygon lookup failed for {ticker}: {exc}")
        return None


def company_name(ticker: str) -> Optional[str]:
    """Registrant/issuer name for *ticker*, or ``None`` when no source knows it.
    SEC first (bulk), Polygon reference second (per symbol, cached, negatives
    re-asked after ``_NEGATIVE_TTL_DAYS``)."""
    sym = (ticker or "").strip().upper()
    if not sym:
        return None
    st = _load_state()
    with _lock:
        if _sec_stale(st):
            _refresh_sec(st)
            _token_freq_reset()
            _save_state()
        rec = st["names"].get(sym)
        if rec is not None:
            if rec.get("name"):
                return rec["name"]
            ts = _parse_iso(rec.get("ts"))
            if ts is not None and (_now() - ts) < timedelta(days=_NEGATIVE_TTL_DAYS):
                return None                      # known-unknown, still fresh
        name = _polygon_name(sym)
        st["names"][sym] = {"name": name, "source": "polygon" if name else "none",
                            "ts": _iso(_now())}
        _save_state()
        return name


def _token_freq_reset() -> None:
    global _token_freq
    _token_freq = None
    _keyword_cache.clear()
    _pattern_cache.clear()


def _registrant_token_freq() -> Counter:
    """How many SEC registrants use each normalised name token — the
    self-maintaining "generic word" test ("general" 14, "therapeutics" 202,
    "disney" 1)."""
    global _token_freq
    if _token_freq is not None:
        return _token_freq
    st = _load_state()
    freq: Counter = Counter()
    seen: Set[str] = set()
    for rec in st["names"].values():
        name = (rec or {}).get("name")
        if not name or (rec or {}).get("source") != "sec" or name in seen:
            continue
        seen.add(name)
        freq.update(set(_name_tokens(name)))
    _token_freq = freq
    return freq


# ---------------------------------------------------------------------------
# Name normalisation → keywords
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'&.-]*")
_INTERIOR_NOISE = frozenset({"inc", "corp", "corporation", "co", "ltd", "plc", "llc", "&", "and"})


def _name_tokens(name: str) -> List[str]:
    """Lowercase tokens with the SEC's state/ADR markers and trailing corporate
    boilerplate removed: ``"The Walt Disney Company"`` → ``["walt", "disney"]``;
    ``"ELI LILLY & Co"`` → ``["eli", "lilly"]``; ``"US BANCORP \\DE\\"`` →
    ``["us"]``; ``"AT&T Inc"`` → ``["at&t"]``; ``"Hut 8 Corp."`` → ``["hut", "8"]``."""
    text = re.split(r"[/\\]", (name or "").lower(), maxsplit=1)[0]
    toks = [t.strip(".,'-&") for t in _WORD_RE.findall(text)]
    toks = [t for t in toks if t]
    if toks and toks[0] == "the":
        toks = toks[1:]
    while len(toks) > 1 and toks[-1] in _SUFFIX_TOKENS:
        toks.pop()
    if len(toks) > 1:
        core = [t for t in toks if t not in _INTERIOR_NOISE]
        if core:
            toks = core
    return toks


# Curated aliases: nicknames and product/brand names articles use instead of the
# registrant title, plus the sector phrases that make ETF coverage matchable.
# All are PHRASE-strength evidence, so keep them specific.
TICKER_ALIASES: Dict[str, List[str]] = {
    "AAPL": ["apple", "iphone", "ipad"],
    "MSFT": ["microsoft", "azure", "copilot"],
    "NVDA": ["nvidia", "jensen huang"],
    "TSLA": ["tesla", "elon musk"],
    "AMZN": ["amazon", "aws"],
    "META": ["meta platforms", "facebook", "instagram", "whatsapp", "zuckerberg"],
    "GOOGL": ["google", "alphabet", "youtube", "waymo"],
    "GOOG": ["google", "alphabet", "youtube", "waymo"],
    "NFLX": ["netflix"], "ORCL": ["oracle"], "AMD": ["advanced micro devices"],
    "INTC": ["intel"], "CRM": ["salesforce"], "ADBE": ["adobe"],
    "PYPL": ["paypal"], "UBER": ["uber"], "LYFT": ["lyft"],
    "JPM": ["jpmorgan", "jp morgan", "jamie dimon"],
    "BAC": ["bank of america"], "GS": ["goldman sachs", "goldman"],
    "MS": ["morgan stanley"], "WFC": ["wells fargo"], "C": ["citigroup", "citi"],
    "USB": ["u.s. bancorp", "us bancorp", "u.s. bank"],
    "BRK-B": ["berkshire hathaway", "berkshire", "warren buffett"],
    "BRK.B": ["berkshire hathaway", "berkshire", "warren buffett"],
    "DIS": ["disney"], "KO": ["coca-cola"], "PEP": ["pepsico", "pepsi"],
    "MCD": ["mcdonald's", "mcdonalds"], "SBUX": ["starbucks"], "NKE": ["nike"],
    "WMT": ["walmart"], "TGT": ["target corp", "target corporation", "target stores"],
    "HD": ["home depot"], "LOW": ["lowe's", "lowes"], "COST": ["costco"],
    "BA": ["boeing"], "LMT": ["lockheed martin", "lockheed"], "RTX": ["raytheon"],
    "GE": ["ge aerospace", "general electric"], "F": ["ford motor", "ford"],
    "GM": ["general motors"], "XOM": ["exxon mobil", "exxon", "exxonmobil"],
    "CVX": ["chevron"], "T": ["at&t"], "VZ": ["verizon"], "TMUS": ["t-mobile"],
    "V": ["visa inc", "visa card", "visa and mastercard", "mastercard and visa"],
    "MA": ["mastercard"], "AXP": ["american express", "amex"],
    "SHEL": ["shell plc", "royal dutch shell"], "ZM": ["zoom video", "zoom communications"],
    "DAL": ["delta air lines", "delta airlines"], "UAL": ["united airlines"],
    "AAL": ["american airlines"], "LUV": ["southwest airlines"],
    "CAT": ["caterpillar"], "DE": ["john deere", "deere"], "MMM": ["3m"],
    "MSTR": ["microstrategy", "strategy inc", "michael saylor"],
    "SLB": ["schlumberger"], "ROOT": ["root insurance"], "VFC": ["vf corp", "v.f. corp"],
    "UNH": ["unitedhealth"], "JNJ": ["johnson & johnson", "johnson and johnson"],
    "PFE": ["pfizer"], "MRK": ["merck"], "LLY": ["eli lilly", "lilly"],
    "ABBV": ["abbvie"], "BMY": ["bristol myers", "bristol-myers"],
    "AVGO": ["broadcom"], "QCOM": ["qualcomm"], "TXN": ["texas instruments"],
    "MU": ["micron"], "TSM": ["tsmc", "taiwan semiconductor"], "ASML": ["asml"],
    "PLTR": ["palantir"], "COIN": ["coinbase"], "HOOD": ["robinhood"],
    "SQ": ["block inc", "square"], "XYZ": ["block inc", "square"],
    "SNAP": ["snap inc", "snapchat"], "MTCH": ["match group", "tinder", "hinge"],
    "GPS": ["gap inc", "old navy", "banana republic"],
    "SPY": ["s&p 500", "sp500", "s&p500"], "QQQ": ["nasdaq-100", "nasdaq 100"],
    "XLK": ["technology sector", "tech sector", "tech etf"],
    "XLF": ["financials", "financial sector", "bank stocks", "banks"],
    "XLE": ["energy sector", "oil prices", "crude oil", "exxon", "chevron"],
    "XLV": ["health care sector", "healthcare sector", "biotech", "pharma"],
    "XLY": ["consumer discretionary"], "XLP": ["consumer staples"],
    "XLI": ["industrials"], "XLB": ["materials sector", "basic materials"],
    "XLU": ["utilities"], "XLRE": ["real estate sector", "reit"],
    "XLC": ["communication services"],
    "GDX": ["gold miners", "gold price", "gold prices"],
    "GLD": ["gold price", "gold prices", "spdr gold"],
    "SLV": ["silver price", "silver prices"],
    "IBB": ["biotech"], "XBI": ["biotech"], "SMH": ["semiconductor", "chip stocks"],
    "SOXX": ["semiconductor", "chip stocks"], "ITA": ["defense stocks", "defense contractors"],
    "JETS": ["airline stocks", "airlines"], "KRE": ["regional banks", "regional bank"],
    "LIT": ["lithium"], "TAN": ["solar stocks", "solar energy"],
    "XHB": ["homebuilders", "homebuilder"], "IGV": ["software stocks"],
    "ARKK": ["ark innovation", "cathie wood"], "IWM": ["russell 2000", "small caps", "small-cap"],
    "DIA": ["dow jones"], "TLT": ["treasury yields", "long bond", "20-year treasury"],
    "USO": ["crude oil", "oil prices"], "UNG": ["natural gas"],
    "IBIT": ["bitcoin"], "BITO": ["bitcoin"], "ETHA": ["ethereum"],
}


def _symbol_forms(sym: str) -> Set[str]:
    """Spellings of a class-share symbol as articles print them:
    ``BRK-B`` → {BRK-B, BRK.B, BRK B}."""
    forms = {sym}
    if "-" in sym or "." in sym:
        base = sym.replace(".", "-")
        forms.update({base, base.replace("-", "."), base.replace("-", " ")})
    return forms


def _letters(tok: str) -> str:
    return re.sub(r"[^a-z]", "", tok)


def _generic(tok: str, freq: Counter) -> bool:
    return (tok in _ENGLISH_STOPLIST or tok in _SUFFIX_TOKENS
            or freq.get(tok, 0) >= _GENERIC_TOKEN_MIN_REGISTRANTS)


def _distinctive(tok: str, min_letters: int, freq: Counter) -> bool:
    """A token that identifies a company on its own: long enough, not an
    ordinary word, not shared across many registrants."""
    return (len(_letters(tok)) >= min_letters
            and re.fullmatch(r"[a-z0-9&.'-]+", tok) is not None
            and not _generic(tok, freq))


def name_keywords(ticker: str) -> dict:
    """Evidence vocabulary for *ticker*:
    ``{"phrases": [...], "tokens": [...], "symbol_word": bool, "name": str|None,
    "fund": bool}``.

    * ``phrases`` — whole-phrase evidence: the normalised name, the name with
      trailing generic words dropped (stopping at a single ordinary word),
      and every curated alias.
    * ``tokens`` — single distinctive tokens of a multi-word name (the
      Capitalised-mention tier, confirmation only).
    * ``symbol_word`` — whether the bare symbol as a case-sensitive word counts.
    * ``fund`` — the registrant name reads as an ETF/trust/fund, i.e. the
      "company" behind the symbol is what it holds (the sentiment prompt
      tells the scorer so; ``False`` when no name is known).
    """
    sym = (ticker or "").strip().upper()
    if not sym:
        return {"phrases": [], "tokens": [], "symbol_word": False, "name": None, "fund": False}
    hit = _keyword_cache.get(sym)
    if hit is not None:
        return hit
    name = company_name(sym)
    toks = _name_tokens(name) if name else []
    freq = _registrant_token_freq()
    phrases: List[str] = []
    tokens: List[str] = []

    # A fund's name is its issuer + a theme ("State Street Energy Select Sector
    # SPDR"): dropping the theme would leave the ISSUER, which identifies
    # nothing, so funds keep only the full phrase and the alias map.
    raw = set(_WORD_RE.findall((name or "").lower()))
    is_fund = bool(raw & {"etf", "trust", "fund", "index", "portfolio", "ishares", "spdr"})

    def _add_phrase(parts: List[str]) -> None:
        if not parts:
            return
        if len(parts) == 1 and not _distinctive(parts[0], 4, freq):
            return
        if sum(len(re.sub(r"[^a-z0-9]", "", p)) for p in parts) < 4:   # "v f"
            return
        ph = " ".join(parts)
        if ph not in phrases:
            phrases.append(ph)

    if len(toks) >= 2:
        _add_phrase(toks)
        if not is_fund:
            # Articles drop the trailing generic words ("Regeneron", "Super
            # Micro", "Thermo Fisher"), so the name minus its generic tail is a
            # phrase too — never shortened below two words by this loop, and
            # to ONE word only when that word identifies the company by itself
            # ("regeneron", "antero"; not "super", not "range").
            core = list(toks)
            while len(core) >= 3 and _generic(core[-1], freq):
                core.pop()
            if len(core) < len(toks):
                _add_phrase(core)
            if (len(core) == 2 and _generic(core[-1], freq)
                    and _distinctive(core[0], 4, freq)
                    and freq.get(core[0], 0) <= 2):     # "nano" (3 registrants) stays a phrase-only word
                _add_phrase([core[0]])
            for t in toks:
                if _distinctive(t, 6, freq):
                    tokens.append(t)
    elif len(toks) == 1:
        _add_phrase(toks)
    for alias in TICKER_ALIASES.get(sym, []):
        if alias not in phrases:
            phrases.append(alias)
    # The symbol as a bare word is evidence only when it cannot be an ordinary
    # acronym: ≥ 3 letters and not in the extractor's stoplist (AI, EV, ALL, …).
    plain = re.sub(r"[^A-Z]", "", sym)
    symbol_word = len(plain) >= 3 and sym not in _ACRONYM_STOPWORDS and plain == sym
    out = {"phrases": phrases, "tokens": tokens, "symbol_word": symbol_word, "name": name,
           "fund": is_fund}
    _keyword_cache[sym] = out
    return out


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

_EXCHANGES = (r"(?i:nyse(?:\s*american|\s*arca)?|nasdaq(?:\s*(?:gs|gm|cm|global\s+select(?:\s+market)?"
              r"|global\s+market|capital\s+market))?|nysearca|amex|cboe|otc(?:qb|qx|mkts)?|tsx(?:v)?|lse|asx)")


def _phrase_pattern(phrase: str) -> str:
    """Whole-word, case-insensitive pattern for a phrase; each internal gap
    matches any run of spaces/hyphens so "coca-cola" and "coca cola" agree."""
    parts = [re.escape(p) for p in re.split(r"[\s\-]+", phrase) if p]
    body = r"[\s\-]+".join(parts)
    return rf"(?<![a-z0-9]){body}(?![a-z0-9])"


def _compile(ticker: str) -> dict:
    sym = (ticker or "").strip().upper()
    hit = _pattern_cache.get(sym)
    if hit is not None:
        return hit
    kw = name_keywords(sym)
    forms = sorted(_symbol_forms(sym), key=len, reverse=True)
    alt = "|".join(re.escape(f) for f in forms)
    explicit = re.compile(
        rf"(?:\(\s*(?:{_EXCHANGES}\s*:\s*)?(?:{alt})\s*\)"      # (AR) / (NYSE: AR)
        rf"|\$(?:{alt})(?![A-Za-z0-9])"                          # $AR
        rf"|\b{_EXCHANGES}\s*:\s*(?:{alt})(?![A-Za-z0-9]))"      # NYSE: AR
    )
    word = (re.compile(rf"(?<![A-Za-z0-9$])(?:{alt})(?![A-Za-z0-9])")
            if kw["symbol_word"] else None)
    phrase = (re.compile("|".join(_phrase_pattern(p) for p in kw["phrases"]), re.I)
              if kw["phrases"] else None)
    # Tokens must be written Capitalised: "Disney", not "disney"; the ordinary
    # word ("option", "cruise") is lowercase in prose.
    token = (re.compile("|".join(
                 rf"(?<![A-Za-z0-9]){re.escape(t[0].upper() + t[1:])}(?![a-z0-9])"
                 for t in kw["tokens"]))
             if kw["tokens"] else None)
    pats = {"explicit": explicit, "word": word, "phrase": phrase, "token": token}
    _pattern_cache[sym] = pats
    return pats


def _all_caps_around(text: str, start: int, end: int) -> bool:
    """An all-caps headline makes any word look like a symbol. Judged on the
    text AROUND the match — the symbol's own letters are upper-case by
    definition and would dominate a short headline ("AAPL up")."""
    window = text[max(0, start - 40): start] + text[end: end + 40]
    letters = [c for c in window if c.isalpha()]
    return bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.6


def mention_evidence(ticker: str, text: str, allow_token: bool = False) -> Optional[str]:
    """Strongest evidence that *text* is about *ticker*:
    ``"symbol_explicit" | "name_phrase" | "symbol_word" | "name_token" | None``.
    ``name_token`` (the weakest tier) is only consulted with ``allow_token``."""
    if not text or not ticker:
        return None
    pats = _compile(ticker)
    if pats["explicit"].search(text):
        return "symbol_explicit"
    if pats["phrase"] is not None and pats["phrase"].search(text):
        return "name_phrase"
    if pats["word"] is not None:
        m = pats["word"].search(text)
        if m and not _all_caps_around(text, m.start(), m.end()):
            return "symbol_word"
    if allow_token and pats["token"] is not None and pats["token"].search(text):
        return "name_token"
    return None


def mentions(ticker: str, text: str, allow_token: bool = False) -> bool:
    return mention_evidence(ticker, text, allow_token=allow_token) is not None


_SIC_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")


def _title_sic(sic: str) -> str:
    """Polygon serves SIC lines ALL CAPS (``CRUDE PETROLEUM & NATURAL GAS``);
    the header wants prose (``Crude Petroleum & Natural Gas``). Only an
    all-caps line is touched, word by word so ``WOMEN'S`` and
    ``SERVICES-PREPACKAGED`` both come out right."""
    if not sic.isupper():
        return sic
    return _SIC_WORD_RE.sub(lambda m: m.group(0).capitalize(), sic)


# Polygon's non-SIC security types → the industry line rendered for them when
# the reference record carries no SIC description. Every one is a wrapper whose
# "company" is what it holds, so ``industry_is_fund`` reads exactly these lines
# as funds even when the registrant name lacks a fund word (``SPDR Gold
# Shares``). One table, so the renderer and the detector cannot drift apart.
_FUND_TYPE_LINES = {"ETF": "exchange-traded fund", "ETN": "exchange-traded note",
                    "ETV": "exchange-traded vehicle", "FUND": "closed-end fund",
                    "ETS": "exchange-traded share"}
_FUND_INDUSTRIES = frozenset(_FUND_TYPE_LINES.values())

# Industry lines that mean OPERATING COMPANY even though the registrant name
# carries a fund word. `is_fund` matches the bare token "trust", and a REIT is a
# trust by construction — Medical Properties Trust, Americold Realty Trust,
# Healthcare Realty Trust, Community Healthcare Trust, Postal Realty Trust were
# all typed as funds (5 of 635 universe tickers, 2026-09-11 scan).
#
# That is not cosmetic. The same verdict (a) drives `fund_catalyst_override`,
# which rewrites every company-event catalyst on those names to `macro_sector` —
# polluting the BEST-reading class and the `catalyst_tilt` fit with mislabelled
# REIT earnings — and (b) puts a FUND line in the scoring prompt header, telling
# the model a hospital REIT's news is about its holdings. (b) changes the SCORE,
# not just the label.
#
# Deliberately narrow: a SIC line is NOT evidence of an operating company on its
# own, because commodity ETFs carry real ones (GLD and SLV read "Commodity
# Contracts Brokers & Dealers", XLF reads "State Commercial Banks"). Only the
# REIT line is vetoed, because a REIT is an operating company by definition.
# A bank named "... Trust" would be the same class of miss and is not in the
# current universe; re-run `scratchpad/fundscan.py` to check.
_OPERATING_TRUST_MARKERS = ("real estate investment trust",)


def industry_is_operating_trust(industry: Optional[str]) -> bool:
    """True when the industry line says the entity OPERATES despite a fund word
    in its registrant name. Vetoes the name-derived fund verdict only — a
    Polygon fund TYPE still wins, since that is a direct statement about the
    security rather than an inference from its name."""
    return any(m in str(industry or "").strip().lower()
               for m in _OPERATING_TRUST_MARKERS)


# ── Security-type table (2026-09-11) ────────────────────────────────────────
# Polygon's LIST endpoint carries `type` (CS / ETF / ETN / FUND / ETV / ETS) for
# the whole market in one paginated sweep — measured 13,174 tickers in 14 pages
# in 5.9 s, covering 97.2% of this universe. It does NOT carry `sic_code`, which
# is why the per-ticker detail call survives for operating companies.
#
# What the table buys, stated honestly: the FUND verdict becomes a statement
# about the INSTRUMENT instead of an inference from its name. `is_fund` matches
# the bare token "trust", which every REIT carries — that is how five operating
# companies came to be scored as funds. The speed win is the smaller half: only
# ~6% of this universe are fund types, so the cold industry prime still makes a
# detail call for nearly every name.
#
# It is a FIRST tier, not a replacement. Polygon types closed-end funds as `CS`
# (CCD, CHI, CHY, CSQ — Calamos), so the name heuristic still has to run behind
# it or those four stop being funds. Verified: those 4 are the ONLY
# disagreements across 617 covered universe tickers.
_SEC_TYPE_FILE = "security_types.json"
_SEC_TYPE_TTL_DAYS = 7
_SEC_TYPES: Optional[dict] = None
_FUND_SECURITY_TYPES = frozenset({"ETF", "ETN", "ETV", "FUND", "ETS"})


def _sec_type_path():
    return CACHE_DIR / _SEC_TYPE_FILE


def prime_security_types(force: bool = False) -> int:
    """Sweep Polygon's ticker list into ``{ticker: type}``. Returns the count.

    Cheap enough to be unconditional at the TTL (14 pages, ~6 s) and fail-soft:
    on any error the table stays whatever it was and every caller falls back to
    the name/industry path that predates it."""
    global _SEC_TYPES
    path = _sec_type_path()
    if not force:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            ts = datetime.fromisoformat(str(raw.get("ts")))
            if (_now() - ts).days < _SEC_TYPE_TTL_DAYS and raw.get("types"):
                _SEC_TYPES = dict(raw["types"])
                return len(_SEC_TYPES)
        except Exception:
            pass
    out: dict = {}
    try:
        from src.data import polygon_client as pc
        cursor, pages = None, 0
        while pages < 40:
            params = {"market": "stocks", "active": "true", "limit": 1000, "sort": "ticker"}
            if cursor:
                params["cursor"] = cursor
            j = pc._get("/v3/reference/tickers", params) or {}
            res = j.get("results") or []
            for r in res:
                tk = str(r.get("ticker") or "").strip().upper()
                ty = str(r.get("type") or "").strip().upper()
                if tk and ty:
                    out[tk] = ty
            pages += 1
            nxt = str(j.get("next_url") or "")
            cursor = nxt.split("cursor=")[-1] if "cursor=" in nxt else None
            if not cursor or not res:
                break
    except Exception as exc:                                    # noqa: BLE001
        logger.warning(f"[company_names] security-type sweep failed ({exc}) — "
                       f"falling back to the name/industry path")
        return len(_SEC_TYPES or {})
    if not out:
        return len(_SEC_TYPES or {})
    _SEC_TYPES = out
    try:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"ts": _iso(_now()), "types": out}), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass
    logger.info(f"[company_names] security types: {len(out)} tickers")
    return len(out)


def security_type(ticker: str) -> Optional[str]:
    """Polygon's security type for *ticker*, or None when the table has not been
    primed or does not cover it. Tries the Polygon symbol form too — class
    shares are ``BRK-B`` here and ``BRK.B`` there, and the hyphen form silently
    matches nothing."""
    global _SEC_TYPES
    sym = (ticker or "").strip().upper()
    if not sym:
        return None
    if _SEC_TYPES is None:
        try:
            raw = json.loads(_sec_type_path().read_text(encoding="utf-8"))
            _SEC_TYPES = dict(raw.get("types") or {})
        except Exception:
            _SEC_TYPES = {}
    hit = _SEC_TYPES.get(sym)
    if hit:
        return hit
    try:
        from src.data.polygon_client import to_polygon_symbol
        alt = to_polygon_symbol(sym)
    except Exception:
        return None
    return _SEC_TYPES.get(alt) if alt != sym else None


def type_is_fund(sec_type: Optional[str]) -> bool:
    """True for the wrapper types. A security TYPE is a statement about the
    instrument; a fund word in a NAME is an inference from it."""
    return str(sec_type or "").strip().upper() in _FUND_SECURITY_TYPES


def _industry_from_reference(res: Optional[dict]) -> Optional[str]:
    """The industry line for one Polygon reference record: its
    ``sic_description`` (title-cased) when it has one, else the fund line for
    its security ``type``, else None."""
    res = res if isinstance(res, dict) else {}
    sic = str(res.get("sic_description") or "").strip()
    if sic:
        return _title_sic(sic)
    typ = str(res.get("type") or "").strip().upper()
    return _FUND_TYPE_LINES.get(typ) or None


def _polygon_industry(ticker: str) -> Optional[str]:
    """One-line industry description from Polygon's reference endpoint —
    ``sic_description`` for an operating company (``"Crude Petroleum & Natural
    Gas"``), the security TYPE for a fund, which carries no SIC line."""
    try:
        from src.data import polygon_client as pc
        j = pc._get(f"/v3/reference/tickers/{pc.to_polygon_symbol(ticker)}", {})
        return _industry_from_reference((j or {}).get("results"))
    except Exception as exc:
        logger.debug(f"[company_names] polygon industry lookup failed for {ticker}: {exc}")
        return None


def industry_of(ticker: str) -> Optional[str]:
    """A short industry line for *ticker* (``"Crude Petroleum & Natural Gas"``),
    or ``None`` when no source knows one. Polygon reference, cached beside the
    names (positives re-asked after ``_INDUSTRY_TTL_DAYS``, unknowns after
    ``_NEGATIVE_TTL_DAYS``). Read by the sentiment TARGET HEADER (the hook that
    lets a model notice "EQT AB" is not "EQT Corp — Crude Petroleum & Natural
    Gas") and by the catalyst specialist's header. The scoring header salts the
    verdict cache key, so an industry resolving LATER re-scores that ticker
    once — the same one-time cost a name resolving later already carries; prime
    the universe up front (``prime_industries``) so the scoring path stays
    cache-only."""
    sym = (ticker or "").strip().upper()
    if not sym:
        return None
    st = _load_state()
    with _lock:
        ind = st.setdefault("industry", {})
        rec = ind.get(sym)
        if _industry_fresh(rec):
            return rec.get("industry") or None
        # SIC FIRST, security type as the fallback — the order
        # `_industry_from_reference` documents, and reversing it was a real (if
        # latent) regression: many wrappers DO carry a real SIC line (GLD, SLV
        # and USO read "Commodity Contracts Brokers & Dealers", XLF reads "State
        # Commercial Banks"), so a type-first short-circuit silently replaced
        # those with "exchange-traded vehicle". It had not fired yet only
        # because the industry cache is 180 days warm; it would have hit every
        # newly discovered fund, changing the prompt header and re-keying that
        # ticker's verdict cache with no epoch.
        #
        # The type table still earns its place here as RESILIENCE, not speed: a
        # failed or empty reference call now yields a usable line for a wrapper
        # instead of None.
        line = _polygon_industry(sym)
        if line is None:
            line = _FUND_TYPE_LINES.get(str(security_type(sym) or "").upper())
        ind[sym] = {"industry": line, "ts": _iso(_now())}
        _save_state()
        return line


def prime(tickers: Iterable[str]) -> int:
    """Resolve names for a universe up front (one SEC download + a Polygon call
    per symbol the SEC list lacks). Returns the number of symbols with a name."""
    n = 0
    for t in tickers:
        if company_name(t):
            n += 1
    return n


def _industry_fresh(rec: Optional[dict]) -> bool:
    if rec is None:
        return False
    ts = _parse_iso(rec.get("ts"))
    ttl = _INDUSTRY_TTL_DAYS if rec.get("industry") else _NEGATIVE_TTL_DAYS
    return ts is not None and (_now() - ts) < timedelta(days=ttl)


def prime_industries(tickers: Iterable[str], workers: int = 8) -> int:
    """Resolve industry lines for a universe up front. Unlike ``industry_of``
    (one symbol, lock held across its Polygon call — the ``company_name``
    idiom), the cold fetches here run in a small thread pool OUTSIDE the lock
    and the cache is written ONCE: a ~950-name universe is ~950 reference calls
    the first time, cached for ``_INDUSTRY_TTL_DAYS`` afterwards, and a serial
    sweep with a file write per symbol would sit minutes inside the Step-1
    pool. Returns the number of symbols with an industry line."""
    syms = []
    seen: Set[str] = set()
    for t in tickers:
        sym = (t or "").strip().upper()
        if sym and sym not in seen:
            seen.add(sym)
            syms.append(sym)
    if not syms:
        return 0
    st = _load_state()
    with _lock:
        ind = st.setdefault("industry", {})
        todo = [s for s in syms if not _industry_fresh(ind.get(s))]
    if todo:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(todo)))) as pool:
            fetched = list(pool.map(_polygon_industry, todo))
        ts = _iso(_now())
        with _lock:
            for sym, line in zip(todo, fetched):
                ind[sym] = {"industry": line, "ts": ts}
            _save_state()
    with _lock:
        return sum(1 for s in syms if (ind.get(s) or {}).get("industry"))


def industry_is_fund(line: Optional[str]) -> bool:
    """True when *line* is one of the fund lines ``_polygon_industry`` renders
    for a non-SIC security type (see ``_FUND_TYPE_LINES``)."""
    return bool(line) and str(line).strip().lower() in _FUND_INDUSTRIES


def _reset_for_tests() -> None:
    global _state, _token_freq
    with _lock:
        _state = None
        _token_freq = None
        _keyword_cache.clear()
        _pattern_cache.clear()


def _seed_for_tests(names: dict, industries: Optional[dict] = None) -> None:
    """Install ``{SYMBOL: registrant name}`` as the whole known universe (SEC
    provenance, freshly loaded) so a test never reaches the network. Every
    seeded symbol also gets a FRESH industry record — the line from
    *industries* when given, else a known-unknown — so ``industry_of`` never
    looks one up mid-test."""
    global _state, _token_freq
    with _lock:
        ts = _iso(_now())
        ind = {str(k).upper(): v for k, v in (industries or {}).items()}
        _state = {"sec_loaded_at": ts,
                  "names": {str(k).upper(): {"name": v, "source": "sec", "ts": ts}
                            for k, v in names.items()},
                  "industry": {str(k).upper(): {"industry": ind.get(str(k).upper()), "ts": ts}
                               for k in names}}
        _token_freq = None
        _keyword_cache.clear()
        _pattern_cache.clear()
