"""Company-name relevance engine (`src/data/company_names.py`, 2026-09-04).

The per-ticker news digest used to be selected by ``symbol.lower() in text`` —
a bare substring, so ``"ar" in "market"`` was a hit and every short symbol
received the entire article pool; the sentiment scorer then correctly
reported "about other companies" and abstained on 73–79% of calls. These tests
pin the replacement: evidence TIERS over the SEC registrant name, curated
aliases and the explicit symbol, with a name-normalisation that never leaves a
generic word standing for a company.

Every test runs OFFLINE — the autouse conftest fixture empties the resolver's
state and stubs both sources; names are seeded with ``_seed_for_tests``.
"""
import json

import pytest

from src.data import company_names as cn


def _seed(**names):
    cn._seed_for_tests({k.replace("_", "-"): v for k, v in names.items()})


# ── Name normalisation ───────────────────────────────────────────────────────

@pytest.mark.parametrize("name, toks", [
    ("The Walt Disney Company", ["walt", "disney"]),
    ("ELI LILLY & Co", ["eli", "lilly"]),
    ("US BANCORP \\DE\\", ["us"]),                       # SEC state marker cut
    ("BOEING CO /DE/", ["boeing"]),
    ("AT&T Inc.", ["at&t"]),
    ("Hut 8 Corp.", ["hut", "8"]),
    ("Alphabet Inc. Class C", ["alphabet"]),            # class-share tail stripped
    ("Apple Inc.", ["apple"]),
    ("Antero Resources Corp", ["antero", "resources"]),
    ("Regeneron Pharmaceuticals, Inc.", ["regeneron", "pharmaceuticals"]),
    ("Super Micro Computer, Inc.", ["super", "micro", "computer"]),
    ("Brown-Forman Corp Class B", ["brown-forman"]),
])
def test_name_tokens_strip_markers_and_boilerplate(name, toks):
    assert cn._name_tokens(name) == toks


# ── Keyword construction ─────────────────────────────────────────────────────

def test_keywords_shorten_generic_tail_but_never_to_a_generic_word():
    _seed(REGN="Regeneron Pharmaceuticals, Inc.", RRC="Range Resources Corp",
          SMCI="Super Micro Computer, Inc.")
    regn = cn.name_keywords("REGN")["phrases"]
    assert "regeneron pharmaceuticals" in regn and "regeneron" in regn   # distinctive alone
    rrc = cn.name_keywords("RRC")["phrases"]
    assert "range resources" in rrc and "range" not in rrc              # ordinary word
    smci = cn.name_keywords("SMCI")["phrases"]
    assert "super micro computer" in smci and "super micro" in smci     # generic tail dropped
    assert "super" not in smci and "micro" not in smci                  # never one generic word


def test_keywords_token_shared_by_many_registrants_is_generic():
    """A name token used by ≥ 6 registrants identifies nothing — the
    self-maintaining generic test, independent of the hand stoplist."""
    others = {f"P{i}": f"Planet {w} Inc" for i, w in enumerate(
        ["labs", "fitness", "green", "payment", "13", "home"])}
    _seed(PL="Planet Labs PBC", **others)
    kw = cn.name_keywords("PL")
    assert "planet labs" in kw["phrases"] and "planet" not in kw["phrases"]
    assert "planet" not in kw["tokens"]


def test_keywords_single_token_only_when_rare_and_distinctive():
    _seed(AR="Antero Resources Corp", NNE="Nano Nuclear Energy Inc",
          N1="Nano Dimension Ltd", N2="Nano-X Imaging Ltd", N3="Nano Labs Ltd")
    assert "antero" in cn.name_keywords("AR")["phrases"]          # one registrant, 6 letters
    nne = cn.name_keywords("NNE")["phrases"]
    assert "nano nuclear energy" in nne and "nano" not in nne     # shared + stoplisted


def test_keywords_funds_keep_the_full_phrase_only():
    """Shortening a fund's name would leave its ISSUER ("state street"),
    which is not the fund."""
    _seed(XLE="The Energy Select Sector SPDR Fund",
          SPYX="State Street Energy Select Sector SPDR Trust")
    for sym in ("XLE", "SPYX"):
        ph = cn.name_keywords(sym)["phrases"]
        assert not any(p in ("state street", "energy select", "state") for p in ph), ph
    assert "energy sector" in cn.name_keywords("XLE")["phrases"]  # the alias map covers ETFs


def test_keywords_aliases_and_symbol_word():
    _seed(AAPL="Apple Inc.", DIS="The Walt Disney Company")
    kw = cn.name_keywords("AAPL")
    assert {"apple", "iphone", "ipad"} <= set(kw["phrases"])
    assert kw["symbol_word"] is True                            # AAPL: 4 letters, no acronym
    dis = cn.name_keywords("DIS")
    assert "walt disney" in dis["phrases"] and "disney" in dis["phrases"]
    assert "disney" in dis["tokens"]
    for sym in ("AI", "EV", "ALL", "AR", "GS"):                  # acronym / too short
        assert cn.name_keywords(sym)["symbol_word"] is False, sym


def test_keywords_unnamed_symbol_still_has_the_symbol_tiers():
    kw = cn.name_keywords("CRDO")                                 # nothing seeded
    assert kw["name"] is None and kw["phrases"] == [] and kw["symbol_word"] is True


def test_keywords_two_letter_phrase_rejected():
    """"V F Corp" must not produce the phrase "v f" (matches every "V. F." initial)."""
    _seed(VFC="V F Corp")
    assert all(len(p.replace(" ", "")) >= 4 for p in cn.name_keywords("VFC")["phrases"])


# ── Evidence tiers ───────────────────────────────────────────────────────────

def test_evidence_tiers_in_strength_order():
    _seed(AR="Antero Resources Corp", DIS="The Walt Disney Company",
          LULU="Lululemon Athletica Inc")
    ev = cn.mention_evidence
    assert ev("AR", "Antero (NYSE: AR) raised guidance") == "symbol_explicit"
    assert ev("AR", "$AR breaks out") == "symbol_explicit"
    assert ev("AR", "NYSE:AR volume spikes") == "symbol_explicit"
    assert ev("AR", "Antero Resources raised guidance") == "name_phrase"
    assert ev("AR", "shares of Antero rose") == "name_phrase"      # rare single token → phrase
    assert ev("DIS", "Disney+ raises prices") == "name_phrase"     # curated alias
    # A distinctive token of a two-word name that does not shorten is the
    # weakest tier — opt-in, Capitalised only.
    assert ev("LULU", "Lululemon beats on margins", allow_token=True) == "name_token"
    assert ev("LULU", "Lululemon beats on margins") is None
    assert ev("LULU", "a lululemon-style rally", allow_token=True) is None
    assert ev("AAPL", "AAPL slips after hours") == "symbol_word"
    assert ev("AR", "AR headsets, the market, Barclays") is None   # the old substring sweep


def test_symbol_word_is_case_sensitive_and_refuses_all_caps_headlines():
    _seed(AAPL="Apple Inc.")
    assert cn.mention_evidence("AAPL", "AAPL slips") == "symbol_word"
    assert cn.mention_evidence("AAPL", "aapl slips") is None
    assert cn.mention_evidence("HOOD", "HOOD ORNAMENT THEFTS SURGE IN CHICAGO") is None
    assert cn.mention_evidence("HOOD", "Shares of HOOD jump after hours") == "symbol_word"
    assert cn.mention_evidence("HOOD", "Robinhood's HOOD stock jumps") == "name_phrase"  # alias outranks


def test_phrase_matching_is_whole_word_and_hyphen_tolerant():
    _seed(KO="Coca-Cola Co", AR="Antero Resources Corp")
    assert cn.mention_evidence("KO", "Coca Cola lifts outlook") == "name_phrase"
    assert cn.mention_evidence("KO", "coca-cola bottlers") == "name_phrase"
    assert cn.mention_evidence("AR", "Anterograde amnesia study") is None   # not a whole word


def test_class_share_symbol_forms():
    _seed(BRK_B="BERKSHIRE HATHAWAY INC")
    for txt in ("(NYSE: BRK.B)", "(BRK-B)", "$BRK.B", "NYSE: BRK B"):
        assert cn.mention_evidence("BRK-B", f"Buffett's firm {txt} sold Apple") == "symbol_explicit", txt
    assert cn.mention_evidence("BRK.B", "Berkshire Hathaway sold Apple") == "name_phrase"


def test_explicit_symbol_requires_a_delimiter():
    """``(AR)`` / ``$AR`` / ``NYSE: AR`` are explicit; a bare two-letter "AR" is not."""
    assert cn.mention_evidence("AR", "AR and VR headsets") is None
    assert cn.mention_evidence("AR", "Antero Resources (AR) said") == "symbol_explicit"


def test_mentions_wrapper_and_empty_inputs():
    assert cn.mentions("AAPL", "AAPL up") is True
    assert cn.mentions("AAPL", "") is False
    assert cn.mentions("", "AAPL up") is False
    assert cn.mention_evidence("AAPL", None) is None


# ── Resolver / cache behaviour ───────────────────────────────────────────────

def test_company_name_reads_seeded_state_without_network():
    _seed(AAPL="Apple Inc.")
    assert cn.company_name("AAPL") == "Apple Inc."
    assert cn.company_name("aapl ") == "Apple Inc."
    assert cn.company_name("ZZZZ") is None                        # polygon stub → None
    assert cn.company_name("") is None


def test_unknown_symbol_is_cached_as_negative(monkeypatch):
    _seed(AAPL="Apple Inc.")
    calls = []
    monkeypatch.setattr(cn, "_polygon_name", lambda t: calls.append(t) or None)
    assert cn.company_name("ZZZZ") is None
    assert cn.company_name("ZZZZ") is None
    assert calls == ["ZZZZ"]                                      # asked once, then the negative TTL


def test_polygon_fallback_fills_names_the_sec_list_lacks(monkeypatch):
    _seed(AAPL="Apple Inc.")
    monkeypatch.setattr(cn, "_polygon_name", lambda t: "Credo Technology Group Holding Ltd" if t == "CRDO" else None)
    assert cn.company_name("CRDO") == "Credo Technology Group Holding Ltd"
    assert "credo technology" in cn.name_keywords("CRDO")["phrases"]


def test_cache_file_roundtrip(tmp_path, monkeypatch):
    """The disk cache is what makes a warm tick ~0.4 s: state written by one
    process is read back by the next without touching either source."""
    monkeypatch.setattr(cn, "_CACHE_PATH", tmp_path / "cn.json")
    _seed(AAPL="Apple Inc.")
    cn._save_state()
    raw = json.loads((tmp_path / "cn.json").read_text(encoding="utf-8"))
    assert raw["names"]["AAPL"]["name"] == "Apple Inc."
    cn._reset_for_tests()
    assert cn.company_name("AAPL") == "Apple Inc."                # loaded from disk


def test_prime_counts_named_symbols():
    _seed(AAPL="Apple Inc.", DIS="The Walt Disney Company")
    assert cn.prime(["AAPL", "DIS", "ZZZZ"]) == 2


def test_seed_resets_keyword_caches():
    _seed(AR="Antero Resources Corp")
    assert "antero resources" in cn.name_keywords("AR")["phrases"]
    _seed(AR="Argan Inc")
    assert cn.name_keywords("AR")["phrases"] == ["argan"]


# ── Industry line (Polygon reference, cached beside the names) ──────────────
# The sentiment TARGET HEADER renders it ("AR — Antero Resources Corp
# (industry: Crude Petroleum & Natural Gas)") as the hook that lets a model
# notice "EQT AB" is not "EQT Corp"; the header text salts the verdict cache
# key, so the line must be primed for the universe up front and served
# cache-only from the scoring path.

def test_industry_of_reads_seeded_state_and_caches_unknowns(monkeypatch):
    cn._seed_for_tests({"AR": "Antero Resources Corp", "XYZ": "Xyz Corp"},
                       industries={"AR": "Crude Petroleum & Natural Gas"})
    assert cn.industry_of("AR") == "Crude Petroleum & Natural Gas"
    assert cn.industry_of(" ar ") == "Crude Petroleum & Natural Gas"
    assert cn.industry_of("XYZ") is None                          # seeded known-unknown
    assert cn.industry_of("") is None
    calls = []
    monkeypatch.setattr(cn, "_polygon_industry", lambda t: calls.append(t) or None)
    assert cn.industry_of("ZZZZ") is None
    assert cn.industry_of("ZZZZ") is None
    assert calls == ["ZZZZ"]                                      # asked once, then the negative TTL


def test_industry_records_expire_on_their_own_ttls():
    from datetime import timedelta
    now = cn._now()

    def rec(line, age_days):
        return {"industry": line, "ts": cn._iso(now - timedelta(days=age_days))}

    assert cn._industry_fresh(rec("Widgets", cn._INDUSTRY_TTL_DAYS - 1))
    assert not cn._industry_fresh(rec("Widgets", cn._INDUSTRY_TTL_DAYS + 1))
    assert cn._industry_fresh(rec(None, cn._NEGATIVE_TTL_DAYS - 1))
    assert not cn._industry_fresh(rec(None, cn._NEGATIVE_TTL_DAYS + 1))
    assert not cn._industry_fresh(None)
    assert not cn._industry_fresh({"industry": "Widgets"})        # no timestamp ⇒ re-ask


def test_prime_industries_pools_cold_fetches_and_writes_once(tmp_path, monkeypatch):
    """Step 1 primes the universe so the scoring path stays cache-only. Only
    symbols without a fresh record are fetched, each once (input deduped and
    case-folded), the count is the symbols that carry a line, and the result
    is on disk for the next process."""
    monkeypatch.setattr(cn, "_CACHE_PATH", tmp_path / "cn.json")
    cn._seed_for_tests({"AAA": "Aaa Inc", "BBB": "Bbb Inc", "CCC": "Ccc Inc"},
                       industries={"AAA": "Ind-AAA"})
    st = cn._load_state()
    st["industry"].pop("BBB")
    st["industry"].pop("CCC")
    calls = []
    monkeypatch.setattr(cn, "_polygon_industry",
                        lambda t: calls.append(t) or ("Ind-BBB" if t == "BBB" else None))
    assert cn.prime_industries(["AAA", "BBB", "bbb", "CCC", "", "AAA"]) == 2
    assert sorted(calls) == ["BBB", "CCC"]
    assert cn.industry_of("BBB") == "Ind-BBB"
    assert cn.industry_of("CCC") is None
    assert sorted(calls) == ["BBB", "CCC"]                        # cache-only afterwards
    raw = json.loads((tmp_path / "cn.json").read_text(encoding="utf-8"))
    assert raw["industry"]["BBB"]["industry"] == "Ind-BBB"
    assert raw["industry"]["CCC"]["industry"] is None
    assert cn.prime_industries([]) == 0
    assert cn.prime_industries(["AAA", "BBB", "CCC"]) == 2        # warm: no fetch
    assert sorted(calls) == ["BBB", "CCC"]


@pytest.mark.parametrize("record, line", [
    ({"sic_description": "CRUDE PETROLEUM & NATURAL GAS", "type": "CS"}, "Crude Petroleum & Natural Gas"),
    ({"sic_description": "SERVICES-PREPACKAGED SOFTWARE"}, "Services-Prepackaged Software"),
    ({"sic_description": "WOMEN'S, MISSES', AND JUNIORS OUTERWEAR"}, "Women's, Misses', And Juniors Outerwear"),
    ({"sic_description": "Crude Petroleum & Natural Gas"}, "Crude Petroleum & Natural Gas"),  # prose kept
    ({"sic_description": "", "type": "ETF"}, "exchange-traded fund"),          # fund: no SIC line
    ({"type": "etn"}, "exchange-traded note"),
    ({"type": "FUND"}, "closed-end fund"),
    ({"sic_description": "COMMODITY CONTRACTS BROKERS & DEALERS", "type": "ETF"},
     "Commodity Contracts Brokers & Dealers"),                              # SIC wins when present
    ({"type": "CS"}, None),
    ({}, None),
    (None, None),
])
def test_industry_from_reference_prefers_sic_then_fund_type(record, line):
    assert cn._industry_from_reference(record) == line


def test_industry_is_fund_matches_exactly_the_rendered_fund_lines():
    for typ, line in cn._FUND_TYPE_LINES.items():
        assert cn.industry_is_fund(line), typ
        assert cn.industry_is_fund(f" {line.upper()} ")
    assert not cn.industry_is_fund("Crude Petroleum & Natural Gas")
    assert not cn.industry_is_fund("Commodity Contracts Brokers & Dealers")   # GLD's SIC: name flag covers it
    assert not cn.industry_is_fund(None)
    assert not cn.industry_is_fund("")


# ── REITs are not funds (2026-09-11) ────────────────────────────────────────

def test_a_REIT_industry_line_reads_as_an_operating_company():
    """`is_fund` matches the bare token "trust", and a REIT is a trust by
    construction: Medical Properties Trust, Americold Realty Trust, Healthcare
    Realty Trust, Community Healthcare Trust and Postal Realty Trust were all
    typed as funds (5 of 635 universe tickers)."""
    from src.data.company_names import industry_is_operating_trust as op
    assert op("Real Estate Investment Trusts")
    assert op("real estate investment trust")


def test_the_veto_is_NARROW_because_a_SIC_line_is_not_evidence_on_its_own():
    """Commodity ETFs carry real SIC descriptions — GLD and SLV read "Commodity
    Contracts Brokers & Dealers", XLF reads "State Commercial Banks" — so
    "has an operating-looking industry" would un-fund genuine funds. Only the
    REIT line is vetoed."""
    from src.data.company_names import industry_is_operating_trust as op
    for ind in ("Commodity Contracts Brokers & Dealers", "State Commercial Banks",
                "Investment Advice", "exchange-traded fund", "closed-end fund", "", None):
        assert not op(ind), ind


def test_identity_vetoes_a_name_fund_word_on_a_REIT(monkeypatch):
    """The whole point: the FUND verdict reaches the scoring prompt header, so a
    false positive tells the model a hospital REIT's news is about its holdings
    — it moves the SCORE, not just the catalyst label."""
    import src.analysis.sentiment as sent
    import src.data.company_names as cn
    monkeypatch.setattr(cn, "name_keywords",
                        lambda s: {"name": "MEDICAL PROPERTIES TRUST INC", "fund": True,
                                   "phrases": [], "tokens": [], "symbol_word": False},
                        raising=False)
    monkeypatch.setattr(cn, "industry_of", lambda s: "Real Estate Investment Trusts",
                        raising=False)
    _sym, name, industry, fund = sent._target_identity("MPT")
    assert name and industry == "Real Estate Investment Trusts"
    assert fund is False
    assert sent._is_fund_target("MPT") is False
    # and the label override therefore leaves a company event alone
    assert sent.fund_catalyst_override("MPT", "earnings") == "earnings"


def test_a_genuine_commodity_trust_is_still_a_fund(monkeypatch):
    """`ProShares Trust II` and `Amplify Commodity Trust` are flagged by the same
    bare "trust" token and ARE funds; their industry line must not veto them."""
    import src.analysis.sentiment as sent
    import src.data.company_names as cn
    monkeypatch.setattr(cn, "name_keywords",
                        lambda s: {"name": "Amplify Commodity Trust", "fund": True,
                                   "phrases": [], "tokens": [], "symbol_word": False},
                        raising=False)
    monkeypatch.setattr(cn, "industry_of",
                        lambda s: "Commodity Contracts Brokers & Dealers", raising=False)
    assert sent._target_identity("BWET")[3] is True
    assert sent.fund_catalyst_override("BWET", "earnings") == "macro_sector"


def test_a_polygon_fund_TYPE_still_wins_over_the_veto(monkeypatch):
    """A security TYPE is a direct statement about the instrument; the name is an
    inference. Precedence is made explicit so a later edit cannot quietly invert
    it."""
    import src.analysis.sentiment as sent
    import src.data.company_names as cn
    monkeypatch.setattr(cn, "name_keywords",
                        lambda s: {"name": "Some Realty Trust", "fund": False,
                                   "phrases": [], "tokens": [], "symbol_word": False},
                        raising=False)
    monkeypatch.setattr(cn, "industry_of", lambda s: "exchange-traded fund", raising=False)
    assert sent._target_identity("X")[3] is True


def test_no_scorer_epoch_is_registered_for_this():
    """Deliberate. The fix changes the verdict for 5 of 635 tickers (0.8%), and a
    news-family epoch masks the WHOLE family's history — which is already down to
    days post-2026-09-11. Discarding every news row to correct 0.8% of them is
    the wrong trade, and the header is in the cache salt so those five re-score
    on their own."""
    from src.signals.method_epochs import METHOD_SCORER_EPOCH
    # the family boundary moves whenever the verdict changes; what this test
    # asserts is that the REIT fix did not add one of its own, so it pins the
    # date rather than the exact instant.
    assert str(METHOD_SCORER_EPOCH["news"]).startswith("2026-09-11")


# ── security-type table (2026-09-11) ────────────────────────────────────────

def test_wrapper_types_are_funds_and_ordinary_ones_are_not():
    from src.data.company_names import type_is_fund
    for t in ("ETF", "ETN", "ETV", "FUND", "ETS", "etf"):
        assert type_is_fund(t), t
    for t in ("CS", "PFD", "WARRANT", "ADRC", "UNIT", "SP", "", None):
        assert not type_is_fund(t), t


def test_security_type_tries_the_polygon_symbol_form(monkeypatch):
    """Class shares are `BRK-B` internally and `BRK.B` at Polygon, and the
    hyphen form silently matches nothing — the same defect that burned 238 of
    976 snapshot misses before `to_polygon_symbol` was applied at every call
    site."""
    import src.data.company_names as cn
    monkeypatch.setattr(cn, "_SEC_TYPES", {"BRK.B": "CS", "SPY": "ETF"}, raising=False)
    assert cn.security_type("BRK-B") == "CS"
    assert cn.security_type("SPY") == "ETF"
    assert cn.security_type("NOPE") is None


def test_the_TYPE_tier_beats_the_name_heuristic(monkeypatch):
    """A security type is a statement about the INSTRUMENT; a fund word in a
    name is an inference from it. MPT is `CS` and must not be a fund however
    loudly "MEDICAL PROPERTIES TRUST INC" reads like one."""
    import src.analysis.sentiment as sent
    import src.data.company_names as cn
    monkeypatch.setattr(cn, "_SEC_TYPES", {"MPT": "CS"}, raising=False)
    monkeypatch.setattr(cn, "name_keywords",
                        lambda s: {"name": "MEDICAL PROPERTIES TRUST INC", "fund": True,
                                   "phrases": [], "tokens": [], "symbol_word": False},
                        raising=False)
    monkeypatch.setattr(cn, "industry_of", lambda s: "Real Estate Investment Trusts",
                        raising=False)
    assert sent._target_identity("MPT")[3] is False


def test_a_closed_end_fund_typed_CS_is_still_a_fund(monkeypatch):
    """The table is a FIRST tier, not a replacement: Polygon types closed-end
    funds as `CS` (CCD, CHI, CHY, CSQ — the ONLY 4 disagreements across 617
    covered universe tickers), so the name heuristic still has to run behind it
    or those four stop being funds."""
    import src.analysis.sentiment as sent
    import src.data.company_names as cn
    monkeypatch.setattr(cn, "_SEC_TYPES", {"CCD": "CS"}, raising=False)
    monkeypatch.setattr(cn, "name_keywords",
                        lambda s: {"name": "Calamos Dynamic Convertible & Income Fund",
                                   "fund": True, "phrases": [], "tokens": [],
                                   "symbol_word": False}, raising=False)
    monkeypatch.setattr(cn, "industry_of", lambda s: None, raising=False)
    assert sent._target_identity("CCD")[3] is True


def test_the_TYPE_line_is_a_FALLBACK_and_never_overrides_a_real_SIC(monkeypatch, tmp_path):
    """The precedence `_industry_from_reference` documents: `sic_description`
    when it has one, else the fund line for its security type.

    This test originally asserted the OPPOSITE — that a wrapper type skips the
    detail call — which pinned a regression rather than catching one. Many
    wrappers carry a real SIC line (GLD/SLV/USO read "Commodity Contracts
    Brokers & Dealers", XLF reads "State Commercial Banks"), so a type-first
    short-circuit silently replaced those with "exchange-traded vehicle",
    changing the prompt header and re-keying that ticker's verdict cache."""
    import src.data.company_names as cn
    monkeypatch.setattr(cn, "_SEC_TYPES", {"GLD": "ETV", "SPY": "ETF"}, raising=False)
    monkeypatch.setattr(cn, "_CACHE_PATH", tmp_path / "cn.json", raising=False)
    monkeypatch.setattr(cn, "_STATE", None, raising=False)

    # a wrapper WITH a real SIC keeps the SIC
    monkeypatch.setattr(cn, "_polygon_industry",
                        lambda s: "Commodity Contracts Brokers & Dealers", raising=False)
    assert cn.industry_of("GLD") == "Commodity Contracts Brokers & Dealers"

    # a wrapper with NO usable reference line falls back to its TYPE — the
    # resilience the table actually buys here
    monkeypatch.setattr(cn, "_polygon_industry", lambda s: None, raising=False)
    assert cn.industry_of("SPY") == "exchange-traded fund"


def test_the_sweep_is_fail_soft(monkeypatch):
    """A failed sweep must leave the previous table alone and let every caller
    fall back to the name/industry path that predates it — never zero the fund
    verdict for the whole universe."""
    import src.data.company_names as cn
    from src.data import polygon_client as pc
    monkeypatch.setattr(cn, "_SEC_TYPES", {"SPY": "ETF"}, raising=False)

    def boom(*a, **k):
        raise RuntimeError("polygon down")

    # patch the CLIENT, not sys.modules: `from src.data import polygon_client`
    # reads the attribute off the already-imported package, so a sys.modules
    # swap does not intercept it — and the test would quietly hit the network.
    monkeypatch.setattr(pc, "_get", boom)
    assert cn.prime_security_types(force=True) == 1
    assert cn.security_type("SPY") == "ETF"
