"""News-family continuity pass (2026-08-14) — quantization fixes + news_shock.

Measured motivation (45-day panel): `insider` had 81% within-run tie mass (331
views sharing 62 distinct values — one value covered 3,027 rows), and `news`
had ~38% (round LLM verdicts × step scalers). The rank transform hands a tie
group one average rank, so most of both methods' cross-sections collapsed to
the median and the payoff-shaping deciles were fit on ties. These tests pin the
continuity mechanisms and the new abnormal-attention method.
"""

from datetime import date, datetime, timedelta, timezone

import pytest

from src.models import InsiderTrade, NewsArticle


def _art(hours_ago: float, source: str = "rss", title: str = "t") -> NewsArticle:
    return NewsArticle(
        title=title, summary="s", url=f"http://x/{source}/{title}/{hours_ago}",
        source=source,
        published_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
    )


def _trade(days_ago: int = 0, amount: str = "$15,001 - $50,000",
           role: str = "Director", tx: str = "purchase") -> InsiderTrade:
    return InsiderTrade(
        ticker="AAA", trader_name="X", trader_type="corporate_insider", role=role,
        transaction_type=tx, amount_range=amount,
        transaction_date=date.today() - timedelta(days=days_ago),
        disclosure_date=date.today() - timedelta(days=max(0, days_ago - 1)),
    )


# ── insider continuity ──────────────────────────────────────────────────────

def test_insider_recency_breaks_ties():
    """Same filing pattern, different WHEN → different scores. This is the tie
    the old scorer could not break (no time term at all)."""
    from src.signals.aggregator import _insider_score
    fresh = _insider_score("AAA", [_trade(days_ago=0)])[0]
    week = _insider_score("AAA", [_trade(days_ago=7)])[0]
    month = _insider_score("AAA", [_trade(days_ago=30)])[0]
    assert fresh > week > month > 0
    assert week == pytest.approx(fresh / 2, rel=0.05)      # 7d half-life
    assert _insider_score("AAA", [_trade(days_ago=60)])[0] == 0.0  # beyond max age


def test_exact_notional_beats_the_bucket():
    """The largest quantizer was self-inflicted: 13F / Form 4 / options-sweep
    rows KNOW the exact dollar value and called `_notional_to_amount_range()`
    to squeeze it into 8 buckets before scoring, so a $60k and a $95k purchase
    produced the identical number. The exact value now rides the model."""
    from src.signals.aggregator import _amount_weight_log
    bucket = "$50,001 - $100,000"
    w60 = _amount_weight_log(bucket, 60_000.0)
    w95 = _amount_weight_log(bucket, 95_000.0)
    assert w60 != w95 and w60 < w95           # same bucket, different weight
    # No exact value → the bucket midpoint (congressional/13D-G keep this path).
    assert _amount_weight_log(bucket) == _amount_weight_log(bucket, None)
    assert _amount_weight_log(bucket, 0.0) == _amount_weight_log(bucket)


def test_the_three_known_value_sources_carry_notional():
    """A regression guard: if a source stops setting notional_usd the method
    silently re-quantizes, which looks like nothing at all."""
    import inspect
    from src.data import options_flow, sec_filings
    for mod in (options_flow, sec_filings):
        src = inspect.getsource(mod)
        assert "notional_usd=" in src, f"{mod.__name__} dropped notional_usd"
    assert inspect.getsource(sec_filings).count("notional_usd=") >= 2   # 13F + Form 4


def test_disclosure_lag_weight():
    """Fast disclosure is a stronger tell; same-day sources are unaffected."""
    from src.signals.aggregator import _disclosure_lag_weight
    same = _trade(days_ago=3)                       # disclosure == transaction+... see helper
    fast = InsiderTrade(ticker="A", trader_name="r", trader_type="politician",
                        role="Senator", transaction_type="purchase",
                        amount_range="$15,001 - $50,000",
                        transaction_date=date.today() - timedelta(days=30),
                        disclosure_date=date.today() - timedelta(days=27))   # 3d lag
    slow = InsiderTrade(ticker="A", trader_name="r", trader_type="politician",
                        role="Senator", transaction_type="purchase",
                        amount_range="$15,001 - $50,000",
                        transaction_date=date.today() - timedelta(days=30),
                        disclosure_date=date.today() + timedelta(days=14))   # 44d lag
    assert _disclosure_lag_weight(fast) > _disclosure_lag_weight(slow)
    assert 0.70 < _disclosure_lag_weight(slow) < 1.0        # bounded prior
    # Sources whose two dates are identical are untouched.
    zero_lag = InsiderTrade(ticker="A", trader_name="r", trader_type="institutional",
                            role="x", transaction_type="13f_increase",
                            amount_range="unknown", transaction_date=date.today(),
                            disclosure_date=date.today())
    assert _disclosure_lag_weight(zero_lag) == 1.0


def test_persisted_scores_are_not_requantized_at_3dp():
    """The PERSISTED value is what the rank transform orders on. Both scores
    were rounded to 3dp on the TickerSignal — downstream of all the continuity
    work — which silently re-merged reads that the scorers had just separated
    (insider scores cluster near zero, where 3dp is coarse). Measured live:
    the 3dp value 0.027 alone covered 17% of nonzero insider rows."""
    import ast
    import inspect
    from src.signals import aggregator

    tree = ast.parse(inspect.getsource(aggregator))
    bad = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "TickerSignal"):
            continue
        for kw in node.keywords:
            if kw.arg not in ("sentiment_score", "insider_score"):
                continue
            v = kw.value
            if (isinstance(v, ast.Call) and isinstance(v.func, ast.Name)
                    and v.func.id == "round" and len(v.args) > 1
                    and isinstance(v.args[1], ast.Constant)
                    and int(v.args[1].value) < 4):
                bad.append((kw.arg, v.args[1].value))
    assert not bad, f"persisted score re-quantized too coarsely: {bad}"


def test_insider_amount_weight_is_log_scaled():
    from src.signals.aggregator import _amount_weight_log
    small = _amount_weight_log("$1,001 - $15,000")
    mid = _amount_weight_log("$100,001 - $250,000")
    big = _amount_weight_log("Over $5,000,000")
    assert 0 < small < mid < big == 1.0
    # log, not linear: the $10M/$4k dollar ratio is ~2,500x, the weight ratio ~7x.
    assert big / small < 8
    assert _amount_weight_log("garbage") == pytest.approx(0.10)


def test_insider_role_weight():
    from src.signals.aggregator import _insider_role_weight
    assert _insider_role_weight("Chief Executive Officer") == pytest.approx(1.30)
    assert _insider_role_weight("CFO") == pytest.approx(1.30)
    assert _insider_role_weight("Director") == 1.0
    assert _insider_role_weight("") == 1.0
    ceo = _insider_score_of(role="CEO")
    dirc = _insider_score_of(role="Director")
    assert ceo > dirc


def _insider_score_of(**kw):
    from src.signals.aggregator import _insider_score
    return _insider_score("AAA", [_trade(**kw)])[0]


def test_insider_epoch_registered():
    from src.signals.method_epochs import METHOD_SCORER_EPOCH
    assert METHOD_SCORER_EPOCH["insider"] >= datetime(2026, 8, 14, tzinfo=timezone.utc)
    assert METHOD_SCORER_EPOCH["news"] >= datetime(2026, 8, 14, tzinfo=timezone.utc)


# ── news continuity ─────────────────────────────────────────────────────────

def test_evidence_scale_is_continuous_in_article_age():
    """3 fresh articles ≠ 3 stale ones — the count scale said they were equal."""
    from src.analysis.sentiment import _evidence_scale, attention_mass
    n_f, fresh_mass = attention_mass([_art(1), _art(2), _art(3)])
    n_s, stale_mass = attention_mass([_art(60), _art(70, "b"), _art(80, "c")])
    assert n_f == n_s == 3
    assert fresh_mass > stale_mass
    assert _evidence_scale(fresh_mass) > _evidence_scale(stale_mass)
    assert _evidence_scale(0.0) == 0.0
    assert _evidence_scale(1e9) == 1.0


def test_diversity_scale_smooth():
    from src.analysis.sentiment import _source_diversity_scale
    vals = [_source_diversity_scale([_art(1, s) for s in
                                     [f"src{i}" for i in range(k)]])
            for k in (1, 2, 3, 4, 6)]
    assert vals[0] == pytest.approx(0.70)
    assert vals[1] == pytest.approx(0.85)
    assert vals == sorted(vals)                 # monotone
    assert 0.90 < vals[2] < vals[3] < 1.0       # no longer a hard 1.0 at k=3


def test_sentiment_cache_key_salted_by_prompt_version(monkeypatch):
    """A prompt edit must invalidate cached raw verdicts — before the salt, the
    cache kept serving old-prompt verdicts for the TTL after a prompt change.

    Salted on the ACTIVE version, not on `_SENT_PROMPT_VERSION` specifically:
    once a variant is live (v7dir since 2026-09-11) editing the v6 constant no
    longer changes what is sent, so a test pinned to that constant would pass
    while the real invariant was broken. Patching `_prompt_pair` checks the
    property whichever prompt is in force."""
    import src.analysis.sentiment as sent
    arts = [_art(1)]
    k1 = sent._sentiment_cache_key("AAA", "deepseek", arts)
    prefix, version = sent._prompt_pair()
    monkeypatch.setattr(sent, "_prompt_pair", lambda: (prefix, version + "-bump"))
    k2 = sent._sentiment_cache_key("AAA", "deepseek", arts)
    assert k1 != k2
    # and the LIVE prompt must be the one whose version is salted in
    monkeypatch.undo()
    assert sent._prompt_pair()[1] == version


def test_sentiment_prompt_has_precision_mandate():
    from src.analysis.sentiment import _SENTIMENT_PREFIX
    assert "TWO-decimal" in _SENTIMENT_PREFIX
    assert "CROSS-SECTIONALLY" in _SENTIMENT_PREFIX


def test_sentiment_prompt_v5_zero_is_abstention():
    """v5 (2026-09-04) changed what 0.0 MEANS. v4's mandate ("when in doubt,
    output 0.0") made zero the modal verdict on 73-79% of calls, and a zero is
    an ABSTENTION in the rank-consumed combine — so every soft read silently
    removed its ticker from the 0.40-weight method. The prompt must (1) say
    so, (2) reserve 0.0 for the two abstention cases, (3) give uncertainty a
    MAGNITUDE home (the LEAN band) and (4) carry no worked example, because a
    number written into a prompt becomes a favourite answer (v2 sentiment,
    v1 confidence placement, the local model's 0.15 pile-up under v4)."""
    import re
    from src.analysis.sentiment import _SENTIMENT_PREFIX as P, _SENT_PROMPT_VERSION
    assert _SENT_PROMPT_VERSION >= "v6-"                    # v5 semantics kept by v6+
    assert "ABSTENTION" in P and "0.0 is reserved" in P
    assert "LEAN" in P and "0.01" in P                      # the soft band exists
    # No worked example: the only JSON object in the prompt is the skeleton,
    # and its score slot is a placeholder rather than a number.
    objs = re.findall(r"\{[^{}]*\}", P)
    assert len(objs) == 1, objs
    assert '"score": <' in objs[0] and not re.search(r'"score":\s*-?\d', objs[0])
    # The catalyst taxonomy still interpolates (the skeleton is not the only
    # thing the news-event dataset depends on).
    assert "__CATALYST_TYPES__" not in P and '"catalyst"' in P


def test_sentiment_prompt_v6_routine_indirect_and_funds_are_leans():
    """v6 (2026-09-04) NARROWED the abstention cases. Paired on the v5 zeros,
    three legitimate-looking families remained — routine company-specific
    items (stakes, insider sales, dividend declarations, index changes),
    sector/index FUNDS whose digest is "about the sector", and peer/customer
    read-throughs — each sent to 0.0 by v5's "score ONLY the company itself"
    clause. v6 makes each a signed LEAN (typed, so `catalyst_tilt` can learn
    its orientation) and keeps 0.0 for "nothing connects" / "nil move"."""
    from src.analysis.sentiment import _SENTIMENT_PREFIX as P
    assert "ROUTINE" in P and "never 0.0" in P
    assert "INDIRECT" in P and "read-through" in P
    assert "FUNDS:" in P and "what it holds" in P
    # v5's blanket exclusion is gone.
    assert "Score ONLY information about the target ticker itself" not in P
    # Routine items keep their class even at LEAN size.
    assert '"insider_activity"' in P and '"capital_structure"' in P and '"index_membership"' in P


def test_sentiment_target_header_names_the_company_and_flags_funds():
    """v6's per-ticker header: articles say "Antero", not "AR", and a symbol →
    company mapping is what a small local model cannot be trusted to know for
    a mid-cap; a FUND gets told its holdings ARE the company. No name ⇒ the
    bare v5 header (fail-soft)."""
    from src.data import company_names
    from src.analysis import sentiment as sent
    company_names._seed_for_tests({"AR": "Antero Resources Corp",
                                   "XLE": "Energy Select Sector SPDR Fund"})
    assert sent._target_header("AR") == "TARGET TICKER: AR — Antero Resources Corp"
    fund = sent._target_header("XLE")
    assert fund.startswith("TARGET TICKER: XLE — Energy Select Sector SPDR Fund")
    assert "FUND" in fund and "what it holds" in fund
    assert sent._target_header("ZQZX") == "TARGET TICKER: ZQZX"
    assert sent._target_header("ar") == "TARGET TICKER: AR — Antero Resources Corp"


def test_sentiment_target_header_carries_the_industry_line_and_polygon_fund_type():
    """2026-09-06: the header renders the cached Polygon industry line — the
    hook that lets a model notice "EQT AB" is not "EQT Corp" — and a target
    whose registrant name has no fund word is still flagged as a FUND when
    Polygon types it as one (``SPDR Gold Shares``). The override reads the
    SAME identity as the header, so the two cannot disagree; an unknown
    industry (the 7-day negative record) leaves the line out, fail-soft."""
    from src.data import company_names
    from src.analysis import sentiment as sent
    company_names._seed_for_tests(
        {"AR": "Antero Resources Corp", "GLD": "SPDR Gold Shares",
         "XLE": "Energy Select Sector SPDR Fund", "EQT": "EQT Corp"},
        industries={"AR": "Crude Petroleum & Natural Gas",
                    "GLD": "exchange-traded fund",
                    "EQT": "Natural Gas Transmission"})
    assert sent._target_header("AR") == (
        "TARGET TICKER: AR — Antero Resources Corp (industry: Crude Petroleum & Natural Gas)")
    assert sent._target_header("eqt").startswith(
        "TARGET TICKER: EQT — EQT Corp (industry: Natural Gas Transmission)")
    assert "FUND" not in sent._target_header("EQT")
    gld = sent._target_header("GLD")
    assert gld.startswith("TARGET TICKER: GLD — SPDR Gold Shares (industry: exchange-traded fund)")
    assert "FUND" in gld and "what it holds" in gld
    # Name-word fund with no industry record: flagged, no industry parenthesis.
    xle = sent._target_header("XLE")
    assert xle.startswith("TARGET TICKER: XLE — Energy Select Sector SPDR Fund\n")
    assert "(industry:" not in xle
    # The override sees exactly what the header rendered.
    assert sent._is_fund_target("GLD") and sent._is_fund_target("XLE")
    assert not sent._is_fund_target("AR") and not sent._is_fund_target("ZQZX")
    # The industry line is part of the salt: resolving it re-scores the ticker once.
    arts = [_art(1)]
    k_bare = sent._sentiment_cache_key("AR", "deepseek", arts,
                                       extra="TARGET TICKER: AR — Antero Resources Corp")
    k_ind = sent._sentiment_cache_key("AR", "deepseek", arts, extra=sent._target_header("AR"))
    assert k_bare != k_ind


def test_fund_catalyst_override_retypes_company_events_on_fund_targets(monkeypatch):
    """A fund has no earnings, insiders, trials or offerings of its own, so a
    company-event class on a fund target is a read-through of its holdings —
    ``macro_sector`` by the taxonomy's own rule. Label-only, deterministic,
    inert on ``none``/``macro_sector``/missing labels, on non-funds, and with
    the flag off."""
    from config.settings import settings
    from src.data import company_names
    from src.analysis import sentiment as sent
    company_names._seed_for_tests({"XLV": "Health Care Select Sector SPDR Fund",
                                   "GLD": "SPDR Gold Shares", "LLY": "Eli Lilly & Co"},
                                  industries={"GLD": "exchange-traded fund"})
    monkeypatch.setattr(settings, "enable_catalyst_fund_override", True, raising=False)
    assert sent.fund_catalyst_override("XLV", "fda_clinical") == "macro_sector"
    assert sent.fund_catalyst_override("GLD", "earnings") == "macro_sector"
    assert sent.fund_catalyst_override("xlv", "insider_activity") == "macro_sector"
    assert sent.fund_catalyst_override("XLV", "none") == "none"
    assert sent.fund_catalyst_override("XLV", "macro_sector") == "macro_sector"
    assert sent.fund_catalyst_override("XLV", None) is None
    assert sent.fund_catalyst_override("LLY", "fda_clinical") == "fda_clinical"
    assert sent.fund_catalyst_override("ZQZX", "earnings") == "earnings"
    monkeypatch.setattr(settings, "enable_catalyst_fund_override", False, raising=False)
    assert sent.fund_catalyst_override("XLV", "fda_clinical") == "fda_clinical"


def test_sentiment_cache_key_salted_by_target_header():
    """The header is part of the prompt, so a name resolving later (or a fund
    flag appearing) must re-score rather than serve the bare-header verdict."""
    import src.analysis.sentiment as sent
    arts = [_art(1)]
    k0 = sent._sentiment_cache_key("AR", "deepseek", arts)
    k1 = sent._sentiment_cache_key("AR", "deepseek", arts, extra="TARGET TICKER: AR")
    k2 = sent._sentiment_cache_key("AR", "deepseek", arts,
                                   extra="TARGET TICKER: AR — Antero Resources Corp")
    assert len({k0, k1, k2}) == 3


# ── news_shock ──────────────────────────────────────────────────────────────

def test_news_shock_scoring():
    from src.signals.news_shock import compute_news_shock
    # 8x normal attention with bullish news → full +1.
    assert compute_news_shock(0.5, 8.0, 1.0) == pytest.approx(1.0)
    # 2x → log2(2)/3 = 1/3, signed by the news read.
    assert compute_news_shock(0.5, 2.0, 1.0) == pytest.approx(1 / 3, abs=1e-3)
    assert compute_news_shock(-0.5, 2.0, 1.0) == pytest.approx(-1 / 3, abs=1e-3)
    # Abstentions: quiet is not a fade, no news is no direction, no baseline is no verdict.
    assert compute_news_shock(0.5, 0.9, 1.0) == 0.0     # below baseline
    assert compute_news_shock(0.0, 8.0, 1.0) == 0.0     # no news direction
    assert compute_news_shock(0.5, 8.0, None) == 0.0    # no baseline yet
    assert compute_news_shock(0.5, 0.0, 1.0) == 0.0     # no articles


def test_news_shock_baseline_failsoft(monkeypatch):
    """A DB failure reads as 'no baseline' — the method abstains, never raises."""
    import src.signals.news_shock as ns
    ns.reset_cache()
    from src.db import repo

    def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(repo, "fetch_df", boom)
    assert ns.load_attention_baselines(force=True) == {}
    ns.reset_cache()


def test_news_shock_wiring_complete():
    """The full add-method checklist, mechanically."""
    from src.analysis.code_version import METHOD_SOURCES, unmapped_methods
    from src.db.schema import (SIGNAL_BASE_METHOD_COLUMNS,
                               SIGNAL_NEWS_ATTENTION_COLUMNS, _ADD_COLUMNS)
    from src.models import TickerSignal
    from src.performance.tracker import _ALL_METHODS, METHOD_CATEGORIES, METHOD_LABELS
    from src.signals.agreement import FAMILY_OF
    from src.signals.aggregator import _BASE_WEIGHTS

    assert "news_shock" in _ALL_METHODS
    assert "news_shock" in SIGNAL_BASE_METHOD_COLUMNS
    assert "news_shock" in METHOD_CATEGORIES["Sentiment"]
    assert "news_shock" in METHOD_LABELS
    assert "news_shock" in METHOD_SOURCES and not unmapped_methods()
    migrated = {(t, c) for t, c, _ in _ADD_COLUMNS}
    assert ("signals", "news_shock") in migrated
    for c in SIGNAL_NEWS_ATTENTION_COLUMNS:
        assert ("signals", c) in migrated
        assert c in TickerSignal.model_fields
    assert "news_shock_score" in TickerSignal.model_fields
    # PANEL-FIRST: not weighted, not a family voter — mirrors squeeze.
    # Promoted 2026-09-11 (user request) to the SMALLEST weight in the book.
    # It reaches coherence / `sources_agreeing` / the Sentiment family vote —
    # i.e. confidence and position size — and not direction, which the stackers
    # decide. Smallest because, unlike news_bear_fresh (t +2.96) and
    # catalyst_tilt (t +2.22), it has NO measured IC at all.
    assert 0 < _BASE_WEIGHTS["news_shock"] <= 0.05
    # Joined the Sentiment family on 2026-09-11 with its weight: a weight-0
    # method is excluded from the family vote entirely, and participating in it
    # is the stated reason for the promotion. Sentiment rather than a family of
    # its own — it is a function of the same verdict as `news`, and the family
    # layer exists so correlated methods are ONE voter (the family COUNT is
    # unchanged at 7).
    assert FAMILY_OF["news_shock"] == "Sentiment"


# ── source-tier filter (2026-09-08) ─────────────────────────────────────────

def test_source_tier_drops_aggregators_but_never_empties_a_digest(monkeypatch):
    """The mechanism is kept and tested even though it ships OFF: it failed its
    12-day paired re-test (t -0.55, opposite-sign halves) after a one-day read
    of +0.256, so the code stays one flag from live and repeatable.

    The FALLBACK is the load-bearing half. An empty digest is an ABSTENTION that
    removes the ticker from the cross-section — a bigger change than a thinner
    digest, and the failure the 2026-09-04 relevance rework ended."""
    from datetime import datetime, timezone
    from config.settings import Settings, settings
    from src.analysis.sentiment import apply_source_tier
    from src.models import NewsArticle
    now = datetime.now(timezone.utc)

    def art(src, i):
        return NewsArticle(title="t", summary="s" * 40, url=f"u{i}", source=src,
                           published_at=now)

    # ships off — a fresh environment must not inherit a rejected filter
    assert Settings.model_fields["enable_source_tier_filter"].default is False
    monkeypatch.setattr(settings, "enable_source_tier_filter", True, raising=False)
    mixed = [art("Motley Fool", 1), art("Reuters", 2), art("Zacks", 3)]
    assert [a.source for a in apply_source_tier(mixed)] == ["Reuters"]
    # every article is an aggregator -> keep them all rather than abstain
    only_agg = [art("Motley Fool", 1), art("Zacks", 2)]
    assert len(apply_source_tier(only_agg)) == 2
    assert apply_source_tier([]) == []


def test_source_tier_runs_before_the_top_20_cut():
    """Filtering after the cut would let listicles consume the 20 slots and then
    be removed, leaving a digest thinner than it needed to be."""
    import inspect
    from src.analysis import sentiment as sent
    src = inspect.getsource(sent.analyse_sentiment)
    assert "digest_articles(apply_source_tier(fresh_articles), as_of)" in src


def test_the_news_family_shares_one_boundary():
    """The derived methods consume the verdict, so they move together
    (CLAUDE.md's news-family rule) — whatever the boundary is.

    It sat at 2026-09-08 briefly for the source-tier filter; that filter failed
    its 12-day paired re-test and was defaulted off, so the boundary went back
    to the v6 deploy. What is pinned is the SHARING, not the date."""
    from src.signals.method_epochs import METHOD_SCORER_EPOCH as E
    fam = ("news", "sent_velocity", "news_shock", "news_bear_fresh", "catalyst_tilt")
    assert len({E[m] for m in fam}) == 1, {m: E[m].isoformat() for m in fam}


# ── passing-mention abstention (2026-09-10) ─────────────────────────────────

def test_a_digest_of_passing_mentions_abstains_without_an_llm_call(monkeypatch):
    """An article naming the target only in its BODY is a round-up or an
    ETF-holdings piece — it MENTIONS the company without being ABOUT it.

    Measured over 655 ticker-days: by passing-mention share, oriented pivot
    return goes +0.903 pp (hit 54.5%) below 20%, -0.573 pp (48.1%) at 20-50%,
    and -2.223 pp (41.9%) at >=50%. The high cohort is ANTI-predictive, not
    merely weak, which is why this abstains instead of capping the magnitude:
    capping would preserve a sign the data says is wrong.
    """
    from datetime import datetime, timedelta, timezone

    import src.analysis.sentiment as sent
    from src.models import NewsArticle
    now = datetime.now(timezone.utc)

    def art(title):
        return NewsArticle(title=title, summary="AbbVie is among them.",
                           source="Motley Fool", url=title[:12],
                           published_at=now - timedelta(hours=2))

    from config.settings import settings
    monkeypatch.setattr(settings, "enable_passing_mention_abstention", True, raising=False)
    called = []
    monkeypatch.setattr(sent, "_provider_sentiment_score",
                        lambda *a, **k: called.append("provider") or None)
    arts = [art("Why Wall Street Cannot Get Enough of This Dividend King"),
            art("3 Dividend Stocks I Would Buy Right Now"),
            art("Meet the Vanguard ETF That Crushed the S&P 500")]
    score, rationale, meta = sent.analyse_sentiment("ABBV", arts)
    assert score == 0.0 and "passing" in rationale.lower()
    assert not called, "the abstention must short-circuit BEFORE any scoring path"


def test_a_headline_digest_is_untouched():
    """The other ~98% of digests must be byte-identical — this is a narrow
    relevance refinement, not a change to what a verdict means."""
    from datetime import datetime, timedelta, timezone

    import src.analysis.sentiment as sent
    from src.models import NewsArticle
    now = datetime.now(timezone.utc)
    arts = [NewsArticle(title=f"AbbVie wins FDA approval {i}", summary="s",
                        source="Reuters", url=f"u{i}",
                        published_at=now - timedelta(hours=2))
            for i in range(3)]
    assert sent._passing_mention_share("ABBV", arts) == 0.0


def test_the_share_helper_fails_soft():
    """A broken name lookup must never trigger an abstention: a silent zero
    inside the 0.40-weight `news` method is exactly what the 2026-09-04
    relevance rework existed to end."""
    import src.analysis.sentiment as sent
    assert sent._passing_mention_share("ABBV", []) is None
    assert sent._passing_mention_share("ABBV", None) is None


def test_a_ticker_the_name_matcher_cannot_see_is_SKIPPED(monkeypatch):
    """THE bug this nearly shipped with. `mention_evidence` is blind to a
    company whose registrant name is an ordinary word (ARM -> phrases [],
    tokens []), so every one of its headlines reads as a passing mention, the
    share reads 1.00, and the rule would abstain on that ticker ALWAYS — not
    because its coverage is round-ups but because the matcher cannot see it.

    Measured: 6 of 130 tickers, 21 of 655 rows, and abstaining on them wholesale
    scores -0.0048 IC. Guarding them out drops the shipped effect from +0.0093
    (t +2.09) to +0.0075 (t +1.81) — the honest number."""
    from datetime import datetime, timedelta, timezone

    import src.analysis.sentiment as sent
    from src.models import NewsArticle
    now = datetime.now(timezone.utc)
    arts = [NewsArticle(title=f"Some market story {i}", summary="s", source="Zacks",
                        url=f"u{i}", published_at=now - timedelta(hours=2))
            for i in range(3)]
    monkeypatch.setattr("src.data.company_names.name_keywords", lambda t: {})
    assert sent._passing_mention_share("ARM", arts) is None
    monkeypatch.setattr("src.data.company_names.name_keywords",
                        lambda t: {"phrases": ["acme corp"], "tokens": ["acme"],
                                   "symbol_word": True, "name": "Acme Corp",
                                   "fund": False})
    assert sent._passing_mention_share("ACME", arts) == 1.0


def test_the_shipped_threshold_fires_at_the_measured_RATE():
    """The measured effect is "gate the ~13% of digests with the highest
    passing-mention share" (+0.0242 IC, t +1.31, same-sign halves). The
    THRESHOLD that produces that rate depends on the digest population, and the
    tuning population (replayed 7-day union pools) is not the live one: 0.35
    fires on 13% there and 35.4% in production. What must be pinned is the RATE,
    so the constant is checked against the live-calibrated value."""
    from config.settings import Settings
    assert Settings.model_fields["passing_mention_abstain_share"].default == 0.75


def test_a_wide_abstention_carries_a_news_family_EPOCH():
    """At 0.65 the filter touched ~2% of digests and needed no boundary: a weak
    read becoming an ABSTENTION is a state the panel already treats as "no
    view", and the other 98% of scores were byte-identical.

    At 0.35 it touches 13% — an eighth of the cross-section stops entering the
    news rank — so a calibration pooling both sides of the deploy would be
    fitting two different populations. Below 0.5 the epoch is MANDATORY, and the
    whole news family shares ONE boundary (CLAUDE.md's news-family rule)."""
    from config.settings import settings
    from src.signals.method_epochs import METHOD_SCORER_EPOCH
    share = float(getattr(settings, "passing_mention_abstain_share", 0.65))
    if share >= 0.5:
        return
    fam = ["news", "sent_velocity", "news_shock", "news_bear_fresh", "catalyst_tilt"]
    stamps = {m: METHOD_SCORER_EPOCH.get(m) for m in fam}
    assert all(stamps.values()), f"news family missing an epoch: {stamps}"
    assert len({str(v) for v in stamps.values()}) == 1,         f"the news family must share ONE boundary, got {stamps}"
    assert str(stamps["news"])[:10] >= "2026-09-10",         "the epoch predates the abstention widening it exists for"



def test_the_abstain_threshold_is_calibrated_on_LIVE_digest_shape():
    """The 0.35 threshold was tuned on REPLAYED digests (7-day union pools, a
    deliberately unfaithful reconstruction) and over-fired 2.7x in production:
    35.4% of live digests against the 13% the +0.0242 IC was measured at.

    The live fire rate by threshold, from 3,316 rows of `sentiment_digests`:
        >= 0.35  35.4% | >= 0.50  29.0% | >= 0.65  17.2%
        >= 0.75  12.3% | >= 0.85   7.7% | >= 1.00   6.0%

    0.75 is the constant whose LIVE rate matches the measured cohort. This test
    pins the lower bound so the threshold cannot drift back into over-reach
    without someone re-deriving it from live digests."""
    from config.settings import settings
    share = float(getattr(settings, "passing_mention_abstain_share", 0.75))
    assert share >= 0.65, (
        f"{share} fires on far more than the ~13% of live digests the effect was "
        f"measured at — re-derive the rate from `sentiment_digests` before lowering it")
