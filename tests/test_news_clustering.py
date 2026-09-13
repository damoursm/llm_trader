"""Story clustering: group a digest by what the articles SAY, not when they ran.

THE DEFECT (measured 2026-09-09, 146 re-scored clusters / 38 ticker-days): the
relative time-gap partition cuts one running story into pieces and the scorer
judges each piece as an independent event — 44% of multi-cluster ticker-days had
every cluster carrying the same catalyst class, and 44% took opposite signs.
`AGIO 2026-07-10` scored the same FDA priority review -0.15 in one piece and
+0.25 in the other.

These tests pin the design decisions that make content clustering work at all —
above all the one the first implementation got backwards.
"""
from datetime import datetime, timedelta, timezone

import pytest

from src.analysis import news_clustering as nc
from src.models import NewsArticle

NOW = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)


def _a(hours_ago, title, summary="", source="Reuters"):
    return NewsArticle(title=title, summary=summary, source=source,
                       url=f"u{hours_ago}{title[:8]}",
                       published_at=NOW - timedelta(hours=hours_ago))


@pytest.fixture(autouse=True)
def _clear_corpus():
    nc._CORPUS["df"], nc._CORPUS["n"] = None, 0
    yield
    nc._CORPUS["df"], nc._CORPUS["n"] = None, 0


def _corpus(n=200):
    """A background pool where 'stock'/'shares'/'earnings' are everywhere and the
    story words are rare — what the live tick's several-hundred-article pool
    looks like."""
    filler = [_a(50 + i, f"Stock market update {i} shares earnings outlook",
                 "Shares of various companies moved on earnings and market news.")
              for i in range(n)]
    nc.set_corpus(filler)


# ── the decision the first attempt got backwards ────────────────────────────

def test_idf_must_come_from_the_pool_not_the_digest():
    """Within-digest IDF zeroes exactly the terms that identify a shared story:
    the words appearing in several of the digest's articles ARE the story. On
    the real `AGIO 2026-07-10` digest that scored two unmistakably-same-story
    articles at 0.04 and split nine articles into nine clusters.

    Pinned on the mechanism: a term shared by every digest article must keep a
    NON-ZERO weight when the corpus says it is rare in the world.
    """
    _corpus()
    digest = [_a(1, "Mitapivat wins FDA priority review"),
              _a(3, "FDA grants priority review to mitapivat")]
    vecs = nc._weights(digest, frozenset())
    assert vecs[0].get("mitapivat", 0) > 0, "pool-rare shared term was zeroed"
    assert vecs[0].get("fda", 0) > 0
    # and the pool-common words are down-weighted relative to the rare ones
    common = nc._weights([_a(1, "Stock shares earnings outlook mitapivat")], frozenset())[0]
    assert common["mitapivat"] > common["earnings"]


def test_without_a_corpus_it_falls_back_to_flat_weights_not_digest_idf():
    """The fallback must be safe, not wrong. Digest IDF is the wrong answer, so
    an absent corpus gets flat weights plus the stoplist."""
    assert nc.corpus_size() == 0
    vecs = nc._weights([_a(1, "Mitapivat priority review"),
                        _a(2, "Mitapivat priority review")], frozenset())
    assert set(vecs[0].values()) == {1.0}


def test_the_targets_own_name_is_stripped(monkeypatch):
    """A company's name is in every article of its digest by construction, and
    pool IDF cannot remove it (a company name is pool-RARE). Left in, it merges
    the whole digest into one story.

    The name is monkeypatched rather than resolved: this pins the STRIPPING, not
    the availability of the registrant list — see the test below for what
    happens when the name cannot be resolved.
    """
    _corpus()
    monkeypatch.setattr("src.data.company_names.name_keywords",
                        lambda t: {"phrases": ["agios pharmaceuticals"], "tokens": ["agios"]})
    arts = [_a(1, "Agios Pharmaceuticals wins FDA priority review"),
            _a(2, "Agios Pharmaceuticals names new chief financial officer")]
    with_name = nc.content_groups(arts, threshold=0.25)
    stripped = nc.content_groups(arts, threshold=0.25, ticker="AGIO")
    assert len(stripped) == 2, "two unrelated stories merged on the company name"
    assert len(with_name) == 1, "fixture no longer exercises the name collision"


def test_an_unresolved_company_name_over_merges_and_that_is_known(monkeypatch):
    """Content mode DEPENDS on `company_names.name_keywords`. When the name
    cannot be resolved (an unknown symbol, a cold cache) only the bare ticker is
    stripped, the registrant name stays shared across the digest, and unrelated
    stories merge.

    That is a fidelity loss, not a crash, and it is why `news_cluster_mode`
    defaults to "time": the failure is silent, so it must be a known and tested
    property before content mode carries anything.
    """
    _corpus()
    monkeypatch.setattr("src.data.company_names.name_keywords", lambda t: {})
    arts = [_a(1, "Agios Pharmaceuticals wins FDA priority review"),
            _a(2, "Agios Pharmaceuticals names new chief financial officer")]
    assert len(nc.content_groups(arts, threshold=0.25, ticker="AGIO")) == 1
    assert nc._target_tokens("AGIO") == frozenset({"agio"})


def test_similarity_is_overlap_not_cosine():
    """Digest articles differ wildly in length — a 25-token blurb against an
    88-token feature — and cosine scores that asymmetry rather than the content.
    A short article fully contained in a long one must score high."""
    short = {"mitapivat": 3.0, "priority_review": 3.0}
    long = dict(short, **{f"filler{i}": 1.0 for i in range(30)})
    assert nc._overlap(short, long) == pytest.approx(1.0)
    assert nc._overlap(short, {"unrelated": 3.0}) == 0.0


# ── the grouping ────────────────────────────────────────────────────────────

def test_one_running_story_becomes_one_cluster_across_days():
    """The defect itself. Time would cut this into two clusters (a >24h gap);
    content must keep it as one, because that is what it is."""
    _corpus()
    arts = [_a(2, "Analysts weigh mitapivat priority review for sickle cell disease"),
            _a(4, "FDA grants mitapivat priority review in sickle cell disease"),
            _a(80, "Mitapivat sickle cell disease filing accepted with priority review")]
    time_groups = nc.cluster_articles(arts, mode="time", as_of=NOW)
    content = nc.cluster_articles(arts, mode="content", ticker="AGIO")
    assert len(time_groups) == 2, "fixture no longer exercises the time split"
    assert len(content) == 1, [[x.title for x in g] for g in content]


def test_unrelated_stories_stay_apart():
    """Merging everything would be the opposite failure and would be worse:
    the whole digest would carry one anchor."""
    _corpus()
    arts = [_a(1, "FDA grants mitapivat priority review in sickle cell disease"),
            _a(2, "Company prices $500 million convertible notes offering"),
            _a(3, "Q3 revenue rose 12% on strong volumes")]
    groups = nc.cluster_articles(arts, mode="content", ticker="AGIO")
    assert len(groups) == 3


def test_syndicated_copies_merge():
    """The easiest true positive, and the one the time rule already gets right
    only by accident (they run minutes apart)."""
    _corpus()
    t = "Here is why Agios Pharmaceuticals stock soared today on its FDA catalyst"
    arts = [_a(1, t, source="Motley Fool"), _a(30, t, source="The Motley Fool")]
    assert len(nc.cluster_articles(arts, mode="content", ticker="AGIO")) == 1


def test_groups_come_back_freshest_first():
    """Every consumer assumes it — `cluster_bounds[0]` is "the freshest story",
    which is what `news_unpriced` anchors on."""
    _corpus()
    arts = [_a(90, "FDA grants mitapivat priority review sickle cell"),
            _a(1, "Company prices convertible notes offering")]
    groups = nc.cluster_articles(arts, mode="content", ticker="AGIO")
    assert groups[0][0].published_at > groups[-1][0].published_at


# ── safety ──────────────────────────────────────────────────────────────────

def test_time_mode_is_byte_identical_to_the_legacy_partition():
    """`mode="time"` must reproduce the old behaviour exactly — the shipped
    `news_quiet` depends on it, and the whole 50-day measurement was made on
    it."""
    from src.analysis.sentiment import recent_cluster
    arts = [_a(h, f"Story {h}") for h in (1, 2, 3, 40, 90, 140)]
    got = nc.cluster_articles(arts, mode="time", as_of=NOW)
    rest, expected = list(arts), []
    while rest:
        g = recent_cluster(rest, as_of=NOW)
        if not g:
            break
        expected.append(g)
        keep = {id(x) for x in g}
        rest = [x for x in rest if id(x) not in keep]
    assert [[a.url for a in g] for g in got] == [[a.url for a in g] for g in expected]


def test_content_mode_is_not_the_default():
    """Rebuilding the partition from content was MEASURED to make the defect
    worse (15.8% -> 77.5% at 0.25, 65.2% at 0.10, over 73 ticker-days scored on
    the same digests). It stays reachable for re-testing and must never be the
    default again."""
    from config.settings import Settings
    assert Settings.model_fields["news_cluster_mode"].default in ("time", "hybrid")


def test_every_content_aware_mode_installs_the_pool_corpus():
    """Without a corpus the module falls back to FLAT term weights and merges on
    the wrong evidence — silently. Gating the install on "content" alone was a
    live defect the moment `hybrid` shipped."""
    import inspect

    import src.signals.aggregator as agg
    src = inspect.getsource(agg.build_signals)
    i = src.index("set_corpus(")
    gate = src[max(0, i - 400):i]
    for mode in ("content", "hybrid"):
        assert f'"{mode}"' in gate, f"{mode} does not install the clustering corpus"


def test_a_content_failure_falls_back_to_the_time_partition(monkeypatch):
    """A clustering bug must cost fidelity, never a tick."""
    _corpus()
    monkeypatch.setattr(nc, "content_groups",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    arts = [_a(h, f"Story {h}") for h in (1, 2, 90)]
    groups = nc.cluster_articles(arts, mode="content", as_of=NOW)
    assert len(groups) == 2                      # the time partition


def test_degenerate_inputs_never_raise():
    _corpus()
    assert nc.cluster_articles([], mode="content") == []
    assert nc.cluster_articles(None, mode="content") == []
    one = [_a(1, "Only article")]
    assert len(nc.cluster_articles(one, mode="content")) == 1
    assert nc.content_groups([_a(1, ""), _a(2, "")]) is not None


def test_news_quiet_pins_the_time_rule():
    """`news_quiet`'s measured quantity is the age of the last coverage BURST.
    A content cluster spanning a week would make a name with news TODAY read as
    quiet, inverting the method — so it pins the mode instead of inheriting the
    setting."""
    import inspect

    import src.signals.news_quiet as nq
    src = inspect.getsource(nq.freshest_cluster_age_hours)
    assert 'mode="time"' in src

    _corpus()
    arts = [_a(1, "FDA grants mitapivat priority review sickle cell disease"),
            _a(100, "Mitapivat priority review sickle cell disease filing accepted")]
    # content would call this ONE story starting 100h ago; time must not
    age = nq.freshest_cluster_age_hours(arts, as_of=NOW)
    assert age is not None and age < 24, age


# ── hybrid: time partition, same-story pieces merged, never split ───────────

def test_hybrid_merges_time_pieces_of_one_story():
    """The defect proper: a story cut into two TIME pieces that then contradict
    each other. Hybrid asks content the one question it can answer — "are these
    two pieces the same story?" — and never re-partitions."""
    _corpus()
    arts = [_a(2, "Analysts weigh mitapivat priority review for sickle cell disease"),
            _a(4, "FDA grants mitapivat priority review in sickle cell disease"),
            _a(80, "Mitapivat sickle cell disease filing accepted with priority review")]
    assert len(nc._time_groups(arts, as_of=NOW)) == 2
    assert len(nc.hybrid_groups(arts, threshold=0.25, ticker="AGIO", as_of=NOW)) == 1


def test_hybrid_leaves_genuinely_different_stories_split():
    _corpus()
    arts = [_a(2, "Company prices $500 million convertible notes offering"),
            _a(80, "FDA grants mitapivat priority review in sickle cell disease")]
    assert len(nc.hybrid_groups(arts, threshold=0.25, ticker="AGIO", as_of=NOW)) == 2


def test_hybrid_can_only_REDUCE_the_cluster_count():
    """The property that makes hybrid safe where rebuilding from content was
    not: it never splits, so it cannot shatter a digest into singletons — the
    mechanism that made content mode's contradiction rate explode."""
    _corpus()
    import random
    random.seed(3)
    words = ["fda review", "notes offering", "revenue rose", "chief financial",
             "price target", "index inclusion"]
    for _ in range(20):
        arts = [_a(random.randint(1, 150), random.choice(words) + f" {random.randint(0,3)}")
                for _ in range(random.randint(3, 9))]
        t = len(nc._time_groups(arts, as_of=NOW))
        h = len(nc.hybrid_groups(arts, threshold=0.25, ticker="AGIO", as_of=NOW))
        assert h <= t, (h, t)


def test_hybrid_is_a_noop_on_a_single_time_cluster():
    _corpus()
    arts = [_a(1, "FDA review"), _a(2, "Notes offering"), _a(3, "Revenue rose")]
    assert len(nc._time_groups(arts, as_of=NOW)) == 1
    assert len(nc.hybrid_groups(arts, threshold=0.25, ticker="AGIO", as_of=NOW)) == 1


def test_hybrid_is_reachable_through_the_mode_switch(monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "news_cluster_mode", "hybrid", raising=False)
    assert nc.cluster_mode() == "hybrid"
    _corpus()
    arts = [_a(2, "FDA grants mitapivat priority review sickle cell disease"),
            _a(80, "Mitapivat priority review sickle cell disease filing accepted")]
    assert len(nc.cluster_articles(arts, ticker="AGIO", as_of=NOW)) == 1
    monkeypatch.setattr(settings, "news_cluster_mode", "nonsense", raising=False)
    assert nc.cluster_mode() == "time"       # unknown value must not silently cluster


def test_no_corpus_means_no_merge():
    """Flat weights are a safe REPRESENTATION fallback — they are at least not
    digest IDF, which is actively wrong — but they are NOT a safe basis for a
    merge DECISION: every shared common word counts in full, so "shares fell
    today" matches "shares rose today" and the merge fires on boilerplate.

    A missing corpus therefore degrades to the TIME partition. Caught by four
    existing fixtures the moment `hybrid` became the default: they build
    digests with identical placeholder titles and no corpus, and hybrid merged
    them on that boilerplate.
    """
    assert nc.corpus_size() == 0
    arts = [_a(1, "headline"), _a(3, "headline"), _a(96, "headline"),
            _a(100, "headline"), _a(101, "headline")]
    # identical text + flat weights would merge everything; the guard must not
    assert len(nc.cluster_articles(arts, mode="hybrid", as_of=NOW)) == 2
    assert len(nc.cluster_articles(arts, mode="content", as_of=NOW)) == 2
    # WITH a corpus, identical text is a genuine merge (syndicated copies)
    _corpus()
    assert len(nc.cluster_articles(arts, mode="hybrid", ticker="AGIO", as_of=NOW)) == 1
