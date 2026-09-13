"""Group a digest's articles by the STORY they are about, not by when they ran.

THE DEFECT (measured 2026-09-09, 146 re-scored clusters over 38 days with
rationales kept). The digest is partitioned by a relative TIME GAP
(`sentiment.recent_cluster` applied repeatedly). On a story that runs for days
that rule cuts one story into several pieces and the scorer then judges each
piece as an independent event:

* **44%** of multi-cluster ticker-days had every cluster carrying the SAME
  catalyst class — one story, cut up.
* **44%** of them had clusters taking OPPOSITE signs.

`AGIO 2026-07-10` is the clean example: the FDA priority review for mitapivat is
in both clusters, and the model scored the fresher piece -0.15 ("already largely
priced in, stock up 13.8%") and the older piece +0.25 ("strong catalyst... a
minor lean upwards"). Same event, two verdicts, opposite signs, because the
partition told the scorer they were two events.

THE FIX (user directive 2026-09-09, "it'd be best if they were clustered based
on the catalyst or content"). Group on what the articles SAY. The catalyst class
itself is not available — it is an OUTPUT of the scoring this clustering feeds —
so content is the usable proxy, and it is the right one: syndicated copies and
follow-ups of one story share their distinctive vocabulary.

HOW. Unigrams + bigrams of the title and the head of the summary, weighted by
IDF taken from the TICK'S WHOLE ARTICLE POOL, scored by IDF-weighted overlap,
single-link connected components above `news_cluster_min_similarity`.

Four properties are deliberate, and the first is the one the first attempt got
backwards:

* **IDF comes from the POOL, never from the digest.** Weighting by
  within-digest IDF is the natural-looking choice and it is exactly wrong here:
  the terms that identify a shared story are, by definition, the ones appearing
  in several of that digest's articles, so digest-IDF drives their weight to
  zero and leaves each article's unique boilerplate dominating. Measured on
  `AGIO 2026-07-10`: articles sharing `mitapivat`, `priority_review`,
  `sickle_cell` and `fda` scored a cosine of **0.04**, and the whole digest
  split into nine one-article clusters. Pool IDF asks the right question —
  `mitapivat` is rare across today's several-hundred-article pool while
  `stock`, `shares` and `earnings` are everywhere. Without a corpus the module
  falls back to flat weights plus the stoplist rather than to digest IDF.
* **The target's own name is stripped.** It appears in every article of its
  digest by construction, and pool IDF will not remove it (a company name is
  pool-RARE), so leaving it in merges the whole digest into one story.
* **OVERLAP, not cosine.** Digest articles differ wildly in length — a
  25-token Stocktwits blurb against an 88-token Motley Fool piece — and cosine
  punishes that asymmetry rather than the content. Shared IDF mass over the
  SMALLER article's mass asks "is the short one contained in the long one",
  which is what a syndicated follow-up actually looks like.
* **SINGLE-link, not average-link.** A story genuinely chains: the follow-up
  shares vocabulary with the original, the analyst reaction shares it with the
  follow-up, and the ends may share little. Chaining is the correct behaviour
  here and over-splitting is the defect being fixed. The threshold bounds it.
* **Time does NOT constrain a merge.** The whole point is that a story running
  for a week is ONE story; re-imposing a gap would re-create the defect.

RESULT (2026-09-09/10, 73 ticker-days, every arm scored on the SAME digests).
Rebuilding the partition from content FAILED: it made the defect worse, not
better, because a 13-article digest holds ~8-12 lexically distinct items and
re-partitioning shatters it into singletons that each carry too little context
for a stable read. `hybrid` — the time partition with same-story pieces MERGED,
never split — is the design that survives. See the mode table in CLAUDE.md.

SCOPE — and one method deliberately excluded. `news_quiet` (shipped 2026-09-09)
scores on the age of the freshest cluster, and its +1.90 pp/decision was
measured with the TIME rule, where that age means "hours since the last burst of
coverage began". Under content clustering the same field means something else: a
story that broke a week ago and got a fresh article this morning becomes ONE
cluster starting a week ago, so a name with news TODAY would read as quiet. That
inverts the method. `news_quiet` therefore pins `mode="time"` explicitly and
says so at the call site — it measures news FLOW, which is a time concept, while
`news_unpriced` measures a STORY's anchor, which is a content concept.

Fail-soft: any error returns the time partition, so a clustering bug costs
fidelity, never a tick.
"""

import math
import re
from collections import Counter
from datetime import datetime
from typing import List, Optional, Sequence

from loguru import logger

from config.settings import settings

# Only the words per-digest IDF cannot down-weight on its own: a digest of two
# articles has almost no IDF signal, and these carry no story identity anywhere.
_STOP = frozenset("""
a an the and or but if then than that this these those of in on at to for from by with
about into over after before under above is are was were be been being has have had
do does did will would can could should may might must not no nor so such as it its
he she they them his her their you your we our i me my
stock stocks share shares market markets price prices investor investors company companies
inc corp corporation ltd plc co group holdings today yesterday week month year years
new news report reports says said according update updated more most best top why how
what when where who buy sell hold buying selling analyst analysts wall street nasdaq nyse
""".split())

_WORD = re.compile(r"[a-z0-9][a-z0-9'&.-]{1,}")
_SUMMARY_CHARS = 400        # the lede carries the story; the tail is boilerplate
_MAX_ARTICLES = 40          # O(n^2); a digest is capped at 20 well below this

# Pool document frequencies, set once per tick by `set_corpus`. Module state on
# purpose: threading the pool through `cluster_bounds` -> every consumer would
# touch six call sites for a value that is constant across a tick, and the
# fallback when it is absent is safe (flat weights) rather than wrong.
_CORPUS: dict = {"df": None, "n": 0}


def set_corpus(articles: Sequence) -> int:
    """Install the tick's whole article pool as the IDF corpus. Returns the
    document count. Call once per run, before any clustering."""
    docs = 0
    df: Counter = Counter()
    for a in (articles or []):
        toks = set(_tokens(a))
        if toks:
            docs += 1
            df.update(toks)
    _CORPUS["df"], _CORPUS["n"] = (df, docs) if docs >= 20 else (None, 0)
    return _CORPUS["n"]


def corpus_size() -> int:
    return int(_CORPUS["n"] or 0)


def _tokens(article, drop: frozenset = frozenset()) -> List[str]:
    """Unigrams + bigrams of the title and the head of the summary.

    Bigrams matter more than usual here: "priority review", "class action",
    "price target" identify a story where their halves do not.
    """
    title = str(getattr(article, "title", "") or "")
    summary = str(getattr(article, "summary", "") or "")[:_SUMMARY_CHARS]
    words = [w for w in _WORD.findall(f"{title} {summary}".lower())
             if len(w) > 2 and w not in _STOP and w not in drop]
    grams = list(words)
    grams += [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return grams


def _target_tokens(ticker: Optional[str]) -> frozenset:
    """The company's own name and symbol — in every article of its digest by
    construction, and pool IDF cannot remove them because a company name is
    pool-RARE. Left in, they merge the whole digest into one story."""
    if not ticker:
        return frozenset()
    out = {str(ticker).lower()}
    try:
        from src.data.company_names import name_keywords
        kw = name_keywords(ticker) or {}
        for phrase in (kw.get("phrases") or []):
            out.update(w for w in str(phrase).lower().split() if len(w) > 2)
        out.update(str(t).lower() for t in (kw.get("tokens") or []))
    except Exception:                                           # noqa: BLE001
        pass
    return frozenset(out)


def _weights(articles: Sequence, drop: frozenset) -> List[dict]:
    """``[{term: idf}]`` per article, from the POOL corpus when one is
    installed and flat weights when it is not.

    Never digest IDF: see the module docstring — it zeroes exactly the terms
    that identify a shared story.
    """
    df, n = _CORPUS["df"], _CORPUS["n"]
    out = []
    for a in articles:
        vec = {}
        for t in set(_tokens(a, drop)):
            if df is not None:
                vec[t] = math.log((n + 1.0) / (df.get(t, 0) + 1.0))
            else:
                vec[t] = 1.0                    # flat fallback, stoplist only
        out.append(vec)
    return out


def _overlap(a: dict, b: dict) -> float:
    """Shared IDF mass over the SMALLER article's mass.

    Not cosine: digest articles differ wildly in length, and cosine scores that
    asymmetry rather than the content. "Is the short one contained in the long
    one" is what a syndicated follow-up looks like.
    """
    if not a or not b:
        return 0.0
    ma, mb = sum(a.values()), sum(b.values())
    if ma <= 0 or mb <= 0:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    shared = sum(v for t, v in a.items() if t in b)
    return shared / min(ma, mb)


def content_groups(articles: Sequence, threshold: Optional[float] = None,
                   ticker: Optional[str] = None) -> List[List]:
    """Single-link connected components over cosine similarity, freshest first.

    Freshest FIRST means ordered by each group's most recent article, which is
    the order every consumer assumes (`cluster_bounds`, the cluster arm) and the
    order `recent_cluster` produced.
    """
    arts = [a for a in (articles or []) if getattr(a, "published_at", None) is not None]
    if len(arts) < 2:
        return [list(arts)] if arts else []
    if len(arts) > _MAX_ARTICLES:                       # newest first, bounded
        arts = sorted(arts, key=lambda a: a.published_at, reverse=True)[:_MAX_ARTICLES]
    thr = float(threshold if threshold is not None
                else getattr(settings, "news_cluster_min_similarity", 0.30))
    vecs = _weights(arts, _target_tokens(ticker))
    parent = list(range(len(arts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(arts)):
        for j in range(i + 1, len(arts)):
            if _overlap(vecs[i], vecs[j]) >= thr:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups: dict = {}
    for i, a in enumerate(arts):
        groups.setdefault(find(i), []).append(a)
    out = [sorted(g, key=lambda a: a.published_at, reverse=True) for g in groups.values()]
    out.sort(key=lambda g: g[0].published_at, reverse=True)
    return out


def hybrid_groups(articles: Sequence, threshold: Optional[float] = None,
                  ticker: Optional[str] = None,
                  as_of: Optional[datetime] = None,
                  ratio: float = 3.0, floor_hours: float = 24.0) -> List[List]:
    """TIME partition first, then MERGE the pieces that are the same story.

    Content clustering from scratch was measured (2026-09-09, 73 ticker-days,
    both arms scored on the same digests) to make the defect it was built for
    dramatically WORSE — same-catalyst-opposite-sign went 15.8% -> 77.5%, fixing
    1 ticker-day and introducing 36 — because a 13-article digest holds ~8-12
    lexically distinct items, so re-partitioning shatters it into singletons and
    a singleton carries too little context for a stable read (per-PAIR
    contradiction 11.4% -> 17.8%, abstention 3.4% -> 9.4%).

    The defect is narrower than "the partition is wrong": it is ONE story cut
    into two TIME pieces that then contradict each other. So use content only
    where it is asked a question it can answer — "are these two pieces the same
    story?" — and never to split. Cluster count can only go DOWN from the time
    baseline, which is the direction the scoring evidence wants.

    Cluster-to-cluster similarity is SINGLE-LINK over the articles (the best
    cross pair): syndicated coverage of one story runs through a chain of
    near-copies, and requiring the average to clear the bar would miss it.
    """
    groups = _time_groups(articles, ratio=ratio, floor_hours=floor_hours, as_of=as_of)
    if len(groups) < 2:
        return groups
    thr = float(threshold if threshold is not None
                else getattr(settings, "news_cluster_min_similarity", 0.25))
    drop = _target_tokens(ticker)
    vecs = {}
    for gi, grp in enumerate(groups):
        for ai, vec in enumerate(_weights(grp, drop)):
            vecs[(gi, ai)] = vec
    n = len(groups)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            best = max((_overlap(vecs[(i, ai)], vecs[(j, aj)])
                        for ai in range(len(groups[i]))
                        for aj in range(len(groups[j]))), default=0.0)
            if best >= thr:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    merged: dict = {}
    for i, grp in enumerate(groups):
        merged.setdefault(find(i), []).extend(grp)
    out = [sorted(g, key=lambda a: a.published_at, reverse=True) for g in merged.values()]
    out.sort(key=lambda g: g[0].published_at, reverse=True)
    return out


def cluster_mode() -> str:
    """``"time"`` | ``"content"`` | ``"hybrid"`` — the configured default for
    consumers that do not pin one."""
    v = str(getattr(settings, "news_cluster_mode", "time") or "time").strip().lower()
    return v if v in ("time", "content", "hybrid") else "time"


def cluster_articles(articles: Sequence, mode: Optional[str] = None,
                     as_of: Optional[datetime] = None, ticker: Optional[str] = None,
                     ratio: float = 3.0, floor_hours: float = 24.0) -> List[List]:
    """``[[article, ...], ...]`` freshest group first, under ``mode``.

    ``mode=None`` reads `news_cluster_mode`. Errors fall back to the TIME
    partition — a clustering bug must cost fidelity, never a tick.
    """
    arts = [a for a in (articles or []) if getattr(a, "published_at", None) is not None]
    if not arts:
        return []
    use = (mode or cluster_mode()).strip().lower()
    if use in ("content", "hybrid") and corpus_size() == 0:
        # NO CORPUS, NO MERGE. Without pool document frequencies `_weights`
        # returns FLAT weights, where every shared common word counts in full —
        # so "shares fell today" matches "shares rose today" and the merge fires
        # on boilerplate. Flat weights are a safe REPRESENTATION fallback (they
        # are at least not digest IDF, which is actively wrong) but they are not
        # a safe basis for a merge DECISION, so a missing corpus degrades to the
        # time partition instead. The aggregator installs it once per run; a
        # pool under 20 documents installs nothing on purpose.
        use = "time"
    if use in ("content", "hybrid"):
        try:
            groups = (content_groups(arts, ticker=ticker) if use == "content"
                      else hybrid_groups(arts, ticker=ticker, as_of=as_of,
                                         ratio=ratio, floor_hours=floor_hours))
            if groups:
                return groups
        except Exception as e:                                  # noqa: BLE001
            logger.debug(f"[news_clustering] {use} mode failed ({e}) — time partition")
    return _time_groups(arts, ratio=ratio, floor_hours=floor_hours, as_of=as_of)


def _time_groups(articles: Sequence, ratio: float = 3.0, floor_hours: float = 24.0,
                 as_of: Optional[datetime] = None) -> List[List]:
    """The legacy relative-time partition, byte-for-byte: `recent_cluster`
    applied repeatedly to what is left."""
    from src.analysis.sentiment import recent_cluster
    rest, out, guard = list(articles), [], 0
    while rest and guard < 20:
        guard += 1
        group = recent_cluster(rest, ratio=ratio, floor_hours=floor_hours, as_of=as_of)
        if not group:
            break
        out.append(group)
        keep = {id(a) for a in group}
        rest = [a for a in rest if id(a) not in keep]
    return out
