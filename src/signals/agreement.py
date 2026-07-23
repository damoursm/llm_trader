"""Cross-family agreement + raw-tape confirmation (2026-07-19).

Two upgrades to how "the methods agree" is measured and consumed:

1.  METHOD FAMILIES — the flat ``sources_agreeing`` count treats every method as an
    independent voter, but many are highly correlated: the OHLCV-derived technical
    methods all read the same tape, news and sent_velocity share the same articles.
    Agreement among correlated methods is pseudo-replication, not confirmation.
    ``METHOD_FAMILIES`` groups the weighted pool into independent INFORMATION
    families; ``compute_family_agreement`` rolls each family's members into one
    magnitude-weighted family vote and counts how many FAMILIES align with /
    oppose the combined direction. Three families aligned (e.g. Sentiment +
    Options + Smart-Money) is materially stronger evidence than five technical
    methods agreeing — and now scores (and prompts) that way.

2.  TAPE CONFIRMATION — agreement was score-only; nothing checked whether the raw
    MARKET DATA backs the direction. ``compute_tape_confirmation`` computes a
    score-independent price/volume state composite from the cached daily OHLCV:
      • 20d range position   — where the last close sits in its 20-day range
        (top = bullish structure, bottom = bearish);
      • signed volume share  — up-day volume vs down-day volume over the last 10
        completed bars (is the VOLUME on the up days or the down days?);
      • RVOL × last direction — last bar's volume vs 20d average, signed by that
        bar's close-over-close direction (heavy volume on an up close = bullish
        participation; heavy on a down close = distribution).
    Composite ∈ [-1, +1] with the house sign convention (+ = bullish state).
    It is NOT an alpha method in the weighted combine — it is an agreement
    QUALIFIER: a small confidence factor when it confirms/diverges from the
    combined direction, a per-ticker prompt line for synthesis, and a panel
    pseudo-method (``tape``) so its forward IC is monitored like any method.

Both computations are fail-soft (no cache / thin data → neutral) and respect the
win-rate filter + method inversion: the aggregator passes only ACTIVE methods'
effective (inversion-applied) scores into the family rollup.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from loguru import logger

# Family taxonomy over the WEIGHTED pool (aggregator._BASE_WEIGHTS — the 8
# panel-first methods are outside the pool and outside this map by design).
# Grouped by INFORMATION SOURCE, not by indicator type: two methods share a
# family when they read substantially the same underlying data and would
# therefore co-move (agreeing together is one confirmation, not two).
# tests/test_family_agreement.py drift-tests this against _BASE_WEIGHTS.
METHOD_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "Sentiment":    ("news", "sent_velocity"),                    # article flow (level + Δ)
    "Price/Trend":  ("tech", "massive", "momentum",               # the ticker's own OHLCV tape
                     "trend_strength", "pattern", "vwap", "ext_gap"),
    "Rel-Strength": ("sector_momentum", "market_momentum"),       # ticker vs benchmark tape
    "Volume-Flow":  ("money_flow",),                              # volume-weighted accumulation
    "Options":      ("put_call", "max_pain", "oi_skew",           # options-chain positioning
                     "iv_rank", "iv_expr"),
    "Smart-Money":  ("insider", "broker_advisor"),                # insider filings + borrow state
    "Event/Arb":    ("pead", "coint"),                            # event drift + stat-arb spread
}

# method -> family reverse index (built once)
FAMILY_OF: Dict[str, str] = {
    m: fam for fam, members in METHOD_FAMILIES.items() for m in members
}


@dataclass
class FamilyAgreement:
    """Cross-family agreement snapshot for one ticker."""
    agreeing: int = 0            # families voting WITH the combined direction
    opposing: int = 0            # families voting AGAINST it
    coherence: float = 0.0       # magnitude-weighted family-level agreement ratio ∈ [0,1]
    net_score: float = 0.0       # magnitude-weighted mean of family votes ∈ [-1,+1] (IC-able)
    detail: str = ""             # compact rollup, e.g. "Sentiment:+0.42|Options:-0.12"
    family_scores: Dict[str, float] = field(default_factory=dict)

    def factor(self, span: float) -> float:
        """Confidence multiplier ∈ [1-span, 1+span].

        Rewards BREADTH of independent confirmation, penalises conflict:
          g = clamp((agreeing − 1.5·opposing − 1) / 3, −1, +1);  factor = 1 + span·g
        Anchors: 1 family alone → 1.00 (no cross-confirmation, no penalty — the
        single-source discount is coherence's job); 2 aligned → +span/3;
        4+ aligned, none opposing → 1+span; any opposing family drags it down
        1.5× as hard as an aligned one helps. No family votes at all → 1.00.
        """
        if (self.agreeing + self.opposing) == 0:
            return 1.0
        g = (self.agreeing - 1.5 * self.opposing - 1.0) / 3.0
        g = max(-1.0, min(1.0, g))
        return round(1.0 + float(span) * g, 4)


def compute_family_agreement(effective_scores: Dict[str, float], combined: float,
                             vote_threshold: float = 0.05) -> FamilyAgreement:
    """Roll per-method scores up into per-family votes and count alignment.

    ``effective_scores`` — {method: score} for ACTIVE methods only, with the
    method-inversion sign ALREADY applied (so the rollup reflects what the
    combine actually consumed; the caller owns filtering + inversion).
    Zero scores are no-views and may be omitted or included (ignored either way).

    Per family: magnitude-weighted mean of member views (Σ s·|s| / Σ|s| — a
    strong member dominates a weak dissenter, mirroring ``_coherence_factor``'s
    weighting). A family VOTES when |family score| ≥ ``vote_threshold`` — unlike
    the legacy ``sources_agreeing`` (threshold 0.0), a hairline 0.01 lean is not
    counted as confirmation. Alignment is judged against ``sign(combined)``;
    with a ~zero combined score nothing is aligned/opposed (counts stay 0) but
    the family scores/detail are still reported for the prompt.
    """
    fam_scores: Dict[str, float] = {}
    for fam, members in METHOD_FAMILIES.items():
        num = 0.0
        den = 0.0
        for m in members:
            s = float(effective_scores.get(m, 0.0) or 0.0)
            if s == 0.0:
                continue                      # no view
            num += s * abs(s)
            den += abs(s)
        if den > 0.0:
            fam_scores[fam] = num / den

    votes = {f: s for f, s in fam_scores.items() if abs(s) >= float(vote_threshold)}

    agreeing = opposing = 0
    agree_w = total_w = 0.0
    if abs(combined) > 1e-9:
        for s in votes.values():
            total_w += abs(s)
            if (combined > 0) == (s > 0):
                agreeing += 1
                agree_w += abs(s)
            else:
                opposing += 1
    coherence = (agree_w / total_w) if total_w > 0 else 0.0

    net_num = sum(s * abs(s) for s in votes.values())
    net_den = sum(abs(s) for s in votes.values())
    net = (net_num / net_den) if net_den > 0 else 0.0

    detail = "|".join(
        f"{f}:{s:+.2f}"
        for f, s in sorted(votes.items(), key=lambda kv: -abs(kv[1]))
    )
    return FamilyAgreement(
        agreeing=agreeing, opposing=opposing,
        coherence=round(coherence, 3), net_score=round(net, 3),
        detail=detail, family_scores={f: round(s, 3) for f, s in fam_scores.items()},
    )


@dataclass
class TapeCheck:
    """Raw price/volume structure state for one ticker (score-independent)."""
    score: float = 0.0           # ∈ [-1,+1]; + = bullish structure
    label: str = "NO_DATA"       # BULLISH_TAPE | BEARISH_TAPE | MIXED_TAPE | NO_DATA
    detail: str = ""             # human-readable component summary for the prompt


_TAPE_MIN_BARS = 21              # 20d range/RVOL window + 1 for close-over-close


def compute_tape_confirmation(ticker: str, df=None) -> TapeCheck:
    """Score-independent bullish/bearish read of the raw tape from cached OHLCV.

    Cache-only by design (``load_ohlcv``; pass ``df=`` to reuse an in-scope
    frame): a confirmation overlay must never add network fetches to the hot
    per-ticker scoring path, and a cold cache simply yields NO_DATA (neutral,
    factor 1.0) — the same fail-closed posture as ext_gap. Components (each
    ∈ [-1,+1], mean of those computable):

      range_pos    2·((close − 20d_low) / (20d_high − 20d_low)) − 1
      vol_balance  (up-day vol − down-day vol) / (up+down vol) over last 10 bars
      rvol_dir     sign(close-over-close) · min(last_vol / 20d_avg_vol, 2) / 2

    NaN/zero-guarded throughout (a zero/NaN close row is skipped — the SOLS
    zero-close class); degenerate inputs (flat range, zero volume) drop that
    component rather than fabricating a view.
    """
    try:
        if df is None:
            from src.data.cache import load_ohlcv
            df = load_ohlcv(ticker)
        if df is None or len(df) < _TAPE_MIN_BARS:
            return TapeCheck()
        cols = {str(c).lower(): c for c in df.columns}
        c_close, c_high = cols.get("close"), cols.get("high")
        c_low, c_vol = cols.get("low"), cols.get("volume")
        if not all((c_close, c_high, c_low, c_vol)):
            return TapeCheck()

        import pandas as pd  # noqa: F401 (df ops)
        tail = df.tail(_TAPE_MIN_BARS)
        closes = tail[c_close].astype(float)
        # NaN/zero-close guard: a bad row invalidates the simple vector math —
        # fail closed to NO_DATA rather than scoring on a corrupt tape.
        if closes.isna().any() or (closes <= 0).any():
            return TapeCheck()
        last_close = float(closes.iloc[-1])
        prev_close = float(closes.iloc[-2])

        components: list = []
        details: list = []

        # 1 — 20d range position
        w20 = tail.tail(20)
        hi = float(w20[c_high].astype(float).max())
        lo = float(w20[c_low].astype(float).min())
        if hi > lo and hi == hi and lo == lo:          # non-degenerate, non-NaN
            pos = (last_close - lo) / (hi - lo)        # ∈ [0,1] (close within H/L range)
            pos = max(0.0, min(1.0, pos))
            components.append(2.0 * pos - 1.0)
            details.append(f"close at {pos * 100:.0f}% of 20d range")

        # 2 — signed volume balance over the last 10 completed bars
        vols = tail[c_vol].astype(float)
        up_vol = down_vol = 0.0
        for i in range(len(tail) - 10, len(tail)):
            v = float(vols.iloc[i])
            if v != v or v <= 0:                       # NaN / zero volume bar
                continue
            delta = float(closes.iloc[i]) - float(closes.iloc[i - 1])
            if delta > 0:
                up_vol += v
            elif delta < 0:
                down_vol += v
        tot = up_vol + down_vol
        if tot > 0:
            bal = (up_vol - down_vol) / tot
            components.append(bal)
            details.append(f"up-day volume {((bal + 1) / 2) * 100:.0f}% of 10d")

        # 3 — last-bar relative volume, signed by that bar's direction
        last_vol = float(vols.iloc[-1])
        avg_vol = float(vols.iloc[-21:-1].mean()) if len(vols) >= 21 else float("nan")
        if last_vol == last_vol and avg_vol == avg_vol and avg_vol > 0 and last_vol > 0:
            rvol = last_vol / avg_vol
            direction = 1.0 if last_close > prev_close else (-1.0 if last_close < prev_close else 0.0)
            components.append(direction * min(rvol, 2.0) / 2.0)
            details.append(f"RVOL {rvol:.1f} on {'an up' if direction > 0 else 'a down' if direction < 0 else 'a flat'} close")

        if not components:
            return TapeCheck()
        score = sum(components) / len(components)
        score = max(-1.0, min(1.0, score))
        label = ("BULLISH_TAPE" if score >= 0.25
                 else "BEARISH_TAPE" if score <= -0.25
                 else "MIXED_TAPE")
        return TapeCheck(score=round(score, 3), label=label, detail="; ".join(details))
    except Exception as e:  # pragma: no cover — defensive (odd cache shapes)
        logger.debug(f"[agreement] tape confirmation unavailable for {ticker}: {e}")
        return TapeCheck()


def tape_factor(tape: Optional[TapeCheck], combined: float, span: float) -> float:
    """Confidence multiplier ∈ [1-span, 1+span] from tape-vs-direction alignment.

    Tape CONFIRMING the combined direction lifts confidence, diverging cuts it,
    proportionally to the tape score. Neutral (no data, mixed-zero tape, or a
    near-zero combined score with no direction to confirm) → 1.0.
    """
    if tape is None or tape.label == "NO_DATA" or abs(combined) < 0.05:
        return 1.0
    direction = 1.0 if combined > 0 else -1.0
    return round(1.0 + float(span) * (tape.score * direction), 4)
