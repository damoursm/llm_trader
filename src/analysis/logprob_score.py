"""Recover a CONTINUOUS verdict from the model's own token distribution.

THE DEFECT (measured 2026-09-10, 317 live production verdicts): local qwen3:8b
emits **19 distinct raw values, 100% of them on a 0.05 grid**, modal +0.15 on 15%
of calls. `news` is rank-consumed and a tie group collapses to one shared average
rank, so every collision is cross-section the method cannot order.

THE INSIGHT: that grid is an ARGMAX artifact, not the model's actual belief. The
distribution over the score's decimal digits is not a point mass — it is being
collapsed to its mode by greedy decoding. Reading `top_logprobs` at the digit
positions and taking the EXPECTATION recovers a continuous value:

    measured on 35 production digests, argmax -> expected
        distinct values     12 -> 35
        on the 0.05 grid   100% -> 0%
        rank correlation   +0.9912, ZERO sign flips
        mean |shift|        0.024
        among rows sharing an argmax value, 100% become distinct

WHY THIS AND NOT A CONFIDENCE FIELD. The obvious form of the same idea — ask the
model for a confidence and use it to break ties — is the one thing this project
has learned not to do: `memory/confidence-anchoring-2026-08` records a stated
confidence SCALE becoming a modal OUTPUT (28 distinct values collapsed to 8,
20-35% landing on exactly 1.00), and sentiment v2 did the same with example
numbers. A token distribution is not something the model chooses to report, so
it cannot be anchored that way. It also needs NO prompt change: no version bump,
no scorer epoch, no cache flush — every digest is scored exactly as before and
only READ more precisely.

STATUS: ACCRUAL ONLY. `news_expected_score` is persisted beside the argmax and
consumed by nothing. It cannot be validated yet — settled pivot labels stop at
2026-09-04, the day prompt v6 deployed, so there are no v6-era rows with labels.
The open question is whether the finer value beats the argmax on per-day pivot
IC, and showing that it is merely FINER is not evidence: the v7d priced-in work
produced exactly that illusion (100% -> 6% off the grid) by multiplying a good
read by noise. Bounded upside regardless — the consumed `news` score is only
11.1% tied within a run, because the mass x diversity scaler already breaks most
collisions.

Fail-soft everywhere: any parsing difficulty returns None and the caller keeps
the argmax.
"""

from __future__ import annotations

import math
import re
from typing import List, Optional, Sequence, Tuple

from loguru import logger

# A correction larger than this means the token walk latched onto the wrong
# span (a number inside the rationale, a truncated response). Refuse it rather
# than emit a verdict that is not a finer reading of the same one.
MAX_SHIFT = 0.25
_SCORE_RE = re.compile(r'"score"\s*:\s*([-+]?\d*\.?\d+)')


def _spans(tokens: Sequence) -> Tuple[str, List[Tuple[int, int]]]:
    """``(reconstructed_text, [(start, end), ...])`` — one char span per token.

    Char offsets are what make this robust to TOKENISATION. qwen splits a number
    inconsistently (`0.85` arrives as one token, or `0`/`.`/`85`, or `0`/`.85`),
    and the first version of this walked token TYPES and failed on 56% of
    responses. Mapping the regex match back through spans works whatever the
    split.
    """
    text, spans, off = [], [], 0
    for t in tokens:
        tok = getattr(t, "token", "") or ""
        text.append(tok)
        spans.append((off, off + len(tok)))
        off += len(tok)
    return "".join(text), spans


def _alt_probs(tok) -> List[Tuple[str, float]]:
    out = []
    for a in (getattr(tok, "top_logprobs", None) or []):
        try:
            out.append((getattr(a, "token", "") or "", math.exp(a.logprob)))
        except Exception:                                       # noqa: BLE001
            continue
    return out


def expected_score(tokens: Optional[Sequence], argmax: Optional[float] = None
                   ) -> Optional[float]:
    """Expectation of the `score` value under the model's own token distribution.

    Substitutes each alternative at each token position of the VALUE (holding
    the others fixed), re-parses, and sums the per-position deviations onto the
    argmax — a first-order correction that needs no assumption about how the
    number was split.

    None whenever the answer cannot be read confidently: no tokens, no `score`
    field, no usable alternatives, or a correction beyond `MAX_SHIFT`.
    """
    if not tokens:
        return None
    try:
        text, spans = _spans(tokens)
        m = _SCORE_RE.search(text)
        if not m:
            return None
        lo, hi = m.span(1)
        base_s = m.group(1)
        try:
            base = float(base_s)
        except ValueError:
            return None
        if argmax is not None and abs(base - float(argmax)) > 1e-6:
            # the logprobs describe a different response than the parsed one
            return None
        idx = [i for i, (a, b) in enumerate(spans) if a < hi and b > lo]
        if not idx:
            return None
        delta = 0.0
        for i in idx:
            alts = _alt_probs(tokens[i])
            if len(alts) < 2:
                continue
            a, b = spans[i]
            head, tail = text[lo:max(lo, a)], text[min(hi, b):hi]
            num, den = 0.0, 0.0
            for cand, p in alts:
                try:
                    val = float(head + cand.strip() + tail)
                except ValueError:
                    continue                     # not a number with this token
                if abs(val) > 1.0:
                    continue                     # outside the contract
                num += p * val
                den += p
            if den > 0:
                delta += (num / den) - base
        out = base + delta
        if abs(out - base) > MAX_SHIFT:
            logger.debug(f"[logprob_score] correction {out - base:+.3f} beyond "
                         f"{MAX_SHIFT} — keeping the argmax")
            return None
        return round(max(-1.0, min(1.0, out)), 6)
    except Exception as e:                                      # noqa: BLE001
        logger.debug(f"[logprob_score] {e} — keeping the argmax")
        return None


def tokens_of(response) -> Optional[Sequence]:
    """The per-token logprob list off an OpenAI-shaped response, or None."""
    try:
        return response.choices[0].logprobs.content
    except Exception:                                           # noqa: BLE001
        return None
