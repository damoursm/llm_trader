"""Shared guards for the SELF-HOSTED (local) LLM routes — 2026-09-04.

Two very different jobs run against the same local server: per-ticker SENTIMENT
(measured 4.4k-5.7k token prompts at 14-20 articles, ~68 fresh calls a tick,
of which a fixed 2,176-token rubric prefix; wants throughput) and the
final SYNTHESIS (one call a tick, tens of thousands of tokens, wants context).
They are configured separately (`local_sentiment_*` for sentiment, `local_synthesis_*`
for synthesis, each inheriting the sentiment value when unset), but they share
ONE failure mode, which is why these helpers live in one module rather than
being copied into both call sites:

**Ollama accepts a prompt larger than its context, silently truncates the
OLDEST tokens — where the instructions are — and answers anyway.** Probed
2026-09-03 with a canary at the head of the prompt: found at 2k, LOST at 5k and
8k, `prompt_tokens` pinned at 2050, no error raised anywhere. A verdict
produced that way is indistinguishable from a real one at every layer that
consumes it.

(That probe ran at `OLLAMA_CONTEXT_LENGTH=4096` and left open whether the
context is DIVIDED across parallel slots. Settled 2026-09-05 by reading the
child process's own arguments: `llama-server ... -c 16384 -np 2`, i.e. Ollama
allocates context_length x num_parallel and each request gets the full 8192.
The guards below are therefore not currently firing on the sentiment route —
2,263 local verdicts on 2026-09-04, zero refusals — which is the point: they
are the net for the next config change, not for today's.)

So both routes get the same two nets, in this order:

  1. a PRE-FLIGHT against the server's configured context — refuses before the
     call, so an over-long prompt costs nothing rather than a full
     prefill+decode. The OpenAI-compatible endpoint does not report the
     server's context (and ignores `options.num_ctx`), so the context is a
     SETTING that must be kept in step with `scripts/run_ollama.bat`;
  2. a POST-CALL check of the server's own reported `prompt_tokens` — the net
     for a context smaller than the setting claims, or a chars-per-token
     estimate that is simply wrong.

Both RAISE. On the sentiment route the exception falls through to the next
engine; on the synthesis route it falls through to the hosted engines. Neither
may be downgraded to a warning: a truncated answer that still parses would be
scored as this engine's opinion — a 0.40-weight news verdict, or a trade
decision — and the whole point of running a local engine beside a hosted one is
that the comparison means something.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

# Rough size of a prompt in the local model's tokens. An ESTIMATE, used only to
# size a prompt BEFORE the call; the server's own `prompt_tokens` is read back
# afterwards, which is how a wrong constant surfaces instead of silently
# widening the guard.
#
# MEASURED 2026-09-05, not guessed: the sentiment prefix is 9,528 chars and
# DeepSeek's usage record counts it at 2,176 prompt tokens on the identical
# string (`_log_cache_hit`, prefix-cache line) = **4.38 chars/token** on this
# prompt's English-plus-punctuation mix. 4.0 is deliberately a shade below that,
# so the estimate still runs ~10% HIGH and the pre-flight stays the conservative
# side of the real count. The previous 3.2 over-estimated by ~37%, which is what
# made the ratio below unusable (see there).
CHARS_PER_TOKEN = 4.0
# A server-reported prompt_tokens below this fraction of the estimate reads as
# silent truncation — the count pins at the effective server context — rather
# than as a generous tokenizer.
#
# The threshold has to separate two ratios that were nearly touching at the
# 20-article digest cap (2026-09-05): a truncation to a 4,096-token effective
# context on a ~6.2k-token prompt reports **0.66** of the estimate, while the
# legitimate estimate-vs-reported gap at the OLD 3.2 chars/token was **0.73** —
# i.e. the old 0.6 threshold could not have fired on exactly the largest, most
# consequential digests, and raising it alone would have fired on healthy ones.
# Recalibrating CHARS_PER_TOKEN first is what makes the ratio meaningful: at 4.0
# a healthy call reports ~0.91 of the estimate. Move BOTH constants together or
# not at all.
#
# A ratio test alone can never close the window completely, which is why the
# exact context test below exists. Truncation to an effective context E happens
# once the REAL count passes E (estimate ~1.10*E here), but a ratio test only
# sees it once the estimate passes E/ratio — so prompts in between report a
# healthy-looking fraction. At 0.85 that residual window is estimate in
# (1.10*E, 1.18*E), ~7%; the false-positive cost on the other side is one
# fall-through to a hosted engine, logged, against a silent rubric-less verdict
# in the 0.40-weight news method — so the tighter side is the right error.
TRUNCATION_RATIO = 0.85


def estimate_tokens(text: str) -> int:
    return int(len(text or "") / CHARS_PER_TOKEN) + 1


def check_fits(prompt: str, *, context_tokens: int, label: str) -> int:
    """Pre-flight. Returns the token estimate; raises when the prompt cannot fit
    the server's configured context. ``context_tokens`` <= 0 disables the check
    (an unknown server, e.g. a non-Ollama backend that sizes itself)."""
    est = estimate_tokens(prompt)
    ctx = int(context_tokens or 0)
    if ctx and est > ctx:
        raise RuntimeError(
            f"{label}: prompt is ~{est} tok against a server context of {ctx} — the OLDEST "
            f"tokens, i.e. the instructions, would be truncated silently; raise "
            f"OLLAMA_CONTEXT_LENGTH (and the matching *_context_tokens setting) or send "
            f"less prompt")
    return est


def check_reported(reported: Optional[int], *, estimate: int, label: str,
                   budget: Optional[int] = None,
                   context_tokens: Optional[int] = None) -> None:
    """Post-call. Raises when the server truncated the prompt; warns when it
    counted more than the budget the prompt was built against (the estimate
    under-counts this prompt).

    Two independent truncation tests, because neither covers the other:

      1. EXACT — a prompt that FIT reports strictly fewer tokens than the
         context it fit in. A count that lands AT (or above) the configured
         context is the server telling us it filled the window, which for a
         prompt we already pre-flighted means it dropped the head. No estimate
         is involved, so this one cannot be defeated by a wrong
         ``CHARS_PER_TOKEN``.
      2. RATIO — for the case the first test cannot see: an EFFECTIVE context
         smaller than the configured one (a server reconfigured behind the
         setting, or a runtime that divides the window across slots). The count
         then pins well below the setting, and only the estimate reveals it.
    """
    if not reported:
        return
    ctx = int(context_tokens or 0)
    if ctx and reported >= ctx:
        raise RuntimeError(
            f"{label}: server counted {reported} prompt tok against a context of {ctx} — a "
            f"prompt that fit would report FEWER; the window was filled and the OLDEST tokens "
            f"(the instructions) were dropped. Raise OLLAMA_CONTEXT_LENGTH (and the matching "
            f"*_context_tokens setting) or send less prompt")
    if budget and reported > budget:
        logger.warning(f"{label}: server counted {reported} prompt tok (> budget {budget}, "
                       f"estimate {estimate}) — CHARS_PER_TOKEN under-counts this prompt; "
                       f"lower the budget or refit the constant")
        return
    if reported < TRUNCATION_RATIO * estimate:
        raise RuntimeError(
            f"{label}: server counted only {reported} prompt tok against an estimate of "
            f"{estimate} — the server context is smaller than the prompt and the OLDEST "
            f"tokens (the instructions) were truncated silently; raise OLLAMA_CONTEXT_LENGTH "
            f"or send less prompt")
