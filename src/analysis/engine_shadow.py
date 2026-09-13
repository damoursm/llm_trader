"""Paired synthesis engines — every engine's decision on every tick.

Why
---
User directive (2026-09-04): route the final synthesis to the LOCAL engine
AND to the remote engine, store each engine's decision, and A/B which one the
pipeline acts on. The A/B half is the per-RUN flip in
`claude_analyst.generate_recommendations` (`synthesis_local_share`): the
engine it picks is LIVE — its decisions flow through the gate cascade, the
ledger and the broker sync. This module is the pairing half: after the live
engine has answered, the OTHER engine is asked the same question about the
same signal cross-section and its decisions are recorded and acted on by
nobody. Judging two engines requires the same ticker, the same tick, the same
inputs — the unpaired design (grouping trades by the engine that opened them)
is exactly what produced the 2026-07-22 bake-off's window artifacts, where
every apparent engine difference vanished under paired testing.

Design constraints
------------------
* OFF THE CRITICAL PATH. The shadow call runs in a daemon thread started
  after the live recommendations exist. Orders never wait on it, and the
  persist step drains NON-BLOCKING: a local synthesis is minutes of work, so
  whatever is still in flight carries its own ``run_id`` and lands on the next
  tick's write (hence `repo.insert_engine_recommendations` has no run-wide
  delete — idempotency is per (run_id, engine, ticker)).
* EACH ENGINE GETS THE PROMPT IT CAN RUN. The remote engine takes the FULL
  synthesis prompt (p50 ~60k tokens, measured 2026-09-04); the local engine
  the COMPACT one (`claude_analyst._compact_synthesis_prompt`, capped by
  `local_synthesis_max_prompt_tokens`) — the full prompt cannot fit beside
  8B weights in 8 GB of VRAM. So the pair CONFOUNDS engine with prompt;
  `synthesis_shadow_extra="deepseek:compact"` adds a third arm (the remote
  engine on the compact prompt) that decomposes it. Every row stamps its
  ``prompt_variant`` so the eval can split on it.
* PINNED, PROVENANCE-SAFE. Shadow calls use ``force_engine``, which never
  writes the run/sentiment provenance and never rule-fills missing tickers —
  so a shadow row is always the model's own answer (``rule_filled`` is only
  ever True on LIVE rows, where the fill exists so open positions never fall
  silent), and a shadow call can never mis-stamp the run or its trades.
* BOUNDED + FAIL-SOFT. `synthesis_shadow_max_pending` branches may be in
  flight at once (a local call queues behind the shadow SENTIMENT pass on the
  same server, so a slow tick must not pile up); a failure yields no rows and
  never touches the run. A LOCAL shadow call also waits (bounded) for the
  sentiment shadow to drain first, so the two background consumers of the
  local server take turns instead of fighting over its slots.

Read the result with ``python -m src.analysis.engine_eval``.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from config.settings import settings

from loguru import logger  # project configures loguru sinks only

_ENGINES = ("anthropic", "deepseek", "qwen", "local")
_VARIANTS = ("auto", "full", "compact")

# How long a LOCAL shadow call politely waits for the sentiment shadow pass to
# finish before taking the server — bounded so a wedged sentiment queue can
# never hold the pairing hostage for the whole tick.
_COURTESY_WAIT_SECONDS = 600.0
_COURTESY_POLL_SECONDS = 15.0

_LOCK = threading.Lock()
_ROWS: List[dict] = []          # decisions waiting for the next persist drain
_PENDING = 0                    # shadow branches currently running


# ── rows ──────────────────────────────────────────────────────────────────────

def resolve_variant(engine: str, variant: str = "auto") -> str:
    """The prompt variant a call with ``prompt_variant=variant`` actually
    renders for ``engine`` (mirrors `claude_analyst.generate_recommendations`:
    ``auto`` = compact for the local engine, full otherwise)."""
    v = (variant or "auto").lower()
    if v == "auto":
        return "compact" if engine == "local" else "full"
    return v


def rows_for(*, engine: Optional[str], model: Optional[str], prompt_variant: str,
             live: bool, recs: List[Any], prices: Dict[str, Any],
             latency_s: Optional[float], n_signals: int,
             run_id: str, generated_at: str, signal_date: str) -> List[dict]:
    """Flatten one engine's recommendations into persistable rows."""
    out: List[dict] = []
    # A rec without a ticker is not a decision — it is dropped, and `n_recs`
    # counts what was PERSISTED so the column can be read as this engine's
    # coverage of the cross-section against `n_signals`.
    recs = [r for r in (recs or []) if getattr(r, "ticker", None)]
    for r in recs:
        tic = getattr(r, "ticker", None)
        out.append({
            "run_id": run_id,
            "generated_at": generated_at,
            "signal_date": signal_date,
            "engine": engine,
            "model": model,
            "prompt_variant": prompt_variant,
            "live": bool(live),
            "ticker": tic,
            "action": getattr(r, "action", None),
            "direction": getattr(getattr(r, "direction", None), "value",
                                 getattr(r, "direction", None)),
            "confidence": getattr(r, "confidence", None),
            "time_horizon": getattr(r, "time_horizon", None),
            "rationale": getattr(r, "rationale", None),
            "snap_price": prices.get(tic),
            "rule_filled": bool(getattr(r, "rule_filled", False)),
            "latency_s": latency_s,
            "n_signals": int(n_signals),
            "n_recs": len(recs),
        })
    return out


def _append_rows(rows: List[dict]) -> None:
    if not rows:
        return
    with _LOCK:
        _ROWS.extend(rows)


def pop_engine_shadow_rows() -> List[dict]:
    """Drain every decision row queued so far (live and shadow)."""
    with _LOCK:
        out = list(_ROWS)
        _ROWS.clear()
    return out


def engine_shadow_pending() -> int:
    """Shadow branches still running (their rows land on a later drain)."""
    with _LOCK:
        return _PENDING


def _pending_delta(d: int) -> None:
    global _PENDING
    with _LOCK:
        _PENDING = max(0, _PENDING + d)


# ── arm resolution ────────────────────────────────────────────────────────────

def _local_available() -> bool:
    from src.analysis.claude_analyst import _local_synthesis_available
    return bool(_local_synthesis_available())


def shadow_arms(live_engine: Optional[str]) -> List[Tuple[str, str]]:
    """The ``(engine, prompt_variant)`` arms to run in the background for a run
    whose LIVE decision came from ``live_engine``.

    ``synthesis_shadow_engine`` "auto" pairs the two engines of the A/B: the
    local engine when the remote one was live, DeepSeek when the local one
    was — so the pair exists whichever way the per-run flip landed (and on a
    rule-based run, which had no engine at all). An explicit engine name pins
    it. ``synthesis_shadow_extra`` ("<engine>:<variant>", comma-separated)
    adds decomposition arms. An arm equal to the live one (same engine, same
    variant) is dropped — the live rows already carry that answer — and a
    local arm is dropped while the local LLM is disabled."""
    live = (live_engine or "").lower()
    live_variant = resolve_variant(live, "auto") if live in _ENGINES else None
    arms: List[Tuple[str, str]] = []

    primary = (settings.synthesis_shadow_engine or "auto").strip().lower()
    if primary == "auto":
        primary = "local" if live != "local" else "deepseek"
    if primary in _ENGINES:
        arms.append((primary, "auto"))
    elif primary not in ("", "none", "off"):
        logger.warning(f"[engine_shadow] unknown synthesis_shadow_engine={primary!r} — ignored")

    for spec in (settings.synthesis_shadow_extra or "").split(","):
        spec = spec.strip().lower()
        if not spec:
            continue
        eng, _, var = spec.partition(":")
        var = var or "auto"
        if eng not in _ENGINES or var not in _VARIANTS:
            logger.warning(f"[engine_shadow] bad synthesis_shadow_extra arm {spec!r} — ignored")
            continue
        arms.append((eng, var))

    out: List[Tuple[str, str]] = []
    seen = set()
    for eng, var in arms:
        key = (eng, resolve_variant(eng, var))
        if key in seen:
            continue
        if eng == live and key[1] == live_variant:
            continue                                    # the live rows ARE this arm
        if eng == "local" and not _local_available():
            continue
        seen.add(key)
        out.append((eng, var))
    return out


# ── the branch ────────────────────────────────────────────────────────────────

class EngineShadowBranch:
    """Runs the shadow arms in the background; rows are queued for the
    non-blocking persist drain (never joined)."""

    def __init__(self, *, signals, arms: List[Tuple[str, str]], synth_kwargs: dict,
                 generate: Callable[..., List[Any]], open_positions=None,
                 session: Optional[str] = None, arm_kwargs: Optional[dict] = None,
                 run_id: str, generated_at: str, signal_date: str,
                 wait_for: Optional[Callable[[], int]] = None):
        self._signals = signals
        self._arms = list(arms)
        self._synth_kwargs = dict(synth_kwargs or {})
        self._generate = generate
        self._open_positions = open_positions
        self._session = session
        self._arm_kwargs = dict(arm_kwargs or {})
        self._run_id = run_id
        self._generated_at = generated_at
        self._signal_date = signal_date
        self._wait_for = wait_for
        self._prices = {getattr(s, "ticker", None): getattr(s, "price", None)
                        for s in (signals or [])}
        self._thread: Optional[threading.Thread] = None

    def _courtesy_wait(self) -> None:
        """Let the sentiment shadow pass finish before a LOCAL call takes the
        server (bounded)."""
        if self._wait_for is None:
            return
        deadline = time.monotonic() + _COURTESY_WAIT_SECONDS
        waited = False
        while time.monotonic() < deadline:
            try:
                pending = int(self._wait_for() or 0)
            except Exception:
                return
            if pending <= 0:
                break
            waited = True
            time.sleep(_COURTESY_POLL_SECONDS)
        else:
            logger.warning("[engine_shadow] sentiment shadow still busy after "
                           f"{_COURTESY_WAIT_SECONDS:.0f}s — proceeding with the local call")
        if waited:
            logger.info("[engine_shadow] sentiment shadow drained — starting the local synthesis")

    def _run_arm(self, engine: str, variant: str) -> None:
        from src.analysis.claude_analyst import forced_synthesis_model
        rendered = resolve_variant(engine, variant)
        if engine == "local":
            self._courtesy_wait()
        t0 = time.monotonic()
        try:
            recs = self._generate(
                self._signals,
                open_positions=self._open_positions,
                session=self._session,
                force_engine=engine,
                prompt_variant=variant,
                **self._arm_kwargs,
                **self._synth_kwargs,
            )
        except Exception as exc:                    # never let a shadow break a run
            logger.warning(f"[engine_shadow] {engine}:{rendered} failed: {exc}")
            return
        latency = time.monotonic() - t0
        if not recs:
            # A forced-engine failure returns [] — record nothing rather than
            # a fabricated call.
            logger.warning(f"[engine_shadow] {engine}:{rendered} returned no "
                           f"recommendations ({latency:.0f}s)")
            return
        try:
            model = forced_synthesis_model(engine)
        except Exception:
            model = None
        rows = rows_for(engine=engine, model=model, prompt_variant=rendered, live=False,
                        recs=recs, prices=self._prices, latency_s=latency,
                        n_signals=len(self._signals or []), run_id=self._run_id,
                        generated_at=self._generated_at, signal_date=self._signal_date)
        _append_rows(rows)
        logger.info(f"[engine_shadow] {engine}:{rendered}: {len(rows)} shadow decision(s) "
                    f"in {latency:.0f}s (run {self._run_id})")

    def _run(self) -> None:
        try:
            for engine, variant in self._arms:
                self._run_arm(engine, variant)
        finally:
            _pending_delta(-1)

    def start(self) -> "EngineShadowBranch":
        _pending_delta(+1)
        self._thread = threading.Thread(target=self._run, name="engine-shadow", daemon=True)
        self._thread.start()
        return self

    def join(self, timeout: Optional[float] = None) -> bool:
        """Test hook — production never joins (the drain is non-blocking)."""
        if self._thread is not None:
            self._thread.join(timeout)
            return not self._thread.is_alive()
        return True


def maybe_start(*, signals, live_engine: Optional[str], live_model: Optional[str],
                live_recs, live_latency_s: Optional[float], synth_kwargs: dict,
                generate: Callable[..., List[Any]], run_id: str, generated_at: str,
                signal_date: str, open_positions=None, session: Optional[str] = None,
                arm_kwargs: Optional[dict] = None,
                wait_for: Optional[Callable[[], int]] = None) -> Optional[EngineShadowBranch]:
    """Queue the LIVE engine's decisions and start the shadow arms, when enabled.

    The live rows are queued synchronously (they exist already), so they land
    on THIS tick's drain even when every shadow arm is still running. Returns
    the branch, or ``None`` when the pass is disabled, there is nothing to
    pair, no arm resolves, or `synthesis_shadow_max_pending` branches are
    already in flight (single-flight — a slow local server must not pile up
    minutes-long calls tick after tick)."""
    if not getattr(settings, "enable_synthesis_shadow", False):
        return None
    if not signals or not live_recs:
        return None
    prices = {getattr(s, "ticker", None): getattr(s, "price", None) for s in (signals or [])}
    live = (live_engine or "").lower() or None
    _append_rows(rows_for(
        engine=live, model=live_model,
        prompt_variant=resolve_variant(live or "", "auto"), live=True,
        recs=live_recs, prices=prices, latency_s=live_latency_s,
        n_signals=len(signals), run_id=run_id, generated_at=generated_at,
        signal_date=signal_date,
    ))
    arms = shadow_arms(live)
    if not arms:
        logger.info(f"[engine_shadow] no shadow arm for live engine={live} — live rows only")
        return None
    cap = int(getattr(settings, "synthesis_shadow_max_pending", 2) or 0)
    if cap > 0 and engine_shadow_pending() >= cap:
        logger.warning(f"[engine_shadow] {engine_shadow_pending()} branch(es) still running "
                       f"(cap {cap}) — skipping the shadow arms this tick "
                       f"({', '.join(f'{e}:{resolve_variant(e, v)}' for e, v in arms)})")
        return None
    logger.info(f"[engine_shadow] live={live}; shadow arm(s): "
                f"{', '.join(f'{e}:{resolve_variant(e, v)}' for e, v in arms)}")
    return EngineShadowBranch(
        signals=signals, arms=arms, synth_kwargs=synth_kwargs, generate=generate,
        open_positions=open_positions, session=session, arm_kwargs=arm_kwargs,
        run_id=run_id, generated_at=generated_at, signal_date=signal_date,
        wait_for=wait_for,
    ).start()
