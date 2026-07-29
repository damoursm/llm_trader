"""Shadow synthesis arms — every prompt arm's call on every ticker, every tick.

The problem this solves
-----------------------
The blind-synthesis A/B was evaluated by stamping each run's arm onto the trades
it OPENED and grouping closed-trade outcomes by that stamp. That sample is tiny
and doubly biased: only gate-surviving calls become trades, and each run
contributes to exactly ONE arm, so the arms are compared over DIFFERENT
ticker-days. The 2026-07-22 bake-off established how badly that misleads here --
Qwen's apparent lead and pro-thinking's apparent collapse were BOTH pure window
artifacts (4 days vs 23), and every difference vanished under paired testing.

So an arm is only judgeable against another arm's call on the SAME ticker, the
same tick, with the same context and the same engine. That requires actually
asking each arm, which is what this module does: after the live arm has produced
the recommendations that drive the run, the other arms are asked the same
question about the same tickers and their answers are recorded and acted on by
nobody.

Design constraints
------------------
* OFF THE CRITICAL PATH. Shadow calls run in a background thread started after
  the live recommendations exist, and are joined at persist time. Orders never
  wait on a shadow arm.
* ENGINE-CONTROLLED. Shadow calls are pinned (``force_engine``) to the engine
  the live arm actually used, so the ARM is the only thing that differs. This
  also suppresses the run/sentiment provenance writes, so a shadow call can
  never mis-stamp the run or its trades (same guarantee the pinned hold-review
  relies on).
* FAIL-SOFT. Any failure yields no shadow rows and never touches the run.

Cost: two extra synthesis calls per tick (synthesis is ONE call for the whole
top-N ticker set, unlike per-ticker sentiment), disable with
``enable_shadow_arms``.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional

from config.settings import settings

from loguru import logger  # project configures loguru sinks only

# The three prompt arms. Kept as an explicit tuple (not derived from settings)
# so the evaluation surface is stable even when a share is set to 0.
ARMS = ("dual", "blind", "sighted")

# Arm -> the generate_recommendations kwargs that select it.
_ARM_KWARGS: Dict[str, Dict[str, bool]] = {
    "dual":    {"dual_case": True,  "blind_synthesis": False},
    "blind":   {"dual_case": False, "blind_synthesis": True},
    "sighted": {"dual_case": False, "blind_synthesis": False},
}


def live_arm_name(dual_case: bool, blind_synthesis: bool) -> str:
    """The arm name for this run's flip (mirrors pipeline's own precedence)."""
    if dual_case:
        return "dual"
    return "blind" if blind_synthesis else "sighted"


def _rows_for(arm: str, live: bool, recs: List[Any],
              prices: Dict[str, float]) -> List[dict]:
    """Flatten one arm's recommendations into persistable rows."""
    out: List[dict] = []
    for r in recs or []:
        tic = getattr(r, "ticker", None)
        if not tic:
            continue
        out.append({
            "arm": arm,
            "live": live,
            "ticker": tic,
            "action": getattr(r, "action", None),
            "direction": getattr(r, "direction", None),
            "confidence": getattr(r, "confidence", None),
            "snap_price": prices.get(tic),
        })
    return out


class ShadowArmBranch:
    """Runs the non-live arms in the background; join before persisting.

    Usage mirrors ``_HoldReviewBranch``: construct + ``start()`` once the live
    recommendations exist, then ``rows()`` at persist time (joins, bounded).
    """

    def __init__(self, *, signals, live_arm: str, live_recs: List[Any],
                 synth_kwargs: dict, generate: Callable[..., List[Any]],
                 force_engine: Optional[str] = None,
                 open_positions=None, session: Optional[str] = None):
        self._signals = signals
        self._live_arm = live_arm
        self._live_recs = list(live_recs or [])
        self._synth_kwargs = synth_kwargs
        self._generate = generate
        self._force_engine = force_engine
        self._open_positions = open_positions
        self._session = session
        self._prices = {getattr(s, "ticker", None): getattr(s, "price", None)
                        for s in (signals or [])}
        self._rows: List[dict] = []
        self._thread: Optional[threading.Thread] = None

    # ── background work ────────────────────────────────────────────────────
    def _run(self) -> None:
        rows = _rows_for(self._live_arm, True, self._live_recs, self._prices)
        for arm in ARMS:
            if arm == self._live_arm:
                continue
            try:
                recs = self._generate(
                    self._signals,
                    open_positions=self._open_positions,
                    session=self._session,
                    force_engine=self._force_engine,
                    **_ARM_KWARGS[arm],
                    **self._synth_kwargs,
                )
            except Exception as exc:            # never let a shadow break a run
                logger.warning(f"[shadow_arm] {arm} failed: {exc}")
                continue
            if not recs:
                # A forced-engine failure returns [] — record nothing rather
                # than a fabricated call.
                logger.warning(f"[shadow_arm] {arm} returned no recommendations")
                continue
            rows.extend(_rows_for(arm, False, recs, self._prices))
            logger.info(f"[shadow_arm] {arm}: {len(recs)} shadow calls")
        self._rows = rows

    def start(self) -> "ShadowArmBranch":
        self._thread = threading.Thread(target=self._run, name="shadow-arms",
                                        daemon=True)
        self._thread.start()
        return self

    def rows(self, timeout: Optional[float] = None) -> List[dict]:
        """Join and return every arm's rows (live arm included)."""
        if self._thread is not None:
            self._thread.join(timeout if timeout is not None
                              else float(settings.shadow_arms_join_timeout_seconds))
            if self._thread.is_alive():
                logger.warning("[shadow_arm] join timed out — persisting live arm only")
                return _rows_for(self._live_arm, True, self._live_recs, self._prices)
        return self._rows


def maybe_start(*, signals, live_arm: str, live_recs, synth_kwargs: dict,
                generate: Callable[..., List[Any]],
                force_engine: Optional[str] = None,
                open_positions=None,
                session: Optional[str] = None) -> Optional[ShadowArmBranch]:
    """Start the shadow branch when enabled, else ``None``.

    Returns None (rather than a no-op branch) so the caller can also skip the
    persist step entirely on a disabled run.
    """
    if not getattr(settings, "enable_shadow_arms", False):
        return None
    if not signals or not live_recs:
        return None
    return ShadowArmBranch(
        signals=signals, live_arm=live_arm, live_recs=live_recs,
        synth_kwargs=synth_kwargs, generate=generate,
        force_engine=force_engine, open_positions=open_positions,
        session=session,
    ).start()
