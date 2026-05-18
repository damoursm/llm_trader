"""Seed cache/pattern_registry.json from the existing closed trades.

For each closed trade that lacks ``pattern_at_entry``, this script reconstructs
the chart pattern that was active on its ``entry_date`` by:

  1. Loading the ticker's full OHLCV cache,
  2. Slicing to bars strictly before ``entry_date`` (no look-ahead),
  3. Running ``pattern_recognition._detect_pattern`` on the most recent
     _DETECT_WINDOW bars of that slice,
  4. Recording the detected pattern + the score that would have been emitted
     under the *synthetic-only* library at that time (no live blend, since
     the registry is what we're trying to seed).

Side effects
────────────
- ``cache/trades.json`` is updated in place with ``pattern_at_entry`` and
  ``pattern_score_at_entry`` set on every trade that produced a detection.
  A timestamped backup is written to ``cache/trades.json.bak.<ISO>`` first.
- ``cache/pattern_registry.json`` is updated via ``record_batch`` (idempotent).

Run
───
    python scripts/backfill_pattern_registry.py            # do it
    python scripts/backfill_pattern_registry.py --dry-run  # show what would happen
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from src.data.cache import load_ohlcv  # noqa: E402
from src.signals.pattern_recognition import (  # noqa: E402
    _detect_pattern,
    _load_library,
    _PATTERN_DIR,
    _DETECT_WINDOW,
    _WEAK_PRIOR,
    _MIN_PATTERN_N,
)
from src.signals.pattern_registry import record_batch  # noqa: E402


TRADES_FILE = PROJECT_ROOT / "cache" / "trades.json"


def detect_pattern_at_entry(ticker: str, entry_date_str: str) -> tuple[str, float]:
    """Re-detect the pattern that was active just before ``entry_date``.

    Returns ``(pattern_name, score)`` or ``("", 0.0)`` when no pattern was
    present or the OHLCV cache lacks enough history.

    The score uses the synthetic library only (no live blend) — same formula
    as ``compute_pattern_score`` would have produced at entry time IF the
    registry didn't exist yet.
    """
    df = load_ohlcv(ticker)
    if df is None or df.empty:
        return "", 0.0
    try:
        ed = date.fromisoformat(entry_date_str)
    except (TypeError, ValueError):
        return "", 0.0

    # Slice strictly before entry_date — no look-ahead
    try:
        idx_dates = df.index.date  # type: ignore[attr-defined]
        mask = idx_dates < ed
        df_prev = df.loc[mask]
    except Exception:
        return "", 0.0
    if len(df_prev) < 20:
        return "", 0.0

    recent_prices = df_prev["Close"].values[-_DETECT_WINDOW:].astype(float)
    pattern = _detect_pattern(recent_prices)
    if pattern is None:
        return "", 0.0

    inherent = _PATTERN_DIR.get(pattern, 0)
    if inherent == 0:
        return "", 0.0

    # Score under the synthetic-only formula. If a library exists for this
    # ticker, use its per-ticker win rate; otherwise fall back to the weak
    # prior. We do not consult the registry here on purpose — this is the
    # score the system would have stored at entry-time *under the old rules*.
    library = _load_library(ticker)
    pstats = (library or {}).get("patterns", {}).get(pattern)
    if pstats and pstats.get("count", 0) >= _MIN_PATTERN_N:
        wr = float(pstats["win_rate"])
        edge = (wr - 0.5) * 2.0
        score = round(max(-1.0, min(1.0, edge * inherent)), 3)
    else:
        score = round(_WEAK_PRIOR * inherent, 3)

    return pattern, score


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill pattern_at_entry on closed trades and seed pattern_registry.json.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing.")
    args = parser.parse_args()

    if not TRADES_FILE.exists():
        print(f"No trades file at {TRADES_FILE}", file=sys.stderr)
        return 1

    trades = json.loads(TRADES_FILE.read_text(encoding="utf-8"))

    # Find candidates: any trade (closed or open) without pattern_at_entry
    candidates = [t for t in trades if not t.get("pattern_at_entry")]
    if not candidates:
        print(f"No trades need backfill — {len(trades)} already tagged.")
        return 0

    print(f"{'TICKER':<8}{'ACTION':<7}{'ENTRY_DATE':<12}  {'PATTERN':<22}{'SCORE':>8}{'STATUS':>10}")
    print("-" * 78)

    matched = 0
    for t in candidates:
        ticker = t.get("ticker", "?")
        action = t.get("action", "?")
        entry  = t.get("entry_date", "?")
        status = t.get("status", "?")
        pattern, score = detect_pattern_at_entry(ticker, entry)
        if pattern:
            t["pattern_at_entry"]       = pattern
            t["pattern_score_at_entry"] = score
            matched += 1
            tag = "matched"
        else:
            tag = "no_pattern"
        print(f"  {ticker:<6}{action:<7}{entry:<12}  {pattern:<22}{score:>+8.3f}  {tag:<10} ({status})")

    print("-" * 78)
    print(f"Detected pattern on {matched} of {len(candidates)} candidates.\n")

    if args.dry_run:
        print("[dry-run] No files modified.")
        return 0

    # Backup before writing
    if matched > 0:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = TRADES_FILE.with_suffix(f".json.bak.{stamp}")
        shutil.copy(TRADES_FILE, backup)
        print(f"Backed up trades.json to {backup}")

        TRADES_FILE.write_text(json.dumps(trades, indent=2, default=str), encoding="utf-8")
        print(f"Wrote {matched} new pattern_at_entry tags to {TRADES_FILE}")

    # Seed the registry from all (now-tagged) closed trades
    closed = [t for t in trades if t.get("status") == "CLOSED"]
    added = record_batch(closed)
    print(f"\nRegistered {added} closed outcome(s) into cache/pattern_registry.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
