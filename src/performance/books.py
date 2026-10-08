"""The ledger's LIVE books — the entry mechanisms that trade. Every other open trade belongs to a shadow
book: the legacy exit stack and the signal reversal never touch a live book's trade, the one-shot legacy
flatten skips it, and the broker sends a live book's entries while the legacy books are shadow
(`reconcile.sync`). One definition, read by the tracker and the reconciler."""
from __future__ import annotations

SEL_SHORT = "sel_short"      # the selection short (src/signals/sel_short.py, from 2026-09-28)
DIP_LONG = "dip_long"        # the mega-cap dip long book (src/signals/dip_long.py, from 2026-10-08)
LIVE_MECHANISMS = (SEL_SHORT, DIP_LONG)


def is_live(trade: dict) -> bool:
    """The trade belongs to a live book (it owns its exits; the broker sends its entries)."""
    return trade.get("entry_mechanism") in LIVE_MECHANISMS
