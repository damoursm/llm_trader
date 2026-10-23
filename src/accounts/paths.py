"""THE ACCOUNTS' OWN STATE (from 2026-10-23: the second account — its own ledger, calculated from the main account;
`docs/SECOND_ACCOUNT.md`).

Every live book's DECISIONS are made once, by the main account's scheduler, and SHARED: the selection short's
picks journal and its live-arm markers (the scorer subprocess), the dip book's daily signals, the crash book's daily
decision, the capitulation book's breadth and signals. An account's own STATE is its own: its ledger database
(`db_path`), the picks it consumed, its entry journals and trade logs, its dip / capitulation orders, its books'
entries and exits journals, its margin check's state and liquidations and its fresh-start marker (the legacy
flatten's stays shared: the main account's one-shot). `state(p)` is where account-specific path ``p`` lives for the account this process trades:
the main account's root is "" and every path stays where it always was (``state(p) == Path(p)``, the default — no
path of the main account moves); another account's under its ``account_state_root``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

from config.settings import settings

MAIN = "main"


def name() -> str:
    """The account this process trades ("main" for the main account)."""
    return str(getattr(settings, "account_name", MAIN) or MAIN)


def root() -> str:
    """This process's account state root ("" for the main account)."""
    return str(getattr(settings, "account_state_root", "") or "")


def is_main() -> bool:
    return not root()


def state(path: Union[str, Path]) -> Path:
    """Where account-specific ``path`` lives for this process's account: ``path`` itself for the main account, else
    the same relative path under the account's root (an absolute path — a test's temporary folder — by its parts
    after the drive)."""
    p = Path(path)
    r = root()
    if not r:
        return p
    if p.is_absolute():
        p = Path(*p.parts[1:])
    return Path(r) / p


class NotReady(RuntimeError):
    """A shared decision the main account's tick has not made yet. An extra account never makes it — it would race
    the main account's computation (and its EDGAR, Polygon and store traffic): its step fails soft and the account's
    next tick reads the decision once the main account has made it."""


def decide_here(what: str) -> None:
    """Raise `NotReady` in an extra account's process: ``what`` is the main account's decision to make."""
    if not is_main():
        raise NotReady(f"{what} is the main account's decision and is not made yet — the {name()} account waits "
                       f"for it")
