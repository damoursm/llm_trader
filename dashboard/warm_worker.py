"""Compute the dashboard's heavy caches in a CHILD process, off the server's GIL.

Why this exists (measured 2026-08-14). The warm sweep is ~400 s of pure pandas/
DuckDB CPU spread over 9 accessors — there is no single hog to optimise away —
and it ran on a background thread INSIDE the waitress process. Python's GIL made
that fatal: a `py-spy` dump during an outage showed all 8 worker threads idle and
one thread `active+gil` in `warm_caches`, while a **static 4 KB file timed out
from localhost**. The workers were never the problem; waitress's main loop could
not get the GIL to accept the connection. With the sweep grown to 1,587 s against
a ~30-minute pipeline tick, the dashboard was unreachable nearly all the time.

A child process has its own GIL, so the parent stays responsive throughout.

Contract: ``python -m dashboard.warm_worker <out_path>`` warms every target and
writes ``{"ver", "cache": {key: pickled_entry}, "dropped": [...]}`` to out_path.
Entries are pickled INDIVIDUALLY so one unpicklable value costs that accessor
only, instead of the whole sweep. Keys and entry shape come from running the very
same ``warm_caches()`` the parent would have run, which is what guarantees the
parent can merge them straight into ``_perf_cache`` — key parity by construction
rather than by a reimplementation that could silently drift.
"""

from __future__ import annotations

import pickle
import sys


def main(out_path: str) -> int:
    from dashboard import data          # sets repo read-only at import

    ver = data._data_version()
    data.warm_caches(reason="subprocess")

    cache: dict = {}
    dropped: list = []
    for key, entry in list(data._perf_cache.items()):
        try:
            cache[key] = pickle.dumps(entry, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:                       # value isn't transferable
            dropped.append(f"{key!r}: {type(e).__name__}")

    with open(out_path, "wb") as fh:
        # "code" lets the parent reject a snapshot written by DIFFERENT code:
        # cached values can change shape across a deploy, and a restored old
        # shape renders as empty with nothing to explain why (see
        # data._code_fingerprint).
        pickle.dump({"ver": ver, "code": data._code_fingerprint(),
                     "cache": cache, "dropped": dropped}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"warm_worker: {len(cache)} entries written, {len(dropped)} dropped", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m dashboard.warm_worker <out_path>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
