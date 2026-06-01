"""DuckDB persistence layer — single source of truth for trades, recommendations,
and per-run metadata (including which APIs/sources were used).

The daily pipeline is the sole writer; the dashboard connects read-only. See
`connection.connect` for the concurrency model and `repo` for the read/write API.
"""

from src.db import connection, repo, schema

__all__ = ["connection", "repo", "schema"]
