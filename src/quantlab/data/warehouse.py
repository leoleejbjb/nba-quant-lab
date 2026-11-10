"""DuckDB views over the processed Parquet tables.

Unified tables (``player_games``, ``team_games``, ``games``, ``availability``,
``players``) come from ``data/processed/all``; per-season tables that are only
needed for lineup work (``stints``, ``possessions``, ``box_compare``) are
exposed as views over all seasons with a ``season`` column.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from ..config import PATHS

PER_SEASON = ("stints", "possessions", "box_compare", "starters", "contradictions")


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for path in sorted(PATHS.processed("all").glob("*.parquet")):
        if not path.name.endswith(".tmp.parquet"):
            con.execute(f"CREATE VIEW {path.stem} AS SELECT * FROM read_parquet('{path.as_posix()}')")
    root = PATHS.root / "processed"
    for name in PER_SEASON:
        files = sorted(root.glob(f"*/{name}.parquet"))
        files = [f for f in files if f.parent.name != "all"]
        if files:
            lst = ", ".join(f"'{f.as_posix()}'" for f in files)
            con.execute(f"CREATE VIEW {name} AS SELECT *, regexp_extract(filename, '(\\d{{4}}-\\d{{2}})', 1) AS season "
                        f"FROM read_parquet([{lst}], filename = true, union_by_name = true)")
    return con


def query(sql: str) -> pd.DataFrame:
    con = connect()
    try:
        return con.execute(sql).df()
    finally:
        con.close()
