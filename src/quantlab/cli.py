"""``quantlab`` command line interface."""

from __future__ import annotations

import argparse
import sys
import time

from .config import ALL_SEASONS, PATHS


def _seasons(arg: str | None) -> list[str]:
    if not arg or arg == "all":
        return list(ALL_SEASONS)
    return [s.strip() for s in arg.split(",")]


def cmd_fetch(a) -> None:
    from .data.download import fetch
    from .data.sources import all_files
    fetch(all_files(_seasons(a.season)), PATHS.raw)


def cmd_build(a) -> None:
    from .data.build import build_season
    for s in _seasons(a.season):
        build_season(s, workers=a.workers)


def cmd_database(a) -> None:
    from .data.database import build_database
    build_database()


def cmd_sql(a) -> None:
    import pandas as pd
    from .data.warehouse import query
    with pd.option_context("display.width", 200, "display.max_columns", 50, "display.max_rows", 200):
        print(query(a.query))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="quantlab", description="NBA Quant Lab")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download + verify pinned raw files")
    f.add_argument("--season", default="all")
    f.set_defaults(fn=cmd_fetch)
    b = sub.add_parser("build", help="reconstruct seasons into player/team box scores, stints, possessions")
    b.add_argument("--season", default="all")
    b.add_argument("--workers", type=int, default=4)
    b.set_defaults(fn=cmd_build)
    d = sub.add_parser("database", help="unify seasons into data/processed/all")
    d.set_defaults(fn=cmd_database)
    q = sub.add_parser("sql", help="run a DuckDB query over the processed tables")
    q.add_argument("query")
    q.set_defaults(fn=cmd_sql)
    a = p.parse_args(argv)
    t0 = time.perf_counter()
    a.fn(a)
    print(f"[quantlab] {a.cmd} finished in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
