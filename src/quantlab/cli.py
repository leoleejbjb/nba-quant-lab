"""``quantlab`` command line interface.

Pipeline (each stage reads the previous stage's files; ``quantlab pipeline`` runs all of them):

    fetch -> build -> database -> features -> team -> minutes -> rates -> opponent -> constants
          -> backtest -> baselines -> topdown -> report

Use ``quantlab project --game-id 22501201`` to print a projected box score, or
``quantlab dashboard`` for the interactive dashboard.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from .config import ALL_SEASONS, PATHS, TEST_SEASONS, TUNING_SEASON


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


def cmd_features(a) -> None:
    from .features.asof import build_features
    build_features()


def cmd_team(a) -> None:
    from .models.team import run_team_model
    run_team_model()


def cmd_minutes(a) -> None:
    import pandas as pd
    from .models.minutes import run_minutes, save_season_models
    ev = run_minutes()
    save_season_models()
    with pd.option_context("display.width", 200):
        print(ev[ev["season"] == "all"].sort_values("mae").round(3).to_string(index=False))


def cmd_rates(a) -> None:
    from .models.rates import run_rates
    run_rates()


def cmd_opponent(a) -> None:
    from .models.opponent import run_opponent
    run_opponent()


def cmd_constants(a) -> None:
    from .sim.constants import compute_constants
    compute_constants()


def cmd_backtest(a) -> None:
    from .eval.backtest import run_backtest
    from .sim.inputs import InputStore
    store = InputStore()
    for season in _seasons(a.season):
        ids = store.games[store.games["season"] == season].sort_values("game_date").index.tolist()
        if a.limit:
            ids = ids[: a.limit]
        for mode in a.modes.split(","):
            t0 = time.time()
            run_backtest(ids, n_sims=a.n_sims, environment=mode, store=store, tag=f"sim_{mode}_{season}")
            print(f"[backtest] {season} {mode}: {len(ids)} games in {time.time() - t0:.0f}s")


def cmd_baselines(a) -> None:
    from .eval.baselines import (STATS, add_model_inputs, baseline_predictions, gbm_predictions, load_frame)
    seasons = [TUNING_SEASON, *TEST_SEASONS]
    df = load_frame()
    d = add_model_inputs(df[df["played"] & df["season"].isin(seasons)].copy()).reset_index(drop=True)
    preds = baseline_predictions(d)
    g = gbm_predictions(d, {TEST_SEASONS[0]: [TUNING_SEASON], TEST_SEASONS[1]: [TUNING_SEASON, TEST_SEASONS[0]]})
    keep = d[["game_id", "person_id", "team_id", "season", "rotation", "starter", "pred_gbm"] + STATS].copy()
    for s in STATS:
        preds[s]["gbm"] = g[s]
        for m, v in preds[s].items():
            keep[f"{s}__{m}"] = v
    PATHS.ensure(PATHS.backtest)
    keep.to_parquet(PATHS.backtest / "baseline_predictions.parquet", index=False)
    print(f"[baselines] {len(keep):,} player-games")


def cmd_topdown(a) -> None:
    from .eval.topdown import run_topdown
    run_topdown([TUNING_SEASON])
    run_topdown(list(TEST_SEASONS))


def cmd_report(a) -> None:
    from .eval.report import run_report
    from .eval.writeup import write_findings
    summ = run_report()
    write_findings(summ)


def cmd_project(a) -> None:
    import pandas as pd
    from .live import Projector, box_score_table
    from .sim import props
    from .sim.scenario import Scenario
    pr = Projector()
    sc = Scenario(out=[int(x) for x in a.out.split(",")] if a.out else [])
    proj = pr.project(a.game_id, n=a.n_sims, environment=a.environment, scenario=sc)
    res = proj.result
    sc_ = res.score.astype(float)
    mg = sc_[:, 0] - sc_[:, 1]
    print(f"{res.game.away.name} @ {res.game.home.name}: {sc_[:, 1].mean():.1f} - {sc_[:, 0].mean():.1f}, "
          f"P(home win) {(mg > 0).mean():.1%}, total {sc_.sum(axis=1).mean():.1f}, P(OT) {(res.n_ot > 0).mean():.1%}")
    cols = ["Player", "Start", "MIN", "PTS", "PTS_lo", "PTS_hi", "REB", "AST", "3PM", "FGA", "FTA", "STL", "BLK", "TOV"]
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        for t in (1, 0):
            print(box_score_table(res, t)[cols].round(1).to_string(index=False))
    star = max(((t, i) for t in (0, 1) for i in range(len(res.players[t]))),
               key=lambda ti: res.stat("pts")[:, ti[0], ti[1]].mean())
    pid = res.players[star[0]][star[1]].person_id
    x = props.samples(res, pid, "pts")
    L = float(round(x.mean() - 0.5) + 0.5)
    print(f"{res.players[star[0]][star[1]].name}: over {L} pts {props.line_probs(x, L)['over']:.1%}, "
          f"double-double {props.double_double(res, pid):.1%}; invariants: {res.check_invariants()['all']}")


def cmd_dashboard(a) -> None:
    app = Path(__file__).resolve().parent / "dashboard" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(a.port), "--server.headless", "true"]
    subprocess.run(cmd, check=False)


def cmd_pipeline(a) -> None:
    for fn in (cmd_fetch, cmd_build, cmd_database, cmd_features, cmd_team, cmd_minutes, cmd_rates, cmd_opponent,
               cmd_constants):
        fn(a)
    a.modes = "none,team"
    a.season = ",".join([TUNING_SEASON, *TEST_SEASONS])
    for fn in (cmd_backtest, cmd_baselines, cmd_topdown, cmd_report):
        fn(a)


def cmd_sql(a) -> None:
    import pandas as pd
    from .data.warehouse import query
    with pd.option_context("display.width", 200, "display.max_columns", 50, "display.max_rows", 200):
        print(query(a.query))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="quantlab", description="NBA Quant Lab")
    sub = p.add_subparsers(dest="cmd", required=True)
    simple = {"database": cmd_database, "features": cmd_features, "team": cmd_team, "minutes": cmd_minutes,
              "rates": cmd_rates, "opponent": cmd_opponent, "constants": cmd_constants, "baselines": cmd_baselines,
              "topdown": cmd_topdown, "report": cmd_report}
    f = sub.add_parser("fetch", help="download + verify pinned raw files")
    f.add_argument("--season", default="all")
    f.set_defaults(fn=cmd_fetch)
    b = sub.add_parser("build", help="reconstruct seasons into player/team box scores, stints, possessions")
    b.add_argument("--season", default="all")
    b.add_argument("--workers", type=int, default=4)
    b.set_defaults(fn=cmd_build)
    for name, fn in simple.items():
        sub.add_parser(name, help=(fn.__doc__ or name)).set_defaults(fn=fn)
    bt = sub.add_parser("backtest", help="walk-forward simulation backtest")
    bt.add_argument("--season", default=",".join([TUNING_SEASON, *TEST_SEASONS]))
    bt.add_argument("--modes", default="none,team")
    bt.add_argument("--n-sims", type=int, default=1200)
    bt.add_argument("--limit", type=int, default=0)
    bt.set_defaults(fn=cmd_backtest)
    pj = sub.add_parser("project", help="projected box score for one game (pregame information only)")
    pj.add_argument("--game-id", type=int, required=True)
    pj.add_argument("--n-sims", type=int, default=20000)
    pj.add_argument("--environment", default="blend", choices=["blend", "team", "none"])
    pj.add_argument("--out", default="", help="comma-separated person ids ruled out")
    pj.set_defaults(fn=cmd_project)
    d = sub.add_parser("dashboard", help="launch the Streamlit dashboard")
    d.add_argument("--port", type=int, default=8501)
    d.set_defaults(fn=cmd_dashboard)
    pl = sub.add_parser("pipeline", help="run every stage end to end")
    pl.add_argument("--season", default="all")
    pl.add_argument("--workers", type=int, default=4)
    pl.add_argument("--n-sims", type=int, default=1200)
    pl.add_argument("--limit", type=int, default=0)
    pl.set_defaults(fn=cmd_pipeline)
    q = sub.add_parser("sql", help="run a DuckDB query over the processed tables")
    q.add_argument("query")
    q.set_defaults(fn=cmd_sql)
    a = p.parse_args(argv)
    t0 = time.perf_counter()
    a.fn(a)
    print(f"[quantlab] {a.cmd} finished in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
