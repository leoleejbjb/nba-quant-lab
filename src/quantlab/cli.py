"""``quantlab`` command line interface.

Pipeline (each stage reads the previous stage's files; ``quantlab pipeline`` runs all of them):

    fetch -> build -> database -> features -> team -> minutes -> rates -> opponent -> constants
          -> backtest (2023-24 none, team) -> blendweight -> backtest (blend; 2024-26 all modes)
          -> baselines -> ablation -> topdown -> report

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

from .config import ALL_SEASONS, PATHS, PINNED_SEASONS, TEST_SEASONS, TUNING_SEASON


def _seasons(arg: str | None) -> list[str]:
    if not arg or arg == "all":
        return list(ALL_SEASONS)
    return [s.strip() for s in arg.split(",")]


def cmd_fetch(a) -> None:
    from .config import LIVE_SEASON
    from .data.download import fetch
    from .data.sources import all_files
    seasons = _seasons(a.season)
    feed = [s for s in seasons if s == LIVE_SEASON or s not in PINNED_SEASONS]
    if feed:
        raise SystemExit(f"{', '.join(feed)}: not in the pinned mirror. The current season ({LIVE_SEASON}) is fetched "
                         "game by game by `quantlab live-update` (from cdn.nba.com)")
    fetch(all_files(seasons), PATHS.raw)


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
    from .models.minutes import exact_availability_experiment, run_minutes, save_season_models
    ev = run_minutes()
    save_season_models()
    exact_availability_experiment(TUNING_SEASON)
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
    from .live import default_blend_weight
    from .sim.inputs import InputStore
    store = InputStore()
    w = a.blend_weight if getattr(a, "blend_weight", None) is not None else default_blend_weight()
    for season in _seasons(a.season):
        ids = store.games[store.games["season"] == season].sort_values("game_date").index.tolist()
        if a.limit:
            ids = ids[: a.limit]
        for mode in a.modes.split(","):
            t0 = time.time()
            run_backtest(ids, n_sims=a.n_sims, environment=mode, env_weight=w, store=store, tag=f"sim_{mode}_{season}")
            print(f"[backtest] {season} {mode}{f' (w={w:.2f})' if mode == 'blend' else ''}: {len(ids)} games in "
                  f"{time.time() - t0:.0f}s")


def cmd_blendweight(a) -> None:
    """Choose the team-model weight of the simulator's blend environment on the tuning season."""
    from .eval.report import choose_blend_weight
    choose_blend_weight()


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


def cmd_ablation(a) -> None:
    """Opponent / matchup ablation of the gradient-boosting player models (test seasons)."""
    import json
    from .eval.baselines import opponent_ablation
    out = opponent_ablation()
    PATHS.ensure(PATHS.backtest)
    (PATHS.backtest / "opponent_ablation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


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
    from .sim.inputs import PregameInputsMissing
    from .sim.scenario import Scenario
    pr = Projector()
    if pr.overrides.get("error"):
        print(f"[warning] {pr.overrides['error']}")
    sc = Scenario(out=[int(x) for x in a.out.split(",")] if a.out else [])
    try:
        proj = pr.project(a.game_id, n=a.n_sims, environment=a.environment, scenario=sc)
    except KeyError:
        raise SystemExit(f"game {a.game_id} is not in the database or the current season's schedule "
                         "(run `quantlab live-update` for the current season)") from None
    except PregameInputsMissing as e:
        raise SystemExit(f"game {a.game_id} cannot be projected yet: {e}") from None
    res = proj.result
    if proj.calibration.get("information_set") == "pregame":
        g = pr.game_info(a.game_id)
        when = ("final, but its play-by-play is not built yet" if str(g.get("status", "")) == "final"
                else "not played yet")
        print(f"[pregame] {pd.Timestamp(g['game_date']).date()}: {when} - projected from earlier games only, "
              "with projected starters and availability (players out: --out, or the injury report in "
              "data/live_overrides.json)")
        for side in ("away", "home"):
            m = proj.calibration["pregame"][side]
            names = [pr.store.names.get(x, str(x)) for x in m["starters"]]
            out = [pr.store.names.get(x, str(x)) for x in m["presumed_out"]]
            print(f"  {getattr(res.game, side).name}: starters ({m['starters_source']}) {', '.join(names)}; "
                  f"presumed out {', '.join(out) or 'none'}")
    if proj.calibration.get("minutes_reprojection"):
        print(f"[scenario] {proj.calibration['minutes_reprojection']}")
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


def cmd_slate(a) -> None:
    """Projected box scores of every game on a date: one CSV row per player (means and 10th-90th percentiles,
    conditional on playing, and P(play)), one printed line per game."""
    import numpy as np
    import pandas as pd
    from .live import Projector
    from .sim.inputs import PregameInputsMissing
    pr = Projector()
    if pr.overrides.get("error"):
        print(f"[warning] {pr.overrides['error']}")
    date = (pd.Timestamp.now(tz="America/New_York").date() if a.date == "today" else pd.Timestamp(a.date).date())
    sl = pr.slate(date)
    if sl.empty:
        raise SystemExit(f"no games on {date}")
    rows = []
    for g in sl.itertuples(index=False):
        label = f"{g.away_team} @ {g.home_team} (game {int(g.game_id)})"
        if not g.projectable:
            print(f"  {label}: not projected - {pr.why_not_projectable(int(g.game_id))}")
            continue
        try:
            proj = pr.project(int(g.game_id), n=a.n_sims, environment=a.environment)
        except PregameInputsMissing as e:          # one game's missing inputs must not stop the slate
            print(f"  {label}: not projected - {e}")
            continue
        res, spec = proj.result, proj.spec
        sc = res.score.astype(float)
        mg = sc[:, 0] - sc[:, 1]
        kind = ("finished game, as backtested" if proj.calibration.get("information_set") != "pregame" else
                "final, box score not built yet: pregame" if str(getattr(g, "status", "")) == "final" else "pregame")
        print(f"  {label}: {sc[:, 1].mean():.1f} - {sc[:, 0].mean():.1f}, home win {(mg > 0).mean():.1%}, "
              f"spread {-mg.mean():+.1f}, total {sc.sum(axis=1).mean():.1f} ({kind})")
        for t, team in enumerate((spec.home, spec.away)):
            opp = (spec.away if t == 0 else spec.home).name
            for i, p in enumerate(team.players):
                play = res.stat("min")[:, t, i] > 0
                if not play.any():
                    continue
                r = {"date": str(date), "game_id": int(g.game_id), "team": team.name, "opponent": opp,
                     "home": t == 0, "player": p.name, "person_id": p.person_id, "starter": bool(p.starter),
                     "p_play": round(float(play.mean()), 3), "information": kind}
                for st in ("min", "pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "pra"):
                    x = res.stat(st)[play, t, i]
                    r[st] = round(float(x.mean()), 2)
                    r[f"{st}_p10"], r[f"{st}_p90"] = (round(float(v), 1) for v in np.percentile(x, [10, 90]))
                rows.append(r)
    out = pd.DataFrame(rows)
    path = Path(a.out or f"projections_{date}.csv")
    out.to_csv(path, index=False, encoding="utf-8")
    print(f"[slate] {len(out)} player rows for {out['game_id'].nunique() if len(out) else 0} games -> {path}")


def cmd_live_update(a) -> None:
    """Current season: fetch new games from cdn.nba.com, rebuild the season, refresh features and models with the
    parameters fitted by the full pipeline (no re-tuning, no backtests), and give unplayed games pregame inputs."""
    import datetime as dt
    import json
    import pandas as pd
    from .config import LIVE_SEASON, UPCOMING_DAYS
    from .data.build import build_season
    from .data.cdn import FetchError, fetch_live
    from .data.database import build_database, upcoming_team_games
    from .data.download import AccessDenied
    from .features.asof import build_features
    from .models.minutes import run_minutes, save_season_models
    from .models.opponent import run_opponent
    from .models.rates import run_rates
    from .models.team import run_team_model
    need = [f for f in ("rate_params.json", "team_factor_params.json", "sim_constants.json")
            if not (PATHS.models / f).exists()]
    if need:
        raise SystemExit(f"live-update needs the fitted models first ({', '.join(need)} missing): run the full "
                         "pipeline once (`make all`, or `.\\make.ps1 all` on Windows).")
    if not (PATHS.models / "integration.json").exists():
        print("[live] note: no blend weight yet (data/models/integration.json, written by `quantlab blendweight` "
              "in the backtest step); the Blend environment falls back to 0.5")
    fetched = None
    if not a.skip_fetch:
        try:
            fetched = fetch_live(LIVE_SEASON, refetch_hours=a.refetch_hours)
        except AccessDenied as e:
            raise SystemExit(f"[live] {e}\nThe request was refused (by cdn.nba.com, or by a proxy on the way). Requests "
                             "are never retried under another identity: run `quantlab live-update` on your own "
                             "computer (cloud machines are often refused) without a proxy that blocks it, or rebuild "
                             "from the files already fetched with --skip-fetch.") from None
        except FetchError as e:
            raise SystemExit(f"[live] could not download the schedule: {e}\nCheck the internet connection and run "
                             "`quantlab live-update` again, or rebuild from the files already fetched with "
                             "--skip-fetch.") from None
    q = build_season(LIVE_SEASON, workers=a.workers)
    build_database()
    build_features()
    run_team_model()
    run_minutes()
    save_season_models(seasons=[LIVE_SEASON])
    run_rates(tune=False)
    run_opponent(tune=False)
    up = upcoming_team_games()
    # projectable: both teams have pregame rows (a team that has not played this season has none) and a team-model line
    fs = pd.read_parquet(PATHS.features / "player_asof.parquet", columns=["game_id", "team_id", "scheduled"])
    fs = fs[fs["scheduled"].eq(True)]
    tp_ids = set(pd.read_parquet(PATHS.models / "team_predictions.parquet", columns=["game_id"])["game_id"])
    both = fs.groupby("game_id")["team_id"].nunique()
    n_proj = int(sum(1 for g in up["game_id"].unique() if both.get(g, 0) == 2 and g in tp_ids))
    status = {"season": LIVE_SEASON, "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "games_built": int(q.get("games", 0)), "last_game_date": q.get("live", {}).get("last_game_date"),
              "upcoming_games": int(up["game_id"].nunique()), "projectable_games": n_proj,
              "upcoming_days": UPCOMING_DAYS,
              "upcoming_dates": sorted({str(pd.Timestamp(d).date()) for d in up["game_date"]}),
              "fetch": fetched}
    (PATHS.root / "live_status.json").write_text(json.dumps(status, indent=1, default=str), encoding="utf-8")
    if not status["games_built"]:
        print(f"[live] {LIVE_SEASON}: no finished games yet - a team's games are projected from its first game on")
    print(f"[live] {LIVE_SEASON}: {status['games_built']} games built through {status['last_game_date'] or '-'}; "
          f"{status['upcoming_games']} upcoming games in the next {UPCOMING_DAYS} days "
          f"({status['upcoming_dates'][0] if status['upcoming_dates'] else '-'} .. "
          f"{status['upcoming_dates'][-1] if status['upcoming_dates'] else '-'}), "
          f"{n_proj} of them projectable (both teams have played this season)")


def cmd_dashboard(a) -> None:
    app = Path(__file__).resolve().parent / "dashboard" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(a.port), "--server.headless", "true",
           "--theme.primaryColor", "#2a78d6", "--browser.gatherUsageStats", "false"]
    subprocess.run(cmd, check=False)


def cmd_pipeline(a) -> None:
    for fn in (cmd_fetch, cmd_build, cmd_database, cmd_features, cmd_team, cmd_minutes, cmd_rates, cmd_opponent,
               cmd_constants):
        fn(a)
    # bottom-up and team-constrained backtests; the blend weight is chosen on the tuning season from them,
    # then the shipped blend environment is backtested with that weight
    a.blend_weight = None
    a.modes, a.season = "none,team", TUNING_SEASON
    cmd_backtest(a)
    cmd_blendweight(a)
    a.modes, a.season = "blend", TUNING_SEASON
    cmd_backtest(a)
    a.modes, a.season = "none,team,blend", ",".join(TEST_SEASONS)
    cmd_backtest(a)
    for fn in (cmd_baselines, cmd_ablation, cmd_topdown, cmd_report):
        fn(a)


def cmd_sql(a) -> None:
    import pandas as pd
    from .data.warehouse import query
    with pd.option_context("display.width", 200, "display.max_columns", 50, "display.max_rows", 200):
        print(query(a.query))


def main(argv: list[str] | None = None) -> int:
    # output redirected to a file or a pipe: UTF-8 (player names, the starter check mark) instead of failing on the
    # Windows code page
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and not stream.isatty():
            stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="quantlab", description="NBA Quant Lab")
    sub = p.add_subparsers(dest="cmd", required=True)
    simple = {"database": cmd_database, "features": cmd_features, "team": cmd_team, "minutes": cmd_minutes,
              "rates": cmd_rates, "opponent": cmd_opponent, "constants": cmd_constants, "baselines": cmd_baselines,
              "topdown": cmd_topdown, "report": cmd_report, "blendweight": cmd_blendweight,
              "ablation": cmd_ablation}
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
    bt.add_argument("--blend-weight", type=float, default=None,
                    help="team-model weight for --modes blend (default: data/models/integration.json)")
    bt.set_defaults(fn=cmd_backtest)
    pj = sub.add_parser("project", help="projected box score for one game: an unplayed game of the current season "
                                        "truly pregame (projected starters and availability; --out for the injury "
                                        "report), a finished game as backtested")
    pj.add_argument("--game-id", type=int, required=True)
    pj.add_argument("--n-sims", type=int, default=20000)
    pj.add_argument("--environment", default="blend", choices=["blend", "team", "none"])
    pj.add_argument("--out", default="", help="comma-separated person ids ruled out")
    pj.set_defaults(fn=cmd_project)
    sl = sub.add_parser("slate", help="projected box scores of every game on a date, written to a CSV")
    sl.add_argument("--date", default="today", help="YYYY-MM-DD or 'today' (US Eastern)")
    sl.add_argument("--n-sims", type=int, default=10000)
    sl.add_argument("--environment", default="blend", choices=["blend", "team", "none"])
    sl.add_argument("--out", default="", help="CSV path (default projections_<date>.csv)")
    sl.set_defaults(fn=cmd_slate)
    lu = sub.add_parser("live-update", help="current season: fetch new games from cdn.nba.com, rebuild the season, "
                                            "refresh features and models (run it on your own computer)")
    lu.add_argument("--skip-fetch", action="store_true", help="rebuild from the files already fetched")
    lu.add_argument("--refetch-hours", type=float, default=36.0,
                    help="download games again until this many hours after tip-off (post-game stat corrections)")
    lu.add_argument("--workers", type=int, default=4)
    lu.set_defaults(fn=cmd_live_update)
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
