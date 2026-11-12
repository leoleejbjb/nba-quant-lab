"""Unified multi-season database built from the per-season processed tables.

Tables written to ``data/processed/all/`` (also exposed as DuckDB views by
``warehouse.connect``):

* ``player_games``  every player-game 2020-21..2025-26 (box score + on-court exposures).
  ``min`` is the official NBA.com minutes when available (2020-21..2023-24),
  otherwise the play-by-play reconstruction (``min_rec``, within 1 s of the
  official value for 99.9% of 2023-24 player-games). Coaches and staff (people
  who never played, charged with bench technicals) are not player-games.
* ``availability``  per team-game roster status. 2020-21..2023-24, from the official box-score
  comments: ``played``, ``dnp_cd`` (dressed but unused: "DNP - Coach's Decision" or listed at 0:00)
  or ``out`` (every DND / NWT / DNP-injury / rest / suspension comment). 2024-25 and 2025-26 have no
  official box scores, so only ``played`` rows (from play-by-play) exist there: who was dressed but
  unused, and who was out, is unknown for those seasons (``source`` = ``pbp``).
* ``team_games``    team box score with the opponent's columns joined (``opp_*``), rest days.
* ``games``         one row per game.
* ``players``       one row per player: name, career totals, last team.
* ``schedule``, ``team_schedule``  the live season's schedule (one row per game / per team-game, with rest days and
  back-to-backs over the full schedule); ``built`` marks games reconstructed into the tables above. Unplayed games
  are never written into ``games`` / ``player_games`` / ``team_games``.

The database holds the pinned seasons plus the live season once it has been built (``config.db_seasons``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import LIVE_SEASON, PATHS, UPCOMING_DAYS, db_seasons
from .build import _write

SCHEDULE_TEAM_COLS = ["game_id", "team_id", "opp_team_id", "is_home", "season", "game_date", "tip_utc", "rest_days",
                      "b2b", "status", "built"]

def availability_status(comment: pd.Series, box_seconds: pd.Series) -> pd.Series:
    """played / dnp_cd (dressed, unused) / out. 'DND - Coach's Decision' (did not dress) is out."""
    c = comment.fillna("").str.strip()
    out = np.where(box_seconds.fillna(0) > 0, "played",
                   np.where(c.str.startswith("DNP - Coach") | (c == ""), "dnp_cd", "out"))
    return pd.Series(out, index=comment.index)


def build_schedule_tables(seasons, out_dir) -> tuple[int, pd.DataFrame | None]:
    """all/schedule.parquet and all/team_schedule.parquet from the per-season schedules (seasons built from the
    live feed). Returns the number of scheduled games and the team schedule."""
    paths = [PATHS.processed(s) / "schedule.parquet" for s in seasons]
    scheds = [pd.read_parquet(p) for p in paths if p.exists()]
    if not scheds:
        for name in ("schedule", "team_schedule"):
            (out_dir / f"{name}.parquet").unlink(missing_ok=True)
        return 0, None
    sc = pd.concat(scheds, ignore_index=True)
    sc["game_date"] = pd.to_datetime(sc["game_date"])
    known = sc[(sc["home_team_id"] > 0) & (sc["away_team_id"] > 0) & ~sc["status"].isin(["tbd", "postponed"])]
    keep = ["game_id", "season", "game_date", "tip_utc", "status", "built"]
    home = known[keep].assign(team_id=known["home_team_id"], opp_team_id=known["away_team_id"], is_home=True)
    away = known[keep].assign(team_id=known["away_team_id"], opp_team_id=known["home_team_id"], is_home=False)
    ts = pd.concat([home, away], ignore_index=True).sort_values(["team_id", "game_date", "game_id"])
    prev = ts.groupby(["team_id", "season"])["game_date"].shift(1)
    ts["rest_days"] = (ts["game_date"] - prev).dt.days.fillna(7).clip(upper=7)   # the rule used for team_games
    ts["b2b"] = ts["rest_days"] == 1
    ts = ts[SCHEDULE_TEAM_COLS].reset_index(drop=True)
    _write(sc, out_dir / "schedule.parquet")
    _write(ts, out_dir / "team_schedule.parquet")
    return int(len(sc)), ts


def upcoming_team_games(horizon_days: int = UPCOMING_DAYS) -> pd.DataFrame:
    """Team-games of the live season that are not built yet (scheduled, in progress, or final without a usable
    play-by-play file) and are dated no later than ``horizon_days`` after its last built game: the games that get
    pregame projections."""
    p = PATHS.processed("all") / "team_schedule.parquet"
    if not p.exists():
        return pd.DataFrame(columns=SCHEDULE_TEAM_COLS)
    ts = pd.read_parquet(p)
    ts = ts[ts["season"] == LIVE_SEASON]
    built = ts[ts["built"]]
    start = built["game_date"].max() if len(built) else ts["game_date"].min() - pd.Timedelta(days=1)
    up = ts[~ts["built"] & (ts["game_date"] <= start + pd.Timedelta(days=horizon_days))]
    return up.reset_index(drop=True)


def build_database(seasons=None, log=print) -> dict:
    seasons = tuple(seasons) if seasons is not None else db_seasons()
    out_dir = PATHS.processed("all")
    PATHS.ensure(out_dir)
    pgs, tgs, gms, avs, names = [], [], [], [], []
    for s in seasons:
        d = PATHS.processed(s)
        pg = pd.read_parquet(d / "player_games.parquet")
        pgs.append(pg)
        tgs.append(pd.read_parquet(d / "team_games.parquet"))
        gms.append(pd.read_parquet(d / "games.parquet"))
        ev = pd.read_parquet(PATHS.staged(s) / "events.parquet", columns=["person_id", "player_name"])
        names.append(ev[ev["person_id"] > 0].drop_duplicates())
        bp = PATHS.staged(s) / "box_players.parquet"
        if bp.exists():
            b = pd.read_parquet(bp)
            b["status"] = availability_status(b["comment"], b["box_seconds"])
            b["season"] = s
            avs.append(b[["game_id", "season", "team_id", "person_id", "player_name", "status", "comment",
                          "position", "box_seconds"]])
    pg = pd.concat(pgs, ignore_index=True)
    tg = pd.concat(tgs, ignore_index=True)
    games = pd.concat(gms, ignore_index=True)
    games["game_date"] = pd.to_datetime(games["game_date"])
    av = pd.concat(avs, ignore_index=True) if avs else pd.DataFrame()

    # ---- minutes: official where available ----
    pg["min_rec"] = pg["sec"] / 60.0
    has_box = pg["box_seconds"].notna() if "box_seconds" in pg else pd.Series(False, index=pg.index)
    pg["min"] = np.where(has_box, pg["box_seconds"].fillna(0) / 60.0, pg["min_rec"])
    pg["min_source"] = np.where(has_box, "official", "pbp")
    pg["played"] = pg["min"] > 0
    # coaches / staff: a bench technical gives them an event but they never play. Judged per row (a former
    # player who now coaches is caught too): zero seconds, not in the official box, nothing but technicals
    ever = set(pg.loc[pg["played"], "person_id"])
    box_cols = [c for c in ("fga2", "fga3", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf") if c in pg]
    no_box = pg["box_seconds"].isna() if "box_seconds" in pg else pd.Series(True, index=pg.index)
    bench_only = ~pg["played"] & no_box & (pg[box_cols].sum(axis=1) == 0) & (pg["tech"] > 0)
    staff = (~pg["person_id"].isin(ever) & ~pg["played"]) | bench_only
    n_staff = int(staff.sum())
    pg = pg[~staff].reset_index(drop=True)
    pg["game_date"] = pd.to_datetime(pg["game_date"])

    # ---- names ----
    nm = pd.concat(names).groupby("person_id")["player_name"].agg(lambda s: s.mode().iat[0])
    full = av.groupby("person_id")["player_name"].agg(lambda s: s.mode().iat[0]) if len(av) else pd.Series(dtype=str)
    pg["name"] = pg["person_id"].map(full).fillna(pg["person_id"].map(nm)).fillna(pg["person_id"].astype(str))
    tri = tg.merge(games[["game_id", "home_team_id", "home_team", "away_team_id", "away_team"]], on="game_id")
    team_abbr = pd.concat([tri[["home_team_id", "home_team"]].set_axis(["team_id", "team"], axis=1),
                           tri[["away_team_id", "away_team"]].set_axis(["team_id", "team"], axis=1)])
    team_abbr = team_abbr.groupby("team_id")["team"].agg(lambda s: s.mode().iat[0])
    pg["team"] = pg["team_id"].map(team_abbr)
    pg["opp"] = pg["opp_team_id"].map(team_abbr)

    # ---- team games with opponent columns and rest ----
    tg["game_date"] = pd.to_datetime(tg["game_date"])
    tg["team"] = tg["team_id"].map(team_abbr)
    num = [c for c in tg.columns if c not in ("game_id", "team_id", "opp_team_id", "is_home", "game_date",
                                             "season", "team")]
    opp = tg[["game_id", "team_id"] + num].rename(columns={"team_id": "opp_team_id", **{c: f"opp_{c}" for c in num}})
    tg = tg.merge(opp, on=["game_id", "opp_team_id"], how="left")
    tg = tg.merge(games[["game_id", "ot", "n_periods", "analysis_ok"]], on="game_id", how="left")
    tg = tg.sort_values(["team_id", "game_date"]).reset_index(drop=True)
    prev = tg.groupby(["team_id", "season"])["game_date"].shift(1)
    tg["rest_days"] = (tg["game_date"] - prev).dt.days.fillna(7).clip(upper=7)
    tg["b2b"] = tg["rest_days"] == 1
    # seasons from the live feed: rest from the full schedule (the live season's schedule is read even before its
    # first game is built), so a finished game whose play-by-play is missing still counts
    n_sched, ts = build_schedule_tables(tuple(dict.fromkeys((*seasons, LIVE_SEASON))), out_dir)
    if ts is not None and len(ts):
        rs = ts[["game_id", "team_id", "rest_days", "b2b"]].rename(columns={"rest_days": "_rest", "b2b": "_b2b"})
        tg = tg.merge(rs, on=["game_id", "team_id"], how="left")
        has = tg["_rest"].notna()
        tg.loc[has, "rest_days"] = tg.loc[has, "_rest"]
        tg.loc[has, "b2b"] = tg.loc[has, "_b2b"].astype(bool)
        tg = tg.drop(columns=["_rest", "_b2b"])
    tg["margin"] = tg["pts"] - tg["opp_pts"]
    tg["total"] = tg["pts"] + tg["opp_pts"]
    tg["win"] = tg["margin"] > 0

    # ---- availability: official statuses; 'played' rows only for seasons without official box scores ----
    if len(av):
        av = av.merge(games[["game_id", "game_date"]], on="game_id", how="left")
        av["source"] = "official"
    box_seasons = set(av["season"]) if len(av) else set()
    no_box = pg[~pg["season"].isin(box_seasons) & pg["played"]]
    av_pbp = no_box[["game_id", "season", "team_id", "person_id", "name"]].rename(columns={"name": "player_name"})
    av_pbp = av_pbp.assign(status="played", comment="", position=None, box_seconds=no_box["sec"],
                           source="pbp").merge(games[["game_id", "game_date"]], on="game_id")
    av = pd.concat([av, av_pbp], ignore_index=True)

    players = pg[pg["played"]].groupby("person_id").agg(
        name=("name", "first"), games=("game_id", "nunique"), minutes=("min", "sum"),
        first_date=("game_date", "min"), last_date=("game_date", "max"))
    last = pg[pg["played"]].sort_values("game_date").groupby("person_id").agg(team_id=("team_id", "last"),
                                                                              team=("team", "last"))
    players = players.join(last).reset_index()
    pg = pg.sort_values(["game_date", "game_id", "team_id", "person_id"]).reset_index(drop=True)
    for name, df in [("player_games", pg), ("team_games", tg), ("games", games), ("availability", av),
                     ("players", players)]:
        _write(df, out_dir / f"{name}.parquet")
    summary = {"player_games": int(len(pg)), "played": int(pg["played"].sum()), "games": int(len(games)),
               "players": int(len(players)), "seasons": list(seasons),
               "availability_rows": int(len(av)), "staff_rows_dropped": n_staff, "scheduled_games": n_sched,
               "status_counts": {k: int(v) for k, v in av["status"].value_counts().items()}}
    log(f"[database] {summary}")
    return summary
