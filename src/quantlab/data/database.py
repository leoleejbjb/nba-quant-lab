"""Unified multi-season database built from the per-season processed tables.

Tables written to ``data/processed/all/`` (also exposed as DuckDB views by
``warehouse.connect``):

* ``player_games``  every player-game 2020-21..2025-26 (box score + on-court exposures).
  ``min`` is the official NBA.com minutes when available (2020-21..2023-24),
  otherwise the play-by-play reconstruction (``min_rec``, within 1 s of the
  official value for 99.9% of 2023-24 player-games).
* ``availability``  per team-game roster status for seasons with official box scores:
  ``played``, ``dnp_cd`` (dressed, coach's decision) or ``out`` (DND / NWT / injury).
* ``team_games``    team box score with the opponent's columns joined (``opp_*``), rest days.
* ``games``         one row per game.
* ``players``       one row per player: name, career totals, last team.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ALL_SEASONS, PATHS
from .build import _write

def availability_status(comment: pd.Series, box_seconds: pd.Series) -> pd.Series:
    c = comment.fillna("")
    out = np.where(box_seconds.fillna(0) > 0, "played",
                   np.where(c.str.startswith("DNP - Coach") | c.str.startswith("DND - Coach"), "dnp_cd",
                            np.where(c == "", "dnp_cd", "out")))
    return pd.Series(out, index=comment.index)


def build_database(seasons=ALL_SEASONS, log=print) -> dict:
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
    tg["margin"] = tg["pts"] - tg["opp_pts"]
    tg["total"] = tg["pts"] + tg["opp_pts"]
    tg["win"] = tg["margin"] > 0

    # ---- availability: add 'played' rows for seasons without official box scores ----
    if len(av):
        av = av.merge(games[["game_id", "game_date"]], on="game_id", how="left")
    no_box = pg[pg["min_source"] == "pbp"]
    av_pbp = no_box[["game_id", "season", "team_id", "person_id", "name"]].rename(columns={"name": "player_name"})
    av_pbp = av_pbp.assign(status=np.where(no_box["played"], "played", "dnp_cd"), comment="", position=None,
                           box_seconds=no_box["sec"]).merge(games[["game_id", "game_date"]], on="game_id")
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
               "availability_rows": int(len(av)),
               "status_counts": {k: int(v) for k, v in av["status"].value_counts().items()}}
    log(f"[database] {summary}")
    return summary
