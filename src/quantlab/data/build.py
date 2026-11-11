"""Season build: raw files -> staged tables -> reconstructed games -> processed tables.

Outputs in ``data/processed/<season>/``:

* ``games``         one row per game (teams, date, score, OT, reconciliation flags)
* ``player_games``  one row per player-game: full box score + on-court exposures
* ``team_games``    one row per team-game: team box score incl. team rebounds/turnovers
* ``stints``        lineup stints (both five-man units, duration, points, possessions)
* ``possessions``   one row per possession (offense, start/end time, points, reason)
* ``starters``, ``contradictions``  reconstruction diagnostics
* ``box_compare``   derived vs official box score per player-game (seasons with box scores)
* ``quality.json``  reconstruction + reconciliation summary
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Callable

import duckdb
import numpy as np
import pandas as pd

from ..config import PATHS, Season
from .events import game_table, normalize_events, read_raw_pbp
from .game import reconstruct_game
from .sources import box_files, pbp_file, team_totals_file

BOX_STATS = ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb", "ast", "stl", "blk",
             "tov", "pf", "pts", "plus_minus"]


def _write(df: pd.DataFrame, path) -> None:
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (pd.Timestamp, dt.date)):
        return str(o)
    raise TypeError(type(o))


def write_json(obj, path) -> None:
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)


def parse_minutes(s) -> float:
    if s is None or (isinstance(s, float) and np.isnan(s)) or s == "":
        return 0.0
    s = str(s)
    if ":" in s:
        m, sec = s.split(":", 1)
        return int(m) * 60 + float(sec)
    return float(s) * 60.0


def stage_official_box(season: Season) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = PATHS.raw
    paths = [str(raw / f.filename) for f in box_files()]
    con = duckdb.connect()
    players = con.execute("""
        SELECT CAST(gameId AS BIGINT) game_id, CAST(game_date AS DATE) game_date, matchup,
               CAST(teamId AS BIGINT) team_id, teamTricode team, CAST(personId AS BIGINT) person_id,
               personName player_name, position, comment, CAST(minutes AS VARCHAR) minutes_str,
               CAST(fieldGoalsMade AS INT) fgm, CAST(fieldGoalsAttempted AS INT) fga,
               CAST(threePointersMade AS INT) fg3m, CAST(threePointersAttempted AS INT) fg3a,
               CAST(freeThrowsMade AS INT) ftm, CAST(freeThrowsAttempted AS INT) fta,
               CAST(reboundsOffensive AS INT) oreb, CAST(reboundsDefensive AS INT) dreb,
               CAST(reboundsTotal AS INT) reb, CAST(assists AS INT) ast, CAST(steals AS INT) stl,
               CAST(blocks AS INT) blk, CAST(turnovers AS INT) tov, CAST(foulsPersonal AS INT) pf,
               CAST(points AS INT) pts, CAST(CAST(plusMinusPoints AS DOUBLE) AS INT) plus_minus
        FROM read_csv(?, union_by_name = true, all_varchar = true) WHERE season_year = ?
    """, [paths, season.label]).df()
    teams = con.execute("""
        SELECT CAST(GAME_ID AS BIGINT) game_id, CAST(TEAM_ID AS BIGINT) team_id, TEAM_ABBREVIATION team,
               TEAM_NAME team_name, CAST(SUBSTR(GAME_DATE, 1, 10) AS DATE) game_date, MATCHUP matchup,
               WL wl, CAST(MIN AS DOUBLE) AS team_minutes, CAST(FGM AS INT) fgm, CAST(FGA AS INT) fga,
               CAST(FG3M AS INT) fg3m, CAST(FG3A AS INT) fg3a, CAST(FTM AS INT) ftm, CAST(FTA AS INT) fta,
               CAST(OREB AS INT) oreb, CAST(DREB AS INT) dreb, CAST(REB AS INT) reb, CAST(AST AS INT) ast,
               CAST(CAST(TOV AS DOUBLE) AS INT) tov, CAST(STL AS INT) stl, CAST(BLK AS INT) blk,
               CAST(PF AS INT) pf, CAST(PTS AS INT) pts
        FROM read_csv(?, all_varchar = true) WHERE SEASON_YEAR = ?
    """, [str(raw / team_totals_file().filename), season.label]).df()
    con.close()
    players["box_seconds"] = players["minutes_str"].map(parse_minutes)
    players["is_home"] = players["matchup"].str.contains(" vs. ", regex=False)
    teams["is_home"] = teams["matchup"].str.contains(" vs. ", regex=False)
    return players, teams


def _reconstruct_chunk(items):
    return [reconstruct_game(ev, h, a, box) for ev, h, a, box in items]


def build_season(label: str, *, workers: int = 4, log: Callable[[str], None] = print) -> dict:
    t0 = time.perf_counter()
    season = Season(label)
    sd, pd_dir = PATHS.staged(label), PATHS.processed(label)
    PATHS.ensure(sd, pd_dir)
    rf = pbp_file(season.start_year)
    raw = read_raw_pbp(PATHS.raw / rf.filename, rf.member, season.game_prefix)
    events = normalize_events(raw)
    _write(events, sd / "events.parquet")
    games = game_table(events)
    box_p = box_t = None
    if season.has_box:
        box_p, box_t = stage_official_box(season)
        _write(box_p, sd / "box_players.parquet")
        _write(box_t, sd / "box_teams.parquet")
        bh = box_t[box_t["is_home"]][["game_id", "team_id", "game_date"]].rename(
            columns={"team_id": "box_home_team_id", "game_date": "box_date"})
        games = games.merge(bh, on="game_id", how="left")
        swap = games["box_home_team_id"].notna() & (games["box_home_team_id"] != games["home_team_id"])
        games.loc[swap, ["home_team_id", "away_team_id"]] = games.loc[swap, ["away_team_id", "home_team_id"]].to_numpy()
        games.loc[swap, ["home_team", "away_team"]] = games.loc[swap, ["away_team", "home_team"]].to_numpy()
        games.loc[swap, ["final_home", "final_away"]] = games.loc[swap, ["final_away", "final_home"]].to_numpy()
        games["home_away_swapped"] = swap
        games["game_date"] = pd.to_datetime(games["box_date"]).fillna(games["game_date"])
        games = games.drop(columns=["box_home_team_id", "box_date"])
    log(f"[build {label}] {len(games):,} games, {len(events):,} events, official box scores: "
        f"{'yes' if season.has_box else 'no'}")

    box_by_game = dict(tuple(box_p.groupby("game_id"))) if box_p is not None else {}
    gmeta = games.set_index("game_id")
    items = []
    for gid, g in events.groupby("game_id", sort=True):
        r = gmeta.loc[gid]
        items.append((g, int(r["home_team_id"]), int(r["away_team_id"]), box_by_game.get(gid)))
    nchunk = max(workers, 1) * 6
    chunks = [items[i::nchunk] for i in range(nchunk)]
    results = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for part in ex.map(_reconstruct_chunk, chunks):
                results.extend(part)
    else:
        for c in chunks:
            results.extend(_reconstruct_chunk(c))
    results.sort(key=lambda r: r["meta"]["game_id"])
    cat = lambda k: pd.concat([r[k] for r in results if len(r[k])], ignore_index=True)
    players, teams, stints, poss = cat("players"), cat("teams"), cat("stints"), cat("possessions")
    starters, contra = cat("starters"), (cat("contradictions") if any(len(r["contradictions"]) for r in results)
                                         else results[0]["contradictions"])
    meta = pd.DataFrame([r["meta"] for r in results]).fillna(0)
    games = games.merge(meta[["game_id", "n_periods", "game_len_s", "ot"]].rename(
        columns={"n_periods": "n_periods_rec"}), on="game_id", how="left")

    # ----- derived quantities -----
    players = add_box_columns(players)
    teams = add_team_box_columns(teams, players)
    games = add_game_scores(games, teams)

    # ----- validity of each game's reconstruction -----
    sv = stints.assign(valid_s=stints["duration_s"].where(stints["valid"], 0.0))
    cov = sv.groupby("game_id").agg(total_s=("duration_s", "sum"), valid_s=("valid_s", "sum"))
    games = games.merge(cov, on="game_id", how="left")
    games["valid_share"] = games["valid_s"] / games["total_s"]
    games["points_conserved"] = (games["home_pts"] == games["final_home"]) & (games["away_pts"] == games["final_away"])
    secsum = players.groupby(["game_id", "team_id"])["sec"].sum().groupby("game_id").agg(["min", "max"])
    games = games.merge(secsum.rename(columns={"min": "team_sec_min", "max": "team_sec_max"}), on="game_id", how="left")
    games["minutes_reconcile"] = (np.abs(games["team_sec_min"] - 5 * games["game_len_s"]) < 1) & \
                                 (np.abs(games["team_sec_max"] - 5 * games["game_len_s"]) < 1)
    compare = None
    if box_p is not None:
        compare, gflags = reconcile(players, box_p)
        games = games.merge(gflags, on="game_id", how="left")
        games["analysis_ok"] = games["points_conserved"] & games["box_minutes_ok"].fillna(False) & \
            (games["valid_share"] > 0.99)
        players = attach_official(players, box_p)
    else:
        games["analysis_ok"] = games["points_conserved"] & games["minutes_reconcile"] & (games["valid_share"] > 0.99)
    games["season"] = label
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    players = players.merge(games[["game_id", "game_date", "analysis_ok"]], on="game_id", how="left")
    players["season"] = label
    teams = teams.merge(games[["game_id", "game_date"]], on="game_id", how="left")
    teams["season"] = label

    for name, df in [("games", games), ("player_games", players), ("team_games", teams), ("stints", stints),
                     ("possessions", poss), ("starters", starters), ("contradictions", contra)]:
        _write(df, pd_dir / f"{name}.parquet")
    if compare is not None:
        _write(compare, pd_dir / "box_compare.parquet")
    q = quality(label, games, stints, players, starters, contra, compare, time.perf_counter() - t0)
    write_json(q, pd_dir / "quality.json")
    log(f"[build {label}] done in {q['seconds']:.1f}s: {len(players):,} player-games, {len(stints):,} stints, "
        f"{len(poss):,} possessions; valid lineup time {q['valid_time_share']:.3%}; "
        f"games usable {q['games_ok']}/{q['games']}")
    return q


def add_box_columns(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    p["min"] = p["sec"] / 60.0
    p["fga"] = p["fga2"] + p["fga3"]
    p["fgm"] = p["fgm2"] + p["fgm3"]
    p["fg3a"], p["fg3m"] = p["fga3"], p["fgm3"]
    p["reb"] = p["oreb"] + p["dreb"]
    p["pts"] = 2 * p["fgm2"] + 3 * p["fgm3"] + p["ftm"]
    p["pra"] = p["pts"] + p["reb"] + p["ast"]
    tsa = p["fga"] + 0.44 * p["fta"]
    p["ts_pct"] = np.where(tsa > 0, p["pts"] / (2 * tsa), np.nan)
    p["efg_pct"] = np.where(p["fga"] > 0, (p["fgm"] + 0.5 * p["fg3m"]) / p["fga"], np.nan)
    # Usage rate on exact on-court team totals (not the box-score minutes approximation):
    used = p["fga"] + 0.44 * p["fta"] + p["tov"]
    team_used = p["on_fga"] + 0.44 * p["on_fta"] + p["on_tov"]
    p["usg_pct"] = np.where(team_used > 0, used / team_used, np.nan)
    p["played"] = p["sec"] > 0
    cats = (p[["pts", "reb", "ast", "stl", "blk"]] >= 10).sum(axis=1)
    p["double_double"] = cats >= 2
    p["triple_double"] = cats >= 3
    return p


def add_team_box_columns(t: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    sums = p.groupby(["game_id", "team_id"])[["fga2", "fgm2", "fga3", "fgm3", "fta", "ftm", "oreb", "dreb", "ast",
                                               "stl", "blk", "tov", "pf", "pts", "sec"]].sum()
    sums = sums.add_prefix("pl_").reset_index()
    t = t.merge(sums, on=["game_id", "team_id"], how="left")
    t["team_reb_total"] = t["team_oreb"] + t["team_dreb"]
    return t


def add_game_scores(games: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    h = teams[teams["is_home"]][["game_id", "pts", "poss"]].rename(columns={"pts": "home_pts", "poss": "home_poss"})
    a = teams[~teams["is_home"]][["game_id", "pts", "poss"]].rename(columns={"pts": "away_pts", "poss": "away_poss"})
    return games.merge(h, on="game_id", how="left").merge(a, on="game_id", how="left")


def reconcile(players: pd.DataFrame, box: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derived vs official box score for every player-game (outer join)."""
    der = players[["game_id", "person_id", "team_id", "sec"] + [c for c in BOX_STATS]].copy()
    off = box[["game_id", "person_id", "team_id", "box_seconds", "comment"] + BOX_STATS].copy()
    m = der.merge(off, on=["game_id", "person_id"], how="outer", suffixes=("", "_box"), indicator=True)
    played = (m["sec"].fillna(0) > 0) | (m["box_seconds"].fillna(0) > 0)
    m = m[played | (m["_merge"] != "right_only")].copy()
    m["sec_diff"] = m["sec"].fillna(0) - m["box_seconds"].fillna(0)
    for c in BOX_STATS:
        m[f"{c}_diff"] = m[c].fillna(0) - m[f"{c}_box"].fillna(0)
    g = m.groupby("game_id").agg(
        box_max_abs_sec_diff=("sec_diff", lambda s: float(np.abs(s).max())),
        box_players_minutes_off=("sec_diff", lambda s: int((np.abs(s) > 3).sum())),
        box_players_pm_off=("plus_minus_diff", lambda s: int((s != 0).sum())),
        box_stat_mismatch=("pts_diff", lambda s: int((s != 0).sum())),
    ).reset_index()
    g["box_minutes_ok"] = g["box_players_minutes_off"] == 0
    return m.drop(columns="_merge"), g


def attach_official(players: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
    b = box[["game_id", "person_id", "position", "comment", "box_seconds"]]
    return players.merge(b, on=["game_id", "person_id"], how="left")


def quality(label, games, stints, players, starters, contra, compare, seconds) -> dict:
    tot = float(stints["duration_s"].sum())
    val = float(stints.loc[stints["valid"], "duration_s"].sum())
    q = {"season": label, "seconds": round(seconds, 1), "games": int(len(games)),
         "games_ok": int(games["analysis_ok"].sum()),
         "excluded_games": games.loc[~games["analysis_ok"], "game_id"].astype(int).tolist(),
         "points_conserved_games": int(games["points_conserved"].sum()),
         "valid_time_share": val / tot if tot else float("nan"),
         "starter_status": {k: int(v) for k, v in starters["status"].value_counts().items()},
         "contradictions": {k: int(v) for k, v in contra["reason"].value_counts().items()} if len(contra) else {},
         "player_games": int(len(players)), "player_games_played": int(players["played"].sum()),
         "ot_games": int(games["ot"].sum())}
    if compare is not None:
        c = compare[(compare["sec"].fillna(0) > 0) | (compare["box_seconds"].fillna(0) > 0)]
        q["official_box"] = {"player_games_compared": int(len(c)),
                             "minutes_within_1s": float((c["sec_diff"].abs() <= 1).mean()),
                             "minutes_within_3s": float((c["sec_diff"].abs() <= 3).mean()),
                             **{f"{s}_exact": float((c[f'{s}_diff'] == 0).mean()) for s in BOX_STATS},
                             "all_stats_exact": float((c[[f'{s}_diff' for s in BOX_STATS if s != 'plus_minus']] == 0)
                                                      .all(axis=1).mean())}
    return q
