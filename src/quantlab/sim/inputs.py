"""Assemble simulator inputs (``GameSpec``) for a game from the database and pregame model outputs.

Roster for a team-game (pregame information only):

* *rotation participants*: rotation players (baseline minutes >= 8, expected minutes >= 12, or
  starters) who are active for the game. In the historical backtest this set is taken from who
  actually appeared - the stand-in for the injury report / inactive list that is public before
  tip-off. Their minutes come from the walk-forward minutes model.
* *fringe players*: the team's other recent players (previous 10 games, still with the team), plus
  any fringe player who appeared. They have no planned minutes and a participation probability
  equal to their baseline-minutes share; they mostly play in garbage time, exactly as in reality.

Player rates are the pregame empirical-Bayes estimates; opponent factors and the expected pace come
from the walk-forward team models. Nothing here reads a box-score number from the game itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PATHS
from .engine import DEF_FACTORS, RATE_KEYS, GameSpec, PlayerSpec, TeamSpec


class InputStore:
    """Lazily loaded tables used to build game specs (one instance per process)."""

    def __init__(self):
        allp = PATHS.processed("all")
        self.games = pd.read_parquet(allp / "games.parquet").set_index("game_id")
        self.pg = pd.read_parquet(allp / "player_games.parquet",
                                  columns=["game_id", "person_id", "team_id", "name", "team", "played", "min",
                                           "starter", "game_date", "season"])
        feats = pd.read_parquet(PATHS.features / "player_asof.parquet",
                                columns=["game_id", "person_id", "team_id", "game_date", "exp_min", "base_min",
                                         "rotation", "played", "starter", "season_min_avg", "min_l10"])
        self.feats = feats
        self.rates = pd.read_parquet(PATHS.models / "player_rates.parquet")
        self.rate_cols = [f"r_{k}" for k in RATE_KEYS]
        mp = PATHS.models / "minutes_predictions.parquet"
        self.minutes = pd.read_parquet(mp) if mp.exists() else None
        self.tf = pd.read_parquet(PATHS.models / "team_factors.parquet").set_index(["game_id", "team_id"])
        self.tp = pd.read_parquet(PATHS.models / "team_predictions.parquet").set_index("game_id")
        ctx = pd.read_parquet(PATHS.features / "team_context.parquet")
        self.ctx = ctx.set_index(["game_id", "team_id"])
        tg = pd.read_parquet(allp / "team_games.parquet", columns=["game_id", "team_id", "game_date", "tov", "poss"])
        daily = tg.groupby("game_date")[["tov", "poss"]].sum().sort_index()
        roll = daily.rolling(60, min_periods=1).sum().shift(1).bfill()
        self.lg_tov = (roll["tov"] / roll["poss"]).to_dict()
        self.by_game = {g: d for g, d in self.feats.groupby("game_id")}
        self.rates_by_game = {g: d.set_index("person_id") for g, d in self.rates.groupby("game_id")}
        self.minutes_by_game = ({g: d.set_index("person_id") for g, d in self.minutes.groupby("game_id")}
                                if self.minutes is not None else {})
        # latest pregame rates per player (for fringe players without a row in this game)
        self.rates_sorted = self.rates.sort_values("game_date")
        self.names = self.pg.drop_duplicates("person_id").set_index("person_id")["name"].to_dict()

    def latest_rates(self, person_id: int, before) -> pd.Series | None:
        r = self.rates_sorted
        sub = r[(r["person_id"] == person_id) & (r["game_date"] < before)]
        return sub.iloc[-1] if len(sub) else None


def _rates_dict(row: pd.Series) -> dict:
    return {k: float(row[f"r_{k}"]) for k in RATE_KEYS}


def team_spec(store: InputStore, game_id: int, team_id: int, minutes_col: str = "pred_gbm",
              min_sd: float | None = None) -> TeamSpec:
    g = store.games.loc[game_id]
    date = g["game_date"]
    f = store.by_game[game_id]
    f = f[f["team_id"] == team_id]
    rates = store.rates_by_game.get(game_id)
    mins = store.minutes_by_game.get(game_id)
    players = []
    seen = set()
    for r in f.itertuples(index=False):
        if not r.played:
            continue
        rot = bool(r.rotation) or bool(r.starter)
        rr = rates.loc[r.person_id] if rates is not None and r.person_id in rates.index else None
        if rr is None:
            continue
        if mins is not None and r.person_id in mins.index:
            m = float(mins.loc[r.person_id, minutes_col])
        else:
            m = float(np.nan_to_num(r.exp_min, nan=12.0))
        if rot:
            players.append(PlayerSpec(int(r.person_id), store.names.get(r.person_id, str(r.person_id)),
                                      _rates_dict(rr), min_mean=m, min_sd=minutes_sd(m, bool(r.starter), min_sd),
                                      p_play=1.0, starter=bool(r.starter), rotation=True))
        else:
            base = float(np.nan_to_num(r.base_min, nan=2.0))
            share = np.clip(base / max(np.nan_to_num(r.exp_min, nan=8.0), 1.0), 0.05, 0.95)
            players.append(PlayerSpec(int(r.person_id), store.names.get(r.person_id, str(r.person_id)),
                                      _rates_dict(rr), min_mean=max(base, 0.3), min_sd=3.0, p_play=float(share),
                                      starter=False, rotation=False))
        seen.add(int(r.person_id))
    # fringe players from the baseline who did not appear (pregame they were available-but-uncertain)
    ctx = store.ctx.loc[(game_id, team_id)] if (game_id, team_id) in store.ctx.index else None
    if ctx is not None:
        # baseline fringe: recent players of this team not in today's rotation list and not absent-rotation
        absent_rot = {int(x) for x in str(ctx["absent"]).split("-") if x}
        recent = store.pg[(store.pg["team_id"] == team_id) & (store.pg["game_date"] < date) &
                          (store.pg["game_date"] >= date - pd.Timedelta(days=25)) & store.pg["played"]]
        for pid, grp in recent.groupby("person_id"):
            pid = int(pid)
            if pid in seen or pid in absent_rot or len(players) >= 15:
                continue
            later = store.pg[(store.pg["person_id"] == pid) & (store.pg["game_date"] < date) & store.pg["played"]]
            if len(later) and int(later.iloc[-1]["team_id"]) != team_id:
                continue                          # traded since
            rr = store.latest_rates(pid, date)
            if rr is None:
                continue
            n_team_games = max(recent["game_id"].nunique(), 1)
            share = float(np.clip(len(grp) / n_team_games, 0.05, 0.9))
            per_team_game = float(grp["min"].sum() / n_team_games)
            players.append(PlayerSpec(pid, store.names.get(pid, str(pid)), _rates_dict(rr),
                                      min_mean=max(per_team_game, 0.3), min_sd=3.0, p_play=share, starter=False,
                                      rotation=False))
            seen.add(pid)
    tf = store.tf.loc[(game_id, team_id)] if (game_id, team_id) in store.tf.index else None
    dfac = {k: float(tf[f"def_{k}"]) for k in DEF_FACTORS} if tf is not None else {}
    name = str(g["home_team"] if team_id == g["home_team_id"] else g["away_team"])
    return TeamSpec(int(team_id), name, players, def_factors=dfac)


def minutes_sd(m: float, starter: bool, override: float | None = None) -> float:
    """Planned-minutes noise (minutes). Calibrated on 2023-24 (see eval/calibrate.py)."""
    if override is not None:
        return override
    return float(np.clip(2.2 + 0.08 * m, 2.5, 5.0)) if starter else float(np.clip(2.4 + 0.10 * m, 2.5, 5.5))


def game_spec(store: InputStore, game_id: int, minutes_col: str = "pred_gbm") -> GameSpec:
    g = store.games.loc[game_id]
    home = team_spec(store, game_id, int(g["home_team_id"]), minutes_col)
    away = team_spec(store, game_id, int(g["away_team_id"]), minutes_col)
    tp = store.tp.loc[game_id] if game_id in store.tp.index else None
    pace = float(tp["pace"] * 1.0) if tp is not None else 99.0
    if tp is not None:
        home.target_ppp, away.target_ppp = float(tp["ppp_home"] / 100), float(tp["ppp_away"] / 100)
    rates = store.rates_by_game.get(game_id)
    lg = {}
    if rates is not None and len(rates):
        lg = {"blk": float(rates["lg_blk"].mean()), "ast_rate": float(rates["lg_ast_rate"].mean())}
    lg["tov_per_poss"] = float(store.lg_tov.get(g["game_date"], 0.142))
    return GameSpec(home, away, pace=pace, league=lg)
