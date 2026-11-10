"""Assemble simulator inputs (``GameSpec``) for a game from the database and pregame model outputs.

Roster for a team-game (pregame information only):

* *rotation participants*: rotation players (baseline minutes >= 8, expected minutes >= 12, or
  starters) who are available for the game. The sources have no inactive list for the test seasons,
  so in every season this set is taken from who appeared - the stand-in for the injury report /
  inactive list that is public before tip-off. The stand-in also reveals rotation players who were
  dressed but not used (coach's decision); ``models.minutes.exact_availability_experiment`` measures
  what that is worth on 2023-24, where the official inactive list exists. Their minutes come from the
  walk-forward minutes model.
* *fringe candidates*: one pregame rule for everyone - every player who appeared for the team in the
  previous 25 days and has not appeared for another team since. Participation probability = his share
  of the team's games in that window; minutes *if he plays* = his minutes per appearance in it. Whether
  he appears today never changes his inclusion or his numbers; ``GameSimulator`` then scales the team's
  fringe participation so expected minutes fill 240 x the expected game length.
* a player who appeared but is in neither group (a new signing, a return from a long absence; he would
  be on the pregame active list) is added at the participation floor (5%) so a conditional forecast
  exists; such players are flagged ``in_pool=False`` and left out of participation calibration.

Player rates are the pregame empirical-Bayes estimates; opponent factors and the expected pace come
from the walk-forward team models. Nothing here reads a box-score number from the game itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PATHS
from .engine import DEF_FACTORS, MAX_PLAYERS, RATE_KEYS, GameSpec, PlayerSpec, TeamSpec


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

        self.rates_by_person = {p: d for p, d in self.rates_sorted.groupby("person_id")}
        played = self.pg[self.pg["played"]].sort_values("game_date")
        self.played_by_person = {p: (d["game_date"].to_numpy(), d["team_id"].to_numpy())
                                 for p, d in played.groupby("person_id")}

    def latest_rates(self, person_id: int, before) -> pd.Series | None:
        sub = self.rates_by_person.get(person_id)
        if sub is None:
            return None
        sub = sub[sub["game_date"] < before]
        return sub.iloc[-1] if len(sub) else None

    def last_team_before(self, person_id: int, before) -> int | None:
        """Team of the player's last appearance before ``before``."""
        v = self.played_by_person.get(person_id)
        if v is None:
            return None
        k = int(np.searchsorted(v[0], np.datetime64(before), side="left")) - 1
        return int(v[1][k]) if k >= 0 else None


def _rates_dict(row: pd.Series) -> dict:
    return {k: float(row[f"r_{k}"]) for k in RATE_KEYS}


FRINGE_WINDOW_DAYS = 25
FRINGE_P_FLOOR, FRINGE_P_CAP = 0.05, 0.9


def fringe_pool(store: InputStore, team_id: int, date, exclude: set) -> list:
    """Pregame fringe candidates: (person_id, participation share, minutes per appearance), largest expected
    minutes first. Uses only games before ``date``; today's appearance plays no part."""
    pg = store.pg
    recent = pg[(pg["team_id"] == team_id) & (pg["game_date"] < date) &
                (pg["game_date"] >= date - pd.Timedelta(days=FRINGE_WINDOW_DAYS))]
    n_team_games = max(recent["game_id"].nunique(), 1)
    out = []
    for pid, grp in recent[recent["played"]].groupby("person_id"):
        pid = int(pid)
        if pid in exclude:
            continue
        last = store.last_team_before(pid, date)
        if last is not None and last != team_id:
            continue                          # appeared for another team since (traded / waived)
        share = float(np.clip(len(grp) / n_team_games, FRINGE_P_FLOOR, FRINGE_P_CAP))
        cond = float(grp["min"].sum() / len(grp))
        out.append((pid, share, cond))
    return sorted(out, key=lambda x: -x[1] * x[2])


def team_spec(store: InputStore, game_id: int, team_id: int, minutes_col: str = "pred_gbm",
              min_sd: float | None = None, exclude: set | None = None, max_players: int = MAX_PLAYERS) -> TeamSpec:
    """``exclude``: players appearing for the opponent in this game (e.g. just traded) - never fringe here."""
    exclude = exclude or set()
    g = store.games.loc[game_id]
    date = g["game_date"]
    f = store.by_game[game_id]
    f = f[f["team_id"] == team_id]
    rates = store.rates_by_game.get(game_id)
    mins = store.minutes_by_game.get(game_id)
    ctx = store.ctx.loc[(game_id, team_id)] if (game_id, team_id) in store.ctx.index else None
    absent_rot = {int(x) for x in str(ctx["absent"]).split("-") if x} if ctx is not None else set()
    players, seen, appeared_fringe = [], set(), {}
    # 1. rotation participants (available rotation players; stand-in: who appeared)
    for r in f.itertuples(index=False):
        if not r.played:
            continue
        rr = rates.loc[r.person_id] if rates is not None and r.person_id in rates.index else None
        if rr is None:
            continue
        if mins is not None and r.person_id in mins.index:
            m = float(mins.loc[r.person_id, minutes_col])
        else:
            m = float(np.nan_to_num(r.exp_min, nan=12.0))
        if bool(r.rotation) or bool(r.starter):
            players.append(PlayerSpec(int(r.person_id), store.names.get(r.person_id, str(r.person_id)),
                                      _rates_dict(rr), min_mean=m, min_sd=minutes_sd(m, bool(r.starter), min_sd),
                                      p_play=1.0, starter=bool(r.starter), rotation=True))
            seen.add(int(r.person_id))
        else:
            appeared_fringe[int(r.person_id)] = (rr, m)
    # 2. fringe candidates: one pregame rule (recent players still with the team), same for all
    pool = fringe_pool(store, team_id, date, exclude | seen | absent_rot)
    for pid, share, cond in pool:
        if len(players) >= max_players:
            break
        rr = store.latest_rates(pid, date)
        if rr is None:
            continue
        players.append(PlayerSpec(pid, store.names.get(pid, str(pid)), _rates_dict(rr),
                                  min_mean=max(share * cond, 0.3), min_sd=3.0, p_play=share, starter=False,
                                  rotation=False))
        seen.add(pid)
    in_pool = {pid for pid, _, _ in pool}
    # 3. appeared, but in no pregame group: at the participation floor (conditional minutes from the model)
    for pid, (rr, m) in appeared_fringe.items():
        if pid in seen or pid in in_pool or len(players) >= max_players:
            continue
        cond = m if np.isfinite(m) and m > 0 else 8.0
        players.append(PlayerSpec(pid, store.names.get(pid, str(pid)), _rates_dict(rr),
                                  min_mean=FRINGE_P_FLOOR * cond, min_sd=3.0, p_play=FRINGE_P_FLOOR,
                                  starter=False, rotation=False, in_pool=False))
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
    f = store.by_game[game_id]
    playing = {int(t): set(f.loc[(f["team_id"] == t) & f["played"], "person_id"].astype(int))
               for t in (g["home_team_id"], g["away_team_id"])}
    home = team_spec(store, game_id, int(g["home_team_id"]), minutes_col, exclude=playing[int(g["away_team_id"])])
    away = team_spec(store, game_id, int(g["away_team_id"]), minutes_col, exclude=playing[int(g["home_team_id"])])
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
