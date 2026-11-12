"""Assemble simulator inputs (``GameSpec``) for a game from the database and pregame model outputs.

Roster for a team-game (pregame information, with who appeared standing in for the inactive list):

* *rotation participants*: rotation players (baseline minutes >= 8, expected minutes >= 12, or
  starters) who are available for the game. The sources have no inactive list for the test seasons,
  so in every season this set is taken from who appeared - the stand-in for the injury report /
  inactive list that is public before tip-off. The stand-in also reveals rotation players who were
  dressed but not used (coach's decision); ``models.minutes.exact_availability_experiment`` measures
  what that is worth on 2023-24, where the official inactive list exists. Their minutes come from the
  walk-forward minutes model.
* *fringe candidates*: one pregame rule for everyone (``features.asof.fringe_pool_table``) - every player
  who appeared for the team in the previous 25 days and has not appeared for another team since.
  Participation probability = his share of the team's games in that window; minutes *if he plays* = the
  walk-forward minutes model's projection (every candidate has a pregame feature row, appeared or not).
  Whether he appears today never changes his inclusion or his numbers; ``GameSimulator`` then scales the
  team's fringe participation so expected minutes fill 240 x the expected game length.
* a player who appeared but is in neither group (a new signing, a return from a long absence; he would
  be on the pregame active list) is added at the participation floor (5%) so a conditional forecast
  exists; such players are flagged ``in_pool=False`` and left out of participation calibration.

Player rates are the pregame empirical-Bayes estimates; opponent factors and the expected pace come
from the walk-forward team models. Nothing here reads a box-score number from the game itself.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import PATHS
from ..features.asof import FRINGE_P_FLOOR, PRESUMED_AVAILABLE_STREAK, streak_bucket
from .engine import DEF_FACTORS, MAX_PLAYERS, RATE_KEYS, GameSpec, PlayerSpec, TeamSpec


class PregameInputsMissing(RuntimeError):
    """An unplayed game without pregame rows (outside the projection horizon, or inputs not rebuilt)."""


class InputStore:
    """Lazily loaded tables used to build game specs (one instance per process)."""

    def __init__(self):
        allp = PATHS.processed("all")
        self.games = pd.read_parquet(allp / "games.parquet").set_index("game_id")
        sp = allp / "schedule.parquet"         # the live season's schedule (unplayed games included)
        self.schedule = pd.read_parquet(sp).set_index("game_id") if sp.exists() else None
        pp = PATHS.features / "availability_priors.json"
        self.priors = json.loads(pp.read_text(encoding="utf-8"))["p_appear"] if pp.exists() else {}
        self.pg = pd.read_parquet(allp / "player_games.parquet",
                                  columns=["game_id", "person_id", "team_id", "name", "team", "played", "min",
                                           "starter", "game_date", "season"])
        want = ["game_id", "person_id", "team_id", "game_date", "exp_min", "base_min", "rotation", "played",
                "starter", "season_min_avg", "min_l10", "rotation_pre", "scheduled", "missed_streak", "starts_m20",
                "gp_m20"]
        import pyarrow.parquet as pq
        have = set(pq.ParquetFile(PATHS.features / "player_asof.parquet").schema_arrow.names)
        feats = pd.read_parquet(PATHS.features / "player_asof.parquet", columns=[c for c in want if c in have])
        self.feats = feats
        self.rates = pd.read_parquet(PATHS.models / "player_rates.parquet")
        self.rate_cols = [f"r_{k}" for k in RATE_KEYS]
        mp = PATHS.models / "minutes_predictions.parquet"
        self.minutes = pd.read_parquet(mp) if mp.exists() else None
        self.tf = pd.read_parquet(PATHS.models / "team_factors.parquet").set_index(["game_id", "team_id"])
        self.tp = pd.read_parquet(PATHS.models / "team_predictions.parquet").set_index("game_id")
        ctx = pd.read_parquet(PATHS.features / "team_context.parquet")
        self.ctx = ctx.set_index(["game_id", "team_id"])
        pool = pd.read_parquet(PATHS.features / "fringe_pool.parquet")
        self.pool = {k: d for k, d in pool.groupby(["game_id", "team_id"])}
        tg = pd.read_parquet(allp / "team_games.parquet", columns=["game_id", "team_id", "game_date", "tov", "poss"])
        daily = tg.groupby("game_date")[["tov", "poss"]].sum().sort_index()
        roll = daily.rolling(60, min_periods=1).sum().shift(1).bfill()
        self.lg_tov = (roll["tov"] / roll["poss"]).to_dict()
        self.lg_tov_dates = np.array(sorted(daily.index), dtype="datetime64[ns]")
        self.lg_tov_after = (daily.rolling(60, min_periods=1).sum()).pipe(lambda r: (r["tov"] / r["poss"])).to_numpy()
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

    # ------------------------------------------------------------------ games: built or scheduled
    def is_final(self, game_id: int) -> bool:
        """Built from its play-by-play (a finished game); otherwise an unplayed game of the live season."""
        return int(game_id) in self.games.index

    def game_row(self, game_id: int) -> pd.Series:
        """The game's row: the built game, else its schedule row (date, tip time, teams, status)."""
        gid = int(game_id)
        if gid in self.games.index:
            return self.games.loc[gid]
        if self.schedule is not None and gid in self.schedule.index:
            return self.schedule.loc[gid]
        raise KeyError(f"game {gid} is neither built nor in the live schedule")

    def pregame_problem(self, game_id: int) -> str | None:
        """Why an unplayed game cannot be projected pregame yet (None: it can)."""
        gid = int(game_id)
        g = self.game_row(gid)
        home, away = int(g["home_team_id"]), int(g["away_team_id"])
        if home <= 0 or away <= 0:
            return "the teams are not known yet"
        if gid not in self.tp.index:                    # the team model covers every game in the horizon
            return "no pregame inputs yet (beyond the projection horizon: run `quantlab live-update` closer to the date)"
        f = self.by_game.get(gid)
        have = set(f["team_id"].astype(int)) if f is not None else set()
        missing = [str(g["home_team"] if t == home else g["away_team"]) for t in (home, away) if t not in have]
        if missing:
            return (f"{' and '.join(missing)} {'has' if len(missing) == 1 else 'have'} not played this season yet "
                    "(no recent roster to project from: projections start after a team's first game)")
        return None

    def league_tov(self, date) -> float:
        """League turnovers per possession over the 60 game-days strictly before ``date``."""
        if pd.Timestamp(date) in self.lg_tov:
            return float(self.lg_tov[pd.Timestamp(date)])
        k = int(np.searchsorted(self.lg_tov_dates, np.datetime64(pd.Timestamp(date)), side="left")) - 1
        return float(self.lg_tov_after[k]) if k >= 0 else 0.142

    def last_starters(self, team_id: int, before) -> list:
        """The starting five of the team's last finished game before ``before``."""
        pg = self.pg
        d = pg[(pg["team_id"] == team_id) & (pg["game_date"] < pd.Timestamp(before)) & pg["played"]]
        if d.empty:
            return []
        last = d[d["game_date"] == d["game_date"].max()]
        return [int(x) for x in last.loc[last["starter"], "person_id"]]

    def latest_minutes(self, person_id: int, before) -> tuple[float, float]:
        """Expected minutes when he plays and start rate as of the player's latest game before ``before`` (for a
        player outside his team's pool whom the injury report brings back)."""
        f = self.feats
        d = f[(f["person_id"] == int(person_id)) & (f["game_date"] < pd.Timestamp(before)) & f["played"].eq(True)]
        if d.empty:
            return 20.0, 0.0
        r = d.sort_values("game_date").iloc[-1]
        m = next((float(r[c]) for c in ("exp_min", "min_l10", "season_min_avg")
                  if c in r.index and pd.notna(r[c]) and float(r[c]) > 0), 20.0)
        gp = float(r["gp_m20"]) if "gp_m20" in r.index and pd.notna(r["gp_m20"]) else 0.0
        return m, (float(r["starts_m20"]) / gp if gp > 0 else 0.0)

    def recent_absentees(self, team_ids, before, exclude=(), days: int = 365) -> list:
        """(person id, team id, last date) of players whose last game before ``before`` was for one of ``team_ids``
        within ``days`` and who are not in ``exclude`` - e.g. a long-term absentee outside the 25-day pool."""
        b, teams, skip, out = np.datetime64(pd.Timestamp(before)), {int(t) for t in team_ids}, set(exclude), []
        for pid, (dates, tids) in self.played_by_person.items():
            k = int(np.searchsorted(dates, b, side="left")) - 1
            if k >= 0 and int(tids[k]) in teams and pid not in skip and (b - dates[k]) <= np.timedelta64(days, "D"):
                out.append((int(pid), int(tids[k]), pd.Timestamp(dates[k])))
        return out

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


def fringe_pool(store, team_id: int, game_id: int, exclude: set) -> list:
    """Pregame fringe candidates of a team-game: (person_id, participation share, minutes per appearance in the
    window), largest expected minutes first. Precomputed from games before the date only."""
    d = store.pool.get((game_id, team_id))
    if d is None:
        return []
    out = [(int(p), float(sh), float(c)) for p, sh, c in zip(d["person_id"], d["share"], d["cond_window"])
           if int(p) not in exclude]
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
    # 2. fringe candidates: one pregame rule (recent players still with the team), same for all; minutes if he
    #    plays from the minutes model (pregame row for every candidate), else his window average
    pool = fringe_pool(store, team_id, game_id, exclude | seen | absent_rot)
    for pid, share, cond_w in pool:
        if len(players) >= max_players:
            break
        rr = rates.loc[pid] if rates is not None and pid in rates.index else store.latest_rates(pid, date)
        if rr is None:
            continue
        cond = float(mins.loc[pid, minutes_col]) if mins is not None and pid in mins.index else cond_w
        cond = cond if np.isfinite(cond) and cond > 0 else max(cond_w, 1.0)
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


def availability_prior(store: InputStore, streak: float) -> float:
    """P(appears) of a rotation player by the team games he has missed in a row (estimated on the tuning season)."""
    if not store.priors:
        return 1.0 if streak == 0 else 0.0
    return float(store.priors.get(str(streak_bucket([0 if streak != streak else int(streak)])[0]), 0.0))


def pregame_team_spec(store: InputStore, game_id: int, team_id: int, overrides: dict | None = None,
                      max_players: int = MAX_PLAYERS) -> tuple[TeamSpec, dict]:
    """A team spec for an unplayed game from pregame information only: nothing from the game itself.

    * roster: the team's pool (players who appeared for it in the previous 25 days and not elsewhere since);
    * availability: no injury report - a rotation player who missed the team's last three games or more is presumed
      out, other rotation players play with the appearance rate of their missed-game streak (tuning season: about
      0.9 after playing the previous game), fringe players with their share of recent games; ``overrides`` (``out``,
      ``available``, ``p_play``, ``starters``, ``teams``) are the user's injury report - an ``available`` player
      outside the pool (back from a long absence) is added from his latest rates and minutes, on the team he last
      played for or ``teams[person id]``;
    * minutes: the lineup-blind model (starters are not known), conditional on playing;
    * starters: ``overrides['starters']`` if given (an announced starter plays), else the last game's five among the
      available rotation players, vacancies filled by start rate and then by projected minutes.
    """
    ov = overrides or {}
    out, forced, pset = set(ov.get("out", ())), set(ov.get("available", ())), dict(ov.get("p_play", {}))
    g = store.game_row(game_id)
    date = g["game_date"]
    f = store.by_game.get(game_id)
    f = f[f["team_id"] == team_id] if f is not None else None
    if f is None or f.empty:
        raise PregameInputsMissing(f"no pregame rows for game {game_id} (team {team_id}): it is outside the "
                                   "projection horizon, or the inputs need rebuilding (quantlab live-update)")
    rates = store.rates_by_game.get(game_id)
    mins = store.minutes_by_game.get(game_id)
    pool = store.pool.get((game_id, team_id))
    share = dict(zip(pool["person_id"].astype(int), pool["share"])) if pool is not None else {}
    cond_w = dict(zip(pool["person_id"].astype(int), pool["cond_window"])) if pool is not None else {}
    rot_rows, fringe_rows, presumed_out = [], [], []
    for r in f.itertuples(index=False):
        pid = int(r.person_id)
        if pid in out:
            continue
        rr = rates.loc[pid] if rates is not None and pid in rates.index else store.latest_rates(pid, date)
        if rr is None:
            continue
        streak = getattr(r, "missed_streak", 0.0)
        streak = 0.0 if streak != streak else float(streak)
        m = float(mins.loc[pid, "pred_gbm_pre"]) if mins is not None and pid in mins.index else np.nan
        if not np.isfinite(m) or m <= 0:
            m = float(np.nan_to_num(r.exp_min, nan=cond_w.get(pid, 8.0)))
        start_rate = (r.starts_m20 / r.gp_m20) if getattr(r, "gp_m20", 0) else 0.0
        rot = bool(getattr(r, "rotation_pre", False)) or pid in forced
        if rot:
            if pid in forced:
                p = float(pset.get(pid, 1.0))
            elif pid in pset:
                p = float(pset[pid])
            elif streak > PRESUMED_AVAILABLE_STREAK:
                presumed_out.append(pid)
                continue
            else:
                p = availability_prior(store, streak)
            rot_rows.append((pid, rr, m, p, start_rate))
        elif pid in share:
            fringe_rows.append((pid, rr, m, float(pset.get(pid, share[pid])), cond_w.get(pid, m)))
    # 'available' players without a row in this game (outside the 25-day pool: a return from a long absence, a
    # player who has not played for his new team yet): their latest rates and minutes, on the team they last played
    # for or the one the report gives (overrides['teams'])
    all_rows = store.by_game.get(game_id)
    have = set(all_rows["person_id"].astype(int)) if all_rows is not None else set()
    placed = {int(k): int(v) for k, v in (ov.get("teams") or {}).items()}
    for pid in sorted(forced - have - out):
        if placed.get(pid, store.last_team_before(pid, date)) != int(team_id):
            continue
        rr = store.latest_rates(pid, date)
        if rr is None:
            continue
        m, start_rate = store.latest_minutes(pid, date)
        rot_rows.append((pid, rr, m, float(pset.get(pid, 1.0)), start_rate))
    # starters
    want = [int(x) for x in (ov.get("starters") or {}).get(int(team_id), [])]
    avail_rot = {pid for pid, *_ in rot_rows}
    if want:
        five, source = [p for p in want if p in avail_rot][:5], "user"
    else:
        five, source = [p for p in store.last_starters(team_id, date) if p in avail_rot], "last game"
    if len(five) < 5:
        rest = sorted([x for x in rot_rows if x[0] not in five], key=lambda x: (-x[4], -x[2]))
        five += [x[0] for x in rest[:5 - len(five)]]
    five = set(five[:5])
    players = []
    for pid, rr, m, p, _ in sorted(rot_rows, key=lambda x: -x[2]):
        st = pid in five
        if st and source == "user" and pid not in pset:
            p = 1.0                                    # an announced starter plays
        players.append(PlayerSpec(pid, store.names.get(pid, str(pid)), _rates_dict(rr), min_mean=m,
                                  min_sd=minutes_sd(m, st), p_play=p, starter=st, rotation=True))
    for pid, rr, m, sh, cw in sorted(fringe_rows, key=lambda x: -x[3] * x[2]):
        if len(players) >= max_players:
            break
        cond = m if np.isfinite(m) and m > 0 else max(cw, 1.0)
        players.append(PlayerSpec(pid, store.names.get(pid, str(pid)), _rates_dict(rr),
                                  min_mean=max(sh * cond, 0.3), min_sd=3.0, p_play=sh, starter=False,
                                  rotation=False))
    players = players[:max_players]
    tf = store.tf.loc[(game_id, team_id)] if (game_id, team_id) in store.tf.index else None
    dfac = {k: float(tf[f"def_{k}"]) for k in DEF_FACTORS} if tf is not None else {}
    name = str(g["home_team"] if team_id == g["home_team_id"] else g["away_team"])
    meta = {"starters": sorted(five), "starters_source": source, "presumed_out": presumed_out,
            "out": sorted(out), "factors": tf is not None}
    return TeamSpec(int(team_id), name, players, def_factors=dfac), meta


def game_spec(store: InputStore, game_id: int, minutes_col: str = "pred_gbm", mode: str = "auto",
              overrides: dict | None = None) -> GameSpec:
    """The simulator's inputs for a game. ``mode``: 'asof' (a finished game, as backtested: the game's own
    appearances stand in for the inactive list and its starters are known), 'pregame' (an unplayed game:
    pregame information only, see ``pregame_team_spec``), 'auto' (pregame unless the game is finished)."""
    if mode == "pregame" or (mode == "auto" and not store.is_final(game_id)):
        return pregame_game_spec(store, int(game_id), overrides)
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
    return GameSpec(home, away, pace=pace, league=lg, meta={"mode": "asof"})


def pregame_game_spec(store: InputStore, game_id: int, overrides: dict | None = None) -> GameSpec:
    g = store.game_row(game_id)
    problem = store.pregame_problem(game_id)
    if problem:
        raise PregameInputsMissing(f"game {game_id}: {problem}")
    home, mh = pregame_team_spec(store, game_id, int(g["home_team_id"]), overrides)
    away, ma = pregame_team_spec(store, game_id, int(g["away_team_id"]), overrides)
    tp = store.tp.loc[game_id] if game_id in store.tp.index else None
    if tp is None:
        raise PregameInputsMissing(f"no team-model prediction for game {game_id}: rebuild the inputs "
                                   "(quantlab live-update)")
    home.target_ppp, away.target_ppp = float(tp["ppp_home"] / 100), float(tp["ppp_away"] / 100)
    rates = store.rates_by_game.get(game_id)
    lg = {}
    if rates is not None and len(rates):
        lg = {"blk": float(rates["lg_blk"].mean()), "ast_rate": float(rates["lg_ast_rate"].mean())}
    lg["tov_per_poss"] = store.league_tov(g["game_date"])
    return GameSpec(home, away, pace=float(tp["pace"]), league=lg,
                    meta={"mode": "pregame", "home": mh, "away": ma})
