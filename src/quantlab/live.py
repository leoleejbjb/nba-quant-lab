"""Pregame projection service used by the dashboard and the CLI.

"Upcoming games" are defined by an *as-of date*: every game on that date is projected using only
information from earlier dates (the same walk-forward inputs as the backtest). For historical as-of
dates the actual results are shown next to the projections; the live 2026-27 feed is not reachable
from the build environment (cdn.nba.com returns "Access Denied"), see docs/DATA.md.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import PATHS
from .models.minutes import FEATURES, STARTER_FEATURES
from .sim.engine import GameSimulator, GameSpec, SimParams, SimResult
from .sim.inputs import InputStore, game_spec
from .sim.scenario import Scenario, apply


class MinutesReprojector:
    """Re-run the season-appropriate minutes model after removing players from a team."""

    def __init__(self):
        self.frame = pd.read_parquet(PATHS.models / "minutes_features.parquet")
        self.by_game = {g: d for g, d in self.frame.groupby("game_id")}
        self.models: dict = {}
        self.last_status = ""

    def model(self, season: str):
        """The minutes model trained on seasons before ``season`` (None if there is none: the first two
        seasons have no walk-forward model, and the all-seasons model would use future games)."""
        if season not in self.models:
            path = PATHS.models / f"minutes_gbm_{season}.txt"
            self.models[season] = lgb.Booster(model_file=str(path)) if path.exists() else None
        return self.models[season]

    def __call__(self, game_id: int, team, removed: list) -> None:
        f = self.by_game.get(game_id)
        if f is None:
            self.last_status = "no minutes features for this game: minutes not re-projected"
            return
        season = str(f["season"].iloc[0])
        if self.model(season) is None:
            self.last_status = (f"no walk-forward minutes model for {season}: minutes not re-projected "
                                "(removed players' minutes are shared out by the simulator's 240-minute budget)")
            return
        self.last_status = f"minutes re-projected with the model trained on seasons before {season}"
        tf = f[f["team_id"] == team.team_id].set_index("person_id")
        gone = tf.loc[tf.index.intersection(removed)]
        rest = tf.loc[tf.index.intersection([p.person_id for p in team.players])].copy()
        if rest.empty:
            return
        vac = float(gone["base_min"].fillna(0).sum())
        used_share = float((gone["base_used"].fillna(0).sum() / max(rest["team_used_base"].iloc[0], 1e-9))
                           if "team_used_base" in rest else 0.0)
        rest["vacated_min"] = rest["vacated_min"].fillna(0) + vac
        rest["slack_min"] = rest["slack_min"].fillna(0) + vac
        rest["n_absent_15"] = rest["n_absent_15"].fillna(0) + int((gone["base_min"].fillna(0) >= 15).sum())
        rest["vacated_used_share"] = rest["vacated_used_share"].fillna(0) + used_share
        avail = rest["avail_exp_min"].fillna(240) - float(gone["exp_min"].fillna(0).sum())
        rest["avail_exp_min"] = avail.clip(lower=60)
        rest["share_of_avail"] = rest["exp_min"] / rest["avail_exp_min"]
        # a removed starter is replaced by the remaining non-starter with the highest start rate / minutes
        starters_gone = int(gone["starter_raw"].fillna(0).sum())
        cur_starters = {p.person_id for p in team.players if p.starter}
        if starters_gone:
            cand = rest[~rest.index.isin(cur_starters)].copy()
            cand["key"] = cand["start_rate"].fillna(0) * 10 + cand["exp_min"].fillna(0)
            for pid in cand.sort_values("key", ascending=False).index[:starters_gone]:
                cur_starters.add(int(pid))
        rest["starter"] = rest.index.isin(cur_starters).astype(float)
        st_min = rest.loc[rest["starter"] == 1, "exp_min"].fillna(0).sum()
        rest["n_starters_known_min"] = st_min
        mdl = self.model(str(rest["season"].iloc[0]))
        pred = mdl.predict(rest[FEATURES + STARTER_FEATURES].astype(np.float32))
        newmin = dict(zip(rest.index, pred))
        for p in team.players:
            if p.person_id in newmin and p.rotation:
                p.min_mean = float(max(newmin[p.person_id], 0.5))
            p.starter = p.person_id in cur_starters
            if p.starter:
                p.rotation, p.p_play = True, 1.0


@dataclass
class Projection:
    game_id: int
    spec: GameSpec
    result: SimResult
    calibration: dict


class Projector:
    def __init__(self, params: SimParams | None = None):
        self.store = InputStore()
        self.reproj = MinutesReprojector()
        self.params = params or SimParams()
        self._shift_cache: dict = {}

    # ------------------------------------------------------------------ slates
    def dates(self, season: str | None = None) -> list:
        g = self.store.games
        if season:
            g = g[g["season"] == season]
        return sorted(g["game_date"].dt.date.unique())

    def slate(self, date) -> pd.DataFrame:
        g = self.store.games
        d = g[g["game_date"].dt.date == pd.Timestamp(date).date()].copy()
        tp = self.store.tp
        d = d.join(tp[["exp_home_pts", "exp_away_pts", "exp_margin", "exp_total", "sigma_margin", "p_home_win",
                       "pace"]], how="left")
        return d.reset_index()

    # ------------------------------------------------------------------ projections
    def spec(self, game_id: int) -> GameSpec:
        return game_spec(self.store, int(game_id))

    def minutes_fn(self, game_id: int):
        return lambda team, removed: self.reproj(int(game_id), team, removed)

    def environment_shift(self, game_id: int, environment: str, env_weight: float) -> tuple:
        """Team environment shifts of the *baseline* projection (cached): scenarios reuse them so that a
        roster change acts bottom-up instead of being re-anchored to a team model that ignores it."""
        key = (int(game_id), environment, round(float(env_weight), 4))
        if key not in self._shift_cache:
            sim = GameSimulator(self.spec(game_id), self.params)
            sim.calibrate(environment=environment, env_weight=env_weight, n_pilot=1500)
            self._shift_cache[key] = (sim.teams[0].eff_shift, sim.teams[1].eff_shift)
        return self._shift_cache[key]

    def project(self, game_id: int, n: int = 20000, seed: int = 7, environment: str = "blend",
                env_weight: float | None = None, scenario: Scenario | None = None,
                base_spec: GameSpec | None = None) -> Projection:
        w = env_weight if env_weight is not None else default_blend_weight()
        spec = copy.deepcopy(base_spec) if base_spec is not None else self.spec(game_id)
        shift = self.environment_shift(game_id, environment, w) if environment != "none" else (0.0, 0.0)
        self.reproj.last_status = ""
        if scenario is not None and not scenario.is_empty():
            spec = apply(spec, scenario, self.minutes_fn(game_id))
        spec.home.eff_shift, spec.away.eff_shift = shift
        sim = GameSimulator(spec, self.params)
        info = sim.calibrate(environment="none")          # minutes only; environment fixed above
        info["environment_shift"] = list(shift)
        if self.reproj.last_status:
            info["minutes_reprojection"] = self.reproj.last_status
        res = sim.run(n, seed=seed)
        return Projection(int(game_id), spec, res, info)

    def actual_box(self, game_id: int) -> pd.DataFrame:
        pg = pd.read_parquet(PATHS.processed("all") / "player_games.parquet",
                             filters=[("game_id", "==", int(game_id))])
        return pg


def default_blend_weight() -> float:
    """Team-model weight in the environment blend, chosen on the tuning season (eval/architecture.py)."""
    path = PATHS.models / "integration.json"
    if path.exists():
        import json
        return float(json.loads(path.read_text()).get("blend_weight", 0.5))
    return 0.5


def box_score_table(res: SimResult, team_index: int) -> pd.DataFrame:
    """NBA-style projected box score (means + 10th/90th percentiles) for one team, conditional on playing."""
    rows = []
    players = res.players[team_index]
    for i, pl in enumerate(players):
        played = res.secs[:, team_index, i] > 0
        if played.mean() < 0.02:
            continue
        def m(stat):
            x = res.stat(stat)[played, team_index, i]
            return float(x.mean())
        def q(stat, p):
            x = res.stat(stat)[played, team_index, i]
            return float(np.percentile(x, p))
        rows.append({
            "person_id": pl.person_id, "Player": pl.name, "Start": "✓" if pl.starter else "",
            "P(play)": float(played.mean()),
            "MIN": m("min"), "MIN_lo": q("min", 10), "MIN_hi": q("min", 90),
            "PTS": m("pts"), "PTS_lo": q("pts", 10), "PTS_hi": q("pts", 90),
            "REB": m("reb"), "REB_lo": q("reb", 10), "REB_hi": q("reb", 90),
            "AST": m("ast"), "AST_lo": q("ast", 10), "AST_hi": q("ast", 90),
            "FGM": m("fgm"), "FGA": m("fga"), "3PM": m("fg3m"), "3PA": m("fg3a"),
            "FTM": m("ftm"), "FTA": m("fta"), "OREB": m("oreb"), "DREB": m("dreb"),
            "STL": m("stl"), "BLK": m("blk"), "TOV": m("tov"), "PF": m("pf"), "+/-": m("pm"),
        })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["Start", "MIN"], ascending=[False, False]).reset_index(drop=True)
    return df


def team_totals(res: SimResult) -> pd.DataFrame:
    rows = []
    for t in range(2):
        team = res.game.home if t == 0 else res.game.away
        st = {s: res.stat(s)[:, t, :].sum(axis=1) for s in ("pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb",
                                                           "dreb", "ast", "stl", "blk", "tov", "pf")}
        tr = res.team_reb[:, t, :].sum(axis=1)
        rows.append({"Team": team.name, **{k.upper(): float(v.mean()) for k, v in st.items()},
                     "TEAM_REB": float(tr.mean()), "REB": float((st["oreb"] + st["dreb"] + tr).mean()),
                     "POSS": float(res.poss[:, t].mean())})
    return pd.DataFrame(rows)
