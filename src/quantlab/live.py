"""Pregame projection service used by the dashboard and the CLI.

Two kinds of games:

* **unplayed games of the live season** (``quantlab live-update`` fetches the season from cdn.nba.com): projected
  truly pregame - roster from the team's recent games, availability from missed-game streaks plus the user's
  overrides (data/live_overrides.json), projected starters and the lineup-blind minutes model
  (``sim.inputs.pregame_team_spec``);
* **finished games** (any date of the database): projected as in the backtest - earlier games, plus the game's
  starting fives and who appeared as the stand-in for the inactive list - with the actual result alongside.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import PATHS
from .models.minutes import FEATURES, FEATURES_PRE, STARTER_FEATURES
from .sim.engine import GameSimulator, GameSpec, SimParams, SimResult
from .sim.inputs import InputStore, game_spec
from .sim.scenario import Scenario, apply

OVERRIDES_PATH = PATHS.root / "live_overrides.json"


def load_overrides() -> dict:
    """The user's injury report for unplayed games: {'out': [ids], 'available': [ids], 'p_play': {id: p}}."""
    if not OVERRIDES_PATH.exists():
        return {"out": [], "available": [], "p_play": {}}
    raw = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
    return {"out": [int(x) for x in raw.get("out", [])], "available": [int(x) for x in raw.get("available", [])],
            "p_play": {int(k): float(v) for k, v in (raw.get("p_play") or {}).items()}}


def save_overrides(ov: dict) -> None:
    OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean = {"out": sorted({int(x) for x in ov.get("out", [])}),
             "available": sorted({int(x) for x in ov.get("available", [])}),
             "p_play": {str(int(k)): float(v) for k, v in (ov.get("p_play") or {}).items()}}
    OVERRIDES_PATH.write_text(json.dumps(clean, indent=1), encoding="utf-8")


class MinutesReprojector:
    """Re-run the season-appropriate minutes model after removing players from a team."""

    def __init__(self):
        self.frame = pd.read_parquet(PATHS.models / "minutes_features.parquet")
        self.by_game = {g: d for g, d in self.frame.groupby("game_id")}
        self.models: dict = {}
        self.last_status = ""

    def model(self, season: str, pregame: bool = False):
        """The minutes model trained on seasons before ``season`` (None if there is none: the first two
        seasons have no walk-forward model, and the all-seasons model would use future games). ``pregame``: the
        lineup-blind model, for unplayed games."""
        key = (season, pregame)
        if key not in self.models:
            path = PATHS.models / (f"minutes_gbm_pre_{season}.txt" if pregame else f"minutes_gbm_{season}.txt")
            self.models[key] = (lgb.Booster(model_str=path.read_text(encoding="utf-8"))
                                if path.exists() else None)
        return self.models[key]

    def __call__(self, game_id: int, team, removed: list) -> None:
        f = self.by_game.get(game_id)
        if f is None:
            self.last_status = "no minutes features for this game: minutes not re-projected"
            return
        season = str(f["season"].iloc[0])
        pregame = bool(f["scheduled"].fillna(False).any()) if "scheduled" in f else False
        if self.model(season, pregame) is None:
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
        if pregame:                    # unplayed game: no starters known, lineup-blind model and context
            gone_exp = float(gone.loc[gone.get("rotation_pre", pd.Series(True, index=gone.index)).astype(bool),
                                      "exp_min"].fillna(0).sum())
            pre = rest["avail_exp_min_pre"].fillna(240) - gone_exp
            rest["avail_exp_min_pre"] = pre.clip(lower=60)
            rest["share_of_avail_pre"] = rest["exp_min"] / rest["avail_exp_min_pre"]
            pred = self.model(season, True).predict(rest[FEATURES_PRE].astype(np.float32))
            newmin = dict(zip(rest.index, pred))
            for p in team.players:
                if p.person_id in newmin and p.rotation and not p.locked:
                    p.min_mean = float(max(newmin[p.person_id], 0.5))
            self.last_status = f"minutes re-projected with the lineup-blind model trained on seasons before {season}"
            return
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
        self.overrides = load_overrides()

    # ------------------------------------------------------------------ slates
    def _all_games(self) -> pd.DataFrame:
        """Built games plus the live season's unplayed games (status, tip time; no scores)."""
        g = self.store.games.assign(status="final", is_final=True)
        sc = self.store.schedule
        if sc is not None and len(sc):
            up = sc[~sc.index.isin(g.index) & (sc["home_team_id"] > 0) & (sc["away_team_id"] > 0)].assign(
                is_final=False)
            g = pd.concat([g, up[[c for c in up.columns if c in g.columns or c in ("status", "is_final",
                                                                                   "tip_utc")]]])
        return g

    def dates(self, season: str | None = None) -> list:
        g = self._all_games()
        if season:
            g = g[g["season"] == season]
        return sorted(pd.to_datetime(g["game_date"]).dt.date.unique())

    def slate(self, date) -> pd.DataFrame:
        g = self._all_games()
        d = g[pd.to_datetime(g["game_date"]).dt.date == pd.Timestamp(date).date()].copy()
        tp = self.store.tp
        d = d.join(tp[["exp_home_pts", "exp_away_pts", "exp_margin", "exp_total", "sigma_margin", "p_home_win",
                       "pace"]], how="left")
        d["projectable"] = d["is_final"] | d.index.isin(self.store.by_game.keys())
        return d.reset_index().rename(columns={"index": "game_id"})

    def game_info(self, game_id: int) -> pd.Series:
        return self.store.game_row(int(game_id))

    def is_final(self, game_id: int) -> bool:
        return self.store.is_final(int(game_id))

    def set_overrides(self, ov: dict) -> None:
        """Replace the injury report used for unplayed games (persisted to data/live_overrides.json)."""
        save_overrides(ov)
        self.overrides = load_overrides()
        self._shift_cache = {k: v for k, v in self._shift_cache.items() if self.is_final(k[0])}

    # ------------------------------------------------------------------ projections
    def spec(self, game_id: int) -> GameSpec:
        gid = int(game_id)
        if self.is_final(gid):
            return game_spec(self.store, gid)
        spec = game_spec(self.store, gid, mode="pregame", overrides=self.overrides)
        out = set(self.overrides.get("out", ()))
        rows = self.store.by_game.get(gid)
        if out and rows is not None:   # injury report: teammates' minutes re-projected without them
            from .features.asof import PRESUMED_AVAILABLE_STREAK
            presumed = rows[rows["missed_streak"].fillna(0) <= PRESUMED_AVAILABLE_STREAK] \
                if "missed_streak" in rows else rows
            for team in (spec.home, spec.away):
                # only players the pregame context counted as available (a long absence is already absent)
                removed = sorted(out & {int(x) for x in presumed.loc[presumed["team_id"] == team.team_id,
                                                                       "person_id"]})
                if removed:
                    self.reproj(gid, team, removed)
                    spec.meta["minutes_reprojection"] = self.reproj.last_status
        return spec

    def minutes_fn(self, game_id: int):
        return lambda team, removed: self.reproj(int(game_id), team, removed)

    def environment_shift(self, game_id: int, environment: str, env_weight: float) -> tuple:
        """Team environment shifts of the *baseline* projection (cached): scenarios reuse them so that a
        roster change acts bottom-up instead of being re-anchored to a team model that ignores it."""
        key = (int(game_id), environment, round(float(env_weight), 4),
               None if self.is_final(game_id) else json.dumps(self.overrides, sort_keys=True))
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
        info["information_set"] = spec.meta.get("mode", "asof")
        if spec.meta.get("mode") == "pregame":
            info["pregame"] = {k: spec.meta[k] for k in ("home", "away") if k in spec.meta}
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
        return float(json.loads(path.read_text(encoding="utf-8")).get("blend_weight", 0.5))
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
