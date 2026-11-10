"""Matchup / injury scenario simulator.

A ``Scenario`` edits a pregame ``GameSpec``:

* ``out``             players removed (injury, rest) - the minutes model is re-run for the remaining
                      players with the updated teammate-absence features, so minutes are redistributed
                      the way they historically were (not pro rata);
* ``minutes``         manual projected-minutes overrides (honoured in expectation by calibration);
* ``starters``        manual starting five per team;
* ``availability``    participation probabilities (e.g. 0.5 for a questionable player: each simulated
                      game samples whether he plays, giving the correct mixture distribution);
* ``usage``           usage multipliers (role changes); ``caps`` minutes restrictions.

Shots, usage, assists and rebounds are then re-allocated inside the simulation (usage competition
among the five on court, passing and rebounding weights of the actual lineups), the opponent's
defensive matchups change through the defenders on court, and full score distributions and prop
probabilities are recomputed. Both versions are simulated with the same seed (common random numbers)
and compared.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .engine import GameSimulator, GameSpec, SimParams, SimResult


@dataclass
class Scenario:
    out: list = field(default_factory=list)
    minutes: dict = field(default_factory=dict)
    starters: dict = field(default_factory=dict)       # team_id -> [5 person_ids]
    availability: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    caps: dict = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (self.out or self.minutes or self.starters or self.availability or self.usage or self.caps)


def apply(spec: GameSpec, sc: Scenario, minutes_fn=None) -> GameSpec:
    """Return a modified deep copy of ``spec``. ``minutes_fn(team_spec, removed_ids)`` may re-project
    minutes for a team after removals (see ``quantlab.live.reproject_minutes``)."""
    new = copy.deepcopy(spec)
    for team in (new.home, new.away):
        removed = [p.person_id for p in team.players if p.person_id in set(sc.out)]
        team.players = [p for p in team.players if p.person_id not in set(sc.out)]
        if removed and minutes_fn is not None:
            minutes_fn(team, removed)
        for p in team.players:
            if p.person_id in sc.minutes:
                p.min_mean = float(sc.minutes[p.person_id])
                p.rotation = p.min_mean >= 6 or p.rotation
                p.p_play = 1.0 if p.rotation else p.p_play
            if p.person_id in sc.availability:
                p.p_play = float(sc.availability[p.person_id])
            if p.person_id in sc.usage:
                p.usage_mult = float(sc.usage[p.person_id])
            if p.person_id in sc.caps:
                p.minutes_cap = float(sc.caps[p.person_id])
                p.min_mean = min(p.min_mean, p.minutes_cap)
        if team.team_id in sc.starters:
            five = set(sc.starters[team.team_id])
            for p in team.players:
                p.starter = p.person_id in five
                if p.starter:
                    p.rotation, p.p_play = True, max(p.p_play, 1.0 if p.person_id not in sc.availability else p.p_play)
    return new


def run_pair(spec: GameSpec, sc: Scenario, n: int = 20000, seed: int = 7, params: SimParams | None = None,
             environment: str = "none", minutes_fn=None) -> tuple[SimResult, SimResult]:
    base = GameSimulator(copy.deepcopy(spec), params)
    base.calibrate(environment=environment)
    r0 = base.run(n, seed=seed)
    mod = GameSimulator(apply(spec, sc, minutes_fn), params)
    mod.calibrate(environment=environment)
    r1 = mod.run(n, seed=seed)
    return r0, r1


def player_table(res: SimResult, stats=("min", "pts", "reb", "ast", "fg3m", "stl", "blk", "tov")) -> pd.DataFrame:
    rows = []
    for t in range(2):
        team = res.game.home if t == 0 else res.game.away
        for i, pl in enumerate(res.players[t]):
            played = res.secs[:, t, i] > 0
            row = {"team": team.name, "person_id": pl.person_id, "name": pl.name, "p_play": float(played.mean())}
            for s in stats:
                x = res.stat(s)[:, t, i]
                row[s] = float(x.mean())
            rows.append(row)
    return pd.DataFrame(rows)


def compare(r0: SimResult, r1: SimResult) -> dict:
    a, b = player_table(r0), player_table(r1)
    m = a.merge(b, on=["team", "person_id", "name"], how="outer", suffixes=("_base", "_scen"))
    for s in ("min", "pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "p_play"):
        m[f"d_{s}"] = m[f"{s}_scen"].fillna(0) - m[f"{s}_base"].fillna(0)
    def team_sum(r):
        sc = r.score.astype(float)
        mg = sc[:, 0] - sc[:, 1]
        return {"home_pts": float(sc[:, 0].mean()), "away_pts": float(sc[:, 1].mean()), "margin": float(mg.mean()),
                "total": float(sc.sum(axis=1).mean()), "p_home_win": float((mg > 0).mean()),
                "margin_sd": float(mg.std()), "total_sd": float(sc.sum(axis=1).std())}
    t0, t1 = team_sum(r0), team_sum(r1)
    return {"players": m, "team_base": t0, "team_scenario": t1,
            "team_delta": {k: t1[k] - t0[k] for k in t0}}
