"""Matchup / injury scenario simulator.

A ``Scenario`` edits a pregame ``GameSpec``:

* ``out``             players removed (injury, rest) - the minutes model is re-run for the remaining
                      players with the updated teammate-absence features, so minutes are redistributed
                      the way they historically were (not pro rata);
* ``minutes``         manual projected-minutes overrides (honoured in expectation by calibration and never
                      rescaled by the team's minutes budget);
* ``starters``        manual starting five per team;
* ``availability``    participation probabilities (e.g. 0.5 for a questionable player: each simulated
                      game samples whether he plays, giving the correct mixture distribution; his minutes
                      when he plays stay at his projection and teammates absorb them when he sits);
* ``usage``           usage multipliers (role changes); ``caps`` minutes restrictions.

Shots, usage, assists and rebounds are then re-allocated inside the simulation (usage competition
among the five on court, passing and rebounding weights of the actual lineups), the opponent's
defensive matchups change through the defenders on court, and full score distributions and prop
probabilities are recomputed. Both versions are simulated with the same seed (common random numbers)
and compared.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .engine import GameSimulator, GameSpec, SimParams, SimResult

# Usage-efficiency tradeoff, estimated on 2021-22..2023-24 rotation player-games with teammate absences as a
# natural experiment (docs/REPORT.md): +1 unit of relative usage changes true shooting by -0.088
# (se 0.020), i.e. a logit shift of about -0.33 on the make probability. Naive OLS gets the opposite sign
# (+0.02) because hot shooters both shoot more and make more.
USAGE_EFFICIENCY_LOGIT = -0.33


def expected_shares(team, alpha: float = 1.3) -> dict:
    """Expected share of team plays each player uses while on court, given his teammates' planned minutes
    and usage (the same weights the simulator uses: usage^alpha x usage multiplier)."""
    w = {p.person_id: (max(p.rates["usage"], 1e-4) ** alpha) * p.usage_mult for p in team.players}
    m = {p.person_id: max(p.min_mean, 0.0) * p.p_play for p in team.players}
    out = {}
    for p in team.players:
        others = [q.person_id for q in team.players if q.person_id != p.person_id]
        tot = sum(m[q] for q in others)
        wbar = sum(m[q] * w[q] for q in others) / tot if tot > 0 else 0.0
        out[p.person_id] = w[p.person_id] / (w[p.person_id] + 4.0 * wbar) if wbar > 0 else 1.0
    return out


def _shift_rate(r: float, d: float) -> float:
    r = min(max(r, 1e-4), 1 - 1e-4)
    return 1.0 / (1.0 + math.exp(-(math.log(r / (1 - r)) + d)))


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
    base_shares = [expected_shares(spec.home), expected_shares(spec.away)]
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
                p.locked = True                   # typed minutes are honoured, never rescaled by the budget
            if p.person_id in sc.availability:
                new_p = float(sc.availability[p.person_id])
                if not p.rotation and p.person_id not in sc.minutes and p.p_play > 0:
                    p.min_mean *= new_p / p.p_play    # fringe: same minutes when he plays, new participation
                p.p_play = new_p
                if not p.rotation:
                    p.locked = True
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
    # usage-efficiency tradeoff: players whose expected load rises shoot less efficiently (and vice versa)
    for t, team in enumerate((new.home, new.away)):
        shares = expected_shares(team)
        for p in team.players:
            b = base_shares[t].get(p.person_id)
            if not b:
                continue
            rel = float(np.clip(shares[p.person_id] / b - 1.0, -0.5, 1.0))
            d = USAGE_EFFICIENCY_LOGIT * rel
            if abs(d) > 1e-6:
                p.rates = dict(p.rates)
                p.rates["fg2"] = _shift_rate(p.rates["fg2"], d)
                p.rates["fg3"] = _shift_rate(p.rates["fg3"], d)
    return new


def run_pair(spec: GameSpec, sc: Scenario, n: int = 20000, seed: int = 7, params: SimParams | None = None,
             environment: str = "none", env_weight: float = 0.5, minutes_fn=None) -> tuple[SimResult, SimResult]:
    """Baseline and scenario with common random numbers. The scenario reuses the baseline's team
    environment shift (a property of the matchup): roster changes then act bottom-up, through minutes,
    usage, efficiency, rebounding and defence, instead of being re-anchored to a team model that does not
    know about them."""
    base = GameSimulator(copy.deepcopy(spec), params)
    base.calibrate(environment=environment, env_weight=env_weight)
    r0 = base.run(n, seed=seed)
    new = apply(spec, sc, minutes_fn)
    new.home.eff_shift, new.away.eff_shift = base.teams[0].eff_shift, base.teams[1].eff_shift
    mod = GameSimulator(new, params)
    mod.calibrate(environment="none")
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
