"""Shared fixtures: a synthetic two-team game spec and simulator constants (no downloaded data needed)."""

from __future__ import annotations

import numpy as np
import pytest

from quantlab.sim.engine import GameSpec, PlayerSpec, TeamSpec

TIERS = [0, 12, 20, 28, 34, 60]


def flat_constants() -> dict:
    starter = [1.0] * 6 + [0.5] * 12 + [1.0] * 6 + [0.5] * 12 + [0.8] * 12
    bench = [0.0] * 6 + [0.6] * 12 + [0.0] * 6 + [0.6] * 12 + [0.3] * 12
    templates = {}
    for tier in range(5):
        templates[f"1_{tier}"] = starter
        templates[f"0_{tier}"] = bench
    return {
        "templates": templates, "template_tiers": TIERS,
        "garbage_ratio": {str(k): [1.0, 1.0, 0.95, 0.7, 0.3, 0.2, 0.1] for k in range(36, 48)},
        "margin_buckets": [0, 6, 11, 16, 21, 26, 31, 100],
        "poss_time": {"a": 14.0, "b_per_oreb": 5.1, "gamma_shape": 4.5, "mean_no_oreb": 14.0, "mean_all": 14.6,
                      "oreb_per_poss": 0.125},
        "team_reb_share": 0.111, "team_oreb_share_of_team_reb": 0.61, "team_tov_share": 0.05,
        "nonft_def_fouls_per_opp_play": 0.05, "tech_fta_per_team_game": 0.58,
        "late_game": {"baseline_p_trip_q4": 0.097,
                      "leading": {"1-3": {"p_trip": 0.41}, "4-6": {"p_trip": 0.54}, "7-9": {"p_trip": 0.50},
                                  "10-99": {"p_trip": 0.13}}},
    }


def player(pid: int, minutes: float, starter: bool, usage: float = 0.2, fg3_share: float = 0.37, rotation=True,
           p_play=1.0) -> PlayerSpec:
    rates = {"usage": usage, "tov_share": 0.12, "fga3_share": fg3_share, "trip_share": 0.10, "fg2": 0.55, "fg3": 0.36,
             "ft": 0.78, "and1": 0.06, "trip3_share": 0.04, "ast2_share": 0.54, "ast3_share": 0.85, "ast_rate": 0.16,
             "oreb": 0.045, "dreb": 0.13, "stl": 0.017, "blk": 0.011, "pf_def": 0.036, "off_foul_share": 0.1,
             "blocked": 0.09}
    return PlayerSpec(pid, f"P{pid}", rates, min_mean=minutes, min_sd=4.0, p_play=p_play, starter=starter,
                      rotation=rotation)


def team(team_id: int, base_pid: int, star_usage: float = 0.3) -> TeamSpec:
    mins = [36, 34, 32, 30, 28, 24, 20, 16, 12, 8]
    players = []
    for k, m in enumerate(mins):
        u = star_usage if k == 0 else 0.19
        players.append(player(base_pid + k, m, k < 5, usage=u, rotation=m >= 12))
    players.append(player(base_pid + 10, 2.0, False, rotation=False, p_play=0.3))
    return TeamSpec(team_id, f"T{team_id}", players, def_factors={})


@pytest.fixture
def constants() -> dict:
    return flat_constants()


@pytest.fixture
def game() -> GameSpec:
    return GameSpec(team(1, 100), team(2, 200), pace=99.0,
                    league={"blk": 0.011, "ast_rate": 0.16, "tov_per_poss": 0.142, "fg2": 0.55, "fg3": 0.36})


@pytest.fixture
def rng():
    return np.random.default_rng(0)
