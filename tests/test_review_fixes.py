"""Regression tests for defects found in review: minutes calibration with questionable players, scenario
overrides, the fringe minutes budget, the over/under line, the pregame fringe pool, the team-first
baseline's participants, the NB dispersion and the end-of-game logic."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from quantlab.eval.backtest import discrete_scores
from quantlab.eval.baselines import BinnedDispersion, nb_dispersion
from quantlab.eval.topdown import per_minute_expectations, simulate_game
from quantlab.sim.engine import GameSimulator, SimParams
from quantlab.features.asof import fringe_pool_table
from quantlab.sim.scenario import Scenario, apply


def test_questionable_rotation_player_keeps_his_minutes_when_he_plays(game, constants):
    """p_play < 1 must not inflate a rotation player's plan toward the 48-minute cap (calibration basis)."""
    g = copy.deepcopy(game)
    star = g.home.players[0]                     # 36 projected minutes
    star.p_play = 0.5
    sim = GameSimulator(g, SimParams(), constants)
    sim.calibrate(n_pilot=600)
    assert sim.plan[0, 0] < 42                   # was pushed to 48 before the fix
    res = sim.run(4000, seed=3)
    mins = res.stat("min")[:, 0, 0]
    plays = mins > 0
    assert 0.45 < plays.mean() < 0.55
    assert abs(mins[plays].mean() - 36.0) < 2.0
    # teammates absorb his minutes when he sits
    others_sit = res.stat("min")[~plays, 0, 1].mean()
    others_play = res.stat("min")[plays, 0, 1].mean()
    assert others_sit > others_play


def test_scenario_overrides_for_fringe_players_are_not_rescaled(game, constants):
    g = copy.deepcopy(game)
    fringe = g.home.players[-1]                 # rotation=False, p_play 0.3, 2 expected minutes
    assert not fringe.rotation
    new = apply(g, Scenario(availability={fringe.person_id: 0.6}))
    sim = GameSimulator(new, SimParams(), constants)
    i = [p.person_id for p in sim.teams[0].players].index(fringe.person_id)
    assert abs(sim.p_play[0, i] - 0.6) < 1e-9                          # typed probability kept
    assert abs(sim.min_mean[0, i] / sim.p_play[0, i] - 2.0 / 0.3) < 1e-6  # same minutes when he plays
    new = apply(g, Scenario(minutes={fringe.person_id: 4.0}))
    sim = GameSimulator(new, SimParams(), constants)
    assert abs(sim.min_mean[0, i] - 4.0) < 1e-9


def test_fringe_budget_is_continuous_in_rotation_minutes(game, constants):
    """A deeper projected rotation means less fringe participation, down to the p_min floor (no jump at 241.5)."""
    exp_fringe = []
    for total in (238.0, 241.3, 241.7, 250.0):
        g = copy.deepcopy(game)
        rot = [p for p in g.home.players if p.rotation]
        s = sum(p.min_mean for p in rot)
        for p in rot:
            p.min_mean *= total / s
        sim = GameSimulator(g, SimParams(), constants)
        fr = sim.exists[0] & ~sim.rotation[0]
        exp_fringe.append(float(sim.min_mean[0, fr].sum()))
    assert exp_fringe[0] > exp_fringe[1] >= exp_fringe[2] >= exp_fringe[3]
    assert exp_fringe[2] < 0.5 and exp_fringe[3] < 0.5


def test_over_under_line_is_the_nearest_half_point():
    rng = np.random.default_rng(0)
    for mean_target in (4.2, 4.7, 0.2):
        s = rng.poisson(mean_target, 5000)
        out = discrete_scores(s, 3, rng)
        assert out["line"] == max(np.floor(s.mean()) + 0.5, 0.5)
        assert abs(out["line"] - s.mean()) <= 0.5 + 1e-9 or out["line"] == 0.5


def test_fringe_pool_does_not_depend_on_todays_appearance():
    """One pregame rule: today's game never changes who is a candidate or his numbers."""
    pg = pd.DataFrame({
        "team_id": [1] * 7, "person_id": [7, 7, 8, 9, 7, 8, 5],
        "game_id": [1, 2, 2, 3, 4, 4, 4],
        "game_date": pd.to_datetime(["2024-01-05", "2024-01-10", "2024-01-10", "2024-01-15", "2024-01-20",
                                     "2024-01-20", "2024-01-20"]),
        "played": [True] * 7, "min": [5.0, 3.0, 9.0, 2.0, 30.0, 12.0, 20.0]})
    pool = fringe_pool_table(pg)
    today = pool[pool["game_id"] == 4].sort_values("person_id")
    assert today["person_id"].tolist() == [7, 8, 9]                   # 5 appears today only: not a candidate
    # dropping today's appearances changes nothing for today
    pg2 = pg[pg["game_id"] != 4]
    pool2 = fringe_pool_table(pd.concat([pg2, pd.DataFrame({"team_id": [1], "person_id": [7], "game_id": [4],
                                                            "game_date": [pd.Timestamp("2024-01-20")],
                                                            "played": [False], "min": [0.0]})]))
    today2 = pool2[pool2["game_id"] == 4].sort_values("person_id")
    assert today[["person_id", "share", "cond_window"]].to_numpy().tolist() == \
        today2[["person_id", "share", "cond_window"]].to_numpy().tolist()
    assert abs(today.loc[today["person_id"] == 7, "share"].iloc[0] - 2 / 3) < 1e-9   # 2 of the team's 3 games


def test_team_first_baseline_only_credits_sampled_participants():
    rates = {"usage": 0.2, "tov_share": 0.12, "fga3_share": 0.37, "trip_share": 0.1, "fg2": 0.55, "fg3": 0.36,
             "ft": 0.78, "and1": 0.06, "trip3_share": 0.04, "ast_rate": 0.16, "oreb": 0.045, "dreb": 0.13,
             "stl": 0.017, "blk": 0.011}
    e = per_minute_expectations(rates, {}, 99.0)
    rows = []
    for side in (0, 1):
        for k in range(10):
            rows.append({"team_side": side, "person_id": side * 100 + k, "rotation": k < 8,
                         "p_play": 1.0 if k < 8 else (0.0 if k == 9 else 0.4), "cond_min": 30.0 if k < 8 else 6.0, **e})
    P = pd.DataFrame(rows)
    tp = pd.Series({"exp_home_pts": 112.0, "exp_away_pts": 108.0, "sigma_margin": 13.0, "sigma_total": 18.0,
                    "rho_mt": 0.0})
    out = simulate_game(tp, P, 400, np.random.default_rng(1))
    for side, (pids, res, part) in out.items():
        assert np.allclose(res["min"].sum(axis=1), 240.0)
        assert not part[:, 9].any() and (res["pts"][:, 9] == 0).all()          # p_play 0 never plays
        assert 0.25 < part[:, 8].mean() < 0.55                                   # sampled participation
        assert ((res["pts"] == 0) | part).all()


def test_binned_dispersion_tracks_mean_dependent_overdispersion():
    rng = np.random.default_rng(2)
    mu = rng.uniform(0.5, 25, 60000)
    r_true = 0.5 + 0.6 * mu                                   # dispersion grows with the mean, as for points
    y = rng.negative_binomial(r_true, r_true / (r_true + mu))
    disp = BinnedDispersion(mu, y)
    assert disp(np.array([2.0]))[0] < disp(np.array([20.0]))[0]
    pooled = nb_dispersion(mu, y)
    assert disp(np.array([2.0]))[0] < pooled < disp(np.array([20.0]))[0]


def test_tied_last_possession_rarely_loses(game, constants):
    """A tied team holding for the last shot leaves the opponent (almost) no time: ties go to overtime."""
    sim = GameSimulator(copy.deepcopy(game), SimParams(), constants)
    res = sim.run(6000, seed=11)
    lead24, end = res.marks[:, 2].astype(int), (res.marks[:, 1] * res.marks[:, 3]).astype(int)
    tied = (lead24 == 0) & (res.marks[:, 3] != 0)
    assert tied.sum() > 50
    assert (end[tied] < 0).mean() < 0.10                      # the defense rarely wins from a tied last possession
    assert (end[tied] == 0).mean() > 0.5


def test_player_added_because_he_appeared_does_not_move_anyone_else(game, constants):
    """A player listed only because he appeared (in_pool False) keeps his floor participation and stays out of the
    minutes budget, so the other players' pregame numbers are identical with or without him."""
    from quantlab.sim.engine import PlayerSpec
    base = GameSimulator(copy.deepcopy(game), SimParams(), constants)
    g = copy.deepcopy(game)
    extra = copy.deepcopy(g.home.players[-1])
    extra.person_id, extra.in_pool, extra.p_play, extra.min_mean = 999, False, 0.05, 0.05 * 9.0
    assert isinstance(extra, PlayerSpec)
    g.home.players.append(extra)
    sim = GameSimulator(g, SimParams(), constants)
    n = len(game.home.players)
    assert np.allclose(sim.p_play[0, :n], base.p_play[0, :n])
    assert np.allclose(sim.min_mean[0, :n], base.min_mean[0, :n])
    assert abs(sim.p_play[0, n] - 0.05) < 1e-12
