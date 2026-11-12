"""Simulator invariants and behaviour on a synthetic game (no downloaded data required)."""

from __future__ import annotations

import copy

import numpy as np

from quantlab.sim.engine import GameSimulator, SimParams, S
from quantlab.sim.scenario import Scenario, apply


def run(game, constants, n=3000, seed=1, **kw):
    sim = GameSimulator(copy.deepcopy(game), SimParams(**kw), constants)
    return sim, sim.run(n, seed=seed)


def test_box_score_invariants_hold_in_every_simulated_game(game, constants):
    _, res = run(game, constants)
    inv = res.check_invariants()
    assert inv["all"], inv


def test_team_rebounds_and_misses_reconcile(game, constants):
    _, res = run(game, constants)
    st = res.stats.astype(np.int64)
    for t in (0, 1):
        o = 1 - t
        misses = (st[:, t, :, S["fga2"]] - st[:, t, :, S["fgm2"]] + st[:, t, :, S["fga3"]] -
                  st[:, t, :, S["fgm3"]]).sum(axis=1)
        rebounds_of_misses = (st[:, t, :, S["oreb"]].sum(axis=1) + st[:, o, :, S["dreb"]].sum(axis=1)
                              + res.team_reb[:, t, 0] + res.team_reb[:, o, 1])
        # every rebound follows a live miss (FG or final FT); FG misses alone can't exceed rebounds by much
        assert (rebounds_of_misses <= misses + st[:, t, :, S["fta"]].sum(axis=1)).all()
        assert (rebounds_of_misses >= 0.8 * misses).mean() > 0.95


def test_same_seed_is_deterministic(game, constants):
    _, a = run(game, constants, n=800, seed=5)
    _, b = run(game, constants, n=800, seed=5)
    assert (a.stats == b.stats).all() and (a.score == b.score).all()


def test_possessions_track_pace(game, constants):
    _, res = run(game, constants)
    per48 = res.poss.mean(axis=1) * 2880 / (2880 + 300 * res.n_ot)
    assert abs(per48.mean() - game.pace) < 3.0


def test_minutes_calibration_honours_projections(game, constants):
    sim = GameSimulator(copy.deepcopy(game), SimParams(), constants)
    sim.calibrate(n_pilot=600)
    res = sim.run(3000, seed=2)
    m = res.secs.mean(axis=0) / 60
    for t in (0, 1):
        for i, p in enumerate(res.players[t]):
            if p.rotation:
                assert abs(m[t, i] - sim.min_mean[t, i]) < 2.0, (p.name, m[t, i], sim.min_mean[t, i])


def test_availability_probability_gives_mixture(game, constants):
    g = copy.deepcopy(game)
    g.home.players[1].p_play = 0.5
    _, res = run(g, constants, n=4000)
    played = (res.secs[:, 0, 1] > 0).mean()
    assert 0.45 < played < 0.55


def test_removing_a_star_redistributes_minutes_and_shots(game, constants):
    _, base = run(game, constants, n=4000, seed=3)
    g2 = apply(game, Scenario(out=[100]))
    assert all(p.person_id != 100 for p in g2.home.players)
    _, scen = run(g2, constants, n=4000, seed=3)
    assert np.allclose(scen.secs[:, 0].sum(axis=1), 5 * (2880 + 300 * scen.n_ot))
    fga = lambda r, t, i: (r.stats[:, t, i, S["fga2"]] + r.stats[:, t, i, S["fga3"]]).mean()
    # the second option (index 1 in base, index 0 after removal) shoots more without the star
    assert fga(scen, 0, 0) > fga(base, 0, 1) + 1.0
    # home team scores less without its best player
    assert scen.score[:, 0].mean() < base.score[:, 0].mean()


def test_usage_multiplier_increases_attempts(game, constants):
    _, base = run(game, constants, n=3000, seed=4)
    g2 = apply(game, Scenario(usage={102: 1.5}))
    _, scen = run(g2, constants, n=3000, seed=4)
    f = lambda r: (r.stats[:, 0, 2, S["fga2"]] + r.stats[:, 0, 2, S["fga3"]]).mean()
    assert f(scen) > f(base) * 1.2


def test_minutes_cap_is_respected(game, constants):
    g2 = apply(game, Scenario(caps={100: 20.0}))
    sim = GameSimulator(g2, SimParams(), constants)
    sim.calibrate(n_pilot=400)
    res = sim.run(2000, seed=6)
    assert (res.secs[:, 0, 0] / 60).mean() < 22.0


def test_overtime_occurs_and_breaks_ties(game, constants):
    _, res = run(game, constants, n=6000)
    assert 0.01 < (res.n_ot > 0).mean() < 0.12
    assert (res.score[:, 0] != res.score[:, 1]).all()


def test_blowouts_reduce_starter_minutes(game, constants):
    g = copy.deepcopy(game)
    for p in g.away.players:            # make the away team much worse
        p.rates["fg2"], p.rates["fg3"] = 0.42, 0.26
    _, res = run(g, constants, n=4000)
    margin = res.score[:, 0] - res.score[:, 1]
    star_min = res.secs[:, 0, 0] / 60
    big = margin >= 25
    close = np.abs(margin) <= 5
    assert big.sum() > 50 and close.sum() > 50
    assert star_min[big].mean() < star_min[close].mean() - 2.0


def test_a_sitting_questionable_player_gives_the_deep_bench_a_share(game, constants):
    """A rotation player with p_play < 1 who sits leaves part of his minutes to fringe players (fringe_take); with
    every rotation player certain to play, fringe_take changes nothing - not even the random numbers."""
    _, a = run(game, constants, n=1500, seed=4, fringe_take=0.0)
    _, b = run(game, constants, n=1500, seed=4, fringe_take=0.25)
    assert (a.stats == b.stats).all() and (a.secs == b.secs).all()
    g = copy.deepcopy(game)
    g.home.players[1].p_play = 0.5                         # a questionable 34-minute player
    fringe = [i for i, p in enumerate(g.home.players) if not p.rotation]
    out = {}
    for take in (0.0, 0.25):
        _, r = run(g, constants, n=4000, seed=4, fringe_take=take)
        sat = r.secs[:, 0, 1] == 0
        out[take] = (r.secs[:, 0, fringe].sum(axis=1)[sat].mean() / 60, r.secs[:, 0, fringe].sum(axis=1)[~sat].mean() / 60)
    assert out[0.25][0] > out[0.0][0] + 2.0                # more deep-bench minutes on the nights he sits ...
    assert abs(out[0.25][1] - out[0.0][1]) < 0.5           # ... and the same on the nights he plays
