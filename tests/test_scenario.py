"""Scenario engine: usage-efficiency tradeoff and environment reuse."""

from __future__ import annotations

import copy

from quantlab.sim.engine import GameSimulator, SimParams
from quantlab.sim.scenario import USAGE_EFFICIENCY_LOGIT, Scenario, apply, expected_shares, run_pair


def test_removing_a_star_raises_teammates_load_and_lowers_their_efficiency(game):
    before = expected_shares(game.home)
    new = apply(game, Scenario(out=[100]))
    after = expected_shares(new.home)
    for p_old, p_new in zip(game.home.players[1:], new.home.players):
        assert p_new.person_id == p_old.person_id
        assert after[p_new.person_id] > before[p_old.person_id]          # bigger share of team plays
        assert p_new.rates["fg2"] < p_old.rates["fg2"]                   # ... at lower efficiency
        assert p_new.rates["fg3"] < p_old.rates["fg3"]
    assert USAGE_EFFICIENCY_LOGIT < 0
    # the opponent is untouched
    assert [p.rates for p in new.away.players] == [p.rates for p in game.away.players]


def test_no_change_means_no_efficiency_adjustment(game):
    new = apply(game, Scenario(minutes={101: game.home.players[1].min_mean}))
    assert [p.rates for p in new.home.players] == [p.rates for p in game.home.players]


def test_star_removal_lowers_his_teams_scoring(game, constants, monkeypatch):
    import quantlab.sim.engine as eng
    monkeypatch.setattr(eng, "load_constants", lambda: constants)
    r0, r1 = run_pair(copy.deepcopy(game), Scenario(out=[100]), n=6000, seed=4)
    assert r1.score[:, 0].mean() < r0.score[:, 0].mean()


def test_scenario_reuses_baseline_environment(game, constants, monkeypatch):
    import quantlab.sim.engine as eng
    monkeypatch.setattr(eng, "load_constants", lambda: constants)
    g = copy.deepcopy(game)
    g.home.target_ppp, g.away.target_ppp = 1.25, 1.05
    base = GameSimulator(copy.deepcopy(g), SimParams(), constants)
    base.calibrate(environment="team", n_pilot=400)
    r0, r1 = run_pair(g, Scenario(out=[100]), n=500, seed=1, environment="team")
    # the scenario spec carries the baseline's (non-zero) environment shift
    assert abs(r1.game.home.eff_shift) > 0 and abs(r1.game.home.eff_shift - r0.game.home.eff_shift) < 0.15
