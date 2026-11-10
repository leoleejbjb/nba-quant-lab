"""Prop probability math: pushes on whole-number lines, joint vs independent probabilities."""

from __future__ import annotations

import copy

import numpy as np

from quantlab.sim import props
from quantlab.sim.engine import GameSimulator, SimParams


def test_half_point_line_has_no_push():
    x = np.array([20, 24, 25, 25, 30])
    lp = props.line_probs(x, 24.5)
    assert lp["push"] == 0 and np.isclose(lp["over"], 0.6) and np.isclose(lp["under"], 0.4)


def test_whole_number_line_push():
    x = np.array([20, 24, 25, 25, 30])
    lp = props.line_probs(x, 25)
    assert np.isclose(lp["push"], 0.4) and np.isclose(lp["over"], 0.2) and np.isclose(lp["under"], 0.4)
    assert np.isclose(lp["over"] + lp["under"] + lp["push"], 1.0)


def test_fair_odds():
    assert np.isclose(props.american(0.5), -100)
    assert np.isclose(props.american(0.25), 300)


def test_joint_probability_uses_joint_draws(game, constants):
    sim = GameSimulator(copy.deepcopy(game), SimParams(), constants)
    res = sim.run(4000, seed=9)
    star = res.players[0][0].person_id
    pts = props.samples(res, star, "pts")
    ast = props.samples(res, star, "ast")
    legs = [props.Leg(star, "pts", "over", float(np.median(pts))), props.Leg(star, "ast", "over", float(np.median(ast)))]
    jp = props.joint_probability(res, legs)
    # same-player points and assists are positively correlated (minutes, activity night)
    assert jp["joint"] > jp["independent_product"]
    t, i = props.find_player(res, star)
    manual = ((res.stat("pts")[:, t, i] > legs[0].line) & (res.stat("ast")[:, t, i] > legs[1].line))
    manual = manual[res.secs[:, t, i] > 0].mean()
    assert np.isclose(jp["joint"], manual)


def test_double_double_definition(game, constants):
    sim = GameSimulator(copy.deepcopy(game), SimParams(), constants)
    res = sim.run(2000, seed=3)
    pid = res.players[0][0].person_id
    t, i = props.find_player(res, pid)
    cats = np.stack([res.stat(c)[:, t, i] for c in ("pts", "reb", "ast", "stl", "blk")], 1)
    exp = ((cats >= 10).sum(1) >= 2)[res.secs[:, t, i] > 0].mean()
    assert np.isclose(props.double_double(res, pid), exp)
    assert props.triple_double(res, pid) <= props.double_double(res, pid)
