"""Player prop probabilities from the joint simulation.

Every probability is computed from the *joint* simulated games, never by multiplying marginals:
P(A and B) is the share of simulated games in which both happened.

Discrete statistics and pushes: for a line L,

* half-point line (24.5): over = P(X >= 25), under = P(X <= 24), push = 0;
* whole-number line (25): over = P(X > 25), under = P(X < 25), push = P(X = 25).

By default probabilities are conditional on the player appearing (the standard prop rule: a player
who does not play voids the bet); ``conditional=False`` treats non-appearance as zeros.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .engine import SimResult

DD_CATS = ("pts", "reb", "ast", "stl", "blk")


def find_player(res: SimResult, person_id: int) -> tuple[int, int]:
    for t in range(2):
        for i, pl in enumerate(res.players[t]):
            if pl.person_id == person_id:
                return t, i
    raise KeyError(f"player {person_id} not in this simulation")


def samples(res: SimResult, person_id: int, stat: str, conditional: bool = True) -> np.ndarray:
    t, i = find_player(res, person_id)
    x = res.stat(stat)[:, t, i]
    if conditional:
        x = x[res.secs[:, t, i] > 0]
    return x


def line_probs(x: np.ndarray, line: float) -> dict:
    x = np.asarray(x)
    if len(x) == 0:
        return {"over": np.nan, "under": np.nan, "push": np.nan, "n": 0}
    if float(line).is_integer():
        over, push = float((x > line).mean()), float((x == line).mean())
        under = float((x < line).mean())
    else:
        over = float((x > line).mean())
        under, push = 1.0 - over, 0.0
    return {"over": over, "under": under, "push": push, "n": int(len(x)),
            "fair_over_american": american(over / max(over + under, 1e-12)),
            "fair_under_american": american(under / max(over + under, 1e-12))}


def american(p: float) -> float:
    """Fair American odds for win probability p (ignoring pushes)."""
    if not 0 < p < 1:
        return np.nan
    return -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p


def double_double(res: SimResult, person_id: int, conditional: bool = True, k: int = 2) -> float:
    t, i = find_player(res, person_id)
    cats = np.stack([res.stat(c)[:, t, i] for c in DD_CATS], axis=1)
    hit = (cats >= 10).sum(axis=1) >= k
    if conditional:
        hit = hit[res.secs[:, t, i] > 0]
    return float(hit.mean()) if len(hit) else np.nan


def triple_double(res: SimResult, person_id: int, conditional: bool = True) -> float:
    return double_double(res, person_id, conditional, k=3)


@dataclass
class Leg:
    person_id: int
    stat: str
    side: str        # 'over' | 'under'
    line: float


def joint_probability(res: SimResult, legs: list[Leg], conditional: bool = True) -> dict:
    """P(all legs hit) from the joint draws, vs. the product of marginals (independence)."""
    hit = np.ones(res.n, dtype=bool)
    plays = np.ones(res.n, dtype=bool)
    marg = []
    for leg in legs:
        t, i = find_player(res, leg.person_id)
        x = res.stat(leg.stat)[:, t, i]
        h = x > leg.line if leg.side == "over" else x < leg.line
        p_ = res.secs[:, t, i] > 0
        plays &= p_
        hit &= h
        marg.append(float(h[p_].mean()) if conditional else float(h.mean()))
    base = plays if conditional else np.ones(res.n, dtype=bool)
    joint = float(hit[base].mean()) if base.any() else np.nan
    indep = float(np.prod(marg))
    return {"joint": joint, "independent_product": indep,
            "correlation_lift": joint / indep if indep > 0 else np.nan, "n": int(base.sum())}


def distribution_summary(res: SimResult, person_id: int, stat: str, conditional: bool = True) -> dict:
    x = samples(res, person_id, stat, conditional)
    if len(x) == 0:
        return {}
    q = np.percentile(x, [5, 10, 25, 50, 75, 90, 95])
    return {"mean": float(x.mean()), "sd": float(x.std()), "p05": q[0], "p10": q[1], "p25": q[2], "median": q[3],
            "p75": q[4], "p90": q[5], "p95": q[6], "n": int(len(x))}


def pmf(x: np.ndarray, max_k: int | None = None) -> np.ndarray:
    x = np.asarray(x).astype(int)
    m = int(max_k if max_k is not None else max(x.max(initial=0), 0))
    return np.bincount(np.clip(x, 0, m), minlength=m + 1) / max(len(x), 1)


def standard_lines(mean: float) -> list[float]:
    """Half-point lines around a projection (for prop ladders)."""
    c = np.floor(mean) + 0.5
    return [float(c + k) for k in range(-3, 4) if c + k > 0]
