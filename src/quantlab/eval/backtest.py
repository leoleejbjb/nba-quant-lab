"""Walk-forward simulation backtests and distributional scoring.

For every game, inputs are built from pregame information only (``sim.inputs``), the
simulator is calibrated (planned minutes; optional team environment), and ``n`` games are
simulated. Each player's simulated distribution is scored against his actual box score:

* point forecast: mean (MAE / RMSE computed later)
* CRPS of the discrete distribution, randomized PIT, log score, 50/80/90% interval coverage
* over/under: P(X > L) at the half-point line L closest to the projection (floor(mean) + 0.5) and at
  L - 2, L - 1, L + 1, L + 2 (lines below 0.5 are skipped), and the outcomes

Players are scored conditional on appearing (the standard prop convention: a player who does
not play voids the bet); fringe players' simulated distributions are conditioned on minutes > 0.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..config import PATHS
from ..sim.engine import GameSimulator, SimParams, load_constants
from ..sim.inputs import InputStore, game_spec

EVAL_STATS = ["min", "pts", "reb", "ast", "fg3m", "fga", "fta", "stl", "blk", "tov", "pra", "oreb", "dreb", "pf"]
TEAM_STATS = ["fga2", "fga3", "fgm2", "fgm3", "fta", "ftm", "tov", "oreb", "dreb", "ast", "stl", "blk", "pf"]
LINE_OFFSETS = (-2, -1, 1, 2)          # extra over/under lines: nearest half point -2 .. +2
ACTUAL_COLS = {"min": "min", "pts": "pts", "reb": "reb", "ast": "ast", "fg3m": "fg3m", "fga": "fga", "fta": "fta",
               "stl": "stl", "blk": "blk", "tov": "tov", "pra": "pra", "oreb": "oreb", "dreb": "dreb", "pf": "pf"}


def discrete_scores(samples: np.ndarray, y: float, rng) -> dict:
    """Scores of an integer-valued sample distribution against outcome y."""
    s = np.asarray(samples, dtype=np.int64)
    n = len(s)
    y = int(round(y))
    hi = int(max(s.max(initial=0), y)) + 2
    pmf = np.bincount(s, minlength=hi).astype(float) / n
    cdf = np.cumsum(pmf)
    k = np.arange(hi)
    crps = float(((cdf - (k >= y)) ** 2).sum())
    f_lo = cdf[y - 1] if y >= 1 else 0.0
    pit = float(f_lo + rng.random() * (cdf[y] - f_lo))
    logp = float(np.log((pmf[y] * n + 0.1) / (n + 0.1 * hi)))
    mean = s.mean()
    line = max(np.floor(mean) + 0.5, 0.5)          # the half-point line nearest the projection
    p_over = float((s > line).mean())
    q = np.quantile(s, [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    out = {"mean": float(mean), "sd": float(s.std()), "crps": crps, "pit": pit, "logp": logp,
           "q05": q[0], "q10": q[1], "q25": q[2], "q50": q[3], "q75": q[4], "q90": q[5], "q95": q[6],
           "line": float(line), "p_over": p_over, "over": float(y > line)}
    # a ladder of lines around the projection (props are offered away from the median too)
    for k in LINE_OFFSETS:
        L = np.floor(mean) + 0.5 + k
        ok = L >= 0.5
        out[f"p_over_{k:+d}"] = float((s > L).mean()) if ok else np.nan
        out[f"over_{k:+d}"] = float(y > L) if ok else np.nan
    return out


def continuous_scores(samples: np.ndarray, y: float, rng) -> dict:
    s = np.sort(np.asarray(samples, dtype=float))
    n = len(s)
    crps = float(np.abs(s - y).mean() - (2 * np.arange(1, n + 1) - n - 1).dot(s) / (n * n))
    pit = float((s < y).mean() + rng.random() * (s == y).mean())
    q = np.quantile(s, [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    out = {"mean": float(s.mean()), "sd": float(s.std()), "crps": crps, "pit": pit, "logp": np.nan,
           "q05": q[0], "q10": q[1], "q25": q[2], "q50": q[3], "q75": q[4], "q90": q[5], "q95": q[6],
           "line": np.nan, "p_over": np.nan, "over": np.nan}
    for k in LINE_OFFSETS:
        out[f"p_over_{k:+d}"] = np.nan
        out[f"over_{k:+d}"] = np.nan
    return out


def summarize_game(res, actual: pd.DataFrame, rng, game_id: int) -> tuple[list, dict]:
    """Per-player rows (one per player x stat) and the team-level row for one simulated game."""
    rows = []
    mins = res.stat("min")
    for t in range(2):
        for i, pl in enumerate(res.players[t]):
            if pl.person_id not in actual.index:
                continue
            a = actual.loc[pl.person_id]
            if not a["played"]:
                continue
            playing = mins[:, t, i] > 0
            p_play_sim = float(playing.mean())
            if playing.sum() < 20:
                continue
            for st in EVAL_STATS:
                x = res.stat(st)[playing, t, i]
                y = float(a[ACTUAL_COLS[st]])
                sc = continuous_scores(x, y, rng) if st == "min" else discrete_scores(x, y, rng)
                rows.append({"game_id": game_id, "person_id": pl.person_id,
                             "team_id": res.game.home.team_id if t == 0 else res.game.away.team_id,
                             "rotation": pl.rotation, "starter": pl.starter, "proj_min": pl.min_mean,
                             "p_play_sim": p_play_sim, "in_pool": pl.in_pool, "stat": st, "actual": y, **sc})
    sc = res.score.astype(float)
    margin, total = sc[:, 0] - sc[:, 1], sc.sum(axis=1)
    team = {"game_id": game_id, "sim_home_pts": sc[:, 0].mean(), "sim_away_pts": sc[:, 1].mean(),
            "sim_margin": margin.mean(), "sim_total": total.mean(), "sim_margin_sd": margin.std(),
            "sim_total_sd": total.std(), "sim_p_home_win": float((margin > 0).mean()),
            "sim_ot": float((res.n_ot > 0).mean()), "sim_poss": float(res.poss.mean()),
            "sim_reg_tie": float((res.marks[:, 1] == 0).mean()) if res.marks is not None else np.nan,
            "sim_home_pts_sd": sc[:, 0].std(), "sim_away_pts_sd": sc[:, 1].std()}
    for st in TEAM_STATS:
        v = res.stat(st).sum(axis=2).mean(axis=0)
        team[f"sim_home_{st}"], team[f"sim_away_{st}"] = float(v[0]), float(v[1])
    tr = res.team_reb.mean(axis=0)
    team["sim_home_team_reb"], team["sim_away_team_reb"] = float(tr[0].sum()), float(tr[1].sum())
    return rows, team, margin, total


def run_backtest(game_ids: list, n_sims: int = 1500, environment: str = "none", env_weight: float = 1.0,
                 params: SimParams | None = None, store: InputStore | None = None, tag: str = "bt",
                 n_pilot: int = 600, log=print, save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    store = store or InputStore()
    constants = load_constants()
    pg = pd.read_parquet(PATHS.processed("all") / "player_games.parquet",
                         columns=["game_id", "person_id", "played"] + sorted(set(ACTUAL_COLS.values())))
    pg = pg[pg["game_id"].isin(set(game_ids))]
    act_by_game = {g: d.set_index("person_id") for g, d in pg.groupby("game_id")}
    rng = np.random.default_rng(12345)
    prow, trow = [], []
    t0 = time.perf_counter()
    for j, gid in enumerate(game_ids):
        spec = game_spec(store, int(gid))
        sim = GameSimulator(spec, params, constants)
        info = sim.calibrate(n_pilot=n_pilot, environment=environment, env_weight=env_weight, seed=int(gid) % 9973)
        res = sim.run(n_sims, seed=int(gid) % 100003, track_lineups=False)
        inv = res.check_invariants()
        if not inv["all"]:                      # every simulated game must satisfy the box-score identities
            raise AssertionError(f"game {gid}: invariants violated {[k for k, v in inv.items() if not v]}")
        rows, team, margin, total = summarize_game(res, act_by_game[gid], rng, int(gid))
        g = store.games.loc[gid]
        y_m, y_t = float(g["home_pts"] - g["away_pts"]), float(g["home_pts"] + g["away_pts"])
        team.update({"margin": y_m, "total": y_t,
                     "sim_margin_crps": continuous_scores(margin, y_m, rng)["crps"],
                     "sim_total_crps": continuous_scores(total, y_t, rng)["crps"],
                     "sim_margin_pit": float((margin < y_m).mean()), "sim_total_pit": float((total < y_t).mean()),
                     "shift_home": spec.home.eff_shift, "shift_away": spec.away.eff_shift,
                     "season": g["season"], "game_date": g["game_date"]})
        prow.extend(rows)
        trow.append(team)
        if (j + 1) % 100 == 0:
            el = time.perf_counter() - t0
            log(f"[backtest {tag}] {j + 1}/{len(game_ids)} games, {el:.0f}s ({el / (j + 1):.2f}s/game)")
    P = pd.DataFrame(prow)
    T = pd.DataFrame(trow)
    if save:
        PATHS.ensure(PATHS.backtest)
        P.to_parquet(PATHS.backtest / f"{tag}_players.parquet", index=False)
        T.to_parquet(PATHS.backtest / f"{tag}_teams.parquet", index=False)
    return P, T
