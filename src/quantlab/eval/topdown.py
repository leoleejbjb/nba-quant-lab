"""Team-first reconciliation baseline (the alternative architecture to the possession simulator).

1. Team scores come from the team model: a bivariate normal for (home, away) points built from the
   expected scores, margin/total standard deviations and their correlation (walk-forward).
2. The roster and its participation are the simulator's pregame roster (``sim.inputs.game_spec`` and the
   simulator's minutes budget): rotation players with their projected minutes, fringe candidates with
   their pregame participation probabilities - who appears is *sampled* in every simulated game, never
   taken from the box score. Players are projected *independently* per minute (closed-form
   empirical-Bayes rates, the opponent's defensive factors and the expected pace).
3. Each simulated team score is decomposed into made threes, free throws and made twos
   (2*FG2M + 3*FG3M + FTM = team points, with parity fixed through FTM), and every team-level count
   (made shots, misses, rebounds, assists <= FGM, steals, blocks, turnovers, minutes = 240) is allocated
   to that game's participants multinomially, with shares proportional to their per-minute projections x
   their simulated minutes.

The result is internally consistent box scores that *match the team model exactly*, but without
lineups, rotations, matchups or game-state dynamics - the comparison isolates what the
possession-level simulation adds. Players are scored conditional on appearing, like the simulator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PATHS
from .backtest import continuous_scores, discrete_scores

STATS = ["min", "pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "fga", "fta", "pra"]
E_COLS = ["e_fgm2", "e_fgm3", "e_ftm", "e_fta", "e_miss2", "e_miss3", "e_reb", "e_ast", "e_stl", "e_blk", "e_tov"]


def _bvn_cov(sm, st, rho):
    c_tm = rho * st * sm
    vh = (st ** 2 + sm ** 2 + 2 * c_tm) / 4
    va = (st ** 2 + sm ** 2 - 2 * c_tm) / 4
    cov = (st ** 2 - sm ** 2) / 4
    return np.array([[vh, cov], [cov, va]])


def per_minute_expectations(rates: dict, opp_def: dict, pace: float) -> dict:
    """Closed-form EB expectations per minute on court (same algebra as ``baselines.eb_expectations``)."""
    f = lambda k: float(opp_def.get(k, 1.0))
    poss = pace / 48.0
    used = rates["usage"] * poss * 1.13
    tov = used * rates["tov_share"] * f("tov_forced")
    non = used - tov
    fga3 = non * min(max(rates["fga3_share"] * f("fga3_share_allowed"), 0.0), 0.95)
    trips = non * min(max(rates["trip_share"] * f("trips_allowed"), 0.0), 0.6)
    fga2 = max(non - fga3 - trips, 0.0)
    fgm2 = fga2 * min(rates["fg2"] * f("fg2_allowed"), 0.95)
    fgm3 = fga3 * min(rates["fg3"] * f("fg3_allowed"), 0.95)
    fta = trips * (2 + rates["trip3_share"]) + (fgm2 + fgm3) * rates["and1"]
    return {"e_fgm2": fgm2, "e_fgm3": fgm3, "e_ftm": fta * rates["ft"], "e_fta": fta, "e_miss2": fga2 - fgm2,
            "e_miss3": fga3 - fgm3, "e_reb": (rates["oreb"] * f("oreb_allowed") + rates["dreb"]) * poss * 0.53,
            "e_ast": rates["ast_rate"] * max(poss * 0.42 - fgm2 - fgm3, 0.0) * f("ast_allowed"),
            "e_stl": rates["stl"] * poss, "e_blk": rates["blk"] * poss * 0.88, "e_tov": tov}


def roster_frame(spec, sim) -> pd.DataFrame:
    """Pregame roster with participation, minutes if playing and per-minute expectations, both teams."""
    rows = []
    for t, team in enumerate((spec.home, spec.away)):
        opp = spec.away if t == 0 else spec.home
        for i, pl in enumerate(team.players):
            p = float(sim.p_play[t, i])
            if p <= 0:
                continue
            cond = float(sim.min_mean[t, i] if pl.rotation else sim.min_mean[t, i] / max(p, 1e-6))
            rows.append({"team_side": t, "person_id": pl.person_id, "rotation": bool(pl.rotation), "p_play": p,
                         "cond_min": cond, **per_minute_expectations(pl.rates, opp.def_factors, spec.pace)})
    return pd.DataFrame(rows)


def simulate_game(team_pred: pd.Series, players: pd.DataFrame, n: int, rng, fringe_sd: float = 0.7) -> dict:
    """players: ``roster_frame`` rows (team_side 0 home / 1 away, p_play, cond_min, per-minute e_* columns)."""
    mu = np.array([team_pred["exp_home_pts"], team_pred["exp_away_pts"]])
    cov = _bvn_cov(team_pred["sigma_margin"], team_pred["sigma_total"], team_pred["rho_mt"])
    pts = np.rint(rng.multivariate_normal(mu, cov, size=n)).astype(int)
    pts = np.maximum(pts, 60)
    out = {}
    for side in (0, 1):
        P = players[players["team_side"] == side]
        k = len(P)
        if k == 0:
            continue
        cond = P["cond_min"].to_numpy()
        rot = P["rotation"].to_numpy()
        # participation (sampled), minutes when playing (rotation: normal noise; fringe: lognormal), sum = 240
        part = rng.random((n, k)) < P["p_play"].to_numpy()[None, :]
        noise_rot = cond[None, :] + rng.normal(0, 1, (n, k)) * np.clip(0.18 * cond, 2.5, 6.5)[None, :]
        noise_fr = cond[None, :] * np.exp(fringe_sd * rng.normal(0, 1, (n, k)) - 0.5 * fringe_sd ** 2)
        m = np.where(rot[None, :], noise_rot, noise_fr)
        m = np.clip(m, 0.5, 48) * part
        m = m * 240.0 / np.maximum(m.sum(axis=1, keepdims=True), 1)
        E = {c: P[c].to_numpy() for c in E_COLS}
        tot = {c: (E[c][None, :] * m).sum(axis=1) for c in E_COLS}          # team expectation in this game

        def shares(col):
            w = np.maximum(E[col][None, :], 1e-9) * m
            return w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        tp = pts[:, side]
        scale = tp / np.maximum(2 * tot["e_fgm2"] + 3 * tot["e_fgm3"] + tot["e_ftm"], 1.0)
        fg3m = rng.poisson(tot["e_fgm3"] * scale)
        ftm = rng.poisson(tot["e_ftm"] * scale)
        rem = tp - 3 * fg3m - ftm
        ftm = np.where(rem % 2 == 1, ftm + 1, ftm)         # parity through free throws
        rem = tp - 3 * fg3m - ftm
        bad = rem < 0
        fg3m = np.where(bad, np.maximum((tp - ftm) // 3, 0), fg3m)
        rem = tp - 3 * fg3m - ftm
        ftm = np.where(rem % 2 == 1, ftm + 1, ftm)
        rem = np.maximum(tp - 3 * fg3m - ftm, 0)
        fg2m = rem // 2
        ftm = tp - 3 * fg3m - 2 * fg2m
        alloc = lambda total, col: rng.multinomial(np.asarray(total, dtype=np.int64), shares(col))
        p_fg3m, p_fg2m, p_ftm = alloc(fg3m, "e_fgm3"), alloc(fg2m, "e_fgm2"), alloc(ftm, "e_ftm")
        ft_pct = tot["e_ftm"] / np.maximum(tot["e_fta"], 1e-6)
        p_fta = p_ftm + alloc(rng.poisson(tot["e_fta"] * (1 - ft_pct)), "e_fta")
        p_miss3 = alloc(rng.poisson(tot["e_miss3"]), "e_miss3")
        p_miss2 = alloc(rng.poisson(tot["e_miss2"]), "e_miss2")
        fgm = fg3m + fg2m
        ast_rate = np.minimum(tot["e_ast"] / np.maximum(tot["e_fgm2"] + tot["e_fgm3"], 1e-6), 0.95)
        p_ast = alloc(rng.binomial(fgm, ast_rate), "e_ast")
        res = {"min": m, "pts": 3 * p_fg3m + 2 * p_fg2m + p_ftm, "fg3m": p_fg3m,
               "fga": p_fg3m + p_fg2m + p_miss3 + p_miss2, "fta": p_fta, "ast": p_ast,
               "reb": alloc(rng.poisson(tot["e_reb"]), "e_reb"),
               "stl": alloc(rng.poisson(tot["e_stl"]), "e_stl"),
               "blk": alloc(rng.poisson(tot["e_blk"]), "e_blk"),
               "tov": alloc(rng.poisson(tot["e_tov"]), "e_tov")}
        res["pra"] = res["pts"] + res["reb"] + res["ast"]
        assert (res["pts"].sum(axis=1) == tp).all()
        assert ((res["pts"] == 0) | (m > 0)).all()          # only participants score
        out[side] = (P["person_id"].to_numpy(), res, m > 0)
    return out


def run_topdown(seasons, n: int = 1200, log=print) -> pd.DataFrame:
    """``n`` draws per game - the simulation backtests' count, so sample-based CRPS carries the same finite-draw
    penalty for both."""
    from ..sim.engine import GameSimulator, load_constants
    from ..sim.inputs import InputStore, game_spec
    store = InputStore()
    constants = load_constants()
    tp = pd.read_parquet(PATHS.models / "team_predictions.parquet").set_index("game_id")
    pg = pd.read_parquet(PATHS.processed("all") / "player_games.parquet",
                         columns=["game_id", "person_id", "played", "min", "pts", "reb", "ast", "fg3m", "stl", "blk",
                                  "tov", "fga", "fta", "pra"])
    act = {g: d.set_index("person_id") for g, d in pg[pg["played"]].groupby("game_id")}
    ids = store.games[store.games["season"].isin(list(seasons))].sort_values("game_date").index.tolist()
    rng = np.random.default_rng(2024)
    rows = []
    for j, gid in enumerate(ids):
        if gid not in tp.index or gid not in act:
            continue
        spec = game_spec(store, int(gid))
        sim = GameSimulator(spec, constants=constants)        # pregame roster + minutes budget (not run)
        P = roster_frame(spec, sim)
        sims = simulate_game(tp.loc[gid], P, n, rng)
        a = act[gid]
        for side, (pids, res, part) in sims.items():
            for k, pid in enumerate(pids):
                if pid not in a.index or part[:, k].sum() < 20:
                    continue
                on = part[:, k]
                for st in STATS:
                    x = res[st][on, k]
                    y = float(a.loc[pid, st])
                    sc = continuous_scores(x, y, rng) if st == "min" else discrete_scores(x, y, rng)
                    rows.append({"game_id": gid, "person_id": pid, "stat": st, "actual": y,
                                 "p_play_sim": float(on.mean()), **sc})
        if (j + 1) % 500 == 0:
            log(f"[topdown] {j + 1} games")
    out = pd.DataFrame(rows)
    PATHS.ensure(PATHS.backtest)
    out.to_parquet(PATHS.backtest / f"topdown_{'_'.join(seasons)}.parquet", index=False)
    return out
