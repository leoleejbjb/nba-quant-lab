"""Team-first reconciliation baseline (the alternative architecture to the possession simulator).

1. Team scores come from the team model: a bivariate normal for (home, away) points built from the
   expected scores, margin/total standard deviations and their correlation (walk-forward).
2. Players are projected *independently* (closed-form empirical-Bayes expectations x projected minutes).
3. Each simulated team score is decomposed into made threes, free throws and made twos
   (2*FG2M + 3*FG3M + FTM = team points, with parity fixed through FTM), and every team-level count
   (made shots, misses, rebounds, assists <= FGM, steals, blocks, turnovers, minutes = 240) is allocated
   to players multinomially with shares proportional to their independent projections x their
   simulated minutes.

The result is internally consistent box scores that *match the team model exactly*, but without
lineups, rotations, matchups or game-state dynamics - the comparison isolates what the
possession-level simulation adds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PATHS
from .backtest import continuous_scores, discrete_scores

STATS = ["min", "pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "fga", "fta", "pra"]


def _bvn_cov(sm, st, rho):
    c_tm = rho * st * sm
    vh = (st ** 2 + sm ** 2 + 2 * c_tm) / 4
    va = (st ** 2 + sm ** 2 - 2 * c_tm) / 4
    cov = (st ** 2 - sm ** 2) / 4
    return np.array([[vh, cov], [cov, va]])


def simulate_game(team_pred: pd.Series, players: pd.DataFrame, n: int, rng) -> dict:
    """players: rows for both teams with columns team_side (0 home / 1 away), proj_min, e_fgm2, e_fgm3,
    e_ftm, e_fta, e_miss2, e_miss3, e_reb, e_ast, e_stl, e_blk, e_tov."""
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
        # minutes: normal around projections, normalized to 240
        m = np.clip(P["proj_min"].to_numpy()[None, :] + rng.normal(0, 1, (n, k)) *
                    np.clip(0.18 * P["proj_min"].to_numpy(), 2.5, 6.5)[None, :], 0, 48)
        m = m * 240.0 / np.maximum(m.sum(axis=1, keepdims=True), 1)
        ratio = m / np.maximum(P["proj_min"].to_numpy()[None, :], 0.5)
        def shares(col):
            w = np.maximum(P[col].to_numpy()[None, :], 1e-6) * ratio
            return w / w.sum(axis=1, keepdims=True)
        tp = pts[:, side]
        exp_pts = mu[side]
        scale = tp / max(exp_pts, 1)
        fg3m = rng.poisson(P["e_fgm3"].sum() * scale)
        ftm = rng.poisson(P["e_ftm"].sum() * scale)
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
        alloc = lambda total, col: _multinomial_rows(rng, total, shares(col))
        p_fg3m, p_fg2m, p_ftm = alloc(fg3m, "e_fgm3"), alloc(fg2m, "e_fgm2"), alloc(ftm, "e_ftm")
        ft_pct = (P["e_ftm"].sum() / max(P["e_fta"].sum(), 1e-6))
        p_fta = p_ftm + _multinomial_rows(rng, rng.poisson(P["e_fta"].sum() * (1 - ft_pct), size=n), shares("e_fta"))
        p_miss3 = alloc(rng.poisson(P["e_miss3"].sum(), size=n), "e_miss3")
        p_miss2 = alloc(rng.poisson(P["e_miss2"].sum(), size=n), "e_miss2")
        fgm = fg3m + fg2m
        ast_rate = P["e_ast"].sum() / max(P["e_fgm2"].sum() + P["e_fgm3"].sum(), 1e-6)
        p_ast = alloc(rng.binomial(fgm, min(ast_rate, 0.95)), "e_ast")
        res = {"min": m, "pts": 3 * p_fg3m + 2 * p_fg2m + p_ftm, "fg3m": p_fg3m,
               "fga": p_fg3m + p_fg2m + p_miss3 + p_miss2, "fta": p_fta, "ast": p_ast,
               "reb": alloc(rng.poisson(P["e_reb"].sum(), size=n), "e_reb"),
               "stl": alloc(rng.poisson(P["e_stl"].sum(), size=n), "e_stl"),
               "blk": alloc(rng.poisson(P["e_blk"].sum(), size=n), "e_blk"),
               "tov": alloc(rng.poisson(P["e_tov"].sum(), size=n), "e_tov")}
        res["pra"] = res["pts"] + res["reb"] + res["ast"]
        assert (res["pts"].sum(axis=1) == tp).all()
        out[side] = (P["person_id"].to_numpy(), res)
    return out


def _multinomial_rows(rng, totals, probs):
    totals = np.asarray(totals, dtype=np.int64)
    out = np.zeros(probs.shape, dtype=np.int64)
    for j in range(len(totals)):
        out[j] = rng.multinomial(int(totals[j]), probs[j])
    return out


def player_expectations(season_list) -> pd.DataFrame:
    """Independent per-player expectations (EB closed form) needed by the allocation."""
    from .baselines import add_model_inputs, load_frame
    df = load_frame()
    d = add_model_inputs(df[df["played"] & df["season"].isin(season_list)].copy()).reset_index(drop=True)
    m = d["pred_gbm"].clip(lower=0)
    poss_on = m / 48.0 * d["pace"].fillna(99)
    used = d["r_usage"] * poss_on * 1.13
    tov = used * d["r_tov_share"] * d["opp_def_tov_forced"].fillna(1)
    non = used - tov
    fga3 = non * (d["r_fga3_share"] * d["opp_def_fga3_share_allowed"].fillna(1)).clip(0, 0.95)
    trips = non * (d["r_trip_share"] * d["opp_def_trips_allowed"].fillna(1)).clip(0, 0.6)
    fga2 = (non - fga3 - trips).clip(lower=0)
    fgm2 = fga2 * d["r_fg2"]
    fgm3 = fga3 * d["r_fg3"]
    fta = trips * (2 + d["r_trip3_share"]) + (fgm2 + fgm3) * d["r_and1"]
    e = pd.DataFrame({"game_id": d["game_id"], "person_id": d["person_id"], "team_id": d["team_id"],
                      "season": d["season"], "rotation": d["rotation"], "proj_min": m,
                      "e_fgm2": fgm2, "e_fgm3": fgm3, "e_ftm": fta * d["r_ft"], "e_fta": fta,
                      "e_miss2": fga2 - fgm2, "e_miss3": fga3 - fgm3,
                      "e_reb": (d["r_oreb"] + d["r_dreb"]) * poss_on * 0.53,
                      "e_ast": d["r_ast_rate"] * (poss_on * 0.42 - fgm2 - fgm3).clip(lower=0),
                      "e_stl": d["r_stl"] * poss_on, "e_blk": d["r_blk"] * poss_on * 0.88, "e_tov": tov})
    return e


def run_topdown(seasons, n: int = 1500, log=print) -> pd.DataFrame:
    tp = pd.read_parquet(PATHS.models / "team_predictions.parquet").set_index("game_id")
    pg = pd.read_parquet(PATHS.processed("all") / "player_games.parquet",
                         columns=["game_id", "person_id", "played", "min", "pts", "reb", "ast", "fg3m", "stl", "blk",
                                  "tov", "fga", "fta", "pra"])
    act = {g: d.set_index("person_id") for g, d in pg[pg["played"]].groupby("game_id")}
    E = player_expectations(list(seasons))
    rng = np.random.default_rng(2024)
    rows = []
    for j, (gid, P) in enumerate(E.groupby("game_id")):
        t = tp.loc[gid]
        P = P.assign(team_side=np.where(P["team_id"] == t["home_team_id"], 0, 1))
        sims = simulate_game(t, P, n, rng)
        a = act[gid]
        for side, (pids, res) in sims.items():
            for k, pid in enumerate(pids):
                if pid not in a.index:
                    continue
                for st in STATS:
                    x = res[st][:, k]
                    y = float(a.loc[pid, st])
                    sc = continuous_scores(x, y, rng) if st == "min" else discrete_scores(x, y, rng)
                    rows.append({"game_id": gid, "person_id": pid, "stat": st, "actual": y, **sc})
        if (j + 1) % 500 == 0:
            log(f"[topdown] {j + 1} games")
    out = pd.DataFrame(rows)
    PATHS.ensure(PATHS.backtest)
    out.to_parquet(PATHS.backtest / f"topdown_{'_'.join(seasons)}.parquet", index=False)
    return out
