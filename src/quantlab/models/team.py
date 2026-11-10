"""Team-level game model: walk-forward, time-decayed, opponent-adjusted ratings.

For a game on date d, ratings are fit on all earlier games with weights
``2^{-(t_d - t_j)/H}`` on the effective-time axis (offseasons compressed):

* Efficiency (points per 100 possessions of the offense)::

      ppp = mu + o[offense] + dfn[defense] + eta * home + beta_o * b2b_off + beta_d * b2b_def

  weighted by possessions; ridge penalty ``lam`` on the team effects only.
* Pace (possessions per 48 minutes, mean of both teams)::

      pace = pi + p[home] + p[away]

Prediction: ``pts = ppp/100 * pace * L`` where ``L`` is the average game length
relative to 48 minutes (overtime), margin and total follow, and score
uncertainty is a bivariate normal whose covariance is estimated from
walk-forward residuals of earlier seasons. The normal equations are maintained
incrementally (the decay factorizes), so each date costs O(params^2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..config import PATHS
from ..features.asof import effective_time


@dataclass
class TeamModelConfig:
    halflife: float = 50.0      # effective days (tuned on 2023-24)
    lam: float = 2.0            # ridge on team effects (possession-weighted game units; tuned on 2023-24)
    pace_halflife: float = 12.0
    pace_lam: float = 1.0


def _team_index(tg: pd.DataFrame) -> dict:
    return {t: i for i, t in enumerate(sorted(tg["team_id"].unique()))}


def walk_forward(tg: pd.DataFrame, games: pd.DataFrame, cfg: TeamModelConfig) -> pd.DataFrame:
    """Pregame predictions for every game (home perspective) using only earlier dates."""
    ti = _team_index(tg)
    nt = len(ti)
    g = games[["game_id", "game_date", "season", "home_team_id", "away_team_id", "game_len_s"]].copy()
    g["t"] = effective_time(g["game_date"], games).to_numpy()
    # per-team rest from team_games
    rest = tg.set_index(["game_id", "team_id"])["b2b"].astype(float)
    g["b2b_home"] = [rest.get((a, b), 0.0) for a, b in zip(g["game_id"], g["home_team_id"])]
    g["b2b_away"] = [rest.get((a, b), 0.0) for a, b in zip(g["game_id"], g["away_team_id"])]
    pts = tg.set_index(["game_id", "team_id"])[["pts", "poss"]]
    hp = pts.reindex(list(zip(g["game_id"], g["home_team_id"]))).to_numpy()
    ap = pts.reindex(list(zip(g["game_id"], g["away_team_id"]))).to_numpy()
    g["home_pts"], g["home_poss"], g["away_pts"], g["away_poss"] = hp[:, 0], hp[:, 1], ap[:, 0], ap[:, 1]
    g = g.sort_values(["t", "game_id"]).reset_index(drop=True)

    # efficiency design: [mu, eta, beta_o, beta_d, o(nt), dfn(nt)]
    k_e = 4 + 2 * nt
    k_p = 1 + nt
    lam_e, lam_p = np.log(2) / cfg.halflife, np.log(2) / cfg.pace_halflife
    A_e = np.zeros((k_e, k_e)); b_e = np.zeros(k_e); W_e = 0.0
    A_p = np.zeros((k_p, k_p)); b_p = np.zeros(k_p); W_p = 0.0
    pen_e = np.zeros(k_e); pen_e[4:] = 1.0
    pen_p = np.zeros(k_p); pen_p[1:] = 1.0
    t_ref = g["t"].min()
    out = []
    len_hist = []
    for t, day in g.groupby("t", sort=True):
        # ---- predict (state contains only earlier dates) ----
        se = np.exp(-lam_e * (t - t_ref)); sp = np.exp(-lam_p * (t - t_ref))
        if W_e > 0:
            coef_e = np.linalg.solve(A_e * se + cfg.lam * np.diag(pen_e) + 1e-9 * np.eye(k_e), b_e * se)
            coef_p = np.linalg.solve(A_p * sp + cfg.pace_lam * np.diag(pen_p) + 1e-9 * np.eye(k_p), b_p * sp)
        else:
            coef_e = np.r_[112.0, 1.5, 0, 0, np.zeros(2 * nt)]
            coef_p = np.r_[49.5, np.zeros(nt)]
        L = np.mean(len_hist[-2000:]) if len_hist else 1.006
        for r in day.itertuples(index=False):
            h, a = ti[r.home_team_id], ti[r.away_team_id]
            ppp_h = coef_e[0] + coef_e[1] + coef_e[2] * r.b2b_home + coef_e[3] * r.b2b_away + coef_e[4 + h] + coef_e[4 + nt + a]
            ppp_a = coef_e[0] + coef_e[2] * r.b2b_away + coef_e[3] * r.b2b_home + coef_e[4 + a] + coef_e[4 + nt + h]
            pace = coef_p[0] + coef_p[1 + h] + coef_p[1 + a]
            out.append((r.game_id, ppp_h, ppp_a, pace, L))
        # ---- update with today's results ----
        we = np.exp(lam_e * (t - t_ref)); wp = np.exp(lam_p * (t - t_ref))
        for r in day.itertuples(index=False):
            if not np.isfinite(r.home_pts) or not np.isfinite(r.away_pts):
                continue
            h, a = ti[r.home_team_id], ti[r.away_team_id]
            gl = r.game_len_s / 2880.0
            len_hist.append(gl)
            for off, dfn, home, b_o, b_d, pts_, poss_ in ((h, a, 1.0, r.b2b_home, r.b2b_away, r.home_pts, r.home_poss),
                                                          (a, h, 0.0, r.b2b_away, r.b2b_home, r.away_pts, r.away_poss)):
                x = np.zeros(k_e)
                x[0], x[1], x[2], x[3] = 1.0, home, b_o, b_d
                x[4 + off] = 1.0
                x[4 + nt + dfn] = 1.0
                y = 100.0 * pts_ / poss_
                w = we * poss_ / 100.0
                A_e += w * np.outer(x, x)
                b_e += w * x * y
                W_e += w
            xp = np.zeros(k_p)
            xp[0], xp[1 + h], xp[1 + a] = 1.0, 1.0, 1.0
            yp = 0.5 * (r.home_poss + r.away_poss) / gl
            A_p += wp * np.outer(xp, xp)
            b_p += wp * xp * yp
            W_p += wp
    pr = pd.DataFrame(out, columns=["game_id", "ppp_home", "ppp_away", "pace", "len_factor"])
    pr = g.merge(pr, on="game_id")
    pr["exp_home_pts"] = pr["ppp_home"] / 100 * pr["pace"] * pr["len_factor"]
    pr["exp_away_pts"] = pr["ppp_away"] / 100 * pr["pace"] * pr["len_factor"]
    pr["exp_margin"] = pr["exp_home_pts"] - pr["exp_away_pts"]
    pr["exp_total"] = pr["exp_home_pts"] + pr["exp_away_pts"]
    pr["margin"] = pr["home_pts"] - pr["away_pts"]
    pr["total"] = pr["home_pts"] + pr["away_pts"]
    return pr


def attach_uncertainty(pr: pd.DataFrame) -> pd.DataFrame:
    """Score covariance per season from residuals of *earlier* seasons (first season: league defaults)."""
    pr = pr.copy()
    seasons = sorted(pr["season"].unique())
    sm, st, rho = np.full(len(pr), 12.5), np.full(len(pr), 18.5), np.full(len(pr), 0.0)
    for i, s in enumerate(seasons):
        prior = pr[pr["season"].isin(seasons[max(0, i - 2):i])]
        prior = prior[prior.groupby("season")["game_date"].rank(pct=True) > 0.15]   # skip cold starts
        m = (pr["season"] == s).to_numpy()
        if len(prior) > 300:
            em = prior["margin"] - prior["exp_margin"]
            et = prior["total"] - prior["exp_total"]
            sm[m], st[m], rho[m] = em.std(), et.std(), np.corrcoef(em, et)[0, 1]
    pr["sigma_margin"], pr["sigma_total"], pr["rho_mt"] = sm, st, rho
    pr["p_home_win"] = norm.cdf(pr["exp_margin"] / pr["sigma_margin"])
    return pr


def evaluate(pr: pd.DataFrame, seasons) -> dict:
    d = pr[pr["season"].isin(seasons) & pr["margin"].notna()]
    em, et = d["margin"] - d["exp_margin"], d["total"] - d["exp_total"]
    y = (d["margin"] > 0).astype(float)
    p = d["p_home_win"].clip(1e-4, 1 - 1e-4)
    return {"games": int(len(d)), "margin_mae": float(em.abs().mean()), "margin_rmse": float(np.sqrt((em ** 2).mean())),
            "total_mae": float(et.abs().mean()), "total_rmse": float(np.sqrt((et ** 2).mean())),
            "total_bias": float(et.mean()), "margin_bias": float(em.mean()),
            "win_logloss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
            "win_brier": float(((p - y) ** 2).mean()), "win_accuracy": float(((p > 0.5) == (y == 1)).mean())}


def baseline_predictions(tg: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Naive baselines: (a) home edge only; (b) season-to-date net rating + pace, no opponent adjustment."""
    t = tg.sort_values(["game_date", "game_id"]).copy()
    grp = t.groupby(["team_id", "season"])
    for c in ("pts", "opp_pts", "poss"):
        t[f"cum_{c}"] = grp[c].cumsum() - t[c]
    t["n_prev"] = grp.cumcount()
    t["ortg"] = 100 * t["cum_pts"] / t["cum_poss"].replace(0, np.nan)
    t["drtg"] = 100 * t["cum_opp_pts"] / t["cum_poss"].replace(0, np.nan)
    t["pace_td"] = t["cum_poss"] / t["n_prev"].replace(0, np.nan)
    h = t[t["is_home"]].set_index("game_id")
    a = t[~t["is_home"]].set_index("game_id")
    b = pd.DataFrame(index=h.index)
    lg_pts = t.groupby("season")["pts"].transform("mean")
    b["season"] = h["season"]
    pace = (h["pace_td"].fillna(99) + a["pace_td"].fillna(99)) / 2
    b["naive_exp_home_pts"] = (h["ortg"].fillna(112) + a["drtg"].fillna(112)) / 200 * pace + 1.0
    b["naive_exp_away_pts"] = (a["ortg"].fillna(112) + h["drtg"].fillna(112)) / 200 * pace - 1.0
    del lg_pts
    return b.reset_index()


def run_team_model(cfg: TeamModelConfig | None = None, log=print) -> pd.DataFrame:
    cfg = cfg or TeamModelConfig()
    d = PATHS.processed("all")
    tg = pd.read_parquet(d / "team_games.parquet")
    games = pd.read_parquet(d / "games.parquet")
    pr = attach_uncertainty(walk_forward(tg, games, cfg))
    PATHS.ensure(PATHS.models)
    pr.to_parquet(PATHS.models / "team_predictions.parquet", index=False)
    res = {s: evaluate(pr, [s]) for s in sorted(pr["season"].unique())}
    (PATHS.models / "team_model_eval.json").write_text(json.dumps({"config": cfg.__dict__, "by_season": res}, indent=2))
    for s, r in res.items():
        log(f"[team] {s}: margin MAE {r['margin_mae']:.2f} RMSE {r['margin_rmse']:.2f} | total MAE {r['total_mae']:.2f} "
            f"RMSE {r['total_rmse']:.2f} bias {r['total_bias']:+.2f} | logloss {r['win_logloss']:.4f} acc {r['win_accuracy']:.3f}")
    return pr
