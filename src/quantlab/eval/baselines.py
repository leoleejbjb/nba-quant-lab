"""Player-stat baselines and the count-model / gradient-boosting comparison.

All models predict a played player-game's stat from pregame information and share the same
projected minutes (the walk-forward minutes model, "lineups known"), so that differences reflect the
rate model, not the minutes model:

* ``season_avg``   season-to-date per-game average (previous season early on)
* ``l10``          last 10 games average
* ``per_min``      season-to-date per-minute rate x projected minutes
* ``opp_adj``      ``per_min`` x the opponent's stat-specific defensive factor
* ``eb``           closed-form expectation from the empirical-Bayes per-possession rates
                   (the simulator's inputs, without lineup interactions or game dynamics)
* ``gbm``          LightGBM Poisson model on EB rates, recent per-minute rates, projected minutes,
                   pace, opponent factors, teammate absences, rest, home (trained on earlier seasons)

Distributions for the non-simulation models: negative binomial around the point forecast with a
dispersion fitted on the training season (method of moments in mean bins).
"""

from __future__ import annotations

import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats as sps

from ..config import PATHS

STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "fga", "fta", "pra"]
GBM_PARAMS = dict(objective="poisson", learning_rate=0.03, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=11,
                  num_threads=int(os.environ.get("QUANTLAB_THREADS", "4")))
PLAYS_PER_POSS = 1.13


def load_frame() -> pd.DataFrame:
    allp = PATHS.processed("all")
    pg = pd.read_parquet(allp / "player_games.parquet")
    pg = pg.sort_values(["person_id", "game_date", "game_id"]).reset_index(drop=True)
    played = pg["played"]
    g_season = pg.groupby(["person_id", "season"])
    out = pg[["game_id", "person_id", "team_id", "opp_team_id", "season", "game_date", "played", "min", "starter",
              "is_home"] + [s for s in STATS] + ["oreb", "dreb", "fgm", "fgm3", "fgm2"]].copy()
    for s_ in STATS + ["min"]:
        v = pg[s_].where(played, 0.0)
        cum = v.groupby([pg["person_id"], pg["season"]]).cumsum() - v
        n = played.astype(int).groupby([pg["person_id"], pg["season"]]).cumsum() - played.astype(int)
        out[f"sa_{s_}"] = cum / n.replace(0, np.nan)
        out[f"st_{s_}"] = cum
        # last-10 played games mean
        pv = pg.loc[played, ["person_id", s_]]
        r = pv.groupby("person_id")[s_].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        col = pd.Series(np.nan, index=pg.index)
        col.loc[pv.index] = r
        out[f"l10_{s_}"] = col
    out["sa_n"] = played.astype(int).groupby([pg["person_id"], pg["season"]]).cumsum() - played.astype(int)
    # previous-season per-game averages as early-season fallback
    prev = pg[played].groupby(["person_id", "season"])[STATS + ["min"]].mean()
    prev = prev.reset_index()
    prev["season"] = prev["season"].map(lambda s: f"{int(s[:4]) + 1}-{(int(s[:4]) + 2) % 100:02d}")
    out = out.merge(prev.rename(columns={c: f"prev_{c}" for c in STATS + ["min"]}), on=["person_id", "season"],
                    how="left")
    return out


def add_model_inputs(df: pd.DataFrame) -> pd.DataFrame:
    mp = pd.read_parquet(PATHS.models / "minutes_predictions.parquet")[["game_id", "person_id", "pred_gbm", "rotation"]]
    rates = pd.read_parquet(PATHS.models / "player_rates.parquet")
    rcols = [c for c in rates.columns if c.startswith("r_")]
    tf = pd.read_parquet(PATHS.models / "team_factors.parquet")
    dcols = [c for c in tf.columns if c.startswith("def_") or c.startswith("off_")]
    tp = pd.read_parquet(PATHS.models / "team_predictions.parquet")[["game_id", "pace", "ppp_home", "ppp_away",
                                                                    "home_team_id", "exp_margin"]]
    feats = pd.read_parquet(PATHS.features / "player_asof.parquet",
                            columns=["game_id", "person_id", "vacated_min", "vacated_used_share", "slack_min",
                                     "n_absent_15", "days_since_played", "exp_min", "exp_usg", "career_games"])
    d = df.merge(mp, on=["game_id", "person_id"], how="inner")
    d = d.merge(rates[["game_id", "person_id"] + rcols], on=["game_id", "person_id"], how="left")
    opp = tf[["game_id", "team_id"] + dcols].rename(columns={"team_id": "opp_team_id",
                                                            **{c: f"opp_{c}" for c in dcols}})
    d = d.merge(opp, on=["game_id", "opp_team_id"], how="left")
    own = tf[["game_id", "team_id"] + [c for c in dcols if c.startswith("off_")]].rename(
        columns={c: f"own_{c}" for c in dcols if c.startswith("off_")})
    d = d.merge(own, on=["game_id", "team_id"], how="left")
    d = d.merge(tp, on="game_id", how="left")
    d = d.merge(feats, on=["game_id", "person_id"], how="left")
    d["team_ppp"] = np.where(d["team_id"] == d["home_team_id"], d["ppp_home"], d["ppp_away"]) / 100
    d["opp_ppp"] = np.where(d["team_id"] == d["home_team_id"], d["ppp_away"], d["ppp_home"]) / 100
    d["team_margin"] = np.where(d["team_id"] == d["home_team_id"], d["exp_margin"], -d["exp_margin"])
    return d


def eb_expectations(d: pd.DataFrame) -> pd.DataFrame:
    """Closed-form expected box score from EB per-possession rates and projected minutes."""
    m = d["pred_gbm"].clip(lower=0)
    poss_on = m / 48.0 * d["pace"].fillna(99)
    plays_on = poss_on * PLAYS_PER_POSS
    used = d["r_usage"] * plays_on
    tov = used * d["r_tov_share"] * d["opp_def_tov_forced"].fillna(1)
    non = used - tov
    fga3 = non * (d["r_fga3_share"] * d["opp_def_fga3_share_allowed"].fillna(1)).clip(0, 0.95)
    trips = non * (d["r_trip_share"] * d["opp_def_trips_allowed"].fillna(1)).clip(0, 0.6)
    fga2 = (non - fga3 - trips).clip(lower=0)
    fgm2 = fga2 * (d["r_fg2"] * d["opp_def_fg2_allowed"].fillna(1)).clip(0, 0.95)
    fgm3 = fga3 * (d["r_fg3"] * d["opp_def_fg3_allowed"].fillna(1)).clip(0, 0.95)
    fta = trips * (2 + d["r_trip3_share"]) + (fgm2 + fgm3) * d["r_and1"]
    ftm = fta * d["r_ft"]
    team_miss = poss_on * 0.53                           # live rebound chances per team possession (league)
    oreb = d["r_oreb"] * team_miss * d["opp_def_oreb_allowed"].fillna(1)
    dreb = d["r_dreb"] * team_miss
    team_fgm_on = poss_on * 0.42
    ast = d["r_ast_rate"] * (team_fgm_on - (fgm2 + fgm3)).clip(lower=0) * d["opp_def_ast_allowed"].fillna(1)
    out = pd.DataFrame(index=d.index)
    out["pts"] = 2 * fgm2 + 3 * fgm3 + ftm
    out["reb"] = oreb + dreb
    out["ast"] = ast
    out["fg3m"] = fgm3
    out["stl"] = d["r_stl"] * poss_on
    out["blk"] = d["r_blk"] * poss_on * 0.88
    out["tov"] = tov
    out["fga"] = fga2 + fga3
    out["fta"] = fta
    out["pra"] = out["pts"] + out["reb"] + out["ast"]
    return out


OPP_FACTOR = {"pts": None, "reb": "opp_def_oreb_allowed", "ast": "opp_def_ast_allowed",
              "fg3m": "opp_def_fga3_share_allowed", "stl": "own_off_stl_allowed", "blk": "own_off_blk_allowed",
              "tov": "opp_def_tov_forced", "fga": None, "fta": "opp_def_trips_allowed", "pra": None}


def baseline_predictions(d: pd.DataFrame) -> dict:
    preds = {}
    m = d["pred_gbm"].clip(lower=0)
    lg_ppp = d.groupby("season")["opp_ppp"].transform("mean")
    for s_ in STATS:
        sa = d[f"sa_{s_}"].where(d["sa_n"] >= 3).fillna(d[f"prev_{s_}"]).fillna(d[f"sa_{s_}"]).fillna(0.0)
        l10 = d[f"l10_{s_}"].fillna(sa)
        smin = d["sa_min"].where(d["sa_n"] >= 3).fillna(d["prev_min"])
        rate = (sa / smin.replace(0, np.nan)).fillna(0.0)
        per_min = rate * m
        fac = d[OPP_FACTOR[s_]].fillna(1.0) if OPP_FACTOR[s_] else None
        if s_ in ("pts", "fga", "pra"):
            fac = (d["opp_ppp"] / lg_ppp).fillna(1.0) if s_ != "fga" else 1.0
        preds[s_] = {"season_avg": sa.to_numpy(), "l10": l10.to_numpy(), "per_min": per_min.to_numpy(),
                     "opp_adj": (per_min * fac).to_numpy() if fac is not None else per_min.to_numpy()}
    eb = eb_expectations(d)
    for s_ in STATS:
        preds[s_]["eb"] = eb[s_].to_numpy()
    return preds


GBM_FEATURES = ["pred_gbm", "pace", "team_ppp", "opp_ppp", "team_margin", "is_home", "starter", "rotation",
                "vacated_min", "vacated_used_share", "slack_min", "n_absent_15", "days_since_played", "exp_min",
                "exp_usg", "career_games", "sa_n"]


def gbm_predictions(d: pd.DataFrame, train_seasons_for: dict, log=print) -> dict:
    eb = eb_expectations(d)
    rcols = [c for c in d.columns if c.startswith("r_")]
    ocols = [c for c in d.columns if c.startswith("opp_def_") or c.startswith("own_off_")]
    out = {s_: np.full(len(d), np.nan) for s_ in STATS}
    for s_ in STATS:
        X = pd.concat([d[GBM_FEATURES + rcols + ocols].astype(float),
                       eb[[s_]].rename(columns={s_: "eb_exp"}),
                       (d[f"sa_{s_}"] / d["sa_min"].replace(0, np.nan)).rename("sa_rate"),
                       (d[f"l10_{s_}"] / d["l10_min"].replace(0, np.nan)).rename("l10_rate")], axis=1)
        y = d[s_].astype(float)
        for test_season, train_seasons in train_seasons_for.items():
            tr = d["season"].isin(train_seasons).to_numpy()
            te = (d["season"] == test_season).to_numpy()
            mdl = lgb.train(GBM_PARAMS, lgb.Dataset(X[tr].astype(np.float32), label=y[tr]), num_boost_round=500)
            out[s_][te] = mdl.predict(X[te].astype(np.float32))
        log(f"[gbm] {s_} done")
    return out


def nb_dispersion(mu: np.ndarray, y: np.ndarray) -> float:
    """Method-of-moments NB size r (var = mu + mu^2 / r) pooled over mean bins."""
    ok = np.isfinite(mu) & (mu > 0.05)
    mu, y = mu[ok], y[ok]
    excess = ((y - mu) ** 2 - mu).sum()
    r = (mu ** 2).sum() / max(excess, 1e-6)
    return float(np.clip(r, 0.5, 1e4))


def nb_scores(mu: np.ndarray, r: float, y: np.ndarray, rng) -> dict:
    mu = np.clip(mu, 1e-3, None)
    p = r / (r + mu)
    ymax = int(max(y.max(), np.percentile(mu, 99.9) * 4 + 20))
    k = np.arange(ymax + 1)
    crps = np.zeros(len(mu))
    pit = np.zeros(len(mu))
    logp = np.zeros(len(mu))
    cov80 = np.zeros(len(mu), dtype=bool)
    for i0 in range(0, len(mu), 5000):
        sl = slice(i0, i0 + 5000)
        pm = sps.nbinom.pmf(k[None, :], r, p[sl, None])
        cdf = np.cumsum(pm, axis=1)
        yy = y[sl].astype(int)
        crps[sl] = ((cdf - (k[None, :] >= yy[:, None])) ** 2).sum(axis=1)
        idx = np.arange(len(yy))
        f_lo = np.where(yy > 0, cdf[idx, np.maximum(yy - 1, 0)], 0.0)
        pit[sl] = f_lo + rng.random(len(yy)) * (cdf[idx, yy] - f_lo)
        logp[sl] = np.log(np.maximum(pm[idx, yy], 1e-12))
        q10 = (cdf < 0.1).sum(axis=1)
        q90 = (cdf < 0.9).sum(axis=1)
        cov80[sl] = (yy >= q10) & (yy <= q90)
    return {"crps": crps, "pit": pit, "logp": logp, "cov80": cov80}


def opponent_ablation(log=print) -> dict:
    """Do opponent / matchup features improve player projections out of sample? LightGBM with and without
    every opponent feature (team style factors, opponent efficiency), same training protocol, test seasons."""
    from ..config import TEST_SEASONS, TUNING_SEASON
    df = load_frame()
    seasons = [TUNING_SEASON, *TEST_SEASONS]
    d = add_model_inputs(df[df["played"] & df["season"].isin(seasons)].copy()).reset_index(drop=True)
    eb = eb_expectations(d)
    rcols = [c for c in d.columns if c.startswith("r_")]
    ocols = [c for c in d.columns if c.startswith("opp_def_") or c.startswith("own_off_")]
    plan = {TEST_SEASONS[0]: [TUNING_SEASON], TEST_SEASONS[1]: [TUNING_SEASON, TEST_SEASONS[0]]}
    out = {}
    for s_ in ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov"]:
        base_cols = GBM_FEATURES + rcols
        X_full = pd.concat([d[base_cols + ocols].astype(float), eb[[s_]].rename(columns={s_: "eb_exp"})], axis=1)
        X_no = X_full.drop(columns=ocols + ["opp_ppp"])
        y = d[s_].astype(float)
        res = {}
        test_rows = d["season"].isin(list(plan)).to_numpy()
        for name, X in (("with_opponent", X_full), ("without_opponent", X_no)):
            pred = np.full(len(d), np.nan)
            for te_s, tr_s in plan.items():
                tr, te = d["season"].isin(tr_s).to_numpy(), (d["season"] == te_s).to_numpy()
                m = lgb.train(GBM_PARAMS, lgb.Dataset(X[tr].astype(np.float32), label=y[tr]), num_boost_round=500)
                pred[te] = m.predict(X[te].astype(np.float32))
            e = pred[test_rows] - y.to_numpy()[test_rows]
            res[name] = {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean()))}
            res[f"_err_{name}"] = e
        e1, e0 = res.pop("_err_with_opponent"), res.pop("_err_without_opponent")
        diff = np.abs(e0) - np.abs(e1)                       # > 0: opponent features help
        rng = np.random.default_rng(0)
        g = d["game_id"].to_numpy()[test_rows]               # cluster bootstrap by game
        uniq = np.unique(g)
        idx = {u: np.flatnonzero(g == u) for u in uniq}
        boots = []
        for _ in range(300):
            pick = rng.choice(uniq, len(uniq))
            boots.append(np.concatenate([diff[idx[u]] for u in pick]).mean())
        res["mae_gain"] = float(diff.mean())
        res["mae_gain_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        out[s_] = res
        log(f"[ablation] {s_}: MAE with {res['with_opponent']['mae']:.4f} without {res['without_opponent']['mae']:.4f} "
            f"gain {res['mae_gain']:+.4f} CI {res['mae_gain_ci95'][0]:+.4f}..{res['mae_gain_ci95'][1]:+.4f}")
    return out
