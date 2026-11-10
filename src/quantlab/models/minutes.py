"""Pregame minutes projections.

Information set at tip-off (see docs/METHODOLOGY.md): every feature is from
earlier dates, plus the two pieces of same-day information that are public
before tip-off - who is active (oracle participation in the backtest: the
players who appeared) and, for the "lineups known" variant, the starting five.

Models compared (walk-forward; for season S a model is trained on seasons < S):

* ``season_avg``  season-to-date minutes per game played (falls back to last season)
* ``l10``         last-10-games average
* ``ewma``        decayed mean (half-life 20 effective days)
* ``gbm``         LightGBM on recent minutes, role, rest, teammate absences, blowout risk
* ``gbm_norm``    GBM scaled so each team's projections sum to 240 x expected game length

The residual distribution (for the simulator's planned-minutes noise) is a
table of residual quantiles by projected-minutes bin and starter role.
"""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..config import PATHS

FEATURES = ["exp_min", "min_l3", "min_l5", "min_l10", "last_min", "season_min_avg", "base_min", "rotation",
            "min7", "min60", "start_rate", "last_start", "days_since_played", "rest_days", "b2b",
            "vacated_min", "slack_min", "n_absent_15", "vacated_used_share", "n_baseline", "share_of_avail",
            "new_team", "career_games", "season_games", "abs_spread", "fav", "is_home", "exp_usg", "gp_m20"]
STARTER_FEATURES = ["starter", "n_starters_known_min"]
GBM_PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1,
                  num_threads=4, seed=7, deterministic=True, force_row_wise=True)
N_ROUNDS = 700
DEBUT_MINUTES = 12.0   # baseline guess for a player with no NBA history


def minutes_frame(feats: pd.DataFrame, team_pred: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    f = feats.copy()
    f["min7"] = f["min_sum_m7"] / f["gp_m7"].replace(0, np.nan)
    f["min60"] = f["min_sum_m60"] / f["gp_m60"].replace(0, np.nan)
    f["start_rate"] = f["starts_m20"] / f["gp_m20"].replace(0, np.nan)
    # pregame-legit denominator from team_context (rotation participants + fringe baseline minutes)
    f["share_of_avail"] = f["exp_min"] / f["avail_exp_min"].replace(0, np.nan)
    rest = tg[["game_id", "team_id", "rest_days", "b2b"]]
    f = f.merge(rest, on=["game_id", "team_id"], how="left")
    tp = team_pred[["game_id", "home_team_id", "exp_margin"]]
    f = f.merge(tp, on="game_id", how="left")
    team_margin = np.where(f["team_id"] == f["home_team_id"], f["exp_margin"], -f["exp_margin"])
    f["abs_spread"] = np.abs(team_margin)
    f["fav"] = (team_margin > 0).astype(np.int8)
    f["b2b"] = f["b2b"].astype(float)
    f["is_home"] = f["is_home"].astype(float)
    f["starter"] = f["starter"].astype(float)
    f["rotation"] = f["rotation"].astype(float)
    # minutes of known starters on the team (lineups-known variant)
    st = f[f["starter"] == 1].groupby(["game_id", "team_id"])["exp_min"].sum().rename("n_starters_known_min")
    f = f.merge(st.reset_index(), on=["game_id", "team_id"], how="left")
    return f.drop(columns=["home_team_id", "exp_margin"])


def baselines(f: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=f.index)
    fallback = f["exp_min"].fillna(DEBUT_MINUTES)
    out["season_avg"] = f["season_min_avg"].fillna(f["min60"]).fillna(fallback)
    out["l10"] = f["min_l10"].fillna(fallback)
    out["ewma"] = f["exp_min"]
    return out


def _fit(train: pd.DataFrame, cols: list[str]) -> lgb.Booster:
    ds = lgb.Dataset(train[cols].astype(np.float32), label=train["min"].astype(np.float32))
    return lgb.train(GBM_PARAMS, ds, num_boost_round=N_ROUNDS)


def normalize_team(f: pd.DataFrame, pred: pd.Series, total: float = 241.5) -> pd.Series:
    """Scale rotation players so that rotation + expected fringe minutes = 240 x expected game length.

    Fringe players (whose participation is not known pregame) keep their conditional projection and
    enter the budget only through their baseline minutes, so normalization never uses who played in
    garbage time.
    """
    rot = f["rotation"] == 1
    rot_sum = pred.where(rot, 0.0).groupby([f["game_id"], f["team_id"]]).transform("sum")
    budget = (total - f["fringe_base_min"].fillna(0)).clip(lower=150)
    scale = (budget / rot_sum.replace(0, np.nan)).clip(0.8, 1.25).fillna(1.0)
    return pred.where(~rot, pred * scale)


def walk_forward_minutes(f: pd.DataFrame, seasons: list[str], log=print) -> tuple[pd.DataFrame, dict]:
    """Predictions for played player-games of ``seasons`` (each season trained on all earlier seasons)."""
    played = f[f["played"]].copy()
    all_seasons = sorted(played["season"].unique())
    outs, models = [], {}
    for s in seasons:
        train = played[(played["season"] < s) & (played["season"] > all_seasons[0])]   # skip the cold-start season
        test = played[played["season"] == s].copy()
        b = baselines(test)
        for c in b:
            test[f"pred_{c}"] = b[c].to_numpy()
        m_a = _fit(train, FEATURES)
        m_b = _fit(train, FEATURES + STARTER_FEATURES)
        test["pred_gbm_pre"] = m_a.predict(test[FEATURES].astype(np.float32))
        test["pred_gbm"] = m_b.predict(test[FEATURES + STARTER_FEATURES].astype(np.float32))
        test["pred_gbm_norm"] = normalize_team(test, test["pred_gbm"])
        test["pred_gbm_pre_norm"] = normalize_team(test, test["pred_gbm_pre"])
        models[s] = {"lineups_known": m_b, "pregame": m_a}
        outs.append(test)
        log(f"[minutes] {s}: trained on {len(train):,} player-games, predicted {len(test):,}")
    return pd.concat(outs), models


def evaluate_minutes(pred: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in pred.columns if c.startswith("pred_")]
    rows = []
    for (s, grp) in [("all", pred)] + list(pred.groupby("season")):
        for c in cols:
            e = grp[c] - grp["min"]
            rows.append({"season": s, "model": c[5:], "mae": e.abs().mean(), "rmse": np.sqrt((e ** 2).mean()),
                         "bias": e.mean(), "n": len(grp),
                         "mae_starters": e[grp["starter"] == 1].abs().mean(),
                         "mae_bench": e[grp["starter"] == 0].abs().mean(),
                         "mae_rotation": e[grp["rotation"] == True].abs().mean(),
                         "mae_fringe": e[grp["rotation"] != True].abs().mean()})
    return pd.DataFrame(rows)


def residual_table(pred: pd.DataFrame, col: str = "pred_gbm_norm") -> pd.DataFrame:
    """Empirical residual quantiles by projected-minutes bin and starter role (for planned-minutes noise)."""
    d = pred[[col, "min", "starter"]].dropna().copy()
    d["bin"] = pd.cut(d[col], [0, 8, 14, 20, 26, 31, 35, 60], labels=False)
    q = np.linspace(0.02, 0.98, 25)
    rows = []
    for (b, st), g in d.groupby(["bin", "starter"]):
        r = (g["min"] - g[col]).quantile(q).to_numpy()
        rows.append({"bin": int(b), "starter": int(st), "n": len(g), "sd": float((g["min"] - g[col]).std()),
                     **{f"q{int(round(x * 100)):02d}": v for x, v in zip(q, r)}})
    return pd.DataFrame(rows)


def run_minutes(log=print) -> pd.DataFrame:
    feats = pd.read_parquet(PATHS.features / "player_asof.parquet")
    tp = pd.read_parquet(PATHS.models / "team_predictions.parquet")
    tg = pd.read_parquet(PATHS.processed("all") / "team_games.parquet")
    f = minutes_frame(feats, tp, tg)
    pred, models = walk_forward_minutes(f, ["2023-24", "2024-25", "2025-26"], log=log)
    ev = evaluate_minutes(pred)
    PATHS.ensure(PATHS.models)
    keep = ["game_id", "person_id", "team_id", "season", "game_date", "min", "starter", "rotation"] + \
           [c for c in pred.columns if c.startswith("pred_")]
    pred[keep].to_parquet(PATHS.models / "minutes_predictions.parquet", index=False)
    ev.to_csv(PATHS.models / "minutes_eval.csv", index=False)
    residual_table(pred[pred["season"] == "2023-24"], col="pred_gbm").to_csv(PATHS.models / "minutes_residuals.csv", index=False)
    # final model for live use: trained on everything up to the last season
    played = f[f["played"] & (f["season"] > sorted(f["season"].unique())[0])]
    for name, cols in (("lineups_known", FEATURES + STARTER_FEATURES), ("pregame", FEATURES)):
        _fit(played, cols).save_model(str(PATHS.models / f"minutes_gbm_{name}.txt"))
    imp = pd.Series(models["2025-26"]["lineups_known"].feature_importance("gain"),
                    index=FEATURES + STARTER_FEATURES).sort_values(ascending=False)
    (PATHS.models / "minutes_importance.json").write_text(json.dumps((imp / imp.sum()).round(4).to_dict(), indent=2))
    return ev


def save_season_models(log=print) -> None:
    """Persist, per season S, the lineups-known minutes model trained on seasons < S, plus the feature
    frame, so the scenario engine can re-project minutes after a lineup change without leakage."""
    feats = pd.read_parquet(PATHS.features / "player_asof.parquet")
    tp = pd.read_parquet(PATHS.models / "team_predictions.parquet")
    tg = pd.read_parquet(PATHS.processed("all") / "team_games.parquet")
    f = minutes_frame(feats, tp, tg)
    cols = FEATURES + STARTER_FEATURES
    keep = ["game_id", "person_id", "team_id", "season", "played", "min", "starter_raw"] + \
           sorted(set(cols) | {"avail_exp_min", "base_used", "team_used_base", "fringe_base_min"})
    f["starter_raw"] = f["starter"]
    f[[c for c in keep if c in f.columns]].to_parquet(PATHS.models / "minutes_features.parquet", index=False)
    played = f[f["played"]]
    seasons = sorted(played["season"].unique())
    for s in seasons[2:]:
        train = played[(played["season"] < s) & (played["season"] > seasons[0])]
        _fit(train, cols).save_model(str(PATHS.models / f"minutes_gbm_{s}.txt"))
        log(f"[minutes] saved model for {s} ({len(train):,} rows)")
