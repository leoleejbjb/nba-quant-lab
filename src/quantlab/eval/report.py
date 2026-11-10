"""Evaluation summaries for docs/REPORT.md and the dashboard's Validation page.

All comparisons are on identical player-games (inner join across models). Seasons:
2023-24 = tuning (hyper-parameters, simulator calibration, blend weight); 2024-25 and 2025-26 = test.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from ..config import PATHS, TEST_SEASONS, TUNING_SEASON
from .baselines import STATS as BASE_STATS
from .baselines import nb_dispersion, nb_scores

PLAYER_STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "pra"]
BASE_MODELS = ["season_avg", "l10", "per_min", "opp_adj", "eb", "gbm"]


def load_sim(mode: str, seasons) -> tuple[pd.DataFrame, pd.DataFrame]:
    P, T = [], []
    for s in seasons:
        pp = PATHS.backtest / f"sim_{mode}_{s}_players.parquet"
        tt = PATHS.backtest / f"sim_{mode}_{s}_teams.parquet"
        if pp.exists():
            P.append(pd.read_parquet(pp).assign(season=s))
            T.append(pd.read_parquet(tt))
    return (pd.concat(P, ignore_index=True) if P else pd.DataFrame(),
            pd.concat(T, ignore_index=True) if T else pd.DataFrame())


def pit_dev(p: np.ndarray) -> float:
    h = np.histogram(p, bins=10, range=(0, 1))[0] / max(len(p), 1)
    return float(np.abs(h - 0.1).sum())


def player_comparison(seasons, rng=None) -> tuple[pd.DataFrame, dict]:
    """Model x stat metrics on the common set of played player-games (rotation + fringe)."""
    rng = rng or np.random.default_rng(5)
    base = pd.read_parquet(PATHS.backtest / "baseline_predictions.parquet")
    base = base[base["season"].isin(seasons)]
    sims = {m: load_sim(m, seasons)[0] for m in ("none", "team")}
    td_path = PATHS.backtest / f"topdown_{'_'.join(seasons)}.parquet"
    topdown = pd.read_parquet(td_path) if td_path.exists() else None
    rows, pits, rel = [], {}, {}
    # NB dispersion per (stat, model) fitted on the tuning season predictions (never on test)
    tune = pd.read_parquet(PATHS.backtest / "baseline_predictions.parquet")
    tune = tune[tune["season"] == TUNING_SEASON]
    for st in PLAYER_STATS:
        keys = base[["game_id", "person_id"]].copy()
        y = base[st].to_numpy(dtype=float)
        common = keys.assign(y=y)
        for m, S in sims.items():
            if S.empty:
                continue
            s = S[S["stat"] == st][["game_id", "person_id", "mean"]].rename(columns={"mean": f"sim_{m}"})
            s = s.drop_duplicates(["game_id", "person_id"], keep=False)    # (a traded player listed twice)
            common = common.merge(s, on=["game_id", "person_id"], how="inner")
        if topdown is not None:
            s = topdown[topdown["stat"] == st][["game_id", "person_id", "mean"]].rename(columns={"mean": "topdown"})
            common = common.merge(s, on=["game_id", "person_id"], how="inner")
        b = base.merge(common[["game_id", "person_id"]], on=["game_id", "person_id"], how="inner")
        b = b.set_index(["game_id", "person_id"]).loc[common.set_index(["game_id", "person_id"]).index]
        yy = common["y"].to_numpy()
        for m in BASE_MODELS:
            col = f"{st}__{m}"
            if col not in b or b[col].isna().all():
                continue
            mu = b[col].to_numpy(dtype=float)
            ok = np.isfinite(mu)
            # dispersion from the tuning season only; GBM has no tuning-season predictions (it is trained on
            # earlier seasons), so it borrows the EB model's tuning-season dispersion
            src = col if (col in tune and tune[col].notna().any()) else f"{st}__eb"
            r = nb_dispersion(tune[src].to_numpy(dtype=float), tune[st].to_numpy(dtype=float))
            sc = nb_scores(mu[ok], r, yy[ok], rng)
            rows.append({"stat": st, "model": m, "n": int(ok.sum()), "mae": float(np.abs(mu[ok] - yy[ok]).mean()),
                         "rmse": float(np.sqrt(((mu[ok] - yy[ok]) ** 2).mean())), "bias": float((mu[ok] - yy[ok]).mean()),
                         "crps": float(sc["crps"].mean()), "pit_dev": pit_dev(sc["pit"]),
                         "cov80": float(sc["cov80"].mean()), "logscore": float(sc["logp"].mean())})
            pits[f"{st}|{m}"] = np.histogram(sc["pit"], bins=10, range=(0, 1))[0].tolist()
        for m, S in list(sims.items()) + ([("topdown", topdown)] if topdown is not None else []):
            if S is None or S.empty:
                continue
            s = S[S["stat"] == st].drop_duplicates(["game_id", "person_id"], keep=False)
            s = s.merge(common[["game_id", "person_id"]], on=["game_id", "person_id"])
            name = f"sim_{m}" if m != "topdown" else "topdown"
            e = s["mean"] - s["actual"]
            cov80 = ((s["actual"] >= s["q10"]) & (s["actual"] <= s["q90"])).mean()
            cov50 = ((s["actual"] >= s["q25"]) & (s["actual"] <= s["q75"])).mean()
            cov90 = ((s["actual"] >= s["q05"]) & (s["actual"] <= s["q95"])).mean()
            rows.append({"stat": st, "model": name, "n": int(len(s)), "mae": float(e.abs().mean()),
                         "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean()),
                         "crps": float(s["crps"].mean()), "pit_dev": pit_dev(s["pit"].to_numpy()),
                         "cov80": float(cov80), "cov50": float(cov50), "cov90": float(cov90),
                         "logscore": float(s["logp"].mean())})
            pits[f"{st}|{name}"] = np.histogram(s["pit"], bins=10, range=(0, 1))[0].tolist()
            if "p_over" in s and s["p_over"].notna().any():
                q = pd.qcut(s["p_over"], 10, duplicates="drop")
                g = s.groupby(q, observed=True).agg(pred=("p_over", "mean"), obs=("over", "mean"), n=("over", "size"))
                rel[f"{st}|{name}"] = g.reset_index(drop=True).to_dict("list")
    return pd.DataFrame(rows), {"pit": pits, "reliability": rel}


def minutes_comparison(seasons) -> pd.DataFrame:
    ev = pd.read_csv(PATHS.models / "minutes_eval.csv")
    mp = pd.read_parquet(PATHS.models / "minutes_predictions.parquet")
    mp = mp[mp["season"].isin(seasons)]
    rows = []
    for m in [c for c in mp.columns if c.startswith("pred_")]:
        e = mp[m] - mp["min"]
        rows.append({"model": m[5:], "mae": float(e.abs().mean()), "rmse": float(np.sqrt((e ** 2).mean())),
                     "bias": float(e.mean()), "n": int(len(e)),
                     "mae_rotation": float(e[mp["rotation"] == True].abs().mean())})
    for mode in ("none", "team"):
        S, _ = load_sim(mode, seasons)
        if S.empty:
            continue
        s = S[S["stat"] == "min"]
        e = s["mean"] - s["actual"]
        rows.append({"model": f"sim_{mode}", "mae": float(e.abs().mean()), "rmse": float(np.sqrt((e ** 2).mean())),
                     "bias": float(e.mean()), "n": int(len(s)),
                     "mae_rotation": float(e[s["rotation"]].abs().mean()),
                     "cov80": float(((s["actual"] >= s["q10"]) & (s["actual"] <= s["q90"])).mean()),
                     "pit_dev": pit_dev(s["pit"].to_numpy()), "crps": float(s["crps"].mean())})
    del ev
    return pd.DataFrame(rows)


def team_comparison(tune_season: str, test_seasons, n_boot: int = 2000) -> dict:
    tp = pd.read_parquet(PATHS.models / "team_predictions.parquet")
    out = {}
    from scipy.stats import norm

    def metrics(margin_hat, total_hat, sm, st_, y_m, y_t, p_win=None):
        em, et = y_m - margin_hat, y_t - total_hat
        p = p_win if p_win is not None else norm.cdf(margin_hat / sm)
        p = np.clip(p, 1e-4, 1 - 1e-4)
        win = (y_m > 0).astype(float)
        crps_m = crps_normal(y_m, margin_hat, sm)
        crps_t = crps_normal(y_t, total_hat, st_)
        return {"margin_mae": float(np.abs(em).mean()), "margin_rmse": float(np.sqrt((em ** 2).mean())),
                "total_mae": float(np.abs(et).mean()), "total_rmse": float(np.sqrt((et ** 2).mean())),
                "margin_crps": float(crps_m.mean()), "total_crps": float(crps_t.mean()),
                "win_logloss": float(-(win * np.log(p) + (1 - win) * np.log(1 - p)).mean()),
                "win_brier": float(((p - win) ** 2).mean()), "games": int(len(em))}

    def frame(seasons):
        _, Tn = load_sim("none", seasons)
        _, Tt = load_sim("team", seasons)
        d = tp[tp["season"].isin(seasons)][["game_id", "season", "exp_margin", "exp_total", "sigma_margin",
                                             "sigma_total", "margin", "total", "p_home_win"]]
        if not Tn.empty:
            d = d.merge(Tn[["game_id", "sim_margin", "sim_total", "sim_margin_sd", "sim_total_sd", "sim_p_home_win",
                            "sim_margin_crps", "sim_total_crps"]].add_suffix("_none").rename(
                columns={"game_id_none": "game_id"}), on="game_id")
        if not Tt.empty:
            d = d.merge(Tt[["game_id", "sim_margin", "sim_total", "sim_margin_sd", "sim_total_sd", "sim_p_home_win",
                            "sim_margin_crps", "sim_total_crps"]].add_suffix("_team").rename(
                columns={"game_id_team": "game_id"}), on="game_id")
        return d.dropna(subset=["margin"])

    tune = frame([tune_season])
    test = frame(list(test_seasons))
    # blend weight (team-model share) chosen on the tuning season by margin + total RMSE
    ws = np.round(np.linspace(0, 1, 21), 2)
    def blend_err(d, w):
        m = w * d["exp_margin"] + (1 - w) * d["sim_margin_none"]
        t = w * d["exp_total"] + (1 - w) * d["sim_total_none"]
        return float(np.sqrt(((d["margin"] - m) ** 2).mean()) + np.sqrt(((d["total"] - t) ** 2).mean()))
    curve = {float(w): blend_err(tune, w) for w in ws} if "sim_margin_none" in tune else {}
    w_star = float(min(curve, key=curve.get)) if curve else 1.0
    for name, d in (("tuning", tune), ("test", test)):
        res = {"team_model": metrics(d["exp_margin"], d["exp_total"], d["sigma_margin"], d["sigma_total"],
                                     d["margin"].to_numpy(), d["total"].to_numpy(), d["p_home_win"].to_numpy())}
        for m in ("none", "team"):
            if f"sim_margin_{m}" in d:
                res[f"sim_{m}"] = metrics(d[f"sim_margin_{m}"], d[f"sim_total_{m}"], d[f"sim_margin_sd_{m}"],
                                          d[f"sim_total_sd_{m}"], d["margin"].to_numpy(), d["total"].to_numpy(),
                                          d[f"sim_p_home_win_{m}"].to_numpy())
                res[f"sim_{m}"]["margin_crps_empirical"] = float(d[f"sim_margin_crps_{m}"].mean())
                res[f"sim_{m}"]["total_crps_empirical"] = float(d[f"sim_total_crps_{m}"].mean())
        if "sim_margin_none" in d:
            mb = w_star * d["exp_margin"] + (1 - w_star) * d["sim_margin_none"]
            tb = w_star * d["exp_total"] + (1 - w_star) * d["sim_total_none"]
            sm = np.sqrt(w_star * d["sigma_margin"] ** 2 + (1 - w_star) * d["sim_margin_sd_none"] ** 2)
            st_ = np.sqrt(w_star * d["sigma_total"] ** 2 + (1 - w_star) * d["sim_total_sd_none"] ** 2)
            res["blend"] = metrics(mb, tb, sm, st_, d["margin"].to_numpy(), d["total"].to_numpy())
            res["blend"]["weight_team_model"] = w_star
            # paired bootstrap: blend vs team model (squared-error differences)
            rng = np.random.default_rng(1)
            dm = (d["margin"] - d["exp_margin"]) ** 2 - (d["margin"] - mb) ** 2
            dt = (d["total"] - d["exp_total"]) ** 2 - (d["total"] - tb) ** 2
            dn = (d["margin"] - d["exp_margin"]) ** 2 - (d["margin"] - d["sim_margin_none"]) ** 2
            bt = lambda x: np.percentile([x.sample(frac=1, replace=True, random_state=int(rng.integers(1e9))).mean()
                                          for _ in range(n_boot)], [2.5, 97.5]).tolist()
            res["bootstrap_mse_gain_vs_team_model"] = {
                "margin_blend": {"mean": float(dm.mean()), "ci95": bt(dm)},
                "total_blend": {"mean": float(dt.mean()), "ci95": bt(dt)},
                "margin_bottom_up": {"mean": float(dn.mean()), "ci95": bt(dn)}}
        out[name] = res
    out["blend_curve_tuning"] = curve
    out["blend_weight"] = w_star
    return out


def crps_normal(y, mu, sd):
    from scipy.stats import norm
    z = (np.asarray(y) - np.asarray(mu)) / np.asarray(sd)
    return np.asarray(sd) * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def cost_benchmark(n: int = 20000) -> dict:
    """Wall-clock cost of the two architectures for one game (4 cores)."""
    from ..sim.engine import GameSimulator
    from ..sim.inputs import InputStore, game_spec
    store = InputStore()
    gid = int(store.games[store.games["season"] == "2025-26"].index[300])
    sim = GameSimulator(game_spec(store, gid))
    sim.run(200, seed=0)
    t0 = time.perf_counter(); sim.calibrate(); t_cal = time.perf_counter() - t0
    t0 = time.perf_counter(); res = sim.run(n, seed=1); t_run = time.perf_counter() - t0
    inv = res.check_invariants()
    return {"n_sims": n, "calibration_s": t_cal, "simulate_s": t_run, "us_per_game": 1e6 * t_run / n,
            "invariants": inv}


def run_report(log=print) -> dict:
    summary = {"tuning_season": TUNING_SEASON, "test_seasons": list(TEST_SEASONS)}
    for name, seasons in (("tuning", [TUNING_SEASON]), ("test", list(TEST_SEASONS))):
        pc, extra = player_comparison(seasons)
        summary[f"players_{name}"] = pc.to_dict("records")
        summary[f"pit_{name}"] = extra["pit"]
        summary[f"reliability_{name}"] = extra["reliability"]
        summary[f"minutes_{name}"] = minutes_comparison(seasons).to_dict("records")
        log(f"[report] player comparison {name}: {len(pc)} rows")
    summary["teams"] = team_comparison(TUNING_SEASON, TEST_SEASONS)
    summary["consistency"] = consistency_analysis()
    abl = PATHS.backtest / "opponent_ablation.json"
    if abl.exists():
        summary["opponent_ablation"] = json.loads(abl.read_text())
    try:
        summary["participation"] = participation_analysis()
    except Exception as e:      # pragma: no cover
        summary["participation"] = {"error": str(e)}
    try:
        summary["cost"] = cost_benchmark()
    except Exception as e:      # pragma: no cover - benchmark is informative only
        summary["cost"] = {"error": str(e)}
    (PATHS.backtest / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    (PATHS.models / "integration.json").write_text(json.dumps({"blend_weight": summary["teams"]["blend_weight"]}))
    return summary


def participation_analysis(season: str = TUNING_SEASON, n_games: int = 600, seed: int = 3) -> dict:
    """Selection effects: how well pregame participation probabilities for *fringe* players (whose appearance
    is not known before tip-off) match who actually appeared, using the official 2020-24 availability data
    (played / dressed-but-unused / out). Rotation players are treated as known from the injury report."""
    from ..sim.inputs import InputStore, game_spec
    store = InputStore()
    av = pd.read_parquet(PATHS.processed("all") / "availability.parquet")
    av = av[av["season"] == season]
    ids = store.games[store.games["season"] == season].index.to_numpy()
    ids = np.random.default_rng(seed).choice(ids, min(n_games, len(ids)), replace=False)
    rows = []
    for gid in ids:
        spec = game_spec(store, int(gid))
        status = av[av["game_id"] == gid].set_index("person_id")["status"].to_dict()
        for team in (spec.home, spec.away):
            for p in team.players:
                rows.append({"game_id": int(gid), "person_id": p.person_id, "rotation": p.rotation,
                             "p_play": p.p_play, "status": status.get(p.person_id, "absent_from_box")})
    d = pd.DataFrame(rows)
    fr = d[~d["rotation"]].copy()
    fr["played"] = (fr["status"] == "played").astype(float)
    fr["bin"] = pd.cut(fr["p_play"], [0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
    cal = fr.groupby("bin", observed=True).agg(pred=("p_play", "mean"), obs=("played", "mean"), n=("played", "size"))
    av_counts = av["status"].value_counts().to_dict()
    return {"season": season, "games": int(len(ids)),
            "fringe_rows": int(len(fr)), "fringe_played_share": float(fr["played"].mean()),
            "fringe_mean_p_play": float(fr["p_play"].mean()),
            "fringe_status_counts": {k: int(v) for k, v in fr["status"].value_counts().items()},
            "calibration": cal.reset_index(drop=True).to_dict("list"),
            "season_status_counts": {k: int(v) for k, v in av_counts.items()},
            "rotation_rows": int(d["rotation"].sum())}


def consistency_analysis(seasons=TEST_SEASONS) -> dict:
    """What happens when independently projected player statistics are added up?

    For each team-game: the sum of independent per-player point projections vs the actual team score and vs the team
    model's expectation, and the team-score uncertainty implied by treating players' distributions as independent
    (sum of negative-binomial variances, dispersion fitted on the tuning season) vs the realised error. The possession
    simulator's team score is consistent with its player box scores by construction (asserted in every game)."""
    base = pd.read_parquet(PATHS.backtest / "baseline_predictions.parquet")
    tune = base[base["season"] == TUNING_SEASON]
    test = base[base["season"].isin(list(seasons))].copy()
    tg = pd.read_parquet(PATHS.processed("all") / "team_games.parquet", columns=["game_id", "team_id", "pts", "is_home"])
    tp = pd.read_parquet(PATHS.models / "team_predictions.parquet")[["game_id", "exp_home_pts", "exp_away_pts"]]
    tg = tg.merge(tp, on="game_id")
    tg["team_model"] = np.where(tg["is_home"], tg["exp_home_pts"], tg["exp_away_pts"])
    out = {}
    for m in ("eb", "gbm"):
        col = f"pts__{m}"
        if test[col].isna().all():
            continue
        r = nb_dispersion(tune[f"pts__eb"].to_numpy(float), tune["pts"].to_numpy(float))
        test["_var"] = test[col] + test[col] ** 2 / r
        g = test.groupby(["game_id", "team_id"]).agg(sum_pred=(col, "sum"), sum_var=("_var", "sum")).reset_index()
        g = g.merge(tg, on=["game_id", "team_id"])
        err = g["pts"] - g["sum_pred"]
        out[m] = {"team_games": int(len(g)), "mean_gap_vs_team_model": float((g["sum_pred"] - g["team_model"]).mean()),
                  "mae_gap_vs_team_model": float((g["sum_pred"] - g["team_model"]).abs().mean()),
                  "rmse_team_points_from_sum": float(np.sqrt((err ** 2).mean())),
                  "implied_sd_if_independent": float(np.sqrt(g["sum_var"]).mean()),
                  "realised_sd": float(err.std()),
                  "coverage80_if_independent": float((np.abs(err) <= 1.2816 * np.sqrt(g["sum_var"])).mean())}
    t = tg.merge(test[["game_id"]].drop_duplicates(), on="game_id")
    out["team_model_rmse_team_points"] = float(np.sqrt(((t["pts"] - t["team_model"]) ** 2).mean()))
    return out
