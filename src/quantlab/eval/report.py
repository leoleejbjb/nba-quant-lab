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
from .backtest import LINE_OFFSETS
from .baselines import STATS as BASE_STATS
from .baselines import BinnedDispersion, nb_scores

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


SIM_MODES = ("none", "team", "blend")


def pit_dev(p: np.ndarray) -> float:
    h = np.histogram(p, bins=10, range=(0, 1))[0] / max(len(p), 1)
    return float(np.abs(h - 0.1).sum())


def cov_pit(p: np.ndarray, level: float = 0.8) -> float:
    """Share of outcomes inside the central ``level`` of the forecast distribution, ties randomized (the
    randomized PIT falls in [(1-level)/2, (1+level)/2]): exactly ``level`` for a calibrated forecast of a
    discrete statistic, unlike quantile intervals with inclusive ends, which over-cover by construction."""
    lo = (1 - level) / 2
    p = np.asarray(p, dtype=float)
    return float(((p >= lo) & (p <= 1 - lo)).mean())


def reliability(pred: np.ndarray, obs: np.ndarray, bins: int = 10) -> dict:
    ok = np.isfinite(pred) & np.isfinite(obs)
    q = pd.qcut(pd.Series(pred[ok]), bins, duplicates="drop")
    g = pd.DataFrame({"p": pred[ok], "o": obs[ok]}).groupby(q.to_numpy(), observed=True).agg(
        pred=("p", "mean"), obs=("o", "mean"), n=("o", "size"))
    return g.reset_index(drop=True).to_dict("list")


def player_comparison(seasons, rng=None) -> tuple[pd.DataFrame, dict]:
    """Model x stat metrics on the common set of played player-games (rotation + fringe)."""
    rng = rng or np.random.default_rng(5)
    base = pd.read_parquet(PATHS.backtest / "baseline_predictions.parquet")
    base = base[base["season"].isin(seasons)]
    sims = {m: load_sim(m, seasons)[0] for m in SIM_MODES}
    sims = {m: S for m, S in sims.items() if not S.empty}
    td_path = PATHS.backtest / f"topdown_{'_'.join(seasons)}.parquet"
    topdown = pd.read_parquet(td_path) if td_path.exists() else None
    rows, pits, rel = [], {}, {}
    # NB dispersion per (stat, model), mean-binned, fitted on the tuning season predictions (never on test)
    tune = pd.read_parquet(PATHS.backtest / "baseline_predictions.parquet")
    tune = tune[tune["season"] == TUNING_SEASON]
    for st in PLAYER_STATS:
        keys = base[["game_id", "person_id"]].copy()
        y = base[st].to_numpy(dtype=float)
        common = keys.assign(y=y)
        for m, S in sims.items():
            s = S[S["stat"] == st][["game_id", "person_id", "mean"]].rename(columns={"mean": f"sim_{m}"})
            s = s.drop_duplicates(["game_id", "person_id"], keep=False)    # (a traded player listed twice)
            common = common.merge(s, on=["game_id", "person_id"], how="inner")
        if topdown is not None:
            s = topdown[topdown["stat"] == st][["game_id", "person_id", "mean"]].rename(columns={"mean": "topdown"})
            s = s.drop_duplicates(["game_id", "person_id"], keep=False)
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
            # earlier seasons), so it borrows the EB model's tuning-season dispersion curve
            src = col if (col in tune and tune[col].notna().any()) else f"{st}__eb"
            disp = BinnedDispersion(tune[src].to_numpy(dtype=float), tune[st].to_numpy(dtype=float))
            sc = nb_scores(mu[ok], disp(mu[ok]), yy[ok], rng)
            rows.append({"stat": st, "model": m, "n": int(ok.sum()), "mae": float(np.abs(mu[ok] - yy[ok]).mean()),
                         "rmse": float(np.sqrt(((mu[ok] - yy[ok]) ** 2).mean())), "bias": float((mu[ok] - yy[ok]).mean()),
                         "crps": float(sc["crps"].mean()), "pit_dev": pit_dev(sc["pit"]),
                         "cov80": cov_pit(sc["pit"]), "cov80_inclusive": float(sc["cov80"].mean()),
                         "p_pit_below_half": float((sc["pit"] < 0.5).mean()), "logscore": float(sc["logp"].mean())})
            pits[f"{st}|{m}"] = np.histogram(sc["pit"], bins=10, range=(0, 1))[0].tolist()
        for m, S in list(sims.items()) + ([("topdown", topdown)] if topdown is not None else []):
            s = S[S["stat"] == st].drop_duplicates(["game_id", "person_id"], keep=False)
            s = s.merge(common[["game_id", "person_id"]], on=["game_id", "person_id"])
            name = f"sim_{m}" if m != "topdown" else "topdown"
            e = s["mean"] - s["actual"]
            cov80i = ((s["actual"] >= s["q10"]) & (s["actual"] <= s["q90"])).mean()
            rows.append({"stat": st, "model": name, "n": int(len(s)), "mae": float(e.abs().mean()),
                         "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean()),
                         "crps": float(s["crps"].mean()), "pit_dev": pit_dev(s["pit"].to_numpy()),
                         "cov80": cov_pit(s["pit"].to_numpy()), "cov80_inclusive": float(cov80i),
                         "cov50": cov_pit(s["pit"].to_numpy(), 0.5), "cov90": cov_pit(s["pit"].to_numpy(), 0.9),
                         "p_pit_below_half": float((s["pit"] < 0.5).mean()), "logscore": float(s["logp"].mean())})
            pits[f"{st}|{name}"] = np.histogram(s["pit"], bins=10, range=(0, 1))[0].tolist()
            if "p_over" in s and s["p_over"].notna().any():
                rel[f"{st}|{name}"] = reliability(s["p_over"].to_numpy(float), s["over"].to_numpy(float))
                # every line of the ladder (nearest half point -2 .. +2) pooled
                pp = [s["p_over"].to_numpy(float)] + [s[f"p_over_{k:+d}"].to_numpy(float) for k in LINE_OFFSETS
                                                       if f"p_over_{k:+d}" in s]
                oo = [s["over"].to_numpy(float)] + [s[f"over_{k:+d}"].to_numpy(float) for k in LINE_OFFSETS
                                                     if f"over_{k:+d}" in s]
                pp, oo = np.concatenate(pp), np.concatenate(oo)
                ok = np.isfinite(pp) & np.isfinite(oo)
                rel[f"{st}|{name}|ladder"] = reliability(pp, oo)
                rel[f"{st}|{name}|ladder"]["mean_gap"] = float(pp[ok].mean() - oo[ok].mean())
                rel[f"{st}|{name}"]["mean_gap"] = float(np.nanmean(s["p_over"]) - np.nanmean(s["over"]))
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
    for mode in SIM_MODES:
        S, _ = load_sim(mode, seasons)
        if S.empty:
            continue
        s = S[S["stat"] == "min"]
        e = s["mean"] - s["actual"]
        rows.append({"model": f"sim_{mode}", "mae": float(e.abs().mean()), "rmse": float(np.sqrt((e ** 2).mean())),
                     "bias": float(e.mean()), "n": int(len(s)),
                     "mae_rotation": float(e[s["rotation"]].abs().mean()),
                     "cov80": cov_pit(s["pit"].to_numpy()),
                     "pit_dev": pit_dev(s["pit"].to_numpy()), "crps": float(s["crps"].mean())})
    del ev
    return pd.DataFrame(rows)


def team_comparison(tune_season: str, test_seasons, n_boot: int = 2000) -> dict:
    """Spread / total / win forecasts: team model, the three simulator environments (the shipped one is
    ``blend``, backtested as such) and, for reference, a post-hoc average of team-model and bottom-up means."""
    tp = pd.read_parquet(PATHS.models / "team_predictions.parquet")
    integ = json.loads((PATHS.models / "integration.json").read_text()) if (PATHS.models / "integration.json").exists() else {}
    w_star = float(integ.get("blend_weight", 0.5))
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
                "total_bias": float(-et.mean()),
                "margin_crps": float(crps_m.mean()), "total_crps": float(crps_t.mean()),
                "win_logloss": float(-(win * np.log(p) + (1 - win) * np.log(1 - p)).mean()),
                "win_brier": float(((p - win) ** 2).mean()), "games": int(len(em))}

    cols = ["game_id", "sim_margin", "sim_total", "sim_margin_sd", "sim_total_sd", "sim_p_home_win",
            "sim_margin_crps", "sim_total_crps", "sim_ot", "sim_poss"]

    def frame(seasons):
        d = tp[tp["season"].isin(seasons)][["game_id", "season", "exp_margin", "exp_total", "sigma_margin",
                                             "sigma_total", "margin", "total", "p_home_win"]]
        for m in SIM_MODES:
            _, T = load_sim(m, seasons)
            if T.empty:
                continue
            d = d.merge(T[[c for c in cols if c in T]].add_suffix(f"_{m}").rename(columns={f"game_id_{m}": "game_id"}),
                        on="game_id")
        return d.dropna(subset=["margin"])

    games = pd.read_parquet(PATHS.processed("all") / "games.parquet")[["game_id", "ot"]]
    for name, d in (("tuning", frame([tune_season])), ("test", frame(list(test_seasons)))):
        y_m, y_t = d["margin"].to_numpy(), d["total"].to_numpy()
        res = {"team_model": metrics(d["exp_margin"], d["exp_total"], d["sigma_margin"], d["sigma_total"],
                                     y_m, y_t, d["p_home_win"].to_numpy())}
        for m in SIM_MODES:
            if f"sim_margin_{m}" in d:
                res[f"sim_{m}"] = metrics(d[f"sim_margin_{m}"], d[f"sim_total_{m}"], d[f"sim_margin_sd_{m}"],
                                          d[f"sim_total_sd_{m}"], y_m, y_t, d[f"sim_p_home_win_{m}"].to_numpy())
                res[f"sim_{m}"]["margin_crps_empirical"] = float(d[f"sim_margin_crps_{m}"].mean())
                res[f"sim_{m}"]["total_crps_empirical"] = float(d[f"sim_total_crps_{m}"].mean())
                res[f"sim_{m}"]["p_ot"] = float(d[f"sim_ot_{m}"].mean())
                res[f"sim_{m}"]["poss_per_team"] = float(d[f"sim_poss_{m}"].mean())
        res["actual_ot_rate"] = float(games[games["game_id"].isin(d["game_id"])]["ot"].mean())
        rng = np.random.default_rng(1)
        bt = lambda x: np.percentile([x.sample(frac=1, replace=True, random_state=int(rng.integers(1e9))).mean()
                                      for _ in range(n_boot)], [2.5, 97.5]).tolist()
        se_team_m = (d["margin"] - d["exp_margin"]) ** 2
        se_team_t = (d["total"] - d["exp_total"]) ** 2
        gains = {}
        if "sim_margin_none" in d:
            mb = w_star * d["exp_margin"] + (1 - w_star) * d["sim_margin_none"]
            tb = w_star * d["exp_total"] + (1 - w_star) * d["sim_total_none"]
            sm = np.sqrt(w_star * d["sigma_margin"] ** 2 + (1 - w_star) * d["sim_margin_sd_none"] ** 2)
            st_ = np.sqrt(w_star * d["sigma_total"] ** 2 + (1 - w_star) * d["sim_total_sd_none"] ** 2)
            res["posthoc_blend"] = metrics(mb, tb, sm, st_, y_m, y_t)
            res["posthoc_blend"]["weight_team_model"] = w_star
            gains["margin_posthoc_blend"] = se_team_m - (d["margin"] - mb) ** 2
            gains["total_posthoc_blend"] = se_team_t - (d["total"] - tb) ** 2
        for m in SIM_MODES:
            if f"sim_margin_{m}" in d:
                gains[f"margin_sim_{m}"] = se_team_m - (d["margin"] - d[f"sim_margin_{m}"]) ** 2
                gains[f"total_sim_{m}"] = se_team_t - (d["total"] - d[f"sim_total_{m}"]) ** 2
        # paired bootstrap of the squared-error reduction vs the team model (> 0: better than the team model)
        res["bootstrap_mse_gain_vs_team_model"] = {k: {"mean": float(v.mean()), "ci95": bt(v)} for k, v in gains.items()}
        out[name] = res
    out["blend_weight"] = w_star
    out["blend_curve_tuning"] = integ.get("curve", {})
    return out


def choose_blend_weight(season: str = TUNING_SEASON, log=print) -> dict:
    """Team-model weight ``w`` of the simulator's blend environment (target PPP = w x team model + (1 - w) x
    bottom-up), chosen on the tuning season. The blend's mean margin and total are, to first order, the same
    mixture of the team-constrained and bottom-up backtests' means, so the weight is chosen on that mixture
    (minimising margin RMSE + total RMSE); the shipped blend environment is then backtested with it."""
    _, Tn = load_sim("none", [season])
    _, Tt = load_sim("team", [season])
    d = Tn[["game_id", "margin", "total", "sim_margin", "sim_total"]].merge(
        Tt[["game_id", "sim_margin", "sim_total"]], on="game_id", suffixes=("_none", "_team"))
    rmse = lambda e: float(np.sqrt((e ** 2).mean()))
    curve = {}
    for w in np.round(np.linspace(0, 1, 21), 2):
        m = w * d["sim_margin_team"] + (1 - w) * d["sim_margin_none"]
        t = w * d["sim_total_team"] + (1 - w) * d["sim_total_none"]
        curve[float(w)] = rmse(d["margin"] - m) + rmse(d["total"] - t)
    w_star = float(min(curve, key=curve.get))
    out = {"blend_weight": w_star, "season": season, "games": int(len(d)), "curve": curve,
           "method": "argmin over w of margin RMSE + total RMSE of w*team-constrained + (1-w)*bottom-up means"}
    PATHS.ensure(PATHS.models)
    (PATHS.models / "integration.json").write_text(json.dumps(out, indent=1))
    log(f"[blend weight] {season}: w = {w_star:.2f} (team-model share), {len(d)} games")
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
    ax = PATHS.models / "availability_experiment.json"
    if ax.exists():
        summary["availability_experiment"] = json.loads(ax.read_text())
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
    return summary


def participation_analysis(season: str = TUNING_SEASON, n_games: int = 600, seed: int = 3) -> dict:
    """Selection effects: how well pregame participation probabilities for *fringe* players (whose appearance
    is not known before tip-off) match who actually appeared, using the official 2020-24 availability data
    (played / dressed-but-unused / out). Rows are the fringe candidates listed by the pregame rule; players
    added only because they appeared are counted separately and excluded. Probabilities are the simulated
    ones (after the team's minutes budget). Rotation players are treated as known from the injury report."""
    from ..sim.engine import GameSimulator
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
        sim = GameSimulator(spec)                         # p_play after the minutes budget, as simulated
        for t, team in enumerate((spec.home, spec.away)):
            for i, p in enumerate(team.players):
                rows.append({"game_id": int(gid), "person_id": p.person_id, "rotation": p.rotation,
                             "in_pool": p.in_pool, "p_play": float(sim.p_play[t, i]), "p_play_raw": p.p_play,
                             "status": status.get(p.person_id, "absent_from_box")})
    d = pd.DataFrame(rows)
    n_added = int((~d["rotation"] & ~d["in_pool"]).sum())
    fr = d[~d["rotation"] & d["in_pool"]].copy()     # listed by the pregame rule (not because they appeared)
    fr["played"] = (fr["status"] == "played").astype(float)
    fr["bin"] = pd.cut(fr["p_play"], [0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
    cal = fr.groupby("bin", observed=True).agg(pred=("p_play", "mean"), obs=("played", "mean"), n=("played", "size"))
    av_counts = av["status"].value_counts().to_dict()
    return {"season": season, "games": int(len(ids)), "fringe_added_because_appeared": n_added,
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
    (sum of negative-binomial variances, mean-binned dispersion fitted on the tuning season) vs the realised error. The possession
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
        disp = BinnedDispersion(tune["pts__eb"].to_numpy(float), tune["pts"].to_numpy(float))
        test["_var"] = test[col] + test[col] ** 2 / disp(test[col].to_numpy(float))
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
