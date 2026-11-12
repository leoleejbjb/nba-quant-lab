"""How good are truly pregame projections? Scored on a rehearsal of the live season (scripts/rehearse_live.py).

Run inside a rehearsal folder (QUANTLAB_DATA_DIR / QUANTLAB_LIVE_SEASON set by the script): every unplayed game with
pregame inputs is projected the way an upcoming game is (``Projector`` in pregame mode: projected starters,
availability from missed-game streaks, lineup-blind minutes, nothing from the game), then compared with what
happened and with the backtested as-of projection of the same game (who appeared and who started known), both read
from the main data folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STATS = ("min", "pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "pra")


def project_all(n_sims: int, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    from ..live import NO_REPORT, Projector
    pr = Projector()
    pr.overrides = dict(NO_REPORT)      # nothing a user saved in the folder (it may know what happened)
    sc = pr.store.schedule
    gids = [int(g) for g in sc.index if not pr.store.is_final(g) and pr.store.pregame_problem(g) is None]
    cutoff = pr.store.games["game_date"].max()
    players, teams = [], []
    for k, gid in enumerate(sorted(gids)):
        proj = pr.project(gid, n=n_sims, environment="blend")
        res, spec = proj.result, proj.spec
        info = pr.game_info(gid)
        days = int((pd.Timestamp(info["game_date"]) - cutoff).days) - 1
        sc_ = res.score.astype(float)
        teams.append({"game_id": gid, "days_ahead": days, "pre_margin": float((sc_[:, 0] - sc_[:, 1]).mean()),
                      "pre_total": float(sc_.sum(axis=1).mean()), "pre_p_home": float((sc_[:, 0] > sc_[:, 1]).mean())})
        for t, team in enumerate((spec.home, spec.away)):
            for i, p in enumerate(team.players):
                m = res.stat("min")[:, t, i]
                play = m > 0
                row = {"game_id": gid, "team_id": team.team_id, "person_id": p.person_id, "days_ahead": days,
                       "starter_proj": p.starter, "rotation_proj": p.rotation, "p_play": float(play.mean())}
                for s in STATS:
                    x = res.stat(s)[:, t, i]
                    row[f"pre_{s}"] = float(x[play].mean()) if play.any() else np.nan
                row["pre_min_uncond"] = float(m.mean())
                players.append(row)
        if (k + 1) % 10 == 0:
            log(f"[pregame] projected {k + 1}/{len(gids)} games")
    return pd.DataFrame(players), pd.DataFrame(teams)


def score(players: pd.DataFrame, teams: pd.DataFrame, main: Path, season: str) -> dict:
    pg = pd.read_parquet(main / "processed" / "all" / "player_games.parquet",
                         columns=["game_id", "team_id", "person_id", "played", "starter", "min", "pts", "reb", "ast",
                                  "fg3m", "stl", "blk", "tov"])
    pg = pg[pg["game_id"].isin(teams["game_id"])]
    pg["pra"] = pg["pts"] + pg["reb"] + pg["ast"]
    act = pg[pg["played"]]
    # backtested as-of projections (who appeared / who started known) of the same games
    bt = pd.read_parquet(main / "backtest" / f"sim_blend_{season}_players.parquet",
                         columns=["game_id", "person_id", "stat", "mean"])
    bt = bt[bt["game_id"].isin(teams["game_id"])].pivot_table(index=["game_id", "person_id"], columns="stat",
                                                              values="mean").add_prefix("asof_").reset_index()
    j = act.merge(players, on=["game_id", "team_id", "person_id"], how="left").merge(
        bt, on=["game_id", "person_id"], how="left")
    covered = j["pre_min"].notna()
    out = {"games": int(teams["game_id"].nunique()), "player_games_played": int(len(act)),
           "roster_coverage": float(covered.mean()),
           "minutes_share_covered": float(j.loc[covered, "min"].sum() / j["min"].sum())}
    common = j[covered & j["asof_pts"].notna()]
    out["common_player_games"] = int(len(common))
    out["mae"] = {s: {"pregame": float((common[f"pre_{s}"] - common[s]).abs().mean()),
                      "asof": float((common[f"asof_{s}"] - common[s]).abs().mean())}
                  for s in STATS if f"asof_{s}" in common}
    # every appearance, a player the pregame projection left out counting as projected 0
    out["mae_pregame_all_appearances"] = {s: float((j[f"pre_{s}"].fillna(0) - j[s]).abs().mean()) for s in STATS}
    out["bias"] = {s: {"pregame": float((common[f"pre_{s}"] - common[s]).mean()),
                       "asof": float((common[f"asof_{s}"] - common[s]).mean())} for s in ("min", "pts", "pra")}
    out["minutes_bias_by_actual_role"] = {
        role: {"pregame": float((grp["pre_min"] - grp["min"]).mean()),
               "asof": float((grp["asof_min"] - grp["min"]).mean()), "n": int(len(grp))}
        for role, grp in (("starters", common[common["starter"]]), ("bench", common[~common["starter"]]))}
    # players who appeared but were not projected to: presumed out after a missed-game streak, or not in the pool
    from ..config import PATHS
    from ..features.asof import PRESUMED_AVAILABLE_STREAK
    cand = pd.read_parquet(PATHS.features / "player_asof.parquet",
                           columns=["game_id", "team_id", "person_id", "missed_streak"])
    miss = j[~covered].merge(cand, on=["game_id", "team_id", "person_id"], how="left")
    out["not_projected"] = {"player_games": int(len(miss)),
                            "presumed_out_after_streak": int((miss["missed_streak"] > PRESUMED_AVAILABLE_STREAK).sum()),
                            "not_in_pool": int(miss["missed_streak"].isna().sum()),
                            "minutes_share": float(miss["min"].sum() / j["min"].sum())}
    # starters: projected five vs actual five
    st_act = pg[pg["starter"] & pg["played"]].groupby(["game_id", "team_id"])["person_id"].apply(set)
    st_pre = players[players["starter_proj"]].groupby(["game_id", "team_id"])["person_id"].apply(set)
    hits = [len(st_pre.get(k, set()) & v) for k, v in st_act.items()]
    out["starters_correct_of_5"] = float(np.mean(hits)) if hits else float("nan")
    # availability: projected rotation players who did not appear
    rot = players[players["rotation_proj"]].merge(act[["game_id", "person_id"]].assign(appeared=True),
                                                  on=["game_id", "person_id"], how="left")
    rot["appeared"] = rot["appeared"].fillna(False).astype(bool)
    out["rotation_projected_not_appearing"] = float((~rot["appeared"]).mean())
    out["mean_p_play_rotation"] = float(rot["p_play"].mean())
    # team level: pregame vs backtested as-of vs the team model
    g = pd.read_parquet(main / "processed" / "all" / "games.parquet", columns=["game_id", "home_pts", "away_pts"])
    tb = pd.read_parquet(main / "backtest" / f"sim_blend_{season}_teams.parquet",
                         columns=["game_id", "sim_margin", "sim_total"])
    tp = pd.read_parquet(main / "models" / "team_predictions.parquet", columns=["game_id", "exp_margin", "exp_total"])
    t = teams.merge(g, on="game_id").merge(tb, on="game_id", how="left").merge(tp, on="game_id", how="left")
    y_m, y_t = t["home_pts"] - t["away_pts"], t["home_pts"] + t["away_pts"]
    rmse = lambda e: float(np.sqrt((e ** 2).mean()))     # noqa: E731
    out["team"] = {"margin_rmse": {"pregame": rmse(t["pre_margin"] - y_m), "asof": rmse(t["sim_margin"] - y_m),
                                   "team_model_day_before": rmse(t["exp_margin"] - y_m)},
                   "total_rmse": {"pregame": rmse(t["pre_total"] - y_t), "asof": rmse(t["sim_total"] - y_t),
                                  "team_model_day_before": rmse(t["exp_total"] - y_t)}}
    by_day = []
    for d, grp in common.groupby("days_ahead"):
        by_day.append({"days_ahead": int(d), "n": int(len(grp)),
                       "pts_mae_pregame": float((grp["pre_pts"] - grp["pts"]).abs().mean()),
                       "pts_mae_asof": float((grp["asof_pts"] - grp["pts"]).abs().mean()),
                       "min_mae_pregame": float((grp["pre_min"] - grp["min"]).abs().mean()),
                       "min_mae_asof": float((grp["asof_min"] - grp["min"]).abs().mean())})
    out["by_days_ahead"] = by_day
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--main-data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--score-only", action="store_true", help="re-score the projections saved next to --out")
    a = ap.parse_args()
    from ..config import LIVE_SEASON
    pp, tp = a.out.with_suffix(".players.parquet"), a.out.with_suffix(".teams.parquet")
    if a.score_only:
        players, teams = pd.read_parquet(pp), pd.read_parquet(tp)
    else:
        players, teams = project_all(a.sims)
        players.to_parquet(pp, index=False)
        teams.to_parquet(tp, index=False)
    res = score(players, teams, a.main_data, LIVE_SEASON)
    a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
