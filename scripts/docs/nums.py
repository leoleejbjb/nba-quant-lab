"""Every number the narrative docs quote, flattened from data/backtest/summary.json (plus a few model files).

    from nums import ROOT, load;  N = load()

Keys look like ``test.gbm.pts.mae``, ``test.team.sim_blend.margin_rmse``, ``test.gain.margin_sim_blend``,
``test.rel.reb|sim_blend`` (points), ``lv.test.player.blk.rel``; ``python scripts/docs/nums.py [summary.json] [substr..]``
prints them.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def load(path=None) -> dict:
    s = json.loads(Path(path or ROOT / "data/backtest/summary.json").read_text())
    N = {}
    for split in ("test", "tuning"):
        pc = pd.DataFrame(s[f"players_{split}"])
        N[f"n_players_{split}"] = int(pc["n"].max())
        for r in pc.to_dict("records"):
            for k in ("mae", "rmse", "crps", "pit_dev", "cov80", "cov80_inclusive", "p_pit_below_half", "bias", "logscore",
                      "mean_pred", "mean_obs", "n"):
                if k in r and r[k] == r[k]:
                    N[f"{split}.{r['model']}.{r['stat']}.{k}"] = r[k]
        for r in s[f"minutes_{split}"]:
            for k in ("mae", "rmse", "bias", "mae_rotation", "n"):
                N[f"{split}.min.{r['model']}.{k}"] = r[k]
        for k, v in s.get(f"reliability_{split}", {}).items():
            if "mean_gap" in v:
                N[f"{split}.rel.{k}"] = 100 * v["mean_gap"]
            for off, gap in (v.get("line_gaps") or {}).items():
                N[f"{split}.rel.{k}.line{off}"] = None if gap is None else 100 * gap
                N[f"{split}.rel.{k}.line{off}.n"] = v["line_n"][off]
    tm = s["teams"]
    for split in ("test", "tuning"):
        for k, v in tm[split].items():
            if isinstance(v, dict) and "margin_rmse" in v:
                for m, x in v.items():
                    N[f"{split}.team.{k}.{m}"] = x
        N[f"{split}.actual_ot"] = tm[split]["actual_ot_rate"]
        for k, v in tm[split].get("bootstrap_mse_gain_vs_team_model", {}).items():
            N[f"{split}.gain.{k}"] = v["mean"]
            N[f"{split}.gain.{k}.lo"], N[f"{split}.gain.{k}.hi"] = v["ci95"]
    N["blend_weight"] = tm["blend_weight"]
    for split, lv in (s.get("levels") or {}).items():
        for st, v in lv.get("player", {}).items():
            for k, x in v.items():
                N[f"lv.{split}.player.{st}.{k}"] = x
        for st, v in lv.get("team", {}).items():
            if isinstance(v, dict):
                for k, x in v.items():
                    N[f"lv.{split}.team.{st}.{k}"] = x
        for k, x in lv.get("overtime", {}).items():
            N[f"lv.{split}.ot.{k}"] = x
    co = s["consistency"]
    for m in ("eb", "gbm"):
        for k, v in co[m].items():
            N[f"cons.{m}.{k}"] = v
    N["cons.team_model_rmse"] = co["team_model_rmse_team_points"]
    for st, v in s["opponent_ablation"].items():
        N[f"abl.{st}.gain"] = v["mae_gain"]
        N[f"abl.{st}.pct"] = v["mae_gain_pct"]
        N[f"abl.{st}.lo"], N[f"abl.{st}.hi"] = v["mae_gain_ci95"]
        N[f"abl.{st}.with"], N[f"abl.{st}.without"] = v["with_opponent"]["mae"], v["without_opponent"]["mae"]
    pa = s["participation"]
    N["part.rows"], N["part.played"], N["part.mean_p"] = pa["fringe_rows"], pa["fringe_played_share"], pa["fringe_mean_p_play"]
    N["part.added"] = pa.get("fringe_added_because_appeared")
    N["part.pred"], N["part.obs"], N["part.n"] = pa["calibration"]["pred"], pa["calibration"]["obs"], pa["calibration"]["n"]
    ax = s["availability_experiment"]
    for k, v in ax.items():
        if isinstance(v, (int, float)):
            N[f"ax.{k}"] = v
    N["ax.dnp_share"] = ax["absent_rotation_status_share"].get("dnp_cd")
    c = s["cost"]
    N["cost.n"], N["cost.us"], N["cost.sim_s"], N["cost.cal_s"] = c["n_sims"], c["us_per_game"], c["simulate_s"], c["calibration_s"]
    N["cost.inv_all"] = c["invariants"]["all"]
    # data reconstruction quality (official box-score comparison, 2020-24)
    for season in ("2020-21", "2021-22", "2022-23", "2023-24"):
        p = ROOT / f"data/processed/{season}/quality.json"
        if p.exists():
            q = json.loads(p.read_text()).get("official_box", {})
            N.update({f"q.{season}.{k}": v for k, v in q.items() if isinstance(v, (int, float))})
    N.update({f"q.{k[len('q.2023-24.'):]}": v for k, v in list(N.items()) if k.startswith("q.2023-24.")})
    return N


if __name__ == "__main__":
    import sys
    N = load(sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".json") else None)
    keys = [a for a in sys.argv[1:] if not a.endswith(".json")]
    for k, v in N.items():
        if not keys or any(x in k for x in keys):
            print(k, v)
