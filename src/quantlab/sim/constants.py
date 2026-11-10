"""Structural constants of the possession simulator, estimated from pre-test seasons only.

All quantities come from 2021-22 .. 2023-24 (history + tuning seasons); the
2024-25 and 2025-26 test seasons are never read here.

* rotation templates: P(on court | minute of game, starter, projected-minutes tier)
* garbage time: starters' on-court rate in Q4 as a function of |margin| and time left,
  relative to close games
* possession time: duration model ``a + b * n_oreb`` and the Gamma shape of the base part
* league event rates not tied to a player's usage: team rebounds, non-FT defensive fouls,
  technical free throws, late-game intentional fouling
* closing: P(on court) in the last 4 minutes of close games by minutes rank
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import PATHS

CONST_SEASONS = ("2021-22", "2022-23", "2023-24")
N_SLOTS = 48                                   # one-minute slots in regulation
TIERS = [0, 12, 20, 28, 34, 60]                # minutes tiers for templates
MARGIN_BUCKETS = [0, 6, 11, 16, 21, 26, 31, 100]


def slot_occupancy(stints: pd.DataFrame) -> pd.DataFrame:
    """Seconds on court per player-game per regulation minute (columns s0..s47)."""
    rows = []
    for side in ("home", "away"):
        lin = stints[[f"{side}_lineup", "game_id", "start_s", "end_s", "valid"]].copy()
        lin = lin[lin["valid"] & (lin["start_s"] < 2880)]
        lin["pid"] = lin[f"{side}_lineup"].str.split("-")
        lin = lin.explode("pid")
        lin["pid"] = lin["pid"].astype("int64")
        rows.append(lin[["game_id", "pid", "start_s", "end_s"]])
    x = pd.concat(rows, ignore_index=True)
    edges = np.arange(N_SLOTS + 1) * 60.0
    st = x["start_s"].to_numpy()[:, None]
    en = np.minimum(x["end_s"].to_numpy(), 2880.0)[:, None]
    occ = np.clip(np.minimum(en, edges[1:]) - np.maximum(st, edges[:-1]), 0, 60).astype(np.float32)
    o = pd.DataFrame(occ, columns=[f"s{k}" for k in range(N_SLOTS)])
    o["game_id"], o["person_id"] = x["game_id"].to_numpy(), x["pid"].to_numpy()
    return o.groupby(["game_id", "person_id"], sort=False).sum().reset_index()


def late_game_fouling(poss: pd.DataFrame) -> dict:
    """P(a possession of the *leading* team includes a non-and-one FT trip) in the final minute of regulation
    by lead size, vs. the same rate earlier in the 4th quarter (the excess is intentional fouling)."""
    out_rows = []
    for s in CONST_SEASONS:
        ev = pd.read_parquet(PATHS.staged(s) / "events.parquet",
                             columns=["game_id", "period", "elapsed_s", "action_type", "sub_type", "descriptor",
                                      "team_id", "score_home", "score_away"])
        g = pd.read_parquet(PATHS.processed(s) / "games.parquet")[["game_id", "home_team_id"]]
        ev = ev[ev["period"] == 4]
        trips = ev[(ev["action_type"] == "freethrow") & ev["sub_type"].isin(["1 of 2", "1 of 3"]) &
                   (ev["descriptor"] != "technical")][["game_id", "elapsed_s", "team_id"]]
        ps = poss[(poss["game_id"].isin(g["game_id"])) & (poss["period"] == 4)].copy()
        ps = ps.merge(g, on="game_id")
        # running score at possession start
        ev = ev.sort_values(["game_id", "elapsed_s"])
        res = []
        for gid, pg_ in ps.groupby("game_id"):
            e = ev[ev["game_id"] == gid]
            if e.empty:
                continue
            idx = np.searchsorted(e["elapsed_s"].to_numpy(), pg_["start_s"].to_numpy(), side="right") - 1
            idx = np.clip(idx, 0, len(e) - 1)
            sh, sa = e["score_home"].to_numpy()[idx], e["score_away"].to_numpy()[idx]
            home_off = (pg_["offense_team_id"] == pg_["home_team_id"]).to_numpy()
            lead = np.where(home_off, sh - sa, sa - sh)
            t = trips[trips["game_id"] == gid]
            has = np.zeros(len(pg_), dtype=bool)
            for j, (st_, en_, off) in enumerate(zip(pg_["start_s"], pg_["end_s"], pg_["offense_team_id"])):
                has[j] = ((t["elapsed_s"] >= st_) & (t["elapsed_s"] <= en_) & (t["team_id"] == off)).any()
            res.append(pd.DataFrame({"start_s": pg_["start_s"].to_numpy(), "lead": lead, "trip": has}))
        out_rows.append(pd.concat(res))
    d = pd.concat(out_rows)
    late = d[d["start_s"] >= 2820]
    early = d[(d["start_s"] >= 2160) & (d["start_s"] < 2700)]
    base = float(early["trip"].mean())
    rates = {}
    for lo, hi in ((1, 3), (4, 6), (7, 9), (10, 99)):
        x = late[(late["lead"] >= lo) & (late["lead"] <= hi)]
        rates[f"{lo}-{hi}"] = {"p_trip": float(x["trip"].mean()), "n": int(len(x))}
    tie = late[late["lead"] <= 0]
    return {"baseline_p_trip_q4": base, "leading": rates, "not_leading_p_trip": float(tie["trip"].mean())}


def compute_constants(log=print) -> dict:
    allp = PATHS.processed("all")
    pg = pd.read_parquet(allp / "player_games.parquet")
    tg = pd.read_parquet(allp / "team_games.parquet")
    pg = pg[pg["season"].isin(CONST_SEASONS) & pg["played"] & pg["analysis_ok"]]
    tg = tg[tg["season"].isin(CONST_SEASONS)]
    occs, poss_all, st_all = [], [], []
    for s in CONST_SEASONS:
        st = pd.read_parquet(PATHS.processed(s) / "stints.parquet")
        st_all.append(st)
        occs.append(slot_occupancy(st))
        poss_all.append(pd.read_parquet(PATHS.processed(s) / "possessions.parquet"))
    occ = pd.concat(occs, ignore_index=True)
    poss = pd.concat(poss_all, ignore_index=True)
    stints = pd.concat(st_all, ignore_index=True)
    c: dict = {"seasons": list(CONST_SEASONS)}

    # ---------------- rotation templates ----------------
    m = pg[["game_id", "person_id", "team_id", "starter", "min"]].merge(occ, on=["game_id", "person_id"])
    m["tier"] = pd.cut(m["min"], TIERS, labels=False, right=False)
    sc = [f"s{k}" for k in range(N_SLOTS)]
    tmpl = m.groupby(["starter", "tier"])[sc].mean() / 60.0
    c["templates"] = {f"{int(st)}_{int(t)}": [round(float(v), 4) for v in row]
                      for (st, t), row in tmpl.iterrows()}
    c["template_tiers"] = TIERS

    # ---------------- garbage time ----------------
    # margin (team perspective) at each Q4 minute from stints; starters on court per team-minute
    starters = pg[pg["starter"]][["game_id", "person_id", "team_id"]]
    so = starters.merge(occ, on=["game_id", "person_id"])
    son = so.groupby(["game_id", "team_id"])[sc].sum() / 60.0          # starters on court (0..5)
    games = pd.read_parquet(allp / "games.parquet")[["game_id", "home_team_id"]]
    st = stints.merge(games, on="game_id")
    st = st[st["valid"]]
    # margin at minute k = margin_start of the stint covering 60k+30 (home perspective)
    rows = []
    for gid, g in st.groupby("game_id"):
        starts, marg = g["start_s"].to_numpy(), g["margin_start"].to_numpy()
        pts_h, pts_a = g["home_pts"].to_numpy(), g["away_pts"].to_numpy()
        ends = g["end_s"].to_numpy()
        mid = np.arange(36, N_SLOTS) * 60.0 + 30.0
        idx = np.searchsorted(starts, mid, side="right") - 1
        frac = np.clip((mid - starts[idx]) / np.maximum(ends[idx] - starts[idx], 1), 0, 1)
        mm = marg[idx] + frac * (pts_h[idx] - pts_a[idx])
        rows.append((gid, mm))
    marg = pd.DataFrame({"game_id": [r[0] for r in rows], "m": [r[1] for r in rows]}).set_index("game_id")["m"]
    rec = []
    home_of = games.set_index("game_id")["home_team_id"].to_dict()
    marg_d = marg.to_dict()
    vals = son[[f"s{k}" for k in range(36, N_SLOTS)]].to_numpy()
    for (gid, team), row in zip(son.index, vals):
        if gid not in marg_d:
            continue
        hm = marg_d[gid]
        sign = 1.0 if home_of[gid] == team else -1.0
        for j, k in enumerate(range(36, N_SLOTS)):
            rec.append((k, abs(sign * hm[j]), row[j]))
    gdf = pd.DataFrame(rec, columns=["slot", "abs_margin", "starters_on"])
    gdf["mb"] = pd.cut(gdf["abs_margin"], MARGIN_BUCKETS, labels=False, right=False)
    base = gdf[gdf["mb"] == 0].groupby("slot")["starters_on"].mean()
    ratio = gdf.groupby(["slot", "mb"])["starters_on"].mean().unstack() / base.to_numpy()[:, None]
    c["garbage_ratio"] = {int(k): [round(float(v), 4) if v == v else 1.0 for v in row] for k, row in ratio.iterrows()}
    c["margin_buckets"] = MARGIN_BUCKETS

    # ---------------- possession time ----------------
    # duration = time since the previous possession (either team) ended; first of a period from its start
    p = poss[(poss["period"] <= 4)].sort_values(["game_id", "period", "end_s"]).copy()
    prev_end = p.groupby(["game_id", "period"])["end_s"].shift(1)
    p["dur"] = p["end_s"] - prev_end.fillna((p["period"] - 1) * 720.0)
    p = p[(p["dur"] > 0) & (p["dur"] <= 35) & (p["reason"] != "period_end")]
    X = np.column_stack([np.ones(len(p)), p["n_oreb"].to_numpy()])
    a, b = np.linalg.lstsq(X, p["dur"].to_numpy(), rcond=None)[0]
    base_dur = p.loc[p["n_oreb"] == 0, "dur"]
    shape = base_dur.mean() ** 2 / base_dur.var()
    c["poss_time"] = {"a": float(a), "b_per_oreb": float(b), "gamma_shape": float(shape),
                      "mean_no_oreb": float(base_dur.mean()), "mean_all": float(p["dur"].mean()),
                      "oreb_per_poss": float(p["n_oreb"].mean())}

    # ---------------- league event rates ----------------
    plays = tg["fga"] + tg["trips"] + tg["tov"]
    c["team_reb_share"] = float((tg["team_oreb"] + tg["team_dreb"]).sum() / tg["oreb_ch"].sum())
    c["team_oreb_share_of_team_reb"] = float(tg["team_oreb"].sum() / (tg["team_oreb"] + tg["team_dreb"]).sum())
    c["team_tov_share"] = float(tg["team_tov"].sum() / tg["tov"].sum())
    pf_ft = tg["opp_trips"] + tg["opp_and1"]                     # defensive fouls that produced FTs
    pf_off = pg.groupby(["game_id", "team_id"])["pf_offensive"].sum()
    tg_i = tg.set_index(["game_id", "team_id"])
    other_pf = tg_i["pf"] - pf_off.reindex(tg_i.index).fillna(0) - (tg_i["opp_trips"] + tg_i["opp_and1"])
    c["nonft_def_fouls_per_opp_play"] = float(other_pf.clip(lower=0).sum() / (tg_i["opp_fga"] + tg_i["opp_trips"] +
                                                                              tg_i["opp_tov"]).sum())
    c["tech_fta_per_team_game"] = float(pg["fta_tech"].sum() / len(tg))
    del pf_ft, plays
    c["late_game"] = late_game_fouling(poss)
    log(f"[sim constants] templates {len(c['templates'])}, poss time a={a:.2f}s b={b:.2f}s/OREB shape={shape:.2f}, "
        f"team reb share {c['team_reb_share']:.3f}, non-FT def fouls/play {c['nonft_def_fouls_per_opp_play']:.4f}, "
        f"tech FTA/team-game {c['tech_fta_per_team_game']:.3f}")
    PATHS.ensure(PATHS.models)
    (PATHS.models / "sim_constants.json").write_text(json.dumps(c, indent=1))
    return c
