"""Structural constants of the possession simulator, estimated from pre-test seasons only.

All quantities come from 2021-22 .. 2023-24 (two history seasons and the 2023-24 tuning season);
the 2024-25 and 2025-26 test seasons are never read here.

* rotation templates: P(on court | minute of game, starter, projected-minutes tier)
* garbage time: starters' on-court rate in Q4 as a function of |margin| and time left,
  relative to close games
* possession time: duration model ``a + b * n_oreb`` and the Gamma shape of the base part
* league event rates not tied to a player's usage: team rebounds, non-FT defensive fouls,
  technical free throws, late-game intentional fouling
* end of games: possession durations by score and time left, last-shot holds, shot selection and
  accuracy in the final 24 seconds
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


LATE_LEAD_BUCKETS = ((1, 3), (4, 6), (7, 9), (10, 99))


def _lead_group(lead: np.ndarray) -> np.ndarray:
    """0 leading, 1 tied, 2 down 1-3, 3 down 4+ (offense perspective)."""
    return np.where(lead >= 1, 0, np.where(lead == 0, 1, np.where(lead >= -3, 2, 3)))


def _lead_index(lead: np.ndarray) -> np.ndarray:
    """0 leading, 1 tied, 2 down 1, 3 down 2, 4 down 3, 5 down 4+ (for shot selection)."""
    return np.where(lead >= 1, 0, np.where(lead >= -3, 1 - lead, 5))


def late_game_fouling(poss: pd.DataFrame) -> dict:
    """P(a possession of the *leading* team includes a non-and-one FT trip) late in the 4th quarter by lead size
    and time left, vs. the same rate earlier in the 4th quarter (the excess is intentional fouling).

    Windows: the final minute (``leading``, kept for reference) and its two parts, 24-60 s left and the final 24 s
    (``leading_by_window``). The lead is the offense's lead before the possession's first event (its own points
    excluded) and the possession starts when the previous one ended (``possessions.start_s``), matching the
    simulator, which decides on intentional fouls at the start of a possession from the score at that moment."""
    ps = poss[poss["period"] == 4]
    d = pd.DataFrame({"start_s": ps["start_s"].to_numpy(), "lead": ps["lead_start"].to_numpy(),
                      "trip": (ps["n_trips"] > 0).to_numpy()})
    late = d[d["start_s"] >= 2820]
    early = d[(d["start_s"] >= 2160) & (d["start_s"] < 2700)]
    base = float(early["trip"].mean())
    rates, by_window = {}, {"24-60": {}, "0-24": {}}
    for lo, hi in LATE_LEAD_BUCKETS:
        x = late[(late["lead"] >= lo) & (late["lead"] <= hi)]
        rates[f"{lo}-{hi}"] = {"p_trip": float(x["trip"].mean()), "n": int(len(x))}
        for w, (a, b) in (("24-60", (2820, 2856)), ("0-24", (2856, 2881))):
            y = x[(x["start_s"] >= a) & (x["start_s"] < b)]
            by_window[w][f"{lo}-{hi}"] = {"p_trip": float(y["trip"].mean()) if len(y) else base, "n": int(len(y))}
    tie = late[late["lead"] <= 0]
    return {"baseline_p_trip_q4": base, "leading": rates, "leading_by_window": by_window,
            "not_leading_p_trip": float(tie["trip"].mean())}


def end_game(poss: pd.DataFrame) -> dict:
    """End-of-game behaviour in the 4th quarter and overtime (pre-test seasons):

    * ``dur``: mean possession duration (s) of possessions *not* ending in free throws, by time left
      (60-120 s, 24-60 s) and score (leading / tied / down 1-3 / down 4+): leading teams run the clock,
      trailing teams hurry;
    * ``hold``: share of possessions starting with 4-24 s left that end at the buzzer (last shot), tied and
      down 1-3; ``quick_dur``: mean duration of the others;
    * ``three_share``: share of field-goal attempts that are threes with 4-24 s and 0-4 s left, by lead
      (leading / tied / down 1 / down 2 / down 3 / down 4+);
    * ``fg``: make rates with 0-4 s left (2PA, 3PA) and of threes with 4-24 s left (late, contested shots).
    """
    pend = np.where(poss["period"] <= 4, 2880.0, 2880.0 + 300.0 * (poss["period"] - 4))
    p = poss[poss["period"] >= 4].copy()
    p["pend"] = pend[poss["period"].to_numpy() >= 4]
    p["tleft"] = p["pend"] - p["start_s"]
    p["dur"] = p["end_s"] - p["start_s"]
    p["grp"] = _lead_group(p["lead_start"].to_numpy())
    out: dict = {"groups": ["leading", "tied", "down 1-3", "down 4+"]}
    nf = p[p["n_trips"] == 0]
    dur = {}
    for w, (lo, hi) in (("60-120", (60, 120)), ("24-60", (24, 60))):
        x = nf[(nf["tleft"] > lo) & (nf["tleft"] <= hi)]
        dur[w] = [float(x.loc[x["grp"] == k, "dur"].mean()) for k in range(4)]
    out["dur"] = dur
    w3 = p[(p["tleft"] > 4) & (p["tleft"] <= 24) & (p["n_trips"] == 0)]
    buzz = w3["end_s"] >= w3["pend"] - 3.0
    out["hold"] = {"tied": float(buzz[w3["grp"] == 1].mean()), "down_1_3": float(buzz[w3["grp"] == 2].mean())}
    out["quick_dur"] = float(w3.loc[~buzz & (w3["grp"] >= 2), "dur"].mean())
    # shot selection and accuracy from the events
    shots = []
    for s in CONST_SEASONS:
        ev = pd.read_parquet(PATHS.staged(s) / "events.parquet",
                             columns=["game_id", "period", "elapsed_s", "action_type", "shot_result", "points",
                                      "team_id", "score_home", "score_away"])
        g = pd.read_parquet(PATHS.processed(s) / "games.parquet")[["game_id", "home_team_id"]]
        ev = ev[(ev["period"] >= 4) & ev["action_type"].isin(["2pt", "3pt"])].merge(g, on="game_id")
        tl = np.where(ev["period"] == 4, 2880.0, 2880.0 + 300.0 * (ev["period"] - 4)) - ev["elapsed_s"]
        ev = ev[tl <= 600].assign(tleft=tl[tl <= 600])
        home = ev["team_id"] == ev["home_team_id"]
        made = ev["shot_result"] == "Made"
        after = np.where(home, ev["score_home"] - ev["score_away"], ev["score_away"] - ev["score_home"])
        shots.append(pd.DataFrame({"tleft": ev["tleft"].to_numpy(), "three": (ev["action_type"] == "3pt").to_numpy(),
                                   "made": made.to_numpy(),
                                   "lead": after - np.where(made, ev["points"].fillna(0), 0)}))
    sh_all = pd.concat(shots, ignore_index=True)
    # late-game make-rate shifts (logit) by time left x score group x shot type, relative to the same group
    # with 2-10 minutes left in the 4th quarter / overtime (shrunk toward 0 with 50 pseudo-attempts)
    sh_all["grp"] = _lead_group(sh_all["lead"].to_numpy())
    base_w = sh_all[sh_all["tleft"] > 120]
    late_off = []
    for lo, hi in ((60, 120), (24, 60), (4, 24)):
        x = sh_all[(sh_all["tleft"] > lo) & (sh_all["tleft"] <= hi)]
        rows = []
        for grp in range(4):
            cell = []
            for three in (False, True):
                b = base_w[(base_w["grp"] == grp) & (base_w["three"] == three)]["made"].mean()
                y = x[(x["grp"] == grp) & (x["three"] == three)]["made"]
                r = (y.sum() + 50 * b) / (len(y) + 50)
                cell.append(float(np.log(r / (1 - r)) - np.log(b / (1 - b))))
            rows.append(cell)
        late_off.append(rows)
    out["late_make_logit"] = late_off        # [window 60-120 / 24-60 / 4-24][group][2PA, 3PA]
    sh = sh_all[sh_all["tleft"] <= 24].copy()
    sh["li"] = _lead_index(sh["lead"].to_numpy())
    out["three_share"] = {w: [float(x.loc[x["li"] == k, "three"].mean()) for k in range(6)]
                          for w, x in (("4-24", sh[sh["tleft"] > 4]), ("0-4", sh[sh["tleft"] <= 4]))}
    out["lead_index"] = ["leading", "tied", "down 1", "down 2", "down 3", "down 4+"]
    w4 = sh[sh["tleft"] <= 4]
    out["fg"] = {"heave2": float(w4.loc[~w4["three"], "made"].mean()), "heave3": float(w4.loc[w4["three"], "made"].mean()),
                 "late3": float(sh.loc[(sh["tleft"] > 4) & sh["three"], "made"].mean()),
                 "late2": float(sh.loc[(sh["tleft"] > 4) & ~sh["three"], "made"].mean())}
    # final-seconds make rates by score situation and shot type (shrunk toward the pooled rate, 20 attempts):
    # a contested last shot in a close game goes in far less often than a garbage-time layup
    k = 20.0
    heave = []
    for li in range(6):
        row = []
        for three, pooled in ((False, out["fg"]["heave2"]), (True, out["fg"]["heave3"])):
            x = w4[(w4["li"] == li) & (w4["three"] == three)]
            row.append(float((x["made"].sum() + k * pooled) / (len(x) + k)))
        heave.append(row)
    out["heave_fg"] = heave          # [lead index][2PA, 3PA]
    return out


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
    c["end_game"] = end_game(poss)
    log(f"[sim constants] templates {len(c['templates'])}, poss time a={a:.2f}s b={b:.2f}s/OREB shape={shape:.2f}, "
        f"team reb share {c['team_reb_share']:.3f}, non-FT def fouls/play {c['nonft_def_fouls_per_opp_play']:.4f}, "
        f"tech FTA/team-game {c['tech_fta_per_team_game']:.3f}")
    PATHS.ensure(PATHS.models)
    (PATHS.models / "sim_constants.json").write_text(json.dumps(c, indent=1))
    return c
