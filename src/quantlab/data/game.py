"""Reconstruct one game: lineups, possessions, player & team box scores, on-court exposures.

Player box-score rules (validated against official NBA.com box scores, see
``reconcile.py`` and docs/DATA.md):

* FGA/FGM/3PA/3PM from ``2pt``/``3pt`` actions by the shooter; AST from the
  shot's ``assistPersonId``. 2025-26 records missed end-of-period heaves as
  team-level ``heave`` actions that are not field-goal attempts.
* FTA/FTM from every ``freethrow`` action (technical free throws included).
* OREB/DREB from ``rebound`` actions credited to a player.
* STL / BLK from the separate ``steal`` / ``block`` actions.
* TOV from ``turnover`` actions credited to a player (team turnovers are
  team-only). PF from ``foul`` actions other than technical fouls.
* MIN from reconstructed stints; plus-minus from stint points with free throws
  credited to the lineup at the foul.

On-court exposures (``on_*`` for the player's team, ``op_*`` for the opponent)
count events that happened while the player was on the floor: possessions,
shots, free throws, turnovers, points, live rebound chances ...
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from ..config import REGULATION_S, period_length, period_start
from .lineups import GameArrays, build_roster, player_seconds, reconstruct_lineups
from .possessions import RETAIN_FT, attribution_stints, count_possessions, ft_trip, live_rebound_mask

# Per-stint team event features accumulated for on-court exposures.
EXPO = ["poss", "pts", "fga", "fgm", "fga2", "fga3", "fgm2", "fgm3", "fta", "ftm", "tov", "ast",
        "oreb_ch", "oreb", "dreb", "pf", "blk", "stl", "trips", "and1"]
EI = {k: i for i, k in enumerate(EXPO)}

PLAYER_STATS = ["fga2", "fgm2", "fga3", "fgm3", "fgm2_ast", "fgm3_ast", "fta", "ftm", "fta_tech",
                "trips2", "trips3", "and1", "oreb", "dreb", "ast", "stl", "blk", "tov", "tov_off_foul",
                "pf", "pf_shooting", "pf_offensive", "pfd", "pfd_shooting", "blocked", "tech"]
SI = {k: i for i, k in enumerate(PLAYER_STATS)}


def _and_one_flags(g: GameArrays) -> np.ndarray:
    """True for a '1 of 1' free throw that completes an and-one (made FG by the same team, same clock)."""
    out = np.zeros(g.n, dtype=bool)
    last_make: dict = {}
    for i in range(g.n):
        at = g.at[i]
        if at in ("2pt", "3pt") and g.shot[i] == "Made":
            last_make[int(g.tid[i])] = (g.period[i], g.clock[i])
        elif at == "freethrow" and g.st[i] == "1 of 1" and g.desc[i] not in RETAIN_FT and g.desc[i] != "technical":
            if last_make.get(int(g.tid[i])) == (g.period[i], g.clock[i]):
                out[i] = True
    return out


def reconstruct_game(ev: pd.DataFrame, home: int, away: int, box: pd.DataFrame | None = None) -> dict:
    g = GameArrays.from_frame(ev, home, away)
    teams = (home, away)
    side = {home: 0, away: 1}
    box_roster, box_seconds = None, None
    if box is not None and len(box):
        box_roster = {int(p): int(t) for p, t in zip(box["person_id"], box["team_id"])}
        box_seconds = {int(p): float(s) for p, s in zip(box["person_id"], box["box_seconds"]) if s > 0}
    roster = build_roster(g, box_roster)
    stints, ev_stint, team_periods, contra = reconstruct_lineups(g, roster, box_seconds)
    lineups = [s.lineups for s in stints]
    attr = attribution_stints(g, ev_stint, lineups)
    poss, pstats = count_possessions(g)
    live_reb = live_rebound_mask(g)
    and1 = _and_one_flags(g)
    n_st = len(stints)

    # ---------------- per-stint team features (for exposures) ----------------
    F = np.zeros((n_st, 2, len(EXPO)), dtype=np.int64)
    # ---------------- per-player counting stats ----------------
    stats: dict = defaultdict(lambda: np.zeros(len(PLAYER_STATS), dtype=np.int64))
    team_only = {t: defaultdict(int) for t in teams}

    def addF(s, t, key, v=1):
        if t in side:
            F[s, side[t], EI[key]] += v

    for i in range(g.n):
        at = g.at[i]
        t = int(g.tid[i])
        p = int(g.pid[i])
        s = ev_stint[i]
        if at in ("2pt", "3pt"):
            three = at == "3pt"
            made = g.shot[i] == "Made"
            k = "3" if three else "2"
            addF(s, t, "fga")
            addF(s, t, "fga" + k)
            if p:
                stats[p][SI["fga" + k]] += 1
            if made:
                addF(attr[i], t, "pts", 3 if three else 2)
                addF(s, t, "fgm")
                addF(s, t, "fgm" + k)
                if p:
                    stats[p][SI["fgm" + k]] += 1
                a = int(g.assist[i])
                if a:
                    stats[a][SI["ast"]] += 1
                    addF(s, t, "ast")
                    if p:
                        stats[p][SI[f"fgm{k}_ast"]] += 1
            elif g.block[i] and p:
                stats[p][SI["blocked"]] += 1
        elif at == "freethrow":
            made = g.shot[i] == "Made"
            addF(attr[i], t, "fta")
            if made:
                addF(attr[i], t, "ftm")
                addF(attr[i], t, "pts", 1)
            kk, nn = ft_trip(g.st[i])
            tech = g.desc[i] == "technical"
            if kk == 1 and not tech:
                addF(attr[i], t, "and1" if and1[i] else "trips")
            if p:
                st = stats[p]
                st[SI["fta"]] += 1
                st[SI["ftm"]] += made
                if tech:
                    st[SI["fta_tech"]] += 1
                elif kk == 1:
                    if and1[i]:
                        st[SI["and1"]] += 1
                    elif nn == 2:
                        st[SI["trips2"]] += 1
                    elif nn == 3:
                        st[SI["trips3"]] += 1
        elif at == "rebound":
            off = g.st[i] == "offensive"
            if live_reb[i] and t in side:
                # a live rebound chance for the team that missed
                shooter_team = t if off else (away if t == home else home)
                addF(s, shooter_team, "oreb_ch")
                addF(s, t, "oreb" if off else "dreb")
            if p:
                stats[p][SI["oreb" if off else "dreb"]] += 1
            elif live_reb[i] and t in side:
                team_only[t]["team_oreb" if off else "team_dreb"] += 1
        elif at == "turnover":
            addF(s, t, "tov")
            if p:
                stats[p][SI["tov"]] += 1
                if g.st[i] == "offensive foul":
                    stats[p][SI["tov_off_foul"]] += 1
            elif t in side:
                team_only[t]["team_tov"] += 1
        elif at == "steal":
            if p:
                stats[p][SI["stl"]] += 1
            addF(s, t, "stl")
        elif at == "block":
            if p:
                stats[p][SI["blk"]] += 1
            addF(s, t, "blk")
        elif at == "foul":
            if g.st[i] == "technical":
                if p:
                    stats[p][SI["tech"]] += 1
                continue
            addF(s, t, "pf")
            if p:
                st = stats[p]
                st[SI["pf"]] += 1
                if g.st[i] == "offensive":
                    st[SI["pf_offensive"]] += 1
                elif g.desc[i] == "shooting":
                    st[SI["pf_shooting"]] += 1
            d = int(g.foul_drawn[i])
            if d:
                stats[d][SI["pfd"]] += 1
                if g.desc[i] == "shooting":
                    stats[d][SI["pfd_shooting"]] += 1

    poss_stint = np.array([attr[pp.attr_idx] for pp in poss], dtype=int)
    for pp, s in zip(poss, poss_stint):
        addF(s, pp.offense, "poss")

    # ---------------- stints table ----------------
    dur = np.array([s.end - s.start for s in stints])
    # running score at stint start (from stint points in order)
    hp = F[:, 0, EI["pts"]]
    ap = F[:, 1, EI["pts"]]
    margin_start = np.concatenate([[0], np.cumsum(hp - ap)[:-1]])
    st_rows = []
    for idx, s in enumerate(stints):
        h, a = sorted(s.lineups[home]), sorted(s.lineups[away])
        ih, ia = s.issue[home], s.issue[away]
        if set(h) & set(a):
            ih = ia = "player_on_both_teams"
        st_rows.append((g.game_id, idx, s.period, s.start, s.end, s.end - s.start,
                        "-".join(map(str, h)), "-".join(map(str, a)), ih, ia,
                        int(hp[idx]), int(ap[idx]), int(F[idx, 0, EI["poss"]]), int(F[idx, 1, EI["poss"]]),
                        int(margin_start[idx])))
    stints_df = pd.DataFrame(st_rows, columns=[
        "game_id", "stint", "period", "start_s", "end_s", "duration_s", "home_lineup", "away_lineup",
        "home_issue", "away_issue", "home_pts", "away_pts", "home_poss", "away_poss", "margin_start"])
    stints_df["valid"] = (stints_df["home_issue"] == "") & (stints_df["away_issue"] == "")

    # ---------------- player table: box + exposures ----------------
    secs = player_seconds(stints)
    players = sorted(set(stats) | set(secs) | set(p for p, t in roster.items() if t in side and secs.get(p, 0) > 0))
    players = [p for p in players if roster.get(p) in side]
    pidx = {p: i for i, p in enumerate(players)}
    M = np.zeros((len(players), n_st), dtype=np.float64)          # on-court incidence
    for si, s in enumerate(stints):
        for t in teams:
            for p in s.lineups[t]:
                if p in pidx:
                    M[pidx[p], si] = 1.0
    on_sec = M @ dur
    team_of = np.array([roster[p] for p in players], dtype=np.int64)
    own_side = np.array([side[t] for t in team_of])
    Fh, Fa = F[:, 0, :].astype(float), F[:, 1, :].astype(float)
    own = np.where(own_side[:, None] == 0, M @ Fh, M @ Fa)
    opp = np.where(own_side[:, None] == 0, M @ Fa, M @ Fh)
    pm = own[:, EI["pts"]] - opp[:, EI["pts"]]
    p1 = team_periods.get((1, home)), team_periods.get((1, away))
    starters = set()
    for tp in p1:
        if tp is not None and tp.status in ("observed", "resolved_box"):
            starters |= tp.starters
    stat_mat = np.array([stats[p] if p in stats else np.zeros(len(PLAYER_STATS), dtype=np.int64)
                         for p in players]).reshape(len(players), len(PLAYER_STATS))
    pl = pd.DataFrame(stat_mat, columns=PLAYER_STATS)
    pl.insert(0, "game_id", g.game_id)
    pl.insert(1, "person_id", players)
    pl.insert(2, "team_id", team_of)
    pl["opp_team_id"] = np.where(team_of == home, away, home)
    pl["is_home"] = team_of == home
    pl["starter"] = [p in starters for p in players]
    pl["sec"] = on_sec
    pl["plus_minus"] = pm.astype(int)
    for k in EXPO:
        pl[f"on_{k}"] = own[:, EI[k]].astype(int)
        pl[f"op_{k}"] = opp[:, EI[k]].astype(int)
    # ---------------- team table ----------------
    trow = []
    for t in teams:
        o = side[t]
        tot = F[:, o, :].sum(axis=0)
        trow.append({"game_id": g.game_id, "team_id": t, "opp_team_id": away if t == home else home,
                     "is_home": t == home, **{k: int(tot[EI[k]]) for k in EXPO},
                     "team_oreb": team_only[t]["team_oreb"], "team_dreb": team_only[t]["team_dreb"],
                     "team_tov": team_only[t]["team_tov"]})
    teams_df = pd.DataFrame(trow)
    elapsed = g.elapsed
    # running score (through each event) for the offense's lead at possession start
    ev_pts = np.nan_to_num(np.asarray(g.points, dtype=float))
    cum = {t: np.cumsum(np.where(g.tid == t, ev_pts, 0.0)) for t in teams}

    def score_before(t: int, i: int) -> float:
        return float(cum[t][i - 1]) if i > 0 else 0.0

    # possession start = end of the previous possession in the period (either team), else the period start;
    # first_s is the time of the possession's first recorded event (often later: an inbound has no event)
    true_start, prev_end, prev_period = [], None, None
    for pp in poss:
        if pp.period != prev_period:
            prev_end, prev_period = float(period_start(pp.period)), pp.period
        true_start.append(prev_end)
        prev_end = float(elapsed[pp.end_idx])
    trip_ft = np.array([g.at[i] == "freethrow" and g.st[i] in ("1 of 2", "1 of 3") and g.desc[i] != "technical"
                        for i in range(g.n)], dtype=bool)
    poss_df = pd.DataFrame({
        "game_id": g.game_id,
        "period": [pp.period for pp in poss],
        "offense_team_id": [pp.offense for pp in poss],
        "stint": poss_stint,
        "reason": [pp.reason for pp in poss],
        "points": [pp.points for pp in poss],
        "n_oreb": [pp.n_oreb for pp in poss],
        "start_s": true_start,
        "first_s": [float(elapsed[min(pp.start_idx, g.n - 1)]) for pp in poss],
        "end_s": [float(elapsed[pp.end_idx]) for pp in poss],
        "start_idx": [int(pp.start_idx) for pp in poss],
        "end_idx": [int(pp.end_idx) for pp in poss],
        # offense lead before the possession's first event (its own points excluded)
        "lead_start": [int(round(score_before(pp.offense, pp.start_idx)
                                 - score_before(away if pp.offense == home else home, pp.start_idx)))
                       for pp in poss],
        # non-technical free-throw trips (first FT of 2 or 3) by the offense during the possession
        "n_trips": [int(sum(trip_ft[i] and g.tid[i] == pp.offense for i in range(pp.start_idx, pp.end_idx + 1)))
                    for pp in poss],
    })
    starters_df = pd.DataFrame([{"game_id": g.game_id, "period": k[0], "team_id": k[1], "status": v.status,
                                 "filled": "-".join(map(str, sorted(v.filled)))}
                                for k, v in sorted(team_periods.items())])
    contra_df = pd.DataFrame(contra, columns=["period", "team_id", "person_id", "reason", "elapsed_s",
                                              "since_s", "description"])
    contra_df.insert(0, "game_id", g.game_id)
    n_periods = int(g.period.max()) if g.n else 4
    game_len = period_start(n_periods) + period_length(n_periods)
    return {"players": pl, "teams": teams_df, "stints": stints_df, "possessions": poss_df,
            "starters": starters_df, "contradictions": contra_df,
            "meta": {"game_id": g.game_id, "n_periods": n_periods, "game_len_s": game_len,
                     "ot": n_periods > 4, "reg_s": REGULATION_S, **{f"poss_{k}": v for k, v in pstats.items()}}}
