"""Compiled (numba) possession-by-possession game simulation kernel.

``simulate`` runs ``n`` independent games in parallel chunks. Each game is
played possession by possession with the rules documented in ``engine.py``;
this module only contains the per-game scalar logic.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

# stat columns
PTS, FGA2, FGM2, FGA3, FGM3, FTA, FTM, OREB, DREB, AST, STL, BLK, TOV, PF, PM = range(15)
N_STATS = 15
# rate columns (must match engine.RATE_KEYS)
(USAGE, TOV_SH, FGA3_SH, TRIP_SH, FG2, FG3, FT, AND1, TRIP3, AST2, AST3, ASTR, OREB_R, DREB_R, STL_R, BLK_R,
 PF_R, OFFF, BLOCKED) = range(19)
# defense factor columns (engine.DEF_FACTORS)
F_TOV, F_3SH, F_FG2, F_FG3, F_TRIP, F_OREB, F_AST = range(7)
# scalar parameter vector indices
(P_ALPHA, P_SD_PACE, P_SD_TEAM, P_SD_SHOOT, P_SD_USG, P_SD_REB, P_SD_AST, P_HOME, P_GAMMA, P_PERSIST, P_NOISE,
 P_INJ, P_BASE_MEAN, P_SHAPE, P_B_OREB, P_TECH, P_TEAM_TOV, P_NSF, P_TEAM_REB, P_TEAM_OREB_FRAC, P_LG_TOV,
 P_LG_BLK, P_LG_AST, P_SHIFT_H, P_SHIFT_A, P_CORE, P_FOUL1, P_FOUL2, P_FOUL3, P_FOUL4, P_LG_FG2, P_LG_FG3,
 P_FRINGE_PEN, P_SCORE_EFF, P_TRIP_MULT, P_STL_MULT, P_BLK_MULT, P_FG2_OFF, P_FG3_OFF, P_OWN_EFF, P_RE_RHO,
 P_PPP_REF_H, P_PPP_REF_A, P_FRINGE_SD, P_HOLD_TIED, P_HOLD_DOWN, P_HOLD_U, P_QUICK_DUR, P_HEAVE2, P_HEAVE3,
 P_LATE3_OFF, P_HEAVE_T, P_AVOID4, P_AVOID5, P_TOV_MULT, P_OREB_MULT, P_FRINGE_TAKE) = range(57)
N_PARAMS = 57
# end-of-game lead groups (offense perspective): 0 leading, 1 tied, 2 down 1-3, 3 down 4+


@njit(cache=True, inline="always")
def _sig(x):
    return 1.0 / (1.0 + math.exp(-x))


@njit(cache=True, inline="always")
def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


@njit(cache=True)
def _choice(w, n):
    tot = 0.0
    for i in range(n):
        tot += w[i]
    if tot <= 0:
        return int(np.random.random() * n)
    u = np.random.random() * tot
    acc = 0.0
    for i in range(n):
        acc += w[i]
        if u < acc:
            return i
    return n - 1


@njit(cache=True)
def _foul_weights(rates, lineup, d, fouls, prm, w5b):
    """Defender charged with a foul: foul propensity, lower for players already in foul trouble."""
    for k in range(5):
        pl = lineup[d, k]
        w = rates[d, pl, PF_R]
        if fouls[d, pl] >= 6:
            w *= 1e-9        # disqualified mid-possession: leaves at the next stoppage, fouls no more
        elif fouls[d, pl] == 5:
            w *= prm[P_AVOID5]
        elif fouls[d, pl] == 4:
            w *= prm[P_AVOID4]
        w5b[k] = w


@njit(cache=True)
def _period_end(per):
    if per <= 4:
        return per * 720.0
    return 2880.0 + (per - 4) * 300.0


@njit(cache=True)
def _game(rates, exists, p_play, plan, min_sd, starter, rotation, boost, usage_mult, cap, q, cumfrac, dfac,
          garbage, mb_edges, p_int, late_dur, late3, heave_fg, late_off, prm, P,
          stats, secs, score, team_reb, team_tov, poss, hist, marks):
    """Simulate one game in place (stats/secs/score/... are this game's output slices).
    ``marks``: home-minus-away score with 60 s left in regulation and at the end of regulation; offense lead
    and side at the first possession with <= 24 s left; 1 if a team ran out of eligible players."""
    # ---------------- participation, planned minutes ----------------
    played = np.zeros((2, P), dtype=np.bool_)
    target = np.zeros((2, P))
    for t in range(2):
        tot = 0.0
        # a questionable rotation player (p_play < 1: unplayed games, scenarios) who sits leaves minutes that teams
        # give partly to the deep bench: that night the participation of the fringe players who may be used
        # (``boost``: not set by the user, p_play > 0) is scaled up in proportion to their expected minutes, so the
        # next men up absorb most of it, until P_FRINGE_TAKE x his planned minutes is added in expectation (rotation
        # players are listed first; nothing changes - not even the random numbers - when nobody sits)
        fexp = 0.0
        for i in range(P):
            if exists[t, i] and boost[t, i] > 0.0:
                fexp += p_play[t, i] * plan[t, i]
        vac = 0.0
        for i in range(P):
            if not exists[t, i]:
                continue
            p = p_play[t, i]
            if vac > 0.0 and fexp > 0.0 and boost[t, i] > 0.0:
                p = min(p * (1.0 + prm[P_FRINGE_TAKE] * vac / fexp), 1.0)
            if np.random.random() >= p:
                if rotation[t, i] and p_play[t, i] > 0.0:
                    vac += plan[t, i]
            else:
                played[t, i] = True
                if rotation[t, i]:
                    v = plan[t, i] + min_sd[t, i] * np.random.standard_normal()
                else:
                    # fringe minutes when used vary widely (2-20 min): mean-preserving lognormal noise
                    sf = prm[P_FRINGE_SD]
                    v = plan[t, i] * math.exp(sf * np.random.standard_normal() - 0.5 * sf * sf)
                v = min(max(v, 0.5), min(cap[t, i], 46.0))
                target[t, i] = v
                tot += v
        if tot > 0:
            for i in range(P):
                target[t, i] = min(target[t, i] * 240.0 / tot, min(cap[t, i], 47.0))
    # ---------------- random effects ----------------
    pace_m = math.exp(prm[P_SD_PACE] * np.random.standard_normal())
    team_shoot = np.zeros(2)
    eta = np.zeros((2, P))
    dlt = np.zeros((2, P))
    zeta = np.zeros((2, P))
    xi = np.zeros((2, P))
    inj = np.full((2, P), 1e9)
    gu = np.zeros((2, P))
    rho = prm[P_RE_RHO]
    a_c, a_i = math.sqrt(rho), math.sqrt(1.0 - rho)
    for t in range(2):
        team_shoot[t] = prm[P_SD_TEAM] * np.random.standard_normal()
        for i in range(P):
            act = np.random.standard_normal()           # shared "activity" night: usage, boards, passing
            eta[t, i] = prm[P_SD_USG] * (a_c * act + a_i * np.random.standard_normal())
            dlt[t, i] = prm[P_SD_SHOOT] * np.random.standard_normal()
            zeta[t, i] = prm[P_SD_REB] * (a_c * act + a_i * np.random.standard_normal())
            xi[t, i] = prm[P_SD_AST] * (a_c * act + a_i * np.random.standard_normal())
            if rotation[t, i] and np.random.random() < prm[P_INJ]:
                inj[t, i] = np.random.random() * 2880.0
            gu[t, i] = np.random.random()
    # core players: top planned minutes (garbage-time benching applies to them)
    core = np.zeros((2, P), dtype=np.bool_)
    ncore = int(prm[P_CORE])
    for t in range(2):
        order = np.argsort(-target[t])
        for j in range(min(ncore, P)):
            if target[t, order[j]] > 0:
                core[t, order[j]] = True
    # usage weights (static part)
    wuse_static = np.zeros((2, P))
    for t in range(2):
        for i in range(P):
            wuse_static[t, i] = (max(rates[t, i, USAGE], 1e-4) ** prm[P_ALPHA]) * math.exp(eta[t, i]) * usage_mult[t, i]
    # ---------------- state ----------------
    fouls = np.zeros((2, P), dtype=np.int32)
    oncourt = np.zeros((2, P), dtype=np.bool_)
    lineup = np.zeros((2, 5), dtype=np.int64)
    sc = np.zeros(P)
    clock = 0.0
    period = 1
    tip = 1 if np.random.random() < 0.5 else 0
    offense = tip
    last_slot = -1
    resel = False
    base_mean = prm[P_BASE_MEAN]
    shape = prm[P_SHAPE]
    flim = np.array([prm[P_FOUL1], prm[P_FOUL2], prm[P_FOUL3], prm[P_FOUL4]])
    w5 = np.zeros(5)
    w5b = np.zeros(5)
    marked = False
    marked24 = False
    while True:
        # ---------------- lineups ----------------
        slot = min(int(clock // 60.0), 47) if period <= 4 else 100 + period
        if slot != last_slot or resel:
            resel = False
            pend_ = _period_end(period)
            for t in range(2):
                n_ok = 0
                for i in range(P):
                    ok = exists[t, i] and played[t, i] and fouls[t, i] < 6 and inj[t, i] > clock
                    if not ok:
                        sc[i] = -1e9
                        continue
                    n_ok += 1
                    if period <= 4:
                        planned = target[t, i] * 60.0 * cumfrac[t, i, slot]
                        defi = prm[P_GAMMA] * (planned - secs[t, i]) / 60.0
                        crunch = period >= 4 and pend_ - clock <= 360.0
                        if fouls[t, i] >= flim[period - 1] and not crunch:
                            # foul trouble: sits until the period's limit rises (minutes owed do not pull him back)
                            defi = min(defi, 0.0) - 2.5
                        s = q[t, i, slot] + defi + prm[P_PERSIST] * oncourt[t, i] \
                            + prm[P_NOISE] * np.random.standard_normal()
                        if not rotation[t, i]:
                            s -= prm[P_FRINGE_PEN]
                        if (period == 1 or period == 3) and (clock % 720.0) < 1e-6 and starter[t, i]:
                            s += 4.0
                        if period == 4:
                            m = abs(score[t] - score[1 - t])
                            mb = 0
                            for k in range(len(mb_edges) - 1):
                                if m >= mb_edges[k]:
                                    mb = k
                            ratio = garbage[slot - 36, mb] if slot >= 36 else 1.0
                            if core[t, i] and gu[t, i] > ratio:
                                s -= 5.0
                            if ratio < 0.6 and not core[t, i]:
                                s += 1.5
                    else:
                        s = target[t, i] / 10.0 + 0.5 * oncourt[t, i] + 0.2 * np.random.standard_normal()
                        if fouls[t, i] >= 5:
                            s -= 3.0
                    sc[i] = s
                if n_ok < 5:   # emergency: anyone not disqualified / injured
                    for i in range(P):
                        if sc[i] <= -1e8 and exists[t, i] and fouls[t, i] < 6 and inj[t, i] > clock:
                            sc[i] = -1e6
                            n_ok += 1
                    if n_ok < 5:
                        marks[4] = 1     # no eligible substitute left: a disqualified player stays in (NBA rule)
                order = np.argsort(-sc[:P])
                for i in range(P):
                    oncourt[t, i] = False
                for k in range(5):
                    lineup[t, k] = order[k]
                    oncourt[t, order[k]] = True
            last_slot = slot
        o = offense
        d = 1 - o
        pend = _period_end(period)
        # ---------------- possession ----------------
        t_left = pend - clock
        if period == 4 and t_left <= 60.0 and not marked:
            marks[0] = score[0] - score[1]
            marked = True
        if period == 4 and t_left <= 24.0 and not marked24:
            marks[2] = score[o] - score[d]          # offense lead at the first possession with <= 24 s left
            marks[3] = 1 if o == 0 else -1
            marked24 = True
        lead = score[o] - score[d]
        grp = 0 if lead >= 1 else (1 if lead == 0 else (2 if lead >= -3 else 3))
        endgame = period >= 4 and t_left <= 24.0
        heave = endgame and t_left <= prm[P_HEAVE_T]
        p3_late = -1.0
        li = 0
        buzzer = False                                  # the first shot comes in the final seconds
        if period >= 4 and t_left > 24.0 and t_left <= 120.0:
            # last two minutes: leading teams run the clock, trailing teams hurry (data-driven means)
            dur = np.random.gamma(shape, late_dur[0 if t_left > 60.0 else 1, grp] / shape)
        else:
            dur = np.random.gamma(shape, base_mean * pace_m / shape)
        if endgame:
            li = 0 if lead >= 1 else (1 - lead if lead >= -3 else 5)
            p3_late = late3[1 if heave else 0, li]
            if grp == 0:
                dur = t_left                                   # leading: run out the clock unless fouled
            elif (grp == 1 and np.random.random() < prm[P_HOLD_TIED]) or \
                    (grp == 2 and np.random.random() < prm[P_HOLD_DOWN]):
                dur = max(t_left - prm[P_HOLD_U] * np.random.random(), 0.2)   # hold for the last shot
                buzzer = True
                p3_late = late3[1, li]
            else:
                dur = min(np.random.gamma(shape, prm[P_QUICK_DUR] / shape), t_left)
        pint = 0.0
        if period >= 4 and t_left <= 60.0 and lead >= 1:
            wi = 1 if t_left <= 24.0 else 0
            if lead <= 3:
                pint = p_int[wi, 0]
            elif lead <= 6:
                pint = p_int[wi, 1]
            elif lead <= 9:
                pint = p_int[wi, 2]
            else:
                pint = p_int[wi, 3]
        n_oreb = 0
        alive = True
        # helpers inlined: weights for current five
        for k in range(5):
            w5[k] = wuse_static[o, lineup[o, k]]
        # technical free throw (no change of possession)
        if np.random.random() < prm[P_TECH]:
            best = 0
            for k in range(1, 5):
                if rates[o, lineup[o, k], FT] > rates[o, lineup[o, best], FT]:
                    best = k
            pl = lineup[o, best]
            stats[o, pl, FTA] += 1
            if np.random.random() < rates[o, pl, FT]:
                stats[o, pl, FTM] += 1
                stats[o, pl, PTS] += 1
                score[o] += 1
                for k in range(5):
                    stats[o, lineup[o, k], PM] += 1
                    stats[d, lineup[d, k], PM] -= 1
        if np.random.random() < pint:
            dur = min(2.0 + 4.0 * np.random.random(), max(t_left, 0.1))
            heave = False
            # foul on defense
            _foul_weights(rates, lineup, d, fouls, prm, w5b)
            j = _choice(w5b, 5)
            stats[d, lineup[d, j], PF] += 1
            fouls[d, lineup[d, j]] += 1
            for k in range(5):
                w5b[k] = w5[k] * rates[o, lineup[o, k], FT]
            ks = _choice(w5b, 5)
            pl = lineup[o, ks]
            made_last = False
            for kk in range(2):
                stats[o, pl, FTA] += 1
                mk = np.random.random() < rates[o, pl, FT]
                if mk:
                    stats[o, pl, FTM] += 1
                    stats[o, pl, PTS] += 1
                    score[o] += 1
                    for k in range(5):
                        stats[o, lineup[o, k], PM] += 1
                        stats[d, lineup[d, k], PM] -= 1
                made_last = mk
            if made_last:
                alive = False
            else:
                alive = _rebound(rates, dfac, zeta, lineup, o, d, prm, stats, team_reb, w5b)
                if alive:
                    n_oreb += 1
        if alive and heave:
            # final seconds: one hurried shot (a heave or a quick catch-and-shoot), or the clock runs out
            alive = False
            dur = t_left
            if grp != 0:
                ku = _choice(w5, 5)
                user = lineup[o, ku]
                three = np.random.random() < p3_late
                # less time, worse shot: from a set play with ~4 s to a full-court heave with under a second
                tf = min(max((t_left - 0.3) / (prm[P_HEAVE_T] - 0.3), 0.0), 1.0)
                if three:
                    stats[o, user, FGA3] += 1
                    made = np.random.random() < heave_fg[li, 1] * tf
                else:
                    stats[o, user, FGA2] += 1
                    made = np.random.random() < heave_fg[li, 0] * tf
                if made:
                    pts = 3 if three else 2
                    stats[o, user, FGM3 if three else FGM2] += 1
                    stats[o, user, PTS] += pts
                    score[o] += pts
                    for k in range(5):
                        stats[o, lineup[o, k], PM] += pts
                        stats[d, lineup[d, k], PM] -= pts
                else:
                    if np.random.random() < prm[P_TEAM_REB]:
                        team_reb[d, 1] += 1
                    else:
                        for k in range(5):
                            w5b[k] = rates[d, lineup[d, k], DREB_R] * math.exp(zeta[d, lineup[d, k]])
                        j = _choice(w5b, 5)
                        stats[d, lineup[d, j], DREB] += 1
        if alive and np.random.random() < prm[P_TEAM_TOV]:
            team_tov[o] += 1
            alive = False
        plays = 0
        while alive and plays < 6:
            plays += 1
            if np.random.random() < prm[P_NSF]:
                _foul_weights(rates, lineup, d, fouls, prm, w5b)
                j = _choice(w5b, 5)
                stats[d, lineup[d, j], PF] += 1
                fouls[d, lineup[d, j]] += 1
            ku = _choice(w5, 5)
            user = lineup[o, ku]
            p_tov = min(max(rates[o, user, TOV_SH] * dfac[d, F_TOV] * prm[P_TOV_MULT], 0.01), 0.6)
            s3 = min(max(rates[o, user, FGA3_SH] * dfac[d, F_3SH], 0.0), 0.95)
            st_ = min(max(rates[o, user, TRIP_SH] * dfac[d, F_TRIP] * prm[P_TRIP_MULT], 0.0), 0.6)
            last_shot = buzzer and plays == 1
            if last_shot:
                st_ *= 0.5                                # fewer shooting fouls called on last-second shots
            s2 = max(1.0 - s3 - st_, 0.02)
            tot = s2 + s3 + st_
            if np.random.random() < p_tov:
                stats[o, user, TOV] += 1
                ssum = 0.0
                for k in range(5):
                    w5b[k] = rates[d, lineup[d, k], STL_R]
                    ssum += w5b[k]
                p_stl = min(max(prm[P_STL_MULT] * ssum / prm[P_LG_TOV], 0.05), 0.9)
                if np.random.random() < p_stl:
                    j = _choice(w5b, 5)
                    stats[d, lineup[d, j], STL] += 1
                elif fouls[o, user] < 6 and np.random.random() < min(rates[o, user, OFFF] / max(1.0 - p_stl, 0.1), 1.0):
                    stats[o, user, PF] += 1
                    fouls[o, user] += 1
                alive = False
                continue
            u2 = np.random.random() * tot
            if u2 >= s3 and u2 < s3 + st_:
                # FT trip (shooting foul on a miss / penalty foul)
                _foul_weights(rates, lineup, d, fouls, prm, w5b)
                j = _choice(w5b, 5)
                stats[d, lineup[d, j], PF] += 1
                fouls[d, lineup[d, j]] += 1
                nft = 3 if np.random.random() < rates[o, user, TRIP3] else 2
                made_last = False
                for kk in range(nft):
                    stats[o, user, FTA] += 1
                    mk = np.random.random() < rates[o, user, FT]
                    if mk:
                        stats[o, user, FTM] += 1
                        stats[o, user, PTS] += 1
                        score[o] += 1
                        for k in range(5):
                            stats[o, lineup[o, k], PM] += 1
                            stats[d, lineup[d, k], PM] -= 1
                    made_last = mk
                if made_last:
                    alive = False
                else:
                    alive = _rebound(rates, dfac, zeta, lineup, o, d, prm, stats, team_reb, w5b)
                    if alive:
                        n_oreb += 1
                continue
            late_shot = p3_late >= 0.0 and plays == 1
            if late_shot:
                three = np.random.random() < p3_late       # end-of-game shot selection by score
            else:
                three = u2 < s3
            if three:
                stats[o, user, FGA3] += 1
                base_p = rates[o, user, FG3]
                dsh = _logit(dfac[d, F_FG3] * prm[P_LG_FG3]) - _logit(prm[P_LG_FG3]) + prm[P_FG3_OFF]
            else:
                stats[o, user, FGA2] += 1
                base_p = rates[o, user, FG2]
                dsh = _logit(dfac[d, F_FG2] * prm[P_LG_FG2]) - _logit(prm[P_LG_FG2]) + prm[P_FG2_OFF]
            if period >= 4 and t_left <= 120.0:
                # late-game shooting by time left and score (contested shots, shot-clock pressure)
                dsh += late_off[0 if t_left > 60.0 else (1 if t_left > 24.0 else 2), grp, 1 if three else 0]
            if last_shot:
                # a held last shot is taken with the clock expiring: final-seconds make rates by score
                made = False
                blocked = False
                if np.random.random() < heave_fg[li, 1 if three else 0]:
                    made = True
                if made:
                    pts = 3 if three else 2
                    stats[o, user, FGM3 if three else FGM2] += 1
                    stats[o, user, PTS] += pts
                    score[o] += pts
                    for k in range(5):
                        stats[o, lineup[o, k], PM] += pts
                        stats[d, lineup[d, k], PM] -= pts
                else:
                    _rebound(rates, dfac, zeta, lineup, o, d, prm, stats, team_reb, w5b)
                alive = False                             # the clock runs out on a miss
                continue
            home_sign = 1.0 if o == 0 else -1.0
            shift = prm[P_SHIFT_H] if o == 0 else prm[P_SHIFT_A]
            # score effect: efficiency drifts against the leading team (within-game mean reversion)
            lead_o = max(min(score[o] - score[d], 25), -25)
            ppp_ref = prm[P_PPP_REF_H] if o == 0 else prm[P_PPP_REF_A]
            hot = max(min(score[o] - ppp_ref * poss[o], 25.0), -25.0)    # own scoring vs expectation so far
            lo = _logit(base_p) + dlt[o, user] + team_shoot[o] + dsh + prm[P_HOME] * home_sign + shift \
                - prm[P_SCORE_EFF] * lead_o / 10.0 - prm[P_OWN_EFF] * hot / 10.0
            p_make = _sig(lo)
            made = False
            blocked = False
            if not three:
                bsum = 0.0
                for k in range(5):
                    w5b[k] = rates[d, lineup[d, k], BLK_R]
                    bsum += w5b[k]
                p_blk = min(max(prm[P_BLK_MULT] * rates[o, user, BLOCKED] * bsum / (5.0 * prm[P_LG_BLK]), 0.0), 0.4)
                if np.random.random() < p_blk:
                    blocked = True
                    j = _choice(w5b, 5)
                    stats[d, lineup[d, j], BLK] += 1
                else:
                    made = np.random.random() < min(p_make / max(1.0 - p_blk, 1e-3), 0.98)
            else:
                made = np.random.random() < p_make
            if made:
                pts = 3 if three else 2
                stats[o, user, FGM3 if three else FGM2] += 1
                stats[o, user, PTS] += pts
                score[o] += pts
                for k in range(5):
                    stats[o, lineup[o, k], PM] += pts
                    stats[d, lineup[d, k], PM] -= pts
                psum = 0.0
                for k in range(5):
                    if k == ku:
                        w5b[k] = 0.0
                    else:
                        w5b[k] = rates[o, lineup[o, k], ASTR] * math.exp(xi[o, lineup[o, k]])
                    psum += w5b[k]
                team_pass = psum / (4.0 * prm[P_LG_AST])
                ash = rates[o, user, AST3] if three else rates[o, user, AST2]
                p_ast = min(max(ash * math.sqrt(max(team_pass, 0.05)) * dfac[d, F_AST], 0.0), 0.98)
                if np.random.random() < p_ast:
                    j = _choice(w5b, 5)
                    stats[o, lineup[o, j], AST] += 1
                if np.random.random() < rates[o, user, AND1]:
                    _foul_weights(rates, lineup, d, fouls, prm, w5b)
                    j = _choice(w5b, 5)
                    stats[d, lineup[d, j], PF] += 1
                    fouls[d, lineup[d, j]] += 1
                    stats[o, user, FTA] += 1
                    if np.random.random() < rates[o, user, FT]:
                        stats[o, user, FTM] += 1
                        stats[o, user, PTS] += 1
                        score[o] += 1
                        for k in range(5):
                            stats[o, lineup[o, k], PM] += 1
                            stats[d, lineup[d, k], PM] -= 1
                        alive = False
                    else:
                        alive = _rebound(rates, dfac, zeta, lineup, o, d, prm, stats, team_reb, w5b)
                        if alive:
                            n_oreb += 1
                else:
                    alive = False
            else:
                alive = _rebound(rates, dfac, zeta, lineup, o, d, prm, stats, team_reb, w5b)
                if alive:
                    n_oreb += 1
        # ---------------- clock ----------------
        dur += prm[P_B_OREB] * n_oreb
        dur = min(dur, max(pend - clock, 0.0))
        poss[o] += 1
        mask0 = 0
        mask1 = 0
        for k in range(5):
            secs[0, lineup[0, k]] += dur
            secs[1, lineup[1, k]] += dur
            mask0 |= (1 << lineup[0, k])
            mask1 |= (1 << lineup[1, k])
        if hist.shape[1] > 1:
            hist[0, mask0] += dur
            hist[1, mask1] += dur
        clock += dur
        offense = d
        for t in range(2):                     # disqualification (6th foul): substitute immediately
            for k in range(5):
                if fouls[t, lineup[t, k]] >= 6:
                    resel = True
        if clock >= pend - 1e-9:
            if period == 4:
                marks[1] = score[0] - score[1]
            if period >= 4 and score[0] != score[1]:
                break
            if period >= 12:             # safety valve after 8 tied overtimes: a decisive free throw, booked
                tv = 0 if np.random.random() < 0.5 else 1      # to an on-court player so every identity holds
                pl = lineup[tv, int(np.random.random() * 5)]
                stats[tv, pl, FTA] += 1
                stats[tv, pl, FTM] += 1
                stats[tv, pl, PTS] += 1
                score[tv] += 1
                for k in range(5):
                    stats[tv, lineup[tv, k], PM] += 1
                    stats[1 - tv, lineup[1 - tv, k], PM] -= 1
                break
            period += 1
            clock = pend
            if period == 2 or period == 3:
                offense = 1 - tip
            elif period == 4:
                offense = tip
            else:
                offense = 1 if np.random.random() < 0.5 else 0
            last_slot = -1
    return period


@njit(cache=True)
def _rebound(rates, dfac, zeta, lineup, o, d, prm, stats, team_reb, w5b):
    """Resolve a live miss; returns True if the offense keeps the ball."""
    if np.random.random() < prm[P_TEAM_REB]:
        if np.random.random() < prm[P_TEAM_OREB_FRAC]:
            team_reb[o, 0] += 1
            return True
        team_reb[d, 1] += 1
        return False
    osum = 0.0
    dsum = 0.0
    for k in range(5):
        osum += rates[o, lineup[o, k], OREB_R] * math.exp(zeta[o, lineup[o, k]])
        dsum += rates[d, lineup[d, k], DREB_R] * math.exp(zeta[d, lineup[d, k]])
    osum *= dfac[d, F_OREB] * prm[P_OREB_MULT]
    if np.random.random() < osum / (osum + dsum):
        for k in range(5):
            w5b[k] = rates[o, lineup[o, k], OREB_R] * math.exp(zeta[o, lineup[o, k]])
        j = _choice(w5b, 5)
        stats[o, lineup[o, j], OREB] += 1
        return True
    for k in range(5):
        w5b[k] = rates[d, lineup[d, k], DREB_R] * math.exp(zeta[d, lineup[d, k]])
    j = _choice(w5b, 5)
    stats[d, lineup[d, j], DREB] += 1
    return False


@njit(cache=True, parallel=True)
def simulate(n, n_chunks, seed, rates, exists, p_play, plan, min_sd, starter, rotation, boost, usage_mult, cap, q,
             cumfrac, dfac, garbage, mb_edges, p_int, late_dur, late3, heave_fg, late_off, prm, P, track_lineups):
    stats = np.zeros((n, 2, P, N_STATS), dtype=np.int16)
    secs = np.zeros((n, 2, P), dtype=np.float32)
    score = np.zeros((n, 2), dtype=np.int32)
    team_reb = np.zeros((n, 2, 2), dtype=np.int16)
    team_tov = np.zeros((n, 2), dtype=np.int16)
    poss = np.zeros((n, 2), dtype=np.int16)
    periods = np.zeros(n, dtype=np.int16)
    marks = np.zeros((n, 5), dtype=np.int16)
    hsize = (1 << P) if track_lineups else 1
    hist = np.zeros((n_chunks, 2, hsize))
    per_chunk = (n + n_chunks - 1) // n_chunks
    for c in prange(n_chunks):
        np.random.seed(seed * 7919 + c * 104729 + 17)
        lo = c * per_chunk
        hi = min(n, lo + per_chunk)
        sc64 = np.zeros((2, P))
        sstat = np.zeros((2, P, N_STATS), dtype=np.int32)
        sc_ = np.zeros(2, dtype=np.int64)
        tr = np.zeros((2, 2), dtype=np.int64)
        tt = np.zeros(2, dtype=np.int64)
        ps = np.zeros(2, dtype=np.int64)
        mk = np.zeros(5, dtype=np.int64)
        hc = hist[c]                     # this chunk's lineup histogram (single thread -> no races)
        for g in range(lo, hi):
            sc64[:] = 0.0
            sstat[:] = 0
            sc_[:] = 0
            tr[:] = 0
            tt[:] = 0
            ps[:] = 0
            mk[:] = 0
            per = _game(rates, exists, p_play, plan, min_sd, starter, rotation, boost, usage_mult, cap, q, cumfrac, dfac,
                        garbage, mb_edges, p_int, late_dur, late3, heave_fg, late_off, prm, P, sstat, sc64, sc_, tr,
                        tt, ps, hc, mk)
            for k in range(5):
                marks[g, k] = mk[k]
            for t in range(2):
                score[g, t] = sc_[t]
                team_tov[g, t] = tt[t]
                poss[g, t] = ps[t]
                team_reb[g, t, 0] = tr[t, 0]
                team_reb[g, t, 1] = tr[t, 1]
                for i in range(P):
                    secs[g, t, i] = sc64[t, i]
                    for s in range(N_STATS):
                        stats[g, t, i, s] = sstat[t, i, s]
            periods[g] = per
    return stats, secs, score, team_reb, team_tov, poss, periods, hist, marks
