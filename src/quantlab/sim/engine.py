"""Possession-level NBA game simulator: specification, calibration and results.

The compiled per-game logic lives in ``core.py``. A possession (offense O,
defense D, ten players on court):

1. side events: late-game intentional foul (trailing defense, final minute, data-driven rates),
   technical FT, team turnover, non-shooting defensive foul (PF, no FTs);
2. a *user* is drawn from O's five with weights ``usage_i^alpha * exp(eta_i) * usage_mult_i``;
3. the user's play type (TOV / 2PA / 3PA / FT trip) from his tendencies scaled by D's factors;
4. resolution: steals and offensive fouls on TOVs; blocks, makes (logit = player skill + player
   night + team night + defense + home + environment shift), assists (assister among the other four
   by passing weights), and-ones; FT trips of 2 or 3 shots with the fouler drawn from D;
5. misses go to the rebound model (team-rebound share, else OREB with probability O/(O+D) from the
   on-court rebounding weights); an offensive rebound starts a new play.

Rotations: exactly five per team at all times, re-chosen every game minute by planned on-court
propensity (rotation templates fitted by IPF to the planned minutes) + error diffusion toward each
player's sampled minutes target + persistence + noise, subject to foul trouble, disqualification at
six fouls, in-game injury, data-driven garbage-time benching and overtime closers. Every box-score
number is accumulated event by event, so each simulated game satisfies points = 2*2PM + 3*3PM + FTM,
team points = sum of player points = final score, minutes = 5 x game length, assists <= made FGs,
plus-minus sums to 5 x margin, and rebounds reconcile with misses.

Calibration (``GameSimulator.calibrate``):
* minutes: short pilot runs adjust each player's *planned* minutes so that simulated mean minutes
  equal the minutes model's projection (the projection already includes average blowout effects);
* environment (optional): a team-level logit shift on make probabilities so that simulated points
  per possession match a target (e.g. the team model's) - the hierarchical "team-constrained" mode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from ..config import PATHS
from . import core

STATS = ["pts", "fga2", "fgm2", "fga3", "fgm3", "fta", "ftm", "oreb", "dreb", "ast", "stl", "blk", "tov",
         "pf", "pm"]
S = {k: i for i, k in enumerate(STATS)}
RATE_KEYS = ["usage", "tov_share", "fga3_share", "trip_share", "fg2", "fg3", "ft", "and1", "trip3_share",
             "ast2_share", "ast3_share", "ast_rate", "oreb", "dreb", "stl", "blk", "pf_def", "off_foul_share",
             "blocked"]
R = {k: i for i, k in enumerate(RATE_KEYS)}
DEF_FACTORS = ["tov_forced", "fga3_share_allowed", "fg2_allowed", "fg3_allowed", "trips_allowed",
               "oreb_allowed", "ast_allowed"]
MAX_PLAYERS = 17


@dataclass
class PlayerSpec:
    person_id: int
    name: str
    rates: dict
    min_mean: float = 0.0          # projected minutes (rotation: if he plays; fringe: expected per team game)
    min_sd: float = 4.0            # planned-minutes noise
    p_play: float = 1.0            # availability / participation probability
    starter: bool = False
    rotation: bool = True          # rotation player (planned minutes) vs fringe (mostly garbage time)
    usage_mult: float = 1.0        # scenario knob: scales usage weight
    minutes_cap: float = 48.0      # e.g. injury minutes restriction
    in_pool: bool = True           # fringe: listed by the pregame rule (False: added only because he appeared)
    locked: bool = False           # minutes / participation set by a scenario: never rescaled by the budget


@dataclass
class TeamSpec:
    team_id: int
    name: str
    players: list
    def_factors: dict = field(default_factory=dict)   # this team on defense
    target_ppp: float | None = None                   # team-model expectation (points per possession)
    eff_shift: float = 0.0                            # logit shift on make probability (environment)


@dataclass
class GameSpec:
    home: TeamSpec
    away: TeamSpec
    pace: float = 99.0                 # expected possessions per team per 48 minutes
    league: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)     # how the inputs were built (mode, starters, availability)


@dataclass
class SimParams:
    """Simulator parameters. Defaults were calibrated on the 2023-24 tuning season only
    (scripts/calibrate_sim.py, docs/METHODOLOGY.md); the 2024-25 / 2025-26 test seasons were not used."""
    usage_alpha: float = 1.3        # shot shares in a lineup ~ usage^alpha (super-linear; tuned on 2023-24)
    sd_pace: float = 0.015          # game-level pace night (log scale)
    sd_team_shoot: float = 0.0      # team shooting night (logit); calibration drove it to zero
    sd_player_shoot: float = 0.09   # player shooting night (logit)
    sd_usage: float = 0.17          # player usage night (log scale)
    sd_reb: float = 0.15
    sd_ast: float = 0.18
    re_rho: float = 0.4             # correlation of a player's usage / rebounding / passing nights
    home_logit: float = 0.025
    rot_gamma: float = 0.35         # error-diffusion strength (per minute of deficit)
    rot_persist: float = 0.30       # persistence bonus for players already on court
    rot_noise: float = 0.12
    fringe_penalty: float = 0.3     # fringe players are chosen only when their deficit is clearly larger
    p_injury: float = 0.02          # per rotation player per game (exits early, does not return); 2023-24 PIT
    fringe_sd: float = 0.7          # lognormal SD of a fringe player's minutes when he is used
    foul_limits: tuple = (2, 3, 4, 5)
    garbage_core: int = 7           # players subject to garbage-time benching (by planned minutes)
    score_effect: float = 0.034     # logit drift against the leading team per 10 points of lead
    own_effect: float = 0.034       # logit drift against a team scoring above its expected pace (per 10 pts)
    tov_mult: float = 1.0           # nuisance calibrations: team-level event rates matched to the 2023-24
    trip_mult: float = 0.918        # tuning season (full 2023-24 backtest: makes, FTA, fouls, steals, blocks,
    stl_mult: float = 1.064         # possessions, OREB%)
    blk_mult: float = 1.05
    oreb_mult: float = 0.975
    fg2_offset: float = -0.035
    fg3_offset: float = 0.005
    duration_scale: float = 1.0124
    nsf_scale: float = 0.93
    min_sd_scale: float = 1.35      # scales every player's planned-minutes noise
    hold_u: float = 1.5             # a last-shot hold leaves U(0, hold_u) seconds on the clock
    heave_window: float = 4.0       # possessions starting with less than this left get one hurried shot
    foul_avoid4: float = 0.55       # foul propensity multiplier at 4 fouls / at 5 fouls (players avoid fouling out;
                                    # calibrated on 2023-24 foul-out and 5-foul rates)
    foul_avoid5: float = 0.45
    fringe_take: float = 0.25       # share of a sitting questionable rotation player's minutes that goes to fringe
                                    # players (2021-24: fringe minutes = 2.4 + 0.255 x vacated rotation minutes;
                                    # per season 0.23-0.30); acts only when a rotation player with p_play < 1 sits


def load_constants() -> dict:
    return json.loads((PATHS.models / "sim_constants.json").read_text(encoding="utf-8"))


@dataclass
class SimResult:
    stats: np.ndarray          # (N, 2, P, S) int16
    secs: np.ndarray           # (N, 2, P) float32 seconds played
    score: np.ndarray          # (N, 2)
    n_ot: np.ndarray           # (N,)
    team_reb: np.ndarray       # (N, 2, 2) team OREB, team DREB
    team_tov: np.ndarray       # (N, 2)
    poss: np.ndarray           # (N, 2) possessions
    lineup_secs: np.ndarray    # (2, 2**P) total seconds per five-man unit (bitmask), summed over sims
    players: list              # [[PlayerSpec ...], [PlayerSpec ...]]
    game: GameSpec
    marks: np.ndarray | None = None   # (N, 5) see core._game: margins at 60 s left / end of regulation, last-24 s
                                      # offense lead and side, no-eligible-substitute flag

    @property
    def n(self) -> int:
        return len(self.score)

    def stat(self, name: str) -> np.ndarray:
        st = self.stats.astype(np.int32)
        if name == "min":
            return self.secs / 60.0
        if name == "reb":
            return st[..., S["oreb"]] + st[..., S["dreb"]]
        if name == "fga":
            return st[..., S["fga2"]] + st[..., S["fga3"]]
        if name == "fgm":
            return st[..., S["fgm2"]] + st[..., S["fgm3"]]
        if name == "fg3m":
            return st[..., S["fgm3"]]
        if name == "fg3a":
            return st[..., S["fga3"]]
        if name == "pra":
            return st[..., S["pts"]] + st[..., S["oreb"]] + st[..., S["dreb"]] + st[..., S["ast"]]
        if name == "stocks":
            return st[..., S["stl"]] + st[..., S["blk"]]
        return st[..., S[name]]

    def check_invariants(self) -> dict:
        """Assert the basketball accounting identities on every simulated game; returns a summary."""
        st = self.stats.astype(np.int64)
        pts = 2 * st[..., S["fgm2"]] + 3 * st[..., S["fgm3"]] + st[..., S["ftm"]]
        out = {
            "points_identity": bool((pts == st[..., S["pts"]]).all()),
            "team_points_equal_score": bool((st[..., S["pts"]].sum(axis=2) == self.score).all()),
            "minutes_equal_5x_game": bool(np.allclose(self.secs.sum(axis=2),
                                                      5 * (2880 + 300 * self.n_ot)[:, None], atol=0.05)),
            "assists_le_made_fg": bool((st[..., S["ast"]].sum(axis=2) <=
                                        (st[..., S["fgm2"]] + st[..., S["fgm3"]]).sum(axis=2)).all()),
            "makes_le_attempts": bool(((st[..., S["fgm2"]] <= st[..., S["fga2"]]) &
                                       (st[..., S["fgm3"]] <= st[..., S["fga3"]]) &
                                       (st[..., S["ftm"]] <= st[..., S["fta"]])).all()),
            "plus_minus_sums": bool((st[..., S["pm"]].sum(axis=2) ==
                                     5 * (self.score - self.score[:, ::-1])).all()),
            "no_ties": bool((self.score[:, 0] != self.score[:, 1]).all()),
            "rebounds_reconcile_with_misses": self._rebounds_ok(st),
            # disqualification at six fouls; a disqualified player stays in only when his team has no eligible
            # substitute left (NBA rule), the one case where more than six fouls are possible
            "fouls_le_6": bool((st[..., S["pf"]] <= 6).all(axis=(1, 2))[self._no_emergency()].all()),
        }
        out["all"] = all(out.values())
        return out

    def _no_emergency(self) -> np.ndarray:
        if self.marks is None or self.marks.shape[1] < 5:
            return np.ones(len(self.score), dtype=bool)
        return self.marks[:, 4] == 0

    def _rebounds_ok(self, st: np.ndarray) -> bool:
        """Every missed field goal is rebounded (a player or team rebound) and nothing else is, except missed
        final free throws: per team, missed FG <= own OREB + opponent DREB <= missed FG + missed FT."""
        tr = self.team_reb.astype(np.int64)
        ok = True
        for t in (0, 1):
            miss_fg = (st[:, t, :, S["fga2"]] + st[:, t, :, S["fga3"]] - st[:, t, :, S["fgm2"]]
                       - st[:, t, :, S["fgm3"]]).sum(axis=1)
            miss_ft = (st[:, t, :, S["fta"]] - st[:, t, :, S["ftm"]]).sum(axis=1)
            reb = st[:, t, :, S["oreb"]].sum(axis=1) + tr[:, t, 0] + st[:, 1 - t, :, S["dreb"]].sum(axis=1) \
                + tr[:, 1 - t, 1]
            ok = ok and bool(((reb >= miss_fg) & (reb <= miss_fg + miss_ft)).all())
        return ok


class GameSimulator:
    def __init__(self, game: GameSpec, params: SimParams | None = None, constants: dict | None = None):
        self.g = game
        self.p = params or SimParams()
        self.c = constants or load_constants()
        self.teams = [game.home, game.away]
        for t in self.teams:
            if len(t.players) > MAX_PLAYERS:
                t.players = sorted(t.players, key=lambda pl: (-pl.rotation, -pl.min_mean))[:MAX_PLAYERS]
        self.P = max(5, max(len(t.players) for t in self.teams))
        self.plan = None
        self.dur_scale = 1.0          # per-game possession-length factor set by calibrate() (pace step)
        self._prepare()

    def _prepare(self) -> None:
        P = self.P
        self.rates = np.zeros((2, P, len(RATE_KEYS)))
        self.exists = np.zeros((2, P), dtype=np.bool_)
        self.p_play = np.zeros((2, P))
        self.min_mean = np.zeros((2, P))
        self.min_sd = np.zeros((2, P))
        self.starter = np.zeros((2, P), dtype=np.bool_)
        self.rotation = np.zeros((2, P), dtype=np.bool_)
        self.usage_mult = np.ones((2, P))
        self.cap = np.full((2, P), 48.0)
        self.locked = np.zeros((2, P), dtype=np.bool_)
        self.in_pool = np.ones((2, P), dtype=np.bool_)
        for t, team in enumerate(self.teams):
            for i, pl in enumerate(team.players):
                self.exists[t, i] = True
                self.rates[t, i] = [pl.rates[k] for k in RATE_KEYS]
                self.p_play[t, i] = pl.p_play
                self.min_mean[t, i] = pl.min_mean
                self.min_sd[t, i] = pl.min_sd * self.p.min_sd_scale
                self.starter[t, i] = pl.starter
                self.rotation[t, i] = pl.rotation
                self.usage_mult[t, i] = pl.usage_mult
                self.cap[t, i] = pl.minutes_cap
                self.locked[t, i] = pl.locked
                self.in_pool[t, i] = pl.in_pool
            avail = self.exists[t] & (self.p_play[t] > 0)
            if (self.starter[t] & avail).sum() != 5:
                key = self.min_mean[t] + 100 * (self.starter[t] & avail) - 1e6 * ~avail
                order = np.argsort(-key)
                self.starter[t] = False
                self.starter[t, order[:5]] = True
        r = self.rates
        r[..., R["fga3_share"]] = np.clip(r[..., R["fga3_share"]], 0.0, 0.95)
        r[..., R["trip_share"]] = np.clip(r[..., R["trip_share"]], 0.0, 0.6)
        self.dfac = np.ones((2, len(DEF_FACTORS)))
        for t, team in enumerate(self.teams):
            for j, k in enumerate(DEF_FACTORS):
                self.dfac[t, j] = team.def_factors.get(k, 1.0)
        lg = {"blk": 0.0108, "ast_rate": 0.159, "tov_per_poss": 0.142, "fg2": 0.545, "fg3": 0.36, **self.g.league}
        self.lg = lg
        self._minutes_budget()
        # planned minutes: rotation players their projection; fringe players their conditional minutes
        self.plan = np.where(self.rotation, self.min_mean,
                             np.where(self.p_play > 0, self.min_mean / np.maximum(self.p_play, 1e-6), 0.0))
        self.plan = np.where(self.exists, np.minimum(self.plan, self.cap), 0.0)
        self._templates()

    def _minutes_budget(self, total: float = 241.5, p_max: float = 0.95, p_min: float = 0.03) -> None:
        """Honour the minutes model for rotation players; fringe players fill the remainder by
        *appearing more often*, not by playing longer when they appear.

        The minutes model is unbiased for rotation players, so their projections are kept. Projections are
        minutes *if he plays*; the budget is drawn up for the case where every available rotation player
        plays, and when a questionable one (p_play < 1) sits, the kernel's per-game normalisation gives his
        minutes to the players who do play. The residual ``total - sum(rotation)`` goes to fringe players:
        each keeps his conditional minutes (his typical minutes when he plays) and his participation
        probability is scaled (capped at ``p_max``) until the expected fringe minutes match the residual - so
        a deeper projected rotation always means less fringe participation, down to none; whatever fringe
        players cannot absorb is spread over the rotation. Players whose minutes or participation were set by
        a scenario (``locked``) keep them: their minutes come off the budget first and they are never rescaled.
        Fringe participation never falls below ``p_min`` (deep-bench players still appear now and then in games
        whose rotation projections fill the budget). Players listed only because they appeared in the game
        being forecast (``in_pool`` False) keep their floor participation and stay out of the budget, so their
        presence does not move anyone else's numbers.
        (Diagnosed on 2023-24: without this the simulator over-allocated rotation minutes by ~2.5 per
        team-game; inflating fringe *conditional* minutes instead biased them by +4.9.)
        """
        for t in range(2):
            ex = self.exists[t] & (self.p_play[t] > 0)
            lk = ex & self.locked[t]
            rot = ex & self.rotation[t] & ~lk
            fr = ex & ~self.rotation[t] & ~lk & self.in_pool[t]
            avail = max(total - float(self.min_mean[t, lk].sum()), 0.0)
            rot_sum = float(self.min_mean[t, rot].sum())
            if rot_sum > avail and rot_sum > 0:
                self.min_mean[t, rot] *= avail / rot_sum
                rot_sum = avail
            budget = avail - rot_sum
            if fr.any():
                cond = self.min_mean[t, fr] / np.maximum(self.p_play[t, fr], 1e-6)   # minutes when he plays
                cur = float((self.p_play[t, fr] * cond).sum())
                k = max(budget, 0.0) / max(cur, 1e-6)
                newp = np.clip(self.p_play[t, fr] * k, min(p_min, p_max), p_max)
                self.p_play[t, fr] = newp
                self.min_mean[t, fr] = newp * cond
                budget -= float((newp * cond).sum())
            if budget > 0.5 and rot_sum > 0:
                self.min_mean[t, rot] *= (rot_sum + budget) / rot_sum

    def _templates(self) -> None:
        """Planned on-court propensity per regulation minute: rotation templates fitted by IPF."""
        P = self.P
        self.q = np.zeros((2, P, 48))
        tiers = self.c["template_tiers"]
        for t in range(2):
            base = np.zeros((P, 48))
            exp_min = self.plan[t] * np.where(self.rotation[t], 1.0, self.p_play[t])
            for i in range(P):
                if not self.exists[t, i] or exp_min[i] <= 0:
                    continue
                tier = int(np.clip(np.searchsorted(tiers, self.plan[t, i], side="right") - 1, 0, len(tiers) - 2))
                key = f"{int(self.starter[t, i])}_{tier}"
                base[i] = np.maximum(np.asarray(self.c["templates"].get(key, self.c["templates"]["0_1"])), 0.01)
            tgt = exp_min * 240.0 / max(exp_min.sum(), 1e-9)
            self.q[t] = ipf(base, np.minimum(tgt, 47.0), 5.0)
        cq = np.cumsum(self.q, axis=2)
        tot = np.maximum(cq[..., -1:], 1e-9)
        self.cumfrac = np.ascontiguousarray(np.concatenate([np.zeros((2, P, 1)), cq[..., :-1] / tot], axis=2))

    def _prm(self, shift) -> np.ndarray:
        c, p = self.c, self.p
        mean_dur = 2880.0 / (2.0 * self.g.pace)
        prm = np.zeros(core.N_PARAMS)
        prm[core.P_ALPHA] = p.usage_alpha
        prm[core.P_SD_PACE] = p.sd_pace
        prm[core.P_SD_TEAM] = p.sd_team_shoot
        prm[core.P_SD_SHOOT] = p.sd_player_shoot
        prm[core.P_SD_USG] = p.sd_usage
        prm[core.P_SD_REB] = p.sd_reb
        prm[core.P_SD_AST] = p.sd_ast
        prm[core.P_HOME] = p.home_logit
        prm[core.P_GAMMA] = p.rot_gamma
        prm[core.P_PERSIST] = p.rot_persist
        prm[core.P_NOISE] = p.rot_noise
        prm[core.P_INJ] = p.p_injury
        prm[core.P_BASE_MEAN] = (mean_dur - c["poss_time"]["b_per_oreb"] * c["poss_time"]["oreb_per_poss"]) * \
            p.duration_scale * self.dur_scale
        prm[core.P_SHAPE] = c["poss_time"]["gamma_shape"]
        prm[core.P_B_OREB] = c["poss_time"]["b_per_oreb"]
        prm[core.P_TECH] = c["tech_fta_per_team_game"] / max(self.g.pace, 1.0)
        prm[core.P_TEAM_TOV] = c["team_tov_share"] * self.lg["tov_per_poss"]
        prm[core.P_NSF] = c["nonft_def_fouls_per_opp_play"] * p.nsf_scale
        prm[core.P_TEAM_REB] = c["team_reb_share"]
        prm[core.P_TEAM_OREB_FRAC] = c["team_oreb_share_of_team_reb"]
        prm[core.P_LG_TOV] = self.lg["tov_per_poss"]
        prm[core.P_LG_BLK] = self.lg["blk"]
        prm[core.P_LG_AST] = self.lg["ast_rate"]
        prm[core.P_SHIFT_H], prm[core.P_SHIFT_A] = shift
        prm[core.P_CORE] = p.garbage_core
        prm[core.P_FOUL1:core.P_FOUL4 + 1] = p.foul_limits
        prm[core.P_LG_FG2] = self.lg["fg2"]
        prm[core.P_LG_FG3] = self.lg["fg3"]
        prm[core.P_FRINGE_PEN] = p.fringe_penalty
        prm[core.P_SCORE_EFF] = p.score_effect
        prm[core.P_TRIP_MULT] = p.trip_mult
        prm[core.P_STL_MULT] = p.stl_mult
        prm[core.P_BLK_MULT] = p.blk_mult
        prm[core.P_FG2_OFF] = p.fg2_offset
        prm[core.P_FG3_OFF] = p.fg3_offset
        prm[core.P_OWN_EFF] = p.own_effect
        prm[core.P_RE_RHO] = p.re_rho
        prm[core.P_PPP_REF_H] = self.teams[0].target_ppp or 1.15
        prm[core.P_PPP_REF_A] = self.teams[1].target_ppp or 1.15
        prm[core.P_FRINGE_SD] = p.fringe_sd
        eg = c["end_game"]
        prm[core.P_HOLD_TIED] = eg["hold"]["tied"]
        prm[core.P_HOLD_DOWN] = eg["hold"]["down_1_3"]
        prm[core.P_HOLD_U] = p.hold_u
        prm[core.P_QUICK_DUR] = eg["quick_dur"]
        prm[core.P_HEAVE2] = eg["fg"]["heave2"]
        prm[core.P_HEAVE3] = eg["fg"]["heave3"]
        lg3 = 0.36
        prm[core.P_LATE3_OFF] = float(np.log(eg["fg"]["late3"] / (1 - eg["fg"]["late3"])) - np.log(lg3 / (1 - lg3)))
        prm[core.P_HEAVE_T] = p.heave_window
        prm[core.P_AVOID4] = p.foul_avoid4
        prm[core.P_AVOID5] = p.foul_avoid5
        prm[core.P_TOV_MULT] = p.tov_mult
        prm[core.P_OREB_MULT] = p.oreb_mult
        prm[core.P_FRINGE_TAKE] = p.fringe_take
        return prm

    def _arrays(self):
        c = self.c
        garbage = np.array([c["garbage_ratio"][str(k)] if str(k) in c["garbage_ratio"] else c["garbage_ratio"][k]
                            for k in range(36, 48)], dtype=np.float64)
        mb_edges = np.asarray(c["margin_buckets"], dtype=np.float64)
        lw = c["late_game"]["leading_by_window"]
        base = c["late_game"]["baseline_p_trip_q4"]
        p_int = np.array([[max(0.0, (lw[w][k]["p_trip"] - base) / (1 - base)) for k in ("1-3", "4-6", "7-9", "10-99")]
                          for w in ("24-60", "0-24")])
        eg = c["end_game"]
        late_dur = np.array([eg["dur"]["60-120"], eg["dur"]["24-60"]], dtype=np.float64)
        late3 = np.array([eg["three_share"]["4-24"], eg["three_share"]["0-4"]], dtype=np.float64)
        heave_fg = np.array(eg["heave_fg"], dtype=np.float64) if "heave_fg" in eg else \
            np.tile([eg["fg"]["heave2"], eg["fg"]["heave3"]], (6, 1))
        late_off = np.asarray(eg.get("late_make_logit", np.zeros((3, 4, 2))), dtype=np.float64)
        return garbage, mb_edges, p_int, late_dur, late3, heave_fg, late_off

    def _run_raw(self, n, seed, shift, plan, track, p_play=None):
        garbage, mb_edges, p_int, late_dur, late3, heave_fg, late_off = self._arrays()
        # lineup histograms hold 2**P cells per chunk: fewer chunks when tracking a large roster
        n_chunks = int(min(max(4, n // 64), 32 if (track and self.P > 15) else 256))
        return core.simulate(int(n), n_chunks, int(seed), self.rates, self.exists,
                             self.p_play if p_play is None else p_play, plan, self.min_sd,
                             self.starter, self.rotation, self.usage_mult, self.cap, self.q, self.cumfrac, self.dfac,
                             garbage, mb_edges, p_int, late_dur, late3, heave_fg, late_off, self._prm(shift), self.P,
                             track)

    def calibrate(self, n_pilot: int = 800, iters: int = 3, seed: int = 99, minutes: bool = True,
                  environment: str = "none", env_weight: float = 1.0, pace: bool = True) -> dict:
        """Calibrate pace, planned minutes and (optionally) the team environment with short pilot runs.

        pace: possession length is scaled so that simulated regulation possessions per team per 48 minutes equal
        the game's pace (the team model's), in every environment - the late-game logic and end-of-period
        truncation otherwise leave the simulator ~0.5% short of the team model's possessions.
        environment: 'none' (bottom-up), 'team' (match each team's target_ppp), or 'blend'
        (target = env_weight * team-model PPP + (1 - env_weight) * bottom-up PPP).
        """
        shift = np.array([self.teams[0].eff_shift, self.teams[1].eff_shift], dtype=float)
        plan = self.plan.copy()
        info = {}
        if pace and self.g.pace > 0:
            *_, poss, periods, _, _ = self._run_raw(n_pilot, seed + 70, shift, plan, False)
            n_ot = np.maximum(periods.astype(float) - 4.0, 0.0)
            rate = float((poss.mean(axis=1) / (1.0 + 5.0 * n_ot / 48.0)).mean())
            self.dur_scale *= rate / self.g.pace
            info["pace_rate_before"] = rate
        # targets: rotation players their minutes if they play, fringe players their expected minutes per team
        # game. Pilot runs let every available rotation player play, so a questionable player's target and his
        # simulated mean are on the same basis (otherwise games he sits out would inflate his plan).
        want = self.min_mean.copy()
        p_pilot = np.where(self.rotation & (self.p_play > 0), 1.0, self.p_play)
        for it in range(iters if minutes else 0):
            st, secs, score, *_ = self._run_raw(n_pilot, seed + it, shift, plan, False, p_play=p_pilot)
            sim_mean = secs.mean(axis=0) / 60.0
            ok = self.exists & (want > 0.3) & (sim_mean > 0.05)
            ratio = np.where(ok, want / np.maximum(sim_mean, 1e-3), 1.0)
            plan = np.where(ok, plan * np.clip(ratio, 0.6, 1.6) ** 0.9, plan)
            plan = np.minimum(plan, self.cap)
            info[f"minutes_mae_iter{it}"] = float(np.abs(sim_mean - want)[self.exists & (want > 0.5)].mean())
        self.plan = plan
        if environment != "none":
            tgt = np.array([t.target_ppp or np.nan for t in self.teams])
            st, secs, score, tr, tt, poss, *_ = self._run_raw(n_pilot, seed + 50, shift, plan, False)
            ppp0 = score.mean(axis=0) / poss.mean(axis=0)
            st, secs, score, tr, tt, poss, *_ = self._run_raw(n_pilot, seed + 50, shift + 0.1, plan, False)
            ppp1 = score.mean(axis=0) / poss.mean(axis=0)
            sens = np.maximum((ppp1 - ppp0) / 0.1, 0.2)
            goal = env_weight * tgt + (1 - env_weight) * ppp0 if environment == "blend" else tgt
            delta = np.where(np.isfinite(goal), (goal - ppp0) / sens, 0.0)
            shift = shift + np.clip(delta, -0.6, 0.6)
            info.update({"ppp_bottom_up": ppp0.tolist(), "ppp_target": goal.tolist(), "shift": shift.tolist()})
        self.teams[0].eff_shift, self.teams[1].eff_shift = float(shift[0]), float(shift[1])
        return info

    def run(self, n: int = 10000, seed: int = 0, track_lineups: bool = True) -> SimResult:
        shift = (self.teams[0].eff_shift, self.teams[1].eff_shift)
        stats, secs, score, team_reb, team_tov, poss, periods, hist, marks = self._run_raw(n, seed, shift, self.plan,
                                                                                          track_lineups)
        return SimResult(stats, secs, score, np.maximum(periods.astype(int) - 4, 0), team_reb, team_tov, poss,
                         hist.sum(axis=0), [t.players for t in self.teams], self.g, marks)


def ipf(base: np.ndarray, row_targets: np.ndarray, col_target: float, iters: int = 80) -> np.ndarray:
    """Iterative proportional fitting: rows sum to ``row_targets`` (minutes), columns to ``col_target``
    (five on court), entries capped at 1 (a player cannot be on court more than the whole minute)."""
    q = base.copy().astype(float)
    if row_targets.sum() <= 0:
        return q * 0
    for _ in range(iters):
        rs = q.sum(axis=1)
        q *= np.where(rs > 0, row_targets / np.maximum(rs, 1e-12), 0)[:, None]
        q = np.minimum(q, 1.0)
        cs = q.sum(axis=0)
        q *= (col_target / np.maximum(cs, 1e-12))[None, :]
        q = np.minimum(q, 1.0)
    return q
