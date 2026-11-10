"""Reconstruct the five players on the floor for both teams from play-by-play.

Rules (the conventions validated against official minutes and plus-minus in
the FiveMan lineup project, re-implemented here):

1. *Period starters* are not announced. A player whose first appearance in a
   period is anything other than ``SUB in`` (a shot, rebound, foul,
   ``SUB out`` ...) was on the floor when the period began.
2. Within a period substitutions are explicit, so lineups are walked forward.
3. Never guess silently. A starter who records no event in a whole period is
   invisible; if official box-score minutes identify them *uniquely* they are
   filled in (``resolved_box``), otherwise the team's stints in that period
   are flagged invalid. Substitution contradictions invalidate stints from the
   moment the player was last known to be off the floor.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from ..config import period_length, period_start

EVIDENCE_ACTIONS = frozenset({"2pt", "3pt", "freethrow", "rebound", "turnover", "steal",
                              "block", "foul", "violation", "jumpball"})
MATCH_TOLERANCE_S = 10.0


@dataclass
class GameArrays:
    """Column arrays for one game's ordered events (fast per-event access)."""
    game_id: int
    home: int
    away: int
    period: np.ndarray
    clock: np.ndarray
    elapsed: np.ndarray
    at: list
    st: list
    desc: list
    qual: list
    pid: np.ndarray
    tid: np.ndarray
    shot: list
    points: np.ndarray
    assist: np.ndarray
    foul_drawn: np.ndarray
    steal: np.ndarray
    block: np.ndarray
    jb_won: np.ndarray
    jb_lost: np.ndarray
    description: list

    @property
    def n(self) -> int:
        return len(self.at)

    @classmethod
    def from_frame(cls, g, home: int, away: int) -> "GameArrays":
        return cls(int(g["game_id"].iat[0]), int(home), int(away),
                   g["period"].to_numpy(), g["clock_s"].to_numpy(), g["elapsed_s"].to_numpy(),
                   g["action_type"].tolist(), g["sub_type"].tolist(), g["descriptor"].tolist(),
                   g["qualifiers"].tolist(), g["person_id"].to_numpy(), g["team_id"].to_numpy(),
                   g["shot_result"].tolist(), g["points"].to_numpy(), g["assist_id"].to_numpy(),
                   g["foul_drawn_id"].to_numpy(), g["steal_id"].to_numpy(), g["block_id"].to_numpy(),
                   g["jb_won_id"].to_numpy(), g["jb_lost_id"].to_numpy(), g["description"].tolist())


def evidence_people(g: GameArrays, i: int) -> tuple:
    """Players whose involvement in event ``i`` proves they were on the floor.

    Technical fouls/free throws are excluded (bench players and coaches receive
    them; shooters may have just been substituted at a period break).
    """
    at = g.at[i]
    if at not in EVIDENCE_ACTIONS:
        return ()
    if (at == "foul" and g.st[i] == "technical") or (at == "freethrow" and g.desc[i] == "technical"):
        return ()
    p = int(g.pid[i])
    people = [p] if p else []
    if at in ("2pt", "3pt") and g.assist[i]:
        people.append(int(g.assist[i]))
    elif at == "foul" and g.foul_drawn[i]:
        people.append(int(g.foul_drawn[i]))
    elif at == "jumpball":
        people.extend(int(x) for x in (g.jb_won[i], g.jb_lost[i]) if x)
    return tuple(people)


@dataclass
class TeamPeriod:
    period: int
    team: int
    starters: set
    status: str                      # observed | resolved_box | unresolved_missing_starter | too_many_starters
    silent: set = field(default_factory=set)   # roster players with no event in the period
    filled: set = field(default_factory=set)


def infer_starters(g: GameArrays, idx: np.ndarray, roster: dict) -> dict:
    first: dict = {}
    for i in idx:
        if g.at[i] == "substitution":
            p = int(g.pid[i])
            if p and p not in first:
                first[p] = "starter" if g.st[i] == "out" else "entered"
        else:
            for p in evidence_people(g, i):
                first.setdefault(p, "starter")
    period = int(g.period[idx[0]])
    out = {}
    for t in (g.home, g.away):
        team_players = {p for p, tt in roster.items() if tt == t}
        st = {p for p, s in first.items() if s == "starter" and roster.get(p) == t}
        status = "observed" if len(st) == 5 else ("too_many_starters" if len(st) > 5
                                                  else "unresolved_missing_starter")
        out[t] = TeamPeriod(period, t, st, status, silent=team_players - set(first))
    return out


@dataclass
class Stint:
    period: int
    start: float
    end: float = 0.0
    lineups: dict = field(default_factory=dict)
    issue: dict = field(default_factory=dict)
    n_events: int = 0


def walk_period(g: GameArrays, idx: np.ndarray, period: int, starters: dict, roster: dict,
                base_issue: dict):
    teams = (g.home, g.away)
    pstart = period_start(period)
    pend = pstart + period_length(period)
    on = {t: set(starters[t]) for t in teams}
    broken = dict(base_issue)
    last_off: dict = {}
    stints: list = []
    ev_stint: list = []
    contra: list = []

    def open_stint(t0: float) -> Stint:
        s = Stint(period, t0, lineups={t: frozenset(on[t]) for t in teams})
        for t in teams:
            s.issue[t] = broken[t] or ("" if len(on[t]) == 5 else "lineup_size")
        stints.append(s)
        return s

    def contradiction(t: int, p: int, reason: str, i: int) -> None:
        since = last_off.get(p, pstart)
        contra.append({"period": period, "team_id": t, "person_id": p, "reason": reason,
                       "elapsed_s": float(g.elapsed[i]), "since_s": float(since),
                       "description": g.description[i]})
        if not broken[t]:
            broken[t] = reason
        for s in stints:
            if s.end > since or s is stints[-1]:
                if not s.issue[t] or s.issue[t] == "lineup_size":
                    s.issue[t] = reason

    cur = open_stint(pstart)
    for i in idx:
        at = g.at[i]
        if at == "substitution":
            p = int(g.pid[i])
            t = roster.get(p)
            if t not in on:
                contra.append({"period": period, "team_id": 0, "person_id": p, "reason": "sub_unknown_player",
                               "elapsed_s": float(g.elapsed[i]), "since_s": float(g.elapsed[i]),
                               "description": g.description[i]})
                ev_stint.append(len(stints) - 1)
                continue
            cur.end = float(g.elapsed[i])
            if g.st[i] == "out":
                if p in on[t]:
                    on[t].discard(p)
                    last_off[p] = float(g.elapsed[i])
                else:
                    contradiction(t, p, "sub_out_not_on_court", i)
            else:
                if p not in on[t]:
                    on[t].add(p)
                else:
                    contradiction(t, p, "sub_in_already_on_court", i)
            cur = open_stint(float(g.elapsed[i]))
            ev_stint.append(len(stints) - 1)
            continue
        for p in evidence_people(g, i):
            t = roster.get(p)
            if t in on and p not in on[t] and not broken[t]:
                contradiction(t, p, "event_by_off_court_player", i)
        cur.n_events += at in EVIDENCE_ACTIONS
        ev_stint.append(len(stints) - 1)
    cur.end = pend
    return stints, ev_stint, contra


def player_seconds(stints: list) -> dict:
    secs: dict = defaultdict(float)
    for s in stints:
        d = s.end - s.start
        for lineup in s.lineups.values():
            for p in lineup:
                secs[p] += d
    return secs


def resolve_missing_starters(team_periods: dict, deficits: dict, box_seconds: dict,
                             tol: float = MATCH_TOLERANCE_S) -> None:
    """Fill invisible starters from box-score minutes, only when the answer is unique."""
    incomplete = {k: v for k, v in team_periods.items() if v.status == "unresolved_missing_starter"}
    if not incomplete:
        return
    cand: dict = defaultdict(list)
    for key, tp in incomplete.items():
        for c in tp.silent:
            if box_seconds.get(c, 0) > 0:
                cand[c].append(key)
    assigned: dict = defaultdict(set)
    for c, keys in cand.items():
        d = deficits.get(c, 0.0)
        matches = [sub for r in range(1, len(keys) + 1) for sub in combinations(keys, r)
                   if abs(d - sum(period_length(k[0]) for k in sub)) <= tol]
        if len(matches) == 1:
            for k in matches[0]:
                assigned[k].add(c)
    for key, tp in incomplete.items():
        if len(assigned[key]) == 5 - len(tp.starters):
            tp.filled = set(assigned[key])
            tp.starters = tp.starters | tp.filled
            tp.status = "resolved_box"


def build_roster(g: GameArrays, box_roster: dict | None) -> dict:
    roster: dict = {}
    mask = (g.pid > 0) & ((g.tid == g.home) | (g.tid == g.away))
    if mask.any():
        pids, tids = g.pid[mask], g.tid[mask]
        counts: dict = defaultdict(lambda: defaultdict(int))
        for p, t in zip(pids.tolist(), tids.tolist()):
            counts[p][t] += 1
        for p, c in counts.items():
            roster[p] = max(c.items(), key=lambda kv: kv[1])[0]
    if box_roster:
        for p, t in box_roster.items():
            roster.setdefault(p, t)
    return roster


def reconstruct_lineups(g: GameArrays, roster: dict, box_seconds: dict | None):
    """Return (stints, event_stint, team_periods, contradictions) after dropping transient stints."""
    teams = (g.home, g.away)
    periods = sorted(set(g.period.tolist()))
    by_period = {p: np.flatnonzero(g.period == p) for p in periods}
    team_periods: dict = {}
    for p in periods:
        for t, tp in infer_starters(g, by_period[p], roster).items():
            team_periods[(p, t)] = tp

    def run():
        all_st, all_ev, all_c = [], [], []
        for p in periods:
            starters = {t: team_periods[(p, t)].starters for t in teams}
            base = {t: ("" if team_periods[(p, t)].status in ("observed", "resolved_box")
                        else team_periods[(p, t)].status) for t in teams}
            st, ev, c = walk_period(g, by_period[p], p, starters, roster, base)
            off = len(all_st)
            all_st.extend(st)
            all_ev.extend(i + off for i in ev)
            all_c.extend(c)
        return all_st, all_ev, all_c

    stints, ev_stint, contra = run()
    if box_seconds and any(tp.status == "unresolved_missing_starter" for tp in team_periods.values()):
        secs = player_seconds(stints)
        deficits = {p: box_seconds[p] - secs.get(p, 0.0) for p in box_seconds}
        resolve_missing_starters(team_periods, deficits, box_seconds)
        if any(tp.status == "resolved_box" for tp in team_periods.values()):
            stints, ev_stint, contra = run()

    # Drop zero-length, event-free transient stints (inside substitution batches); their events
    # (the substitutions) map to the next kept stint, trailing ones to the previous kept stint.
    keep = [i for i, s in enumerate(stints) if (s.end - s.start) > 0 or s.n_events > 0]
    remap = {old: new for new, old in enumerate(keep)}
    mapped = np.full(len(stints), -1)
    nxt = -1
    for i in range(len(stints) - 1, -1, -1):
        nxt = remap.get(i, nxt)
        mapped[i] = nxt
    prv = 0
    for i in range(len(stints)):
        if mapped[i] == -1:
            mapped[i] = prv
        prv = mapped[i]
    event_stint = mapped[np.asarray(ev_stint, dtype=int)]
    return [stints[i] for i in keep], event_stint, team_periods, contra
