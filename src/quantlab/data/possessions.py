"""Possession counting and free-throw crediting on the ordered event stream.

A possession ends when the other team gains control; offensive rebounds do
not start a new possession. End events for offense T: a made FG (unless an
and-one free throw follows at the same clock), a defensive rebound of a live
miss, a turnover, a made final free throw (not technical / flagrant /
clear-path / take / away-from-play, after which the shooting team keeps the
ball), a held ball won by the defense, or the end of the period (if T had an
action). Technical free throws never start or end possessions; dead-ball
team "rebounds" after a missed non-final free throw are placeholders.

Free throws are credited to the lineup on the floor at the *foul* (the
NBA.com plus-minus convention), except when the shooter was not on the floor
at the foul (a substitute shooter), in which case the shooting lineup is used.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .lineups import GameArrays

RETAIN_FT = frozenset({"flagrant", "clear-path", "transition take", "away-from-play"})


@dataclass
class Possession:
    period: int
    offense: int
    start_idx: int
    end_idx: int
    attr_idx: int
    reason: str
    points: int
    n_oreb: int


def ft_trip(sub_type: str) -> tuple:
    try:
        k, n = sub_type.split(" of ")
        return int(k), int(n)
    except (ValueError, AttributeError):
        return 1, 1


def is_and_one(g: GameArrays, i: int) -> bool:
    team, clock, period = g.tid[i], g.clock[i], g.period[i]
    for j in range(i + 1, g.n):
        if g.period[j] != period or g.clock[j] != clock:
            return False
        at = g.at[j]
        if at == "freethrow" and g.tid[j] == team:
            if g.desc[j] == "technical" or g.desc[j] in RETAIN_FT:
                continue
            return g.st[j] == "1 of 1"
        if at in ("2pt", "3pt", "turnover", "jumpball"):
            return False
    return False


def count_possessions(g: GameArrays) -> tuple:
    home, away = g.home, g.away
    other = {home: away, away: home}
    out: list = []
    stats: Counter = Counter()
    s = {"off": None, "act": False, "live": False, "mft": None, "start": 0, "pts": 0, "period": None, "oreb": 0}

    def end(i: int, team: int, reason: str, attr: int) -> None:
        out.append(Possession(s["period"], team, s["start"], i, attr, reason, s["pts"], s["oreb"]))
        stats[reason] += 1
        s.update(start=i + 1, pts=0, act=False, live=False, mft=None, oreb=0)

    def switch_to(t: int, i: int, reason: str = "implicit") -> None:
        if s["off"] is None:
            s["off"] = t
        elif s["off"] != t:
            if s["act"]:
                end(i, s["off"], reason, i)
            else:
                stats[f"{reason}_no_action"] += 1
                s.update(start=i, pts=0, live=False, mft=None, oreb=0)
            s["off"] = t

    for i in range(g.n):
        if g.period[i] != s["period"]:
            s.update(off=None, act=False, live=False, mft=None, start=i, pts=0, period=int(g.period[i]), oreb=0)
        at, t = g.at[i], int(g.tid[i])
        if at == "period":
            if g.st[i] == "end":
                if s["off"] is not None and s["act"]:
                    end(i, s["off"], "period_end", i)
                s.update(off=None, act=False, live=False, mft=None)
            continue
        if t not in other:
            continue
        if at in ("2pt", "3pt"):
            switch_to(t, i)
            s["act"] = True
            if g.shot[i] == "Made":
                s["pts"] += int(g.points[i])
                s["live"] = False
                if is_and_one(g, i):
                    stats["and_one"] += 1
                else:
                    end(i, t, "made_fg", i)
                    s["off"] = other[t]
            else:
                s["live"], s["mft"] = True, None
        elif at == "freethrow":
            if g.desc[i] == "technical":
                stats["technical_ft"] += 1
                continue
            if g.desc[i] in RETAIN_FT:
                switch_to(t, i, reason="foul_retained")
                s["act"] = True
                s["pts"] += int(g.points[i])
                s["live"] = False
                continue
            switch_to(t, i)
            s["act"] = True
            s["pts"] += int(g.points[i])
            k, n = ft_trip(g.st[i])
            if k >= n:
                if g.shot[i] == "Made":
                    end(i, t, "ft_made", i)
                    s["off"] = other[t]
                else:
                    s["live"], s["mft"] = True, i
            else:
                s["live"] = False
        elif at == "rebound":
            if not s["live"]:
                stats["ignored_rebound"] += 1
                continue
            attr = s["mft"] if s["mft"] is not None else i
            s["live"], s["mft"] = False, None
            if t == s["off"]:
                s["act"] = True
                s["oreb"] += 1
            else:
                end(i, s["off"], "def_rebound", attr)
                s["off"] = t
        elif at == "turnover":
            switch_to(t, i)
            s["act"] = True
            end(i, t, "turnover", i)
            s["off"] = other[t]
        elif at == "jumpball":
            if s["off"] is None:
                s["off"] = t
            elif s["off"] != t:
                if s["act"]:
                    end(i, s["off"], "jumpball", i)
                else:
                    s.update(start=i, pts=0)
                s["off"] = t
            s["live"], s["mft"] = False, None
    return out, stats


def live_rebound_mask(g: GameArrays) -> np.ndarray:
    """True for rebounds that follow a live miss (real rebounds, incl. team rebounds)."""
    mask = np.zeros(g.n, dtype=bool)
    live = False
    period = None
    for i in range(g.n):
        if g.period[i] != period:
            period, live = g.period[i], False
        at = g.at[i]
        if at in ("2pt", "3pt"):
            live = g.shot[i] != "Made"
        elif at == "freethrow":
            if g.desc[i] == "technical":
                continue
            k, n = ft_trip(g.st[i])
            live = (k >= n) and g.shot[i] != "Made" and g.desc[i] not in RETAIN_FT
        elif at == "rebound":
            mask[i] = live
            live = False
        elif at in ("turnover", "jumpball") or (at == "period" and g.st[i] == "end"):
            live = False
    return mask


def attribution_stints(g: GameArrays, event_stint: np.ndarray, stint_lineups: list) -> np.ndarray:
    """Stint credited with each event: free throws -> the stint of the foul that awarded them."""
    attr = np.asarray(event_stint).copy()
    last_foul: dict = {}
    period = None
    for i in range(g.n):
        if g.period[i] != period:
            last_foul.clear()
            period = g.period[i]
        at = g.at[i]
        if at == "foul" and g.tid[i]:
            last_foul[(int(g.tid[i]), g.st[i] == "technical")] = attr[i]
        elif at == "freethrow" and g.tid[i] in (g.home, g.away):
            opp = g.away if g.tid[i] == g.home else g.home
            key = (opp, g.desc[i] == "technical")
            if key in last_foul:
                fs = last_foul[key]
                p = int(g.pid[i])
                if p and p not in stint_lineups[fs].get(int(g.tid[i]), ()):
                    continue
                attr[i] = fs
    return attr
