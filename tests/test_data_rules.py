"""Play-by-play rules on small hand-built event streams: possessions, free-throw trips, and-ones,
starters and lineup walking, derived box score identities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.data.events import parse_clock
from quantlab.data.game import reconstruct_game
from quantlab.data.lineups import GameArrays
from quantlab.data.possessions import count_possessions, ft_trip

H, A = 1610612737, 1610612738          # team ids
HP = [1, 2, 3, 4, 5]                   # home starters
AP = [11, 12, 13, 14, 15]              # away starters


def ev(rows):
    """rows: (period, clock_s, action, sub_type, descriptor, person, team, shot_result, assist, ...)."""
    out = []
    for k, r in enumerate(rows):
        period, clock, at, st, desc, pid, tid, res = r[:8]
        extra = r[8] if len(r) > 8 else {}
        made = res == "Made"
        pts = (2 if at == "2pt" else 3 if at == "3pt" else 1 if at == "freethrow" else 0) if made else 0
        out.append({"game_id": 1, "order": k, "period": period, "clock_s": float(clock), "action_type": at,
                    "sub_type": st, "descriptor": desc, "qualifiers": "", "person_id": pid, "team_id": tid,
                    "team": "", "score_home": 0, "score_away": 0, "shot_result": res, "description": "",
                    "player_name": "", "time_actual": "", "shot_distance": np.nan, "area": "",
                    "assist_id": extra.get("assist", 0), "foul_drawn_id": extra.get("drawn", 0),
                    "steal_id": extra.get("steal", 0), "block_id": extra.get("block", 0), "jb_won_id": 0,
                    "jb_lost_id": 0, "points": pts,
                    "elapsed_s": (period - 1) * 720.0 + (720.0 - clock)})
    return pd.DataFrame(out)


def starters_touch():
    rows = [(1, 720, "period", "start", "", 0, 0, "")]
    t = 719
    for p in HP:
        rows.append((1, t, "rebound", "defensive", "", p, H, "")); t -= 1
    for p in AP:
        rows.append((1, t, "rebound", "defensive", "", p, A, "")); t -= 1
    return rows, t


def test_parse_clock():
    s = parse_clock(pd.Series(["PT11M41.00S", "PT00M05.40S"]))
    assert list(s) == [701.0, 5.4]


def test_ft_trip_parsing():
    assert ft_trip("2 of 3") == (2, 3) and ft_trip("weird") == (1, 1)


def test_possession_rules():
    rows, t = starters_touch()
    rows += [
        (1, 600, "2pt", "", "", 1, H, "Made", {"assist": 2}),            # H possession ends (made FG)
        (1, 590, "3pt", "", "", 11, A, "Missed"),                        # A misses
        (1, 589, "rebound", "offensive", "", 12, A, ""),                 # OREB: same possession
        (1, 585, "2pt", "", "", 12, A, "Made"),                          # and-one make
        (1, 585, "foul", "personal", "shooting", 3, H, "", {"drawn": 12}),
        (1, 585, "freethrow", "1 of 1", "", 12, A, "Made"),              # A possession ends on the FT
        (1, 570, "turnover", "bad pass", "", 4, H, "", {"steal": 13}),   # H ends (turnover)
        (1, 560, "freethrow", "1 of 2", "", 14, A, "Made"),
        (1, 560, "freethrow", "2 of 2", "", 14, A, "Missed"),            # last FT missed -> live
        (1, 559, "rebound", "defensive", "", 5, H, ""),                  # A ends (def rebound)
        (1, 550, "freethrow", "1 of 1", "technical", 1, H, "Made"),      # technical: no possession effect
        (1, 0, "period", "end", "", 0, 0, ""),
    ]
    df = ev(rows)
    g = GameArrays.from_frame(df, H, A)
    poss, stats = count_possessions(g)
    reasons = [p.reason for p in poss]
    assert reasons[:4] == ["made_fg", "ft_made", "turnover", "def_rebound"]
    assert poss[1].n_oreb == 1 and poss[1].points == 3
    assert stats["and_one"] == 1 and stats["technical_ft"] == 1


def test_lineups_and_box_identities():
    rows, t = starters_touch()
    rows += [
        (1, 650, "2pt", "", "", 1, H, "Made", {"assist": 2}),
        (1, 640, "substitution", "out", "", 5, H, ""),
        (1, 640, "substitution", "in", "", 6, H, ""),
        (1, 630, "3pt", "", "", 6, H, "Made"),
        (1, 620, "3pt", "", "", 11, A, "Missed"),
        (1, 619, "rebound", "defensive", "", 6, H, ""),
        (1, 610, "freethrow", "1 of 2", "", 2, H, "Made"),
        (1, 610, "freethrow", "2 of 2", "", 2, H, "Made"),
        (1, 0, "period", "end", "", 0, 0, ""),
    ]
    out = reconstruct_game(ev(rows), H, A)
    pl = out["players"].set_index("person_id")
    assert pl.loc[1, "fgm2"] == 1 and pl.loc[2, "ast"] == 1 and pl.loc[6, "fgm3"] == 1
    assert np.isclose(pl.loc[5, "sec"], 80.0) and np.isclose(pl.loc[6, "sec"], 640.0)
    home = pl[pl["team_id"] == H]
    assert np.isclose(home["sec"].sum(), 5 * 720.0)       # five on court for the whole period
    pts = 2 * home["fgm2"] + 3 * home["fgm3"] + home["ftm"]
    assert int(pts.sum()) == int(out["teams"].set_index("team_id").loc[H, "pts"]) == 7
    assert out["stints"]["valid"].all()
