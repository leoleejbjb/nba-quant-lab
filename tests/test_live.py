"""The current season: live-feed readers, the fetcher's access policy, the schedule tables, the missed-game streak,
and truly pregame specs (no information from the game itself)."""

from __future__ import annotations

import io
import tarfile
import urllib.error

import numpy as np
import pandas as pd
import pytest

from quantlab.config import PATHS, Paths
from quantlab.data import cdn
from quantlab.data.events import actions_frame, normalize_events, read_live_pbp, read_raw_pbp, read_schedule
from quantlab.features.asof import missed_streaks, streak_bucket

MIRROR = PATHS.raw / "cdnnba_2025.tar.xz"
BUILT = (PATHS.features / "player_asof.parquet").exists() and (PATHS.models / "player_rates.parquet").exists()


def _actions(gid=22500001, end=True):
    acts = [{"actionNumber": 1, "orderNumber": 10000, "clock": "PT12M00.00S", "period": 1, "actionType": "period",
             "subType": "start", "personId": 0, "scoreHome": "0", "scoreAway": "0", "qualifiers": []},
            {"actionNumber": 2, "orderNumber": 20000, "clock": "PT11M40.00S", "period": 1, "actionType": "2pt",
             "subType": "Jump Shot", "personId": 7, "teamId": 1610612760, "teamTricode": "OKC", "shotResult": "Made",
             "scoreHome": "2", "scoreAway": "0", "qualifiers": ["pointsinthepaint", "2ndchance"]},
            {"actionNumber": 3, "orderNumber": 30000, "clock": "PT00M00.00S", "period": 4, "actionType": "period",
             "subType": "end", "personId": 0, "scoreHome": "2", "scoreAway": "0", "qualifiers": []}]
    if end:
        acts.append({"actionNumber": 4, "orderNumber": 40000, "clock": "PT00M00.00S", "period": 4,
                     "actionType": "game", "subType": "end", "personId": 0, "scoreHome": "2", "scoreAway": "0"})
    return {"game": {"gameId": f"{gid:010d}", "actions": acts}}


def test_actions_frame_is_the_mirror_row_format():
    df = actions_frame(_actions())
    assert list(df["gameId"].unique()) == [22500001]
    assert df.loc[1, "qualifiers"] == "pointsinthepaint, 2ndchance"       # lists joined as in the mirror
    assert pd.isna(df.loc[0, "qualifiers"]) and pd.isna(df.loc[0, "teamId"])
    ev = normalize_events(read_live_pbp([_actions()], 225))
    assert int(ev["points"].sum()) == 2 and int(ev["score_home"].max()) == 2


@pytest.mark.skipif(not MIRROR.exists(), reason="needs the pinned 2025-26 mirror file")
def test_live_documents_rebuild_the_mirror_exactly():
    with tarfile.open(MIRROR, "r:xz") as t:
        full = pd.read_csv(io.BufferedReader(t.extractfile("cdnnba_2025.csv")), low_memory=False, nrows=60000)
    gids = sorted(full["gameId"].unique())[:3]
    full = full[full["gameId"].isin(gids)]
    raw = read_raw_pbp(MIRROR, "cdnnba_2025.csv", 225)
    a = normalize_events(raw[raw["gameId"].isin(gids)]).reset_index(drop=True)
    b = normalize_events(read_live_pbp([cdn.doc_from_mirror_rows(full[full.gameId == g]) for g in gids], 225))
    pd.testing.assert_frame_equal(a, b.reset_index(drop=True), check_dtype=False)


def test_check_pbp_rejects_in_progress_games():
    assert cdn.check_pbp(_actions(), 22500001) == []
    assert any("game end" in p for p in cdn.check_pbp(_actions(end=False), 22500001))
    assert any("gameId" in p for p in cdn.check_pbp(_actions(), 22500002))


def _schedule_doc():
    def game(gid, status, home, away, tip, hs=0, as_=0, text="", ppd="A"):
        return {"gameId": f"{gid:010d}", "gameStatus": status, "gameStatusText": text, "postponedStatus": ppd,
                "gameDateEst": tip[:10] + "T00:00:00Z", "gameDateTimeUTC": tip,
                "homeTeam": {"teamId": home, "teamTricode": "HOM", "score": hs},
                "awayTeam": {"teamId": away, "teamTricode": "AWY", "score": as_}}
    games = [game(12600001, 3, 1, 2, "2026-10-05T23:00:00Z", 100, 90),                     # preseason: dropped
             game(22600001, 3, 1610612760, 1610612745, "2026-10-21T23:30:00Z", 110, 101, "Final"),
             game(22600002, 1, 1610612738, 1610612752, "2026-11-12T00:30:00Z", text="7:30 pm ET"),
             game(22600003, 1, 0, 0, "2026-12-14T01:00:00Z", text="TBD"),
             game(22600004, 1, 1610612747, 1610612744, "2026-11-13T03:00:00Z", text="PPD", ppd="P")]
    return {"leagueSchedule": {"gameDates": [{"gameDate": "x", "games": games}]}}


def test_schedule_parser():
    s = read_schedule(_schedule_doc(), "2026-27", 226).set_index("game_id")
    assert 12600001 not in s.index                                         # regular season only
    assert s.loc[22600001, "status"] == "final" and s.loc[22600001, "home_score"] == 110
    assert s.loc[22600002, "status"] == "scheduled" and np.isnan(s.loc[22600002, "home_score"])
    assert str(s.loc[22600002, "game_date"].date()) == "2026-11-12"        # the Eastern date
    assert s.loc[22600003, "status"] == "tbd" and s.loc[22600004, "status"] == "postponed"
    with pytest.raises(ValueError):
        read_schedule({"unexpected": 1}, "2026-27", 226)


def test_access_refusal_stops_the_fetch_without_retrying(monkeypatch):
    calls = []

    def refuse(req, timeout=None):
        calls.append(req.full_url)
        raise urllib.error.HTTPError(req.full_url, 403, "Access Denied", {}, None)
    monkeypatch.setattr(cdn.urllib.request, "urlopen", refuse)
    with pytest.raises(cdn.AccessDenied):
        cdn._get_json(cdn.SCHEDULE_URL)
    assert len(calls) == 1                                                # reported, never retried


def test_missed_streaks():
    d = pd.to_datetime(["2026-11-01", "2026-11-03", "2026-11-05", "2026-11-07"])
    pg = pd.DataFrame({"team_id": [1] * 7, "season": ["2026-27"] * 7, "game_id": [1, 1, 2, 2, 3, 4, 4],
                       "person_id": [10, 11, 10, 11, 11, 10, 11],
                       "game_date": [d[0], d[0], d[1], d[1], d[2], d[3], d[3]], "played": [True] * 7})
    rows = pd.DataFrame({"team_id": [1, 1, 1, 1], "season": ["2026-27"] * 4, "person_id": [10, 11, 10, 12],
                         "game_date": pd.to_datetime(["2026-11-07", "2026-11-07", "2026-11-09", "2026-11-09"])})
    # 10 missed game 3 -> 1 before game 4; 11 played game 3 -> 0; on 11-09, 10 played game 4 -> 0; 12 never -> 4
    assert missed_streaks(pg, rows).tolist() == [1, 0, 0, 4]
    assert list(streak_bucket([0, 1, 2, 3, 7, 12])) == ["0", "1", "2", "3-4", "5-9", "10+"]


def test_upcoming_games_follow_the_last_built_game(monkeypatch, tmp_path):
    from quantlab.data import database
    paths = Paths(root=tmp_path)
    (tmp_path / "processed" / "all").mkdir(parents=True)
    ts = pd.DataFrame({"game_id": [1, 1, 2, 2, 3, 3], "team_id": [10, 20, 10, 20, 10, 20],
                       "opp_team_id": [20, 10, 20, 10, 20, 10], "is_home": [True, False] * 3,
                       "season": ["2026-27"] * 6,
                       "game_date": pd.to_datetime(["2026-11-01"] * 2 + ["2026-11-05"] * 2 + ["2026-11-30"] * 2),
                       "tip_utc": pd.NaT, "rest_days": 2.0, "b2b": False, "status": ["final"] * 2 + ["scheduled"] * 4,
                       "built": [True] * 2 + [False] * 4})
    ts.to_parquet(tmp_path / "processed" / "all" / "team_schedule.parquet")
    monkeypatch.setattr(database, "PATHS", paths)
    monkeypatch.setattr(database, "LIVE_SEASON", "2026-27")
    up = database.upcoming_team_games(horizon_days=10)
    assert sorted(up["game_id"].unique()) == [2]                          # game 3 is beyond the horizon


def test_fetch_of_the_live_season_points_to_live_update():
    from quantlab.cli import main
    with pytest.raises(SystemExit) as e:
        main(["fetch", "--season", "2026-27"])
    assert "live-update" in str(e.value)


@pytest.mark.skipif(not BUILT, reason="needs the built feature store and models")
def test_pregame_spec_uses_nothing_from_the_game():
    """A pregame spec of a finished game is unchanged when the game's own appearances and starters are scrambled."""
    from quantlab.sim.inputs import InputStore, game_spec
    store = InputStore()
    gid = int(store.games[store.games["season"] == "2025-26"].index[400])
    a = game_spec(store, gid, mode="pregame")
    f = store.by_game[gid].copy()
    rng = np.random.default_rng(0)
    f["played"] = rng.permutation(f["played"].to_numpy())
    f["starter"] = rng.permutation(f["starter"].to_numpy())
    store.by_game[gid] = f
    b = game_spec(store, gid, mode="pregame")
    for ta, tb in ((a.home, b.home), (a.away, b.away)):
        assert [(p.person_id, p.starter, round(p.min_mean, 6), p.p_play) for p in ta.players] == \
               [(p.person_id, p.starter, round(p.min_mean, 6), p.p_play) for p in tb.players]
        assert sum(p.starter for p in ta.players) == 5
    assert a.meta["mode"] == "pregame"


def test_result_label_shows_tip_time_for_unplayed_games():
    from quantlab.dashboard.view_slate import result_label
    sched = pd.Series({"is_final": False, "status": "scheduled", "home_pts": np.nan, "away_pts": np.nan,
                       "tip_utc": pd.Timestamp("2026-11-12 00:30", tz="UTC"), "home_team": "BOS", "away_team": "MEM"})
    assert result_label(sched) == "7:30 PM ET"
    fin = pd.Series({"is_final": True, "status": "final", "home_pts": 110.0, "away_pts": 101.0,
                     "home_team": "OKC", "away_team": "HOU"})
    assert result_label(fin) == "HOU 101 - 110 OKC"
    assert result_label(pd.Series({**sched.to_dict(), "status": "postponed"})) == "Postponed"
