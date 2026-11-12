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
    def game(gid, status, home, away, tip, hs=0, as_=0, text="", ppd="A", est=True):
        g = {"gameId": f"{gid:010d}", "gameStatus": status, "gameStatusText": text, "postponedStatus": ppd,
             "gameDateTimeUTC": tip,
             "homeTeam": {"teamId": home, "teamTricode": "HOM", "score": hs},
             "awayTeam": {"teamId": away, "teamTricode": "AWY", "score": as_}}
        if est:                                    # the feed's Eastern calendar date of the game, at midnight
            g["gameDateEst"] = pd.Timestamp(tip).tz_convert("America/New_York").strftime("%Y-%m-%dT00:00:00Z")
        return g
    games = [game(12600001, 3, 1, 2, "2026-10-05T23:00:00Z", 100, 90),                     # preseason: dropped
             game(22600001, 3, 1610612760, 1610612745, "2026-10-21T23:30:00Z", 110, 101, "Final"),
             game(22600002, 1, 1610612738, 1610612752, "2026-11-12T00:30:00Z", text="7:30 pm ET"),
             game(22600003, 1, 0, 0, "2026-12-14T01:00:00Z", text="TBD"),
             game(22600004, 1, 1610612747, 1610612744, "2026-11-13T03:00:00Z", text="PPD", ppd="P"),
             game(22600005, 1, 1610612737, 1610612751, "2026-11-14T03:30:00Z", text="10:30 pm ET", est=False)]
    return {"leagueSchedule": {"gameDates": [{"gameDate": "x", "games": games}]}}


def test_schedule_parser():
    s = read_schedule(_schedule_doc(), "2026-27", 226).set_index("game_id")
    assert 12600001 not in s.index                                         # regular season only
    assert s.loc[22600001, "status"] == "final" and s.loc[22600001, "home_score"] == 110
    assert s.loc[22600002, "status"] == "scheduled" and np.isnan(s.loc[22600002, "home_score"])
    assert str(s.loc[22600002, "game_date"].date()) == "2026-11-11"        # 7:30 pm ET on the 11th (00:30 UTC)
    assert str(s.loc[22600005, "game_date"].date()) == "2026-11-13"        # no Eastern date: tip - 6 h
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
def test_pregame_starters_and_availability_use_earlier_games_only():
    """The pregame spec's own inputs - the last starting five, who appeared, the missed-game streaks - come from games
    strictly before the date: scrambling the game's own appearances and starters, in the player-game table and in
    its feature rows, changes nothing. (Its minutes and rates are the feature store's rows; an unplayed game's are
    built as of its information date, a finished game's are the backtest's as-of ones - pregame mode is meant for
    unplayed games, and the cross-cutoff test in test_rehearsal_slow.py checks those end to end.)"""
    from quantlab.sim.inputs import InputStore, game_spec
    store = InputStore()
    gid = int(store.games[store.games["season"] == "2025-26"].index[400])
    a = game_spec(store, gid, mode="pregame")
    rng = np.random.default_rng(0)
    m = store.pg["game_id"] == gid
    for col in ("played", "starter"):
        store.pg.loc[m, col] = rng.permutation(store.pg.loc[m, col].to_numpy())
    f = store.by_game[gid].copy()
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


def test_info_dates_anchor_games_ahead_on_the_teams_next_game():
    from quantlab.features.asof import info_dates
    # built: 11-08 team 20, 11-09 team 10, 11-10 teams 50/60 (the league's last built date)
    games = pd.DataFrame({"season": ["2026-27"] * 3, "game_date": pd.to_datetime(["2026-11-08", "2026-11-09",
                                                                                   "2026-11-10"]),
                          "home_team_id": [20, 10, 50], "away_team_id": [70, 30, 60]})
    up = pd.DataFrame({"game_id": [1, 2, 3, 4, 5, 6], "team_id": [10, 10, 10, 20, 20, 40],
                       "season": ["2026-27"] * 6,
                       "game_date": pd.to_datetime(["2026-11-08", "2026-11-12", "2026-11-15", "2026-11-10",
                                                    "2026-11-11", "2026-11-12"])})
    d = info_dates(up, games)
    assert str(d[(1, 10)].date()) == "2026-11-08"          # final but not built, before its team's last game: own date
    assert str(d[(2, 10)].date()) == "2026-11-12" and str(d[(3, 10)].date()) == "2026-11-12"   # team's next game
    # team 20's game of 11-10 is still in progress: its back-to-back is anchored at it, not at its own date
    assert str(d[(4, 20)].date()) == "2026-11-10" and str(d[(5, 20)].date()) == "2026-11-10"
    assert str(d[(6, 40)].date()) == "2026-11-12"          # a team with no built game: its first game


def test_pregame_problem_names_a_team_without_a_game_yet():
    from quantlab.sim.inputs import InputStore
    st = object.__new__(InputStore)
    st.games = pd.DataFrame(index=pd.Index([], name="game_id"))
    st.schedule = pd.DataFrame({"home_team_id": [1, 1, 0, 1, 1], "away_team_id": [2, 2, 0, 2, 2],
                                "home_team": ["HOM"] * 5, "away_team": ["AWY"] * 5},
                               index=pd.Index([7, 8, 9, 10, 11], name="game_id"))
    st.tp = pd.DataFrame(index=pd.Index([7, 8, 10], name="game_id"))
    st.by_game = {7: pd.DataFrame({"team_id": [1, 2]}), 8: pd.DataFrame({"team_id": [1, 1]})}
    assert st.pregame_problem(7) is None
    assert "AWY has not played" in st.pregame_problem(8)
    assert "not known" in st.pregame_problem(9)
    assert "HOM and AWY have not played" in st.pregame_problem(10)       # opening night: no rosters at all
    assert "beyond the projection horizon" in st.pregame_problem(11)      # no team-model line either
    st.schedule["status"] = ["scheduled"] * 4 + ["postponed"]
    assert "postponed" in st.pregame_problem(11)


def test_overrides_round_trip_keeps_integer_keys(monkeypatch, tmp_path):
    from quantlab import live
    monkeypatch.setattr(live, "OVERRIDES_PATH", tmp_path / "live_overrides.json")
    live.save_overrides({"out": [5, 3], "available": [], "p_play": {7: 0.5}, "starters": {1610612738: [1, 2, 3, 4, 5]}})
    ov = live.load_overrides()
    assert ov["out"] == [3, 5] and ov["p_play"] == {7: 0.5} and ov["starters"] == {1610612738: [1, 2, 3, 4, 5]}


def test_overrides_file_in_any_encoding_or_broken(monkeypatch, tmp_path):
    from quantlab import live
    path = tmp_path / "live_overrides.json"
    monkeypatch.setattr(live, "OVERRIDES_PATH", path)
    body = '{"out": [7], "available": null}'
    for raw in (body.encode("utf-8-sig"), body.encode("utf-16"), body.encode("utf-8")):
        path.write_bytes(raw)
        ov = live.load_overrides()
        assert ov["out"] == [7] and ov["available"] == [] and "error" not in ov
    path.write_text('{"out": [7],}', encoding="utf-8")                  # a typo: ignored, with the reason
    ov = live.load_overrides()
    assert ov["out"] == [] and "could not be read" in ov["error"]


def _unplayed(pr, gid, monkeypatch):
    """Treat a finished game of the main data as unplayed (pregame inputs from its own feature rows)."""
    monkeypatch.setattr(pr.store, "is_final", lambda g: int(g) != gid and int(g) in pr.store.games.index)


@pytest.mark.skipif(not BUILT, reason="needs the built feature store and models")
def test_a_scenario_removal_on_an_unplayed_game_equals_the_injury_report(monkeypatch):
    from quantlab.live import Projector
    from quantlab.sim.scenario import Scenario
    pr = Projector()
    pr.overrides = {"out": [], "available": [], "p_play": {}, "starters": {}, "teams": {}}
    gid = int(pr.store.games[pr.store.games["season"] == "2025-26"].index[400])
    _unplayed(pr, gid, monkeypatch)
    base = pr.spec(gid)
    starter = next(p.person_id for p in base.home.players if p.starter)
    via_report = pr.spec(gid, overrides={**pr.overrides, "out": [starter]})
    via_scenario = pr.project(gid, n=200, environment="none", scenario=Scenario(out=[starter])).spec
    for a, b in ((via_report.home, via_scenario.home), (via_report.away, via_scenario.away)):
        assert [(p.person_id, p.starter, round(p.min_mean, 6)) for p in a.players] == \
               [(p.person_id, p.starter, round(p.min_mean, 6)) for p in b.players]
    assert starter not in {p.person_id for p in via_scenario.home.players}
    assert sum(p.starter for p in via_scenario.home.players) == 5          # the five is re-picked


@pytest.mark.skipif(not BUILT, reason="needs the built feature store and models")
def test_the_injury_report_can_bring_back_a_player_outside_the_pool(monkeypatch):
    from quantlab.live import Projector
    pr = Projector()
    gid = int(pr.store.games[pr.store.games["season"] == "2025-26"].index[400])
    _unplayed(pr, gid, monkeypatch)
    g = pr.store.games.loc[gid]
    have = set(pr.store.by_game[gid]["person_id"].astype(int))
    date = pd.Timestamp(g["game_date"])
    cand = [p for p, t, d in pr.store.recent_absentees([int(g["home_team_id"])], date, exclude=have)
            if pr.store.latest_rates(p, date) is not None]
    assert cand, "a long-absent player of the home team"
    pid = cand[0]
    ov = {"out": [], "available": [pid], "p_play": {}, "starters": {}, "teams": {pid: int(g["home_team_id"])}}
    spec = pr.spec(gid, overrides=ov)
    back = [p for p in spec.home.players if p.person_id == pid]
    assert back and back[0].p_play == 1.0 and back[0].rotation


@pytest.mark.skipif(not BUILT, reason="needs the built feature store and models")
def test_the_injury_report_acts_like_a_scenario(monkeypatch):
    """An injury-report 'out' and Scenario(out=[the same player]) give the same simulation, and ruling out the
    team's top scorer lowers its projected points (the team model does not know about the report)."""
    from quantlab.live import NO_REPORT, Projector
    from quantlab.sim.scenario import Scenario
    pr = Projector()
    gid = int(pr.store.games[pr.store.games["season"] == "2025-26"].index[400])
    _unplayed(pr, gid, monkeypatch)
    pr.overrides = dict(NO_REPORT)
    base = pr.project(gid, n=1000, seed=3, environment="blend")
    star = max(base.spec.home.players, key=lambda p: p.rates["usage"] * p.min_mean).person_id
    via_scenario = pr.project(gid, n=1000, seed=3, environment="blend", scenario=Scenario(out=[star]))
    pr.overrides = {**NO_REPORT, "out": [star]}
    via_report = pr.project(gid, n=1000, seed=3, environment="blend")
    assert (via_report.result.score == via_scenario.result.score).all()
    assert via_report.result.score[:, 0].mean() < base.result.score[:, 0].mean() - 0.3
    # the team model does not know about the report: the environment shift stays the default roster's
    assert via_report.calibration["environment_shift"] == base.calibration["environment_shift"]


@pytest.mark.skipif(not BUILT, reason="needs the built feature store and models")
def test_announced_starters_start_and_a_trade_moves_a_player(monkeypatch):
    from quantlab.live import NO_REPORT, Projector
    pr = Projector()
    gid = int(pr.store.games[pr.store.games["season"] == "2025-26"].index[400])
    _unplayed(pr, gid, monkeypatch)
    g = pr.store.games.loc[gid]
    home, away = int(g["home_team_id"]), int(g["away_team_id"])
    base = pr.default_spec(gid)
    deep = next(p for p in base.home.players if not p.rotation)        # a deep-bench player of the home team
    five = [p.person_id for p in base.home.players if p.starter][:4] + [deep.person_id]
    spec = pr.spec(gid, overrides={**NO_REPORT, "starters": {home: five}})
    st = {p.person_id: p for p in spec.home.players}
    assert {pid for pid, p in st.items() if p.starter} == set(five)
    assert st[deep.person_id].rotation and st[deep.person_id].p_play == 1.0
    # 'teams' moves a player: off his team's roster, onto the other team's
    mover = next(p for p in base.home.players if p.rotation and not p.starter).person_id
    spec = pr.spec(gid, overrides={**NO_REPORT, "available": [mover], "teams": {mover: away}})
    assert mover not in {p.person_id for p in spec.home.players}
    assert mover in {p.person_id for p in spec.away.players}
