"""The current-season fetcher against a fake cdn.nba.com: what is kept, what is retried, what is cached."""

from __future__ import annotations

import io
import json
import urllib.error

import pandas as pd

from quantlab.config import Paths
from quantlab.data import cdn


def _game(gid, status, tip, home=1610612760, away=1610612745, hs=0, as_=0):
    return {"gameId": f"{gid:010d}", "gameStatus": status, "gameStatusText": "Final" if status == 3 else "7:00 pm ET",
            "postponedStatus": "A", "gameDateEst": tip[:10] + "T00:00:00Z", "gameDateTimeUTC": tip,
            "homeTeam": {"teamId": home, "teamTricode": "OKC", "score": hs},
            "awayTeam": {"teamId": away, "teamTricode": "HOU", "score": as_}}


def _pbp(gid, end=True):
    acts = [{"actionNumber": 1, "orderNumber": 10000, "clock": "PT12M00.00S", "period": 1, "actionType": "period",
             "subType": "start", "personId": 0},
            {"actionNumber": 2, "orderNumber": 20000, "clock": "PT00M00.00S", "period": 4, "actionType": "period",
             "subType": "end", "personId": 0}]
    if end:
        acts.append({"actionNumber": 3, "orderNumber": 30000, "clock": "PT00M00.00S", "period": 4,
                     "actionType": "game", "subType": "end", "personId": 0})
    return {"game": {"gameId": f"{gid:010d}", "actions": acts}}


class _Resp(io.BytesIO):
    def __init__(self, obj):
        super().__init__(json.dumps(obj).encode("utf-8"))
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def test_fetch_keeps_complete_games_and_retries_the_rest(monkeypatch, tmp_path):
    old = "2026-10-21T23:30:00Z"                                   # long finished: cached after the first fetch
    sched = {"leagueSchedule": {"gameDates": [{"gameDate": "x", "games": [
        _game(22600001, 3, old, hs=110, as_=101),                   # complete play-by-play
        _game(22600002, 3, old, hs=99, as_=98),                     # final, but the file has no 'game end' yet
        _game(22600003, 3, old, hs=120, as_=100),                   # final, play-by-play not published (404)
        _game(22600004, 1, "2099-11-12T00:30:00Z")]}]}}             # scheduled: never requested
    calls = []

    def fake(req, timeout=None):
        url = req.full_url
        calls.append(url)
        assert req.get_header("User-agent") == cdn.USER_AGENT     # the program identifies itself
        if url == cdn.SCHEDULE_URL:
            return _Resp(sched)
        gid = int(url.rsplit("_", 1)[1].split(".")[0])
        if gid == 22600003:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        return _Resp(_pbp(gid, end=gid != 22600002))

    monkeypatch.setattr(cdn, "PATHS", Paths(root=tmp_path))
    monkeypatch.setattr(cdn.urllib.request, "urlopen", fake)
    s = cdn.fetch_live("2026-27", pause_s=0, log=lambda m: None)
    assert s["final"] == 3 and s["downloaded"] == 1 and set(s["failed"]) == {"22600002", "22600003"}
    assert "game end" in s["failed"]["22600002"] and s["failed"]["22600003"] == "HTTP 404"
    assert cdn.pbp_path("2026-27", 22600001).exists() and not cdn.pbp_path("2026-27", 22600002).exists()
    assert not any("0022600004" in u for u in calls)
    man = json.loads((tmp_path / "raw" / "live" / "2026-27" / "manifest.json").read_text(encoding="utf-8"))
    assert man["games"]["22600001"]["complete"] and man["schedule"]["games"] == 4
    assert cdn.read_json_gz(cdn.pbp_path("2026-27", 22600001))["game"]["gameId"] == "0022600001"
    sc = cdn.load_schedule("2026-27").set_index("game_id")
    assert sc.loc[22600001, "status"] == "final" and pd.isna(sc.loc[22600004, "home_score"])

    calls.clear()                                                  # second run: the finished game is cached
    s2 = cdn.fetch_live("2026-27", pause_s=0, log=lambda m: None)
    assert s2["cached"] == 1 and s2["downloaded"] == 0
    assert not any("0022600001" in u for u in calls) and any("0022600002" in u for u in calls)


def test_recent_games_are_fetched_again_for_stat_corrections(monkeypatch, tmp_path):
    tip = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    sched = {"leagueSchedule": {"gameDates": [{"gameDate": "x", "games": [_game(22600010, 3, tip, hs=100, as_=90)]}]}}
    n = {"pbp": 0}

    def fake(req, timeout=None):
        if req.full_url == cdn.SCHEDULE_URL:
            return _Resp(sched)
        n["pbp"] += 1
        return _Resp(_pbp(22600010))

    monkeypatch.setattr(cdn, "PATHS", Paths(root=tmp_path))
    monkeypatch.setattr(cdn.urllib.request, "urlopen", fake)
    cdn.fetch_live("2026-27", pause_s=0, log=lambda m: None)
    cdn.fetch_live("2026-27", pause_s=0, log=lambda m: None)
    assert n["pbp"] == 2                                           # within 36 h of tip-off: downloaded again
    cdn.fetch_live("2026-27", refetch_hours=1, pause_s=0, log=lambda m: None)
    assert n["pbp"] == 2                                           # outside the window: read from disk
