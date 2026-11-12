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


def test_network_failures_are_recorded_and_the_fetch_goes_on(monkeypatch, tmp_path):
    old = "2026-10-21T23:30:00Z"
    gids = list(range(22600101, 22600111))                         # 10 finished games
    sched = {"leagueSchedule": {"gameDates": [{"gameDate": "x", "games": [_game(g, 3, old, hs=100, as_=90)
                                                                            for g in gids]}]}}
    bad = {22600102} | set(gids[3:])                               # one flaky game, then the network goes down

    def fake(req, timeout=None):
        if req.full_url == cdn.SCHEDULE_URL:
            return _Resp(sched)
        gid = int(req.full_url.rsplit("_", 1)[1].split(".")[0])
        if gid in bad:
            raise urllib.error.URLError(ConnectionResetError("connection reset"))
        return _Resp(_pbp(gid))

    monkeypatch.setattr(cdn, "PATHS", Paths(root=tmp_path))
    monkeypatch.setattr(cdn.urllib.request, "urlopen", fake)
    monkeypatch.setattr(cdn.time, "sleep", lambda s: None)
    s = cdn.fetch_live("2026-27", pause_s=0, log=lambda m: None)
    assert s["downloaded"] == 2 and s["stopped"]                   # 22600101 and 22600103 kept
    assert "FetchError" not in s["failed"]["22600102"] and "connection reset" in s["failed"]["22600102"]
    assert s["failed"][str(gids[-1])] == "not tried (fetch stopped)"   # stopped after 5 failures in a row
    man = json.loads((tmp_path / "raw" / "live" / "2026-27" / "manifest.json").read_text(encoding="utf-8"))
    assert set(man["games"]) == {"22600101", "22600103"}


def test_a_refusal_mid_fetch_keeps_the_manifest(monkeypatch, tmp_path):
    old = "2026-10-21T23:30:00Z"
    sched = {"leagueSchedule": {"gameDates": [{"gameDate": "x", "games": [
        _game(22600201, 3, old, hs=100, as_=90), _game(22600202, 3, old, hs=100, as_=90)]}]}}
    calls = []

    def fake(req, timeout=None):
        calls.append(req.full_url)
        if req.full_url == cdn.SCHEDULE_URL:
            return _Resp(sched)
        if "0022600202" in req.full_url:
            raise urllib.error.HTTPError(req.full_url, 403, "Access Denied", {}, None)
        return _Resp(_pbp(22600201))

    monkeypatch.setattr(cdn, "PATHS", Paths(root=tmp_path))
    monkeypatch.setattr(cdn.urllib.request, "urlopen", fake)
    try:
        cdn.fetch_live("2026-27", pause_s=0, log=lambda m: None)
        raise AssertionError("expected AccessDenied")
    except cdn.AccessDenied:
        pass
    assert sum("0022600202" in u for u in calls) == 1              # refused once, never retried
    man = json.loads((tmp_path / "raw" / "live" / "2026-27" / "manifest.json").read_text(encoding="utf-8"))
    assert "22600201" in man["games"]                              # the game saved before the refusal is known


def test_a_proxy_refusal_is_an_access_refusal(monkeypatch):
    calls = []

    def refuse(req, timeout=None):
        calls.append(req.full_url)
        raise urllib.error.URLError(OSError("Tunnel connection failed: 407 Proxy Authentication Required"))
    monkeypatch.setattr(cdn.urllib.request, "urlopen", refuse)
    try:
        cdn._get_json(cdn.SCHEDULE_URL)
        raise AssertionError("expected AccessDenied")
    except cdn.AccessDenied as e:
        assert "proxy" in str(e)
    assert len(calls) == 1


def test_gzip_responses_are_decompressed(monkeypatch):
    import gzip

    class Gz(_Resp):
        def __init__(self, obj):
            io.BytesIO.__init__(self, gzip.compress(json.dumps(obj).encode("utf-8")))
            self.headers = {"Content-Encoding": "gzip"}

    seen = {}

    def fake(req, timeout=None):
        seen["accept"] = req.get_header("Accept-encoding")
        return Gz({"ok": 1})
    monkeypatch.setattr(cdn.urllib.request, "urlopen", fake)
    assert cdn._get_json(cdn.SCHEDULE_URL) == {"ok": 1} and seen["accept"] == "gzip"
