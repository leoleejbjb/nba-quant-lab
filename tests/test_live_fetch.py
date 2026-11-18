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
        cdn.fetch_live("2026-27", pause_s=0, log=lambda m: None, source="cdn")      # CDN only: a refusal stops it
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


def test_a_corrupt_body_is_retried_and_a_404_resets_the_failure_count(monkeypatch, tmp_path):
    import gzip

    class Bad(_Resp):
        def __init__(self):
            io.BytesIO.__init__(self, gzip.compress(b'{"x": 1}')[:12] + b"\xff" * 20)   # gzip header, bad stream
            self.headers = {"Content-Encoding": "gzip"}

    calls = []
    monkeypatch.setattr(cdn.time, "sleep", lambda s: None)
    monkeypatch.setattr(cdn.urllib.request, "urlopen", lambda req, timeout=None: calls.append(1) or Bad())
    try:
        cdn._get_json(cdn.SCHEDULE_URL)
        raise AssertionError("expected FetchError")
    except cdn.FetchError:
        pass
    assert len(calls) == 3                                         # retried, then reported
    old = "2026-10-21T23:30:00Z"
    gids = list(range(22600401, 22600411))
    sched = {"leagueSchedule": {"gameDates": [{"gameDate": "x", "games": [_game(g, 3, old, hs=100, as_=90)
                                                                            for g in gids]}]}}

    def fake(req, timeout=None):                                   # network failures alternate with 404s
        if req.full_url == cdn.SCHEDULE_URL:
            return _Resp(sched)
        gid = int(req.full_url.rsplit("_", 1)[1].split(".")[0])
        if gid % 2:
            raise urllib.error.URLError(ConnectionResetError("reset"))
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)
    monkeypatch.setattr(cdn, "PATHS", Paths(root=tmp_path))
    monkeypatch.setattr(cdn.urllib.request, "urlopen", fake)
    s = cdn.fetch_live("2026-27", pause_s=0, log=lambda m: None)
    assert not s["stopped"] and "not tried" not in "".join(s["failed"].values())


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


# ---------------------------------------------------------------------------------------------- the mirror fallback
class _Raw(io.BytesIO):
    def __init__(self, body: bytes, headers=None):
        super().__init__(body)
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _mirror_tar(docs, key="cdnnba_2026") -> bytes:
    import tarfile
    from quantlab.data.events import actions_frame
    csv = pd.concat([actions_frame(d) for d in docs], ignore_index=True).to_csv(index=False).encode("utf-8")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:xz") as t:
        info = tarfile.TarInfo(f"{key}.csv")
        info.size = len(csv)
        t.addfile(info, io.BytesIO(csv))
    return buf.getvalue()


def _fallback_world(monkeypatch, tmp_path, listed=True):
    """cdn.nba.com refuses; the mirror lists 2026-27 with two complete games and one cut short; a saved schedule."""
    old = "2026-10-21T23:30:00Z"
    sched = {"leagueSchedule": {"gameDates": [{"gameDate": "x", "games": [
        _game(22600501, 3, old, hs=2, as_=0), _game(22600502, 1, old),      # 502: the saved copy predates its end
        _game(22600503, 3, old, hs=2, as_=0), _game(22600504, 1, "2099-11-12T00:30:00Z")]}]}}
    saved = tmp_path / "Downloads" / "scheduleLeagueV2.json"
    saved.parent.mkdir(parents=True)
    saved.write_text(json.dumps(sched), encoding="utf-8")
    tar = _mirror_tar([_pbp(22600501), _pbp(22600502), _pbp(22600503, end=False)])
    url = "https://example.invalid/cdnnba_2026.tar.xz"
    seen = {"etag_sent": [], "calls": []}

    def fake(req, timeout=None):
        u = req.full_url
        seen["calls"].append(u)
        if u.startswith("https://cdn.nba.com/"):
            raise urllib.error.HTTPError(u, 403, "Forbidden", {}, None)
        if u == cdn.MIRROR_LIST_URL:
            return _Raw(("cdnnba_2025=https://example.invalid/cdnnba_2025.tar.xz\n" +
                         (f"cdnnba_2026={url}\n" if listed else "")).encode("utf-8"))
        if u == url:
            seen["etag_sent"].append(req.get_header("If-none-match"))
            if req.get_header("If-none-match") == '"v1"':
                raise urllib.error.HTTPError(u, 304, "Not Modified", {}, None)
            return _Raw(tar, {"ETag": '"v1"'})
        raise AssertionError(u)

    monkeypatch.setattr(cdn, "PATHS", Paths(root=tmp_path / "data"))
    monkeypatch.setattr(cdn.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(cdn.urllib.request, "urlopen", fake)
    return seen


def test_a_refusal_falls_back_to_the_mirror_and_a_saved_schedule(monkeypatch, tmp_path):
    seen = _fallback_world(monkeypatch, tmp_path)
    s = cdn.fetch_live("2026-27", pause_s=0, log=lambda m: None)
    assert s["source"] == "mirror" and "403" in s["refused"] and s["schedule"]["file"].endswith("scheduleLeagueV2.json")
    assert s["games_in_mirror"] == 3 and s["downloaded"] == 2 and "game end" in s["failed"]["22600503"]
    assert cdn.pbp_path("2026-27", 22600501).exists() and cdn.pbp_path("2026-27", 22600502).exists()
    assert sum(u.startswith("https://cdn.nba.com/") for u in seen["calls"]) == 1      # refused once, not retried
    man = json.loads((tmp_path / "data" / "raw" / "live" / "2026-27" / "manifest.json").read_text(encoding="utf-8"))
    assert man["mirror"]["etag"] == '"v1"' and man["games"]["22600501"]["source"] == "mirror"
    s2 = cdn.fetch_live("2026-27", pause_s=0, log=lambda m: None)    # unchanged mirror: a 304, nothing rewritten
    assert seen["etag_sent"][-1] == '"v1"' and s2["downloaded"] == 0
    # the build takes every complete game file, also one the saved schedule does not show as final yet
    from quantlab.data import build
    raw, sch = build.live_season_inputs("2026-27", log=lambda m: None)
    assert set(raw["gameId"].astype(int)) == {22600501, 22600502}
    assert sch.set_index("game_id").loc[22600502, "status"] == "final"
    assert sch.attrs["missing_pbp"] == [22600503]               # cut short: never written, so missing


def test_without_a_saved_schedule_the_refusal_says_how_to_save_one(monkeypatch, tmp_path):
    _fallback_world(monkeypatch, tmp_path)
    (tmp_path / "Downloads" / "scheduleLeagueV2.json").unlink()
    try:
        cdn.fetch_live("2026-27", pause_s=0, log=lambda m: None)
        raise AssertionError("expected ScheduleNeeded")
    except cdn.ScheduleNeeded as e:
        assert "scheduleLeagueV2.json" in str(e) and "Ctrl+S" in str(e)


def test_the_mirror_has_not_published_the_season_yet(monkeypatch, tmp_path):
    _fallback_world(monkeypatch, tmp_path, listed=False)
    s = cdn.fetch_live("2026-27", pause_s=0, log=lambda m: None)
    assert s["mirror"] == "not published yet" and s["downloaded"] == 0


def test_the_newest_saved_schedule_is_used(monkeypatch, tmp_path):
    import os as _os
    monkeypatch.setattr(cdn, "PATHS", Paths(root=tmp_path / "data"))
    monkeypatch.setattr(cdn.Path, "home", classmethod(lambda cls: tmp_path))
    d = tmp_path / "Downloads"
    d.mkdir()
    (d / "scheduleLeagueV2.json").write_text("{}", encoding="utf-8")
    (d / "scheduleLeagueV2 (1).json").write_text("{}", encoding="utf-8")
    _os.utime(d / "scheduleLeagueV2.json", (1_000_000_000, 1_000_000_000))
    assert cdn.saved_schedule_file("2026-27").name == "scheduleLeagueV2 (1).json"
    assert cdn.saved_schedule_file("2026-27", d / "scheduleLeagueV2.json").name == "scheduleLeagueV2.json"


def test_a_refusal_of_game_files_falls_back_to_the_mirror_for_the_rest(monkeypatch, tmp_path):
    seen = _fallback_world(monkeypatch, tmp_path)
    old = "2026-10-21T23:30:00Z"
    sched = {"leagueSchedule": {"gameDates": [{"gameDate": "x", "games": [_game(22600501, 3, old, hs=2, as_=0),
                                                                            _game(22600502, 3, old, hs=2, as_=0)]}]}}
    inner = cdn.urllib.request.urlopen

    def fake(req, timeout=None):                                   # the schedule answers, the game files are refused
        if req.full_url == cdn.SCHEDULE_URL:
            return _Raw(json.dumps(sched).encode("utf-8"))
        return inner(req, timeout)
    monkeypatch.setattr(cdn.urllib.request, "urlopen", fake)
    s = cdn.fetch_live("2026-27", pause_s=0, log=lambda m: None)
    assert s["source"] == "cdn+mirror" and s["downloaded"] == 2
    assert sum("playbyplay" in u for u in seen["calls"]) == 1      # the first refusal stops the CDN requests
