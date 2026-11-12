"""The current season from the NBA's live CDN (cdn.nba.com).

``fetch_live`` downloads the league schedule and, for every finished regular-season game, its liveData play-by-play -
the same feed the pinned mirror archives once a season is over, so the season is reconstructed by the same code.
Files are not pinned (the season is still being played): each game file is checked for completeness (a 'game end'
action) before it is kept, stored gzip-compressed under ``data/raw/live/<season>/`` with its SHA-256 in a manifest, and
fetched again for a while after the game to pick up the league's post-game stat corrections.

Like every download in this project, requests identify themselves honestly (``download.USER_AGENT``). An access refusal
(HTTP 401/403/407/451 from the CDN, or a proxy refusing the connection) stops the fetch: it is reported and never
retried under another identity. cdn.nba.com refuses some cloud machines; run ``quantlab live-update`` on your own
computer. Other failures of a game file (not published yet, a dropped connection, a server error after retries) are
recorded and that game is tried again on the next update; after several network failures in a row the fetch stops
and the update goes on with the files already on disk.

``doc_from_mirror_rows`` and ``schedule_doc`` turn the pinned mirror back into live-format files; they let the whole
live path be rehearsed on a finished season (scripts/rehearse_live.py) and tested without network access.
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import http.client
import json
import os
import re
import time
import urllib.error
import zlib
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..config import LIVE_SEASON, PATHS, Season
from .download import DENIED, USER_AGENT, AccessDenied
from .events import parse_clock, read_schedule

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
PBP_URL = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id:010d}.json"
LIC_LIVE = "NBA.com live data feed; NBA.com Terms of Use (personal, non-commercial)"
# a proxy that refuses the HTTPS tunnel surfaces as a URLError, not an HTTPError
_TUNNEL_REFUSED = re.compile(r"Tunnel connection failed: (401|403|407|451)\b")
STOP_AFTER_FAILURES = 5          # network failures of game files in a row before the fetch gives up for this run


class FetchError(RuntimeError):
    """A request that still failed after its retries (network trouble, server errors, a corrupt response)."""


def schedule_path(season: str) -> Path:
    return PATHS.live_raw(season) / "schedule.json.gz"


def pbp_path(season: str, game_id: int) -> Path:
    return PATHS.live_raw(season) / "pbp" / f"{int(game_id):010d}.json.gz"


def write_json_gz(obj: dict, path: Path) -> str:
    """Write ``obj`` as gzip-compressed JSON (atomically); return the SHA-256 of the compressed file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = gzip.compress(json.dumps(obj, separators=(",", ":")).encode("utf-8"), mtime=0)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def read_json_gz(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def load_schedule(season: str = LIVE_SEASON) -> pd.DataFrame:
    path = schedule_path(season)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist: run `quantlab live-update` (it fetches the {season} schedule)")
    return read_schedule(read_json_gz(path), season, Season(season).game_prefix)


def _get_json(url: str, timeout: float = 60.0, retries: int = 3) -> dict:
    """GET a JSON document (gzip-compressed in transit). An access refusal raises ``AccessDenied`` at once and a 404
    is re-raised as is; other failures are retried with back-off and end in ``FetchError``."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json",
                                                       "Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
                if str(r.headers.get("Content-Encoding") or "").lower() == "gzip":
                    body = gzip.decompress(body)
            return json.loads(body.decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in DENIED:
                raise AccessDenied(f"{url}: HTTP {e.code} (access refused; not retried)") from e
            if e.code == 404:
                raise
            last = e
        except urllib.error.URLError as e:
            if _TUNNEL_REFUSED.search(str(e.reason)):
                raise AccessDenied(f"{url}: a proxy refused the connection ({e.reason}; not retried)") from e
            last = e
        except (OSError, EOFError, ValueError, zlib.error, http.client.HTTPException) as e:
            # timeouts, connections cut off mid-transfer, TLS errors, truncated or corrupt bodies
            last = e
        time.sleep(2 ** (attempt + 1))
    raise FetchError(f"{url}: {type(last).__name__}: {last}")


def check_pbp(doc: dict, game_id: int) -> list[str]:
    """Reasons a play-by-play document cannot be used as a finished game (empty list = complete)."""
    try:
        game = doc["game"]
        acts = game.get("actions") or []
    except (KeyError, TypeError, AttributeError):
        return ["no game.actions"]
    problems = []
    if int(game.get("gameId", 0)) != int(game_id):
        problems.append(f"gameId {game.get('gameId')} != {game_id}")
    if not acts:
        return problems + ["no actions"]
    if not any(a.get("actionType") == "game" and a.get("subType") == "end" for a in acts):
        problems.append("no 'game end' action (game in progress or truncated)")
    if max(int(a.get("period") or 0) for a in acts) < 4:
        problems.append("fewer than four periods")
    try:
        parse_clock(pd.Series([a.get("clock") for a in acts]))
    except ValueError as e:
        problems.append(str(e))
    return problems


def fetch_live(season: str = LIVE_SEASON, refetch_hours: float = 36.0, pause_s: float = 1.0,
               log: Callable[[str], None] = print) -> dict:
    """Refresh the schedule, then download every finished game's play-by-play that is missing, incomplete, or
    recent enough (``refetch_hours`` after tip-off) to still receive stat corrections."""
    base = PATHS.live_raw(season)
    base.mkdir(parents=True, exist_ok=True)
    manifest_path = base / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    now = dt.datetime.now(dt.timezone.utc)
    stamp = now.isoformat(timespec="seconds")

    log(f"[live] schedule <- {SCHEDULE_URL}")
    doc = _get_json(SCHEDULE_URL)
    sched = read_schedule(doc, season, Season(season).game_prefix)
    if sched.empty:
        raise RuntimeError(f"the schedule at {SCHEDULE_URL} has no {season} regular-season games "
                           "(has the season changed? set QUANTLAB_LIVE_SEASON)")
    manifest["schedule"] = {"url": SCHEDULE_URL, "sha256": write_json_gz(doc, schedule_path(season)),
                            "fetched_at": stamp, "games": int(len(sched)), "license": LIC_LIVE,
                            "status_counts": {k: int(v) for k, v in sched["status"].value_counts().items()}}

    games = manifest.setdefault("games", {})
    fetched, cached, failed, in_a_row, stopped = 0, 0, {}, 0, None
    final = sched[sched["status"] == "final"]
    try:
        for gid, tip in zip(final["game_id"].to_numpy(), final["tip_utc"]):
            key, path = str(int(gid)), pbp_path(season, gid)
            recent = pd.notna(tip) and (now - tip.to_pydatetime()) < dt.timedelta(hours=refetch_hours)
            if path.exists() and games.get(key, {}).get("complete") and not recent:
                cached += 1
                continue
            if stopped:
                failed[key] = "not tried (fetch stopped)"
                continue
            url = PBP_URL.format(game_id=int(gid))
            try:
                pdoc = _get_json(url)
            except urllib.error.HTTPError as e:        # e.g. 404: not published yet - try again next time
                failed[key] = f"HTTP {e.code}"
                in_a_row = 0                           # the server answered: not a network failure
                continue
            except FetchError as e:                    # network trouble or server errors after retries
                failed[key] = str(e)
                in_a_row += 1
                if in_a_row >= STOP_AFTER_FAILURES:
                    stopped = f"{in_a_row} game files failed in a row (last: {e})"
                    log(f"[live] stopping the fetch: {stopped}; the update goes on with the files on disk")
                continue
            finally:
                time.sleep(pause_s)                    # every request is paced, failed ones too
            in_a_row = 0
            problems = check_pbp(pdoc, int(gid))
            if problems:
                failed[key] = "; ".join(problems)
                continue
            games[key] = {"url": url, "sha256": write_json_gz(pdoc, path), "fetched_at": stamp,
                          "actions": len(pdoc["game"]["actions"]), "complete": True}
            fetched += 1
            if fetched % 25 == 0:
                log(f"[live] {fetched} games downloaded ...")
                _write_manifest(manifest, failed, manifest_path)
    finally:                                           # also on a refusal or Ctrl+C: what is on disk stays known
        _write_manifest(manifest, failed, manifest_path)
    summary = {"season": season, "scheduled_games": int(len(sched)), "final": int(len(final)), "downloaded": fetched,
               "cached": cached, "failed": failed, "stopped": stopped}
    log(f"[live] {season}: {len(final)} of {len(sched)} games final; {fetched} downloaded, {cached} cached, "
        f"{len(failed)} not available yet")
    return summary


def _write_manifest(manifest: dict, failed: dict, path: Path) -> None:
    manifest["failed"] = failed
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------ rehearsal / tests: mirror -> live format
_INT_FIELDS = {"actionNumber", "orderNumber", "period", "personId", "isFieldGoal", "possession"}
_ID_FIELDS = {"teamId", "assistPersonId", "foulDrawnPersonId", "stealPersonId", "blockPersonId",
              "jumpBallWonPersonId", "jumpBallLostPersonId", "jumpBallRecoverdPersonId", "shotActionNumber",
              "officialId"}


def doc_from_mirror_rows(rows: pd.DataFrame) -> dict:
    """The liveData document of one game rebuilt from the mirror's CSV rows (the inverse of ``events.actions_frame``):
    absent fields are left out, list fields are lists and the running score is a string, as in the feed."""
    gid = int(rows["gameId"].iloc[0])
    acts = []
    for rec in rows.drop(columns=["gameId"]).to_dict("records"):
        a = {}
        for k, v in rec.items():
            if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
                continue
            if k in _INT_FIELDS or k in _ID_FIELDS:
                v = int(v)
            elif k in ("scoreHome", "scoreAway"):
                v = str(int(v))
            elif k in ("qualifiers", "personIdsFilter"):
                v = [x.strip() for x in str(v).split(",") if x.strip()]
                if k == "personIdsFilter":
                    v = [int(float(x)) for x in v]
            elif isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = float(v)
            elif isinstance(v, (np.bool_,)):
                v = bool(v)
            a[k] = v
        acts.append(a)
    return {"meta": {"version": 1}, "game": {"gameId": f"{gid:010d}", "actions": acts}}


def schedule_doc(games: pd.DataFrame, final_ids: set) -> dict:
    """A scheduleLeagueV2-style document for ``games`` (game_id, game_date, tip_utc, home/away ids, tricodes and
    final scores): ids in ``final_ids`` are final with their score, the rest scheduled."""
    by_date: dict = {}
    for r in games.sort_values(["game_date", "game_id"]).itertuples(index=False):
        fin = int(r.game_id) in final_ids
        tip = pd.Timestamp(r.tip_utc)
        tip = tip.tz_localize("UTC") if tip.tzinfo is None else tip.tz_convert("UTC")
        g = {"gameId": f"{int(r.game_id):010d}", "gameStatus": 3 if fin else 1,
             "gameStatusText": "Final" if fin else tip.strftime("%I:%M %p UTC"),
             "gameDateEst": pd.Timestamp(r.game_date).strftime("%Y-%m-%dT00:00:00Z"),
             "gameDateTimeUTC": tip.strftime("%Y-%m-%dT%H:%M:%SZ"), "postponedStatus": "A",
             "homeTeam": {"teamId": int(r.home_team_id), "teamTricode": str(r.home_team),
                          "score": int(r.final_home) if fin else 0},
             "awayTeam": {"teamId": int(r.away_team_id), "teamTricode": str(r.away_team),
                          "score": int(r.final_away) if fin else 0}}
        by_date.setdefault(pd.Timestamp(r.game_date).strftime("%m/%d/%Y 00:00:00"), []).append(g)
    return {"meta": {"version": 1}, "leagueSchedule": {"gameDates": [{"gameDate": d, "games": gs}
                                                                    for d, gs in by_date.items()]}}
