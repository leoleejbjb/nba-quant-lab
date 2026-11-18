"""The current season from the NBA's live CDN (cdn.nba.com), or from the open mirror when the CDN refuses.

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

When the CDN refuses (``source="auto"``, the default) the season comes from elsewhere instead: the schedule from a
copy of ``scheduleLeagueV2.json`` the user saved in a browser (``saved_schedule_file``: the live folder, the Downloads
folder, or ``--schedule``), and the finished games from the open mirror the pinned seasons come from
(shufinskiy/nba_data, ``cdnnba_<year>.tar.xz``, updated every few weeks during a season), converted to the same
per-game documents. Finished games then lag by up to the mirror's update interval.

``doc_from_mirror_rows`` and ``schedule_doc`` turn the pinned mirror back into live-format files; they let the whole
live path be rehearsed on a finished season (scripts/rehearse_live.py) and tested without network access.
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import http.client
import io
import json
import os
import re
import tarfile
import time
import urllib.error
import urllib.request
import zlib
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
MIRROR_REPO = "shufinskiy/nba_data"
MIRROR_LIST_URL = f"https://raw.githubusercontent.com/{MIRROR_REPO}/main/list_data.txt"
LIC_MIRROR = f"{MIRROR_REPO} (Apache-2.0) mirror of the NBA liveData feed; data NBA.com Terms of Use"
SAVED_SCHEDULE = "scheduleLeagueV2.json"       # the name a browser gives the schedule when it is saved
# a proxy that refuses the HTTPS tunnel surfaces as a URLError, not an HTTPError
_TUNNEL_REFUSED = re.compile(r"Tunnel connection failed: (401|403|407|451)\b")
STOP_AFTER_FAILURES = 5          # network failures of game files in a row before the fetch gives up for this run


class FetchError(RuntimeError):
    """A request that still failed after its retries (network trouble, server errors, a corrupt response)."""


class ScheduleNeeded(RuntimeError):
    """The CDN refused and no saved copy of the schedule was found: the message says how to save one."""


def schedule_help(season: str) -> str:
    return (f"cdn.nba.com refused this machine, so the {season} schedule has to come from your browser: open "
            f"{SCHEDULE_URL} and save the page (Ctrl+S) as {SAVED_SCHEDULE} - into your Downloads folder or "
            f"{PATHS.live_raw(season)}, where live-update looks for it - then run it again (or pass --schedule <file>). "
            "Save it again whenever you want newer results and game statuses. If the browser shows 'Access Denied' too, "
            "the refusal is about your network (a VPN, the region, security software): try another network.")


def saved_schedule_file(season: str, given: str | Path | None = None) -> Path | None:
    """The browser-saved schedule to use: ``given``, else the newest of the live folder's copy and the Downloads
    folder's ``scheduleLeagueV2*.json``."""
    if given:
        path = Path(given).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"--schedule {given}: no such file")
        return path
    found = [PATHS.live_raw(season) / SAVED_SCHEDULE]
    downloads = Path.home() / "Downloads"
    if downloads.is_dir():
        found += sorted(downloads.glob("scheduleLeagueV2*.json"))
    found = [f for f in found if f.is_file()]
    return max(found, key=lambda f: f.stat().st_mtime) if found else None


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
    body, _ = _get(url, {"Accept": "application/json"}, timeout, retries, decode=lambda b: json.loads(b.decode("utf-8")))
    return body


def _get(url: str, headers: dict | None = None, timeout: float = 60.0, retries: int = 3, decode=None):
    """GET ``url`` with the project's User-Agent -> (body, response headers); the body is None for 304 Not Modified.
    ``decode`` turns the (decompressed) bytes into the result, inside the retry loop. The access policy of
    ``_get_json`` applies."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip",
                                                       **(headers or {})})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
                if str(r.headers.get("Content-Encoding") or "").lower() == "gzip":
                    body = gzip.decompress(body)
                info = dict(r.headers.items()) if hasattr(r.headers, "items") else dict(r.headers)
            return (decode(body) if decode else body), info
        except urllib.error.HTTPError as e:
            if e.code in DENIED:
                raise AccessDenied(f"{url}: HTTP {e.code} (access refused; not retried)") from e
            if e.code == 304:
                return None, {}
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
               log: Callable[[str], None] = print, source: str = "auto", schedule_file: str | Path | None = None) -> dict:
    """Refresh the schedule, then download every finished game's play-by-play that is missing, incomplete, or
    recent enough (``refetch_hours`` after tip-off) to still receive stat corrections. ``source``: 'cdn' (cdn.nba.com
    only), 'mirror' (a saved schedule and the open mirror), 'auto' (the CDN, and the mirror if it refuses)."""
    if source not in ("auto", "cdn", "mirror"):
        raise ValueError(f"source must be auto, cdn or mirror, not {source!r}")
    base = PATHS.live_raw(season)
    base.mkdir(parents=True, exist_ok=True)
    manifest_path = base / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    now = dt.datetime.now(dt.timezone.utc)
    stamp = now.isoformat(timespec="seconds")

    doc, refused = None, None
    if source in ("auto", "cdn"):
        log(f"[live] schedule <- {SCHEDULE_URL}")
        try:
            doc = _get_json(SCHEDULE_URL)
            sched_src = {"url": SCHEDULE_URL, "fetched_at": stamp, "license": LIC_LIVE}
        except AccessDenied as e:
            if source == "cdn":
                raise
            refused = str(e)
            log(f"[live] {e}\n[live] cdn.nba.com refused this machine: the schedule comes from a copy saved in a "
                f"browser and the finished games from the open mirror ({MIRROR_REPO}); requests are never retried "
                "under another identity")
    if doc is None:
        f = saved_schedule_file(season, schedule_file)
        if f is None:
            raise ScheduleNeeded(schedule_help(season))
        try:
            doc = json.loads(f.read_text(encoding="utf-8-sig"))
        except (ValueError, UnicodeDecodeError) as e:
            raise ScheduleNeeded(f"{f} is not the schedule JSON ({e}); {schedule_help(season)}") from None
        saved_t = dt.datetime.fromtimestamp(f.stat().st_mtime, dt.timezone.utc)
        saved = saved_t.isoformat(timespec="seconds")
        sched_src = {"file": str(f), "saved_at": saved, "license": LIC_LIVE}
        log(f"[live] schedule <- {f} (saved {saved[:16].replace('T', ' ')} UTC)")
        if now - saved_t > dt.timedelta(hours=36):
            log(f"[live] note: the saved schedule is {(now - saved_t).days} days old - its game statuses and the "
                "projection window follow it: save it again from your browser for today's slate")
    try:
        sched = read_schedule(doc, season, Season(season).game_prefix)
    except ValueError as e:
        raise ScheduleNeeded(f"the schedule is not in the expected format ({e}); {schedule_help(season)}") from None
    if sched.empty:
        raise RuntimeError(f"the schedule has no {season} regular-season games "
                           "(has the season changed? set QUANTLAB_LIVE_SEASON)")
    manifest["schedule"] = {**sched_src, "sha256": write_json_gz(doc, schedule_path(season)),
                            "games": int(len(sched)),
                            "status_counts": {k: int(v) for k, v in sched["status"].value_counts().items()}}
    if refused is not None or source == "mirror":
        try:
            out = fetch_mirror(season, manifest, log)
        finally:
            _write_manifest(manifest, manifest.get("failed", {}), manifest_path)
        return {"season": season, "scheduled_games": int(len(sched)), "source": "mirror", "refused": refused,
                "schedule": sched_src, **out}

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
            except AccessDenied as e:
                if source != "auto":
                    raise
                refused = str(e)                       # the CDN stopped answering: the mirror for the rest
                log(f"[live] {e}\n[live] the remaining finished games come from the open mirror ({MIRROR_REPO})")
                break
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
    if refused is not None:
        try:
            out = fetch_mirror(season, manifest, log)
        finally:
            _write_manifest(manifest, manifest.get("failed", {}), manifest_path)
        return {"season": season, "scheduled_games": int(len(sched)), "source": "cdn+mirror", "refused": refused,
                "schedule": sched_src, "cdn_downloaded": fetched, **out}
    summary = {"season": season, "scheduled_games": int(len(sched)), "final": int(len(final)), "downloaded": fetched,
               "cached": cached, "failed": failed, "stopped": stopped, "source": "cdn", "schedule": sched_src}
    log(f"[live] {season}: {len(final)} of {len(sched)} games final; {fetched} downloaded, {cached} cached, "
        f"{len(failed)} not available yet")
    return summary


def mirror_key(season: str) -> str:
    return f"cdnnba_{Season(season).start_year}"


def mirror_path(season: str) -> Path:
    return PATHS.live_raw(season) / "mirror" / f"{mirror_key(season)}.tar.xz"


def fetch_mirror(season: str, manifest: dict, log: Callable[[str], None] = print) -> dict:
    """The season's finished games from the open mirror: download ``cdnnba_<year>.tar.xz`` when it changed
    (conditional request), then write a per-game document for every complete game that is new or longer than the
    copy on disk (a game fetched from the CDN is kept). Returns what was done."""
    key = mirror_key(season)
    listing, _ = _get(MIRROR_LIST_URL, decode=lambda b: b.decode("utf-8"))
    urls = dict(line.strip().split("=", 1) for line in listing.splitlines() if "=" in line)
    m = manifest.setdefault("mirror", {})
    path = mirror_path(season)
    if key not in urls:
        log(f"[live] the mirror ({MIRROR_REPO}) has not published {season} yet - it usually adds a season in "
            "November and then updates it every few weeks; nothing new to build")
        return {"mirror": "not published yet", "downloaded": 0, "games_in_mirror": 0, "failed": {}}
    # the list gives github.com/<repo>/raw/<branch>/<file>; the raw host serves it directly (no redirect), with an ETag
    url = re.sub(r"^https://github\.com/([^/]+/[^/]+)/raw/", r"https://raw.githubusercontent.com/\1/", urls[key])
    headers = {"If-None-Match": m["etag"]} if m.get("etag") and path.exists() else {}
    log(f"[live] finished games <- {url}")
    body, info = _get(url, headers, timeout=300)
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    if body is None:
        log(f"[live] the mirror's {key} is unchanged since {m.get('fetched_at', '?')[:10]}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(body)
        os.replace(tmp, path)
        hdr = {k.lower(): v for k, v in info.items()}
        m.update({"url": url, "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body), "fetched_at": stamp,
                  "etag": hdr.get("etag"), "last_modified": hdr.get("last-modified"), "license": LIC_MIRROR})
    if not path.exists():
        return {"mirror": "unavailable", "downloaded": 0, "games_in_mirror": 0, "failed": {}}
    with tarfile.open(path, "r:xz") as t:
        member = next((n for n in t.getnames() if n.endswith(f"{key}.csv")), None)
        if member is None:
            raise FetchError(f"{path}: no {key}.csv inside (has the mirror's format changed?)")
        rows = pd.read_csv(io.BufferedReader(t.extractfile(member)), low_memory=False)
    prefix = Season(season).game_prefix
    gid_col = pd.to_numeric(rows["gameId"], errors="coerce").fillna(0).astype("int64")
    rows = rows[(gid_col // 100_000) == prefix]
    games = manifest.setdefault("games", {})
    failed = manifest.setdefault("failed", {})
    written, last = 0, None
    for gid, g in rows.groupby("gameId"):
        k = str(int(gid))
        have = games.get(k, {})
        if pbp_path(season, gid).exists() and have.get("complete") and (
                have.get("source", "cdn") == "cdn" or have.get("actions", 0) >= len(g)):
            continue
        doc = doc_from_mirror_rows(g)
        problems = check_pbp(doc, int(gid))
        if problems:
            failed[k] = "mirror: " + "; ".join(problems)
            continue
        games[k] = {"source": "mirror", "sha256": write_json_gz(doc, pbp_path(season, gid)), "fetched_at": stamp,
                    "actions": int(len(g)), "complete": True}
        failed.pop(k, None)
        written += 1
    n = int(rows["gameId"].nunique())
    log(f"[live] mirror: {n} finished {season} games, {written} new or longer")
    return {"mirror": m.get("url"), "games_in_mirror": n, "downloaded": written, "failed": dict(failed)}


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
