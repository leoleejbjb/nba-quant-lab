"""Normalize the raw liveData play-by-play into a tidy, chronologically ordered event table.

Ordering follows the convention validated against official box scores in the
FiveMan project: events are ordered by (period, game clock descending,
``orderNumber``). The clock decides chronology; the league's official order
breaks ties at the same clock (which matters e.g. for a substitution between
two free throws). ``actionNumber`` is creation order and is not used.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import OVERTIME_PERIOD_S, REGULATION_PERIOD_S, TEAM_ID_MIN

RAW_COLUMNS = [
    "gameId", "actionNumber", "orderNumber", "clock", "timeActual", "period",
    "actionType", "subType", "qualifiers", "descriptor", "personId", "teamId", "teamTricode",
    "scoreHome", "scoreAway", "shotResult", "isFieldGoal", "description", "playerNameI",
    "assistPersonId", "foulDrawnPersonId", "stealPersonId", "blockPersonId",
    "jumpBallWonPersonId", "jumpBallLostPersonId", "shotDistance", "x", "y", "area",
]
ID_COLUMNS = {"assistPersonId": "assist_id", "foulDrawnPersonId": "foul_drawn_id",
              "stealPersonId": "steal_id", "blockPersonId": "block_id",
              "jumpBallWonPersonId": "jb_won_id", "jumpBallLostPersonId": "jb_lost_id"}


def read_raw_pbp(archive: Path, member: str, game_prefix: int) -> pd.DataFrame:
    with tarfile.open(archive, mode="r:xz") as tar:
        fobj = tar.extractfile(member)
        if fobj is None:
            raise FileNotFoundError(f"{member} not found in {archive}")
        df = pd.read_csv(io.BufferedReader(fobj), low_memory=False,
                         usecols=lambda c: c in RAW_COLUMNS)
    for c in RAW_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    gid = pd.to_numeric(df["gameId"], errors="coerce").fillna(0).astype("int64")
    return df[(gid // 100_000) == game_prefix].reset_index(drop=True)


def actions_frame(doc: dict) -> pd.DataFrame:
    """One liveData play-by-play document (``cdn.nba.com/.../playbyplay_<id>.json``) as the mirror's CSV rows: the
    mirror is a flat copy of ``game.actions`` with the game id added and list fields joined with ', '."""
    game = doc["game"]
    df = pd.DataFrame(game.get("actions") or [])
    if "qualifiers" in df:
        df["qualifiers"] = df["qualifiers"].map(
            lambda q: ", ".join(str(x) for x in q) if isinstance(q, list) and q else (q if isinstance(q, str) and q
                                                                                    else pd.NA))
    df["gameId"] = int(game["gameId"])
    for c in RAW_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[RAW_COLUMNS]


def read_live_pbp(docs, game_prefix: int) -> pd.DataFrame:
    """Live play-by-play documents -> the same raw table ``read_raw_pbp`` returns for the pinned mirror."""
    frames = [actions_frame(d) for d in docs]
    if not frames:
        return pd.DataFrame(columns=RAW_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    gid = pd.to_numeric(df["gameId"], errors="coerce").fillna(0).astype("int64")
    return df[(gid // 100_000) == game_prefix].reset_index(drop=True)


SCHEDULE_STATUS = {1: "scheduled", 2: "live", 3: "final"}


def read_schedule(doc: dict, season: str, game_prefix: int) -> pd.DataFrame:
    """The league schedule (``cdn.nba.com/static/json/staticData/scheduleLeagueV2.json``) -> one row per
    regular-season game: date (US Eastern), tip-off time (UTC), teams, status (scheduled / live / final / postponed
    / tbd) and the score of finished games."""
    try:
        dates = doc["leagueSchedule"]["gameDates"]
    except (KeyError, TypeError):
        raise ValueError("unexpected schedule format: no leagueSchedule.gameDates (top-level keys: "
                         f"{sorted(doc) if isinstance(doc, dict) else type(doc).__name__})") from None
    rows = []
    for gd in dates:
        for g in gd.get("games") or []:
            gid = int(g["gameId"])
            if gid // 100_000 != game_prefix:
                continue                                   # preseason, play-in, playoffs, Cup final
            home, away = g.get("homeTeam") or {}, g.get("awayTeam") or {}
            code = int(g.get("gameStatus") or 0)
            text = str(g.get("gameStatusText") or "")
            status = SCHEDULE_STATUS.get(code, "unknown")
            if str(g.get("postponedStatus") or "").upper() == "P" or "PPD" in text.upper() or "POSTPONED" in text.upper():
                status = "postponed"
            hid, aid = int(home.get("teamId") or 0), int(away.get("teamId") or 0)
            if not hid or not aid:
                status = "tbd"                             # e.g. NBA Cup knockout placeholders
            rows.append({"game_id": gid, "season": season, "status": status, "status_text": text,
                         "date_est": g.get("gameDateEst"), "tip": g.get("gameDateTimeUTC") or g.get("gameDateUTC"),
                         "home_team_id": hid, "away_team_id": aid,
                         "home_team": home.get("teamTricode"), "away_team": away.get("teamTricode"),
                         "home_score": home.get("score"), "away_score": away.get("score")})
    cols = ["game_id", "season", "game_date", "tip_utc", "home_team_id", "away_team_id", "home_team", "away_team",
            "status", "status_text", "home_score", "away_score"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    df["tip_utc"] = pd.to_datetime(df["tip"], utc=True, errors="coerce", format="ISO8601")
    est = pd.to_datetime(df["date_est"], errors="coerce", format="ISO8601", utc=True).dt.tz_localize(None).dt.normalize()
    # the Eastern date; when it is missing, the same -6 h rule game_table applies to tip times
    df["game_date"] = est.fillna((df["tip_utc"] - pd.Timedelta(hours=6)).dt.tz_localize(None).dt.normalize())
    for c in ("home_score", "away_score"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df.loc[df["status"] != "final", ["home_score", "away_score"]] = np.nan
    return df[cols].drop_duplicates("game_id").sort_values(["game_date", "tip_utc", "game_id"]).reset_index(drop=True)


def parse_clock(clock: pd.Series) -> pd.Series:
    """'PT11M41.00S' -> 701.0 seconds remaining in the period."""
    parts = clock.astype("string").str.extract(r"^PT(\d+)M(\d+(?:\.\d+)?)S$")
    secs = parts[0].astype(float) * 60 + parts[1].astype(float)
    if secs.isna().any():
        raise ValueError(f"Unparseable clock values: {clock[secs.isna()].unique()[:5]}")
    return secs


def _int(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype("int64").to_numpy()


def _str(s: pd.Series) -> np.ndarray:
    return s.astype("string").fillna("").astype(str).to_numpy()


def normalize_events(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({
        "game_id": _int(raw["gameId"]),
        "order": _int(raw["orderNumber"]),
        "period": _int(raw["period"]),
        "clock_s": parse_clock(raw["clock"]).to_numpy(),
        "action_type": _str(raw["actionType"]),
        "sub_type": _str(raw["subType"]),
        "descriptor": _str(raw["descriptor"]),
        "qualifiers": _str(raw["qualifiers"]),
        "person_id": _int(raw["personId"]),
        "team_id": _int(raw["teamId"]),
        "team": _str(raw["teamTricode"]),
        "score_home": _int(raw["scoreHome"]),
        "score_away": _int(raw["scoreAway"]),
        "shot_result": _str(raw["shotResult"]),
        "description": _str(raw["description"]),
        "player_name": _str(raw["playerNameI"]),
        "time_actual": _str(raw["timeActual"]),
        "shot_distance": pd.to_numeric(raw["shotDistance"], errors="coerce").to_numpy(),
        "area": _str(raw["area"]),
    })
    for src, dst in ID_COLUMNS.items():
        df[dst] = _int(raw[src])
    df.loc[df["person_id"] >= TEAM_ID_MIN, "person_id"] = 0
    df = df.sort_values(["game_id", "period", "clock_s", "order"],
                        ascending=[True, True, False, True], kind="mergesort").reset_index(drop=True)
    p = df["period"].to_numpy()
    plen = np.where(p <= 4, REGULATION_PERIOD_S, OVERTIME_PERIOD_S)
    pstart = np.where(p <= 4, (p - 1) * REGULATION_PERIOD_S,
                      4 * REGULATION_PERIOD_S + (p - 5) * OVERTIME_PERIOD_S)
    df["clock_s"] = np.clip(df["clock_s"].to_numpy(), 0, plen)
    df["elapsed_s"] = pstart + (plen - df["clock_s"].to_numpy())
    made = df["shot_result"].to_numpy() == "Made"
    at = df["action_type"].to_numpy()
    df["points"] = np.select([made & (at == "2pt"), made & (at == "3pt"), made & (at == "freethrow")],
                             [2, 3, 1], default=0).astype(np.int64)
    return df


def game_table(events: pd.DataFrame) -> pd.DataFrame:
    """One row per game: home/away team ids (inferred from scoreboard increments), date, final score."""
    ev = events
    tid = ev["team_id"].to_numpy()
    pts = ev["points"].to_numpy()
    d_home = ev.groupby("game_id")["score_home"].diff().fillna(ev["score_home"]).to_numpy()
    d_away = ev.groupby("game_id")["score_away"].diff().fillna(ev["score_away"]).to_numpy()
    sc = (tid > 0) & (pts > 0) & ((d_home > 0) | (d_away > 0))
    votes = pd.DataFrame({"game_id": ev["game_id"].to_numpy()[sc], "team_id": tid[sc],
                          "home": (d_home[sc] > 0).astype(float)})
    v = votes.groupby(["game_id", "team_id"])["home"].mean().reset_index()
    v = v.sort_values(["game_id", "home"], ascending=[True, False])
    first = v.groupby("game_id").nth(0).set_index("game_id")
    second = v.groupby("game_id").nth(1).set_index("game_id")
    g = pd.DataFrame({"home_team_id": first["team_id"], "away_team_id": second["team_id"],
                      "home_vote": first["home"], "away_vote": second["home"]})
    g["home_away_clean"] = (g["home_vote"] == 1.0) & (g["away_vote"] == 0.0)
    tri = ev[ev["team_id"] > 0].groupby("team_id")["team"].agg(lambda s: s.mode().iat[0])
    g["home_team"] = g["home_team_id"].map(tri)
    g["away_team"] = g["away_team_id"].map(tri)
    last = ev.groupby("game_id").agg(final_home=("score_home", "max"), final_away=("score_away", "max"),
                                     n_periods=("period", "max"), first_time=("time_actual", "first"))
    g = g.join(last, how="right").reset_index().rename(columns={"index": "game_id"})
    # timeActual is UTC; NBA games tip 12:00-22:30 ET, i.e. 16:00-03:30 UTC. Shift by -6h so that
    # late-evening (Pacific) games keep their local calendar date.
    t = pd.to_datetime(g["first_time"], utc=True, errors="coerce", format="ISO8601")
    g["tip_utc"] = t
    g["game_date"] = (t - pd.Timedelta(hours=6)).dt.tz_localize(None).dt.normalize()
    return g.drop(columns=["first_time", "home_vote", "away_vote"]).sort_values("game_id").reset_index(drop=True)
