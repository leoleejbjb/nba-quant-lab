"""Walk-forward pregame feature store.

Every feature for a game on date ``d`` is computed from games played on
dates strictly before ``d`` (players and teams play at most once per day, so
"previous rows" are always earlier dates; league-level quantities are lagged
by one date).

Time decay uses an *effective time* axis: days within a season, with each
offseason compressed to ``OFFSEASON_GAP_DAYS`` - player ability does not decay
over the summer the way a calendar-time half-life would imply.

Decayed sums: ``S_k = sum_{j<k} x_j * 2^{-(t_k - t_j)/h}`` for several
half-lives ``h``, computed with a vectorized cumulative-sum identity.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import PATHS, TUNING_SEASON

OFFSEASON_GAP_DAYS = 30.0
RATE_HALFLIVES = (15, 40, 100, 250, 600)
MIN_HALFLIVES = (7, 20, 60)

COUNT_COLS = ["fga2", "fgm2", "fga3", "fgm3", "fta", "ftm", "trips2", "trips3", "and1", "oreb", "dreb",
              "ast", "stl", "blk", "tov", "tov_off_foul", "pf", "pf_offensive", "pf_shooting", "fgm2_ast",
              "fgm3_ast", "pts", "blocked", "pfd", "fta_tech"]
EXPO_COLS = ["on_poss", "op_poss", "on_fga", "on_fgm", "on_trips", "on_and1", "on_tov", "on_oreb_ch",
             "op_oreb_ch", "op_fga", "op_fga2", "on_pts", "op_pts", "min", "one"]


def time_axis(games: pd.DataFrame, upcoming: pd.DataFrame | None = None) -> pd.DataFrame:
    """Season and date of every built game plus the live season's upcoming games: the calendar ``effective_time``
    maps (so scheduled dates - even of a season with no finished game yet - get an effective time)."""
    parts = [games[["season", "game_date"]]]
    if upcoming is not None and len(upcoming):
        parts.append(upcoming[["season", "game_date"]])
    ax = pd.concat(parts, ignore_index=True)
    ax["game_date"] = pd.to_datetime(ax["game_date"])
    return ax


def effective_time(dates: pd.Series, games: pd.DataFrame) -> pd.Series:
    """Map calendar dates to the effective-time axis (days)."""
    seasons = games.groupby("season")["game_date"].agg(["min", "max"]).sort_index()
    offset, offsets = 0.0, {}
    for s, r in seasons.iterrows():
        offsets[s] = (r["min"], offset)
        offset += (r["max"] - r["min"]).days + OFFSEASON_GAP_DAYS
    d = pd.to_datetime(dates)
    out = np.full(len(d), np.nan)
    for s, (start, off) in offsets.items():
        end = seasons.loc[s, "max"]
        m = (d >= start - pd.Timedelta(days=60)) & (d <= end + pd.Timedelta(days=60))
        out[m.to_numpy()] = off + (d[m] - start).dt.days.to_numpy()
    return pd.Series(out, index=dates.index)


def decayed_sums(keys: np.ndarray, t: np.ndarray, X: np.ndarray, halflife: float) -> np.ndarray:
    """Pregame decayed sums per key (rows sorted by key, then time); the current row is excluded.

    Weights are taken relative to each key's first time and cumulated per key, so magnitudes from
    different keys never mix (no cancellation); the exclusive sum is a per-key shift, not a subtraction.
    """
    lam = np.log(2.0) / halflife
    first_t = pd.Series(t).groupby(keys).transform("first").to_numpy()
    rel = t - first_t
    W = X * np.exp(lam * rel)[:, None]
    C = pd.DataFrame(W).groupby(keys).cumsum()
    C_excl = C.groupby(keys).shift(1).fillna(0.0).to_numpy()
    return C_excl * np.exp(-lam * rel)[:, None]


def player_features(pg: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Pregame player features for every row: played player-games, zero-minute rows and pregame fringe candidates
    who did not appear (not-played rows get features but add nothing to anyone's history)."""
    df = pg.copy()
    df["one"] = 1.0
    df["t_eff"] = effective_time(df["game_date"], games).to_numpy()
    df = df.sort_values(["person_id", "t_eff", "game_id"]).reset_index(drop=True)
    played = df["played"].to_numpy()
    keys = df["person_id"].to_numpy()
    t = df["t_eff"].to_numpy()
    cols = COUNT_COLS + EXPO_COLS
    X = df[cols].to_numpy(dtype=np.float64) * played[:, None]   # DNP rows contribute nothing
    feats = {}
    for h in RATE_HALFLIVES:
        S = decayed_sums(keys, t, X, h)
        for j, c in enumerate(cols):
            feats[f"{c}_h{h}"] = S[:, j].astype(np.float32)
    # minutes features (played games only)
    Xm = np.column_stack([df["min"].to_numpy() * played, played.astype(float),
                          (df["starter"].to_numpy() & played).astype(float)])
    for h in MIN_HALFLIVES:
        S = decayed_sums(keys, t, Xm, h)
        feats[f"min_sum_m{h}"] = S[:, 0].astype(np.float32)
        feats[f"gp_m{h}"] = S[:, 1].astype(np.float32)
        feats[f"starts_m{h}"] = S[:, 2].astype(np.float32)
    F = pd.DataFrame(feats)
    out = pd.concat([df[["game_id", "person_id", "team_id", "opp_team_id", "season", "game_date", "t_eff",
                         "is_home", "starter", "played", "min"]], F], axis=1)
    # last played game (any team) and previous-game facts
    g = df.groupby("person_id", sort=False)
    prev_played_t = df["t_eff"].where(df["played"]).groupby(df["person_id"]).shift(1)
    prev_played_t = prev_played_t.groupby(df["person_id"]).ffill()
    out["days_since_played"] = (df["t_eff"] - prev_played_t).astype(np.float32)
    last_min = df["min"].where(df["played"]).groupby(df["person_id"]).shift(1).groupby(df["person_id"]).ffill()
    out["last_min"] = last_min.astype(np.float32)
    last_start = df["starter"].where(df["played"]).astype(float).groupby(df["person_id"]).shift(1)
    out["last_start"] = last_start.groupby(df["person_id"]).ffill().astype(np.float32)
    last_team = df["team_id"].where(df["played"]).groupby(df["person_id"]).shift(1).groupby(df["person_id"]).ffill()
    out["new_team"] = (last_team.notna() & (last_team != df["team_id"])).astype(np.int8)
    out["career_games"] = g["played"].cumsum().sub(df["played"].astype(int)).astype(np.int32)
    season_gp = df.groupby(["person_id", "season"])["played"].cumsum() - df["played"].astype(int)
    out["season_games"] = season_gp.astype(np.int32)
    s_min = (df["min"] * played).groupby([df["person_id"], df["season"]]).cumsum() - df["min"] * played
    out["season_min_avg"] = (s_min / season_gp.replace(0, np.nan)).astype(np.float32)
    # last-3/5/10 played-game minutes: the mean through each played game, carried to every later row (played or
    # not), so a not-played candidate row sees exactly what a played row on that date would
    pm = df.loc[df["played"], ["person_id", "min"]]
    for n in (3, 5, 10):
        r = pm.groupby("person_id")["min"].transform(lambda s: s.rolling(n, min_periods=1).mean())
        col = pd.Series(np.nan, index=df.index)
        col.loc[pm.index] = r
        prev = col.groupby(df["person_id"]).shift(1)
        out[f"min_l{n}"] = prev.groupby(df["person_id"]).ffill().astype(np.float32)
    return out


ROTATION_BASE_MIN = 8.0    # baseline minutes per team game that make a player "rotation"
ROTATION_EXP_MIN = 12.0    # ... or expected minutes when he plays


def team_context(pg: pd.DataFrame, feats: pd.DataFrame, active: dict | None = None,
                 use_starters: bool = True, scheduled: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Teammate-availability context per team-game, from information public before tip-off - except that, without
    ``active``, availability is the who-appeared stand-in for the inactive list (below).

    Baseline = the team's previous 10 games: a player's baseline minutes / plays used are his
    per-team-game averages over those games (0 when he did not play), if he has not appeared for
    another team since. Players are *rotation* players if their baseline minutes >= 8 or their
    expected minutes when playing >= 12, or (``use_starters``) they start today.

    Who is *available* is the injury report / inactive list. ``active`` maps (game_id, team_id) to the
    set of players known to be dressed (official box-score statuses: played or DNP-coach's decision);
    for team-games not in ``active`` the stand-in is who appeared, which also reveals the rotation
    players a coach chose not to use (game-dependent: see docs/REPORT.md). Fringe players enter every
    pregame quantity through their baseline minutes, never through today's participation:

    * ``avail_exp_min``  sum of today's available rotation players' expected minutes + fringe baseline minutes
    * ``vacated_min``    baseline minutes of unavailable *rotation* players (``vacated_used`` for plays used)
    * ``slack_min``      240 - (baseline minutes of available rotation players + fringe baseline minutes)

    ``scheduled`` maps the (game_id, team_id) of unplayed games to the players presumed available (the pregame
    availability rule in ``build_features``); such team-games get context rows but add nothing to any baseline.
    """
    scheduled = scheduled or {}
    exp_min = (feats["min_sum_m20"] / feats["gp_m20"].replace(0, np.nan)).fillna(feats["season_min_avg"])
    h = 40
    used = sum(feats[f"{c}_h{h}"] for c in ("fga2", "fga3", "trips2", "trips3", "tov"))
    plays = feats[f"on_fga_h{h}"] + feats[f"on_trips_h{h}"] + feats[f"on_tov_h{h}"]
    f = feats[["game_id", "person_id", "team_id", "game_date", "starter"]].assign(
        exp_min=exp_min.to_numpy(), exp_usg=((used + 0.2 * 20) / (plays + 20)).to_numpy())
    pl = pg[pg["played"]].copy()
    pl["used"] = pl["fga2"] + pl["fga3"] + pl["trips2"] + pl["trips3"] + pl["tov"]
    pl = pl.merge(f[["game_id", "person_id", "exp_min"]], on=["game_id", "person_id"], how="left")
    by_gt = {k: {p: (m, u, e, st) for p, m, u, e, st in zip(v["person_id"], v["min"], v["used"], v["exp_min"],
                                                             v["starter"])}
             for k, v in pl.groupby(["game_id", "team_id"])}
    own_exp = {(g, p): e for g, p, e in zip(f["game_id"], f["person_id"], f["exp_min"])}
    order = pl.drop_duplicates(["game_id", "team_id"])[["game_id", "team_id", "game_date"]]
    if scheduled:
        sk = pd.DataFrame(list(scheduled), columns=["game_id", "team_id"]).merge(
            f[["game_id", "team_id", "game_date"]].drop_duplicates(["game_id", "team_id"]), on=["game_id", "team_id"])
        order = pd.concat([order, sk], ignore_index=True)
    order = order.sort_values(["game_date", "game_id"])
    ctx_rows, player_rows = [], []
    recent: dict = {}
    last_team: dict = {}
    last_exp: dict = {}      # latest pregame expected minutes per player
    for date, day in order.groupby("game_date", sort=True):
        updates = []
        for gid, team in day[["game_id", "team_id"]].itertuples(index=False):
            hist = recent.get(team, [])
            ng = len(hist)
            base_min: dict = {}
            base_used: dict = {}
            for gm in hist:
                for p, (m, u) in gm.items():
                    if last_team.get(p) == team:
                        base_min[p] = base_min.get(p, 0.0) + m / ng
                        base_used[p] = base_used.get(p, 0.0) + u / ng
            sched = (gid, team) in scheduled
            now = by_gt.get((gid, team), {})
            act = active.get((gid, team)) if active is not None else None
            if sched:
                avail = sorted(scheduled[(gid, team)])
            else:
                avail = list(now) if act is None else sorted(set(now) | set(act))

            def expected(p):
                e = own_exp.get((gid, p), np.nan) if sched else (now[p][2] if p in now else np.nan)
                return e if e == e else last_exp.get(p, np.nan)

            def is_rot(p):
                e = expected(p)
                st = use_starters and p in now and bool(now[p][3])
                return st or base_min.get(p, 0.0) >= ROTATION_BASE_MIN or (e == e and e >= ROTATION_EXP_MIN)

            avail_set = set(avail)
            rot_now = [p for p in avail if is_rot(p)]
            fringe_base = [p for p in base_min if not is_rot(p)]
            absent_rot = [p for p in base_min if p not in avail_set and is_rot(p)]
            vac_min = sum(base_min[p] for p in absent_rot)
            vac_used = sum(base_used[p] for p in absent_rot)
            team_used = sum(base_used.values())
            fringe_min = sum(base_min[p] for p in fringe_base)
            avail_exp = sum((expected(p) if expected(p) == expected(p) else 12.0) for p in rot_now) + fringe_min
            avail_base = sum(base_min.get(p, 0.0) for p in rot_now) + fringe_min
            ctx_rows.append((gid, team, ng, len(base_min), len(absent_rot),
                             sum(1 for p in absent_rot if base_min[p] >= 15), vac_min, vac_used, team_used,
                             avail_base, (240.0 - avail_base) if ng else np.nan, avail_exp, len(rot_now), fringe_min,
                             "-".join(str(p) for p in sorted(absent_rot, key=lambda p: -base_min[p]))))
            for p in avail + [q for q in base_min if q not in avail_set]:   # + baseline players not available
                player_rows.append((gid, team, p, base_min.get(p, 0.0) if ng else np.nan,
                                    base_used.get(p, 0.0) if ng else np.nan, is_rot(p), p in now))
            if now:                      # unplayed games add nothing to the baselines
                updates.append((team, {p: (v[0], v[1]) for p, v in now.items()}, now))
        for team, mins, now in updates:      # same-day games never see each other
            recent[team] = (recent.get(team, []) + [mins])[-10:]
            for p, v in now.items():
                last_team[p] = team
                if v[2] == v[2]:
                    last_exp[p] = v[2]
    ctx = pd.DataFrame(ctx_rows, columns=["game_id", "team_id", "base_games", "n_baseline", "n_absent",
                                          "n_absent_15", "vacated_min", "vacated_used", "team_used_base",
                                          "avail_base_min", "slack_min", "avail_exp_min", "n_rotation_playing",
                                          "fringe_base_min", "absent"])
    ctx["vacated_used_share"] = ctx["vacated_used"] / ctx["team_used_base"].replace(0, np.nan)
    pctx = pd.DataFrame(player_rows, columns=["game_id", "team_id", "person_id", "base_min", "base_used", "rotation",
                                              "appeared"])
    if active is None:     # keyed by team too: a just-traded player can sit in two teams' lists for one game
        return ctx, f.merge(pctx.drop(columns=["appeared"]), on=["game_id", "team_id", "person_id"], how="left")
    return ctx, pctx


FRINGE_WINDOW_DAYS = 25
FRINGE_P_FLOOR, FRINGE_P_CAP = 0.05, 0.9


def fringe_pool_table(pg: pd.DataFrame, window_days: int = FRINGE_WINDOW_DAYS,
                      team_games: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pregame fringe candidates for every team-game - one rule for everyone, today's game plays no part:
    players who appeared for the team in the previous ``window_days`` days and whose last appearance before
    the game was for this team. ``share`` = their appearances / the team's games in the window (clipped to
    [0.05, 0.9]); ``cond_window`` = their minutes per appearance in the window. ``team_games`` (game_id,
    team_id, game_date) lists the team-games to build pools for (default: every team-game in ``pg``; add the
    live season's unplayed games to give them pools)."""
    pl = pg[pg["played"]][["game_id", "team_id", "person_id", "game_date", "min"]].sort_values("game_date")
    if team_games is None:
        team_games = pg[["game_id", "team_id", "game_date"]].drop_duplicates(["game_id", "team_id"])
    last = {p: (d["game_date"].to_numpy(), d["team_id"].to_numpy()) for p, d in pl.groupby("person_id")}
    # a player who plays for another team on the game date (just traded) is not a candidate of his old team
    elsewhere = set(zip(pg["game_date"].to_numpy(), pg["person_id"].to_numpy(), pg["team_id"].to_numpy()))
    on_date = {}
    for d_, p_, t_ in elsewhere:
        on_date.setdefault((d_, p_), set()).add(t_)
    win = np.timedelta64(window_days, "D")
    rows = []
    for t, games in team_games.groupby("team_id"):
        games = games.sort_values("game_date")
        app = pl[pl["team_id"] == t]
        a_dates = app["game_date"].to_numpy()
        a_pid, a_min, a_gid = app["person_id"].to_numpy(), app["min"].to_numpy(), app["game_id"].to_numpy()
        g_dates = games["game_date"].to_numpy()
        for gid, date in zip(games["game_id"].to_numpy(), g_dates):
            lo_a, hi_a = np.searchsorted(a_dates, date - win, "left"), np.searchsorted(a_dates, date, "left")
            if hi_a <= lo_a:
                continue
            n_team = len(np.unique(a_gid[lo_a:hi_a]))
            pids, inv = np.unique(a_pid[lo_a:hi_a], return_inverse=True)
            cnt = np.bincount(inv)
            mins = np.bincount(inv, weights=a_min[lo_a:hi_a])
            for p, c, m in zip(pids, cnt, mins):
                d, tt = last[p]
                k = int(np.searchsorted(d, date, "left")) - 1
                if k >= 0 and tt[k] != t:
                    continue                      # appeared for another team since (traded / waived)
                if on_date.get((date, p), {t}) - {t}:
                    continue                      # on another team's sheet that day
                rows.append((gid, t, int(p), float(np.clip(c / max(n_team, 1), FRINGE_P_FLOOR, FRINGE_P_CAP)),
                             float(m / c), int(c), n_team))
    return pd.DataFrame(rows, columns=["game_id", "team_id", "person_id", "share", "cond_window", "appearances",
                                       "team_games"])


def candidate_rows(pg: pd.DataFrame, pool: pd.DataFrame, ctx: pd.DataFrame | None = None) -> pd.DataFrame:
    """Zero-stat, not-played rows for pool candidates who have no player-game row in that game, so the feature
    store, the minutes model and the rate model produce pregame numbers for every candidate. ``ctx`` (game_id,
    team_id, opp_team_id, season, game_date, is_home) defaults to the team-games in ``pg``."""
    have = pd.MultiIndex.from_frame(pg[["game_id", "person_id"]])
    c = pool[~pd.MultiIndex.from_frame(pool[["game_id", "person_id"]]).isin(have)][["game_id", "team_id", "person_id"]]
    if ctx is None:
        ctx = pg[["game_id", "team_id", "opp_team_id", "season", "game_date", "is_home"]].drop_duplicates(
            ["game_id", "team_id"])
    c = c.merge(ctx, on=["game_id", "team_id"], how="inner")
    num = [col for col in pg.columns if col not in c.columns and pd.api.types.is_numeric_dtype(pg[col])
           and not pd.api.types.is_bool_dtype(pg[col])]
    for col in num:
        c[col] = 0
    c["played"] = False
    c["starter"] = False
    return c


# context columns that depend on the rotation rule / the availability information set
CTX_VARIANT_COLS = ["n_absent", "n_absent_15", "vacated_min", "vacated_used", "avail_base_min", "slack_min",
                    "avail_exp_min", "n_rotation_playing", "fringe_base_min", "vacated_used_share"]


def official_active(av: pd.DataFrame) -> dict:
    """(game_id, team_id) -> players dressed (played or DNP-coach's decision), official statuses only."""
    a = av[(av.get("source", "official") == "official") & av["status"].isin(["played", "dnp_cd"])]
    return {k: set(v) for k, v in a.groupby(["game_id", "team_id"])["person_id"]}


def missed_streaks(pg: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    """For each row (team_id, season, person_id, game_date): the team's games (same season, before the date) since
    the player last appeared for it - 0 if he played the team's previous game; the team's games so far if he has
    not appeared for it this season."""
    app = pg[pg["played"]][["team_id", "season", "person_id", "game_date", "game_id"]]
    tgs = app.drop_duplicates(["game_id", "team_id"])
    team_dates = {k: np.sort(v["game_date"].to_numpy()) for k, v in tgs.groupby(["team_id", "season"])}
    pdates = {k: np.sort(v["game_date"].to_numpy()) for k, v in app.groupby(["team_id", "season", "person_id"])}
    out = np.zeros(len(rows), dtype=np.int32)
    for i, (t, s, p, d) in enumerate(zip(rows["team_id"].to_numpy(), rows["season"].to_numpy(),
                                         rows["person_id"].to_numpy(), rows["game_date"].to_numpy())):
        dates = team_dates.get((t, s))
        if dates is None:
            continue
        n_before = int(np.searchsorted(dates, d, "left"))
        mine = pdates.get((t, s, p))
        j = int(np.searchsorted(mine, d, "left")) - 1 if mine is not None else -1
        out[i] = n_before if j < 0 else n_before - int(np.searchsorted(dates, mine[j], "left")) - 1
    return out


STREAK_BUCKETS = (("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), ("5-9", 5, 9), ("10+", 10, 10 ** 6))
PRESUMED_AVAILABLE_STREAK = 2      # unplayed games: presumed available if he appeared in one of the team's last 3


def streak_bucket(streak) -> np.ndarray:
    s = np.asarray(streak)
    out = np.full(s.shape, "10+", dtype=object)
    for name, lo, hi in STREAK_BUCKETS:
        out[(s >= lo) & (s <= hi)] = name
    return out


def availability_priors(feats: pd.DataFrame, pool: pd.DataFrame, pg: pd.DataFrame,
                        season: str = TUNING_SEASON) -> dict:
    """P(appears) of a rotation player (lineup-blind rule) in his team's pregame pool, by team games missed in a
    row - the availability prior for unplayed games, which have no injury report. Estimated on ``season``."""
    f = feats[(feats["season"] == season) & feats["rotation_pre"].eq(True)]
    f = f.merge(pool[["game_id", "team_id", "person_id"]], on=["game_id", "team_id", "person_id"])
    b = streak_bucket(missed_streaks(pg, f))
    out = {"season": season, "rule": "rotation players (lineup-blind rule) in the team's 25-day pool",
           "p_appear": {}, "n": {}}
    for name, _, _ in STREAK_BUCKETS:
        m = b == name
        out["p_appear"][name] = round(float(f.loc[m, "played"].mean()), 4) if m.any() else 0.0
        out["n"][name] = int(m.sum())
    return out


def build_features(log=print) -> dict:
    from ..data.database import upcoming_team_games
    d = PATHS.processed("all")
    pg = pd.read_parquet(d / "player_games.parquet")
    games = pd.read_parquet(d / "games.parquet")
    tg = pd.read_parquet(d / "team_games.parquet")
    # the live season's unplayed games (within the projection horizon) get pregame rows too
    up = upcoming_team_games()
    axis = time_axis(games, up)
    team_games = pg[["game_id", "team_id", "game_date"]].drop_duplicates(["game_id", "team_id"])
    ctx_cols = ["game_id", "team_id", "opp_team_id", "season", "game_date", "is_home"]
    ctx_all = pg[ctx_cols].drop_duplicates(["game_id", "team_id"])
    if len(up):
        team_games = pd.concat([team_games, up[["game_id", "team_id", "game_date"]]], ignore_index=True)
        ctx_all = pd.concat([ctx_all, up[ctx_cols]], ignore_index=True)
    # pregame fringe candidates who did not appear get a (not-played) row, so every candidate is projected
    # the same way whether or not he appears; for unplayed games every pool player is such a candidate
    pool = fringe_pool_table(pg, team_games=team_games)
    cand = candidate_rows(pg, pool, ctx=ctx_all)
    cand["scheduled"] = cand["game_id"].isin(set(up["game_id"]))
    pg = pd.concat([pg.assign(candidate=False, scheduled=False), cand.assign(candidate=True)], ignore_index=True)
    # unplayed games: presumed available = appeared in one of the team's last three games
    sched_rows = pg[pg["scheduled"]]
    streak = missed_streaks(pg, sched_rows) if len(sched_rows) else np.zeros(0, dtype=np.int32)
    scheduled = {}
    for (gid, tid, pid), st in zip(sched_rows[["game_id", "team_id", "person_id"]].itertuples(index=False, name=None),
                                   streak):
        if st <= PRESUMED_AVAILABLE_STREAK:
            scheduled.setdefault((gid, tid), set()).add(pid)
    for key in set(zip(up["game_id"], up["team_id"])) - set(scheduled):
        scheduled[key] = set()            # no recent players at all: context rows still exist
    feats = player_features(pg, axis)
    feats = feats.merge(pg[["game_id", "team_id", "person_id", "candidate", "scheduled"]],
                        on=["game_id", "team_id", "person_id"], how="left")
    if len(sched_rows):
        ms = sched_rows[["game_id", "person_id"]].assign(missed_streak=streak)
        feats = feats.merge(ms, on=["game_id", "person_id"], how="left")
    else:
        feats["missed_streak"] = np.nan
    ctx, f = team_context(pg, feats, scheduled=scheduled)
    feats = feats.merge(f[["game_id", "team_id", "person_id", "exp_min", "exp_usg", "base_min", "base_used",
                           "rotation"]], on=["game_id", "team_id", "person_id"], how="left")
    feats = feats.merge(ctx.drop(columns=["absent"]), on=["game_id", "team_id"], how="left")
    # lineup-blind variant (the "no lineup information" minutes model): rotation ignores today's starters
    ctx_pre, f_pre = team_context(pg, feats, use_starters=False, scheduled=scheduled)
    feats = feats.merge(f_pre[["game_id", "team_id", "person_id", "rotation"]].rename(
        columns={"rotation": "rotation_pre"}), on=["game_id", "team_id", "person_id"], how="left")
    feats = feats.merge(ctx_pre[["game_id", "team_id"] + CTX_VARIANT_COLS].rename(
        columns={c: f"{c}_pre" for c in CTX_VARIANT_COLS}), on=["game_id", "team_id"], how="left")
    # candidates outside the 10-game baseline: no baseline minutes; rotation by expected minutes only
    nb = feats["candidate"] & feats["base_min"].isna()
    feats.loc[nb, "base_min"] = 0.0
    for col in ("rotation", "rotation_pre"):
        feats[col] = feats[col].where(~nb, feats["exp_min"].fillna(0) >= ROTATION_EXP_MIN)
    assert not feats.duplicated(["game_id", "person_id"]).any(), "one feature row per player-game"
    PATHS.ensure(PATHS.features)
    feats.to_parquet(PATHS.features / "player_asof.parquet", index=False)
    ctx.to_parquet(PATHS.features / "team_context.parquet", index=False)
    pool.to_parquet(PATHS.features / "fringe_pool.parquet", index=False)
    pri = availability_priors(feats, pool, pg)
    (PATHS.features / "availability_priors.json").write_text(json.dumps(pri, indent=1), encoding="utf-8")
    # exact information set where the official inactive list exists (2020-21 .. 2023-24): dressed players
    # (played or DNP-coach's decision) are available; used to measure the appearance proxy's leak
    av = pd.read_parquet(d / "availability.parquet")
    act = official_active(av)
    if act:
        ctx_x, pctx_x = team_context(pg, feats, active=act)
        keys = pd.DataFrame(list(act.keys()), columns=["game_id", "team_id"])
        ctx_x = ctx_x.merge(keys, on=["game_id", "team_id"])
        pctx_x = pctx_x[pctx_x["game_id"].isin(keys["game_id"])]
        ctx_x.to_parquet(PATHS.features / "team_context_exact.parquet", index=False)
        pctx_x.to_parquet(PATHS.features / "player_context_exact.parquet", index=False)
    log(f"[features] {len(feats):,} rows ({int(feats['candidate'].sum()):,} pregame fringe candidates who did not "
        f"appear; {int(feats['scheduled'].sum()):,} rows of {up['game_id'].nunique()} unplayed games), "
        f"{feats.shape[1]} columns; availability priors {pri['p_appear']}")
    return {"rows": int(len(feats)), "cols": int(feats.shape[1])}
