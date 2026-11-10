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

import numpy as np
import pandas as pd

from ..config import PATHS

OFFSEASON_GAP_DAYS = 30.0
RATE_HALFLIVES = (15, 40, 100, 250)
MIN_HALFLIVES = (7, 20, 60)

COUNT_COLS = ["fga2", "fgm2", "fga3", "fgm3", "fta", "ftm", "trips2", "trips3", "and1", "oreb", "dreb",
              "ast", "stl", "blk", "tov", "tov_off_foul", "pf", "pf_offensive", "pf_shooting", "fgm2_ast",
              "fgm3_ast", "pts", "blocked", "pfd", "fta_tech"]
EXPO_COLS = ["on_poss", "op_poss", "on_fga", "on_fgm", "on_trips", "on_and1", "on_tov", "on_oreb_ch",
             "op_oreb_ch", "op_fga", "op_fga2", "on_pts", "op_pts", "min", "one"]


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
    """Pregame player features for every played player-game (plus DNP rows, which get features but no history)."""
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
    # last-10 and last-5 played-game minutes (rolling over played games only)
    pm = df.loc[df["played"], ["person_id", "min"]]
    for n in (3, 5, 10):
        r = pm.groupby("person_id")["min"].transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
        col = pd.Series(np.nan, index=df.index)
        col.loc[pm.index] = r
        out[f"min_l{n}"] = col.groupby(df["person_id"]).ffill().astype(np.float32)
    return out


def team_context(pg: pd.DataFrame, feats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Teammate-availability context per player-game (oracle participation = who actually played).

    For each team-game, the baseline is the team's previous 10 games: a player's *baseline minutes*
    is his minutes per team game over those games (0 when he did not play), counted only if he has
    not appeared for another team since. ``vacated_min`` / ``vacated_used`` sum the baseline minutes /
    plays used (FGA + FT trips + TOV) of baseline players who are not playing today; ``slack_min`` =
    240 - baseline minutes of today's participants (minutes that must be redistributed).
    """
    exp_min = (feats["min_sum_m20"] / feats["gp_m20"].replace(0, np.nan)).fillna(feats["season_min_avg"])
    h = 40
    used = sum(feats[f"{c}_h{h}"] for c in ("fga2", "fga3", "trips2", "trips3", "tov"))
    plays = feats[f"on_fga_h{h}"] + feats[f"on_trips_h{h}"] + feats[f"on_tov_h{h}"]
    f = feats[["game_id", "person_id", "team_id", "game_date"]].assign(
        exp_min=exp_min.fillna(0).to_numpy(), exp_usg=((used + 0.2 * 20) / (plays + 20)).to_numpy())
    pl = pg[pg["played"]].copy()
    pl["used"] = pl["fga2"] + pl["fga3"] + pl["trips2"] + pl["trips3"] + pl["tov"]
    by_gt = {k: dict(zip(v["person_id"], zip(v["min"], v["used"])))
             for k, v in pl.groupby(["game_id", "team_id"])}
    order = pl.drop_duplicates(["game_id", "team_id"]).sort_values(["game_date", "game_id"])
    ctx_rows, player_rows = [], []
    recent: dict = {}
    last_team: dict = {}
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
            now = by_gt[(gid, team)]
            absent = [p for p in base_min if p not in now]
            vac_min = sum(base_min[p] for p in absent)
            vac_used = sum(base_used[p] for p in absent)
            team_used = sum(base_used.values())
            avail = sum(base_min.get(p, 0.0) for p in now)
            sig_absent = sum(1 for p in absent if base_min[p] >= 15)
            ctx_rows.append((gid, team, ng, len(base_min), len(absent), sig_absent, vac_min, vac_used,
                             team_used, avail, 240.0 - avail if ng else np.nan,
                             "-".join(str(p) for p in sorted(absent, key=lambda p: -base_min[p]))))
            for p in now:
                player_rows.append((gid, p, base_min.get(p, 0.0) if ng else np.nan,
                                    base_used.get(p, 0.0) if ng else np.nan))
            updates.append((team, now))
        for team, now in updates:            # same-day games never see each other
            recent[team] = (recent.get(team, []) + [now])[-10:]
            for p in now:
                last_team[p] = team
    ctx = pd.DataFrame(ctx_rows, columns=["game_id", "team_id", "base_games", "n_baseline", "n_absent",
                                          "n_absent_15", "vacated_min", "vacated_used", "team_used_base",
                                          "avail_base_min", "slack_min", "absent"])
    ctx["vacated_used_share"] = ctx["vacated_used"] / ctx["team_used_base"].replace(0, np.nan)
    pctx = pd.DataFrame(player_rows, columns=["game_id", "person_id", "base_min", "base_used"])
    return ctx, f.merge(pctx, on=["game_id", "person_id"], how="left")


def build_features(log=print) -> dict:
    d = PATHS.processed("all")
    pg = pd.read_parquet(d / "player_games.parquet")
    games = pd.read_parquet(d / "games.parquet")
    tg = pd.read_parquet(d / "team_games.parquet")
    feats = player_features(pg, games)
    ctx, f = team_context(pg, feats)
    feats = feats.merge(f[["game_id", "person_id", "exp_min", "exp_usg", "base_min", "base_used"]],
                        on=["game_id", "person_id"], how="left")
    feats = feats.merge(ctx.drop(columns=["absent"]), on=["game_id", "team_id"], how="left")
    PATHS.ensure(PATHS.features)
    feats.to_parquet(PATHS.features / "player_asof.parquet", index=False)
    ctx.to_parquet(PATHS.features / "team_context.parquet", index=False)
    log(f"[features] {len(feats):,} player-game rows, {feats.shape[1]} columns")
    return {"rows": int(len(feats)), "cols": int(feats.shape[1])}
