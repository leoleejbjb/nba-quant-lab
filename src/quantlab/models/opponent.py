"""Team defensive / offensive style factors (walk-forward, decayed, shrunk to 1.0).

For defense D and stat s the factor is the rate D allows relative to the league::

    f_D(s) = [(A_h + k * L) / (E_h + k)] / L

with ``A_h``/``E_h`` the decayed sums (half-life ``h``) of what D's opponents
produced against it (numerator / exposure) before the game and ``L`` the
decayed league rate. Offense factors are the same construction on a team's
own production. ``k`` and ``h`` are tuned on 2023-24 by the predictive
log-likelihood of the opponent's counts in the next game.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import PATHS
from ..features.asof import decayed_sums, effective_time
from .rates import loglik


@dataclass(frozen=True)
class TeamRate:
    name: str
    num: tuple
    den: tuple
    kind: str


DEF_RATES = [
    TeamRate("tov_forced", ("tov",), ("fga", "trips", "tov"), "binom"),
    TeamRate("fga3_share_allowed", ("fga3",), ("fga",), "binom"),
    TeamRate("fg2_allowed", ("fgm2",), ("fga2",), "binom"),
    TeamRate("fg3_allowed", ("fgm3",), ("fga3",), "binom"),
    TeamRate("trips_allowed", ("trips",), ("fga", "trips", "tov"), "binom"),
    TeamRate("oreb_allowed", ("oreb",), ("oreb_ch",), "binom"),
    TeamRate("ast_allowed", ("ast",), ("fgm",), "binom"),
    TeamRate("stl_allowed", ("stl",), ("poss",), "pois"),     # opponent steals vs this offense (offense style)
    TeamRate("blk_allowed", ("blk",), ("fga",), "pois"),
]
HALFLIVES = (20, 50, 120, 300)
KS = (25, 50, 100, 200, 400, 800, 1600, 3200)


def _sum(df, cols, prefix=""):
    return sum(df[prefix + c].to_numpy(float) for c in cols)


def decayed_league_rate(dates: pd.Series, num: np.ndarray, den: np.ndarray, games: pd.DataFrame,
                        halflife: float = 60.0) -> pd.Series:
    """League rate as of each date from strictly earlier dates, decayed on effective time."""
    daily = pd.DataFrame({"d": dates.to_numpy(), "x": num, "e": den}).groupby("d")[["x", "e"]].sum().sort_index()
    te = effective_time(pd.Series(daily.index, index=daily.index), games).to_numpy()
    lam = np.log(2) / halflife
    sx = se = 0.0
    out = np.full(len(daily), np.nan)
    prev = te[0]
    for i, (x, e) in enumerate(daily.to_numpy()):
        dec = np.exp(-lam * (te[i] - prev))
        sx, se = sx * dec, se * dec
        out[i] = sx / se if se > 0 else np.nan
        sx, se, prev = sx + x, se + e, te[i]
    return pd.Series(out, index=daily.index).bfill()


def team_factor_frame(tg: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    t = tg.sort_values(["team_id", "game_date", "game_id"]).reset_index(drop=True).copy()
    t["t_eff"] = effective_time(t["game_date"], games).to_numpy()
    return t


def compute_factors(t: pd.DataFrame, games: pd.DataFrame, params: dict | None = None, season: str = "2023-24",
                    log=print):
    """Return (frame with def_<rate> / off_<rate> factors per team-game, tuned params)."""
    keys, tt = t["team_id"].to_numpy(), t["t_eff"].to_numpy()
    out = t[["game_id", "team_id", "opp_team_id", "season", "game_date"]].copy()
    tuned = {} if params is None else dict(params)
    # league rate by date (decayed, earlier dates only)
    for side, src_prefix in (("def", "opp_"), ("off", "")):
        for r in DEF_RATES:
            # defense: what opponents did against this team; offense: what this team did
            num = _sum(t, r.num, src_prefix)
            den = _sum(t, r.den, src_prefix)
            L = t["game_date"].map(decayed_league_rate(t["game_date"], num, den, games)).to_numpy()
            key = f"{side}_{r.name}"
            # target for tuning: opponent's (def) / own (off) production in *this* game
            m = (t["season"] == season).to_numpy()
            if key not in tuned:
                best = None
                for h in HALFLIVES:
                    S = decayed_sums(keys, tt, np.column_stack([num, den]), h)
                    for k in KS:
                        rate = (S[:, 0] + k * L) / (S[:, 1] + k)
                        ll = loglik(r.kind, rate[m], num[m], den[m]).sum()
                        if best is None or ll > best[0]:
                            best = (ll, h, k)
                ll0 = loglik(r.kind, L[m], num[m], den[m]).sum()
                tuned[key] = {"halflife": best[1], "k": best[2], "gain_per_game": float((best[0] - ll0) / m.sum())}
                log(f"[opp] {key:24s} h={best[1]:4d} k={best[2]:5d} gain {tuned[key]['gain_per_game']:+.4f}/game")
            h, k = tuned[key]["halflife"], tuned[key]["k"]
            S = decayed_sums(keys, tt, np.column_stack([num, den]), h)
            out[key] = (((S[:, 0] + k * L) / (S[:, 1] + k)) / L).astype(np.float32)
            out[f"lg_{r.name}"] = L.astype(np.float32)
    return out, tuned


def run_opponent(tune: bool = True, log=print) -> dict:
    """Team style factors for every team-game, including the live season's unplayed games (zero-count rows: they get
    factors from earlier games and add nothing). ``tune=False`` reuses the saved half-lives / strengths."""
    from ..data.database import upcoming_team_games
    from ..features.asof import time_axis
    d = PATHS.processed("all")
    tg = pd.read_parquet(d / "team_games.parquet")
    up = upcoming_team_games()
    games = time_axis(pd.read_parquet(d / "games.parquet"), up)
    if len(up):
        rows = up[["game_id", "team_id", "opp_team_id", "season", "game_date"]].copy()
        num = [c for c in tg.columns if c not in rows.columns and pd.api.types.is_numeric_dtype(tg[c])
               and not pd.api.types.is_bool_dtype(tg[c])]
        for c in num:
            rows[c] = 0.0
        tg = pd.concat([tg, rows], ignore_index=True)
    t = team_factor_frame(tg, games)
    params = None
    if not tune:
        params = json.loads((PATHS.models / "team_factor_params.json").read_text(encoding="utf-8"))
    fac, tuned = compute_factors(t, games, params=params, log=log)
    PATHS.ensure(PATHS.models)
    fac.to_parquet(PATHS.models / "team_factors.parquet", index=False)
    if tune:
        (PATHS.models / "team_factor_params.json").write_text(json.dumps(tuned, indent=2), encoding="utf-8")
    return tuned
