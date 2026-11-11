"""Empirical-Bayes per-possession player rates (the inputs of the possession simulator).

Every rate is ``numerator / exposure`` where both are *decayed* pregame sums
(effective-time half-life ``h``) and the estimate is the conjugate posterior
mean under a prior centred on the decayed league average::

    rate = (X_h + k * mu_league) / (E_h + k)

i.e. Beta-Binomial for shares (shooting %, rebound %, usage share ...) and
Gamma-Poisson for counts per exposure (steals per opponent possession ...).
``k`` (prior strength, in exposure units) and ``h`` are chosen per rate by
maximising the predictive log-likelihood of the *next* game's count given its
actual exposure on the tuning season (2023-24); recent form therefore enters
through ``h`` and regression to the mean through ``k``.

Shooting priors are additionally conditioned on the player's long-horizon
shot profile (``prior_model='style'``) when that improves the tuning score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import gammaln

from ..config import PATHS
from ..features.asof import RATE_HALFLIVES, effective_time


@dataclass(frozen=True)
class RateSpec:
    name: str
    num: tuple            # numerator columns (summed)
    den: tuple            # exposure columns (summed); a leading '-' subtracts
    kind: str             # 'binom' | 'pois'
    group: str            # 'offense' | 'defense' | 'shooting' | 'passing' | 'rebounding'


def _plays(prefix: str) -> tuple:
    return (f"{prefix}fga", f"{prefix}trips", f"{prefix}tov")


RATES = [
    RateSpec("usage", ("fga2", "fga3", "trips2", "trips3", "tov"), ("on_fga", "on_trips", "on_tov"), "binom", "offense"),
    RateSpec("tov_share", ("tov",), ("fga2", "fga3", "trips2", "trips3", "tov"), "binom", "offense"),
    RateSpec("fga3_share", ("fga3",), ("fga2", "fga3", "trips2", "trips3"), "binom", "offense"),
    RateSpec("trip_share", ("trips2", "trips3"), ("fga2", "fga3", "trips2", "trips3"), "binom", "offense"),
    RateSpec("fg2", ("fgm2",), ("fga2",), "binom", "shooting"),
    RateSpec("fg3", ("fgm3",), ("fga3",), "binom", "shooting"),
    RateSpec("ft", ("ftm",), ("fta",), "binom", "shooting"),
    RateSpec("and1", ("and1",), ("fgm2", "fgm3"), "binom", "shooting"),
    RateSpec("trip3_share", ("trips3",), ("trips2", "trips3"), "binom", "shooting"),
    RateSpec("ast2_share", ("fgm2_ast",), ("fgm2",), "binom", "passing"),
    RateSpec("ast3_share", ("fgm3_ast",), ("fgm3",), "binom", "passing"),
    RateSpec("ast_rate", ("ast",), ("on_fgm", "-fgm2", "-fgm3"), "binom", "passing"),
    RateSpec("oreb", ("oreb",), ("on_oreb_ch",), "binom", "rebounding"),
    RateSpec("dreb", ("dreb",), ("op_oreb_ch",), "binom", "rebounding"),
    RateSpec("stl", ("stl",), ("op_poss",), "pois", "defense"),
    RateSpec("blk", ("blk",), ("op_fga",), "pois", "defense"),
    RateSpec("pf_def", ("pf", "-pf_offensive"), ("op_poss",), "pois", "defense"),
    RateSpec("off_foul_share", ("tov_off_foul",), ("tov",), "binom", "offense"),
    RateSpec("blocked", ("blocked",), ("fga2",), "binom", "shooting"),
]
SPEC = {r.name: r for r in RATES}
K_GRID = (2, 5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560)


def _combine(df: pd.DataFrame, cols: tuple, suffix: str = "", prefix: str = "") -> np.ndarray:
    """Sum (or subtract, for '-col') columns ``prefix + col + suffix``; floored at zero."""
    out = np.zeros(len(df))
    for c in cols:
        sign = -1.0 if c.startswith("-") else 1.0
        out += sign * df[prefix + c.lstrip("-") + suffix].to_numpy(dtype=float)
    return np.maximum(out, 0.0)


def league_prior(pg: pd.DataFrame, games: pd.DataFrame, spec: RateSpec, halflife: float = 60.0) -> pd.Series:
    """Decayed league-average rate as of each date (strictly earlier dates), indexed by game_date."""
    d = pg[pg["played"]]
    daily = pd.DataFrame({"x": _combine(d, spec.num), "e": _combine(d, spec.den), "game_date": d["game_date"]})
    daily = daily.groupby("game_date")[["x", "e"]].sum().sort_index()
    t = effective_time(pd.Series(daily.index, index=daily.index), games).to_numpy()
    lam = np.log(2) / halflife
    sx = se = 0.0
    out = np.zeros(len(daily))
    prev_t = t[0]
    for i, (x, e) in enumerate(daily.to_numpy()):
        decay = np.exp(-lam * (t[i] - prev_t))
        sx, se = sx * decay, se * decay
        out[i] = sx / se if se > 0 else np.nan
        sx, se, prev_t = sx + x, se + e, t[i]
    s = pd.Series(out, index=daily.index)
    return s.bfill()


def eb_estimate(X: np.ndarray, E: np.ndarray, mu: np.ndarray, k: float) -> np.ndarray:
    return (X + k * mu) / (E + k)


def loglik(kind: str, rate: np.ndarray, x: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Per-row predictive log-likelihood of count x given exposure e (constant terms kept for comparability)."""
    eps = 1e-9
    if kind == "binom":
        p = np.clip(rate, eps, 1 - eps)
        n = np.maximum(e, x)
        return (gammaln(n + 1) - gammaln(x + 1) - gammaln(n - x + 1) + x * np.log(p) + (n - x) * np.log(1 - p))
    lam = np.maximum(rate * e, eps)
    return x * np.log(lam) - lam - gammaln(x + 1)


def build_rate_frame(feats: pd.DataFrame, pg: pd.DataFrame) -> pd.DataFrame:
    """Rows = played player-games, with game-level actual numerators/exposures for scoring."""
    cols = ["game_id", "person_id"] + sorted({c.lstrip("-") for r in RATES for c in r.num + r.den})
    g = pg.loc[pg["played"], cols + ["game_date"]].rename(columns={c: f"g_{c}" for c in cols[2:]})
    return feats.merge(g.drop(columns=["game_date"]), on=["game_id", "person_id"], how="inner")


def tune_rates(rf: pd.DataFrame, pg: pd.DataFrame, games: pd.DataFrame, season: str = "2023-24", log=print) -> dict:
    """Choose (half-life, k) per rate by next-game predictive log-likelihood on ``season``."""
    sub = rf[rf["season"] == season]
    best = {}
    for spec in RATES:
        mu = sub["game_date"].map(league_prior(pg, games, spec)).to_numpy()
        xg = _combine(sub, spec.num, prefix="g_")
        eg = _combine(sub, spec.den, prefix="g_")
        ok = eg > 0
        res = []
        for h in RATE_HALFLIVES:
            X = _combine(sub, spec.num, f"_h{h}")
            E = _combine(sub, spec.den, f"_h{h}")
            for k in K_GRID:
                ll = loglik(spec.kind, eb_estimate(X, E, mu, k)[ok], xg[ok], eg[ok]).sum()
                res.append((ll, h, k))
        ll0 = loglik(spec.kind, mu[ok], xg[ok], eg[ok]).sum()        # league mean only
        ll, h, k = max(res)
        best[spec.name] = {"halflife": int(h), "k": int(k), "loglik_gain_per_row": float((ll - ll0) / ok.sum()),
                           "rows": int(ok.sum()), "kind": spec.kind}
        log(f"[rates] {spec.name:15s} h={h:4d} k={k:4d}  gain over league mean {best[spec.name]['loglik_gain_per_row']:+.4f}/row")
    return best


def compute_rates(rf: pd.DataFrame, pg: pd.DataFrame, games: pd.DataFrame, params: dict) -> pd.DataFrame:
    """EB rate estimates for every row of ``rf`` (pregame), plus the league priors used."""
    out = rf[["game_id", "person_id", "team_id", "opp_team_id", "season", "game_date"]].copy()
    for spec in RATES:
        p = params[spec.name]
        prior = league_prior(pg, games, spec)
        mu = rf["game_date"].map(prior).to_numpy()
        X = _combine(rf, spec.num, f"_h{p['halflife']}")
        E = _combine(rf, spec.den, f"_h{p['halflife']}")
        out[f"r_{spec.name}"] = eb_estimate(X, E, mu, p["k"]).astype(np.float32)
        out[f"lg_{spec.name}"] = mu.astype(np.float32)
        out[f"n_{spec.name}"] = E.astype(np.float32)       # effective sample size behind the estimate
    return out


def run_rates(log=print) -> dict:
    d = PATHS.processed("all")
    pg = pd.read_parquet(d / "player_games.parquet")
    games = pd.read_parquet(d / "games.parquet")
    feats = pd.read_parquet(PATHS.features / "player_asof.parquet")
    rf = build_rate_frame(feats, pg)
    params = tune_rates(rf, pg, games, log=log)
    PATHS.ensure(PATHS.models)
    (PATHS.models / "rate_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    # rates for all candidate rows (played or not) so that the simulator can use them pregame
    allrows = feats.copy()
    rates = compute_rates(allrows, pg, games, params)
    rates.to_parquet(PATHS.models / "player_rates.parquet", index=False)
    log(f"[rates] wrote {len(rates):,} rows")
    return params
