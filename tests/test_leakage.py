"""Walk-forward guarantees: features and team predictions never use same-day or future results."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.config import PATHS
from quantlab.features.asof import decayed_sums
from quantlab.models.rates import eb_estimate


def test_decayed_sums_exclude_current_and_future_rows():
    keys = np.array([1, 1, 1, 2, 2])
    t = np.array([0.0, 10.0, 20.0, 0.0, 5.0])
    X = np.array([[1.0], [2.0], [4.0], [8.0], [16.0]])
    S = decayed_sums(keys, t, X, halflife=10.0)
    assert S[0, 0] == 0.0 and S[3, 0] == 0.0                      # first game of each key: no history
    assert np.isclose(S[1, 0], 1.0 * 0.5)                         # one half-life earlier
    assert np.isclose(S[2, 0], 1.0 * 0.25 + 2.0 * 0.5)
    X2 = X.copy()
    X2[2, 0] = 1000.0                                             # change the last game
    assert np.allclose(decayed_sums(keys, t, X2, 10.0)[:3], S[:3])  # earlier pregame features unchanged


def test_eb_estimate_shrinks_toward_prior():
    assert np.isclose(eb_estimate(np.array([0.0]), np.array([0.0]), np.array([0.36]), 100)[0], 0.36)
    big = eb_estimate(np.array([450.0]), np.array([1000.0]), np.array([0.36]), 100)[0]
    assert 0.36 < big < 0.45


@pytest.mark.skipif(not (PATHS.processed("all") / "team_games.parquet").exists(), reason="needs built data")
def test_team_model_ignores_same_day_results():
    from quantlab.models.team import TeamModelConfig, walk_forward
    d = PATHS.processed("all")
    tg = pd.read_parquet(d / "team_games.parquet")
    games = pd.read_parquet(d / "games.parquet")
    tg = tg[tg["season"] == "2023-24"]
    games = games[games["season"] == "2023-24"]
    date = sorted(games["game_date"].unique())[60]
    a = walk_forward(tg, games, TeamModelConfig())
    tg2 = tg.copy()
    same_day = tg2["game_date"] == date
    tg2.loc[same_day, "pts"] = tg2.loc[same_day, "pts"] + 50              # corrupt that day's results
    b = walk_forward(tg2, games, TeamModelConfig())
    ids = games.loc[games["game_date"] <= date, "game_id"]
    pa = a.set_index("game_id").loc[ids, "exp_margin"]
    pb = b.set_index("game_id").loc[ids, "exp_margin"]
    assert np.allclose(pa, pb)                                            # predictions up to that date unchanged
    later = games.loc[games["game_date"] > date, "game_id"]
    assert not np.allclose(a.set_index("game_id").loc[later, "exp_total"], b.set_index("game_id").loc[later, "exp_total"])


@pytest.mark.skipif(not (PATHS.features / "player_asof.parquet").exists(), reason="needs built features")
def test_player_features_use_only_earlier_games():
    f = pd.read_parquet(PATHS.features / "player_asof.parquet",
                        columns=["person_id", "game_date", "season", "one_h40", "career_games", "played"])
    f = f.sort_values(["person_id", "game_date"])
    first = f.groupby("person_id").head(1)
    assert (first["one_h40"] == 0).all() and (first["career_games"] == 0).all()


@pytest.mark.skipif(not (PATHS.processed("2023-24") / "quality.json").exists(), reason="needs built data")
def test_reconstruction_matches_official_box_scores():
    import json
    q = json.loads((PATHS.processed("2023-24") / "quality.json").read_text())["official_box"]
    assert q["minutes_within_1s"] > 0.999 and q["pts_exact"] > 0.999 and q["plus_minus_exact"] > 0.995
    assert q["all_stats_exact"] > 0.99
