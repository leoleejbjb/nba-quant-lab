"""The cross-cutoff invariant, end to end (slow - opt in with QUANTLAB_SLOW_TESTS=1 - and it needs the full
pipeline's data): rehearse the live season on 2025-26 with the games from a cutoff hidden; every finished game's
model outputs, and the team-level inputs of the first unplayed date, must equal the full build's - so nothing from a
game reaches its own inputs or those of earlier games."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "data"
NEEDS = [MAIN / "raw" / "cdnnba_2025.tar.xz", MAIN / "processed" / "all" / "games.parquet",
         MAIN / "models" / "rate_params.json", MAIN / "models" / "team_factors.parquet"]
RUN = os.environ.get("QUANTLAB_SLOW_TESTS") == "1" and all(p.exists() for p in NEEDS)


def _same(a: pd.DataFrame, b: pd.DataFrame, keys: list, ids: set, label: str, full: bool) -> None:
    a, b = a[a["game_id"].isin(ids)], b[b["game_id"].isin(ids)]
    m = a.merge(b, on=keys, suffixes=("_full", "_rehearsal"))
    if full:
        assert len(m) == len(a) == len(b), f"{label}: rows differ ({len(a)} vs {len(b)})"
    assert len(m), f"{label}: nothing to compare"
    for c in a.columns:
        if c in keys or f"{c}_rehearsal" not in m or not pd.api.types.is_numeric_dtype(a[c]):
            continue
        if not full and m[f"{c}_rehearsal"].isna().all():
            continue                                       # an outcome (points, possessions): unknown before the game
        x, y = m[f"{c}_full"].to_numpy(float), m[f"{c}_rehearsal"].to_numpy(float)
        assert np.allclose(x, y, rtol=0, atol=1e-9, equal_nan=True), f"{label}: column {c} differs"


@pytest.mark.skipif(not RUN, reason="slow: set QUANTLAB_SLOW_TESTS=1 (needs the full pipeline's data)")
def test_hiding_the_games_from_a_cutoff_changes_nothing_before_it(tmp_path):
    cutoff = "2025-11-05"
    for step in ("prepare", "update"):
        subprocess.run([sys.executable, str(ROOT / "scripts" / "rehearse_live.py"), step, "--dir", str(tmp_path),
                        "--cutoff", cutoff, "--horizon", "3"], cwd=ROOT, check=True, timeout=7200)
    games = pd.read_parquet(MAIN / "processed" / "all" / "games.parquet", columns=["game_id", "game_date"])
    c = pd.Timestamp(cutoff)
    before = set(games.loc[games["game_date"] < c, "game_id"])
    first = set(games.loc[games["game_date"] == c, "game_id"])
    R = tmp_path / "data" / "models"
    for name, keys in (("team_predictions.parquet", ["game_id"]), ("player_rates.parquet", ["game_id", "person_id"]),
                       ("team_factors.parquet", ["game_id", "team_id"]),
                       ("minutes_predictions.parquet", ["game_id", "person_id"])):
        a, b = pd.read_parquet(MAIN / "models" / name), pd.read_parquet(R / name)
        _same(a, b, keys, before, f"{name}, finished games", full=True)
        if name != "minutes_predictions.parquet":          # unplayed games' minutes use the lineup-blind model
            _same(a, b, keys, first, f"{name}, first unplayed date", full=False)
