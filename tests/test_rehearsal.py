"""The rehearsal's scoring (eval/pregame.py) on a tiny synthetic game, and the rehearsal script's folder guard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab import config
from quantlab.eval.pregame import STATS, score

ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_score_counts_coverage_starters_and_the_unprojected(monkeypatch, tmp_path):
    main = tmp_path / "main"
    # game 1: home team 10 (players 1-7, starters 1-5; 7 is not in the pool), away team 20 (players 11-16, only four
    # recorded starters 11-14; 16 was presumed out after a three-game streak and played anyway)
    home, away = list(range(1, 8)), list(range(11, 17))
    pg = pd.DataFrame({"game_id": 1, "team_id": [10] * 7 + [20] * 6, "person_id": home + away, "played": True,
                       "starter": [p in (1, 2, 3, 4, 5, 11, 12, 13, 14) for p in home + away],
                       "min": 20.0, "pts": 10.0, "reb": 4.0, "ast": 2.0, "fg3m": 1.0, "stl": 1.0, "blk": 0.0,
                       "tov": 1.0})
    _write(main / "processed" / "all" / "player_games.parquet", pg)
    _write(main / "processed" / "all" / "games.parquet", pd.DataFrame({"game_id": [1], "home_pts": [110],
                                                                         "away_pts": [100]}))
    _write(main / "models" / "team_predictions.parquet", pd.DataFrame({"game_id": [1], "exp_margin": [8.0],
                                                                         "exp_total": [212.0]}))
    asof = [{"game_id": 1, "person_id": p, "stat": s, "mean": 2.0 if s != "pra" else 6.0}
            for p in home + away for s in STATS]
    _write(main / "backtest" / "sim_blend_2026-27_players.parquet", pd.DataFrame(asof))
    _write(main / "backtest" / "sim_blend_2026-27_teams.parquet", pd.DataFrame({"game_id": [1], "sim_margin": [9.0],
                                                                                  "sim_total": [211.0]}))
    projected = [p for p in home + away if p not in (7, 16)]
    players = pd.DataFrame({"game_id": 1, "team_id": [10 if p < 10 else 20 for p in projected],
                            "person_id": projected, "days_ahead": 0,
                            "starter_proj": [p in (1, 2, 3, 4, 5, 11, 12, 13, 14, 15) for p in projected],
                            "rotation_proj": True, "p_play": 0.9, "pre_min_uncond": 18.0,
                            **{f"pre_{s}": 0.0 for s in STATS}})
    players["pre_min"] = 22.0
    teams = pd.DataFrame({"game_id": [1], "days_ahead": [0], "pre_margin": [7.0], "pre_total": [215.0],
                          "pre_p_home": [0.7]})
    feats = tmp_path / "rehearsal" / "data"
    _write(feats / "features" / "player_asof.parquet",
           pd.DataFrame({"game_id": 1, "team_id": [10 if p < 10 else 20 for p in projected + [16]],
                         "person_id": projected + [16], "missed_streak": [0] * len(projected) + [3]}))
    monkeypatch.setattr(config, "PATHS", config.Paths(root=feats))
    out = score(players, teams, main, "2026-27")
    assert out["player_games_played"] == 13 and out["common_player_games"] == 11
    assert out["roster_coverage"] == pytest.approx(11 / 13)
    assert out["not_projected"] == {"player_games": 2, "presumed_out_after_streak": 1, "not_in_pool": 1,
                                    "minutes_share": pytest.approx(2 / 13)}
    assert out["starters_correct_of_5"] == pytest.approx((5 + 4) / 2)      # four recorded starters on one side
    assert out["mae"]["min"]["pregame"] == pytest.approx(2.0) and out["bias"]["min"]["pregame"] == pytest.approx(2.0)
    # every appearance: the two unprojected count as projected 0 (error 20 minutes each)
    assert out["mae_pregame_all_appearances"]["min"] == pytest.approx((11 * 2 + 2 * 20) / 13)
    assert out["team"]["margin_rmse"]["pregame"] == pytest.approx(3.0)


@pytest.mark.parametrize("bad", [".", "data"])
def test_rehearsal_refuses_the_projects_data_folder(bad):
    before = sorted(p.name for p in (ROOT / "data").iterdir()) if (ROOT / "data").is_dir() else None
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "rehearse_live.py"), "prepare", "--dir", bad],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "not the project's data" in (r.stdout + r.stderr)
    after = sorted(p.name for p in (ROOT / "data").iterdir()) if (ROOT / "data").is_dir() else None
    assert after == before                                                 # nothing deleted


def test_rehearsal_refuses_a_folder_it_did_not_make(tmp_path):
    (tmp_path / "data" / "important").mkdir(parents=True)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "rehearse_live.py"), "prepare", "--dir", str(tmp_path)],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "not made by a rehearsal" in (r.stdout + r.stderr)
    assert (tmp_path / "data" / "important").exists()
