"""Cached resources and simulation runs shared by the dashboard pages."""

from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd
import streamlit as st

from quantlab.config import LIVE_SEASON, PATHS, TEST_SEASONS, TUNING_SEASON
from quantlab.live import OVERRIDES_PATH, Projector
from quantlab.sim.scenario import Scenario

PAGES: dict = {}                               # filled by app.py: name -> st.Page (for st.switch_page)
_VERSION_FILES = [PATHS.processed("all") / "games.parquet", PATHS.processed("all") / "schedule.parquet",
                  PATHS.processed("all") / "player_games.parquet", PATHS.features / "player_asof.parquet",
                  PATHS.features / "team_context.parquet", PATHS.features / "fringe_pool.parquet",
                  PATHS.models / "player_rates.parquet", PATHS.models / "minutes_predictions.parquet",
                  PATHS.models / "minutes_features.parquet", PATHS.models / "team_predictions.parquet",
                  PATHS.models / "team_factors.parquet", PATHS.models / "integration.json",
                  PATHS.root / "live_status.json", OVERRIDES_PATH]      # live_status.json: written last by live-update


def data_version() -> str:
    """Changes whenever the pipeline / live-update rewrites the inputs or the injury report is edited: the cache key
    of the loaded models and of every simulation."""
    return "-".join(str(f.stat().st_mtime_ns) if f.exists() else "0" for f in _VERSION_FILES)


@st.cache_resource(show_spinner="Loading models and the database ...", max_entries=1)
def _projector(version: str) -> Projector:
    return Projector()


def projector() -> Projector:
    return _projector(data_version())


def seasons(pr: Projector) -> list:
    """Seasons with walk-forward minutes projections: the tuning and test seasons, then the seasons from the live
    feed (the current one last) once they have data."""
    have = set(pr._all_games()["season"])
    out = [s for s in (TUNING_SEASON, *TEST_SEASONS) if s in have]
    out += sorted(s for s in have if s > TEST_SEASONS[-1] and s not in out)
    if LIVE_SEASON in have and LIVE_SEASON not in out:
        out.append(LIVE_SEASON)
    return out


def live_status() -> dict | None:
    p = PATHS.root / "live_status.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data(show_spinner=False, max_entries=4)
def _players_table(version: str) -> pd.DataFrame:
    return pd.read_parquet(PATHS.processed("all") / "players.parquet")


def players_table() -> pd.DataFrame:
    return _players_table(data_version())


def game_logs(season: str) -> pd.DataFrame:
    return _game_logs(season, data_version())


@st.cache_data(show_spinner=False, max_entries=8)
def _game_logs(season: str, version: str) -> pd.DataFrame:
    cols = ["game_id", "game_date", "person_id", "name", "team", "opp", "is_home", "starter", "min", "pts", "reb",
            "ast", "fg3m", "fga", "fgm", "fta", "ftm", "stl", "blk", "tov", "pf", "plus_minus", "usg_pct", "ts_pct",
            "played", "season"]
    pg = pd.read_parquet(PATHS.processed("all") / "player_games.parquet", columns=cols)
    return pg[pg["season"] <= season]


def _scenario_key(sc: Scenario | None) -> str:
    return json.dumps(asdict(sc), sort_keys=True, default=str) if sc is not None else ""


@st.cache_data(show_spinner=False, max_entries=24, hash_funcs={Scenario: _scenario_key})
def _run_projection(game_id: int, n: int, environment: str, scenario: Scenario | None, seed: int, version: str):
    pr = _projector(version)
    return pr.project(int(game_id), n=n, seed=seed, environment=environment, scenario=scenario)


def run_projection(game_id: int, n: int, environment: str, scenario: Scenario | None = None, seed: int = 7):
    return _run_projection(int(game_id), n, environment, scenario, seed, data_version())


def reload_data() -> None:
    st.cache_resource.clear()
    st.cache_data.clear()


def backtest_summary() -> dict | None:
    path = PATHS.backtest / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def select_game(game_id: int) -> None:
    st.session_state["game_id"] = int(game_id)


def selected_game() -> int | None:
    return st.session_state.get("game_id")
