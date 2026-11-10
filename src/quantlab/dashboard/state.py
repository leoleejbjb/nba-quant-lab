"""Cached resources and simulation runs shared by the dashboard pages."""

from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd
import streamlit as st

from quantlab.config import PATHS
from quantlab.live import Projector
from quantlab.sim.scenario import Scenario

SEASONS = ("2023-24", "2024-25", "2025-26")   # seasons with walk-forward minutes projections
PAGES: dict = {}                               # filled by app.py: name -> st.Page (for st.switch_page)


@st.cache_resource(show_spinner="Loading models and the 2020-26 database ...")
def projector() -> Projector:
    return Projector()


@st.cache_data(show_spinner=False, max_entries=64)
def players_table() -> pd.DataFrame:
    return pd.read_parquet(PATHS.processed("all") / "players.parquet")


@st.cache_data(show_spinner=False, max_entries=8)
def game_logs(season: str) -> pd.DataFrame:
    cols = ["game_id", "game_date", "person_id", "name", "team", "opp", "is_home", "starter", "min", "pts", "reb",
            "ast", "fg3m", "fga", "fgm", "fta", "ftm", "stl", "blk", "tov", "pf", "plus_minus", "usg_pct", "ts_pct",
            "played", "season"]
    pg = pd.read_parquet(PATHS.processed("all") / "player_games.parquet", columns=cols)
    return pg[pg["season"] <= season]


def _scenario_key(sc: Scenario | None) -> str:
    return json.dumps(asdict(sc), sort_keys=True, default=str) if sc is not None else ""


@st.cache_data(show_spinner=False, max_entries=24, hash_funcs={Scenario: _scenario_key})
def run_projection(game_id: int, n: int, environment: str, scenario: Scenario | None = None, seed: int = 7):
    pr = projector()
    return pr.project(int(game_id), n=n, seed=seed, environment=environment, scenario=scenario)


def backtest_summary() -> dict | None:
    path = PATHS.backtest / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def select_game(game_id: int) -> None:
    st.session_state["game_id"] = int(game_id)


def selected_game() -> int | None:
    return st.session_state.get("game_id")
