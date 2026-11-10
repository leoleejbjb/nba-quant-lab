"""Data & methods: coverage, reconciliation against official box scores, and method notes."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from quantlab.config import ALL_SEASONS, PATHS


def render() -> None:
    st.title("Data & methods")
    rows = []
    for s in ALL_SEASONS:
        p = PATHS.processed(s) / "quality.json"
        if not p.exists():
            continue
        q = json.loads(p.read_text())
        o = q.get("official_box", {})
        rows.append({"Season": s, "Games": q["games"], "Usable games": q["games_ok"],
                     "Valid lineup time": q["valid_time_share"], "Player-games": q["player_games_played"],
                     "Minutes within 1 s": o.get("minutes_within_1s"), "PTS exact": o.get("pts_exact"),
                     "REB exact": o.get("reb_exact"), "AST exact": o.get("ast_exact"),
                     "+/- exact": o.get("plus_minus_exact"), "All stats exact": o.get("all_stats_exact")})
    df = pd.DataFrame(rows)
    st.subheader("Play-by-play reconstruction vs official NBA.com box scores")
    st.dataframe(df, hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(format="percent") for c in df.columns
                                if c not in ("Season", "Games", "Usable games", "Player-games")})
    st.caption("Official box scores exist in the pinned sources only through 2023-24; later seasons are checked for "
               "internal consistency (points conserved, 5 x game length minutes per team).")
    st.subheader("Method in brief")
    st.markdown("""
* **Data**: NBA liveData play-by-play 2020-21 .. 2025-26 (pinned mirror, SHA-256 verified). Player box scores,
  on-court exposures, lineups (stints) and possessions are reconstructed from the events and checked against
  official box scores.
* **Team model**: time-decayed, opponent-adjusted ridge ratings for offense, defense and pace (walk-forward).
* **Minutes**: LightGBM on recent minutes, role, rest and *pregame* teammate absences (fringe players'
  participation is never used as a feature, because it depends on blowouts).
* **Rates**: empirical-Bayes (Beta-Binomial / Gamma-Poisson) per-possession rates, half-life and prior strength
  tuned per statistic on 2023-24.
* **Simulation**: possession-by-possession Monte Carlo with five-man rotations, fouls, garbage time, overtime and
  correlated player/game effects; every simulated box score is internally consistent by construction.
* Full details: `docs/METHODOLOGY.md`; results: `docs/REPORT.md`.
""")
