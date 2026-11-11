"""Data & methods: coverage, reconciliation against official box scores, and method notes."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from quantlab.config import LIVE_SEASON, PATHS, db_seasons


def render() -> None:
    st.title("Data & methods")
    rows = []
    for s in dict.fromkeys((*db_seasons(), LIVE_SEASON)):
        p = PATHS.processed(s) / "quality.json"
        if not p.exists():
            continue
        q = json.loads(p.read_text(encoding="utf-8"))
        o = q.get("official_box", {})
        rows.append({"Season": s, "Games": q["games"], "Usable games": q["games_ok"],
                     "Valid lineup time": q["valid_time_share"], "Player-games": q["player_games_played"],
                     "Minutes within 1 s": o.get("minutes_within_1s", float("nan")),
                     "PTS exact": o.get("pts_exact", float("nan")), "REB exact": o.get("reb_exact", float("nan")),
                     "AST exact": o.get("ast_exact", float("nan")),
                     "+/- exact": o.get("plus_minus_exact", float("nan")),
                     "All stats exact": o.get("all_stats_exact", float("nan"))})
    df = pd.DataFrame(rows)
    st.subheader("Play-by-play reconstruction vs official NBA.com box scores")
    st.dataframe(df, hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(format="percent") for c in df.columns
                                if c not in ("Season", "Games", "Usable games", "Player-games")})
    st.caption("Official box scores exist in the pinned sources only through 2023-24; later seasons are checked for "
               "internal consistency (points conserved, 5 x game length minutes per team).")
    lq = PATHS.processed(LIVE_SEASON) / "quality.json"
    if lq.exists():
        live = json.loads(lq.read_text(encoding="utf-8")).get("live", {})
        st.subheader(f"Current season ({LIVE_SEASON})")
        st.markdown(
            f"Fetched from the NBA's live CDN (cdn.nba.com) by `quantlab live-update` on this computer - the same "
            f"play-by-play feed the pinned mirror archives, not pinned because the season is still being played. "
            f"{live.get('built', 0)} of {live.get('final_in_schedule', 0)} finished games built (through "
            f"{live.get('last_game_date', '-')}), {live.get('scheduled_games', 0)} games in the schedule. Unplayed games "
            "are projected truly pregame: no inactive list and no starting fives exist yet, so availability is "
            "estimated from missed-game streaks (plus the injury report you enter on the Slate page) and starters "
            "are projected; this is less information than the backtest used (see docs/METHODOLOGY.md, current "
            "season).")
    ax_path = PATHS.models / "availability_experiment.json"
    if ax_path.exists():
        ax = json.loads(ax_path.read_text(encoding="utf-8"))
        st.subheader("Who is available: the inactive-list stand-in")
        st.markdown(
            f"The injury report / inactive list is public before tip-off but is in the sources only for 2020-24. "
            f"Every season therefore uses *who appeared* as the stand-in for rotation players' availability, which "
            f"also reveals the rotation players a coach chose not to use: "
            f"{ax['absent_rotation_status_share'].get('dnp_cd', 0):.1%} of the rotation players it treats as "
            f"unavailable (2021-24) were dressed (DNP - coach's decision). On {ax['season']}, with the official "
            f"statuses, the minutes model's MAE is **{ax['mae_appearance_proxy']:.3f}** with the stand-in and "
            f"**{ax['mae_inactive_list']:.3f}** with the real inactive list (+{ax['mae_leak']:.3f}): the test-season "
            "numbers carry about this much optimism.")
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
