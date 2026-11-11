"""NBA Quant Lab dashboard. Run with ``quantlab dashboard`` or
``streamlit run src/quantlab/dashboard/app.py``."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from quantlab.config import LIVE_SEASON
from quantlab.dashboard import (state, style, view_compare, view_data, view_game, view_player, view_scenario,
                                view_slate, view_validation)

st.set_page_config(page_title="NBA Quant Lab", page_icon=":material/sports_basketball:", layout="wide")
style.css()

pr = state.projector()
LIVE = pr.store.schedule is not None
with st.sidebar:
    st.markdown("### NBA Quant Lab")
    seasons = state.seasons(pr)
    season = st.selectbox("Season", seasons, index=len(seasons) - 1, key="season",
                          format_func=lambda s: f"{s} (current)" if LIVE and s == LIVE_SEASON else s)
    allg = pr._all_games()
    allg = allg[allg["season"] == season]
    dates = pr.dates(season)
    if not dates:
        st.warning(f"No games for {season} yet.")
        st.stop()
    unplayed = set(pd.to_datetime(allg.loc[~allg["is_final"], "game_date"]).dt.date)
    projectable = set(pd.to_datetime(allg.loc[~allg["is_final"] & allg.index.isin(pr.store.by_game.keys()),
                                              "game_date"]).dt.date)
    default_date = st.session_state.get("asof_date")
    if default_date not in dates:
        if projectable:                                    # the current season: the next slate to be played
            today = pd.Timestamp.now(tz="America/New_York").date()
            later = sorted(d for d in projectable if d >= today)
            default_date = later[0] if later else min(projectable)
        else:
            counts = pd.to_datetime(allg["game_date"]).dt.date.value_counts()
            full = [d for d in dates if counts.get(d, 0) >= 8 and (dates[-1] - d).days >= 7]
            default_date = full[-1] if full else dates[-1]     # latest full slate (not the rest-heavy final week)

    def _label(d) -> str:
        tag = " · upcoming" if d in projectable else (" · not projected yet" if d in unplayed else "")
        return pd.Timestamp(d).strftime("%a %b %d, %Y") + tag

    date = st.selectbox("Date (games on this date)", dates, index=dates.index(default_date), key="asof_date",
                        format_func=_label)
    st.select_slider("Simulations per game", options=[5000, 10000, 20000, 40000], value=20000, key="n_sims",
                     help="The Slate page runs at most 10,000 per game.")
    st.radio("Game environment", ["blend", "team", "none"], key="environment", horizontal=True,
             format_func={"blend": "Blend", "team": "Team model", "none": "Bottom-up"}.get,
             help="How the team-level model constrains the simulation (pace always comes from it): 'Blend' "
                  "(default, the backtested forecast) targets a mix of the team model's and the players' points "
                  "per possession with the weight chosen on the 2023-24 tuning season; 'Team model' matches the "
                  "team model's efficiency; 'Bottom-up' lets player projections alone decide (see Validation).")
    if date in unplayed:
        st.caption("Unplayed games are projected truly pregame: from earlier games only, with projected starting "
                   "fives and availability estimated from recent games (there is no injury report: mark players out "
                   "on the Slate page). Finished games are shown as backtested.")
    else:
        st.caption("Finished games are projected as in the backtest: features from earlier dates, plus the game's "
                   "starting fives and who appeared, the stand-in for the pregame inactive list (see Data & "
                   "methods). The actual results are shown for comparison.")
    ls = state.live_status()
    if ls and LIVE:
        st.caption(f"Current season data updated {str(ls.get('updated_at', ''))[:16].replace('T', ' ')} UTC, "
                   f"games through {ls.get('last_game_date')}.")
    if st.button("Reload data", help="After `quantlab live-update`: load the new games and models"):
        state.reload_data()
        st.rerun()

pages = [
    st.Page(view_slate.render, title="Slate", icon=":material/calendar_month:", default=True),
    st.Page(view_game.render, title="Projected box score", icon=":material/table_chart:", url_path="game"),
    st.Page(view_player.render, title="Player detail & props", icon=":material/person:", url_path="player"),
    st.Page(view_scenario.render, title="Scenario lab", icon=":material/tune:", url_path="scenario"),
    st.Page(view_compare.render, title="Player comparison", icon=":material/compare_arrows:", url_path="compare"),
    st.Page(view_validation.render, title="Validation", icon=":material/fact_check:", url_path="validation"),
    st.Page(view_data.render, title="Data & methods", icon=":material/menu_book:", url_path="data"),
]
state.PAGES.update({"slate": pages[0], "game": pages[1], "player": pages[2], "scenario": pages[3],
                    "compare": pages[4], "validation": pages[5], "data": pages[6]})
st.navigation({"Player Projections": pages[:5], "Evidence": pages[5:]}).run()
