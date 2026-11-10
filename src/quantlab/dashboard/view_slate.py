"""Slate: all games on the as-of date with team-model lines and simulated outcomes."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from quantlab.dashboard import state, style


def render() -> None:
    pr = state.projector()
    date = st.session_state["asof_date"]
    st.title("Slate")
    st.caption(f"Games on {pd.Timestamp(date):%A %B %d, %Y} - projected with information available before tip-off.")
    sl = pr.slate(date)
    if sl.empty:
        st.info("No games on this date.")
        return
    n = st.session_state["n_sims"]
    env = st.session_state["environment"]
    rows = []
    prog = st.progress(0.0, text="Simulating the slate ...")
    for k, g in enumerate(sl.itertuples(index=False)):
        proj = state.run_projection(int(g.game_id), min(n, 10000), env)
        sc = proj.result.score.astype(float)
        mg = sc[:, 0] - sc[:, 1]
        rows.append({"game_id": int(g.game_id), "Matchup": f"{g.away_team} @ {g.home_team}",
                     "Home win": float((mg > 0).mean()), "Proj. score": f"{g.away_team} {sc[:, 1].mean():.1f} - "
                     f"{sc[:, 0].mean():.1f} {g.home_team}", "Spread (home)": -float(mg.mean()),
                     "Total": float(sc.sum(axis=1).mean()), "P(OT)": float((proj.result.n_ot > 0).mean()),
                     "Team model spread": -float(g.exp_margin), "Team model total": float(g.exp_total),
                     "Final": f"{g.away_team} {int(g.away_pts)} - {int(g.home_pts)} {g.home_team}"})
        prog.progress((k + 1) / len(sl), text=f"Simulated {k + 1}/{len(sl)} games")
    prog.empty()
    df = pd.DataFrame(rows)
    st.dataframe(df.drop(columns=["game_id"]), hide_index=True, use_container_width=True,
                 height=36 * (len(df) + 1) + 4,
                 column_config={
                     "Home win": st.column_config.ProgressColumn("Home win", format="percent", min_value=0, max_value=1),
                     "Spread (home)": st.column_config.NumberColumn(format="%+.1f",
                                                                    help="Negative = home team favoured"),
                     "Total": st.column_config.NumberColumn(format="%.1f"),
                     "P(OT)": st.column_config.NumberColumn(format="percent",
                                                            help="Simulated; understates real overtime rates "
                                                                 "(see the Validation page)"),
                     "Team model spread": st.column_config.NumberColumn(format="%+.1f"),
                     "Team model total": st.column_config.NumberColumn(format="%.1f")})
    st.markdown("##### Open a game")
    cols = st.columns(min(4, len(df)))
    for k, r in df.iterrows():
        with cols[k % len(cols)]:
            if st.button(r["Matchup"], key=f"open_{r['game_id']}", use_container_width=True):
                state.select_game(r["game_id"])
                st.switch_page(state.PAGES["game"])
    if state.selected_game() not in set(df["game_id"]):
        state.select_game(int(df["game_id"].iloc[0]))
    st.caption("Spread convention: negative numbers mean the home team is favoured. The simulation's environment "
               "mode is set in the sidebar.")
