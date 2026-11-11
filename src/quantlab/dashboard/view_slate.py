"""Slate: all games on the as-of date with team-model lines and simulated outcomes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from quantlab.dashboard import state, style


def result_label(g) -> str:
    """Final score of a finished game; tip-off time (US Eastern) or status of an unplayed one."""
    if bool(getattr(g, "is_final", True)) and pd.notna(getattr(g, "home_pts", np.nan)):
        return f"{g.away_team} {int(g.away_pts)} - {int(g.home_pts)} {g.home_team}"
    status = str(getattr(g, "status", "") or "")
    if status == "live":
        return "In progress"
    if status == "postponed":
        return "Postponed"
    tip = pd.Timestamp(getattr(g, "tip_utc", pd.NaT))
    if pd.isna(tip):
        return "Scheduled"
    tip = tip.tz_localize("UTC") if tip.tzinfo is None else tip
    return tip.tz_convert("America/New_York").strftime("%I:%M %p ET").lstrip("0")


def injury_report(pr, sl: pd.DataFrame) -> None:
    """The user's injury report for unplayed games (all of them, not just this date)."""
    gids = [int(g) for g in sl.loc[~sl["is_final"] & sl["projectable"], "game_id"]]
    rows = pd.concat([pr.store.by_game[g] for g in gids if g in pr.store.by_game]) if gids else pd.DataFrame()
    if rows.empty:
        return
    tri = {}
    for r in sl.itertuples():
        tri[int(r.home_team_id)], tri[int(r.away_team_id)] = r.home_team, r.away_team
    rows = rows.drop_duplicates("person_id")
    label = {int(p): f"{pr.store.names.get(int(p), p)} ({tri.get(int(t), '')})"
             for p, t in zip(rows["person_id"], rows["team_id"])}
    ov = pr.overrides
    opts = sorted(set(label) | set(ov["out"]) | set(ov["available"]), key=lambda x: label.get(x, str(x)))
    with st.expander("Injury report for unplayed games"):
        st.caption("There is no injury feed: a player who missed his team's last three games or more is presumed out, "
                   "other rotation players play with the appearance rate of their missed-game streak (about 90% after "
                   "playing the previous game). Mark the official injury report here; it applies to every unplayed "
                   "game and re-projects teammates' minutes.")
        c1, c2 = st.columns(2)
        out = c1.multiselect("Out", opts, default=[x for x in ov["out"] if x in opts],
                             format_func=lambda x: label.get(x, str(x)), key="ir_out")
        avail = c2.multiselect("Available (back from a presumed absence)", opts,
                               default=[x for x in ov["available"] if x in opts],
                               format_func=lambda x: label.get(x, str(x)), key="ir_avail")
        if st.button("Save injury report", type="primary"):
            pr.set_overrides({"out": out, "available": avail, "p_play": ov.get("p_play", {})})
            state.reload_data()
            st.rerun()


def render() -> None:
    pr = state.projector()
    date = st.session_state["asof_date"]
    st.title("Slate")
    sl = pr.slate(date)
    if sl.empty:
        st.info("No games on this date.")
        return
    unplayed = bool((~sl["is_final"]).any())
    if unplayed:
        st.caption(f"Games on {pd.Timestamp(date):%A %B %d, %Y}. Unplayed games are projected truly pregame (earlier "
                   "games only; projected starters and availability, see the injury report below). At most 10,000 "
                   "simulations per game on this page.")
        injury_report(pr, sl)
    else:
        st.caption(f"Games on {pd.Timestamp(date):%A %B %d, %Y} - projected with pregame features and the starting "
                   "fives; who is available comes from who appeared, a stand-in for the pregame inactive list (see "
                   "Data & methods). At most 10,000 simulations per game on this page.")
    n = st.session_state["n_sims"]
    env = st.session_state["environment"]
    rows = []
    prog = st.progress(0.0, text="Simulating the slate ...")
    for k, g in enumerate(sl.itertuples(index=False)):
        row = {"game_id": int(g.game_id), "Matchup": f"{g.away_team} @ {g.home_team}", "Home win": np.nan,
               "Proj. score": "", "Spread (home)": np.nan, "Total": np.nan, "P(OT)": np.nan,
               "Team model spread": -float(g.exp_margin) if pd.notna(g.exp_margin) else np.nan,
               "Team model total": float(g.exp_total) if pd.notna(g.exp_total) else np.nan,
               "Result": result_label(g), "projected": False}
        if g.projectable:
            try:
                proj = state.run_projection(int(g.game_id), min(n, 10000), env)
                sc = proj.result.score.astype(float)
                mg = sc[:, 0] - sc[:, 1]
                row.update({"Home win": float((mg > 0).mean()),
                            "Proj. score": f"{g.away_team} {sc[:, 1].mean():.1f} - {sc[:, 0].mean():.1f} {g.home_team}",
                            "Spread (home)": -float(mg.mean()), "Total": float(sc.sum(axis=1).mean()),
                            "P(OT)": float((proj.result.n_ot > 0).mean()), "projected": True})
            except Exception as e:                     # one game's missing inputs must not break the slate
                row["Proj. score"] = f"projection unavailable ({type(e).__name__})"
        else:
            row["Proj. score"] = "projection available closer to the date"
        rows.append(row)
        prog.progress((k + 1) / len(sl), text=f"Simulated {k + 1}/{len(sl)} games")
    prog.empty()
    df = pd.DataFrame(rows)
    st.dataframe(df.drop(columns=["game_id", "projected"]), hide_index=True, use_container_width=True,
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
    df = df[df["projected"]].reset_index(drop=True)
    if df.empty:
        return
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
