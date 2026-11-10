"""Scenario lab: injuries, rest, minutes restrictions, lineup and usage changes.

Removing a player re-runs the minutes model for his teammates with the updated absence features
(minutes go where they historically went, not pro rata); shots, usage, assists and rebounds are then
re-allocated inside the possession simulation, and both versions are simulated with the same seed.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quantlab.dashboard import state, style
from quantlab.dashboard.view_game import _game_picker
from quantlab.sim import props
from quantlab.sim.engine import GameSimulator
from quantlab.sim.scenario import Scenario, compare


def roster_frame(spec) -> pd.DataFrame:
    """Editable roster with the values the simulation actually uses (after the team's minutes budget):
    rotation players' minutes if they play, fringe players' expected minutes and participation."""
    sim = GameSimulator(copy.deepcopy(spec))
    rows = []
    for t, team in ((1, spec.away), (0, spec.home)):
        for i, p in enumerate(team.players):
            rows.append({"team_id": team.team_id, "Team": team.name, "person_id": p.person_id, "Player": p.name,
                         "Out": False, "Starter": bool(p.starter), "Proj MIN": round(float(sim.min_mean[t, i]), 1),
                         "P(play)": round(float(sim.p_play[t, i]), 2), "Usage x": 1.0, "MIN cap": 48.0,
                         "rotation": bool(p.rotation)})
    return pd.DataFrame(rows)


def build_scenario(base: pd.DataFrame, ed: pd.DataFrame) -> Scenario:
    """Diff the edited roster table against the baseline table (rows aligned) into a Scenario."""
    sc = Scenario()
    for (_, b), (_, e) in zip(base.iterrows(), ed.iterrows()):
        pid = int(b["person_id"])
        if bool(e["Out"]):
            sc.out.append(pid)
            continue
        if abs(float(e["Proj MIN"]) - float(b["Proj MIN"])) > 0.05:
            sc.minutes[pid] = float(e["Proj MIN"])
        if abs(float(e["P(play)"]) - float(b["P(play)"])) > 1e-6:
            sc.availability[pid] = float(e["P(play)"])
        elif pid in sc.minutes and not bool(b["rotation"]) and sc.minutes[pid] < 6:
            # a deep-bench minutes edit keeps the participation shown in the table (the simulated, post-budget one)
            sc.availability[pid] = float(b["P(play)"])
        if abs(float(e["Usage x"]) - 1.0) > 1e-6:
            sc.usage[pid] = float(e["Usage x"])
        if float(e["MIN cap"]) < 48.0:
            sc.caps[pid] = float(e["MIN cap"])
    for tid in base["team_id"].unique():
        b5 = set(base.loc[(base["team_id"] == tid) & base["Starter"], "person_id"])
        e5 = set(ed.loc[(ed["team_id"] == tid) & ed["Starter"].astype(bool) & ~ed["Out"].astype(bool), "person_id"])
        if e5 != b5 and len(e5) == 5:
            sc.starters[int(tid)] = [int(x) for x in e5]
    return sc


def render() -> None:
    pr = state.projector()
    st.title("Scenario lab")
    st.caption("Edit the rosters and run the scenario. The baseline and the scenario are simulated with the same "
               "random seed (common random numbers), which reduces but does not remove Monte Carlo noise in the "
               "differences; small changes can still be noise.")
    gid = _game_picker(pr)
    if gid is None:
        return
    n = st.session_state["n_sims"]
    env = st.session_state["environment"]
    base_proj = state.run_projection(gid, n, env)
    base = roster_frame(pr.spec(gid))
    initial = base.copy()
    q_out = {int(x) for x in str(st.query_params.get("out", "")).split(",") if x.strip().isdigit()}
    if q_out:                                   # shareable links: ?out=<person_id>,<person_id>
        initial["Out"] = initial["person_id"].isin(q_out)
    ed = st.data_editor(initial.drop(columns=["team_id", "person_id", "rotation"]), hide_index=True,
                        use_container_width=True, key=f"editor_{gid}",
                        disabled=["Team", "Player"],
                        column_config={"Out": st.column_config.CheckboxColumn(help="Ruled out (injury / rest)"),
                                       "Starter": st.column_config.CheckboxColumn(help="Five per team"),
                                       "Proj MIN": st.column_config.NumberColumn(
                                           min_value=0.0, max_value=48.0, step=0.5,
                                           help="Rotation players: minutes if he plays. Deep bench: expected minutes "
                                                "per game. Edited values are used as typed."),
                                       "P(play)": st.column_config.NumberColumn(
                                           min_value=0.0, max_value=1.0, step=0.05,
                                           help="Probability he plays (e.g. 0.5 = questionable; deep bench: the "
                                                "chance he gets into the game). Edited values are used as typed."),
                                       "Usage x": st.column_config.NumberColumn(min_value=0.3, max_value=2.0, step=0.05),
                                       "MIN cap": st.column_config.NumberColumn(min_value=0.0, max_value=48.0, step=1.0,
                                                                              help="Minutes restriction")})
    ed = pd.concat([base[["team_id", "person_id"]].reset_index(drop=True), ed.reset_index(drop=True)], axis=1)
    sc = build_scenario(base.reset_index(drop=True), ed[base.columns.drop("rotation")])
    st.caption(f"Scenario: {len(sc.out)} out · {len(sc.minutes)} minutes overrides · {len(sc.availability)} "
               f"availability changes · {len(sc.usage)} usage changes · {len(sc.caps)} caps · "
               f"{len(sc.starters)} lineup changes")
    if sc.is_empty():
        st.info("Change something in the table (e.g. tick **Out** for a starter) to compare.")
        return
    with st.spinner("Simulating the scenario ..."):
        scen_proj = state.run_projection(gid, n, env, scenario=sc)
    cmp = compare(base_proj.result, scen_proj.result)
    tb, ts, td = cmp["team_base"], cmp["team_scenario"], cmp["team_delta"]
    home, away = base_proj.result.game.home.name, base_proj.result.game.away.name
    c = st.columns(4)
    c[0].metric(f"{home} win probability", style.pct(ts["p_home_win"]), f"{100 * td['p_home_win']:+.1f} pts")
    c[1].metric("Spread (home)", f"{-ts['margin']:+.1f}", f"{-td['margin']:+.1f}", delta_color="inverse")
    c[2].metric("Total", f"{ts['total']:.1f}", f"{td['total']:+.1f}")
    c[3].metric(f"{home} / {away} points", f"{ts['home_pts']:.1f} / {ts['away_pts']:.1f}",
                f"{td['home_pts']:+.1f} / {td['away_pts']:+.1f}", delta_color="off")
    m = cmp["players"]
    m = m[(m["min_base"].fillna(0) + m["min_scen"].fillna(0)) > 1].copy()
    show = m[["person_id", "team", "name", "min_base", "d_min", "pts_base", "d_pts", "reb_base", "d_reb", "ast_base",
              "d_ast", "fg3m_base", "d_fg3m"]].sort_values("d_pts", key=np.abs, ascending=False)
    st.markdown("##### Player changes (scenario minus baseline, expected values)")
    st.dataframe(show.drop(columns=["person_id"]).rename(columns={"team": "Team", "name": "Player"}), hide_index=True,
                 use_container_width=True,
                 column_config={**{c_: st.column_config.NumberColumn(format="%.1f") for c_ in show.columns
                                   if c_.endswith("_base")},
                                **{c_: st.column_config.NumberColumn(format="%+.2f") for c_ in show.columns
                                   if c_.startswith("d_")}})
    tok = style.tokens()
    top = show.head(14).iloc[::-1]
    fig = go.Figure(go.Bar(x=top["d_pts"], y=top["name"] + " (" + top["team"] + ")", orientation="h",
                           marker_color=np.where(top["d_pts"] >= 0, tok["pos"], tok["neg"]),
                           hovertemplate="%{y}: %{x:+.2f} points<extra></extra>"))
    fig.add_vline(x=0, line_color=tok["axis"], line_width=1)
    style.base_layout(fig, 420, "Change in expected points", "points (scenario - baseline)", None, legend=False)
    fig.update_xaxes(showgrid=True, gridcolor=tok["grid"])
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("##### Prop probabilities: baseline vs scenario (same line)")
    rows = []
    for pid in show["person_id"].head(8):
        try:
            x0 = props.samples(base_proj.result, int(pid), "pts")
            x1 = props.samples(scen_proj.result, int(pid), "pts")
        except KeyError:
            continue
        L = float(np.floor(x0.mean()) + 0.5)
        rows.append({"Player": m.loc[m.person_id == pid, "name"].iloc[0], "Points line": L,
                     "Over (baseline)": props.line_probs(x0, L)["over"], "Over (scenario)": props.line_probs(x1, L)["over"]})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                     column_config={"Over (baseline)": st.column_config.NumberColumn(format="percent"),
                                    "Over (scenario)": st.column_config.NumberColumn(format="percent")})
