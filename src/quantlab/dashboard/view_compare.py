"""Player comparison: two or more players side by side - projections for the selected game (if they
play in it), season-to-date per-36 production and recent form, and overlaid simulated distributions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quantlab.dashboard import state, style
from quantlab.sim import props

STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov"]


def render() -> None:
    st.title("Player comparison")
    season = st.session_state["season"]
    date = pd.Timestamp(st.session_state["asof_date"])
    logs = state.game_logs(season)
    lg = logs[(logs["season"] == season) & (logs["game_date"] < date) & logs["played"]]
    agg = lg.groupby(["person_id", "name"]).agg(games=("game_id", "nunique"), team=("team", "last"),
                                                min=("min", "sum"), **{s: (s, "sum") for s in STATS},
                                                fga=("fga", "sum"), fta=("fta", "sum"), mpg=("min", "mean"))
    agg = agg.reset_index()
    agg = agg[agg["games"] >= 5].sort_values("min", ascending=False)
    if agg.empty:
        st.info("Not enough games before this date.")
        return
    opts = agg["person_id"].tolist()
    labels = dict(zip(agg["person_id"], agg["name"] + " (" + agg["team"] + ")"))
    default = [p for p in [st.session_state.get("person_id")] if p in labels] or opts[:2]
    pick = st.multiselect("Players (2-3 recommended; colors follow the order you pick)", opts,
                          default=default[:3] if len(default) >= 2 else opts[:2], format_func=labels.get,
                          max_selections=6, key="cmp_players")
    if len(pick) < 2:
        st.info("Pick at least two players.")
        return
    a = agg.set_index("person_id").loc[pick]
    per36 = pd.DataFrame({"Player": [labels[p] for p in pick], "GP": a["games"].to_numpy(),
                          "MPG": a["mpg"].to_numpy()})
    for s_ in STATS:
        per36[f"{s_.upper()}/36"] = (a[s_] / a["min"] * 36).to_numpy()
    per36["TS%"] = (a["pts"] / (2 * (a["fga"] + 0.44 * a["fta"]))).to_numpy()
    recent = []
    for p in pick:
        r = lg[lg["person_id"] == p].sort_values("game_date").tail(10)
        recent.append({f"L10 {s_.upper()}": r[s_].mean() for s_ in ("min", "pts", "reb", "ast")})
    per36 = pd.concat([per36, pd.DataFrame(recent)], axis=1)
    st.markdown(f"##### Season to date (before {date:%b %d, %Y})")
    st.dataframe(per36, hide_index=True, use_container_width=True,
                 column_config={**{c: st.column_config.NumberColumn(format="%.1f") for c in per36.columns
                                   if c not in ("Player", "GP", "TS%")},
                                "TS%": st.column_config.NumberColumn(format="percent")})
    # projections for the selected game, when the players are in it
    gid = state.selected_game()
    if gid is None:
        return
    proj = state.run_projection(gid, st.session_state["n_sims"], st.session_state["environment"])
    res = proj.result
    in_game = []
    for p in pick:
        try:
            props.find_player(res, p)
            in_game.append(p)
        except KeyError:
            pass
    if not in_game:
        st.caption("None of these players is in the selected game; open a game on the Slate page to compare "
                   "their projected distributions.")
        return
    st.markdown("##### Projected distributions in the selected game")
    stat = st.selectbox("Statistic", ["pts", "reb", "ast", "fg3m", "pra", "min"], key="cmp_stat",
                        format_func={"pts": "Points", "reb": "Rebounds", "ast": "Assists", "fg3m": "3PM",
                                     "pra": "PRA", "min": "Minutes"}.get)
    tok = style.tokens()
    colors = [tok["s1"], tok["s2"], tok["s3"]]
    fig = go.Figure()
    rows = []
    for k, p in enumerate(in_game[:3]):
        x = props.samples(res, p, stat)
        if stat == "min":
            h, e = np.histogram(x, bins=np.arange(0, 50, 1.0))
            kk = e[:-1] + 0.5
        else:
            kk = np.arange(0, int(x.max()) + 1)
            h = np.bincount(x.astype(int), minlength=len(kk))
        fig.add_trace(go.Scatter(x=kk, y=h / len(x), mode="lines+markers", name=labels[p], line=dict(width=2,
                                 color=colors[k]), marker=dict(size=8, line=dict(width=2, color=tok["surface"])),
                                 hovertemplate=f"{labels[p]}: %{{x}} → %{{y:.1%}}<extra></extra>"))
        d = props.distribution_summary(res, p, stat)
        rows.append({"Player": labels[p], "Mean": d["mean"], "Median": d["median"], "10th": d["p10"], "90th": d["p90"],
                     "SD": d["sd"]})
    style.base_layout(fig, 340, None, stat, "probability")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(format="%.1f") for c in ("Mean", "Median", "10th",
                                                                                         "90th", "SD")})
