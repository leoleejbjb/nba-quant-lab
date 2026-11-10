"""Player detail: simulated distributions, a prop calculator for arbitrary lines (with pushes on whole
numbers), double/triple-double odds, a joint-probability builder and the recent game log."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quantlab.dashboard import state, style
from quantlab.dashboard.view_game import _game_picker
from quantlab.sim import props

STAT_LABELS = {"pts": "Points", "reb": "Rebounds", "ast": "Assists", "fg3m": "3-pointers made", "pra": "Pts+Reb+Ast",
               "stl": "Steals", "blk": "Blocks", "tov": "Turnovers", "min": "Minutes", "fga": "FG attempts",
               "fta": "FT attempts", "stocks": "Steals+Blocks"}


def roster(res):
    out = []
    for t in (0, 1):
        team = res.game.home if t == 0 else res.game.away
        for pl in res.players[t]:
            out.append((pl.person_id, f"{pl.name} ({team.name})"))
    return out


def dist_chart(x: np.ndarray, line: float, label: str) -> go.Figure:
    tok = style.tokens()
    is_min = label == "Minutes"
    if is_min:
        bins = np.arange(0, np.ceil(x.max()) + 2, 1.0)
        h, e = np.histogram(x, bins=bins)
        k = e[:-1] + 0.5
    else:
        xi = x.astype(int)
        k = np.arange(0, xi.max() + 1)
        h = np.bincount(xi, minlength=len(k))
    p = h / max(len(x), 1)
    over = k > line
    push = (k == line) if float(line).is_integer() else np.zeros_like(over)
    fig = go.Figure()
    for mask, name, color in ((over, f"over {line:g}", tok["s1"]), (~over & ~push, f"under {line:g}", tok["gray"]),
                              (push, f"push ({line:g})", tok["s4"])):
        if mask.any():
            fig.add_trace(go.Bar(x=k[mask], y=p[mask], name=name, marker_color=color,
                                 hovertemplate=f"{label} %{{x}}: %{{y:.1%}}<extra>{name}</extra>"))
    fig.add_vline(x=line, line_color=tok["ink2"], line_width=1)
    style.base_layout(fig, 300, f"{label}: simulated distribution", label.lower(), "probability")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(barmode="overlay")
    return fig


def render() -> None:
    pr = state.projector()
    st.title("Player detail & props")
    gid = _game_picker(pr)
    if gid is None:
        return
    proj = state.run_projection(gid, st.session_state["n_sims"], st.session_state["environment"])
    res = proj.result
    ros = roster(res)
    ids = [r[0] for r in ros]
    cur = st.session_state.get("person_id")
    if cur not in ids:
        mins = []
        for pid_ in ids:
            t_, i_ = props.find_player(res, pid_)
            mins.append(float(res.stat("min")[:, t_, i_].mean()))
        cur = ids[int(np.argmax(mins))]
    pid = st.selectbox("Player", ids, index=ids.index(cur), format_func=dict(ros).get, key="player_pick")
    st.session_state["person_id"] = pid
    t, i = props.find_player(res, pid)
    played = res.secs[:, t, i] > 0
    st.caption(f"Distributions are conditional on the player appearing (he appears in "
               f"{style.pct(played.mean())} of simulated games); a player who does not play voids a prop.")
    c = st.columns(6)
    for col, s_ in zip(c, ("min", "pts", "reb", "ast", "fg3m", "pra")):
        d = props.distribution_summary(res, pid, s_)
        col.metric(STAT_LABELS[s_], f"{d['mean']:.1f}", help=f"median {d['median']:.0f} · 80% range "
                                                            f"{d['p10']:.0f}–{d['p90']:.0f}")
    st.markdown("##### Prop calculator")
    cl, cr = st.columns([1, 2])
    with cl:
        stat = st.selectbox("Statistic", list(STAT_LABELS), format_func=STAT_LABELS.get, key="prop_stat")
        x = props.samples(res, pid, stat)
        default = float(np.floor(x.mean()) + 0.5)
        line = st.number_input("Line", value=default, step=0.5, min_value=0.0, key=f"line_{pid}_{stat}")
        lp = props.line_probs(x, line)
        st.metric(f"Over {line:g}", style.pct(lp["over"]), help=f"fair odds {style.odds(lp['fair_over_american'])}")
        st.metric(f"Under {line:g}", style.pct(lp["under"]), help=f"fair odds {style.odds(lp['fair_under_american'])}")
        if lp["push"] > 0:
            st.metric("Push", style.pct(lp["push"]), help="Whole-number line: the stake is returned on exactly this value")
        st.metric("Double-double", style.pct(props.double_double(res, pid)))
        st.metric("Triple-double", style.pct(props.triple_double(res, pid)))
    with cr:
        st.plotly_chart(dist_chart(x, line, STAT_LABELS[stat]), use_container_width=True)
    st.markdown("##### Prop ladder")
    rows = []
    for s_ in ("pts", "reb", "ast", "fg3m", "pra", "stl", "blk", "tov"):
        xs = props.samples(res, pid, s_)
        for L in props.standard_lines(xs.mean())[1:6]:
            lp = props.line_probs(xs, L)
            rows.append({"Stat": STAT_LABELS[s_], "Line": L, "Over": lp["over"], "Under": lp["under"],
                         "Fair over": lp["fair_over_american"], "Fair under": lp["fair_under_american"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=320,
                 column_config={"Over": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                                "Under": st.column_config.NumberColumn(format="percent"),
                                "Fair over": st.column_config.NumberColumn(format="%+.0f"),
                                "Fair under": st.column_config.NumberColumn(format="%+.0f")})

    st.markdown("##### Joint probability (same simulated games, not multiplied marginals)")
    n_legs = st.radio("Legs", [2, 3], horizontal=True, key="n_legs")
    legs = []
    cols = st.columns(n_legs)
    for k, col in enumerate(cols):
        with col:
            lp_id = st.selectbox(f"Player {k + 1}", ids, index=ids.index(pid) if k == 0 else min(k, len(ids) - 1),
                                 format_func=dict(ros).get, key=f"leg_p{k}")
            ls = st.selectbox("Stat", ["pts", "reb", "ast", "fg3m", "pra", "stl", "blk"], index=k % 3,
                              format_func=STAT_LABELS.get, key=f"leg_s{k}")
            side = st.radio("Side", ["over", "under"], horizontal=True, key=f"leg_side{k}")
            xs = props.samples(res, lp_id, ls)
            lv = st.number_input("Line", value=float(np.floor(xs.mean()) + 0.5), step=0.5, key=f"leg_l{k}_{lp_id}_{ls}")
            legs.append(props.Leg(lp_id, ls, side, lv))
    jp = props.joint_probability(res, legs)
    c = st.columns(3)
    c[0].metric("Joint probability", style.pct(jp["joint"]))
    c[1].metric("If independent", style.pct(jp["independent_product"]))
    c[2].metric("Correlation lift", f"{jp['correlation_lift']:.2f}x",
                help="Joint / product of marginals. Teammates competing for shots usually push this below 1; one "
                     "player's points and assists, or a player's points with the game total, push it above 1.")

    st.markdown("##### Recent games (before this date)")
    logs = state.game_logs(st.session_state["season"])
    date = pd.Timestamp(st.session_state["asof_date"])
    lg = logs[(logs["person_id"] == pid) & (logs["game_date"] < date) & logs["played"]].sort_values(
        "game_date", ascending=False).head(10)
    st.dataframe(lg[["game_date", "team", "opp", "min", "pts", "reb", "ast", "fg3m", "fgm", "fga", "ftm", "fta", "stl",
                     "blk", "tov", "plus_minus"]],
                 hide_index=True, use_container_width=True,
                 column_config={"game_date": st.column_config.DateColumn("Date"),
                                "min": st.column_config.NumberColumn("MIN", format="%.1f")})
