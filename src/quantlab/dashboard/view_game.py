"""Projected box score for one game: both teams, uncertainty ranges, score distributions,
scoring contributions, lineup allocation and (for historical dates) the actual box score."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quantlab.dashboard import state, style
from quantlab.live import box_score_table, team_totals


def _game_picker(pr) -> int | None:
    sl = pr.slate(st.session_state["asof_date"])
    sl = sl[sl["projectable"]] if "projectable" in sl else sl
    if sl.empty:
        st.info("No games to project on the selected date.")
        return None
    labels = {int(r.game_id): f"{r.away_team} @ {r.home_team}" for r in sl.itertuples()}
    cur = state.selected_game()
    if cur not in labels:
        cur = next(iter(labels))
    gid = st.selectbox("Game", list(labels), index=list(labels).index(cur), format_func=labels.get, key="game_pick")
    state.select_game(gid)
    return gid


def box_table(df: pd.DataFrame) -> pd.DataFrame:
    """NBA-style box score of projected means."""
    if df.empty:
        return df
    return pd.DataFrame({
        "Player": df["Player"], "": df["Start"], "MIN": df["MIN"], "PTS": df["PTS"], "REB": df["REB"],
        "AST": df["AST"],
        "FG": [f"{a:.1f}-{b:.1f}" for a, b in zip(df["FGM"], df["FGA"])],
        "3P": [f"{a:.1f}-{b:.1f}" for a, b in zip(df["3PM"], df["3PA"])],
        "FT": [f"{a:.1f}-{b:.1f}" for a, b in zip(df["FTM"], df["FTA"])],
        "OREB": df["OREB"], "DREB": df["DREB"], "STL": df["STL"], "BLK": df["BLK"], "TOV": df["TOV"],
        "PF": df["PF"], "+/-": df["+/-"]})


def range_table(df: pd.DataFrame) -> pd.DataFrame:
    """10th-90th percentile ranges (the 80% interval) and participation."""
    if df.empty:
        return df
    rng = lambda k: [f"{a:.0f}–{b:.0f}" for a, b in zip(df[f"{k}_lo"], df[f"{k}_hi"])]
    return pd.DataFrame({"Player": df["Player"], "P(play)": df["P(play)"], "MIN": rng("MIN"), "PTS": rng("PTS"),
                         "REB": rng("REB"), "AST": rng("AST")})


BOX_CFG = {c: st.column_config.NumberColumn(format="%.1f") for c in
           ["MIN", "PTS", "REB", "AST", "OREB", "DREB", "STL", "BLK", "TOV", "PF"]}
BOX_CFG["+/-"] = st.column_config.NumberColumn(format="%+.1f")
BOX_CFG[""] = st.column_config.TextColumn(width="small", help="Projected starter")
RANGE_CFG = {"P(play)": st.column_config.NumberColumn(format="percent", help="Share of simulated games he appears in")}
for c in ["MIN", "PTS", "REB", "AST"]:
    RANGE_CFG[c] = st.column_config.TextColumn(help="10th–90th percentile of the simulated distribution "
                                                    "(conditional on playing)")


def render() -> None:
    pr = state.projector()
    st.title("Projected box score")
    gid = _game_picker(pr)
    if gid is None:
        return
    proj = state.run_projection(gid, st.session_state["n_sims"], st.session_state["environment"])
    res = proj.result
    g = pr.game_info(gid)
    final = pr.is_final(gid)
    home, away = res.game.home, res.game.away
    sc = res.score.astype(float)
    mg, tot = sc[:, 0] - sc[:, 1], sc.sum(axis=1)
    c = st.columns(5)
    c[0].metric(f"{home.name} win probability", style.pct(float((mg > 0).mean())))
    c[1].metric(f"Score ({away.name} – {home.name})", f"{sc[:, 1].mean():.1f} – {sc[:, 0].mean():.1f}")
    c[2].metric("Spread (home)", f"{-mg.mean():+.1f}",
                help="Negative = home team favoured. 80% interval of the margin: "
                     f"{np.percentile(mg, 10):+.0f} to {np.percentile(mg, 90):+.0f}")
    c[3].metric("Total", f"{tot.mean():.1f}", help=f"80% interval {np.percentile(tot, 10):.0f}–{np.percentile(tot, 90):.0f}")
    c[4].metric("Overtime", style.pct(float((res.n_ot > 0).mean())),
                help="Share of simulated games that go to overtime. The simulator understates overtime: regulation "
                     "ties are rarer than in real games (see the Validation page).")
    tp = pr.store.tp.loc[gid] if gid in pr.store.tp.index else None
    team_line = (f"team model: spread {-tp['exp_margin']:+.1f}, total {tp['exp_total']:.1f}, pace {tp['pace']:.1f}"
                 if tp is not None else "no team-model line")
    if final:
        result = f"**final {away.name} {int(g['away_pts'])} – {int(g['home_pts'])} {home.name}**"
    else:
        from quantlab.dashboard.view_slate import result_label
        result = f"**not played yet ({result_label(pd.Series({**g.to_dict(), 'is_final': False}))})**"
    st.caption(f"{pd.Timestamp(g['game_date']):%a %b %d, %Y} · {res.n:,} simulated games · environment: "
               f"{st.session_state['environment']} · {team_line} · {result}")
    if not final:
        meta = proj.calibration.get("pregame", {})
        names = pr.store.names
        lines = []
        for t, team in ((1, away), (0, home)):
            m = meta.get("home" if t == 0 else "away", {})
            st_ = ", ".join(names.get(p, str(p)) for p in m.get("starters", []))
            po = ", ".join(names.get(p, str(p)) for p in m.get("presumed_out", [])) or "none"
            lines.append(f"**{team.name}** projected starters ({m.get('starters_source', '?')}): {st_}; presumed out "
                         f"(missed three or more games in a row): {po}")
        st.info("Pregame projection: nothing from this game is used. " + " · ".join(lines) + ". Mark the injury "
                "report on the Slate page; P(play) below includes each player's chance of not playing.")

    for t, team in ((1, away), (0, home)):
        st.subheader(f"{team.name} {'(home)' if t == 0 else '(away)'}")
        bt = box_score_table(res, t)
        tab_box, tab_rng = st.tabs(["Box score (projected means)", "Uncertainty (80% ranges)"])
        h = 36 * (len(bt) + 1) + 4
        tab_box.dataframe(box_table(bt), hide_index=True, use_container_width=True, column_config=BOX_CFG, height=h)
        tab_rng.dataframe(range_table(bt), hide_index=True, use_container_width=True, column_config=RANGE_CFG,
                          height=h)
    st.markdown("##### Team totals (projected means)")
    tt = team_totals(res)
    st.dataframe(tt, hide_index=True, use_container_width=True,
                 column_config={k: st.column_config.NumberColumn(format="%.1f") for k in tt.columns if k != "Team"})

    tok = style.tokens()
    c1, c2 = st.columns(2)
    with c1:
        bins = np.arange(np.floor(mg.min()) - 0.5, np.ceil(mg.max()) + 1.5, 1.0)
        h, e = np.histogram(mg, bins=bins)
        x = (e[:-1] + e[1:]) / 2
        fig = go.Figure(go.Bar(x=x, y=h / len(mg), marker_color=np.where(x > 0, tok["s1"], tok["s2"]),
                               hovertemplate="margin %{x:+.0f}: %{y:.1%}<extra></extra>", showlegend=False))
        fig.add_vline(x=0, line_color=tok["axis"], line_width=1)
        style.base_layout(fig, 300, f"Final margin ({home.name} minus {away.name})", "points", "share of games",
                          legend=False)
        fig.update_yaxes(tickformat=".0%")
        fig.add_annotation(x=x.max(), y=max(h / len(mg)), text=f"{home.name} wins {style.pct((mg > 0).mean())}",
                           showarrow=False, xanchor="right", font=dict(color=tok["ink2"]))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        bins = np.arange(np.floor(tot.min()) - 0.5, np.ceil(tot.max()) + 1.5, 2.0)
        h, e = np.histogram(tot, bins=bins)
        fig = go.Figure(go.Bar(x=(e[:-1] + e[1:]) / 2, y=h / len(tot), marker_color=tok["s1"],
                               hovertemplate="total %{x:.0f}: %{y:.1%}<extra></extra>"))
        if tp is not None:
            fig.add_vline(x=float(tp["exp_total"]), line_color=tok["ink2"], line_width=1)
            fig.add_annotation(x=float(tp["exp_total"]), y=1.02, yref="paper", text="team model", showarrow=False,
                               font=dict(color=tok["ink2"]))
        style.base_layout(fig, 300, "Total points", "points", "share of games", legend=False)
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Contribution to team scoring")
    c1, c2 = st.columns(2)
    for col, (t, team) in zip((c1, c2), ((0, home), (1, away))):
        pts = res.stat("pts")[:, t, :]
        m = pts.mean(axis=0)
        names = [p.name for p in res.players[t]]
        order = np.argsort(-m)[: min(10, len(names))][::-1]
        team_mean = float(pts.sum(axis=1).mean())
        fig = go.Figure(go.Bar(
            x=m[order], y=[names[i] for i in order], orientation="h",
            marker_color=tok["s1"] if t == 0 else tok["s2"],
            error_x=dict(type="data", symmetric=False, array=np.percentile(pts, 90, axis=0)[order] - m[order],
                         arrayminus=m[order] - np.percentile(pts, 10, axis=0)[order], color=tok["muted"],
                         thickness=1, width=0),
            customdata=np.c_[m[order] / team_mean],
            hovertemplate="%{y}: %{x:.1f} pts (%{customdata[0]:.0%} of team)<extra></extra>"))
        style.base_layout(fig, 360, f"{team.name}: expected points (bar) and 10–90% range", "points", None,
                          legend=False)
        fig.update_xaxes(showgrid=True, gridcolor=tok["grid"])
        fig.update_yaxes(showgrid=False)
        col.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Minutes and lineup allocation")
    c1, c2 = st.columns(2)
    for col, (t, team) in zip((c1, c2), ((0, home), (1, away))):
        ls = res.lineup_secs[t]
        top = np.argsort(-ls)[:6]
        names = [p.name.split(" ")[-1] for p in res.players[t]]
        rows = []
        for mask in top:
            if ls[mask] <= 0:
                continue
            members = [names[i] for i in range(len(names)) if (int(mask) >> i) & 1]
            rows.append({"Lineup": " · ".join(members), "Min / game": ls[mask] / 60.0 / res.n})
        col.markdown(f"**{team.name}** most-used five-man units")
        col.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                      column_config={"Min / game": st.column_config.NumberColumn(format="%.1f")})

    if not final:
        return
    with st.expander("Projection vs. actual box score (historical game)"):
        act = pr.actual_box(gid)
        rows = []
        for t in (0, 1):
            for i, pl in enumerate(res.players[t]):
                a = act[act["person_id"] == pl.person_id]
                if a.empty or not bool(a["played"].iloc[0]):
                    continue
                played = res.secs[:, t, i] > 0
                if played.sum() < 10:
                    continue
                r = {"Team": (home if t == 0 else away).name, "Player": pl.name}
                for s_ in ("min", "pts", "reb", "ast", "fg3m"):
                    x = res.stat(s_)[played, t, i]
                    y = float(a[s_].iloc[0])
                    r[f"{s_.upper()} proj"] = float(x.mean())
                    r[f"{s_.upper()} act"] = y
                    r[f"{s_.upper()} pct"] = float((x < y).mean() + 0.5 * (x == y).mean())
                rows.append(r)
        d = pd.DataFrame(rows)
        cfg = {c_: st.column_config.NumberColumn(format="%.1f") for c_ in d.columns if c_.endswith("proj")}
        cfg.update({c_: st.column_config.NumberColumn(format="percent",
                                                      help="Percentile of the actual value in the simulated "
                                                           "distribution (uniform if calibrated)")
                    for c_ in d.columns if c_.endswith("pct")})
        st.dataframe(d, hide_index=True, use_container_width=True, column_config=cfg)
