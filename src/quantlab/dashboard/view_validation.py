"""Validation: walk-forward backtest evidence (player stats, minutes, calibration, team forecasts,
architecture comparison). Reads data/backtest/summary.json written by ``quantlab report``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quantlab.dashboard import state, style

MODEL_LABELS = {"season_avg": "Season average", "l10": "Last-10 average", "per_min": "Per-minute x proj. minutes",
                "opp_adj": "Opponent-adjusted per-minute", "eb": "EB rates (closed form)", "gbm": "Gradient boosting",
                "sim_none": "Simulation (bottom-up)", "sim_team": "Simulation (team-constrained)",
                "topdown": "Team-first reconciliation"}


def render() -> None:
    st.title("Validation")
    summ = state.backtest_summary()
    if summ is None:
        st.warning("No backtest summary yet. Run `quantlab backtest` and `quantlab report`.")
        return
    split = st.radio("Seasons", ["test", "tuning"], horizontal=True,
                     format_func={"test": f"Test: {', '.join(summ['test_seasons'])}",
                                  "tuning": f"Tuning: {summ['tuning_season']}"}.get)
    st.caption("Walk-forward: every prediction uses only games before its date. Hyper-parameters, simulator "
               "calibration and the blend weight were chosen on the tuning season; the test seasons were scored once.")
    pc = pd.DataFrame(summ[f"players_{split}"])
    st.subheader("Player statistics")
    metric = st.radio("Metric", ["mae", "rmse", "crps", "pit_dev", "cov80", "bias"], horizontal=True,
                      format_func={"mae": "MAE", "rmse": "RMSE", "crps": "CRPS (distribution)", "pit_dev":
                                   "Calibration error (PIT L1)", "cov80": "80% interval coverage", "bias": "Bias"}.get)
    tab = pc.pivot(index="model", columns="stat", values=metric)
    order = [m for m in MODEL_LABELS if m in tab.index]
    tab = tab.loc[order, [c for c in ["pts", "reb", "ast", "fg3m", "pra", "stl", "blk", "tov"] if c in tab.columns]]
    tab.index = [MODEL_LABELS[m] for m in tab.index]
    lower_better = metric in ("mae", "rmse", "crps", "pit_dev")
    if metric == "cov80":
        styled = tab.style.format("{:.1%}")
    elif metric == "bias":
        styled = tab.style.format("{:+.3f}")
    else:
        styled = tab.style.format("{:.3f}").highlight_min(axis=0, props="font-weight:700") if lower_better \
            else tab.style.format("{:.3f}")
    st.dataframe(styled, use_container_width=True)
    st.caption("Bold = best model for that statistic. Distributions for the baseline models are negative "
               "binomial around the point forecast with dispersion fitted on the tuning season. 80% coverage of "
               "discrete statistics is computed with inclusive interval ends, so values slightly above 80% are "
               "expected for a calibrated forecast; the PIT calibration error is the fairer check (0 = perfect).")
    st.subheader("Minutes")
    mt = pd.DataFrame(summ[f"minutes_{split}"])
    min_labels = {"season_avg": "Season average", "l10": "Last-10 average", "ewma": "Decayed average (EWMA)",
                  "gbm_pre": "LightGBM (no lineup info)", "gbm": "LightGBM (starting lineups known)",
                  "gbm_norm": "LightGBM, normalized to 240", "gbm_pre_norm": "LightGBM (no lineups), normalized",
                  "sim_none": "Simulation (bottom-up)", "sim_team": "Simulation (team-constrained)"}
    mt["model"] = mt["model"].map(min_labels).fillna(mt["model"])
    cfg = {c: st.column_config.NumberColumn(format="%.3f") for c in mt.columns if c not in ("model", "n")}
    cfg["n"] = st.column_config.NumberColumn("player-games", format="%d")
    st.dataframe(mt, hide_index=True, use_container_width=True, column_config=cfg)

    st.subheader("Calibration")
    pits = summ[f"pit_{split}"]
    tok = style.tokens()
    cols = st.columns(4)
    for k, (col, s_) in enumerate(zip(cols, ["pts", "reb", "ast", "pra"])):
        fig = go.Figure()
        top = 0.16
        for key, name, color in ((f"{s_}|eb", "EB + negative binomial", tok["gray"]),
                                 (f"{s_}|sim_none", "Simulation", tok["s1"])):
            if key not in pits:
                continue
            h = np.asarray(pits[key], dtype=float)
            h = h / h.sum()
            top = max(top, h.max() * 1.1)
            fig.add_trace(go.Bar(x=np.arange(10) / 10 + 0.05, y=h, name=name, marker_color=color,
                                 hovertemplate=f"{name}: PIT %{{x:.2f}} → %{{y:.1%}}<extra></extra>"))
        if not fig.data:
            continue
        fig.add_hline(y=0.1, line_color=tok["ink2"], line_width=1)
        style.base_layout(fig, 250, f"{s_.upper()}: PIT histogram", None, None, legend=(k == 0))
        fig.update_layout(barmode="group", bargap=0.15, bargroupgap=0.05)
        fig.update_yaxes(tickformat=".0%", range=[0, top])
        col.plotly_chart(fig, use_container_width=True)
    st.caption("A calibrated forecast has a flat PIT histogram at 10% per decile (the line). A U shape means "
               "too-narrow distributions, a hump means too-wide ones. Gray = the empirical-Bayes point forecast with a "
               "negative-binomial distribution; blue = the possession simulation.")
    rel = summ[f"reliability_{split}"]
    fig = go.Figure()
    colors = [tok["s1"], tok["s2"], tok["s3"], tok["s4"]]
    for k, s_ in enumerate(["pts", "reb", "ast", "fg3m"]):
        key = f"{s_}|sim_none"
        if key in rel:
            r = rel[key]
            fig.add_trace(go.Scatter(x=r["pred"], y=r["obs"], mode="lines+markers", name=s_.upper(),
                                     line=dict(width=2, color=colors[k]),
                                     marker=dict(size=8, line=dict(width=2, color=tok["surface"])),
                                     hovertemplate=f"{s_.upper()}: predicted %{{x:.0%}}, observed %{{y:.0%}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect", line=dict(width=1, color=tok["muted"])))
    style.base_layout(fig, 360, "Over/under reliability at the half-point line nearest the projection",
                      "predicted P(over)", "observed frequency")
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Team forecasts: does player-level simulation help?")
    tm = summ["teams"][split]
    rows = []
    for k, v in tm.items():
        if isinstance(v, dict) and "margin_rmse" in v:
            rows.append({"Forecast": {"team_model": "Team model (ratings)", "sim_none": "Simulation (bottom-up)",
                                      "sim_team": "Simulation (team-constrained)", "blend": "Blend"}[k],
                         **{x: v[x] for x in ("margin_mae", "margin_rmse", "total_mae", "total_rmse", "margin_crps",
                                              "total_crps", "win_logloss", "win_brier", "games")}})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in
                                ("margin_mae", "margin_rmse", "total_mae", "total_rmse", "margin_crps", "total_crps",
                                 "win_logloss", "win_brier")})
    bs = tm.get("bootstrap_mse_gain_vs_team_model")
    if bs:
        st.caption("Paired bootstrap over games (2,000 resamples), mean squared-error reduction vs the team model "
                   "(positive = better than the team model): " + "; ".join(
                       f"{k.replace('_', ' ')} {v['mean']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]"
                       for k, v in bs.items()))
    st.caption(f"Blend weight on the team model, chosen on the tuning season: {summ['teams']['blend_weight']:.2f}.")
    cost = summ.get("cost", {})
    if "us_per_game" in cost:
        st.subheader("Computational cost")
        st.markdown(f"{cost['n_sims']:,} simulated games in {cost['simulate_s']:.2f} s "
                    f"({cost['us_per_game']:.0f} µs per game on 4 cores) plus {cost['calibration_s']:.2f} s of "
                    f"pilot calibration; every invariant held: **{cost['invariants'].get('all')}**.")
