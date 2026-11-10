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
                "sim_blend": "Simulation (blend, shipped)", "topdown": "Team-first reconciliation"}
TEAM_LABELS = {"team_model": "Team model (ratings)", "sim_none": "Simulation (bottom-up)",
               "sim_team": "Simulation (team-constrained)", "sim_blend": "Simulation (blend, shipped)",
               "posthoc_blend": "Post-hoc weighted average of means (reference)"}


def render() -> None:
    st.title("Validation")
    summ = state.backtest_summary()
    if summ is None:
        st.warning("No backtest summary yet. Run `make backtest` and `make report` (quantlab backtest, blendweight, "
                   "baselines, ablation, topdown, report).")
        return
    split = st.radio("Seasons", ["test", "tuning"], horizontal=True,
                     format_func={"test": f"Test: {', '.join(summ['test_seasons'])}",
                                  "tuning": f"Tuning: {summ['tuning_season']}"}.get)
    leak = (summ.get("availability_experiment") or {}).get("mae_leak")
    st.caption("Walk-forward: every feature uses only games before its date, plus the game's starting fives and who "
               "appeared (the stand-in for the pregame inactive list" +
               (f", worth about {leak:.2f} minutes of minutes MAE on 2023-24" if leak is not None else "") +
               "; see Data & methods). Hyper-parameters, simulator calibration and the blend weight were chosen on "
               "the tuning season; the final version was scored on the test seasons with everything frozen.")
    pc = pd.DataFrame(summ[f"players_{split}"])
    st.subheader("Player statistics")
    metric = st.radio("Metric", ["mae", "rmse", "crps", "pit_dev", "logscore", "cov80", "p_pit_below_half", "bias"],
                      horizontal=True,
                      format_func={"mae": "MAE", "rmse": "RMSE", "crps": "CRPS (distribution)", "pit_dev":
                                   "Calibration error (PIT L1)", "logscore": "Log score",
                                   "cov80": "80% interval coverage", "p_pit_below_half": "Share below median",
                                   "bias": "Bias"}.get)
    tab = pc.pivot(index="model", columns="stat", values=metric)
    order = [m for m in MODEL_LABELS if m in tab.index]
    tab = tab.loc[order, [c for c in ["pts", "reb", "ast", "fg3m", "pra", "stl", "blk", "tov"] if c in tab.columns]]
    tab.index = [MODEL_LABELS[m] for m in tab.index]
    lower_better = metric in ("mae", "rmse", "crps", "pit_dev")
    if metric in ("cov80", "p_pit_below_half"):
        styled = tab.style.format("{:.1%}")
    elif metric == "bias":
        styled = tab.style.format("{:+.3f}")
    elif metric == "logscore":
        styled = tab.style.format("{:.4f}").highlight_max(axis=0, props="font-weight:700")
    else:
        styled = tab.style.format("{:.3f}").highlight_min(axis=0, props="font-weight:700") if lower_better \
            else tab.style.format("{:.3f}")
    st.dataframe(styled, use_container_width=True)
    st.caption("Bold = lowest value for that statistic before rounding (for the log score: highest); values that "
               "tie at the decimals shown are effectively level. Distributions for the six baseline models (season "
               "average to gradient boosting) are negative binomial around the point forecast, with a mean-dependent "
               "dispersion fitted on the tuning season (gradient boosting, which has no tuning-season predictions, "
               "uses the empirical-Bayes model's curve); the team-first reconciliation and the simulations use their "
               "own simulated draws. Log score = mean log probability of the observed value. 80% coverage = share of outcomes in the central 80% of the forecast distribution with ties "
               "randomized (exactly 80% for a calibrated forecast); share below median: 50% when calibrated, more "
               "means the forecast is too high.")
    st.subheader("Minutes")
    mt = pd.DataFrame(summ[f"minutes_{split}"])
    min_labels = {"season_avg": "Season average", "l10": "Last-10 average", "ewma": "Decayed average (EWMA)",
                  "gbm_pre": "LightGBM (no lineup info)", "gbm": "LightGBM (starting lineups known)",
                  "gbm_norm": "LightGBM, scaled to team budget", "gbm_pre_norm": "LightGBM (no lineups), scaled",
                  "sim_none": "Simulation (bottom-up)", "sim_team": "Simulation (team-constrained)",
                  "sim_blend": "Simulation (blend, shipped)"}
    mt["model"] = mt["model"].map(min_labels).fillna(mt["model"])
    mt = mt.apply(lambda c: pd.to_numeric(c, errors="coerce") if c.name != "model" else c)
    # distribution columns exist only for the simulation: show them as text so the other rows stay blank
    for c in ("cov80", "pit_dev", "crps"):
        if c in mt:
            mt[c] = mt[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    names = {"mae": "MAE", "rmse": "RMSE", "bias": "Bias", "mae_rotation": "MAE (rotation players)",
             "cov80": "80% coverage", "pit_dev": "PIT error", "crps": "CRPS"}
    cfg = {c: st.column_config.NumberColumn(names.get(c, c), format="%.3f") for c in ("mae", "rmse", "bias",
                                                                                     "mae_rotation") if c in mt}
    cfg.update({c: st.column_config.TextColumn(names[c]) for c in ("cov80", "pit_dev", "crps") if c in mt})
    cfg["model"] = st.column_config.TextColumn("Model")
    cfg["n"] = st.column_config.NumberColumn("player-games", format="%d")
    st.dataframe(mt, hide_index=True, use_container_width=True, column_config=cfg)
    st.caption("Minutes if the player appears. 'Scaled' rows multiply each team's rotation projections so that, with "
               "the fringe players' baseline minutes, they fill the team budget (240 x the expected game length).")

    st.subheader("Calibration")
    pits = summ[f"pit_{split}"]
    tok = style.tokens()
    cols = st.columns(4)
    for k, (col, s_) in enumerate(zip(cols, ["pts", "reb", "ast", "pra"])):
        fig = go.Figure()
        top = 0.16
        for key, name, color in ((f"{s_}|gbm", "Gradient boosting + negative binomial", tok["gray"]),
                                 (f"{s_}|sim_blend", "Simulation", tok["s1"])):
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
        style.base_layout(fig, 250, f"{s_.upper()}: PIT histogram", None, None, legend=False)
        fig.update_layout(barmode="group", bargap=0.15, bargroupgap=0.05)
        fig.update_yaxes(tickformat=".0%", range=[0, top])
        col.plotly_chart(fig, use_container_width=True)
    has_gbm = any(f"{s_}|gbm" in pits for s_ in ("pts", "reb", "ast", "pra"))
    st.markdown((f"<span style='color:{tok['gray']}'>■</span> gradient boosting + negative binomial &nbsp;&nbsp; "
                 if has_gbm else "") +
                f"<span style='color:{tok['s1']}'>■</span> possession simulation (shipped blend)", unsafe_allow_html=True)
    st.caption("A calibrated forecast has a flat PIT histogram at 10% per decile (the line). A U shape means "
               "too-narrow distributions, a hump means too-wide ones, a tilt means a biased mean. " +
               ("Gray = the gradient-boosting point forecast with a mean-dependent negative-binomial distribution; "
                "blue = the possession simulation." if has_gbm else
                "Blue = the possession simulation; gradient boosting has no tuning-season predictions (2023-24 is "
                "its first training season), so it is shown for the test seasons only."))
    rel = summ[f"reliability_{split}"]
    fig = go.Figure()
    colors = [tok["s1"], tok["s2"], tok["s3"], tok["s4"]]
    for k, s_ in enumerate(["pts", "reb", "ast", "fg3m"]):
        key = f"{s_}|sim_blend|ladder"
        if key in rel:
            r = rel[key]
            fig.add_trace(go.Scatter(x=r["pred"], y=r["obs"], mode="lines+markers", name=s_.upper(),
                                     line=dict(width=2, color=colors[k]),
                                     marker=dict(size=8, line=dict(width=2, color=tok["surface"])),
                                     hovertemplate=f"{s_.upper()}: predicted %{{x:.0%}}, observed %{{y:.0%}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect", line=dict(width=1, color=tok["muted"])))
    style.base_layout(fig, 380, "Over/under reliability, lines from 2 below to 2 above the nearest half point",
                      "predicted P(over)", "observed frequency")
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
                      margin=dict(l=8, r=8, t=40, b=60))
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    stats8 = ["pts", "reb", "ast", "fg3m", "pra", "stl", "blk", "tov"]
    near = {s_: rel.get(f"{s_}|sim_blend", {}).get("mean_gap") for s_ in stats8}
    gaps = {s_: rel.get(f"{s_}|sim_blend|ladder", {}).get("mean_gap") for s_ in stats8}
    if any(v is not None for v in near.values()):
        st.caption("Mean predicted P(over) minus observed over rate (percentage points; 0 = calibrated). At the line "
                   "nearest the projection: " + ", ".join(f"{k.upper()} {100 * v:+.1f}" for k, v in near.items()
                                                          if v is not None) +
                   ". Pooled over the ladder (up to five lines, lines below 0.5 skipped; the gaps differ by line, "
                   "see docs/REPORT.md): " + ", ".join(f"{k.upper()} {100 * v:+.1f}" for k, v in gaps.items()
                                                       if v is not None) + ".")

    st.subheader("Team forecasts: does player-level simulation help?")
    tm = summ["teams"][split]
    rows = []
    for k, v in tm.items():
        if isinstance(v, dict) and "margin_rmse" in v:
            rows.append({"Forecast": TEAM_LABELS.get(k, k),
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
    st.caption(f"Blend weight on the team model, chosen on the tuning season: {summ['teams']['blend_weight']:.2f}; "
               "the blend row is the shipped simulator environment, backtested as such.")
    ots = {k: v["p_ot"] for k, v in tm.items() if isinstance(v, dict) and "p_ot" in v}
    if ots:
        st.caption(f"Overtime: actual {tm['actual_ot_rate']:.1%} of games; simulated mean P(OT) " +
                   ", ".join(f"{TEAM_LABELS[k]} {v:.1%}" for k, v in ots.items()) +
                   " - the simulator understates overtime (regulation ties are too rare).")
    co = summ.get("consistency")
    if co:
        st.subheader("Why a joint simulation: adding up independent player projections")
        rows = []
        for m, name in (("eb", "EB rates"), ("gbm", "Gradient boosting")):
            if m in co:
                r = co[m]
                rows.append({"Player model": name, "Gap vs team model (MAE)": r["mae_gap_vs_team_model"],
                             "Team-points RMSE of the sum": r["rmse_team_points_from_sum"],
                             "Implied SD if independent": r["implied_sd_if_independent"],
                             "Realised SD": r["realised_sd"], "80% coverage if independent": r["coverage80_if_independent"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                     column_config={"80% coverage if independent": st.column_config.NumberColumn(format="percent"),
                                    **{c: st.column_config.NumberColumn(format="%.2f") for c in
                                       ("Gap vs team model (MAE)", "Team-points RMSE of the sum",
                                        "Implied SD if independent", "Realised SD")}})
        st.caption(f"Test seasons. The team model's team-points RMSE on the same games is "
                   f"{co['team_model_rmse_team_points']:.2f}. Teammates compete for the same possessions, so their "
                   "points are negatively correlated; treating them as independent overstates team-score uncertainty. "
                   "In the simulation the team score equals the sum of its player points in every simulated game.")
    ab = summ.get("opponent_ablation")
    gbm_pts = next((r["mae"] for r in summ.get("players_test", []) if r["model"] == "gbm" and r["stat"] == "pts"), None)
    if ab and "pts" in ab:
        st.subheader("Matchups: do opponent features help?")
        rows = [{"Stat": k.upper(), "MAE with opponent": v["with_opponent"]["mae"],
                 "MAE without": v["without_opponent"]["mae"], "Gain": v["mae_gain"],
                 "95% CI": f"{v['mae_gain_ci95'][0]:+.4f} to {v['mae_gain_ci95'][1]:+.4f}"} for k, v in ab.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(format="%.4f") for c in
                                    ("MAE with opponent", "MAE without")} | {"Gain": st.column_config.NumberColumn(format="%+.4f")})
        st.caption("Gradient boosting retrained without the explicit opponent information (opponent style factors, "
                   "the EB expectation's opponent factors and the team-model environment, replaced by the team's own "
                   "season-to-date averages; projected minutes still reflect the team-model spread), then with it "
                   "added back; test seasons; game-clustered bootstrap. Both refits' features differ from the shipped "
                   "model's (no recent per-minute rates; the own-team averages are kept in both), so the with-opponent "
                   f"points MAE ({ab['pts']['with_opponent']['mae']:.3f}) differs slightly from the shipped "
                   "gradient-boosting model's test-season MAE" + (f" ({gbm_pts:.3f})." if gbm_pts is not None else "."))
    pa = summ.get("participation")
    if pa and "calibration" in pa:
        st.subheader("Uncertain participation (deep bench)")
        c_ = pa["calibration"]
        st.dataframe(pd.DataFrame({"Predicted P(play)": c_["pred"], "Observed": c_["obs"], "Player-games": c_["n"]}),
                     hide_index=True, use_container_width=True,
                     column_config={"Predicted P(play)": st.column_config.NumberColumn(format="%.2f"),
                                    "Observed": st.column_config.NumberColumn(format="%.2f")})
        st.caption(f"{pa['season']}: simulated participation probability of fringe candidates listed by the "
                   f"pregame rule vs who appeared ({pa['games']} games). Rotation players' participation is taken "
                   "from who appeared, the stand-in for the injury report (see the Data & methods page for what that "
                   "stand-in is worth).")
    cost = summ.get("cost", {})
    if "us_per_game" in cost:
        st.subheader("Computational cost")
        st.markdown(f"{cost['n_sims']:,} simulated games in {cost['simulate_s']:.2f} s "
                    f"({cost['us_per_game']:.0f} µs per game on 4 cores) plus {cost['calibration_s']:.2f} s of "
                    f"pilot calibration; every invariant held: **{cost['invariants'].get('all')}**.")
