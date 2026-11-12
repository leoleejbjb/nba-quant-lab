"""Render docs/REPORT.md from data/backtest/summary.json (written by `quantlab report`).

    .venv/bin/python scripts/docs/render_report.py [--summary PATH] [--check]

Numbers come from the summary, the model files (simulator constants, EB rate parameters) and the official box-score
comparison; the calibration-study figures of section 4 are fixed text documented in docs/METHODOLOGY.md. Every
qualitative claim the text makes about the numbers (which model is lowest, which differences are significant, ...) is
asserted, so a re-run whose results contradict the prose stops here instead of rendering a wrong report. ``--check``
runs everything without writing (``make docs`` runs both renderers' checks before anything is written;
scripts/make_report_page.py, which has no checks, runs last).
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from nums import ROOT, load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--summary", help="summary.json to render from (default data/backtest/summary.json)")
ap.add_argument("--check", action="store_true", help="compute and assert, write nothing")
ARGS = ap.parse_args()
N = load(ARGS.summary)
T = "test"
LBL = {"pts": "points", "reb": "rebounds", "ast": "assists", "fg3m": "threes", "pra": "PRA", "stl": "steals",
       "blk": "blocks", "tov": "turnovers", "fta": "free-throw attempts", "pf": "personal fouls"}
MODELS = [("season_avg", "Season average"), ("l10", "Last-10 average"), ("per_min", "Per-minute x projected minutes"),
          ("eb", "Empirical-Bayes rates (closed form)"), ("gbm", "Gradient boosting (LightGBM)"),
          ("topdown", "Team-first reconciliation"), ("sim_none", "Possession simulation, bottom-up"),
          ("sim_team", "Possession simulation, team-constrained"), ("sim_blend", "**Possession simulation, blend (shipped)**")]
SIMS = ("sim_none", "sim_team", "sim_blend")
NB = ("season_avg", "l10", "per_min", "eb", "gbm")


def v(model, stat, metric, split=T):
    return N[f"{split}.{model}.{stat}.{metric}"]


def tv(model, metric, split=T):
    return N[f"{split}.team.{model}.{metric}"]


def gtxt(name, split=T):
    m, lo, hi = N[f"{split}.gain.{name}"], N[f"{split}.gain.{name}.lo"], N[f"{split}.gain.{name}.hi"]
    return f"{m:+.2f} (95% CI {lo:+.2f} to {hi:+.2f})"


def is_sig(name, split=T):
    lo, hi = N[f"{split}.gain.{name}.lo"], N[f"{split}.gain.{name}.hi"]
    return lo > 0 or hi < 0


def sg(x, f="+.1f"):
    """Signed format without a negative zero."""
    s = format(x, f)
    return s.replace("-0.0", "0.0").replace("+0.0", "0.0") if abs(x) < 0.05 else s


def lowest(stat, metric, models):
    return min(models, key=lambda m: v(m, stat, metric))


def highest(stat, metric, models):
    return max(models, key=lambda m: v(m, stat, metric))


# ---------------------------------------------------------------- main table
cols = [("pts", "mae", "Points MAE", True), ("pts", "crps", "Points CRPS", True), ("pts", "pit_dev", "Points PIT error", True),
        ("pra", "crps", "PRA CRPS", True), ("pra", "cov80", "PRA 80% coverage", None)]
best = {}
for st, mt, _, low in cols:
    vals = [v(m, st, mt) for m, _ in MODELS]
    best[(st, mt)] = (min(vals) if low else None)
rows = []
for m, name in MODELS:
    cells = []
    for st, mt, _, low in cols:
        x = v(m, st, mt)
        txt = f"{100 * x:.1f}%" if mt == "cov80" else f"{x:.3f}"
        if best[(st, mt)] is not None and abs(x - best[(st, mt)]) < 1e-12:
            txt = f"**{txt}**"
        cells.append(txt)
    team = ["", ""]
    if m.startswith("sim_"):
        team = [f"{tv(m, 'margin_rmse'):.3f}", f"{tv(m, 'total_rmse'):.3f}"]
    rows.append(f"| {name} | " + " | ".join(cells + team) + " |")
FORECASTS = ("team_model",) + SIMS
sp_best = min(tv(m, "margin_rmse") for m in FORECASTS)
to_best = min(tv(m, "total_rmse") for m in FORECASTS)


def bold_if(x, b):
    return f"**{x:.3f}**" if abs(x - b) < 1e-12 else f"{x:.3f}"


for i, (m, name) in enumerate(MODELS):
    if m in SIMS:
        rows[i] = rows[i].replace(f"| {tv(m, 'margin_rmse'):.3f} | {tv(m, 'total_rmse'):.3f} |",
                                  f"| {bold_if(tv(m, 'margin_rmse'), sp_best)} | {bold_if(tv(m, 'total_rmse'), to_best)} |")
w = N["blend_weight"]
rows.append(f"| Team model (ratings) | | | | | | {bold_if(tv('team_model', 'margin_rmse'), sp_best)} | "
            f"{bold_if(tv('team_model', 'total_rmse'), to_best)} |")
POSTHOC = f"Post-hoc {w:.2f}/{1 - w:.2f} weighted average of team-model and bottom-up means"
rows.append(f"| {POSTHOC} (reference) | | | | | | "
            f"{tv('posthoc_blend', 'margin_rmse'):.3f} | {tv('posthoc_blend', 'total_rmse'):.3f} |")
table = "\n".join(["| | Points MAE | Points CRPS | Points calibration error (PIT) | PRA CRPS | PRA 80% coverage | "
                   "Spread RMSE | Total RMSE |", "|---|---:|---:|---:|---:|---:|---:|---:|"] + rows)
ph_m, ph_t = tv("posthoc_blend", "margin_rmse"), tv("posthoc_blend", "total_rmse")
ph_low = [lbl for lbl, x, b in (("spread", ph_m, sp_best), ("total", ph_t, to_best)) if x < b]
bold_note = ("Bold = lowest in the column before rounding (values that tie at the three decimals shown are effectively "
             "level; coverage is not ranked)" +
             (f"; the post-hoc reference row, lower on {' and '.join(ph_low)} RMSE "
              f"({', '.join(f'{x:.3f}' for lbl, x in (('spread', ph_m), ('total', ph_t)) if lbl in ph_low)}), is excluded."
              if ph_low else "; the post-hoc reference row is not a forecast and is not ranked."))

# ---------------------------------------------------------------- derived numbers
mae = {m: v(m, "pts", "mae") for m, _ in MODELS}
rmse = {m: v(m, "pts", "rmse") for m, _ in MODELS}
crps = {m: v(m, "pts", "crps") for m, _ in MODELS}
pra = {m: v(m, "pra", "crps") for m, _ in MODELS}
ALL = [m for m, _ in MODELS]
assert lowest("pts", "mae", ALL) == "gbm" and lowest("pts", "rmse", ALL) == "topdown"
assert lowest("pra", "rmse", ALL + []) in ("topdown",)
MODEL_BASED = ("eb", "gbm", "topdown") + SIMS
g_sa = [1 - mae[m] / mae["season_avg"] for m in MODEL_BASED]
g_l10 = [1 - mae[m] / mae["l10"] for m in MODEL_BASED]
assert min(g_sa) > 1 - mae["per_min"] / mae["season_avg"]          # per-minute scaling is the weakest model-based one

rel = {s: N.get(f"{T}.rel.{s}|sim_blend") for s in ("pts", "reb", "ast", "fg3m", "pra", "stl", "blk", "tov")}
rel_l = {s: N.get(f"{T}.rel.{s}|sim_blend|ladder") for s in rel}
line = lambda s, k: N[f"{T}.rel.{s}|sim_blend|ladder.line{k:+d}"]      # noqa: E731
over = sorted([s for s in rel if rel[s] > 1.0], key=lambda s: -rel[s])
calib = [s for s in rel if abs(rel[s]) <= 0.75]
assert set(over) == {"reb", "blk", "fg3m", "ast"} and set(calib) == {"pts", "pra", "stl", "tov"}
reb_lines = [line("reb", k) for k in (-2, -1, 0, 1, 2)]
assert min(reb_lines) > 0                                            # every rebounds line too high
assert abs(line("blk", 1)) < 1.0 and abs(line("blk", 2)) < 1.0 and line("blk", 0) > 2.0
assert line("ast", -2) < 0 < line("ast", 0) and line("ast", 2) < 0     # assists: gaps of opposite sign
assert line("fg3m", 1) < 0 and line("fg3m", 2) < 0 < line("fg3m", 0)
assert all(abs(rel_l[s]) < rel[s] for s in over)                      # the ladder's average gaps are smaller
assert min(v("sim_blend", s, "p_pit_below_half") for s in ("ast", "fg3m")) > 0.5    # medians too high
lv = lambda split, st, k="rel": N[f"lv.{split}.player.{st}.{k}"]          # noqa: E731
lt = lambda split, st, k="rel_per_poss": N[f"lv.{split}.team.{st}.{k}"]   # noqa: E731
fga2_pp, fgm2_pp = lt(T, "fga2"), lt(T, "fgm2")
assert 0.01 < fga2_pp < 0.03 and abs(fgm2_pp) < 0.005
blk_rel, blk_tune = lv(T, "blk"), lv("tuning", "blk")
assert blk_rel > 0.05 and abs(blk_tune) < 0.01
ot_part = N["lv.test.ot.total_bias_from_ot_shortfall"]
sig_m, sig_t = is_sig("margin_sim_blend"), is_sig("total_sim_blend")
assert not sig_m and not sig_t, "the team-forecast prose assumes neither difference is significant"
team_head, team_sig_txt = "level with the team model", "neither difference is significant"
assert N[f"{T}.gain.margin_sim_none.hi"] < 0 and N[f"{T}.gain.total_sim_none.hi"] < 0    # bottom-up: worse on both
assert N[f"{T}.gain.margin_posthoc_blend.lo"] > 0                                        # post-hoc: better spreads
tm_m, tm_t = tv("team_model", "margin_rmse"), tv("team_model", "total_rmse")
bl_m, bl_t = tv("sim_blend", "margin_rmse"), tv("sim_blend", "total_rmse")
ot_sim, ot_act = tv("sim_blend", "p_ot"), N[f"{T}.actual_ot"]
bias_bl, bias_tm = tv("sim_blend", "total_bias"), tv("team_model", "total_bias")
# overtime too rare (but up from 2.6% before the end-of-game logic), totals lower than the team model's, and the
# overtime part no larger than the gap
assert 0.026 < ot_sim < ot_act and ot_part < 0 and bias_bl < bias_tm and -ot_part <= bias_tm - bias_bl
# win probabilities level with the team model
assert abs(tv("sim_blend", "win_logloss") - tv("team_model", "win_logloss")) < 0.002
assert abs(tv("sim_blend", "win_brier") - tv("team_model", "win_brier")) < 0.001
# adding up independent players: worse than the team model, and too uncertain
assert N["cons.gbm.rmse_team_points_from_sum"] > N["cons.team_model_rmse"]
assert N["cons.gbm.implied_sd_if_independent"] > N["cons.gbm.realised_sd"]
# the inactive-list stand-in flatters both the minutes model and the closed-form EB points
assert N["ax.mae_leak"] > 0 and N["ax.eb_pts_mae_inactive_list"] > N["ax.eb_pts_mae_appearance_proxy"]
# every model-based approach beats both averages, and projected minutes are most of the gain
assert min(g_l10) > 0 and 1 - mae["per_min"] / mae["season_avg"] > 0.5 * max(g_sa)
m = {k: N[f"{T}.min.{k}.mae"] for k in ("season_avg", "l10", "ewma", "gbm_pre", "gbm", "gbm_norm", "sim_blend")}
r = {k: N[f"{T}.min.{k}.rmse"] for k in m}
n_min = N[f"{T}.min.gbm.n"]
n_min_sim = N[f"{T}.min.sim_blend.n"]
crps_best_pts = lowest("pts", "crps", ALL)
assert crps_best_pts == "topdown"
# minutes table: the lineups-known LightGBM is best (bolded) and forcing 240 minutes did not help
assert min(m, key=m.get) == "gbm" and min(r, key=r.get) == "gbm" and m["gbm_norm"] > m["gbm"]
NAME = {"gbm": "gradient boosting", "topdown": "the team-first reconciliation", "sim_blend": "the simulation",
        "eb": "the empirical-Bayes rates"}
# points: the simulation is best calibrated and has the best log score; CRPS is a near tie
NON_SIM = [k for k in ALL if k not in SIMS]
assert v("sim_blend", "pts", "pit_dev") < min(v(k, "pts", "pit_dev") for k in NON_SIM)
assert highest("pts", "logscore", ALL) == "sim_blend"
assert max(crps.values()) - min(crps[k] for k in ("gbm", "topdown", "sim_blend", "eb")) < 0.3 and \
    abs(crps["gbm"] - crps["sim_blend"]) < 0.01
assert pra["gbm"] < pra["sim_blend"]
# rebounds: CRPS ties, negative binomial better calibrated
assert abs(v("gbm", "reb", "crps") - v("sim_blend", "reb", "crps")) < 0.003
assert v("gbm", "reb", "pit_dev") < v("sim_blend", "reb", "pit_dev")
# steals / blocks / turnovers
assert lowest("stl", "crps", ALL) in SIMS and lowest("tov", "crps", ALL) in SIMS and lowest("blk", "crps", ALL) == "eb"
for s_ in ("stl", "blk", "tov"):
    assert highest(s_, "logscore", ALL) in NB
    assert max(v(k, s_, "crps") for k in ("eb", "gbm", "sim_blend")) - min(v(k, s_, "crps") for k in ALL) < 0.002
assert v("gbm", "stl", "pit_dev") < v("sim_blend", "stl", "pit_dev")
assert max(v("eb", "blk", "pit_dev"), v("gbm", "blk", "pit_dev")) < v("sim_blend", "blk", "pit_dev")
assert v("sim_blend", "tov", "pit_dev") < v("gbm", "tov", "pit_dev")
# gradient boosting's marginals: as good for points, better for PRA, rebounds, assists, threes
for s_ in ("pra", "reb", "ast", "fg3m"):
    assert v("gbm", s_, "crps") < v("sim_blend", s_, "crps") and v("gbm", s_, "logscore") > v("sim_blend", s_, "logscore")
part = list(zip(N["part.pred"], N["part.obs"], N["part.n"]))
# ablation: every statistic improves
abl_stats = ("pts", "reb", "ast", "fg3m", "stl", "blk", "tov")
assert all(N[f"abl.{s}.gain"] > 0 for s in abl_stats)
# the official box-score comparison (2023-24): residual differences
bc = pd.read_parquet(ROOT / "data/processed/2023-24/box_compare.parquet")
bc = bc[(bc["sec"].fillna(0) > 0) | (bc["box_seconds"].fillna(0) > 0)]          # build.quality()'s comparison set
COUNTING = ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf", "pts"]
n_resid = int((bc[[f"{s}_diff" for s in COUNTING]] != 0).any(axis=1).sum())
SIGNED = ["fga", "fg3a", "ftm", "fta", "reb", "ast", "stl", "blk", "tov", "pf", "pts"]   # derived - official
off_hi = sum(int((bc[f"{s}_diff"] < 0).sum()) for s in SIGNED)
off_lo = sum(int((bc[f"{s}_diff"] > 0).sum()) for s in SIGNED)
blk_hi, blk_n = int((bc["blk_diff"] < 0).sum()), int((bc["blk_diff"] != 0).sum())
assert off_hi > off_lo and 2 * blk_hi > blk_n                       # lean toward the official box being higher
# EB rate stability (data/models/rate_params.json, chosen on 2023-24)
rp = json.loads((ROOT / "data/models/rate_params.json").read_text(encoding="utf-8"))
SHOOT = {"fg3": "threes", "fg2": "twos", "ft": "free throws"}
assert max(SHOOT, key=lambda k: rp[k]["k"]) == "fg3"
FAST = {"usage": "usage", "fga3_share": "shot profile", "ast2_share": "assisted share of twos", "ast_rate": "assist rate"}
h_min = min(x["halflife"] for x in rp.values())
fast = [k for k, x in rp.items() if x["halflife"] == h_min]
assert set(fast) <= set(FAST) and {"usage", "fga3_share"} <= set(fast)
fast_names = list(dict.fromkeys(FAST[k] for k in fast))
fast_k = [rp[k]["k"] for k in fast]
q_range = [N[f"q.{s}.all_stats_exact"] for s in ("2020-21", "2021-22", "2022-23")]
# end of games: tied teams' final-four-second make rates, raw (2021-24, the selection in sim.constants.end_game:
# shots with <= 4 s left in the 4th quarter / overtime, score tied before the shot) and as the simulator uses them
TIED_RAW = (37 / 145, 24 / 127)
const = json.loads((ROOT / "data/models/sim_constants.json").read_text(encoding="utf-8"))
tied_sim = const["end_game"]["heave_fg"][const["end_game"]["lead_index"].index("tied")]
# known player-level biases (test seasons, shipped blend)
BIAS_STATS = ("blk", "reb", "fg3m", "fta", "pf", "stl", "ast")
hi = [s for s in BIAS_STATS if lv(T, s) > 0.005]
lo = [s for s in BIAS_STATS if lv(T, s) < -0.005]
assert set(hi) == {"blk", "reb", "fg3m"} and set(lo) == {"fta", "pf", "stl", "ast"}


def about(s):
    return f"about {abs(100 * lv(T, s)):.0f}%"


bias_hi = ", ".join(f"{LBL[s]} {about(s)} high" + (" (too many two-point attempts per possession)" if s == "reb" else "")
                    for s in sorted(hi, key=lambda s: -lv(T, s)))
bias_lo = ", ".join(f"{LBL[s]} {about(s)} low" for s in sorted(lo, key=lambda s: lv(T, s)))
tune_reb, tune_ast = lv("tuning", "reb"), lv("tuning", "ast")
tune_rest = max(abs(lv("tuning", s)) for s in ("pts", "fg3m", "fta", "pf", "stl", "blk", "tov"))
assert tune_rest < 0.012

report = f"""# NBA Quant Lab: player box-score forecasting and game simulation

**Question.** Can a possession-level simulation forecast every player's box score - minutes, points, rebounds,
assists, threes, steals, blocks, turnovers - as calibrated, internally consistent probability distributions, and does
modelling players improve the forecast of the game itself?

**Design.** Six regular seasons (2020-21 .. 2025-26, 7,230 games) reconstructed from play-by-play. History seasons
2020-23 give priors and training data; the tuned hyper-parameters, the simulator's calibration, its structural
constants (estimated on 2021-24) and the blend weight were chosen on **2023-24** (tuning); **2024-25 and 2025-26**
(test, {tv('team_model', 'games'):,.0f} games) were scored with those settings frozen, while the models are refit
walk-forward: each season's minutes model is trained on all earlier seasons (from 2021-22; the cold-start 2020-21 is
skipped), and the gradient-boosting player models (fixed hyper-parameters, no 2023-24 forecasts) are trained on
2023-24, plus 2024-25 for 2025-26. All forecasts are walk-forward: player and team histories use only games on earlier
dates. Teammate availability, however, is taken from who appeared in the game itself (the stand-in for the inactive
list, which also reveals coach's-decision DNPs), and today's starting fives (public about 30 minutes before tip-off)
are used by the shipped minutes model, the gradient-boosting player models and the simulator's starting lineups.
Measured on 2023-24, the stand-in flatters minutes MAE by {N['ax.mae_leak']:.3f} and closed-form EB points MAE by
{N['ax.eb_pts_mae_inactive_list'] - N['ax.eb_pts_mae_appearance_proxy']:.3f} (section 2). Numbers below come from
data/backtest/summary.json (tabulated in [FINDINGS_DATA.md](FINDINGS_DATA.md); both regenerated by `quantlab
report`), the model files and the official box-score comparison, rendered by `scripts/docs/render_report.py`; the
study figures in sections 2 and 4 (the fringe-participation leak, usage-exponent sweep, usage-efficiency regression,
score SDs, end-of-game rates) and the review notes' descriptions of earlier versions are fixed text from the studies
documented in [METHODOLOGY.md](METHODOLOGY.md). The first version of this
report was reviewed adversarially and several of its conclusions did not survive; the corrections are listed at the
end, and every number here is the corrected one.

## Results in one table

Test seasons 2024-25 and 2025-26, every model scored on the same {N['n_players_test']:,} player-games (conditional on
the player appearing) and {tv('team_model', 'games'):,.0f} games. Lower is better except coverage (80.0% = calibrated:
the share of outcomes in the central 80% of the forecast distribution, ties randomized). Distributions for the
season-average, last-10, per-minute, EB and gradient-boosting models are negative binomials around their point
forecasts, with a dispersion that depends on the predicted mean (fitted on 2023-24; gradient boosting, which has no
2023-24 forecasts, uses the EB model's curve); the team-first reconciliation, like the simulations, is scored from
1,200 simulated draws of each game.

{table}

{bold_note}

**What this says.**

1. **Point forecasts: gradient boosting has the lowest MAE (the team-first reconciliation the lowest RMSE,
   {rmse['topdown']:.3f} vs {rmse['gbm']:.3f}); the rest are close.** Points MAE {mae['gbm']:.3f} for gradient boosting,
   {mae['topdown']:.3f} for the team-first reconciliation, {mae['sim_blend']:.3f} for the shipped simulation and
   {mae['eb']:.3f} for closed-form empirical-Bayes rates. The EB rates, gradient boosting, the team-first reconciliation
   and the three simulations beat the season average by {100 * min(g_sa):.1f}-{100 * max(g_sa):.1f}% and the last-10
   average by {100 * min(g_l10):.1f}-{100 * max(g_l10):.1f}%, and most of that gain is projected minutes (per-minute rates
   x projected minutes alone reach {mae['per_min']:.3f}, {100 * (1 - mae['per_min'] / mae['season_avg']):.1f}% better than
   the season average).
2. **Distributions: a contest decided statistic by statistic.** With a mean-dependent dispersion the negative-binomial
   baselines are strong. The simulation is still the best calibrated for points (PIT error
   {v('sim_blend', 'pts', 'pit_dev'):.3f} vs {v('gbm', 'pts', 'pit_dev'):.3f}: around the same mean, a negative binomial
   does not follow the lumpy points distribution of 2s, 3s, free throws and minutes risk as well) and has the best points
   log score ({v('sim_blend', 'pts', 'logscore'):.3f} vs {v('gbm', 'pts', 'logscore'):.3f}), but on CRPS points are a near
   tie ({NAME[crps_best_pts]} is lowest at {crps[crps_best_pts]:.3f}; gradient boosting {crps['gbm']:.3f}, simulation
   {crps['sim_blend']:.3f}) and PRA favours gradient boosting ({pra['gbm']:.3f} vs {pra['sim_blend']:.3f}; 80% coverage
   {100 * v('gbm', 'pra', 'cov80'):.1f}% vs {100 * v('sim_blend', 'pra', 'cov80'):.1f}%). For rebounds CRPS ties and the
   negative binomial is better calibrated ({v('gbm', 'reb', 'pit_dev'):.3f} vs {v('sim_blend', 'reb', 'pit_dev'):.3f}).
   For steals, blocks and turnovers CRPS is a near tie (a simulation is lowest for steals and turnovers, the EB rates
   for blocks), but the negative binomials have the better log score for all three and are better calibrated for steals
   (gradient boosting: PIT error {v('gbm', 'stl', 'pit_dev'):.3f} vs {v('sim_blend', 'stl', 'pit_dev'):.3f}) and blocks
   ({min(v('eb', 'blk', 'pit_dev'), v('gbm', 'blk', 'pit_dev')):.3f}-{max(v('eb', 'blk', 'pit_dev'), v('gbm', 'blk', 'pit_dev')):.3f}
   vs {v('sim_blend', 'blk', 'pit_dev'):.3f}); the simulation is better calibrated only for turnovers
   ({v('sim_blend', 'tov', 'pit_dev'):.3f} vs {v('gbm', 'tov', 'pit_dev'):.3f}).
3. **Over/under probabilities.** At the half-point line nearest the projection, simulated P(over) is within 0.75
   percentage points of the observed over rate for {', '.join(LBL[s] for s in calib)}, and too high for
   {', '.join(f"{LBL[s]} ({rel[s]:+.1f})" for s in over)}. Over a ladder of five lines (nearest -2 to +2) the average gaps
   are smaller ({', '.join(f"{LBL[s]} {rel_l[s]:+.1f}" for s in over)}), mostly because the over-prediction shrinks at
   the higher lines: for rebounds every line is too high ({min(reb_lines):+.1f} to {max(reb_lines):+.1f}), and for blocks
   the +1 and +2 lines are about right ({sg(line('blk', 1))}, {sg(line('blk', 2))}); only for assists, and slightly for
   threes, do gaps of opposite sign cancel. Rebounds run high because the simulator takes about
   {100 * fga2_pp:.0f}% too many two-point attempts per possession (makes match), and blocks because the simulated block
   rate is about {100 * blk_rel:.0f}% too high in the test seasons ({lv(T, 'blk', 'pred'):.3f} vs
   {lv(T, 'blk', 'actual'):.3f} per player-game; unbiased in 2023-24, where the blocks gap is only
   {N['tuning.rel.blk|sim_blend']:+.1f}); for assists and threes the simulated distributions are not right-skewed enough
   (their medians are too high: the share of outcomes below the simulated median is
   {v('sim_blend', 'ast', 'p_pit_below_half'):.3f} and {v('sim_blend', 'fg3m', 'p_pit_below_half'):.3f}).
4. **Team forecasts: {team_head}.** The shipped simulation (blend environment: target efficiency =
   {w:.2f} x team model + {1 - w:.2f} x bottom-up, weight chosen on 2023-24, backtested as shipped) has spread RMSE
   {bl_m:.3f} against {tm_m:.3f} for the team model - squared-error gain {gtxt('margin_sim_blend')} - and total RMSE
   {bl_t:.3f} against {tm_t:.3f} (gain {gtxt('total_sim_blend')}); {team_sig_txt}. Win probabilities are level
   (log-loss {tv('sim_blend', 'win_logloss'):.4f} vs {tv('team_model', 'win_logloss'):.4f}, Brier
   {tv('sim_blend', 'win_brier'):.4f} vs {tv('team_model', 'win_brier'):.4f}). The simulation's totals average
   {bias_bl:+.2f} points against {bias_tm:+.2f} for the team model; about {abs(ot_part):.1f} points of the gap is overtime,
   which it plays too rarely ({100 * ot_sim:.1f}% of games against {100 * ot_act:.1f}%). The bottom-up simulation alone is
   significantly worse than the team model on both (spread gain {gtxt('margin_sim_none')}, total {gtxt('total_sim_none')}).
   The post-hoc {w:.2f}/{1 - w:.2f} weighted average of team-model and bottom-up means - the number the first version of
   this report quoted as "the blend" - improves spreads significantly (gain {gtxt('margin_posthoc_blend')}), so
   player-level information carries some signal the team model lacks; but it is not a joint simulation and not what the
   product runs, and the shipped simulator does not turn that signal into a significantly better score forecast. (A
   first version of this backtest, before the simulator's possessions were calibrated to the team model's pace, showed a
   significant spread gain; most of it came from the possession shortfall shrinking every simulated margin toward zero.)
5. **Architecture.** We keep the *hierarchical* design as the product - the team model sets pace and (with weight
   {w:.2f}) efficiency, the possession simulator allocates opportunities to lineups and players, and team scores are the
   sum of player scores. It is the only approach here whose every simulated game is a fully consistent box score
   (rebounds reconcile with misses, plus-minus sums, no ties, minutes include overtime) and that supports injury and
   lineup scenarios mechanistically (the team-first reconciliation also prices combinations from joint draws whose
   points add up, but without these identities, lineups or game dynamics), and it matches the team model on spreads and
   totals; it costs {N['cost.us']:.0f} µs per simulated game on 4 cores ({N['cost.sim_s']:.1f} s for
   {N['cost.n']:,}, plus {N['cost.cal_s']:.2f} s of pilot calibration per projection). For single-statistic marginals it
   is not the best model: a gradient-boosting point forecast with a mean-dependent negative binomial is as good for
   points and better for PRA, rebounds, assists and threes (and better calibrated for steals and blocks), so a
   production system should use such marginals where they win and the simulation for everything joint, while the
   simulation remains the better-calibrated source for points.

## 1. The data foundation is exact enough to model from

Player box scores derived from the play-by-play match NBA.com's official box scores: for 2023-24, minutes within one
second for {100 * N['q.minutes_within_1s']:.2f}% of player-games, points exactly for {100 * N['q.pts_exact']:.2f}%,
plus-minus {100 * N['q.plus_minus_exact']:.2f}%, every counting stat exactly for {100 * N['q.all_stats_exact']:.2f}% (the
residual differences, {n_resid} player-games and mostly single steals, rebounds, turnovers, field-goal attempts and
blocks, go both ways but lean toward the official box being higher: {off_hi} of {off_hi + off_lo} stat differences,
{blk_hi} of {blk_n} for blocks. This is consistent with post-game stat corrections the archived feed lacks). In 2020-23
every counting stat matches for {100 * min(q_range):.1f}-{100 * max(q_range):.1f}% of player-games; 2024-25 and 2025-26
have no official box scores in the sources to check against. The reconstruction also yields what box scores cannot:
exact on-court team totals (true usage rate), live rebound chances, five-man stints and possessions with timestamps and
the score at their start. Zero-minute rows of people charged only with a bench technical and absent from that game's
official box score (coaches, staff, players charged from the bench in games they sat out) are not player-games. See
[DATA.md](DATA.md).

## 2. Minutes: the hard part, and where most of the gain comes from

| Minutes model (test seasons, {n_min:,} player-games; simulation row {n_min_sim:,}) | MAE | RMSE |
|---|---:|---:|
| Season average | {m['season_avg']:.2f} | {r['season_avg']:.2f} |
| Last 10 games | {m['l10']:.2f} | {r['l10']:.2f} |
| Decayed average | {m['ewma']:.2f} | {r['ewma']:.2f} |
| LightGBM, no lineup information | {m['gbm_pre']:.2f} | {r['gbm_pre']:.2f} |
| **LightGBM, starting lineups known** | **{m['gbm']:.2f}** | **{r['gbm']:.2f}** |
| Simulation, shipped (enforces 240 minutes, rotations, fouls, blowouts, overtime) | {m['sim_blend']:.2f} | {r['sim_blend']:.2f} |

* The gradient-boosting model is {100 * (1 - m['gbm'] / m['ewma']):.0f}% better than a decayed average and
  {100 * (1 - m['gbm'] / m['season_avg']):.0f}% better than a season average. Knowing the starting lineups is worth
  {m['gbm_pre'] - m['gbm']:.2f} minutes of MAE (measured with a rotation rule that ignores today's starters in the
  "no lineup" model; the first version's "no lineup" model saw them).
* **Two participation leaks.** Using everyone who appeared to describe teammate availability leaks the outcome:
  deep-bench players appear because a game became a blowout. Treating only rotation players' participation as known
  pregame and fringe players through their baseline minutes removed that leak (worth 0.07 minutes of MAE). The second
  cannot be removed for the test seasons: who is *available* (the injury report / inactive list, public before tip-off)
  is in the sources only for 2020-24, so every season uses who appeared as its stand-in, and that also reveals rotation
  players a coach chose not to use ({100 * N['ax.dnp_share']:.1f}% of the "absent" rotation players in 2021-24 were
  dressed but unused, more often in close games). Measured on 2023-24 with the official statuses, the minutes model's
  MAE is {N['ax.mae_appearance_proxy']:.3f} with the stand-in and {N['ax.mae_inactive_list']:.3f} with the inactive
  list ({N['ax.mae_leak']:+.3f}); carried through the closed-form EB projection, points MAE goes from
  {N['ax.eb_pts_mae_appearance_proxy']:.3f} to {N['ax.eb_pts_mae_inactive_list']:.3f}
  ({N['ax.eb_pts_mae_inactive_list'] - N['ax.eb_pts_mae_appearance_proxy']:+.3f}) and PRA from
  {N['ax.eb_pra_mae_appearance_proxy']:.3f} to {N['ax.eb_pra_mae_inactive_list']:.3f}. The test numbers of every model
  that uses projected minutes presumably carry about this much optimism (measured only for the minutes model and the
  closed-form EB projection; the gradient-boosting player models and the simulator, which read the same context, were
  not measured separately); season and last-10 averages are not affected.
* **Forcing 240 minutes per team did not help** a model that already sees teammate absences (MAE
  {m['gbm_norm']:.2f} vs {m['gbm']:.2f}); the simulator enforces the constraint instead, with five players on the floor
  at every moment.

## 3. Player rates and matchups

Empirical-Bayes rates with per-statistic decay and shrinkage (chosen on 2023-24) reproduce known stability: among the
shooting percentages, three-point percentage needs the most shrinkage (a prior worth {rp['fg3']['k']} attempts, against
{rp['fg2']['k']} for twos and {rp['ft']['k']} for free throws; half-life {rp['fg3']['halflife']} days), while
{', '.join(fast_names[:-1]) + ' and ' + fast_names[-1]} follow recent games most closely (half-life {h_min} days, priors
worth {min(fast_k)}-{max(fast_k)} observations). Opponent effects are real but small. In a nested ablation of the
gradient-boosting player models - without the explicit opponent information (the team's own season-to-date averages in
place of its team-model environment, the closed-form expectation with neutral opponent factors at its own pace) against
the same model plus that information (the game's team-model environment, the opponent's defensive style factors) -
opponent information improves every statistic, by
{min(N[f'abl.{s}.pct'] for s in abl_stats):.2f}-{max(N[f'abl.{s}.pct'] for s in abl_stats):.2f}% of MAE: points
{N['abl.pts.gain']:+.4f} ({N['abl.pts.pct']:.2f}%, 95% CI {N['abl.pts.lo']:+.4f} to {N['abl.pts.hi']:+.4f}), turnovers
{N['abl.tov.gain']:+.4f} ({N['abl.tov.pct']:.2f}%), blocks {N['abl.blk.gain']:+.4f} ({N['abl.blk.pct']:.2f}%), steals
{N['abl.stl.gain']:+.4f} ({N['abl.stl.pct']:.2f}%, CI {N['abl.stl.lo']:+.4f} to {N['abl.stl.hi']:+.4f}). The gain also
contains what the team model's current own-team ratings know beyond the lagged averages (small after the first weeks).
Both arms are refits whose features differ from the shipped model's (no recent per-minute rates; the own-team averages
and the neutral-opponent expectation are kept in both), so the with-opponent points MAE is {N['abl.pts.with']:.3f}
against {mae['gbm']:.3f} for the shipped model; and both keep the projected minutes, whose blowout-risk features come
from the team model's expected margin against this opponent, so a little opponent information remains in the
"without" model.

## 4. Empirical regularities the simulator needed

* **Usage is super-linear.** When the simulator shares a lineup's plays in proportion to usage^alpha, points MAE on
  tuning games was best at alpha = 1.3 (4.571 / 4.440 / 4.406 / 4.457 / 4.588 for alpha = 0.7 / 1.0 / 1.3 / 1.6 / 1.9):
  the offense runs through its highest-usage players more than proportionally.
* **The usage-efficiency tradeoff.** With teammate absences as a natural experiment (59,494 rotation player-games,
  2021-24), each +10% of relative usage costs a player 0.88 points of true shooting (SE 0.20). The naive regression
  has the opposite sign (+0.20 per +10%), because hot shooters both shoot more and shoot better. Without this
  tradeoff, removing a star *raised* his team's simulated scoring (his shots went to teammates at their usual
  efficiency).
* **Real scores are less variable than independent possessions imply.** Independent possessions give a team-points
  SD of about 12.8; a good pregame model's residual SD is about 11.5 *including* its own error. The simulator therefore
  has two small mean-reversion terms (efficiency drifts against a leading team and against a team running hot), and
  the team-level shooting "night" that seemed natural was calibrated to zero.
* **The end of games is its own regime** (all rates from 2021-24): leading teams run the clock and are fouled (in the
  final 24 seconds on 77% of possessions when ahead by 1-3, counting the possessions in which a team simply dribbles out
  the clock), trailing teams hurry and shoot threes, tied teams hold for the last shot, and last-second shots go in far
  less often (tied teams make {100 * TIED_RAW[0]:.0f}% of twos and {100 * TIED_RAW[1]:.0f}% of threes in the final four
  seconds; the simulator uses {100 * tied_sim[0]:.0f}% and {100 * tied_sim[1]:.0f}% after shrinking toward the pooled
  final-seconds rates). A tied team with the ball and 24 seconds or less goes to overtime 72.6% of the time in real
  games and 71.6% in the simulation.

## 5. Why model the game jointly

Adding up independently projected players does not give a sensible team forecast (test seasons,
{N['cons.gbm.team_games']:,} team-games): summing gradient-boosting point projections misses the team model's
expectation by {N['cons.gbm.mae_gap_vs_team_model']:.1f} points on average and predicts the actual team score worse than the
team model does (RMSE {N['cons.gbm.rmse_team_points_from_sum']:.1f} vs {N['cons.team_model_rmse']:.1f}); treating players as
independent implies a team-score SD of {N['cons.gbm.implied_sd_if_independent']:.1f} points against a realised
{N['cons.gbm.realised_sd']:.1f} (80% intervals cover {100 * N['cons.gbm.coverage80_if_independent']:.0f}%). Teammates
compete for the same possessions, so their points are negatively correlated. In the simulation every team score is the
sum of its players' points, every player's points are 2·2PM + 3·3PM + FTM, minutes add to five times the game length,
assists never exceed made field goals and rebounds reconcile with misses - checked on every simulated game of every
backtest, which stops if one fails.

"""

review = f"""## What an adversarial review changed

An independent review of the first version (25 findings, 23 confirmed) found errors that flattered some results.
All but one were fixed, overtime only partly (see Limitations); the exception, the coach's-decision participation
leak, cannot be fixed for 2024-26 and is measured on 2023-24 instead. Everything was re-run; the numbers above are the
corrected ones.

* **Baseline distributions were too weak.** The negative-binomial baselines used one dispersion for every player;
  the true dispersion varies ~15x with the mean. With a mean-dependent dispersion (fitted on 2023-24) the
  gradient-boosting baseline's distributions improve sharply, and the earlier claim that "a negative binomial cannot
  represent" points and PRA distributions was withdrawn for PRA (the negative binomial now has the lower PRA CRPS,
  {pra['gbm']:.3f} vs {pra['sim_blend']:.3f}) and narrowed for points to calibration (PIT error
  {v('gbm', 'pts', 'pit_dev'):.3f} vs {v('sim_blend', 'pts', 'pit_dev'):.3f} for the simulation; CRPS a near tie,
  {crps['gbm']:.3f} vs {crps['sim_blend']:.3f}).
* **The team-first baseline knew who played.** It split minutes and points over the players who actually appeared
  (garbage-time appearances reveal blowouts). It now uses the simulator's pregame roster with sampled participation.
* **The reported blend was not the shipped forecast.** It was a post-hoc average of team-model and simulation means;
  the dashboard and CLI run a different (simulator) blend. The blend environment is now backtested as shipped.
* **Over/under lines and reliability.** Lines were placed at round(mean) - 0.5 instead of the nearest half point, and
  "reliable" overstated the evidence; lines are now the nearest half point plus a ladder of four more, and the
  over-predictions are quantified.
* **Simulator bugs.** A questionable player's minutes plan was pushed to the 48-minute cap during calibration;
  scenario edits for deep-bench players were silently rescaled; fringe minutes jumped at a budget threshold;
  overtime happened about half as often as in reality; players fouled out twice as often. All fixed (overtime only
  partly, see Limitations).
* **Information leaks, measured.** Rotation players' coach's-decision DNPs were treated as known pregame; the "no
  lineup" minutes model saw today's starters; the deep-bench participation probabilities used a formula chosen by
  the outcome. The last two are fixed; the first cannot be fixed for 2024-26 (no inactive list in the sources) and is
  measured on 2023-24 instead.
* **Smaller corrections.** Late-game foul rates were bucketed by a lead that included the possession's own points;
  coaches with bench technicals were counted as player-games; the matchup ablation left most opponent information in
  the "without opponent" model; scenario re-projection for 2020-22 used a model trained on later seasons; several
  documentation figures (injury rate, speed, constants seasons) did not match the code.
* **A second, independent check of the fixes** (ten reviewers, one per group of findings) confirmed them and found
  further minor issues, also fixed before the final run: the final-24-second foul rates left out possessions in
  which a team dribbles out the clock (no recorded event), late-game possession lengths counted offensive-rebound
  time twice, the deep-bench candidates' conditional minutes came from a cruder estimate than the minutes model, the
  simulator's calibration fitted on a 200-game sample did not hold on the full tuning season, its possessions ran
  ~0.5% short of the team model's pace, and the matchup ablation was not nested.
* **A third check of the documents** against the code and the data corrected 75 statements (wording that overstated
  what was pregame, wrong or imprecise figures, and descriptions of the evaluation, the code and the dashboard that did
  not match them); no forecast or accuracy result changed (only the speed benchmark, re-timed when the summary was
  regenerated, moved from 66 to {N['cost.us']:.0f} µs per simulated game).
"""

RH = json.loads((ROOT / "docs" / "pregame_rehearsal.json").read_text(encoding="utf-8"))
RM = RH["mae"]
assert all(RM[k]["pregame"] > RM[k]["asof"] for k in ("min", "pts", "pra")), "rehearsal: pregame no longer worse"

limitations = f"""
## Limitations

* **The current season is projected, not evaluated.** 2026-27 is fetched from the NBA's live CDN by
  `quantlab live-update` (the CDN refused this build environment, so the fetch itself is untested here) and its
  unplayed games are projected with pregame information only: projected starters, availability from missed-game
  streaks (there is no injury feed; the user enters the injury report). Rehearsed on 2025-26 with the games from
  {RH['cutoff']} on hidden ({RH['games']} games in the following week), minutes MAE was {RM['min']['pregame']:.2f}
  against {RM['min']['asof']:.2f} for the backtest's projections of the same games, points {RM['pts']['pregame']:.2f}
  against {RM['pts']['asof']:.2f} and PRA {RM['pra']['pregame']:.2f} against {RM['pra']['asof']:.2f}, on the
  {RH['common_player_games']:,} player-games both projected ({RH['not_projected']['player_games']} more appearances,
  {100 * RH['not_projected']['minutes_share']:.1f}% of the minutes, were by players the pregame projection left out):
  the results above are an optimistic bound for a real slate (METHODOLOGY section 11, `docs/pregame_rehearsal.json`).
* **Availability comes from who appeared** for rotation players (the inactive list is not in the sources for the test
  seasons); measured on 2023-24, this flatters minutes by {N['ax.mae_leak']:.3f} MAE and closed-form EB points by
  {N['ax.eb_pts_mae_inactive_list'] - N['ax.eb_pts_mae_appearance_proxy']:.3f} (section 2).
* **Overtime is still too rare**: {100 * ot_sim:.1f}% of simulated games against {100 * ot_act:.1f}% in the test seasons (up
  from 2.6% before the end-of-game logic). Regulation ties from one-to-three-point games in the last minute remain
  under-represented.
* **Known player-level biases** (test seasons, per player-game): {bias_hi}; {bias_lo}. On 2023-24 rebounds are
  {100 * tune_reb:+.0f}%, assists {100 * tune_ast:+.0f}% and the rest within about 1%. Over probabilities for
  {', '.join(LBL[s] for s in over)} are {min(rel[s] for s in over):.1f}-{max(rel[s] for s in over):.1f} points too high
  near the projection ("What this says", 3). Starters' points are slightly under-projected and bench players' slightly
  over-projected by the usage allocation. Level checks for every statistic: FINDINGS_DATA.md.
* **Deep-bench participation is hard to predict.** For fringe candidates listed by the pregame rule
  ({N['part.rows']:,} player-games, 2023-24 sample), the simulated participation probabilities
  {', '.join(f'{p:.2f}' for p, _, _ in part)} correspond to observed appearance rates
  {', '.join(f'{o:.2f}' for _, o, _ in part)}: ordered correctly but compressed toward the middle. Fringe-player props
  should be treated with caution.
* **Defense beyond the box score is not modelled at the player level.** A player's defensive value enters through
  steals, blocks, rebounding and fouls, plus team-level style factors.
* **Matchup-driven substitutions and coaching changes** are not modelled; rotations follow each player's role,
  minutes plan, fouls, score and time.
* **Single-season tuning.** The simulator's calibration was fitted on one season (2023-24) and held fixed; drifts in
  league style are absorbed by the decayed league priors, not by re-tuning (the test-season block, free-throw and foul
  biases above, absent in 2023-24, show the limits of that).
* **Prop probabilities are model probabilities**, without market prices; no claim is made about beating a sportsbook.
"""

if ARGS.check:
    print("docs/REPORT.md: every check passed (not written)")
else:
    (ROOT / "docs" / "REPORT.md").write_text(report + review + limitations, encoding="utf-8", newline="\n")
    print("docs/REPORT.md rendered", len(report + review + limitations))
