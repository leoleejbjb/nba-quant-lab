"""Render the README's results block and docs/headline.json (the report page's headline bullets) from
data/backtest/summary.json (written by `quantlab report`).

    .venv/bin/python scripts/docs/render_docs.py [--summary PATH] [--check]

Qualitative claims made about the numbers are asserted, so a re-run that contradicts the prose fails here;
``--check`` runs everything without writing (``make docs`` runs both renderers' checks before anything is written;
scripts/make_report_page.py, which has no checks, runs last).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nums import ROOT, load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--summary", help="summary.json to render from (default data/backtest/summary.json)")
ap.add_argument("--check", action="store_true", help="compute and assert, write nothing")
ARGS = ap.parse_args()
SUMMARY = Path(ARGS.summary) if ARGS.summary else ROOT / "data/backtest/summary.json"
N = load(SUMMARY)
T = "test"


def g(key, f=".3f"):
    return format(N[key], f)


def P(model, stat, metric, split=T, f=".3f"):
    return format(N[f"{split}.{model}.{stat}.{metric}"], f)


def pct(x, f=".1f"):
    return format(100 * x, f) + "%"


def better(a, b, lower=True):
    return (a < b) if lower else (a > b)


def team(model, metric, split=T, f=".3f"):
    return format(N[f"{split}.team.{model}.{metric}"], f)


def gain(name, split=T):
    m, lo, hi = N[f"{split}.gain.{name}"], N[f"{split}.gain.{name}.lo"], N[f"{split}.gain.{name}.hi"]
    return m, lo, hi


def gtxt(name, split=T):
    m, lo, hi = gain(name, split)
    return f"{m:+.2f} (95% CI {lo:+.2f} to {hi:+.2f})"


def sig(name, split=T):
    m, lo, hi = gain(name, split)
    return "significant" if (lo > 0 or hi < 0) else "not significant"


# ---------------------------------------------------------------- derived numbers
mae = {m: N[f"{T}.{m}.pts.mae"] for m in ("season_avg", "l10", "per_min", "opp_adj", "eb", "gbm", "topdown", "sim_none",
                                         "sim_team", "sim_blend")}
best_pts_mae = min(mae, key=mae.get)
crps_pts = {m: N[f"{T}.{m}.pts.crps"] for m in mae}
crps_pra = {m: N[f"{T}.{m}.pra.crps"] for m in mae}
gain_vs_sa = 1 - mae["gbm"] / mae["season_avg"]
sim_gain_vs_sa = 1 - mae["sim_blend"] / mae["season_avg"]
m_gbm, m_sa = N[f"{T}.min.gbm.mae"], N[f"{T}.min.season_avg.mae"]
m_ewma, m_pre = N[f"{T}.min.ewma.mae"], N[f"{T}.min.gbm_pre.mae"]
leak = N["ax.mae_leak"]
corrected_gbm = m_gbm + leak
rel = {s: N.get(f"{T}.rel.{s}|sim_blend") for s in ("pts", "reb", "ast", "fg3m", "pra", "stl", "blk", "tov")}
over_stats = [s for s, v in rel.items() if v is not None and v > 1.0]
calib_stats = [s for s, v in rel.items() if v is not None and abs(v) <= 0.75]
LBL = {"pts": "points", "reb": "rebounds", "ast": "assists", "fg3m": "threes", "pra": "PRA", "stl": "steals",
       "blk": "blocks", "tov": "turnovers"}
cons = N["cons.gbm.implied_sd_if_independent"] / N["cons.gbm.realised_sd"] - 1
abl_pcts = [N[f"abl.{s}.pct"] for s in ("pts", "reb", "ast", "fg3m", "stl", "blk", "tov")]
abl_sig = [s for s in ("pts", "reb", "ast", "fg3m", "stl", "blk", "tov") if N[f"abl.{s}.lo"] > 0]
w = N["blend_weight"]
tm_m, tm_t = N[f"{T}.team.team_model.margin_rmse"], N[f"{T}.team.team_model.total_rmse"]
bl_m, bl_t = N[f"{T}.team.sim_blend.margin_rmse"], N[f"{T}.team.sim_blend.total_rmse"]
ph_m, ph_t = N[f"{T}.team.posthoc_blend.margin_rmse"], N[f"{T}.team.posthoc_blend.total_rmse"]
spread_word = "improves" if bl_m < tm_m else "does not improve"
sig_m = N[f"{T}.gain.margin_sim_blend.lo"] > 0 or N[f"{T}.gain.margin_sim_blend.hi"] < 0
sig_t = N[f"{T}.gain.total_sim_blend.lo"] > 0 or N[f"{T}.gain.total_sim_blend.hi"] < 0
assert not sig_m and not sig_t, "the team-forecast prose assumes neither difference is significant"
team_head, team_sig_txt = "level with the team model", "neither difference is significant"
assert N[f"{T}.gain.margin_sim_none.hi"] < 0 and N[f"{T}.gain.total_sim_none.hi"] < 0    # bottom-up: worse on both
assert N[f"{T}.gain.margin_posthoc_blend.lo"] > 0                                        # post-hoc: better spreads
assert cons > 0 and min(abl_pcts) > 0                  # independence overstates uncertainty; opponent info helps all
total_cmp = ("slightly better" if bl_t < tm_t else "no better") + f" ({bl_t:.3f} vs {tm_t:.3f}; squared-error gain {gtxt('total_sim_blend')})"
ot_sim, ot_act = N[f"{T}.team.sim_blend.p_ot"], N[f"{T}.actual_ot"]
part_pred, part_obs = N["part.pred"], N["part.obs"]
rmse = {m: N[f"{T}.{m}.pts.rmse"] for m in mae}
rmse_pra = {m: N[f"{T}.{m}.pra.rmse"] for m in mae}
assert best_pts_mae == "gbm" and min(rmse, key=rmse.get) == "topdown" and min(rmse_pra, key=rmse_pra.get) == "topdown"
mae_wins = [s for s in LBL if min((N[f"{T}.{m}.{s}.mae"], m) for m in mae)[1] == "gbm"]
assert len(mae_wins) == 6 and "stl" not in mae_wins and "blk" not in mae_wins
q_old = [N[f"q.{s}.all_stats_exact"] for s in ("2020-21", "2021-22", "2022-23")]
q_all = q_old + [N["q.all_stats_exact"]]
m_pre_gain = 1 - m_pre / m_sa
summ = json.loads(SUMMARY.read_text(encoding="utf-8"))
# points: the shipped simulation is the best calibrated and has the best log score; CRPS a near tie; PRA favours GBM
pit_pts = {m: N[f"{T}.{m}.pts.pit_dev"] for m in mae}
log_pts = {m: N[f"{T}.{m}.pts.logscore"] for m in mae}
assert min(pit_pts, key=pit_pts.get) == "sim_blend" and max(log_pts, key=log_pts.get) == "sim_blend"
assert abs(crps_pts["gbm"] - crps_pts["sim_blend"]) < 0.01 and crps_pra["gbm"] < crps_pra["sim_blend"]
# minutes: lineups-known < no-lineup < season average; the stand-in flatters; projected minutes are most of the gain
assert m_gbm < m_pre < m_sa and leak > 0
assert (mae["season_avg"] - mae["per_min"]) > 0.5 * (mae["season_avg"] - mae["gbm"])


def decile_max_gap(stat):
    r = summ["reliability_test"][f"{stat}|sim_blend"]
    return max(100 * (p - o) for p, o in zip(r["pred"], r["obs"]))


assert set(calib_stats) == {"pts", "pra", "stl", "tov"} and set(over_stats) == {"reb", "blk", "fg3m", "ast"}


def stat_list(stats, key="v"):
    stats = sorted(stats, key=lambda s: -rel[s])
    return ", ".join(f"{LBL[s]} {rel[s]:+.1f}" for s in stats)


crps_order = sorted(["gbm", "topdown", "sim_blend", "eb"], key=lambda m: crps_pts[m])
MODEL = {"gbm": "gradient boosting", "topdown": "team-first", "sim_blend": "simulation", "eb": "empirical-Bayes rates"}
pts_crps_txt = ", ".join(f"{MODEL[m]} {crps_pts[m]:.3f}" for m in crps_order)


# ---------------------------------------------------------------- README results block
readme_block = f"""## Results (walk-forward, test seasons 2024-25 and 2025-26)

Everything was tuned on 2023-24; the final version was scored, with everything frozen, on
{N[f'{T}.team.team_model.games']:,.0f} test games / {N['n_players_test']:,} player-games (earlier versions had also been
scored on them before the review fixes). Full tables: [docs/REPORT.md](docs/REPORT.md) and the auto-generated
[docs/FINDINGS_DATA.md](docs/FINDINGS_DATA.md). An adversarial review of the first version found errors that flattered
some results; they were fixed - except the who-appeared availability stand-in, which cannot be fixed for the test
seasons and is measured instead (item 2), and overtime frequency, fixed only partly - and everything was re-run
(REPORT, "What an adversarial review changed").

1. **The box-score database is near-exact where it can be checked.** Player box scores derived from play-by-play match
   the official NBA.com box scores on every counting stat for {pct(N['q.all_stats_exact'], '.2f')} of 2023-24
   player-games (minutes within 1 s for {pct(N['q.minutes_within_1s'], '.2f')}, plus-minus
   {pct(N['q.plus_minus_exact'], '.2f')}) and for {100 * min(q_old):.1f}-{100 * max(q_old):.1f}% in 2020-23; 2024-25 and 2025-26
   have no official box scores to check against.
2. **Minutes are where accuracy is won.** A LightGBM minutes model with pregame teammate-absence features and the
   announced starting five has MAE {m_gbm:.2f} minutes vs {m_sa:.2f} for season averages
   ({100 * (1 - m_gbm / m_sa):.0f}% better; {m_pre:.2f}, {pct(m_pre_gain)} better, without lineup information). Who is available is
   taken from who appeared (the sources have no inactive list for the test seasons), which also reveals rotation
   players a coach chose not to use; on 2023-24, where the official list exists, it flatters the minutes model by
   {leak:.3f} minutes ({100 * leak / N['ax.mae_appearance_proxy']:.1f}%). A second leak - deep-bench appearances reveal
   blowouts - was found and removed.
3. **Point forecasts: gradient boosting has the lowest MAE, the others are close** (on RMSE the team-first
   reconciliation is lowest for points and PRA). Points MAE: gradient boosting
   {mae['gbm']:.3f}, team-first reconciliation {mae['topdown']:.3f}, simulation {mae['sim_blend']:.3f}, closed-form
   empirical-Bayes rates {mae['eb']:.3f}; season averages {mae['season_avg']:.3f}. Most of the gain is projected minutes
   (per-minute rates x projected minutes alone: {mae['per_min']:.3f}).
4. **Distributions: the simulation is best calibrated for points, not better overall.** Against negative-binomial
   distributions with a mean-dependent dispersion around the same point forecasts, the simulation's points PIT error is
   {P('sim_blend', 'pts', 'pit_dev')} vs {P('gbm', 'pts', 'pit_dev')} and it has the best points log score, but CRPS is a
   near tie for points ({pts_crps_txt}) and favours gradient boosting for PRA ({P('gbm', 'pra', 'crps')} vs
   {P('sim_blend', 'pra', 'crps')}). Simulated over/under probabilities at the line nearest the projection are on
   average within 0.75 percentage points of the observed over rate for {', '.join(LBL[s] for s in calib_stats)}
   (within probability deciles the gaps reach {decile_max_gap('pts'):+.1f} points for points and
   {decile_max_gap('pra'):+.1f} for PRA), and overstated for {stat_list(over_stats)} points.
5. **Team forecasts: {team_head}.** The shipped simulation (team-model weight {w:.2f} in its efficiency target) has
   spread RMSE {bl_m:.3f} vs {tm_m:.3f} for the team model (squared-error gain {gtxt('margin_sim_blend')}) and total RMSE
   {bl_t:.3f} vs {tm_t:.3f} ({gtxt('total_sim_blend')}): {team_sig_txt}. The bottom-up simulation alone is significantly
   worse on both. A post-hoc {100 * w:.0f}/{100 * (1 - w):.0f} weighted average of team-model and bottom-up means (the
   blend weight chosen on 2023-24) does improve spreads ({gtxt('margin_posthoc_blend')}), so player-level information
   carries some signal the team model lacks, but the
   simulator's value is the consistent joint box score rather than a better score forecast.
6. **Joint modelling matters.** Summing independently projected players misses the team forecast by
   {N['cons.gbm.mae_gap_vs_team_model']:.1f} points on average and overstates team-score uncertainty by
   {100 * cons:.0f}% (implied SD {N['cons.gbm.implied_sd_if_independent']:.1f} vs realised
   {N['cons.gbm.realised_sd']:.1f}): teammates share possessions. Every simulated game satisfies every box-score
   identity (checked on each backtest game).
7. **Two data-backed mechanisms; matchups help a little.** Shot shares grow super-linearly with usage (alpha = 1.3),
   and a player's true shooting falls 0.88 points per +10% relative usage (natural experiment via teammate absences;
   naive regression gets the sign wrong). Opponent information improves every statistic, by
   {min(abl_pcts):.1f}-{max(abl_pcts):.1f}% of MAE.

| | Points MAE | Points CRPS | Points PIT error | PRA CRPS | Spread RMSE | Total RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Season average | {mae['season_avg']:.3f} | {crps_pts['season_avg']:.3f} | {P('season_avg', 'pts', 'pit_dev')} | {crps_pra['season_avg']:.3f} | | |
| Gradient boosting | {mae['gbm']:.3f} | {crps_pts['gbm']:.3f} | {P('gbm', 'pts', 'pit_dev')} | {crps_pra['gbm']:.3f} | | |
| Team-first reconciliation | {mae['topdown']:.3f} | {crps_pts['topdown']:.3f} | {P('topdown', 'pts', 'pit_dev')} | {crps_pra['topdown']:.3f} | | |
| **Possession simulation (shipped blend)** | {mae['sim_blend']:.3f} | {crps_pts['sim_blend']:.3f} | {P('sim_blend', 'pts', 'pit_dev')} | {crps_pra['sim_blend']:.3f} | {bl_m:.3f} | {bl_t:.3f} |
| Team model | | | | | {tm_m:.3f} | {tm_t:.3f} |

**Live data.** The current 2026-27 season is not in the pinned sources and the NBA's CDN refused this environment, so
"upcoming games" are the slate of any as-of date in 2023-24 .. 2025-26, projected from earlier games plus that game's
starting five and who appeared (the inactive-list stand-in, Results 2) - the same inputs and simulator as the
backtest - with the advantage that it can be checked against what happened ([docs/DATA.md](docs/DATA.md)).

"""

# ---------------------------------------------------------------- headline.json (report page)
headline = [
    f"<b>Point forecasts: gradient boosting has the lowest MAE for {len(mae_wins)} of 8 statistics, the others are "
    f"close.</b> Points MAE {mae['gbm']:.3f} (gradient boosting), {mae['topdown']:.3f} (team-first), "
    f"{mae['sim_blend']:.3f} (simulation), {mae['eb']:.3f} (closed-form empirical-Bayes rates), against "
    f"{mae['season_avg']:.3f} for season averages; on RMSE the team-first reconciliation is lowest for points "
    f"({rmse['topdown']:.3f} against {rmse['gbm']:.3f}). Most of the gain is the minutes model.",
    f"<b>Distributions: best calibrated for points, not better overall.</b> The simulation's points PIT error is "
    f"{P('sim_blend', 'pts', 'pit_dev')} against {P('gbm', 'pts', 'pit_dev')} for a mean-dependent negative binomial "
    f"around the gradient-boosting forecast, but CRPS is a near tie for points and favours gradient boosting for PRA.",
    f"<b>Team forecasts: {team_head}.</b> The shipped simulation's spread RMSE is {bl_m:.3f} against {tm_m:.3f} for the "
    f"team model and its total RMSE {bl_t:.3f} against {tm_t:.3f}; {team_sig_txt}. The bottom-up simulation alone is "
    f"worse on both.",
    f"<b>Players must be modelled jointly.</b> Summing independent player projections misses the team forecast by "
    f"{N['cons.gbm.mae_gap_vs_team_model']:.1f} points on average and implies a team-score SD of "
    f"{N['cons.gbm.implied_sd_if_independent']:.1f} against a realised {N['cons.gbm.realised_sd']:.1f}. In the "
    f"simulation every team score is the sum of its players' points.",
    f"<b>Honest gaps.</b> At the line nearest the projection, over/under probabilities are overstated for "
    f"{', '.join(LBL[s] for s in over_stats)} by {min(rel[s] for s in over_stats):.1f}-{max(rel[s] for s in over_stats):.1f} "
    f"points; the simulator plays overtime in {pct(ot_sim)} of games against {pct(ot_act)}; who is available comes from who "
    f"appeared, which flatters minutes by {leak:.3f} (measured on 2023-24).",
]

# ---------------------------------------------------------------- write
readme = (ROOT / "README.md").read_text(encoding="utf-8")
start = readme.index("## Results (walk-forward")
end = readme.index("## Quick start")
if ARGS.check:
    print("README results / docs/headline.json: every check passed (not written)")
else:
    (ROOT / "README.md").write_text(readme[:start] + readme_block + readme[end:], encoding="utf-8", newline="\n")
    (ROOT / "docs" / "headline.json").write_text(json.dumps(headline, indent=1), encoding="utf-8", newline="\n")
    print("README results and headline.json rendered")
