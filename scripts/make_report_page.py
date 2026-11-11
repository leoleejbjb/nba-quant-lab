"""Render docs/report.html (a self-contained validation report) from data/backtest/summary.json.

    python scripts/make_report_page.py [out.html]

Every number on the page comes from the summary written by ``quantlab report``.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "data" / "backtest" / "summary.json"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "report.html"

LABEL = {"season_avg": "Season average", "l10": "Last-10 average", "per_min": "Per-minute × projected minutes",
         "opp_adj": "Opponent-adjusted per-minute", "eb": "Empirical-Bayes rates", "gbm": "Gradient boosting",
         "sim_none": "Simulation, bottom-up", "sim_team": "Simulation, team-constrained",
         "sim_blend": "Simulation, blend (shipped)", "topdown": "Team-first reconciliation"}
SIM = {"sim_none", "sim_team", "sim_blend"}
STATS = ["pts", "reb", "ast", "fg3m", "pra", "stl", "blk", "tov"]
STAT_LABEL = {"pts": "PTS", "reb": "REB", "ast": "AST", "fg3m": "3PM", "pra": "PRA", "stl": "STL", "blk": "BLK",
              "tov": "TOV"}


def esc(x) -> str:
    return html.escape(str(x))


def fmt(v, f=".3f"):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, f)


def player_table(rows: list, metric: str, f: str, lower_better=True) -> str:
    by = {}
    for r in rows:
        by.setdefault(r["model"], {})[r["stat"]] = r.get(metric)
    models = [m for m in LABEL if m in by]
    best = {}
    for s in STATS:
        vals = [by[m].get(s) for m in models if by[m].get(s) is not None]
        if vals and metric not in ("bias", "cov80", "p_pit_below_half"):
            best[s] = min(vals) if lower_better else max(vals)
    out = ['<div class="scroll"><table class="num"><thead><tr><th>Model</th>' +
           "".join(f"<th>{STAT_LABEL[s]}</th>" for s in STATS) + "</tr></thead><tbody>"]
    for m in models:
        cls = ' class="sim"' if m in SIM else ""
        cells = []
        for s in STATS:
            v = by[m].get(s)
            txt = fmt(v, f)
            if s in best and v is not None and abs(v - best[s]) < 1e-12:
                txt = f"<b>{txt}</b>"
            cells.append(f"<td>{txt}</td>")
        out.append(f"<tr{cls}><th scope=\"row\">{esc(LABEL[m])}</th>{''.join(cells)}</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def pit_svg(pits: dict, stat: str, w=300, h=150) -> str:
    """Grouped bars: PIT deciles for gradient boosting + NB (gray) and the shipped simulation (blue); line at 10%."""
    series = [("gbm", "Gradient boosting + neg. binomial", "c-gray"), ("sim_blend", "Simulation", "c-s1")]
    data = []
    for key, name, cls in series:
        k = f"{stat}|{key}"
        if k in pits:
            a = np.asarray(pits[k], float)
            data.append((name, cls, a / a.sum()))
    if not data:
        return ""
    pad_l, pad_b, pad_t = 34, 22, 10
    top = max(0.2, max(d[2].max() for d in data) * 1.08)
    pw, ph = w - pad_l - 6, h - pad_b - pad_t
    y = lambda v: pad_t + ph * (1 - v / top)
    g = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="PIT histogram for {STAT_LABEL[stat]}">']
    for tick in (0, 0.1, 0.2):
        if tick <= top:
            g.append(f'<line x1="{pad_l}" x2="{w - 6}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" class="grid"/>'
                     f'<text x="{pad_l - 6}" y="{y(tick) + 4:.1f}" class="tick" text-anchor="end">{int(tick * 100)}%</text>')
    bw = pw / 10
    for i in range(10):
        for j, (name, cls, a) in enumerate(data):
            x = pad_l + i * bw + 2 + j * (bw - 4) / len(data)
            bh = y(0) - y(a[i])
            g.append(f'<rect x="{x:.1f}" y="{y(a[i]):.1f}" width="{(bw - 4) / len(data) - 1:.1f}" height="{bh:.1f}" '
                     f'rx="1.5" class="{cls}"><title>{name}: decile {i + 1} = {a[i]:.1%}</title></rect>')
    g.append(f'<line x1="{pad_l}" x2="{w - 6}" y1="{y(0.1):.1f}" y2="{y(0.1):.1f}" class="ref"/>')
    g.append(f'<text x="{pad_l}" y="{h - 5}" class="tick">0</text><text x="{w - 6}" y="{h - 5}" class="tick" '
             f'text-anchor="end">1</text><text x="{pad_l + pw / 2:.0f}" y="{h - 5}" class="tick" text-anchor="middle">'
             f'PIT</text>')
    g.append("</svg>")
    return "".join(g)


def reliability_svg(rel: dict, w=420, h=300) -> str:
    cols = {"pts": "c-s1", "reb": "c-s2", "ast": "c-s3", "fg3m": "c-s4"}
    pad = 40
    pw, ph = w - pad - 12, h - pad - 14
    X = lambda v: pad + pw * v
    Y = lambda v: 14 + ph * (1 - v)
    g = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Over/under reliability diagram">']
    for t in (0, .25, .5, .75, 1):
        g.append(f'<line x1="{pad}" x2="{w - 12}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/>'
                 f'<text x="{pad - 6}" y="{Y(t) + 4:.1f}" class="tick" text-anchor="end">{int(t * 100)}%</text>'
                 f'<text x="{X(t):.1f}" y="{h - 18}" class="tick" text-anchor="middle">{int(t * 100)}%</text>')
    g.append(f'<line x1="{X(0)}" y1="{Y(0)}" x2="{X(1)}" y2="{Y(1)}" class="ref"/>')
    for s, cls in cols.items():
        k = f"{s}|sim_blend|ladder"
        if k not in rel:
            continue
        r = rel[k]
        pts = " ".join(f"{X(p):.1f},{Y(o):.1f}" for p, o in zip(r["pred"], r["obs"]))
        g.append(f'<polyline points="{pts}" class="line {cls}"/>')
        for p, o, n in zip(r["pred"], r["obs"], r["n"]):
            g.append(f'<circle cx="{X(p):.1f}" cy="{Y(o):.1f}" r="4" class="dot {cls}"><title>{STAT_LABEL[s]}: '
                     f'predicted {p:.0%}, observed {o:.0%} (n={n:,})</title></circle>')
    g.append(f'<text x="{pad + pw / 2:.0f}" y="{h - 2}" class="tick" text-anchor="middle">predicted P(over)</text>')
    g.append("</svg>")
    legend = "".join(f'<span class="key"><i class="{c}"></i>{STAT_LABEL[s]}</span>' for s, c in cols.items())
    return "".join(g) + f'<div class="legend">{legend}<span class="key"><i class="c-ref"></i>perfect</span></div>'


def team_table(tm: dict) -> str:
    names = {"team_model": "Team model (ratings)", "sim_none": "Simulation, bottom-up",
             "sim_team": "Simulation, team-constrained", "sim_blend": "Simulation, blend (shipped)",
             "posthoc_blend": "Post-hoc weighted average of means (reference)"}
    keys = [("margin_mae", "Spread MAE"), ("margin_rmse", "Spread RMSE"), ("total_mae", "Total MAE"),
            ("total_rmse", "Total RMSE"), ("margin_crps", "Spread CRPS"), ("total_crps", "Total CRPS"),
            ("win_logloss", "Win log-loss")]
    rows = [(k, v) for k, v in tm.items() if isinstance(v, dict) and "margin_rmse" in v]
    best = {k: min(v[k] for name, v in rows if name != "posthoc_blend") for k, _ in keys}   # reference row excluded
    out = ['<div class="scroll"><table class="num"><thead><tr><th>Forecast</th>' +
           "".join(f"<th>{t}</th>" for _, t in keys) + "</tr></thead><tbody>"]
    for k, v in rows:
        cls = ' class="sim"' if k in ("sim_none", "sim_team", "sim_blend") else ""
        hit = lambda m: k != "posthoc_blend" and abs(v[m] - best[m]) < 1e-12
        cells = "".join(f"<td>{'<b>' if hit(m) else ''}{v[m]:.3f}{'</b>' if hit(m) else ''}</td>" for m, _ in keys)
        out.append(f"<tr{cls}><th scope=\"row\">{names[k]}</th>{cells}</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def main() -> None:
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    test_rows = s["players_test"]
    n_test = max(r["n"] for r in test_rows)
    find = lambda rows, m, st, k: next((r[k] for r in rows if r["model"] == m and r["stat"] == st), float("nan"))
    tm = s["teams"]
    t_test = tm["test"]
    bs = t_test.get("bootstrap_mse_gain_vs_team_model", {})
    minutes = {r["model"]: r for r in s["minutes_test"]}
    abl = s.get("opponent_ablation", {})
    part = s.get("participation", {})
    cost = s.get("cost", {})
    hl = ROOT / "docs" / "headline.json"
    findings = json.loads(hl.read_text(encoding="utf-8")) if hl.exists() else []
    head = f"""<title>NBA Quant Lab Validation</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@500;600;700&family=Inter:wght@400;500;600&display=swap">
<style>
/* Layout: one reading column of evidence; tables scroll inside their own frames; charts in a wrapping grid */
:root {{
  --bg: #f7f7f4; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781; --line: #e1e0d9;
  --axis: #c3c2b7; --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --gray: #b9b8b0;
  --tint: rgba(42,120,214,.07);
  --display: "Barlow Semi Condensed", "Arial Narrow", system-ui, sans-serif;
  --body: "Inter", system-ui, -apple-system, "Segoe UI", sans-serif;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #121211; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781; --line: #2c2c2a;
  --axis: #383835; --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --gray: #5d5c57;
  --tint: rgba(57,135,229,.12); color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #121211; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781; --line: #2c2c2a;
  --axis: #383835; --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --gray: #5d5c57;
  --tint: rgba(57,135,229,.12); color-scheme: dark; }}
body {{ background: var(--bg); color: var(--ink); font: 15px/1.55 var(--body);
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }}
main {{ max-width: 1040px; margin: 0 auto; padding-block: 40px 64px; display: grid; gap: 40px; }}
header {{ display: grid; gap: 10px; }}
.eyebrow {{ font: 600 13px/1 var(--display); letter-spacing: .08em; text-transform: uppercase; color: var(--s1); }}
h1 {{ font: 700 clamp(34px, 5vw, 52px)/1.02 var(--display); letter-spacing: -.01em; margin: 0; text-wrap: balance; }}
h2 {{ font: 600 26px/1.15 var(--display); margin: 0; text-wrap: balance; }}
h3 {{ font: 600 17px/1.2 var(--display); margin: 0; letter-spacing: .01em; }}
p {{ margin: 0; max-width: 68ch; color: var(--ink2); }}
section {{ display: grid; gap: 14px; min-width: 0; }}
.lede {{ font-size: 17px; color: var(--ink2); max-width: 70ch; }}
.facts {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.chip {{ font: 500 13px/1 var(--body); padding: 6px 10px; border: 1px solid var(--line); border-radius: 999px; color: var(--ink2);
  background: var(--surface); font-variant-numeric: tabular-nums; }}
ol.find {{ margin: 0; padding-left: 1.3em; display: grid; gap: 10px; max-width: 78ch; }}
ol.find li::marker {{ font: 600 15px var(--display); color: var(--s1); }}
ol.find b {{ color: var(--ink); }}
.scroll {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--surface); }}
table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; }}
th, td {{ padding: 7px 12px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--line); }}
thead th {{ font: 600 12px/1.2 var(--display); letter-spacing: .06em; text-transform: uppercase; color: var(--muted);
  background: var(--surface); position: sticky; top: 0; }}
th[scope="row"], thead th:first-child {{ text-align: left; font-weight: 500; color: var(--ink); }}
table.num td {{ font-variant-numeric: tabular-nums; color: var(--ink2); }}
td b {{ color: var(--ink); font-weight: 650; }}
tr.sim th[scope="row"] {{ box-shadow: inset 3px 0 0 var(--s1); }}
tr.sim {{ background: var(--tint); }}
tbody tr:last-child th, tbody tr:last-child td {{ border-bottom: 0; }}
.grid4 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }}
.grid2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 24px; align-items: start; }}
figure {{ margin: 0; display: grid; gap: 6px; min-width: 0; }}
figcaption {{ font-size: 13px; color: var(--muted); }}
svg {{ width: 100%; height: auto; display: block; }}
svg .grid {{ stroke: var(--line); stroke-width: 1; }}
svg .ref {{ stroke: var(--ink2); stroke-width: 1; }}
svg .tick {{ fill: var(--muted); font: 11px var(--body); }}
svg .c-s1 {{ fill: var(--s1); stroke: var(--s1); }} svg .c-s2 {{ fill: var(--s2); stroke: var(--s2); }}
svg .c-s3 {{ fill: var(--s3); stroke: var(--s3); }} svg .c-s4 {{ fill: var(--s4); stroke: var(--s4); }}
svg .c-gray {{ fill: var(--gray); stroke: none; }}
svg rect {{ stroke: none !important; }}
svg .line {{ fill: none !important; stroke-width: 2; }}
svg .dot {{ stroke: var(--surface) !important; stroke-width: 2; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 14px; font-size: 12.5px; color: var(--ink2); }}
.key {{ display: inline-flex; align-items: center; gap: 6px; }}
.key i {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; }}
.key i.c-s1 {{ background: var(--s1); }} .key i.c-s2 {{ background: var(--s2); }} .key i.c-s3 {{ background: var(--s3); }}
.key i.c-s4 {{ background: var(--s4); }} .key i.c-gray {{ background: var(--gray); }} .key i.c-ref {{ background: var(--ink2); height: 2px; }}
.note {{ font-size: 13px; color: var(--muted); max-width: 80ch; }}
.callout {{ border-left: 3px solid var(--s2); padding: 4px 0 4px 14px; display: grid; gap: 6px; }}
footer {{ font-size: 12.5px; color: var(--muted); border-top: 1px solid var(--line); padding-top: 16px; }}
</style>"""
    n_games = t_test["team_model"]["games"]
    body = [f"""<main>
<header>
  <div class="eyebrow">Walk-forward backtest · NBA regular season</div>
  <h1>NBA Quant Lab Validation</h1>
  <p class="lede">Every player's minutes, points, rebounds, assists and the rest, forecast walk-forward from earlier
  games and the starting fives (who appeared stands in for the inactive list) and simulated possession by possession,
  scored against what actually happened. Tuned on 2023-24, then frozen and scored on 2024-25 and 2025-26.</p>
  <div class="facts"><span class="chip">{n_games:,} test games</span><span class="chip">{n_test:,} test player-games</span>
  <span class="chip">7,230 games reconstructed from play-by-play</span>
  <span class="chip">{cost.get('us_per_game', 0):.0f} µs per simulated game</span></div>
</header>"""]
    if findings:
        body.append('<section><h2>What the evidence says</h2><ol class="find">' +
                    "".join(f"<li>{f}</li>" for f in findings) + "</ol></section>")
    body.append(f"""<section><h2>Player statistics: point accuracy</h2>
<p>Mean absolute error on identical player-games, conditional on the player appearing. Bold marks the lowest value
per statistic before rounding; values that tie at the three decimals shown are effectively level. Shaded rows are the
possession simulator.</p>
{player_table(test_rows, "mae", ".3f")}</section>""")
    body.append(f"""<section><h2>Player statistics: full distributions</h2>
<p>CRPS scores the whole predicted distribution (lower is better). Calibration error is the distance of the PIT
histogram from flat (0 is perfect). The six baseline models (season average to gradient boosting) use a
negative-binomial distribution around their point forecast whose dispersion depends on the predicted mean (fitted on
the tuning season; gradient boosting, which has no tuning-season predictions, uses the empirical-Bayes model's curve).
The team-first reconciliation's distributions come from its own simulated draws.</p>
<h3>CRPS</h3>{player_table(test_rows, "crps", ".3f")}
<h3>Calibration error (PIT, L1 from uniform)</h3>{player_table(test_rows, "pit_dev", ".3f")}
<h3>80% interval coverage (central 80% of the forecast, ties randomized; 0.800 = calibrated)</h3>{player_table(test_rows, "cov80", ".3f", lower_better=False)}
<h3>Share of outcomes below the forecast median (0.500 = calibrated; higher = forecast too high)</h3>{player_table(test_rows, "p_pit_below_half", ".3f", lower_better=False)}
<div class="grid4">""" + "".join(
        f'<figure>{pit_svg(s["pit_test"], st)}<figcaption>{STAT_LABEL[st]}: PIT deciles, flat line = calibrated</figcaption></figure>'
        for st in ("pts", "reb", "ast", "pra")) + """</div>
<div class="legend"><span class="key"><i class="c-gray"></i>Gradient boosting + negative binomial</span><span class="key"><i class="c-s1"></i>Possession simulation (shipped blend)</span></div>
</section>""")
    rel = s.get("reliability_test", {})
    body.append(f"""<section><h2>Over/under reliability</h2>
<div class="grid2"><figure>{reliability_svg(rel)}<figcaption>Simulated P(over) at up to five lines per player-game
(the half-point line nearest the projection and 1-2 points either side; lines below 0.5 are skipped), binned by
decile, against how often the over hit.</figcaption></figure>
<div class="callout"><h3>Reading it</h3><p>Points on the diagonal mean a stated 60% over happens 60% of the time.
Mean gap, predicted minus observed, at the line nearest the projection (percentage points):
{", ".join(f"{STAT_LABEL[k]} {100 * rel[f'{k}|sim_blend']['mean_gap']:+.1f}" for k in STATS if f"{k}|sim_blend" in rel)};
pooled over the ladder (the gaps differ by line; for rebounds, assists, threes and blocks the over-prediction
shrinks at the lines above the projection):
{", ".join(f"{STAT_LABEL[k]} {100 * rel[f'{k}|sim_blend|ladder']['mean_gap']:+.1f}" for k in STATS if f"{k}|sim_blend|ladder" in rel)}.
Every probability comes from the joint simulated games, so a two-leg parlay is the share of games in which both legs
hit, never a product of marginals.</p></div></div></section>""")
    mrows = [("season_avg", "Season average"), ("l10", "Last-10 average"), ("ewma", "Decayed average"),
             ("gbm_pre", "LightGBM, no lineup info"), ("gbm", "LightGBM, starting lineups known"),
             ("sim_none", "Simulation, bottom-up"), ("sim_blend", "Simulation, blend (shipped)")]
    mt = ['<div class="scroll"><table class="num"><thead><tr><th>Minutes model</th><th>MAE</th><th>RMSE</th><th>Bias</th>'
          '</tr></thead><tbody>']
    for k, name in mrows:
        if k in minutes:
            r = minutes[k]
            cls = ' class="sim"' if k.startswith("sim") else ""
            mt.append(f"<tr{cls}><th scope=\"row\">{name}</th><td>{r['mae']:.3f}</td><td>{r['rmse']:.3f}</td>"
                      f"<td>{r['bias']:+.3f}</td></tr>")
    mt.append("</tbody></table></div>")
    ax = s.get("availability_experiment", {})
    axp = (f"""<p class="note">Who is available is taken from who appeared (the sources have no inactive list for the
test seasons), which also reveals rotation players a coach chose not to use. On {ax['season']}, where the official
statuses exist, the lineups-known model's MAE is {ax['mae_appearance_proxy']:.3f} with that stand-in and
{ax['mae_inactive_list']:.3f} with the real inactive list: the test numbers presumably carry about
{ax['mae_leak']:.2f} minutes of optimism.</p>""" if ax else "")
    body.append(f"""<section><h2>Minutes</h2><p>Projected minutes for every player who appeared. The simulator enforces
240 player-minutes per team (plus overtime) with five on the floor at all times.</p>{''.join(mt)}{axp}</section>""")
    body.append(f"""<section><h2>Team forecasts: does the player layer help?</h2>
<p>Spread and total accuracy on {n_games:,} test games. The shipped forecast is the simulator's blend environment
(team-model weight {tm['blend_weight']:.2f}, chosen on 2023-24), backtested as such; the post-hoc
{tm['blend_weight']:.2f}/{1 - tm['blend_weight']:.2f} weighted average of the team-model and bottom-up means is shown
for reference only. Bold marks the lowest value per column before rounding, the reference row excluded. Overtime: {t_test['actual_ot_rate']:.1%} of test games, simulated
{t_test.get('sim_blend', {}).get('p_ot', float('nan')):.1%}.</p>
{team_table(t_test)}
<p class="note">Paired bootstrap over games, reduction in squared error versus the team model (positive is better):
{"; ".join(f"{k.replace('_', ' ')} {v['mean']:+.2f} (95% CI {v['ci95'][0]:+.2f} to {v['ci95'][1]:+.2f})" for k, v in bs.items())}.</p>
</section>""")
    if abl:
        rows = "".join(f"<tr><th scope=\"row\">{STAT_LABEL.get(k, k)}</th><td>{v['with_opponent']['mae']:.4f}</td>"
                       f"<td>{v['without_opponent']['mae']:.4f}</td><td>{v['mae_gain']:+.4f}</td>"
                       f"<td>{v['mae_gain_ci95'][0]:+.4f} to {v['mae_gain_ci95'][1]:+.4f}</td></tr>" for k, v in abl.items())
        gbm_pts = next((r["mae"] for r in test_rows if r["model"] == "gbm" and r["stat"] == "pts"), float("nan"))
        body.append(f"""<section><h2>Matchups</h2><p>Gradient boosting refitted twice, nested: once without the
explicit opponent information (opponent style factors, the opponent factors inside the closed-form expectation, and
the team-model environment, replaced by the team's own season-to-date averages) and once with it added back. Both
refits' features differ from the shipped model's (no recent per-minute rates; the own-team averages are kept in both),
so the with-opponent points MAE is {abl['pts']['with_opponent']['mae']:.3f} against {gbm_pts:.3f} for the shipped
model. Both refits also keep the projected minutes, whose blowout-risk features come from the team-model spread, so a
little opponent information remains in the "without" refit. Gains are the reduction in MAE from adding the opponent
information, with game-clustered bootstrap intervals.</p>
<div class="scroll"><table class="num"><thead><tr><th>Stat</th><th>With opponent</th><th>Without</th><th>Gain</th>
<th>95% CI</th></tr></thead><tbody>{rows}</tbody></table></div></section>""")
    if part.get("calibration"):
        c = part["calibration"]
        rows = "".join(f"<tr><th scope=\"row\">{p:.2f}</th><td>{o:.2f}</td><td>{n:,}</td></tr>"
                       for p, o, n in zip(c["pred"], c["obs"], c["n"]))
        body.append(f"""<section><h2>Uncertain participation</h2><p>Rotation players' availability comes from who
appeared (the stand-in for the injury report). Deep-bench appearances are uncertain: they depend on blowouts. Simulated participation
probabilities of the fringe candidates listed by the pregame rule against who actually appeared ({part['season']},
{part['games']} games):</p>
<div class="scroll"><table class="num"><thead><tr><th>Predicted P(play)</th><th>Observed</th><th>Player-games</th></tr></thead>
<tbody>{rows}</tbody></table></div></section>""")
    body.append(f"""<footer>Data: NBA liveData play-by-play 2020-21 to 2025-26 and NBA.com box scores through 2023-24 (pinned
public mirrors, SHA-256 verified), non-commercial research use. Generated by scripts/make_report_page.py from
data/backtest/summary.json.</footer></main>""")
    OUT.write_text(head + "\n" + "\n".join(body), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
