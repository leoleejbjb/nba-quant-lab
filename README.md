# NBA Quant Lab

**Simulate an NBA game possession by possession and get the final score *and* every player's box score - as
probability distributions that are internally consistent in every simulated game.**

NBA Quant Lab reconstructs six regular seasons (2020-21 .. 2025-26, 7,230 games) from raw play-by-play into a
player box-score database that matches the official NBA.com numbers, projects every player's minutes and
per-possession rates from pregame information only, and plays each game 20,000 times through a compiled possession
simulator with real five-man rotations, fouls, garbage time and overtime. From the joint simulations it prices player
props at any line (pushes included), double/triple-doubles and same-game combinations, and re-runs any matchup with
injuries, minutes limits or lineup changes. Everything is validated walk-forward against what actually happened.

![Projected box score](docs/screenshots/game.png)

<!-- RESULTS -->

## Quick start

```bash
git clone <this repo> && cd nba-quant-lab
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"     # Python 3.10+
```

```bash
make data        # download (pinned, SHA-256 verified) + reconstruct 6 seasons + database   (~3 min)
make models      # walk-forward features, team model, minutes, rates, opponent factors     (~5 min)
make backtest    # walk-forward simulation backtests, 2023-24 tuning + 2024-26 test        (~1 h, 4 cores)
make report      # baselines, gradient boosting, team-first model, summary, FINDINGS_DATA  (~10 min)
make dashboard   # http://localhost:8501
make test        # 29 tests
```

Project one game from the command line (only information available before tip-off is used):

```bash
.venv/bin/quantlab project --game-id 22501132 --n-sims 20000
```

```bash
.venv/bin/quantlab project --game-id 22501132 --out 1627759      # same game, Jaylen Brown ruled out
```

## Commands

| Command | What it does |
|---|---|
| `quantlab fetch` | Download the pinned play-by-play and box-score files (cached, checksum-verified). |
| `quantlab build` | Reconstruct each season: events, five-man lineups (stints), possessions, player and team box scores with on-court exposures, reconciliation against official box scores. |
| `quantlab database` | Unify seasons: `player_games`, `team_games`, `games`, `availability`, `players` (+ DuckDB views). |
| `quantlab features` | Walk-forward pregame feature store (decayed rates, minutes history, teammate absences). |
| `quantlab team` | Team model: walk-forward opponent-adjusted efficiency and pace ratings, spread/total distributions. |
| `quantlab minutes` | Minutes models (LightGBM vs baselines), per-season model files for scenarios. |
| `quantlab rates` / `opponent` | Empirical-Bayes per-possession player rates; team defensive/offensive style factors. |
| `quantlab constants` | Simulator structural constants from pre-test seasons (rotation templates, garbage time, possession time, late-game fouling). |
| `quantlab backtest` | Walk-forward simulation backtests (bottom-up and team-constrained). |
| `quantlab baselines` / `topdown` | Baselines + gradient boosting; the team-first reconciliation architecture. |
| `quantlab report` | `data/backtest/summary.json` and `docs/FINDINGS_DATA.md`. |
| `quantlab project` | Projected box scores, score distribution and a prop for one game; `--out` for scenarios. |
| `quantlab dashboard` | Streamlit dashboard. |
| `quantlab sql "SELECT ..."` | DuckDB over every processed table. |

## Architecture

```mermaid
flowchart LR
  A[Pinned mirrors of NBA.com data<br/>SHA-256 verified] --> B[Play-by-play reconstruction<br/>lineups · possessions · box scores]
  B --> C[(Player-game database<br/>2020-21 .. 2025-26)]
  C --> D[Walk-forward features]
  D --> E[Team model<br/>pace + efficiency]
  D --> F[Minutes model<br/>LightGBM]
  D --> G[EB per-possession rates<br/>+ opponent factors]
  E -- pace, environment --> H{{Possession simulator<br/>rotations · fouls · OT}}
  F -- minutes targets --> H
  G -- player tendencies --> H
  H --> I[Box scores · score distributions<br/>props · scenarios]
  I --> J[Dashboard]
  H --> K[Walk-forward validation<br/>vs baselines, GBM, team-first model]
```

```
src/quantlab/
  data/        sources, download, events, lineups, possessions, game (box scores + exposures), build, database
  features/    asof.py - walk-forward pregame feature store
  models/      team.py, minutes.py, rates.py, opponent.py
  sim/         core.py (numba kernel), engine.py (spec, calibration, invariants), inputs.py, props.py,
               scenario.py, constants.py
  eval/        backtest.py, baselines.py, topdown.py, report.py, writeup.py
  dashboard/   Streamlit app (slate, box score, player & props, scenario lab, comparison, validation, data)
  live.py      projection service (as-of-date slates, scenarios with minutes re-projection)
tests/         29 tests: PBP rules, leakage, simulator invariants, props, scenarios
docs/          REPORT.md, METHODOLOGY.md, DATA.md, FINDINGS_DATA.md (auto-generated), screenshots
```

## Dashboard

| | |
|---|---|
| ![Slate](docs/screenshots/slate.png) **Slate**: every game on the as-of date with win probability, spread, total and the final score | ![Player](docs/screenshots/player.png) **Player detail & props**: distributions, any line (pushes on whole numbers), ladders with fair odds, double/triple-doubles, joint probabilities |
| ![Scenario](docs/screenshots/scenario.png) **Scenario lab**: rule players out, change minutes, starters, availability or usage; compare with common random numbers | ![Compare](docs/screenshots/compare.png) **Player comparison**: per-36 production, recent form and overlaid projected distributions |

The Validation page shows the backtest evidence below; the Data & methods page shows the reconciliation against
official box scores.

## Documentation

* [docs/REPORT.md](docs/REPORT.md) - results and conclusions
* [docs/METHODOLOGY.md](docs/METHODOLOGY.md) - every rule, model, parameter and what it was fitted on
* [docs/DATA.md](docs/DATA.md) - sources, checksums, coverage, reconciliation, the live-data limitation
* [docs/FINDINGS_DATA.md](docs/FINDINGS_DATA.md) - auto-generated tables behind every number

## Data and licensing

Play-by-play: NBA liveData feed via [shufinskiy/nba_data](https://github.com/shufinskiy/nba_data) (Apache-2.0). Box
scores: NBA.com via [NocturneBear/NBA-Data-2010-2024](https://github.com/NocturneBear/NBA-Data-2010-2024) (MIT).
Both pinned to exact commits. The statistics belong to the NBA (NBA.com Terms of Use: personal, non-commercial use);
raw and processed data are not committed. Code: MIT.
