# Data

## Sources (pinned)

| Role | Mirror @ commit | Underlying source | License |
|---|---|---|---|
| Play-by-play 2020-21 .. 2025-26 | [shufinskiy/nba_data](https://github.com/shufinskiy/nba_data) @ `e829d4678be1e075f99e5d41a1c5f97089be446b`, `datasets/cdnnba_<year>.tar.xz` | NBA liveData feed (cdn.nba.com) | Mirror Apache-2.0; data NBA.com Terms of Use |
| Player box scores 2010-24 | [NocturneBear/NBA-Data-2010-2024](https://github.com/NocturneBear/NBA-Data-2010-2024) @ `a5f108b5b1f08074d78b9e8e901926a9ce4c06c5`, `regular_season_box_scores_2010_2024_part_{1,2,3}.csv` | NBA.com box scores | Mirror MIT; data NBA.com Terms of Use |
| Team game totals 2010-24 | same, `regular_season_totals_2010_2024.csv` | NBA.com team game logs | same |

Checksums (also in `src/quantlab/data/sources.py`; `quantlab fetch` refuses a file that does not match):

| File | SHA-256 |
|---|---|
| `cdnnba_2020.tar.xz` | `e45e1b4a70a33ce93525c58f04362b6c275f7e514789772d4a8360f22a13f4f2` |
| `cdnnba_2021.tar.xz` | `0c116aff50b829de70902023f108ee78c0f44bec569521ebf1fa292e01bda28f` |
| `cdnnba_2022.tar.xz` | `de895e3474365fc14b028f514eb0f8c9c1503b95d3f1a5aef5a408305ddfb41f` |
| `cdnnba_2023.tar.xz` | `33d49fefc809f73d5d6cbad6d1d6690e0df6c89fe2b5a4d814ecc9ea4df6101b` |
| `cdnnba_2024.tar.xz` | `35a82e6b29821adb22b8474dea8c51e08847d8e00690d1d5ffde7d000e0087d1` |
| `cdnnba_2025.tar.xz` | `a5d1bed30778430816b755676649cb2e3690ea62df5d96f4d9d6e9a8b8839339` |
| `regular_season_box_scores_2010_2024_part_1.csv` | `ca9636d08c577e3d752f69e32f90b173f4c3374c15f4dc8494b655049fdfc557` |
| `regular_season_box_scores_2010_2024_part_2.csv` | `5643af8a9f15ce3540dced759375ca79e77c71644e23270136f23b7a57fc182d` |
| `regular_season_box_scores_2010_2024_part_3.csv` | `65a01fee3df865a272e578cf4b365dce7dbc62eef2ce6f15f0844cf6694ef375` |
| `regular_season_totals_2010_2024.csv` | `b63367be8c58155f978c841bb966f5b19e334cec833a37ddae2915a29cd352d9` |

The 2023-24 .. 2025-26 play-by-play and the box-score files are byte-identical to the files used by the FiveMan
lineup project.

## What the database contains

`quantlab build` + `quantlab database` produce (Parquet, also queryable with `quantlab sql`):

| Table | Rows | Content |
|---|---:|---|
| `player_games` | 154,357 | every player-game 2020-21..2025-26: MIN, PTS, FGA/FGM by 2/3, FTA/FTM, OREB/DREB, AST, STL, BLK, TOV, PF, +/-, starter, assisted makes, FT trips, and-ones, fouls drawn, usage, TS%, eFG%, double/triple-double flags, and on-court exposures (team/opponent possessions, shots, FT trips, turnovers, points, rebound chances). Zero-minute rows of people charged only with a bench technical who are not listed in that game's official box score (2024-26 has none) are excluded: 768 rows, of which 734 belong to coaches and staff who never played in 2020-26 and 34 to 22 people who played in other games (players charged from the bench in games they sat out, and former player JJ Redick in 8 games as Lakers coach); 16 such rows of players listed in the 2020-24 official box as DNP or DND are kept (not played) |
| `team_games` | 14,460 | team box score incl. team rebounds and team turnovers, possessions, opponent columns, rest days |
| `games` | 7,230 | teams, date, tip time, score, overtime, reconciliation flags |
| `availability` | 177,057 | 2020-21..2023-24 from the official box-score comments: `played`, `dnp_cd` (dressed but unused: "DNP - Coach's Decision" or listed at 0:00) or `out` (every DND - including "DND - Coach's Decision" - NWT, DNP-injury, rest and suspension comment). 2024-25 and 2025-26 have no official box scores, so only `played` rows exist (from the play-by-play, `source = pbp`): who was dressed but unused, or out, is unknown there |
| `stints` (per season) | ~37k / season | both five-man units, start/end, duration, points, possessions, score margin at the start, validity |
| `possessions` (per season) | ~243k / season | offense, period, `start_s` (when the previous possession ended, or the period start), `first_s` (time of its first recorded event), `end_s`, event indices, `lead_start` (offense lead before its first event), points, end reason, offensive rebounds, FT trips |
| `box_compare` (2020-24) | | derived vs official value of every stat for every player-game |

## Coverage and reconciliation

| Season | Games | Usable for lineup work | Valid lineup time | Minutes within 1 s | PTS exact | +/- exact | All stats exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2020-21 | 1,080 (72-game season) | 1,080 | 100.000% | 99.99% | 99.96% | 99.90% | 97.68% |
| 2021-22 | 1,230 | 1,225 | 99.989% | 99.95% | 99.96% | 99.90% | 97.41% |
| 2022-23 | 1,230 | 1,140 | 100.000% | 99.16% | 99.96% | 99.28% | 99.07% |
| 2023-24 | 1,230 | 1,227 | 99.987% | 99.97% | 99.98% | 99.80% | 99.27% |
| 2024-25 | 1,230 | 1,219 | 99.901% | no official box scores in the sources |
| 2025-26 | 1,230 | 1,226 | 99.966% | no official box scores in the sources |

* **2020-22 stat mismatches** (all-stats-exact 97.4-97.7%) go both ways almost equally (e.g. 255 steal disagreements
  in 2021-22 with a net of -12): post-game stat corrections that the archived feed lacks, not parsing errors. Every
  stat still matches >= 99%. In 2023-24 the 194 player-games with a difference lean slightly toward the official box
  being higher (132 of 231 stat differences; see REPORT section 1).
* **2022-23 minutes**: 90 games have equal-and-opposite minute pairs (e.g. +60 s / -60 s) - a substitution logged at a
  different clock time in the feed than in the box score. Counting stats are unaffected; those games are kept for
  player stats (official minutes are used) and flagged for lineup-level work.
* **Timestamps**: the feed mixes two ISO-8601 formats (with and without fractional seconds); parsing them with an
  inferred format silently dropped 232 game dates in 2024-26 before it was fixed.
* **Traded players** can appear in their former team's recent rotation; they are excluded from that team's
  fringe list when they appear for the opponent.
* **Availability.** Who is available for a game (the injury report / inactive list) is public before tip-off but is
  in the sources only through the 2020-24 box-score comments. Every season therefore uses who appeared as the
  stand-in for rotation players' availability, so that training and test features mean the same thing. The stand-in
  also reveals the rotation players a coach chose not to use: on 2021-24, 24.8% of the rotation players it treats as
  unavailable were in fact dressed (DNP - coach's decision), and how often that happens depends on how the game went.
  On 2023-24, where the official statuses exist, the minutes model is 0.066 minutes (1.5%) less accurate with the
  official inactive list than with the stand-in, and closed-form EB points projections 0.011 (0.24%) - the size of the
  optimism this leaves in the 2024-26 test numbers (`quantlab minutes` writes the minutes result and the 24.8% share
  to `data/models/availability_experiment.json`; `quantlab report` adds the points propagation to
  `data/backtest/summary.json`).

## Live 2026-27 data

The current season is not in the pinned mirrors. `cdn.nba.com` (schedule, live box scores and play-by-play) was added
to the environment's network allowlist, but the NBA's CDN answered **"Access Denied"** to this client; per project
policy a refusal is reported and never retried under a different identity (e.g. a spoofed browser). The dashboard
therefore defines "upcoming games" by an as-of date: every game on that date is projected from earlier games plus
who appeared and who started in it, the stand-in for the pregame inactive list and announced starting lineups (see
Availability; this makes the projections slightly optimistic compared with a true live projection). Projecting a past
date this way has the advantage that results can be checked. When newer seasons are published by the mirror, add the
file's checksum to `PBP_SHA256` in `sources.py` and move the pinned mirror commit (`SHUFINSKIY` in `sources.py`, and
the Sources table above) to one that contains it (a fresh fetch then re-verifies the older files at that commit), add
the season to `ALL_SEASONS` in `config.py` (used by `fetch`, `build` and `database`), and extend the hard-coded season
lists in `models/minutes.py` (`run_minutes`) and `dashboard/state.py` (`SEASONS`).

## Terms

The statistics belong to the NBA. NBA.com's Terms of Use permit personal, non-commercial use; this is a
non-commercial research project. Raw and processed data are not committed - `quantlab fetch` re-downloads the raw
files and `quantlab build` + `quantlab database` (`make data`) rebuild the processed tables.
