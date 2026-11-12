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

## The current season (2026-27)

The current season is not in the pinned mirrors; `quantlab live-update` (`make live`, `.\make.ps1 live`) fetches it from
the NBA's live CDN, the feed the play-by-play mirror archives once a season is over, so it is reconstructed by the
same code:

| File | URL | Stored as |
|---|---|---|
| League schedule (dates, tip-off times, teams, status, final scores) | `cdn.nba.com/static/json/staticData/scheduleLeagueV2.json` | `data/raw/live/2026-27/schedule.json.gz` |
| Play-by-play of each finished regular-season game | `cdn.nba.com/static/json/liveData/playbyplay/playbyplay_<game id>.json` | `data/raw/live/2026-27/pbp/<game id>.json.gz` |

* **Not pinned**: the season is still being played, so there is no fixed checksum to verify against. Each file's
  SHA-256, URL and fetch time go to `data/raw/live/2026-27/manifest.json`, and a game file is kept only if it is
  complete: the right game id, a 'game end' action, at least four periods and clocks that parse. A game in progress, a
  truncated file, a file not published yet (404) or a download that still fails after three attempts is recorded in
  the manifest (`failed`) and tried again on the next update; after five network failures in a row the fetch stops
  for that run and the update goes on with the files already on disk. Games are downloaded again for 36 hours after
  tip-off (`--refetch-hours`) to pick up the league's post-game stat corrections; older ones are read from disk.
* **Checks**: there are no official box scores for the season, so the schedule is the reference for each game's date
  and home team, and a game whose reconstructed final score differs from the schedule's is flagged as not usable for
  lineup-level work (like the pinned seasons' games that fail a check). `data/processed/2026-27/quality.json` (`live`) counts
  the scheduled and finished games, the games built, missing or incomplete play-by-play and score mismatches; the
  dashboard's Data & methods page shows it.
* **Schedule tables**: `data/processed/all/schedule.parquet` (every regular-season game of the season, with tip-off
  time, status and whether it was built) and `team_schedule.parquet` (one row per team and game, with rest days and
  back-to-backs computed on the whole schedule, so an upcoming game's rest is known; postponed games and games whose
  teams are not known yet are left out). Unplayed games never enter `games`, `player_games` or `team_games`; the
  ones in the next 10 days after the last finished game (`QUANTLAB_UPCOMING_DAYS`) get pregame rows in the feature
  store and the models (METHODOLOGY, section 11).
* **Access policy**: requests identify the program honestly (the same User-Agent as every download here, a
  one-second pause after every game-file request, gzip-compressed transfers), and an access refusal (HTTP
  401/403/407/451, or a proxy refusing the connection) stops the update: it is reported and never retried under
  another identity. This build environment was refused ("Access Denied"), so the live path
  was exercised by a rehearsal instead: `scripts/rehearse_live.py` writes a finished season (2025-26) in the live
  format from the pinned mirror (games before a cutoff final, the rest scheduled; a test checks that the live reader
  reproduces the mirror's events exactly) and runs the real `quantlab live-update --skip-fetch` on it (results in
  METHODOLOGY, section 11). On a desktop computer the CDN normally answers; cloud machines are often refused.
* **When the season changes**: set `QUANTLAB_LIVE_SEASON` to the new season (or change `LIVE_SEASON` in `config.py`).
  The finished season stays in the database, built from the files already fetched (`quantlab build --season 2026-27`
  rebuilds it). Once the mirror publishes it, pin it like the others: add the file's checksum to `PBP_SHA256` in
  `sources.py` and move the pinned mirror commit (`SHUFINSKIY` in `sources.py`, and the Sources table above) to one
  that contains it (a fresh fetch then re-verifies the older files at that commit), and add the season to
  `PINNED_SEASONS` in `config.py`.

## Terms

The statistics belong to the NBA. NBA.com's Terms of Use permit personal, non-commercial use; this is a
non-commercial research project. Raw and processed data are not committed - `quantlab fetch` re-downloads the raw
files and `quantlab build` + `quantlab database` (`make data`) rebuild the processed tables.
