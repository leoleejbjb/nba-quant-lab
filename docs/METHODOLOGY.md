# Methodology

Every rule, model and parameter in NBA Quant Lab, and which data each was fitted on.

**Experimental design (fixed before any model was scored).**

| Role | Seasons | Used for |
|---|---|---|
| History | 2020-21, 2021-22, 2022-23 | priors, training data, simulator structural constants (2021-22 and 2022-23, with 2023-24) |
| Tuning | 2023-24 | the tuned hyper-parameters (half-lives and prior strengths of the EB player rates and the team style factors, team-model grid, usage exponent), the baselines' negative-binomial dispersion curves, the simulator's calibration and structural constants (with 2021-23), the blend weight; the gradient-boosting player models use fixed hyper-parameters |
| Test | 2024-25, 2025-26 | scored with everything frozen (the final version; earlier versions were also scored on them before the review fixes) |
| Current season | 2026-27 | projected with everything frozen (section 11); not scored or tuned, beyond a rehearsal of the projection path on 2025-26 |

All predictions are **walk-forward**: the features of a forecast for a game on date *d* use only games on dates
strictly before *d* (players and teams play at most once per day; league-level quantities are lagged one date;
same-day games never see each other). Two pieces of same-day information are used on purpose: the starting five (for
the lineups-known minutes model, the rotation rule behind the teammate context, the gradient-boosting player models'
starter feature and the simulator's starting lineups; public about 30 minutes before tip-off) and teammate
availability, which comes from who appeared in the game - the stand-in for the pregame inactive list, a remaining and
measured leak (section 2). Unplayed games of the current season have neither: they are projected from pregame
information only (section 11).

---

## 1. Data reconstruction (`src/quantlab/data`)

**Sources.** NBA liveData play-by-play for six regular seasons (pinned mirror, SHA-256 verified) and official NBA.com
box scores through 2023-24 (pinned mirror). See [DATA.md](DATA.md).

**Event order.** (period, game clock descending, `orderNumber`). The clock decides chronology and the official order
breaks ties - the convention validated against official plus-minus in the FiveMan lineup project, re-implemented here.

**Lineups.** A player whose first appearance in a period is anything other than `SUB in` started the period; within a
period substitutions are walked forward. Contradictions invalidate stints rather than being guessed around. A starter
who records no event in a whole period is filled from official box-score minutes only when the match is unique
(2020-24); later seasons flag such team-periods.

**Possessions.** A rule-based state machine: made FG (unless an and-one FT follows at the same clock), defensive
rebound of a live miss, turnover, made final FT (not technical / flagrant / clear-path / take / away-from-play), held
ball won by the defense, end of period. Offensive rebounds continue a possession. Free throws are credited to the lineup
on the floor at the *foul* (the NBA.com plus-minus convention).

**Player box scores** are derived from events (FGA/FGM by type, FT, OREB/DREB, AST, STL, BLK, TOV, PF excluding
technicals, minutes from stints, plus-minus from stint points), together with **on-court exposures**: team and opponent
possessions, shots, FT trips, turnovers, points and live rebound chances while the player was on the floor. Usage rate
uses the exact on-court team totals (not the box-score minutes approximation). 2025-26 records missed end-of-period
heaves as team actions that are not field-goal attempts; they are excluded from player stats as in the official rules.

**Validation.** Every stat is compared with the official box score for 2020-21..2023-24 (`box_compare` tables):
2023-24 minutes are within 1 s for 99.97% of player-games, points exact for 99.98%, plus-minus 99.80%, every counting
stat (all but minutes and plus-minus) exact for 99.27% (99.07% including plus-minus). Residual mismatches go both ways
(almost equally in 2020-23; in 2023-24 they lean toward the official box being higher, 132 of 231 stat differences),
consistent with post-game stat corrections that the archived feed does not contain. For seasons with official box scores, `min` is the official value.

## 2. Pregame feature store (`features/asof.py`)

* **Decayed sums** `S_k = sum_{j<k} x_j 2^{-(t_k - t_j)/h}` of every counting stat and exposure, for half-lives
  h in {15, 40, 100, 250, 600} *effective* days. Effective time counts days within a season and compresses each
  offseason to 30 days (ability does not decay over the summer). Computed per player with relative exponents and an
  exclusive shift (no cross-player cancellation; tested in `tests/test_leakage.py`).
* **Minutes history**: decayed means (7/20/60 days), last 3/5/10 games, season average, days since last game, starts.
* **Teammate availability (the pregame rule).** For each team-game the baseline is the previous 10 games. *Rotation*
  players (baseline minutes >= 8, expected minutes >= 12, or starting) are *available* or not according to the
  injury report / inactive list, which is public before tip-off; *fringe* players' participation is uncertain,
  because whether the 13th man plays depends on blowouts, so fringe players enter every pregame quantity through
  their baseline minutes, never through today's participation. An earlier version that used everyone who appeared
  leaked the game's outcome (garbage-time appearances signal a blowout); removing it cost 0.07 minutes of MAE.
* **The inactive-list stand-in (a remaining, measured leak).** The sources contain the inactive list only for
  2020-24 (box-score comments), not for the 2024-26 test seasons. So that training and test features mean the same
  thing, every season uses *who appeared* as the stand-in for rotation players' availability. The stand-in also
  reveals rotation players who were dressed but not used - 24.8% of the "absent" rotation players in 2021-24 were
  DNP-coach's decision, more often in close games - which is game information. Measured on 2023-24 with the official
  statuses (`models.minutes.exact_availability_experiment`): the minutes model's MAE is 4.362 with the stand-in and
  4.428 with the true inactive list (+0.066, 1.5%); carried through the closed-form EB projection, points MAE rises
  from 4.368 to 4.379 (+0.011, 0.24%) and PRA from 5.738 to 5.763. The 2024-26 test numbers of every model that uses
  projected minutes presumably carry about this much optimism; it was measured only for the minutes model and the
  closed-form EB projection (the gradient-boosting player models and the simulator, which read the same context, were
  not measured separately).
* **Lineup-blind variant.** The "no lineup information" minutes model uses a rotation rule and availability share
  that ignore today's starters (`rotation_pre`, `share_of_avail_pre`).
* **Pregame candidates.** Deep-bench players on the team's recent list who do not appear get a not-played row with
  the same pregame features, so the minutes and rate models project every candidate identically, appeared or not.

## 3. Team model (`models/team.py`) - the game-level engine

Time-decayed (half-life 50 effective days), opponent-adjusted ridge (lambda 2) ratings fitted walk-forward on every
earlier game: `points per 100 possessions = mu + offense + defense + home + back-to-back terms`, and pace
`= pi + p_home + p_away` (half-life 12, lambda 1). The normal equations are updated incrementally (the decay
factorizes), so each date costs one O(params^3) solve plus O(params^2) per game, independent of the number of earlier
games. Score uncertainty is a bivariate normal for (home, away) points whose
margin/total standard deviations and correlation come from earlier seasons' residuals. Hyper-parameters: grid search on
2023-24 only.

## 4. Minutes (`models/minutes.py`)

LightGBM (L2) on recent minutes, role (start rate, rotation flag), rest, days since last game, the pregame teammate
context (vacated minutes and usage of absent rotation players, available rotation minutes, share of available minutes),
blowout risk (|expected margin| from the team model) and home. Two variants: *pregame* (no lineup information, with
the lineup-blind rotation rule) and *lineups known* (today's starting five - public ~30 minutes before tip-off). For
season S the model is trained on 2021-22 .. S-1 (2020-21 is the cold-start season). Baselines: season average, last
10, decayed mean. Normalizing each team's projections to 240 minutes made MAE *worse* (4.367 vs 4.297 on the test
seasons) once teammate absences are features, so the 240-minute constraint is enforced by the simulator's rotation
engine instead. Scenario re-projection uses the per-season model trained on earlier seasons only; 2020-21 and 2021-22
have none, and their scenarios are not re-projected (the all-seasons model would use future games).

## 5. Per-possession player rates (`models/rates.py`)

Empirical-Bayes posterior means `(X_h + k mu) / (E_h + k)` (Beta-Binomial for shares, Gamma-Poisson for counts per
exposure) for 19 rates: usage (share of on-court team plays = FGA + FT trips + TOV), play-type shares (TOV, 3PA, FT
trip), 2P%, 3P%, FT%, and-one rate, 3-shot-trip share, assisted shares on 2s and 3s, assist rate (share of teammates'
made FGs assisted), OREB% and DREB% of live rebound chances, steals per opponent possession, blocks per opponent FGA,
defensive fouls per opponent possession, offensive-foul share of turnovers and the share of the player's 2PA that are
blocked. `mu` is the decayed (60-day) league rate as of the date; the half-life `h` and prior strength `k` are chosen per
rate by the predictive log-likelihood of the *next* game's count given its actual exposure on 2023-24. The fitted values
match known stability: 3P% (h = 250 days, k = 160 attempts) and steals (k = 640 opponent possessions) are slow and
noisy; usage and 3PA share (h = 40) move quickly.

**Team style factors** (`models/opponent.py`): what each defense allows relative to the league (turnovers forced, 3PA
share, 2P% and 3P% allowed, FT trips, offensive rebounding, assisted share) and what each offense concedes (steals,
blocks), decayed and shrunk to 1.0 with half-life and strength tuned on 2023-24. 3P% allowed is shrunk hard
(between-team SD 1.1%) - it is mostly noise.

## 6. The possession simulator (`sim/core.py`, `sim/engine.py`)

A compiled (numba) kernel plays each game possession by possession (the measured cost is in
[FINDINGS_DATA.md](FINDINGS_DATA.md), "Computational cost").

**Possession.** Side events first: late-game intentional fouls (a trailing defense in the final minute, by lead and
time left; rates estimated from 2021-24 with the lead taken before the possession's first event, counting the
unrecorded final possessions in which a leading team dribbles out the clock: a team leading by 1-3 shoots free throws
on 26% of possessions starting 24-60 s before the end and on 77% of those starting in the final 24 s, vs 9.6% earlier
in the quarter; a team up 10+ in the final 24 s is almost never fouled, 2%), technical free throws, team turnovers and
non-shooting defensive fouls. Then a *user* is drawn
from the five on offense with weights `usage^alpha x exp(eta) x usage multiplier`; his play type (TOV / 2PA / 3PA / FT
trip) follows his own tendencies scaled by the defense's factors; the outcome is resolved: steals and offensive fouls on
turnovers, blocks (shooter's blocked share x defenders' shot blocking), makes (logit = player skill + player shooting
night + defense + home + environment shift - score effects), assists (the assister drawn among the other four by
passing weights; the shooter's assisted share scaled by the passers on court), and-ones, FT trips of two or three shots
with the fouler drawn from the five defenders. Misses go to the rebound model: a team-rebound share (11.1%), else an
offensive rebound with probability O/(O+D) of the on-court rebounding weights; offensive rebounds start a new play.
Possession time is Gamma (shape 4.6) with mean set by the team model's pace plus 5.1 s per offensive rebound.

**End of games** (4th quarter and overtime; every rate estimated on 2021-24 possessions and shots). In the last two
minutes possession length depends on the score (base length before offensive-rebound time: leading teams run the
clock, 17.2 s with 24-60 s left; teams down 4+ hurry, 9.1 s) and shooting changes with time left and score (logit
shifts by window x leading / tied / down 1-3 / down 4+ x 2PA / 3PA, mostly negative: e.g. with 24-60 s left threes go
in less often for leading and trailing teams, -0.25 / -0.20 / -0.19, but more often for tied teams, +0.33). With 24 seconds or less: a leading
team runs out the clock unless fouled; a tied team holds for the last shot (91% of real tied possessions end at the
buzzer; a team down 1-3 holds 51% of the time, otherwise attacks quickly); shot selection follows the score (down 3:
62% threes, 89% in the final 4 s); a held last shot and any shot in the final 4 seconds use final-seconds make rates
by score (tied teams: 27% of twos and 19% of threes, shrunk toward the pooled final-seconds rates from the observed
26% and 19%), and a heave's chances shrink with the time left. Calibrated
target: real tied teams with the ball and <= 24 s left go to overtime 72.6% of the time and win 23.9%; the simulator
gives 71.6% and 24.9%. The overall regulation-tie rate is still too low (see REPORT, Limitations).

**Rotations.** Exactly five per team at all times. Every game minute, players are ranked by: planned on-court
propensity (rotation templates by starter role and minutes tier, fitted by iterative proportional fitting so that rows
sum to the planned minutes and columns to five) + error diffusion toward the player's sampled minutes target +
persistence + noise, subject to: starters open the first and third quarters; foul trouble (2/3/4/5 fouls by quarter,
lifted in the last six minutes): a player in foul trouble sits until the quarter's limit rises - minutes he is owed do
not pull him back - and players with 4 / 5 fouls commit fouls at 0.55 / 0.45 times their usual rate (calibrated on
2023-24 games: P(6 fouls) 0.94% simulated vs 0.91% real, P(5+) 4.7% vs 4.6%, at calibration); disqualification at six fouls (he fouls no
more and is replaced at the next possession; a disqualified player stays in only if no eligible substitute is left,
as under NBA rules); in-game injury (2% per rotation player per game, from the 2023-24 PIT of minutes); data-driven
garbage time (with 4 minutes left and a 21-25 point margin, starters are on the floor 29% as often as in close
games); overtime closers. Overtime is played until the tie is broken.

**Minutes budget and calibration.** The minutes model is unbiased for rotation players, so their projections
(minutes *if he plays*) are honoured; fringe players fill the remainder (241.5 minus rotation projections) by appearing
more often - their participation probability is scaled, their minutes-when-playing are not (inflating those instead
biased them by +4.9 minutes). The scaling is continuous: a deeper projected rotation means less fringe
participation, down to a floor of 3% per fringe player (`p_min`). The budget is drawn up for every available rotation player playing; when a questionable
rotation player (participation < 1: an unplayed game's availability, or a scenario setting) sits, a quarter of his
planned minutes goes to the deep bench - that night the fringe players' participation rises until their expected extra
minutes equal 25% of his (`fringe_take`; on 2021-24, fringe minutes = 2.4 + 0.255 x the rotation minutes vacated, by
season 0.23-0.30) - and the kernel's per-game 240-minute normalisation gives the rest to the players who do play.
Nothing changes when nobody sits, so the backtests, where every rotation player is known to play, are unaffected
(identical random numbers). Players whose minutes or participation were set in a scenario are never rescaled. Short
pilot runs (in which every available rotation player plays, so targets and simulated means are on the same basis)
then adjust each player's planned minutes so that simulated mean minutes equal the projection (which already includes
average blowout effects).

**Correlation structure.** Game-level pace night; per-player shooting night (logit SD 0.09); per-player usage,
rebounding and passing nights that share a common "activity" factor (correlation 0.4) so that a player's points,
rebounds and assists move together; usage competition makes teammates' shots negatively correlated; score effects make
leads mean-revert. A *team* shooting night was included initially and calibration drove it to zero.

**Calibration (2023-24 only).** Real NBA scores are *less* variable than independent possessions imply (independent
possessions give a team-points SD of ~12.8; the residual SD of a good pregame model is ~11.5 including model error), so
the simulator has two mean-reversion terms: a logit drift against the leading team (0.034 per 10 points of lead) and
against a team scoring above its expected pace (0.034 per 10 points). These, the player-night SDs and a handful of
nuisance multipliers (FT trips 0.918, non-shooting fouls 0.93, steals 1.064, blocks 1.05, offensive rebounding odds
0.975, possession length 1.0124, 2P/3P logit offsets -0.035 / +0.005) were set by matching league aggregates on the
full 2023-24 backtest (a first fit on a 200-game sample did not hold on the full season and was replaced),
margin/total dispersion and player-level variance ratios / PIT histograms; the usage exponent was chosen by points MAE
on 150 tuning games (below). Check on the full 2023-24 bottom-up backtest (1,230 games) with the final values,
simulated minus actual per team-game: points +0.00, made twos +0.09, made threes -0.01, FTA -0.25, player turnovers
+0.09, steals +0.04, blocks +0.01, fouls -0.07, possessions +0.05 (`quantlab report` writes these level checks to
FINDINGS_DATA.md). Two gaps remain: the simulator takes about 2% more two-point attempts per possession than real teams
(makes match, so rebounds run about 2.5% high), and assists are about 1.5% low; in the test seasons blocks (+8%),
free-throw attempts (-5%), fouls (-4%), steals (-3%) and threes (+2%) drift as well, and rebounds reach about +3%. **Usage is super-linear**: points MAE on 150 tuning games was
4.571 / 4.440 / **4.406** / 4.457 / 4.588 for alpha = 0.7 / 1.0 / **1.3** / 1.6 / 1.9.

**Accounting identities (asserted on every simulated game, `SimResult.check_invariants`, tested):**
points = 2·2PM + 3·3PM + FTM for every player; team points = sum of player points = final score; minutes = 5 x game
length per team; assists <= made field goals; makes <= attempts; plus-minus sums to 5 x margin; no ties; per team,
missed FG <= own offensive + opponent defensive rebounds (player + team) <= missed FG + missed FT; no player above six
fouls unless his team ran out of eligible players. The backtest checks them on every simulated game and stops if
one fails.

## 7. Integration with the team model

Three environments (`GameSimulator.calibrate`):

* **bottom-up** - player projections alone determine scoring;
* **team-constrained** - a team-level logit shift on make probabilities (two pilot runs estimate its sensitivity)
  makes simulated points per possession equal the team model's; each simulated game still varies naturally;
* **blend** (shipped: dashboard and CLI default) - the target is `w x team model + (1 - w) x bottom-up`. `w` is chosen
  on 2023-24 (`quantlab blendweight`): the blend's mean margin and total are, to first order, the same mixture of the
  team-constrained and bottom-up backtests' means, so `w` minimises margin RMSE + total RMSE of that mixture; the blend
  environment itself is then backtested with that weight on the tuning season and both test seasons (2023-24,
  2024-25, 2025-26).

Pace always comes from the team model. Shooting is shifted, not individual scores forced, so distributions stay
realistic. The alternative architecture (`eval/topdown.py`) samples team scores from the team model, draws each
participant's minutes around his projection (rescaled so that the team's minutes sum to 240) and allocates made
baskets, free throws, misses, rebounds, assists, steals, blocks and turnovers multinomially to that simulated game's
participants - the simulator's pregame roster, with fringe participation sampled from the same pregame probabilities -
in proportion to their per-minute projections x simulated minutes: box scores whose points add up, without lineups or
game dynamics (rebounds are not tied to misses, there is no plus-minus or overtime, and rounded scores can tie).

## 8. Scenarios (`sim/scenario.py`)

Removing a player re-runs the minutes model with updated absence features (minutes go where they historically went);
shots, assists and rebounds are re-allocated inside the simulation. The **usage-efficiency tradeoff** - players whose
load rises shoot less efficiently - is estimated with teammate absences as a natural experiment on 59,494 rotation
player-games (2021-24): vacated usage raises relative usage (first stage 0.30, SE 0.011) and lowers true shooting
(reduced form -0.026, SE 0.006): -0.088 TS per unit of relative usage (SE 0.020), a logit shift of -0.33. Naive OLS
gives +0.020 - the hot-hand confound (hot shooters shoot more) - which is why the instrument matters. In a scenario
each player's FG% is shifted by -0.33 x the relative change of his expected share of team plays. Scenarios reuse the
baseline game's environment shift and are simulated with the same seed.

## 9. Props (`sim/props.py`)

Probabilities are shares of the *joint* simulated games, conditional on the player appearing (a void otherwise).
Half-point lines have no push; whole-number lines report P(over), P(under) and P(push) = P(X = line). Joint
probabilities count the games in which every leg hits (never products of marginals); the dashboard shows the
correlation lift (joint / product).

## 10. Validation protocol (`eval/`)

* Point accuracy: MAE, RMSE, bias. Distributions: CRPS of the discrete distribution, randomized PIT histograms
  (calibration error = L1 distance from uniform), interval coverage, log score, and over/under reliability. Coverage is
  the share of outcomes in the central 80% (50%, 90%) of the forecast distribution with ties randomized (randomized PIT
  in [0.1, 0.9]) - exactly 80% for a calibrated forecast of a discrete statistic; quantile intervals with inclusive ends
  over-cover by construction (a calibrated forecast with the simulator's PRA means and SDs covers ~84%), so they are
  reported only as a secondary column. Over/under reliability uses the half-point line nearest the projection
  (floor(mean) + 0.5) and the ladder from 2 below to 2 above it.
* The minutes-based baselines (per-minute, opponent-adjusted, EB, gradient boosting) share the projected minutes, while
  season and last-10 averages are per-game averages; these six models get negative-binomial distributions whose size `r`
  depends on the predicted mean (method of moments in deciles of the tuning-season predictions; a single pooled `r`
  is far too narrow for bench players and too wide for stars - for points `r` ranges from 0.9 to 13.8 across deciles).
* Players are scored conditional on appearing; participation follows the pregame information set (rotation
  players' availability from the inactive-list stand-in, which also reveals coach's-decision DNPs; fringe players
  uncertain). Fringe candidates come from one
  pregame rule (`features.asof.fringe_pool_table`: players who appeared for the team in the previous 25 days and
  have not appeared elsewhere since; participation = share of the team's games in that window). Every candidate has
  a pregame feature row whether or not he appears (66,914 not-played candidate rows are added to the feature store),
  so his minutes-if-playing come from the same walk-forward minutes model and his rates from the same EB model. A
  player who appeared without being on that list (a new signing, a return from a long absence - he would be on the
  pregame active list) is added at a 5% participation floor and excluded from the participation calibration. Team metrics:
  margin/total MAE, RMSE and CRPS, win log-loss and Brier, with paired bootstrap intervals over games.
* Matchup ablation: gradient boosting retrained without the direct opponent information (opponent style factors, the
  EB expectation recomputed with neutral opponent factors, and the team-model environment replaced by the team's own
  season-to-date averages; the player's own team's style factors are kept, and so are the projected minutes, whose
  blowout-risk features come from the team model's expected margin against this opponent). The design is nested: the
  with-opponent arm is the without-opponent arm plus the opponent information. Both are refits whose features differ
  from the shipped model's (neither has the season-to-date / last-10 per-minute rates; both keep the own-team
  season-to-date averages and the neutral-opponent EB expectation), so the with-opponent MAE differs slightly from the
  shipped gradient-boosting model's (points 4.383 against 4.376).

## 11. The current season (`quantlab live-update`, `live.py`, `sim/inputs.py`)

**Update.** `quantlab live-update` fetches the season's new games (DATA.md, "The current season"), rebuilds the season
and the database and re-runs the walk-forward pipeline over all seasons with every fitted parameter frozen: the
feature store, the team model, the minutes models (for the current season trained on 2021-22 .. 2025-26, both
variants, saved for re-projection), and the EB rates and team style factors with the half-lives and prior strengths
read from `rate_params.json` and `team_factor_params.json`; the simulator constants and the blend weight are reused.
Nothing is re-tuned and nothing about the current season is scored. On the rehearsal (602 finished games) an update
without the download took about 90 s.

**Unplayed games.** The unplayed games up to 10 days after the last finished game (`QUANTLAB_UPCOMING_DAYS`) get rows
in every walk-forward table with nothing from the game: the team model's ratings through the last finished game and
the schedule's rest days and back-to-backs; one feature row per candidate player - the team's pool, the players who
appeared for it in the previous 25 days and not for another team since (the fringe-candidate rule of section 10,
applied to everyone); EB rates and style factors as of the last finished game. Baselines are updated only by played
games, so an unplayed game changes no one's history; its rows are recomputed by every update. A game several days
ahead is computed as of its team's next unplayed game (`features.asof.info_dates`): nothing is known about the games
in between, so they must not count as games the player sat out (days since he played, the decay of his history, the
pool's 25-day window). A team that has not played yet this season has no pool, so its games are projected from its
first game on (the slate says so).

**Information set** (`sim.inputs.pregame_team_spec`). The backtest's two pieces of same-day information do not exist
yet, so each is projected:

* **Availability.** There is no injury feed. A player's *missed-game streak* is the number of his team's games since
  he last appeared (0 = he played the previous game). A rotation player (the lineup-blind rule of section 2) with a
  streak of 3 or more is presumed out, and his teammates' availability context (section 2) counts him as absent; the
  others are counted as available and play with the appearance rate of their streak, estimated on 2023-24 for
  rotation players in their team's pool (`data/features/availability_priors.json`):

  | Missed in a row | 0 | 1 | 2 | 3-4 | 5-9 | 10+ |
  |---|---:|---:|---:|---:|---:|---:|
  | Appeared (2023-24) | 91.2% | 49.3% | 29.8% | 24.3% | 11.0% | 5.1% |
  | Player-games | 22,461 | 1,878 | 886 | 1,012 | 1,205 | 216 |

  Fringe candidates play with their share of the team's recent games, as in the backtest. In each simulated game a
  player drawn as not playing sits; a quarter of his minutes goes to the deep bench and the rest to the players who
  play (section 6).
* **The injury report** is the user's: the Slate page's injury report, or `data/live_overrides.json`, e.g.
  `{"out": [1628369], "available": [1630532], "p_play": {"1629029": 0.5}, "starters": {"1610612738": [five ids]},
  "teams": {"1630532": 1610612753}}`, applies to every unplayed game. An `out` player is removed; an `available`
  player plays with probability 1 (or his `p_play`) - also one outside the 25-day pool, back from a long absence,
  who is added from his latest rates and minutes on the team he last played for (or the one in `teams`, e.g. after a
  trade). The report is applied to the default pregame roster the way a scenario is (section 8): whenever it removes a
  player the pregame context counted as available, or brings back one it counted as absent, his teammates' minutes
  are re-projected (the lineup-blind per-season model with the availability features recomputed) and their shot
  efficiency follows their new share of the plays (the usage-efficiency tradeoff), while the environment shift stays
  the one calibrated on the default roster - the team model's ratings come from games with that roster and know
  nothing about the report, so re-anchoring to them would undo most of a star's absence. A scenario's removals on an
  unplayed game join the report, so the Scenario lab and the report give the same projection. A file that cannot be
  read is ignored with a warning.
* **Starters.** The starting five is announced about 30 minutes before tip-off, so it is projected: the team's five
  in its last game, among the available rotation players, with vacancies filled by start rate (decayed, 20-day
  half-life) and then by projected minutes. The Scenario lab, or `starters` in `data/live_overrides.json`, sets the announced five
  (they play; minutes are not re-projected for it).
* **Minutes** come from the lineup-blind model (section 4), conditional on playing; the team level (pace, efficiency,
  the blend) is unchanged.

**Rehearsal** (`scripts/rehearse_live.py`, results in `docs/pregame_rehearsal.json`). The CDN refused this build
environment, so the live path was rehearsed on 2025-26: the season was written in the live feed's format with the
games before 2026-01-15 final and the rest scheduled, `quantlab live-update --skip-fetch` ran on it unchanged, and
every unplayed game in the following 7 days (53 games) was projected pregame (2,000 simulations each) and scored
against what happened, next to the backtest's projections of the same games (`sim_blend`, which know who appeared and
who started). Per player-game of the players who appeared (1,096 in both):

| MAE | MIN | PTS | REB | AST | 3PM | STL | BLK | TOV | PRA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pregame (rehearsal) | 4.921 | 4.560 | 1.870 | 1.373 | 0.925 | 0.716 | 0.522 | 0.891 | 5.934 |
| Backtest (appearances and starters known) | 4.102 | 4.329 | 1.793 | 1.362 | 0.915 | 0.708 | 0.523 | 0.883 | 5.489 |

* **Minutes are the gap** (+20%; points +5%, PRA +8%, the other statistics within -0.3% to +4%). Projected minutes
  run 0.36 high on average (backtest 0.19): +0.42 for the players who started (backtest +0.47) and +0.31 for the
  bench (-0.07); by the lineup-blind rule, rotation players are 0.7 minutes high when they play and fringe players
  2.0 low (points +0.16 per player-game, backtest +0.06). Two corrections made during the review produced these
  numbers: games several days ahead computed as of the team's next game (above), and the deep bench's share of a
  questionable player's minutes (section 6); before them, minutes MAE was 5.153, points 4.575 and the bias of the
  starters' minutes +1.03.
* **Starters**: 4.0 of the 5 projected starters started, on average.
* **Availability**: 16.0% of the projected rotation players did not appear, against 14.6% implied by their
  appearance probabilities (mean 85.4%). 56 of the 1,152 player-games (4.9%, 3.8% of the minutes) were not projected:
  25 by players presumed out after three or more missed games who returned, 31 by players not in the pool (signings,
  call-ups, players traded in before their first game for the team, returns after more than 25 days).
* **Team level**: spread RMSE 14.77 (backtest simulation 14.79, team model 14.81) and total RMSE 18.50 (18.61, 18.49) -
  indistinguishable on 53 games.
* **Days ahead**: within the week there is no clear trend with the days ahead (points MAE by day 4.29 - 5.14, minutes
  4.37 - 5.30, in no order; the slates differ more than a few days of staleness matter). Updating daily still helps:
  the streaks and the last starting five change with every game.

The backtest numbers in REPORT are therefore an optimistic bound for a real slate; entering the injury report and the
announced starters closes part of the gap.
