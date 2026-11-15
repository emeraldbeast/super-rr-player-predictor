# Phase 2 and Phase 3 — Corrected pressure models and player scores

## Status and scope

The independent review reports were used as audit evidence and checked against source data and code. Confirmed findings have been corrected and Phase 1 tables, Phase 2 state, expected-outcome validation, rolling-origin predictions, and player rankings regenerated. Phase 4 has not started.

The input does not record event-time revisions to DLS targets/allocations. To avoid retrospective target and overs information entering a pre-ball analysis, all DLS-tagged matches are excluded from primary pressure references, expected-outcome fitting, and player scores. The DLS marker remains in the delivery and innings audit tables for traceability; it is not a model predictor.

## Phase 1 correction carried forward

The source fields `first_inn_revised_overs` and `second_inn_revised_overs` are legal-ball counts despite their names. The corrected Phase 1 innings builder and Phase 2 state builder now use their values directly. Regular T20 innings default to 120 legal balls and Super Overs to 6. The regenerated innings schedule has no regular T20 limit above 120.

Two review examples now reconcile directly:

| Match / innings | Source revised count | Actual legal balls | Scheduled legal balls |
|---|---:|---:|---:|
| 13612 / 2 | 114 | 114 | 114 |
| 21586 / 2 | 90 | 90 | 90 |

Match 21586 innings 2 is separately labelled `target reached`; its scheduled limit remains 90 even though the innings ended before using that allocation. The analytical delivery table also drops `bowl_speed` and its raw copy after the source speed audit.

## Phase 2 — state and pressure

Pre-ball score, wickets lost, and legal balls are calculated as cumulative innings events minus the current delivery. Remaining balls and required rate use the corrected legal-ball schedule. Phase boundaries are powerplay before ball 36, middle overs from ball 36 through 89, and death overs from ball 90. The pre-ball regression test mutates runs, wicket, and legality on current and later deliveries across several matches and checks the anchored state does not change.

Historical par rates use earlier matches only. The exact lookup is competition pool × innings × phase × wickets bucket; it falls back to pool × innings × phase and then to the configured 8 runs per over. The current match is excluded from its own reference. Among the pressure-scored delivery rows, 194,427 use the exact lookup, 583 the broader lookup, and 697 the cold start.

The configured pressure score remains a 0–1 blend of pre-ball state: chase pressure combines required-rate strain (60%), wickets lost (25%), and time elapsed (15%); first-innings adversity combines par-trajectory deficit (55%), wickets lost (30%), and time elapsed (15%). A separate knockout tag is retained. The 75th-percentile high-pressure cutoff is calculated per competition pool. The rebuilt table has 200,999 deliveries; 195,707 are pressure-eligible, 5,254 DLS-tagged delivery rows and 38 Super Over rows are excluded, and 48,959 eligible rows are high pressure.

The pressure diagnostic is now stratified by pool, chase versus first-innings adversity, innings phase, and high-pressure flag. It also applies the same pool-specific context × phase weights to both cohorts:

| Pool | Cohort | Raw runs / legal ball | Standardized runs / legal ball | Raw wickets / 100 deliveries | Standardized wickets / 100 deliveries |
|---|---|---:|---:|---:|---:|
| IPL | Ordinary | 1.533 | 1.584 | 4.43 | 4.71 |
| IPL | High pressure | 1.609 | 1.432 | 7.00 | 5.23 |
| T20I | Ordinary | 1.411 | 1.465 | 4.90 | 5.18 |
| T20I | High pressure | 1.455 | 1.301 | 7.73 | 5.95 |
| Other T20 (diagnostic only) | Ordinary | 1.321 | 1.401 | 4.90 | 5.30 |
| Other T20 (diagnostic only) | High pressure | 1.315 | 1.087 | 7.68 | 6.68 |

The pooled raw scoring rate is still higher for IPL and T20I high-pressure balls, while the context/phase-standardized rate is lower and standardized wicket frequency is higher. This supports the expected descriptive direction after accounting for cohort composition; it is not causal evidence. Both the stratified table and standardized comparison are saved for review.

## Phase 3 — expected outcomes and player ratings

Three outcomes are modelled independently: batter runs, bowler-conceded runs, and bowler-credited wicket probability. The predictor allow-list contains only pre-ball state and available context. It excludes player IDs, current outcomes, result text, `is_dls`, and `is_dls_text`. DLS result flags are used only to exclude retrospectively identified affected matches from the primary sample. Wides are excluded from batter-run training, validation, calibration, and prediction; no-balls, byes, and leg-byes remain batter opportunities as recorded by the source.

The benchmark is a smoothed contextual grouped average. Forward-validation folds cover 2023–2026 and each fold trains on earlier seasons; folds are capped at 30,000 rows for runtime. The table below summarizes the latest three test seasons (2024–2026). Run outcomes are compared by MAE and wicket probabilities by Brier score. High-pressure metrics are also saved in the validation CSV. For each held-out player-rating season, model selection uses only validation seasons earlier than that held-out year; the earliest fold defaults to the grouped baseline because no prior model-comparison fold exists.

| Pool / target | Grouped baseline | CatBoost | Selected |
|---|---:|---:|---|
| IPL batter runs — MAE | 1.3946 | 1.4083 | Grouped baseline |
| IPL bowler conceded — MAE | 1.3548 | 1.3724 | Grouped baseline |
| IPL wicket probability — Brier | 0.04558 | 0.04543 | CatBoost |
| T20I batter runs — MAE | 1.2890 | 1.2710 | CatBoost |
| T20I bowler conceded — MAE | 1.2584 | 1.2412 | CatBoost |
| T20I wicket probability — Brier | 0.05164 | 0.05138 | CatBoost |

Player expected outcomes are generated with rolling-origin folds: each held-out season is predicted using prior seasons only. The earliest season in each pool has no prior training sample and is omitted from the player residual estimates. The run produced 147,084 expected-outcome rows across IPL and T20I. `phase3_model_choices.csv` records each held-out season and its latest training season.

For each player-match, the numerator is the pressure-weighted sum of observed-minus-expected batter runs or expected-minus-observed bowler runs. The player rate is the ratio of total weighted residuals to total weighted opportunities, scaled to 100 opportunities. Matches are weighted by their actual pressure exposure, not equally. Zero-exposure matches are removed before rates, match counts, and uncertainty are calculated.

Uncertainty uses a match-cluster variance for the weighted ratio estimator. The role prior mean is the player-level role mean; role between-player variance is estimated by method of moments after subtracting mean player sampling variance, with the configured variance floor. Normal-normal shrinkage produces the posterior mean and a 20th-percentile lower-tail summary. This is an empirical-Bayes approximation, not a validated 95% confidence interval; the percentile is labelled and described as a 20th percentile.

The primary ranking eligibility is at least 10 pressure-exposure matches and 100 pressure-weighted balls. A relaxed candidate pool (3 matches and 30 balls) is retained separately for discovery. The strict IPL lists currently contain 62 eligible batters and 61 eligible bowlers. Results are residual performance versus the expected model, not auction availability or a roster recommendation.

High-pressure uplift is now a separate secondary measure: for each player it is the high-pressure adjusted residual rate minus the ordinary-pressure adjusted residual rate. It is reported only when both cohorts have at least 50 pressure-weighted opportunities. Bowling wicket uplift remains a separate observed-minus-expected wicket rate; it is not combined with runs using an arbitrary conversion.

## Current IPL top tens

Rates below are per 100 pressure-weighted opportunities. A positive batting value means runs above expected; a positive bowling value means runs saved versus expected. The lower-tail estimate orders the table, but opportunity counts and separate wicket/uplift measures should also be considered.

### Batters

| Rank | Player | Shrunk rate | 20th percentile | Pressure matches | Weighted balls |
|---:|---|---:|---:|---:|---:|
| 1 | Abhishek Sharma | 39.87 | 33.26 | 53 | 118.2 |
| 2 | Suryakumar Yadav | 27.60 | 21.45 | 54 | 271.3 |
| 3 | Prabhsimran Singh | 26.75 | 21.14 | 56 | 144.7 |
| 4 | Heinrich Klaasen | 23.10 | 18.31 | 54 | 341.0 |
| 5 | Ashutosh Sharma | 27.03 | 17.55 | 26 | 148.2 |
| 6 | Shubman Gill | 21.54 | 16.66 | 58 | 210.2 |
| 7 | Shreyas Iyer | 20.78 | 14.46 | 43 | 173.7 |
| 8 | Rajat Patidar | 22.45 | 14.35 | 39 | 145.9 |
| 9 | Rinku Singh | 22.72 | 14.31 | 46 | 283.8 |
| 10 | Naman Dhir | 19.70 | 13.05 | 31 | 126.4 |

### Bowlers

| Rank | Player | Shrunk runs saved | 20th percentile | Wicket uplift / 100 | Pressure matches | Weighted balls |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Jasprit Bumrah | 37.94 | 32.06 | 0.50 | 36 | 210.2 |
| 2 | Sunil Narine | 24.63 | 20.16 | 0.61 | 52 | 289.2 |
| 3 | Matheesha Pathirana | 24.72 | 18.91 | 1.27 | 30 | 272.2 |
| 4 | Axar Patel | 8.87 | 3.77 | -1.43 | 52 | 196.6 |
| 5 | Krunal Pandya | 10.11 | 2.69 | -0.17 | 52 | 246.3 |
| 6 | Mohammed Siraj | 7.05 | 0.85 | -0.21 | 59 | 309.3 |
| 7 | Noor Ahmad | 6.26 | 0.23 | 1.17 | 50 | 283.0 |
| 8 | Ravindra Jadeja | 5.86 | -0.91 | 0.23 | 55 | 241.6 |
| 9 | Kuldeep Yadav | 3.94 | -1.42 | -0.30 | 50 | 272.3 |
| 10 | Eshan Malinga | 2.89 | -2.88 | 2.14 | 22 | 147.8 |

These are dataset-level performance rankings, not confirmation that a player will be available in the 2027 auction. Current roster, retention, role need, price, and availability checks remain necessary before making auction selections.

## Reproducibility and outputs

Run Phase 1 with `python scripts/00_audit.py` and `python scripts/01_build_deliveries.py`, then run `python scripts/run_phase2_3.py`. The run manifest binds the YAML config, delivery Parquet, source files, runtime versions, and generated artifact hashes. The config is captured for this run; the manifest does not claim that no earlier exploratory rankings were seen.

- `data/interim/innings.parquet` — corrected scheduled legal-ball limits and termination labels.
- `data/processed/model_table.parquet` — pre-ball state, pressure weights, and separate DLS exclusion tags.
- `data/processed/expected_outcomes_oof.parquet` — rolling-origin expected outcomes.
- `outputs/phase2_pressure_diagnostics.csv` — pool/context/phase pressure cohort detail.
- `outputs/phase2_pressure_standardized.csv` — common-mix comparison of high and ordinary pressure cohorts.
- `outputs/phase3_validation_metrics.csv` — forward model metrics, including high-pressure metrics.
- `outputs/phase3_model_choices.csv` — selected models and rolling training windows.
- `outputs/phase3_batting_rankings_all_pools.csv`, `outputs/phase3_bowling_rankings_all_pools.csv` — strict IPL/T20I rankings.
- `outputs/phase3_batting_candidate_rankings.csv`, `outputs/phase3_bowling_candidate_rankings.csv` — relaxed discovery pools.
- `outputs/phase3_high_pressure_uplift.csv` — separate high-versus-ordinary comparison.
- `outputs/top10_batters.csv`, `outputs/top10_bowlers.csv` — strict IPL top tens.
- `outputs/phase23_run_manifest.json` — run identity and file hashes.

## Checks and remaining limitations

All 21 project tests pass after regeneration. They cover the corrected DLS limits (30, 42, 54, 72, 90, and 114 balls), target-reached schedule, no bowl-speed columns in analytical deliveries, current/future-row pre-ball mutation invariance, DLS predictor exclusion, wide-population alignment, ratio-of-sums estimation, rolling-origin metadata, and Phase 1 attribution/audit cases.

The rankings remain provisional pending Phase 4 sensitivity checks, calibration uncertainty, a simulation/recovery check for the shrinkage model, DLS-exclusion sensitivity, rank stability under stricter eligibility, and auction availability/role-fit review. The 20th-percentile lower-tail score and the high-versus-ordinary uplift are descriptive model summaries; neither should be presented as a validated causal effect or 95% confidence bound.
