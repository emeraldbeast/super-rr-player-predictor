# Super RR Selector — Round 1 Build Plan

## 1. Objective and scope

Produce two ordered, evidence-based lists of the **10 most reliable batters and 10 most reliable bowlers under pressure**, plus a methodology report. The report will define pressure, explain the performance and reliability measures, compare IPL and T20I evidence fairly, support each selection, and disclose limitations.

The primary candidate ranking is IPL-based. T20I results are a separate supporting view; they are not blended into the primary score by arbitrary format weights. Auction availability, retention status, purse, price, and squad optimisation are out of scope because those data are not supplied.

## 2. Data facts and scope controls

The pack contains six CSVs and the problem-statement PDF. The 401 `T20` matches include both IPL and 30 other T20 matches. Pool coverage is 371 IPL (367 with delivery data), 482 T20I (470 with delivery data), and 30 other T20 (all with delivery data): 883 matches and 867 delivery-covered matches total, with 200,999 deliveries. The 16 matches without delivery rows are marked abandoned. Six matches have Super Overs (38 delivery rows in innings 3 and 4); keep those out of ordinary innings rankings.

Known issues to carry into the audit:

- `extras_type` includes `wide`, `legbyes`, `byes`, `Overstep`, `other`, and `Penalty`. Confirm the run columns’ meanings before deriving batter and bowler runs.
- There are 23 matches with `revised_target`; use the revised target for chase-state reconstruction where available. Do not silently treat revised-target chases as ordinary targets.
- Forty-one matches have no `win_team_id`; the core player model does not need match winner labels.
- There are 12 rows with `dismissed_player_id` but no dismissal type, all with `is_wicket=False`, plus one `is_wicket=True` row without a dismissed-player ID. Do not infer a bowler wicket from the unresolved rows.
- Seven player names map to multiple `player_id` values. Key all modelling and joins by ID; display player name, latest team, and ID together.
- `extras_2_runs` has no schema definition. Keep it in team runs and exclude it from bowler-conceded runs in the primary calculation; keep charging it as a sensitivity setting.
- Four DLS/D-L result-text matches have no revised-target or revised-overs fields (20182, 23674, 24822, 28334). Include result text in `is_dls` detection and do not invent their chase targets.
- One inning-margin discrepancy remains: match 20190 reports Lucknow Super Giants winning by 6 wickets, while delivery rows show 5 wickets lost. Direct result-margin checks otherwise agree in 802 of 803 standard matches.
- Five non-final overs fail the six-legal-ball structure check. Preserve these named exceptions for review in the Phase 1 over-structure table.
- Dismissal labels include `Retire Out`, `RunOut`, `Stump`, and misspelled `Obstracting the field`. Normalize spelling while retaining the source value.
- `bowl_speed` is nullified in the analytical field for zero, negative, or values above 160; retain `bowl_speed_raw` for auditing.
- `delivery_style` is missing on 391 rows. Exclude it from version 1 models; consider only a later retrospective sensitivity if its annotation quality is established. Exclude `stroke` from expected-outcome features.
- `innings_ball_num` is the contiguous delivery ordinal, including wides and no-balls. It is not a legal-ball count; derive `legal_ball_no`. `delivery_ist_time` is not a reliable sequence.
- `is_impact` and `is_impact_off` describe substitutions. Preserve them as roster-context fields; they are not player performance outcomes.
- There are 22 matches with only one recorded normal innings (19 no-results and 3 abandoned); later chase analysis must not treat them as completed chases.

## 3. Statistical design

### 3.1 Join and delivery table

Use `match_id`, `comp_id`, `team_id`, and `player_id` as keys. Keep raw columns and add normalized derived columns; never overwrite source values. Build one delivery row per source ball with competition, season, teams, player names, innings, and stage context.

Audit duplicate keys, join coverage, match availability, innings totals, legal-ball counts, wickets, revised targets, and extras. Store all exclusions and unresolved cases in `outputs/audit/` and `docs/decisions_log.md`.

Derive the following only after reconciling source semantics:

- `bat_runs`: runs credited to the striker.
- `bowler_conceded`: striker runs plus bowler-charged extras; exclude byes, leg-byes, and penalty runs.
- `is_legal`: false for wides and no-balls.
- `bowler_wicket`: dismissal credited to the bowler according to an explicit, reviewed dismissal-type mapping. Run-outs and unresolved dismissals are not bowler wickets.
- `striker_out`: striker is the dismissed player, regardless of whether the dismissal is credited to the bowler.
- `is_super_over`, `is_ipl`, `is_t20i`, `is_knockout`, and `is_dls`.

Reconcile derived innings totals to match records where possible. Stop and document any irreconcilable totals rather than silently coercing them.

### 3.2 Pre-delivery state

Within each `match_id` and innings, sort deliveries by the audited sequence fields. Calculate cumulative runs, wickets, and legal balls, then shift each cumulative value by one delivery so the state at delivery *t* contains no outcome from delivery *t* or later.

Create `score_before`, `wickets_lost_before`, `legal_balls_before`, `balls_remaining`, `innings_phase`, `current_run_rate`, and, for chases, `target`, `runs_needed`, and `required_run_rate`. Cap remaining balls to the applicable innings length and explicitly handle shortened innings and Super Overs.

### 3.3 Pressure definition

Keep **match-state pressure** separate from **delivery difficulty** and from knockout status.

- **Chases:** calculate pressure from pre-delivery required rate relative to a historical scoring benchmark for the same phase and wickets remaining, plus balls remaining. Estimate benchmarks from prior matches only. Use a continuous 0–1 weight and a high-pressure flag for reporting; define the top-quartile rule from the training/reference population and freeze it before player rankings are inspected.
- **First innings:** create a separate adversity measure from score relative to a historical expected scoring trajectory at the same phase and wickets lost. Estimate that trajectory from prior matches only. Treat phase and wickets lost as state dimensions, not as proof by themselves that a delivery was pressured.
- **Knockout:** keep as an additional tag, not as a multiplier in the primary pressure weight.

Report chase and first-innings results separately. The primary top-10 score may combine the two only if both are adequately represented and the combination rule is configured before reviewing player ranks. Give each context a defined contribution, and publish sensitivity results for chase-only and combined rankings. Do not validate pressure by requiring high-pressure balls to have lower run rates; that would confuse the situation with the player outcome being evaluated. Instead, verify state distributions, example situations, absence of look-ahead, and rank stability under alternative definitions.

### 3.4 Expected-outcome models

First build a transparent benchmark: grouped means for runs and grouped wicket rates by format, innings phase, pre-delivery state buckets, batting hand, and pace/spin. Then compare CatBoost models to these benchmarks. Keep a complex model only if it improves held-out performance and calibration.

Fit separate expected values:

\[
\hat\mu^{bat}_t = E[\text{bat\_runs}_t \mid x_t]
\]

\[
\hat\mu^{bowl}_t = E[\text{bowler\_conceded}_t \mid x_t]
\]

\[
\hat p^{wkt}_t = P(\text{bowler\_wicket}_t=1 \mid x_t)
\]

Candidate features include format/competition, season, innings phase, pre-delivery score and resource state, required rate for chases, batting hand, bowling type, and pace/spin. Exclude player IDs. Do not use post-delivery outcomes (`stroke`, dismissal result, resulting score) as predictors. Keep `delivery_style` and `bowl_speed` out of version 1.

Train format-aware models, but calculate and publish IPL and T20I scores separately. IPL is the primary ranking pool. If a combined cross-format view is shown, combine within-format standardized evidence using an explicit rule and label it secondary; never let T20I volume silently determine an IPL ranking.

### 3.5 Residuals and player scores

Generate predictions without training on the same season’s delivery outcomes used to score the player. For descriptive player ratings, use season-held-out predictions. For prospective validation, use rolling-origin splits: train only on earlier seasons and evaluate a later season. Do not describe a model trained on future seasons as prospective validation.

Per delivery residuals:

\[
r^{bat}_t = \text{bat\_runs}_t - \hat\mu^{bat}_t
\]

\[
r^{bowl}_t = \hat\mu^{bowl}_t - \text{bowler\_conceded}_t
\]

\[
r^{wkt}_t = \text{bowler\_wicket}_t - \hat p^{wkt}_t
\]

Aggregate residuals by player-match before estimating uncertainty. For player *i*, report pressure-weighted runs above expected per 100 weighted balls:

\[
S_i = 100\,\frac{\sum_t w_t r_t}{\sum_t w_t}
\]

Use the same structure for batting and bowling with their respective residuals. Keep wicket impact separate from run impact; do not add them without a justified run-value conversion. Also report pressure opportunities, normal-situation score, pressure uplift, share of pressure matches with positive impact, and match-to-match spread.

Use empirical-Bayes shrinkage or a hierarchical player effect toward a role/format mean. Model uncertainty at the player-match level, not as if every delivery were independent. Rank eligible players by a conservative posterior estimate (for example, the 20th percentile) and show the posterior mean and interval alongside it. Eligibility limits for pressure matches and weighted balls live in config and are frozen before ranks are reviewed.

## 4. Validation and sensitivity

Use chronological, match-grouped splits. For forward validation, train on earlier seasons and evaluate on a later season. Compare baseline and CatBoost using MAE/RMSE and calibration for run expectations; use Brier score or log loss plus calibration for wicket probabilities. Report results overall and within pressure cohorts.

Run predeclared sensitivity variants:

1. Continuous pressure vs binary high-pressure.
2. Chase-only vs first-innings plus chase.
3. Grouped baseline vs CatBoost residuals.
4. IPL primary vs secondary format-standardized cross-format view.
5. Equal-season vs recency-weighted score.
6. Alternative eligibility thresholds and pressure-reference windows.

Publish rank ranges and flag players whose top-10 status is unstable. SHAP is not a substitute for validation and is deferred unless it adds a clear explanation benefit.

## 5. Directory plan

```text
Phase1/
├── README.md
├── requirements.txt           # pinned runtime dependencies
├── Makefile
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/                  # six CSVs + problem statement; read-only
│   ├── interim/              # normalized delivery table
│   └── processed/            # state, predictions, residuals
├── src/superr/
│   ├── io.py
│   ├── audit.py
│   ├── deliveries.py
│   ├── tables.py
│   ├── state.py
│   ├── pressure.py
│   ├── models.py
│   ├── scoring.py
│   ├── validation.py
│   └── reporting.py
├── scripts/
│   ├── 00_audit.py
│   ├── 01_build_deliveries.py
│   ├── 02_build_state.py
│   ├── 03_score_and_validate.py
│   └── 04_make_report_assets.py
├── tests/                    # critical data-attribution and no-leakage checks
├── docs/
│   ├── data_dictionary.md
│   ├── decisions_log.md
│   └── methodology_report.docx
└── outputs/
    ├── audit/
    ├── tables/
    ├── figures/
    └── final/
        ├── top10_batters.csv
        ├── top10_bowlers.csv
        └── full_rankings.csv
```

Keep notebooks optional and exploratory; production logic belongs in `src/`. `config.yaml` stores tunable definitions, thresholds, format handling, eligibility, recency weights, sensitivity variants, and random seeds. Freeze these before inspecting player rankings. Raw data are copied once, checksummed, and never edited.

## 6. Build sequence and deliverables

1. **Audit:** loaders, schemas, joins, missing matches, extras, dismissals, Super Overs, DLS fields, legal balls, and score reconciliation. Deliver audit CSV/JSON and decision log.
2. **Delivery normalization:** construct and reconcile batter runs, bowler-conceded runs, legal balls, and wicket attribution. Deliver `deliveries.parquet`.
3. **State and pressure:** add shifted pre-delivery state; derive first-innings and chase pressure from prior-match references. Deliver `model_table.parquet` and pressure diagnostics.
4. **Expected outcomes:** grouped baselines first, then CatBoost comparison; save held-out predictions and residuals.
5. **Player scoring:** player-match aggregation, shrinkage, confidence intervals, eligibility, separate IPL/T20I scores, and rank sensitivity.
6. **Submission assets:** top-10 CSVs that match the submission form choices, full rankings, pressure distribution, top-10 interval plot, rank-stability figure, and methodology report using the brief’s required headings.

Do not start the dashboard or auction-fit layer until the Round 1 tables and report are stable.
