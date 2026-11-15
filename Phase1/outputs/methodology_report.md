# Rajasthan Royals: pressure-performance shortlist for IPL 2027

**Phase 4 final report | Data cut: supplied IPL and T20I ball-by-ball files through 2026**

## Executive summary

This analysis ranks players by how much better or worse they performed than the expected outcome of comparable pre-ball situations, with greater weight assigned to high-pressure deliveries. The primary lists use IPL evidence and a conservative, uncertainty-adjusted score. The submitted lists are performance targets, not a verified auction availability list: the supplied files do not identify the 2027 auction pool, RR's retained squad, or likely prices.

Primary eligibility required at least 10 pressure-exposed matches and 100 pressure-weighted opportunities. The primary pressure score uses both chase strain and first-innings adversity. Super Overs and matches with DLS revisions are excluded from scoring because the data does not provide event-time revised targets consistently enough to reconstruct fair pre-ball chase states.

## 1. Pressure definition

Pressure is reconstructed from information available before each delivery. The state contains the score, wickets lost, legal balls used, balls remaining and innings context. For chases, the index combines required-rate strain, wickets in hand and time remaining. For first innings, it combines scoring relative to a contextual par, wickets lost and innings phase. The component weights and high-pressure cut point are held in `configs/config.yaml`.

The continuous pressure weight lies from 0 to 1 and is the primary scoring weight. A separate high-pressure flag marks deliveries at or above the configured within-pool pressure threshold. A knockout tag is retained separately; it does not itself make a delivery high-pressure. Pre-ball state is used, and post-match result/DLS fields are excluded from model predictors. Super Overs are omitted. DLS matches are omitted from primary player scoring because a final revised target cannot safely stand in for the target known at each earlier ball.

Pressure diagnostics are contextualized by competition, innings context and phase rather than relying on pooled raw run rates. Expected-outcome adjustment is used to compare deliveries with similar pre-ball states; raw scoring differences between cohorts are not interpreted as a player skill effect.

## 2. Performance measurement

Three expected-outcome targets are modelled separately inside IPL and T20I pools: batter runs from a non-wide batting opportunity, runs charged to the bowler, and bowler-creditable wicket probability. Predictors are restricted to pre-ball state, competition/season context, phase, venue and player handedness/bowling style. No player identity, post-match outcome, or final DLS flag is supplied to the model.

A smoothed grouped contextual mean is the baseline. CatBoost is retained only where it improves prior-season chronological validation for that pool and target; otherwise the grouped baseline is used. Cross-fitted predictions are generated with the evaluated season wholly held out and training restricted to earlier seasons. This reduces optimistic in-sample residuals and prevents later seasons from informing earlier player scores.

For batters, the event residual is actual batter runs minus expected batter runs. For bowlers, it is expected bowler-conceded runs minus actual runs charged to the bowler. Deliveries are weighted by continuous pressure and aggregated as a ratio of weighted residual sum to weighted exposure, so matches with more pressure opportunity contribute proportionally more. The player score is expressed per 100 weighted opportunities.

Role-specific empirical-Bayes shrinkage pulls estimates toward the role mean. The prior variance is estimated from between-player dispersion after subtracting average sampling noise. Uncertainty uses a match-cluster ratio variance, rather than treating each delivery as independent. The ordering score is the 20th-percentile posterior lower bound; this favours evidence-backed contributions over a high but imprecise raw rate. Positive-match share and high-versus-ordinary-pressure uplift are supporting diagnostics, not independent selection criteria. Bowling ranks are based on runs saved; wicket uplift is reported separately.

## 3. Fair comparisons

IPL is the primary ranking pool because the decision concerns an IPL franchise. IPL and T20I are fitted and ranked separately so differences in formats, opponents, venues and opportunity rates do not silently determine one pooled leaderboard. The cross-format corroboration output standardizes each player's posterior score against the player's role distribution within each format, then averages the IPL and T20I standardized values only for players with the same source player name in both pools. Source IDs are pool-specific in some records, so exact source names are used for this corroboration join. This is a secondary sensitivity view, not the submission ranking.

Eligibility is based on weighted exposure and the number of distinct pressure-exposed matches. The primary threshold is 10 matches and 100 pressure-weighted opportunities. Sensitivities use 3/30 and 15/200 thresholds. Equal exposure-weighted aggregation is primary; a three-year recency half-life is also tested. Within a role and format, players receive the same expected-outcome baseline/model selection process and the same scoring/shrinkage procedure.

Chronological validation uses held-out seasons 2023–2026; the recent comparison below averages seasons 2024–2026. It is not a claim of future 2027 performance. Model comparison over those recent folds gives:

| Pool | Target | Grouped baseline | CatBoost | Preferred recent-fold average |
|---|---|---:|---:|---|
| IPL | Batter runs MAE | 1.395 | 1.408 | Grouped baseline |
| IPL | Bowler-conceded runs MAE | 1.355 | 1.372 | Grouped baseline |
| IPL | Bowler wicket Brier | 0.04558 | 0.04543 | CatBoost, slight edge |
| T20I | Batter runs MAE | 1.289 | 1.271 | CatBoost |
| T20I | Bowler-conceded runs MAE | 1.258 | 1.241 | CatBoost |
| T20I | Bowler wicket Brier | 0.05164 | 0.05138 | CatBoost, slight edge |

Fold-specific selection uses only validation seasons earlier than the held-out season. The table above summarizes the recent evaluation period; it does not override that leakage-safe fold selection.

## 4. Evidence for the top 10

### Batters

Scores are conservative posterior lower bounds of adjusted batter runs per 100 pressure-weighted batting opportunities. Exposure is shown to communicate how much evidence supports each estimate.

| Rank | Player | Inferred role | Pressure matches | Weighted opportunities | Lower-bound score | Positive match share | RR role contribution to assess |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | Abhishek Sharma | Top order | 53 | 118 | 33.26 | 68% | Left-handed top-order scoring option; could add powerplay aggression if roster/availability align. |
| 2 | Suryakumar Yadav | Middle order | 54 | 271 | 21.45 | 65% | High-volume middle-order option with strong adjusted pressure scoring; may cover a high-impact middle-order slot. |
| 3 | Prabhsimran Singh | Top order | 56 | 145 | 21.14 | 64% | Top-order option with positive adjusted pressure output across many matches. |
| 4 | Heinrich Klaasen | Middle order | 54 | 341 | 18.31 | 69% | Large pressure sample and stable positive contribution; potential middle-order finishing / spin-hitting profile. |
| 5 | Ashutosh Sharma | Lower order | 26 | 148 | 17.55 | 62% | Lower-order hitter with positive pressure-adjusted output; estimate is less certain due to fewer matches. |
| 6 | Shubman Gill | Top order | 58 | 210 | 16.66 | 53% | Broad top-order sample and positive conservative score; could provide a top-order anchor/attacking option. |
| 7 | Shreyas Iyer | Middle order | 43 | 174 | 14.46 | 53% | Middle-order coverage supported by a multi-season sample, though the match-level positive share is close to even. |
| 8 | Rajat Patidar | Middle order | 39 | 146 | 14.35 | 62% | Positive adjusted middle-order contribution and a useful role fit if a middle-order slot is needed. |
| 9 | Rinku Singh | Middle order | 46 | 284 | 14.31 | 48% | Highest-volume evidence among the lower half of this list; high-versus-ordinary uplift is strong, but match consistency is mixed. |
| 10 | Naman Dhir | Top order | 31 | 126 | 13.05 | 58% | Top-order candidate with a smaller but positive pressure sample; worth monitoring against roster needs. |

### Bowlers

Scores are conservative posterior lower bounds of bowler-conceded runs saved per 100 pressure-weighted legal deliveries. Positive wicket uplift is a supporting, separate diagnostic and does not determine the runs-saved ranking.

| Rank | Player | Bowling type | Pressure matches | Weighted legal deliveries | Lower-bound score | Positive match share | RR role contribution to assess |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | Jasprit Bumrah | Right-arm fast | 36 | 210 | 32.06 | 81% | Elite pressure overs profile in the data; could anchor death/high-leverage pace overs if available. |
| 2 | Sunil Narine | Right-arm offbreak | 52 | 289 | 20.16 | 73% | Experienced spin option with a large pressure sample; potential control and matchup value. |
| 3 | Matheesha Pathirana | Right-arm medium | 30 | 272 | 18.91 | 70% | Pressure overs profile with meaningful exposure; could address a specialist pace/death role. |
| 4 | Axar Patel | Slow left-arm orthodox | 52 | 197 | 3.77 | 60% | Spin-bowling role with a substantial sample; positive conservative run-saving estimate, wicket uplift is negative. |
| 5 | Krunal Pandya | Slow left-arm orthodox | 52 | 246 | 2.69 | 52% | Spin/all-round role coverage; estimate is positive but close to neutral and uncertain. |
| 6 | Mohammed Siraj | Right-arm medium-fast | 59 | 309 | 0.85 | 51% | Large pressure sample and slightly positive conservative score; use as pace-depth consideration, not an elite-score claim. |
| 7 | Noor Ahmad | Left-arm Chinaman | 50 | 283 | 0.23 | 52% | Wrist-spin variation with a broad sample and positive wicket uplift; run-saving score is near neutral. |
| 8 | Ravindra Jadeja | Slow left-arm orthodox | 55 | 242 | -0.91 | 51% | All-round/spin profile; broad sample, but adjusted pressure run saving is slightly negative on the conservative bound. |
| 9 | Kuldeep Yadav | Left-arm Chinaman | 50 | 272 | -1.42 | 58% | Wrist-spin matchup option with positive match share; conservative run-saving bound remains slightly negative. |
| 10 | Eshan Malinga | Right-arm fast-medium | 22 | 148 | -2.88 | 55% | Pace option with fewer pressure matches; wicket uplift is positive, but the lower-bound run-saving score is negative and uncertain. |

These role descriptions are inferred from the supplied player/ball-by-ball fields. They do not assert that any player will be available or that RR has a vacancy in that role. Auction availability, overseas-player slots, purse, age/injury information and the 2027 retained squad must be checked before converting this evidence shortlist into bids.

## 5. Reliability and limitations

The primary IPL top tens are fully reproducible under the same primary specification. Sensitivity analyses show where ordering depends on assumptions:

| IPL sensitivity | Batter top-10 overlap vs primary | Batter Spearman | Bowler top-10 overlap vs primary | Bowler Spearman |
|---|---:|---:|---:|---:|
| Baseline expectations only | 8/10 | 0.944 | 8/10 | 0.891 |
| Binary high-pressure weighting | 5/10 | 0.805 | 6/10 | 0.848 |
| Chase-only pressure | 3/10 | 0.733 | 6/10 | 0.695 |
| Three-year recency half-life | 8/10 | 0.972 | 10/10 | 0.955 |
| Relaxed eligibility (3 matches / 30 opportunities) | 4/10 | 1.000 among shared players | 6/10 | 1.000 among shared players |
| Strict eligibility (15 matches / 200 opportunities) | 4/10 | 1.000 among shared players | 8/10 | 1.000 among shared players |

Eligibility variants change who qualifies; their Spearman values describe ordering only among players eligible in both lists. They should not be read as evidence that membership is unchanged. Binary and chase-only variants also have lower pressure exposure and fewer eligible players, so their lower overlap partly reflects less information. The cross-format standardized view is available in `phase4_variant_rankings.csv`; it is restricted to exact source-name matches between pools and is corroborative, not the auction list.

Across held-out seasons, high-pressure predictions are generally harder than ordinary-pressure predictions. The 2023–2026 mean MAE was 1.406 vs 1.367 for IPL batter runs, 1.357 vs 1.336 for IPL bowler-conceded runs, 1.276 vs 1.257 for T20I batter runs, and 1.244 vs 1.235 for T20I bowler-conceded runs. High-pressure wicket Brier score averaged 0.0584 vs 0.0413 for IPL and 0.0670 vs 0.0454 for T20I. This confirms that the high-pressure cohort is harder to predict; it does not invalidate scoring because player residuals are compared with expected outcomes, but uncertainty should remain visible.

Important limits:

- No verified 2027 auction list, RR roster/retentions, purse, salary estimates, injury availability, or dropdown mapping was supplied. These are performance targets pending operational checks.
- DLS matches are excluded from player scoring because event-time revised targets are incomplete; their omission may affect players with distinctive DLS exposure.
- Super Overs are excluded because their format and tiny samples are not comparable with regular innings.
- The available match history ends in 2026; roles, form and availability can change before the 2027 auction.
- Data is ball-by-ball and may contain source attribution errors. Phase 1 reconciliation and exception tables are retained in `outputs/audit/`.
- Pressure residuals measure performance relative to modelled delivery expectation, not total player value, replacement value, price, fielding, leadership, fitness, or strategic fit.
- Role inference from historical batting order and bowling type is approximate. Cross-format evidence helps assess repeatability but cannot replace IPL-specific matchup and role analysis.

## Reproducibility and handoff

Run from `Phase1/` after installing `requirements.txt` and making the project's dependency environment available:

```powershell
python scripts/05_phase4_robustness.py
```

Key outputs: `top10_batters.csv`, `top10_bowlers.csv`, `phase4_variant_rankings.csv`, `phase4_rank_stability.csv`, `phase4_validation_by_pressure_cohort.csv`, `phase4_wicket_reliability.csv`, `phase4_recent_model_comparison.csv`, and PNG figures under `outputs/figures/`. The final submission CSVs contain the rank and exact source player name only; the full ranking and exposure evidence remain in the Phase 3 ranking tables.
