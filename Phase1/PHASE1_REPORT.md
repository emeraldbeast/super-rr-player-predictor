# Phase 1 Report — Data Foundation and Audit

**Status:** Complete with named source-data exceptions  
**Primary output:** `data/interim/deliveries.parquet`  
**Audit summary:** `outputs/audit/audit_summary.json`  
**Run record:** `outputs/audit/run_manifest.json`

## 1. Files and reproducibility

The six source CSVs and problem-statement PDF are copied to `data/raw/`; the original `Dataset/` files remain untouched. `checksums.txt` records SHA-256 values. Every loader call verifies those hashes before reading inputs and fails on a mismatch.

Implemented:

- `src/superr/io.py`: typed CSV loading, YAML config loading, checksum verification, Parquet writing.
- `src/superr/deliveries.py`: competition joins, delivery outcomes, dismissal attribution, DLS/pool flags, and delivery ordering.
- `src/superr/tables.py`: player, innings, and player-match table builders.
- `src/superr/audit.py`: coverage, result-margin, over-structure, roster, and column-policy audits; writes the run manifest.
- `scripts/00_audit.py` and `scripts/01_build_deliveries.py`: both load `configs/config.yaml`; the build uses the settings for extras, wicket credit and DLS tagging. Speed remains in the raw source audit but is dropped from the analytical delivery table.
- `tests/`: fixture tests and data-level integration checks against the generated Parquet and audit tables.

The run manifest stores the resolved config, verified input checksums, and runtime versions: Python 3.13.0, pandas 2.3.3, PyArrow 26.0.1 and PyYAML 6.0.3.

## 2. Competition coverage

The 401 matches labelled `T20` include 30 non-IPL T20 matches. The ranking pools are therefore separated as follows:

| Pool | Matches | With delivery data | Delivery rows | Policy |
|---|---:|---:|---:|---|
| IPL | 371 | 367 | 87,687 | Primary pool |
| T20I | 482 | 470 | 106,327 | Separate supporting pool |
| Other T20 | 30 | 30 | 6,985 | Separate; excluded from primary ranking |
| **Total** | **883** | **867** | **200,999** | |

The 16 matches without delivery records are listed in `outputs/audit/matches_missing_deliveries.csv`; all are marked abandoned in the match file. Four are IPL matches. The report no longer labels all 401 T20 matches as IPL.

The feed has 1,724 innings groups, including 12 Super Over innings (six matches, 38 delivery rows). Super Overs are flagged and should remain outside normal innings scoring. There are 22 matches with only one recorded normal innings: 19 `No Result` and 3 `Match Abandoned`. They are listed in `outputs/audit/matches_with_incomplete_innings.csv` so later chase logic can distinguish incomplete matches.

## 3. Delivery attribution and ordering

### Runs and extras

The source totals by extras type are in `outputs/audit/extras_reconciliation.csv`. The output derives:

- `team_runs = runs + extras1_runs + extras_2_runs`.
- `bat_runs = runs`.
- `bowler_conceded = bat_runs + extras1_runs` on bowler-charged categories, with `byes`, `legbyes`, and `Penalty` extras excluded. `extras_2_runs` stays in team totals but is **not charged to the bowler** in the primary output.

There are 36 non-zero `extras_2_runs` rows, totalling 84 runs. Thirty-five are on `Overstep`/`other` rows and one has no extras type; all have `runs=0`, with extra values 1 or 4. The source does not define this field. The primary assumption is to exclude it from bowler charges, as this pattern is consistent with uncharged extras; `bowler_charged_extras2_assumption: true` remains available in config as a sensitivity. All 36 rows carry `extras2_attribution_unverified=True`.

The primary delivery table sums to 284,257 team runs and 279,565 bowler-conceded runs under that assumption.

`wide` is illegal. `Overstep` is a no-ball. All 264 `other` rows have one run in `extras1_runs`; treating them as no-balls keeps regular innings within the 120-legal-ball cap. The two rows with no extras type but a positive extra amount remain flagged; one with `extras1_runs=1` is treated as an inferred no-ball. `Penalty` remains legal and is not charged to the bowler.

### Wickets and batter-out flags

`striker_out` and `nonstriker_out` require both `is_wicket=True` and a matching dismissed-player ID. `bowler_wicket` uses the config list (Bowled, Caught, LBW, Stump, Hit Wicket). Run Outs, Retire Outs, and Obstructing the field are not credited to the bowler.

`batter_failure` excludes `Retire Out`. Twelve rows have a dismissed-player ID with `is_wicket=False` and no dismissal label; these are flagged as `retired_not_out` candidates, not automatically treated as known retirements or batter failures. One `is_wicket=True` row has no dismissed-player ID. The source contains 10,881 wicket events, 10,272 bowler-credited wickets, 10,606 striker-out flags, and 274 non-striker-out flags. The 12 untyped, non-wicket rows no longer create false striker dismissals.

`Obstracting the field` is normalized to `Obstructing the field`; `RunOut` is normalized to `Run Out`. Original values remain available.

### Delivery sequence and over structure

The source `innings_ball_num` counts delivery rows, including wides and no-balls; it is not a legal-ball count. It is unique and contiguous from 1 to N within every match innings. `delivery_id` is a deterministic global sequence after sorting by match, innings, `innings_ball_num`, `overid`, and `over_ball_num`. `legal_ball_no` is the within-innings cumulative legal-ball count; illegal deliveries do not increment it. `delivery_ist_time` is not used for ordering.

An over-structure audit checks each non-final over for six legal deliveries. It finds five named exceptions:

| Match | Innings | Over | Legal balls | Finding |
|---:|---:|---:|---:|---|
| 14130 | 1 | 14 | 5 | Short non-final over |
| 14130 | 2 | 14 | 5 | Short non-final over |
| 14164 | 1 | 14 | 5 | Short non-final over |
| 36097 | 2 | 8 | 5 | Short non-final over |
| 41509 | 1 | 12 | 7 | Seventh legal ball has no extras flag |

The exceptions are preserved in `outputs/audit/over_structure.csv`; later phases should not silently normalize them. All regular innings are at or below their scheduled legal-ball limit, and all innings have at most ten `is_wicket` events.

## 4. Independent score and result checks

Every innings’ derived team-run total equals the sum of the three source run fields. In addition, result text is parsed case-insensitively across variants such as `Won By`, `Wkts`, `17runs`, `D/L`, and `DLS`, then compared with delivery-derived run or wicket margins. DLS results are retained but excluded from direct-margin accuracy because revised targets are not always supplied.

- 803 non-DLS result margins were directly checkable.
- 802 matched exactly; one did not: **match 20190**. The result says Lucknow Super Giants won by 6 wickets, while the second-innings delivery rows show 5 wickets lost (5 remaining). This exception is in `outputs/audit/result_margin_reconciliation.csv`.
- Exact agreement is **99.875%**, above the configured 99% threshold.

The result text contains 19 D/L or DLS matches. Thirty-four matches have revised-target or revised-overs fields; the union gives 38 matches flagged as DLS. Four DLS result-text matches lack all revised fields: 20182, 23674, 24822 and 28334. The additional 23674 case was not in the initial review note and is listed in `outputs/audit/dls_text_without_revision_fields.csv`.

## 5. Players, roster and derived tables

Created Parquet tables:

| Table | Rows | Key fields |
|---|---:|---|
| `data/interim/deliveries.parquet` | 200,999 | delivery and match outcomes, sequence IDs, pool and policy fields |
| `data/interim/innings.parquet` | 1,724 | batting/bowling teams, total, wickets, legal balls, target, termination reason |
| `data/interim/player_match.parquet` | 28,347 | player participation, batted/bowled, balls faced/bowled, played-XI flag |
| `data/interim/players.parquet` | 1,217 | one row per player ID and disambiguated display name |

Names are never used as player keys. Seven player names map to multiple IDs: Abdul Samad, Bashir Ahmad, Basil Hameed, Haider Ali, Mohammad Ali, Muhammad Farooq, and Rashid Khan. `display_name` includes the player name, latest associated team, and player ID so these identities remain distinct. There are no multiple name variants within a single player ID.

There are 14 event-role slots not marked in the played XI, across six distinct player-match keys: 6 striker, 5 non-striker, and 3 bowler roles. The audit lists them in `participant_role_exceptions.csv`. The playing-eleven source also has 75 repeated rows across 35 match/team/player keys; the normalized player-match table collapses those to one key and the repeated rows are retained in the audit. The maximum observed number of distinct played slots per team is 13.

## 6. Feature policy and data-quality exclusions

`outputs/audit/delivery_column_policy.csv` labels every delivery column as `pre-ball safe`, `outcome`, or `post-ball` and gives a short use note. Stroke, delivery-style, speed, result, and other post-delivery fields must not enter pre-ball expected-outcome predictors.

The source speed field is audited from the raw CSV, then `bowl_speed` and its raw/cleaned variants are dropped from the analytical delivery table. The speed audit identifies 77,882 unusable values (77,861 zeros, one negative value, and 20 values above 160). There are 558 `is_impact` and 558 `is_impact_off` flags in the roster file; these remain roster context.

### Reviewed correction: revised DLS schedules

The source fields named `first_inn_revised_overs` and `second_inn_revised_overs` contain revised legal-ball counts. Both Phase 1 and Phase 2 now use those values directly; ordinary T20 innings default to 120 legal balls and Super Overs to 6. The innings table and audit were regenerated. Matches 13612/innings 2 and 21586/innings 2 now use limits of 114 and 90 legal balls, respectively, and the complete innings table has no regular schedule above 120. The primary Phase 2/3 scoring population excludes all DLS-tagged matches because event-time target/overs revisions cannot be reconstructed from this feed.

## 7. Tests and run status

Run from `Phase1/` after installing the pinned dependencies:

```powershell
python scripts/00_audit.py
python scripts/01_build_deliveries.py
python -m unittest discover -s tests -v
python scripts/package_phase1.py
```

**Test result:** the updated full suite passes 21 tests. It includes fixture and data-level checks for extras, dismissals, run/ball attribution, revised legal-ball limits, target-reached termination, bowl-speed removal, competition coverage, DLS exclusion, model feature leakage, batter wide-population alignment, rolling-origin predictions, pressure-state invariance, and ratio-of-weighted-sums scoring.

## 8. Gate

**Phase 1 passes with named source exceptions.** The pipeline verifies input checksums, writes the four analytical tables, and generates result and over-structure audit tables. Remaining data caveats are the undocumented meaning of `extras_2_runs`, four result-text DLS matches without revision fields, and the specific score/wicket/over and roster exceptions listed above. DLS matches are excluded from primary pressure/player scoring.
