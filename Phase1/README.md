# Super RR Pressure Analysis

Data preparation and Phase 2/3 modelling for the Rajasthan Royals pressure-performance project.

## Environment and data

Install Python dependencies with `python -m pip install -r requirements.txt`. Unzip the supplied pack into `data/raw/`; its six CSVs and `Problem_Statement.pdf` must match `data/raw/checksums.txt`.

## Run

Run Phase 1 with `python scripts/00_audit.py` followed by `python scripts/01_build_deliveries.py`. Then run `python scripts/run_phase2_3.py` to build pre-ball state and pressure, compare expected-outcome models, and export player scores.

Run checks with `python -m unittest discover -s tests -v`.

Main outputs:

- `data/processed/model_table.parquet`: Phase 2 match state and pressure weights.
- `outputs/phase2_pressure_diagnostics.csv` and `outputs/phase2_pressure_standardized.csv`: stratified and standardized pressure cohort checks.
- `outputs/phase3_validation_metrics.csv`: forward-season model comparison.
- `outputs/top10_batters.csv` and `outputs/top10_bowlers.csv`: stricter IPL rankings (10 pressure matches and 100 weighted balls minimum).
- `outputs/phase23_run_manifest.json`: config, input, source, and artifact hashes for the run.
- `PHASE2_3_REPORT.md`: methodology, results, and limitations.

The packaging script excludes `.venv/`, `.deps/`, caches, and VCS metadata.
