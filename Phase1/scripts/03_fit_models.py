"""Fit chronological expected-outcome models and create cross-fitted predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from superr.expected import add_model_features, chronological_validation, leave_one_season_out_predictions
from superr.io import load_config


def main() -> None:
    config = load_config(ROOT / "configs" / "config.yaml")
    table = pd.read_parquet(ROOT / "data" / "processed" / "model_table.parquet")
    table = add_model_features(table)
    validation, validation_predictions = chronological_validation(table, config["phase3"])
    out = ROOT / "outputs"
    validation.to_csv(out / "phase3_validation_metrics.csv", index=False)
    validation_predictions.to_parquet(ROOT / "data" / "processed" / "chronological_validation_predictions.parquet", index=False)
    predictions, choices = leave_one_season_out_predictions(table, config["phase3"], validation)
    validation_years = sorted(int(y) for y in validation["test_season"].dropna().unique()) if not validation.empty else []
    report_years = validation_years[-int(config["phase3"].get("validation_last_seasons", 3)):]
    reported_validation = validation.loc[validation["test_season"].isin(report_years)]
    predictions.to_parquet(ROOT / "data" / "processed" / "expected_outcomes_oof.parquet", index=False)
    choices.to_csv(out / "phase3_model_choices.csv", index=False)
    summary = {
        "rows_with_expected_outcomes": int(len(predictions)),
        "validated_fold_rows": int(len(validation_predictions)),
        "validation_folds": int(validation[["pool", "test_season"]].drop_duplicates().shape[0]) if not validation.empty else 0,
        "reported_validation_seasons": report_years,
        "catboost_selected_share": float(choices["selected_model"].eq("catboost").mean()) if not choices.empty else 0,
        "model_choice_rows": int(len(choices)),
    }
    (out / "phase3_model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Validation metrics:")
    print(reported_validation.groupby(["pool", "target", "model"])[["mae", "rmse", "brier", "log_loss"]].mean(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
