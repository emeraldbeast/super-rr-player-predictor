"""Build Phase 2 pre-ball match state and pressure diagnostics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from superr.io import load_config
from superr.pressure import add_pressure
from superr.state import build_match_state


def main() -> None:
    config = load_config(ROOT / "configs" / "config.yaml")
    deliveries = pd.read_parquet(ROOT / config["data"]["deliveries_file"])
    p2 = config["phase2"]
    state = build_match_state(
        deliveries,
        regular_innings_overs=int(p2["regular_innings_overs"]),
        super_over_balls=int(p2["super_over_balls"]),
    )
    model_table, diagnostic, coverage, standardized = add_pressure(
        state,
        cold_start_rate=float(p2["cold_start_par_runs_per_over"]),
        high_pressure_quantile=float(p2["high_pressure_quantile"]),
        chase_weights=p2["chase_weights"],
        first_innings_weights=p2["first_innings_weights"],
        exclude_dls_matches=bool(p2.get("exclude_dls_matches_from_primary_scoring", True)),
    )
    # Basic executable state/reconciliation assertions.
    first_delivery = model_table.groupby(["match_id", "innings"], sort=False).head(1)
    if not first_delivery["score_before"].eq(0).all() or not first_delivery["legal_balls_before"].eq(0).all():
        raise AssertionError("Opening delivery state must be zero before its outcome")
    if (model_table["pressure_weight"].dropna().lt(0) | model_table["pressure_weight"].dropna().gt(1)).any():
        raise AssertionError("Pressure weights must be between zero and one")
    if model_table.loc[model_table["is_dls"], "pressure_available"].any():
        raise AssertionError("DLS matches without revision event timing must be excluded from primary pressure scoring")
    if model_table.loc[~model_table["is_super_over"], "scheduled_legal_balls"].gt(120).any():
        raise AssertionError("T20 innings schedule exceeds 120 legal balls")

    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    model_table.to_parquet(out / "model_table.parquet", index=False)
    diagnostic.to_csv(ROOT / "outputs" / "phase2_pressure_diagnostics.csv", index=False)
    standardized.to_csv(ROOT / "outputs" / "phase2_pressure_standardized.csv", index=False)
    coverage.to_csv(ROOT / "outputs" / "phase2_reference_coverage.csv", index=False)
    summary = {
        "delivery_rows": int(len(model_table)),
        "matches": int(model_table["match_id"].nunique()),
        "pressure_available_rows": int(model_table["pressure_available"].sum()),
        "high_pressure_rows": int(model_table["high_pressure"].sum()),
        "super_over_rows_excluded_from_pressure": int(model_table["is_super_over"].sum()),
        "dls_matches_excluded_from_pressure": int(model_table.loc[model_table["is_dls"], "match_id"].nunique()),
        "maximum_scheduled_legal_balls_regular_innings": int(model_table.loc[~model_table["is_super_over"], "scheduled_legal_balls"].max()),
        "pressure_quantile": float(p2["high_pressure_quantile"]),
    }
    (ROOT / "outputs" / "phase2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Saved", out / "model_table.parquet")


if __name__ == "__main__":
    main()
