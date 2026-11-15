"""Run Phases 2/3 and write a config/data/source-bound run manifest."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import catboost
import numpy
import pandas

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest() -> None:
    config = ROOT / "configs" / "config.yaml"
    data = ROOT / "data" / "interim" / "deliveries.parquet"
    source_files = (
        sorted((ROOT / "src" / "superr").glob("*.py"))
        + sorted((ROOT / "scripts").glob("*.py"))
        + sorted((ROOT / "tests").glob("*.py"))
    )
    config_hash = sha256(config)
    data_hash = sha256(data)
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in source_files}
    artifacts = [
        "data/processed/model_table.parquet",
        "data/processed/expected_outcomes_oof.parquet",
        "data/processed/chronological_validation_predictions.parquet",
        "outputs/phase2_pressure_diagnostics.csv",
        "outputs/phase2_pressure_standardized.csv",
        "outputs/phase3_validation_metrics.csv",
        "outputs/phase3_model_choices.csv",
        "outputs/phase3_high_pressure_uplift.csv",
        "outputs/phase3_batting_candidate_rankings.csv",
        "outputs/phase3_bowling_candidate_rankings.csv",
        "outputs/top10_batters.csv",
        "outputs/top10_bowlers.csv",
    ]
    artifact_hashes = {relative: sha256(ROOT / relative) for relative in artifacts}
    run_material = json.dumps({"config": config_hash, "data": data_hash, "sources": source_hashes}, sort_keys=True).encode()
    manifest = {
        "run_id": hashlib.sha256(run_material).hexdigest()[:16],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config.relative_to(ROOT)),
        "config_sha256": config_hash,
        "deliveries_path": str(data.relative_to(ROOT)),
        "deliveries_sha256": data_hash,
        "source_sha256": source_hashes,
        "artifact_sha256": artifact_hashes,
        "runtime": {
            "python": sys.version,
            "pandas": pandas.__version__,
            "numpy": numpy.__version__,
            "catboost": catboost.__version__,
        },
    }
    (ROOT / "outputs" / "phase23_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote Phase 2/3 run manifest {manifest['run_id']}")


def main() -> None:
    if "--manifest-only" not in sys.argv:
        for script in ("02_build_pressure.py", "03_fit_models.py", "04_score_players.py"):
            print(f"\n=== {script} ===", flush=True)
            subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True)
    write_manifest()


if __name__ == "__main__":
    main()
