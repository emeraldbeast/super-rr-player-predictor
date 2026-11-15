"""Typed CSV loading helpers for the supplied Super RR data pack."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


CSV_FILES = {
    "competitions": "competitions.csv",
    "teams": "teams.csv",
    "matches": "matches.csv",
    "playing_eleven": "playing_eleven.csv",
    "player_mapping": "competition_team_player_mapping.csv",
    "ball_by_ball": "ball_by_ball.csv",
}


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and minimally validate the project YAML configuration."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {config_path}")
    return config


def verify_checksums(raw_dir: str | Path) -> dict[str, str]:
    """Verify each raw input against checksums.txt and return verified hashes."""
    raw_dir = Path(raw_dir)
    checksum_file = raw_dir / "checksums.txt"
    if not checksum_file.is_file():
        raise FileNotFoundError(f"Required checksum manifest not found: {checksum_file}")
    expected: dict[str, str] = {}
    for line_number, line in enumerate(checksum_file.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Malformed checksum entry on line {line_number}: {line!r}")
        expected[parts[1].strip()] = parts[0].strip().upper()

    required = set(CSV_FILES.values()) | {"Problem_Statement.pdf"}
    missing_entries = required - set(expected)
    if missing_entries:
        raise ValueError(f"Checksum manifest omits required inputs: {sorted(missing_entries)}")

    verified: dict[str, str] = {}
    for filename, expected_hash in expected.items():
        path = raw_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Checksummed raw input not found: {path}")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        if actual_hash != expected_hash:
            raise ValueError(f"Checksum mismatch for {filename}: expected {expected_hash}, got {actual_hash}")
        verified[filename] = actual_hash
    return verified


def load_csvs(raw_dir: str | Path, *, verify: bool = True) -> dict[str, pd.DataFrame]:
    """Load all six CSVs, verifying immutable raw inputs by default."""
    raw_dir = Path(raw_dir)
    if verify:
        verify_checksums(raw_dir)
    tables: dict[str, pd.DataFrame] = {}
    for key, filename in CSV_FILES.items():
        path = raw_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")
        tables[key] = pd.read_csv(path, low_memory=False)

    matches = tables["matches"]
    matches["match_date"] = pd.to_datetime(matches["match_date"], errors="coerce")

    for col in ("is_four", "is_six", "is_wicket"):
        tables["ball_by_ball"][col] = tables["ball_by_ball"][col].astype("boolean")
    tables["playing_eleven"]["is_played"] = tables["playing_eleven"]["is_played"].astype("boolean")

    # Parse source values robustly while retaining nulls in the raw input.
    for col in ("is_captain", "is_wk"):
        if col in tables["playing_eleven"]:
            tables["playing_eleven"][col] = tables["playing_eleven"][col].map(_nullable_bool).astype("boolean")
    return tables


def _nullable_bool(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return pd.NA


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Write Parquet and raise a clear message if no engine is installed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        raise RuntimeError(
            "Writing deliveries.parquet requires pyarrow or fastparquet. "
            "Install project dependencies with: python -m pip install -r requirements.txt"
        ) from exc
