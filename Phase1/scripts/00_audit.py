from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from superr.audit import run_audit
from superr.io import load_config, load_csvs
from superr.audit import write_run_manifest


def main() -> None:
    config = load_config(ROOT / "configs" / "config.yaml")
    tables = load_csvs(ROOT / "data" / "raw", verify=True)
    outputs = run_audit(
        tables,
        ROOT / "outputs" / "audit",
        config=config,
        interim_dir=ROOT / "data" / "interim",
    )
    write_run_manifest(ROOT / "data" / "raw", ROOT / "outputs" / "audit" / "run_manifest.json", config)
    summary = (ROOT / "outputs" / "audit" / "audit_summary.json").read_text(encoding="utf-8")
    print(summary)
    print(f"Wrote {len(outputs)} audit tables to {ROOT / 'outputs' / 'audit'}")


if __name__ == "__main__":
    main()
