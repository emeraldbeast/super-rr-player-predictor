"""Create Phase1.zip without local environments, caches, or VCS metadata."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT.parent / "Phase1.zip"
TEMP_ARCHIVE = ROOT.parent / "Phase1.zip.tmp"
EXCLUDED_PARTS = {".venv", ".deps", ".git", "__pycache__", ".pytest_cache"}


def main() -> None:
    if TEMP_ARCHIVE.exists():
        TEMP_ARCHIVE.unlink()
    with ZipFile(TEMP_ARCHIVE, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            if any(part in EXCLUDED_PARTS for part in relative.parts):
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            archive.write(path, Path("Phase1") / relative)
    TEMP_ARCHIVE.replace(ARCHIVE)
    print(f"Created {ARCHIVE} ({ARCHIVE.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
