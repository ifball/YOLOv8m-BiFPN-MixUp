"""Apply the BiFPN / attention-module patches to the installed ultralytics package.

The patches were prepared against ultralytics 8.4.37. The script copies the
patched files over the installed package and keeps a `.orig` backup of every
replaced file (only on the first run, so the original files are never lost).

Usage:
    python ultralytics_patch/apply_patch.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import ultralytics

PATCH_VERSION = "8.4.37"


def main() -> None:
    patch_root = Path(__file__).resolve().parent / PATCH_VERSION / "ultralytics"
    pkg_root = Path(ultralytics.__file__).resolve().parent

    print(f"Installed ultralytics: {pkg_root}")
    print(f"Reported version:      {ultralytics.__version__}")
    if ultralytics.__version__ != PATCH_VERSION:
        print(
            f"WARNING: the patches were prepared for ultralytics {PATCH_VERSION} "
            f"but {ultralytics.__version__} is installed. Review the diff before continuing."
        )

    files = sorted(patch_root.rglob("*.py"))
    if not files:
        raise SystemExit(f"No patch files found under {patch_root}")

    for src in files:
        rel = src.relative_to(patch_root)
        dst = pkg_root / rel
        if not dst.exists():
            raise SystemExit(f"Target file does not exist, aborting: {dst}")

        backup = dst.with_suffix(dst.suffix + ".orig")
        if not backup.exists():
            shutil.copy2(dst, backup)
            print(f"  backup  {backup.name}")
        shutil.copy2(src, dst)
        print(f"  patched {rel.as_posix()}")

    print("Done. Restart Python so the patched modules take effect.")


if __name__ == "__main__":
    main()
