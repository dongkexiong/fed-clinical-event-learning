"""Verify external experiment inputs without copying them into the repository."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path, help="directory containing selected Synthea CSV files")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PACKAGE_ROOT / "INPUT_DATA_MANIFEST.json",
        help="input manifest; defaults to the manifest beside this script",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    expected = manifest["selected_input"]["files"]
    failed = False
    for name, wanted in expected.items():
        path = args.input_dir / name
        if not path.is_file():
            print(f"MISSING {name}")
            failed = True
            continue
        actual = sha256(path)
        status = "OK" if actual == wanted else "MISMATCH"
        print(f"{status} {name} {actual}")
        failed |= actual != wanted
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

