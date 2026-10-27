"""Verify PACKAGE_SHA256SUMS.txt on Windows, Linux, or macOS.

This uses only Python's standard library, so it does not require the
``sha256sum`` command that is normally available on Linux but not in a
standard Windows PowerShell installation.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_entries(manifest: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line_number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            expected, relative = line.split(None, 1)
        except ValueError as exc:
            raise ValueError(f"Malformed manifest line {line_number}: {raw!r}") from exc
        relative = relative.lstrip("*")
        entries.append((expected.lower(), relative.replace("/", "/")))
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PACKAGE_ROOT / "PACKAGE_SHA256SUMS.txt",
        help="checksum manifest; defaults to the manifest beside this script",
    )
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    root = manifest.parent
    failures = 0
    checked = 0
    for expected, relative in read_entries(manifest):
        path = root.joinpath(*relative.replace("\\", "/").split("/"))
        if not path.is_file():
            print(f"MISSING {relative}")
            failures += 1
            continue
        actual = sha256(path)
        checked += 1
        if actual == expected:
            print(f"OK {relative}")
        else:
            print(f"MISMATCH {relative}\n  expected {expected}\n  actual   {actual}")
            failures += 1
    print(f"Checked {checked} files; failures {failures}.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
