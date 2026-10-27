"""Dependency-free structural validator for generated synthetic artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

SITE_RE = re.compile(r"^logical-site-[0-9]+$")


def _required(row: Mapping[str, Any], names: Iterable[str], errors: List[str], line: int) -> None:
    for name in names:
        if name not in row:
            errors.append(f"line {line}: missing {name}")


def validate_site_row(row: Mapping[str, Any], line: int) -> List[str]:
    errors: List[str] = []
    variant = row.get("schema_variant")
    specs = {
        "canonical": ("patient_id", "event_id", "site_id", "event_type", "concept_code", "concept_system", "target_label"),
        "wide": ("patientId", "eventId", "site_id", "eventType", "local_code", "code_system", "target_label"),
        "offset": ("patient_id", "event_id", "site_id", "type", "code", "system", "target_label"),
    }
    if variant not in specs:
        errors.append(f"line {line}: invalid schema_variant {variant!r}")
        return errors
    _required(row, specs[variant], errors, line)
    if not isinstance(row.get("site_id"), str) or not SITE_RE.fullmatch(row.get("site_id", "")):
        errors.append(f"line {line}: invalid site_id")
    if "gold_patient_id" in row or "gold_event_id" in row or "duplicate_of" in row:
        errors.append(f"line {line}: generator truth leaked into site artifact")
    if row.get("target_label") not in (0, 1):
        errors.append(f"line {line}: target_label must be 0 or 1")
    return errors


def validate_truth_row(row: Mapping[str, Any], line: int) -> List[str]:
    errors: List[str] = []
    _required(row, ("site_id", "event_id", "gold_patient_id", "gold_event_id", "target_label"), errors, line)
    if not isinstance(row.get("site_id"), str) or not SITE_RE.fullmatch(row.get("site_id", "")):
        errors.append(f"line {line}: invalid truth site_id")
    if row.get("target_label") not in (0, 1):
        errors.append(f"line {line}: truth target_label must be 0 or 1")
    return errors


def validate_jsonl(path: Path, truth: bool = False) -> Dict[str, Any]:
    errors: List[str] = []
    records = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            records += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON ({exc.msg})")
                continue
            if not isinstance(row, dict):
                errors.append(f"line {line_number}: row is not an object")
                continue
            errors.extend(validate_truth_row(row, line_number) if truth else validate_site_row(row, line_number))
    return {"path": str(path), "records": records, "valid": not errors, "errors": errors[:100]}


def validate_directory(directory: Path) -> Dict[str, Any]:
    directory = Path(directory)
    site_reports = [validate_jsonl(path) for path in sorted((directory / "sites").glob("site_*.jsonl"))]
    truth = validate_jsonl(directory / "gold_truth.jsonl", truth=True)
    return {"valid": bool(site_reports) and all(report["valid"] for report in site_reports) and truth["valid"], "sites": site_reports, "truth": truth}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_directory(args.artifact_dir)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
