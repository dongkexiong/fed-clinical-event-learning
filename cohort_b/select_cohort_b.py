"""Select cohort B with the same per-patient caps used for the reported cohort A.

Usage:
    python select_cohort_b.py [--source RAW_CSV_DIR] [--output SELECTED_DIR]
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / "synthea_raw" / "csv"
DEFAULT_OUTPUT = HERE / "selected_csv"

CAPS = {
    "conditions.csv": 10,
    "observations.csv": 25,
    "encounters.csv": 10,
    "medications.csv": 10,
    "procedures.csv": 10,
    "allergies.csv": 5,
    "immunizations.csv": 5,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="raw Synthea CSV directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="directory for the selected CSV files")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, cap in CAPS.items():
        seen = defaultdict(int)
        kept = 0
        with (args.source / name).open("r", encoding="utf-8-sig", newline="") as src, \
             (args.output / name).open("w", encoding="utf-8", newline="") as dst:
            reader = csv.DictReader(src)
            writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                patient = row["PATIENT"]
                retain_target = name == "conditions.csv" and "hypertension" in row["DESCRIPTION"].casefold()
                if seen[patient] < cap or retain_target:
                    writer.writerow(row)
                    kept += 1
                seen[patient] += 1
        stats[name] = kept
        print(f"{name}: kept {kept}")

    rows = list(csv.DictReader((args.source / "patients.csv").open(encoding="utf-8-sig")))
    with (args.output / "patients.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"patients.csv: {len(rows)} patients")

    positive = set()
    for row in csv.DictReader((args.output / "conditions.csv").open(encoding="utf-8-sig")):
        if "hypertension" in row["DESCRIPTION"].casefold():
            positive.add(row["PATIENT"])
    stats["patients.csv"] = len(rows)
    stats["positive_patients"] = len(positive)
    (args.output / "selection_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print("positive patients:", len(positive))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
