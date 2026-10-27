"""Check audit completion and selected numerical results reported in the paper."""
import argparse
import json
import math
from pathlib import Path
import sys


EXPECTED = {
    "raw": (0.5979, 0.8283, 0.6269),
    "normalized_events": (0.6693, 0.8742, 0.7059),
    "alignment_aware": (0.6567, 0.8714, 0.7012),
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def check(output):
    audit = read_json(output / "integrity_audit.json")
    if audit["passed"] is not True:
        raise ValueError("integrity_audit.json: passed must be true")
    for field, expected in (("runs_checked", 15), ("sites_checked", 75),
                            ("input_csv_hashes_checked", 8)):
        if type(audit[field]) is not int or audit[field] != expected:
            raise ValueError(f"Audit {field}: expected {expected}, got {audit[field]!r}")
    result = read_json(output / "missing_10" / "summary.json")
    if result["seeds"] != [11, 22, 33, 44, 55]:
        raise ValueError("Unexpected experiment seeds")
    for mode, paper_values in EXPECTED.items():
        metrics = result["summary"]["federated_metrics"][mode]["fedavg"]
        actual = []
        for metric in ("f1", "auroc", "auprc"):
            value = metrics[metric]["mean"]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Invalid mean: {mode}/{metric}: {value!r}")
            actual.append(round(value, 4))
        if tuple(actual) != paper_values:
            raise ValueError(f"{mode}: F1/AUROC/AP {tuple(actual)} != paper {paper_values}")
        print(f"OK {mode}: F1/AUROC/AP = {tuple(actual)}")
    print("PASS: audit counts and selected paper values match.")
    print("Scope: three FedAvg modes at 10% missingness; see README for all tables and figures.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path("new_reproduction_output"),
                        help="Output directory; relative paths are resolved against this package")
    args = parser.parse_args()
    output = args.study_dir
    if not output.is_absolute():
        output = Path(__file__).resolve().parent / output
    try:
        check(output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
