"""Check saved study artifacts without rerunning model fitting or linkage."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit_run(directory: Path, run: dict, summary: dict, source_labels: dict) -> dict:
    manifest = read(directory / "manifest.json")
    validation = read(directory / "validation.json")
    federated = read(directory / "federated_results.json")
    alignment = read(directory / "alignment_results.json")
    require(manifest == run["manifest"], f"Manifest differs from summary: {directory}")
    require(alignment == run["alignment"], f"Alignment differs from summary: {directory}")
    require(validation["valid"] and len(validation["sites"]) == 5,
            f"Saved validation failed: {directory}")
    require(manifest["config"]["seed"] == run["seed"], "Seed mismatch")
    require(manifest["config"]["missing_rate"] == summary["config"]["missing_rate"], "Missingness mismatch")
    config = federated["config"]
    for key in ("feature_dim", "rounds", "local_epochs", "lr", "prox_mu"):
        require(config[key] == summary["config"][key], f"Training config differs: {key}")
    require(config["exclude_event_types"] == ["condition"], "Condition exclusion changed")
    require(config["split_policy"] == "gold_patient_grouped", "Split policy changed")
    require(federated["entity_linkage"]["truth_used_for_matching"] is False, "Linkage uses truth")

    truth_path = directory / manifest["gold_truth"]["path"]
    require(digest(truth_path) == manifest["gold_truth"]["sha256"], "Truth hash mismatch")
    truth = {}
    duplicates = {}
    patient_gold = {}
    golds = set()
    with truth_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["site_id"], row["event_id"])
            require(key not in truth, "Repeated truth event ID")
            patient, gold, label = row.get("patient_id"), row["gold_patient_id"], row["target_label"]
            require(gold in source_labels and label == source_labels[gold], "Source/replica label mismatch")
            golds.add(gold)
            truth[key] = (patient, gold, label)
            if patient is not None:
                pk = (row["site_id"], patient)
                require(patient_gold.get(pk, gold) == gold, "Patient ID maps to multiple source patients")
                patient_gold[pk] = gold
            if row.get("duplicate_of"):
                duplicates[key] = (row["site_id"], row["duplicate_of"])
                require("-dup" not in row["event_id"], "Duplicate ID reveals duplicate status")
    require(len(truth) == manifest["gold_truth"]["records"] == manifest["output_records"], "Truth count mismatch")
    require(golds == set(source_labels), "Not all source patients are represented")
    for key, original in duplicates.items():
        require(original in truth and truth[key] == truth[original], "Invalid duplicate truth reference")
    ordered = sorted(set(patient_gold.values()),
                     key=lambda gold: hashlib.sha256(f"{run['seed']}:{gold}".encode()).hexdigest())
    test_gold = set(ordered[:round(len(ordered) * 0.2)])
    train_gold = set(ordered) - test_gold
    actual_train, actual_test = set(), set()
    site_counts = []
    for position, site in enumerate(manifest["sites"]):
        path = directory / site["path"]
        require(digest(path) == site["sha256"], f"Site hash mismatch: {path}")
        observed = set()
        patients = {}
        missing = excluded = count = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                require(not {"gold_patient_id", "gold_event_id", "duplicate_of"} & row.keys(), "Truth field in site data")
                event_id = row.get("event_id", row.get("eventId"))
                key = (row["site_id"], event_id)
                require(key in truth and key not in observed, "Missing truth or repeated site event ID")
                observed.add(key)
                patient = row.get("patient_id", row.get("patientId"))
                require(patient == truth[key][0] and row["target_label"] == truth[key][2], "Site/truth disagreement")
                event_type = row.get("event_type", row.get("eventType", row.get("type")))
                excluded += int(event_type == "condition")
                count += 1
                if patient is None:
                    missing += 1
                    continue
                gold, label = truth[key][1:]
                require(patients.get(patient, (gold, label)) == (gold, label), "Inconsistent patient label")
                patients[patient] = (gold, label)
                (actual_test if gold in test_gold else actual_train).add(gold)
        require(count == site["records"] == validation["sites"][position]["records"], "Site count mismatch")
        # The executed generator's legacy manifest counts None as a distinct
        # patient-ID value. Training correctly excludes that value.
        require(len(patients) + int(missing > 0) == site["patients"], "Manifest patient-ID cardinality mismatch")
        site_truth_keys = {key for key in truth if key[0] == site["site_id"]}
        require(observed == site_truth_keys, "Truth/site event coverage mismatch")
        n_test = sum(gold in test_gold for gold, _ in patients.values())
        expected = {"records": count, "patients": len(patients), "test": n_test,
                    "train": len(patients) - n_test,
                    "positive": sum(label for _, label in patients.values()),
                    "missing_patient_records": missing, "excluded_records": excluded}
        for mode in federated["modes"].values():
            info = mode["sites"][position]
            for key, value in expected.items():
                require(info[key] == value, f"Site feature audit differs: {key}")
        site_counts.append({"site_id": site["site_id"], **expected})
    require(not actual_train & actual_test, "Cross-site patient overlap between folds")
    require(actual_train == train_gold and actual_test == test_gold, "Holdout coverage differs")
    split = federated["split_audit"]
    require(split["train_gold_patients"] == len(train_gold) == 924 and
            split["test_gold_patients"] == len(test_gold) == 231 and
            split["train_test_gold_overlap"] == 0, "Saved split audit differs")
    for mode, mode_data in federated["modes"].items():
        for method, result in mode_data["methods"].items():
            for metric in ("accuracy", "balanced_accuracy", "f1", "auroc", "auprc", "logloss"):
                values = [r[metric] for r in result["per_site"] if r.get(metric) is not None]
                require(values and all(math.isfinite(v) for v in values), "Nonfinite or missing metric")
                mean = sum(values) / len(values)
                require(math.isclose(mean, result["metrics"][metric], abs_tol=1e-12), "Site mean differs")
                row = summary["summary"]["federated_metrics"][mode][method][metric]
                seed_position = summary["seeds"].index(run["seed"])
                require(math.isclose(mean, row["values"][seed_position], abs_tol=1e-12), "Summary value differs")
                require(math.isclose(sum(row["values"]) / len(row["values"]), row["mean"], abs_tol=1e-12), "Summary mean differs")
    return {"seed": run["seed"], "passed": True, "records": len(truth),
            "train_gold_patients": len(train_gold), "test_gold_patients": len(test_gold),
            "train_test_gold_overlap": len(actual_train & actual_test), "site_counts": site_counts,
            "artifact_sha256": {name: digest(directory / name) for name in
                                ("manifest.json", "validation.json", "federated_results.json", "alignment_results.json")}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    index = read(args.study_dir / "study_summary.json")
    selection = read(args.input_dir.parent / "selection_manifest.json")
    for name, expected in selection["sha256"].items():
        require(digest(args.input_dir / name) == expected, f"Selected input hash mismatch: {name}")
    with (args.input_dir / "patients.csv").open(encoding="utf-8-sig", newline="") as handle:
        source_labels = {row["Id"]: 0 for row in csv.DictReader(handle)}
    with (args.input_dir / "conditions.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if "hypertension" in row["DESCRIPTION"].casefold():
                source_labels[row["PATIENT"]] = 1
    require(len(source_labels) == 1155 and sum(source_labels.values()) == 295, "Source patient denominator differs")
    results = {}
    code_hashes = None
    for scenario, path in index["summaries"].items():
        summary = read(args.study_dir / path)
        require(summary["seeds"] == index["seeds"] == [11, 22, 33, 44, 55], "Seed set differs")
        require(summary["config"]["missing_rate"] == index["scenarios"][scenario], "Scenario differs")
        for path, expected in summary["reproducibility"]["input_sha256"].items():
            # Historical runs stored the original machine-relative input path.
            # Resolve by the CSV basename under the caller-supplied public input
            # directory so the audit also works after a clean download.
            input_path = args.input_dir / Path(path).name
            require(digest(input_path) == expected, f"Run input hash differs: {input_path}")
        current = summary["reproducibility"]["code_sha256"]
        if code_hashes is None:
            code_hashes = current
        require(current == code_hashes, "Code changed across scenarios")
        for name, expected in current.items():
            require(digest(Path(__file__).resolve().parent / name) == expected,
                    f"Current code differs from executed code: {name}")
        results[scenario] = []
        for run in summary["runs"]:
            directory = args.study_dir / scenario / f"seed_{run['seed']}"
            results[scenario].append(audit_run(directory, run, summary, source_labels))
            print(f"Audited {scenario}/seed_{run['seed']}", flush=True)
    require(sum(map(len, results.values())) == 15, "Incomplete experiment grid")
    report = {"passed": True, "scope": "artifact hashes, truth isolation, source-consistent labels, independent grouped-fold reconstruction, recorded metrics; no model refitting",
              "manifest_patient_count_note": "Legacy manifest sites[].patients includes one None value when missing IDs occur; site_counts below and federated_results count identified patients only.",
              "runs_checked": 15, "sites_checked": 75, "input_csv_hashes_checked": len(selection["sha256"]),
              "study_summary_sha256": digest(args.study_dir / "study_summary.json"),
              "selection_manifest_sha256": digest(args.input_dir.parent / "selection_manifest.json"),
              "audit_script_sha256": digest(Path(__file__)), "executed_code_sha256": code_hashes,
              "scenarios": results}
    (args.study_dir / "integrity_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("All 15 runs and 75 sites passed.", flush=True)


if __name__ == "__main__":
    main()
