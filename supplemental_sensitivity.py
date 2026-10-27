"""Targeted supplemental checks for the manuscript's main negative contrast.

This script keeps the frozen primary experiment unchanged.  It generates one
temporary site bundle at a time and reports:

* source-patient-equal evaluation by averaging test-replica probabilities;
* a lower-overlap setting (shared_rate=0.3); and
* a longer FedAvg optimization budget (80 rounds).

The reported threshold is selected by the existing training-only federated
count rule.  Source-patient metrics then apply that same threshold after
averaging the test-replica probabilities for each source patient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from alignment_aware import build_entity_weights
from federated_baselines import (
    f1_score,
    load_group_splits,
    load_site,
    local_update,
    metrics,
    rank_metrics,
    select_federated_threshold,
    sigmoid,
    unpack,
)
from generate_synthea_sites import apply_target_condition, generate_sites, load_synthea
from validate_synthetic_artifacts import validate_directory


SEEDS = (11, 22, 33, 44, 55)
MODES = ("normalized_events", "alignment_aware")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def site_patient_keys(path: Path) -> List[str]:
    keys = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("patient_id", row.get("patientId"))
            if key not in (None, ""):
                keys.add(str(key))
    return sorted(keys)


def load_truth(path: Path) -> Dict[Tuple[str, str], Tuple[str, int]]:
    result: Dict[Tuple[str, str], Tuple[str, int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            site = row.get("site_id")
            patient = row.get("patient_id")
            gold = row.get("gold_patient_id")
            if site and patient and gold:
                result.setdefault((str(site), str(patient)), (str(gold), int(row.get("target_label") or 0)))
    return result


def source_metrics(groups: Mapping[str, Sequence[float]], labels: Mapping[str, int], threshold: float) -> Dict[str, Any]:
    gold = np.asarray(sorted(labels), dtype=object)
    y = np.asarray([labels[key] for key in gold], dtype=np.int8)
    p = np.asarray([float(np.mean(groups[key])) for key in gold], dtype=float)
    predicted = (p >= threshold).astype(np.int8)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    tp = int(((y == 1) & (predicted == 1)).sum())
    tn = int(((y == 0) & (predicted == 0)).sum())
    sensitivity = tp / positives if positives else None
    specificity = tn / negatives if negatives else None
    auroc, auprc = rank_metrics(y, p)
    return {
        "n": int(len(y)),
        "positive": positives,
        "replicas": int(sum(len(groups[key]) for key in gold)),
        "mean_replicas_per_source": float(np.mean([len(groups[key]) for key in gold])) if len(gold) else None,
        "accuracy": float(np.mean(predicted == y)) if len(y) else None,
        "balanced_accuracy": float((sensitivity + specificity) / 2.0)
        if sensitivity is not None and specificity is not None else None,
        "f1": float(f1_score(y, predicted)) if len(y) else None,
        "auroc": auroc,
        "auprc": auprc,
        "logloss": float(-np.mean(y * np.log(np.clip(p, 1e-8, 1 - 1e-8))
                                  + (1 - y) * np.log(np.clip(1 - p, 1e-8, 1 - 1e-8)))) if len(y) else None,
        "threshold": float(threshold),
    }


def train_fed_capture(
    sites: Sequence[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]],
    site_meta: Sequence[Tuple[str, Sequence[str], Sequence[str]]],
    feature_dim: int,
    rounds: int,
    local_epochs: int,
    lr: float,
    prox_mu: float,
    mode: str,
    truth: Mapping[Tuple[str, str], Tuple[str, int]],
) -> Dict[str, Any]:
    weighted = mode == "alignment_aware"
    w = np.zeros(feature_dim, dtype=np.float64)
    b = 0.0
    for _round in range(rounds):
        updates = []
        masses = []
        for site in sites:
            x_train, y_train, _x_test, _y_test, info = unpack(site)
            weights = np.asarray(info.get("sample_weights", np.ones(len(site[1]))))[:len(y_train)]
            local_w, local_b = local_update(
                w, b, x_train, y_train, local_epochs, lr,
                global_w=w if False else None,
                global_b=b if False else None,
                prox_mu=0.0,
                sample_weight=weights,
            )
            updates.append((local_w, local_b))
            masses.append(float(weights.sum()))
        total = float(sum(masses))
        w = sum(local_w * (mass / total) for (local_w, _), mass in zip(updates, masses))
        b = sum(local_b * (mass / total) for (_, local_b), mass in zip(updates, masses))

    threshold = select_federated_threshold(w, b, sites)
    site_rows = []
    groups: Dict[str, List[float]] = defaultdict(list)
    labels: Dict[str, int] = {}
    replica_counts = []
    for site, (stem, _train_keys, test_keys) in zip(sites, site_meta):
        _x_train, _y_train, x_test, y_test, _info = unpack(site)
        site_rows.append(metrics(w, b, x_test, y_test, threshold=threshold))
        probabilities = sigmoid(x_test @ w + b)
        if len(probabilities) != len(test_keys):
            raise ValueError(f"Test key count differs for {stem}: {len(test_keys)} != {len(probabilities)}")
        logical_site = f"logical-site-{stem.split('_')[-1]}"
        for patient, probability in zip(test_keys, probabilities):
            record = truth.get((logical_site, patient))
            if record is None:
                continue
            gold, label = record
            groups[gold].append(float(probability))
            labels[gold] = int(label)
    patient = source_metrics(groups, labels, threshold)
    patient["source_patients_with_identified_test_replica"] = len(labels)
    patient["source_patients_missing_all_test_ids"] = None
    return {
        "site_macro": {key: value for key, value in metrics_average(site_rows).items()},
        "source_patient_equal": patient,
        "threshold": float(threshold),
        "rounds": int(rounds),
        "local_epochs": int(local_epochs),
        "mode": mode,
        "weighted_training": weighted,
    }


def metrics_average(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [row for row in rows if row.get("n", 0)]
    if not valid:
        return {"n": 0}
    result = {"n": int(sum(row["n"] for row in valid))}
    for key in ("accuracy", "balanced_accuracy", "f1", "auroc", "auprc", "logloss"):
        values = [row[key] for row in valid if row.get(key) is not None]
        result[key] = float(np.mean(values)) if values else None
    return result


def summarize(values: Sequence[float]) -> Dict[str, Any]:
    values = [float(value) for value in values]
    return {
        "n": len(values),
        "values": values,
        "mean": float(mean(values)) if values else None,
        "sd": float(stdev(values)) if len(values) > 1 else 0.0 if values else None,
    }


def run_condition(
    events: Sequence[Mapping[str, Any]],
    target_definition: Mapping[str, Any],
    input_hashes: Mapping[str, str],
    output_dir: Path,
    name: str,
    shared_rate: float,
    rounds: int,
    seeds: Sequence[int],
    missing_rate: float = 0.1,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed = []
    temp_root = Path(tempfile.mkdtemp(prefix=f"{name}_", dir=str(output_dir)))
    try:
        for seed in seeds:
            seed_dir = temp_root / f"seed_{seed}"
            manifest = generate_sites(
                events, seed_dir, sites=5, seed=seed, shared_rate=shared_rate,
                missing_rate=missing_rate, format_noise_rate=0.1,
                duplicate_rate=0.05, time_jitter_minutes=5,
                target_definition=target_definition,
            )
            validation = validate_directory(seed_dir)
            if not validation["valid"]:
                raise ValueError(f"Invalid supplemental artifacts for seed {seed}")
            paths = sorted((seed_dir / "sites").glob("site_*.jsonl"))
            truth_path = seed_dir / "gold_truth.jsonl"
            split_maps, split_audit = __import__("federated_baselines").load_group_splits(truth_path, paths, seed)
            entity_weights, linkage_audit = build_entity_weights(paths, split_maps)
            truth = load_truth(truth_path)
            mode_results = {}
            for mode in MODES:
                sites = []
                site_meta = []
                for path in paths:
                    keys = site_patient_keys(path)
                    split = split_maps[path.stem]
                    train_keys = [key for key in keys if not split[key]]
                    test_keys = [key for key in keys if split[key]]
                    data = load_site(
                        path, 128, True, patient_splits=split,
                        normalize_events=True,
                        entity_weights=entity_weights[path.stem] if mode == "alignment_aware" else None,
                    )
                    if len(train_keys) != data[2]["train_count"] or len(test_keys) != data[2]["test"]:
                        raise ValueError(f"Key ordering mismatch for {path}")
                    sites.append(data)
                    site_meta.append((path.stem, train_keys, test_keys))
                mode_results[mode] = train_fed_capture(
                    sites, site_meta, 128, rounds, 2, 0.5, 0.01,
                    mode, truth,
                )
            per_seed.append({
                "seed": seed,
                "manifest": {"output_records": manifest["output_records"], "input_patients": manifest["input_patients"]},
                "split_audit": split_audit,
                "linkage_audit": linkage_audit,
                "modes": mode_results,
            })
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    summary: Dict[str, Any] = {
        "name": name,
        "config": {
            "sites": 5, "shared_rate": shared_rate, "missing_rate": missing_rate,
            "format_noise_rate": 0.1, "duplicate_rate": 0.05,
            "time_jitter_minutes": 5, "feature_dim": 128,
            "rounds": rounds, "local_epochs": 2, "learning_rate": 0.5,
            "prox_mu": 0.01, "seeds": list(seeds),
            "source_patient_aggregation": "mean probability over identified test replicas; inherited training-only threshold",
            "input_sha256": dict(input_hashes),
        },
        "per_seed": per_seed,
        "summary": {},
    }
    for mode in MODES:
        summary["summary"][mode] = {}
        for evaluation in ("site_macro", "source_patient_equal"):
            summary["summary"][mode][evaluation] = {}
            for metric in ("f1", "auroc", "auprc", "accuracy", "balanced_accuracy", "logloss"):
                summary["summary"][mode][evaluation][metric] = summarize([
                    row["modes"][mode][evaluation][metric]
                    for row in per_seed if row["modes"][mode][evaluation].get(metric) is not None
                ])
    deltas = {}
    for evaluation in ("site_macro", "source_patient_equal"):
        deltas[evaluation] = {}
        for metric in ("f1", "auroc", "auprc"):
            deltas[evaluation][metric] = summarize([
                row["modes"]["alignment_aware"][evaluation][metric]
                - row["modes"]["normalized_events"][evaluation][metric]
                for row in per_seed
            ])
    summary["summary"]["alignment_aware_minus_normalized"] = deltas
    (output_dir / f"{name}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    input_hashes = {path.name: sha256(path) for path in sorted(input_dir.glob("*.csv"))}
    events = load_synthea(input_dir)
    events, target_definition = apply_target_condition(events, "hypertension")
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = [
        ("baseline_patient_equal", 0.7, 20),
        ("low_overlap_shared_030", 0.3, 20),
        ("long_optimization_rounds_080", 0.7, 80),
    ]
    index = {"script": Path(__file__).name, "input_sha256": input_hashes, "conditions": []}
    for name, shared_rate, rounds in conditions:
        result = run_condition(events, target_definition, input_hashes, output_dir, name, shared_rate, rounds, seeds)
        index["conditions"].append({"name": name, "shared_rate": shared_rate, "rounds": rounds,
                                    "summary_file": f"{name}.json"})
        print(json.dumps({"name": name, "source_patient_delta_f1_mean": result["summary"]["alignment_aware_minus_normalized"]["source_patient_equal"]["f1"]["mean"]}, ensure_ascii=False))
    (output_dir / "manifest.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
