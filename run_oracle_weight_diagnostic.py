"""Post-hoc oracle-replica-weight diagnostic on the frozen synthetic sites.

This diagnostic uses generator truth only to construct oracle 1/k weights. It
does not change the primary v3 results or claim that oracle weights are
available in a real federated deployment.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from federated_baselines import load_group_splits, load_site, train_federated
from run_synthetic_experiment import bootstrap_ci


METRICS = ("accuracy", "balanced_accuracy", "f1", "auroc", "auprc", "logloss")
SCENARIOS = ("missing_00", "missing_10", "missing_30")
SEEDS = (11, 22, 33, 44, 55)


def oracle_weights(seed_dir: Path, split_maps: dict[str, dict[str, bool]]) -> dict[str, dict[str, float]]:
    """Return fold-specific 1/k weights over identifiable local identities."""
    site_names = {p.stem: f"logical-site-{p.stem.split('_')[-1]}"
                  for p in sorted((seed_dir / "sites").glob("site_*.jsonl"))}
    identities: dict[str, dict[str, set[tuple[str, str]]]] = {
        "train": defaultdict(set), "test": defaultdict(set)
    }
    rows = []
    with (seed_dir / "gold_truth.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            site = str(row.get("site_id") or "")
            patient = row.get("patient_id")
            gold = row.get("gold_patient_id")
            if not site or patient in (None, "") or not gold:
                continue
            stem = next((s for s, name in site_names.items() if name == site), None)
            if stem is None or str(patient) not in split_maps.get(stem, {}):
                continue
            fold = "test" if split_maps[stem][str(patient)] else "train"
            rows.append((stem, str(patient), str(gold), fold))
            identities[fold][str(gold)].add((site, str(patient)))
    counts = {fold: {gold: len(keys) for gold, keys in by_gold.items()}
              for fold, by_gold in identities.items()}
    result = {stem: {} for stem in site_names}
    for stem, patient, gold, fold in rows:
        result[stem][patient] = 1.0 / counts[fold][gold]
    return result


def run_one(seed_dir: Path, seed: int) -> dict[str, Any]:
    paths = sorted((seed_dir / "sites").glob("site_*.jsonl"))
    split_maps, _ = load_group_splits(seed_dir / "gold_truth.jsonl", paths, seed)
    oracle = oracle_weights(seed_dir, split_maps)
    normal = [load_site(path, 128, True, patient_splits=split_maps[path.stem],
                         normalize_events=True) for path in paths]
    oracle_sites = [load_site(path, 128, True, patient_splits=split_maps[path.stem],
                              normalize_events=True, entity_weights=oracle[path.stem])
                    for path in paths]
    inferred = json.loads((seed_dir / "federated_results.json").read_text(encoding="utf-8"))
    inferred_metrics = inferred["modes"]["alignment_aware"]["methods"]["fedavg"]["metrics"]
    normal_result = train_federated(normal, 128, "fedavg", 20, 2, 0.5, 0.01)
    oracle_result = train_federated(oracle_sites, 128, "fedavg", 20, 2, 0.5, 0.01)
    return {
        "seed": seed,
        "normal_metrics": {k: normal_result["metrics"][k] for k in METRICS},
        "inferred_metrics": {k: inferred_metrics[k] for k in METRICS},
        "oracle_metrics": {k: oracle_result["metrics"][k] for k in METRICS},
        "oracle_minus_normal": {k: oracle_result["metrics"][k] - normal_result["metrics"][k]
                                for k in METRICS},
        "oracle_minus_inferred": {k: oracle_result["metrics"][k] - inferred_metrics[k]
                                  for k in METRICS},
        "oracle_identity_count": sum(len(v) for v in oracle.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path,
                        default=Path("reproduction_output"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("reproduction_output/oracle_weight_diagnostic"))
    args = parser.parse_args()
    root = args.study_dir
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, Any] = {}
    for scenario in SCENARIOS:
        rows = []
        for seed in SEEDS:
            seed_dir = root / scenario / f"seed_{seed}"
            row = run_one(seed_dir, seed)
            rows.append(row)
            (out / f"{scenario}_seed_{seed}.json").write_text(
                json.dumps(row, indent=2) + "\n", encoding="utf-8")
        all_results[scenario] = rows
        summary[scenario] = {}
        for contrast in ("oracle_minus_normal", "oracle_minus_inferred"):
            summary[scenario][contrast] = {}
            for metric in METRICS:
                values = [r[contrast][metric] for r in rows]
                summary[scenario][contrast][metric] = {
                    "values": values,
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values),
                    "bootstrap_95_ci": bootstrap_ci(values, 11),
                }
    payload = {
        "status": "post-hoc diagnostic; generator truth used only for oracle weights; primary results unchanged",
        "protocol": {"feature_dim": 128, "rounds": 20, "local_epochs": 2,
                      "learning_rate": 0.5, "prox_mu": 0.01,
                      "weights": "oracle 1/k over identifiable source-patient replicas within each fold"},
        "source_study": str(root),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "results": all_results,
        "summary": summary,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out / "summary.json"), "scenarios": list(summary)}, indent=2))


if __name__ == "__main__":
    main()
