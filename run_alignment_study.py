"""Run the explicitly documented exploratory missingness sensitivity study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_synthetic_experiment import bootstrap_ci, run

SCENARIOS = {"missing_00": 0.0, "missing_10": 0.1, "missing_30": 0.3}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for scenario, missing_rate in SCENARIOS.items():
        directory = args.output_dir / scenario
        if (directory / "summary.json").exists():
            raise ValueError(f"Refusing to overwrite an existing completed study: {directory}")
        print(f"Scenario {scenario}", flush=True)
        results[scenario] = run(args.input_dir, directory, [11, 22, 33, 44, 55], 5,
                                0.7, missing_rate, 0.1, 0.05, 5, 128, 20, 2, 0.5, 0.01, "hypertension")
    contrasts = {}
    for scenario, result in results.items():
        metrics = result["summary"]["federated_metrics"]
        contrasts[scenario] = {}
        for control in ("raw", "aligned", "normalized_events"):
            contrasts[scenario][control] = {}
            for metric in ("f1", "auroc", "auprc", "logloss"):
                combined = metrics["alignment_aware"]["fedavg"][metric]["values"]
                baseline = metrics[control]["fedavg"][metric]["values"]
                differences = [a - b for a, b in zip(combined, baseline)]
                contrasts[scenario][control][metric] = {"values": differences, "mean": sum(differences) / len(differences),
                                                       "bootstrap_95_ci": bootstrap_ci(differences, 11)}
    index = {"status": "exploratory; protocol revised after diagnostics; not preregistered",
             "scenarios": SCENARIOS, "seeds": [11, 22, 33, 44, 55],
             "summaries": {name: str((args.output_dir / name / "summary.json").relative_to(args.output_dir)) for name in results},
             "paired_alignment_aware_fedavg_minus_control": contrasts}
    (args.output_dir / "study_summary.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print("Study complete", flush=True)


if __name__ == "__main__":
    main()
