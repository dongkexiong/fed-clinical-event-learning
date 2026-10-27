"""Run the frozen synthetic pipeline over multiple seeds and summarize metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from evaluate_synthetic_alignment import evaluate
from federated_baselines import run_experiment
from generate_synthea_sites import DEFAULT_SEED, apply_target_condition, generate_sites, load_synthea
from validate_synthetic_artifacts import validate_directory


def bootstrap_ci(values: Iterable[float], seed: int, samples: int = 2000) -> List[float]:
    values = np.asarray(list(values), dtype=float)
    if len(values) == 0:
        return [None, None]
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def summarize(runs: List[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for mode in runs[0]["federated"]["modes"]:
        summary[mode] = {}
        methods = runs[0]["federated"]["modes"][mode]["methods"]
        for method in methods:
            summary[mode][method] = {}
            for metric in ("accuracy", "balanced_accuracy", "f1", "auroc", "auprc", "logloss"):
                values = [run["federated"]["modes"][mode]["methods"][method]["metrics"].get(metric)
                          for run in runs]
                values = [float(value) for value in values if value is not None]
                summary[mode][method][metric] = {
                    "n": len(values),
                    "values": values,
                    "mean": float(np.mean(values)) if values else None,
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None,
                    "bootstrap_95_ci": bootstrap_ci(values, seed),
                }
    align_values = {}
    for view in runs[0]["alignment"]["entity_alignment"]:
        for metric in ("precision", "recall", "f1"):
            values = [run["alignment"]["entity_alignment"][view][metric] for run in runs]
            align_values.setdefault(view, {})[metric] = {
                "n": len(values), "values": values, "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "bootstrap_95_ci": bootstrap_ci(values, seed),
            }
    duplicates = {}
    for metric in ("precision", "recall", "f1"):
        values = [run["alignment"]["duplicate_detection"][metric] for run in runs]
        duplicates[metric] = {"n": len(values), "values": values, "mean": float(np.mean(values)),
                              "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                              "bootstrap_95_ci": bootstrap_ci(values, seed)}
    return {"federated_metrics": summary, "entity_alignment": align_values, "duplicate_detection": duplicates}


def run(input_path: Path, output_dir: Path, seeds: List[int], sites: int,
        shared_rate: float, missing_rate: float, format_noise_rate: float,
        duplicate_rate: float, time_jitter_minutes: int, feature_dim: int,
        rounds: int, local_epochs: int, lr: float, prox_mu: float,
        target_condition_keyword: str) -> Dict[str, Any]:
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    source_files = sorted(input_path.glob("*.csv")) if input_path.is_dir() else [input_path]
    input_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    code_dir = Path(__file__).resolve().parent
    code_hashes = {name: hashlib.sha256((code_dir / name).read_bytes()).hexdigest()
                   for name in ("generate_synthea_sites.py", "alignment_aware.py", "federated_baselines.py",
                                "evaluate_synthetic_alignment.py", "run_synthetic_experiment.py", "validate_synthetic_artifacts.py")}
    events = load_synthea(input_path)
    events, target_definition = apply_target_condition(events, target_condition_keyword)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in seeds:
        print(f"seed {seed}: generate / validate / train / align", flush=True)
        seed_dir = output_dir / f"seed_{seed}"
        manifest = generate_sites(events, seed_dir, sites=sites, seed=seed,
                                  shared_rate=shared_rate, missing_rate=missing_rate,
                                   format_noise_rate=format_noise_rate,
                                   duplicate_rate=duplicate_rate,
                                   time_jitter_minutes=time_jitter_minutes,
                                   target_definition=target_definition)
        validation = validate_directory(seed_dir)
        (seed_dir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
        if not validation["valid"]:
            raise ValueError(f"Invalid site/truth artifacts for seed {seed}")
        federated_path = seed_dir / "federated_results.json"
        federated = run_experiment(seed_dir / "sites", federated_path,
                                   feature_dim=feature_dim, rounds=rounds,
                                   local_epochs=local_epochs, lr=lr,
                                   prox_mu=prox_mu, seed=seed)
        alignment = evaluate(seed_dir / "gold_truth.jsonl", seed_dir / "sites")
        (seed_dir / "alignment_results.json").write_text(json.dumps(alignment, indent=2) + "\n", encoding="utf-8")
        runs.append({"seed": seed, "manifest": manifest,
                     "federated": federated, "alignment": alignment})
        print(f"seed {seed}: complete", flush=True)
    result = {
        "source": "Synthea synthetic records",
        "synthetic_ground_truth": True,
        "study_status": "exploratory protocol revision after earlier diagnostic runs; not preregistered",
        "reproducibility": {"started_utc": started_utc, "elapsed_seconds": time.perf_counter() - started,
                            "python": platform.python_version(), "numpy": np.__version__,
                            "platform": platform.platform(), "input_sha256": input_hashes, "code_sha256": code_hashes},
        "seeds": seeds,
        "config": {"sites": sites, "shared_rate": shared_rate, "missing_rate": missing_rate,
                   "format_noise_rate": format_noise_rate, "duplicate_rate": duplicate_rate,
                   "time_jitter_minutes": time_jitter_minutes, "feature_dim": feature_dim,
                   "rounds": rounds, "local_epochs": local_epochs, "lr": lr, "prox_mu": prox_mu,
                   "missing_patient_policy": "exclude_from_patient_task",
                   "threshold_selection": "local_train_quantile_f1_or_federated_train_count_f1",
                   "split_policy": "gold_patient_grouped"},
                   "target_definition": target_definition,
        "runs": [{"seed": run["seed"], "manifest": run["manifest"],
                  "federated_results": str((output_dir / f"seed_{run['seed']}" / "federated_results.json").relative_to(output_dir)),
                  "alignment": run["alignment"]} for run in runs],
        "summary": summarize(runs, seeds[0] if seeds else DEFAULT_SEED),
    }
    (output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--sites", type=int, default=5)
    parser.add_argument("--shared-rate", type=float, default=0.7)
    parser.add_argument("--missing-rate", type=float, default=0.1)
    parser.add_argument("--format-noise-rate", type=float, default=0.1)
    parser.add_argument("--duplicate-rate", type=float, default=0.05)
    parser.add_argument("--time-jitter-minutes", type=int, default=5)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--prox-mu", type=float, default=0.01)
    parser.add_argument("--target-condition-keyword", default="hypertension")
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    result = run(args.input_dir, args.output_dir, seeds, args.sites, args.shared_rate,
                 args.missing_rate, args.format_noise_rate, args.duplicate_rate,
                 args.time_jitter_minutes, args.feature_dim, args.rounds,
                 args.local_epochs, args.lr, args.prox_mu, args.target_condition_keyword)
    print(json.dumps({"output": str(args.output_dir / "summary.json"), "seeds": result["seeds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
