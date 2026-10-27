"""Post hoc audit of the fold-specific identity clusters actually used in v3.

No classification model is fitted. Generator truth is read only for scoring
after truth-blind links and clusters are constructed from the frozen code.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from alignment_aware import load_linkage_profiles, mutual_best_links
from federated_baselines import load_group_splits


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pairs(n):
    return n * (n - 1) // 2


def audit_seed(directory, seed):
    paths = sorted((directory / "sites").glob("site_*.jsonl"))
    truth_path = directory / "gold_truth.jsonl"
    splits, _ = load_group_splits(truth_path, paths, seed)
    profiles = load_linkage_profiles(paths)
    saved = read(directory / "federated_results.json")
    truths, labels = {}, {}
    with truth_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("patient_id") is None:
                continue
            key = (f"site_{row['site_id'].split('-')[-1]}", row["patient_id"])
            truths[key] = row["gold_patient_id"]
            labels[key] = row["target_label"]
    folds = {}
    for fold, test in (("train", False), ("test", True)):
        subset = {key: tokens for key, tokens in profiles.items() if splits[key[0]][key[1]] == test}
        edges, matcher = mutual_best_links(subset)
        parents = {key: key for key in subset}
        members = {key: {key} for key in subset}

        def root(key):
            while parents[key] != key:
                parents[key] = parents[parents[key]]
                key = parents[key]
            return key

        rejected = 0
        for left, right, score in sorted(edges, key=lambda edge: (-edge[2], edge[0], edge[1])):
            a, b = root(left), root(right)
            if a == b:
                continue
            if {key[0] for key in members[a]} & {key[0] for key in members[b]}:
                rejected += 1
                continue
            parents[b] = a
            members[a] |= members.pop(b)
        sizes = Counter(truths[key] for key in subset)
        true_pairs = sum(pairs(n) for n in sizes.values())
        predicted = tp = impure = cross_label_pairs = 0
        weights = {}
        for cluster in members.values():
            gold_counts = Counter(truths[key] for key in cluster)
            label_counts = Counter(labels[key] for key in cluster)
            predicted += pairs(len(cluster))
            tp += sum(pairs(n) for n in gold_counts.values())
            impure += int(len(gold_counts) > 1)
            cross_label_pairs += pairs(len(cluster)) - sum(pairs(n) for n in label_counts.values())
            for key in cluster:
                weights[key] = 1.0 / len(cluster)
        fp = predicted - tp
        fn = true_pairs - tp
        precision = tp / predicted if predicted else 0.0
        recall = tp / true_pairs if true_pairs else 0.0
        per_site = defaultdict(float)
        for key, weight in weights.items():
            per_site[key[0]] += weight
        expected = saved["entity_linkage"]["folds"][fold]
        assert len(members) == expected["clusters"]
        assert rejected == expected["rejected_site_conflicts"]
        assert matcher["candidate_pairs"] == expected["candidate_pairs"]
        assert math.isclose(sum(weights.values()), expected["effective_patient_weight"], abs_tol=1e-7)
        if fold == "train":
            for path, info in zip(paths, saved["modes"]["alignment_aware"]["sites"]):
                assert math.isclose(per_site[path.stem], info["effective_train_weight"], abs_tol=1e-7)
        error = sum(abs(weight - 1.0 / sizes[truths[key]]) for key, weight in weights.items()) / len(weights)
        folds[fold] = {"profiles": len(subset), "clusters": len(members), "oracle_source_patients": len(sizes),
                       "impure_clusters": impure, "rejected_site_conflicts": rejected,
                       "predicted_cluster_pairs": predicted, "true_pairs": true_pairs,
                       "tp": tp, "fp": fp, "fn": fn, "cross_label_false_pairs": cross_label_pairs,
                       "precision": precision, "recall": recall,
                       "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                       "weight_mae_vs_oracle": error, "effective_mass_matches_saved": True}
    return {"seed": seed, "folds": folds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    integrity = read(args.study_dir / "integrity_audit.json")
    assert integrity["passed"]
    for name in ("alignment_aware.py", "federated_baselines.py"):
        assert sha(Path(__file__).resolve().parent / name) == integrity["executed_code_sha256"][name], "Core code changed"
    index = read(args.study_dir / "study_summary.json")
    summaries = {}
    for scenario in index["scenarios"]:
        rows = []
        for seed in index["seeds"]:
            output = args.output_dir / f"{scenario}_seed_{seed}.json"
            if output.exists():
                raise ValueError(f"Refusing to overwrite: {output}")
            result = audit_seed(args.study_dir / scenario / f"seed_{seed}", seed)
            result["status"] = "post hoc diagnostic; no model refitting"
            output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            rows.append(result)
            print(f"Cluster audit {scenario}/seed_{seed}: complete", flush=True)
        summaries[scenario] = {}
        for fold in ("train", "test"):
            summaries[scenario][fold] = {}
            for metric in rows[0]["folds"][fold]:
                if metric == "effective_mass_matches_saved":
                    continue
                values = [row["folds"][fold][metric] for row in rows]
                mean = sum(values) / len(values)
                std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
                summaries[scenario][fold][metric] = {"values": values, "mean": mean, "std": std}
    report = {"status": "post hoc diagnostic added after v3 classification results; no model refitting",
              "study_summary_sha256": sha(args.study_dir / "study_summary.json"),
              "integrity_audit_sha256": sha(args.study_dir / "integrity_audit.json"),
              "script_sha256": sha(Path(__file__)), "summary": summaries,
              "evidence_scope": "Final clusters rebuilt from the frozen matcher; truth used only for scoring. Pairwise metrics include transitive cluster pairs. Cross-label counts are diagnostic, not causal attribution of prediction differences."}
    (args.output_dir / "cluster_audit_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("All 15 fold-specific cluster reconstructions match saved effective training masses.")


if __name__ == "__main__":
    main()
