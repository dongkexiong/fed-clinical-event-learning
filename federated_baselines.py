"""NumPy-only federated learning baselines for generated logical sites.

The script is intentionally small and auditable. It trains a binary logistic
regression on patient-level event aggregates and compares Local-only,
FedAvg, FedProx and a centralized upper bound.  Running it on the generated
fixture demonstrates the mechanics; it is not a clinical performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from alignment_aware import build_entity_weights, canonical_event, event_fingerprint


def _first(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def normalize_code(value: Any, aligned: bool) -> Optional[str]:
    if value in (None, ""):
        return None
    code = str(value).strip().casefold()
    if aligned:
        # The generator encodes site-local code namespaces as L2-<code>.
        # Removing only this explicit namespace is the alignment operation;
        # arbitrary semantic synonymy is not inferred.
        while len(code) > 3 and code[0] == "l" and code[2] == "-" and code[1].isdigit():
            code = code[3:]
    return code


def _row_fields(row: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[int]]:
    patient = _first(row, "patient_id", "patientId")
    event_id = _first(row, "event_id", "eventId")
    event_type = _first(row, "event_type", "eventType", "type")
    code = _first(row, "concept_code", "local_code", "code")
    label = _first(row, "target_label")
    try:
        label_int = int(label) if label is not None else None
    except (TypeError, ValueError):
        label_int = None
    return (str(patient) if patient is not None else None,
            str(event_id) if event_id is not None else None,
            str(event_type) if event_type is not None else None,
            str(code) if code is not None else None,
            label_int)


def load_site(path: Path, feature_dim: int, aligned: bool,
              exclude_event_types: Sequence[str] = ("condition",),
              patient_splits: Optional[Mapping[str, bool]] = None,
              normalize_events: bool = False,
              entity_weights: Optional[Mapping[str, float]] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    patients: Dict[str, Dict[str, Any]] = {}
    records = 0
    excluded_records = 0
    missing_patient_records = 0
    missing_patient_positive_records = 0
    seen = set()
    collapsed_records = 0
    excluded = {str(value).casefold() for value in exclude_event_types}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            patient, event_id, event_type, code, label = _row_fields(row)
            is_excluded = bool(event_type and event_type.casefold() in excluded)
            # A patient-level target cannot be assigned to an event without a
            # patient identifier. Keep the record in the site artifact and
            # report it, but do not turn it into a synthetic patient bucket.
            if patient is None:
                missing_patient_records += 1
                missing_patient_positive_records += int(label == 1)
                excluded_records += int(is_excluded)
                records += 1
                continue
            key = patient
            item = patients.setdefault(key, {"tokens": [], "label": 0})
            if label is not None:
                item["label"] = max(item["label"], int(label))
            if is_excluded:
                # The event determines the synthetic task label, but its type,
                # code, value, and description are withheld from features to
                # prevent direct label leakage.
                excluded_records += 1
                records += 1
                continue
            if normalize_events:
                event = canonical_event(row, records)
                fingerprint = event_fingerprint(event)
                if fingerprint in seen:
                    collapsed_records += 1
                    records += 1
                    continue
                seen.add(fingerprint)
                code = event["code"]
                unit = event["unit"]
            else:
                unit = _first(row, "unit")
            if event_type:
                item["tokens"].append(f"event:{event_type.casefold()}")
            code_norm = normalize_code(code, aligned)
            if code_norm:
                item["tokens"].append(f"code:{code_norm}")
            if unit:
                item["tokens"].append(f"unit:{str(unit).casefold()}")
            records += 1
    keys = sorted(patients)
    x = np.zeros((len(keys), feature_dim), dtype=np.float64)
    y = np.zeros(len(keys), dtype=np.float64)
    for index, key in enumerate(keys):
        y[index] = patients[key]["label"]
        for token in patients[key]["tokens"]:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "big") % feature_dim
            x[index, bucket] += 1.0
        norm = np.linalg.norm(x[index])
        if norm:
            x[index] /= norm
    if patient_splits is not None:
        missing = set(keys) - set(patient_splits)
        if missing:
            raise ValueError(f"Split map missing {len(missing)} patients for {path}")
        test_mask = np.array([patient_splits[k] for k in keys], dtype=bool)
    else:
        # Direct loader use only. Formal runs require a gold-patient grouped
        # split, prepared offline by run_experiment below.
        test_mask = np.array([(int(hashlib.sha256(k.encode()).hexdigest()[:8], 16) % 5) == 0 for k in keys], dtype=bool)
        if len(keys) >= 3 and not test_mask.any():
            test_mask[int(np.argmax([int(hashlib.sha256(k.encode()).hexdigest()[:8], 16) for k in keys]))] = True
        if len(keys) >= 3 and test_mask.all():
            test_mask[0] = False
    train_mask = ~test_mask
    if patient_splits is None and not train_mask.any() and len(keys):
        train_mask[0] = True
        test_mask[0] = False
    info = {"path": str(path), "patients": len(keys), "records": records,
            "train": int(train_mask.sum()), "test": int(test_mask.sum()),
            "positive": int(y.sum()), "excluded_event_types": sorted(excluded),
            "excluded_records": excluded_records,
            "missing_patient_records": missing_patient_records,
            "missing_patient_positive_records": missing_patient_positive_records,
            "collapsed_feature_records": collapsed_records,
            "sample_weights": [float((entity_weights or {}).get(k, 1.0)) for k in list(np.array(keys)[train_mask]) + list(np.array(keys)[test_mask])],
            "split_policy": "gold_patient_grouped" if patient_splits is not None else "site_local_id"}
    # Return train/test packed in a fixed structure for simple federated loops.
    return np.concatenate([x[train_mask], x[test_mask]], axis=0), np.concatenate([y[train_mask], y[test_mask]]), {
        **info, "train_count": int(train_mask.sum()), "x_train_end": int(train_mask.sum()),
    }


def unpack(data: Tuple[np.ndarray, np.ndarray, Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    x, y, info = data
    cut = info["x_train_end"]
    return x[:cut], y[:cut], x[cut:], y[cut:], info


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def local_update(w: np.ndarray, b: float, x: np.ndarray, y: np.ndarray,
                epochs: int, lr: float, global_w: Optional[np.ndarray] = None,
                global_b: Optional[float] = None, prox_mu: float = 0.0,
                sample_weight: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    if len(y) == 0:
        return w.copy(), float(b)
    out_w, out_b = w.copy(), float(b)
    weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight)
    denominator = float(weights.sum())
    for _ in range(epochs):
        p = sigmoid(x @ out_w + out_b)
        err = (p - y) * weights
        grad_w = (x.T @ err) / denominator
        grad_b = float(err.sum() / denominator)
        if global_w is not None and prox_mu:
            grad_w += prox_mu * (out_w - global_w)
            grad_b += prox_mu * (out_b - float(global_b))
        out_w -= lr * grad_w
        out_b -= lr * grad_b
    return out_w, out_b


def f1_score(y: np.ndarray, p: np.ndarray) -> float:
    tp = float(((y == 1) & (p == 1)).sum())
    fp = float(((y == 0) & (p == 1)).sum())
    fn = float(((y == 1) & (p == 0)).sum())
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def rank_metrics(y: np.ndarray, scores: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Return AUROC and average precision without a third-party metrics package."""
    y = np.asarray(y, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return None, None
    order = np.argsort(-scores, kind="mergesort")
    ranked_y = y[order]
    ranked_scores = scores[order]
    # Average precision is computed at score-group boundaries so tied scores
    # do not acquire an arbitrary order from the input rows.
    cumulative_tp = np.cumsum(ranked_y)
    group_ends = np.r_[np.flatnonzero(np.diff(ranked_scores) != 0), len(scores) - 1]
    tp_at_end = cumulative_tp[group_ends]
    positive_in_group = np.diff(np.r_[0, tp_at_end])
    average_precision = float(np.sum((tp_at_end / (group_ends + 1)) * positive_in_group) / positives)
    # Tie-aware AUROC from the Mann-Whitney U statistic.  Ranks must be
    # ascending here so a larger score receives a larger rank.
    ascending_order = np.argsort(scores, kind="mergesort")
    ascending_scores = scores[ascending_order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and ascending_scores[end] == ascending_scores[start]:
            end += 1
        ranks[ascending_order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[y == 1].sum())
    auroc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auroc), average_precision


def select_threshold(w: np.ndarray, b: float, x: np.ndarray, y: np.ndarray) -> float:
    """Select a fixed decision threshold from training data only."""
    if len(y) == 0 or int(y.sum()) == 0 or int(y.sum()) == len(y):
        return 0.5
    scores = sigmoid(x @ w + b)
    candidates = np.unique(np.concatenate([np.quantile(scores, np.linspace(0.01, 0.99, 99)), [0.5]]))
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in candidates:
        score = f1_score(y.astype(int), (scores >= threshold).astype(int))
        if score > best_f1 or (score == best_f1 and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)):
            best_threshold, best_f1 = float(threshold), float(score)
    return best_threshold


def select_federated_threshold(w: np.ndarray, b: float,
                               sites: Sequence[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]) -> float:
    """Choose one threshold using only site-level confusion counts."""
    candidates = np.linspace(0.01, 0.99, 99)
    tp = np.zeros(len(candidates), dtype=np.float64)
    fp = np.zeros(len(candidates), dtype=np.float64)
    fn = np.zeros(len(candidates), dtype=np.float64)
    for site in sites:
        x_train, y_train, _x_test, _y_test, _info = unpack(site)
        if not len(y_train):
            continue
        labels = (sigmoid(x_train @ w + b)[:, None] >= candidates[None, :])
        positives = (y_train == 1)[:, None]
        weights = np.array(_info.get("sample_weights", np.ones(len(site[1]))))[:len(y_train), None]
        tp += np.sum((labels & positives) * weights, axis=0)
        fp += np.sum((labels & ~positives) * weights, axis=0)
        fn += np.sum((~labels & positives) * weights, axis=0)
    denominators = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominators, out=np.zeros(len(candidates), dtype=float), where=denominators > 0)
    best = np.flatnonzero(f1 == f1.max())
    return float(candidates[best[np.argmin(np.abs(candidates[best] - 0.5))]])


def metrics(w: np.ndarray, b: float, x: np.ndarray, y: np.ndarray,
            threshold: float = 0.5) -> Dict[str, Optional[float]]:
    if len(y) == 0:
        return {"n": 0, "positive": 0, "predicted_positive": 0,
                "accuracy": None, "balanced_accuracy": None, "f1": None,
                "auroc": None, "auprc": None, "logloss": None,
                "threshold": threshold}
    p = sigmoid(x @ w + b)
    labels = (p >= threshold).astype(int)
    loss = -np.mean(y * np.log(np.clip(p, 1e-8, 1 - 1e-8)) + (1 - y) * np.log(np.clip(1 - p, 1e-8, 1 - 1e-8)))
    tp = float(((y == 1) & (labels == 1)).sum())
    tn = float(((y == 0) & (labels == 0)).sum())
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    sensitivity = tp / positives if positives else None
    specificity = tn / negatives if negatives else None
    balanced = ((sensitivity + specificity) / 2.0
                if sensitivity is not None and specificity is not None else None)
    auroc, auprc = rank_metrics(y, p)
    return {"n": int(len(y)), "accuracy": float(np.mean(labels == y)),
            "positive": positives, "predicted_positive": int(labels.sum()),
            "balanced_accuracy": float(balanced) if balanced is not None else None,
            "f1": float(f1_score(y.astype(int), labels)),
            "auroc": auroc, "auprc": auprc, "logloss": float(loss),
            "threshold": float(threshold)}


def average_metrics(rows: Sequence[Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    valid = [row for row in rows if row.get("n", 0)]
    if not valid:
        return {"n": 0, "positive": 0, "predicted_positive": 0,
                "accuracy": None, "balanced_accuracy": None, "f1": None,
                "auroc": None, "auprc": None, "logloss": None,
                "threshold": None}
    averaged = {}
    for key in ("accuracy", "balanced_accuracy", "f1", "auroc", "auprc", "logloss"):
        values = [row[key] for row in valid if row.get(key) is not None]
        averaged[key] = float(np.mean(values)) if values else None
    return {"n": int(sum(row["n"] for row in valid)),
            "positive": int(sum(row.get("positive", 0) for row in valid)),
            "predicted_positive": int(sum(row.get("predicted_positive", 0) for row in valid)),
            **averaged}


def train_federated(sites: Sequence[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]], feature_dim: int,
                    method: str, rounds: int, local_epochs: int, lr: float, prox_mu: float) -> Dict[str, Any]:
    w = np.zeros(feature_dim, dtype=np.float64)
    b = 0.0
    history = []
    for _round in range(rounds):
        updates = []
        weights = []
        for site in sites:
            x_train, y_train, _x_test, _y_test, info = unpack(site)
            local_w, local_b = local_update(w, b, x_train, y_train, local_epochs, lr,
                                            global_w=w if method == "fedprox" else None,
                                            global_b=b if method == "fedprox" else None,
                                            prox_mu=prox_mu if method == "fedprox" else 0.0,
                                            sample_weight=np.array(info.get("sample_weights", np.ones(len(site[1]))))[:len(y_train)])
            updates.append((local_w, local_b))
            weights.append(float(sum(info.get("sample_weights", np.ones(len(site[1])))[:len(y_train)])))
        total = float(sum(weights))
        w = sum(local_w * (weight / total) for (local_w, _), weight in zip(updates, weights))
        b = sum(local_b * (weight / total) for (_, local_b), weight in zip(updates, weights))
        history.append({"round": _round + 1, "sites": len(updates)})
    threshold = select_federated_threshold(w, b, sites)
    site_metrics = []
    for site in sites:
        _x_train, _y_train, x_test, y_test, _ = unpack(site)
        site_metrics.append(metrics(w, b, x_test, y_test, threshold=threshold))
    return {"metrics": average_metrics(site_metrics), "per_site": site_metrics,
            "history": history,
            "communication_bytes": 2 * rounds * len(sites) * (feature_dim + 1) * 8 + len(sites) * 99 * 3 * 8 + len(sites) * 8,
            "model_update_upload_bytes": rounds * len(sites) * (feature_dim + 1) * 8,
            "model_broadcast_bytes": rounds * len(sites) * (feature_dim + 1) * 8,
            "threshold_aggregation_bytes": len(sites) * 99 * 3 * 8 + len(sites) * 8}


def train_local(sites: Sequence[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]], feature_dim: int,
                rounds: int, local_epochs: int, lr: float) -> Dict[str, Any]:
    rows = []
    for site in sites:
        x_train, y_train, x_test, y_test, _ = unpack(site)
        w, b = local_update(np.zeros(feature_dim), 0.0, x_train, y_train, rounds * local_epochs, lr)
        threshold = select_threshold(w, b, x_train, y_train)
        rows.append(metrics(w, b, x_test, y_test, threshold=threshold))
    return {"metrics": average_metrics(rows), "per_site": rows, "history": [], "communication_bytes": 0}


def train_centralized(sites: Sequence[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]], feature_dim: int,
                      rounds: int, local_epochs: int, lr: float) -> Dict[str, Any]:
    trains = [unpack(site) for site in sites]
    x_train = np.concatenate([row[0] for row in trains if len(row[0])], axis=0)
    y_train = np.concatenate([row[1] for row in trains if len(row[1])], axis=0)
    w, b = local_update(np.zeros(feature_dim), 0.0, x_train, y_train, rounds * local_epochs, lr)
    threshold = select_threshold(w, b, x_train, y_train)
    rows = [metrics(w, b, row[2], row[3], threshold=threshold) for row in trains]
    # Centralized transfer is reported as an estimate to make its distinction
    # from model-update communication explicit.
    data_bytes = int(sum(info["records"] for _, _, info in sites) * 160)
    return {"metrics": average_metrics(rows), "per_site": rows, "history": [], "communication_bytes": data_bytes}


def load_group_splits(truth_path: Path, paths: Sequence[Path], seed: int) -> Tuple[Dict[str, Dict[str, bool]], Dict[str, Any]]:
    """Create an offline holdout with the same synthetic patient in one fold."""
    if not truth_path.is_file():
        raise FileNotFoundError(f"Grouped evaluation requires generator truth: {truth_path}")
    site_names = {path.stem: f"logical-site-{path.stem.split('_')[-1]}" for path in paths}
    gold_by_site_patient: Dict[Tuple[str, str], str] = {}
    with truth_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            site, patient, gold = row.get("site_id"), row.get("patient_id"), row.get("gold_patient_id")
            if not site or not patient or not gold:
                continue
            key = (str(site), str(patient))
            if key in gold_by_site_patient and gold_by_site_patient[key] != str(gold):
                raise ValueError(f"Conflicting generator truth for {key}")
            gold_by_site_patient[key] = str(gold)
    golds = sorted(set(gold_by_site_patient.values()),
                   key=lambda gold: hashlib.sha256(f"{seed}:{gold}".encode()).hexdigest())
    if len(golds) < 2:
        raise ValueError("Grouped evaluation requires at least two gold patients")
    n_test = min(len(golds) - 1, max(1, round(len(golds) * 0.2)))
    test_golds = set(golds[:n_test])
    splits: Dict[str, Dict[str, bool]] = {}
    for stem, site_name in site_names.items():
        splits[stem] = {patient: gold in test_golds for (site, patient), gold in gold_by_site_patient.items()
                        if site == site_name}
        if not splits[stem]:
            raise ValueError(f"No generator truth mappings for {site_name}")
    audit = {"policy": "gold_patient_grouped", "truth_path": str(truth_path),
             "unique_gold_patients": len(golds), "train_gold_patients": len(golds) - n_test,
             "test_gold_patients": n_test, "train_test_gold_overlap": 0,
             "site_patient_mappings": sum(len(v) for v in splits.values())}
    return splits, audit


def run_experiment(sites_dir: Path, output: Path, feature_dim: int = 128, rounds: int = 10,
                   local_epochs: int = 2, lr: float = 0.5, prox_mu: float = 0.01,
                   seed: int = 20261011,
                   exclude_event_types: Sequence[str] = ("condition",),
                   truth_path: Optional[Path] = None) -> Dict[str, Any]:
    np.random.seed(seed)
    paths = sorted(sites_dir.glob("site_*.jsonl"))
    if not paths:
        raise ValueError(f"No site_*.jsonl files found under {sites_dir}")
    split_maps, split_audit = load_group_splits(truth_path or sites_dir.parent / "gold_truth.jsonl", paths, seed)
    results: Dict[str, Any] = {"config": {"feature_dim": feature_dim, "rounds": rounds,
                                           "local_epochs": local_epochs, "lr": lr,
                                            "prox_mu": prox_mu, "seed": seed,
                                            "exclude_event_types": list(exclude_event_types),
                                            "missing_patient_policy": "exclude_from_patient_task",
                                            "threshold_selection": "local_train_quantile_f1_or_federated_train_count_f1",
                                            "split_policy": "gold_patient_grouped"},
                               "split_audit": split_audit, "modes": {}}
    entity_weights, linkage_audit = build_entity_weights(paths, split_maps)
    results["entity_linkage"] = linkage_audit
    for mode, aligned in (("raw", False), ("aligned", True), ("normalized_events", True), ("alignment_aware", True)):
        sites = [load_site(path, feature_dim, aligned, exclude_event_types=exclude_event_types,
                           patient_splits=split_maps[path.stem],
                           normalize_events=mode in {"normalized_events", "alignment_aware"},
                           entity_weights=entity_weights[path.stem] if mode == "alignment_aware" else None) for path in paths]
        methods = {
            "local_only": train_local(sites, feature_dim, rounds, local_epochs, lr),
            "fedavg": train_federated(sites, feature_dim, "fedavg", rounds, local_epochs, lr, prox_mu),
            "fedprox": train_federated(sites, feature_dim, "fedprox", rounds, local_epochs, lr, prox_mu),
            "centralized_upper_bound": train_centralized(sites, feature_dim, rounds, local_epochs, lr),
        }
        results["modes"][mode] = {"sites": [info for _, _, info in sites], "methods": methods}
        for method in ("fedavg", "fedprox"):
            result = methods[method]
            sketch_bytes = linkage_audit["sketch_digest_bytes"] if mode == "alignment_aware" else 0
            weight_bytes = sum(fold["profiles"] for fold in linkage_audit["folds"].values()) * 8 if mode == "alignment_aware" else 0
            result["linkage_sketch_upload_bytes"] = sketch_bytes
            result["linkage_weight_download_bytes"] = weight_bytes
            result["communication_bytes"] += sketch_bytes + weight_bytes
            result["communication_scope"] = "numeric/digest payload estimate; excludes identifiers, headers, transport and cryptographic overhead"
        # Weight arrays are local implementation state, not report payloads.
        for info in results["modes"][mode]["sites"]:
            info["effective_train_weight"] = sum(info["sample_weights"][:info["train_count"]])
            info.pop("sample_weights", None)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--prox-mu", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()
    result = run_experiment(args.sites_dir, args.output, feature_dim=args.feature_dim,
                            rounds=args.rounds, local_epochs=args.local_epochs,
                            lr=args.lr, prox_mu=args.prox_mu, seed=args.seed)
    print(json.dumps({"output": str(args.output), "modes": list(result["modes"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
