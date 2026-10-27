"""Auditable synthetic-event normalization and truth-blind entity linkage.

This is an in-process emulation. Token sketches are not a privacy protocol.
Generator truth, outcome labels, and condition events are never linkage inputs.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def first(row: Mapping[str, Any], *names: str) -> Any:
    return next((row[name] for name in names if row.get(name) not in (None, "")), None)


def canonical_event(row: Mapping[str, Any], index: int = 0) -> dict[str, Any]:
    code = first(row, "concept_code", "local_code", "code")
    code = re.sub(r"^l\d+-", "", re.sub(r"\s+", "", str(code).casefold())) if code else None
    unit = first(row, "unit")
    unit = re.sub(r"\s+", "", str(unit).casefold()) if unit else None
    label = str(first(row, "label", "description") or "").strip().casefold()
    numeric = first(row, "value_numeric", "value_num", "numeric")
    try:
        numeric = float(numeric) if numeric is not None else None
    except (ValueError, TypeError):
        numeric = None
    # The conversion factor is specific to glucose, never arbitrary mg/dL.
    if "glucose" in label or code in {"2339-0", "2345-7"}:
        if unit in {"mmol/l", "mmolperliter"}:
            numeric = numeric * 18.0 if numeric is not None else None
            unit = "mg/dl"
        elif unit in {"mg/dl", "mgdl"}:
            unit = "mg/dl"
    timestamp = first(row, "event_time", "timestamp")
    offset = first(row, "event_offset_min")
    return {"patient_id": first(row, "patient_id", "patientId"),
            "event_type": str(first(row, "event_type", "eventType", "type") or "unknown").casefold(),
            "code": code, "label": label, "numeric": round(numeric, 4) if numeric is not None else None,
            "text": first(row, "value_text", "text"), "unit": unit,
            "time": timestamp, "offset": offset, "index": index}


def event_fingerprint(event: Mapping[str, Any]) -> tuple:
    # Distinct unknown-time events cannot be safely collapsed.
    time = ("absolute", event["time"]) if event["time"] is not None else ("offset", event["offset"])
    if time[1] is None:
        time = ("unknown", event["index"])
    return (event["patient_id"], event["event_type"], event["code"], event["label"],
            event["numeric"], event["text"], event["unit"], time)


def load_linkage_profiles(paths: Sequence[Path]) -> dict[tuple[str, str], set[str]]:
    profiles: dict[tuple[str, str], set[str]] = defaultdict(set)
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                event = canonical_event(row, index)
                patient = event["patient_id"]
                if patient is None:
                    continue
                key = (path.stem, str(patient))
                tokens = profiles[key]
                if event["event_type"] == "condition":
                    continue
                for kind, value in (("code", event["code"]), ("label", event["label"])):
                    if value:
                        token = f"{event['event_type']}:{kind}:{value}"
                        tokens.add(hashlib.sha256(token.encode()).hexdigest())
    return dict(profiles)


def profile_score_blocks(profiles: Mapping[tuple[str, str], set[str]]):
    """Yield bounded cross-site Jaccard score matrices."""
    keys = sorted(profiles)
    vocabulary = {token: idx for idx, token in enumerate(sorted(set().union(*profiles.values())))}
    matrix = np.zeros((len(keys), len(vocabulary)), dtype=np.float32)
    for index, key in enumerate(keys):
        if profiles[key]:
            matrix[index, [vocabulary[token] for token in profiles[key]]] = 1
    sizes = matrix.sum(axis=1)
    by_site: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        by_site[key[0]].append(index)
    site_names = sorted(by_site)
    for i, site in enumerate(site_names):
        left_indices = np.array(by_site[site])
        for other in site_names[i + 1:]:
            right_indices = np.array(by_site[other])
            intersection = matrix[left_indices] @ matrix[right_indices].T
            union = sizes[left_indices, None] + sizes[right_indices][None, :] - intersection
            scores = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
            yield [keys[idx] for idx in left_indices], [keys[idx] for idx in right_indices], scores


def mutual_best_links(profiles: Mapping[tuple[str, str], set[str]], threshold: float = 0.5) -> tuple[list, dict]:
    """Unique reciprocal maxima, computed by logical-site matrix blocks."""
    edges = []
    pairs = 0
    for left_keys, right_keys, scores in profile_score_blocks(profiles):
        pairs += scores.size
        right_best = scores.argmax(axis=1)
        left_best = scores.argmax(axis=0)
        row_max = scores.max(axis=1)
        col_max = scores.max(axis=0)
        unique_row = np.sum(np.isclose(scores, row_max[:, None], atol=1e-7, rtol=0), axis=1) == 1
        unique_col = np.sum(np.isclose(scores, col_max[None, :], atol=1e-7, rtol=0), axis=0) == 1
        for li, ri in enumerate(right_best):
            if row_max[li] >= threshold and unique_row[li] and unique_col[ri] and left_best[ri] == li:
                edges.append((left_keys[li], right_keys[ri], float(row_max[li])))
    return edges, {"candidate_pairs": pairs, "predicted_pairs": len(edges),
                   "profiles": len(profiles), "sketch_digest_bytes": sum(len(tokens) for tokens in profiles.values()) * 32,
                   "excluded_linkage_event_types": ["condition"], "threshold": threshold}


def build_entity_weights(paths: Sequence[Path], splits: Mapping[str, Mapping[str, bool]]) -> tuple[dict, dict]:
    """Weight shared replicas by 1/cluster_size with one identity/site/cluster."""
    profiles = load_linkage_profiles(paths)
    weights = {stem: {} for stem in splits}
    audits = {}
    for fold, test in (("train", False), ("test", True)):
        subset = {k: v for k, v in profiles.items() if splits[k[0]][k[1]] == test}
        edges, audit = mutual_best_links(subset)
        parent = {key: key for key in subset}
        members = {key: {key} for key in subset}

        def root(key: tuple) -> tuple:
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        rejected = 0
        for left, right, score in sorted(edges, key=lambda e: (-e[2], e[0], e[1])):
            a, b = root(left), root(right)
            if a == b:
                continue
            if {key[0] for key in members[a]} & {key[0] for key in members[b]}:
                rejected += 1
                continue
            parent[b] = a
            members[a] |= members.pop(b)
        for key in subset:
            weights[key[0]][key[1]] = 1.0 / len(members[root(key)])
        audits[fold] = {**audit, "clusters": len(members), "rejected_site_conflicts": rejected,
                        "effective_patient_weight": sum(weights[k[0]][k[1]] for k in subset)}
    return weights, {"method": "noncondition_mutual_best_inverse_cluster_weight",
                     "truth_used_for_matching": False, "folds": audits,
                     "sketch_digest_bytes": sum(a["sketch_digest_bytes"] for a in audits.values())}
