"""Generate reproducible logical sites from Synthea CSV or FHIR JSON output.

This module deliberately treats Synthea as a synthetic source.  The generated
site files contain site-local identifiers and perturbations; the separate truth
file is only for offline evaluation.  No patient identity is inferred across
independent datasets.

Examples
--------
    python generate_synthea_sites.py --input-dir ./synthea/output/csv \
        --output-dir ./synthea_sites --sites 3 --seed 20261011

    python generate_synthea_sites.py --input-dir ./tests/fixture_synthea \
        --output-dir ./synthea_sites_fixture --sites 3 --shared-rate 0.8
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


DEFAULT_SEED = 20261011
CSV_EVENT_FILES = {
    "conditions.csv": "condition",
    "observations.csv": "observation",
    "encounters.csv": "encounter",
    "medications.csv": "medication",
    "procedures.csv": "procedure",
    "allergies.csv": "allergy",
    "immunizations.csv": "immunization",
}


def clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    return value if value and value.lower() not in {"null", "none", "nan", "\\n"} else None


def first(row: Mapping[str, Any], *names: str) -> Optional[str]:
    lower = {str(k).casefold(): v for k, v in row.items()}
    for name in names:
        value = clean(row.get(name))
        if value is None:
            value = clean(lower.get(name.casefold()))
        if value is not None:
            return value
    return None


def normalize_datetime(value: Optional[str]) -> Optional[str]:
    value = clean(value)
    if not value:
        return None
    # Synthea CSV uses ISO dates/timestamps. Keep second precision and UTC-less
    # semantics explicit; this is a synthetic event clock, not clinical time.
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.replace(microsecond=0).isoformat()
    except ValueError:
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(value, fmt).isoformat()
            except ValueError:
                continue
    return value


def stable_hash(*parts: Any, length: int = 16) -> str:
    payload = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def parse_number(value: Optional[str]) -> Optional[float]:
    value = clean(value)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def canonical_from_csv(path: Path, event_type: str) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader):
            source_patient = first(row, "PATIENT", "patient", "patient_id", "patientId")
            if not source_patient:
                # A row without a patient cannot participate in entity alignment,
                # but retaining it makes missingness behavior observable.
                source_patient = f"missing-patient-row-{row_number}"
            event_id = first(row, "Id", "ID", "id", "event_id") or f"{path.name}:{row_number}"
            code = first(row, "CODE", "code", "ICD10", "ICD-10", "SNOMED")
            label = first(row, "DESCRIPTION", "description", "REASONDESCRIPTION", "REASON")
            if event_type == "observation":
                label = first(row, "DESCRIPTION", "description", "TYPE", "type") or label
            event_time = first(row, "DATE", "date", "START", "start", "START_DATETIME", "effectiveDateTime")
            stop_time = first(row, "STOP", "stop", "END", "end")
            value_text = first(row, "VALUE", "value", "VALUE_STRING", "valueString")
            numeric = parse_number(value_text)
            unit = first(row, "UNITS", "unit", "UNIT", "valueUnit")
            yield {
                "gold_patient_id": source_patient,
                "gold_event_id": f"{path.name}:{event_id}",
                "event_type": event_type,
                "concept_code": code,
                "concept_system": "synthea-source",
                "label": label,
                "value_text": value_text,
                "value_numeric": numeric,
                "unit": unit,
                "event_time": normalize_datetime(event_time),
                "event_end_time": normalize_datetime(stop_time),
                # This synthetic downstream label is intentionally simple and
                # generator-defined: a patient is positive when a condition
                # resource is present. It is not a clinical outcome label.
                "target_label": 1 if event_type == "condition" else 0,
                "source_file": path.name,
                "source_row": row_number,
            }


def _fhir_reference(resource: Mapping[str, Any], field: str) -> Optional[str]:
    value = resource.get(field)
    if isinstance(value, Mapping):
        reference = clean(value.get("reference"))
        if reference:
            return reference.split("/")[-1]
    return None


def _fhir_code(resource: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    codeable = resource.get("code") or resource.get("valueCodeableConcept")
    if not isinstance(codeable, Mapping):
        return None, None, None
    codings = codeable.get("coding")
    coding = codings[0] if isinstance(codings, list) and codings else codeable
    if not isinstance(coding, Mapping):
        return None, None, None
    return clean(coding.get("code")), clean(coding.get("system")), clean(coding.get("display"))


def _iter_fhir_resources(value: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if isinstance(value.get("resourceType"), str):
            yield value
        if isinstance(value.get("entry"), list):
            for entry in value["entry"]:
                if isinstance(entry, Mapping):
                    resource = entry.get("resource", entry)
                    yield from _iter_fhir_resources(resource)
        # Also support a directory-level JSON array or a nested bundle.
        for child in value.values():
            if isinstance(child, (Mapping, list)) and child is not value:
                yield from _iter_fhir_resources(child)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_fhir_resources(item)


def canonical_from_fhir(path: Path) -> Iterator[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    seen: set[Tuple[str, str]] = set()
    for resource in _iter_fhir_resources(data):
        resource_type = clean(resource.get("resourceType")) or "resource"
        resource_id = clean(resource.get("id")) or stable_hash(path.name, json.dumps(resource, sort_keys=True))
        key = (resource_type, resource_id)
        if key in seen:
            continue
        seen.add(key)
        patient = _fhir_reference(resource, "subject") or _fhir_reference(resource, "patient")
        if not patient:
            # Patient resources themselves are retained as an entity anchor.
            patient = resource_id if resource_type == "Patient" else f"missing-patient-{resource_id}"
        code, system, display = _fhir_code(resource)
        value = resource.get("valueQuantity") if isinstance(resource.get("valueQuantity"), Mapping) else {}
        value_numeric = value.get("value") if isinstance(value, Mapping) else None
        unit = clean(value.get("unit")) if isinstance(value, Mapping) else None
        event_time = resource.get("effectiveDateTime") or resource.get("issued")
        if not event_time and isinstance(resource.get("effectivePeriod"), Mapping):
            event_time = resource["effectivePeriod"].get("start")
        event_type = {
            "Condition": "condition", "Observation": "observation", "Encounter": "encounter",
            "MedicationRequest": "medication", "Procedure": "procedure", "AllergyIntolerance": "allergy",
        }.get(resource_type, resource_type.casefold())
        yield {
            "gold_patient_id": patient,
            "gold_event_id": f"{resource_type}:{resource_id}",
            "event_type": event_type,
            "concept_code": code,
            "concept_system": system or "fhir-source",
            "label": display or clean(resource.get("text")),
            "value_text": clean(value.get("value")) if isinstance(value, Mapping) else None,
            "value_numeric": parse_number(str(value_numeric)) if value_numeric is not None else None,
            "unit": unit,
            "event_time": normalize_datetime(str(event_time)) if event_time else None,
            "event_end_time": None,
            "target_label": 1 if event_type == "condition" else 0,
            "source_file": path.name,
            "source_row": None,
        }


def load_synthea(input_path: Path) -> List[Dict[str, Any]]:
    """Load Synthea CSV directory, a CSV file, or FHIR JSON file(s)."""
    paths: List[Path]
    if input_path.is_dir():
        paths = sorted(input_path.iterdir())
    else:
        paths = [input_path]
    events: List[Dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            continue
        lower = path.name.casefold()
        if lower.endswith(".csv"):
            event_type = CSV_EVENT_FILES.get(lower)
            if event_type:
                events.extend(canonical_from_csv(path, event_type))
        elif lower.endswith(".json"):
            events.extend(canonical_from_fhir(path))
    if not events:
        raise ValueError(f"No supported Synthea CSV/FHIR JSON event files found under {input_path}")
    return events


def apply_target_condition(events: Sequence[Mapping[str, Any]], keyword: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Derive a reproducible patient task label from a condition description.

    The original loader marked every condition resource as positive.  That
    produces an almost-all-positive task for Synthea and leaks the event type
    into the feature matrix.  The formal experiment instead freezes one
    condition-description keyword (for example, ``hypertension``), then copies
    the resulting patient-level status to the patient's supervised records.
    Labels are excluded from feature extraction and entity linkage.
    """
    needle = clean(keyword)
    if not needle:
        raise ValueError("target condition keyword must not be empty")
    needle = needle.casefold()
    updated: List[Dict[str, Any]] = []
    positive_events = 0
    positive_patients: set[str] = set()
    patients: set[str] = set()
    for event in events:
        row = dict(event)
        patient = clean(row.get("gold_patient_id")) or "missing"
        patients.add(patient)
        event_type = clean(row.get("event_type"))
        label = clean(row.get("label")) or ""
        is_positive = event_type == "condition" and needle in label.casefold()
        row["target_label"] = 1 if is_positive else 0
        if is_positive:
            positive_events += 1
            positive_patients.add(patient)
        updated.append(row)
    # Supervised labels describe the patient, including when a diagnosis row
    # loses its patient ID at one logical site. They are never feature tokens.
    for row in updated:
        row["target_label"] = int((clean(row.get("gold_patient_id")) or "missing") in positive_patients)
    return updated, {
        "keyword": keyword,
        "event_type": "condition",
        "matching_rule": "casefold substring in label",
        "patient_label_rule": "any matching condition in retained source input; copied to all patient events",
        "events": len(updated),
        "positive_events": positive_events,
        "patients": len(patients),
        "positive_patients": len(positive_patients),
    }


def _site_public_patient(seed: int, site_index: int, gold_patient: Optional[str]) -> str:
    return f"site{site_index + 1}-p-{stable_hash(seed, site_index, gold_patient)}"


def _jitter_time(value: Optional[str], minutes: int, rng: random.Random) -> Optional[str]:
    if not value or not minutes:
        return value
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    delta = rng.randint(-minutes, minutes)
    return (parsed + timedelta(minutes=delta)).isoformat()


def _convert_value(value: Optional[float], unit: Optional[str], site_index: int,
                   description: Optional[str] = None) -> Tuple[Optional[float], Optional[str]]:
    if value is None or not unit:
        return value, unit
    normalized = unit.casefold().replace(" ", "")
    if "glucose" not in str(description or "").casefold():
        return value, unit
    # A transparent, reversible unit perturbation used only when the source
    # observation is recognizably glucose.  Other units are left untouched.
    if site_index % 2 == 1 and normalized in {"mg/dl", "mgdl"}:
        return round(value / 18.0, 5), "mmol/L"
    if site_index % 2 == 1 and normalized in {"mmol/l", "mmolperliter"}:
        return round(value * 18.0, 5), "mg/dL"
    return value, unit


def _variant_record(record: Dict[str, Any], site_index: int, variant: str) -> Dict[str, Any]:
    if variant == "wide":
        return {
            "patientId": record["patient_id"], "eventId": record["event_id"],
            "eventType": record["event_type"], "timestamp": record["event_time"],
            "local_code": record["concept_code"], "code_system": record["concept_system"],
            "description": record["label"], "value_num": record["value_numeric"],
            "value_text": record["value_text"], "unit": record["unit"],
            "target_label": record["target_label"], "site_id": record["site_id"],
            "schema_variant": variant,
        }
    if variant == "offset":
        return {
            "patient_id": record["patient_id"], "event_id": record["event_id"],
            "type": record["event_type"], "event_offset_min": record["event_offset_min"],
            "code": record["concept_code"], "system": record["concept_system"],
            "label": record["label"], "numeric": record["value_numeric"],
            "text": record["value_text"], "unit": record["unit"],
            "target_label": record["target_label"], "site_id": record["site_id"],
            "schema_variant": variant,
        }
    # Keep generator identifiers exclusively in gold_truth.jsonl.  Site files
    # may contain a synthetic task label, but never gold patient/event keys.
    return {key: value for key, value in record.items() if key != "gold_event_id"} | {"schema_variant": variant}


def generate_sites(
    events: Sequence[Mapping[str, Any]],
    output_dir: Path,
    sites: int = 3,
    seed: int = DEFAULT_SEED,
    shared_rate: float = 0.7,
    missing_rate: float = 0.1,
    format_noise_rate: float = 0.1,
    duplicate_rate: float = 0.05,
    time_jitter_minutes: int = 5,
    target_definition: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if not 1 <= sites <= 20:
        raise ValueError("sites must be between 1 and 20")
    for name, value in (("shared_rate", shared_rate), ("missing_rate", missing_rate),
                        ("format_noise_rate", format_noise_rate), ("duplicate_rate", duplicate_rate)):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    output_dir.mkdir(parents=True, exist_ok=True)
    site_dir = output_dir / "sites"
    site_dir.mkdir(exist_ok=True)
    truth_path = output_dir / "gold_truth.jsonl"
    rng = random.Random(seed)
    by_patient: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        by_patient[str(event.get("gold_patient_id") or "missing")].append(event)

    site_events: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    truth_rows: List[Dict[str, Any]] = []
    event_count = Counter()
    for gold_patient, patient_events in sorted(by_patient.items()):
        primary = stable_hash(seed, gold_patient, length=8)
        primary_site = int(primary, 16) % sites
        assigned = {primary_site}
        for site_index in range(sites):
            if site_index != primary_site and rng.random() < shared_rate:
                assigned.add(site_index)
        for site_index in sorted(assigned):
            variant = ("canonical", "wide", "offset")[site_index % 3]
            patient_id = _site_public_patient(seed, site_index, gold_patient)
            for source_event in patient_events:
                if rng.random() < 0.02 and site_index > 0:
                    # Site-specific non-IID thinning; the truth remains intact.
                    continue
                event_id = f"site{site_index + 1}-e-{stable_hash(seed, site_index, source_event.get('gold_event_id'))}"
                code = clean(source_event.get("concept_code"))
                if code and rng.random() < format_noise_rate:
                    code = re.sub(r"\s+", "", code).casefold()
                if code and site_index > 0:
                    code = f"L{site_index + 1}-{code}"
                unit = clean(source_event.get("unit"))
                numeric, unit = _convert_value(source_event.get("value_numeric"), unit, site_index,
                                               clean(source_event.get("label")))
                event_time = _jitter_time(clean(source_event.get("event_time")), time_jitter_minutes, rng)
                missing = rng.random() < missing_rate
                if missing:
                    # Missingness is field-level, leaving provenance and labels.
                    code = None
                    numeric = None
                    unit = None
                event_offset = None
                if event_time:
                    event_offset = int((int(stable_hash(event_time, site_index, length=6), 16) % 2880) - 1440)
                base = {
                    "patient_id": None if rng.random() < missing_rate / 2 else patient_id,
                    "event_id": event_id,
                    "site_id": f"logical-site-{site_index + 1}",
                    "event_type": clean(source_event.get("event_type")) or "unknown",
                    "event_time": event_time,
                    "event_offset_min": event_offset,
                    "concept_code": code,
                    "concept_system": "site-local" if site_index > 0 else clean(source_event.get("concept_system")),
                    "label": clean(source_event.get("label")),
                    "value_numeric": numeric,
                    "value_text": clean(source_event.get("value_text")),
                    "unit": unit,
                    "target_label": int(source_event.get("target_label") or 0),
                    "gold_event_id": source_event.get("gold_event_id"),
                }
                site_record = _variant_record(base, site_index, variant)
                site_events[site_index].append(site_record)
                truth_rows.append({
                    "site_id": base["site_id"], "patient_id": base["patient_id"],
                    "event_id": event_id, "gold_patient_id": gold_patient,
                    "gold_event_id": source_event.get("gold_event_id"),
                    "target_label": base["target_label"],
                })
                event_count[base["event_type"]] += 1
                if rng.random() < duplicate_rate:
                    duplicate = dict(site_record)
                    duplicate_id = f"site{site_index + 1}-e-{stable_hash(seed, site_index, source_event.get('gold_event_id'), 'duplicate')}"
                    duplicate["eventId" if variant == "wide" else "event_id"] = duplicate_id
                    site_events[site_index].append(duplicate)
                    truth_rows.append({
                        "site_id": base["site_id"], "patient_id": base["patient_id"],
                        "event_id": duplicate_id, "gold_patient_id": gold_patient,
                        "gold_event_id": source_event.get("gold_event_id"),
                        "duplicate_of": event_id, "target_label": base["target_label"],
                    })

    def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
        digest = hashlib.sha256()
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                line = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                handle.write(line)
                digest.update(line.encode("utf-8"))
        return digest.hexdigest()

    site_manifest = []
    for site_index in range(sites):
        path = site_dir / f"site_{site_index + 1}.jsonl"
        rows = site_events.get(site_index, [])
        digest = write_jsonl(path, rows)
        site_manifest.append({
            "site_id": f"logical-site-{site_index + 1}",
            "path": str(path.relative_to(output_dir)),
            "records": len(rows),
            "patients": len({row.get("patientId", row.get("patient_id")) for row in rows}),
            "sha256": digest,
            "schema_variant": ("canonical", "wide", "offset")[site_index % 3],
        })
    truth_digest = write_jsonl(truth_path, truth_rows)
    config = {
        "seed": seed, "sites": sites, "shared_rate": shared_rate,
        "missing_rate": missing_rate, "format_noise_rate": format_noise_rate,
        "duplicate_rate": duplicate_rate, "time_jitter_minutes": time_jitter_minutes,
    }
    if target_definition is not None:
        config["target_definition"] = dict(target_definition)
    manifest = {
        "generator": "generate_synthea_sites.py",
        "source": "Synthea CSV/FHIR synthetic records",
        "synthetic_ground_truth": True,
        "config": config,
        "input_events": len(events),
        "input_patients": len(by_patient),
        "output_records": sum(len(x) for x in site_events.values()),
        "event_type_counts": dict(sorted(event_count.items())),
        "sites": site_manifest,
        "gold_truth": {"path": str(truth_path.relative_to(output_dir)), "records": len(truth_rows), "sha256": truth_digest},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Synthea CSV directory or FHIR JSON file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sites", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--shared-rate", type=float, default=0.7)
    parser.add_argument("--missing-rate", type=float, default=0.1)
    parser.add_argument("--format-noise-rate", type=float, default=0.1)
    parser.add_argument("--duplicate-rate", type=float, default=0.05)
    parser.add_argument("--time-jitter-minutes", type=int, default=5)
    parser.add_argument("--target-condition-keyword", default="hypertension",
                        help="condition label substring used for the synthetic task label")
    args = parser.parse_args()
    events = load_synthea(args.input_dir)
    events, target_definition = apply_target_condition(events, args.target_condition_keyword)
    manifest = generate_sites(events, args.output_dir, sites=args.sites, seed=args.seed,
                              shared_rate=args.shared_rate, missing_rate=args.missing_rate,
                              format_noise_rate=args.format_noise_rate, duplicate_rate=args.duplicate_rate,
                              time_jitter_minutes=args.time_jitter_minutes,
                              target_definition=target_definition)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
