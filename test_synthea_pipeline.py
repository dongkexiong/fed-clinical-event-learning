"""Code-level checks for the synthetic multi-site experiment fixture."""

import json
import tempfile
import unittest
from pathlib import Path

from generate_synthea_sites import apply_target_condition, generate_sites, load_synthea
from federated_baselines import load_group_splits, load_site, local_update, rank_metrics, run_experiment, select_federated_threshold
from evaluate_synthetic_alignment import evaluate
from validate_synthetic_artifacts import validate_directory
from alignment_aware import canonical_event, event_fingerprint, load_linkage_profiles, mutual_best_links


ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "testdata" / "synthea_fixture"


class SyntheaSiteGenerationTests(unittest.TestCase):
    def test_loader_reads_fixture_and_preserves_synthetic_truth_labels(self):
        events = load_synthea(FIXTURE)
        self.assertEqual(len(events), 7)
        self.assertEqual({event["gold_patient_id"] for event in events}, {"p1", "p2", "p3", "p4"})
        self.assertGreaterEqual(sum(event["target_label"] for event in events), 3)

    def test_target_definition_is_specific_and_condition_events_are_not_features(self):
        events = load_synthea(FIXTURE)
        events, definition = apply_target_condition(events, "diabetes")
        self.assertEqual(definition["keyword"], "diabetes")
        self.assertTrue(all(event["target_label"] in (0, 1) for event in events))
        labels_by_patient = {}
        for event in events:
            labels_by_patient.setdefault(event["gold_patient_id"], set()).add(event["target_label"])
        self.assertTrue(all(len(labels) == 1 for labels in labels_by_patient.values()))
        with tempfile.TemporaryDirectory() as output:
            output_path = Path(output)
            generate_sites(events, output_path, sites=2, seed=31, duplicate_rate=0,
                           target_definition=definition)
            result = run_experiment(output_path / "sites", output_path / "results.json",
                                    feature_dim=32, rounds=2, local_epochs=1)
            for mode in result["modes"].values():
                for info in mode["sites"]:
                    self.assertGreater(info["excluded_records"], 0)

    def test_generation_is_deterministic_and_uses_site_local_ids(self):
        events = load_synthea(FIXTURE)
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = generate_sites(events, Path(first_dir), sites=3, seed=7, duplicate_rate=0)
            second = generate_sites(events, Path(second_dir), sites=3, seed=7, duplicate_rate=0)
            self.assertEqual(first["output_records"], second["output_records"])
            for site in first["sites"]:
                first_bytes = (Path(first_dir) / site["path"]).read_bytes()
                second_bytes = (Path(second_dir) / site["path"]).read_bytes()
                self.assertEqual(first_bytes, second_bytes)
            rows = [json.loads(line) for line in (Path(first_dir) / "sites" / "site_1.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(rows)
            self.assertTrue(all("gold_patient_id" not in row and "gold_event_id" not in row for row in rows))
            self.assertTrue(all(str(row.get("patient_id", row.get("patientId", ""))).startswith("site1-") for row in rows))

    def test_duplicate_truth_is_explicit(self):
        events = load_synthea(FIXTURE)
        with tempfile.TemporaryDirectory() as output:
            manifest = generate_sites(events, Path(output), sites=2, seed=3, duplicate_rate=1.0)
            truth_lines = (Path(output) / "gold_truth.jsonl").read_text(encoding="utf-8").splitlines()
            truth = [json.loads(line) for line in truth_lines]
            self.assertEqual(manifest["gold_truth"]["records"], len(truth))
            self.assertTrue(any("duplicate_of" in row for row in truth))
            for site in (Path(output) / "sites").glob("site_*.jsonl"):
                rows = [json.loads(line) for line in site.read_text(encoding="utf-8").splitlines()]
                self.assertTrue(all("duplicate_of" not in row for row in rows))
                event_ids = [row.get("eventId", row.get("event_id")) for row in rows]
                self.assertEqual(len(event_ids), len(set(event_ids)))

    def test_federated_baselines_emit_raw_and_aligned_modes(self):
        events = load_synthea(FIXTURE)
        with tempfile.TemporaryDirectory() as output:
            output_path = Path(output)
            generate_sites(events, output_path / "sites_out", sites=3, seed=11, duplicate_rate=0)
            result = run_experiment(output_path / "sites_out" / "sites", output_path / "results.json",
                                    feature_dim=32, rounds=2, local_epochs=1)
            self.assertEqual(set(result["modes"]), {"raw", "aligned", "normalized_events", "alignment_aware"})
            for mode in result["modes"].values():
                self.assertIn("fedavg", mode["methods"])
                self.assertIn("fedprox", mode["methods"])
                self.assertGreaterEqual(mode["methods"]["fedavg"]["communication_bytes"], 1)
            self.assertTrue((output_path / "results.json").exists())

    def test_alignment_evaluator_uses_generator_truth_only(self):
        events = load_synthea(FIXTURE)
        with tempfile.TemporaryDirectory() as output:
            output_path = Path(output)
            generate_sites(events, output_path / "sites_out", sites=3, seed=17, duplicate_rate=1.0)
            result = evaluate(output_path / "sites_out" / "gold_truth.jsonl",
                              output_path / "sites_out" / "sites", threshold=0.2)
            self.assertTrue(result["entity_alignment"]["raw"]["synthetic_ground_truth"])
            self.assertTrue(result["entity_alignment"]["aligned"]["synthetic_ground_truth"])
            self.assertTrue(result["entity_alignment"]["mutual_best"]["synthetic_ground_truth"])
            self.assertGreater(result["duplicate_detection"]["truth_duplicates"], 0)

    def test_artifact_validator_rejects_truth_leakage(self):
        events = load_synthea(FIXTURE)
        with tempfile.TemporaryDirectory() as output:
            output_path = Path(output)
            generate_sites(events, output_path, sites=3, seed=19, duplicate_rate=0)
            self.assertTrue(validate_directory(output)["valid"])
            site_path = output_path / "sites" / "site_1.jsonl"
            text = site_path.read_text(encoding="utf-8")
            site_path.write_text(text.replace("\"schema_variant\":\"canonical\"", "\"gold_event_id\":\"leak\",\"schema_variant\":\"canonical\"", 1), encoding="utf-8")
            self.assertFalse(validate_directory(output)["valid"])

    def test_missing_patient_records_are_reported_not_promoted_to_patients(self):
        with tempfile.TemporaryDirectory() as output:
            site = Path(output) / "site_1.jsonl"
            site.write_text(
                '{"patient_id":"p1","event_type":"observation","concept_code":"A","target_label":1}\n'
                '{"event_type":"condition","concept_code":"B","target_label":1}\n',
                encoding="utf-8",
            )
            _x, _y, info = load_site(site, feature_dim=8, aligned=True)
            self.assertEqual(info["patients"], 1)
            self.assertEqual(info["missing_patient_records"], 1)
            self.assertEqual(info["missing_patient_positive_records"], 1)

    def test_rank_metrics_uses_high_score_as_positive_evidence(self):
        import numpy as np
        auroc, auprc = rank_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
        self.assertAlmostEqual(auroc, 1.0)
        self.assertAlmostEqual(auprc, 1.0)
        tied_auc, tied_ap = rank_metrics(np.array([0, 0, 1, 1]), np.ones(4))
        self.assertAlmostEqual(tied_auc, 0.5)
        self.assertAlmostEqual(tied_ap, 0.5)

    def test_grouped_split_keeps_shared_synthetic_patient_in_one_fold(self):
        events = load_synthea(FIXTURE)
        with tempfile.TemporaryDirectory() as output:
            directory = Path(output)
            generate_sites(events, directory, sites=3, seed=11, shared_rate=1.0, duplicate_rate=0)
            paths = sorted((directory / "sites").glob("site_*.jsonl"))
            split_maps, audit = load_group_splits(directory / "gold_truth.jsonl", paths, seed=11)
            self.assertEqual(audit["train_test_gold_overlap"], 0)
            by_gold = {}
            for line in (directory / "gold_truth.jsonl").read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row.get("patient_id"):
                    stem = "site_" + row["site_id"].split("-")[-1]
                    by_gold.setdefault(row["gold_patient_id"], set()).add(split_maps[stem][row["patient_id"]])
            self.assertTrue(all(len(folds) == 1 for folds in by_gold.values()))

    def test_federated_threshold_uses_aggregate_training_counts(self):
        import numpy as np
        sites = [
            (np.array([[0.1], [0.8], [0.0]]), np.array([0., 1., 0.]), {"x_train_end": 2}),
            (np.array([[0.2], [0.9], [0.0]]), np.array([0., 1., 0.]), {"x_train_end": 2}),
        ]
        threshold = select_federated_threshold(np.array([1.0]), 0.0, sites)
        self.assertTrue(0.55 <= threshold <= 0.69)

    def test_unit_conversion_and_duplicate_key_preserve_distinct_observations(self):
        base = {"patient_id": "p", "event_type": "observation", "label": "Glucose", "concept_code": "2345-7",
                "value_numeric": 5.0, "unit": "mmol/L", "event_time": "2020-01-01"}
        event = canonical_event(base)
        self.assertEqual((event["numeric"], event["unit"]), (90.0, "mg/dl"))
        other = canonical_event({**base, "value_numeric": 6.0})
        self.assertNotEqual(event_fingerprint(event), event_fingerprint(other))
        cholesterol = canonical_event({**base, "label": "Cholesterol", "concept_code": "2093-3"})
        self.assertEqual((cholesterol["numeric"], cholesterol["unit"]), (5.0, "mmol/l"))

    def test_linkage_ignores_outcomes_and_condition_events_and_abstains_on_ties(self):
        with tempfile.TemporaryDirectory() as output:
            root = Path(output)
            paths = []
            for site in (1, 2):
                path = root / f"site_{site}.jsonl"
                row = {"patient_id": "p", "event_type": "observation", "concept_code": "A", "target_label": site - 1}
                condition = {"patient_id": "p", "event_type": "condition", "concept_code": f"C{site}", "label": "hypertension"}
                path.write_text(json.dumps(row) + '\n' + json.dumps(condition) + '\n', encoding="utf-8")
                paths.append(path)
            profiles = load_linkage_profiles(paths)
            self.assertEqual(profiles[("site_1", "p")], profiles[("site_2", "p")])
        edges, _audit = mutual_best_links({("a", "1"): {"x"}, ("b", "2"): {"x"}, ("b", "3"): {"x"}})
        self.assertEqual(edges, [])

    def test_inverse_replica_weights_restore_single_patient_gradient(self):
        import numpy as np
        x = np.array([[1.0], [0.0]])
        y = np.array([1.0, 0.0])
        original = local_update(np.zeros(1), 0.0, x, y, epochs=1, lr=0.5)
        duplicated = local_update(np.zeros(1), 0.0, np.array([[1.0], [1.0], [0.0]]),
                                  np.array([1., 1., 0.]), epochs=1, lr=0.5,
                                  sample_weight=np.array([0.5, 0.5, 1.]))
        np.testing.assert_allclose(original[0], duplicated[0])
        self.assertAlmostEqual(original[1], duplicated[1])


if __name__ == "__main__":
    unittest.main()
