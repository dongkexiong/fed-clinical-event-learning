# Cohort B replication (v1.1.0)

This directory contains the independently generated Synthea cohort used for Section 4.9 and
Table 10 of the manuscript. It supplements, and does not replace, the primary cohort in
`data/synthea_input/`.

## Contents

| Path | Content |
|---|---|
| `select_cohort_b.py` | Applies the same per-patient caps used for the primary cohort |
| `input/` | Eight selected CSV files and `selection_stats.json` |
| `results/` | `study_summary.json`, one `summary.json` per scenario, and per-seed `federated_results.json`, `alignment_results.json`, `manifest.json` and `validation.json` |
| `logs/` | Synthea generation log and analysis pipeline log |
| `RUN_COHORT_B.ps1` / `RUN_COHORT_B.sh` | End-to-end regeneration |
| `cohort_b_provenance.json` | Generator settings, selection caps, commands, input hashes and environment |
| `../CHANGELOG.md` | v1.1.0 change record |

The per-site CSV files and `gold_truth.jsonl` (about 2.7 GB in total) are reproducible outputs
and are not included. The raw Synthea export (about 1.5 GB) is also excluded; it is regenerated
by the command below.

## Provenance

- Generator: Synthea 4.0.1-SNAPSHOT, the same runnable JAR used for the primary cohort
  (SHA-256 `37596e412f9095164711d2d29b0a2466934a42c846c67551b1e45b1ec95fa7b6`).
- Generation: patient and clinician seeds `20261012`; reference and end date `20261011`;
  requested population 1,000; CSV export only. The export contains 1,152 patients.
- Selection caps: conditions 10, observations 25, encounters 10, medications 10, procedures 10,
  allergies 5, immunizations 5, plus all condition rows whose description contains
  "hypertension" (case-insensitive).
- Selected input: 1,152 patients, 269 positive patients and 78,092 retained events.

Exact commands and per-file SHA-256 values are in `cohort_b_provenance.json`.

## Re-running the analysis

The analysis code is unchanged from the v1.0.0 archive. From the package root:

```bash
python run_alignment_study.py --input-dir cohort_b/input --output-dir cohort_b/results_reproduced
```

This runs the three missingness scenarios and five seeds (15 runs) with the same protocol,
experiment seeds and evaluation rules used for the primary cohort. The resulting
`results_reproduced/study_summary.json` reproduces the values in `results/study_summary.json`
within rounding; site-level CSV files will differ in byte content across platforms.

## Regenerating the input from Synthea

```bash
java -Xmx8g -jar <synthea-4.0.1-SNAPSHOT-with-dependencies.jar> \
  -p 1000 -s 20261012 -cs 20261012 -r 20261011 -e 20261011 \
  --exporter.baseDirectory="<work>/synthea_raw" \
  --exporter.csv.export=true --exporter.fhir.export=false
python cohort_b/select_cohort_b.py --source <work>/synthea_raw/csv --output <work>/input
```

`RUN_COHORT_B.ps1` and `RUN_COHORT_B.sh` wrap both steps.

## Mapping to Table 10

- Cohort B F1 mean and SD: `results/missing_*/summary.json` →
  `summary.federated_metrics.<mode>.fedavg.f1`, with mode keys `raw`, `normalized_events`
  and `alignment_aware`.
- Cohort B paired differences and 95% bootstrap intervals: `results/study_summary.json` →
  `paired_alignment_aware_fedavg_minus_control.<scenario>.normalized_events.<metric>`,
  with metric keys `f1`, `auroc` and `auprc`.
- Cohort A values are the ones documented in the main README and produced by the v1.0.0
  workflow.

## Verification

```bash
python verify_package_hashes.py
```

Every line should report `OK`, and the final line should report `failures 0`.

## Data statement

Both cohorts are synthetic. No real patient data were used, and no real-world clinical
claims are supported by these files.
