# Revised Synthetic Alignment Study Protocol v3

## Status and Rationale

This is an exploratory revision following earlier diagnostic runs, **not a prospective preregistration**. Earlier v1/v2 outputs served as historical diagnostics; subsequent paper reporting uses v3. The revision addresses cross-site train/test patient overlap caused by splitting site-local IDs, leakage of event-level duplicate truth, application of mg/dL conversion beyond glucose, and changes to patient labels caused by missing diagnostic records.

## Settings Fixed Before This Study Run

- Input: `data/synthea_input/csv`; 1,155 synthetic patients and 78,515 retained events after truncation; all hypertension condition events are retained.
- Target: patient status defined by whether the retained input contains a hypertension condition description. The supervised label is copied to every event for that patient. Labels and all condition events are excluded from predictive features and training linkage.
- Split by source synthetic patient: 924 training / 231 test patients, rather than splitting site-local IDs. Ground truth is used only for offline splitting and scoring.
- Three missingness scenarios: 0%, 10%, and 30%. Patient-ID missingness is injected at half the scenario probability; joint code/value/unit missingness uses the full scenario probability.
- Five fixed seeds: 11, 22, 33, 44, and 55; five sites; shared-patient rate 0.7; format perturbation rate 0.1; duplicate rate 0.05; timestamp jitter of ±5 minutes.
- NumPy logistic regression; 128 hashed features; 20 rounds; two full-batch gradient steps per round; lr=0.5; FedProx μ=0.01.
- Local-only, FedAvg, FedProx, and a centralized reference are retained for every scenario. The centralized reference is not a guaranteed performance upper bound.

## Ablations and Combined Method

| Mode | Processing |
|---|---|
| raw | Read fields consistently; retain site-specific code namespaces |
| aligned | Remove generator-injected `L<site>-` code prefixes |
| normalized_events | aligned + glucose unit normalization + within-site deduplication using complete event fingerprints |
| alignment_aware | normalized_events + reciprocal unique-best matching on non-condition code/description tokens + inverse cluster-size training weights for shared patients |

Entity matching uses Jaccard similarity ≥0.5 and abstains when the best similarity is tied. Each linked cluster contains at most one identity per site; conflicting joins are rejected. Training and test folds are matched separately; **test labels are not used for matching or training**. Supervised features contain only counts of non-condition event types, codes, and units. Numerical values are used only for normalization and duplicate detection, not for classification features. Local-only and centralized references retain the same feature modes. Inverse cluster weights are applied to federated gradients and threshold counts.

Site communication and computation are simulated in one process. Hashed token digests may disclose concepts. No PSI, secure aggregation, differential privacy, deployment isolation, or cryptographic guarantee is implemented. Model uploads, broadcasts, threshold statistics, digest uploads, and weight downloads are estimated as payload bytes, excluding identifiers, headers, transport, and encryption overhead. Digests implement a reproducible alignment mechanism and must not be presented as providing privacy protection.

## Thresholds, Metrics, and Limits of Inference

Local-only selects thresholds from training-set quantile candidates. FedAvg/FedProx select a global threshold on a fixed 0.01–0.99 grid using site training TP/FP/FN counts. Thresholds are fitted using training counts without an independent validation fold; this limitation must be reported. All test metrics use held-out synthetic patients. Site metrics receive equal weight; patient replicas are not independent real patients.

Report accuracy, balanced accuracy, F1, AUROC, average precision (AP in the tables, distinct from trapezoidal PR-AUC), and log loss; per-seed results, SD, 2,000-resample bootstrap intervals, and paired differences. Intervals describe split and perturbation variation in one fixed synthetic population, **not uncertainty across independent clinical populations**. Labels describe synthetic record status, and features may contain treatment information after diagnosis; the task must not be described as future disease risk prediction.

## Execution and Checks

The historical protocol used different workspace paths. In this release package, use the included `data/synthea_input/csv` input and the run commands in `README.md`.

```powershell
python -m unittest -v test_synthea_pipeline.py
python run_alignment_study.py --input-dir data\synthea_input\csv --output-dir reproduction_output
```

Each seed must pass structural and truth-isolation checks before training and scoring. Outputs retain code/input SHA-256 hashes, execution environment, site/truth manifests, validation, federated results, and entity/duplicate scores. If the combined method does not improve metrics, retain the result and narrow the paper's conclusions accordingly. Do not adjust test-set thresholds to obtain higher scores.
