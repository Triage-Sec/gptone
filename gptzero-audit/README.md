# GPTZero Audit: Reproducible Evaluation Pipeline

This project evaluates GPTZero detector behavior with:

- extraction probes,
- surrogate model training,
- held-out validation,
- ground-truth accuracy checks on verified-provenance text,
- reproducibility manifests with checksums.

## Quick Start (Local)

```bash
cd gptzero-audit
pip install -r requirements.txt --break-system-packages
python main.py --api-key YOUR_GPTZERO_API_KEY
```

Deterministic run with explicit metadata:

```bash
python main.py \
  --api-key YOUR_GPTZERO_API_KEY \
  --seed 42 \
  --dataset-version v1 \
  --run-id run-20260220-001 \
  --output-dir .
```

## Modal Setup (Per Attached Flow)

Install and configure Modal locally:

```bash
pip install modal --break-system-packages
python3 -m modal setup
```

Create secret used by `modal_app.py`:

```bash
modal secret create gptzero-audit-secrets GPTZERO_API_KEY=YOUR_GPTZERO_API_KEY
```

The runtime accepts either `GPTZERO_API_KEY` or `GPT_ZERO_API_KEY` in Modal secrets.

Run remote pipeline:

```bash
modal run modal_app.py \
  --run-id modal-run-001 \
  --seed 42 \
  --dataset-version v1
```

## Artifacts And Manifest Structure

For each run:

- `data/extraction_results.json`
- `data/test_results.json`
- `models/surrogate_model.joblib`
- `models/surrogate_regressor.joblib`
- `models/model_meta.json`
- `report/gptzero_extraction_report.txt`
- `artifacts/<run_id>/snapshots/*.json|*.joblib|*.txt`
- `artifacts/<run_id>/run_manifest.json`

Run manifest includes:

- CLI args and seed,
- dataset manifest hash,
- git commit hash,
- Python/platform/dependency versions,
- SHA256 checksums for immutable per-run artifact snapshots.

## Dataset Versioning

Versioned benchmark data lives in:

- `datasets/v1/probes.jsonl`
- `datasets/v1/train_verified.jsonl`
- `datasets/v1/test_verified.jsonl`
- `datasets/v1/manifest.json`

`datasets/v1/manifest.json` contains checksums and record counts used for integrity checks.

Build the reason-guided augmentation dataset (`v2`) from `v1`:

```bash
python scripts/build_dataset_v2.py --source-version v1 --target-version v2
```

Then run the pipeline against `v2`:

```bash
python main.py \
  --api-key YOUR_GPTZERO_API_KEY \
  --seed 42 \
  --dataset-version v2 \
  --run-id run-v2-001 \
  --output-dir .
```

Mine near-boundary probes and build `v3` from `v2`:

```bash
python scripts/mine_boundary_probes.py \
  --source-version v2 \
  --target-version v3 \
  --max-base-samples 12 \
  --max-mined-samples 24 \
  --max-steps 3 \
  --branching-factor 4 \
  --boundary-low 0.35 \
  --boundary-high 0.65
```

Run the pipeline on boundary-mined data:

```bash
python main.py \
  --api-key YOUR_GPTZERO_API_KEY \
  --seed 42 \
  --dataset-version v3 \
  --run-id run-v3-001 \
  --output-dir .
```

Build the adaptive/quota boundary-mined dataset (`v4`) from `v2`:

```bash
python scripts/mine_boundary_probes.py \
  --source-version v2 \
  --target-version v4 \
  --max-base-samples 16 \
  --base-screen-size 48 \
  --max-mined-samples 18 \
  --fallback-top-k 10 \
  --max-steps 3 \
  --branching-factor 4 \
  --boundary-low 0.30 \
  --boundary-high 0.70 \
  --tight-boundary-low 0.40 \
  --tight-boundary-high 0.60 \
  --adaptive-band \
  --quota-ai 6 \
  --quota-human 6 \
  --quota-mixed 6
```

Build the empirical corpus-matrix dataset (`v5`) from `v4`:

```bash
python scripts/build_dataset_v5.py \
  --source-version v4 \
  --target-version v5 \
  --target-per-stratum 120 \
  --variants-per-augmented-base 3 \
  --seed 1337
```

Include external corpora (optional):

```bash
python scripts/build_dataset_v5.py \
  --source-version v4 \
  --target-version v5 \
  --external-corpus-config configs/external_corpora.example.json \
  --target-per-stratum 200
```

Run the probability-objective surrogate configuration:

```bash
python main.py \
  --api-key YOUR_GPTZERO_API_KEY \
  --seed 42 \
  --dataset-version v4 \
  --run-id run-v4-001 \
  --output-dir . \
  --use-api-features \
  --probability-objective \
  --oversample \
  --calibration-method temperature \
  --cv-repeats 5
```

Run perturbation-based pattern benchmark (writes `data/pattern_benchmark_results.json`):

```bash
python scripts/run_pattern_benchmark.py \
  --api-key YOUR_GPTZERO_API_KEY \
  --output-dir . \
  --dataset-version v5 \
  --max-base-per-stratum 80 \
  --variants-per-base 8 \
  --candidate-multiplier 6 \
  --boundary-focus \
  --blend-fit-split base \
  --stratum-calibration affine \
  --seed 1337
```

## Modal Matrix Execution

Run the strict empirical matrix from config:

```bash
modal run modal_app.py::matrix --config-path configs/modal_empirical_matrix.json
```

Single run + pattern benchmark (manual sequence):

```bash
modal run modal_app.py \
  --run-id modal-m5-001 \
  --seed 42 \
  --dataset-version v5

modal run modal_app.py::run_pattern_benchmark --config '{"run_id":"modal-m5-001","output_dir":"/artifacts/gptzero-audit/m5","dataset_version":"v5"}'
```

## Reproducibility Verification

Compare two or more runs:

```bash
python scripts/verify_run.py \
  --run-ids run-20260220-001 run-20260220-002 \
  --output-dir .
```

Or compare explicit manifests:

```bash
python scripts/verify_run.py \
  --manifest-paths \
    artifacts/run-20260220-001/run_manifest.json \
    artifacts/run-20260220-002/run_manifest.json
```

Enforce strict promotion gates (>=99% agreement, MAE/CI/per-class floors, stability):

```bash
python scripts/verify_run.py \
  --manifest-paths \
    artifacts/run-seed-42/run_manifest.json \
    artifacts/run-seed-1337/run_manifest.json \
    artifacts/run-seed-2026/run_manifest.json \
  --enforce-promotion-targets \
  --agreement-min 0.99 \
  --probability-mae-max 0.02 \
  --ci-lower-min 0.975 \
  --per-class-min-ai 0.99 \
  --per-class-min-human 0.95 \
  --per-class-min-mixed 0.90 \
  --clone-spread-max 0.02 \
  --mae-spread-max 0.02 \
  --require-stability-runs 3
```

Also enforce pattern-benchmark gates (overall + per-stratum):

```bash
python scripts/verify_run.py \
  --manifest-paths \
    artifacts/M5-seed-42/run_manifest.json \
    artifacts/M5-seed-1337/run_manifest.json \
    artifacts/M5-seed-2026/run_manifest.json \
  --enforce-promotion-targets \
  --enforce-pattern-targets \
  --pattern-overall-agreement-min 0.99 \
  --pattern-overall-probability-mae-max 0.02 \
  --pattern-overall-transition-min 0.97 \
  --pattern-stratum-agreement-min 0.97 \
  --pattern-stratum-probability-mae-max 0.03 \
  --pattern-stratum-transition-min 0.94 \
  --pattern-stratum-min-samples 30 \
  --pattern-required-strata human_pre_llm human_conversational ai_instructional ai_formal mixed_blended boundary_near
```

The verifier checks:

- artifact checksum validity,
- dataset manifest hash equality,
- clone-score and accuracy metric drift within tolerances,
- strict promotion targets (optional),
- pattern benchmark overall + stratum targets (optional).

## Tests

Run test suite:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Tests cover:

- dataset manifest integrity,
- run manifest structure/checksum writing,
- reproducibility verifier pass/fail behavior,
- UI-reason-guided probe taxonomy coverage.
