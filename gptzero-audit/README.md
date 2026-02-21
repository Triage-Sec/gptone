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
- `artifacts/<run_id>/run_manifest.json`

Run manifest includes:

- CLI args and seed,
- dataset manifest hash,
- git commit hash,
- Python/platform/dependency versions,
- SHA256 checksums for each artifact.

## Dataset Versioning

Versioned benchmark data lives in:

- `datasets/v1/probes.jsonl`
- `datasets/v1/train_verified.jsonl`
- `datasets/v1/test_verified.jsonl`
- `datasets/v1/manifest.json`

`datasets/v1/manifest.json` contains checksums and record counts used for integrity checks.

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

The verifier checks:

- artifact checksum validity,
- dataset manifest hash equality,
- clone-score and accuracy metric drift within tolerances.

## Tests

Run test suite:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Tests cover:

- dataset manifest integrity,
- run manifest structure/checksum writing,
- reproducibility verifier pass/fail behavior.
