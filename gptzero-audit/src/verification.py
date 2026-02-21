"""Run-to-run reproducibility verification utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

from src.reproducibility import file_sha256


DEFAULT_TOLERANCES = {
    "clone_score_delta_max": 0.03,
    "gptzero_accuracy_delta_max": 0.05,
    "surrogate_accuracy_delta_max": 0.05,
}


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_metric_bundle(manifest: Mapping[str, object]) -> Dict[str, float]:
    artifacts = manifest.get("artifacts", {})
    test_info = artifacts.get("test_results", {})
    test_path = Path(test_info.get("path", ""))
    if not test_path.exists():
        return {}
    test_results = _load_json(test_path)
    gt_eval = test_results.get("ground_truth_evaluation", {})
    return {
        "clone_score": float(test_results.get("agreement_rate", 0.0)),
        "gptzero_accuracy": float(gt_eval.get("gptzero_accuracy", 0.0) or 0.0),
        "surrogate_accuracy": float(gt_eval.get("surrogate_accuracy", 0.0) or 0.0),
    }


def _artifact_checksum_failures(manifest: Mapping[str, object]) -> List[str]:
    failures: List[str] = []
    for key, info in manifest.get("artifacts", {}).items():
        expected = info.get("sha256")
        path = Path(info.get("path", ""))
        if expected is None:
            continue
        if not path.exists():
            failures.append(f"{key}:missing")
            continue
        actual = file_sha256(path)
        if actual != expected:
            failures.append(f"{key}:checksum_mismatch")
    return failures


def verify_runs(
    *,
    manifest_paths: Iterable[Path],
    tolerances: Mapping[str, float] | None = None,
) -> Dict[str, object]:
    """Verify reproducibility constraints across two or more run manifests."""
    tol = dict(DEFAULT_TOLERANCES)
    if tolerances:
        tol.update(tolerances)

    paths = [Path(p) for p in manifest_paths]
    if len(paths) < 2:
        raise ValueError("Need at least two manifests for reproducibility checks.")

    manifests = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing manifest: {path}")
        manifests.append(_load_json(path))

    checks = []

    # Check 1: all artifact checksums validate.
    checksum_issues = {}
    for path, manifest in zip(paths, manifests):
        issues = _artifact_checksum_failures(manifest)
        if issues:
            checksum_issues[str(path)] = issues
    checks.append(
        {
            "name": "artifact_checksums_valid",
            "passed": not checksum_issues,
            "details": checksum_issues,
        }
    )

    # Check 2: dataset manifest hash equality.
    dataset_hashes = [
        m.get("run_metadata", {})
        .get("dataset", {})
        .get("manifest_sha256")
        for m in manifests
    ]
    dataset_match = len(set(dataset_hashes)) == 1
    checks.append(
        {
            "name": "dataset_manifest_hash_equal",
            "passed": dataset_match,
            "details": dataset_hashes,
        }
    )

    # Check 3+: metric deltas within tolerance.
    metric_bundles = [_read_metric_bundle(m) for m in manifests]
    baseline = metric_bundles[0]
    metric_delta_details: Dict[str, object] = {"baseline": baseline, "deltas": []}
    metric_pass = True

    for idx, bundle in enumerate(metric_bundles[1:], start=1):
        delta = {
            "run_index": idx,
            "clone_score_delta": abs(bundle.get("clone_score", 0.0) - baseline.get("clone_score", 0.0)),
            "gptzero_accuracy_delta": abs(bundle.get("gptzero_accuracy", 0.0) - baseline.get("gptzero_accuracy", 0.0)),
            "surrogate_accuracy_delta": abs(bundle.get("surrogate_accuracy", 0.0) - baseline.get("surrogate_accuracy", 0.0)),
        }
        metric_delta_details["deltas"].append(delta)
        if delta["clone_score_delta"] > tol["clone_score_delta_max"]:
            metric_pass = False
        if delta["gptzero_accuracy_delta"] > tol["gptzero_accuracy_delta_max"]:
            metric_pass = False
        if delta["surrogate_accuracy_delta"] > tol["surrogate_accuracy_delta_max"]:
            metric_pass = False

    checks.append(
        {
            "name": "metric_deltas_within_tolerance",
            "passed": metric_pass,
            "details": {
                "tolerances": tol,
                "metrics": metric_delta_details,
            },
        }
    )

    passed = all(check["passed"] for check in checks)
    return {
        "passed": passed,
        "checks": checks,
        "manifest_paths": [str(p) for p in paths],
    }
