"""Run-to-run reproducibility and promotion-gate verification utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

from src.reproducibility import file_sha256


DEFAULT_TOLERANCES = {
    "clone_score_delta_max": 0.03,
    "gptzero_accuracy_delta_max": 0.05,
    "surrogate_accuracy_delta_max": 0.05,
}

DEFAULT_PROMOTION_TARGETS = {
    "agreement_min": 0.99,
    "probability_mae_max": 0.02,
    "ci_lower_min": 0.975,
    "per_class_min": {
        "ai": 0.99,
        "human": 0.95,
        "mixed": 0.90,
    },
    "clone_score_spread_max": 0.02,
    "mae_spread_max": 0.02,
    "require_runs": 3,
}


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_metric_bundle(manifest: Mapping[str, object]) -> Dict[str, object]:
    artifacts = manifest.get("artifacts", {})
    test_info = artifacts.get("test_results", {})
    test_path = Path(test_info.get("path", ""))
    if not test_path.exists():
        return {}
    test_results = _load_json(test_path)
    gt_eval = test_results.get("ground_truth_evaluation", {})
    ci = test_results.get("agreement_ci_95") or [0.0, 0.0]
    per_class_agreement = test_results.get("per_class_agreement") or {}
    if not isinstance(per_class_agreement, dict):
        per_class_agreement = {}
    return {
        "clone_score": float(test_results.get("agreement_rate", 0.0)),
        "gptzero_accuracy": float(gt_eval.get("gptzero_accuracy", 0.0) or 0.0),
        "surrogate_accuracy": float(gt_eval.get("surrogate_accuracy", 0.0) or 0.0),
        "regressor_mae": float(test_results.get("regressor_mae", 0.0) or 0.0),
        "agreement_ci_low": float(ci[0] if len(ci) >= 1 else 0.0),
        "agreement_ci_high": float(ci[1] if len(ci) >= 2 else 0.0),
        "per_class_agreement": {
            str(cls): float(val)
            for cls, val in per_class_agreement.items()
            if val is not None
        },
    }


def _promotion_failures(
    metric_bundle: Mapping[str, object],
    targets: Mapping[str, object],
) -> List[str]:
    failures: List[str] = []
    agreement = float(metric_bundle.get("clone_score", 0.0) or 0.0)
    mae = float(metric_bundle.get("regressor_mae", 0.0) or 0.0)
    ci_low = float(metric_bundle.get("agreement_ci_low", 0.0) or 0.0)
    per_class = metric_bundle.get("per_class_agreement", {})
    if not isinstance(per_class, dict):
        per_class = {}

    agreement_min = float(targets.get("agreement_min", 0.0))
    mae_max = float(targets.get("probability_mae_max", 1.0))
    ci_lower_min = float(targets.get("ci_lower_min", 0.0))

    if agreement < agreement_min:
        failures.append(f"agreement<{agreement_min:.4f}")
    if mae > mae_max:
        failures.append(f"regressor_mae>{mae_max:.4f}")
    if ci_low < ci_lower_min:
        failures.append(f"ci_low<{ci_lower_min:.4f}")

    per_class_min = targets.get("per_class_min", {})
    if isinstance(per_class_min, dict):
        for cls, min_rate in per_class_min.items():
            key = str(cls)
            if key not in per_class:
                # Skip gates for classes absent from the evaluated slice.
                continue
            actual = float(per_class.get(key, 0.0) or 0.0)
            min_rate_f = float(min_rate)
            if actual < min_rate_f:
                failures.append(f"per_class[{cls}]<{min_rate_f:.4f}")
    return failures


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
    promotion_targets: Mapping[str, object] | None = None,
    enforce_promotion: bool = False,
    require_stability_runs: int | None = None,
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

    if enforce_promotion:
        targets = dict(DEFAULT_PROMOTION_TARGETS)
        if promotion_targets:
            # Merge nested per-class dict with defaults.
            merged = dict(targets.get("per_class_min", {}))
            incoming_per_class = promotion_targets.get("per_class_min", {})
            if isinstance(incoming_per_class, dict):
                merged.update(incoming_per_class)
            targets.update(promotion_targets)
            targets["per_class_min"] = merged

        per_run = []
        promotion_pass = True
        for idx, bundle in enumerate(metric_bundles):
            failures = _promotion_failures(bundle, targets)
            if failures:
                promotion_pass = False
            per_run.append(
                {
                    "run_index": idx,
                    "metrics": bundle,
                    "failures": failures,
                }
            )

        checks.append(
            {
                "name": "promotion_targets_met",
                "passed": promotion_pass,
                "details": {
                    "targets": targets,
                    "runs": per_run,
                },
            }
        )

        required_runs = (
            int(require_stability_runs)
            if require_stability_runs is not None
            else int(targets.get("require_runs", 0))
        )
        stability_pass = True
        stability_details: Dict[str, object] = {"required_runs": required_runs}
        if required_runs > 0 and len(metric_bundles) < required_runs:
            stability_pass = False
            stability_details["error"] = "not_enough_runs"
            stability_details["available_runs"] = len(metric_bundles)
        else:
            considered = metric_bundles[:required_runs] if required_runs > 0 else metric_bundles
            clone_scores = [float(b.get("clone_score", 0.0) or 0.0) for b in considered]
            maes = [float(b.get("regressor_mae", 0.0) or 0.0) for b in considered]
            clone_spread = (max(clone_scores) - min(clone_scores)) if clone_scores else 0.0
            mae_spread = (max(maes) - min(maes)) if maes else 0.0
            clone_spread_max = float(targets.get("clone_score_spread_max", 1.0))
            mae_spread_max = float(targets.get("mae_spread_max", 1.0))
            if clone_spread > clone_spread_max:
                stability_pass = False
            if mae_spread > mae_spread_max:
                stability_pass = False
            stability_details.update(
                {
                    "runs_considered": len(considered),
                    "clone_spread": clone_spread,
                    "clone_spread_max": clone_spread_max,
                    "mae_spread": mae_spread,
                    "mae_spread_max": mae_spread_max,
                }
            )

        checks.append(
            {
                "name": "promotion_stability",
                "passed": stability_pass,
                "details": stability_details,
            }
        )

    passed = all(check["passed"] for check in checks)
    return {
        "passed": passed,
        "checks": checks,
        "manifest_paths": [str(p) for p in paths],
    }
