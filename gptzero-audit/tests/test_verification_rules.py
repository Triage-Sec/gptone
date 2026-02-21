import json
import tempfile
import unittest
from pathlib import Path

from src.reproducibility import file_sha256
from src.verification import verify_runs


def _write_manifest(
    path: Path,
    dataset_hash: str,
    clone: float,
    gz_acc: float,
    sg_acc: float,
    *,
    reg_mae: float = 0.1,
    ci_low: float = 0.8,
    per_class_human: float = 0.8,
    per_class_mixed: float = 0.8,
    per_class_ai: float = 0.95,
) -> Path:
    data_dir = path.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    test_path = data_dir / f"{path.stem}_test_results.json"
    test_payload = {
        "agreement_rate": clone,
        "agreement_ci_95": [ci_low, 0.99],
        "regressor_mae": reg_mae,
        "per_class_agreement": {
            "ai": per_class_ai,
            "human": per_class_human,
            "mixed": per_class_mixed,
        },
        "ground_truth_evaluation": {
            "gptzero_accuracy": gz_acc,
            "surrogate_accuracy": sg_acc,
        },
    }
    test_path.write_text(json.dumps(test_payload), encoding="utf-8")

    manifest = {
        "run_metadata": {
            "dataset": {"manifest_sha256": dataset_hash},
        },
        "artifacts": {
            "test_results": {
                "path": str(test_path),
                "sha256": file_sha256(test_path),
                "size_bytes": test_path.stat().st_size,
            }
        },
        "metrics": {},
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class VerificationRulesTest(unittest.TestCase):
    def test_verify_runs_passes_when_within_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            m1 = _write_manifest(base / "m1.json", "abc", 0.90, 0.45, 0.40)
            m2 = _write_manifest(base / "m2.json", "abc", 0.92, 0.48, 0.43)
            result = verify_runs(manifest_paths=[m1, m2], tolerances={})
            self.assertTrue(result["passed"])

    def test_verify_runs_fails_on_dataset_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            m1 = _write_manifest(base / "m1.json", "abc", 0.90, 0.45, 0.40)
            m2 = _write_manifest(base / "m2.json", "def", 0.91, 0.45, 0.40)
            result = verify_runs(manifest_paths=[m1, m2], tolerances={})
            self.assertFalse(result["passed"])
            hash_check = [c for c in result["checks"] if c["name"] == "dataset_manifest_hash_equal"][0]
            self.assertFalse(hash_check["passed"])

    def test_verify_runs_fails_on_metric_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            m1 = _write_manifest(base / "m1.json", "abc", 0.90, 0.45, 0.40)
            m2 = _write_manifest(base / "m2.json", "abc", 0.70, 0.20, 0.10)
            result = verify_runs(manifest_paths=[m1, m2], tolerances={})
            self.assertFalse(result["passed"])
            metric_check = [c for c in result["checks"] if c["name"] == "metric_deltas_within_tolerance"][0]
            self.assertFalse(metric_check["passed"])

    def test_verify_runs_enforces_promotion_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            m1 = _write_manifest(
                base / "m1.json",
                "abc",
                0.995,
                0.45,
                0.40,
                reg_mae=0.01,
                ci_low=0.98,
                per_class_ai=0.995,
                per_class_human=0.96,
                per_class_mixed=0.91,
            )
            m2 = _write_manifest(
                base / "m2.json",
                "abc",
                0.994,
                0.45,
                0.40,
                reg_mae=0.015,
                ci_low=0.979,
                per_class_ai=0.994,
                per_class_human=0.95,
                per_class_mixed=0.92,
            )
            m3 = _write_manifest(
                base / "m3.json",
                "abc",
                0.993,
                0.45,
                0.40,
                reg_mae=0.018,
                ci_low=0.978,
                per_class_ai=0.993,
                per_class_human=0.955,
                per_class_mixed=0.93,
            )
            result = verify_runs(
                manifest_paths=[m1, m2, m3],
                enforce_promotion=True,
                promotion_targets={},
                require_stability_runs=3,
            )
            self.assertTrue(result["passed"])
            prom_check = [c for c in result["checks"] if c["name"] == "promotion_targets_met"][0]
            self.assertTrue(prom_check["passed"])
            stability_check = [c for c in result["checks"] if c["name"] == "promotion_stability"][0]
            self.assertTrue(stability_check["passed"])

    def test_verify_runs_fails_promotion_on_low_human_class(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            m1 = _write_manifest(
                base / "m1.json",
                "abc",
                0.995,
                0.45,
                0.40,
                reg_mae=0.015,
                ci_low=0.98,
                per_class_human=0.20,
            )
            m2 = _write_manifest(
                base / "m2.json",
                "abc",
                0.994,
                0.45,
                0.40,
                reg_mae=0.016,
                ci_low=0.979,
                per_class_human=0.22,
            )
            m3 = _write_manifest(
                base / "m3.json",
                "abc",
                0.993,
                0.45,
                0.40,
                reg_mae=0.016,
                ci_low=0.978,
                per_class_human=0.21,
            )
            result = verify_runs(
                manifest_paths=[m1, m2, m3],
                enforce_promotion=True,
                promotion_targets={},
                require_stability_runs=3,
            )
            self.assertFalse(result["passed"])
            prom_check = [c for c in result["checks"] if c["name"] == "promotion_targets_met"][0]
            self.assertFalse(prom_check["passed"])


if __name__ == "__main__":
    unittest.main()
