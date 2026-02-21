import json
import tempfile
import unittest
from pathlib import Path

from src.reproducibility import file_sha256
from src.verification import verify_runs


def _write_manifest(path: Path, dataset_hash: str, clone: float, gz_acc: float, sg_acc: float) -> Path:
    data_dir = path.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    test_path = data_dir / f"{path.stem}_test_results.json"
    test_payload = {
        "agreement_rate": clone,
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


if __name__ == "__main__":
    unittest.main()
