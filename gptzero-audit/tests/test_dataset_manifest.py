import json
import unittest
from pathlib import Path

from src.reproducibility import file_sha256


class DatasetManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.base = Path(__file__).resolve().parents[1] / "datasets" / "v1"
        self.manifest_path = self.base / "manifest.json"
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)

    def test_manifest_exists(self) -> None:
        self.assertTrue(self.manifest_path.exists())
        self.assertEqual(self.manifest.get("dataset_version"), "v1")

    def test_file_checksums_match_manifest(self) -> None:
        files = self.manifest.get("files", {})
        self.assertIn("probes.jsonl", files)
        self.assertIn("train_verified.jsonl", files)
        self.assertIn("test_verified.jsonl", files)

        for file_name, info in files.items():
            path = self.base / file_name
            self.assertTrue(path.exists(), msg=f"Missing dataset file {file_name}")
            self.assertEqual(file_sha256(path), info["sha256"])

    def test_manifest_counts_are_consistent(self) -> None:
        counts = self.manifest.get("counts", {})
        self.assertIn("probes_by_collection", counts)
        self.assertIn("train_verified_by_label", counts)
        self.assertIn("test_verified_by_label", counts)

        expected_probe_total = self.manifest["files"]["probes.jsonl"]["record_count"]
        actual_probe_total = sum(counts["probes_by_collection"].values())
        self.assertEqual(expected_probe_total, actual_probe_total)

        expected_train_total = self.manifest["files"]["train_verified.jsonl"]["record_count"]
        actual_train_total = sum(counts["train_verified_by_label"].values())
        self.assertEqual(expected_train_total, actual_train_total)

        expected_test_total = self.manifest["files"]["test_verified.jsonl"]["record_count"]
        actual_test_total = sum(counts["test_verified_by_label"].values())
        self.assertEqual(expected_test_total, actual_test_total)


if __name__ == "__main__":
    unittest.main()
