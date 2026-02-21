import json
import tempfile
import unittest
from pathlib import Path

from src.reproducibility import (
    build_run_metadata,
    configure_run_paths,
    snapshot_artifacts_for_run,
    write_run_manifest,
)


class ReproducibilityManifestTest(unittest.TestCase):
    def test_manifest_contains_required_fields_and_artifact_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run_paths = configure_run_paths(base, "run-test")

            extraction = run_paths.data_dir / "extraction_results.json"
            extraction.write_text(json.dumps({"results": []}), encoding="utf-8")

            test_results = run_paths.data_dir / "test_results.json"
            test_results.write_text(
                json.dumps(
                    {
                        "agreement_rate": 0.5,
                        "ground_truth_evaluation": {
                            "gptzero_accuracy": 0.4,
                            "surrogate_accuracy": 0.35,
                        },
                    }
                ),
                encoding="utf-8",
            )

            metadata = build_run_metadata(
                project_root=Path(__file__).resolve().parents[1],
                seed=42,
                run_id="run-test",
                dataset_version="v1",
                args={"seed": 42, "dataset_version": "v1"},
                api_endpoint="https://api.gptzero.me/v2/predict/text",
            )

            write_run_manifest(
                manifest_path=run_paths.manifest_path,
                run_metadata=metadata,
                artifact_paths={
                    "extraction_results": extraction,
                    "test_results": test_results,
                },
                extra_metrics={"clone_score": 0.5},
            )

            with run_paths.manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)

            self.assertIn("run_metadata", manifest)
            self.assertIn("artifacts", manifest)
            self.assertIn("metrics", manifest)
            self.assertEqual(manifest["run_metadata"]["seed"], 42)
            self.assertEqual(manifest["run_metadata"]["run_id"], "run-test")
            self.assertIsNotNone(manifest["artifacts"]["extraction_results"]["sha256"])
            self.assertIsNotNone(manifest["artifacts"]["test_results"]["sha256"])

    def test_snapshot_artifacts_for_run_creates_immutable_copies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run_paths = configure_run_paths(base, "run-test")
            source_file = run_paths.data_dir / "test_results.json"
            source_file.write_text(json.dumps({"agreement_rate": 0.9}), encoding="utf-8")

            snapped = snapshot_artifacts_for_run(
                run_dir=run_paths.run_dir,
                artifact_paths={"test_results": source_file},
            )
            snapshot_path = snapped["test_results"]

            self.assertTrue(snapshot_path.exists())
            self.assertEqual(run_paths.run_dir / "snapshots", snapshot_path.parent)
            self.assertEqual(
                source_file.read_text(encoding="utf-8"),
                snapshot_path.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
